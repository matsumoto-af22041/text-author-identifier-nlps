"""全体統合実験：特徴量 × チャンク長 × 分割方式 × 類似度 × 推論方式の評価と結果出力．

使い方：
    uv run python scripts/run_experiments.py --config configs/default.yaml
    uv run python scripts/run_experiments.py --features A --chunk-chars 500 --ablation
    uv run python scripts/run_experiments.py --features C --exclude-works 坊っちゃん --run-name excl_bocchan_C

分割・シード・間引き（configs/default.yaml の split）：
    Intra-work は split.intra_seeds の各シードで 1 回ずつ分割．
    split.balance_train が true のとき，同じ run で Train の間引きあり・なしの両方を実行（Test は間引かない）．
        Intra-work：分割と同じシードの独立な子乱数列 default_rng(SeedSequence(seed).spawn(1)[0]) で間引く．
        Inter-work：split.balance_seeds の各シードで，fold ごとに default_rng([seed, fold 番号]) で間引く（014 と同じ）．
    間引きなしの Inter-work は乱数を使わない決定的な分割のため 1 回のみ（seed 欄は空）．

出力（outputs/<run_name>/）：
    run_meta.json          実行時の設定・CLI 引数・git 状態・コーパスハッシュ・fold ごとの n_min（balance_train）
    folds.csv              fold 単位の結果（Inter-work は 1 シードあたり 9 fold）
    summary.csv            条件単位の結果（Inter-work は全 fold の Pooled Out-of-Fold）
                           row_type=seed：シードごとの値，row_type=mean：シード平均（<指標>_std は標本標準偏差，ddof = 1）
                           n_ties：同点（判別不能）件数，n_zero_query：ノルム 0 の照会チャンク数
                           計算時間は再現性の比較を妨げるため timings.csv に分離
    timings.csv            条件・シードごとの特徴抽出・類似度計算の所要時間
    results.json           summary と混同行列（plot_results.py の入力）
    similarity_stats.json  類似度統計（作家間ペア類似度和・ペア数・ヒストグラム，シード別と全シード合算）
    misclassified_best.md  最良モデルの誤判別チャンク抜粋
    report.md, figures/    plot_results.py による図表
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import json
import platform
import random
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.dataset import (
    ChunkSet,
    Corpus,
    Split,
    balance_train_split,
    build_chunks,
    inter_work_splits,
    intra_work_split,
    load_corpus,
)
from src.evaluator import ClassifierSpec, FoldResult, SimilarityStats, aggregate_results, evaluate_fold
from src.features.base import BaseFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureVariant:
    """特徴量設定（本採用，アブレーションで 1 引数のみ差し替えたもの，または combinations で複数引数を差し替えたもの）．"""

    feature: str
    name: str
    cls: type[BaseFeatureExtractor]
    params: dict[str, Any]
    is_default: bool

    def make(self) -> BaseFeatureExtractor:
        return self.cls(**self.params)


# ---------------------------------------------------------------- 設定の解釈


def load_feature_class(key: str, spec: dict[str, Any]) -> type[BaseFeatureExtractor]:
    module_path, _, class_name = spec["class"].rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"特徴量 {key}（{spec.get('description', '')}）のモジュール {module_path} が未実装．"
            f"--features で実装済みの特徴量のみを指定すること"
        ) from e
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseFeatureExtractor)):
        raise TypeError(f"{spec['class']} は BaseFeatureExtractor のサブクラスではない")
    return cls


def build_variants(cfg: dict[str, Any], keys: Sequence[str], with_ablation: bool) -> list[FeatureVariant]:
    unknown = set(keys) - set(cfg["features"])
    if unknown:
        raise KeyError(f"features に未登録の特徴量: {sorted(unknown)}（登録済み: {sorted(cfg['features'])}）")

    def variants_of(key: str) -> list[FeatureVariant]:
        spec = cfg["features"][key]
        cls = load_feature_class(key, spec)
        params = dict(spec["params"])
        default = FeatureVariant(key, key, cls, params, True)
        ablations = [
            FeatureVariant(key, f"{key}[{p}={v}]", cls, params | {p: v}, False)
            for p, values in (spec.get("ablations") or {}).items()
            for v in values
        ]
        # 複数引数の同時差し替え（例：B の内容語対照 × 語ブロックのみ，plans/025）
        combos = spec.get("combinations") or []
        unknown_params = sorted({p for c in combos for p in c} - set(params))
        if unknown_params:
            raise KeyError(f"features.{key}.combinations に params に無い引数: {unknown_params}")
        if any(len(c) < 2 for c in combos):
            raise ValueError(f"features.{key}.combinations の各要素は 2 引数以上（1 引数は ablations に記載）: {combos}")
        combinations = [
            FeatureVariant(key, f"{key}[{','.join(f'{p}={c[p]}' for p in sorted(c))}]", cls, params | c, False)
            for c in combos
        ]
        names = [v.name for v in (default, *ablations, *combinations)]
        if len(names) != len(set(names)):
            raise ValueError(f"特徴量 {key} の条件名が重複: {names}")
        return [default, *((ablations + combinations) if with_ablation else [])]

    return [v for key in keys for v in variants_of(key)]


def classifier_specs(cfg: dict[str, Any]) -> list[ClassifierSpec]:
    c = cfg["classifier"]
    unknown = set(c["methods"]) - {"centroid", "knn"}
    if unknown:
        raise ValueError(f"未知の推論方式: {unknown}")
    centroid = [ClassifierSpec("centroid")] if "centroid" in c["methods"] else []
    knn = [ClassifierSpec("knn", k, m) for m in c["knn_modes"] for k in c["k_values"]] if "knn" in c["methods"] else []
    return [*centroid, *knn]


@dataclass(frozen=True)
class SplitRun:
    """1 シード分の分割群（Intra-work は 1 split，Inter-work は 9 fold）．seed=None は乱数を使わない分割．"""

    split_type: str
    balanced: bool
    seed: int | None
    splits: tuple[Split, ...]

    @property
    def balance_tag(self) -> str:
        return "bal" if self.balanced else "nobal"

    @property
    def seed_tag(self) -> str:
        return "det" if self.seed is None else f"s{self.seed}"


INTRA_BALANCE_RNG = "np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])"
INTER_BALANCE_RNG = "np.random.default_rng([seed, fold_index])"


def intra_balance_rng(seed: int) -> np.random.Generator:
    """Intra-work の間引き用乱数：分割と同じシードの独立な子乱数列．

    default_rng([seed, 0]) は default_rng(seed)（分割の乱数列）と同一の状態になるため使わない．
    """
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def validate_split_config(sc: dict[str, Any]) -> None:
    if not isinstance(sc["balance_train"], bool):
        raise TypeError(f"split.balance_train は bool: {sc['balance_train']!r}")
    for name in ("intra_seeds", "balance_seeds"):
        seeds = sc[name]
        if not seeds or not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError(f"split.{name} は重複のない整数の非空リスト: {seeds!r}")


def make_split_runs(chunk_set: ChunkSet, split_type: str, cfg: dict[str, Any]) -> tuple[list[SplitRun], dict[str, list[dict[str, Any]]]]:
    """分割方式ごとの全シード・間引きあり / なしの分割群と，間引きの記録（シード → fold ごとの記録）．"""
    sc = cfg["split"]
    match split_type:
        case "intra":
            seeds = list(sc["intra_seeds"])
            plain = [SplitRun("intra", False, seed, (intra_work_split(chunk_set, sc["intra_test_ratio"], seed),)) for seed in seeds]
            # シード → (間引き後の Split 列, 記録)
            balanced = {r.seed: balanced_splits(chunk_set, r.splits, lambda i, seed=r.seed: intra_balance_rng(seed)) for r in plain}
        case "inter":
            seeds = list(sc["balance_seeds"])
            plain = [SplitRun("inter", False, None, tuple(inter_work_splits(chunk_set)))]
            balanced = {seed: balanced_inter_splits(chunk_set, seed) for seed in seeds}
        case _:
            raise ValueError(f"未知の分割方式: {split_type}")
    if not sc["balance_train"]:
        return plain, {}
    bal_runs = [SplitRun(split_type, True, seed, tuple(b)) for seed, (b, _) in balanced.items()]
    return [*plain, *bal_runs], {str(seed): rec for seed, (_, rec) in balanced.items()}


def exclude_works(corpus: Corpus, titles: Sequence[str], min_works_per_author: int = 2) -> Corpus:
    """作品名で指定した作品をコーパスから除外（チャンク分割の前に適用し Intra / Inter の両方に効く）．

    存在しない作品名，作品名の重複，除外後に作品数が min_works_per_author 未満となる作家はいずれも ValueError．
    """
    all_titles = [w.title for w in corpus.works]
    dup = sorted(t for t, n in Counter(all_titles).items() if n > 1)
    if dup:
        raise ValueError(f"作品名が一意でないため除外対象を特定できない: {dup}")
    unknown = sorted(set(titles) - set(all_titles))
    if unknown:
        raise ValueError(f"--exclude-works に存在しない作品名: {unknown}（対象作品: {all_titles}）")
    kept = tuple(w for w in corpus.works if w.title not in set(titles))
    counts = Counter(w.author_id for w in kept)
    short = {name: counts[a] for a, name in zip(corpus.author_ids, corpus.author_names, strict=True) if counts[a] < min_works_per_author}
    if short:
        raise ValueError(f"除外後に作品数が {min_works_per_author} 未満となる作家（Inter-work 評価不能）: {short}")
    return dataclasses.replace(corpus, works=kept)


def balanced_inter_splits(chunk_set: ChunkSet, seed: int) -> tuple[list[Split], list[dict[str, Any]]]:
    """Inter-work の各 fold の Train を作家別に n_min 個へ間引く．

    fold ごとの乱数生成器は default_rng([seed, fold 番号]) で，fold の実行順に依存しない．
    戻り値：間引き後の Split 列と，fold ごとの作家別 Train チャンク数（間引き前・後）の記録．
    """
    return balanced_splits(chunk_set, inter_work_splits(chunk_set), lambda i: np.random.default_rng([seed, i]))


def balanced_splits(
    chunk_set: ChunkSet, splits: Sequence[Split], rng_of: Callable[[int], np.random.Generator]
) -> tuple[list[Split], list[dict[str, Any]]]:
    """各 split の Train を作家別に n_min 個へ間引き，fold ごとの記録（014 と同形式）を返す．rng_of(fold 番号) が乱数生成器．"""
    labels, author_ids = chunk_set.labels, chunk_set.author_ids
    num_authors = len(author_ids)
    balanced = [balance_train_split(s, labels, rng_of(i)) for i, s in enumerate(splits)]

    def counts(idx: np.ndarray) -> dict[str, int]:
        return dict(zip(author_ids, np.bincount(labels[idx], minlength=num_authors).tolist(), strict=True))

    def record(i: int, before: Split, after: Split) -> dict[str, Any]:
        n_before = counts(before.train_idx)
        n_min = min(n_before.values())
        test_works = np.unique(chunk_set.work_ids[before.test_idx])
        return {
            "fold_index": i,
            "fold": before.name,
            # Inter-work は Test 作品（1 作品），Intra-work は全作品が Test に現れるため None
            "test_work": str(test_works[0]) if test_works.size == 1 else None,
            "n_min": n_min,
            "n_min_authors": [a for a, n in n_before.items() if n == n_min],
            "n_train_before": n_before,
            "n_train_after": counts(after.train_idx),
            "test_unchanged": bool(np.array_equal(before.test_idx, after.test_idx)),
        }

    records = [record(i, b, a) for i, (b, a) in enumerate(zip(splits, balanced, strict=True))]
    bad = [r["fold"] for r in records if not r["test_unchanged"] or set(r["n_train_after"].values()) != {r["n_min"]}]
    if bad:
        raise AssertionError(f"間引き後の作家別 Train チャンク数が n_min と不一致，または Test が変化: {bad}")
    return balanced, records


# ---------------------------------------------------------------- 追跡情報


def git_state() -> dict[str, Any]:
    run = lambda *a: subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout  # noqa: E731
    dirty = [line for line in run("status", "--porcelain").splitlines() if line.strip()]
    return {"commit": run("rev-parse", "HEAD").strip(), "dirty": bool(dirty), "dirty_files": dirty}


def display_path(path: Path) -> str:
    """プロジェクト内なら相対パス，外なら絶対パス．"""
    p = path.resolve()
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)


def corpus_meta(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        meta = json.load(f)["meta"]
    return {k: meta[k] for k in ("source_sha256", "source_encoding", "git_commit")}


# ---------------------------------------------------------------- 実験本体


def condition_key(
    variant: str, L: int, split_type: str, balance_tag: str, similarity: str, classifier: str | None = None, seed_tag: str | None = None
) -> str:
    """条件キー：variant|L|split|bal or nobal|similarity[|classifier][|mean or s<seed> or det]．"""
    parts = [variant, f"L{L}", split_type, balance_tag, similarity, *([classifier] if classifier else []), *([seed_tag] if seed_tag else [])]
    return "|".join(parts)


def metric_columns(author_ids: Sequence[str]) -> list[str]:
    """シード平均・標準偏差を取る指標列（summary.csv の列順）．"""
    return [
        "accuracy",
        "balanced_accuracy",
        "balanced_accuracy_random",
        "macro_f1",
        "macro_f1_random",
        "mrr",
        "mrr_random",
        "map_macro",
        "map_macro_random",
        "map_micro",
        "map_micro_random",
        *[f"author_acc_{a}" for a in author_ids],
        *[f"author_ap_{a}" for a in author_ids],
        "n_ties",
        "n_zero_query",
        "mu_same_macro",
        "mu_diff_macro",
        "delta_s_macro",
        "mu_same",
        "mu_diff",
        "delta_s_micro",
    ]


def summary_row(
    variant: FeatureVariant, L: int, run: SplitRun, folds: Sequence[FoldResult], author_ids: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """1 シード分の Pooled 行．戻り値：summary 行，results.json 用の詳細，計算時間の行．"""
    agg = aggregate_results(folds)
    r0 = folds[0]
    m, ret = agg.metrics, agg.retrieval
    # 作家別正解率（= 作家別再現率）．Pooled では全作家が Test に現れる
    acc_by_author = m.get("recall", [float("nan")] * len(author_ids))
    values = {
        "accuracy": m["accuracy"],
        "balanced_accuracy": m["balanced_accuracy"],
        "macro_f1": m["macro_f1"],
        "mrr": agg.mrr,
        **agg.baselines,
        **{k: ret[k] for k in ("map_macro", "map_macro_random", "map_micro", "map_micro_random")},
        **{f"author_acc_{a}": v for a, v in zip(author_ids, acc_by_author, strict=True)},
        **{f"author_ap_{a}": v for a, v in zip(author_ids, ret["ap_by_author"], strict=True)},
        "n_ties": agg.n_ties,
        "n_zero_query": agg.n_zero_query,
        **agg.sim_stats.scalars(),
    }
    cols = metric_columns(author_ids)
    assert set(values) == set(cols), f"指標列の不一致: {set(values) ^ set(cols)}"
    row = {
        "key": condition_key(variant.name, L, run.split_type, run.balance_tag, r0.similarity, r0.spec.name, run.seed_tag),
        "row_type": "seed",
        "variant": variant.name,
        "feature": variant.feature,
        "is_default": variant.is_default,
        "params": json.dumps(variant.params, ensure_ascii=False, sort_keys=True),
        "chunk_chars": L,
        "split": run.split_type,
        "balance_train": run.balanced,
        "seed": run.seed,
        "n_seeds": 1,
        "seeds": json.dumps([] if run.seed is None else [run.seed]),
        "similarity": r0.similarity,
        "classifier": r0.spec.name,
        "method": r0.spec.method,
        "k": r0.spec.k,
        "knn_mode": r0.spec.knn_mode,
        "num_folds": agg.num_folds,
        "n_test": agg.n_test,
        "n_train_mean": float(np.mean([r.n_train for r in folds])),
        "n_features": int(np.mean([r.n_features for r in folds])),
        **{c: values[c] for c in cols},
        **{f"{c}_std": None for c in cols},
    }
    detail = {**row, "confusion": agg.confusion.tolist(), **{k: m.get(k) for k in ("precision", "recall", "f1")}}
    timing = {
        "key": row["key"],
        "extract_seconds": float(sum(r.extract_seconds for r in folds)),
        "similarity_seconds": float(sum(r.similarity_seconds for r in folds)),
    }
    return row, detail, timing


def mean_row(details: Sequence[dict[str, Any]], author_ids: Sequence[str]) -> dict[str, Any]:
    """同一条件のシード行からシード平均の行を作成（標準偏差は ddof = 1，1 シードのみなら NaN）．

    混同行列は全シードの合算，作家別 Precision / Recall / F1 はシード平均．
    """
    d0 = details[0]
    invariant = ("variant", "chunk_chars", "split", "balance_train", "similarity", "classifier", "n_test")
    if any(d[k] != d0[k] for d in details for k in invariant):
        raise ValueError(f"異なる条件のシード行が混在: {d0['key']}")
    cols = metric_columns(author_ids)
    # shape: (n_seeds, n_metrics)
    M = np.array([[d[c] for c in cols] for d in details], dtype=np.float64)
    std = M.std(axis=0, ddof=1) if len(details) > 1 else np.full(len(cols), np.nan)
    seeds = [d["seed"] for d in details if d["seed"] is not None]
    tag = condition_key(d0["variant"], d0["chunk_chars"], d0["split"], "bal" if d0["balance_train"] else "nobal", d0["similarity"], d0["classifier"], "mean")
    return {
        **{k: v for k, v in d0.items() if k not in {*cols, *(f"{c}_std" for c in cols), "confusion", "precision", "recall", "f1"}},
        "key": tag,
        "row_type": "mean",
        "seed": None,
        "n_seeds": len(details),
        "seeds": json.dumps(seeds),
        "n_train_mean": float(np.mean([d["n_train_mean"] for d in details])),
        **dict(zip(cols, M.mean(axis=0).tolist(), strict=True)),
        **{f"{c}_std": float(v) for c, v in zip(cols, std.tolist(), strict=True)},
        "confusion": np.sum([d["confusion"] for d in details], axis=0).tolist(),
        **{k: np.mean([d[k] for d in details], axis=0).tolist() if d0.get(k) is not None else None for k in ("precision", "recall", "f1")},
    }


def fold_rows(variant: FeatureVariant, L: int, run: SplitRun, folds: Sequence[FoldResult]) -> list[dict[str, Any]]:
    return [
        {
            "variant": variant.name,
            "chunk_chars": L,
            "split": run.split_type,
            "balance_train": run.balanced,
            "seed": run.seed,
            "fold": r.split_name,
            "similarity": r.similarity,
            "classifier": r.spec.name,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "n_features": r.n_features,
            "accuracy": r.metrics["accuracy"],
            "balanced_accuracy": r.metrics["balanced_accuracy"],
            "macro_f1": r.metrics["macro_f1"],
            "mrr": r.mrr,
            "map_macro": r.retrieval["map_macro"],
            "map_micro": r.retrieval["map_micro"],
            "map_micro_random": r.retrieval["map_micro_random"],
            "n_ties": r.n_ties,
            "n_zero_query": r.n_zero_query,
            **r.sim_stats.scalars(),
            "extract_seconds": r.extract_seconds,
            "similarity_seconds": r.similarity_seconds,
        }
        for r in folds
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}: 出力行が空")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_misclassified(
    path: Path,
    best: dict[str, Any],
    folds: Sequence[FoldResult],
    chunk_set: ChunkSet,
    corpus: Corpus,
    num_examples: int,
    excerpt_chars: int,
) -> None:
    """最良モデルの誤判別を確信度の高い順（最大スコア − 正解作家スコアの降順）に抜粋．`best` は 1 シード分の行．

    同点（予測ラベル C）も誤判別に含め，予測欄は「同点」と表記（margin は最大スコア基準）．
    """
    titles = {w.work_id: w.title for w in corpus.works}
    # 予測ラベル C（同点）の表示名を末尾に追加
    authors = [*corpus.author_names, "同点"]
    # shape: (n_test_total,), (n_test_total, C)
    idx = np.concatenate([r.test_idx for r in folds])
    y_true = np.concatenate([r.y_true for r in folds])
    y_pred = np.concatenate([r.y_pred for r in folds])
    scores = np.concatenate([r.scores for r in folds])
    # shape: (n_test_total,)，同点時も定義される最大スコアを予測側のスコアとする
    margin = scores.max(axis=1) - np.take_along_axis(scores, y_true[:, None], 1)[:, 0]
    wrong = np.flatnonzero(y_true != y_pred)
    order = wrong[np.argsort(-margin[wrong], kind="stable")][:num_examples]
    pair_counts = Counter((authors[t], authors[p]) for t, p in zip(y_true[wrong], y_pred[wrong], strict=True))

    def example(i: int) -> str:
        c = chunk_set.chunks[idx[i]]
        text = c.text[:excerpt_chars] + ("…" if len(c.text) > excerpt_chars else "")
        return (
            f"| {titles[c.work_id]} #{c.index_in_work} | {authors[y_true[i]]} | {authors[y_pred[i]]} | "
            f"{margin[i]:.4f} | {len(c.text)} | {text.replace('|', '｜')} |"
        )

    lines = [
        "# 最良モデルの誤判別チャンク抜粋",
        "",
        f"- 条件：`{best['key']}`（params：`{best['params']}`）",
        f"- 抜粋に用いた分割：`{best['key']}`（シード平均で選んだ条件のうち先頭シード）",
        f"- このシードの Pooled 指標：作家別正解率の平均 {best['balanced_accuracy']:.4f}，作家別 F 値の平均 {best['macro_f1']:.4f}，MRR {best['mrr']:.4f}",
        f"- 誤判別数：{wrong.size} / {y_true.size}（うち同点 {best['n_ties']}，ノルム 0 の照会 {best['n_zero_query']}）",
        "",
        "## 誤判別の内訳（正解 → 予測）",
        "",
        "| 正解 | 予測 | 件数 |",
        "| :--- | :--- | ---: |",
        *[f"| {t} | {p} | {n} |" for (t, p), n in pair_counts.most_common()],
        "",
        f"## 確信度の高い誤判別（上位 {order.size} 件，margin = 最大スコア − 正解作家スコア）",
        "",
        "| 作品 #チャンク | 正解 | 予測 | margin | 文字数 | 抜粋 |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
        *[example(i) for i in order],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--features", nargs="+", help="実行する特徴量キー（既定：experiment.features）")
    parser.add_argument("--chunk-chars", nargs="+", type=int, help="チャンク長 L（既定：chunking.target_chunk_chars）")
    parser.add_argument("--splits", nargs="+", choices=["intra", "inter"], help="分割方式（既定：experiment.splits）")
    parser.add_argument("--ablation", action="store_true", help="アブレーション設定も実行")
    parser.add_argument("--run-name", help="出力ディレクトリ名（既定：日時）")
    parser.add_argument("--no-report", action="store_true", help="図表・report.md の生成を省略")
    parser.add_argument("--exclude-works", nargs="+", help="除外する作品名（チャンク分割前に適用，補助実験用）")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)

    feature_keys = args.features or cfg["experiment"]["features"]
    chunk_chars = args.chunk_chars or cfg["chunking"]["target_chunk_chars"]
    split_types = args.splits or cfg["experiment"]["splits"]
    ablation_L = cfg["experiment"]["ablation_chunk_chars"]
    validate_split_config(cfg["split"])
    if args.ablation and not set(chunk_chars) & set(ablation_L):
        raise ValueError(f"--ablation 指定時は --chunk-chars に experiment.ablation_chunk_chars {ablation_L} のいずれかを含めること")
    # 長時間の実行前に全特徴量クラスの読み込みを検証（Fail-Fast）
    variants = build_variants(cfg, feature_keys, args.ablation)
    specs = classifier_specs(cfg)
    sims = list(cfg["similarity"]["names"])
    sim_kwargs = {"eps": cfg["similarity"]["eps"]}
    num_bins = cfg["evaluation"]["similarity_hist_bins"]

    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    corpus = load_corpus(corpus_path)
    excluded = [w for w in corpus.works if args.exclude_works and w.title in set(args.exclude_works)]
    if args.exclude_works:
        corpus = exclude_works(corpus, args.exclude_works)
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = PROJECT_ROOT / cfg["experiment"]["output_dir"] / run_name
    if run_dir.exists():
        raise FileExistsError(f"出力先が既に存在（上書き防止）: {run_dir}")
    run_dir.mkdir(parents=True)

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": display_path(args.config),
        "config": cfg,
        "cli": {
            "features": feature_keys,
            "chunk_chars": chunk_chars,
            "splits": split_types,
            "ablation": args.ablation,
            **({"exclude_works": list(args.exclude_works)} if args.exclude_works else {}),
        },
        "variants": [{"name": v.name, "class": f"{v.cls.__module__}.{v.cls.__name__}", "params": v.params} for v in variants],
        "classifiers": [s.name for s in specs],
        "similarities": sims,
        "git": git_state(),
        "corpus": corpus_meta(corpus_path),
        "author_ids": list(corpus.author_ids),
        "author_names": list(corpus.author_names),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if args.exclude_works:
        meta["excluded_works"] = [{"work_id": w.work_id, "author_id": w.author_id, "title": w.title} for w in excluded]
        meta["included_works"] = [{"work_id": w.work_id, "author_id": w.author_id, "title": w.title} for w in corpus.works]
        print(f"[info] 除外作品: {[w.title for w in excluded]}（残り {len(corpus.works)} 作品）")
    if cfg["split"]["balance_train"]:
        # folds[L][seed] は 014 と同形式の fold ごとの記録（n_min，n_min_authors，間引き前後の作家別 Train チャンク数）
        meta["balance_train"] = {
            "intra": {"seeds": cfg["split"]["intra_seeds"], "rng": INTRA_BALANCE_RNG, "folds": {}},
            "inter": {"seeds": cfg["split"]["balance_seeds"], "rng": INTER_BALANCE_RNG, "folds": {}},
        }
        print(f"[info] Train 間引きあり・なしの両方を実行（Intra seeds={cfg['split']['intra_seeds']}，Inter seeds={cfg['split']['balance_seeds']}）")
    if meta["git"]["dirty"]:
        print(f"[warn] 未コミットの変更あり（run_meta.json に記録）: {meta['git']['dirty_files']}")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    mean_details: list[dict[str, Any]] = []
    seed_details: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    folds_out: list[dict[str, Any]] = []
    sim_stats: dict[str, Any] = {}
    fold_store: dict[str, list[FoldResult]] = {}
    chunk_sets: dict[int, ChunkSet] = {}
    author_ids = list(corpus.author_ids)
    t_start = time.perf_counter()

    for L in chunk_chars:
        chunk_sets[L] = cs = build_chunks(corpus, L)
        runs: dict[str, list[SplitRun]] = {}
        for split_type in split_types:
            runs[split_type], records = make_split_runs(cs, split_type, cfg)
            if records:
                meta["balance_train"][split_type]["folds"][str(L)] = records
                n_min = sorted({(r["fold"], r["n_min"]) for recs in records.values() for r in recs}, key=lambda x: x[1])
                print(f"[info] L={L} {split_type} n_min（最小 3 件）: {n_min[:3]}")
        active = [v for v in variants if v.is_default or L in cfg["experiment"]["ablation_chunk_chars"]]
        for v in active:
            for split_type in split_types:
                t0 = time.perf_counter()
                cond_details: list[dict[str, Any]] = []
                for run in runs[split_type]:
                    results = [
                        r
                        for s in run.splits
                        for r in evaluate_fold(v.make, cs, s, sims, specs, similarity_kwargs=sim_kwargs, num_bins=num_bins)
                    ]
                    # 類似度 × 推論器ごとに fold をまとめる（results は fold → 類似度 → 推論器の順）
                    conditions = list(dict.fromkeys((r.similarity, r.spec) for r in results))
                    grouped = {c: [r for r in results if (r.similarity, r.spec) == c] for c in conditions}
                    for (sim, _), folds in grouped.items():
                        row, detail, timing = summary_row(v, L, run, folds, author_ids)
                        cond_details.append(detail)
                        timings.append(timing)
                        fold_store[row["key"]] = folds
                        sim_key = condition_key(v.name, L, split_type, run.balance_tag, sim, seed_tag=run.seed_tag)
                        sim_stats.setdefault(sim_key, aggregate_results(folds).sim_stats)
                    folds_out.extend(fold_rows(v, L, run, results))
                seed_details.extend(cond_details)
                # 全シード合算の類似度統計（分布図用）
                for bal in dict.fromkeys(r.balance_tag for r in runs[split_type]):
                    for sim in sims:
                        prefix = condition_key(v.name, L, split_type, bal, sim) + "|"
                        sim_stats[prefix + "all"] = SimilarityStats.merge([st for k, st in sim_stats.items() if k.startswith(prefix) and not k.endswith("|all")])
                means = [
                    mean_row([d for d in cond_details if (d["balance_train"], d["similarity"], d["classifier"]) == c], author_ids)
                    for c in dict.fromkeys((d["balance_train"], d["similarity"], d["classifier"]) for d in cond_details)
                ]
                mean_details.extend(means)
                for bal in (True, False):
                    head = [m for m in means if m["balance_train"] is bal]
                    if not head:
                        continue
                    best = max(head, key=lambda r: r["balanced_accuracy"])
                    print(
                        f"[{time.perf_counter() - t_start:7.1f}s] L={L:<5} {v.name:<22} {split_type:<5} {'bal  ' if bal else 'nobal'} "
                        f"n_seeds={head[0]['n_seeds']} best 作家別正解率の平均={best['balanced_accuracy']:.3f} ({best['similarity']}/{best['classifier']})  "
                        f"作家別 MAP の平均={head[0]['map_macro']:.4f}  ΔS_macro={head[0]['delta_s_macro']:.4f}  ({time.perf_counter() - t0:.1f}s)"
                    )

    details = [*seed_details, *mean_details]
    summary = [{k: v for k, v in d.items() if k not in ("confusion", "precision", "recall", "f1")} for d in details]

    write_csv(run_dir / "summary.csv", summary)
    write_csv(run_dir / "folds.csv", folds_out)
    write_csv(run_dir / "timings.csv", timings)
    (run_dir / "similarity_stats.json").write_text(json.dumps({k: v.to_dict() for k, v in sim_stats.items()}), encoding="utf-8")

    # 最良モデル：Inter-work の Pooled 指標のシード平均（本採用の特徴量設定のみ），同点は作家別 F 値の平均，MRR の順
    metric = cfg["report"]["best_model_metric"]
    candidates = [d for d in mean_details if d["split"] == "inter" and d["is_default"]]
    best = max(candidates, key=lambda d: (d[metric], d["macro_f1"], d["mrr"])) if candidates else None
    if best is not None:
        # 誤判別の抜粋は最良条件の先頭シード（間引きなしの Inter-work は決定的な 1 回）
        first_seed = next(d for d in seed_details if d["key"].rsplit("|", 1)[0] == best["key"].rsplit("|", 1)[0])
        write_misclassified(
            run_dir / "misclassified_best.md",
            first_seed,
            fold_store[first_seed["key"]],
            chunk_sets[best["chunk_chars"]],
            corpus,
            cfg["report"]["num_misclassified_examples"],
            cfg["report"]["excerpt_chars"],
        )
    (run_dir / "results.json").write_text(
        json.dumps({"best_key": best["key"] if best else None, "results": details}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["elapsed_seconds"] = round(time.perf_counter() - t_start, 1)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[output] {run_dir.relative_to(PROJECT_ROOT)}（{len(summary)} 条件，{meta['elapsed_seconds']}s）")
    if best is None:
        print("[info] Inter-work の結果がないため最良モデル・誤判別抜粋は未出力")

    if not args.no_report:
        from plot_results import make_report

        make_report(run_dir)


if __name__ == "__main__":
    main()
