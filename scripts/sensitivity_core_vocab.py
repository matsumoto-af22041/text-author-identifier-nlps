"""主結果（付属語 vs 内容語，B の語ブロック単独）の語彙選択に対する感度分析（plans/029）．

対象：Inter-work（LOWO）の各 fold，B の抽出器で `target_pos` のみが異なる 2 条件（report.core_comparison の
function_words / content_words，blocks は core_comparison.blocks の先頭 = words）．
それ以外の引数は configs/default.yaml の本採用 params．間引きは scripts/run_experiments.py と同じ
`balanced_inter_splits` （fold ごとに np.random.default_rng([seed, fold 番号])）．

条件（cond）と，各条件で fit（語彙選択）・検索対象（AP の並び）・重心に使うデータ：

    cond                 fit（語彙）              検索対象・重心          語彙の追加処理
    bal                  間引き後の Train         間引き後の Train        なし（主結果と同じ）
    nobal                fold の Train 全件       fold の Train 全件      なし（間引きなしと同じ，シードなし）
    fitfull_poolbal      fold の Train 全件       間引き後の Train        なし
    fitbal_poolfull      間引き後の Train         fold の Train 全件      なし
    drop_single_author   間引き後の Train         間引き後の Train        間引き後の Train で 1 作家にしか出ない語を除外

漏洩の有無：いずれの条件も fit・語彙選択・単一作家語の判定に使うのは fold の Train（全件またはその部分集合）
とその作家ラベルのみで，Test チャンクは transform（と類似度計算）を通るだけ．Test の作家・作品の情報は
語彙選択に使わない．028 の「Test 作家の Train 側作品にしか出ない語を除く」診断は fold の Test 作家を使う
ため実装しない．

drop_single_author の列除外：B の語ブロックは count(w, d) / 総形態素数(d) で，分母は語彙によらず IDF もないため，
語の除外は fit 済みの出力から列を削除することと同値．付属語にも同じ規則を適用した値を併記（統制）．
単一作家語の判定は Train チャンクでの出現（相対頻度 > 0）の有無を作家ごとに集計して行う．

評価（全 fold の Test チャンクをまとめた Pooled，シードごと）：
    作家別 AP（照会 = Test チャンク，適合 = 同一著者の検索対象チャンク，src.evaluator.average_precision）
    作家別 MAP の平均 map_macro とランダム基準 map_macro_random（src.evaluator.expected_random_ap）
    重心法の作家別正解率の平均 balanced_accuracy（重心は検索対象と同じデータで計算，同点は誤判別）
    差 = 付属語 − 内容語（シードごとに差を取ってから平均・標準偏差，対応のある比較）

再現の検証（不一致は即停止）：bal・nobal の作家別 AP・map_macro・map_macro_random・balanced_accuracy が
--baseline-run の summary.csv の同条件（Inter-work，centroid，row_type=seed のシードごとの行と row_type=mean の行）
と 1e-12 以内で一致．併せて特徴量の params とコーパスのハッシュが一致することを確認．

使い方：
    uv run python scripts/sensitivity_core_vocab.py --baseline-run full_006 --chunk-chars 500 --out-dir outputs/sensitivity_029_L500

出力（--out-dir．既存のディレクトリには書き込まず停止）：
    seed_metrics.csv    L × 語の種類 × 条件 × シードの指標（nobal はシードなし）
    summary.csv         L × 語の種類 × 条件のシード平均・標本標準偏差（ddof = 1）
    diff.csv            L × 条件の付属語 − 内容語（シードごとの差の平均・標準偏差）
    vocab_drop.csv      drop_single_author の fold × シードごとの語彙数と除外語数
    baseline_check.csv  bal・nobal と --baseline-run の照合結果
    report.md           上記の Markdown 表と漏洩がないことの説明
    run_meta.json       設定・CLI 引数・git 状態・コーパスハッシュ・所要時間
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from run_experiments import (
    INTER_BALANCE_RNG,
    PROJECT_ROOT,
    FeatureVariant,
    balanced_inter_splits,
    build_variants,
    corpus_meta,
    display_path,
    git_state,
    override_name,
)

from src.classifier import CentroidClassifier
from src.dataset import ChunkSet, Split, build_chunks, inter_work_splits, load_corpus
from src.evaluator import (
    average_precision,
    classification_metrics,
    confusion_matrix,
    expected_random_ap,
    macro_mean,
    per_author_mean,
)
from src.similarity import cosine_similarity

CONDITIONS = ("bal", "nobal", "fitfull_poolbal", "fitbal_poolfull", "drop_single_author")
# 各条件の fit（語彙）・検索対象と重心のデータ（report.md の漏洩の説明に使用）
CONDITION_DATA = {
    "bal": ("間引き後の Train", "間引き後の Train", "なし（主結果）"),
    "nobal": ("fold の Train 全件", "fold の Train 全件", "なし（間引きなし）"),
    "fitfull_poolbal": ("fold の Train 全件", "間引き後の Train", "なし"),
    "fitbal_poolfull": ("間引き後の Train", "fold の Train 全件", "なし"),
    "drop_single_author": ("間引き後の Train", "間引き後の Train", "間引き後の Train で 1 作家にしか出ない語を除外"),
}
ROLES = ("function", "content")
# 付属語 − 内容語の差に加えて併記する組（付属語は除外なし，内容語のみ除外．028 の報告値との照合用）
EXTRA_DIFFS = {"drop_single_author_content_only": ("bal", "drop_single_author")}
BASELINE_CONDS = {"bal": "True", "nobal": "False"}
BASELINE_TOL = 1e-12


@dataclass(frozen=True)
class QueryResult:
    """1 fold・1 条件の照会ごとの結果．配列の shape はすべて (n_test,)．"""

    y_test: np.ndarray
    ap: np.ndarray
    ap_random: np.ndarray
    pred: np.ndarray
    n_train: int
    n_features: int
    n_zero_query: int


# ---------------------------------------------------------------- 1 fold の評価


def evaluate(X_pool: np.ndarray, y_pool: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, num_classes: int, eps: float) -> QueryResult:
    """検索対象 X_pool (n_pool, M) に対する照会 X_test (n_test, M) の AP・E[AP] と，X_pool の重心による重心法の予測．"""
    assert X_pool.ndim == X_test.ndim == 2 and X_pool.shape[1] == X_test.shape[1] and y_pool.shape == (X_pool.shape[0],)
    assert np.isfinite(X_pool).all() and np.isfinite(X_test).all()
    similarity = lambda A, B: cosine_similarity(A, B, eps=eps)  # noqa: E731
    # shape: (n_test, n_pool)
    S = similarity(X_test, X_pool)
    ap = average_precision(S, y_test, y_pool)
    # shape: (n_test,)，照会と同一著者の検索対象チャンク数
    R = (y_test[:, None] == y_pool[None, :]).sum(axis=1)
    pred = CentroidClassifier(similarity, num_classes).fit(X_pool, y_pool).predict(X_test)
    return QueryResult(
        y_test=y_test,
        ap=ap,
        ap_random=expected_random_ap(y_pool.size, R),
        pred=pred,
        n_train=int(y_pool.size),
        n_features=int(X_pool.shape[1]),
        n_zero_query=int((np.linalg.norm(X_test, axis=1) == 0).sum()),
    )


def multi_author_columns(X_train: np.ndarray, y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Train で 2 作家以上のチャンクに出現する列のマスク．X_train shape: (n_train, M) → (M,)．Train のみを使用．"""
    # shape: (C, n_train)
    Y = (np.arange(num_classes)[:, None] == y_train[None, :]).astype(np.float64)
    # shape: (C, M)，作家 a のいずれかの Train チャンクに語が出現するか
    present = (Y @ (X_train > 0).astype(np.float64)) > 0
    return present.sum(axis=0) >= 2


def fold_results(
    variant: FeatureVariant,
    chunk_set: ChunkSet,
    split: Split,
    balanced: dict[int, Split],
    eps: float,
) -> tuple[list[tuple[str, int | None, QueryResult]], list[dict[str, Any]]]:
    """1 fold の全条件・全シードの結果と，drop_single_author の語彙の記録．

    抽出器の fit は fold の Train 全件（1 回）と各シードの間引き後の Train（シードごとに 1 回）のみ．
    """
    texts, labels = chunk_set.texts, chunk_set.labels
    C = len(chunk_set.author_ids)
    take = lambda idx: [texts[j] for j in idx]  # noqa: E731
    test_texts, y_test = take(split.test_idx), labels[split.test_idx]
    full_texts, y_full = take(split.train_idx), labels[split.train_idx]

    ex_full = variant.make().fit(full_texts)
    # shape: (n_full, M_full), (n_test, M_full)
    X_full, X_test_full = ex_full.transform(full_texts), ex_full.transform(test_texts)
    results: list[tuple[str, int | None, QueryResult]] = [("nobal", None, evaluate(X_full, y_full, X_test_full, y_test, C, eps))]
    vocab_rows: list[dict[str, Any]] = []
    for seed, bs in balanced.items():
        assert np.array_equal(bs.test_idx, split.test_idx)
        # 間引き後の Train が fold の Train 全件の中で占める行位置，shape: (n_bal,)
        rows = np.flatnonzero(np.isin(split.train_idx, bs.train_idx))
        assert np.array_equal(split.train_idx[rows], bs.train_idx)
        bal_texts, y_bal = take(bs.train_idx), labels[bs.train_idx]
        ex_bal = variant.make().fit(bal_texts)
        # shape: (n_bal, M_bal), (n_test, M_bal), (n_full, M_bal)
        X_bal, X_test_bal, X_full_bal = ex_bal.transform(bal_texts), ex_bal.transform(test_texts), ex_bal.transform(full_texts)
        # shape: (M_bal,)，間引き後の Train のみから判定
        keep = multi_author_columns(X_bal, y_bal, C)
        results += [
            ("bal", seed, evaluate(X_bal, y_bal, X_test_bal, y_test, C, eps)),
            ("fitfull_poolbal", seed, evaluate(X_full[rows], y_bal, X_test_full, y_test, C, eps)),
            ("fitbal_poolfull", seed, evaluate(X_full_bal, y_full, X_test_bal, y_test, C, eps)),
            ("drop_single_author", seed, evaluate(X_bal[:, keep], y_bal, X_test_bal[:, keep], y_test, C, eps)),
        ]
        vocab_rows.append(
            {"fold": split.name, "seed": seed, "n_features": int(keep.size), "n_dropped": int((~keep).sum()), "n_kept": int(keep.sum())}
        )
    return results, vocab_rows


# ---------------------------------------------------------------- 集計


def metric_columns(author_ids: Sequence[str]) -> list[str]:
    return ["map_macro", "map_macro_random", *(f"author_ap_{a}" for a in author_ids), "balanced_accuracy", *(f"author_acc_{a}" for a in author_ids)]


def pooled_metrics(parts: Sequence[QueryResult], author_ids: Sequence[str]) -> dict[str, Any]:
    """全 fold の Test チャンクをまとめた（Pooled）指標．"""
    C = len(author_ids)
    y = np.concatenate([p.y_test for p in parts])
    ap, ap_random, pred = (np.concatenate([getattr(p, k) for p in parts]) for k in ("ap", "ap_random", "pred"))
    ap_by_author = per_author_mean(ap, y, C)
    cm = classification_metrics(confusion_matrix(y, pred, C))
    return {
        "n_test": int(y.size),
        "n_train_mean": float(np.mean([p.n_train for p in parts])),
        "n_features_mean": float(np.mean([p.n_features for p in parts])),
        "n_zero_query": int(sum(p.n_zero_query for p in parts)),
        "n_ties": cm["n_ties"],
        "map_macro": macro_mean(ap_by_author),
        "map_macro_random": macro_mean(per_author_mean(ap_random, y, C)),
        **{f"author_ap_{a}": float(v) for a, v in zip(author_ids, ap_by_author, strict=True)},
        "balanced_accuracy": cm["balanced_accuracy"],
        **{f"author_acc_{a}": float(v) for a, v in zip(author_ids, cm["recall"], strict=True)},
    }


def mean_std(rows: Sequence[dict[str, Any]], cols: Sequence[str]) -> dict[str, float]:
    """シード行の平均と標本標準偏差（ddof = 1，1 行のみなら NaN）．"""
    # shape: (n_seeds, n_cols)
    M = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    std = M.std(axis=0, ddof=1) if len(rows) > 1 else np.full(len(cols), np.nan)
    return {**dict(zip(cols, M.mean(axis=0).tolist(), strict=True)), **{f"{c}_std": float(s) for c, s in zip(cols, std, strict=True)}}


def summarize(seed_rows: Sequence[dict[str, Any]], author_ids: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """L × 語の種類 × 条件のシード平均と，L × 条件の付属語 − 内容語（シードごとの差の平均）．"""
    cols = metric_columns(author_ids)
    groups = sorted({(r["chunk_chars"], r["role"], r["cond"]) for r in seed_rows}, key=lambda g: (g[0], ROLES.index(g[1]), CONDITIONS.index(g[2])))
    pick = lambda L, role, cond: sorted((r for r in seed_rows if (r["chunk_chars"], r["role"], r["cond"]) == (L, role, cond)), key=lambda r: r["seed"] or -1)  # noqa: E731
    summary = [
        {"chunk_chars": L, "role": role, "cond": cond, "n_seeds": len(rs := pick(L, role, cond)), **mean_std(rs, cols)} for L, role, cond in groups
    ]

    def diff_row(L: int, name: str, f_cond: str, c_cond: str) -> dict[str, Any]:
        f_rows, c_rows = pick(L, "function", f_cond), pick(L, "content", c_cond)
        if [r["seed"] for r in f_rows] != [r["seed"] for r in c_rows]:
            raise AssertionError(f"付属語と内容語のシードが不一致: L={L} {name}")
        diffs = [{c: f[c] - g[c] for c in cols} for f, g in zip(f_rows, c_rows, strict=True)]
        return {"chunk_chars": L, "cond": name, "function_cond": f_cond, "content_cond": c_cond, "n_seeds": len(diffs), **mean_std(diffs, cols)}

    Ls = sorted({r["chunk_chars"] for r in seed_rows})
    diffs = [diff_row(L, cond, cond, cond) for L in Ls for cond in CONDITIONS] + [
        diff_row(L, name, f, c) for L in Ls for name, (f, c) in EXTRA_DIFFS.items()
    ]
    return summary, diffs


# ---------------------------------------------------------------- 照合


def load_baseline(outputs_dir: Path, name: str) -> tuple[dict[str, Any], list[dict[str, str]]]:
    run_dir = outputs_dir / name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--baseline-run が存在しない: {run_dir}")
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    with (run_dir / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return meta, rows


def baseline_checks(
    seed_rows: Sequence[dict[str, Any]],
    summary: Sequence[dict[str, Any]],
    variants: dict[str, FeatureVariant],
    baseline_rows: Sequence[dict[str, str]],
    author_ids: Sequence[str],
    seeds: Sequence[int],
    baseline_seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """bal・nobal の値と --baseline-run の summary.csv（Inter-work，centroid）を照合．許容誤差を超えたら停止．"""
    cols = ["map_macro", "map_macro_random", *(f"author_ap_{a}" for a in author_ids), "balanced_accuracy"]

    def ref_row(variant: FeatureVariant, L: int, cond: str, row_type: str, seed: int | None) -> dict[str, str]:
        hits = [
            r
            for r in baseline_rows
            if r["variant"] == variant.name and r["chunk_chars"] == str(L) and r["split"] == "inter" and r["classifier"] == "centroid"
            and r["balance_train"] == BASELINE_CONDS[cond] and r["row_type"] == row_type and r["seed"] == ("" if seed is None else str(seed))
        ]
        if len(hits) != 1:
            raise KeyError(f"--baseline-run に対応する行が {len(hits)} 件: {variant.name} L={L} {cond} {row_type} seed={seed}")
        if json.loads(hits[0]["params"]) != variant.params:
            raise ValueError(f"特徴量の params が --baseline-run と不一致: {variant.name} {hits[0]['params']} vs {variant.params}")
        return hits[0]

    targets = [(r, "seed", r["seed"]) for r in seed_rows if r["cond"] in BASELINE_CONDS]
    # シード平均の照合は --seeds が --baseline-run と同じ場合のみ（nobal は常に 1 回）
    targets += [
        (s, "mean", None) for s in summary if s["cond"] == "nobal" or (s["cond"] == "bal" and list(seeds) == list(baseline_seeds))
    ]
    checks = [
        {
            "chunk_chars": ours["chunk_chars"], "role": ours["role"], "cond": ours["cond"], "row_type": row_type,
            "seed": "" if seed is None else seed, "metric": c, "ours": float(ours[c]), "baseline": float(ref[c]),
            "abs_diff": abs(float(ours[c]) - float(ref[c])),
        }
        for ours, row_type, seed in targets
        for ref in [ref_row(variants[ours["role"]], ours["chunk_chars"], ours["cond"], row_type, seed)]
        for c in cols
    ]
    bad = [c for c in checks if not c["abs_diff"] <= BASELINE_TOL]
    if bad:
        raise AssertionError(f"--baseline-run と不一致（許容誤差 {BASELINE_TOL}）: {bad[:5]}（全 {len(bad)} 件）")
    return checks


# ---------------------------------------------------------------- 出力


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt_ms(row: dict[str, Any], col: str, signed: bool = False) -> str:
    v, s = row[col], row[f"{col}_std"]
    head = f"{v:+.4f}" if signed else f"{v:.4f}"
    return head if np.isnan(s) else f"{head} ± {s:.4f}"


def md_table(header: Sequence[str], body: Sequence[Sequence[str]], n_text: int = 1) -> list[str]:
    """先頭 n_text 列を左寄せ，残りを右寄せにした Markdown 表．"""
    return ["| " + " | ".join(header) + " |", "| " + " | ".join(":---" if i < n_text else "---:" for i in range(len(header))) + " |", *("| " + " | ".join(r) + " |" for r in body), ""]


def report_lines(
    summary: Sequence[dict[str, Any]],
    diffs: Sequence[dict[str, Any]],
    vocab_rows: Sequence[dict[str, Any]],
    checks: Sequence[dict[str, Any]],
    variants: dict[str, FeatureVariant],
    author_ids: Sequence[str],
    meta: dict[str, Any],
) -> list[str]:
    get = lambda L, role, cond: next(s for s in summary if (s["chunk_chars"], s["role"], s["cond"]) == (L, role, cond))  # noqa: E731
    getd = lambda L, cond: next(d for d in diffs if (d["chunk_chars"], d["cond"]) == (L, cond))  # noqa: E731
    lines = [
        "# 主結果の語彙選択に対する感度分析（plans/029）",
        "",
        f"- 照合 run：`{meta['cli']['baseline_run']}`，シード：{meta['cli']['seeds']}，間引きの乱数：`{INTER_BALANCE_RNG}`",
        f"- 付属語：`{variants['function'].name}`，内容語：`{variants['content'].name}`",
        "- 分割：Inter-work（LOWO，9 fold），評価は全 fold の Test チャンクをまとめた Pooled",
        "- 値：5 シードの平均 ± 標本標準偏差（`nobal` はシードによらず 1 回）．差はシードごとに付属語 − 内容語を求めてから平均",
        "",
        "## 各条件で使うデータ（漏洩がないことの説明）",
        "",
        (
            "fit（語彙選択）・単一作家語の判定に使うのは，いずれの条件も fold の Train（全件またはその間引き後の部分集合）とその作家ラベルのみ．"
            "Test チャンクは fit 済みの抽出器の transform と類似度計算を通るだけで，Test の作家・作品の情報は語彙選択に使わない．"
            "028 の「Test 作家の Train 側作品にしか出ない語を除く」診断は Test の作家を使うため実装しない．"
        ),
        "",
        *md_table(["条件", "fit（語彙）", "検索対象・重心", "語彙の追加処理"], [[f"`{c}`", *CONDITION_DATA[c]] for c in CONDITIONS], n_text=4),
        "B の語ブロックは count(w, d) / 総形態素数(d) で分母が語彙によらず IDF もないため，`drop_single_author` の列除外は語彙から除いて fit し直すことと同値．",
        "",
    ]
    for L in sorted({s["chunk_chars"] for s in summary}):
        lines += [
            f"## L = {L}",
            "",
            "### 作家別 MAP の平均と重心法の作家別正解率の平均",
            "",
            *md_table(
                ["条件", "付属語 MAP", "内容語 MAP", "差（付属語 − 内容語）", "基準値 E[AP]", "付属語 作家別正解率", "内容語 作家別正解率", "差"],
                [
                    [
                        f"`{c}`", fmt_ms(get(L, "function", c), "map_macro"), fmt_ms(get(L, "content", c), "map_macro"),
                        fmt_ms(getd(L, c), "map_macro", signed=True), f"{get(L, 'function', c)['map_macro_random']:.4f}",
                        fmt_ms(get(L, "function", c), "balanced_accuracy"), fmt_ms(get(L, "content", c), "balanced_accuracy"),
                        fmt_ms(getd(L, c), "balanced_accuracy", signed=True),
                    ]
                    for c in CONDITIONS
                ],
            ),
            (
                "参考（付属語は除外なし `bal` ，内容語のみ `drop_single_author` ）：差 "
                f"{fmt_ms(getd(L, 'drop_single_author_content_only'), 'map_macro', signed=True)}．"
            ),
            "",
            "### 作家別 AP（平均）",
            "",
            *md_table(
                ["条件", "語の種類", *author_ids],
                [[f"`{c}`", role, *(f"{get(L, role, c)[f'author_ap_{a}']:.4f}" for a in author_ids)] for c in CONDITIONS for role in ROLES],
                n_text=2,
            ),
            "### `drop_single_author` の除外語数（fold ごと，シード平均）",
            "",
            *md_table(
                ["fold", *(f"{role} 除外 / 語彙" for role in ROLES)],
                [
                    [
                        fold,
                        *(
                            f"{np.mean([v['n_dropped'] for v in vs]):.1f} / {np.mean([v['n_features'] for v in vs]):.1f}"
                            for role in ROLES
                            for vs in [[v for v in vocab_rows if (v["chunk_chars"], v["role"], v["fold"]) == (L, role, fold)]]
                        ),
                    ]
                    for fold in dict.fromkeys(v["fold"] for v in vocab_rows if v["chunk_chars"] == L)
                ],
            ),
        ]
    lines += [
        "## 照合（`bal` ・ `nobal` vs 照合 run の summary.csv）",
        "",
        f"- 照合した値：{len(checks)} 件，最大絶対誤差 {max(c['abs_diff'] for c in checks):.2e}（許容 {BASELINE_TOL:.0e}）",
        "",
    ]
    return lines


# ---------------------------------------------------------------- 本体


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--baseline-run", required=True, help="照合する run 名（experiment.output_dir 配下，例：full_006）")
    parser.add_argument("--chunk-chars", nargs="+", type=int, help="チャンク長 L（既定：chunking.target_chunk_chars）")
    parser.add_argument("--seeds", nargs="+", type=int, help="間引きシード（既定：split.balance_seeds）")
    parser.add_argument("--out-dir", type=Path, required=True, help="出力先（相対パスはプロジェクトルート基準．既存なら停止）")
    args = parser.parse_args()

    t_start = time.perf_counter()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chunk_chars = args.chunk_chars or list(cfg["chunking"]["target_chunk_chars"])
    seeds = args.seeds or list(cfg["split"]["balance_seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds に重複: {seeds}")
    eps = cfg["similarity"]["eps"]
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"出力先が既に存在（上書き防止）: {out_dir}")

    # 付属語・内容語の 2 条件（config の report.core_comparison と features.B.all_chunk_chars から取得）
    core = cfg["report"]["core_comparison"]
    word_block = core["blocks"][0]
    names = {
        "function": override_name(core["feature"], {"blocks": word_block}),
        "content": override_name(core["feature"], {core["param"]: core["content_words"], "blocks": word_block}),
    }
    by_name = {v.name: v for v in build_variants(cfg, [core["feature"]], with_ablation=False, with_all_chunk_chars=True)}
    missing = [n for n in names.values() if n not in by_name]
    if missing:
        raise KeyError(f"features.{core['feature']}.all_chunk_chars に無い条件: {missing}")
    variants = {role: by_name[n] for role, n in names.items()}
    if variants["function"].params[core["param"]] != core["function_words"]:
        raise ValueError(f"付属語条件の {core['param']} が report.core_comparison.function_words と不一致: {variants['function'].params}")

    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    corpus = load_corpus(corpus_path)
    author_ids = list(corpus.author_ids)
    outputs_dir = PROJECT_ROOT / cfg["experiment"]["output_dir"]
    base_meta, base_rows = load_baseline(outputs_dir, args.baseline_run)
    if base_meta["corpus"]["source_sha256"] != corpus_meta(corpus_path)["source_sha256"]:
        raise ValueError("コーパスのハッシュが --baseline-run と不一致")
    if base_meta.get("cli", {}).get("exclude_works"):
        raise ValueError(f"--baseline-run は作品除外の run: {base_meta['cli']['exclude_works']}")
    baseline_seeds = list(base_meta["balance_train"]["inter"]["seeds"])
    if not set(seeds) <= set(baseline_seeds):
        raise ValueError(f"--seeds {seeds} が --baseline-run の間引きシード {baseline_seeds} に含まれない")
    if base_meta["author_ids"] != author_ids:
        raise ValueError(f"作家 ID が --baseline-run と不一致: {base_meta['author_ids']} vs {author_ids}")

    seed_rows: list[dict[str, Any]] = []
    vocab_rows: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    for L in chunk_chars:
        chunk_set = build_chunks(corpus, L)
        splits = inter_work_splits(chunk_set)
        # balanced_inter_splits の返す Split 列は inter_work_splits と同じ fold 順
        balanced = {s: balanced_inter_splits(chunk_set, s)[0] for s in seeds}
        assert all([b.name for b in bs] == [s.name for s in splits] for bs in balanced.values())
        for role, variant in variants.items():
            t0 = time.perf_counter()
            per_fold = [fold_results(variant, chunk_set, sp, {s: balanced[s][i] for s in seeds}, eps) for i, sp in enumerate(splits)]
            results = [r for res, _ in per_fold for r in res]
            vocab_rows += [{"chunk_chars": L, "role": role, **v} for _, vs in per_fold for v in vs]
            keys = [("nobal", None), *((c, s) for c in CONDITIONS if c != "nobal" for s in seeds)]
            for cond, seed in keys:
                parts = [q for c, s, q in results if (c, s) == (cond, seed)]
                assert len(parts) == len(splits), f"fold 数が不一致: {cond} {seed} {len(parts)}"
                seed_rows.append({"chunk_chars": L, "role": role, "variant": variant.name, "cond": cond, "seed": seed, **pooled_metrics(parts, author_ids)})
            timings.append({"chunk_chars": L, "role": role, "seconds": time.perf_counter() - t0})
            print(f"[info] L={L} {role} 完了（{timings[-1]['seconds']:.0f}s）", flush=True)

    summary, diffs = summarize(seed_rows, author_ids)
    checks = baseline_checks(seed_rows, summary, variants, base_rows, author_ids, seeds, baseline_seeds)
    print(f"[info] --baseline-run と照合：{len(checks)} 件，最大絶対誤差 {max(c['abs_diff'] for c in checks):.2e}")

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": display_path(args.config),
        "config": cfg,
        "cli": {"baseline_run": args.baseline_run, "chunk_chars": chunk_chars, "seeds": seeds, "out_dir": display_path(out_dir)},
        "variants": {role: {"name": v.name, "params": v.params} for role, v in variants.items()},
        "conditions": {c: dict(zip(("fit", "pool_and_centroid", "vocab_postprocess"), CONDITION_DATA[c], strict=True)) for c in CONDITIONS},
        "balance_rng": INTER_BALANCE_RNG,
        "baseline": {"run": args.baseline_run, "git": base_meta["git"], "seeds": baseline_seeds, "tolerance": BASELINE_TOL},
        "git": git_state(),
        "corpus": corpus_meta(corpus_path),
        "author_ids": author_ids,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "timings": timings,
    }
    if meta["git"]["dirty"]:
        print(f"[warn] 未コミットの変更あり（run_meta.json に記録）: {meta['git']['dirty_files']}")
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "seed_metrics.csv", [{**r, "seed": "" if r["seed"] is None else r["seed"]} for r in seed_rows])
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "diff.csv", diffs)
    write_csv(out_dir / "vocab_drop.csv", vocab_rows)
    write_csv(out_dir / "baseline_check.csv", checks)
    (out_dir / "report.md").write_text("\n".join(report_lines(summary, diffs, vocab_rows, checks, variants, author_ids, meta)), encoding="utf-8")
    meta["elapsed_seconds"] = time.perf_counter() - t_start
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[info] 出力：{display_path(out_dir)}（{meta['elapsed_seconds']:.0f}s）")


if __name__ == "__main__":
    main()
