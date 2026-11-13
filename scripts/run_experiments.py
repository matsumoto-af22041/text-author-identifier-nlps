"""全体統合実験：特徴量 × チャンク長 × 分割方式 × 類似度 × 推論方式の評価と結果出力．

使い方：
    uv run python scripts/run_experiments.py --config configs/default.yaml
    uv run python scripts/run_experiments.py --features A --chunk-chars 500 --ablation

出力（outputs/<run_name>/）：
    run_meta.json          実行時の設定・CLI 引数・git 状態・コーパスハッシュ
    folds.csv              fold 単位の結果（Inter-work は 9 fold）
    summary.csv            条件単位の結果（Inter-work は全 fold の Pooled Out-of-Fold）
    results.json           summary と混同行列（plot_results.py の入力）
    similarity_stats.json  類似度統計（作家間ペア類似度和・ペア数・ヒストグラム）
    misclassified_best.md  最良モデルの誤判別チャンク抜粋
    report.md, figures/    plot_results.py による図表
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import platform
import random
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.dataset import ChunkSet, Corpus, Split, build_chunks, inter_work_splits, intra_work_split, load_corpus
from src.evaluator import ClassifierSpec, FoldResult, aggregate_results, evaluate_fold
from src.features.base import BaseFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FeatureVariant:
    """特徴量設定（本採用またはアブレーションで 1 引数のみ差し替えたもの）．"""

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
        return [default, *(ablations if with_ablation else [])]

    return [v for key in keys for v in variants_of(key)]


def classifier_specs(cfg: dict[str, Any]) -> list[ClassifierSpec]:
    c = cfg["classifier"]
    unknown = set(c["methods"]) - {"centroid", "knn"}
    if unknown:
        raise ValueError(f"未知の推論方式: {unknown}")
    centroid = [ClassifierSpec("centroid")] if "centroid" in c["methods"] else []
    knn = [ClassifierSpec("knn", k, m) for m in c["knn_modes"] for k in c["k_values"]] if "knn" in c["methods"] else []
    return [*centroid, *knn]


def similarities_for(cfg: dict[str, Any], chunk_chars: int, non_negative: bool) -> list[str]:
    exp = cfg["experiment"]
    restricted = set(exp["restricted_similarities"])
    names = [
        s
        for s in cfg["similarity"]["names"]
        if (s not in restricted or chunk_chars in exp["restricted_chunk_chars"])
    ]
    # 非負でない特徴量（LSI）はコサイン類似度のみ
    return names if non_negative else [s for s in names if s == "cosine"]


def make_splits(chunk_set: ChunkSet, split_type: str, cfg: dict[str, Any]) -> list[Split]:
    match split_type:
        case "intra":
            return [intra_work_split(chunk_set, cfg["split"]["intra_test_ratio"], cfg["seed"])]
        case "inter":
            return inter_work_splits(chunk_set)
        case _:
            raise ValueError(f"未知の分割方式: {split_type}")


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


def condition_key(variant: str, L: int, split_type: str, similarity: str, classifier: str | None = None) -> str:
    parts = [variant, f"L{L}", split_type, similarity, *([classifier] if classifier else [])]
    return "|".join(parts)


def summary_row(
    variant: FeatureVariant, L: int, split_type: str, folds: Sequence[FoldResult], seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    agg = aggregate_results(folds)
    r0 = folds[0]
    row = {
        "key": condition_key(variant.name, L, split_type, r0.similarity, r0.spec.name),
        "variant": variant.name,
        "feature": variant.feature,
        "is_default": variant.is_default,
        "params": json.dumps(variant.params, ensure_ascii=False, sort_keys=True),
        "chunk_chars": L,
        "split": split_type,
        "similarity": r0.similarity,
        "classifier": r0.spec.name,
        "method": r0.spec.method,
        "k": r0.spec.k,
        "knn_mode": r0.spec.knn_mode,
        "seed": seed,
        "num_folds": agg.num_folds,
        "n_test": agg.n_test,
        "n_train_mean": float(np.mean([r.n_train for r in folds])),
        "n_features": int(np.mean([r.n_features for r in folds])),
        "accuracy": agg.metrics["accuracy"],
        "macro_f1": agg.metrics["macro_f1"],
        "mrr": agg.mrr,
        **agg.sim_stats.scalars(),
        "extract_seconds": float(sum(r.extract_seconds for r in folds)),
        "similarity_seconds": float(sum(r.similarity_seconds for r in folds)),
    }
    detail = {**row, "confusion": agg.confusion.tolist(), **{k: agg.metrics.get(k) for k in ("precision", "recall", "f1")}}
    return row, detail


def fold_rows(variant: FeatureVariant, L: int, split_type: str, folds: Sequence[FoldResult]) -> list[dict[str, Any]]:
    return [
        {
            "variant": variant.name,
            "chunk_chars": L,
            "split": split_type,
            "fold": r.split_name,
            "similarity": r.similarity,
            "classifier": r.spec.name,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "n_features": r.n_features,
            "accuracy": r.metrics["accuracy"],
            "macro_f1": r.metrics["macro_f1"],
            "mrr": r.mrr,
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
    """最良モデルの誤判別を確信度の高い順（予測作家スコア − 正解作家スコアの降順）に抜粋．"""
    titles = {w.work_id: w.title for w in corpus.works}
    authors = corpus.author_names
    # shape: (n_test_total,), (n_test_total, C)
    idx = np.concatenate([r.test_idx for r in folds])
    y_true = np.concatenate([r.y_true for r in folds])
    y_pred = np.concatenate([r.y_pred for r in folds])
    scores = np.concatenate([r.scores for r in folds])
    margin = np.take_along_axis(scores, y_pred[:, None], 1)[:, 0] - np.take_along_axis(scores, y_true[:, None], 1)[:, 0]
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
        f"- Pooled 指標：Accuracy {best['accuracy']:.4f}，マクロ F1 {best['macro_f1']:.4f}，MRR {best['mrr']:.4f}",
        f"- 誤判別数：{wrong.size} / {y_true.size}",
        "",
        "## 誤判別の内訳（正解 → 予測）",
        "",
        "| 正解 | 予測 | 件数 |",
        "| :--- | :--- | ---: |",
        *[f"| {t} | {p} | {n} |" for (t, p), n in pair_counts.most_common()],
        "",
        f"## 確信度の高い誤判別（上位 {order.size} 件，margin = 予測作家スコア − 正解作家スコア）",
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
    if args.ablation and not set(chunk_chars) & set(ablation_L):
        raise ValueError(f"--ablation 指定時は --chunk-chars に experiment.ablation_chunk_chars {ablation_L} のいずれかを含めること")
    # 長時間の実行前に全特徴量クラスの読み込みを検証（Fail-Fast）
    variants = build_variants(cfg, feature_keys, args.ablation)
    specs = classifier_specs(cfg)
    sim_kwargs = {"eps": cfg["similarity"]["eps"], "max_batch_bytes": cfg["similarity"]["max_batch_bytes"]}
    num_bins = cfg["evaluation"]["similarity_hist_bins"]

    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    corpus = load_corpus(corpus_path)
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
        },
        "variants": [{"name": v.name, "class": f"{v.cls.__module__}.{v.cls.__name__}", "params": v.params} for v in variants],
        "classifiers": [s.name for s in specs],
        "git": git_state(),
        "corpus": corpus_meta(corpus_path),
        "author_ids": list(corpus.author_ids),
        "author_names": list(corpus.author_names),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if meta["git"]["dirty"]:
        print(f"[warn] 未コミットの変更あり（run_meta.json に記録）: {meta['git']['dirty_files']}")
    (run_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    summary: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    folds_out: list[dict[str, Any]] = []
    sim_stats: dict[str, Any] = {}
    fold_store: dict[str, list[FoldResult]] = {}
    chunk_sets: dict[int, ChunkSet] = {}
    t_start = time.perf_counter()

    for L in chunk_chars:
        chunk_sets[L] = cs = build_chunks(corpus, L)
        active = [v for v in variants if v.is_default or L in cfg["experiment"]["ablation_chunk_chars"]]
        for v in active:
            sims = similarities_for(cfg, L, v.cls.non_negative)
            for split_type in split_types:
                t0 = time.perf_counter()
                results = [
                    r
                    for s in make_splits(cs, split_type, cfg)
                    for r in evaluate_fold(v.make, cs, s, sims, specs, similarity_kwargs=sim_kwargs, num_bins=num_bins)
                ]
                # 類似度 × 推論器ごとに fold をまとめる（results は fold → 類似度 → 推論器の順）
                conditions = list(dict.fromkeys((r.similarity, r.spec) for r in results))
                grouped = {c: [r for r in results if (r.similarity, r.spec) == c] for c in conditions}
                for (sim, _), folds in grouped.items():
                    row, detail = summary_row(v, L, split_type, folds, seed)
                    summary.append(row)
                    details.append(detail)
                    fold_store[row["key"]] = folds
                    sim_stats.setdefault(condition_key(v.name, L, split_type, sim), aggregate_results(folds).sim_stats.to_dict())
                folds_out.extend(fold_rows(v, L, split_type, results))
                head = {(r["similarity"], r["classifier"]): r for r in summary if r["key"].startswith(f"{v.name}|L{L}|{split_type}|")}
                best = max(head.values(), key=lambda r: r["macro_f1"])
                print(
                    f"[{time.perf_counter() - t_start:7.1f}s] L={L:<5} {v.name:<22} {split_type:<5} sims={sims} "
                    f"({time.perf_counter() - t0:.1f}s)  best F1={best['macro_f1']:.3f} ({best['similarity']}/{best['classifier']})  "
                    f"ΔS_macro(cos)={head[('cosine', specs[0].name)]['delta_s_macro']:.4f}"
                )

    write_csv(run_dir / "summary.csv", summary)
    write_csv(run_dir / "folds.csv", folds_out)
    (run_dir / "similarity_stats.json").write_text(json.dumps(sim_stats), encoding="utf-8")

    # 最良モデル：Inter-work の Pooled 指標（本採用の特徴量設定のみ），同点は Accuracy，MRR の順
    metric = cfg["report"]["best_model_metric"]
    candidates = [d for d in details if d["split"] == "inter" and d["is_default"]]
    best = max(candidates, key=lambda d: (d[metric], d["accuracy"], d["mrr"])) if candidates else None
    if best is not None:
        write_misclassified(
            run_dir / "misclassified_best.md",
            best,
            fold_store[best["key"]],
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
