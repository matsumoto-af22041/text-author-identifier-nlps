"""特徴量 D の潜在次元数 K の推移図（Slide 12 のグラフ D）．

使い方：
    uv run python scripts/plot_d_k_sweep.py --run-dir outputs/D_ablation [--chunk-chars 500] [--classifier centroid]

入力：
    <run-dir>/summary.csv  `run_experiments.py --features D --ablation` の出力（K の各値と本採用 K の行を含む）
    data/processed/authors_corpus.json  累積寄与率の算出用（Intra-work の間引きあり Train を再現して fit）

出力（<run-dir>/figures/）：
    d_k_sweep.png  (a) 累積寄与率，(b) 作家別正解率の平均，(c) 作家別 MAP の平均（Intra / Inter，間引きあり）
    d_k_sweep.csv  図の数値（シード平均と標本標準偏差）

K の比較は類似度の値の分布に依存しない指標（判別結果のみに依存する作家別正解率，類似度の順位のみに依存する
作家別 MAP）で行い，ΔS は用いない（K により類似度の値の分布が変わるため．research-context の ΔS の使用範囲）．
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.run_experiments import intra_balance_rng
from src.dataset import balance_train_split, build_chunks, intra_work_split, load_corpus
from src.features.lsi_embedding import LSIEmbeddingExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_K = re.compile(r"^D\[n_components=(\d+)\]$")
SPLITS = ("intra", "inter")


def load_rows(summary_csv: Path, chunk_chars: int, classifier: str, default_k: int) -> dict[tuple[str, int], dict[str, Any]]:
    """(split, K) -> 間引きありのシード平均行．本採用（variant == "D"）は default_k として扱う．"""
    with summary_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def k_of(variant: str) -> int | None:
        m = VARIANT_K.match(variant)
        return int(m.group(1)) if m else (default_k if variant == "D" else None)

    picked = {
        (r["split"], k): r
        for r in rows
        if r["row_type"] == "mean"
        and r["balance_train"] == "True"
        and r["classifier"] == classifier
        and int(r["chunk_chars"]) == chunk_chars
        and (k := k_of(r["variant"])) is not None
    }
    if not picked:
        raise ValueError(f"{summary_csv}: L={chunk_chars}, classifier={classifier} の D の行が無い")
    return picked


def explained_ratio(cfg: dict[str, Any], chunk_chars: int, k_values: list[int]) -> np.ndarray:
    """Intra-work の各シード（間引きあり Train，run_experiments.py と同じ乱数）での累積寄与率．shape: (n_seeds, |K|)"""
    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    chunk_set = build_chunks(corpus, chunk_chars)
    texts, labels = chunk_set.texts, chunk_set.labels

    def ratio(seed: int) -> np.ndarray:
        split = intra_work_split(chunk_set, cfg["split"]["intra_test_ratio"], seed)
        split = balance_train_split(split, labels, intra_balance_rng(seed))
        ext = LSIEmbeddingExtractor(n_components=max(k_values)).fit([texts[i] for i in split.train_idx])
        return ext.cumulative_explained_ratio_[np.array(k_values) - 1]

    return np.stack([ratio(s) for s in cfg["split"]["intra_seeds"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chunk-chars", type=int, default=None, help="既定：config の experiment.ablation_chunk_chars の先頭")
    parser.add_argument("--classifier", default="centroid", help="作家別正解率に用いる推論方式（summary.csv の classifier 列）")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    L = args.chunk_chars or cfg["experiment"]["ablation_chunk_chars"][0]
    spec = cfg["features"]["D"]
    default_k = spec["params"]["n_components"]
    k_values = sorted({default_k, *spec["ablations"]["n_components"]})

    rows = load_rows(args.run_dir / "summary.csv", L, args.classifier, default_k)
    missing = [(s, k) for s in SPLITS for k in k_values if (s, k) not in rows]
    if missing:
        raise ValueError(f"summary.csv に無い条件: {missing}")

    def series(split: str, col: str) -> np.ndarray:
        return np.array([float(rows[(split, k)][col]) for k in k_values])

    ratios = explained_ratio(cfg, L, k_values)  # shape: (n_seeds, |K|)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(len(k_values))
    axes[0].errorbar(x, ratios.mean(axis=0), yerr=ratios.std(axis=0, ddof=1), marker="o", capsize=3, color="tab:gray")
    axes[0].set_ylabel("Cumulative explained ratio")
    axes[0].set_title(rf"(a) $\sum_{{j \leq K}} \sigma_j^2 \,/\, \|X\|_F^2$  (intra Train, L={L})")

    panels = [
        (axes[1], "balanced_accuracy", "Per-author accuracy (mean)", f"(b) accuracy / {args.classifier}"),
        (axes[2], "map_macro", "Per-author MAP (mean)", "(c) MAP"),
    ]
    for ax, col, ylabel, title in panels:
        for split, color in zip(SPLITS, ("tab:blue", "tab:orange"), strict=True):
            ax.errorbar(x, series(split, col), yerr=series(split, f"{col}_std"), marker="o", capsize=3, color=color, label=split)
            ax.plot(x, series(split, f"{col}_random"), linestyle=":", color=color, label=f"{split} random")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x, [f"K={k}" + (" *" if k == default_k else "") for k in k_values])
        ax.grid(alpha=0.3)
    fig.suptitle(f"D (LSI): K sweep, L={L}, balanced Train, mean ± std over seeds  (* = default)", fontsize=10)
    fig.tight_layout()

    fig_dir = args.run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "d_k_sweep.png", dpi=200)

    with (fig_dir / "d_k_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "explained_ratio", "explained_ratio_std"]
                   + [f"{s}_{c}" for s in SPLITS for c in ("balanced_accuracy", "balanced_accuracy_std", "map_macro", "map_macro_std", "map_macro_random")])
        for i, k in enumerate(k_values):
            w.writerow([k, f"{ratios[:, i].mean():.4f}", f"{ratios[:, i].std(ddof=1):.4f}"]
                       + [rows[(s, k)][c] for s in SPLITS for c in ("balanced_accuracy", "balanced_accuracy_std", "map_macro", "map_macro_std", "map_macro_random")])
    print(f"[output] {fig_dir / 'd_k_sweep.png'}, {fig_dir / 'd_k_sweep.csv'}")


if __name__ == "__main__":
    main()
