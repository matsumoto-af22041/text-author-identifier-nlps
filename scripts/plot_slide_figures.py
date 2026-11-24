"""発表スライド用の追加図（storyline のグラフ A・C・F）を full_006 の summary.csv から描画．

使い方：
    uv run python scripts/plot_slide_figures.py [--run-dir outputs/full_006] [--fig-dir presentation/figures]
                                                [--values-dir outputs/slide_figures] [--config configs/default.yaml]

入力：<run-dir>/summary.csv （シード平均の行 row_type=mean のみ．実験の再実行なし）
出力：
    <fig-dir>/a_ablation.png             グラフ A（Slide 8）：A のアブレーション，L = 500，重心法，間引きあり
    <fig-dir>/c_intra_inter.png          グラフ C（Slide 14）：C の作品内分割と作品間分割，L 別，重心法，間引きあり
    <fig-dir>/classifier_comparison.png  グラフ F（補助 S5）：推論方式ごとの作家別正解率の平均，L = 500，作品間分割
    <values-dir>/<図の名前>.csv            描画した数値（描画前の値，check_slides.py が summary.csv と照合）

図の表記は scripts/plot_results.py と同じ（031：日本語の研究用語，フォントは config の report.font_candidates）．
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.plot_results import BAL_LABEL, CHUNK_AXIS_LABEL, METRIC_LABEL, SPLIT_LABEL, clf_label, setup_font

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Row = dict[str, Any]
SPLITS = ("intra", "inter")
SPLIT_COLOR = {"intra": "tab:blue", "inter": "tab:orange"}
FEATURE_COLOR = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

# グラフ A の条件（summary.csv の variant → 図の表記）
A_VARIANTS = {"A": "既定", "A[scaling=minmax]": "Min-Max 正規化", "A[blocks=bigram]": "Bi-gram のみ", "A[blocks=symbol]": "記号のみ"}
# グラフ F の推論方式（storyline の表 5 と同じ列）
F_CLASSIFIERS = ("centroid", "knn1_per_author", "knn5_per_author", "knn20_per_author", "knn5_global")
VALUE_FIELDS = ("figure", "panel", "variant", "chunk_chars", "split", "balance", "classifier", "metric", "value", "std", "key")


def row_key(variant: str, L: int, split: str, bal: bool, clf: str) -> str:
    return f"{variant}|L{L}|{split}|{'bal' if bal else 'nobal'}|cosine|{clf}|mean"


class Summary:
    """summary.csv のシード平均の行．無い条件・NaN は停止（黙って欠損を描かない）．"""

    def __init__(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            self.rows = {r["key"]: r for r in csv.DictReader(f) if r["row_type"] == "mean"}
        self.values: list[dict[str, Any]] = []

    def get(self, figure: str, panel: str, variant: str, L: int, split: str, bal: bool, clf: str, metric: str) -> tuple[float, float]:
        """(値, 標本標準偏差)．1 シード（間引きなしの Inter-work）の標準偏差は 0 として描く（誤差棒なし）．"""
        key = row_key(variant, L, split, bal, clf)
        if key not in self.rows:
            raise KeyError(f"summary.csv に無い条件: {key}")
        r = self.rows[key]
        value = float(r[metric])
        single = int(r["n_seeds"]) == 1
        std_s = "" if single else r[f"{metric}_std"]
        std = 0.0 if single else float(std_s)
        if not (np.isfinite(value) and np.isfinite(std)):
            raise ValueError(f"{key} の {metric} が有限でない: {value}, {std}")
        self.values.append({"figure": figure, "panel": panel, "variant": variant, "chunk_chars": L, "split": split,
                            "balance": bal, "classifier": clf, "metric": metric, "value": value, "std": std_s, "key": key})
        return value, std

    def write_values(self, path: Path, figure: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=VALUE_FIELDS)
            w.writeheader()
            w.writerows(v for v in self.values if v["figure"] == figure)


def bar_labels(ax: plt.Axes, bars: Any, values: Sequence[float]) -> None:
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=7, padding=2)


def graph_a(s: Summary, L: int, path: Path) -> None:
    """A のアブレーション：作家別 MAP の平均・作家別正解率の平均（作品内分割・作品間分割，基準値の線）．"""
    variants = list(A_VARIANTS)
    x = np.arange(len(variants))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, metric in zip(axes, ("map_macro", "balanced_accuracy"), strict=True):
        for offset, split in zip((-width / 2, width / 2), SPLITS, strict=True):
            vals = np.array([s.get("a_ablation", metric, v, L, split, True, "centroid", metric) for v in variants])  # shape: (4, 2)
            bars = ax.bar(x + offset, vals[:, 0], width, yerr=vals[:, 1], capsize=3, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
            bar_labels(ax, bars, vals[:, 0])
            rnd, _ = s.get("a_ablation", f"{metric}_random", "A", L, split, True, "centroid", f"{metric}_random")
            ax.axhline(rnd, color=SPLIT_COLOR[split], linestyle=":", linewidth=1.2, label=f"基準値（{SPLIT_LABEL[split]}）")
        ax.set_xticks(x, [A_VARIANTS[v] for v in variants])
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"A のアブレーション（$L = {L}$，重心法，間引きあり，シード平均 ± 標本標準偏差）", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig_keep_layout(fig, path)


def graph_c(s: Summary, Ls: Sequence[int], path: Path) -> None:
    """C の作品内分割と作品間分割：ΔS（マクロ）・作家別 MAP の平均・作家別正解率の平均の L 別推移．"""
    x = np.arange(len(Ls))
    metrics = ("delta_s_macro", "map_macro", "balanced_accuracy")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric in zip(axes, metrics, strict=True):
        for split in SPLITS:
            vals = np.array([s.get("c_intra_inter", metric, "C", L, split, True, "centroid", metric) for L in Ls])  # shape: (|L|, 2)
            ax.errorbar(x, vals[:, 0], yerr=vals[:, 1], marker="o", capsize=3, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
            [ax.annotate(f"{v:.3f}" if metric != "delta_s_macro" else f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7, color=SPLIT_COLOR[split]) for xi, v in zip(x, vals[:, 0], strict=True)]
            if metric != "delta_s_macro":
                rnd = np.array([s.get("c_intra_inter", f"{metric}_random", "C", L, split, True, "centroid", f"{metric}_random")[0] for L in Ls])
                ax.plot(x, rnd, linestyle=":", color=SPLIT_COLOR[split], label=f"基準値（{SPLIT_LABEL[split]}）")
        ax.set_xticks(x, [str(L) for L in Ls])
        ax.margins(x=0.12, y=0.15)
        ax.set_xlabel(CHUNK_AXIS_LABEL)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.grid(alpha=0.3)
    axes[2].set_ylim(0, 1.1)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("C（内容語 TF-IDF）：作品内分割と作品間分割（重心法，間引きあり，シード平均 ± 標本標準偏差）", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save_fig_keep_layout(fig, path)


def graph_f(s: Summary, L: int, features: Sequence[str], path: Path) -> None:
    """推論方式ごとの作家別正解率の平均（作品間分割，間引きあり / なし の 2 パネル，基準値 0.25 の線）．"""
    x = np.arange(len(F_CLASSIFIERS))
    width = 0.8 / len(features)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    metric = "balanced_accuracy"
    for ax, bal in zip(axes, (True, False), strict=True):
        for i, feat in enumerate(features):
            vals = np.array([s.get("classifier_comparison", f"bal={bal}", feat, L, "inter", bal, c, metric) for c in F_CLASSIFIERS])
            ax.bar(x + (i - (len(features) - 1) / 2) * width, vals[:, 0], width, yerr=vals[:, 1] if bal else None, capsize=2,
                   color=FEATURE_COLOR[feat], label=feat)
        rnd, _ = s.get("classifier_comparison", f"bal={bal}", features[0], L, "inter", bal, "centroid", f"{metric}_random")
        ax.axhline(rnd, color="black", linestyle=":", linewidth=1.2, label="基準値")
        ax.set_xticks(x, [tick_label(c) for c in F_CLASSIFIERS], fontsize=8)
        ax.set_title(f"間引き{BAL_LABEL[bal]}", fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(METRIC_LABEL[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"推論方式の比較（{SPLIT_LABEL['inter']}，$L = {L}$，間引きありはシード平均 ± 標本標準偏差）", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig_keep_layout(fig, path)


def tick_label(clf: str) -> str:
    """推論方式の軸ラベル（clf_label の表記を 2 行に折り返す）．"""
    label = clf_label(clf)
    return label.replace(" 平均（", "\n平均（").replace("の重み付き投票（", "の\n重み付き投票（")


def save_fig_keep_layout(fig: plt.Figure, path: Path) -> None:
    """凡例を図の下に置くため tight_layout(rect=...) を保ったまま保存（plot_results.save_fig は tight_layout を再実行する）．"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "full_006")
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "presentation" / "figures")
    parser.add_argument("--values-dir", type=Path, default=PROJECT_ROOT / "outputs" / "slide_figures")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(f"[font] {setup_font(cfg['report']['font_candidates'])}")
    L = cfg["report"]["headline"]["chunk_chars"]
    Ls = sorted(cfg["chunking"]["target_chunk_chars"])
    features = [f for f in cfg["features"] if f != "base"]  # 担当 A～D（ベースラインは図に含めない）

    s = Summary(args.run_dir / "summary.csv")
    figures = {
        "a_ablation": lambda p: graph_a(s, L, p),
        "c_intra_inter": lambda p: graph_c(s, Ls, p),
        "classifier_comparison": lambda p: graph_f(s, L, features, p),
    }
    for name, draw in figures.items():
        draw(args.fig_dir / f"{name}.png")
        s.write_values(args.values_dir / f"{name}.csv", name)
        print(f"[output] {args.fig_dir / f'{name}.png'}, {args.values_dir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
