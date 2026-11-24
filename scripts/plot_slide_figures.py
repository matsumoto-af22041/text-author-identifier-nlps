"""発表スライド用の追加図を full_006 の出力から描画．

使い方：
    uv run python scripts/plot_slide_figures.py [--run-dir outputs/full_006] [--fig-dir presentation/figures]
                                                [--values-dir outputs/slide_figures] [--config configs/default.yaml]

入力：<run-dir>/summary.csv （シード平均の行 row_type=mean のみ），<run-dir>/folds.csv （fold ごとの Test チャンク数），
      <run-dir>/similarity_stats.json （作家間平均類似度行列と類似度のヒストグラム）．実験の再実行なし．
出力：
    <fig-dir>/a_ablation.png             グラフ A（A の結果・考察）：A の条件の比較，L = 500，重心法，間引きあり
    <fig-dir>/b_ablation.png             グラフ B-1（B の結果・考察）：B の条件の比較，グラフ A と同じ形式
    <fig-dir>/c_intra_inter.png          グラフ C（C の結果・考察）：C の作品内分割と作品間分割，L = 500，重心法，間引きあり（グラフ A と同じ形式）
    <fig-dir>/classifier_comparison.png  グラフ F（補助 S2）：推論方式ごとの作家別正解率の平均，L = 500，作品間分割
    <fig-dir>/core_chunk_length_slide.png  図 B（付属語と内容語の統制比較，補助 S11）のスライド用の再描画（文字をスライド上で読める大きさに．
                                         044 で作家別 F 値の平均の行を外し，作家別 MAP の平均と作家別正解率の平均の 2 行）
    <fig-dir>/chunk_length_slide.png     グラフ E（チャンク長による推移，補助 S7）のスライド用の再描画：A～D の作家別 MAP の平均と
                                         作家別正解率の平均（作品内分割・作品間分割，重心法，間引きあり / なし，基準値の線）．
                                         full_006 の report の chunk_length_curves.png から ΔS と作家別 F 値の平均の行を外したもの（044）
    <fig-dir>/test_composition.png       指標の定義のスライド：作品間分割の Test の作家構成と，全件を 1 作家と予測したときの
                                         全 Test チャンクでの正解の割合・作家別正解率の平均の対比（folds.csv の Test チャンク数から計算）
    <fig-dir>/s8_heatmaps.png，s8_distributions.png  補助 S16 の 4×4 作家間平均類似度と類似度分布のスライド用の再描画
    <values-dir>/<図の名前>.csv            描画した数値（描画前の値，check_slides.py が出典と照合）

図の表記は scripts/plot_results.py と同じ（031：日本語の研究用語，フォントは config の report.font_candidates）．
スライド用の再描画（core_chunk_length_slide・chunk_length_slide・s8_*）は，スライドに置く大きさ（インチ）で描き，文字の大きさを pt で指定する．
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import yaml

from scripts.plot_results import BAL_LABEL, CHUNK_AXIS_LABEL, METRIC_LABEL, SPLIT_LABEL, clf_label, setup_font
from src.evaluator import SimilarityStats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Row = dict[str, Any]
SPLITS = ("intra", "inter")
SPLIT_COLOR = {"intra": "tab:blue", "inter": "tab:orange"}
FEATURE_COLOR = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

# グラフ A・B-1 の条件（summary.csv の variant → 図の表記）
A_VARIANTS = {"A": "既定", "A[scaling=minmax]": "Min-Max 正規化", "A[blocks=bigram]": "Bi-gram のみ", "A[blocks=symbol]": "記号のみ"}
B_VARIANTS = {"B": "既定", "B[token_key=lemma]": "語彙素",
              "B[target_pos=['助詞', '助動詞', '接続詞', '連体詞', '感動詞']]": "拡張（接続詞・\n連体詞・感動詞）",
              "B[blocks=pos_bigram]": "品詞 2-gram\n単独"}
# 図 B（統制比較）の系列：(付属語か内容語か, blocks) → summary.csv の variant
CONTENT_POS = "['名詞', '動詞', '形容詞']"
CORE_VARIANTS = {("func", "words"): "B[blocks=words]", ("cont", "words"): f"B[blocks=words,target_pos={CONTENT_POS}]",
                 ("func", "all"): "B", ("cont", "all"): f"B[target_pos={CONTENT_POS}]"}
CORE_KIND = {"func": ("付属語", "tab:blue"), "cont": ("内容語", "tab:red")}
CORE_BLOCKS = {"words": "語ブロック", "all": "結合"}
CORE_METRICS = ("map_macro", "balanced_accuracy")  # 044：作家別 F 値の平均の行を外した
# グラフ F の推論方式（storyline の表 5 と同じ列）
F_CLASSIFIERS = ("centroid", "knn1_per_author", "knn5_per_author", "knn20_per_author", "knn5_global")
# グラフ A・B-1・C の横軸の枠の数（条件の数がこれより少ないグラフも，棒の幅をグラフ A・B-1 とそろえる，044）
ABLATION_SLOTS = 4
# グラフ A・B-1・C の棒の幅（条件の間隔 1 に対する比）．045：グループ内の 2 本（作品内分割・作品間分割）は接し，
# グループの間に棒 1.5 本分強の隙間（1 − 2 × 0.28 = 0.44）を空ける
ABLATION_BAR_WIDTH = 0.28
# チャンク長による推移（スライド用）の指標と線種（間引きあり：実線，なし：破線）
CHUNK_METRICS = ("map_macro", "balanced_accuracy")
BAL_STYLE = {True: "-", False: "--"}
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


def graph_ablation(s: Summary, figure: str, variants: dict[str, str], base: str, L: int, path: Path, title: str) -> None:
    """条件の比較：作家別 MAP の平均・作家別正解率の平均（作品内分割・作品間分割，基準値の線）．base は基準値を取る variant．"""
    names = list(variants)
    x = np.arange(len(names))
    width = ABLATION_BAR_WIDTH
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, metric in zip(axes, ("map_macro", "balanced_accuracy"), strict=True):
        for offset, split in zip((-width / 2, width / 2), SPLITS, strict=True):
            vals = np.array([s.get(figure, metric, v, L, split, True, "centroid", metric) for v in names])  # shape: (len(names), 2)
            bars = ax.bar(x + offset, vals[:, 0], width, yerr=vals[:, 1], capsize=3, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
            bar_labels(ax, bars, vals[:, 0])
            rnd, _ = s.get(figure, f"{metric}_random", base, L, split, True, "centroid", f"{metric}_random")
            ax.axhline(rnd, color=SPLIT_COLOR[split], linestyle=":", linewidth=1.2, label=f"基準値（{SPLIT_LABEL[split]}）")
        ax.set_xticks(x, [variants[v] for v in names])
        center = (len(names) - 1) / 2  # 条件の数によらず横軸の枠を ABLATION_SLOTS 個分にし，棒の幅の見た目をそろえる
        ax.set_xlim(center - ABLATION_SLOTS / 2, center + ABLATION_SLOTS / 2)
        ax.set_ylabel(METRIC_LABEL[metric])
        ax.set_ylim(0, 1.05)
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_fig_keep_layout(fig, path)


def graph_a(s: Summary, L: int, path: Path) -> None:
    """A の条件の比較（グラフ A．041 で図の題名から「アブレーション」を除いた）．"""
    graph_ablation(s, "a_ablation", A_VARIANTS, "A", L, path, f"A の条件の比較（$L = {L}$，重心法，間引きあり，シード平均 ± 標本標準偏差）")


def graph_b(s: Summary, L: int, path: Path) -> None:
    """B の条件の比較（グラフ B-1，グラフ A と同じ形式）．"""
    graph_ablation(s, "b_ablation", B_VARIANTS, "B", L, path, f"B の条件の比較（$L = {L}$，重心法，間引きあり，シード平均 ± 標本標準偏差）")


def graph_c(s: Summary, L: int, path: Path) -> None:
    """C の作品内分割と作品間分割（グラフ C，グラフ A・B-1 と同じ形式．042 で代表条件 L = 500 にそろえ，L 別の推移と ΔS は除いた）．"""
    graph_ablation(s, "c_intra_inter", {"C": "C（内容語 TF-IDF）"}, "C", L, path,
                   f"C の作品内分割と作品間分割（$L = {L}$，重心法，間引きあり，シード平均 ± 標本標準偏差）")


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


def graph_core_slide(s: Summary, Ls: Sequence[int], path: Path, size_in: tuple[float, float], font_pt: float) -> None:
    """図 B（付属語と内容語の統制比較）のスライド用の再描画．plot_results.core_chunk_length と同じ系列（間引きあり，重心法，
    青：付属語，赤：内容語，実線・丸：語ブロック，点線・四角：結合，誤差棒は標本標準偏差，作家別正解率の平均に基準値 0.25 の線）．
    行：指標，列：作品内分割・作品間分割．スライドに置く大きさ size_in（インチ）で描く．"""
    fig, axes = plt.subplots(len(CORE_METRICS), len(SPLITS), figsize=size_in, squeeze=False, sharex=True)
    for metric, ax_row in zip(CORE_METRICS, axes, strict=True):
        for split, ax in zip(SPLITS, ax_row, strict=True):
            for (kind, blocks), variant in CORE_VARIANTS.items():
                vals = np.array([s.get("core_chunk_length_slide", f"{metric}|{split}", variant, L, split, True, "centroid", metric)
                                 for L in Ls])  # shape: (|L|, 2)
                name, color = CORE_KIND[kind]
                main = blocks == "words"
                ax.errorbar(Ls, vals[:, 0], yerr=vals[:, 1], marker="o" if main else "s", markersize=3.5, linestyle="-" if main else ":",
                            linewidth=1.2, capsize=2, color=color, label=f"{name}・{CORE_BLOCKS[blocks]}")
            if metric == "balanced_accuracy":
                rnd, _ = s.get("core_chunk_length_slide", f"{metric}|{split}", CORE_VARIANTS[("func", "words")], Ls[0], split, True,
                               "centroid", f"{metric}_random")
                ax.axhline(rnd, color="gray", linewidth=0.8, label="基準値")
            ax.set_xscale("log")
            ax.set_xticks(list(Ls), [str(L) for L in Ls])
            ax.minorticks_off()
            ax.tick_params(labelsize=font_pt - 1)
            ax.grid(alpha=0.3)
            if split == SPLITS[0]:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=font_pt)
            if metric == CORE_METRICS[0]:
                ax.set_title(SPLIT_LABEL[split], fontsize=font_pt)
            if metric == CORE_METRICS[-1]:
                ax.set_xlabel(CHUNK_AXIS_LABEL, fontsize=font_pt)
    handles, labels = axes[CORE_METRICS.index("balanced_accuracy")][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=font_pt - 1, frameon=False, handlelength=1.6,
               columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.4, w_pad=0.6)
    save_fig_keep_layout(fig, path)


def graph_chunk_length_slide(s: Summary, features: Sequence[str], Ls: Sequence[int], path: Path, size_in: tuple[float, float],
                             font_pt: float) -> None:
    """チャンク長による推移（グラフ E）のスライド用の再描画．行：作家別 MAP の平均・作家別正解率の平均，列：作品内分割・作品間分割．
    色は特徴量（グラフ F と同じ），線種は間引きあり（実線，誤差棒は標本標準偏差）/ なし（破線），黒線は基準値（線種は同じ規則）．"""
    fig, axes = plt.subplots(len(CHUNK_METRICS), len(SPLITS), figsize=size_in, squeeze=False, sharex=True)
    for metric, ax_row in zip(CHUNK_METRICS, axes, strict=True):
        for split, ax in zip(SPLITS, ax_row, strict=True):
            panel = f"{metric}|{split}"
            for bal in (True, False):
                for feat in features:
                    vals = np.array([s.get("chunk_length_slide", panel, feat, L, split, bal, "centroid", metric) for L in Ls])  # shape: (|L|, 2)
                    ax.errorbar(Ls, vals[:, 0], yerr=vals[:, 1] if bal else None, marker="o", markersize=3, linestyle=BAL_STYLE[bal],
                                linewidth=1.1, capsize=2, color=FEATURE_COLOR[feat])
                rnd = np.array([s.get("chunk_length_slide", panel, features[0], L, split, bal, "centroid", f"{metric}_random")[0]
                                for L in Ls])  # shape: (|L|,)
                ax.plot(Ls, rnd, color="black", linestyle=BAL_STYLE[bal], linewidth=0.8)
            ax.set_xscale("log")
            ax.set_xticks(list(Ls), [str(L) for L in Ls])
            ax.minorticks_off()
            ax.tick_params(labelsize=font_pt - 1)
            ax.grid(alpha=0.3)
            if split == SPLITS[0]:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=font_pt)
            if metric == CHUNK_METRICS[0]:
                ax.set_title(SPLIT_LABEL[split], fontsize=font_pt)
            if metric == CHUNK_METRICS[-1]:
                ax.set_xlabel(CHUNK_AXIS_LABEL, fontsize=font_pt)
    handles = ([Line2D([], [], color=FEATURE_COLOR[f], marker="o", markersize=3, linewidth=1.1, label=f) for f in features]
               + [Line2D([], [], color="gray", linestyle=BAL_STYLE[b], linewidth=1.1, label=f"間引き{BAL_LABEL[b]}") for b in (True, False)]
               + [Line2D([], [], color="black", linewidth=0.8, label="基準値")])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_pt - 1, frameon=False, handlelength=1.8,
               columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.4, w_pad=0.6)
    save_fig_keep_layout(fig, path)


def test_composition(folds_path: Path, L: int, authors: Sequence[str], labels: dict[str, str]) -> dict[str, int]:
    """作品間分割（L）の Test の作家別チャンク数：各 fold の Test は 1 作品で，fold 名 lowo_<作家>_<i> の作家に属する．
    全 fold の Test チャンク数を作家ごとに合計（Pooled の Test 全体と同じ）．間引きの有無・シード・特徴量で一致することを確認．"""
    with folds_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "inter" and int(r["chunk_chars"]) == L and r["classifier"] == "centroid"]
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for r in rows:
        author = r["fold"].split("_")[1]
        g = groups.setdefault((r["variant"], r["balance_train"], r["seed"]), {})
        g[author] = g.get(author, 0) + int(r["n_test"])
    counts = list(groups.values())
    if not counts or any(c != counts[0] for c in counts):
        raise ValueError(f"作品間分割 L = {L} の Test の作家別チャンク数が条件によって異なる（または行がない）")
    if sorted(counts[0]) != sorted(authors):
        raise ValueError(f"Test の作家 {sorted(counts[0])} が run_meta の作家 {sorted(authors)} と一致しない")
    return {a: counts[0][a] for a in authors}


def graph_composition(counts: dict[str, int], labels: dict[str, str], majority: str, path: Path, values_path: Path,
                      size_in: tuple[float, float], font_pt: float) -> None:
    """Test の作家構成と，全件を majority と予測したときの 2 つの集計の対比．
    全 Test チャンクでの正解の割合 = n_majority / n_total（Test の作家構成で決まる），
    作家別正解率の平均 = (1 + 0 + 0 + 0) / 4 = 0.25（作家構成によらない）．"""
    authors = list(counts)
    n = np.array([counts[a] for a in authors], dtype=float)  # shape: (C,)
    share = n / n.sum()
    pred = np.array([a == majority for a in authors], dtype=float)  # 全件を majority と予測したときの作家別正解率，shape: (C,)
    overall = float((n * pred).sum() / n.sum())
    per_author_mean = float(pred.mean())
    assert np.isclose(share.sum(), 1.0) and 0 < overall < 1, (share, overall)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=size_in, gridspec_kw={"width_ratios": [1.15, 1]})
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    bars = ax_l.bar([labels[a] for a in authors], share, color=colors[: len(authors)])
    ax_l.bar_label(bars, labels=[f"{int(c)}" for c in n], fontsize=font_pt - 1, padding=1)
    ax_l.set_title(f"Test の作家構成（計 {int(n.sum())} チャンク）", fontsize=font_pt)
    ax_l.set_ylabel("割合", fontsize=font_pt)
    ax_l.set_ylim(0, 0.62)
    names = ["全 Test チャンクでの\n正解の割合", "作家別正解率\nの平均"]
    bars = ax_r.bar(names, [overall, per_author_mean], color=["tab:gray", "tab:blue"])
    ax_r.bar_label(bars, labels=[f"{overall:.3f}", f"{per_author_mean:.2f}"], fontsize=font_pt - 1, padding=1)
    ax_r.set_title(f"全件を{labels[majority]}と予測した場合", fontsize=font_pt)
    ax_r.set_ylim(0, 0.62)
    for ax in (ax_l, ax_r):
        ax.tick_params(labelsize=font_pt - 1)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(w_pad=1.0)
    save_fig_keep_layout(fig, path)
    values_path.parent.mkdir(parents=True, exist_ok=True)
    with values_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "author", "value"])
        w.writerows([["n_test", a, counts[a]] for a in authors])
        w.writerows([["overall_if_all_" + majority, "", repr(overall)], ["per_author_mean_if_all_" + majority, "", repr(per_author_mean)]])


def graph_s8(stats_path: Path, features: Sequence[str], L: int, labels: Sequence[str], heat_path: Path, dist_path: Path,
             values_path: Path, heat_in: tuple[float, float], dist_in: tuple[float, float], font_pt: float) -> None:
    """補助 S16：作品間分割・間引きありの 4×4 作家間平均類似度（全 5 シードの合算，行：照会（Test）作家，列：登録（Train）作家）と
    同一著者・異著者ペアの類似度分布（密度正規化ヒストグラム）．plot_results の heatmap_*・dist_* と同じ描画データ（キー *|all）を，
    特徴量ごとの 2×2 のパネルにまとめてスライドに置く大きさで描く（箱ひげ図は省く）．"""
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = {f: SimilarityStats.from_dict(raw[f"{f}|L{L}|inter|bal|cosine|all"]) for f in features}
    rows: list[list[Any]] = []
    fig, axes = plt.subplots(2, 2, figsize=heat_in)
    for ax, f in zip(axes.flat, features, strict=True):
        A = stats[f].author_matrix  # shape: (C, C)
        assert np.isfinite(A).all(), f"{f} の作家間平均類似度に NaN"
        im = ax.imshow(A, cmap="viridis")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=font_pt - 2)
        ax.set_xticks(range(len(labels)), labels, fontsize=font_pt - 1)
        ax.set_yticks(range(len(labels)), labels, fontsize=font_pt - 1)
        ax.set_title(f, fontsize=font_pt)
        digits = 4 if np.abs(A).max() < 0.1 else 3
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                ax.text(j, i, f"{A[i, j]:.{digits}f}", ha="center", va="center", fontsize=font_pt - 2.5,
                        color="black" if im.norm(A[i, j]) > 0.6 else "white")
                rows.append(["author_matrix", f, i, j, repr(float(A[i, j]))])
    fig.supxlabel("登録（Train）作家", fontsize=font_pt)
    fig.supylabel("照会（Test）作家", fontsize=font_pt)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save_fig_keep_layout(fig, heat_path)

    fig, axes = plt.subplots(2, 2, figsize=dist_in)
    for ax, f in zip(axes.flat, features, strict=True):
        st = stats[f]
        edges = st.bin_edges
        widths = np.diff(edges)
        density = lambda h: h / (h.sum() * widths)  # noqa: E731
        ax.stairs(density(st.hist_same), edges, fill=True, alpha=0.5, color="tab:blue", label="同一著者ペア")
        ax.stairs(density(st.hist_diff), edges, fill=True, alpha=0.5, color="tab:orange", label="異著者ペア")
        ax.set_title(f, fontsize=font_pt)
        ax.tick_params(labelsize=font_pt - 2)
        ax.set_yticks([])
        rows += [["mu_same", f, "", "", repr(st.mu_same)], ["mu_diff", f, "", "", repr(st.mu_diff)]]
    handles, hl = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, hl, loc="lower center", ncol=2, fontsize=font_pt - 1, frameon=False)
    fig.supxlabel("類似度", fontsize=font_pt, y=0.075)
    fig.supylabel("密度", fontsize=font_pt)
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=0.6, w_pad=0.6)
    save_fig_keep_layout(fig, dist_path)
    values_path.parent.mkdir(parents=True, exist_ok=True)
    with values_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "feature", "i", "j", "value"])
        w.writerows(rows)


def tick_label(clf: str) -> str:
    """推論方式の軸ラベル（clf_label の表記を 2 行に折り返す）．"""
    label = clf_label(clf)
    return label.replace(" 平均（", "\n平均（").replace("の重み付き投票（", "の\n重み付き投票（")


def save_fig_keep_layout(fig: plt.Figure, path: Path) -> None:
    """凡例を図の下に置くため tight_layout(rect=...) を保ったまま保存（plot_results.save_fig は tight_layout を再実行する）．"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, metadata={"Software": None})
    plt.close(fig)


def cfg_slide_sizes() -> dict[str, Any]:
    """スライド用の再描画の大きさ（インチ，スライドに置く大きさ）と文字の大きさ（pt）．
    presentation/slides_content.yaml の theme.figures から読む（スライドの配置と同じ場所で管理）．"""
    content = yaml.safe_load((PROJECT_ROOT / "presentation" / "slides_content.yaml").read_text(encoding="utf-8"))
    f = content["theme"]["figures"]
    return {"font_pt": float(f["font_pt"]), **{k: tuple(map(float, v)) for k, v in f.items() if k != "font_pt"}}


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
    slide = cfg_slide_sizes()
    figures = {
        "a_ablation": lambda p: graph_a(s, L, p),
        "b_ablation": lambda p: graph_b(s, L, p),
        "c_intra_inter": lambda p: graph_c(s, L, p),
        "classifier_comparison": lambda p: graph_f(s, L, features, p),
        "core_chunk_length_slide": lambda p: graph_core_slide(s, Ls, p, slide["core"], slide["font_pt"]),
        "chunk_length_slide": lambda p: graph_chunk_length_slide(s, features, Ls, p, slide["chunk_length"], slide["font_pt"]),
    }
    for name, draw in figures.items():
        draw(args.fig_dir / f"{name}.png")
        s.write_values(args.values_dir / f"{name}.csv", name)
        print(f"[output] {args.fig_dir / f'{name}.png'}, {args.values_dir / f'{name}.csv'}")

    meta = json.loads((args.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    authors = meta["author_ids"]
    labels = cfg["report"]["author_labels"]
    counts = test_composition(args.run_dir / "folds.csv", L, authors, labels)
    majority = max(counts, key=counts.__getitem__)
    graph_composition(counts, labels, majority, args.fig_dir / "test_composition.png", args.values_dir / "test_composition.csv",
                      slide["composition"], slide["font_pt"])
    print(f"[output] {args.fig_dir / 'test_composition.png'}（Test の作家別チャンク数 {counts}）")
    graph_s8(args.run_dir / "similarity_stats.json", features, L, [labels[a] for a in authors], args.fig_dir / "s8_heatmaps.png",
             args.fig_dir / "s8_distributions.png", args.values_dir / "s8.csv", slide["s8_heatmaps"], slide["s8_distributions"],
             slide["font_pt"])
    print(f"[output] {args.fig_dir / 's8_heatmaps.png'}, {args.fig_dir / 's8_distributions.png'}")

if __name__ == "__main__":
    main()
