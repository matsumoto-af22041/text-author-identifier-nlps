"""実験結果ディレクトリから図表と report.md を生成．

使い方：
    uv run python scripts/plot_results.py outputs/<run_name>

図表の代表条件は実行時の設定（run_meta.json 内の config.report.headline）に従う．
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluator import (  # noqa: E402
    SimilarityStats,
    plot_author_similarity_matrix,
    plot_confusion_matrix,
    plot_similarity_distribution,
)

Row = dict[str, Any]


def fmt(x: Any, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def md_table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: str | None = None) -> list[str]:
    """align 未指定時は先頭行の値が数値（または数値文字列・欠損記号）の列を右寄せ．"""
    is_num = lambda v: isinstance(v, int | float) or (isinstance(v, str) and re.fullmatch(r"-?[\d.]+|—", v) is not None)  # noqa: E731
    align = align or ("".join("r" if is_num(v) else "l" for v in rows[0]) if rows else "l" * len(header))
    sep = ["---:" if a == "r" else ":---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows],
        "",
    ]


def slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", s).strip("_")


class Results:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        self.best_key: str | None = data["best_key"]
        self.rows: list[Row] = data["results"]
        self.by_key = {r["key"]: r for r in self.rows}
        self.stats = {
            k: SimilarityStats.from_dict(v)
            for k, v in json.loads((run_dir / "similarity_stats.json").read_text(encoding="utf-8")).items()
        }
        self.cfg = self.meta["config"]
        self.authors: list[str] = self.meta["author_ids"]

    def get(self, variant: str, L: int, split: str, sim: str, clf: str) -> Row | None:
        return self.by_key.get(f"{variant}|L{L}|{split}|{sim}|{clf}")

    def unique(self, field: str, pred: Callable[[Row], bool] = lambda r: True) -> list[Any]:
        return list(dict.fromkeys(r[field] for r in self.rows if pred(r)))


# ---------------------------------------------------------------- 各セクション


def section_overview(res: Results, headline: dict[str, Any]) -> list[str]:
    m = res.meta
    return [
        f"# 実験結果レポート：`{res.run_dir.name}`",
        "",
        f"- 実行：{m['started_at']} 〜 {m.get('finished_at', '—')}（{m.get('elapsed_seconds', '—')} 秒）",
        f"- git：`{m['git']['commit'][:10]}`" + ("（**未コミットの変更あり**）" if m["git"]["dirty"] else ""),
        f"- コーパス SHA-256：`{m['corpus']['source_sha256'][:16]}…`，seed：{m['config']['seed']}",
        f"- 実行範囲：特徴量 {m['cli']['features']}，$L \\in$ {m['cli']['chunk_chars']}，分割 {m['cli']['splits']}，アブレーション {m['cli']['ablation']}",
        f"- 代表条件（headline）：$L = {headline['chunk_chars']}$，類似度 `{headline['similarity']}`，推論 `{headline['classifier']}`",
        "- 類似度ギャップの主指標は作家単位のマクロ平均 $\\Delta S_{\\text{macro}}$，Inter-work の指標は全 9 fold の Pooled Out-of-Fold",
        "",
        "## 特徴量設定",
        "",
        *md_table(["設定", "クラス", "パラメータ"], [[v["name"], f"`{v['class']}`", f"`{json.dumps(v['params'], ensure_ascii=False)}`"] for v in m["variants"]], "lll"),
    ]


def section_main_table(res: Results, h: dict[str, Any], features: list[str]) -> list[str]:
    L, sim, clf = h["chunk_chars"], h["similarity"], h["classifier"]
    rows = [
        [f, split, r["mu_same_macro"], r["mu_diff_macro"], r["delta_s_macro"], r["delta_s_micro"], r["accuracy"], r["macro_f1"], r["mrr"]]
        for f in features
        for split in ("intra", "inter")
        if (r := res.get(f, L, split, sim, clf)) is not None
    ]
    drops = [
        [f, b["delta_s_macro"] - a["delta_s_macro"], b["macro_f1"] - a["macro_f1"]]
        for f in features
        if (a := res.get(f, L, "intra", sim, clf)) is not None and (b := res.get(f, L, "inter", sim, clf)) is not None
    ]
    write_rows(res.run_dir / "table_main_comparison.csv", ["feature", "split", "mu_same_macro", "mu_diff_macro", "delta_s_macro", "delta_s_micro", "accuracy", "macro_f1", "mrr"], rows)
    return [
        "## 1. 類似度ギャップと著者判別精度の総合比較（Intra-work vs Inter-work）",
        "",
        f"条件：$L = {L}$，`{sim}`，`{clf}`（$\\mu$ と $\\Delta S$ は推論方式に依存しない）",
        "",
        *md_table(["特徴量", "分割", "μ_same(macro)", "μ_diff(macro)", "ΔS_macro", "ΔS_micro", "Accuracy", "Macro-F1", "MRR"], rows),
        "Intra-work → Inter-work の変化量（負なら未知作品で低下）",
        "",
        *md_table(["特徴量", "Δ(ΔS_macro)", "Δ(Macro-F1)"], drops),
    ]


def section_distributions(res: Results, h: dict[str, Any], features: list[str], fig_dir: Path) -> list[str]:
    L, sim = h["chunk_chars"], h["similarity"]
    lines = [
        "## 2. 同一著者 vs 異著者の類似度分布と作家間平均類似度",
        "",
        f"条件：$L = {L}$，`{sim}`．ヒートマップの行は照会（Test）作家，列は登録（Train）作家",
        "",
    ]
    for f in features:
        for split in ("intra", "inter"):
            st = res.stats.get(f"{f}|L{L}|{split}|{sim}")
            if st is None:
                continue
            base = slug(f"{f}_L{L}_{split}_{sim}")
            plot_similarity_distribution(st, fig_dir / f"dist_{base}.png", title=f"{f} / {split} / L={L} / {sim}")
            plot_author_similarity_matrix(st.author_matrix, res.authors, fig_dir / f"heatmap_{base}.png", title=f"{f} / {split} / L={L} / {sim}")
            lines += [f"### {f}（{split}）", "", f"![dist](figures/dist_{base}.png)", f"![heatmap](figures/heatmap_{base}.png)", ""]
    return lines


def section_chunk_length(res: Results, h: dict[str, Any], features: list[str], fig_dir: Path) -> list[str]:
    sim, clf = h["similarity"], h["classifier"]
    Ls = sorted(res.unique("chunk_chars"))
    splits = [s for s in ("intra", "inter") if s in res.unique("split")]
    metrics = [("delta_s_macro", "ΔS_macro"), ("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]
    fig, axes = plt.subplots(len(metrics), len(splits), figsize=(4.6 * len(splits), 3.0 * len(metrics)), squeeze=False)
    for (mkey, mlabel), ax_row in zip(metrics, axes, strict=True):
        for split, ax in zip(splits, ax_row, strict=True):
            for f in features:
                pts = [(L, r[mkey]) for L in Ls if (r := res.get(f, L, split, sim, clf)) is not None]
                if pts:
                    ax.plot(*zip(*pts, strict=True), marker="o", label=f)
            ax.set(xscale="log", xticks=Ls, xticklabels=[str(L) for L in Ls], xlabel="Chunk length L (chars)", ylabel=mlabel, title=f"{split} / {sim} / {clf}")
            ax.minorticks_off()
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "chunk_length_curves.png", dpi=200)
    plt.close(fig)
    rows = [
        [f, split, L, r["delta_s_macro"], r["accuracy"], r["macro_f1"]]
        for f in features
        for split in splits
        for L in Ls
        if (r := res.get(f, L, split, sim, clf)) is not None
    ]
    write_rows(res.run_dir / "table_chunk_length.csv", ["feature", "split", "chunk_chars", "delta_s_macro", "accuracy", "macro_f1"], rows)
    return [
        "## 3. チャンク長 L による類似度ギャップと判別精度の推移",
        "",
        f"条件：`{sim}`，`{clf}`",
        "",
        "![chunk length](figures/chunk_length_curves.png)",
        "",
        *md_table(["特徴量", "分割", "L", "ΔS_macro", "Accuracy", "Macro-F1"], rows),
    ]


def section_similarity_classifier(res: Results, h: dict[str, Any], features: list[str]) -> list[str]:
    sims = res.unique("similarity")
    clf = h["classifier"]
    sim_rows = [
        [f, L, split, *[fmt(r["macro_f1"]) if (r := res.get(f, L, split, s, clf)) else "—" for s in sims]]
        for f in features
        for L in sorted(res.unique("chunk_chars"))
        for split in ("intra", "inter")
        if any(res.get(f, L, split, s, clf) for s in sims)
    ]
    L, sim = h["chunk_chars"], h["similarity"]
    clfs = res.unique("classifier")
    clf_rows = [
        [f, split, *[fmt(r["macro_f1"]) if (r := res.get(f, L, split, sim, c)) else "—" for c in clfs]]
        for f in features
        for split in ("intra", "inter")
        if any(res.get(f, L, split, sim, c) for c in clfs)
    ]
    return [
        "## 4. 類似度関数および推論方式の比較（Macro-F1）",
        "",
        f"### 4.1 類似度関数（推論 `{clf}`，Jaccard / Dice は計算量の制約により一部の L のみ）",
        "",
        *md_table(["特徴量", "L", "分割", *sims], sim_rows),
        f"### 4.2 推論方式（$L = {L}$，`{sim}`）",
        "",
        *md_table(["特徴量", "分割", *clfs], clf_rows),
        "全条件の Accuracy・MRR・計算時間は `summary.csv` を参照",
        "",
    ]


def section_best(res: Results, fig_dir: Path) -> list[str]:
    if res.best_key is None:
        return ["## 5. 最良モデル", "", "Inter-work の結果がないため未出力", ""]
    b = res.by_key[res.best_key]
    plot_confusion_matrix(
        np.asarray(b["confusion"]), res.authors, fig_dir / "confusion_best.png", title=f"Best (pooled LOWO): {b['variant']} L={b['chunk_chars']}"
    )
    per_author = [[a, p, r, f] for a, p, r, f in zip(res.authors, b["precision"], b["recall"], b["f1"], strict=True)]
    return [
        "## 5. 最良モデルの混同行列と誤判別",
        "",
        f"- 条件：`{b['key']}`（選択基準：Inter-work の Pooled `{res.cfg['report']['best_model_metric']}`，本採用設定のみ）",
        f"- Accuracy {b['accuracy']:.4f}，Macro-F1 {b['macro_f1']:.4f}，MRR {b['mrr']:.4f}",
        "- 注：Test 結果から最良条件を選んでいるため，この値は楽観的な上限であり汎化性能の推定ではない",
        "",
        "![confusion](figures/confusion_best.png)",
        "",
        *md_table(["作家", "Precision", "Recall", "F1"], per_author),
        "誤判別チャンクの抜粋：[misclassified_best.md](misclassified_best.md)",
        "",
    ]


def section_ablation(res: Results, h: dict[str, Any]) -> list[str]:
    ablated = res.unique("feature", lambda r: not r["is_default"])
    if not ablated:
        return []
    sim, clf = h["similarity"], h["classifier"]
    rows = [
        [v, L, split, r["delta_s_macro"], r["accuracy"], r["macro_f1"]]
        for f in ablated
        for v in res.unique("variant", lambda r, f=f: r["feature"] == f)
        for L in sorted(res.unique("chunk_chars", lambda r: not r["is_default"]))
        for split in ("intra", "inter")
        if (r := res.get(v, L, split, sim, clf)) is not None
    ]
    return [
        "## 6. アブレーション（1 引数ずつ本採用値から差し替え）",
        "",
        f"条件：`{sim}`，`{clf}`",
        "",
        *md_table(["設定", "L", "分割", "ΔS_macro", "Accuracy", "Macro-F1"], rows),
    ]


def write_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_report(run_dir: Path) -> Path:
    res = Results(run_dir)
    h = res.cfg["report"]["headline"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    features = res.unique("variant", lambda r: r["is_default"])
    missing = [
        name
        for name, ok in (
            ("headline.chunk_chars", h["chunk_chars"] in res.unique("chunk_chars")),
            ("headline.similarity", h["similarity"] in res.unique("similarity")),
            ("headline.classifier", h["classifier"] in res.unique("classifier")),
        )
        if not ok
    ]
    notice = (
        [f"> **注意**：代表条件 {missing} が今回の実行範囲に含まれないため，該当する表・図は空になる", ""] if missing else []
    )
    lines = [
        *section_overview(res, h),
        *notice,
        *section_main_table(res, h, features),
        *section_distributions(res, h, features, fig_dir),
        *section_chunk_length(res, h, features, fig_dir),
        *section_similarity_classifier(res, h, features),
        *section_best(res, fig_dir),
        *section_ablation(res, h),
    ]
    out = run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] {out}（図 {len(list(fig_dir.glob('*.png')))} 枚）")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run_experiments.py の出力ディレクトリ")
    args = parser.parse_args()
    if not (args.run_dir / "results.json").is_file():
        raise FileNotFoundError(f"results.json がない: {args.run_dir}")
    make_report(args.run_dir)


if __name__ == "__main__":
    main()
