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


BAL_LABEL = {True: "あり", False: "なし"}
BAL_TAG = {True: "bal", False: "nobal"}

# 各図表に使用した数式の出典（講義資料の回・頁，または独自設計）
SOURCE = {
    "cosine": "コサイン係数：第 5 回 p.26，第 3 回 p.25",
    "delta_s": "$\\mu_{\\text{same}}$ ・ $\\mu_{\\text{diff}}$ ・ $\\Delta S$ ・ 4×4 作家間平均類似度行列：独自設計",
    "map": "AP：第 5 回 p.31 を参考にした独自設計（分母は適合文書数 $R$ ，同順位は悲観的順位），作家別 MAP の平均：独自設計",
    "balanced_accuracy": "作家別正解率の平均：第 5 回 p.28 の再現率・第 6 回 p.44 の正解率を作家ごとに求めて単純平均（独自設計）",
    "macro_f1": "作家別 F 値の平均：第 5 回 p.28-29 の F 値を作家ごとに求めて単純平均（独自設計）",
    "mrr": "MRR：第 5 回 p.31（悲観的順位）",
    "random": "基準値：特徴量を使わない一様ランダムな予測・並びの期待値（独自設計）",
    "centroid": "重心法：第 2 回 p.48，$k$-NN の `per_author` ：独自設計，`global` ：比較用",
    "confusion": "混同行列：作家別の適合率・再現率（第 5 回 p.28-29）の集計",
}


def source_line(*keys: str) -> list[str]:
    return [f"出典：{'，'.join(SOURCE[k] for k in keys)}", ""]


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
        # 表・図はシード平均の行（row_type=mean）のみを用いる
        self.mean_rows = [r for r in self.rows if r["row_type"] == "mean"]

    def get(self, variant: str, L: int, split: str, bal: bool, sim: str, clf: str) -> Row | None:
        return self.by_key.get(f"{variant}|L{L}|{split}|{BAL_TAG[bal]}|{sim}|{clf}|mean")

    def unique(self, field: str, pred: Callable[[Row], bool] = lambda r: True) -> list[Any]:
        return list(dict.fromkeys(r[field] for r in self.mean_rows if pred(r)))

    @property
    def balances(self) -> list[bool]:
        """間引きありを先（主結果），なしを後（併記）．"""
        return [b for b in (True, False) if b in self.unique("balance_train")]

    @property
    def num_inter_folds(self) -> list[int]:
        return self.unique("num_folds", lambda r: r["split"] == "inter")


def pm(r: Row, col: str, digits: int = 3) -> str:
    """シード平均 ± 標本標準偏差（1 シードのみなら値のみ）．"""
    std = r.get(f"{col}_std")
    has_std = r["n_seeds"] > 1 and std is not None and not (isinstance(std, float) and math.isnan(std))
    return f"{fmt(r[col], digits)} ± {fmt(std, digits)}" if has_std else fmt(r[col], digits)


# ---------------------------------------------------------------- 各セクション


def section_overview(res: Results, headline: dict[str, Any]) -> list[str]:
    m = res.meta
    sc = res.cfg["split"]
    folds = res.num_inter_folds
    fold_text = f"Inter-work の指標は全 {'・'.join(map(str, folds))} fold の Pooled Out-of-Fold" if folds else "Inter-work の結果なし"
    bal = (
        f"Train チャンク数の間引きあり（主結果）・なし（併記）の両方．Intra-work は分割と間引きにシード {sc['intra_seeds']}，"
        f"Inter-work は間引きにシード {sc['balance_seeds']}（間引きなしの Inter-work は決定的な 1 回）"
        if sc["balance_train"]
        else f"Train チャンク数の間引きなし．Intra-work はシード {sc['intra_seeds']}"
    )
    return [
        f"# 実験結果レポート：`{res.run_dir.name}`",
        "",
        f"- 実行：{m['started_at']} 〜 {m.get('finished_at', '—')}（{m.get('elapsed_seconds', '—')} 秒）",
        f"- git：`{m['git']['commit'][:10]}`" + ("（**未コミットの変更あり**）" if m["git"]["dirty"] else ""),
        f"- コーパス SHA-256：`{m['corpus']['source_sha256'][:16]}…`",
        f"- 実行範囲：特徴量 {m['cli']['features']}，$L \\in$ {m['cli']['chunk_chars']}，分割 {m['cli']['splits']}，アブレーション {m['cli']['ablation']}",
        f"- 分割と間引き：{bal}",
        f"- 代表条件（headline）：$L = {headline['chunk_chars']}$，類似度 `{headline['similarity']}`，推論 `{headline['classifier']}`",
        f"- {fold_text}．複数シードの値は「平均 ± 標本標準偏差（ddof = 1）」",
        "- 特徴量間の比較の主指標は作家別 MAP の平均，著者判別の主指標は作家別正解率の平均と作家別 F 値の平均．$\\Delta S_{\\text{macro}}$ は特徴量内（チャンク長別）の比較に用いる",
        "",
    ]


def section_variants(res: Results) -> list[str]:
    return [
        "## 特徴量設定",
        "",
        *md_table(["設定", "クラス", "パラメータ"], [[v["name"], f"`{v['class']}`", f"`{json.dumps(v['params'], ensure_ascii=False)}`"] for v in res.meta["variants"]], "lll"),
    ]


def variant_name(res: Results, feature: str, params: dict[str, Any]) -> str | None:
    """特徴量 feature のうち params が完全一致する設定名（設定名ではなく params で照合）．"""
    names = [v["name"] for v in res.meta["variants"] if v["name"].split("[", 1)[0] == feature and v["params"] == params]
    if len(names) > 1:
        raise ValueError(f"params が同一の設定が複数: {names}")
    return names[0] if names else None


def find_variant_row(res: Results, feature: str, params: dict[str, Any], L: int, split: str, bal: bool, sim: str, clf: str) -> Row | None:
    """特徴量 feature のうち params が完全一致する設定のシード平均行．"""
    name = variant_name(res, feature, params)
    return res.get(name, L, split, bal, sim, clf) if name else None


class CoreSpec:
    """report.core_comparison の解釈：付属語（func）・内容語対照（cont）× blocks の条件の params と設定名．"""

    def __init__(self, res: Results) -> None:
        cc = res.cfg["report"]["core_comparison"]
        self.feature, self.param, self.blocks = cc["feature"], cc["param"], list(cc["blocks"])
        self.kinds = {"func": cc["function_words"], "cont": cc["content_words"]}
        self.default = next((v["params"] for v in res.meta["variants"] if v["name"] == self.feature), None)
        self.res = res

    def params(self, kind: str, blocks: str) -> dict[str, Any]:
        assert self.default is not None
        return self.default | {self.param: self.kinds[kind], "blocks": blocks}

    def name(self, kind: str, blocks: str) -> str | None:
        return variant_name(self.res, self.feature, self.params(kind, blocks))

    def row(self, kind: str, blocks: str, L: int, split: str, bal: bool, sim: str, clf: str) -> Row | None:
        n = self.name(kind, blocks)
        return self.res.get(n, L, split, bal, sim, clf) if n else None


CORE_COLS = ("map_macro", "balanced_accuracy", "macro_f1")
CORE_LABEL = {"func": "付属語", "cont": "内容語"}


def section_core(res: Results, h: dict[str, Any], fig_dir: Path) -> list[str]:
    """冒頭の主結果表：B の付属語（既定）vs 内容語対照を同一の表現（相対頻度・表層形・同一の品詞 2-gram ブロック）で比較．

    全チャンク長の表・チャンク長別推移の図と，Inter-work の作品ごとの一貫性の表を含む．
    """
    if res.cfg["report"].get("core_comparison") is None:
        return []
    cs = CoreSpec(res)
    title = f"## 主結果：{cs.feature} の付属語 vs 内容語対照（Core Message の検証）"
    if cs.default is None:
        return [title, "", f"今回の実行に特徴量 {cs.feature} が含まれないため未出力", ""]
    sim, clf = h["similarity"], h["classifier"]
    Ls = sorted(res.unique("chunk_chars"))
    found = [
        (blocks, L, split, bal, f, c)
        for blocks in cs.blocks
        for L in Ls
        for split in ("intra", "inter")
        for bal in res.balances
        if (f := cs.row("func", blocks, L, split, bal, sim, clf)) is not None
        # pos_bigram は target_pos が出力に影響しないため内容語対照の行は作らない
        for c in [None if blocks == "pos_bigram" else cs.row("cont", blocks, L, split, bal, sim, clf)]
    ]
    # 内容語対照の行が 1 つも無い場合は比較にならないため表を出さない
    if not any(c is not None for *_, c in found):
        return [
            title,
            "",
            f"今回の実行に比較条件（`{cs.param}` × `blocks` の差し替え）が含まれないため未出力．"
            f"`features.{cs.feature}.all_chunk_chars` に対照条件が設定されていることを確認すること",
            "",
        ]
    rows = [
        [
            f"`{blocks}`",
            L,
            split,
            BAL_LABEL[bal],
            *[x for col in CORE_COLS for x in (pm(f, col), pm(c, col) if c else "—", fmt(f[col] - c[col]) if c else "—")],
            fmt(f["map_macro_random"]),
        ]
        for blocks, L, split, bal, f, c in found
    ]
    write_rows(
        res.run_dir / "table_core_comparison.csv",
        ["blocks", "chunk_chars", "split", "balance_train", "word_type", "variant", "n_seeds", *[x for col in (*CORE_COLS, "map_macro_random") for x in (col, f"{col}_std")]],
        [
            [blocks, L, split, bal, kind, r["variant"], r["n_seeds"], *[r[x] for col in (*CORE_COLS, "map_macro_random") for x in (col, f"{col}_std")]]
            for blocks, L, split, bal, f, c in found
            for kind, r in (("function", f), ("content", c))
            if r is not None
        ],
    )
    header = [
        "blocks",
        "L",
        "分割",
        "間引き",
        *[x for name in ("作家別 MAP の平均", "作家別正解率の平均", "作家別 F 値の平均") for x in (f"{name}（付属語）", f"{name}（内容語）", "差（付属語 − 内容語）")],
        "作家別 MAP の平均の基準値",
    ]
    missing_L = sorted(set(Ls) - {L for b, L, *_, c in found if b == cs.blocks[0] and c is not None})
    return [
        title,
        "",
        f"条件：`{cs.feature}`，`{sim}`，`{clf}`（作家別 MAP の平均は推論方式に依存しない）．"
        f"付属語 `{cs.param}={cs.kinds['func']}`（本採用），内容語対照 `{cs.param}={cs.kinds['cont']}`．それ以外の引数（相対頻度・表層形・品詞 2-gram ブロック）は共通",
        "",
        f"- 主は `blocks={cs.blocks[0]}`（語ブロック単独）．"
        + "，".join(f"`blocks={b}`" for b in cs.blocks[1:])
        + " は併記（`all` は品詞 2-gram との結合版，`pos_bigram` は品詞 2-gram 単独で付属語・内容語の区別なし）",
        f"- `words` ・ `all` の対照条件は全チャンク長（`features.{cs.feature}.all_chunk_chars`），`pos_bigram` は `--ablation` 指定時の "
        f"`experiment.ablation_chunk_chars` {res.cfg['experiment']['ablation_chunk_chars']} のみ",
        "- 作家別正解率の平均の基準値は 0.25，作家別 F 値の平均の基準値は第 1 節の表を参照",
        *([f"- **注意**：`blocks={cs.blocks[0]}` の対照条件が無いチャンク長：{missing_L}"] if missing_L else []),
        "",
        *md_table(header, rows, "lrll" + "r" * (len(header) - 4)),
        *source_line("cosine", "map", "balanced_accuracy", "macro_f1", "random"),
        *core_chunk_length(res, cs, h, Ls, fig_dir),
        *core_per_work(res, cs, h),
    ]


def core_chunk_length(res: Results, cs: CoreSpec, h: dict[str, Any], Ls: list[int], fig_dir: Path) -> list[str]:
    """主結果のチャンク長別推移（間引きあり，words・all の付属語 vs 内容語）．"""
    sim, clf = h["similarity"], h["classifier"]
    splits = [s for s in ("intra", "inter") if s in res.unique("split")]
    blocks = [b for b in cs.blocks if b != "pos_bigram"]
    bal = res.balances[0]
    fig, axes = plt.subplots(len(CORE_COLS), len(splits), figsize=(4.6 * len(splits), 3.0 * len(CORE_COLS)), squeeze=False)
    labels = {"map_macro": "MAP (per-author mean)", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1"}
    for col, ax_row in zip(CORE_COLS, axes, strict=True):
        for split, ax in zip(splits, ax_row, strict=True):
            for b in blocks:
                for kind, color in (("func", "C0"), ("cont", "C3")):
                    pts = [(L, r[col], r[f"{col}_std"]) for L in Ls if (r := cs.row(kind, b, L, split, bal, sim, clf)) is not None]
                    if pts:
                        x, y, e = zip(*pts, strict=True)
                        err = [0.0 if (v is None or math.isnan(v)) else v for v in e]
                        main = b == blocks[0]
                        ax.errorbar(x, y, yerr=err, marker="o" if main else "s", linestyle="-" if main else ":", capsize=3, color=color, label=f"{'function' if kind == 'func' else 'content'} / {b}")
            if col == "balanced_accuracy":
                ax.axhline(0.25, color="gray", linewidth=0.8)
            ax.set(xscale="log", xticks=Ls, xticklabels=[str(L) for L in Ls], xlabel="Chunk length L (chars)", ylabel=labels[col], title=f"{split} / {BAL_TAG[bal]} / {clf}")
            ax.minorticks_off()
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "core_chunk_length.png", dpi=200)
    plt.close(fig)
    return [
        f"### 主結果のチャンク長別推移（間引き{BAL_LABEL[bal]}）",
        "",
        f"青：付属語，赤：内容語対照．実線は `blocks={blocks[0]}`（主），点線・四角印は "
        + "，".join(f"`blocks={b}`" for b in blocks[1:])
        + "．誤差棒は標本標準偏差，灰色の横線は作家別正解率の平均の基準値 0.25．数値は上の表を参照",
        "",
        "![core chunk length](figures/core_chunk_length.png)",
        "",
        *source_line("map", "balanced_accuracy", "macro_f1"),
    ]


def read_folds(run_dir: Path) -> list[Row]:
    with (run_dir / "folds.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def core_per_work(res: Results, cs: CoreSpec, h: dict[str, Any]) -> list[str]:
    """Inter-work（間引きあり）の作品ごとの一貫性：Test 作品（fold）ごとの付属語 vs 内容語（blocks の主）．

    作品ごとの作家別 MAP の平均に相当する値は，その fold の Test チャンクの AP の平均（fold の map_micro．
    Test は 1 作家のみのため作家別平均と一致）．作家別正解率はその fold の正解率（fold の accuracy）．
    """
    clf, b0 = h["classifier"], cs.blocks[0]
    title = f"### 作品ごとの一貫性（Inter-work，間引きあり，`blocks={b0}`）"
    names = {k: cs.name(k, b0) for k in ("func", "cont")}
    if True not in res.balances or "inter" not in res.unique("split") or None in names.values():
        return [title, "", "今回の実行に Inter-work の間引きありの対照条件が含まれないため未出力", ""]
    folds = [
        r for r in read_folds(res.run_dir)
        if r["split"] == "inter" and r["balance_train"] == "True" and r["classifier"] == clf and r["similarity"] == h["similarity"]
    ]
    titles = res.meta.get("work_titles", {})
    Ls = sorted({int(r["chunk_chars"]) for r in folds})
    metrics = (("map_micro", "作家別 MAP の平均（Test チャンクの AP の平均）"), ("accuracy", "作家別正解率"))

    def values(kind: str, L: int, fold: str, col: str) -> np.ndarray:
        # shape: (n_seeds,)，シード昇順
        rows = sorted((r for r in folds if r["variant"] == names[kind] and int(r["chunk_chars"]) == L and r["fold"] == fold), key=lambda r: int(r["seed"]))
        return np.array([float(r[col]) for r in rows], dtype=np.float64)

    def ms(v: np.ndarray) -> str:
        return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if v.size > 1 else f"{v.mean():.3f}"

    table: list[list[Any]] = []
    csv_rows: list[list[Any]] = []
    wins: list[list[Any]] = []
    for L in Ls:
        fold_names = list(dict.fromkeys(r["fold"] for r in folds if int(r["chunk_chars"]) == L and r["variant"] == names["func"]))
        diffs: dict[str, list[float]] = {col: [] for col, _ in metrics}
        for fold in fold_names:
            work = fold.removeprefix("lowo_")
            vals = {(k, col): values(k, L, fold, col) for k in ("func", "cont") for col, _ in metrics}
            n = {v.size for v in vals.values()}
            if len(n) != 1 or 0 in n:
                raise ValueError(f"fold {fold}（L={L}）のシード数が条件間で不一致: {sorted(n)}")
            for col, _ in metrics:
                diffs[col].append(float(vals[("func", col)].mean() - vals[("cont", col)].mean()))
            table.append([
                L,
                titles.get(work, work),
                *[x for col, _ in metrics for x in (ms(vals[("func", col)]), ms(vals[("cont", col)]), fmt(diffs[col][-1]))],
            ])
            csv_rows.append([L, work, titles.get(work, work), n.pop(), *[vals[(k, col)].mean() for col, _ in metrics for k in ("func", "cont")], *[vals[(k, col)].std(ddof=1) for col, _ in metrics for k in ("func", "cont")]])
        wins.append([L, len(fold_names), *[f"{sum(d > 0 for d in diffs[col])} / {len(fold_names)}" for col, _ in metrics], *[f"{sum(d == 0 for d in diffs[col])}" for col, _ in metrics]])
    write_rows(
        res.run_dir / "table_core_per_work.csv",
        ["chunk_chars", "work_id", "title", "n_seeds", *[f"{col}_{k}" for col, _ in metrics for k in ("function", "content")], *[f"{col}_{k}_std" for col, _ in metrics for k in ("function", "content")]],
        csv_rows,
    )
    head = [x for _, name in metrics for x in (f"{name}（付属語）", f"{name}（内容語）", "差")]
    return [
        title,
        "",
        f"条件：`{cs.feature}`，`{h['similarity']}`，`{clf}`（作家別 MAP の平均に相当する値は推論方式に依存しない）．"
        "各 Test 作品（fold）の Test チャンクは 1 作家のみのため，その作品の作家別 MAP の平均は Test チャンクの AP の平均，作家別正解率はその作品の Test チャンクの正解率",
        "",
        "- ± は間引きの乱数（シード）によるばらつきのみで，作品の違いによるばらつきを含まない（作品間のばらつきは下の作品ごとの差の分布で見る）",
        "- 付属語が上回る作品数はシード平均の差（付属語 − 内容語）が正の作品数",
        "",
        *md_table(["L", "作品数", "付属語が上回る作品数（MAP）", "付属語が上回る作品数（作家別正解率）", "同値（MAP）", "同値（作家別正解率）"], wins, "rrrrrr"),
        *md_table(["L", "Test 作品", *head], table, "rl" + "r" * len(head)),
        *source_line("map", "balanced_accuracy"),
    ]


MAIN_HEADER = [
    "作家別 MAP の平均",
    "基準値",
    "全 Test チャンクの AP の平均",
    "基準値",
    "作家別正解率の平均",
    "基準値",
    "作家別 F 値の平均",
    "基準値",
    "MRR",
    "基準値",
]
MAIN_COLS = [
    "map_macro",
    "map_macro_random",
    "map_micro",
    "map_micro_random",
    "balanced_accuracy",
    "balanced_accuracy_random",
    "macro_f1",
    "macro_f1_random",
    "mrr",
    "mrr_random",
]


SIM_COLS = ["mu_same_macro", "mu_diff_macro", "delta_s_macro", "mu_same", "mu_diff", "delta_s_micro"]
SIM_HEADER = ["μ_same（マクロ）", "μ_diff（マクロ）", "ΔS_macro", "μ_same（ミクロ）", "μ_diff（ミクロ）", "ΔS_micro"]


def section_main_table(res: Results, h: dict[str, Any], features: list[str]) -> list[str]:
    L, sim, clf = h["chunk_chars"], h["similarity"], h["classifier"]
    found = [
        (f, split, bal, r)
        for f in features
        for split in ("intra", "inter")
        for bal in res.balances
        if (r := res.get(f, L, split, bal, sim, clf)) is not None
    ]
    # 基準値は Test の作家構成と Train 件数のみで決まり，シード間でほぼ変わらないため平均値のみ表示
    rows = [
        [f, split, BAL_LABEL[bal], r["n_seeds"], *[fmt(r[c]) if c.endswith("_random") else pm(r, c) for c in MAIN_COLS], fmt(r["n_ties"], 1), fmt(r["n_zero_query"], 1)]
        for f, split, bal, r in found
    ]
    drops = [
        [f, BAL_LABEL[bal], *[b[c] - a[c] for c in ("map_macro", "balanced_accuracy", "macro_f1", "delta_s_macro")]]
        for f in features
        for bal in res.balances
        if (a := res.get(f, L, "intra", bal, sim, clf)) is not None and (b := res.get(f, L, "inter", bal, sim, clf)) is not None
    ]
    write_rows(
        res.run_dir / "table_main_comparison.csv",
        ["feature", "split", "balance_train", "n_seeds", *[x for c in (*MAIN_COLS, *SIM_COLS) for x in (c, f"{c}_std")]],
        [[f, split, bal, r["n_seeds"], *[r[x] for c in (*MAIN_COLS, *SIM_COLS) for x in (c, f"{c}_std")]] for f, split, bal, r in found],
    )
    return [
        "## 1. 特徴量間の比較（Intra-work vs Inter-work，間引きあり・なし）",
        "",
        f"条件：$L = {L}$，`{sim}`，`{clf}`（作家別 MAP の平均・全 Test チャンクの AP の平均は推論方式に依存しない）",
        "",
        "- 特徴量間の比較は作家別 MAP の平均を主列とする（類似度の順位のみに依存し，特徴量ごとの類似度の値の分布の違いに左右されない）",
        "- $\\Delta S_{\\text{macro}}$ は類似度の値の分布に依存するため特徴量間の比較には用いず，特徴量内（チャンク長別，第 3 節）の比較に用いる",
        "",
        *md_table(["特徴量", "分割", "間引き", "シード数", *MAIN_HEADER, "同点件数", "ノルム 0 の照会数"], rows, "lllr" + "r" * (len(MAIN_COLS) + 2)),
        f"同点件数・ノルム 0 の照会数は `{clf}` のシード平均（同点は誤判別，008）",
        "",
        *source_line("cosine", "map", "balanced_accuracy", "macro_f1", "mrr", "random"),
        "類似度の直接評価（マクロ版：4×4 作家間平均類似度行列の対角平均・非対角平均，ミクロ版：全ペアの平均．特徴量間の値の比較には用いない）",
        "",
        *md_table(
            ["特徴量", "分割", "間引き", *SIM_HEADER],
            [[f, split, BAL_LABEL[bal], *[pm(r, c, 4) for c in SIM_COLS]] for f, split, bal, r in found],
            "lll" + "r" * len(SIM_COLS),
        ),
        *source_line("cosine", "delta_s"),
        "Intra-work → Inter-work の変化量（負なら未知作品で低下，シード平均の差）",
        "",
        *md_table(["特徴量", "間引き", "Δ(作家別 MAP の平均)", "Δ(作家別正解率の平均)", "Δ(作家別 F 値の平均)", "Δ(ΔS_macro)"], drops, "llrrrr"),
        *source_line("map", "balanced_accuracy", "macro_f1", "delta_s"),
    ]


def section_distributions(res: Results, h: dict[str, Any], features: list[str], fig_dir: Path) -> list[str]:
    L, sim = h["chunk_chars"], h["similarity"]
    lines = [
        "## 2. 同一著者 vs 異著者の類似度分布と作家間平均類似度",
        "",
        f"条件：$L = {L}$，`{sim}`．ヒートマップの行は照会（Test）作家，列は登録（Train）作家．複数シードの条件は全シードのペアを合算した分布（図中の $\\Delta S$ は合算後の値で，表のシード平均とは一致しない場合がある）",
        "",
        *source_line("cosine", "delta_s"),
    ]
    for f in features:
        for split in ("intra", "inter"):
            for bal in res.balances:
                st = res.stats.get(f"{f}|L{L}|{split}|{BAL_TAG[bal]}|{sim}|all")
                if st is None:
                    continue
                base = slug(f"{f}_L{L}_{split}_{BAL_TAG[bal]}_{sim}")
                title = f"{f} / {split} / {BAL_TAG[bal]} / L={L} / {sim}"
                plot_similarity_distribution(st, fig_dir / f"dist_{base}.png", title=title)
                plot_author_similarity_matrix(st.author_matrix, res.authors, fig_dir / f"heatmap_{base}.png", title=title)
                lines += [f"### {f}（{split}，間引き{BAL_LABEL[bal]}）", "", f"![dist](figures/dist_{base}.png)", f"![heatmap](figures/heatmap_{base}.png)", ""]
    return lines


CURVE_METRICS = [
    ("delta_s_macro", "ΔS_macro"),
    ("map_macro", "MAP (per-author mean)"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro-F1"),
]


def section_chunk_length(res: Results, h: dict[str, Any], features: list[str], fig_dir: Path) -> list[str]:
    sim, clf = h["similarity"], h["classifier"]
    Ls = sorted(res.unique("chunk_chars"))
    splits = [s for s in ("intra", "inter") if s in res.unique("split")]
    fig, axes = plt.subplots(len(CURVE_METRICS), len(splits), figsize=(4.6 * len(splits), 3.0 * len(CURVE_METRICS)), squeeze=False)
    for (mkey, mlabel), ax_row in zip(CURVE_METRICS, axes, strict=True):
        for split, ax in zip(splits, ax_row, strict=True):
            for f in features:
                for bal in res.balances:
                    pts = [(L, r[mkey], r[f"{mkey}_std"]) for L in Ls if (r := res.get(f, L, split, bal, sim, clf)) is not None]
                    if pts:
                        x, y, e = zip(*pts, strict=True)
                        err = [0.0 if (v is None or math.isnan(v)) else v for v in e]
                        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linestyle="-" if bal else "--", label=f"{f} ({BAL_TAG[bal]})")
            ax.set(xscale="log", xticks=Ls, xticklabels=[str(L) for L in Ls], xlabel="Chunk length L (chars)", ylabel=mlabel, title=f"{split} / {sim} / {clf}")
            ax.minorticks_off()
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "chunk_length_curves.png", dpi=200)
    plt.close(fig)
    found = [
        (f, split, bal, L, r)
        for f in features
        for split in splits
        for bal in res.balances
        for L in Ls
        if (r := res.get(f, L, split, bal, sim, clf)) is not None
    ]
    cols = ["delta_s_macro", "map_macro", "balanced_accuracy", "macro_f1"]
    write_rows(
        res.run_dir / "table_chunk_length.csv",
        ["feature", "split", "balance_train", "chunk_chars", *[x for c in cols for x in (c, f"{c}_std")]],
        [[f, split, bal, L, *[r[x] for c in cols for x in (c, f"{c}_std")]] for f, split, bal, L, r in found],
    )
    return [
        "## 3. チャンク長 L による類似度ギャップと判別精度の推移（特徴量内の比較）",
        "",
        f"条件：`{sim}`，`{clf}`．実線は間引きあり，破線は間引きなし，誤差棒は標本標準偏差．$\\Delta S_{{\\text{{macro}}}}$ は同じ特徴量のチャンク長別の比較に用いる",
        "",
        "![chunk length](figures/chunk_length_curves.png)",
        "",
        *source_line("delta_s", "map", "balanced_accuracy", "macro_f1"),
        *md_table(
            ["特徴量", "分割", "間引き", "L", "ΔS_macro", "作家別 MAP の平均", "作家別正解率の平均", "作家別 F 値の平均"],
            [[f, split, BAL_LABEL[bal], L, *[pm(r, c, 4 if c == "delta_s_macro" else 3) for c in cols]] for f, split, bal, L, r in found],
            "lllr" + "r" * len(cols),
        ),
    ]


def section_classifier(res: Results, h: dict[str, Any], features: list[str]) -> list[str]:
    L, sim = h["chunk_chars"], h["similarity"]
    clfs = res.unique("classifier")

    def table(col: str) -> list[list[Any]]:
        return [
            [f, split, BAL_LABEL[bal], *[pm(r, col) if (r := res.get(f, L, split, bal, sim, c)) else "—" for c in clfs]]
            for f in features
            for split in ("intra", "inter")
            for bal in res.balances
            if any(res.get(f, L, split, bal, sim, c) for c in clfs)
        ]

    align = "lll" + "r" * len(clfs)
    return [
        f"## 4. 推論方式の比較（$L = {L}$，`{sim}`）",
        "",
        "### 作家別正解率の平均",
        "",
        *md_table(["特徴量", "分割", "間引き", *clfs], table("balanced_accuracy"), align),
        "### 作家別 F 値の平均",
        "",
        *md_table(["特徴量", "分割", "間引き", *clfs], table("macro_f1"), align),
        "### MRR",
        "",
        *md_table(["特徴量", "分割", "間引き", *clfs], table("mrr"), align),
        "- `knn_mode=\"global\"` の MRR は，上位 $k$ 件に入らない作家のスコアがすべて 0 の同点となり，悲観的順位（008）で数えるため低く出る．MRR の考察は重心法と `per_author` を主とする",
        "- 全条件の同点件数（`n_ties`）・ノルム 0 の照会数（`n_zero_query`）・シードごとの値は `summary.csv` ，計算時間は `timings.csv` を参照",
        "",
        *source_line("centroid", "balanced_accuracy", "macro_f1", "mrr"),
    ]


def section_best(res: Results, fig_dir: Path) -> list[str]:
    if res.best_key is None:
        return ["## 5. 最良モデル", "", "Inter-work の結果がないため未出力", ""]
    b = res.by_key[res.best_key]
    plot_confusion_matrix(
        np.asarray(b["confusion"]),
        res.authors,
        fig_dir / "confusion_best.png",
        title=f"Best (pooled LOWO, {b['n_seeds']} seed(s)): {b['variant']} L={b['chunk_chars']} {BAL_TAG[b['balance_train']]}",
    )
    per_author = [
        [a, b[f"author_acc_{a}"], p, f1, b[f"author_ap_{a}"]]
        for a, p, f1 in zip(res.authors, b["precision"], b["f1"], strict=True)
    ]
    return [
        "## 5. 最良モデルの混同行列と誤判別",
        "",
        f"- 条件：`{b['key']}`（選択基準：Inter-work の Pooled `{res.cfg['report']['best_model_metric']}` のシード平均，本採用設定のみ）",
        f"- 作家別正解率の平均 {pm(b, 'balanced_accuracy', 4)}（基準値 {fmt(b['balanced_accuracy_random'], 4)}），"
        f"作家別 F 値の平均 {pm(b, 'macro_f1', 4)}（基準値 {fmt(b['macro_f1_random'], 4)}），"
        f"MRR {pm(b, 'mrr', 4)}（基準値 {fmt(b['mrr_random'], 4)}），"
        f"作家別 MAP の平均 {pm(b, 'map_macro', 4)}（基準値 {fmt(b['map_macro_random'], 4)}）",
        f"- 同点（判別不能，混同行列の `tie` 列）{fmt(b['n_ties'], 1)} 件，ノルム 0 の照会 {fmt(b['n_zero_query'], 1)} 件（シード平均．同点は誤判別，MRR は悲観的順位で集計）",
        f"- 混同行列は全 {b['n_seeds']} シードの合算，作家別の値はシード平均",
        "- 注：Test 結果から最良条件を選んでいるため，この値は楽観的な上限であり汎化性能の推定ではない",
        "",
        "![confusion](figures/confusion_best.png)",
        "",
        *md_table(["作家", "作家別正解率", "適合率", "F 値", "AP の作家別平均"], per_author),
        *source_line("confusion", "balanced_accuracy", "macro_f1", "map"),
        "誤判別チャンクの抜粋：[misclassified_best.md](misclassified_best.md)",
        "",
    ]


def section_ablation(res: Results, h: dict[str, Any]) -> list[str]:
    ablated = res.unique("feature", lambda r: not r["is_default"])
    if not ablated:
        return []
    sim, clf = h["similarity"], h["classifier"]
    cols = ["map_macro", "balanced_accuracy", "macro_f1", "delta_s_macro"]
    rows = [
        [v, L, split, BAL_LABEL[bal], *[pm(r, c, 4 if c == "delta_s_macro" else 3) for c in cols]]
        for f in ablated
        for v in res.unique("variant", lambda r, f=f: r["feature"] == f)
        for L in sorted(res.unique("chunk_chars", lambda r: not r["is_default"]))
        for split in ("intra", "inter")
        for bal in res.balances
        if (r := res.get(v, L, split, bal, sim, clf)) is not None
    ]
    return [
        "## 6. アブレーション（本採用値から 1 引数ずつ，または combinations の複数引数を同時に差し替え）",
        "",
        f"条件：`{sim}`，`{clf}`．$\\Delta S_{{\\text{{macro}}}}$ は同じ特徴量の設定間の比較に限る",
        "",
        *md_table(["設定", "L", "分割", "間引き", "作家別 MAP の平均", "作家別正解率の平均", "作家別 F 値の平均", "ΔS_macro"], rows, "lrll" + "r" * len(cols)),
        *source_line("map", "balanced_accuracy", "macro_f1", "delta_s"),
    ]


def section_notes(res: Results) -> list[str]:
    """解釈上の注記（n_min と per_author の k，作品の多様性の差，global の MRR）．"""
    inter = (res.meta.get("balance_train") or {}).get("inter", {}).get("folds", {})
    nmin = [
        (int(L), min((r["n_min"], r["fold"]) for recs in by_seed.values() for r in recs))
        for L, by_seed in sorted(inter.items(), key=lambda kv: int(kv[0]))
    ]
    k_max = max(res.cfg["classifier"]["k_values"])
    nmin_text = "，".join(f"$L = {L}$ で {n}（`{fold}`）" for L, (n, fold) in nmin) if nmin else "今回の実行では Inter-work の間引きなし"
    return [
        "## 注記",
        "",
        f"- `per_author` の $k = {k_max}$ ：Inter-work の間引き後の作家別 Train チャンク数 $n_{{\\min}}$ の最小値は {nmin_text}．"
        f"$n_{{\\min}}$ が小さい fold では上位 $k$ 件の平均が作家の全チャンクの平均（重心法に近い判定）に近づく",
        "- 作品の多様性の差（015）：Inter-work では Test 作家だけ Train に 1 作品少なく，この差は間引き後も残る．"
        "重心とのコサインは作家内平均類似度 $\\bar{\\rho}_a$ が小さい作家ほど $1 / \\|\\hat{c}_a\\|$ で割り増されるため，残りの偏りとして効く可能性がある",
        "- `knn_mode=\"global\"` の MRR は悲観的順位（008）のため，上位 $k$ 件外の作家の同点で低く出る",
        "- Test は間引かないため作家構成が偏る．正解率（全 Test チャンクでの集計）は漱石の正誤に支配されるため本レポートには載せず，`summary.csv` の `accuracy` 列にのみ出力",
        "",
    ]


def write_rows(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def make_report(run_dir: Path) -> Path:
    res = Results(run_dir)
    configured = res.cfg["report"]["headline"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    features = res.unique("variant", lambda r: r["is_default"])
    # 代表チャンク長が実行範囲に無い場合（例：--chunk-chars 1000 のスモーク）は，最も近い L（同距離なら大きい方）に
    # 置き換え，report.md の冒頭と標準出力に明示する（暗黙の置き換えはしない）
    Ls = res.unique("chunk_chars")
    substituted = configured["chunk_chars"] not in Ls and bool(Ls)
    h = configured | ({"chunk_chars": min(Ls, key=lambda L: (abs(L - configured["chunk_chars"]), -L))} if substituted else {})
    sub_notice = (
        [
            f"> **注意**：設定の代表チャンク長 `report.headline.chunk_chars` = {configured['chunk_chars']} が今回の実行範囲 {sorted(Ls)} に"
            f"含まれないため，本レポートの代表条件は $L = {h['chunk_chars']}$ に置き換えた",
            "",
        ]
        if substituted
        else []
    )
    if substituted:
        print(f"[warn] headline.chunk_chars={configured['chunk_chars']} が実行範囲外のため L={h['chunk_chars']} で代表条件の図表を作成")
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
        *sub_notice,
        *notice,
        *section_core(res, h, fig_dir),
        *section_variants(res),
        *section_main_table(res, h, features),
        *section_distributions(res, h, features, fig_dir),
        *section_chunk_length(res, h, features, fig_dir),
        *section_classifier(res, h, features),
        *section_best(res, fig_dir),
        *section_ablation(res, h),
        *section_notes(res),
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
