"""漱石 2 作品版の補助実験（作品除外 run）と 9 作品版の比較表を作成．

入力は scripts/run_experiments.py の現行形式の run（split.balance_train: true で Train の間引きあり・なしを
同じ run に含み，results.json に row_type=seed（シードごと）と row_type=mean（シード平均）の行を持つ）．
Inter-work の Pooled Out-of-Fold 混同行列（行：正解，列：予測，最右列は同点）から，各チャンク長 L・
間引きの有無・推論器（既定：重心法と knn5_per_author）について，シードごとに以下を求めてシード平均・標準偏差
（ddof = 1，間引きなしは決定的な 1 回のため NaN）を集計：
    balanced_accuracy, macro_f1      作家別正解率の平均・作家別 F 値の平均（results.json の値．作家別正解率は混同行列と照合）
    map_macro, map_macro_random      作家別 MAP の平均とランダム基準（間引きの有無で基準値が異なるため併記）
    recall_<作家>                    作家別正解率（同点は FN として分母のみに算入）
    focus_share_all                  全 Test チャンク中で注目作家（既定：漱石）と予測された割合
    focus_share_others               注目作家以外の作家のチャンク中で注目作家と予測された割合（013 の集中の指標）
    accuracy, majority_author/share  正解率と Test の最多作家の割合（CSV のみ．report.md には載せない）

主結果は間引きあり（bal，5 シード平均），間引きなし（nobal）を併記．

使い方：
    uv run python scripts/summarize_exclusion.py --features A B C D \\
        --baseline <9 作品版の run> --excluded excl_bocchan_ABCD excl_kokoro_ABCD excl_sanshiro_ABCD
    uv run python scripts/summarize_exclusion.py --features base --chunk-chars 1000 \\
        --baseline base_c_022 --excluded excl027_bocchan_base excl027_kokoro_base excl027_sanshiro_base

run の対応付け（outputs/ 以下）：
    --baseline   9 作品版の run（複数可）．各特徴量はちょうど 1 つの run に含まれること
    --excluded   作品除外 run（複数可，scripts/run_experiments.py --exclude-works で作成）．除外作品は run_meta.json の
                 excluded_works から読み，除外作品の組ごと・特徴量ごとにちょうど 1 つの run に含まれること
    特徴量の本採用パラメータ・split 設定は全 run で一致すること（不一致は停止）

出力（--out-dir，既定 outputs/exclusion_summary/．既存のディレクトリには書き込まず停止）：
    summary.csv     全 run × L × 間引きの有無 × 推論器の集計値（シード平均・標準偏差）
    seeds.csv       同上のシードごとの値
    diff.csv        除外版 − 9 作品版の差分（シード平均どうし，majority_author は除外版の値）
    report.md       上記の Markdown 表（間引きあり → 間引きなしの順）
    sources.json    入力 run の git commit・除外作品・生成時刻
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_LABEL = "9works"
BALANCE_LABELS = {True: "bal", False: "nobal"}
# results.json の値と混同行列からの再計算値の照合の許容誤差（同一データの浮動小数点誤差のみ）
MATCH_ATOL = 1e-12


@dataclass(frozen=True)
class Run:
    """現行形式の 1 run．excluded は除外作品名の組（9 作品版は空）．"""

    name: str
    meta: dict[str, Any]
    results: list[dict[str, Any]]

    @property
    def excluded(self) -> tuple[str, ...]:
        return tuple(w["title"] for w in self.meta.get("excluded_works", []))

    @property
    def label(self) -> str:
        return "excl_" + "+".join(self.excluded) if self.excluded else FULL_LABEL


@dataclass(frozen=True)
class RunSpec:
    """集計対象：run × 特徴量．"""

    feature: str
    run: Run


def parse_pairs(items: Sequence[str], what: str) -> dict[str, str]:
    bad = [s for s in items if s.count("=") != 1 or not all(s.split("="))]
    if bad:
        raise ValueError(f"{what} は <キー>=<値> 形式で指定: {bad}")
    return dict(s.split("=") for s in items)


def load_run(outputs_dir: Path, name: str) -> Run:
    """run_meta.json と results.json を読み込み，現行形式（Inter-work・間引きあり / なし）の完了済み run か検証．"""
    run_dir = outputs_dir / name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run が存在しない: {run_dir}")
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if "finished_at" not in meta:
        raise ValueError(f"未完了の run: {run_dir}")
    bal = meta.get("balance_train")
    # 022 以前の形式は balance_train が無いか {seed, folds}（単一シード）
    if not (isinstance(bal, dict) and "inter" in bal and "inter" in meta["cli"]["splits"]):
        raise ValueError(f"{name}: 現行形式（間引きあり・なしを含む Inter-work）の run ではない")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["results"]
    return Run(name, meta, results)


def confusion_stats(cm: np.ndarray, focus: int, author_ids: Sequence[str]) -> dict[str, Any]:
    """混同行列 (C, C + 1) から報告量を算出．"""
    C = len(author_ids)
    assert cm.shape == (C, C + 1), f"混同行列の形状 {cm.shape} が ({C}, {C + 1}) ではない"
    # shape: (C,)
    true_count = cm.sum(axis=1)
    assert np.all(true_count > 0), f"Test に現れない作家がある: {dict(zip(author_ids, true_count.tolist(), strict=True))}"
    n = int(cm.sum())
    others = np.arange(C) != focus
    recall = np.diag(cm[:, :C]) / true_count
    return {
        "n_test": n,
        "n_ties": int(cm[:, C].sum()),
        "accuracy_cm": float(np.trace(cm[:, :C]) / n),
        "balanced_accuracy_cm": float(recall.mean()),
        **{f"recall_{a}": float(r) for a, r in zip(author_ids, recall, strict=True)},
        "focus_share_all": float(cm[:, focus].sum() / n),
        "focus_share_others": float(cm[others, focus].sum() / cm[others].sum()),
        "majority_author": author_ids[int(np.argmax(true_count))],
        "majority_share": float(true_count.max() / n),
    }


def seed_stats(d: dict[str, Any], focus: int, author_ids: Sequence[str]) -> dict[str, Any]:
    """1 シード分の行（row_type=seed）の報告量．results.json の値と混同行列からの再計算値を照合．"""
    stats = confusion_stats(np.asarray(d["confusion"], dtype=np.int64), focus, author_ids)
    checks = {
        "accuracy": (stats.pop("accuracy_cm"), d["accuracy"]),
        "balanced_accuracy": (stats.pop("balanced_accuracy_cm"), d["balanced_accuracy"]),
        **{f"recall_{a}": (stats[f"recall_{a}"], r) for a, r in zip(author_ids, d["recall"], strict=True)},
        **{f"author_acc_{a}": (stats[f"recall_{a}"], d[f"author_acc_{a}"]) for a in author_ids},
    }
    bad = [k for k, (ours, stored) in checks.items() if not np.isclose(ours, stored, rtol=0, atol=MATCH_ATOL)]
    if bad or stats["n_test"] != d["n_test"] or stats["n_ties"] != d["n_ties"]:
        raise ValueError(f"{d['key']}: 混同行列からの再計算値が results.json と不一致: {bad}")
    return {
        "accuracy": d["accuracy"],
        "balanced_accuracy": d["balanced_accuracy"],
        "macro_f1": d["macro_f1"],
        "map_macro": d["map_macro"],
        "map_macro_random": d["map_macro_random"],
        **stats,
    }


def pick(run: Run, feature: str, L: int, balanced: bool, similarity: str, classifier: str, row_type: str) -> list[dict[str, Any]]:
    return [
        d
        for d in run.results
        if d["variant"] == feature
        and d["is_default"]
        and d["split"] == "inter"
        and d["chunk_chars"] == L
        and d["balance_train"] is balanced
        and d["similarity"] == similarity
        and d["classifier"] == classifier
        and d["row_type"] == row_type
    ]


def summarize_condition(
    spec: RunSpec, L: int, balanced: bool, similarity: str, classifier: str, focus: int, author_ids: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """1 条件のシード行を集計し，シード平均の行（row_type=mean）と照合．戻り値：集計行，シード行．"""
    run = spec.run
    seed_rows = pick(run, spec.feature, L, balanced, similarity, classifier, "seed")
    mean_rows = pick(run, spec.feature, L, balanced, similarity, classifier, "mean")
    expected_seeds = sorted(run.meta["balance_train"]["inter"]["seeds"]) if balanced else [None]
    found_seeds = sorted((d["seed"] for d in seed_rows), key=lambda s: -1 if s is None else s)
    cond = f"{run.name} {spec.feature}/L{L}/{BALANCE_LABELS[balanced]}/{classifier}"
    if found_seeds != expected_seeds or len(mean_rows) != 1:
        raise ValueError(f"{cond}: 条件の過不足（seed 行のシード {found_seeds}，想定 {expected_seeds}，mean 行 {len(mean_rows)} 件）")
    per_seed = [seed_stats(d, focus, author_ids) for d in seed_rows]
    numeric = [k for k, v in per_seed[0].items() if isinstance(v, int | float)]
    # shape: (n_seeds, n_metrics)
    M = np.array([[s[k] for k in numeric] for s in per_seed], dtype=np.float64)
    assert np.all(np.isfinite(M)), f"{cond}: NaN / Inf を含む"
    mean = M.mean(axis=0)
    std = M.std(axis=0, ddof=1) if len(per_seed) > 1 else np.full(len(numeric), np.nan)
    agg = dict(zip(numeric, mean.tolist(), strict=True))

    # row_type=mean の行（run_experiments.py のシード平均・混同行列の合算）との照合
    m = mean_rows[0]
    bad = [k for k in ("accuracy", "balanced_accuracy", "macro_f1", "map_macro", "map_macro_random") if not np.isclose(agg[k], m[k], rtol=0, atol=MATCH_ATOL)]
    cm_sum = np.sum([np.asarray(d["confusion"], dtype=np.int64) for d in seed_rows], axis=0)
    if bad or not np.array_equal(cm_sum, np.asarray(m["confusion"], dtype=np.int64)) or m["n_seeds"] != len(seed_rows):
        raise ValueError(f"{cond}: シード行の集計が row_type=mean の行と不一致: {bad}")
    if len({s["majority_author"] for s in per_seed}) != 1:
        raise ValueError(f"{cond}: シード間で Test の最多作家が異なる（Test は間引かないため同一のはず）")

    head = {
        "feature": spec.feature,
        "condition": run.label,
        "excluded": "+".join(run.excluded),
        "run": run.name,
        "chunk_chars": L,
        "balance_train": BALANCE_LABELS[balanced],
        "classifier": classifier,
        "num_folds": m["num_folds"],
        "n_seeds": len(seed_rows),
    }
    row = head | {k: (int(v) if k == "n_test" else v) for k, v in agg.items()} | {f"{k}_std": float(v) for k, v in zip(numeric, std, strict=True)}
    row["majority_author"] = per_seed[0]["majority_author"]
    seeds = [head | {"seed": "" if d["seed"] is None else d["seed"]} | s for d, s in zip(seed_rows, per_seed, strict=True)]
    return row, seeds


NON_DIFF = {"feature", "condition", "excluded", "run", "chunk_chars", "balance_train", "classifier", "majority_author", "num_folds", "n_seeds"}


def diff_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """除外版の各行から，同じ (feature, L, 間引き, classifier) の 9 作品版の値（シード平均）を引く．"""
    key = lambda r: (r["feature"], r["chunk_chars"], r["balance_train"], r["classifier"])  # noqa: E731
    base = {key(r): r for r in rows if r["condition"] == FULL_LABEL}
    missing = sorted({key(r) for r in rows} - set(base))
    if missing:
        raise ValueError(f"9 作品版に対応する条件がない: {missing}")
    return [
        {
            **{k: r[k] for k in ("feature", "condition", "excluded", "chunk_chars", "balance_train", "classifier", "majority_author")},
            **{f"d_{k}": v - base[key(r)][k] for k, v in r.items() if k not in NON_DIFF and not k.endswith("_std")},
        }
        for r in rows
        if r["condition"] != FULL_LABEL
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}: 出力行が空")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: Sequence[dict[str, Any]], cols: Sequence[str], signed: bool = False) -> list[str]:
    fmt = lambda v: (f"{v:+.3f}" if signed else f"{v:.3f}") if isinstance(v, float) else str(v)  # noqa: E731
    return [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---:" if isinstance(rows[0][c], int | float) else ":---" for c in cols) + " |",
        *["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows],
        "",
    ]


def resolve_specs(features: Sequence[str], baseline: Sequence[Run], excluded: Sequence[Run]) -> list[RunSpec]:
    """各特徴量について，9 作品版と除外作品の組ごとにちょうど 1 つの run を対応付け．"""
    bad_base = [r.name for r in baseline if r.excluded]
    bad_excl = [r.name for r in excluded if not r.excluded]
    if bad_base or bad_excl:
        raise ValueError(f"--baseline に作品除外 run {bad_base}，または --excluded に 9 作品版 {bad_excl} が指定された")
    groups = {FULL_LABEL: list(baseline)} | {lab: [r for r in excluded if r.label == lab] for lab in dict.fromkeys(r.label for r in excluded)}

    def one(f: str, label: str) -> RunSpec:
        hits = [r for r in groups[label] if f in r.meta["cli"]["features"]]
        if len(hits) != 1:
            raise ValueError(f"特徴量 {f} の {label} の run が {len(hits)} 件（{[r.name for r in hits]}）")
        return RunSpec(f, hits[0])

    specs = [one(f, label) for f in features for label in groups]
    # 比較の公平性：同一特徴量の本採用パラメータと split 設定が全 run で一致
    for f in features:
        params = {s.run.name: {v["name"]: v["params"] for v in s.run.meta["variants"]}[f] for s in specs if s.feature == f}
        if len({json.dumps(p, sort_keys=True, ensure_ascii=False) for p in params.values()}) != 1:
            raise ValueError(f"特徴量 {f} のパラメータが run 間で不一致: {params}")
    split_cfg = {s.run.name: json.dumps(s.run.meta["config"]["split"], sort_keys=True) for s in specs}
    if len(set(split_cfg.values())) != 1:
        raise ValueError(f"split 設定が run 間で不一致: {sorted(split_cfg)}")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--features", nargs="+", default=["A", "B", "C", "D"], help="対象の特徴量キー（base，A～D）")
    parser.add_argument("--baseline", nargs="+", required=True, help="9 作品版の run 名（複数可）")
    parser.add_argument("--excluded", nargs="+", required=True, help="作品除外 run 名（複数可）")
    parser.add_argument("--chunk-chars", nargs="+", type=int, help="集計する L（既定：全 run に共通の L）")
    parser.add_argument("--classifiers", nargs="+", default=["centroid", "knn5_per_author"])
    parser.add_argument("--similarity", default="cosine")
    parser.add_argument("--focus-author", default="soseki", help="予測集中を測る作家 ID")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exclusion_summary")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"出力先が既に存在（上書き防止）: {args.out_dir}")
    load = lambda names: [load_run(args.outputs_dir, n) for n in names]  # noqa: E731
    specs = resolve_specs(args.features, load(args.baseline), load(args.excluded))
    runs = list({s.run.name: s.run for s in specs}.values())
    common_L = sorted(set.intersection(*(set(r.meta["cli"]["chunk_chars"]) for r in runs)))
    args.chunk_chars = args.chunk_chars or common_L
    missing_L = sorted(set(args.chunk_chars) - set(common_L))
    if not args.chunk_chars or missing_L:
        raise ValueError(f"全 run に共通の L {common_L} にない L が指定された: {missing_L}")
    author_ids = list(runs[0].meta["author_ids"])
    if any(list(r.meta["author_ids"]) != author_ids for r in runs):
        raise ValueError("作家 ID の並びが run 間で不一致")
    focus = author_ids.index(args.focus_author)

    measured = [
        summarize_condition(spec, L, bal, args.similarity, c, focus, author_ids)
        for spec in specs
        for L in args.chunk_chars
        for bal in (True, False)
        for c in args.classifiers
    ]
    rows = [r for r, _ in measured]
    seed_rows = [s for _, ss in measured for s in ss]
    diffs = diff_rows(rows)

    args.out_dir.mkdir(parents=True)
    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "seeds.csv", seed_rows)
    write_csv(args.out_dir / "diff.csv", diffs)

    recall_cols = [f"recall_{a}" for a in author_ids]
    pm_cols = ["balanced_accuracy", "macro_f1", "map_macro", "focus_share_others"]

    def fmt(r: dict[str, Any]) -> dict[str, Any]:
        """シード平均 ± 標準偏差（間引きなしは平均のみ）の表示行．"""
        pm = lambda k: f"{r[k]:.3f}" if np.isnan(r[f"{k}_std"]) else f"{r[k]:.3f} ± {r[f'{k}_std']:.3f}"  # noqa: E731
        return {**r, **{k: pm(k) for k in pm_cols}}

    cols = ["condition", "chunk_chars", "classifier", "n_test", *pm_cols, "map_macro_random", *recall_cols, "focus_share_all", "n_ties"]
    dcols = ["condition", "chunk_chars", "classifier", "d_balanced_accuracy", "d_macro_f1", "d_map_macro", "d_map_macro_random",
             *[f"d_{c}" for c in recall_cols], "d_focus_share_others", "d_focus_share_all"]  # fmt: skip
    lines = [
        "# 漱石 2 作品版の補助実験：集計",
        "",
        f"- 分割：Inter-work（LOWO）の Pooled Out-of-Fold，類似度 `{args.similarity}`，推論器 {args.classifiers}",
        "- 主結果は間引きあり（bal，シード平均 ± 標準偏差，ddof = 1），間引きなし（nobal，決定的な 1 回）を併記",
        f"- focus_share_others / focus_share_all：`{args.focus_author}` 以外の作家のチャンク中 / 全 Test チャンク中で "
        f"`{args.focus_author}` と予測された割合",
        "- map_macro_random：作家別 MAP の平均のランダム基準（間引きの有無・除外の有無で Train 構成が変わるため条件ごとに異なる）",
        "- 差分は 除外版 − 9 作品版（同一特徴量・L・間引き・推論器のシード平均どうし）",
        f"- run：{', '.join(f'{r.label}={r.name}' for r in runs)}",
        "",
        *[
            line
            for f in args.features
            for bal in ("bal", "nobal")
            for sel in [[r for r in rows if r["feature"] == f and r["balance_train"] == bal]]
            for line in [
                f"## 特徴量 {f}（{bal}）：集計値",
                "",
                *md_table([fmt(r) for r in sel], cols),
                f"## 特徴量 {f}（{bal}）：9 作品版との差分",
                "",
                *md_table([r for r in diffs if r["feature"] == f and r["balance_train"] == bal], dcols, signed=True),
            ]
        ],
    ]
    report = "\n".join(lines)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")

    sources = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args) | {"outputs_dir": str(args.outputs_dir), "out_dir": str(args.out_dir)},
        "runs": [{"run": r.name, "condition": r.label, "features": r.meta["cli"]["features"], "excluded": list(r.excluded), "git": r.meta["git"]} for r in runs],
    }
    (args.out_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
