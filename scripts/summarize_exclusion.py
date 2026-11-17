"""漱石 2 作品版の補助実験（作品除外 run）と 9 作品版の比較表を作成．

Inter-work の Pooled Out-of-Fold 混同行列（行：正解，列：予測，最右列は同点）から，各チャンク長 L・
推論器（既定：重心法と knn5_per_author）について以下を集計：
    accuracy, macro_f1          Pooled 指標（results.json の値を混同行列からの再計算値と照合）
    recall_<作家>               作家別の再現率（同点は FN として分母のみに算入）
    focus_share_all             全 Test チャンク中で注目作家（既定：漱石）と予測された割合
    focus_share_others          注目作家以外の作家のチャンク中で注目作家と予測された割合
    majority_author / share     全 Test チャンク中で最多の作家とその割合（全予測をその作家とした場合の Accuracy）

使い方：
    uv run python scripts/summarize_exclusion.py
    uv run python scripts/summarize_exclusion.py --features base C --exclusions bocchan=坊っちゃん kokoro=こころ

前提となる run（outputs/ 以下）：
    9 作品版        --baseline で指定（既定：base=full_base，C=c_only）
    除外版          excl_<略称>_<特徴量>（scripts/run_experiments.py --exclude-works <作品名> で作成）

出力（--out-dir，既定 outputs/exclusion_summary/）：
    summary.csv     全 run × L × 推論器の集計値
    diff.csv        除外版 − 9 作品版の差分（majority_author は除外版の値）
    report.md       上記の Markdown 表
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


@dataclass(frozen=True)
class RunSpec:
    """集計対象の 1 run．excluded_title が None なら 9 作品版．"""

    feature: str
    label: str
    run_dir: Path
    excluded_title: str | None


def parse_pairs(items: Sequence[str], what: str) -> dict[str, str]:
    bad = [s for s in items if s.count("=") != 1 or not all(s.split("="))]
    if bad:
        raise ValueError(f"{what} は <キー>=<値> 形式で指定: {bad}")
    return dict(s.split("=") for s in items)


def load_run(spec: RunSpec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """run_meta.json と results.json を読み込み，除外作品の記録が想定と一致するか検証．"""
    if not spec.run_dir.is_dir():
        raise FileNotFoundError(f"run が存在しない: {spec.run_dir}")
    meta = json.loads((spec.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if "finished_at" not in meta:
        raise ValueError(f"未完了の run: {spec.run_dir}")
    recorded = [w["title"] for w in meta.get("excluded_works", [])]
    expected = [spec.excluded_title] if spec.excluded_title else []
    if recorded != expected:
        raise ValueError(f"{spec.run_dir.name}: 除外作品の記録 {recorded} が想定 {expected} と不一致")
    if spec.feature not in meta["cli"]["features"]:
        raise ValueError(f"{spec.run_dir.name}: 特徴量 {spec.feature} を含まない（{meta['cli']['features']}）")
    results = json.loads((spec.run_dir / "results.json").read_text(encoding="utf-8"))["results"]
    return meta, results


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
        **{f"recall_{a}": float(r) for a, r in zip(author_ids, recall, strict=True)},
        "focus_share_all": float(cm[:, focus].sum() / n),
        "focus_share_others": float(cm[others, focus].sum() / cm[others].sum()),
        "majority_author": author_ids[int(np.argmax(true_count))],
        "majority_share": float(true_count.max() / n),
    }


def summarize_run(spec: RunSpec, classifiers: Sequence[str], similarity: str, focus_author: str) -> list[dict[str, Any]]:
    meta, results = load_run(spec)
    author_ids = list(meta["author_ids"])
    focus = author_ids.index(focus_author)
    picked = [
        d
        for d in results
        if d["feature"] == spec.feature
        and d["is_default"]
        and d["split"] == "inter"
        and d["similarity"] == similarity
        and d["classifier"] in classifiers
    ]
    found = {(d["chunk_chars"], d["classifier"]) for d in picked}
    expected = {(L, c) for L in meta["cli"]["chunk_chars"] for c in classifiers}
    if found != expected or len(picked) != len(expected):
        raise ValueError(f"{spec.run_dir.name}: 条件の過不足（不足 {sorted(expected - found)}，重複 {len(picked) - len(found)}）")

    def row(d: dict[str, Any]) -> dict[str, Any]:
        stats = confusion_stats(np.asarray(d["confusion"], dtype=np.int64), focus, author_ids)
        assert np.isclose(stats.pop("accuracy_cm"), d["accuracy"], rtol=0, atol=1e-12), f"Accuracy 不一致: {d['key']}"
        assert stats["n_test"] == d["n_test"], f"n_test 不一致: {d['key']}"
        recall_stored = np.asarray(d["recall"])
        recall_cm = np.array([stats[f"recall_{a}"] for a in author_ids])
        assert np.allclose(recall_stored, recall_cm, rtol=0, atol=1e-12), f"Recall 不一致: {d['key']}"
        return {
            "feature": spec.feature,
            "condition": spec.label,
            "excluded": spec.excluded_title or "",
            "run": spec.run_dir.name,
            "chunk_chars": d["chunk_chars"],
            "classifier": d["classifier"],
            "num_folds": d["num_folds"],
            "accuracy": d["accuracy"],
            "macro_f1": d["macro_f1"],
            **stats,
        }

    return sorted(map(row, picked), key=lambda r: (r["chunk_chars"], classifiers.index(r["classifier"])))


NON_DIFF = {"feature", "condition", "excluded", "run", "chunk_chars", "classifier", "majority_author"}


def diff_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """除外版の各行から，同じ (feature, L, classifier) の 9 作品版の値を引く．"""
    key = lambda r: (r["feature"], r["chunk_chars"], r["classifier"])  # noqa: E731
    base = {key(r): r for r in rows if r["condition"] == FULL_LABEL}
    return [
        {
            **{k: r[k] for k in ("feature", "condition", "excluded", "chunk_chars", "classifier", "majority_author")},
            **{f"d_{k}": v - base[key(r)][k] for k, v in r.items() if k not in NON_DIFF},
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--features", nargs="+", default=["base", "C"])
    parser.add_argument("--baseline", nargs="+", default=["base=full_base", "C=c_only"], help="<特徴量>=<9 作品版の run 名>")
    parser.add_argument(
        "--exclusions",
        nargs="+",
        default=["bocchan=坊っちゃん", "kokoro=こころ", "sanshiro=三四郎"],
        help="<略称>=<除外作品名>（run 名は excl_<略称>_<特徴量>）",
    )
    parser.add_argument("--classifiers", nargs="+", default=["centroid", "knn5_per_author"])
    parser.add_argument("--similarity", default="cosine")
    parser.add_argument("--focus-author", default="soseki", help="予測集中を測る作家 ID")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "exclusion_summary")
    args = parser.parse_args()

    baseline = parse_pairs(args.baseline, "--baseline")
    exclusions = parse_pairs(args.exclusions, "--exclusions")
    missing = sorted(set(args.features) - set(baseline))
    if missing:
        raise ValueError(f"--baseline に 9 作品版の run が未指定の特徴量: {missing}")

    specs = [
        spec
        for f in args.features
        for spec in [
            RunSpec(f, FULL_LABEL, args.outputs_dir / baseline[f], None),
            *(RunSpec(f, f"excl_{s}", args.outputs_dir / f"excl_{s}_{f}", t) for s, t in exclusions.items()),
        ]
    ]
    rows = [r for spec in specs for r in summarize_run(spec, args.classifiers, args.similarity, args.focus_author)]
    diffs = diff_rows(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "diff.csv", diffs)

    recall_cols = [c for c in rows[0] if c.startswith("recall_")]
    cols = ["condition", "chunk_chars", "classifier", "n_test", "accuracy", "macro_f1", *recall_cols,
            "focus_share_all", "focus_share_others", "majority_author", "majority_share", "n_ties"]  # fmt: skip
    dcols = ["condition", "chunk_chars", "classifier", "d_accuracy", "d_macro_f1", *[f"d_{c}" for c in recall_cols],
             "d_focus_share_all", "d_focus_share_others", "d_majority_share"]  # fmt: skip
    lines = [
        "# 漱石 2 作品版の補助実験：集計",
        "",
        f"- 分割：Inter-work（LOWO）の Pooled Out-of-Fold，類似度 `{args.similarity}`，推論器 {args.classifiers}",
        f"- focus_share_all / focus_share_others：全 Test チャンク中 / `{args.focus_author}` 以外の作家のチャンク中で "
        f"`{args.focus_author}` と予測された割合",
        "- majority_share：全 Test チャンク中で最多の作家の割合（全予測をその作家とした場合の Accuracy）",
        "- 差分は 除外版 − 9 作品版（同一特徴量・L・推論器）",
        "",
        *[
            line
            for f in args.features
            for line in [
                f"## 特徴量 {f}：集計値",
                "",
                *md_table([r for r in rows if r["feature"] == f], cols),
                f"## 特徴量 {f}：9 作品版との差分",
                "",
                *md_table([r for r in diffs if r["feature"] == f], dcols, signed=True),
            ]
        ],
    ]
    report = "\n".join(lines)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")

    def source(spec: RunSpec) -> dict[str, Any]:
        meta = json.loads((spec.run_dir / "run_meta.json").read_text(encoding="utf-8"))
        return {"run": spec.run_dir.name, "feature": spec.feature, "excluded": spec.excluded_title, "git": meta["git"]}

    sources = {"generated_at": datetime.now().isoformat(timespec="seconds"), "args": vars(args) | {
        "outputs_dir": str(args.outputs_dir), "out_dir": str(args.out_dir)}, "runs": list(map(source, specs))}  # fmt: skip
    (args.out_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
