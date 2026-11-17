"""Train チャンク数の間引き補助実験（作家別に n_min 個へ間引き）の集計と 9 作品版（間引きなし）との比較．

Inter-work の Pooled Out-of-Fold 混同行列から，各チャンク長 L・推論器（既定：重心法と knn5_per_author）について
以下を run ごとに算出し（定義は scripts/summarize_exclusion.py の confusion_stats と同一），シード間の平均・標準偏差を求める：
    focus_share_others   注目作家（既定：漱石）以外の作家のチャンク中で注目作家と予測された割合（主指標）
    focus_share_all      全 Test チャンク中で注目作家と予測された割合
    accuracy, macro_f1   Pooled 指標
    recall_<作家>        作家別の再現率（同点は FN として分母のみに算入）
標準偏差は不偏標準偏差（ddof = 1）．差分は「間引き版のシード平均 − 9 作品版」．

使い方：
    uv run python scripts/summarize_balance.py
    uv run python scripts/summarize_balance.py --seeds 42 43 44 45 46 --features base C

前提となる run（outputs/ 以下）：
    9 作品版        --baseline で指定（既定：base=full_base，C=c_only）
    間引き版        bal_s<seed>_<特徴量>（scripts/run_experiments.py --splits inter --balance-train-seeds <seed> で作成）

出力（--out-dir，既定 outputs/balance_summary/）：
    runs.csv        全 run × L × 推論器の集計値（シード別）
    summary.csv     特徴量 × L × 推論器ごとのシード平均・標準偏差・9 作品版の値・差分
    nmin.csv        fold ごとの n_min とそれを決めた作家，間引き前後の作家別 Train チャンク数
    report.md       上記の Markdown 表
    sources.json    入力 run の git 状態・シード・生成時刻
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from summarize_exclusion import FULL_LABEL, RunSpec, md_table, parse_pairs, summarize_run

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_label(seed: int) -> str:
    return f"bal_s{seed}"


def load_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))


def check_balance_meta(meta: dict[str, Any], run_dir: Path, seed: int | None) -> None:
    """間引き版はシード一致と全 fold の間引き後件数 = n_min，9 作品版は間引き・除外なしを検証．"""
    if meta.get("excluded_works"):
        raise ValueError(f"{run_dir.name}: 作品除外 run は対象外（{meta['excluded_works']}）")
    bal = meta.get("balance_train")
    if seed is None:
        if bal is not None:
            raise ValueError(f"{run_dir.name}: 9 作品版に間引き記録がある")
        return
    if bal is None or bal["seed"] != seed:
        raise ValueError(f"{run_dir.name}: 間引きシードの記録 {bal and bal['seed']} が想定 {seed} と不一致")
    if sorted(map(int, bal["folds"])) != sorted(meta["cli"]["chunk_chars"]):
        raise ValueError(f"{run_dir.name}: 間引き記録の L {sorted(bal['folds'])} が実行 L と不一致")
    bad = [
        (L, r["fold"])
        for L, recs in bal["folds"].items()
        for r in recs
        if set(r["n_train_after"].values()) != {r["n_min"]} or not r["test_unchanged"]
    ]
    if bad:
        raise ValueError(f"{run_dir.name}: 間引き後の件数が n_min でない，または Test が変化した fold: {bad}")


def nmin_rows(metas: Sequence[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """fold ごとの n_min 記録を抽出し，全間引き run で同一（シードに依存しない）ことを検証．"""

    def rows_of(meta: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_chars": int(L),
                "fold": r["fold"],
                "test_work": r["test_work"],
                "n_min": r["n_min"],
                "n_min_authors": "+".join(r["n_min_authors"]),
                **{f"before_{a}": n for a, n in r["n_train_before"].items()},
                **{f"after_{a}": n for a, n in r["n_train_after"].items()},
            }
            for L, recs in sorted(meta["balance_train"]["folds"].items(), key=lambda kv: int(kv[0]))
            for r in recs
        ]

    ref_name, ref = metas[0][0], rows_of(metas[0][1])
    mismatched = [name for name, m in metas[1:] if rows_of(m) != ref]
    if mismatched:
        raise ValueError(f"n_min の記録が {ref_name} と異なる run: {mismatched}")
    return ref


METRICS_FIXED = ["focus_share_others", "focus_share_all", "accuracy", "macro_f1"]


def aggregate(rows: Sequence[dict[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """(feature, L, classifier) ごとにシード平均・標準偏差（ddof = 1）と 9 作品版との差分を算出．"""
    metrics = [*METRICS_FIXED, *[c for c in rows[0] if c.startswith("recall_")], "n_ties"]
    key = lambda r: (r["feature"], r["chunk_chars"], r["classifier"])  # noqa: E731
    base = {key(r): r for r in rows if r["condition"] == FULL_LABEL}
    groups = {k: [r for r in rows if r["condition"] != FULL_LABEL and key(r) == k] for k in base}

    def agg(k: tuple[str, int, str]) -> dict[str, Any]:
        g = groups[k]
        if sorted(r["seed"] for r in g) != sorted(seeds):
            raise ValueError(f"{k}: シードの過不足（{sorted(r['seed'] for r in g)}）")
        if len({r["n_test"] for r in g} | {base[k]["n_test"]}) != 1:
            raise ValueError(f"{k}: Test チャンク数が run 間で不一致")
        # shape: (num_metrics, num_seeds)
        values = np.array([[r[m] for r in g] for m in metrics], dtype=np.float64)
        assert np.all(np.isfinite(values)), f"{k}: NaN / Inf を含む"
        mean, std = values.mean(axis=1), values.std(axis=1, ddof=1)
        return {
            "feature": k[0],
            "chunk_chars": k[1],
            "classifier": k[2],
            "num_seeds": len(g),
            "n_test": base[k]["n_test"],
            **{f"{m}_mean": float(v) for m, v in zip(metrics, mean, strict=True)},
            **{f"{m}_std": float(v) for m, v in zip(metrics, std, strict=True)},
            **{f"{m}_9works": float(base[k][m]) for m in metrics},
            **{f"d_{m}": float(v - base[k][m]) for m, v in zip(metrics, mean, strict=True)},
        }

    return [agg(k) for k in base]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}: 出力行が空")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--features", nargs="+", default=["base", "C"])
    parser.add_argument("--baseline", nargs="+", default=["base=full_base", "C=c_only"], help="<特徴量>=<9 作品版の run 名>")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--classifiers", nargs="+", default=["centroid", "knn5_per_author"])
    parser.add_argument("--similarity", default="cosine")
    parser.add_argument("--focus-author", default="soseki", help="予測集中を測る作家 ID")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "balance_summary")
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError(f"--seeds は重複なしの 2 個以上（標準偏差の算出のため）: {args.seeds}")
    baseline = parse_pairs(args.baseline, "--baseline")
    missing = sorted(set(args.features) - set(baseline))
    if missing:
        raise ValueError(f"--baseline に 9 作品版の run が未指定の特徴量: {missing}")

    # (RunSpec, seed)：seed が None なら 9 作品版
    specs = [
        item
        for f in args.features
        for item in [
            (RunSpec(f, FULL_LABEL, args.outputs_dir / baseline[f], None), None),
            *((RunSpec(f, run_label(s), args.outputs_dir / f"{run_label(s)}_{f}", None), s) for s in args.seeds),
        ]
    ]
    metas = {spec.run_dir.name: load_meta(spec.run_dir) for spec, _ in specs}
    for spec, seed in specs:
        check_balance_meta(metas[spec.run_dir.name], spec.run_dir, seed)
    rows = [
        r | {"seed": seed if seed is not None else ""}
        for spec, seed in specs
        for r in summarize_run(spec, args.classifiers, args.similarity, args.focus_author)
    ]
    summary = aggregate(rows, args.seeds)
    bal_metas = [(spec.run_dir.name, metas[spec.run_dir.name]) for spec, seed in specs if seed is not None]
    nmin = nmin_rows(bal_metas)
    k_max = max(bal_metas[0][1]["config"]["classifier"]["k_values"])
    below_k = [r for r in nmin if r["n_min"] < k_max]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "runs.csv", rows)
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "nmin.csv", nmin)

    recall = [c.removeprefix("recall_") for c in rows[0] if c.startswith("recall_")]
    shown = ["focus_share_others", "accuracy", "macro_f1", *[f"recall_{a}" for a in recall]]

    def fmt_rows(f: str) -> tuple[list[dict[str, Any]], list[str]]:
        out = [
            {
                "L": s["chunk_chars"],
                "classifier": s["classifier"],
                **{
                    m: f"{s[f'{m}_mean']:.3f} ± {s[f'{m}_std']:.3f} ({s[f'{m}_9works']:.3f}, {s[f'd_{m}']:+.3f})"
                    for m in shown
                },
            }
            for s in summary
            if s["feature"] == f
        ]
        return out, ["L", "classifier", *shown]

    nmin_cols = ["chunk_chars", "fold", "test_work", "n_min", "n_min_authors", *[f"before_{a}" for a in recall]]
    lines = [
        "# Train チャンク数の間引き補助実験：集計",
        "",
        f"- 分割：Inter-work（LOWO）の Pooled Out-of-Fold，類似度 `{args.similarity}`，推論器 {args.classifiers}",
        f"- 間引き：各 fold で Train を作家別に n_min 個へ一様非復元抽出（シード {args.seeds}，fold 乱数は default_rng([seed, fold 番号])）",
        f"- focus_share_others：`{args.focus_author}` 以外の作家のチャンク中で `{args.focus_author}` と予測された割合（主指標）．"
        "目安は一様予測の 0.25",
        "- 表の各セル：シード平均 ± 不偏標準偏差（9 作品版の値, 平均 − 9 作品版）",
        "",
        *[
            line
            for f in args.features
            for line in [f"## 特徴量 {f}", "", *md_table(*fmt_rows(f))]
        ],
        "## fold ごとの n_min（全シード共通）",
        "",
        *md_table(nmin, nmin_cols),
        f"- n_min が classifier.k_values の最大値 {k_max} を下回る fold：{len(below_k)} 件"
        + (f" {[(r['chunk_chars'], r['fold'], r['n_min']) for r in below_k]}" if below_k else ""),
        "",
    ]
    report = "\n".join(lines)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")

    sources = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args) | {"outputs_dir": str(args.outputs_dir), "out_dir": str(args.out_dir)},
        "runs": [
            {"run": spec.run_dir.name, "feature": spec.feature, "seed": seed, "git": metas[spec.run_dir.name]["git"],
             "elapsed_seconds": metas[spec.run_dir.name]["elapsed_seconds"]}
            for spec, seed in specs
        ],
    }  # fmt: skip
    (args.out_dir / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"[output] {args.out_dir}")


if __name__ == "__main__":
    main()
