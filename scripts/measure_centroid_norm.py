"""重心の長さと作家内類似度の実測（重心法の漱石への予測集中の記述統計）．

長さ 1 に正規化した Train ベクトル d̂ = d / ||d||_2（ゼロベクトルは 0 のまま）の作家別平均を
ĉ_a = (1/n_a) Σ_{d∈D_a} d̂ とすると，

    (1/n_a) Σ_{d∈D_a} cos(q, d) = ||ĉ_a|| cos(q, ĉ_a)                  （恒等式）
    ||ĉ_a||^2 = 1/n_a + (1 - 1/n_a) ρ̄_a                               （ゼロベクトルなしの場合）

ρ̄_a は作家 a の Train チャンク同士（自分自身を除く）の平均コサイン．重心とのコサイン cos(q, ĉ_a) は
作家内の平均類似度を ||ĉ_a|| で割り増しした量で，割り増し係数は 1 / ||ĉ_a||．
実装の重心法（src/classifier.py の CentroidClassifier）は正規化前のベクトルの平均 c_a を使うため，
上の関係は近似．その程度として ||c_a||_2，|cos(q, c_a) - cos(q, ĉ_a)|，両重心による予測の一致率を併記．

対象：Inter-work（LOWO）の各 fold，configs/default.yaml の本採用パラメータで各 fold の Train に fit．
Train 構成は「間引きなし」（none）と間引きあり（bal_s<seed>）．間引きは旧 run に依存せず，
src.dataset.balance_train_split と np.random.default_rng([seed, fold 番号])（scripts/run_experiments.py の
Inter-work と同じ乱数，plans/research-context.md「Train チャンク数の作家間均等化」）でこのスクリプト内で再現．

照合する run（--runs <特徴量>=<run 名>）：scripts/run_experiments.py の現行形式（split.balance_train: true で
間引きあり・なしを同じ run に含む）の 9 作品版．特徴量の本採用パラメータが現在の config と一致すること．
間引きシードの既定は照合 run の run_meta.json（balance_train.inter.seeds）．

再現の検証（不一致は即停止）：
    fold ごとの n_min と作家別 Train 数（間引き前・後）が照合 run の run_meta.json（balance_train.inter.folds）と一致
    正規化前の重心による予測の Pooled 混同行列が照合 run の results.json（centroid，同じ間引き条件・シードの
    row_type=seed の行）と完全一致
    fold ごとの Accuracy・n_train・mu_same・mu_diff（ミクロ）が照合 run の folds.csv（同条件）と一致

使い方：
    uv run python scripts/measure_centroid_norm.py --features base C --chunk-chars 500 --runs base=base_c_022 C=base_c_022
    uv run python scripts/measure_centroid_norm.py --features B --chunk-chars 500 --runs B=smoke_b_allL --seeds 42 43

出力（--out-dir，既定 outputs/centroid_norm/．既存のディレクトリには書き込まず停止）：
    author_stats.csv     特徴量 × L × Train 構成（none / bal_s<seed>）× fold × 作家の全量
    author_summary.csv   特徴量 × L × fold × 作家（間引きありは 5 シードの平均・不偏標準偏差）
    fold_stats.csv       fold ごとの恒等式の最大絶対誤差，予測一致率，漱石予測割合，再現検証値
    focus_vs_inflation.csv  割り増し係数と漱石以外のチャンク中の漱石予測割合の対応表（fold 単位）
    pooled.csv           特徴量 × L × Train 構成の Pooled 漱石予測割合（再計算値と results.json の値）
    report.md            上記の Markdown 表
    run_meta.json        設定・CLI 引数・git 状態・コーパスハッシュ・所要時間
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from run_experiments import INTER_BALANCE_RNG, FeatureVariant, build_variants, corpus_meta, display_path, git_state
from summarize_exclusion import md_table, parse_pairs

from src.classifier import CentroidClassifier, predict_from_scores, tie_label
from src.dataset import ChunkSet, Split, balance_train_split, build_chunks, inter_work_splits, load_corpus
from src.evaluator import confusion_matrix
from src.similarity import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NONE_LABEL = "none"
# 既存 run の値との照合の許容誤差（同一コード・同一データのため浮動小数点の加算順の差のみ）
MATCH_ATOL = 1e-12


@dataclass(frozen=True)
class TrainConfig:
    """Train 構成．seed が None なら間引きなし．"""

    label: str
    seed: int | None

    @property
    def balanced(self) -> bool:
        return self.seed is not None


@dataclass(frozen=True)
class RefRun:
    """照合する run（現行形式）．"""

    name: str
    meta: dict[str, Any]
    results: list[dict[str, Any]]
    folds: list[dict[str, str]]


# ---------------------------------------------------------------- 数値計算


def unit_rows(X: np.ndarray) -> np.ndarray:
    """行ごとに L2 正規化（ノルム 0 の行は 0 のまま）．shape: (n, M) → (n, M)"""
    # shape: (n, 1)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


def author_stats(X_a: np.ndarray, Q: np.ndarray, eps: float) -> dict[str, Any]:
    """作家 a の Train 行列 X_a: (n_a, M) と照会 Q: (n_test, M) から重心の長さ・作家内類似度を算出．"""
    n = X_a.shape[0]
    if n < 2:
        raise ValueError(f"Train チャンクが 2 未満のため ρ̄ を定義できない: n_a = {n}")
    # shape: (n_a, M)
    D_hat = unit_rows(X_a)
    n_zero = int(np.sum(~np.any(D_hat != 0, axis=1)))
    # shape: (n_a, n_a)，対角（自己類似度）を除いた平均
    G = D_hat @ D_hat.T
    rho = float((G.sum() - np.trace(G)) / (n * (n - 1)))
    # shape: (M,)
    c_hat = D_hat.mean(axis=0)
    c_raw = X_a.mean(axis=0)
    c_hat_norm = float(np.linalg.norm(c_hat))
    if c_hat_norm == 0:
        raise ValueError("正規化後の重心がゼロベクトル（全 Train チャンクがゼロベクトル）")
    term_inv_n, term_rho = 1.0 / n, (1.0 - 1.0 / n) * rho
    # 恒等式：左辺 (1/n_a) Σ_d cos(q, d)，右辺 ||ĉ_a|| cos(q, ĉ_a)．shape: (n_test,)
    lhs = cosine_similarity(Q, X_a, eps=eps).mean(axis=1)
    cos_hat = cosine_similarity(Q, c_hat[None, :], eps=eps)[:, 0]
    cos_raw = cosine_similarity(Q, c_raw[None, :], eps=eps)[:, 0]
    rhs = c_hat_norm * cos_hat
    assert np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs)), "恒等式の両辺に NaN / Inf"
    return {
        "n_a": n,
        "n_zero_train": n_zero,
        "rho_bar": rho,
        "c_hat_norm": c_hat_norm,
        "inflation": 1.0 / c_hat_norm,
        "term_inv_n": term_inv_n,
        "term_rho": term_rho,
        # ||ĉ_a||^2 に占める 1/n_a の項の割合
        "inv_n_share": term_inv_n / c_hat_norm**2,
        "norm_formula_abs_err": abs(c_hat_norm**2 - (term_inv_n + term_rho)),
        "c_raw_norm": float(np.linalg.norm(c_raw)),
        "cos_raw_minus_hat_mean_abs": float(np.abs(cos_raw - cos_hat).mean()),
        "cos_raw_minus_hat_max_abs": float(np.abs(cos_raw - cos_hat).max()),
        "_identity_err": float(np.abs(lhs - rhs).max()),
        "_c_hat": c_hat,
    }


# ---------------------------------------------------------------- 再現の検証


def load_run(outputs_dir: Path, name: str) -> RefRun:
    run_dir = outputs_dir / name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"照合する run が存在しない: {run_dir}")
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    if "finished_at" not in meta:
        raise ValueError(f"未完了の run: {run_dir}")
    if meta.get("excluded_works"):
        raise ValueError(f"{name}: 作品除外 run は照合対象外")
    bal = meta.get("balance_train")
    # 022 以前の形式は balance_train が {seed, folds}（単一シード）または無し
    if not (isinstance(bal, dict) and "inter" in bal and "inter" in meta["cli"]["splits"]):
        raise ValueError(f"{name}: 現行形式（間引きあり・なしを含む Inter-work）の run ではない（balance_train={type(bal).__name__}）")
    if bal["inter"]["rng"] != INTER_BALANCE_RNG:
        raise ValueError(f"{name}: Inter-work の間引き乱数の記録 {bal['inter']['rng']!r} が {INTER_BALANCE_RNG!r} と不一致")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["results"]
    with (run_dir / "folds.csv").open(encoding="utf-8") as f:
        folds = list(csv.DictReader(f))
    return RefRun(name, meta, results, folds)


def check_run_compat(run: RefRun, variant: FeatureVariant, chunk_chars: Sequence[int], seeds: Sequence[int]) -> None:
    """照合 run が対象の特徴量（同一の本採用パラメータ）・L・シードを含むか検証．"""
    meta = run.meta
    recorded = {v["name"]: v["params"] for v in meta["variants"]}
    if recorded.get(variant.name) != variant.params:
        raise ValueError(f"{run.name}: {variant.name} のパラメータ {recorded.get(variant.name)} が現在の config {variant.params} と不一致")
    missing_L = sorted(set(chunk_chars) - set(meta["cli"]["chunk_chars"]))
    missing_seeds = sorted(set(seeds) - set(meta["balance_train"]["inter"]["seeds"]))
    if missing_L or missing_seeds:
        raise ValueError(f"{run.name}: 照合に必要な L {missing_L} / 間引きシード {missing_seeds} を含まない")


def check_balance_record(run: RefRun, cfg: TrainConfig, L: int, records: Sequence[dict[str, Any]]) -> None:
    """再現した fold ごとの n_min・作家別 Train 数が照合 run の run_meta.json と一致するか検証．

    間引きありは同じシードの記録と間引き前・後の両方を，間引きなしは全シードの記録と間引き前の件数を照合．
    """
    recorded = run.meta["balance_train"]["inter"]["folds"][str(L)]
    targets = {str(cfg.seed): recorded[str(cfg.seed)]} if cfg.balanced else recorded
    keys = ("fold_index", "fold", "n_min", "n_train_before", *(("n_train_after",) if cfg.balanced else ()))
    bad_len = sorted(s for s, stored in targets.items() if len(stored) != len(records))
    if bad_len:
        raise ValueError(f"{run.name} L={L} {cfg.label}: fold 数が run_meta.json（seed {bad_len}）と不一致（{len(records)}）")
    mismatch = [
        (s, r["fold"]) for s, stored in targets.items() for r, t in zip(records, stored, strict=True) if any(r[k] != t[k] for k in keys)
    ]
    if mismatch:
        raise ValueError(f"{run.name} L={L} {cfg.label}: 再現した作家別 Train 数・n_min が run_meta.json と不一致: {mismatch}")


def stored_centroid(run: RefRun, feature: str, L: int, cfg: TrainConfig) -> dict[str, Any]:
    """results.json のうち同じ間引き条件・シードの Inter-work 重心法の 1 シード分の行．"""
    stored = [
        d
        for d in run.results
        if d["variant"] == feature
        and d["chunk_chars"] == L
        and d["split"] == "inter"
        and d["classifier"] == "centroid"
        and d["similarity"] == "cosine"
        and d["row_type"] == "seed"
        and d["balance_train"] is cfg.balanced
        and d["seed"] == cfg.seed
    ]
    if len(stored) != 1:
        raise ValueError(f"{run.name}: results.json の {feature}/L{L}/inter/{cfg.label}/centroid が {len(stored)} 件")
    return stored[0]


def check_against_run(
    feature: str, L: int, cfg: TrainConfig, fold_rows: Sequence[dict[str, Any]], pooled_cm: np.ndarray, run: RefRun
) -> None:
    """再計算した正規化前の重心法の結果が照合 run（同一 Train）と一致するか検証．"""
    stored = stored_centroid(run, feature, L, cfg)
    if not np.array_equal(np.asarray(stored["confusion"], dtype=np.int64), pooled_cm):
        raise ValueError(f"{run.name} {feature} L={L} {cfg.label}: Pooled 混同行列が results.json と不一致")
    seed_str = "" if cfg.seed is None else str(cfg.seed)
    ref = {
        r["fold"]: r
        for r in run.folds
        if r["variant"] == feature
        and int(r["chunk_chars"]) == L
        and r["split"] == "inter"
        and r["classifier"] == "centroid"
        and r["similarity"] == "cosine"
        and r["balance_train"] == str(cfg.balanced)
        and r["seed"] == seed_str
    }
    if set(ref) != {r["fold"] for r in fold_rows} or len(ref) != len(fold_rows):
        raise ValueError(f"{run.name} {feature} L={L} {cfg.label}: folds.csv の fold 集合が不一致")
    bad = [
        (r["fold"], k)
        for r in fold_rows
        for k, ours in (("n_train", r["n_train"]), ("accuracy", r["accuracy"]), ("mu_same", r["mu_same"]), ("mu_diff", r["mu_diff"]))
        if not np.isclose(float(ref[r["fold"]][k]), ours, rtol=0, atol=MATCH_ATOL)
    ]
    if bad:
        raise ValueError(f"{run.name} {feature} L={L} {cfg.label}: folds.csv と不一致: {bad}")


# ---------------------------------------------------------------- 1 fold の計測


def measure_fold(
    variant: FeatureVariant,
    cs: ChunkSet,
    split: Split,
    eps: float,
    identity_tol: float,
    focus: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    texts, labels = cs.texts, cs.labels
    C = len(cs.author_ids)
    extractor = variant.make()
    # Train のみで fit，Test は transform のみ（run_experiments.py と同一）
    # shape: (n_train, M), (n_test, M)
    X_train = extractor.fit([texts[i] for i in split.train_idx]).transform([texts[i] for i in split.train_idx])
    X_test = extractor.transform([texts[i] for i in split.test_idx])
    y_train, y_test = labels[split.train_idx], labels[split.test_idx]

    stats = [author_stats(X_train[y_train == a], X_test, eps) for a in range(C)]
    identity_err = max(s.pop("_identity_err") for s in stats)
    if not identity_err < identity_tol:
        raise ValueError(f"[{split.name}] 恒等式の最大絶対誤差 {identity_err:.3e} が許容値 {identity_tol:.0e} 以上")
    # shape: (C, M)
    C_hat = np.stack([s.pop("_c_hat") for s in stats])

    similarity = lambda A, B: cosine_similarity(A, B, eps=eps)  # noqa: E731
    # 実装の重心法（正規化前の重心）と正規化後の重心による予測．shape: (n_test,)
    y_raw = predict_from_scores(CentroidClassifier(similarity, C).fit(X_train, y_train).scores(X_test))
    y_hat = predict_from_scores(similarity(X_test, C_hat))
    # 既存 run との照合用の類似度統計（ミクロ平均）．shape: (n_test, n_train)
    S = similarity(X_test, X_train)
    same = y_test[:, None] == y_train[None, :]
    others = y_test != focus
    fold_row = {
        "fold": split.name,
        "test_author_label": int(y_test[0]),
        "n_train": int(split.train_idx.size),
        "n_test": int(split.test_idx.size),
        "n_features": int(X_train.shape[1]),
        "n_zero_test": int(np.sum(~np.any(X_test != 0, axis=1))),
        "train_idx_sha256": hashlib.sha256(np.ascontiguousarray(split.train_idx, dtype=np.int64).tobytes()).hexdigest()[:16],
        "identity_max_abs_err": identity_err,
        "pred_agreement_raw_vs_hat": float(np.mean(y_raw == y_hat)),
        "accuracy": float(np.mean(y_raw == y_test)),
        "accuracy_hat": float(np.mean(y_hat == y_test)),
        "n_ties": int(np.sum(y_raw == tie_label(C))),
        # 漱石以外のチャンクが無い fold（漱石の作品が Test）は NaN
        "focus_share_others": float(np.mean(y_raw[others] == focus)) if others.any() else float("nan"),
        "focus_share_others_hat": float(np.mean(y_hat[others] == focus)) if others.any() else float("nan"),
        "mu_same": float(S[same].mean()),
        "mu_diff": float(S[~same].mean()),
    }
    author_rows = [{"author_label": a, **s} for a, s in enumerate(stats)]
    return author_rows, fold_row, confusion_matrix(y_test, y_raw, C)


def measure_condition(
    variant: FeatureVariant,
    cs: ChunkSet,
    cfg: TrainConfig,
    run: RefRun,
    eps: float,
    identity_tol: float,
    focus: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    labels, author_ids = cs.labels, cs.author_ids
    C = len(author_ids)
    splits = inter_work_splits(cs)
    # 間引き：fold ごとに default_rng([seed, fold 番号])（run_experiments.py の INTER_BALANCE_RNG と同じ）
    used = (
        [balance_train_split(s, labels, np.random.default_rng([cfg.seed, i])) for i, s in enumerate(splits)]
        if cfg.balanced
        else splits
    )

    def counts(idx: np.ndarray) -> dict[str, int]:
        return dict(zip(author_ids, np.bincount(labels[idx], minlength=C).tolist(), strict=True))

    records = [
        {
            "fold_index": i,
            "fold": b.name,
            "n_min": min(counts(b.train_idx).values()),
            "n_train_before": counts(b.train_idx),
            "n_train_after": counts(a.train_idx),
        }
        for i, (b, a) in enumerate(zip(splits, used, strict=True))
    ]
    L = cs.target_chunk_chars
    check_balance_record(run, cfg, L, records)

    measured = [measure_fold(variant, cs, s, eps, identity_tol, focus) for s in used]
    head = {
        "feature": variant.name,
        "chunk_chars": L,
        "train_config": cfg.label,
        "seed": cfg.seed if cfg.seed is not None else "",
        "n_min_mean": float(np.mean([r["n_min"] for r in records])),
    }
    fold_rows = [head | f | {"n_min": r["n_min"]} for (_, f, _), r in zip(measured, records, strict=True)]
    # shape: (C, C + 1)
    pooled_cm = np.sum([cm for _, _, cm in measured], axis=0)
    check_against_run(variant.name, L, cfg, fold_rows, pooled_cm, run)

    author_rows = [
        head | {"fold": f["fold"], "test_author": author_ids[f["test_author_label"]], "author": author_ids[r.pop("author_label")]} | r
        for rows, f, _ in measured
        for r in rows
    ]
    others = np.arange(C) != focus
    stored = stored_centroid(run, variant.name, L, cfg)
    stored_cm = np.asarray(stored["confusion"], dtype=np.int64)
    pooled = head | {
        "focus_share_others": float(pooled_cm[others, focus].sum() / pooled_cm[others].sum()),
        "focus_share_others_results_json": float(stored_cm[others, focus].sum() / stored_cm[others].sum()),
        "accuracy": float(np.trace(pooled_cm[:, :C]) / pooled_cm.sum()),
        "accuracy_results_json": float(stored["accuracy"]),
        "identity_max_abs_err": max(f["identity_max_abs_err"] for f in fold_rows),
        "pred_agreement_raw_vs_hat": float(np.mean([f["pred_agreement_raw_vs_hat"] for f in fold_rows])),
    }
    return author_rows, fold_rows, pooled


# ---------------------------------------------------------------- 集計・出力

AUTHOR_METRICS = [
    "n_a",
    "n_zero_train",
    "rho_bar",
    "c_hat_norm",
    "inflation",
    "term_inv_n",
    "term_rho",
    "inv_n_share",
    "norm_formula_abs_err",
    "c_raw_norm",
    "cos_raw_minus_hat_mean_abs",
    "cos_raw_minus_hat_max_abs",
]


def summarize_authors(rows: Sequence[dict[str, Any]], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """fold × 作家ごとに，none はそのまま，間引きありはシード平均と不偏標準偏差（ddof = 1）．"""
    key = lambda r: (r["feature"], r["chunk_chars"], r["fold"], r["author"])  # noqa: E731
    none = {key(r): r for r in rows if r["train_config"] == NONE_LABEL}

    def row(k: tuple[str, int, str, str]) -> dict[str, Any]:
        bal = [r for r in rows if r["train_config"] != NONE_LABEL and key(r) == k]
        if sorted(r["seed"] for r in bal) != sorted(seeds):
            raise ValueError(f"{k}: シードの過不足（{sorted(r['seed'] for r in bal)}）")
        # shape: (num_metrics, num_seeds)
        values = np.array([[r[m] for r in bal] for m in AUTHOR_METRICS], dtype=np.float64)
        assert np.all(np.isfinite(values)), f"{k}: NaN / Inf を含む"
        mean, std = values.mean(axis=1), values.std(axis=1, ddof=1)
        return {
            "feature": k[0],
            "chunk_chars": k[1],
            "fold": k[2],
            "test_author": none[k]["test_author"],
            "author": k[3],
            **{f"none_{m}": none[k][m] for m in AUTHOR_METRICS},
            **{f"bal_{m}_mean": float(v) for m, v in zip(AUTHOR_METRICS, mean, strict=True)},
            **{f"bal_{m}_std": float(v) for m, v in zip(AUTHOR_METRICS, std, strict=True)},
        }

    return [row(k) for k in none]


def focus_table(
    author_rows: Sequence[dict[str, Any]], fold_rows: Sequence[dict[str, Any]], focus_author: str, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    """漱石以外の作品が Test の fold について，割り増し係数と漱石予測割合の対応（間引きありはシード平均・標準偏差）．"""
    infl = {(r["feature"], r["chunk_chars"], r["train_config"], r["fold"], r["author"]): r["inflation"] for r in author_rows}
    targets = [f for f in fold_rows if f["train_config"] == NONE_LABEL and not np.isnan(f["focus_share_others"])]

    def per_config(f: dict[str, Any], label: str) -> dict[str, float]:
        k = (f["feature"], f["chunk_chars"], label, f["fold"])
        test_author = next(r["test_author"] for r in author_rows if r["fold"] == f["fold"])
        share = next(g["focus_share_others"] for g in fold_rows if (g["feature"], g["chunk_chars"], g["train_config"], g["fold"]) == k)
        return {
            "infl_focus": infl[(*k, focus_author)],
            "infl_test_author": infl[(*k, test_author)],
            "infl_ratio": infl[(*k, focus_author)] / infl[(*k, test_author)],
            "focus_share": share,
        }

    def row(f: dict[str, Any]) -> dict[str, Any]:
        none = per_config(f, NONE_LABEL)
        bal = [per_config(f, f"bal_s{s}") for s in seeds]
        # shape: (num_seeds, num_metrics)
        values = np.array([[b[m] for m in none] for b in bal], dtype=np.float64)
        return {
            "feature": f["feature"],
            "chunk_chars": f["chunk_chars"],
            "fold": f["fold"],
            "test_author": next(r["test_author"] for r in author_rows if r["fold"] == f["fold"]),
            **{f"none_{m}": v for m, v in none.items()},
            **{f"bal_{m}_mean": float(v) for m, v in zip(none, values.mean(axis=0), strict=True)},
            **{f"bal_{m}_std": float(v) for m, v in zip(none, values.std(axis=0, ddof=1), strict=True)},
        }

    return [row(f) for f in targets]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"{path.name}: 出力行が空")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    summary: Sequence[dict[str, Any]],
    focus: Sequence[dict[str, Any]],
    pooled: Sequence[dict[str, Any]],
    fold_rows: Sequence[dict[str, Any]],
    author_ids: Sequence[str],
    args: argparse.Namespace,
    elapsed: float,
) -> str:
    def pm(r: dict[str, Any], m: str, fmt: str = ".4f") -> str:
        return f"{r[f'bal_{m}_mean']:{fmt}} ± {r[f'bal_{m}_std']:{fmt}}"

    def author_avg(feature: str, L: int) -> list[dict[str, Any]]:
        """fold 平均（間引きありはシード平均の fold 平均）．"""

        def one(a: str) -> dict[str, Any]:
            rs = [r for r in summary if r["feature"] == feature and r["chunk_chars"] == L and r["author"] == a]
            avg = lambda c: float(np.mean([r[c] for r in rs]))  # noqa: E731
            return {
                "author": a,
                "n_a (none / bal)": f"{avg('none_n_a'):.1f} / {avg('bal_n_a_mean'):.1f}",
                "ρ̄ (none / bal)": f"{avg('none_rho_bar'):.4f} / {avg('bal_rho_bar_mean'):.4f}",
                "‖ĉ‖ (none / bal)": f"{avg('none_c_hat_norm'):.4f} / {avg('bal_c_hat_norm_mean'):.4f}",
                "1/‖ĉ‖ (none / bal)": f"{avg('none_inflation'):.3f} / {avg('bal_inflation_mean'):.3f}",
                "1/n 項の割合 (none / bal)": f"{avg('none_inv_n_share'):.3f} / {avg('bal_inv_n_share_mean'):.3f}",
                "‖c‖ (none / bal)": f"{avg('none_c_raw_norm'):.4g} / {avg('bal_c_raw_norm_mean'):.4g}",
            }

        return [one(a) for a in author_ids]

    def detail(feature: str, L: int) -> list[dict[str, Any]]:
        return [
            {
                "fold": r["fold"],
                "author": r["author"],
                "n_a": f"{r['none_n_a']} / {r['bal_n_a_mean']:.0f}",
                "ρ̄ none": f"{r['none_rho_bar']:.4f}",
                "ρ̄ bal": pm(r, "rho_bar"),
                "‖ĉ‖ none": f"{r['none_c_hat_norm']:.4f}",
                "‖ĉ‖ bal": pm(r, "c_hat_norm"),
                "1/‖ĉ‖ none": f"{r['none_inflation']:.3f}",
                "1/‖ĉ‖ bal": pm(r, "inflation", ".3f"),
                "1/n : ρ̄ 項 none": f"{r['none_term_inv_n']:.2e} : {r['none_term_rho']:.2e}",
                "1/n : ρ̄ 項 bal": f"{r['bal_term_inv_n_mean']:.2e} : {r['bal_term_rho_mean']:.2e}",
                "‖c‖ none": f"{r['none_c_raw_norm']:.4g}",
            }
            for r in summary
            if r["feature"] == feature and r["chunk_chars"] == L
        ]

    focus_fmt = [
        {
            "feature": r["feature"],
            "L": r["chunk_chars"],
            "fold": r["fold"],
            "test_author": r["test_author"],
            "1/‖ĉ_漱石‖ none": f"{r['none_infl_focus']:.3f}",
            "1/‖ĉ_test‖ none": f"{r['none_infl_test_author']:.3f}",
            "比 none": f"{r['none_infl_ratio']:.3f}",
            "漱石予測割合 none": f"{r['none_focus_share']:.3f}",
            "1/‖ĉ_漱石‖ bal": pm(r, "infl_focus", ".3f"),
            "1/‖ĉ_test‖ bal": pm(r, "infl_test_author", ".3f"),
            "比 bal": pm(r, "infl_ratio", ".3f"),
            "漱石予測割合 bal": pm(r, "focus_share", ".3f"),
        }
        for r in focus
    ]
    pooled_fmt = [
        {
            "feature": p["feature"],
            "L": p["chunk_chars"],
            "train": p["train_config"],
            "漱石予測割合（再計算）": f"{p['focus_share_others']:.4f}",
            "同（results.json）": f"{p['focus_share_others_results_json']:.4f}",
            "Accuracy（再計算 / results.json）": f"{p['accuracy']:.4f} / {p['accuracy_results_json']:.4f}",
            "恒等式 最大絶対誤差": f"{p['identity_max_abs_err']:.2e}",
            "予測一致率 c vs ĉ（fold 平均）": f"{p['pred_agreement_raw_vs_hat']:.4f}",
        }
        for p in pooled
    ]
    max_err = max(f["identity_max_abs_err"] for f in fold_rows)
    min_agree = min(f["pred_agreement_raw_vs_hat"] for f in fold_rows)
    n_zero = sum(r["none_n_zero_train"] for r in summary)
    lines = [
        "# 重心の長さと作家内類似度の実測",
        "",
        "- 分割：Inter-work（LOWO）の 9 fold，類似度はコサイン，特徴抽出は configs/default.yaml の本採用パラメータ（各 fold の Train に fit）",
        f"- Train 構成：none（間引きなし）と bal（間引きあり，fold ごとに default_rng([seed, fold 番号])，シード {args.seeds} の平均 ± 不偏標準偏差）",
        f"- 照合 run：{', '.join(f'{k}={v}' for k, v in args.runs_map.items())}（fold ごとの n_min・作家別 Train 数，Pooled 混同行列，folds.csv の値がすべて一致）",
        "- ĉ_a：長さ 1 に正規化した Train ベクトルの作家別平均．ρ̄_a：作家内 Train チャンク同士（自己除く）の平均コサイン．"
        "1/‖ĉ_a‖：割り増し係数．c_a：実装の重心（正規化前の平均）",
        "- 1/n 項の割合 = (1/n_a) / ‖ĉ_a‖²",
        f"- 恒等式 (1/n_a) Σ_d cos(q, d) = ‖ĉ_a‖ cos(q, ĉ_a) の最大絶対誤差（全条件）：{max_err:.3e}",
        f"- 正規化前・後の重心による予測の一致率の最小値（fold 単位）：{min_agree:.4f}",
        f"- ゼロベクトルの Train チャンク数（間引きなし，全 fold × 作家の合計）：{n_zero}",
        f"- 所要時間：{elapsed:.1f}s",
        "",
        "## 作家別の fold 平均",
        "",
        *[
            line
            for f in args.features
            for L in args.chunk_chars
            for line in [f"### {f}，L = {L}", "", *md_table(author_avg(f, L), list(author_avg(f, L)[0]))]
        ],
        "## 割り増し係数と漱石以外のチャンク中の漱石予測割合（重心法，fold 単位）",
        "",
        "- 比 = (1/‖ĉ_漱石‖) / (1/‖ĉ_test‖)．test は Test 作品の作家",
        "",
        *md_table(focus_fmt, list(focus_fmt[0])),
        "## Pooled（重心法）と再現の照合",
        "",
        *md_table(pooled_fmt, list(pooled_fmt[0])),
        *[
            line
            for f in args.features
            for L in args.chunk_chars
            for line in [f"## fold × 作家の詳細：{f}，L = {L}", "", *md_table(detail(f, L), list(detail(f, L)[0]))]
        ],
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--features", nargs="+", default=["base", "C"], help="対象の特徴量キー（base，A～D）")
    parser.add_argument("--chunk-chars", nargs="+", type=int, help="チャンク長 L（既定：chunking.target_chunk_chars）")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="<特徴量>=<照合する現行形式の 9 作品版 run 名>（例：base=base_c_022 C=base_c_022）"
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="間引きシード（既定：照合 run の balance_train.inter.seeds）")
    parser.add_argument("--focus-author", default="soseki", help="予測集中を測る作家 ID")
    parser.add_argument("--identity-tol", type=float, default=1e-10, help="恒等式の最大絶対誤差の許容値（超えたら停止）")
    parser.add_argument("--outputs-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "centroid_norm")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"出力先が既に存在（上書き防止）: {args.out_dir}")
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args.chunk_chars = args.chunk_chars or cfg["chunking"]["target_chunk_chars"]
    run_of = parse_pairs(args.runs, "--runs")
    missing = sorted(set(args.features) - set(run_of))
    if missing:
        raise ValueError(f"--runs に照合 run が未指定の特徴量: {missing}")
    args.runs_map = {f: run_of[f] for f in args.features}
    eps = float(cfg["similarity"]["eps"])
    variants = build_variants(cfg, args.features, with_ablation=False)
    runs = {name: load_run(args.outputs_dir, name) for name in dict.fromkeys(args.runs_map.values())}
    recorded_seeds = {name: list(r.meta["balance_train"]["inter"]["seeds"]) for name, r in runs.items()}
    if args.seeds is None:
        if len({tuple(s) for s in recorded_seeds.values()}) != 1:
            raise ValueError(f"照合 run 間で間引きシードが異なるため --seeds を指定すること: {recorded_seeds}")
        args.seeds = next(iter(recorded_seeds.values()))
    if len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        raise ValueError(f"--seeds は重複なしの 2 個以上（標準偏差の算出のため）: {args.seeds}")
    for v in variants:
        check_run_compat(runs[args.runs_map[v.name]], v, args.chunk_chars, args.seeds)

    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    corpus = load_corpus(corpus_path)
    if args.focus_author not in corpus.author_ids:
        raise ValueError(f"--focus-author {args.focus_author} が対象作家 {corpus.author_ids} にない")
    mismatched = sorted(n for n, r in runs.items() if r.meta["corpus"]["source_sha256"] != corpus_meta(corpus_path)["source_sha256"])
    if mismatched or any(list(r.meta["author_ids"]) != list(corpus.author_ids) for r in runs.values()):
        raise ValueError(f"照合 run のコーパスハッシュ・作家 ID が現在のコーパスと不一致: {mismatched}")
    focus = corpus.author_ids.index(args.focus_author)
    train_configs = [TrainConfig(NONE_LABEL, None), *(TrainConfig(f"bal_s{s}", s) for s in args.seeds)]

    t_start = time.perf_counter()
    chunk_sets = {L: build_chunks(corpus, L) for L in args.chunk_chars}
    measured = []
    for v in variants:
        for L, cs in chunk_sets.items():
            for c in train_configs:
                t0 = time.perf_counter()
                measured.append(measure_condition(v, cs, c, runs[args.runs_map[v.name]], eps, args.identity_tol, focus))
                p = measured[-1][2]
                print(
                    f"[{time.perf_counter() - t_start:6.1f}s] {v.name:<5} L={L:<5} {c.label:<8} ({time.perf_counter() - t0:.1f}s) "
                    f"identity_err={p['identity_max_abs_err']:.2e} agree={p['pred_agreement_raw_vs_hat']:.4f} "
                    f"focus_share={p['focus_share_others']:.4f}"
                )
    author_rows = [r for a, _, _ in measured for r in a]
    fold_rows = [r for _, f, _ in measured for r in f]
    pooled = [p for _, _, p in measured]
    summary = summarize_authors(author_rows, args.seeds)
    focus_rows = focus_table(author_rows, fold_rows, args.focus_author, args.seeds)
    elapsed = time.perf_counter() - t_start

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "author_stats.csv", author_rows)
    write_csv(args.out_dir / "author_summary.csv", summary)
    write_csv(args.out_dir / "fold_stats.csv", fold_rows)
    write_csv(args.out_dir / "focus_vs_inflation.csv", focus_rows)
    write_csv(args.out_dir / "pooled.csv", pooled)
    report = build_report(summary, focus_rows, pooled, fold_rows, corpus.author_ids, args, elapsed)
    (args.out_dir / "report.md").write_text(report + "\n", encoding="utf-8")
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": display_path(args.config),
        "config": cfg,
        "args": {k: (display_path(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "variants": [{"name": v.name, "class": f"{v.cls.__module__}.{v.cls.__name__}", "params": v.params} for v in variants],
        "reference_runs": {name: r.meta["git"] for name, r in runs.items()},
        "balance_rng": INTER_BALANCE_RNG,
        "git": git_state(),
        "corpus": corpus_meta(corpus_path),
        "author_ids": list(corpus.author_ids),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": round(elapsed, 1),
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(report)
    print(f"[output] {display_path(args.out_dir)}（{elapsed:.1f}s）")


if __name__ == "__main__":
    main()
