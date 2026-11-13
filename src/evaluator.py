"""類似度直接評価と著者判別評価の 2 段階評価器，および漏洩防止付き 1 fold 評価パイプライン．

類似度直接評価は照会（Test）チャンク × 登録（Train）チャンクの類似度行列 S: (n_test, n_train) 上で定義．
    μ_same = mean{ S[i, j] | y_test[i] = y_train[j] }，μ_diff = mean{ S[i, j] | y_test[i] ≠ y_train[j] }
    ΔS = μ_same - μ_diff（ペア単位の平均．チャンク数の多い作家ほど重みが大きい）
    作家間平均類似度マトリクス A[a, b] = mean{ S[i, j] | y_test[i] = a, y_train[j] = b }
    ΔS_macro = mean(diag A) - mean(offdiag A)（作家単位の平均．チャンク数の不均衡に頑健）
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.classifier import KNNClassifier, KnnMode, build_classifier  # noqa: E402
from src.dataset import ChunkSet, Split  # noqa: E402
from src.features.base import BaseFeatureExtractor  # noqa: E402
from src.similarity import NON_NEGATIVE_ONLY, get_similarity  # noqa: E402


# ---------------------------------------------------------------- 著者判別指標


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """行が正解，列が予測．shape: (C, C)．"""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: {y_true.shape} vs {y_pred.shape}")
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_metrics(cm: np.ndarray) -> dict[str, Any]:
    """混同行列から Accuracy，作家別 Precision / Recall / F1，マクロ F1 を算出．

    予測数 0 の作家の Precision（0 / 0）は 0 と定義（該当作家を一度も当てられていないため）．
    """
    tp = np.diag(cm).astype(np.float64)
    # shape: (C,)
    pred_count, true_count = cm.sum(axis=0), cm.sum(axis=1)
    if np.any(true_count == 0):
        raise ValueError(f"Test に存在しない作家があり Recall / マクロ F1 が定義不能: {np.flatnonzero(true_count == 0)}")
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = tp / true_count
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def mean_reciprocal_rank(scores: np.ndarray, y_true: np.ndarray) -> float:
    """正解作家の順位 r_i（自分より厳密に高いスコア数 + 1）の逆数平均．scores shape: (n, C)．"""
    true_score = np.take_along_axis(scores, y_true[:, None], axis=1)
    ranks = 1 + (scores > true_score).sum(axis=1)
    return float(np.mean(1.0 / ranks))


# ---------------------------------------------------------------- 類似度直接評価


@dataclass(frozen=True)
class SimilarityStats:
    mu_same: float
    mu_diff: float
    delta_s: float
    delta_s_macro: float
    # shape: (C, C)，行が照会（Test）作家，列が登録（Train）作家
    author_matrix: np.ndarray
    # 同一著者・異著者ペア類似度のヒストグラム（共通ビン）
    bin_edges: np.ndarray
    hist_same: np.ndarray
    hist_diff: np.ndarray
    # 箱ひげ図用の分位点（min, 25%, 50%, 75%, max）
    quantiles_same: np.ndarray
    quantiles_diff: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def similarity_stats(
    S: np.ndarray, y_test: np.ndarray, y_train: np.ndarray, num_classes: int, num_bins: int = 50
) -> SimilarityStats:
    if S.shape != (y_test.size, y_train.size):
        raise ValueError(f"形状不一致: S{S.shape}, y_test{y_test.shape}, y_train{y_train.shape}")
    # shape: (n_test, n_train)
    same = y_test[:, None] == y_train[None, :]
    if not same.any() or same.all():
        raise ValueError("同一著者ペアまたは異著者ペアが存在しない")
    # shape: (n_test, C), (n_train, C)
    Yt = (y_test[:, None] == np.arange(num_classes)).astype(np.float64)
    Yr = (y_train[:, None] == np.arange(num_classes)).astype(np.float64)
    pair_counts = np.outer(Yt.sum(axis=0), Yr.sum(axis=0))
    present = pair_counts > 0
    # shape: (C, C) = (C, n_test) @ (n_test, n_train) @ (n_train, C)
    A = np.divide(Yt.T @ S @ Yr, pair_counts, out=np.full_like(pair_counts, np.nan), where=present)
    # Test に現れない作家（Inter-work fold 等）の行は NaN とし，マクロ平均から除外
    diag_mask = np.eye(num_classes, dtype=bool) & present
    off_mask = ~np.eye(num_classes, dtype=bool) & present
    mu_same, mu_diff = float(S[same].mean()), float(S[~same].mean())
    lower = 0.0 if S.min() >= 0 else -1.0
    edges = np.linspace(lower, 1.0, num_bins + 1)
    probs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    stats = SimilarityStats(
        mu_same=mu_same,
        mu_diff=mu_diff,
        delta_s=mu_same - mu_diff,
        delta_s_macro=float(A[diag_mask].mean() - A[off_mask].mean()),
        author_matrix=A,
        bin_edges=edges,
        hist_same=np.histogram(S[same], bins=edges)[0],
        hist_diff=np.histogram(S[~same], bins=edges)[0],
        quantiles_same=np.quantile(S[same], probs),
        quantiles_diff=np.quantile(S[~same], probs),
    )
    assert np.isfinite([stats.mu_same, stats.mu_diff, stats.delta_s_macro]).all(), "類似度統計に NaN / Inf"
    return stats


# ---------------------------------------------------------------- 1 fold 評価パイプライン


@dataclass(frozen=True)
class FoldResult:
    split_name: str
    similarity: str
    method: str
    k: int | None
    knn_mode: str | None
    n_train: int
    n_test: int
    n_features: int
    metrics: dict[str, Any]
    mrr: float
    confusion: np.ndarray
    sim_stats: SimilarityStats
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split_name,
            "similarity": self.similarity,
            "method": self.method,
            "k": self.k,
            "knn_mode": self.knn_mode,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "accuracy": self.metrics["accuracy"],
            "macro_f1": self.metrics["macro_f1"],
            "mrr": self.mrr,
            "mu_same": self.sim_stats.mu_same,
            "mu_diff": self.sim_stats.mu_diff,
            "delta_s": self.sim_stats.delta_s,
            "delta_s_macro": self.sim_stats.delta_s_macro,
        }


@dataclass(frozen=True)
class ClassifierSpec:
    method: str
    k: int | None = None
    knn_mode: KnnMode = "per_author"


def evaluate_fold(
    make_extractor: Callable[[], BaseFeatureExtractor],
    chunk_set: ChunkSet,
    split: Split,
    similarity_name: str,
    classifier_specs: Sequence[ClassifierSpec],
    *,
    similarity_kwargs: dict[str, Any] | None = None,
    num_bins: int = 50,
) -> list[FoldResult]:
    """1 つの分割について，特徴量抽出（Train のみで fit）→ 類似度 → 各推論器の評価を実行．

    抽出器は fold ごとに `make_extractor()` で新規生成し，fit 済みの抽出器の使い回しによる漏洩を防止．
    """
    extractor = make_extractor()
    if extractor.is_fitted:
        raise RuntimeError("make_extractor は未 fit の新しい抽出器を返すこと")
    if similarity_name in NON_NEGATIVE_ONLY and not type(extractor).non_negative:
        raise ValueError(f"{type(extractor).__name__} は非負ベクトルではないため {similarity_name} を適用不可")

    sim_fn = get_similarity(similarity_name)
    kwargs = similarity_kwargs or {}
    similarity = lambda A, B: sim_fn(A, B, **kwargs)  # noqa: E731

    texts = chunk_set.texts
    labels = chunk_set.labels
    num_classes = len(chunk_set.author_ids)
    train_texts = [texts[i] for i in split.train_idx]
    test_texts = [texts[i] for i in split.test_idx]
    y_train, y_test = labels[split.train_idx], labels[split.test_idx]

    # Train のみで fit，Test は transform のみ
    X_train = extractor.fit(train_texts).transform(train_texts)
    X_test = extractor.transform(test_texts)

    # 照会 × 登録のチャンク類似度行列は 1 fold で 1 回だけ計算し，類似度直接評価と全 k-NN で共有
    # shape: (n_test, n_train)
    S = similarity(X_test, X_train)
    sim_stats = similarity_stats(S, y_test, y_train, num_classes, num_bins)

    def run(spec: ClassifierSpec) -> FoldResult:
        clf = build_classifier(spec.method, similarity, num_classes, k=spec.k, knn_mode=spec.knn_mode).fit(X_train, y_train)
        scores = clf.scores_from_similarity(S) if isinstance(clf, KNNClassifier) else clf.scores(X_test)
        y_pred = scores.argmax(axis=1)
        cm = confusion_matrix(y_test, y_pred, num_classes)
        return FoldResult(
            split_name=split.name,
            similarity=similarity_name,
            method=spec.method,
            k=spec.k,
            knn_mode=spec.knn_mode if spec.method == "knn" else None,
            n_train=len(train_texts),
            n_test=len(test_texts),
            n_features=X_train.shape[1],
            metrics=classification_metrics(cm) if np.all(cm.sum(axis=1) > 0) else _partial_metrics(cm),
            mrr=mean_reciprocal_rank(scores, y_test),
            confusion=cm,
            sim_stats=sim_stats,
            y_true=y_test,
            y_pred=y_pred,
            test_idx=split.test_idx,
        )

    return [run(spec) for spec in classifier_specs]


def _partial_metrics(cm: np.ndarray) -> dict[str, Any]:
    """Test に一部作家しか現れない fold（Inter-work）用．マクロ F1 は fold 統合後の混同行列で算出すること．"""
    return {"accuracy": float(np.trace(cm) / cm.sum()), "macro_f1": float("nan")}


def aggregate_confusion(results: Sequence[FoldResult]) -> dict[str, Any]:
    """複数 fold の混同行列を合算し，統合後の指標を算出（Leave-One-Work-Out の最終評価用）．"""
    cm = np.sum([r.confusion for r in results], axis=0)
    return {"confusion": cm, **classification_metrics(cm)}


# ---------------------------------------------------------------- 可視化


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path, title: str = "Confusion matrix") -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    threshold = cm.max() / 2
    [
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
    ]
    _save(fig, path)


def plot_author_similarity_matrix(
    A: np.ndarray, labels: Sequence[str], path: Path, title: str = "Mean similarity (query x registered)"
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(np.ma.masked_invalid(A), cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels)
    ax.set(xlabel="Registered author", ylabel="Query author", title=title)
    [
        ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        for i in range(A.shape[0])
        for j in range(A.shape[1])
        if np.isfinite(A[i, j])
    ]
    _save(fig, path)


def plot_similarity_distribution(stats: SimilarityStats, path: Path, title: str = "Same vs different author") -> None:
    """同一著者・異著者ペアの類似度分布（密度正規化ヒストグラム）と分位点の箱ひげ図．"""
    edges = stats.bin_edges
    widths = np.diff(edges)
    density = lambda h: h / (h.sum() * widths)  # noqa: E731
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    ax_h.stairs(density(stats.hist_same), edges, fill=True, alpha=0.5, label=f"same (mean={stats.mu_same:.3f})")
    ax_h.stairs(density(stats.hist_diff), edges, fill=True, alpha=0.5, label=f"different (mean={stats.mu_diff:.3f})")
    ax_h.set(xlabel="Similarity", ylabel="Density", title=f"{title}  ΔS={stats.delta_s:.3f}")
    ax_h.legend()
    ax_b.bxp(
        [
            {"label": name, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4], "fliers": []}
            for name, q in (("same", stats.quantiles_same), ("diff", stats.quantiles_diff))
        ],
        showfliers=False,
    )
    ax_b.set(ylabel="Similarity", title="min / quartiles / max")
    _save(fig, path)
