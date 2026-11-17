"""類似度直接評価と著者判別評価の 2 段階評価器，および漏洩防止付き 1 fold 評価パイプライン．

類似度直接評価は照会（Test）チャンク × 登録（Train）チャンクの類似度行列 S: (n_test, n_train) 上で定義．
    作家間平均類似度マトリクス A[a, b] = mean{ S[i, j] | y_test[i] = a, y_train[j] = b }
    ΔS_macro = mean(diag A) - mean(offdiag A)（作家単位の平均．チャンク数の不均衡に頑健，主指標）
    μ_same = mean{ S[i, j] | y_test[i] = y_train[j] }，μ_diff = mean{ S[i, j] | y_test[i] ≠ y_train[j] }
    ΔS_micro = μ_same - μ_diff（ペア単位の平均．チャンク数の多い作家ほど重みが大きい，併記用）

類似度統計は (C, C) のペア類似度和・ペア数と固定ビンのヒストグラムで保持するため，
Leave-One-Work-Out の全 fold を加算により統合（Pooled Out-of-Fold）可能．

著者判別評価の同点処理（最大スコアを 2 作家以上が共有する照会）．
    予測は同点ラベル C（判別不能）とし，混同行列 (C, C + 1) の列 C に計上．
    Accuracy では誤判別，マクロ F1 では正解作家の FN（Recall の分母のみに算入）．
    MRR の順位は悲観的順位 r_i = |{a | s_ia ≥ s_i,y_i}|（正解作家以上のスコアを持つ作家数，自身を含む）．
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.classifier import KNNClassifier, KnnMode, build_classifier, predict_from_scores, tie_label  # noqa: E402
from src.dataset import ChunkSet, Split  # noqa: E402
from src.features.base import BaseFeatureExtractor  # noqa: E402
from src.similarity import get_similarity  # noqa: E402

QUANTILE_PROBS = (0.0, 0.25, 0.5, 0.75, 1.0)


# ---------------------------------------------------------------- 著者判別指標


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """行が正解，列が予測（列 C は同点）．shape: (C, C + 1)．"""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: {y_true.shape} vs {y_pred.shape}")
    if np.any((y_true < 0) | (y_true >= num_classes)):
        raise ValueError(f"正解ラベルが [0, {num_classes}) の範囲外")
    if np.any((y_pred < 0) | (y_pred > tie_label(num_classes))):
        raise ValueError(f"予測ラベルが [0, {num_classes}]（{num_classes} は同点）の範囲外")
    n_cols = num_classes + 1
    return np.bincount(y_true * n_cols + y_pred, minlength=num_classes * n_cols).reshape(num_classes, n_cols)


def classification_metrics(cm: np.ndarray) -> dict[str, Any]:
    """混同行列 (C, C + 1) から Accuracy，作家別 Precision / Recall / F1，マクロ F1，同点件数を算出．

    同点（列 C）は Accuracy では誤判別，Recall の分母（正解作家の FN）にのみ算入し，Precision には影響しない．
    Test に現れない作家がある場合（Inter-work の単一 fold）はマクロ F1 を NaN とし，
    fold 統合後の混同行列（`aggregate_results`）で算出する．
    予測数 0 の作家の Precision（0 / 0）は 0 と定義（該当作家を一度も当てられていないため）．
    """
    C = cm.shape[0]
    if cm.shape != (C, C + 1):
        raise ValueError(f"混同行列の形状 {cm.shape} が (C, C + 1) ではない")
    tp = np.diag(cm[:, :C]).astype(np.float64)
    # shape: (C,)
    pred_count, true_count = cm[:, :C].sum(axis=0), cm.sum(axis=1)
    accuracy = float(tp.sum() / cm.sum())
    n_ties = int(cm[:, C].sum())
    if np.any(true_count == 0):
        return {"accuracy": accuracy, "macro_f1": float("nan"), "n_ties": n_ties}
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = tp / true_count
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "n_ties": n_ties,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def reciprocal_ranks(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """正解作家の悲観的順位 r_i = |{a | s_ia ≥ s_i,y_i}|（自身を含む）の逆数．scores shape: (n, C) → (n,)．

    全作家が同点の照会は r_i = C（逆順位 1 / C）．
    """
    if scores.ndim != 2 or y_true.shape != (scores.shape[0],):
        raise ValueError(f"形状不一致: scores{scores.shape}, y_true{y_true.shape}")
    # shape: (n, 1)
    true_score = np.take_along_axis(scores, y_true[:, None], axis=1)
    # shape: (n,)，自身を含むため r_i ≥ 1
    return 1.0 / (scores >= true_score).sum(axis=1)


def mean_reciprocal_rank(scores: np.ndarray, y_true: np.ndarray) -> float:
    return float(reciprocal_ranks(scores, y_true).mean())


# ---------------------------------------------------------------- 類似度直接評価


@dataclass(frozen=True)
class SimilarityStats:
    """加算可能な類似度統計（fold 統合は `merge`）．"""

    # shape: (C, C)，行が照会（Test）作家，列が登録（Train）作家
    pair_sum: np.ndarray
    pair_count: np.ndarray
    # shape: (num_bins + 1,), (num_bins,)
    bin_edges: np.ndarray
    hist_same: np.ndarray
    hist_diff: np.ndarray

    @property
    def author_matrix(self) -> np.ndarray:
        # 該当ペアのない成分（Test に現れない作家の行）は NaN
        return np.divide(
            self.pair_sum, self.pair_count, out=np.full_like(self.pair_sum, np.nan), where=self.pair_count > 0
        )

    @property
    def mu_same(self) -> float:
        return float(np.trace(self.pair_sum) / np.trace(self.pair_count))

    @property
    def mu_diff(self) -> float:
        off = ~np.eye(len(self.pair_sum), dtype=bool)
        return float(self.pair_sum[off].sum() / self.pair_count[off].sum())

    @property
    def delta_s_micro(self) -> float:
        return self.mu_same - self.mu_diff

    @property
    def mu_same_macro(self) -> float:
        A, present = self.author_matrix, self.pair_count > 0
        return float(A[np.eye(len(A), dtype=bool) & present].mean())

    @property
    def mu_diff_macro(self) -> float:
        A, present = self.author_matrix, self.pair_count > 0
        return float(A[~np.eye(len(A), dtype=bool) & present].mean())

    @property
    def delta_s_macro(self) -> float:
        return self.mu_same_macro - self.mu_diff_macro

    @property
    def quantiles_same(self) -> np.ndarray:
        return histogram_quantiles(self.hist_same, self.bin_edges)

    @property
    def quantiles_diff(self) -> np.ndarray:
        return histogram_quantiles(self.hist_diff, self.bin_edges)

    @classmethod
    def merge(cls, stats: Sequence[SimilarityStats]) -> SimilarityStats:
        if not stats:
            raise ValueError("統合対象が空")
        edges = stats[0].bin_edges
        if not all(np.array_equal(s.bin_edges, edges) for s in stats):
            raise ValueError("ビン境界の異なる統計は統合不可")
        return cls(
            pair_sum=np.sum([s.pair_sum for s in stats], axis=0),
            pair_count=np.sum([s.pair_count for s in stats], axis=0),
            bin_edges=edges,
            hist_same=np.sum([s.hist_same for s in stats], axis=0),
            hist_diff=np.sum([s.hist_diff for s in stats], axis=0),
        )

    def scalars(self) -> dict[str, float]:
        return {
            "mu_same_macro": self.mu_same_macro,
            "mu_diff_macro": self.mu_diff_macro,
            "delta_s_macro": self.delta_s_macro,
            "mu_same": self.mu_same,
            "mu_diff": self.mu_diff,
            "delta_s_micro": self.delta_s_micro,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_sum": self.pair_sum.tolist(),
            "pair_count": self.pair_count.tolist(),
            "bin_edges": self.bin_edges.tolist(),
            "hist_same": self.hist_same.tolist(),
            "hist_diff": self.hist_diff.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimilarityStats:
        return cls(**{k: np.asarray(v, dtype=np.float64) for k, v in d.items()})


def histogram_quantiles(hist: np.ndarray, edges: np.ndarray, probs: Sequence[float] = QUANTILE_PROBS) -> np.ndarray:
    """ヒストグラムの累積分布を線形補間した近似分位点（min / max は非空ビンの端）．"""
    cdf = np.concatenate([[0.0], np.cumsum(hist)]) / hist.sum()
    nonempty = np.flatnonzero(hist)
    lo, hi = edges[nonempty[0]], edges[nonempty[-1] + 1]
    # cdf は単調非減少，平坦区間は np.interp が左端を返す
    q = np.interp(np.asarray(probs), cdf, edges)
    return np.clip(q, lo, hi)


def similarity_hist_edges(non_negative: bool, num_bins: int) -> np.ndarray:
    return np.linspace(0.0 if non_negative else -1.0, 1.0, num_bins + 1)


def similarity_stats(
    S: np.ndarray, y_test: np.ndarray, y_train: np.ndarray, num_classes: int, bin_edges: np.ndarray
) -> SimilarityStats:
    if S.shape != (y_test.size, y_train.size):
        raise ValueError(f"形状不一致: S{S.shape}, y_test{y_test.shape}, y_train{y_train.shape}")
    if S.min() < bin_edges[0] or S.max() > bin_edges[-1]:
        raise ValueError(f"類似度 [{S.min()}, {S.max()}] がビン範囲 [{bin_edges[0]}, {bin_edges[-1]}] 外")
    # shape: (n_test, n_train)
    same = y_test[:, None] == y_train[None, :]
    if not same.any() or same.all():
        raise ValueError("同一著者ペアまたは異著者ペアが存在しない")
    # shape: (n_test, C), (n_train, C)
    Yt = (y_test[:, None] == np.arange(num_classes)).astype(np.float64)
    Yr = (y_train[:, None] == np.arange(num_classes)).astype(np.float64)
    stats = SimilarityStats(
        # shape: (C, C) = (C, n_test) @ (n_test, n_train) @ (n_train, C)
        pair_sum=Yt.T @ S @ Yr,
        pair_count=np.outer(Yt.sum(axis=0), Yr.sum(axis=0)),
        bin_edges=bin_edges,
        hist_same=np.histogram(S[same], bins=bin_edges)[0].astype(np.float64),
        hist_diff=np.histogram(S[~same], bins=bin_edges)[0].astype(np.float64),
    )
    assert np.isfinite(list(stats.scalars().values())).all(), "類似度統計に NaN / Inf"
    return stats


# ---------------------------------------------------------------- 1 fold 評価パイプライン


@dataclass(frozen=True)
class ClassifierSpec:
    method: str
    k: int | None = None
    knn_mode: KnnMode | None = None

    @property
    def name(self) -> str:
        return "centroid" if self.method == "centroid" else f"knn{self.k}_{self.knn_mode}"


@dataclass(frozen=True)
class FoldResult:
    split_name: str
    similarity: str
    spec: ClassifierSpec
    n_train: int
    n_test: int
    n_features: int
    extract_seconds: float
    similarity_seconds: float
    # shape: (C, C + 1)，列 C は同点
    confusion: np.ndarray
    sim_stats: SimilarityStats
    # 最大スコアを 2 作家以上が共有した照会数（同点ラベル C で予測）
    n_ties: int
    # ノルム 0 の照会（Test）チャンク数（推論方式に依存しない）
    n_zero_query: int
    test_idx: np.ndarray = field(repr=False)
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    # shape: (n_test, C)
    scores: np.ndarray = field(repr=False)

    @property
    def metrics(self) -> dict[str, Any]:
        return classification_metrics(self.confusion)

    @property
    def mrr(self) -> float:
        return mean_reciprocal_rank(self.scores, self.y_true)


def evaluate_fold(
    make_extractor: Callable[[], BaseFeatureExtractor],
    chunk_set: ChunkSet,
    split: Split,
    similarity_names: Sequence[str],
    classifier_specs: Sequence[ClassifierSpec],
    *,
    similarity_kwargs: dict[str, Any] | None = None,
    num_bins: int = 50,
) -> list[FoldResult]:
    """1 つの分割について，特徴量抽出（Train のみで fit，1 回のみ）→ 各類似度 → 各推論器の評価を実行．

    抽出器は fold ごとに `make_extractor()` で新規生成し，fit 済みの抽出器の使い回しによる漏洩を防止．
    """
    extractor = make_extractor()
    if extractor.is_fitted:
        raise RuntimeError("make_extractor は未 fit の新しい抽出器を返すこと")
    non_negative = type(extractor).non_negative

    texts = chunk_set.texts
    labels = chunk_set.labels
    num_classes = len(chunk_set.author_ids)
    train_texts = [texts[i] for i in split.train_idx]
    test_texts = [texts[i] for i in split.test_idx]
    y_train, y_test = labels[split.train_idx], labels[split.test_idx]

    # Train のみで fit，Test は transform のみ
    t0 = time.perf_counter()
    X_train = extractor.fit(train_texts).transform(train_texts)
    X_test = extractor.transform(test_texts)
    extract_seconds = time.perf_counter() - t0
    # 負値を含む特徴量（LSI）のヒストグラム範囲は [-1, 1]
    edges = similarity_hist_edges(non_negative, num_bins)
    n_zero_query = int((np.linalg.norm(X_test, axis=1) == 0).sum())

    def run_similarity(name: str) -> list[FoldResult]:
        sim_fn = get_similarity(name)
        kwargs = similarity_kwargs or {}
        similarity = lambda A, B: sim_fn(A, B, **kwargs)  # noqa: E731
        t1 = time.perf_counter()
        # 照会 × 登録のチャンク類似度行列は 1 回だけ計算し，類似度直接評価と全 k-NN で共有
        # shape: (n_test, n_train)
        S = similarity(X_test, X_train)
        sim_seconds = time.perf_counter() - t1
        stats = similarity_stats(S, y_test, y_train, num_classes, edges)

        def run(spec: ClassifierSpec) -> FoldResult:
            clf = build_classifier(spec.method, similarity, num_classes, k=spec.k, knn_mode=spec.knn_mode or "per_author")
            clf.fit(X_train, y_train)
            scores = clf.scores_from_similarity(S) if isinstance(clf, KNNClassifier) else clf.scores(X_test)
            # shape: (n_test,)，値域 {0, ..., C}（C は同点）
            y_pred = predict_from_scores(scores)
            confusion = confusion_matrix(y_test, y_pred, num_classes)
            n_ties = int((y_pred == tie_label(num_classes)).sum())
            assert n_ties == confusion[:, num_classes].sum(), "同点件数と混同行列の同点列が不一致"
            return FoldResult(
                split_name=split.name,
                similarity=name,
                spec=spec,
                n_train=len(train_texts),
                n_test=len(test_texts),
                n_features=X_train.shape[1],
                extract_seconds=extract_seconds,
                similarity_seconds=sim_seconds,
                confusion=confusion,
                sim_stats=stats,
                n_ties=n_ties,
                n_zero_query=n_zero_query,
                test_idx=split.test_idx,
                y_true=y_test,
                y_pred=y_pred,
                scores=scores,
            )

        return [run(spec) for spec in classifier_specs]

    return [r for name in similarity_names for r in run_similarity(name)]


@dataclass(frozen=True)
class AggregateResult:
    """複数 fold の Pooled Out-of-Fold 統合結果（Intra-work は 1 fold のみ）．"""

    num_folds: int
    n_test: int
    # shape: (C, C + 1)，列 C は同点
    confusion: np.ndarray
    mrr: float
    n_ties: int
    n_zero_query: int
    sim_stats: SimilarityStats

    @property
    def metrics(self) -> dict[str, Any]:
        return classification_metrics(self.confusion)


def aggregate_results(results: Sequence[FoldResult]) -> AggregateResult:
    """混同行列・逆順位・類似度統計を全 fold で合算（同一条件の fold 群を渡すこと）．"""
    keys = {(r.similarity, r.spec) for r in results}
    if len(keys) != 1:
        raise ValueError(f"異なる条件の fold が混在: {keys}")
    rr = np.concatenate([reciprocal_ranks(r.scores, r.y_true) for r in results])
    return AggregateResult(
        num_folds=len(results),
        n_test=int(rr.size),
        confusion=np.sum([r.confusion for r in results], axis=0),
        mrr=float(rr.mean()),
        n_ties=sum(r.n_ties for r in results),
        n_zero_query=sum(r.n_zero_query for r in results),
        sim_stats=SimilarityStats.merge([r.sim_stats for r in results]),
    )


# ---------------------------------------------------------------- 可視化


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: Path, title: str = "Confusion matrix") -> None:
    """混同行列 (C, C + 1) を描画．最右列は同点（判別不能，"tie"）．"""
    C = len(labels)
    if cm.shape != (C, C + 1):
        raise ValueError(f"混同行列の形状 {cm.shape} が (C, C + 1) = ({C}, {C + 1}) ではない")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(C + 1), yticks=range(C), xticklabels=[*labels, "tie"], yticklabels=labels)
    # 同点列の境界線
    ax.axvline(C - 0.5, color="gray", linewidth=1.0, linestyle="--")
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
    # viridis は高値側が明るいため，正規化値 0.6 超のセルは黒文字
    norm = im.norm
    [
        ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", color="black" if norm(A[i, j]) > 0.6 else "white", fontsize=8)
        for i in range(A.shape[0])
        for j in range(A.shape[1])
        if np.isfinite(A[i, j])
    ]
    _save(fig, path)


def plot_similarity_distribution(stats: SimilarityStats, path: Path, title: str = "Same vs different author") -> None:
    """同一著者・異著者ペアの類似度分布（密度正規化ヒストグラム）と近似分位点の箱ひげ図．"""
    edges = stats.bin_edges
    widths = np.diff(edges)
    density = lambda h: h / (h.sum() * widths)  # noqa: E731
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 1]})
    ax_h.stairs(density(stats.hist_same), edges, fill=True, alpha=0.5, label=f"same (mean={stats.mu_same:.3f})")
    ax_h.stairs(density(stats.hist_diff), edges, fill=True, alpha=0.5, label=f"different (mean={stats.mu_diff:.3f})")
    ax_h.set(
        xlabel="Similarity",
        ylabel="Density",
        title=f"{title}  ΔS_macro={stats.delta_s_macro:.3f}  ΔS_micro={stats.delta_s_micro:.3f}",
    )
    ax_h.legend()
    ax_b.bxp(
        [
            {"label": name, "whislo": q[0], "q1": q[1], "med": q[2], "q3": q[3], "whishi": q[4], "fliers": []}
            for name, q in (("same", stats.quantiles_same), ("diff", stats.quantiles_diff))
        ],
        showfliers=False,
    )
    ax_b.set(ylabel="Similarity", title="approx. quartiles")
    _save(fig, path)
