"""類似度に基づく著者推定器（重心法 Centroid，k-近傍類似度法 k-NN）．

推定器は登録ベクトル X_train: (N, M) と作家ラベル y_train: (N,) を保持し，
照会ベクトル X: (n, M) に対する作家スコア行列 (n, C) を返す．予測は argmax（同点は小さいラベルを採用）．
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Self

import numpy as np

SimilarityCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]
KnnMode = Literal["per_author", "global"]


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    # shape: (N, C)
    return (y[:, None] == np.arange(num_classes)[None, :]).astype(np.float64)


class _SimilarityClassifier:
    def __init__(self, similarity: SimilarityCallable, num_classes: int) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes は 2 以上: {num_classes}")
        self.similarity = similarity
        self.num_classes = num_classes

    def _check_fit_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError(f"形状不一致: X{X.shape}, y{y.shape}")
        if np.any((y < 0) | (y >= self.num_classes)):
            raise ValueError(f"ラベルが [0, {self.num_classes}) の範囲外")
        missing = np.setdiff1d(np.arange(self.num_classes), y)
        if missing.size:
            raise ValueError(f"登録データに存在しない作家ラベル: {missing.tolist()}")

    def scores(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # shape: (n,)
        return self.scores(X).argmax(axis=1)


class CentroidClassifier(_SimilarityClassifier):
    """作家 a の重心 μ_a = (1/|D_a|) Σ_{v∈D_a} v との類似度 sim(q, μ_a) をスコアとする．"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self._check_fit_inputs(X, y)
        Y = _one_hot(y, self.num_classes)
        # shape: (C, M) = (C, N) @ (N, M) / (C, 1)
        self.centroids_ = (Y.T @ X) / Y.sum(axis=0)[:, None]
        assert np.all(np.isfinite(self.centroids_)), "重心に NaN / Inf"
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        # shape: (n, C)
        return self.similarity(X, self.centroids_)


class KNNClassifier(_SimilarityClassifier):
    """k-近傍類似度法．

    per_author：各作家 a の登録チャンク D_a 内の上位 k_eff = min(k, |D_a|) 個の平均類似度．
    global：全登録チャンクの上位 k 個について，類似度を重みとした作家別投票の合計．
    """

    def __init__(self, similarity: SimilarityCallable, num_classes: int, k: int, mode: KnnMode = "per_author") -> None:
        super().__init__(similarity, num_classes)
        if k < 1:
            raise ValueError(f"k は 1 以上: {k}")
        if mode not in ("per_author", "global"):
            raise ValueError(f"未知の knn_mode: {mode}")
        self.k = k
        self.mode = mode

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self._check_fit_inputs(X, y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        # shape: (n, N)
        return self.scores_from_similarity(self.similarity(X, self.X_train_))

    def scores_from_similarity(self, S: np.ndarray) -> np.ndarray:
        """事前計算済みの照会 × 登録類似度行列 S: (n, N) から作家スコア (n, C) を算出（k 違いで S を共有）．"""
        if S.ndim != 2 or S.shape[1] != self.y_train_.size:
            raise ValueError(f"S の形状 {S.shape} が登録数 {self.y_train_.size} と不一致")
        return self._per_author_scores(S) if self.mode == "per_author" else self._global_scores(S)

    def _per_author_scores(self, S: np.ndarray) -> np.ndarray:
        def top_k_mean(a: int) -> np.ndarray:
            # shape: (n, |D_a|) → (n,)
            S_a = S[:, self.y_train_ == a]
            k_eff = min(self.k, S_a.shape[1])
            return np.partition(S_a, S_a.shape[1] - k_eff, axis=1)[:, -k_eff:].mean(axis=1)

        # shape: (n, C)
        return np.stack([top_k_mean(a) for a in range(self.num_classes)], axis=1)

    def _global_scores(self, S: np.ndarray) -> np.ndarray:
        n, N = S.shape
        k_eff = min(self.k, N)
        # shape: (n, k_eff)
        nn_idx = np.argpartition(S, N - k_eff, axis=1)[:, -k_eff:]
        weights = np.take_along_axis(S, nn_idx, axis=1)
        # shape: (n, k_eff, C) → (n, C)
        votes = _one_hot(self.y_train_[nn_idx].ravel(), self.num_classes).reshape(n, k_eff, self.num_classes)
        return np.einsum("nk,nkc->nc", weights, votes)


def build_classifier(
    method: str, similarity: SimilarityCallable, num_classes: int, *, k: int | None = None, knn_mode: KnnMode = "per_author"
) -> _SimilarityClassifier:
    match method:
        case "centroid":
            return CentroidClassifier(similarity, num_classes)
        case "knn":
            if k is None:
                raise ValueError("knn には k の指定が必要")
            return KNNClassifier(similarity, num_classes, k=k, mode=knn_mode)
        case _:
            raise ValueError(f"未知の推論方式: {method}")
