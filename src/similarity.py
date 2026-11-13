"""講義準拠の類似度（コサイン類似度，重み付き Jaccard 係数，重み付き Dice 係数）のベクトル化実装．

すべての関数は X: (n, M), Y: (m, M) を受け取り，類似度行列 S: (n, m)（S[i, j] = sim(x_i, y_j)）を返す．
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

DEFAULT_EPS = 1e-12
# Σ min(x, y) のバッチブロードキャストで確保する中間配列 (batch, m, M) の上限バイト数
DEFAULT_MAX_BATCH_BYTES = 256 * 1024**2
# 浮動小数点誤差として許容する [0, 1] 範囲からのはみ出し
_RANGE_TOL = 1e-9


class SimilarityFn(Protocol):
    def __call__(self, X: np.ndarray, Y: np.ndarray, *, eps: float = ..., max_batch_bytes: int = ...) -> np.ndarray: ...


def _check_inputs(X: np.ndarray, Y: np.ndarray, *, non_negative: bool) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"形状不一致: X{X.shape}, Y{Y.shape}（(n, M), (m, M) を要求）")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("入力に NaN / Inf を含む")
    if non_negative and (np.any(X < 0) or np.any(Y < 0)):
        raise ValueError("Jaccard / Dice 係数は非負ベクトル専用（負値を含む入力）")
    return X, Y


def _finalize(S: np.ndarray, lower: float) -> np.ndarray:
    assert np.all(np.isfinite(S)), "類似度行列に NaN / Inf"
    assert np.all((S >= lower - _RANGE_TOL) & (S <= 1.0 + _RANGE_TOL)), (
        f"類似度が [{lower}, 1] の範囲外: [{S.min()}, {S.max()}]"
    )
    return np.clip(S, lower, 1.0)


def _pairwise_min_sum(X: np.ndarray, Y: np.ndarray, max_batch_bytes: int) -> np.ndarray:
    """Σ_j min(x_ij, y_kj) を X の行バッチ単位の 3 次元ブロードキャストで算出．shape: (n, m)．"""
    n, M = X.shape
    m = Y.shape[0]
    if n == 0 or m == 0:
        raise ValueError(f"空の入力: X{X.shape}, Y{Y.shape}")
    batch = max(1, max_batch_bytes // (m * M * X.itemsize))
    # shape: (batch, 1, M) と (1, m, M) → (batch, m, M) → (batch, m)
    blocks = [np.minimum(X[s : s + batch, None, :], Y[None, :, :]).sum(axis=2) for s in range(0, n, batch)]
    return np.concatenate(blocks, axis=0)


def cosine_similarity(
    X: np.ndarray, Y: np.ndarray, *, eps: float = DEFAULT_EPS, max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
) -> np.ndarray:
    """cos(x, y) = x·y / (||x||_2 ||y||_2 + eps)．負値を含む入力（LSI 等）では [-1, 1]．"""
    X, Y = _check_inputs(X, Y, non_negative=False)
    # shape: (n, 1), (1, m)
    norm_x = np.linalg.norm(X, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, axis=1, keepdims=True).T
    # shape: (n, m)
    S = (X @ Y.T) / (norm_x * norm_y + eps)
    lower = 0.0 if (np.all(X >= 0) and np.all(Y >= 0)) else -1.0
    return _finalize(S, lower)


def jaccard_similarity(
    X: np.ndarray, Y: np.ndarray, *, eps: float = DEFAULT_EPS, max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
) -> np.ndarray:
    """J_w(x, y) = Σ min(x_j, y_j) / (Σ max(x_j, y_j) + eps)，Σ max = Σ x + Σ y - Σ min を利用．"""
    X, Y = _check_inputs(X, Y, non_negative=True)
    # shape: (n, m)
    min_sum = _pairwise_min_sum(X, Y, max_batch_bytes)
    max_sum = X.sum(axis=1)[:, None] + Y.sum(axis=1)[None, :] - min_sum
    return _finalize(min_sum / (max_sum + eps), 0.0)


def dice_similarity(
    X: np.ndarray, Y: np.ndarray, *, eps: float = DEFAULT_EPS, max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
) -> np.ndarray:
    """D_w(x, y) = 2 Σ min(x_j, y_j) / (Σ (x_j + y_j) + eps)．"""
    X, Y = _check_inputs(X, Y, non_negative=True)
    # shape: (n, m)
    min_sum = _pairwise_min_sum(X, Y, max_batch_bytes)
    total = X.sum(axis=1)[:, None] + Y.sum(axis=1)[None, :]
    return _finalize(2.0 * min_sum / (total + eps), 0.0)


SIMILARITY_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "cosine": cosine_similarity,
    "jaccard": jaccard_similarity,
    "dice": dice_similarity,
}
# 非負ベクトルを要求する類似度
NON_NEGATIVE_ONLY = frozenset({"jaccard", "dice"})


def get_similarity(name: str) -> Callable[..., np.ndarray]:
    if name not in SIMILARITY_FUNCTIONS:
        raise KeyError(f"未知の類似度: {name}（選択肢: {sorted(SIMILARITY_FUNCTIONS)}）")
    return SIMILARITY_FUNCTIONS[name]
