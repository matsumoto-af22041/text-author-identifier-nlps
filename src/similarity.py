"""コサイン係数（講義第 5 回 p.26，第 3 回 p.25）のベクトル化実装．

cos(d_i, q) = d_i・q / (||d_i||_2 ||q||_2)
関数は X: (n, M), Y: (m, M) を受け取り，類似度行列 S: (n, m)（S[i, j] = cos(x_i, y_j)）を返す．
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

DEFAULT_EPS = 1e-12
# 浮動小数点誤差として許容する値域からのはみ出し
_RANGE_TOL = 1e-9


class SimilarityFn(Protocol):
    def __call__(self, X: np.ndarray, Y: np.ndarray, *, eps: float = ...) -> np.ndarray: ...


def _check_inputs(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(f"形状不一致: X{X.shape}, Y{Y.shape}（(n, M), (m, M) を要求）")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("入力に NaN / Inf を含む")
    return X, Y


def _finalize(S: np.ndarray, lower: float) -> np.ndarray:
    assert np.all(np.isfinite(S)), "類似度行列に NaN / Inf"
    assert np.all((S >= lower - _RANGE_TOL) & (S <= 1.0 + _RANGE_TOL)), (
        f"類似度が [{lower}, 1] の範囲外: [{S.min()}, {S.max()}]"
    )
    return np.clip(S, lower, 1.0)


def cosine_similarity(X: np.ndarray, Y: np.ndarray, *, eps: float = DEFAULT_EPS) -> np.ndarray:
    """cos(x, y) = x·y / (||x||_2 ||y||_2 + eps)．負値を含む入力（LSI 等）では [-1, 1]．

    ノルム 0 のベクトル（ゼロベクトル）との類似度は 0 となる．
    """
    X, Y = _check_inputs(X, Y)
    # shape: (n, 1), (1, m)
    norm_x = np.linalg.norm(X, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, axis=1, keepdims=True).T
    # shape: (n, m)
    S = (X @ Y.T) / (norm_x * norm_y + eps)
    lower = 0.0 if (np.all(X >= 0) and np.all(Y >= 0)) else -1.0
    return _finalize(S, lower)


SIMILARITY_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "cosine": cosine_similarity,
}


def get_similarity(name: str) -> Callable[..., np.ndarray]:
    if name not in SIMILARITY_FUNCTIONS:
        raise KeyError(f"未知の類似度: {name}（選択肢: {sorted(SIMILARITY_FUNCTIONS)}）")
    return SIMILARITY_FUNCTIONS[name]
