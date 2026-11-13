"""全特徴量モジュール（A〜D）が準拠する抽象基底クラス．

サブクラスは `fit` と `transform` を通常どおり実装するだけでよい．
`__init_subclass__` により以下の検査が自動的に付与される．
    - `fit` 前の `transform` 呼び出しを禁止（Test データでの fit による漏洩を構造的に防止）
    - `transform` 出力が 2 次元 `(N_samples, D_features)` かつ全要素有限
    - 次元数 D_features が fit 後の全 `transform` 呼び出しで一定
    - `non_negative = True` のクラスでは出力が非負（Jaccard / Dice 係数の適用条件）
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar, Self

import numpy as np


class NotFittedError(RuntimeError):
    """fit 前に transform が呼ばれた場合の例外．"""


class BaseFeatureExtractor(ABC):
    """テキスト列を特徴量行列へ写像する抽出器．"""

    # 出力が非負ベクトルか（False の場合はコサイン類似度のみ適用可）
    non_negative: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "fit" in cls.__dict__:
            cls.fit = _guard_fit(cls.__dict__["fit"])  # type: ignore[method-assign]
        if "transform" in cls.__dict__:
            cls.transform = _guard_transform(cls.__dict__["transform"])  # type: ignore[method-assign]

    @abstractmethod
    def fit(self, train_texts: Sequence[str]) -> Self:
        """Train テキストのみから語彙・統計量などのパラメータを推定．"""

    @abstractmethod
    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """fit 済みパラメータで特徴量行列（shape: (N_samples, D_features)）を返す．"""

    def fit_transform(self, train_texts: Sequence[str]) -> np.ndarray:
        return self.fit(train_texts).transform(train_texts)

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "_is_fitted", False)

    @property
    def n_features(self) -> int | None:
        return getattr(self, "_n_features", None)


def _validate_texts(texts: Sequence[str], where: str) -> None:
    if isinstance(texts, str):
        raise TypeError(f"{where}: テキスト列（Sequence[str]）を渡すこと（単一 str は不可）")
    if len(texts) == 0:
        raise ValueError(f"{where}: 空のテキスト列")
    if not all(isinstance(t, str) for t in texts):
        raise TypeError(f"{where}: 要素はすべて str であること")


def _guard_fit(fit: Callable[..., object]) -> Callable[..., object]:
    @functools.wraps(fit)
    def wrapper(self: BaseFeatureExtractor, train_texts: Sequence[str]) -> object:
        _validate_texts(train_texts, f"{type(self).__name__}.fit")
        result = fit(self, train_texts)
        if result is not self:
            raise TypeError(f"{type(self).__name__}.fit は self を返すこと")
        self._is_fitted = True  # type: ignore[attr-defined]
        self._n_features = None  # type: ignore[attr-defined]
        return result

    return wrapper


def _guard_transform(transform: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
    @functools.wraps(transform)
    def wrapper(self: BaseFeatureExtractor, texts: Sequence[str]) -> np.ndarray:
        name = f"{type(self).__name__}.transform"
        if not self.is_fitted:
            raise NotFittedError(f"{name}: fit(train_texts) の前に呼ばれた")
        _validate_texts(texts, name)
        X = transform(self, texts)
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError(f"{name}: 2 次元 np.ndarray を返すこと（得られた型: {type(X).__name__}）")
        if X.shape[0] != len(texts):
            raise ValueError(f"{name}: 行数 {X.shape[0]} が入力数 {len(texts)} と不一致")
        if X.shape[1] == 0:
            raise ValueError(f"{name}: 特徴量次元が 0")
        if not np.all(np.isfinite(X)):
            raise ValueError(f"{name}: NaN / Inf を含む")
        if type(self).non_negative and np.any(X < 0):
            raise ValueError(f"{name}: non_negative=True だが負値を含む（最小値 {X.min()}）")
        if self.n_features is None:
            self._n_features = X.shape[1]  # type: ignore[attr-defined]
        elif X.shape[1] != self.n_features:
            raise ValueError(f"{name}: 次元数 {X.shape[1]} が fit 後の初回 {self.n_features} と不一致")
        return X

    return wrapper
