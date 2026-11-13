"""ベースライン：文字 unigram 相対頻度（パイプライン疎通確認および最小構成の比較基準）．"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from src.features.base import BaseFeatureExtractor


class CharUnigramExtractor(BaseFeatureExtractor):
    """Train に出現した文字の相対頻度．Test のみに出現する文字は無視．"""

    def fit(self, train_texts: Sequence[str]) -> Self:
        self.vocab_ = {c: i for i, c in enumerate(sorted(set("".join(train_texts))))}
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        rows = [np.bincount([self.vocab_[c] for c in t if c in self.vocab_], minlength=len(self.vocab_)) for t in texts]
        # shape: (n, |vocab|)
        X = np.stack(rows).astype(np.float64)
        return X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
