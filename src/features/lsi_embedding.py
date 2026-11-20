"""特徴量 D：LSI（潜在的意味解析）による潜在意味ベクトル（Ticket 005，講義第 5 回 p.36-37）．

入力は 004 の内容語 TF-IDF 行列（`WordTFIDFExtractor`，既定パラメータ，Train のみで fit）．
Train の文書-単語行列を特異値分解し，上位 K 個の成分でトピック空間へ射影する．

    X_train = U Σ V^T                       （第 5 回 p.37．中心化は講義にないため行わない）
    projection="u"  ：Z = X V_K Σ_K^{-1}    （Train 文書では U_K の行と一致，下記の導出）
    projection="us" ：Z = X V_K             （Train 文書では U_K Σ_K，特異値で重み付け．独自設計）

導出：X = U Σ V^T の両辺に右から V_K Σ_K^{-1} を掛けると U_K = X V_K Σ_K^{-1}（V の列の直交性による）．
講義 p.37 の「検索質問を V によりトピックの重みベクトルに変換し，U を用いて文書と比較する」を，
検索質問（Test チャンク）にも同じ変換を適用することで実現する．

出力は負値を含む（`non_negative = False`）．LSI は講義第 3 回の分散表現（word2vec 等）とは別物．
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np

from src.features.base import BaseFeatureExtractor
from src.features.word_tfidf import WordTFIDFExtractor


class LSIEmbeddingExtractor(BaseFeatureExtractor):
    """内容語 TF-IDF 行列の上位 K 特異成分によるトピック空間への射影．"""

    non_negative: ClassVar[bool] = False

    projections: ClassVar[frozenset[str]] = frozenset({"u", "us"})
    # σ_K / σ_1 がこの値以下ならランク不足とみなし停止（Σ_K^{-1} の数値的不安定の防止）
    rank_rtol: ClassVar[float] = 1e-10

    def __init__(self, n_components: int = 32, projection: str = "u") -> None:
        if not (isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1):
            raise ValueError(f"n_components は正の整数であること: {n_components!r}")
        if projection not in self.projections:
            raise ValueError(f"projection は {sorted(self.projections)} のいずれか: {projection!r}")
        self.n_components = n_components
        self.projection = projection

    def fit(self, train_texts: Sequence[str]) -> Self:
        K = self.n_components
        # 004 の抽出器を変更せず再利用（Train のみで語彙・idf を算出）
        self.tfidf_: WordTFIDFExtractor = WordTFIDFExtractor()
        # shape: (N, M)
        X = self.tfidf_.fit_transform(train_texts)
        N, M = X.shape
        if K > min(N, M):
            raise ValueError(f"{type(self).__name__}.fit: n_components={K} が min(N={N}, M={M}) を超える")

        # U: (N, r), s: (r,), Vt: (r, M)，r = min(N, M)．s は降順
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        if not s[K - 1] > self.rank_rtol * s[0]:
            raise ValueError(
                f"{type(self).__name__}.fit: ランク不足（σ_{K} = {s[K - 1]:.3e} <= {self.rank_rtol:.0e} * σ_1 = {s[0]:.3e}）"
            )

        # 符号の正規化：各 v_k の絶対値最大成分を正にする（u_k も同時に反転し X = U Σ V^T を保存）
        V_k = Vt[:K].T  # shape: (M, K)
        signs = np.sign(V_k[np.argmax(np.abs(V_k), axis=0), np.arange(K)])  # shape: (K,)
        assert np.all(signs != 0), "単位ベクトルの絶対値最大成分は非零のはず"

        self.components_: np.ndarray = V_k * signs[None, :]  # V_K, shape: (M, K)
        self.singular_values_: np.ndarray = s[:K]  # Σ_K の対角, shape: (K,)
        self.train_u_: np.ndarray = U[:, :K] * signs[None, :]  # U_K（検証用）, shape: (N, K)
        # 累積寄与率 Σ_{j≤k} σ_j^2 / ||X||_F^2（K の選択の考察用）, shape: (K,)
        self.cumulative_explained_ratio_: np.ndarray = np.cumsum(s[:K] ** 2) / np.sum(s**2)

        assert np.allclose(self.components_.T @ self.components_, np.eye(K), atol=1e-8), "V_K が直交正規でない"
        assert np.all(np.isfinite(self.components_)) and np.all(self.singular_values_ > 0)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        # Train で fit 済みの語彙・idf のまま TF-IDF 化（Test を含めた SVD の再計算は行わない）
        # shape: (n_texts, M) @ (M, K) -> (n_texts, K)
        Z = self.tfidf_.transform(texts) @ self.components_
        if self.projection == "u":
            # shape: (n_texts, K) / (1, K)
            Z = Z / self.singular_values_[None, :]
        return Z
