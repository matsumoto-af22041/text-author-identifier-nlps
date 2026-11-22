"""特徴量 C：内容語 TF-IDF（Ticket 4，講義第 5 回 p.11-13, p.25 のベクトル空間モデル）．

索引語は UniDic の `pos1` が名詞・動詞・形容詞の形態素（第 5 回 p.11）．特徴量キーは語彙素
`lemma`（第 2 回 p.16 の原形復元に相当．未知語で `None` の場合は表層形）．

    tf(t, d) = (チャンク d における語 t の出現回数) / (チャンク d の総形態素数)   （第 5 回 p.12）
    idf(t)   = log(N / df(t))                                                      （第 5 回 p.13，自然対数）
    w_dt     = tf(t, d) * idf(t)

総形態素数は補助記号（`pos1 == "補助記号"`）と空白（`pos1 == "空白"`）を除く全形態素数（B と同じ定義）．
Train の全チャンクに出現する語は idf = 0 となり重みが消える（第 5 回 p.11 の「どの文書にも均質に現れる語」）．
補正（+1 等）は行わない．

形態素解析は `src.morphology.morphemes_cached` （テキスト単位のプロセス内キャッシュ，Ticket 26）を使い，
fold・シード・条件の間で同じチャンクを再解析しない．キャッシュは解析結果のみで，語彙・df・IDF は fit ごとに
Train のみから算出．
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np

from src.features.base import BaseFeatureExtractor
from src.morphology import Tagger, morphemes_cached, shared_tagger


@dataclass(frozen=True)
class AnalyzedText:
    """1 テキストの形態素解析結果（005 から再利用）．

    terms：索引語（内容語）の特徴量キー列（出現順，重複あり）
    num_morphemes：補助記号・空白を除く総形態素数（tf の分母）
    """

    terms: tuple[str, ...]
    num_morphemes: int


class WordTFIDFExtractor(BaseFeatureExtractor):
    """Train の文書頻度上位 `max_features` 語に限定した内容語 TF-IDF．"""

    non_negative: ClassVar[bool] = True

    # 索引語とする品詞（UniDic pos1，第 5 回 p.11）
    content_pos: ClassVar[frozenset[str]] = frozenset({"名詞", "動詞", "形容詞"})
    # 総形態素数から除外する品詞（UniDic pos1）
    excluded_pos: ClassVar[frozenset[str]] = frozenset({"補助記号", "空白"})

    def __init__(self, max_features: int = 2000) -> None:
        if not (isinstance(max_features, int) and max_features >= 1):
            raise ValueError(f"max_features は正の整数であること: {max_features!r}")
        self.max_features = max_features
        # transform 呼び出しごとのゼロベクトル行数（語彙に一致する語を持たないチャンク）
        self.n_zero_rows_history_: list[int] = []

    @property
    def tagger(self) -> Tagger:
        """共有 Tagger（検証スクリプトで生の素性を確認する用途．特徴量の計算は analyze のキャッシュ経由）．"""
        return shared_tagger()

    # ---- 形態素解析・頻度集計（005 から再利用する公開メソッド） ----

    def analyze(self, text: str) -> AnalyzedText:
        """テキストを形態素解析し，索引語キー列と総形態素数を返す．総形態素数 0 は ValueError．"""
        # 特徴量キーは語彙素（lemma が None の未知語は表層形，Morpheme で代替済み）
        morphs = [(m.pos1, m.lemma) for m in morphemes_cached(text)]
        num_morphemes = sum(pos1 not in self.excluded_pos for pos1, _ in morphs)
        if num_morphemes == 0:
            raise ValueError(f"{type(self).__name__}: 総形態素数が 0 のテキスト: {text[:50]!r}")
        terms = tuple(key for pos1, key in morphs if pos1 in self.content_pos)
        return AnalyzedText(terms=terms, num_morphemes=num_morphemes)

    def analyze_all(self, texts: Sequence[str]) -> list[AnalyzedText]:
        return [self.analyze(t) for t in texts]

    @staticmethod
    def document_frequency(analyzed: Sequence[AnalyzedText]) -> Counter[str]:
        """df(t)：語 t を 1 回以上含むテキスト数．"""
        return Counter(term for a in analyzed for term in set(a.terms))

    @staticmethod
    def select_vocabulary(df: Counter[str], max_features: int) -> tuple[str, ...]:
        """df 降順・同数は辞書順で上位 max_features 語を決定的に選択．"""
        return tuple(sorted(df, key=lambda t: (-df[t], t))[:max_features])

    def count_matrix(self, analyzed: Sequence[AnalyzedText]) -> np.ndarray:
        """語彙上の出現回数行列（語彙外の語は無視）．shape: (n_texts, |V|)"""
        flat = np.array(
            [(i, self.vocab_[term]) for i, a in enumerate(analyzed) for term in a.terms if term in self.vocab_],
            dtype=np.int64,
        ).reshape(-1, 2)
        counts = np.zeros((len(analyzed), len(self.vocab_)), dtype=np.float64)
        np.add.at(counts, (flat[:, 0], flat[:, 1]), 1.0)
        return counts

    def tf_matrix(self, analyzed: Sequence[AnalyzedText]) -> np.ndarray:
        """tf(t, d) = count(t, d) / 総形態素数(d)．shape: (n_texts, |V|)"""
        # shape: (n_texts, 1)
        num_morphemes = np.array([a.num_morphemes for a in analyzed], dtype=np.float64)[:, None]
        assert np.all(num_morphemes > 0), "総形態素数 0 は analyze で停止済みのはず"
        return self.count_matrix(analyzed) / num_morphemes

    # ---- BaseFeatureExtractor ----

    def fit(self, train_texts: Sequence[str]) -> Self:
        analyzed = self.analyze_all(train_texts)
        df_all = self.document_frequency(analyzed)
        if len(df_all) == 0:
            raise ValueError(f"{type(self).__name__}.fit: Train に索引語（内容語）が 1 語も無い")
        vocab = self.select_vocabulary(df_all, self.max_features)
        self.vocab_: dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.num_train_docs_: int = len(train_texts)  # N
        # shape: (|V|,)
        self.df_: np.ndarray = np.array([df_all[t] for t in vocab], dtype=np.float64)
        assert np.all(self.df_ >= 1), "語彙は Train に出現した語のみのため df(t) >= 1"
        assert np.all(self.df_ <= self.num_train_docs_)
        # shape: (|V|,)．df = N の語は厳密に log(1) = 0
        self.idf_: np.ndarray = np.log(self.num_train_docs_ / self.df_)
        assert np.all(np.isfinite(self.idf_)) and np.all(self.idf_ >= 0)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        # shape: (n_texts, |V|) * (|V|,) -> (n_texts, |V|)
        X = self.tf_matrix(self.analyze_all(texts)) * self.idf_[None, :]
        self.n_zero_rows_: int = int(np.sum(~np.any(X > 0, axis=1)))
        self.n_zero_rows_history_.append(self.n_zero_rows_)
        return X
