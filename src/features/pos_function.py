"""特徴量 B：品詞 2-gram・付属語の相対頻度（Ticket 3，講義第 2 回 p.11, p.16, p.30, p.36，第 5 回 p.11-12）．

2 ブロックを結合した非負ベクトルを出力（ `blocks` で片方のブロックのみの出力も可，Ticket 25）．

    品詞 2-gram ブロック：
        tag(m)        = (pos1, pos2)                                   （第 2 回 p.36 の品詞階層の上位 2 階層）
        x_bigram(b,d) = C(t_{i-1}, t_i) / (チャンク d の品詞 2-gram 総数)  （第 2 回 p.30）
        系列は補助記号・空白を含む全形態素（句読点の位置は統語情報のため保持）
    語ブロック（既定は付属語 = 助詞・助動詞，第 2 回 p.11）：
        x_word(w,d)   = count(w, d) / (チャンク d の総形態素数)        （第 5 回 p.12）
        総形態素数は補助記号・空白を除く形態素数（C と同じ定義）

第 5 回 p.11 で「索引語として不適切な不要語」とされる付属語を，著者識別では内容に依存しない文体の
手がかりとして用いる（Core Message）．`target_pos=["名詞", "動詞", "形容詞"]` は同じ表現のまま語の種類
だけを内容語に差し替えた統制条件．

語彙は Train の文書頻度 df >= min_df の要素から，ブロックごとに df 降順（同数は辞書順）で上位
`max_features` 件を選択．ブロックごとに上限を課すのは，内容語対照条件で語彙数の多い内容語が品詞 2-gram
の枠を奪い，「同じ品詞 2-gram ブロック」という統制が崩れるのを防ぐため．

`blocks`（Ticket 25）：
    all        [品詞 2-gram | 語] の結合（既定）
    pos_bigram 品詞 2-gram ブロックのみ（ `target_pos` ・ `token_key` は出力に影響しない）
    words      語ブロックのみ（付属語と内容語対照を，共通の品詞 2-gram ブロックに薄められずに比較するため）
    語彙の選択・各ブロックの分母は blocks によらず同一（ blocks=all の出力の列の部分集合と一致）．

形態素解析の結果はテキストごとにプロセス内でキャッシュ（ `src.morphology.morphemes_cached` ，Ticket 26 で
B・C・D に共通化）．解析結果は抽出器の引数に依存しないため，fold・シード・アブレーション条件の間で同じチャンクを
再解析しない（数値は不変）．
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

import numpy as np

from src.features.base import BaseFeatureExtractor
from src.morphology import TokenKey, morphemes_cached, morphemes_of, shared_tagger

Blocks = Literal["all", "pos_bigram", "words"]
ALLOWED_BLOCKS: tuple[str, ...] = ("all", "pos_bigram", "words")

# 付属語（第 2 回 p.11）
FUNCTION_POS: frozenset[str] = frozenset({"助詞", "助動詞"})
# 付属語 + 活用のない自立語のうち接続詞・連体詞・感動詞（第 2 回 p.11）
EXTENDED_POS: frozenset[str] = frozenset({"助詞", "助動詞", "接続詞", "連体詞", "感動詞"})
# 内容語対照（C の索引語と同じ品詞，第 5 回 p.11）
CONTENT_POS: frozenset[str] = frozenset({"名詞", "動詞", "形容詞"})
ALLOWED_TARGET_POS: tuple[frozenset[str], ...] = (FUNCTION_POS, EXTENDED_POS, CONTENT_POS)

# 総形態素数から除外する品詞（UniDic pos1，C と同じ）
EXCLUDED_POS: frozenset[str] = frozenset({"補助記号", "空白"})


@dataclass(frozen=True)
class POSAnalyzedText:
    """1 テキストの解析結果．

    pos_bigrams：品詞 2-gram の列（"pos1-pos2|pos1-pos2"，出現順，重複あり）．長さ = 全形態素数 - 1
    words：対象品詞の語の特徴量キー列（出現順，重複あり）
    word_pos1：words と同じ長さの pos1 列（対象品詞の検査用）
    num_morphemes：補助記号・空白を除く総形態素数（語ブロックの分母）
    """

    pos_bigrams: tuple[str, ...]
    words: tuple[str, ...]
    word_pos1: tuple[str, ...]
    num_morphemes: int


def validate_target_pos(target_pos: Iterable[str]) -> frozenset[str]:
    """付属語・拡張・内容語対照の 3 集合のいずれかと完全一致する場合のみ受理．"""
    if isinstance(target_pos, str):
        raise ValueError(f"target_pos は品詞名のリストであること（単一 str は不可）: {target_pos!r}")
    items = tuple(target_pos)
    pos_set = frozenset(items)
    if len(items) != len(pos_set) or pos_set not in ALLOWED_TARGET_POS:
        allowed = [sorted(s) for s in ALLOWED_TARGET_POS]
        raise ValueError(f"target_pos は {allowed} のいずれかと完全一致すること（付属語と内容語の混在は不可）: {items!r}")
    return pos_set


def top_function_words(
    texts: Sequence[str],
    n: int = 12,
    token_key: TokenKey = "surface",
    target_pos: Iterable[str] = ("助詞", "助動詞"),
) -> list[tuple[str, int]]:
    """作品単位の付属語頻度上位 n 語（講義第 5 回 p.14 の頻度表の再現用．評価パイプラインでは不使用）．

    texts は 1 作品分の文やチャンクの列．出現回数降順，同数は辞書順で決定的に並べる．
    """
    if not (isinstance(n, int) and n >= 1):
        raise ValueError(f"n は正の整数であること: {n!r}")
    pos_set = validate_target_pos(target_pos)
    # 作品単位の一回限りの集計のためキャッシュしない
    tagger = shared_tagger()
    counts = Counter(m.key(token_key) for text in texts for m in morphemes_of(tagger, text) if m.pos1 in pos_set)
    if len(counts) == 0:
        raise ValueError("top_function_words: 対象品詞の語が 1 語も無い")
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


class POSFunctionExtractor(BaseFeatureExtractor):
    """[品詞 2-gram 相対頻度 | 対象品詞の語の相対頻度] を結合した非負ベクトル．"""

    non_negative: ClassVar[bool] = True

    def __init__(
        self,
        token_key: TokenKey = "surface",
        target_pos: Sequence[str] = ("助詞", "助動詞"),
        max_features: int = 1500,
        min_df: int = 1,
        blocks: Blocks = "all",
    ) -> None:
        if token_key not in ("surface", "lemma"):
            raise ValueError(f"token_key は 'surface' または 'lemma': {token_key!r}")
        if not (isinstance(max_features, int) and max_features >= 1):
            raise ValueError(f"max_features は正の整数であること: {max_features!r}")
        if not (isinstance(min_df, int) and min_df >= 1):
            raise ValueError(f"min_df は正の整数であること: {min_df!r}")
        if blocks not in ALLOWED_BLOCKS:
            raise ValueError(f"blocks は {list(ALLOWED_BLOCKS)} のいずれか: {blocks!r}")
        self.token_key: TokenKey = token_key
        # config のリストを共有するため書き換えず，不変な tuple / frozenset で保持
        self.target_pos: tuple[str, ...] = tuple(target_pos)
        self.target_pos_set: frozenset[str] = validate_target_pos(self.target_pos)
        self.max_features = max_features
        self.min_df = min_df
        self.blocks: Blocks = blocks

    # ---- 形態素解析 ----

    def analyze(self, text: str) -> POSAnalyzedText:
        morphs = morphemes_cached(text)
        tags = [f"{m.pos1}-{m.pos2}" for m in morphs]
        pos_bigrams = tuple(f"{prev}|{cur}" for prev, cur in zip(tags, tags[1:], strict=False))
        num_morphemes = sum(m.pos1 not in EXCLUDED_POS for m in morphs)
        if num_morphemes == 0:
            raise ValueError(f"{type(self).__name__}: 総形態素数が 0 のテキスト: {text[:50]!r}")
        if len(pos_bigrams) == 0:
            raise ValueError(f"{type(self).__name__}: 品詞 2-gram 総数が 0 のテキスト: {text[:50]!r}")
        targets = [m for m in morphs if m.pos1 in self.target_pos_set]
        word_pos1 = tuple(m.pos1 for m in targets)
        assert set(word_pos1) <= self.target_pos_set, f"対象外の品詞が混入: {set(word_pos1) - self.target_pos_set}"
        return POSAnalyzedText(
            pos_bigrams=pos_bigrams,
            words=tuple(m.key(self.token_key) for m in targets),
            word_pos1=word_pos1,
            num_morphemes=num_morphemes,
        )

    def analyze_all(self, texts: Sequence[str]) -> list[POSAnalyzedText]:
        return [self.analyze(t) for t in texts]

    # ---- 語彙選択・頻度集計 ----

    def select_vocabulary(self, docs: Sequence[tuple[str, ...]]) -> dict[str, int]:
        """df >= min_df の要素を df 降順・同数は辞書順で上位 max_features 件．"""
        df = Counter(item for doc in docs for item in set(doc))
        eligible = sorted((t for t in df if df[t] >= self.min_df), key=lambda t: (-df[t], t))
        return {t: i for i, t in enumerate(eligible[: self.max_features])}

    @staticmethod
    def count_matrix(docs: Sequence[tuple[str, ...]], vocab: dict[str, int]) -> np.ndarray:
        """語彙上の出現回数行列（語彙外は無視）．shape: (n_texts, |vocab|)"""
        flat = np.array(
            [(i, vocab[item]) for i, doc in enumerate(docs) for item in doc if item in vocab],
            dtype=np.int64,
        ).reshape(-1, 2)
        counts = np.zeros((len(docs), len(vocab)), dtype=np.float64)
        np.add.at(counts, (flat[:, 0], flat[:, 1]), 1.0)
        return counts

    def bigram_block(self, analyzed: Sequence[POSAnalyzedText]) -> np.ndarray:
        """C(t_{i-1}, t_i) / 品詞 2-gram 総数（語彙外の 2-gram も分母に含む）．shape: (n_texts, |V_bigram|)"""
        docs = [a.pos_bigrams for a in analyzed]
        # shape: (n_texts, 1)
        totals = np.array([len(d) for d in docs], dtype=np.float64)[:, None]
        assert np.all(totals > 0), "品詞 2-gram 総数 0 は analyze で停止済みのはず"
        return self.count_matrix(docs, self.bigram_vocab_) / totals

    def word_block(self, analyzed: Sequence[POSAnalyzedText]) -> np.ndarray:
        """count(w, d) / 総形態素数(d)．shape: (n_texts, |V_word|)"""
        # shape: (n_texts, 1)
        totals = np.array([a.num_morphemes for a in analyzed], dtype=np.float64)[:, None]
        assert np.all(totals > 0), "総形態素数 0 は analyze で停止済みのはず"
        return self.count_matrix([a.words for a in analyzed], self.word_vocab_) / totals

    # ---- BaseFeatureExtractor ----

    def fit(self, train_texts: Sequence[str]) -> Self:
        analyzed = self.analyze_all(train_texts)
        self.bigram_vocab_: dict[str, int] = self.select_vocabulary([a.pos_bigrams for a in analyzed])
        self.word_vocab_: dict[str, int] = self.select_vocabulary([a.words for a in analyzed])
        if (self.blocks != "words" and len(self.bigram_vocab_) == 0) or (
            self.blocks != "pos_bigram" and len(self.word_vocab_) == 0
        ):
            raise ValueError(
                f"{type(self).__name__}.fit: 語彙が空（品詞 2-gram {len(self.bigram_vocab_)} 件，"
                f"語 {len(self.word_vocab_)} 件，min_df={self.min_df}）"
            )
        # 語彙の各語が Train で取った pos1 の集合（対象品詞の検査・考察用）
        pairs = sorted({(w, p) for a in analyzed for w, p in zip(a.words, a.word_pos1, strict=True)})
        self.word_pos1_: dict[str, frozenset[str]] = {
            w: frozenset(p for _, p in group)
            for w, group in itertools.groupby(pairs, key=lambda wp: wp[0])
            if w in self.word_vocab_
        }
        assert all(ps <= self.target_pos_set for ps in self.word_pos1_.values())
        self.num_bigram_features_: int = len(self.bigram_vocab_)
        self.num_word_features_: int = len(self.word_vocab_)
        return self

    @property
    def num_output_features_(self) -> int:
        """blocks に応じた transform 出力の列数．"""
        return {
            "all": self.num_bigram_features_ + self.num_word_features_,
            "pos_bigram": self.num_bigram_features_,
            "words": self.num_word_features_,
        }[self.blocks]

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        analyzed = self.analyze_all(texts)
        parts = {
            "all": (self.bigram_block, self.word_block),
            "pos_bigram": (self.bigram_block,),
            "words": (self.word_block,),
        }[self.blocks]
        # shape: (n_texts, num_output_features_)
        X = np.hstack([block(analyzed) for block in parts])
        assert X.shape == (len(texts), self.num_output_features_)
        assert np.all(X >= 0) and np.all(np.isfinite(X))
        return X

    def split_blocks(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """blocks=all の transform 出力を (品詞 2-gram ブロック, 語ブロック) に分割．"""
        if self.blocks != "all":
            raise ValueError(f"split_blocks は blocks='all' の出力のみ対象: blocks={self.blocks!r}")
        assert X.ndim == 2 and X.shape[1] == self.num_bigram_features_ + self.num_word_features_
        return X[:, : self.num_bigram_features_], X[:, self.num_bigram_features_ :]
