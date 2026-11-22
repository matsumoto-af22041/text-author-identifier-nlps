"""特徴量 A：文字 Bi-gram・記号の相対頻度（Ticket 002，形態素解析に依存しない文字列としての文書表現，第 5 回 p.10）．

すべての次元は「出現回数 ÷ チャンクの総文字数」の相対出現頻度（第 5 回 p.12 の tf 正規化を文字単位に適用）．
総文字数は空白・改行を除いた文字数．

    bigram ブロック：空白・改行を除いた文字列の文字 2-gram（第 2 回 p.25 の N-gram を文字に適用）．
                     語彙は Train の総出現回数上位 `max_features` 件（同数は文字列の辞書順）．
                     Bi-gram はチャンク内のみで数え，チャンク内の文境界は跨ぐ．
    symbol ブロック：読点「、，」，文末記号「。．？！」，開き括弧類「「『（［【〔〈《(」の 3 次元．
                     文数 = 文末記号の出現数（第 2 回 p.8）とみなすと，文末記号率 = 1 / 平均文長．

字種構成比（漢字・ひらがな・カタカナ・その他）は値が 1 桁大きくコサインを支配するため類似度ベクトルに含めず，
考察用の記述統計として `char_type_ratios` で公開（評価パイプラインでは不使用）．
記号・字種・Bi-gram の定義は 010 の調査スクリプト `scripts/investigate_a_scale.py` と同一．

`scaling="minmax"` は Train の各次元の最小値・最大値で [0, 1] に揃えるアブレーション（講義外の独自設計）．
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import ClassVar, Literal, Self

import numpy as np

from src.features.base import BaseFeatureExtractor

Scaling = Literal["none", "minmax"]
Blocks = Literal["all", "bigram", "symbol"]

WHITESPACE = re.compile(r"\s")
# 2 文字目のコードポイントを格納するビット幅（Unicode の最大コードポイント 0x10FFFF < 2^21）
_CODE_BITS = 21

# 記号クラス（010 と同一）
SYMBOL_CLASSES: dict[str, str] = {
    "comma": "、，",
    "sentence_end": "。．？！",
    "open_bracket": "「『（［【〔〈《(",
}
# 字種（Unicode ブロック．漢字に「々〆〇」，カタカナに長音「ー」（U+30FC）を含む．010 と同一）
CHAR_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "kanji": re.compile(r"[㐀-䶿一-鿿豈-﫿々〆〇]"),
    "hiragana": re.compile(r"[ぁ-ゟ]"),
    "katakana": re.compile(r"[゠-ヿㇰ-ㇿ]"),
}
CHAR_TYPE_NAMES: tuple[str, ...] = (*CHAR_TYPE_PATTERNS, "other")


def strip_whitespace(texts: Sequence[str]) -> list[str]:
    """空白・改行を除去．総文字数 0 のテキストを含む場合は ValueError（Fail-Fast）．"""
    stripped = [WHITESPACE.sub("", t) for t in texts]
    empty = [i for i, t in enumerate(stripped) if not t]
    if empty:
        raise ValueError(f"総文字数（空白・改行を除く）が 0 のテキスト: index {empty[:10]}")
    return stripped


def _lengths(texts: Sequence[str]) -> np.ndarray:
    # shape: (n,)，単位：文字
    return np.fromiter(map(len, texts), dtype=np.int64, count=len(texts))


def _bigram_keys(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """全テキストの文字 Bi-gram をコードポイント対の整数キーで返す（テキスト境界は跨がない）．

    返り値：(pair_key, pair_text)，いずれも shape: (num_bigrams,)．
    キー (cp1 << 21 | cp2) の大小順は 2 文字列の辞書順と一致．
    """
    lengths = _lengths(texts)
    # shape: (total_chars,)
    codes = np.frombuffer("".join(texts).encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    text_id = np.repeat(np.arange(len(texts)), lengths)
    valid = text_id[:-1] == text_id[1:]
    pair_key = (codes[:-1] << _CODE_BITS | codes[1:])[valid]
    return pair_key, text_id[:-1][valid]


def _decode_key(key: int) -> str:
    return chr(key >> _CODE_BITS) + chr(key & ((1 << _CODE_BITS) - 1))


def symbol_counts(texts: Sequence[str]) -> np.ndarray:
    """記号クラスごとの出現回数．shape: (n, 3)，列順は SYMBOL_CLASSES．"""
    return np.array([[sum(map(t.count, chars)) for chars in SYMBOL_CLASSES.values()] for t in texts], dtype=np.float64)


def char_type_ratios(texts: Sequence[str]) -> np.ndarray:
    """字種構成比（考察用の記述統計，評価パイプラインでは不使用）．

    shape: (n, 4)，列順は CHAR_TYPE_NAMES（漢字・ひらがな・カタカナ・その他），各行の和 = 1．
    作品別に求める場合は作品の全文（またはチャンク列を連結した文字列）を 1 要素として渡す．
    """
    if isinstance(texts, str):
        raise TypeError("char_type_ratios: テキスト列（Sequence[str]）を渡すこと")
    stripped = strip_whitespace(texts)
    lengths = _lengths(stripped).astype(np.float64)
    # shape: (n, 3)
    counts = np.array([[len(p.findall(t)) for p in CHAR_TYPE_PATTERNS.values()] for t in stripped], dtype=np.float64)
    other = lengths - counts.sum(axis=1)
    assert np.all(other >= 0), "字種の重複計数"
    # shape: (n, 4)
    ratios = np.column_stack([counts, other]) / lengths[:, None]
    assert np.allclose(ratios.sum(axis=1), 1.0, atol=1e-9)
    return ratios


class StylometryCharExtractor(BaseFeatureExtractor):
    """文字 Bi-gram 相対頻度（上位 `max_features`）と記号 3 次元の相対頻度．列順は [bigram | symbol]．"""

    non_negative: ClassVar[bool] = True

    valid_scalings: ClassVar[tuple[str, ...]] = ("none", "minmax")
    valid_blocks: ClassVar[tuple[str, ...]] = ("all", "bigram", "symbol")
    # Bi-gram 語彙数の上限（002）
    max_features_limit: ClassVar[int] = 1500

    def __init__(self, scaling: Scaling = "none", blocks: Blocks = "all", max_features: int = 1000) -> None:
        if scaling not in self.valid_scalings:
            raise ValueError(f"scaling は {self.valid_scalings} のいずれか: {scaling!r}")
        if blocks not in self.valid_blocks:
            raise ValueError(f"blocks は {self.valid_blocks} のいずれか: {blocks!r}")
        if not (isinstance(max_features, int) and not isinstance(max_features, bool) and 1 <= max_features <= self.max_features_limit):
            raise ValueError(f"max_features は 1 以上 {self.max_features_limit} 以下の整数: {max_features!r}")
        self.scaling = scaling
        self.blocks = blocks
        self.max_features = max_features

    @property
    def use_bigram(self) -> bool:
        return self.blocks in ("all", "bigram")

    @property
    def use_symbol(self) -> bool:
        return self.blocks in ("all", "symbol")

    @property
    def feature_names(self) -> list[str]:
        """出力列の名前（fit 後）．"""
        return [*(self.vocab_ if self.use_bigram else ()), *(SYMBOL_CLASSES if self.use_symbol else ())]

    def fit(self, train_texts: Sequence[str]) -> Self:
        stripped = strip_whitespace(train_texts)
        if self.use_bigram:
            pair_key, _ = _bigram_keys(stripped)
            if pair_key.size == 0:
                raise ValueError(f"{type(self).__name__}.fit: Train に文字 Bi-gram が存在しない")
            uniq, counts = np.unique(pair_key, return_counts=True)
            # 総出現回数の降順，同数は辞書順（uniq は昇順）で決定的に選択
            top = np.lexsort((uniq, -counts))[: self.max_features]
            # 列順 = 出現回数順．transform の検索用に昇順キーと列番号の対応も保持
            self.vocab_keys_ = uniq[top]
            self.vocab_ = tuple(_decode_key(int(k)) for k in self.vocab_keys_)
            self.coverage_ = float(counts[top].sum() / counts.sum())
            order = np.argsort(self.vocab_keys_)
            self._sorted_keys = self.vocab_keys_[order]
            self._sorted_cols = order
        else:
            self.vocab_ = ()
        if self.scaling == "minmax":
            # shape: (N_train, D)
            X = self._relative_frequency(stripped)
            self.min_ = X.min(axis=0)
            span = X.max(axis=0) - self.min_
            # Train で max = min の次元は分母 0 のため 0 に固定
            self.constant_dims_ = span == 0.0
            self.n_constant_dims_ = int(self.constant_dims_.sum())
            self.span_ = np.where(self.constant_dims_, 1.0, span)
        return self

    def _bigram_block(self, stripped: Sequence[str]) -> np.ndarray:
        """shape: (n, |vocab|)，Bi-gram 出現回数 ÷ 総文字数．未知 Bi-gram は無視．"""
        pair_key, pair_text = _bigram_keys(stripped)
        pos = np.searchsorted(self._sorted_keys, pair_key)
        pos_clipped = np.minimum(pos, self._sorted_keys.size - 1)
        in_vocab = self._sorted_keys[pos_clipped] == pair_key
        X = np.zeros((len(stripped), self.vocab_keys_.size), dtype=np.float64)
        np.add.at(X, (pair_text[in_vocab], self._sorted_cols[pos_clipped[in_vocab]]), 1.0)
        return X / _lengths(stripped)[:, None]

    def _relative_frequency(self, stripped: Sequence[str]) -> np.ndarray:
        """スケーリング前の相対頻度行列．shape: (n, D)．"""
        lengths = _lengths(stripped).astype(np.float64)
        blocks = [
            *([self._bigram_block(stripped)] if self.use_bigram else []),
            *([symbol_counts(stripped) / lengths[:, None]] if self.use_symbol else []),
        ]
        X = np.hstack(blocks)
        assert np.all(np.isfinite(X)), "相対頻度に非有限値"
        return X

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        X = self._relative_frequency(strip_whitespace(texts))
        if self.scaling == "minmax":
            assert X.shape[1] == self.min_.size, f"次元不一致: {X.shape[1]} != {self.min_.size}"
            X = np.where(self.constant_dims_, 0.0, np.clip((X - self.min_) / self.span_, 0.0, 1.0))
        return X
