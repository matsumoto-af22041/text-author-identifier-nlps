"""対象作品の読み込み，文境界保持チャンク分割，Intra-work / Inter-work データ分割器．

前提：`scripts/inspect_and_extract_corpus.py` が出力した `authors_corpus.json`（本文 1 行 = 1 文）．
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WorkText:
    """1 作品分の文列．"""

    work_id: str
    author_id: str
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Corpus:
    """対象作家（ラベル順を固定）と作品群．"""

    author_ids: tuple[str, ...]
    author_names: tuple[str, ...]
    works: tuple[WorkText, ...]


@dataclass(frozen=True)
class Chunk:
    """文境界で区切った評価単位．`text` は文を区切り文字なしで連結した文字列．"""

    text: str
    sentences: tuple[str, ...]
    label: int
    author_id: str
    work_id: str
    index_in_work: int


@dataclass(frozen=True)
class ChunkSet:
    """チャンク列と，分割・評価用の整列済み配列．"""

    chunks: tuple[Chunk, ...]
    author_ids: tuple[str, ...]
    target_chunk_chars: int

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]

    @property
    def labels(self) -> np.ndarray:
        # shape: (num_chunks,), 作家ラベル（author_ids のインデックス）
        return np.array([c.label for c in self.chunks], dtype=np.int64)

    @property
    def work_ids(self) -> np.ndarray:
        # shape: (num_chunks,)
        return np.array([c.work_id for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class Split:
    """Train / Test のチャンクインデックス（ChunkSet 内の位置）．"""

    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.train_idx, self.test_idx).size:
            raise ValueError(f"[{self.name}] Train と Test のインデックスが重複")
        if self.train_idx.size == 0 or self.test_idx.size == 0:
            raise ValueError(f"[{self.name}] Train または Test が空")


def load_corpus(path: Path) -> Corpus:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    authors = data["authors"]
    corpus = Corpus(
        author_ids=tuple(a["id"] for a in authors),
        author_names=tuple(a["name"] for a in authors),
        works=tuple(
            WorkText(w["work_id"], w["author_id"], w["title"], tuple(w["sentences"])) for w in data["works"]
        ),
    )
    unknown = {w.author_id for w in corpus.works} - set(corpus.author_ids)
    if unknown:
        raise ValueError(f"authors に未定義の author_id: {unknown}")
    return corpus


def chunk_sentences(sentences: Sequence[str], target_chunk_chars: int) -> list[tuple[str, ...]]:
    """累積文字数が L 以上に達した文末で区切る．末尾の L 未満の端数チャンクは破棄．

    累積値 s_i は直前でチャンクが閉じていれば l_i にリセット，そうでなければ s_{i-1} + l_i．
    """
    if target_chunk_chars <= 0:
        raise ValueError(f"target_chunk_chars は正の整数: {target_chunk_chars}")
    if not sentences:
        return []
    L = target_chunk_chars
    # shape: (num_sentences,)
    lengths = np.fromiter(map(len, sentences), dtype=np.int64, count=len(sentences))
    running = np.fromiter(
        itertools.accumulate(lengths, lambda acc, x: x if acc >= L else acc + x),
        dtype=np.int64,
        count=len(sentences),
    )
    is_end = running >= L
    # 最後の文末より後ろは端数として破棄
    num_complete = int(is_end.nonzero()[0][-1]) + 1 if is_end.any() else 0
    bounds = np.flatnonzero(is_end[:num_complete]) + 1
    starts = np.concatenate([[0], bounds[:-1]])
    return [tuple(sentences[s:e]) for s, e in zip(starts.tolist(), bounds.tolist(), strict=True)]


def build_chunks(corpus: Corpus, target_chunk_chars: int) -> ChunkSet:
    label_of = {a: i for i, a in enumerate(corpus.author_ids)}
    chunks = tuple(
        Chunk(
            text="".join(sents),
            sentences=sents,
            label=label_of[w.author_id],
            author_id=w.author_id,
            work_id=w.work_id,
            index_in_work=i,
        )
        for w in corpus.works
        for i, sents in enumerate(chunk_sentences(w.sentences, target_chunk_chars))
    )
    empty_works = {w.work_id for w in corpus.works} - {c.work_id for c in chunks}
    if empty_works:
        raise ValueError(f"L={target_chunk_chars} でチャンクが 1 つも得られない作品: {empty_works}")
    return ChunkSet(chunks=chunks, author_ids=corpus.author_ids, target_chunk_chars=target_chunk_chars)


def intra_work_split(chunk_set: ChunkSet, test_ratio: float, seed: int) -> Split:
    """同一作品内分割：各作品のチャンクを乱数で並べ替え，先頭 round(n * test_ratio) 個を Test とする．

    作品単位の層化により，全作品が Train / Test の両方に現れる（トピック漏洩ありの比較条件）．
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio は (0, 1) の範囲: {test_ratio}")
    rng = np.random.default_rng(seed)
    # shape: (num_chunks,)
    work_ids = chunk_set.work_ids
    uniq, work_codes, counts = np.unique(work_ids, return_inverse=True, return_counts=True)
    n_test = np.rint(counts * test_ratio).astype(np.int64)
    if np.any(n_test < 1) or np.any(n_test >= counts):
        bad = uniq[(n_test < 1) | (n_test >= counts)].tolist()
        raise ValueError(f"Train / Test の一方が空になる作品（チャンク数不足）: {bad}")
    # 作品コードを第 1 キー，乱数を第 2 キーとして整列し，作品内の乱数順位を算出
    order = np.lexsort((rng.random(len(work_ids)), work_codes))
    group_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rank = np.empty(len(work_ids), dtype=np.int64)
    rank[order] = np.arange(len(work_ids)) - group_start[work_codes[order]]
    is_test = rank < n_test[work_codes]
    return Split(
        name=f"intra_seed{seed}",
        train_idx=np.flatnonzero(~is_test),
        test_idx=np.flatnonzero(is_test),
    )


def inter_work_splits(chunk_set: ChunkSet) -> list[Split]:
    """作品単位分割（Leave-One-Work-Out）：各作品を 1 つずつ Test とし，残り全作品を Train とする．"""
    work_ids, labels = chunk_set.work_ids, chunk_set.labels
    num_authors = len(chunk_set.author_ids)
    splits = [
        Split(name=f"lowo_{w}", train_idx=np.flatnonzero(work_ids != w), test_idx=np.flatnonzero(work_ids == w))
        for w in dict.fromkeys(work_ids.tolist())
    ]
    # 保持作品を除いても全作家が Train に残ること（未登録作家の発生防止）
    missing = [s.name for s in splits if np.unique(labels[s.train_idx]).size != num_authors]
    if missing:
        raise ValueError(f"Train に存在しない作家が生じる fold: {missing}")
    return splits
