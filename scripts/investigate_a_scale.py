"""特徴量 A のブロック別スケール事前調査（Ticket 010，使い捨ての記述統計スクリプト）．

注意：本スクリプトは記述統計であり評価ではない．Bi-gram 語彙と Min-Max の範囲は全チャンクから算出しており
（Train / Test の区別なし），評価パイプラインには使用しないこと．特徴量 A の本実装は `src/features/stylometry_char.py`（002）．

調査対象の 3 構成（すべて「出現回数 ÷ 総文字数」の相対頻度，総文字数は空白・改行を除いた文字数）：
    (i)   案 C：文字 Bi-gram 上位 K + 記号 3 次元（読点・文末記号・開き括弧類）
    (ii)  旧案：(i) + 字種構成比 4 次元（漢字・ひらがな・カタカナ・その他）
    (iii) (ii) に各次元の Min-Max 正規化を適用（max = min の次元は 0 に固定）
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from src.dataset import ChunkSet, build_chunks, load_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHITESPACE = re.compile(r"\s")

# 記号クラス（002 の定義）
SYMBOL_CLASSES: dict[str, str] = {
    "comma": "、，",
    "sentence_end": "。．？！",
    "open_bracket": "「『（［【〔〈《(",
}
# 字種（Unicode ブロック．漢字には繰り返し記号「々」等を含める）
CHAR_TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "kanji": re.compile(r"[㐀-䶿一-鿿豈-﫿々〆〇]"),
    "hiragana": re.compile(r"[ぁ-ゟ]"),
    "katakana": re.compile(r"[゠-ヿㇰ-ㇿ]"),
}


@dataclass(frozen=True)
class FeatureBlocks:
    """1 つの L に対する全チャンクのブロック別特徴量．"""

    # shape: (num_chunks, K)，Bi-gram 出現回数 ÷ 総文字数
    bigram: np.ndarray
    # shape: (num_chunks, 3)
    symbol: np.ndarray
    # shape: (num_chunks, 4)，各行の和 = 1
    char_type: np.ndarray
    # Bi-gram 上位 K 種が全 Bi-gram 出現に占める割合
    coverage: float
    vocab: tuple[str, ...]


def strip_ws(texts: list[str]) -> list[str]:
    stripped = [WHITESPACE.sub("", t) for t in texts]
    empty = [i for i, t in enumerate(stripped) if not t]
    if empty:
        raise ValueError(f"総文字数 0 のチャンク: {empty[:10]}")
    return stripped


def bigram_block(texts: list[str], max_features: int) -> tuple[np.ndarray, float, tuple[str, ...]]:
    """全チャンクの文字 Bi-gram を数え，総出現回数上位 K（同数は辞書順）の相対頻度行列を返す．"""
    lengths = np.fromiter(map(len, texts), dtype=np.int64, count=len(texts))
    # shape: (total_chars,)，Unicode コードポイント
    codes = np.frombuffer("".join(texts).encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    chunk_id = np.repeat(np.arange(len(texts)), lengths)
    # 同一チャンク内で隣接する位置のみ Bi-gram とする（チャンク境界を跨がない）
    valid = chunk_id[:-1] == chunk_id[1:]
    # (第 1 文字, 第 2 文字) のコードポイント対の辞書順 = 2 文字列の辞書順
    pair_key = (codes[:-1] << 21 | codes[1:])[valid]
    pair_chunk = chunk_id[:-1][valid]
    uniq, inverse, counts = np.unique(pair_key, return_inverse=True, return_counts=True)
    # 出現回数の降順，同数は Bi-gram の辞書順（uniq は昇順）で決定的に選択
    top = np.lexsort((uniq, -counts))[:max_features]
    coverage = float(counts[top].sum() / counts.sum())
    vocab_index = np.full(uniq.size, -1, dtype=np.int64)
    vocab_index[top] = np.arange(top.size)
    col = vocab_index[inverse]
    in_vocab = col >= 0
    # shape: (num_chunks, K)
    X = np.zeros((len(texts), top.size), dtype=np.float64)
    np.add.at(X, (pair_chunk[in_vocab], col[in_vocab]), 1.0)
    vocab = tuple(chr(int(k >> 21)) + chr(int(k & ((1 << 21) - 1))) for k in uniq[top])
    return X / lengths[:, None], coverage, vocab


def symbol_block(texts: list[str]) -> np.ndarray:
    lengths = np.array([len(t) for t in texts], dtype=np.float64)
    # shape: (num_chunks, 3)
    counts = np.array([[sum(t.count(c) for c in chars) for chars in SYMBOL_CLASSES.values()] for t in texts], dtype=np.float64)
    return counts / lengths[:, None]


def char_type_block(texts: list[str]) -> np.ndarray:
    lengths = np.array([len(t) for t in texts], dtype=np.float64)
    # shape: (num_chunks, 3)，漢字・ひらがな・カタカナ
    counts = np.array([[len(p.findall(t)) for p in CHAR_TYPE_PATTERNS.values()] for t in texts], dtype=np.float64)
    other = lengths - counts.sum(axis=1)
    assert np.all(other >= 0), "字種の重複計数"
    # shape: (num_chunks, 4)
    ratios = np.column_stack([counts, other]) / lengths[:, None]
    assert np.allclose(ratios.sum(axis=1), 1.0, atol=1e-9)
    return ratios


def build_blocks(chunk_set: ChunkSet, max_features: int) -> FeatureBlocks:
    texts = strip_ws(chunk_set.texts)
    bigram, coverage, vocab = bigram_block(texts, max_features)
    blocks = FeatureBlocks(bigram, symbol_block(texts), char_type_block(texts), coverage, vocab)
    for name in ("bigram", "symbol", "char_type"):
        assert np.all(np.isfinite(getattr(blocks, name))), f"{name}: 非有限値"
    return blocks


def minmax(X: np.ndarray) -> tuple[np.ndarray, int]:
    """全チャンクの各次元の最小・最大で [0, 1] に写像．max = min の次元は 0 に固定し，その次元数を返す．"""
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo
    const = span == 0.0
    return np.where(const, 0.0, (X - lo) / np.where(const, 1.0, span)), int(const.sum())


def build_configs(blocks: FeatureBlocks) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    """構成名 → {ブロック名 → 行列}．"""
    k_bi, k_sym = blocks.bigram.shape[1], blocks.symbol.shape[1]
    full = np.hstack([blocks.bigram, blocks.symbol, blocks.char_type])
    scaled, num_const = minmax(full)
    configs = {
        "(i) C": {"bigram": blocks.bigram, "symbol": blocks.symbol},
        "(ii) old": {"bigram": blocks.bigram, "symbol": blocks.symbol, "char_type": blocks.char_type},
        "(iii) old+minmax": {
            "bigram": scaled[:, :k_bi],
            "symbol": scaled[:, k_bi : k_bi + k_sym],
            "char_type": scaled[:, k_bi + k_sym :],
        },
    }
    return configs, num_const


def norm_share_rows(config: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """各ブロックの二乗ノルムがベクトル全体の二乗ノルムに占める割合（チャンクごと）．"""
    # shape: (num_chunks,) per block
    sq = {name: np.einsum("nd,nd->n", X, X) for name, X in config.items()}
    total = sum(sq.values())
    assert np.all(total > 0.0), "二乗ノルム 0 のチャンク"
    return [(name, s / total) for name, s in sq.items()]


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    assert np.all(norms > 0.0), "ノルム 0 のベクトル（コサイン未定義）"
    U = X / norms[:, None]
    # shape: (n, n)
    return U @ U.T


def pair_correlation(config: dict[str, np.ndarray], block: str, idx: np.ndarray) -> float:
    """全ベクトルのコサイン類似度とブロック単独のコサイン類似度の，全チャンク対（i < j）上の Pearson 相関．"""
    full = np.hstack(list(config.values()))[idx]
    S_full, S_block = cosine_matrix(full), cosine_matrix(config[block][idx])
    iu = np.triu_indices(idx.size, k=1)
    return float(np.corrcoef(S_full[iu], S_block[iu])[0, 1])


def fmt_stats(v: np.ndarray) -> str:
    return f"{v.mean():>9.4f} {np.median(v):>9.4f} {v.min():>9.4f} {v.max():>9.4f}"


def report_L(L: int, chunk_set: ChunkSet, args: argparse.Namespace, rng: np.random.Generator) -> None:
    blocks = build_blocks(chunk_set, args.max_features)
    configs, num_const = build_configs(blocks)
    n = len(chunk_set)
    print(f"\n{'=' * 78}\nL = {L}  (num_chunks = {n}, bigram vocab K = {blocks.bigram.shape[1]})")
    print(f"Bi-gram 上位 {blocks.bigram.shape[1]} 種の被覆率: {blocks.coverage:.4f}")
    print(f"Min-Max で max = min となり 0 に固定した次元数: {num_const}")

    print("\n[二乗ノルム割合 ||x_block||^2 / ||x||^2]")
    print(f"{'config':<18} {'block':<10} {'mean':>9} {'median':>9} {'min':>9} {'max':>9}")
    for cname, config in configs.items():
        for bname, share in norm_share_rows(config):
            print(f"{cname:<18} {bname:<10} {fmt_stats(share)}")

    print("\n[次元ごとの値（チャンク方向）の中央値・最大値]")
    print("  Bi-gram ブロックは 1000 次元のため，次元別統計を次元方向に要約（median / max over dims）")
    print(f"{'config':<18} {'block':<10} {'dim':<14} {'median':>10} {'max':>10}")
    dim_names = {"symbol": list(SYMBOL_CLASSES), "char_type": [*CHAR_TYPE_PATTERNS, "other"]}
    for cname, config in configs.items():
        for bname, X in config.items():
            med, mx = np.median(X, axis=0), X.max(axis=0)
            if bname == "bigram":
                print(f"{cname:<18} {bname:<10} {'(median dim)':<14} {np.median(med):>10.5f} {np.median(mx):>10.5f}")
                print(f"{cname:<18} {bname:<10} {'(max dim)':<14} {med.max():>10.5f} {mx.max():>10.5f}")
                continue
            for d, dname in enumerate(dim_names[bname]):
                print(f"{cname:<18} {bname:<10} {dname:<14} {med[d]:>10.5f} {mx[d]:>10.5f}")
    top_dims = np.argsort(-np.median(blocks.bigram, axis=0), kind="stable")[:5]
    top_str = ", ".join(f"{blocks.vocab[d]!r}:{np.median(blocks.bigram[:, d]):.4f}" for d in top_dims)
    print(f"  参考：(i) Bi-gram 中央値上位 5 次元 {top_str}")

    idx = np.sort(rng.choice(n, size=args.pair_sample, replace=False)) if n > args.pair_sample else np.arange(n)
    print(f"\n[コサイン類似度の相関（全体 vs ブロック単独，{idx.size} チャンク / {idx.size * (idx.size - 1) // 2} 対）]")
    print(f"{'config':<18} {'vs block':<10} {'pearson r':>10}")
    for cname, config in configs.items():
        for bname in config:
            print(f"{cname:<18} {bname:<10} {pair_correlation(config, bname, idx):>10.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    p.add_argument("--max-features", type=int, default=1000, help="文字 Bi-gram 語彙数 K（002 の既定）")
    p.add_argument("--pair-sample", type=int, default=500, help="類似度相関を計算するチャンク数の上限（超過時は無作為抽出）")
    p.add_argument("--seed", type=int, default=None, help="未指定時は config の seed")
    p.add_argument("--chunk-chars", type=int, nargs="+", default=None, help="未指定時は config の chunking.target_chunk_chars")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"] if args.seed is None else args.seed
    chunk_chars = cfg["chunking"]["target_chunk_chars"] if args.chunk_chars is None else args.chunk_chars
    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    print(f"seed = {seed}, L = {chunk_chars}, K = {args.max_features}, pair_sample = {args.pair_sample}")
    print("注意：語彙・Min-Max 範囲は全チャンクから算出（記述統計専用，評価には不使用）")
    rng = np.random.default_rng(seed)
    for L in chunk_chars:
        report_L(L, build_chunks(corpus, L), args, rng)


if __name__ == "__main__":
    main()
