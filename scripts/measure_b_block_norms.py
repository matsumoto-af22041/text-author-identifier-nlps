"""特徴量 B のブロック別二乗ノルム割合の測定（Ticket 025，記述統計スクリプト）．

注意：本スクリプトは記述統計であり評価ではない．語彙は L ごとの全チャンクで fit しており（Train / Test の区別なし），
評価パイプラインには使用しないこと．

指標（010 の `scripts/investigate_a_scale.py` と同じ）：チャンク d ごとに
    share_block(d) = ||x_block(d)||^2 / ||x(d)||^2      （x = [x_bigram | x_word]，blocks=all の出力）
を求め，全チャンクの平均を報告．2 ブロックの割合の和は各チャンクで 1．コサインの内積
x(d)・x(d') = x_bigram(d)・x_bigram(d') + x_word(d)・x_word(d') への各ブロックの寄与の目安．

条件：L ∈ chunking.target_chunk_chars × target_pos 3 条件（付属語，拡張，内容語対照），token_key = surface，
その他の引数は configs/default.yaml の features.B.params．

使い方：
    uv run python scripts/measure_b_block_norms.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from src.dataset import build_chunks, load_corpus
from src.features.pos_function import CONTENT_POS, EXTENDED_POS, FUNCTION_POS, POSFunctionExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_POS_CONDITIONS: dict[str, list[str]] = {
    "function": sorted(FUNCTION_POS),
    "extended": sorted(EXTENDED_POS),
    "content": sorted(CONTENT_POS),
}


def block_norm_shares(ex: POSFunctionExtractor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """チャンクごとの (品詞 2-gram, 語) の二乗ノルム割合．shape: (n_chunks,) × 2"""
    bigram, word = ex.split_blocks(X)
    # shape: (n_chunks,)
    sq_bigram, sq_word = (bigram**2).sum(axis=1), (word**2).sum(axis=1)
    total = sq_bigram + sq_word
    assert np.all(total > 0), "ノルム 0 のチャンク（品詞 2-gram 総数 > 0 のため起こらないはず）"
    share_bigram, share_word = sq_bigram / total, sq_word / total
    assert np.allclose(share_bigram + share_word, 1.0) and np.all(np.isfinite(share_bigram))
    return share_bigram, share_word


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    base_params = cfg["features"]["B"]["params"] | {"token_key": "surface", "blocks": "all"}

    print("| L | target_pos | n_chunks | |V_bigram| | |V_word| | share_bigram (mean) | share_word (mean) | share_word (min / max) |")
    print("|---|---|---|---|---|---|---|---|")
    for L in cfg["chunking"]["target_chunk_chars"]:
        texts = build_chunks(corpus, L).texts
        for name, target_pos in TARGET_POS_CONDITIONS.items():
            ex = POSFunctionExtractor(**(base_params | {"target_pos": target_pos}))
            X = ex.fit_transform(texts)
            share_bigram, share_word = block_norm_shares(ex, X)
            print(
                f"| {L} | {name} | {len(texts)} | {ex.num_bigram_features_} | {ex.num_word_features_} | "
                f"{share_bigram.mean():.3f} | {share_word.mean():.3f} | {share_word.min():.3f} / {share_word.max():.3f} |"
            )


if __name__ == "__main__":
    main()
