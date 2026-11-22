"""特徴量 B（品詞 2-gram・付属語，`src/features/pos_function.py`）の検証スクリプト（Ticket 003，025）．

使い方：
    uv run python scripts/verify_pos_function.py [--config configs/default.yaml] [--chunk-chars 500] [--reference-rev b5c2a54]

検証項目（003）：
    V1 noun_swap        名詞だけが異なる 2 文（猫 / 犬）で品詞 2-gram・語ブロックとも一致しコサイン 1.0
    V2 no_content_words 既定（付属語）の語彙に名詞・動詞・形容詞・副詞が入らない
    V3 content_control  内容語対照では 2 文の語ブロックが異なり，猫・犬が語彙に入る
    V4 target_pos_guard 混在・部分集合・単一 str・重複の 5 通りで ValueError，許可 3 集合は受理
    V5 bocchan_top12    『坊っちゃん』の付属語頻度上位 12 語が講義第 5 回 p.14 の 9 語を含む
    V6 real_chunks      実データ（L = --chunk-chars，偶数番目で fit，奇数番目で transform）の 4 条件で非負・有限
    FF fail_fast        総形態素数 0（「。」のみ）・品詞 2-gram 総数 0（1 形態素）で ValueError
検証項目（025）：
    B1 all_bit_equal    blocks=all の出力が blocks 追加前の実装（--reference-rev）と np.array_equal で一致
    B2 block_subsets    blocks=pos_bigram / words の出力が blocks=all を split_blocks で分けた各ブロックと一致し，
                        列数が各ブロックの次元数．pos_bigram の出力は target_pos・token_key によらず一致
    B3 blocks_guard     blocks="foo" で ValueError

V6・B1・B2 の実データでの fit は偶数番目のチャンク（検証用の分け方であり評価には使わない）．
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from src.dataset import build_chunks, load_corpus
from src.features.pos_function import CONTENT_POS, EXTENDED_POS, POSFunctionExtractor, top_function_words

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = "src/features/pos_function.py"
ATOL_COS = 1e-9

NOUN_PAIR = ("猫は庭で寝ている。", "犬は庭で寝ている。")
CONTENT_TRAIN = ("猫が走る。", "美しい花がゆっくり咲いた。", "猫はゆっくり走るのが美しい。")
CONTENT_TOKENS = frozenset({"猫", "走る", "美しい", "ゆっくり", "咲い"})
CONTENT_LIKE_POS = frozenset({"名詞", "動詞", "形容詞", "副詞"})
# 講義第 5 回 p.14 の『坊っちゃん』頻度上位の付属語
LECTURE_TOP = ("て", "だ", "た", "は", "を", "の", "に", "が", "と")
BAD_TARGET_POS: tuple[Any, ...] = (
    ["助詞", "名詞"],
    ["助詞"],
    ["助詞", "助動詞", "名詞", "動詞", "形容詞"],
    "助詞",
    ["助詞", "助詞", "助動詞"],
)
TARGET_POS_CONDITIONS: tuple[tuple[str, ...], ...] = (("助詞", "助動詞"), tuple(sorted(EXTENDED_POS)), tuple(sorted(CONTENT_POS)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def raises_value_error(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def load_reference_module(rev: str) -> ModuleType:
    """blocks 追加前の pos_function.py を git から取り出して別名で import．"""
    src = subprocess.run(["git", "show", f"{rev}:{MODULE_PATH}"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "pos_function_reference.py"
    path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("pos_function_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if "blocks" in inspect.signature(module.POSFunctionExtractor.__init__).parameters:
        raise RuntimeError(f"{rev} の {MODULE_PATH} は既に blocks 引数を持つ（blocks 追加前のリビジョンを指定すること）")
    return module


# ---------------------------------------------------------------- 003 の検証


def check_noun_swap() -> bool:
    ex = POSFunctionExtractor().fit(list(NOUN_PAIR))
    X = ex.transform(list(NOUN_PAIR))
    bg, wd = ex.split_blocks(X)
    c = cosine(X[0], X[1])
    print(f"[V1] word vocab={list(ex.word_vocab_)}, |V_bigram|={len(ex.bigram_vocab_)}, cos={c!r}")
    return np.array_equal(bg[0], bg[1]) and np.array_equal(wd[0], wd[1]) and abs(c - 1.0) < ATOL_COS


def check_no_content_words() -> bool:
    ex = POSFunctionExtractor().fit(list(CONTENT_TRAIN))
    print(f"[V2] word vocab pos1={ {w: sorted(p) for w, p in ex.word_pos1_.items()} }")
    return all(not (p & CONTENT_LIKE_POS) for p in ex.word_pos1_.values()) and not (CONTENT_TOKENS & set(ex.word_vocab_))


def check_content_control() -> bool:
    ex = POSFunctionExtractor(target_pos=list(sorted(CONTENT_POS))).fit(list(NOUN_PAIR))
    X = ex.transform(list(NOUN_PAIR))
    bg, wd = ex.split_blocks(X)
    print(f"[V3] word vocab={list(ex.word_vocab_)}, bigram equal={np.array_equal(bg[0], bg[1])}, cos={cosine(X[0], X[1]):.6f}")
    return not np.array_equal(wd[0], wd[1]) and {"猫", "犬"} <= set(ex.word_vocab_)


def check_target_pos_guard() -> bool:
    rejected = [raises_value_error(lambda tp=tp: POSFunctionExtractor(target_pos=tp)) for tp in BAD_TARGET_POS]
    accepted = [not raises_value_error(lambda tp=tp: POSFunctionExtractor(target_pos=list(tp))) for tp in TARGET_POS_CONDITIONS]
    print(f"[V4] rejected={rejected}, accepted={accepted}")
    return all(rejected) and all(accepted)


def check_bocchan(corpus_path: Path) -> bool:
    corpus = load_corpus(corpus_path)
    work = next(w for w in corpus.works if w.title == "坊っちゃん")
    top = top_function_words(list(work.sentences), n=12)
    hits = [w for w in LECTURE_TOP if w in {k for k, _ in top}]
    print(f"[V5] top12={top}, lecture overlap={len(hits)}/{len(LECTURE_TOP)}")
    return len(hits) == len(LECTURE_TOP)


def check_fail_fast() -> bool:
    results = [raises_value_error(lambda t=t: POSFunctionExtractor().fit([t])) for t in ("。", "猫")]
    print(f"[FF] ValueError for ['。', '猫']: {results}")
    return all(results)


# ---------------------------------------------------------------- 実データ（003 の V6，025 の B1・B2）


def check_real_chunks(train: Sequence[str], test: Sequence[str], reference: ModuleType) -> dict[str, bool]:
    conditions: tuple[dict[str, Any], ...] = (
        {},
        {"token_key": "lemma"},
        {"target_pos": list(TARGET_POS_CONDITIONS[1])},
        {"target_pos": list(TARGET_POS_CONDITIONS[2])},
    )
    ok = {"V6": True, "B1": True, "B2": True}
    pos_bigram_outputs: list[np.ndarray] = []
    for kw in conditions:
        t0 = time.time()
        ex = POSFunctionExtractor(**kw).fit(list(train))
        X_tr, X_te = ex.transform(list(train)), ex.transform(list(test))
        valid = all(np.all(X >= 0) and np.all(np.isfinite(X)) for X in (X_tr, X_te))
        vocab_pos = set().union(*ex.word_pos1_.values())
        ok["V6"] &= bool(valid) and vocab_pos <= ex.target_pos_set
        bg_tr, wd_tr = ex.split_blocks(X_tr)
        print(
            f"[V6] {kw or 'default'}: shape train={X_tr.shape} test={X_te.shape}, "
            f"|V_bigram|={ex.num_bigram_features_} |V_word|={ex.num_word_features_}, "
            f"zero test rows={int((~np.any(X_te > 0, axis=1)).sum())}, row sum mean bigram={bg_tr.sum(1).mean():.3f} "
            f"word={wd_tr.sum(1).mean():.3f}, vocab pos1={sorted(vocab_pos)}"
        )

        ref = reference.POSFunctionExtractor(**kw).fit(list(train))
        R_tr, R_te = ref.transform(list(train)), ref.transform(list(test))
        equal = np.array_equal(X_tr, R_tr) and np.array_equal(X_te, R_te)
        ok["B1"] &= bool(equal)
        print(f"[B1] {kw or 'default'}: blocks=all == reference: {equal}")

        for blocks, (expected_tr, expected_te), dim in (
            ("pos_bigram", (bg_tr, ex.split_blocks(X_te)[0]), ex.num_bigram_features_),
            ("words", (wd_tr, ex.split_blocks(X_te)[1]), ex.num_word_features_),
        ):
            sub = POSFunctionExtractor(**kw, blocks=blocks).fit(list(train))
            S_tr, S_te = sub.transform(list(train)), sub.transform(list(test))
            match = (
                np.array_equal(S_tr, expected_tr)
                and np.array_equal(S_te, expected_te)
                and S_tr.shape[1] == dim == sub.n_features
            )
            ok["B2"] &= bool(match)
            print(f"[B2] {kw or 'default'}: blocks={blocks} == split_blocks: {match} (cols={S_tr.shape[1]}, block dim={dim})")
            if blocks == "pos_bigram":
                pos_bigram_outputs.append(np.vstack([S_tr, S_te]))
        print(f"     ({time.time() - t0:.1f}s)")
    invariant = all(np.array_equal(pos_bigram_outputs[0], P) for P in pos_bigram_outputs[1:])
    print(f"[B2] blocks=pos_bigram invariant to target_pos / token_key: {invariant}")
    ok["B2"] &= invariant
    return ok


def check_blocks_guard() -> bool:
    rejected = raises_value_error(lambda: POSFunctionExtractor(blocks="foo"))  # type: ignore[arg-type]
    split_rejected = raises_value_error(
        lambda: (lambda e: e.split_blocks(e.transform(list(NOUN_PAIR))))(POSFunctionExtractor(blocks="words").fit(list(NOUN_PAIR)))
    )
    print(f"[B3] blocks='foo' ValueError: {rejected}, split_blocks on blocks='words' ValueError: {split_rejected}")
    return rejected and split_rejected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--chunk-chars", type=int, default=500, help="実データ検証のチャンク長 L")
    parser.add_argument("--reference-rev", default="b5c2a54", help="blocks 追加前の pos_function.py を含む git リビジョン")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    reference = load_reference_module(args.reference_rev)

    texts = build_chunks(load_corpus(corpus_path), args.chunk_chars).texts
    results = {
        "V1": check_noun_swap(),
        "V2": check_no_content_words(),
        "V3": check_content_control(),
        "V4": check_target_pos_guard(),
        "V5": check_bocchan(corpus_path),
        "FF": check_fail_fast(),
        **check_real_chunks(texts[::2], texts[1::2], reference),
        "B3": check_blocks_guard(),
    }
    print("\n".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    if not all(results.values()):
        raise SystemExit(f"検証失敗: {[k for k, v in results.items() if not v]}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
