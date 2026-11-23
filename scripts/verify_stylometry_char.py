"""特徴量 A（文字 Bi-gram・記号，`src/features/stylometry_char.py`）の検証スクリプト（Ticket 002）．

使い方：
    uv run python scripts/verify_stylometry_char.py [--config configs/default.yaml] [--chunk-chars 100 500 1000]

検証項目（002 の Verification Criteria）：
    V1 shape_finite     サンプル 5 件の fit_transform が (5, D_features) で全要素有限
    V2 symbol_by_hand   scaling=none の記号 3 次元が「出現回数 ÷ 総文字数（空白・改行を除く）」と小数点以下 6 桁まで一致
                        （例：「吾輩は猫である。名前はまだ無い。」の文末記号率 2 / 16）
    V3 char_type_sum    char_type_ratios の 4 区分の和が各行で 1.0（許容誤差 1e-9）
    V4 block_subsets    blocks=bigram の次元が Bi-gram 語彙数，blocks=symbol の次元が 3 で，blocks=all の対応列と一致
    V5 minmax_range     scaling=minmax の出力が [0, 1] に収まり，Train に無い文字を含むテキストの transform も同じ次元
追加の確認：
    FF fail_fast        不正な引数（scaling / blocks / max_features の範囲外）と総文字数 0 のテキストで ValueError
    M1 minmax_train     minmax の最小値が Train の相対頻度の列最小値と一致（Test を使わない）
    R1 real_vs_010      実コーパスの各 L で既定出力が 010 の scripts/investigate_a_scale.py の build_blocks と np.array_equal
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import investigate_a_scale  # noqa: E402

from src.dataset import build_chunks, load_corpus  # noqa: E402
from src.features.stylometry_char import SYMBOL_CLASSES, StylometryCharExtractor, char_type_ratios  # noqa: E402

SAMPLES: tuple[str, ...] = (
    "吾輩は猫である。名前はまだ無い。",
    "どこで生れたかとんと見当がつかぬ。何でも薄暗い、じめじめした所で泣いていた事だけは記憶している。",
    "「おい、君。」と彼は言った。『こころ』を読んだか？",
    "親譲りの無鉄砲で小供の時から損ばかりしている。",
    "山路を登りながら、こう考えた！　智に働けば角が立つ。\n情に棹させば流される。",
)
# Train に無い文字（漢字・ギリシャ文字・旧仮名・未登録の括弧）を含むテキスト
UNSEEN: tuple[str, ...] = ("龘龘鬱鬱、Ωαβ。", "全く新しい文字列ばかり；ゑゐ【括弧】！")
BAD_ARGS: tuple[dict[str, Any], ...] = ({"scaling": "std"}, {"blocks": "char"}, {"max_features": 0}, {"max_features": 2000})
BAD_TEXTS: tuple[tuple[str, ...], ...] = (("   \n",), ("abc", ""))
NUM_SYMBOLS = len(SYMBOL_CLASSES)
DIGITS = 6
ATOL_SUM = 1e-9


def raises_value_error(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def hand_symbol_rates(texts: Sequence[str]) -> np.ndarray:
    """記号クラスごとの「出現回数 ÷ 総文字数（空白・改行を除く）」．shape: (n, NUM_SYMBOLS)．"""
    stripped = ["".join(t.split()) for t in texts]
    return np.array([[sum(s.count(c) for c in chars) / len(s) for chars in SYMBOL_CLASSES.values()] for s in stripped])


def check_shape_finite() -> bool:
    ex = StylometryCharExtractor()
    X = ex.fit_transform(list(SAMPLES))
    ok = X.shape == (len(SAMPLES), ex.n_features) and bool(np.all(np.isfinite(X)))
    print(f"[V1] shape={X.shape} (n_features={ex.n_features}, |V_bigram|={len(ex.vocab_)}), finite={bool(np.all(np.isfinite(X)))}")
    return ok


def check_symbol_by_hand() -> bool:
    X = StylometryCharExtractor().fit_transform(list(SAMPLES))
    # shape: (n, NUM_SYMBOLS)，blocks=all の末尾 3 列が記号ブロック
    sym, hand = X[:, -NUM_SYMBOLS:], hand_symbol_rates(SAMPLES)
    match = bool(np.array_equal(np.round(sym, DIGITS), np.round(hand, DIGITS)))
    example = bool(round(float(sym[0, list(SYMBOL_CLASSES).index("sentence_end")]), DIGITS) == round(2 / 16, DIGITS))
    print(f"[V2] max |impl - hand|={np.abs(sym - hand).max():.2e}, 6 桁一致={match}, 例文の文末記号率={sym[0, 1]:.6f}（2/16={2 / 16:.6f}）: {example}")
    return match and example


def check_char_type_sum() -> bool:
    R = char_type_ratios(list(SAMPLES))
    err = float(np.abs(R.sum(axis=1) - 1.0).max())
    print(f"[V3] max |sum - 1|={err:.2e}（許容 {ATOL_SUM:g}）")
    return err <= ATOL_SUM


def check_block_subsets() -> bool:
    full = StylometryCharExtractor()
    X = full.fit_transform(list(SAMPLES))
    k = len(full.vocab_)
    Xb = StylometryCharExtractor(blocks="bigram").fit_transform(list(SAMPLES))
    Xs = StylometryCharExtractor(blocks="symbol").fit_transform(list(SAMPLES))
    ok_b = Xb.shape[1] == k and bool(np.array_equal(Xb, X[:, :k]))
    ok_s = Xs.shape[1] == NUM_SYMBOLS and bool(np.array_equal(Xs, X[:, k:]))
    print(f"[V4] bigram: dim={Xb.shape[1]}（語彙数 {k}），一致={ok_b} / symbol: dim={Xs.shape[1]}，一致={ok_s}")
    return ok_b and ok_s


def check_minmax() -> tuple[bool, bool]:
    mm = StylometryCharExtractor(scaling="minmax")
    Xm = mm.fit_transform(list(SAMPLES))
    Xu = mm.transform(list(UNSEEN))
    in_range = all(float(M.min()) >= 0.0 and float(M.max()) <= 1.0 for M in (Xm, Xu))
    same_dim = Xu.shape[1] == Xm.shape[1] == mm.n_features
    raw = StylometryCharExtractor().fit_transform(list(SAMPLES))
    train_only = bool(np.array_equal(mm.min_, raw.min(axis=0)))
    print(
        f"[V5] Train 値域=[{Xm.min():.3f}, {Xm.max():.3f}]，未知文字 Test 値域=[{Xu.min():.3f}, {Xu.max():.3f}]，"
        f"次元 {Xm.shape[1]} / {Xu.shape[1]}，定数次元 {mm.n_constant_dims_}"
    )
    print(f"[M1] minmax の最小値 = Train の列最小値: {train_only}")
    return in_range and same_dim, train_only


def check_fail_fast() -> bool:
    fitted = StylometryCharExtractor().fit(list(SAMPLES))
    args_ok = {str(kw): raises_value_error(lambda kw=kw: StylometryCharExtractor(**kw)) for kw in BAD_ARGS}
    texts_ok = {repr(t): raises_value_error(lambda t=t: fitted.transform(list(t))) for t in BAD_TEXTS}
    char_type_ok = raises_value_error(lambda: char_type_ratios(["\n"]))
    print(f"[FF] 引数: {args_ok}，総文字数 0: {texts_ok}，char_type_ratios 総文字数 0: {char_type_ok}")
    return all(args_ok.values()) and all(texts_ok.values()) and char_type_ok


def check_real_vs_010(corpus_path: Path, chunk_chars: Sequence[int], max_features: int) -> bool:
    corpus = load_corpus(corpus_path)

    def one(L: int) -> bool:
        t0 = time.perf_counter()
        cs = build_chunks(corpus, L)
        ref = investigate_a_scale.build_blocks(cs, max_features)
        ex = StylometryCharExtractor(max_features=max_features)
        X = ex.fit_transform(cs.texts)
        eq_x = bool(np.array_equal(X, np.hstack([ref.bigram, ref.symbol])))
        eq_v = ex.vocab_ == tuple(ref.vocab)
        eq_c = bool(np.array_equal(char_type_ratios(cs.texts), ref.char_type))
        print(f"[R1] L={L}: chunks={len(cs.texts)} shape={X.shape} 出力一致={eq_x} 語彙一致={eq_v} 字種一致={eq_c} ({time.perf_counter() - t0:.1f}s)")
        return eq_x and eq_v and eq_c

    return all([one(L) for L in chunk_chars])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--chunk-chars", nargs="+", type=int, help="R1 のチャンク長 L（既定：chunking.target_chunk_chars）")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    chunk_chars = args.chunk_chars or cfg["chunking"]["target_chunk_chars"]
    max_features = cfg["features"]["A"]["params"]["max_features"]

    v5, m1 = check_minmax()
    results = {
        "V1": check_shape_finite(),
        "V2": check_symbol_by_hand(),
        "V3": check_char_type_sum(),
        "V4": check_block_subsets(),
        "V5": v5,
        "M1": m1,
        "FF": check_fail_fast(),
        "R1": check_real_vs_010(PROJECT_ROOT / cfg["paths"]["processed_corpus"], chunk_chars, max_features),
    }
    print("\n".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in results.items()))
    if not all(results.values()):
        raise SystemExit(f"検証失敗: {[k for k, v in results.items() if not v]}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
