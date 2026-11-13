"""共通基盤（Ticket 1）の検証：数値的健全性，単一バッチ過学習，乱数シード再現性，実データでの分割・漏洩ガード．"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import yaml

from src.classifier import CentroidClassifier, KNNClassifier
from src.dataset import build_chunks, chunk_sentences, inter_work_splits, intra_work_split, load_corpus
from src.evaluator import ClassifierSpec, aggregate_results, evaluate_fold, plot_similarity_distribution
from src.features.base import BaseFeatureExtractor, NotFittedError
from src.features.char_unigram import CharUnigramExtractor
from src.similarity import SIMILARITY_FUNCTIONS, cosine_similarity, jaccard_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6


class IdentityExtractor(BaseFeatureExtractor):
    """事前計算済みベクトル（テキスト ID → 行）をそのまま返すダミー抽出器．"""

    def __init__(self, table: dict[str, np.ndarray]) -> None:
        self.table = table

    def fit(self, train_texts: Sequence[str]) -> Self:
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.table[t] for t in texts])



def check_numerical_sanity(rng: np.random.Generator) -> None:
    # shape: (10, 50)，非負
    X = rng.random((10, 50))
    X[rng.random(X.shape) < 0.3] = 0.0
    for name, fn in SIMILARITY_FUNCTIONS.items():
        S = fn(X, X)
        assert S.shape == (10, 10), f"{name}: 形状 {S.shape}"
        assert np.allclose(np.diag(S), 1.0, atol=TOL), f"{name}: 対角 {np.diag(S)}"
        assert np.all((S >= 0.0) & (S <= 1.0)), f"{name}: [0, 1] 範囲外"
        assert np.allclose(S, S.T, atol=TOL), f"{name}: 非対称"
        # 素朴な定義式（ペアごと）と一致すること
        i, j = 2, 7
        x, y = X[i], X[j]
        ref = {
            "cosine": x @ y / (np.linalg.norm(x) * np.linalg.norm(y)),
            "jaccard": np.minimum(x, y).sum() / np.maximum(x, y).sum(),
            "dice": 2 * np.minimum(x, y).sum() / (x + y).sum(),
        }[name]
        assert abs(S[i, j] - ref) < TOL, f"{name}: 定義式との差 {S[i, j] - ref}"
        print(f"  [ok] {name:<7} diag=1 (max err {np.abs(np.diag(S) - 1).max():.1e}), range [{S.min():.3f}, {S.max():.3f}]")
    # バッチ分割の有無で Σ min の結果が一致
    S_small = jaccard_similarity(X, X, max_batch_bytes=1)
    assert np.allclose(S_small, jaccard_similarity(X, X), atol=1e-12), "バッチサイズで結果が変化"
    # 負値入力：cosine は [-1, 1]，Jaccard は Fail-Fast
    Z = rng.standard_normal((5, 8))
    assert np.allclose(np.diag(cosine_similarity(Z, Z)), 1.0, atol=TOL)
    try:
        jaccard_similarity(Z, Z)
    except ValueError:
        print("  [ok] 負値入力に対する Jaccard の停止")
    else:
        raise AssertionError("負値入力で Jaccard が停止しない")


def check_single_batch_overfit(rng: np.random.Generator, num_classes: int = 4) -> None:
    # 作家ごとに異なるプロトタイプ + 小ノイズの非負ダミー特徴（Train = Test）
    per_class, dim = 12, 50
    protos = rng.random((num_classes, dim)) ** 4
    y = np.repeat(np.arange(num_classes), per_class)
    X = protos[y] + 0.01 * rng.random((y.size, dim))
    for name, fn in SIMILARITY_FUNCTIONS.items():
        acc_c = (CentroidClassifier(fn, num_classes).fit(X, y).predict(X) == y).mean()
        acc_k = (KNNClassifier(fn, num_classes, k=1).fit(X, y).predict(X) == y).mean()
        acc_g = (KNNClassifier(fn, num_classes, k=1, mode="global").fit(X, y).predict(X) == y).mean()
        assert acc_c == acc_k == acc_g == 1.0, f"{name}: centroid={acc_c}, knn1={acc_k}, global1={acc_g}"
        print(f"  [ok] {name:<7} centroid / knn(k=1, per_author) / knn(k=1, global) accuracy = 1.0")
    # 抽出器経由（IdentityExtractor）でも Train = Test で 100%
    texts = [f"doc{i}" for i in range(y.size)]
    ext = IdentityExtractor(dict(zip(texts, X, strict=True)))
    Xt = ext.fit(texts).transform(texts)
    assert (CentroidClassifier(cosine_similarity, num_classes).fit(Xt, y).predict(Xt) == y).all()
    print("  [ok] IdentityExtractor 経由でも accuracy = 1.0")


def check_seed_reproducibility(chunk_set, ratio: float, seed: int) -> None:
    a, b = intra_work_split(chunk_set, ratio, seed), intra_work_split(chunk_set, ratio, seed)
    assert np.array_equal(a.train_idx, b.train_idx) and np.array_equal(a.test_idx, b.test_idx)
    c = intra_work_split(chunk_set, ratio, seed + 1)
    assert not np.array_equal(a.test_idx, c.test_idx), "異なるシードで同一分割"
    print(f"  [ok] seed={seed} の 2 回実行で Train / Test インデックス完全一致（seed={seed + 1} では相違）")


def check_leakage_guard() -> None:
    ext = CharUnigramExtractor()
    try:
        ext.transform(["テスト"])
    except NotFittedError:
        print("  [ok] fit 前の transform を停止")
    else:
        raise AssertionError("fit 前の transform が通過")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]
    random.seed(seed)
    rng = np.random.default_rng(seed)

    print("[1] 数値的健全性チェック")
    check_numerical_sanity(rng)
    print("[2] 単一バッチ過学習チェック")
    check_single_batch_overfit(rng)
    print("[3] 漏洩ガード")
    check_leakage_guard()

    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    assert chunk_sentences(["あい", "う", "えおか", "き"], 3) == [("あい", "う"), ("えおか",)], "チャンク分割の境界"

    print("[4] 実データのチャンク分割と分割器")
    ratio = cfg["split"]["intra_test_ratio"]
    for L in cfg["chunking"]["target_chunk_chars"]:
        cs = build_chunks(corpus, L)
        lens = np.array([len(c.text) for c in cs.chunks])
        assert lens.min() >= L, f"L={L} 未満のチャンク"
        per_author = Counter(cs.author_ids[c.label] for c in cs.chunks)
        intra = intra_work_split(cs, ratio, seed)
        lowo = inter_work_splits(cs)
        assert all(np.union1d(s.train_idx, s.test_idx).size == len(cs) for s in [intra, *lowo])
        print(
            f"  L={L:>4}: chunks={len(cs):>5}  文字数 median={int(np.median(lens))} max={lens.max()}  "
            f"{dict(per_author)}  intra train/test={intra.train_idx.size}/{intra.test_idx.size}  LOWO folds={len(lowo)}"
        )
    print("[5] 乱数シード再現性検証")
    check_seed_reproducibility(build_chunks(corpus, 500), ratio, seed)

    print("[6] 実データ通し実行（ダミー文字 unigram，L=1000，Intra + LOWO 全 fold）")
    cs = build_chunks(corpus, 1000)
    specs = [
        ClassifierSpec("centroid"),
        ClassifierSpec("knn", k=1, knn_mode="per_author"),
        ClassifierSpec("knn", k=5, knn_mode="global"),
    ]
    t0 = time.perf_counter()
    intra_res = evaluate_fold(CharUnigramExtractor, cs, intra_work_split(cs, ratio, seed), ["cosine"], specs)
    lowo_res = [r for s in inter_work_splits(cs) for r in evaluate_fold(CharUnigramExtractor, cs, s, ["jaccard"], specs)]
    for r in intra_res:
        st = r.sim_stats
        print(f"  intra/cosine  {r.spec.name:<16}: acc={r.metrics['accuracy']:.3f} F1={r.metrics['macro_f1']:.3f} "
              f"MRR={r.mrr:.3f} ΔS_macro={st.delta_s_macro:.4f} ΔS_micro={st.delta_s_micro:.4f}")
    for spec in specs:
        agg = aggregate_results([r for r in lowo_res if r.spec == spec])
        assert agg.num_folds == 9 and agg.n_test == len(cs), "LOWO の pooled 統合で全チャンクが 1 回ずつ Test に入ること"
        print(f"  LOWO/jaccard  {spec.name:<16}: acc={agg.metrics['accuracy']:.3f} F1={agg.metrics['macro_f1']:.3f} "
              f"MRR={agg.mrr:.3f} ΔS_macro={agg.sim_stats.delta_s_macro:.4f}")
    print(f"  所要時間 {time.perf_counter() - t0:.1f}s")
    out = PROJECT_ROOT / "outputs" / "verify" / "similarity_distribution_intra_cosine.png"
    plot_similarity_distribution(intra_res[0].sim_stats, out)
    print(f"  描画確認: {out.relative_to(PROJECT_ROOT)}")
    print("\nすべての検証に合格")


if __name__ == "__main__":
    main()
