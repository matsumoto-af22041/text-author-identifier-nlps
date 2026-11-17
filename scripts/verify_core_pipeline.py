"""共通基盤（001，008）の検証：数値的健全性，単一バッチ過学習，同点処理，乱数シード再現性，実データでの分割・漏洩ガード．"""

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

from src.classifier import CentroidClassifier, KNNClassifier, predict_from_scores, tie_label
from src.dataset import Chunk, ChunkSet, Split, build_chunks, chunk_sentences, inter_work_splits, intra_work_split, load_corpus
from src.evaluator import (
    ClassifierSpec,
    aggregate_results,
    classification_metrics,
    confusion_matrix,
    evaluate_fold,
    plot_confusion_matrix,
    plot_similarity_distribution,
    reciprocal_ranks,
)
from src.features.base import BaseFeatureExtractor, NotFittedError
from src.features.char_unigram import CharUnigramExtractor
from src.similarity import SIMILARITY_FUNCTIONS, cosine_similarity

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
    assert list(SIMILARITY_FUNCTIONS) == ["cosine"], f"類似度はコサインのみ: {list(SIMILARITY_FUNCTIONS)}"
    # shape: (10, 50)，非負
    X = rng.random((10, 50))
    X[rng.random(X.shape) < 0.3] = 0.0
    S = cosine_similarity(X, X)
    assert S.shape == (10, 10), f"形状 {S.shape}"
    assert np.allclose(np.diag(S), 1.0, atol=TOL), f"対角 {np.diag(S)}"
    assert np.all((S >= 0.0) & (S <= 1.0)), "[0, 1] 範囲外"
    assert np.allclose(S, S.T, atol=TOL), "非対称"
    # 素朴な定義式（ペアごと）と一致すること
    i, j = 2, 7
    x, y = X[i], X[j]
    ref = x @ y / (np.linalg.norm(x) * np.linalg.norm(y))
    assert abs(S[i, j] - ref) < TOL, f"定義式との差 {S[i, j] - ref}"
    print(f"  [ok] cosine  diag=1 (max err {np.abs(np.diag(S) - 1).max():.1e}), range [{S.min():.3f}, {S.max():.3f}]")
    # 負値入力：[-1, 1]，対角 1
    Z = rng.standard_normal((5, 8))
    S_z = cosine_similarity(Z, Z)
    assert np.allclose(np.diag(S_z), 1.0, atol=TOL) and np.all((S_z >= -1.0) & (S_z <= 1.0))
    print(f"  [ok] cosine  負値入力 diag=1, range [{S_z.min():.3f}, {S_z.max():.3f}]")
    # ゼロベクトルとの類似度は 0（NaN にならない）
    S_0 = cosine_similarity(np.zeros((1, 50)), X)
    assert np.all(S_0 == 0.0), f"ゼロベクトルの類似度 {S_0}"
    print("  [ok] cosine  ゼロベクトルとの類似度 = 0")


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


def check_tie_handling(rng: np.random.Generator, num_classes: int = 4) -> None:
    """全要素 0 の照会行は全推論器で同点ラベル C に予測され，逆順位が 1 / C になること．"""
    C = num_classes
    per_class, dim = 6, 30
    protos = rng.random((C, dim)) ** 4
    y_train = np.repeat(np.arange(C), per_class)
    X_train = protos[y_train] + 0.01 * rng.random((y_train.size, dim))
    # 照会：各作家のプロトタイプ（正答）2 行ずつ + 全要素 0 の行（正解ラベルは作家 0〜3 を巡回）
    # shape: (2C + C, dim)
    y_test = np.concatenate([np.repeat(np.arange(C), 2), np.arange(C)])
    X_test = np.concatenate([protos[np.repeat(np.arange(C), 2)], np.zeros((C, dim))])
    zero_rows = np.arange(2 * C, 3 * C)
    classifiers = {
        "centroid": CentroidClassifier(cosine_similarity, C),
        "knn5_per_author": KNNClassifier(cosine_similarity, C, k=5),
        "knn5_global": KNNClassifier(cosine_similarity, C, k=5, mode="global"),
    }
    for name, clf in classifiers.items():
        scores = clf.fit(X_train, y_train).scores(X_test)
        y_pred = clf.predict(X_test)
        rr = reciprocal_ranks(scores, y_test)
        cm = confusion_matrix(y_test, y_pred, C)
        m = classification_metrics(cm)
        assert np.all(y_pred[zero_rows] == tie_label(C)), f"{name}: ゼロ行の予測 {y_pred[zero_rows]}"
        assert np.allclose(rr[zero_rows], 1.0 / C), f"{name}: ゼロ行の逆順位 {rr[zero_rows]}"
        assert np.array_equal(y_pred[: 2 * C], y_test[: 2 * C]) and np.allclose(rr[: 2 * C], 1.0)
        assert cm.shape == (C, C + 1) and cm[:, C].sum() == C and m["n_ties"] == C
        # 同点は誤判別（Accuracy = 2C / 3C），Recall の分母に算入（各作家 2 / 3），Precision は 1
        assert np.isclose(m["accuracy"], 2 / 3) and np.allclose(m["recall"], 2 / 3) and np.allclose(m["precision"], 1.0)
        print(
            f"  [ok] {name:<16}: ゼロ行 {zero_rows.size} 件すべて同点列（予測 {y_pred[zero_rows].tolist()}），"
            f"逆順位 {rr[zero_rows].tolist()} = 1/{C}，acc={m['accuracy']:.3f} recall={m['recall'][0]:.3f} "
            f"precision={m['precision'][0]:.3f}"
        )
    # 部分同点：最大スコアを 2 作家が共有 → 同点．正解が同点の一方なら悲観的順位 2
    s = np.array([[0.5, 0.5, 0.2, 0.1], [0.9, 0.5, 0.5, 0.1], [0.1, 0.3, 0.3, 0.3]])
    yt = np.array([1, 2, 0])
    assert predict_from_scores(s).tolist() == [C, 0, C]
    assert np.allclose(reciprocal_ranks(s, yt), [1 / 2, 1 / 3, 1 / 4])
    print(f"  [ok] 部分同点：予測 {predict_from_scores(s).tolist()}，逆順位 {reciprocal_ranks(s, yt).round(4).tolist()}")
    # evaluate_fold 経由で FoldResult に同点件数・ゼロベクトル件数が記録されること
    X_all = np.concatenate([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    texts = [f"doc{i}" for i in range(y_all.size)]
    table = dict(zip(texts, X_all, strict=True))
    author_ids = tuple(f"a{c}" for c in range(C))
    chunk_set = ChunkSet(
        chunks=tuple(Chunk(t, (t,), int(lab), author_ids[lab], "dummy", i) for i, (t, lab) in enumerate(zip(texts, y_all, strict=True))),
        author_ids=author_ids,
        target_chunk_chars=0,
    )
    split = Split("tie_check", np.arange(y_train.size), np.arange(y_train.size, y_all.size))
    specs = [ClassifierSpec("centroid"), ClassifierSpec("knn", k=5, knn_mode="per_author"), ClassifierSpec("knn", k=5, knn_mode="global")]
    results = evaluate_fold(lambda: IdentityExtractor(table), chunk_set, split, ["cosine"], specs)
    for r in results:
        assert r.n_ties == C and r.n_zero_query == C and r.confusion.shape == (C, C + 1), r
        assert np.isclose(r.mrr, (2 * C + C / C) / (3 * C)), f"{r.spec.name}: MRR {r.mrr}"
    print(f"  [ok] evaluate_fold：n_ties={[r.n_ties for r in results]}，n_zero_query={[r.n_zero_query for r in results]}，"
          f"MRR={[round(r.mrr, 4) for r in results]}")
    out = PROJECT_ROOT / "outputs" / "verify" / "confusion_tie_check.png"
    plot_confusion_matrix(results[0].confusion, list(author_ids), out, title="Tie column check")
    print(f"  描画確認: {out.relative_to(PROJECT_ROOT)}")


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
    print("[3] 同点処理（全要素 0 の照会）")
    check_tie_handling(rng)
    print("[4] 漏洩ガード")
    check_leakage_guard()

    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    assert chunk_sentences(["あい", "う", "えおか", "き"], 3) == [("あい", "う"), ("えおか",)], "チャンク分割の境界"

    print("[5] 実データのチャンク分割と分割器")
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
    print("[6] 乱数シード再現性検証")
    check_seed_reproducibility(build_chunks(corpus, 500), ratio, seed)

    print("[7] 実データ通し実行（ダミー文字 unigram，L=1000，Intra + LOWO 全 fold）")
    cs = build_chunks(corpus, 1000)
    specs = [
        ClassifierSpec("centroid"),
        ClassifierSpec("knn", k=1, knn_mode="per_author"),
        ClassifierSpec("knn", k=5, knn_mode="global"),
    ]
    t0 = time.perf_counter()
    intra_res = evaluate_fold(CharUnigramExtractor, cs, intra_work_split(cs, ratio, seed), ["cosine"], specs)
    lowo_res = [r for s in inter_work_splits(cs) for r in evaluate_fold(CharUnigramExtractor, cs, s, ["cosine"], specs)]
    for r in intra_res:
        st = r.sim_stats
        print(f"  intra/cosine  {r.spec.name:<16}: acc={r.metrics['accuracy']:.3f} F1={r.metrics['macro_f1']:.3f} "
              f"MRR={r.mrr:.3f} ΔS_macro={st.delta_s_macro:.4f} ΔS_micro={st.delta_s_micro:.4f} "
              f"ties={r.n_ties} zero={r.n_zero_query}")
    for spec in specs:
        agg = aggregate_results([r for r in lowo_res if r.spec == spec])
        assert agg.num_folds == 9 and agg.n_test == len(cs), "LOWO の pooled 統合で全チャンクが 1 回ずつ Test に入ること"
        print(f"  LOWO/cosine   {spec.name:<16}: acc={agg.metrics['accuracy']:.3f} F1={agg.metrics['macro_f1']:.3f} "
              f"MRR={agg.mrr:.3f} ΔS_macro={agg.sim_stats.delta_s_macro:.4f} ties={agg.n_ties} zero={agg.n_zero_query}")
    print(f"  所要時間 {time.perf_counter() - t0:.1f}s")
    out = PROJECT_ROOT / "outputs" / "verify" / "similarity_distribution_intra_cosine.png"
    plot_similarity_distribution(intra_res[0].sim_stats, out)
    print(f"  描画確認: {out.relative_to(PROJECT_ROOT)}")
    print("\nすべての検証に合格")


if __name__ == "__main__":
    main()
