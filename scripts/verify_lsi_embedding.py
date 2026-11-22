"""特徴量 D（LSI，`src/features/lsi_embedding.py`）の検証スクリプト（Ticket 005 の Verification Criteria）．

使い方：
    uv run python scripts/verify_lsi_embedding.py [--config configs/default.yaml] [--num-train 60] [--num-test 10]

検証項目（トイデータ：内容語を乱数で組み合わせた合成文書．乱数シードは config の `seed`）：
    shape        K = n_components（config の features.D.params）で transform の出力形状が (N_samples, K)
    orthonormal  V_K^T V_K = I（atol 1e-5）
    u_equals_U   projection="u" で Train 自身の変換結果が，独立に再計算した U_K と一致（atol 1e-8）
    us_equals_US projection="us" で Train 自身の変換結果が U_K Σ_K と一致（atol 1e-8）
    k_too_large  K > min(N, M) で ValueError，ランク不足（2 種類の文書の繰り返しで K = 3）で ValueError
    determinism  同一入力で 2 回 fit・transform した出力が完全一致（符号正規化の確認）
    guards       不正な引数で ValueError，Test のみに出現する語を含むテキストでも次元数が K

実コーパス（`data/processed/authors_corpus.json`）が存在する場合は追加で，L ごとの Intra-work（seed）の Train で
fit した際の N，M と，K ∈ {本採用値, ablations} の累積寄与率を表示する．
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.dataset import build_chunks, intra_work_split, load_corpus
from src.features.lsi_embedding import LSIEmbeddingExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATOL_ORTHO = 1e-5
ATOL_U = 1e-8

# 合成文書の語彙（内容語：名詞・動詞）
NOUNS = (
    "猫 犬 鳥 魚 馬 牛 花 木 山 川 海 空 雲 雨 雪 風 星 月 町 村 "
    "家 庭 窓 机 本 紙 筆 茶 酒 米 金 銀 汽車 学校 先生 生徒 手紙 写真 映画 時計"
).split()
VERBS = "見る 読む 書く 買う 売る 持つ 待つ 呼ぶ 運ぶ 探す".split()
SENTENCES_PER_DOC = 6


def synth_docs(rng: np.random.Generator, n_docs: int, nouns: tuple[str, ...] | list[str]) -> list[str]:
    """「{名詞}は{名詞}を{動詞}。」を SENTENCES_PER_DOC 文連結した合成文書．"""
    picks = rng.integers(0, [len(nouns), len(nouns), len(VERBS)], size=(n_docs, SENTENCES_PER_DOC, 3))
    return ["".join(f"{nouns[a]}は{nouns[b]}を{VERBS[v]}。" for a, b, v in doc) for doc in picks]


def independent_u(ext: LSIEmbeddingExtractor, train: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """抽出器の内部状態を使わず，TF-IDF 行列の SVD から符号正規化済みの (U_K, Σ_K) を再計算．"""
    K = ext.n_components
    U, s, Vt = np.linalg.svd(ext.tfidf_.transform(train), full_matrices=False)
    signs = np.sign(Vt[np.arange(K), np.argmax(np.abs(Vt[:K]), axis=1)])
    return U[:, :K] * signs[None, :], s[:K]


def raises_value_error(fn: Callable[[], object]) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def run_toy_checks(params: dict[str, Any], seed: int, num_train: int, num_test: int) -> dict[str, bool]:
    rng = np.random.default_rng(seed)
    train = synth_docs(rng, num_train, NOUNS)
    # Test には Train に無い語（未知語）を含める
    test = synth_docs(rng, num_test, [*NOUNS, "宇宙船", "望遠鏡"])
    K = params["n_components"]

    ext_u = LSIEmbeddingExtractor(**params | {"projection": "u"}).fit(train)
    ext_us = LSIEmbeddingExtractor(**params | {"projection": "us"}).fit(train)
    N, M = len(train), len(ext_u.tfidf_.vocab_)
    print(f"  toy: N={N}, M={M}, K={K}")
    U_k, s_k = independent_u(ext_u, train)
    V_k = ext_u.components_

    rerun = LSIEmbeddingExtractor(**params).fit(train)
    again = LSIEmbeddingExtractor(**params).fit(train)

    return {
        "shape": ext_u.transform(test).shape == (num_test, K) and ext_u.transform(train).shape == (N, K),
        "orthonormal": bool(np.allclose(V_k.T @ V_k, np.eye(K), atol=ATOL_ORTHO)),
        "u_equals_U": bool(
            np.allclose(ext_u.transform(train), U_k, atol=ATOL_U) and np.allclose(ext_u.train_u_, U_k, atol=ATOL_U)
        ),
        "us_equals_US": bool(np.allclose(ext_us.transform(train), U_k * s_k[None, :], atol=ATOL_U)),
        "k_too_large": raises_value_error(lambda: LSIEmbeddingExtractor(n_components=N + 1).fit(train))
        # 2 種類の文書の繰り返し（ランク 2）で K = 3 を指定
        and raises_value_error(lambda: LSIEmbeddingExtractor(n_components=3).fit([train[0], train[1]] * 3)),
        "determinism": bool(
            np.array_equal(rerun.transform(test), again.transform(test))
            and np.array_equal(rerun.components_, again.components_)
        ),
        "guards": raises_value_error(lambda: LSIEmbeddingExtractor(n_components=0))
        and raises_value_error(lambda: LSIEmbeddingExtractor(projection="v"))
        and ext_u.transform(["宇宙船は望遠鏡を見る。"]).shape == (1, K),
    }


def report_corpus(cfg: dict[str, Any]) -> None:
    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    if not corpus_path.exists():
        print(f"\n[corpus] {corpus_path} が無いため実コーパスの集計を省略")
        return
    spec = cfg["features"]["D"]
    k_values = sorted({spec["params"]["n_components"], *spec["ablations"].get("n_components", [])})
    corpus = load_corpus(corpus_path)
    print(f"\n[corpus] Intra-work（seed={cfg['seed']}）の Train で fit，累積寄与率 K={k_values}")
    for L in cfg["chunking"]["target_chunk_chars"]:
        chunk_set = build_chunks(corpus, L)
        split = intra_work_split(chunk_set, cfg["split"]["intra_test_ratio"], cfg["seed"])
        train = [chunk_set.texts[i] for i in split.train_idx]
        ext = LSIEmbeddingExtractor(n_components=max(k_values)).fit(train)
        ratios = ", ".join(f"K={k}: {ext.cumulative_explained_ratio_[k - 1]:.3f}" for k in k_values)
        print(f"  L={L}: N={len(train)}, M={len(ext.tfidf_.vocab_)}, {ratios}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--num-train", type=int, default=60)
    parser.add_argument("--num-test", type=int, default=10)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    print("[toy] Ticket 005 Verification Criteria")
    results = run_toy_checks(cfg["features"]["D"]["params"], cfg["seed"], args.num_train, args.num_test)
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    report_corpus(cfg)
    if not all(results.values()):
        raise SystemExit("検証失敗")


if __name__ == "__main__":
    main()
