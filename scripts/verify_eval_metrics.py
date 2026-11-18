"""022 の評価指標（作家別正解率の平均・AP・作家別 MAP の平均・基準値）の手計算一致を検証．

使い方：
    uv run python scripts/verify_eval_metrics.py [--seed 0] [--num-trials 10000]

検証項目：
    作家構成が偏った混同行列での balanced_accuracy と，全件を作家 0 と予測した場合の 0.25
    作家構成比 0.5, 0.2, 0.15, 0.15 での macro_f1_random と mrr_random = 25 / 48
    同順位を含む手作りの類似度行列での作家別 AP，map_macro，map_micro（悲観的順位）
    一様乱数の類似度での AP の平均と期待値の式の相対誤差（(N, R) = (100, 25), (708, 177)）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluator import (  # noqa: E402
    average_precision,
    classification_metrics,
    confusion_matrix,
    expected_random_ap,
    harmonic_number,
    macro_f1_random,
    mrr_random,
    retrieval_metrics,
)

TOL = 1e-12


def check(name: str, observed: float, expected: float, tol: float = TOL) -> None:
    ok = abs(observed - expected) <= tol
    print(f"  [{'ok' if ok else 'NG'}] {name}: observed={observed:.12f} expected={expected:.12f}")
    if not ok:
        raise AssertionError(f"{name}: {observed} != {expected}")


def verify_balanced_accuracy() -> None:
    print("作家別正解率の平均（Test 100 件，作家 0 が半数）")
    C = 4
    # 正解ラベル：作家 0 が 50 件，作家 1 が 20 件，作家 2・3 が 15 件ずつ
    y_true = np.repeat(np.arange(C), [50, 20, 15, 15])
    # 予測：作家 0 は 40 正解・5 件を作家 1・5 件を同点，作家 1 は 10 正解・10 件を作家 0，
    #       作家 2 は 15 件すべて正解，作家 3 は 3 正解・12 件を作家 0
    y_pred = np.concatenate([
        np.repeat([0, 1, C], [40, 5, 5]),
        np.repeat([1, 0], [10, 10]),
        np.repeat([2], [15]),
        np.repeat([3, 0], [3, 12]),
    ])
    m = classification_metrics(confusion_matrix(y_true, y_pred, C))
    by_hand = (40 / 50 + 10 / 20 + 15 / 15 + 3 / 15) / 4
    check("balanced_accuracy（作家別正解率の単純平均）", m["balanced_accuracy"], by_hand)
    print(f"       作家別正解率 = {m['recall']}，accuracy = {m['accuracy']:.4f}（全件集計，作家 0 の正誤に支配）")
    m0 = classification_metrics(confusion_matrix(y_true, np.zeros_like(y_true), C))
    check("全件を作家 0 と予測した balanced_accuracy", m0["balanced_accuracy"], 0.25)
    print(f"       同条件の accuracy = {m0['accuracy']:.4f}（作家 0 の構成比 0.5 と一致）")


def verify_baselines() -> None:
    print("基準値（作家構成比 0.5, 0.2, 0.15, 0.15）")
    p = np.array([0.5, 0.2, 0.15, 0.15])
    y_true = np.repeat(np.arange(4), (p * 100).astype(int))
    by_hand = float(np.mean([2 * pa * 0.25 / (pa + 0.25) for pa in p]))
    check("macro_f1_random", macro_f1_random(y_true, 4), by_hand)
    check("mrr_random", mrr_random(4), 25 / 48)
    check("H_4", harmonic_number(4), 25 / 12)


def verify_map_by_hand() -> None:
    print("AP・作家別 MAP の平均・全 Test チャンクの AP の平均（Train 5 件・Test 3 件，同順位あり）")
    y_train = np.array([0, 1, 0, 1, 0])
    y_test = np.array([0, 1, 0])
    # shape: (n_test, n_train) = (3, 5)
    S = np.array([
        # 0.9 の同順位（適合 #0 と非適合 #1）→ 非適合が先：rel 列 [0, 1, 0, 1, 1]
        [0.9, 0.9, 0.5, 0.7, 0.1],
        # 0.8 の同順位（適合 #1, #3 と非適合 #2）→ 非適合が先：rel 列 [0, 1, 1, 0, 0]
        [0.2, 0.8, 0.8, 0.8, 0.3],
        # 0.0 の同順位（適合 #2, #4 と非適合 #1, #3）→ 非適合が先：rel 列 [1, 0, 0, 1, 1]
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    ap_hand = np.array([
        (1 / 2 + 2 / 4 + 3 / 5) / 3,
        (1 / 2 + 2 / 3) / 2,
        (1 / 1 + 2 / 4 + 3 / 5) / 3,
    ])
    ap = average_precision(S, y_test, y_train)
    [check(f"AP[{i}]", float(a), float(h)) for i, (a, h) in enumerate(zip(ap, ap_hand, strict=True))]
    ap_random = expected_random_ap(5, np.array([3, 2, 3]))
    ret = retrieval_metrics(ap, ap_random, y_test, 2)
    check("作家 0 の AP（照会 #0, #2 の平均）", ret["ap_by_author"][0], (ap_hand[0] + ap_hand[2]) / 2)
    check("作家 1 の AP（照会 #1）", ret["ap_by_author"][1], ap_hand[1])
    check("map_macro", ret["map_macro"], ((ap_hand[0] + ap_hand[2]) / 2 + ap_hand[1]) / 2)
    check("map_micro", ret["map_micro"], ap_hand.mean())
    # 楽観的順位（適合を先）なら照会 #0 の AP は (1 + 2/4 + 3/5) / 3 となり，悲観的順位と異なること
    optimistic = (1 / 1 + 2 / 4 + 3 / 5) / 3
    print(f"       参考：照会 #0 の楽観的順位での AP = {optimistic:.6f}（悲観的順位 {ap_hand[0]:.6f}）")
    # 基準値の集約：作家 0 は R = 3，作家 1 は R = 2
    e3, e2 = expected_random_ap(5, np.array([3, 2]))
    check("map_macro_random", ret["map_macro_random"], (e3 + e2) / 2)
    check("map_micro_random", ret["map_micro_random"], (2 * e3 + e2) / 3)


def verify_random_ap(seed: int, num_trials: int) -> None:
    print(f"AP のランダム基準（一様乱数の類似度，{num_trials} 回，seed={seed}）")
    rng = np.random.default_rng(seed)
    for N, R in ((100, 25), (708, 177)):
        y_train = np.r_[np.zeros(R, dtype=np.int64), np.ones(N - R, dtype=np.int64)]
        y_test = np.zeros(num_trials, dtype=np.int64)
        # shape: (num_trials, N)
        S = rng.random((num_trials, N))
        observed = float(average_precision(S, y_test, y_train).mean())
        expected = float(expected_random_ap(N, np.array([R]))[0])
        rel_err = abs(observed - expected) / expected
        ok = rel_err <= 0.01
        print(f"  [{'ok' if ok else 'NG'}] (N, R) = ({N}, {R}): 平均 AP={observed:.6f}，期待値={expected:.6f}，相対誤差={rel_err:.4%}")
        if not ok:
            raise AssertionError(f"(N, R) = ({N}, {R}) の相対誤差 {rel_err:.4%} が 1% 超")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="ランダム基準の検証に使う乱数シード")
    parser.add_argument("--num-trials", type=int, default=10000, help="ランダム基準の検証の試行回数")
    args = parser.parse_args()
    verify_balanced_accuracy()
    verify_baselines()
    verify_map_by_hand()
    verify_random_ap(args.seed, args.num_trials)
    print("\nすべての検証に合格")


if __name__ == "__main__":
    main()
