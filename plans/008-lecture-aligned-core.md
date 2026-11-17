# 008: 共通基盤の講義資料準拠化（Jaccard / Dice の削除と同点処理）

## Expected Deliverables & Design Policy

- 担当：システム基盤（リーダー）
- 001 で実装済みの共通基盤を，`plans/research-context.md` の改訂仕様に合わせて修正する．
- **Jaccard / Dice 係数の削除**：類似度はコサイン係数（第 5 回 p.26，第 3 回 p.25）に統一．既存の重み付き Jaccard / Dice（ $\sum \min / \sum \max$ ）は講義資料に式がなく，研究の主張にも寄与しないため，関数・設定・計算量制限の仕組みごと削除．
  - `src/similarity.py` ：`jaccard_similarity` ， `dice_similarity` ， `_pairwise_min_sum` ， `NON_NEGATIVE_ONLY` ， `max_batch_bytes` 関連を削除．`SIMILARITY_FUNCTIONS` は `cosine` のみ．docstring の「講義準拠」はコサインの講義頁（第 5 回 p.26，第 3 回 p.25）の明記に置き換え．
  - `src/evaluator.py` ：`NON_NEGATIVE_ONLY` による組合せ拒否を削除．負値を含む特徴量（D）のヒストグラム範囲 $[-1, 1]$ の扱いは維持．
  - `scripts/run_experiments.py` ：`similarities_for` の制限ロジック（ `restricted_similarities` ， `restricted_chunk_chars` ）を削除．
  - `configs/default.yaml` ：`similarity.names: [cosine]` ，`similarity.max_batch_bytes` と `experiment.restricted_*` を削除．
  - `BaseFeatureExtractor.non_negative` は抽出器の出力の性質の宣言として残す（Jaccard / Dice 適用条件の記述は削除）．
- **同点処理（評価の偏りの除去）**：現状は `argmax` が同点時に最小ラベル（漱石）を返し，`reciprocal_ranks` は自分より「厳密に」高いスコア数で順位を数えるため，ゼロベクトル等で全作家が同点になるチャンクは「漱石と予測」かつ「正解作家の逆順位 1」と楽観的に評価される．
  - 予測：最大スコアを 2 作家以上が共有する場合は判別不能とし，混同行列に「同点」列を追加（形状 `(C, C + 1)` ）．Accuracy では誤判別，マクロ F1 では正解作家の FN として集計．
  - MRR：順位を「正解作家以上のスコアを持つ作家数」（悲観的順位）で数える．
  - fold ごとに同点件数とゼロベクトル（ノルム 0）の照会チャンク数を `FoldResult` に記録し，`folds.csv` / `summary.csv` に出力．

## Target File Paths

- `src/similarity.py`
- `src/evaluator.py`
- `src/features/base.py` （docstring とコメントのみ）
- `src/classifier.py` （予測の同点判定を `evaluator` と共有する場合のみ）
- `scripts/verify_core_pipeline.py`
- `scripts/run_experiments.py`
- `scripts/plot_results.py` （類似度式の比較表の削除と，混同行列の「同点」列の描画）
- `configs/default.yaml` （ `similarity` ， `experiment.restricted_*` のみ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `numpy` のみ．サンプル間の 2 重 `for` ループ禁止．
- `cosine_similarity` の数値結果は変更しない（001 の結果との連続性）．
- 001 の Verification Criteria のうち，数値的健全性（コサインのみ）・単一バッチ過学習・シード再現性が引き続き通ること．
- `grep -rn "jaccard\|dice\|Jaccard\|Dice" src scripts configs` の該当が 0 件になること（`presentation/` と `plans/` は対象外）．

## Verification Criteria

- [ ] `uv run python scripts/verify_core_pipeline.py` が完走（コサインの対角 $1.0$ ，Train = Test で Accuracy $1.0$ ，シード再現性）．
- [ ] 全要素 0 の行を含む照会で，予測が「同点」列に入り，その行の逆順位が $1 / C$ になる．
- [ ] 上記の `grep` が 0 件．
- [ ] `--features base --chunk-chars 500` のスモーク実行が完走し，結果にコサインの行のみと同点件数・ゼロベクトル件数の列が出力される．
