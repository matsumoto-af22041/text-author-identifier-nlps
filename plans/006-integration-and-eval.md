# 006: 全体統合実験スクリプトと可視化・レポート出力

## Expected Deliverables & Design Policy

- 担当：全員（統合はリーダー）
- 特徴量 A～D（類似度はコサインのみ），チャンク長 $L \in \{100, 500, 1000\}$ ，分割方式（Intra-work / Inter-work），推論方式（重心法 / $k$-NN 法 $k \in \{1, 3, 5, 10, 20\}$ ）の全組合せ実験と，アブレーション（A の `scaling=minmax` と `blocks` $\in$ {`bigram`, `symbol`} ，B の `token_key=lemma` と `target_pos` 拡張，D の `n_components` $\in \{8, 16, 64\}$ と `projection=us` ）を一括または条件指定（ `--features` ， `--chunk-chars` ， `--splits` ， `--ablation` ）で実行する実験スクリプト．
- 結果の表（CSV / Markdown）と図（混同行列，チャンク長別推移，同一著者 vs 異著者の類似度ヒストグラム，4×4 作家間ヒートマップ）を自動生成するレポートスクリプト．
- **Intra-work の複数シード化（追加要件）**：現状は `seed=42` の 1 回分割のみで，Intra と Inter の差がシードの偶然か区別できない． `configs/default.yaml` に `split.intra_seeds: [42, 43, 44, 45, 46]` を追加し，Intra-work は 5 シードの平均と標準偏差を報告（Inter-work は決定的分割のため 1 回）．
- 既存実装：両スクリプトは依存チケットに先行して実装済み（ベースライン `base` で動作）．本チケットでは 002～005・008 の完了後に A～D を含めた動作検証と上記追加要件の実装を行い，不具合があれば修正．

## Target File Paths

- `scripts/run_experiments.py`
- `scripts/plot_results.py`
- `configs/default.yaml` （ `features` ， `experiment` ， `report` セクション）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 結果の JSON / CSV には実行時の設定（乱数シード，チャンク長，特徴量ハイパーパラメータ），CLI 引数，git 状態，コーパスハッシュを `run_meta.json` として併記（Experiment Traceability）．
- 主結果表として，B の付属語（既定）と内容語対照（ `target_pos=[名詞, 動詞, 形容詞]` ）の Intra-work / Inter-work 比較を `report.md` の冒頭に出力（Core Message の検証）．
- 類似度ギャップの主指標はマクロ版 `delta_s_macro` （ミクロ版 `delta_s_micro` を併記）．Inter-work（LOWO 9 fold）の Accuracy・マクロ F1 は全 fold の混同行列を合算する Pooled Out-of-Fold で算出．同点件数・ゼロベクトル件数（008）を結果表に併記．
- `report.md` の各図表に，使用した数式の出典（講義資料の回・頁または `plans/research-context.md` の文献）を 1 行で併記．
- 生成物は `outputs/` （git 管理外）に出力．

## Verification Criteria

- [ ] スモーク実行： `uv run python scripts/run_experiments.py --config configs/default.yaml --features A B C D --chunk-chars 1000 --run-name smoke` が 10 分以内に完走し， `outputs/smoke/` に `run_meta.json` ，結果 CSV / JSON，図， `report.md` が生成される．
- [ ] スモーク実行の結果に以下が含まれる．
  - 各特徴量の $\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， `delta_s_macro` ， `delta_s_micro` ，Accuracy，マクロ F1 の Intra-work / Inter-work 比較表
  - 同一著者 vs 異著者の類似度分布ヒストグラムと 4×4 作家間平均類似度ヒートマップ
  - 推論方式（Centroid，$k$-NN $k \in \{1, 3, 5, 10, 20\}$ ）の比較表
  - 最良モデルの混同行列と誤判別チャンクの抜粋ログ
- [ ] 乱数シード再現性：同一設定のスモーク実行を 2 回行い，結果 CSV の数値が完全一致．
- [ ] Intra-work の結果に 5 シードの平均・標準偏差の列が出力される．
- [ ] `--ablation` 付きスモーク実行（ `--chunk-chars 500` ，特徴量は A・B・D のいずれか 1 つ）で，本採用値以外のアブレーション設定行が結果に追加される．

## Full Experiment (optional, run by the researcher)

- **Command**: `uv run python scripts/run_experiments.py --config configs/default.yaml --ablation`
- **Expected Runtime & Resources**: 未計測．Executor がスモーク実行の所要時間から外挿して報告（CPU のみ，RAM 8～16GB 想定）
- **Outputs**: `outputs/<run-name>/` （ `run_meta.json` ，結果 CSV / JSON，図， `report.md` ）
