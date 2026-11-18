# 022: 評価パイプラインの更新（特徴量に依存しない部分の 006 からの先行実装）

## Expected Deliverables & Design Policy

- 目的：006 の追加要件のうち特徴量 A・B・D に依存しない部分を，`base` と C だけで先に実装・検証し，班員が 002・003・005 の完了後にそのまま統合実験（README の Step 3）を実行できるようにする．仕様の正は `plans/research-context.md` の「データ分割」「Train チャンク数の作家間均等化」「主評価指標」．
- **Intra-work の 5 シード化**：`configs/default.yaml` の `split` に `intra_seeds: [42, 43, 44, 45, 46]` を追加し，Intra-work を 5 シードで実行．
- **Train チャンク数の間引き（Intra-work / Inter-work の両方）**：
  - `split` に `balance_train: true` と `balance_seeds: [42, 43, 44, 45, 46]` を追加．`balance_train: true` のとき，同じ run で間引きあり・なしの両方を実行し，結果表に間引きの有無の列を追加．
  - Inter-work：各 fold で `np.random.default_rng([seed, fold 番号])` により `src/dataset.py` の `balance_train_split` を適用（014 と同じ）．シード 42～46 の 5 回．
  - Intra-work：分割と同じシードで，分割後の Train を間引く．現状の `--balance-train-seeds` が Intra-work との併用を `ValueError` で止めている箇所を拡張．既存の CLI 引数は互換のため残すか，設定ファイルに一本化して README と 014 の再現手順との関係を報告．
  - 特徴抽出器の `fit` は間引き後の Train のみ．Test は間引かない．
  - fold ごとの $n_{\min}$ とそれを決めた作家を `run_meta.json` に記録（014 と同形式）．
- **集計**：間引きありの行と Intra-work の行は，シードごとの値と 5 シードの平均・標準偏差を出力（ `summary.csv` に列を追加）．
- **評価指標（ `src/evaluator.py` ）**：
  - 作家別正解率（4 作家）と作家別正解率の平均 `balanced_accuracy` を追加．既存の `accuracy` と `macro_f1` は残す．
  - 作家別 AP（4 作家），作家別 MAP の平均 `map_macro` ，全 Test チャンクの AP の平均 `map_micro` を追加．AP は Train 全件をコサイン類似度の降順に並べ，適合文書数 $R$ で割る．同順位は悲観的順位．Inter-work は全 fold の Test チャンクをまとめて集約（Pooled）．
  - 基準値 `balanced_accuracy_random` （0.25）， `macro_f1_random` （Test の作家構成比 $p_a$ から $\frac{1}{4} \sum_a \frac{2 \cdot p_a \cdot 1/4}{p_a + 1/4}$ ，Inter-work は Pooled の構成）， `mrr_random` （ $25/48$ ）， `map_random` （検索質問ごとに $\frac{1}{N} ( H_N + \frac{R - 1}{N - 1} (N - H_N) )$ を求めて `map_macro` と同じ集約，`map_micro` 用の値も同様に）を追加．
- **レポート（ `scripts/plot_results.py` ）**：
  - 表の列名は「作家別 MAP の平均」「全 Test チャンクの AP の平均」「作家別正解率の平均」「作家別 F 値の平均」「MRR」と各基準値．正解率（ `accuracy` ）は `report.md` に載せない．
  - 特徴量間の比較表は作家別 MAP の平均を主列とし， `delta_s_macro` は特徴量内（チャンク長別）の比較に用いる旨を注記．
  - 「全 9 fold」等の固定文言を実際の fold 数（ `run_meta.json` または `num_folds` ）から生成．
  - 注記：`per_author` の $k = 20$ は $n_{\min}$ が小さい fold で作家の全チャンク平均に近づくこと，作品の多様性の差（015）が間引き後も残ること，`knn_mode="global"` の MRR は悲観的順位で低く出ること（008）．
  - 各図表に使用した数式の出典（講義資料の回・頁，または独自設計）を 1 行で併記．
  - `report.best_model_metric` は `balanced_accuracy` に変更．
- **README**：「チケット 022」の表記を `plans/022-eval-pipeline-update.md` に置き換え，追加した設定キー名が「実験用語の定義」の記載（ `split.intra_seeds` 等）と一致することを確認し，異なれば README を修正．

## Target File Paths

- `scripts/run_experiments.py`
- `scripts/plot_results.py`
- `src/evaluator.py`
- `configs/default.yaml` （ `split` ， `report` セクション）
- `README.md` （022 の参照と設定キー名のみ）
- 検証用スクリプトを追加する場合は `scripts/verify_eval_metrics.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/dataset.py` の `balance_train_split` と分割器の挙動（乱数の消費順を含む）を変えない．間引きなしの結果は従来と数値一致させる．
- `src/features/` は変更しない．A・B・D の抽出器は未実装のため，検証は `base` と C で行う．
- MAP の並べ替えは行列演算でベクトル化（ `np.argsort` ／ `np.lexsort` 等）．同順位の悲観的順位も明示的に実装．
- 同点・ゼロベクトル件数の記録（008）は維持．
- `plans/` は変更しない（ `plans/wbs.md` の 022 の Status 欄を除く）．
- コミットしないこと．

## Verification Criteria

- [ ] 作家別正解率の平均の手計算一致：作家構成が偏った手作りの混同行列（例：作家 0 が Test の半数）で `balanced_accuracy` が作家別正解率の単純平均と一致し，全件を作家 0 と予測した場合に 0.25．
- [ ] 基準値の手計算一致：作家構成比 0.5, 0.2, 0.15, 0.15 で `macro_f1_random` が式と一致し，`mrr_random` が $25/48$ と一致．
- [ ] MAP の手計算一致：Train 5 件・Test 2 件程度の手作りの類似度行列（同順位を含む）で，作家別 AP，`map_macro` ，`map_micro` が手計算値と一致．
- [ ] MAP のランダム基準：$(N, R) = (100, 25)$ と $(708, 177)$ で，類似度を一様乱数（固定シード，1 万回）にしたときの AP の平均が期待値の式（約 0.282，約 0.257）と相対誤差 1% 以内で一致．
- [ ] スモーク実行：`uv run python scripts/run_experiments.py --config configs/default.yaml --features base C --chunk-chars 1000 --run-name smoke022` が 10 分以内に完走し，`summary.csv` に間引きあり・なしの両方，Intra-work と間引きありの 5 シードの平均・標準偏差，追加した全指標と基準値の列が出力される．所要時間を報告．
- [ ] 間引きの確認：間引きありの全分割（Intra-work 5 シード，Inter-work 9 fold × 5 シード）で作家別 Train チャンク数が $n_{\min}$ に等しく，Test が間引きなしと同一．
- [ ] 従来との数値一致：間引きなしの Inter-work の行が，同じ設定の従来の出力（例：`outputs/c_only/summary.csv` の $L = 1000$ の C の Inter-work の行）と既存指標で数値一致．
- [ ] 再現性：同一設定のスモーク実行を 2 回行い，`summary.csv` が完全一致．
- [ ] `report.md` に正解率が載っておらず，fold 数の文言が実際の fold 数から生成され，上記の注記が含まれることを該当行の抜粋で報告．

## Full Experiment (optional, run by the researcher)

- **Command**: `uv run python scripts/run_experiments.py --config configs/default.yaml --features base C --run-name base_c_022`
- **Expected Runtime & Resources**: スモーク実行（ $L = 1000$ ）の所要時間から $L = 100, 500$ を外挿して報告（CPU のみ）
- **Outputs**: `outputs/base_c_022/`
