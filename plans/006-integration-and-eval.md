# 006: 全体統合実験スクリプトと可視化・レポート出力

## Expected Deliverables & Design Policy

- 担当：全員（統合はリーダー）
- 特徴量 A～D（類似度はコサインのみ），チャンク長 $L \in \{100, 500, 1000\}$ ，分割方式（Intra-work / Inter-work），推論方式（重心法 / $k$-NN 法 $k \in \{1, 3, 5, 10, 20\}$ ）の全組合せ実験と，アブレーション（A の `scaling=minmax` と `blocks` $\in$ {`bigram`, `symbol`} ，B の `token_key=lemma` と `target_pos` 拡張，D の `n_components` $\in \{8, 16, 64\}$ と `projection=us` ）を一括または条件指定（ `--features` ， `--chunk-chars` ， `--splits` ， `--ablation` ）で実行する実験スクリプト．
- 結果の表（CSV / Markdown）と図（混同行列，チャンク長別推移，同一著者 vs 異著者の類似度ヒストグラム，4×4 作家間ヒートマップ）を自動生成するレポートスクリプト．
- **Intra-work の複数シード化（追加要件）**：現状は `seed=42` の 1 回分割のみで，Intra と Inter の差がシードの偶然か区別できない． `configs/default.yaml` に `split.intra_seeds: [42, 43, 44, 45, 46]` を追加し，Intra-work は 5 シードの平均と標準偏差を報告（Inter-work は決定的分割のため 1 回）．
- **Train チャンク数の作家間均等化（追加要件，2026-11-18 決定）**：主結果は 014 の間引き（ `src/dataset.py` の `balance_train_split` ）を適用した条件．仕様は `plans/research-context.md` の「Train チャンク数の作家間均等化」．
  - Inter-work：各 fold で `default_rng([seed, fold 番号])` により間引き，シード 42～46 の 5 回．
  - Intra-work：分割と同じシード（42～46）で分割後の Train を間引く．現状の `--balance-train-seeds` は Intra-work との併用を `ValueError` で止めているため拡張する．
  - 間引きの有無とシードは `configs/default.yaml` の `split` セクション（例：`balance_train: true` ， `balance_seeds: [42, 43, 44, 45, 46]` ）で指定し，間引きなし（ `balance_train: false` 相当）の結果も同じ run の結果表に併記（`summary.csv` に間引きの有無の列を追加）．
  - 間引きありの行は 5 シードの平均と標準偏差を出力．fold ごとの $n_{\min}$ とそれを決めた作家を `run_meta.json` に記録（014 と同形式）．
  - 追加した設定キー名が README の「実験用語の定義」（ `split.intra_seeds` 等）と一致することを確認し，異なれば README を修正．
  - `report.md` に，`per_author` の $k = 20$ は $n_{\min}$ が小さい fold で作家の全チャンク平均に近づく旨と，作品の多様性の差（015）が残る旨を注記．
- **判別精度の主指標の変更（追加要件，2026-11-18 決定）**：定義は `plans/research-context.md` の「主評価指標」に従う．
  - `summary.csv` に作家別正解率の平均 `balanced_accuracy` と作家別正解率（4 列）を追加．既存の `accuracy` 列は残す．
  - `report.md` の表と図では，正解率（ `accuracy` ）を載せず，作家別正解率の平均と作家別 F 値の平均を主列とする．表記は「作家別正解率の平均」「作家別 F 値の平均」「作家別 MAP の平均」（Balanced Accuracy，マクロ F1，MAP の名称は使わない）．
  - README 中の正解率の数値（C の Intra 0.955 / Inter 0.351，全予測漱石 0.477 等）を 006 の結果の作家別正解率の平均に置き換え（出典の run 名を併記）．指標の名称と定義の修正は 020 で実施済みの前提．
  - 基準値（`plans/research-context.md` の「基準値」）を `summary.csv` の列（ `balanced_accuracy_random` ， `macro_f1_random` ， `mrr_random` ， `map_random` ）として出力し，`report.md` の各表に併記．`macro_f1_random` は Test の作家構成から算出（Inter-work は Pooled の構成）．
- **作家別 MAP の平均の追加（追加要件，2026-11-18 決定）**：特徴量間比較の主指標．定義（AP の分母，悲観的順位，作家別平均→マクロ，Inter-work の Pooled，ランダム基準の式）は `plans/research-context.md` の「主評価指標」に従う．
  - `src/evaluator.py` に実装し，`summary.csv` に `map_macro` ，全 Test チャンクの AP の単純平均 `map_micro` ，作家別 AP，ランダム基準 `map_random` の列を追加．`report.md` では `map_micro` を「全検索質問の AP の平均（講義の MAP と同じ集約）」として併記．間引きありの行は 5 シードの平均と標準偏差．
  - `report.md` の主結果表（B の付属語 vs 内容語対照）と特徴量 A～D の比較表は `map_macro` を主列とし， `delta_s_macro` は特徴量内の比較（チャンク長別）に用いる旨を注記．
- **022 との分担（2026-11-18）**：上記の追加要件のうち特徴量に依存しない部分（Intra-work の 5 シード化，間引き，評価指標・基準値の追加，`report.md` の列名・fold 数の文言・注記，README の設定キー名の確認）は 022 で `base` と C により先行実装・検証する．本チケットでは 022 の完了を前提に，A～D を含めた動作検証，作品除外の補助実験，重心の割り増しの実測，README の数値の置き換えを行う．Verification Criteria のうち 022 で確認済みの項目は，A～D を含めたスモーク実行で再確認すれば足りる．
- 既存実装：両スクリプトは依存チケットに先行して実装済み（ベースライン `base` で動作）．本チケットでは 002～005・008 の完了後に A～D を含めた動作検証と上記追加要件の実装を行い，不具合があれば修正．

## Target File Paths

- `scripts/run_experiments.py`
- `scripts/plot_results.py`
- `src/evaluator.py` （MAP の追加）
- `README.md` （「実験用語の定義」の MAP と設定キー名）
- `configs/default.yaml` （ `features` ， `experiment` ， `report` セクション）
- `scripts/verify_stylometry_char.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 結果の JSON / CSV には実行時の設定（乱数シード，チャンク長，特徴量ハイパーパラメータ），CLI 引数，git 状態，コーパスハッシュを `run_meta.json` として併記（Experiment Traceability）．
- 主結果表として，B の付属語（既定）と内容語対照（ `target_pos=[名詞, 動詞, 形容詞]` ）の Intra-work / Inter-work 比較を `report.md` の冒頭に出力（Core Message の検証）．比較は `blocks=words` （語ブロック単独）を主とし，`blocks=all` （結合版）と `blocks=pos_bigram` を併記（025，2026-11-23 決定）．README のアブレーションの説明に `combinations` （複数引数の同時差し替え）を追記．B の形態素解析キャッシュ（025）による全 $L$ 実行時の最大 RSS をスモーク実行で測り報告．
- 類似度ギャップの主指標はマクロ版 `delta_s_macro` （ミクロ版 `delta_s_micro` を併記）．Inter-work（LOWO 9 fold）の作家別正解率の平均・作家別 F 値の平均（および参考の `accuracy` ）は全 fold の混同行列を合算する Pooled Out-of-Fold で算出．同点件数・ゼロベクトル件数（008）を結果表に併記．
- `scripts/plot_results.py` の report.md にある「全 9 fold」の固定文言を，実際の fold 数（`run_meta.json` または結果の `num_folds` ）から生成するよう修正（013 の 8 作品版で不正確になるため）．
- **主結果の対照条件を全チャンク長で実行（2026-11-23 決定）**：`report.core_comparison` の B の 3 条件（付属語 × `words` ，内容語 × `words` ，内容語 × `all` ．付属語 × `all` は本採用， `pos_bigram` は併記用）を，`--ablation` なしの通常実行でも $L \in \{100, 500, 1000\}$ のすべてで生成する．他のアブレーションは従来どおり $L = 500$ のみ．実装方法（例：`features.B` に全 $L$ で常時実行する条件の組を指定するキー）は `scripts/run_experiments.py` を確認して決め，README と config のコメントに記載．`report.md` の主結果表とチャンク長別推移に全 $L$ の付属語 vs 内容語を出す．
- **作品ごとの一貫性（2026-11-23 決定）**：主結果の Inter-work（間引きあり）について，Test 作品（9 fold）ごとに付属語 × `words` と内容語 × `words` の作家別 MAP の平均に相当する値（その fold の Test チャンクの AP の平均）と作家別正解率を求め，付属語が上回る作品数を `report.md` に表で出す（全 $L$ ）．5 シードの ± は間引きの乱数によるばらつきのみで作品の違いによるばらつきを含まない旨を注記．
- 作品数の偏りの補助実験：013 で追加した `--exclude-works` により，A～D についても漱石から 1 作品ずつ除いた 3 通りを実行し，`scripts/summarize_exclusion.py` で 9 作品版との差分を報告（スクリプトの現行形式への対応は 027．除外 run は 027 の Full Experiment として研究者が実行）．
- 重心の割り増しの実測（027 で実施）：A～D についても `scripts/measure_centroid_norm.py` （015）を実行し，作家別の $\bar{\rho}_a$ と $1 / \|\hat{c}_a\|$ を報告．
- MRR は悲観的順位（008）のため，`knn_mode="global"` では上位 $k$ 外の作家の同点で値が低く出る． `report.md` の MRR の考察は重心法と `per_author` を主とし，global の MRR にはその旨を注記．
- 008 以前の出力（`outputs/smoke_base/` 等，混同行列が (C, C) の旧形式）は新しい `plot_results.py` で読めないため，再実行で作り直すこと．
- `report.md` の各図表に，使用した数式の出典（講義資料の回・頁，または `plans/research-context.md` の「講義資料で補えない箇所（独自設計）」）を 1 行で併記．
- 002 の検証スクリプト（scratchpad にあり未コミット）に相当するものを `scripts/verify_stylometry_char.py` としてリポジトリに置き，002 の Verification Criteria を再実行できるようにする．
- 生成物は `outputs/` （git 管理外）に出力．

## Verification Criteria

- [ ] スモーク実行： `uv run python scripts/run_experiments.py --config configs/default.yaml --features A B C D --chunk-chars 1000 --run-name smoke` が 10 分以内に完走し， `outputs/smoke/` に `run_meta.json` ，結果 CSV / JSON，図， `report.md` が生成される．
- [ ] スモーク実行の結果に以下が含まれる．
  - 各特徴量の作家別 MAP の平均，作家別正解率の平均，作家別 F 値の平均，$\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， `delta_s_macro` ， `delta_s_micro` の Intra-work / Inter-work 比較表
  - 同一著者 vs 異著者の類似度分布ヒストグラムと 4×4 作家間平均類似度ヒートマップ
  - 推論方式（Centroid，$k$-NN $k \in \{1, 3, 5, 10, 20\}$ ）の比較表
  - 最良モデルの混同行列と誤判別チャンクの抜粋ログ
- [ ] 乱数シード再現性：同一設定のスモーク実行を 2 回行い，結果 CSV の数値が完全一致．
- [ ] Intra-work の結果に 5 シードの平均・標準偏差の列が出力される．
- [ ] 間引きあり・なしの両方の行が結果に出力され，間引きありの全分割（Intra-work 5 シード，Inter-work 9 fold × 5 シード）で作家別 Train チャンク数が $n_{\min}$ に等しく，Test が間引きなしと同一であることを確認．
- [ ] 間引きなしの行が，同じ設定の従来の出力（例：`outputs/c_only/summary.csv` の Inter-work の行）と数値一致．
- [ ] 作家別正解率の平均の手計算一致：作家構成が偏った手作りの混同行列（例：作家 0 が Test の半数）で `balanced_accuracy` が作家別正解率の単純平均と一致し，全件を作家 0 と予測した場合に 0.25 となる．
- [ ] 基準値の手計算一致：手作りの Test 構成（例：作家構成比 0.5, 0.2, 0.15, 0.15）で `macro_f1_random` が $\frac{1}{4} \sum_a \frac{2 \cdot p_a \cdot 1/4}{p_a + 1/4}$ と一致し，`mrr_random` が $25/48$ と一致．
- [ ] MAP の手計算一致：Train 5 件・Test 2 件程度の手作りの類似度行列（同順位を含む）で，`map_macro` と作家別 AP が手計算値と一致．
- [ ] MAP のランダム基準：$(N, R) = (100, 25)$ と $(708, 177)$ で，類似度を一様乱数（固定シード，1 万回）にしたときの AP の平均が，`plans/research-context.md` の期待値の式（約 0.282，約 0.257）と相対誤差 1% 以内で一致．
- [ ] `--ablation` 付きスモーク実行（ `--chunk-chars 500` ，特徴量は A・B・D のいずれか 1 つ）で，本採用値以外のアブレーション設定行が結果に追加される．

## Full Experiment (optional, run by the researcher)

- **Command**: `uv run python scripts/run_experiments.py --config configs/default.yaml --ablation`
- **Expected Runtime & Resources**: 未計測．Executor がスモーク実行の所要時間から外挿して報告（CPU のみ，RAM 8～16GB 想定）
- **Outputs**: `outputs/<run-name>/` （ `run_meta.json` ，結果 CSV / JSON，図， `report.md` ）
- **作品除外の補助実験（上記の後，研究者が実行，027 で準備済み）**：
  - `uv run python scripts/run_experiments.py --config configs/default.yaml --features A B C D --splits inter --exclude-works <作品> --run-name excl_<略称>_ABCD` を 坊っちゃん / こころ / 三四郎 の 3 通り（ `--splits inter` で短縮，B の対照 3 条件を含むため 1 回 11～12 分の見込み）．
  - `uv run python scripts/summarize_exclusion.py --features A B C D --baseline <上記 Full Experiment の run 名> --excluded excl_bocchan_ABCD excl_kokoro_ABCD excl_sanshiro_ABCD --out-dir outputs/exclusion_summary_ABCD`
  - 重心の割り増し：`uv run python scripts/measure_centroid_norm.py --features A B C D --chunk-chars 500 --runs A=<run> B=<run> C=<run> D=<run> --identity-tol 1e-9 --out-dir outputs/centroid_norm_ABCD` （Full Experiment の run で再実行）．
  - Executor は本チケットの残り作業の完了時に，上記コマンドの引数が実装と一致することを確認して報告．
