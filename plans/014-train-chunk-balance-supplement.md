# 014: Train チャンク数の偏りの補助実験（作家別の間引き）

## Expected Deliverables & Design Policy

- **背景**：013 で漱石を 2 作品にしても，C の漱石への予測集中（漱石以外のチャンク中の漱石予測割合，$L = 500$ ，重心法）は 0.583 → 0.375～0.524 と残存．作品数ではなく，作家ごとの Train チャンク数（漱石の作品は長い）と疎な高次元特徴の組合せが原因という仮説を検証する．
  - 重心法：疎なベクトルは平均するチャンクが多いほど重心が密になり，どのチャンクともコサインが高く出やすい．
  - `per_author` の上位 $k$ 平均：候補が多い作家ほど，偶然似たチャンクが見つかりやすい．
- **手順の前半（013 の再実行）**：コード変更の前に，013 の 7 run（`excl_{bocchan,kokoro,sanshiro}_{base,C}` と `full_base`）を 013 に記載のコマンドで再実行し，`uv run python scripts/summarize_exclusion.py` を再実行．コミット `a53fe7b` 以降のコードでの再実行により `run_meta.json` の追跡情報を確定させる．
  - `git_state()` は未追跡ファイルも dirty と数えるため，`docs/` が未追跡の間は `dirty: true` のままになる．`dirty_files` に `src/` ， `scripts/` ， `configs/` 配下のファイルが含まれないことを確認する．
- **手順の後半（間引き実験）**：Inter-work（LOWO）の各 fold で，Train チャンクを作家ごとに無作為に間引き，全作家の Train チャンク数を「その fold で Train チャンクが最少の作家」の数 $n_{\min}$ にそろえる．
  - 間引きは作家内のチャンクから一様に非復元抽出する（作品の内訳は保存しない．作品単位の抽出は行わない）．
  - 特徴抽出器の `fit`（語彙・DF・IDF）も間引き後の Train のみで行う（「学習」の定義を変えない）．Test チャンクは間引かない．
  - 乱数シードは CLI で指定し，5 シード（42, 43, 44, 45, 46）を実行．fold ごとの乱数は `np.random.default_rng([seed, fold 番号])` のようにシードと fold から決定的に生成し，fold の実行順に依存しないこと．
  - 対象：9 作品（除外なし）で `base` と C，$L \in \{100, 500, 1000\}$ ，Inter-work のみ．run 名は `bal_s<seed>_<特徴量>` （例：`bal_s42_C` ）．
- **実装**：
  - `src/dataset.py` に，Split と作家ラベルと乱数生成器を受け取り，Train を作家別に $n_{\min}$ 個へ間引いた新しい Split を返す関数を追加（Test は不変，Train の作家集合は不変）．
  - `scripts/run_experiments.py` に `--balance-train-seeds <seed> ...` を追加．1 個のシードを受け取り（複数シードは run を分けて実行），Inter-work の各 fold に上記の間引きを適用．`--splits intra` との併用は `ValueError` ．使用シードと fold ごとの作家別 Train チャンク数（間引き前・後）を `run_meta.json` に記録．
  - オプション未指定時の挙動・出力は従来と完全に同一．
  - `scripts/summarize_balance.py` （新規）：5 シードの run と 9 作品版（C は `c_only` ， `base` は `full_base` ）を集計．
- **報告する量**（Inter-work の Pooled，各 $L$ ，重心法と `per_author` の $k = 5$ ）：
  - 漱石以外のチャンク中の漱石予測割合（主指標），Accuracy，マクロ F1，作家別の再現率．いずれも 5 シードの平均と標準偏差
  - 9 作品版（間引きなし）との差分
  - fold ごとの $n_{\min}$ と，それを決めた作家
- **解釈の前提**：間引きで漱石以外の作家の Train も減るため，全体の精度の低下は想定内．見るのは漱石への予測集中の変化．目安は，4 作家へ一様に予測した場合の 1/4 と，`base` の 9 作品版の値（013 で 0.19～0.22）．

## Target File Paths

- `src/dataset.py` （間引き関数の追加のみ）
- `scripts/run_experiments.py` （ `--balance-train-seeds` の追加のみ）
- `scripts/summarize_balance.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 013 の再実行は，コードを 1 行も変更していない状態で最初に行う．
- `configs/` と既存の関数の挙動は変更しない．
- $n_{\min}$ が `classifier.k_values` の最大値（20）を下回る fold があれば報告（ `per_author` は `k_eff = min(k, len(D_a))` のため停止はしない）．
- 数値は実行結果からの転記のみ．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] 013 の 7 run の再実行結果で，`summarize_exclusion.py` の出力が再実行前と数値一致．各 run の `run_meta.json` の `commit` と `dirty_files` を報告．
- [ ] `--balance-train-seeds` 未指定の `--features C --chunk-chars 500 --splits inter` の `summary.csv` が `outputs/c_only/summary.csv` の対応行と数値一致．
- [ ] 間引き関数の検証：全 fold・全 $L$ で，間引き後の作家別 Train チャンク数がすべて $n_{\min}$ に等しく，Test インデックスが不変で，Train が間引き前の部分集合であること．同一シードで 2 回呼んで同一の結果，異なるシードで異なる結果．
- [ ] `--splits intra` との併用で `ValueError` ．
- [ ] 10 run（5 シード × 2 特徴量）が各 10 分以内に完走し，所要時間を報告．
- [ ] `uv run python scripts/summarize_balance.py` が上記の報告量を出力．観測値そのものを報告．
