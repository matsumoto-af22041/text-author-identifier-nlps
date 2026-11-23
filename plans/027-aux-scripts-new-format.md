# 027: 作品除外の集計と重心の割り増し測定のスクリプトを現行の出力形式に対応

## Expected Deliverables & Design Policy

- 背景：006 の中間レビュー（2026-11-23）で，次の 2 本が 022 以前の出力形式を前提にしており，現行の `scripts/run_experiments.py` の出力では実行できないと判明．
  - `scripts/summarize_exclusion.py` （013）：1 条件 1 行の `summary.csv` を前提とし，現行の `row_type` （ `seed` / `mean` ）と `balance_train` （間引きあり / なし）の行が同じ run に入るため「条件の過不足」で停止．
  - `scripts/measure_centroid_norm.py` （015）：`bal_s<seed>_<特徴量>` という run 名と旧キー形式を前提とし，その run を作る `--balance-train-seeds` は 022 で廃止済み．
- 両スクリプトを現行の出力形式に対応させる．
  - `summarize_exclusion.py` ：`row_type` と `balance_train` で行を絞って照合．主結果は間引きあり（ `row_type=mean` ，5 シード平均）とし，間引きなしを併記．集中の指標「漱石以外のチャンク中の漱石予測割合」（013）は従来どおり．特徴量は `--features` 等で A～D・`base` を指定可能に．
  - `measure_centroid_norm.py` ：間引きの再現は `src/dataset.py` の `balance_train_split` と現行の乱数（Inter-work は `np.random.default_rng([seed, fold 番号])` ，`plans/research-context.md` の「Train チャンク数の作家間均等化」）で直接行い，旧 run に依存しない．照合用に，現行の run の `run_meta.json` に記録された fold ごとの $n_{\min}$ ・作家別 Train 数と一致することを確認．出力は 015 と同じ（作家別の $\bar{\rho}_a$ ， $1 / \|\hat{c}_a\|$ ，実装の重心との予測一致率）で，A～D・`base` を指定可能に．
- 015 の数値（C， $L = 500$ ，`base` ）との連続性：現行形式でも `base` と C について 015 の主要値が再現されることを確認（Inter-work の間引きあり・なしは 015 と同じ乱数の決め方のため一致する見込み．一致しない場合は原因を報告）．

## Target File Paths

- `scripts/summarize_exclusion.py`
- `scripts/measure_centroid_norm.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `configs/` ・ `scripts/run_experiments.py` は変更しない（必要になった場合は変更せず報告）．
- `data/raw/` のコーパスを直接読まない．
- コミットしない．

## Verification Criteria

- [ ] `measure_centroid_norm.py` を `base` と C の $L = 500$ で実行し（10 分以内），015 の主要値（割り増し係数 $1 / \|\hat{c}_a\|$ ：`base` 1.08～1.11，C 3.4～4.1，間引き後の漱石 $\bar{\rho}_a$ 0.0688，予測一致率 `base` 0.98～0.99 / C 0.93～0.98）との対応を報告．
- [ ] 同スクリプトを A・B・D の $L = 500$ でも実行し，作家別の $\bar{\rho}_a$ と $1 / \|\hat{c}_a\|$ を報告（10 分を超える場合は Full Experiment へ）．
- [ ] `summarize_exclusion.py` の動作確認：`--features base --chunk-chars 1000` 程度の小さな除外 run を 3 通り（ `--exclude-works` で漱石の各作品）作り，9 作品版の run と合わせて集計が完走し，漱石以外のチャンク中の漱石予測割合が出力されること．
- [ ] 間引きの照合：`measure_centroid_norm.py` 内で再現した fold ごとの $n_{\min}$ と作家別 Train 数が，対応する run の `run_meta.json` の記録と一致．

## Full Experiment (optional, run by the researcher)

- **Command**：A～D の作品除外 run を 3 通り（Executor が正確なコマンドを用意．目安：`uv run python scripts/run_experiments.py --config configs/default.yaml --features A B C D --exclude-works <作品> --run-name excl_<略称>_ABCD` ），その後 `summarize_exclusion.py` で 9 作品版の Full Experiment の run と集計．
- **Expected Runtime & Resources**：1 回あたり約 8 分（006 の見積もり），計 25 分程度．Executor が実測から更新．
- **Outputs**：`outputs/excl_*_ABCD/` と集計結果
