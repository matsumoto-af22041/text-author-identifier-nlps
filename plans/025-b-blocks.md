# 025: 特徴量 B のブロック選択（ `blocks` ）の追加

## Expected Deliverables & Design Policy

- 背景：`src/features/pos_function.py` （003）は品詞 2-gram ブロックと語ブロックを結合して出力する．付属語（既定）と内容語対照（ `target_pos=[名詞, 動詞, 形容詞]` ）で品詞 2-gram ブロックは完全に共通で，チャンクごとの値の合計は品詞 2-gram 1.0 に対し語ブロック 0.46（付属語）/ 0.30（内容語）．共通ブロックがコサインの大半を占め，語の種類の差が薄まる可能性がある．Core Message の検証（`plans/research-context.md` の「付属語 vs 内容語の統制比較」）のため，語ブロック単独で比較できるようにする．
- `POSFunctionExtractor` に引数 `blocks: Literal["all", "pos_bigram", "words"] = "all"` を追加．
  - `all` ：現行と同じ結合ベクトル（数値は現行とビット一致させる）．
  - `pos_bigram` ：品詞 2-gram ブロックのみ．
  - `words` ：語ブロックのみ．
  - 語彙の選択・分母の定義は現行のまま変えない．`pos_bigram` では `target_pos` ・ `token_key` は出力に影響しない．
  - 不正な値は `ValueError` ．
- `configs/default.yaml` の `features.B` ：`params` に `blocks: all` を追加，`ablations` に `blocks: [pos_bigram, words]` を追加．
  - Core Message の主結果表には「`target_pos` が付属語 / 内容語 × `blocks=words` 」の 2 条件が必要．既存の `build_variants` （ `scripts/run_experiments.py` ）がアブレーションを 1 キーずつ既定値から変える方式なら，`target_pos=[名詞, 動詞, 形容詞]` かつ `blocks=words` の組合せは生成されない．その場合，組合せを指定できる最小限の拡張（例：`ablations` とは別のキーで条件の組を列挙）を `scripts/run_experiments.py` に加える．方式はコードを確認して決め，報告すること．
- ブロックごとの二乗ノルム割合の測定スクリプト `scripts/measure_b_block_norms.py` を追加．
  - 010（`scripts/investigate_a_scale.py` ）と同じ指標：チャンクごとに $\|x_{\text{block}}\|^2 / \|x\|^2$ を求め，平均を報告．
  - 条件：$L \in \{100, 500, 1000\}$ × `target_pos` 3 条件（付属語，拡張，内容語対照），`token_key=surface` ．fit は全チャンクで行う（記述統計のため．評価には使わない旨をスクリプト冒頭に明記）．
- 003 の検証スクリプト（scratchpad にあり未コミット）は `scripts/verify_pos_function.py` としてリポジトリに置き，`blocks` の検査を加える．

## Target File Paths

- `src/features/pos_function.py`
- `configs/default.yaml` （ `features.B` セクション）
- `scripts/run_experiments.py` （組合せ条件が必要な場合のみ）
- `scripts/measure_b_block_norms.py` （新規）
- `scripts/verify_pos_function.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `blocks=all` の出力は現行の実装と数値が完全一致すること．
- 外部機械学習ライブラリは禁止．`numpy` と `fugashi` のみ．
- `data/raw/` のコーパスを直接読まない．`data/processed/authors_corpus.json` または `src/dataset.py` 経由．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/verify_pos_function.py` が完走し，003 の検証 6 項目に加えて次を確認：
  - `blocks=all` の出力が `blocks` 追加前の実装（ `git show HEAD:src/features/pos_function.py` 相当）と `np.array_equal` で一致．
  - `blocks=pos_bigram` / `words` の出力が `split_blocks` で分けた各ブロックと一致し，列数が各ブロックの次元数．
  - `blocks="foo"` で `ValueError` ．
- [ ] `uv run python scripts/measure_b_block_norms.py` が完走し，9 条件の品詞 2-gram / 語ブロックの二乗ノルム割合（平均）を出力（値を報告）．
- [ ] スモーク実行：`uv run python scripts/run_experiments.py --config configs/default.yaml --features B --chunk-chars 500 --splits inter --ablation --no-report --run-name smoke_025` が 10 分以内に完走し，`summary.csv` に「付属語 × `words` 」と「内容語対照 × `words` 」の両方の行があること．両条件と `blocks=all` の 2 条件の作家別 MAP の平均（間引きあり）を報告．10 分を超える見込みなら $L = 1000$ に変えて実行し，その旨を報告．
- [ ] 間引きなし・`blocks=all` の既定条件の行が `outputs/smoke_003_b/summary.csv` の対応する行と数値一致．
