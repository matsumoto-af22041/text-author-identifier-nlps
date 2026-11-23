# 033: Slide 2 の表 1（4 作家・9 作品の文字数・文数・チャンク数）

## Expected Deliverables & Design Policy

- 背景：スライド作成担当から，storyline の Slide 2「表 1（9 作品の文字数・文数・チャンク数，000 実測，`verify_qa_points` の Q1）」の作品別の値が `presentation/` の資料に載っていないとの問い合わせ（2026-11-24）．`scripts/verify_qa_points.py` の Q1 は全 30 作品の文字数のみで，文数・チャンク数は出力していない．
- `scripts/verify_qa_points.py` の Q1 に，対象 9 作品の表を追加：作家，作品，文字数，文数，チャンク数（ $L = 100, 500, 1000$ ），端数として破棄された文字数．作家ごとの小計と 9 作品の合計も出す．
  - 文字数は既存の Q4 と同じ定義（作品の全文の文字数．`sum(len(s) for s in w.sentences)` ）．既存の Q1 の 30 作品の文字数と一致することをスクリプト内で検査．
  - 文数は `src/dataset.py` が読み込んだ文の数（文の定義は `authors_corpus.json` の `meta.sentence_definition` ）．
  - チャンク数は `src/dataset.py` のチャンク分割（`chunk_sentences` ）による．9 作品の合計が Q3 のチャンク数（7047 / 1679 / 857）と一致することをスクリプト内で検査．
  - 照合：`--baseline-run full_006` 等の引数で指定した run の `folds.csv` から，Inter-work の各 fold（Test が 1 作品）の Test チャンク数を読み，作品別のチャンク数と一致することを検査（不一致なら停止）．
- `presentation/presentation_storyline.md` の Slide 2 に表 1 の値を追記（出典：`scripts/verify_qa_points.py` の Q1，照合した run 名）．チャンク数は 3 つの $L$ をすべて載せ，代表条件の $L = 500$ を主とする旨を書く．
- `presentation/qa_defense_matrix.md` の Q1 に同じ表への参照を追記．

## Target File Paths

- `scripts/verify_qa_points.py`
- `presentation/presentation_storyline.md` （Slide 2 のみ）
- `presentation/qa_defense_matrix.md` （Q1 のみ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `configs/` ・他の `scripts/` を変更しない．既存の Q1～Q10 の出力値を変えない．
- `data/raw/` のコーパスを直接読まない．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/verify_qa_points.py --baseline-run full_006` （引数名は実装に合わせる）が完走し，9 作品の表を出力（値を報告）．文字数の合計 886709，チャンク数の合計 7047 / 1679 / 857，`full_006` の Test チャンク数との一致の検査がすべて通る．
- [ ] 追加前の出力（Q1 の 30 作品の行～Q10，Slide 1a，字種構成比）が追加後も同一（diff で確認）．
- [ ] storyline の表 1 と qa の Q1 の値がスクリプトの出力と一致．
