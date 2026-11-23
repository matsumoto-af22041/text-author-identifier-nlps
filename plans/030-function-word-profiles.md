# 030: 作家ごとに偏りのある付属語の具体例（記述統計）

## Expected Deliverables & Design Policy

- 目的：発表の Slide 7-8（B，Core Message）と質疑「付属語の何が作家ごとに違うのか」に答える具体例を，恣意的な選択を避けた機械的な規則で抽出する．講義第 5 回 p.14 の『坊っちゃん』の頻度表（捨てる語）との対比に使う．評価には使わない記述統計．
- 新規スクリプト `scripts/function_word_profiles.py` ．
  - 対象：4 作家・9 作品の全文（`data/processed/authors_corpus.json` または `src/dataset.py` 経由）．形態素解析は `src/morphology.py` ．付属語の定義は B の既定と同じ（`pos1` が助詞・助動詞，表層形をキー，総形態素数は補助記号・空白を除く．`src/features/pos_function.py` の定義を再利用）．
  - 作品ごと・作家ごとに，各付属語の相対頻度（出現回数 ÷ 総形態素数）を算出．作家の値はその作家の全作品を合算した値．
  - 偏りの指標：作家 $a$ の語 $w$ について $r_a(w) = \frac{f_a(w)}{\bar{f}_{\neg a}(w)}$ （ $f_a$ は作家 $a$ の相対頻度， $\bar{f}_{\neg a}$ は他の 3 作家の相対頻度の単純平均）．
  - 頻度の下限：4 作家合計の出現回数が `--min-count` （既定 100）以上の語のみ．分母 $\bar{f}_{\neg a}(w) = 0$ の語は比を計算せず，「他作家に出現なし」として別表に件数と語を出す（0 で置き換えない）．
  - 作品をまたいだ一貫性：作家 $a$ のすべての作品で，作品単位の比（作品の相対頻度 ÷ $\bar{f}_{\neg a}$ ）が同じ側（すべて > 1 またはすべて < 1）の語だけを「一貫」とする．
  - 出力：作家ごとに，一貫する語のうち $r_a$ の大きい上位 `--top-n` （既定 8）語と小さい上位 8 語．各語について作家の相対頻度，他 3 作家の平均，$r_a$ ，作品ごとの相対頻度と比，出現回数．一貫しない語を除いた件数も出力．`outputs/<--out-dir>/` に CSV と `report.md` （既存の出力先は上書きしない）．
  - 表層形をキーにしているため，同じ表層形で品詞が異なる語（例：「で」の助詞と助動詞）が 1 語にまとまる旨を `report.md` に注記．各語の pos1 の内訳を列に出す．
  - 評価に使わない記述統計であること（全チャンク・全作品を使うため Train / Test の区別がない）をスクリプトの docstring と `report.md` に明記．
- `presentation/presentation_storyline.md` の Slide 7-8 に，各作家の偏りの大きい付属語（上位 3～5 語程度）の表を追加し，出典の出力ディレクトリとスクリプト名，抽出規則（比の定義・頻度の下限・作品をまたいだ一貫性）を併記．「作家間で差が出た付属語・品詞遷移の個別の例は集計していない」の記述を更新（品詞遷移は未集計のまま）．
- `presentation/qa_defense_matrix.md` に追加質問「付属語の何が作家ごとに違うのか」（上記の表と抽出規則，記述統計であり評価に使っていない旨）を追加．

## Target File Paths

- `scripts/function_word_profiles.py` （新規）
- `presentation/presentation_storyline.md`
- `presentation/qa_defense_matrix.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `configs/` ・既存の `scripts/` を変更しない．
- `data/raw/` のコーパスを直接読まない．
- 例の選択に手作業の取捨を入れない（規則のみで決まる）．閾値は CLI 引数で変更可能にし，使った値を `report.md` と `run_meta.json` に記録（git 状態・コーパスハッシュも）．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．仮説と事実を区別する．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/function_word_profiles.py --out-dir outputs/function_word_profiles` が 10 分以内に完走し，4 作家それぞれの上位・下位 8 語の表を出力（値を報告）．
- [ ] 検算：任意の 1 作家・1 語について，相対頻度と $r_a$ を別経路（作品ごとの出現回数と総形態素数からの手計算）で再計算し一致．作品ごとの総形態素数の合計が作家の総形態素数と一致．
- [ ] 同じコマンドを 2 回実行して CSV がバイト一致．
- [ ] 一貫する語の件数と除外された語の件数を作家ごとに報告．
- [ ] storyline と qa の追記内容が出力 CSV の値と一致．
