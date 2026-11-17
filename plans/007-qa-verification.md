# 007: 発表・質疑応答対策ドキュメントと補助検証スクリプト

## Expected Deliverables & Design Policy

- 担当：全員（統合はリーダー）
- 想定質問 Q1～Q12 と追加質問（ `presentation/qa_verification_instructions.md` ，009 で改訂済み）について，本採用仕様・アブレーション設定・回答ロジック・数式とその出典（講義資料の回・頁または `plans/research-context.md` の文献）・検証結果の数値を統合した発表防衛用マスタードキュメント．
- 「要検証」項目の数値を再生成する補助検証スクリプト（Q1 のコーパス内作品分布，Q2 の `unidic-lite` による解析・表記揺れの確認，Q3 のチャンク長実測分布，Q4 の端数チャンク破棄率，Q10 の B の特徴のチャンク出現率）．
- 001 の検証で判明した以下の記述修正を反映．
  - Q3：「相対頻度化と L2 正規化で完全にキャンセル」は過大．スケールの差は正規化で除去されるが，チャンク長に応じた頻度推定の分散の差は残存（ $L = 100$ で変動係数約 26%， $2L$ 以上のチャンクが存在）．
  - Q4：全体の破棄率に加え，作品単位の最大値（約 1.66%）を併記．
  - Q11：per_author で $k \to |D_a|$ の極限は「作家 $a$ の全チャンクとの平均類似度」であり，重心との類似度 $\text{sim}(\vec{q}, \vec{\mu}_a)$ とは一致しない．「全チャンク平均類似度への漸近（重心法に近い大域的評価）」に修正．
  - 新規の想定質問：「表記（漢字 / かな）の差は作家の文体か底本の方針か」．
- 想定質問ごとに，本実験の既定仕様と，対照実験に用いた切り替えフラグ（ `scaling` ， `token_key` ， `knn_mode` ， $k$ ， `n_components` ）および 006 の結果の数値を対応付け．

## Target File Paths

- `presentation/qa_defense_matrix.md`
- `scripts/verify_qa_points.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/qa_verification_instructions.md` は入力資料として読むのみで変更しないこと．
- `data/raw/corpus20150426_with_titles.txt` を Read ツールや全文出力コマンドで直接読み込まないこと．コーパス情報は `data/processed/authors_corpus.json` または `src/dataset.py` 経由で取得．
- `scikit-learn` 等は禁止． `numpy` と `fugashi` のみを使用．
- 実験結果の数値は 006 の Full Experiment の出力（ `outputs/<run-name>/` ）から転記し，出典の run 名を併記．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従うこと．

## Verification Criteria

- [ ] `uv run python scripts/verify_qa_points.py` が完走し，4 作家・9 作品の文字数・文数，各 $L$ のチャンク実測文字数（平均・標準偏差・最大），端数破棄率（全体・作品単位最大），表記揺れ語の漢字 / かな出現数と `lemma` による統一例を出力．
- [ ] 同スクリプト内で，$L \in \{100, 500, 1000\}$ ごとに B（既定設定，全チャンクで fit）の付属語ブロック各次元のチャンク出現率（ $x_j > 0$ のチャンクの割合）を算出し，出現率上位 20 語とその値，全次元の出現率の中央値を出力．
- [ ] `presentation/qa_defense_matrix.md` に Q1～Q12 と追加質問が揃い，Q10（Jaccard / Dice を使わない理由）に上記の出現率が根拠として記載され，Q3・Q4・Q11 の修正後の記述が反映されている．
- [ ] 各想定質問に，既定仕様・切り替えフラグ・ 006 の結果数値（run 名付き）・数式の出典が対応付けられている．
