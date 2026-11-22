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
- 判別精度の主指標の変更（2026-11-18 決定）の反映：`presentation/` の資料（ `qa_verification_instructions.md` を除く）にある正解率（Accuracy）の数値を，006 の結果の作家別正解率の平均に置き換え（名称の修正は 020 で実施済みの前提）．
  - 020 からの申し送り：`presentation/presentation_storyline.md` の Slide 9-10 と `qa_verification_instructions.md` の Q5 に，`c_only` の正解率の数値が名称のみ変更して残る．正解率は発表資料に載せない方針のため，作家別正解率の平均に差し替える．
  - storyline の評価データ定義と表 3～5 に，主指標（作家別 MAP の平均，作家別正解率の平均）と基準値を追加．
  - README の `delta_s_macro` の「（主指標）」は $\Delta S$ のマクロ版とミクロ版の間での主従であることが分かる表記に修正（特徴量間比較の主指標は作家別 MAP の平均）．新規の想定質問「なぜ普通の正解率を使わないのか」（Test の作家構成の偏り，正解率は結果 CSV の `accuracy` 列で回答可能）を追加．
- 023 からの申し送り（D のブランチは手直しなしでマージ，2026-11-23）：
  - `presentation/slides_d.md` の想定質問にある「残りの 75% には個々のチャンク固有の語が多く含まれる」は未検証のため，削除するか仮説と明記．
  - 同ファイルの累積寄与率の表（ $K = 32$ で 0.249）に，間引きありの Train の値である旨を追記（間引きなしでは 0.216）．
  - Q11 の「`knn_mode="per_author"` を既定」は $k$-NN 法の中での既定であり，代表条件は重心法である旨が分かる書き方にする（024 からの申し送り，`qa_defense_matrix.md` 側で記述）．
  - 全スライドの数値が代表条件の重心法（ $L = 500$ ，`plans/research-context.md` の「推論方式」）の値であることを確認し，`knn5_per_author` の値が残っていれば差し替え．
- 想定質問ごとに，本実験の既定仕様と，対照実験に用いた切り替えフラグ（ `scaling` ， `token_key` ， `knn_mode` ， $k$ ， `n_components` ）および 006 の結果の数値を対応付け．

## Target File Paths

- `presentation/qa_defense_matrix.md`
- `scripts/verify_qa_points.py`
- `presentation/presentation_storyline.md` ， `README.md` （020 からの申し送りの修正）
- `presentation/slides_d.md` ほか班員のスライド案（023 からの申し送りの修正）

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
