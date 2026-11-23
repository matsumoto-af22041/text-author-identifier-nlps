# 007: 発表資料の更新・質疑応答対策ドキュメントと補助検証スクリプト

## Expected Deliverables & Design Policy

- 担当：全員（統合はリーダー）
- 目的：006 の Full Experiment（`outputs/full_006/` ，コミット `9668297` ，`dirty: false` ）と補助実験・調査（`outputs/exclusion_summary_ABCD/` ，`outputs/centroid_norm_ABCD/` ，028，029）の結果を発表資料に反映し，想定質問への回答を数値付きで整備する．
- 数値の出典：すべて上記の出力から転記し，run 名（`full_006` 等）を併記．`c_only` ・ `base_c_022` 等の旧 run の数値は残さない．数値の書き方は「平均 ± 標準偏差（5 シード）」，間引きなしは 1 回の値．

### 発表の主張（`plans/research-context.md` の「主結果の報告方針」「推論方式」に従う）

- 代表条件は重心法，$L = 500$ ，Inter-work・間引きあり．表は作家別 MAP の平均（主指標）と作家別正解率の平均・作家別 F 値の平均を並べ，基準値を併記．正解率（ `accuracy` ）は載せない．
- 特徴量間の比較（ $L = 500$ ，Inter-work，間引きあり，作家別 MAP の平均）：A 0.472，B 0.420，C 0.306，D 0.292（基準値 0.260）．Intra-work → Inter-work の低下は B が最小（−0.082），C・D が大きい（−0.202，−0.241）．主張は「内容によらない A・B は未知作品に強く，内容語系の C・D は作品が変わると大きく落ちる」．「B が最良」とは書かない．
- Core Message（Slide 7-8，16）：B の `blocks=words` で付属語 vs 内容語．
  - 作家別正解率の平均：Inter-work で付属語が全 $L$ ・間引きの有無によらず +0.09～+0.22 上回る．Intra-work では内容語が上（トピック漏洩と整合）．
  - 作家別 MAP の平均：Inter-work で付属語が上回るが差は小さく（+0.014～+0.049，間引きあり）．
  - 語彙選択に対する感度（029，`outputs/sensitivity_029_L{100,500,1000}/` ）を両指標で表にする：`bal` ・ `nobal` ・ `fitfull_poolbal` ・ `fitbal_poolfull` ・ `drop_single_author` の付属語 − 内容語の差．特に `drop_single_author` （Train で 1 作家にしか出ない語を除く）では MAP の差が +0.006（ $L = 500$ ）・+0.003（ $L = 1000$ ）とほぼ消え，作家別正解率の平均の差も +0.186 → +0.085（ $L = 500$ ）に縮む．MAP だけでなく作家別正解率の変化も必ず併記．
  - 結論の文言：「未知作品の判別では，内容語の語彙選択を変えても付属語が内容語を上回る（作家別正解率の平均の差は $L = 500, 1000$ で +0.085～+0.227， $L = 100$ で +0.038～+0.103，Executor が 029 の出力から訂正）．同一著者の文章を上位に並べる力（MAP）では差は小さく，作品固有の語を除いた内容語とはほぼ同等」．
  - 作品ごとの一貫性：MAP で付属語が上回るのは 9 作品中 7 作品（各 $L$ ）．例外の寺田寅彦 2 作品（随筆で作品間の題材が近い可能性，仮説）を明記．9 作品のため検定はしない旨．
  - `blocks=all` では差が小さい理由（品詞 2-gram ブロックが二乗ノルムの 65～95% を占める，025）．
- 推論方式（Slide 14-15）：間引きなしで `per_author` が大きく下がり重心法は不変（B， $L = 500$ ：0.577 vs 0.408）．最良モデル（Test で選択）は楽観的な上限である旨を明記．
- 漱石への偏りと重心の割り増し（質疑用）：作品除外での変動（±0.05），漱石以外のチャンク中の漱石予測割合（B 0.146，C 0.351），割り増し係数の比（A・B ほぼ 1，C・D は最大 1.43）．
- 底本の表記の交絡（Slide 5-6，質疑用）：A が最高値である点について，底本の表記方針を拾っている可能性と，文字 Bi-gram が付属語を含む文字列を拾っている可能性（いずれも仮説）を併記．`char_type_ratios` の作品別字種構成比を補助検証スクリプトで出力して根拠に使う．

### 更新対象

- `presentation/presentation_storyline.md` ：「要計測（006）」「要計測（003）」の全箇所を上記の数値で置換．Slide 7-8 の主結果表 3 を `words` 主・ `all` ・ `pos_bigram` 併記に変更．Slide 9-10 の `c_only` の数値と Intra-work 暫定値を `full_006` の値に置換．Slide 13 の仮説の文言を結果に合わせて修正（仮説と異なった点はそのまま記載）．Slide 16 の結論を上記の文言に．参考文献・出典の規則は既存のまま．
- 図：`outputs/full_006/figures/` から発表に使う図（`core_chunk_length.png` ，代表条件の類似度分布・ヒートマップ，`chunk_length_curves.png` ，`confusion_best.png` ）を `presentation/figures/` にコピーし，storyline から参照．
- `presentation/slides_d.md` ：023 からの申し送り（「残りの 75% はチャンク固有の語」の削除または仮説の明記，累積寄与率の表が間引きありの Train である旨の追記（間引きなしは 0.216））．数値が `full_006` の重心法の値と一致することを確認（不一致なら置換）．
- `presentation/qa_defense_matrix.md` （新規）：想定質問 Q1～Q12（ `presentation/qa_verification_instructions.md` ）と追加質問のマスタードキュメント．各質問に既定仕様・切り替えフラグ・数値（run 名付き）・数式の出典（講義資料の回・頁，または `plans/research-context.md` の「講義資料で補えない箇所（独自設計）」）を対応付け．
  - 001 の検証による修正：Q3（正規化で分散の差は残る， $L = 100$ で変動係数約 26%），Q4（作品単位の最大破棄率約 1.66% を併記），Q11（`per_author` で $k \to |D_a|$ は全チャンク平均類似度への漸近で重心との類似度とは一致しない．`per_author` は $k$-NN 法の中での既定で，代表条件は重心法）．
  - 追加質問：表記の差は文体か底本か／なぜ普通の正解率を使わないのか／なぜ Train を間引くのか／MAP の差が小さいのに付属語が有効と言えるのか（028・029）／結果を見てから指標を変えたのでは（主指標は 2026-11-18 に結果の前に決定，変更していない）／寺田寅彦だけ逆なのはなぜか／A の方が B より高いのでは／Jaccard / Dice を使わない理由（Q10，出現率の根拠）．
- `scripts/verify_qa_points.py` （新規）：Q1 のコーパス内作品分布，Q2 の表記揺れ語の漢字 / かな出現数と `lemma` による統一例，Q3 のチャンク実測文字数（平均・標準偏差・最大），Q4 の端数破棄率（全体・作品単位最大），Q10 の B の付属語ブロック各次元のチャンク出現率，A の作品別字種構成比（`char_type_ratios` ）．

## Target File Paths

- `presentation/presentation_storyline.md`
- `presentation/slides_d.md`
- `presentation/qa_defense_matrix.md` （新規）
- `presentation/figures/` （図のコピー）
- `scripts/verify_qa_points.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/qa_verification_instructions.md` は入力資料として読むのみで変更しない．
- `src/` ・ `configs/` ・既存の `scripts/` を変更しない．
- `data/raw/corpus20150426_with_titles.txt` を Read ツールや全文出力で直接読まない．コーパス情報は `data/processed/authors_corpus.json` または `src/dataset.py` 経由．
- `scikit-learn` 等は禁止．`numpy` と `fugashi` （`src/morphology.py` 経由）のみ．
- 数値は上記の出力から転記し run 名を併記．028 の漏洩を含む診断値（Test 作家の Train 側作品にしか出ない語を除いた値）は使わない．
- 仮説と確認済みの事実を区別して書く．講義資料にない指標名（Balanced Accuracy，マクロ F1）や外部文献の出典は書かない．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/verify_qa_points.py` が完走し，上記の各項目と B の付属語ブロックの出現率上位 20 語・全次元の出現率の中央値（ $L \in \{100, 500, 1000\}$ ，B は全チャンクで fit した記述統計である旨を出力に明記）を出力．
- [ ] `git grep -n "要計測" -- presentation/presentation_storyline.md` が 0 件．`git grep -n "c_only\|0\.955" -- presentation/` が 0 件（`qa_verification_instructions.md` を除く）．
- [ ] storyline と `qa_defense_matrix.md` の主要な数値（特徴量間比較 4 値，主結果の `words` の差（全 $L$ ・両指標），作品ごとの 7/9，推論方式の比較の 2 値）を `outputs/full_006/` の CSV と照合した対応表を報告（不一致 0）．029 の感度の数値は 029 の出力と照合．
- [ ] `qa_defense_matrix.md` に Q1～Q12 と追加質問がすべてあり，各質問に既定仕様・数値（run 名付き）・出典が付いている．
- [ ] storyline から参照する図がすべて `presentation/figures/` に存在．
