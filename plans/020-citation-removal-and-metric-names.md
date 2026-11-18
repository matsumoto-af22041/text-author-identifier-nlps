# 020: README・発表資料からの外部文献の削除と評価指標の名称の更新

## Expected Deliverables & Design Policy

- 背景（2026-11-18 決定）：原典を確認できない記憶に基づく外部文献の出典をすべて削除し，講義資料からの導出または独自設計の明示に置き換える．評価指標は講義資料にない名称（Accuracy，Balanced Accuracy，マクロ F1，MAP の無修飾の使用）を避け，素朴な名称に改める．正は `plans/research-context.md` の「Design Principle」「主評価指標」「講義資料で補えない箇所（独自設計）」．
- 使用箇所の一覧は `plans/wbs.md` の 019 の Review Notes と次の対象ファイル．対象ファイル内を自分で検索し，一覧に漏れがあれば同様に修正．
- **外部文献の削除**：次の文献の出典表記・文献リスト・「要確認」の注意書きをすべて削除し，`plans/research-context.md` の表の「扱い」列の内容に置き換える．
  - Deerwester et al. (1990)，Manning et al. (2008)（全節），Han, Kamber & Pei (2011)，Hurlbert (1984)，Mosteller & Wallace (1964)，Stamatatos (2008)，Stamatatos (2009)，伝ほか (2007)，金・樺島・村上 (1993)，金 (2002)，Fawcett (2006)
  - 置き換えの例：
    - LSI の射影式：講義第 5 回 p.37 の $X = U \Sigma V^\top$ から $U_K = X V_K \Sigma_K^{-1}$ を導出し，検索質問に同じ変換を適用．
    - $k$-NN：$k = 1$ は第 5 回 p.25，`per_author` と `global` は独自設計．
    - UniDic の語彙素：第 2 回 p.37 の UniDic と 004 の実測（「云う」「言う」等の統一）．
    - Min-Max 正規化，間引き，A の記号特徴：独自設計（根拠は実測のチケット番号）．
    - 作家別の値の平均，AP の分母：講義の指標を参考にした独自設計．
  - 先行研究で答えていた想定質問（Q7，Q14 等）と Core Message の説明は，「付属語が著者識別に有効か」を本研究の仮説として提示し 006 の実測で検証する，という回答に書き換える．先行研究に関する新たな記述は追加しない．
- **「擬似反復」の用語の置き換え**：「擬似反復」「Pseudoreplication」を使わず，「同じ作品のチャンクは題材・登場人物・底本の表記を共有して互いに独立でないため，同じ作品のチャンクが Train と Test に分かれると作品の一致で識別できてしまい精度が過大に出る（トピック漏洩）」という平易な説明にする．README の「実験用語の定義」の見出し・項目も同様．
- **評価指標の名称と定義（README の「実験用語の定義」）**：次を `plans/research-context.md` の「主評価指標」どおりに定義．
  - 作家別 MAP の平均（特徴量間比較の主指標，`map_macro` ）と全 Test チャンクの AP の平均（講義の MAP と同じ集約，`map_micro` ）．AP の分母を適合文書数とする理由（ $\min(R, N_{\text{out}})$ の解釈）を 2～3 行で．
  - 作家別正解率の平均（ `balanced_accuracy` ）と作家別 F 値の平均（ `macro_f1` ）．作家別の値を単純平均する理由（Test は間引かず作家構成が偏る）．
  - 正解率（ `accuracy` ）：参考値として結果 CSV にのみ出力し，表や発表資料には載せない．
  - 基準値：作家別正解率の平均 0.25，作家別 F 値の平均の一様ランダムな予測での期待値，MRR 約 0.521，MAP のランダム基準（式）．
  - $\Delta S$ は類似度の値の分布に依存するため，同一特徴量内の比較（チャンク長別等）に限定して用いる旨．
- **評価指標の名称（発表資料）**：`presentation/` の「Accuracy」「Acc」は「正解率」に，「マクロ F1」「macro-F1」は「作家別 F 値の平均」に改める．数値は変更しない（006 の結果での数値の置き換えは 007 で実施）．
- 上記以外の記述（環境構築手順，スライド構成等）は変更しない．

## Target File Paths

- `README.md`
- `presentation/presentation_storyline.md`
- `presentation/qa_verification_instructions.md`
- `presentation/note.md`
- 上記以外の `presentation/` 内のファイルで，検索により該当箇所が見つかったもの（報告に列挙）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `plans/` のファイルは変更しない（完了済みチケット 004・009・017 の文献表記は当時の記録として残す）．
- 数値は変更しない．新しい外部文献・先行研究の記述を追加しない．
- `references/` の中身は読まない（講義資料の頁は `plans/research-context.md` の記載を使う）．
- 体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペース．新設・変更する見出しに番号を付けない．
- 個人名・メールアドレス・ローカルの絶対パスを書かない（作家名・作品名は可）．
- 班員の実装方法（使うツール等）を制限する記述は書かない．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] `git grep -n -E "Deerwester|Manning|Kamber|Hurlbert|Mosteller|Federalist|Stamatatos|伝ほか|伝 ほか|金明哲|樺島|金 \(|Fawcett|要確認|擬似反復|Pseudoreplication|§" -- README.md presentation/` の出力をそのまま報告し，0 件であること．
- [ ] `git grep -n -E "Accuracy|Acc |マクロ F1|macro-F1|Balanced" -- README.md presentation/` の出力をそのまま報告し，残った一致がすべてコード上の列名（ `balanced_accuracy` ， `macro_f1` ， `accuracy` ）の説明であること．
- [ ] 修正箇所の一覧（ファイル，行番号，修正前の要旨，修正後の要旨）を報告．
- [ ] README の「実験用語の定義」で，作家別 MAP の平均，全 Test チャンクの AP の平均，作家別正解率の平均，作家別 F 値の平均，正解率，基準値，$\Delta S$ の位置付け，トピック漏洩（擬似反復の言い換え）の各項の行番号を報告し，`plans/research-context.md` の該当箇所と矛盾しないことの対応表を報告．
- [ ] `git diff --stat` で変更ファイルが Target File Paths の範囲内であること．
