# 038: storyline の誤りの修正，氏名の表記，`.gitignore` の追加

## Expected Deliverables & Design Policy

- `.gitignore` ：「講義資料・配布資料」の節の後ろに，発表資料の節を追加（コメント付き）．
  - `presentation/orig.*` （担当から上がった元のスライド．個人名を含む）
  - `presentation/*.pdf` （pptx から書き出す派生物）
  - `presentation/*_提出用.*` （提出直前に氏名を書き込んだ版）
  - 追加後，`git status --short` で `presentation/orig.pptx` ・ `orig.pdf` が表示されないこと，`presentation/figures/*.png` ・ `presentation/*.md` が無視されないことを確認．
- 氏名の表記（研究者の指示，2026-11-24）：`presentation/presentation_storyline.md` ，`presentation/README.md` ，`presentation/qa_defense_matrix.md` の「（氏名：提出直前に記入）」等の氏名のプレースホルダを「担当 A」～「担当 D」「リーダー」に置換．README の「個人名・秘匿情報の扱い」の節を，`plans/research-context.md` の「Data Handling Guardrails」の該当項目（氏名は「担当 A」等，担当の割り当ては従来どおりで実際のスライド作成は研究者，元のスライドは `presentation/orig.*` で Git 管理外，発表スライドは 1 ファイルで Git 管理，提出用の版と PDF は管理外）に合わせて更新．
- storyline の修正（037 の指摘）：
  - 補助 S8 の「A・B では 4 作家中 3 作家で本人がいちばん近い」：C・D も同条件で 4 作家中 3 作家（例外は太宰）であるため，A・B の特徴とする記述を削除し，`full_006` の 4×4 作家間平均類似度行列（$L = 500$ ，Inter-work，間引きあり）の事実（各特徴量で対角が行の最大になる作家数と例外）に置き換え．値はスクリプトで確認．
  - 出典一覧：チケット番号（010 等）を削除（「独自設計」の記載のみ）．本文で使う講義資料の回・頁（第 6 回 p.42，第 2 回 p.36 等）がすべて出典一覧にあるよう，本文の出典と一覧をスクリプトで突き合わせて追加．追加する回・頁は `references/` の講義資料で該当内容があることを確認（034 と同じく `pdftotext` 等．`references/` は読むのみ）．
  - 補助 S7 の表のセル内の `|` を含む条件名（`B|L1000|inter|bal|cosine|knn10_per_author` ）：表が崩れないよう，研究の用語での表記（B，$L = 1000$ ，作品間分割，間引きあり，作家別上位 $k$ 平均（$k = 10$ ））に置換し，run の条件名は表の外の注に移す．
- `presentation/slides_d.md` ：見出しのスライド番号を新構成（Slide 15～17）に合わせて修正．氏名のプレースホルダがあれば同様に置換．

## Target File Paths

- `.gitignore`
- `presentation/presentation_storyline.md`
- `presentation/README.md`
- `presentation/qa_defense_matrix.md`
- `presentation/slides_d.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/orig.*` を読まない・変更しない・ステージしない．
- `references/` は読むのみ．講義資料の文言を資料に長く引用しない．
- 新しい主張を加えない．数値は出典から確認．
- 個人名を書かない．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．
- `plans/` ・ `src/` ・ `scripts/` ・ `configs/` を変更しない．コミットしない．

## Verification Criteria

- [ ] `git status --short` に `presentation/orig.*` が出ない．`git check-ignore -v` で 3 つのパターンが効いていることと，`presentation/figures/` の図・`presentation/*.md` が無視されないことを報告．
- [ ] `presentation/` の Markdown に「氏名：提出直前」等の旧プレースホルダが 0 件．個人名がないことの確認結果（036 と同じ方法）．
- [ ] 補助 S8 の新しい記述の値と，`full_006` の行列から計算した値の対応を報告．
- [ ] 本文の出典（講義資料の回・頁）と出典一覧の突き合わせ結果（追加した回・頁と，`references/` で確認した内容の要約）．出典一覧にチケット番号が 0 件．
- [ ] 034 の監査スクリプトを再実行して不一致 0．
