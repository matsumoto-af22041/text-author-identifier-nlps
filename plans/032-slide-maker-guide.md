# 032: スライド作成担当向けの手引きと 031・032 のコミット

## Expected Deliverables & Design Policy

- `presentation/README.md` （新規）：スライド作成担当（研究の経緯を知らない班員）が最初に読む手引き．1 ページ程度．内容は既存の資料からの要約と案内のみで，新しい数値や主張を加えない．
  - **使う資料**：`presentation_storyline.md` （スライドごとの載せるデータ・主張・数値・出典の設計図），`qa_defense_matrix.md` （想定質問と回答），`slides_d.md` （担当 D のスライド案），`figures/` （発表に使う図）．
  - **使わない資料**：`qa_verification_instructions.md` （2026-11-18 時点の旧値．数値を写さない），`note.md` （形態素解析辞書の選定の背景資料で，結果の数値は含まない）．
  - **数値の写し方**：数値は `presentation_storyline.md` と `qa_defense_matrix.md` から，出典の run 名（`full_006` ，`sensitivity_029_L*_clean` ，`function_word_profiles` 等）ごと写す．`outputs/` を直接見る場合も run 名を併記．古い run（`c_only` ，`base_c_022` 等）の数値は使わない．
  - **図**：`figures/` の図を使う（日本語表記，`full_006` から再生成，031）．`core_chunk_length.png` の作家別正解率の平均のパネルは凡例が線に一部重なるため，拡大表示する場合は注意．
  - **用語**：指標名は「作家別 MAP の平均」「作家別正解率の平均」「作家別 F 値の平均」．Balanced Accuracy，マクロ F1，（単独の）正解率は使わない．外部文献は引用しない．出典は講義資料の回・頁か「独自設計」．
  - **主張の線（言い過ぎないための注意）**：`plans/research-context.md` の「主結果の報告方針」と storyline の Slide 16 に従う旨と，次の要点を箇条で．
    - Core Message は判別（作家別正解率の平均）では支持，同一著者の文章を上位に並べる力（作家別 MAP の平均）では作品固有の語を除いた内容語とほぼ同等．
    - 「B が最良」とは言わない（未知作品で最も高いのは A，作品が変わって最も落ちにくいのは B）．
    - 寺田寅彦の 2 作品は例外として必ず示す．
    - 最良モデルの値は Test で選んだ楽観的な上限．
    - 付属語の具体例（表 B-3）は記述統計で，作家の癖か作品の語りの様式かは区別できない．
    - 仮説と確認済みの事実を区別する（storyline の「仮説」の表記を落とさない）．
  - **質問の窓口**：分からない数値・主張はリーダーに確認．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．
- 作成後，`main` で 2 回コミットする（研究者の承認済み，2026-11-23）．ステージはファイル名指定のみ．
  - 1 つ目：メッセージ `docs: 発表用の図の表記を研究の用語に合わせる` （1 行のみ）．対象：`scripts/plot_results.py` ， `scripts/plot_d_k_sweep.py` ， `configs/default.yaml` ， `presentation/qa_verification_instructions.md` ， `presentation/figures/` の変更済み 13 枚（`git status` で M のもの），`plans/031-presentation-figure-labels.md`
  - 2 つ目：メッセージ `docs: スライド作成担当向けの手引きを追加` （1 行のみ）．対象：`presentation/README.md` ， `plans/032-slide-maker-guide.md` ， `plans/wbs.md`

## Target File Paths

- `presentation/README.md` （新規）
- 上記のコミット対象

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/` の既存ファイル・図の内容を変更しない（README の新規作成のみ）．
- trailer（Co-Authored-By 等）を付けない．`git add -A` ・ `git add .` を使わない．push ・ amend ・ブランチ操作をしない．
- `plans/wbs.md` は変更しない（Orchestrator が更新済みのものをそのままコミット）．

## Verification Criteria

- [ ] `presentation/README.md` の案内するファイル名・run 名・図のファイル名がすべて実在する（`ls` で確認した一覧を報告）．
- [ ] README に新しい数値が含まれていない（記載した数値があれば出典の文書の該当箇所を報告）．
- [ ] `git show --stat HEAD~1` が 1 つ目の対象のみ，`git show --stat HEAD` が 2 つ目の対象のみ．
- [ ] `git status --short` が空．
