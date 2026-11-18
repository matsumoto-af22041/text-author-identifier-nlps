# 018: 講義資料における評価指標の記載箇所の調査

## Expected Deliverables & Design Policy

- 目的：判別精度の指標（Accuracy 等）に講義資料の根拠があるかの確認．第 5 回 p.27-31（再現率・適合率・F 値・AP・MAP・MRR）と第 2 回 p.47-48（文書分類）は Orchestrator が確認済みで，Accuracy は見つかっていない．
- 対象：`references/nlps26-4.pdf` と `references/nlps26-6.pdf` の全ページ．余力があれば `nlps26-1.pdf` ～ `nlps26-3.pdf` も同じ方法で検索（第 5 回は確認済みのため不要）．
- 方法は任意（PDF のテキスト抽出と検索，テキストが取れないページは画像として読む等）．調査用のスクリプトは scratchpad に置き，リポジトリに追加しない．
- 検索語：正解率，正答率，精度，正確度，accuracy（大文字小文字を区別しない），識別率，誤り率，適合率，再現率，F 値，F1，マクロ，混同行列，confusion，MAP，MRR，評価．
- 報告内容：
  - 各 PDF の総ページ数と，テキスト抽出できなかったページ（画像で読んだかどうか）．
  - ヒットごとに，ファイル名・ページ番号・スライドの題名・該当する式または定義文の要約（1～2 行）．
  - 特に，分類の正解率（正しく分類された件数 / 全件）に相当する定義の有無を明記．定義がある場合は式をそのまま書き写す．
  - 「評価」等の一般語のヒットで評価指標と無関係なものは件数のみ．

## Target File Paths

- なし（リポジトリのファイルは変更しない）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `references/` のファイルを変更・移動・コミットしない．
- 報告に受講者・教員の個人名，メールアドレス，ローカルの絶対パスを含めない（ファイル名とページ番号で参照）．
- コミットしない．`plans/wbs.md` は更新しない（Orchestrator が結果を記録）．

## Verification Criteria

- [ ] `nlps26-4.pdf` と `nlps26-6.pdf` の全ページを調べたことを，総ページ数と調べたページ数で報告．
- [ ] 分類の正解率に相当する定義の有無を，PDF ごとに「あり（ページ番号と式）」または「なし」で報告．
- [ ] `git status --porcelain` の出力が調査前から変化していないこと．
