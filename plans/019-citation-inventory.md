# 019: 外部文献の出典の使用箇所の洗い出し

## Expected Deliverables & Design Policy

- 目的：記憶に基づき原典を確認していない外部文献の出典を削除するため，各文献がリポジトリのどこでどの文脈で使われているかを研究者に示す．本チケットは調査と報告のみで，ファイルは変更しない．
- 対象の文献（表記揺れを含めて検索．例：著者名のみ，年のみ，「et al.」，節番号「§」）：
  - Deerwester et al. (1990)
  - Fawcett (2006)
  - Han, Kamber & Pei (2011)
  - Hurlbert (1984)
  - Manning, Raghavan & Schütze (2008)（「Manning」「IIR」「Introduction to Information Retrieval」等）
  - Mosteller & Wallace (1964)（「Federalist」も検索）
  - Stamatatos (2008)，Stamatatos (2009)
  - 伝ほか (2007)（「伝康晴」「伝 ほか」等）
  - 金・樺島・村上 (1993)，金 (2002)（「金明哲」「金 (」等）
  - Brodersen
  - 上記以外に見つかった外部文献の出典（年号付きの著者名，DOI，書名等）
- 検索範囲：git 管理下の全ファイル（`git ls-files` の対象．`plans/`，`README.md`，`presentation/`，`src/`，`scripts/`，`configs/` 等）．`outputs/`，`data/` は対象外．
- 講義資料での言及の確認：`references/nlps26-1.pdf` ～ `nlps26-6.pdf` の全ページのテキストを抽出し，各文献の著者名・書名・手法名の言及（例：LSI の原論文，擬似反復，Federalist）があるかを確認．018 と同じ方法（テキスト抽出，抽出できないページは画像で確認）で可．
- 報告内容：文献ごとに次の表．
  - 使用箇所：ファイル（リポジトリ相対パス）と行番号
  - 文脈：その行が何の主張・手法の根拠として引用されているかの要約（1 行）
  - 講義資料での言及：あり（回・ページと内容の要約）／なし
  - 削除した場合の影響：その主張が講義資料で代替できるか，独自設計の明示に置き換えるか，主張自体を削る必要があるかの見立て（1 行）

## Target File Paths

- なし（リポジトリのファイルは変更しない）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 調査用スクリプトや抽出テキストは scratchpad に置き，リポジトリに追加しない．
- `references/` のファイルを変更・移動・コミットしない．
- 報告に受講者・教員の個人名，メールアドレス，ローカルの絶対パスを含めない（講義資料はファイル名とページ番号で参照）．
- コミットしない．`plans/wbs.md` は更新しない（Orchestrator が結果を記録）．

## Verification Criteria

- [ ] 対象の各文献について，使用箇所が 0 件の場合も含めて表を報告し，検索に使ったパターンを併記．
- [ ] 講義資料 6 回分の総ページ数と調べたページ数を報告．
- [ ] `git status --porcelain` の出力が調査前から変化していないこと．
