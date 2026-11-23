# 034: 発表資料が予定する結果の存在確認（監査）

## Expected Deliverables & Design Policy

- 背景：033 で，storyline の Slide 2 の表 1 が出典（`verify_qa_points` の Q1）を挙げながら値が出力されていなかったことが判明（2026-11-24）．同種の抜けが他にないかを網羅的に確認する．
- 対象：`presentation/presentation_storyline.md` の全スライド（Slide 1～17）と付随する節，`presentation/qa_defense_matrix.md` の全質問，`presentation/slides_d.md` ，`presentation/README.md` ．
- 各資料が「載せる」「示す」「出典」として挙げる結果をすべて列挙し，項目ごとに次を判定する．
  - **種類**：表・図・数値・具体例・再現コマンド・出典の run / スクリプト．
  - **値の有無**：値（または図のファイル）が資料の中に実際に書かれているか．「（予定）」「後で」「要計測」等の未記入，出典だけで値がない箇所を不足とする．
  - **出典の実在**：挙げられたファイル（`presentation/figures/` の図，`scripts/` のスクリプト，`outputs/` の run・ファイル）が存在するか．スクリプトの場合，その出力に該当する値が実際に出るか（実行して確認．各 10 分以内）．
  - **値の一致**：資料の値と出典の値の一致（出典の CSV・スクリプト出力・図と照合．007・030・033 で照合済みの項目は再照合の結果のみ記載でよい）．
  - **再現性**：再現コマンドが書かれている場合，そのコマンドが現在のコードで動くか．
- 結果は `outputs/audit_034/report.md` に表で出す（スライド / 質問ごとに 1 行以上，判定：OK / 値なし / 出典なし / 不一致 / 再現不可）．不足の項目には「何が足りないか」「埋めるのに必要な作業（既存の出力からの転記で済むか，新たな計算が必要か）」を書く．
- 修正は行わない（監査のみ）．

## Target File Paths

- なし（リポジトリのファイルは変更しない．出力は `outputs/audit_034/` と scratchpad）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/` ・ `scripts/` ・ `src/` ・ `configs/` ・ `plans/` を変更しない．
- `data/raw/` のコーパスを直接読まない．
- 照合はスクリプトで行い，目視のみの判定は「目視」と明記．
- コミットしない．

## Verification Criteria

- [ ] `outputs/audit_034/report.md` に，storyline の Slide 1～17（1a を含む）と付随する節，qa の全質問，slides_d，README のすべてが少なくとも 1 行ずつ含まれる（件数を報告）．
- [ ] 判定ごとの件数と，OK 以外の全項目の一覧（資料の箇所，種類，不足の内容，必要な作業）を報告．
- [ ] 実行したスクリプト・照合の方法と所要時間を報告．
