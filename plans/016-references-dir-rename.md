# 016: 参照資料ディレクトリの改名（ `docs/` → `references/` ）への追従と git 除外

## Expected Deliverables & Design Policy

- 研究者が講義資料・配布資料のディレクトリを `docs/` から `references/` に改名済み（ファイルの中身は不変）．リポジトリ内の参照を追従させる．
- `references/` を `.gitignore` に追加する．
  - 理由：講義資料に受講者の実名が含まれ，著作権も大学側にあるため．
  - 理由：未追跡のままだと `scripts/run_experiments.py` の `git_state()` が常に `dirty: true` を記録するため．
- 追加後，`git status --porcelain` が空になり，実験の `run_meta.json` が `dirty: false` を記録できる状態にする．

## Target File Paths

- `.gitignore`
- `git ls-files` に含まれるファイルのうち，ディレクトリとしての `docs/` を参照しているすべてのファイル（ `plans/` ， `presentation/` ， `README.md` ， `CLAUDE.md` ， `.claude/` 配下， `scripts/` ， `src/` ， `configs/` を含む）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 置換対象は，講義資料・配布資料のディレクトリとしての `docs/` （例：`docs/nlps26-5.pdf` ， `docs/C班_自然言語システムv2.pdf` ，ディレクトリ構成の説明の `docs/` ）のみ．コミット prefix の `docs:` など，ディレクトリを指さない文字列は変更しない．各一致箇所を目視で確認してから置換する．
- `plans/` 配下の置換は，パス文字列の置換のみ．それ以外の記述は変更しない（Done のチケットも，参照切れを防ぐためパスのみ置換する）．
- `references/` の中身は読み込み・変更しない．
- 置換以外の内容変更はしない．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] 置換前に `git grep -n "docs/"` の全一致を一覧化し，置換した箇所と置換しなかった箇所（理由付き）を報告．
- [ ] 置換後の `git grep -n "docs/"` の出力をそのまま報告し，残った一致がすべてディレクトリを指さない文字列であること．
- [ ] 置換後のファイルに出てくる `references/` のファイル名が実在すること（`ls references` の出力と照合）．
- [ ] `git status --porcelain` に `references/` が出ないこと．出力をそのまま報告．
- [ ] `git check-ignore -v references/` が `.gitignore` の追加行を示すこと．
