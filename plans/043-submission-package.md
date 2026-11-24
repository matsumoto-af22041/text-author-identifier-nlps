# 043: 提出用のプログラム一式の作成

## Expected Deliverables & Design Policy

- 背景：11/25（水）23:59 までに，プレゼンのファイルと作成したプログラム一式を ScombZ に提出（第 6 回 p.48-49，`plans/research-context.md` の「発表の要件」）．本チケットはプログラム一式の zip を作る．スライドは別に研究者が氏名を記入して提出する．
- 範囲（研究者決定，2026-11-24）：
  - 入れる：`src/` ， `scripts/` ， `configs/` ， `pyproject.toml` ， `uv.lock` ， `.python-version` ，提出用の README．
  - 入れない：`data/raw/` （配布コーパスそのもの）， `plans/` ， `.claude/` ， `references/` （受講者の氏名を含む）， `presentation/` （発表資料は別に提出．`orig.*` を含む）， `outputs/` ， `data/processed/` （コーパスからの派生物）．エージェント用のファイル（`CLAUDE.md` ， `.agents/` ， `llm*.md` ， `llm.txt` ）も入れない．
  - `scripts/` のうち発表スライドの生成・検査（`build_slides.py` ， `check_slides.py` ）は `presentation/` の内容データがないと動かないため入れない．`scripts/` の他のスクリプトで，入れないファイル（`plans/` ・ `presentation/` 等）がないと動かないものがあれば一覧を報告し，README に注記（入れるか外すかは報告で判断を仰ぐ．勝手に外さない）．
- 提出用の README（zip の直下，新規．リポジトリの `README.md` とは別ファイルとして `scripts/` 等と同じく生成元をリポジトリに置く．置き場所は `submission/README.md` 等で提案）：
  - システムの概要（青空文庫の 4 作家・9 作品の著者識別．特徴量 A～D とコサイン類似度・重心法を自作）と，使用したツール（Python 3.13 ・ uv ・ NumPy ・ fugashi と UniDic ・ Matplotlib ・ PyYAML ．scikit-learn 等は不使用）．
  - ディレクトリ構成と各モジュールの役割（`src/` の各ファイル，主な `scripts/` ）．
  - 動かし方：コーパス（配布ファイル名）を `data/raw/` に置く → 前処理 → 実験 → 図・集計の再現コマンド（既存の README・ research-context の記載を確認して正確に）．コーパスの同一性は `meta.source_sha256` の値で確認できること（値を記載）．全実験の所要時間の目安．
  - 個人名・学籍番号・メールアドレス・ローカルパスを書かない（氏名欄が必要なら「（提出直前に記入）」のプレースホルダ）．
- zip の作成スクリプト（`scripts/make_submission.py` 等，新規）：Git で追跡されているファイルのうち上記の範囲だけを集めて `outputs/submission/` （Git 管理外）に zip を作る（作業ツリーの未追跡・未コミットのファイルを混ぜないよう，`git archive` 等で HEAD の内容から作る．未コミットの変更がある場合は警告して停止）．zip のファイル名に個人名を含めない．
- 秘匿情報の検査：zip の中のすべてのファイルについて，メールアドレス・ローカルの絶対パス（`/Users/` 等）・学籍番号の形の文字列・ git のユーザー名・ログイン名の一部がないことを検査（見つけた文字列は出力せず箇所のみ）．zip の中の `run_meta.json` 等の生成物がないことも確認．

## Target File Paths

- `scripts/make_submission.py` （新規）
- 提出用の README の生成元（新規，置き場所は提案）
- `outputs/submission/` （生成物，Git 管理外）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `configs/` ・既存の `scripts/` ・ `plans/` ・ `presentation/` を変更しない（必要なら報告）．
- `references/` ・ `data/raw/` の中身を zip に入れない．生コーパスを全文出力しない．
- 個人名を出力・ログ・報告に書かない．コミットしない．

## Verification Criteria

- [ ] zip の中のファイルの一覧（パスのみ）と合計サイズを報告．入れない範囲のファイルが 0 件．
- [ ] 秘匿情報の検査の結果（検査した種類と件数，見つかった箇所）．
- [ ] zip に `__pycache__` ・ `*.pyc` ・ `.venv` ・ `.DS_Store` 等のキャッシュ・環境のファイルが 0 件．
- [ ] `src/` ・ `scripts/` ・ `configs/` のコメント・docstring にあるチケット番号（`（041）` ， `013～015` 等の，提出物に含まれない `plans/` を指す番号）の件数とファイルの一覧を報告（コードは変更しない．README に「コメント中の 3 桁の番号は開発時の作業記録の番号」と注記するか，研究者の判断を仰ぐ）．
- [ ] zip を scratchpad に展開し，`uv sync` と，コーパスを置いた状態での前処理と小さな実験（既存のスモーク用の設定・引数があればそれ．10 分以内）が README の手順どおりに動くことを確認（コーパスは `data/raw/` のものを展開先に一時的にコピーまたはリンクし，確認後に削除）．
- [ ] 入れないファイルに依存して動かないスクリプトの一覧と扱いの提案．
