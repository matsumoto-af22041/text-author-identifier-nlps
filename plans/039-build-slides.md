# 039: 発表スライドの生成スクリプト（storyline から pptx を作る）

## Expected Deliverables & Design Policy

- 研究者の承認（2026-11-24）：担当のスライドは使わず，`presentation/presentation_storyline.md` （036・038 の新構成）から発表スライドをプログラムで作り直す．研究者と Orchestrator で見栄えを反復して直す前提の初版．
- 依存関係：`python-pptx` を開発用の依存として `pyproject.toml` に追加（`uv add --dev python-pptx` 相当，`uv.lock` を更新）．研究のコード（`src/` ）では使わない．
- `scripts/build_slides.py` （新規）：`presentation/slides.pptx` を生成．
  - スライドの内容はスクリプト内または `presentation/slides_content.yaml` 等の構造化データに持ち，storyline の各スライドの「載せるデータ・主張・出典」から作成（文章は storyline の要約で，新しい数値・主張を加えない）．構造化データを使う場合はそのファイルも Target に含める．
  - 構成：本編 21 枚 → 出典一覧 1 枚 → 補助スライド S1～S14（storyline の順と番号）．
  - 見た目：`presentation/orig.pptx` から色・フォント・スライドの縦横比のみを読み取って引き継ぐ（文字・図・メモは使わない．`orig.*` は読むのみ，個人名を出力・ログに書かない）．
  - 書き方の規則（`presentation/README.md` の「スライドの書き方」）を守る：短い文，列挙は箇条書き，1 スライド 1 主張，定義はスライド 2 と各パートの最初，出典は本文に書かず右肩の [n] とスライド下部の薄い文字（講義資料の回・頁か「独自設計」のみ．プログラム名・run 名・チケット番号は書かない），A～D は定義 / 実験の概要 / 結果・考察の 3 枚．
  - 発表者ノート：storyline の主張・想定時間と，数値の出典の run 名（ノートには書いてよい）．
  - 図：`presentation/figures/` の現行版を使う．storyline で「スライド作成時に作成」とした図のうち，データから描けるもの（グラフ A・C・F）は `full_006` 等の出力から描く図のスクリプトを追加して `presentation/figures/` に保存（031 と同じ日本語表記・フォントの規則）．概念図（Slide 1）と作品内分割・作品間分割の図（Slide 4 の図 1）は pptx の図形で描く．
  - 数式・添字：`_` 等の平文の添字を使わない（上付き・下付きの書式または数式画像）．
  - 氏名：「担当 A」～「担当 D」「リーダー」．pptx のメタデータ（作成者・最終更新者・会社名等）を空にする．
- `scripts/check_slides.py` （新規）：生成した pptx を検査し，違反があれば停止（終了コード 1）．
  - 数値：スライド本文・表・ノートの数値を storyline・qa の値と照合（037 の `verify_slides.py` の方法を流用可）．
  - 書き方：出典行にプログラム名・`.py` ・ run 名・チケット番号がない，本文に「第 N 回 p.」の直書きがない，読点区切りの長い列挙がない（基準はスクリプトに明記），1 段落の文字数の上限（基準を明記）．
  - 個人名・秘匿情報：メタデータが空，メールアドレス・ローカルパス・学籍番号の形の文字列がない，git のユーザー名・ログイン名の一部がない．見つけた文字列は出力しない（箇所のみ）．
  - 構成：枚数と順（本編 21，出典 1，補助 14），A～D の 3 枚構成．
- 再現コマンドを `presentation/README.md` に追記（`uv run python scripts/build_slides.py` ， `uv run python scripts/check_slides.py` ）．README に「発表スライドは `presentation/slides.pptx` の 1 ファイル．修正は生成スクリプト（と内容のデータ）で行う」旨を追記．

## Target File Paths

- `pyproject.toml` ， `uv.lock`
- `scripts/build_slides.py` （新規）， `scripts/check_slides.py` （新規）
- 図を描くスクリプト（新規，名前は実装時に決定）と `presentation/figures/` の追加図
- `presentation/slides_content.yaml` 等の内容データ（使う場合）
- `presentation/slides.pptx` （生成物．Git 管理対象）
- `presentation/README.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `presentation/orig.*` を変更・ステージしない．個人名を出力・ログ・報告に書かない．
- storyline・qa・`src/` ・ `configs/` ・既存の `scripts/` を変更しない（変更が必要なら報告）．
- 新しい数値・主張を加えない．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/build_slides.py` が完走し `presentation/slides.pptx` を生成．枚数と構成（本編 21，出典 1，補助 14）を報告．同じコマンドを 2 回実行して，スライドの内容（テキスト・図の参照）が同一（生成日時等のメタデータを除く）．
- [ ] `uv run python scripts/check_slides.py` が終了コード 0．照合した数値の件数，書き方の検査の基準と結果，個人名の検査の結果（箇所のみ）を報告．
- [ ] 生成した pptx を PDF に書き出せる環境（LibreOffice 等）があれば書き出して主要スライドを画像で確認し，崩れ（はみ出し・重なり・読めない文字）を報告．なければその旨と，pptx の図形の座標から検出したはみ出し（スライドの外に出る要素）の有無を報告．
- [ ] 追加した図の描画値が `full_006` 等の出力と一致（031 と同じく描画前の数値で比較）．
