# 040: 発表スライドの組版の仕組み（数式・定義の一覧・照合の強化）

## Expected Deliverables & Design Policy

- 背景：039 で生成した `presentation/slides.pptx` への研究者のフィードバック（2026-11-24）のうち，スライド全体に効く組版の仕組みを先に整える．内容・構成の修正は 041 で行う．本チケットでは既存の内容（文言・数値）を変えない．
- **数式（研究者の指示：「数式は全部 LaTeX の数式のように」）**
  - 方式（2026-11-24 改訂．PowerPoint のネイティブの数式（OMML）は研究者が見た目を好まないため不採用）：`slides_content.yaml` の文字列中の数式を `$...$` （LaTeX の記法）で書き，ビルド時にローカルの TeX で 1 式ずつコンパイルした画像としてスライドに置く．PowerPoint 上では式を編集できず，修正は yaml とスクリプトで行う（研究者承認済み）．
    - 置き方：独立した数式（定義式等）は行として画像を置く．定義の一覧（後述）の記号の列は記号の画像をセルの位置に合わせて置く．文中の短い記号（1 文字の変数，添字のない記号）は画像にせず，セリフ体のイタリックの文字で書く．文中に複雑な式が残る箇所は報告（041 で文を書き換える）．
    - 画像の形式・コンパイルの経路（2026-11-24 決定．`orig.pptx` の数式も PNG で，研究者はその見た目を良しとした）：
      - 1 式ずつ `standalone` クラス（`amsmath` ，書体は TeX の既定の Computer Modern）の文書を `lualatex` でコンパイルして PDF にし，`pdftocairo -png -transp -r 600` で背景透過の PNG に変換．コマンドのパス・解像度は設定または CLI 引数で与える．
      - 画像の表示の大きさは，TeX の箱の寸法（高さ・深さ・幅，pt）を本文の文字の大きさとの比で換算して決め，深さからベースラインをそろえる．
      - 生成した PNG は式の文字列（とプリアンブル・解像度）のハッシュをキーに `outputs/slides_math/` （Git 管理外）にキャッシュ．pptx に埋め込まれるため，PNG 自体はコミットしない．
      - 画像の代替テキストに元の LaTeX を入れる（照合用）．
    - 画像の大きさは本文の文字の大きさに合わせ（ベースラインの位置をそろえる），画像はビルドごとに式の文字列から決定的に生成（同じ式なら同じ画像）．TeX が見つからない・コンパイルに失敗したら明示的なエラーで停止（平文への黙ったフォールバックはしない）．
  - 対象：本文・表のセル・強調枠のすべての数式と記号（$L$ ，$\cos(\vec{d}_i, \vec{q})$ ，$\mu_{\text{same}}$ ，$\Delta S$ ，$\text{idf}(t) = \log \frac{N}{\text{df}(t)}$ ，$X = U \Sigma V^\top$ ，$K$ ，$k$ 等）．既存の `_{x}` ・ `^{x}` の独自記法と，`‖` ・ `Σ` ・ `⊤` 等を平文で並べた式は廃止し，`$...$` に書き換える（数式でない日本語の文はそのまま）．
  - 発表者ノートは平文のまま（PowerPoint のノートは数式を表示できないため）．
  - 図（`scripts/plot_slide_figures.py` ， `scripts/plot_results.py` ， `scripts/plot_d_k_sweep.py` が描く `presentation/figures/` の図）の軸・凡例・注の記号は Matplotlib の mathtext（`$L$` 等）で書く．図の数値は変えない．
- **定義の一覧の組版（研究者の指示：コロンの前後の行頭をそろえる）**
  - 「用語：説明」の形の箇条書きを，罫線なしの 2 列の表（左列：用語・記号，右列：説明，各列の行頭をそろえる）で描くブロックを追加（例：`deflist: [[用語, 説明], ...]` ）．行の区切りは薄い横線 1 本または行間のみ（研究者に見せて調整する前提の初版）．
  - 定義のスライド（現行の Slide 2，6，9，12，15）と，同じ形の箇条書きがある他のスライドに適用．
- **確認のスクリプトの強化（`scripts/check_slides.py` ）**
  - 現行はスライドの数値を出典の数値の集合（トークン単位）と照合しているため，同じ表の隣の値と入れ替わっても通る．037 の照合の Layer 1 と同様に，スライドの表の各セルと本文の数値を「どの指標・どの条件（特徴量，$L$ ，分割，間引き，推論方式）の値か」まで出典の CSV・storyline の表の該当セルと対応付けて照合する．
    - 対応付けは `slides_content.yaml` 側に明示（例：表の各行・列に指標と条件のキーを持たせる，本文の数値に出典の表のセルを指すキーを付ける）．対応付けのない数値は従来のトークン照合とし，その件数を報告．
  - 数式の検査：`$` の対応が取れていること，ビルド後の pptx に平文の `_` ・ `^` ・ `\` の残りがないこと，各数式が画像として置かれていること（画像の代替テキストに元の LaTeX を入れて照合）．
- **`.gitignore`** ：発表資料の節に `presentation/~$*` （PowerPoint 等が開いている間に作る一時的なロックファイル）を追加．
- `presentation/README.md` の「スライドの書き方」と再現コマンドに，数式の書き方（`$...$` ，ビルドにローカルの TeX が必要，PowerPoint 上では式を編集しない）と定義の一覧のブロックを追記．

## Target File Paths

- `pyproject.toml` ， `uv.lock` （数式の画像化に依存の追加が必要な場合のみ．OMML 用に追加した依存があれば外す）
- `scripts/build_slides.py` ， `scripts/check_slides.py`
- `scripts/plot_slide_figures.py` ， `scripts/plot_results.py` ， `scripts/plot_d_k_sweep.py` （図の記号の表記のみ）と再生成した `presentation/figures/` の図
- `presentation/slides_content.yaml` ， `presentation/slides.pptx`
- `presentation/README.md`
- `.gitignore`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 文言・数値・構成を変えない（記法の書き換えのみ）．新しい数値・主張を加えない．
- `presentation/orig.*` を変更・ステージしない．個人名を出力・ログ・報告に書かない．
- `src/` ・ `configs/` ・ storyline ・ qa を変更しない．図のスクリプトは表記以外を変えない．
- pptx を開いて確認する場合は Microsoft PowerPoint を明示して開く（`open -a "Microsoft PowerPoint"` 等）．Keynote は使わない．開いた文書は確認後に閉じる（他の文書は閉じない）．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/build_slides.py` が完走．画像にした数式の個数と，数式を持つスライドの一覧を報告．2 回の実行でテキスト・数式の画像（バイト列またはハッシュ）・図の参照が同一．
- [ ] TeX のコマンドを存在しないパスにすると明示的なエラーで停止することを確認．
- [ ] PowerPoint で開いて数式の画像が本文の行とそろって表示されること（Slide 2 ・ 12 ・ 15 の数式を含む），定義の一覧の行頭がそろうことを，PowerPoint から書き出した PDF（`presentation/*.pdf` は Git 管理外）の画像で確認して報告（書き出せなければ理由と，XML から確認した内容）．
- [ ] `uv run python scripts/check_slides.py` が終了コード 0．指標・条件まで対応付けて照合した数値の件数とトークン照合のみの件数を報告．表の 2 つのセルの値を入れ替えた一時的な内容で実行して終了コード 1 になることを確認（確認後に元に戻す）．
- [ ] 平文の `_` ・ `^` ・ `\` の残りが 0 件．
- [ ] 再生成した図の描画値が変わっていない（描画前の数値で比較）．
- [ ] `git check-ignore -v "presentation/~\$slides.pptx"` でパターンが効くこと．
