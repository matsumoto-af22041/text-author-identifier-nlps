# 000: 環境構築とコーパス構造・表記の事前検証スクリプト

## Expected Deliverables & Design Policy

- 担当：システム基盤（リーダー）
- `uv` による Python 3.13 仮想環境の初期化（ `uv python pin 3.13` ， `numpy` ， `fugashi-plus[unidic-lite]` ， `matplotlib` ， `pyyaml` の導入）．
- 生コーパスを行単位のストリーム処理で走査し，全 30 作品の「作品名・著者名・行数・文字数」一覧を抽出する前処理スクリプト．
- 対象 4 作家・9 作品について，文字コード（ `cp932` / `utf-8` ）と新旧仮名遣いの状態を検査し，整形済みデータを保存．

## Target File Paths

- `pyproject.toml` （ `uv init` ・ `uv add` による生成・更新）
- `.python-version` （ `3.13` に固定）
- `configs/default.yaml` （対象作家・作品リスト，チャンク長，乱数シード，仮名遣い検査パターン）
- `scripts/inspect_and_extract_corpus.py`
- `data/processed/authors_corpus.json` （4 作家・9 作品の構造化データ出力先）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `data/raw/corpus20150426_with_titles.txt` を Read ツールや全文出力コマンドで直接読み込まないこと（ `plans/research-context.md` の Data Handling Guardrails）．
- 読み込みは `cp932` → `utf-8` の順に全行を厳密デコードして判定し，イテレータで 1 行ずつストリーム処理．空行区切りと作品名行・著者名行のパターンから作品境界を特定．
- 対象作品に旧仮名遣い判定が出た場合は処理を停止（Fail-Fast）．

## Verification Criteria

- [ ] `uv run python scripts/inspect_and_extract_corpus.py` が完走し，全 30 作品のメタ情報サマリ（作家別の作品数一覧を含む）が標準出力に表示される．
- [ ] 対象 4 作家・計 9 作品（夏目漱石 3，太宰治 2，寺田寅彦 2，田山花袋 2）が欠落なく抽出され，各作品の総文字数・総文数が正（ `assert total_chars > 1000` ）．
