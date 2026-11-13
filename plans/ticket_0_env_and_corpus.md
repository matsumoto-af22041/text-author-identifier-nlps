# Ticket 0: 環境構築とコーパス構造・表記の事前検証スクリプト作成（担当：システム基盤／リーダー）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- `uv` を用いた **Python 3.13** 仮想環境の初期化（`uv python pin 3.13` の実行，および `numpy`, `fugashi-plus[unidic-lite]`, `matplotlib`, `pyyaml` の導入）．
- 生コーパスファイルを行単位のストリーム処理で走査し，収録されている全 30 作品の「作品名・著者名・行数・文字数」の一覧を抽出するとともに，対象 4 作家（夏目漱石・太宰治・寺田寅彦・田山花袋，計 9 作品）のテキストにおける文字コード（`cp932` / `utf-8`）および新旧仮名遣いの状態を検査して整形済みデータを保存する前処理スクリプトの構築．

## 2. 対象ファイルパス（Target File Paths）

- `pyproject.toml`（`uv init` および `uv add` により生成・更新）
- `.python-version`（`3.13` に固定）
- `configs/default.yaml`（対象作家・作品リスト，チャンク長 $L \in \{100, 500, 1000\}$，乱数シード等の設定ファイル）
- `scripts/inspect_and_extract_corpus.py`（コーパス走査・対象 9 作品抽出スクリプト）
- `data/processed/authors_corpus.json`（抽出された 4 作家・9 作品の構造化データ出力先）

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，記載されたすべての規約（ベクトル化実装，乱数シード固定，Fail-Fast 原則，禁止ライブラリ規約）を遵守すること．
- **生データファイル（約 100 万語の単一テキストファイル `data/raw/corpus20150426_with_titles.txt`）を AI エージェントがファイル閲覧ツールや全文出力コマンドで直接読み込むことは絶対に行わないこと**．
- ファイル読み込み時は `open(..., encoding="cp932")`（または `utf-8` フォールバック）のイテレータ（`for line in f:`）で 1 行ずつストリーム処理し，空行区切りと「作品名：著者名」のヘッダ行パターンを検出して作品境界を特定すること．

## 4. 完了検証基準（Verification Criteria）

- `uv run python scripts/inspect_and_extract_corpus.py` がエラーなく完走し，全 30 作品のメタ情報サマリ（作家別の作品数一覧を含む）が標準出力に表示されること．
- 対象 4 作家・計 9 作品（夏目漱石 3 作品，太宰治 2 作品，寺田寅彦 2 作品，田山花袋 2 作品）が欠落なく抽出され，各作品の総文字数・総文数が正の値（`assert total_chars > 1000`）であることを確認すること．
