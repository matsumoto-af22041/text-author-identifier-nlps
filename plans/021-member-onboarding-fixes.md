# 021: 班員の着手前の設定・README の修正

## Expected Deliverables & Design Policy

- 目的：班員が 002・003・005 に着手する前に，`configs/default.yaml` と README の古い記述を現行のチケットに合わせ，班員が誤った仕様で実装するのを防ぐ．
- **`configs/default.yaml` の特徴量レジストリ**：各チケットの Target File Paths に記載された値に合わせる（班員のチケットでは自分のセクションの設定も成果物だが，誤誘導を防ぐため先に正しい値を入れておく）．
  - `features.A`：`params: {scaling: none, blocks: all, max_features: 1000}` ， `ablations: {scaling: [minmax], blocks: [bigram, symbol]}` （002）
  - `features.B`：`params: {token_key: surface, target_pos: [助詞, 助動詞], max_features: 1500}` ， `ablations: {token_key: [lemma], target_pos: [[助詞, 助動詞, 接続詞, 連体詞, 感動詞], [名詞, 動詞, 形容詞]]}` （003）
  - `features.D`：`params: {n_components: 32, projection: u}` ， `ablations: {n_components: [8, 16, 64], projection: [us]}` （005）
  - `features.C`：値は変更しない．
  - `description` の「Ticket N」表記を `plans/00N` の表記に改め，内容を現行の特徴量の説明（A は文字 Bi-gram・記号の相対頻度，B は品詞 2-gram・付属語の相対頻度等）に合わせる．
- **README の修正**：
  - 「班員向け TODO チェックリスト」の Step 1 と「環境構築とデータ準備手順」の Step 2 に，前処理の実際のコマンド `uv run python scripts/inspect_and_extract_corpus.py` と，生コーパスの配置先 `data/raw/corpus20150426_with_titles.txt` ，生成物 `data/processed/authors_corpus.json` を明記．スクリプトの実際の引数・出力先はスクリプトを読んで確認し，README と一致させる．
  - Step 2 に，`configs/default.yaml` の自分の特徴量のセクションはチケットどおり設定済みで，コンストラクタ引数名をそれに合わせること，を追記．
  - Step 3 に，統合実験（間引き・Intra-work の 5 シード・作家別 MAP の平均・基準値）はリーダー側の準備（022）の完了後に実行すること，を追記．
  - 「課題の重要ルール」の使用可能ライブラリに PyYAML（実験設定の読み込み）を追加．
  - 「実装インターフェースの共通規約」の `fit` で保存するパラメータの例から「平均・標準偏差」を削除（どの特徴量でも標準化は使わない）．
  - 「実験用語の定義」の `delta_s_macro` の「（主指標）」を，$\Delta S$ のマクロ版とミクロ版のうちの主であることが分かる表記に修正（特徴量間比較の主指標は作家別 MAP の平均）．
- 上記以外の記述は変更しない．

## Target File Paths

- `configs/default.yaml` （ `features` セクションのみ）
- `README.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `scripts/` のコードは変更しない．`plans/` は変更しない（ `plans/wbs.md` の 021 の Status 欄を除く）．
- `data/raw/corpus20150426_with_titles.txt` を Read ツールや全文出力で読まない．
- 体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペース．見出しに新たな番号を付けない．
- 個人名・メールアドレス・ローカルの絶対パスを書かない．
- 班員の実装方法（使うツール等）を制限する記述は書かない．
- コミットしないこと．

## Verification Criteria

- [ ] `uv run python -c "import yaml; c = yaml.safe_load(open('configs/default.yaml', encoding='utf-8')); print({k: (v['params'], v['ablations']) for k, v in c['features'].items()})"` の出力をそのまま報告し，A・B・D がチケットの値と一致すること．
- [ ] 既存の実験スクリプトが設定を読めることの確認：`uv run python scripts/run_experiments.py --config configs/default.yaml --features base --chunk-chars 1000 --splits inter --run-name verify021` が完走（出力は確認後に `outputs/verify021` ごと削除）．
- [ ] `--ablation` の展開で B の `target_pos` （リストのリスト）が条件として扱えるかを，`scripts/run_experiments.py` のアブレーション展開部分のコードを読んで報告（実行は不要．扱えない場合は不具合の箇所を報告し，修正は 022 で行う）．
- [ ] README の修正箇所の一覧（行番号，修正前の要旨，修正後の要旨）と，前処理コマンドがスクリプトの実際の引数・出力先と一致することの根拠を報告．
- [ ] `git diff --stat` で変更ファイルが Target File Paths（と `plans/wbs.md` の Status 欄）の範囲内であること．
