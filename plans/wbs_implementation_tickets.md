# 実装タスク指示書（WBS）インデックス：青空文庫の著者識別システム

## 概要とアーキテクチャ方針

本ドキュメントは，研究計画書（`plans/project_consultation_plan.md`）に基づき，実装担当エージェント（Research Executor）および班員 4 名が自律的に実装と検証を遂行するためのタスク指示書（WBS / Issue チケット群）の全体インデックスである．各チケットの詳細仕様は `plans/` 配下の個別ファイルを参照すること．

- **全体設計**：リーダーが「共通パイプライン（データ分割・類似度計算・推論・評価）」を構築し，班員 4 名が共通インターフェース `BaseFeatureExtractor` を継承した「4 つの特徴量抽出モジュール（A〜D）」を 1 人 1 モジュールずつ実装・統合するプラグイン型アーキテクチャ．
- **厳守すべき共通制約（全チケット共通）**：
  1. 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，記載されたすべての規約（ベクトル化実装，乱数シード固定，Fail-Fast 原則，禁止ライブラリ規約）を遵守すること．
  2. **生データファイル（約 100 万語の単一テキストファイル）を AI エージェントがファイル閲覧ツールや全文出力コマンドで直接読み込むことは絶対に行わないこと**．必ず Python スクリプトの行単位ストリーム処理を介して扱うこと．
  3. `scikit-learn` 等の外部機械学習・類似度計算ライブラリの使用は一切禁止．Python 3.13 環境下で `numpy` と `fugashi-plus[unidic-lite]`（インポート名は `fugashi`）のみを用いてフルスクラッチで実装すること．

---

## チケット一覧（`plans/`）

| チケット番号 | ファイル名 | 担当 | 概要と対象モジュール |
| :--- | :--- | :--- | :--- |
| **Ticket 0** | `ticket_0_env_and_corpus.md` | システム基盤／リーダー | Python 3.13 + `fugashi-plus[unidic-lite]` 環境構築とコーパス構造・表記の事前検証スクリプト作成 |
| **Ticket 1** | `ticket_1_core_pipeline.md` | システム基盤／リーダー | 共通システム基盤（チャンク分割・類似度計算 3 種・推論 Centroid / $k$-NN・評価器）の実装 |
| **Ticket 2** | `ticket_2_feature_a_stylometry.md` | メンバー A | 【担当 A】表層・文字 N-gram・文体統計量と Min-Max 正規化モジュールの実装 |
| **Ticket 3** | `ticket_3_feature_b_pos_function.md` | メンバー B | 【担当 B】統語・品詞 N-gram・機能語（表層形 `surface` デフォルト）モジュールの実装 |
| **Ticket 4** | `ticket_4_feature_c_word_tfidf.md` | メンバー C | 【担当 C】語彙・内容語 TF-IDF（語彙素 `lemma` 固定）モジュールの実装 |
| **Ticket 5** | `ticket_5_feature_d_lsi.md` | メンバー D | 【担当 D】特異値分解（LSI / SVD，デフォルト $K=32$）による潜在意味・分散表現モジュールの実装 |
| **Ticket 6** | `ticket_6_integration_and_eval.md` | 全員／統合はリーダー | 全体統合実験スクリプトと可視化・レポート出力の実装 |
| **Ticket QA-1** | `qa_verification_instructions.md` | 全員／統合はリーダー | 発表・質疑応答対策ドキュメント作成および補助検証スクリプトの実装 |
