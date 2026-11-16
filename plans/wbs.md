# WBS: 青空文庫の文体・表現類似度に基づく著者識別システム

## Roadmap

- **Phase 0：環境とコーパス**：Python 3.13 + `uv` 環境の構築，生コーパスのストリーム走査による対象 4 作家・9 作品の抽出と表記検査
- **Phase 1：共通システム基盤**：チャンク分割，Intra-work / Inter-work 分割器，3 類似度指標，重心法・ $k$-NN 推論器，2 段階評価器，特徴量抽出器の抽象基底クラス
- **Phase 2：特徴量モジュール**：班員 4 名が 1 モジュールずつ担当する特徴量 A～D の実装（相互に独立，並行開発可能だが Executor への委譲は 1 件ずつ）
- **Phase 3：統合実験**：全条件の一括実験，アブレーション，表・図・レポートの自動生成
- **Phase 4：発表・質疑応答対策**：想定質問への定量的根拠の整備と補助検証スクリプト

チケット番号は旧来の「Ticket 0～6」と下 1 桁が一致（README・ `configs/default.yaml` ・ `presentation/` の「Ticket N」表記は `plans/00N-*.md` に対応）．旧「Ticket QA-1」は 007 に対応．

## Tasks

| ID | Ticket | Phase | Depends On | Status |
|---|---|---|---|---|
| 000 | [env-and-corpus](./000-env-and-corpus.md) | 0 | - | Done |
| 001 | [core-pipeline](./001-core-pipeline.md) | 1 | 000 | Done |
| 002 | [feature-a-stylometry](./002-feature-a-stylometry.md) | 2 | 001 | Todo |
| 003 | [feature-b-pos-function](./003-feature-b-pos-function.md) | 2 | 001 | Todo |
| 004 | [feature-c-word-tfidf](./004-feature-c-word-tfidf.md) | 2 | 001 | Todo |
| 005 | [feature-d-lsi](./005-feature-d-lsi.md) | 2 | 001 | Todo |
| 006 | [integration-and-eval](./006-integration-and-eval.md) | 3 | 002, 003, 004, 005 | In Progress |
| 007 | [qa-verification](./007-qa-verification.md) | 4 | 001, 006 | Todo |

## Review Notes

- **000 / 001**：全 Verification Criteria の合格を確認済み．
  - 000：全 30 作品のメタ情報出力，4 作家・9 作品の抽出（最小は『映画芸術』の 26,749 字）
  - 001：3 類似度の対角誤差最大 $1.2 \times 10^{-13}$ ，Train = Test で全推論器の Accuracy $= 1.0$ ， `seed=42` で分割インデックス完全一致
  - 再現コマンド： `uv run python scripts/inspect_and_extract_corpus.py` ， `uv run python scripts/verify_core_pipeline.py`
- **006**：依存チケット 002～005 の完了前に `scripts/run_experiments.py` と `scripts/plot_results.py` を先行実装（ベースライン `base` で動作）．002～005 完了後に Verification Criteria を実施して `Done` 判定．
- **検証で判明し設計に反映済みの事項**：
  - $\Delta S$ の主指標をマクロ版 `delta_s_macro` に決定（006）
  - Inter-work のマクロ F1 を `aggregate_confusion` で算出（006）
  - Jaccard / Dice の計算量対策として各特徴量の `max_features` 上限と $L = 100$ でのコサイン限定を採用（002～004，006）
  - 底本の表記方針による交絡の考察を 002 に追加
  - `outputs/` を `.gitignore` に追加済み
- **発表資料側の未修正事項（007 で対応）**：Q3（チャンク長のばらつき），Q4（端数破棄率），Q11（ $k$ 増大時の漸近先）の記述修正
