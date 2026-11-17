# WBS: 青空文庫の文体・表現類似度に基づく著者識別システム

## Roadmap

- **Phase 0：環境とコーパス**：Python 3.13 + `uv` 環境の構築，生コーパスのストリーム走査による対象 4 作家・9 作品の抽出と表記検査
- **Phase 1：共通システム基盤**：チャンク分割，Intra-work / Inter-work 分割器，コサイン類似度，重心法・ $k$-NN 推論器，2 段階評価器，特徴量抽出器の抽象基底クラス．講義資料準拠化（008）
- **Phase 2：特徴量モジュール**：班員 4 名が 1 モジュールずつ担当する特徴量 A～D の実装（D は C の TF-IDF を入力に使うため 004 の後）．004 はリーダー担当のため Executor に委譲．002・003・005 は班員担当のため Orchestrator は委譲しない．A の仕様確定のため事前調査 010 を実施
- **Phase 3：統合実験**：全条件の一括実験，アブレーション，表・図・レポートの自動生成
- **Phase 4：発表・質疑応答対策**：発表資料の講義資料準拠化と出典明記（009），想定質問への定量的根拠の整備と補助検証スクリプト（007）

チケット番号は旧来の「Ticket 0～6」と下 1 桁が一致（README・ `configs/default.yaml` ・ `presentation/` の「Ticket N」表記は `plans/00N-*.md` に対応，009 で表記を統一）．旧「Ticket QA-1」は 007 に対応．

## Tasks

Orchestrator の委譲対象：000・001・004・006～010・012．002・003・005 は班員担当．

| ID | Ticket | Phase | Depends On | Status |
|---|---|---|---|---|
| 000 | [env-and-corpus](./000-env-and-corpus.md) | 0 | - | Done |
| 001 | [core-pipeline](./001-core-pipeline.md) | 1 | 000 | Done |
| 008 | [lecture-aligned-core](./008-lecture-aligned-core.md) | 1 | 001 | Done |
| 010 | [a-scale-investigation](./010-a-scale-investigation.md) | 2 | 001 | Done |
| 002 | [feature-a-stylometry](./002-feature-a-stylometry.md) | 2 | 001, 010 | Todo |
| 003 | [feature-b-pos-function](./003-feature-b-pos-function.md) | 2 | 001 | Todo |
| 004 | [feature-c-word-tfidf](./004-feature-c-word-tfidf.md) | 2 | 001 | Done |
| 005 | [feature-d-lsi](./005-feature-d-lsi.md) | 2 | 004 | Todo |
| 006 | [integration-and-eval](./006-integration-and-eval.md) | 3 | 002, 003, 004, 005, 008 | In Progress |
| 009 | [presentation-lecture-alignment](./009-presentation-lecture-alignment.md) | 4 | - | Done |
| 012 | [c-results-and-presentation](./012-c-results-and-presentation.md) | 4 | 004, 008, 009 | Done |
| 007 | [qa-verification](./007-qa-verification.md) | 4 | 001, 006, 009 | Todo |

## Review Notes

- **000 / 001**：全 Verification Criteria の合格を確認済み．
  - 000：全 30 作品のメタ情報出力，4 作家・9 作品の抽出（最小は『映画芸術』の 26,749 字）
  - 001：3 類似度の対角誤差最大 $1.2 \times 10^{-13}$ ，Train = Test で全推論器の Accuracy $= 1.0$ ， `seed=42` で分割インデックス完全一致
  - 再現コマンド： `uv run python scripts/inspect_and_extract_corpus.py` ， `uv run python scripts/verify_core_pipeline.py`
- **006**：依存チケット 002～005 の完了前に `scripts/run_experiments.py` と `scripts/plot_results.py` を先行実装（ベースライン `base` で動作）．002～005・008 完了後に Verification Criteria を実施して `Done` 判定．
- **講義資料準拠化（2026-11-16 の見直し）**：
  - 講義資料（第 2・3・5 回）に式がない，または講義と異なる設計を洗い出し，`plans/research-context.md` の「Lecture Grounding & References」に整理．
  - 001 の重み付き Jaccard / Dice は講義に式がなく，有無のみを見る講義版（第 2 回 p.45）も付属語では自明に区別不能で研究の主張に寄与しないため，類似度をコサインに統一して削除（008，2026-11-17 決定）．001 は `Done` のまま，修正は 008 で実施．
  - 発表の軸を「情報検索では捨てる語（第 5 回 p.11, p.14-16）が著者識別では手がかりになる」に設定し，B に内容語対照条件を追加（003，006）．
  - 001 の評価器は同点時に最小ラベル（漱石）を予測し MRR を楽観的に数えるため，同点を誤判別として扱う修正を 008 に追加．
  - 002 の Min-Max 既定採用を撤回し相対頻度に統一．字種構成比は値が 1 桁大きくコサインを支配する見込みのため類似度ベクトルから外し考察用の記述統計へ（案 C，2026-11-17 決定，010 で実測確認），004 の $\text{idf} + 1$ を講義定義に戻し，005 の射影を講義第 5 回 p.37 に合わせて $X V_K \Sigma_K^{-1}$ に変更し入力行列を 004 の TF-IDF に確定．
  - 006 の Intra-work を 5 シード化．
  - 発表資料の修正は Orchestrator の書き込み権限外（ `plans/` のみ）のため 009 として Executor に委譲．
- **010（2026-11-17 レビュー済み）**：二乗ノルム割合（平均）は旧案（字種あり）で字種 0.98～0.99，旧案＋Min-Max で Bi-gram 0.88～0.96，案 C で Bi-gram / 記号 = 0.72 / 0.28（ $L = 100$ ），0.57 / 0.43（500），0.52 / 0.48（1000）．案 C が 3 構成中で唯一，単一ブロックに支配されない．再現コマンド： `uv run python scripts/investigate_a_scale.py`
- **008（2026-11-17 レビュー済み）**：類似度をコサインのみに統一（001 の `cosine_similarity` と出力がビット一致），同点を混同行列 (C, C + 1) の同点列に計上し MRR を悲観的順位で算出．実データ $L = 1000$ で LOWO 重心法 Acc 0.551，F1 0.558，MRR 0.745（ベースライン `base` ），同点・ゼロベクトル 0 件．grep 基準は部分一致（`np.triu_indices`）を除外する単語境界つきに修正．再現コマンド： `uv run python scripts/verify_core_pipeline.py`
  - 注意：悲観的順位では `knn_mode="global"` の上位 $k$ 外の作家（スコア 0 で同点）が順位を押し下げるため，global の MRR は旧値より低下（inter knn1_global 0.722 → 0.583）．global は比較用のため，MRR の考察は `per_author` と重心法を主とする（006）．
- **004（2026-11-17 レビュー済み）**：TF・DF・IDF・TF-IDF が手計算と完全一致（ $\text{df} = N$ の語の重みは厳密に 0 ）．表記統一の根拠例を副詞（「大分」「だいぶ」）から内容語（「云う」「言う」等）に差し替え（詳細は 004 の Review Notes）．検証スクリプトは scratchpad にあり未コミットのため，012 でリポジトリに移す．
- **検証で判明し設計に反映済みの事項**：
  - $\Delta S$ の主指標をマクロ版 `delta_s_macro` に決定（006）
  - Inter-work のマクロ F1 を Pooled Out-of-Fold で算出（006）
  - 底本の表記方針による交絡の考察を 002 に追加
  - `outputs/` を `.gitignore` に追加済み
- **発表資料側の未修正事項（007 で対応）**：Q3（チャンク長のばらつき），Q4（端数破棄率），Q11（ $k$ 増大時の漸近先）の記述修正
