# Research Context

This file is the research specification of the project. It is imported by `CLAUDE.md`, so every session and sub-agent loads it. The Orchestrator owns and updates it as the research design evolves.

## Research Overview

- **Title**: 青空文庫の文体・表現類似度に基づく著者識別システム
- **Research Question / Hypothesis**: 意味内容（トピック・固有名詞）に依存する語彙特徴（内容語 TF-IDF 等）は未知作品（作品間分割）で著者識別精度が大きく低下する一方，文字 N-gram・句読点統計・品詞 N-gram・機能語頻度といった文体特徴（Stylometry）に基づく類似度は，作品が異なっても同一著者の文章間で高い値を保ち，頑健な著者推定を可能にするという仮説．併せて，判定に必要な最小テキスト長（チャンク長），類似度算出式（コサイン係数・Jaccard 係数・Dice 係数），推論方式（重心法・ $k$-NN 法）の最適な組合せの特定．
- **Proposed Approach / Contribution**: 配布サンプルコーパスのうち複数作品を持つ 4 作家（夏目漱石・太宰治・寺田寅彦・田山花袋，計 9 作品）を対象に，文境界を保持した可変長チャンク（ $L \in \{100, 500, 1000\}$ 文字）を構築．外部の類似度計算・分類ライブラリを用いず，観点別の 4 特徴量モジュール（A：表層・文字 N-gram・文体統計，B：品詞 N-gram・機能語，C：内容語 TF-IDF，D：SVD による LSI 潜在意味ベクトル）と講義準拠の 3 類似度指標をフルスクラッチ実装．「同一作品内分割（トピック漏洩あり）」と「作品単位分割（純粋な文体評価）」の比較により文体類似度の有効性を実証．
- **Primary Technical Stack**: Python 3.13（ `uv` ），NumPy（ベクトル・行列演算および SVD ）， `fugashi-plus[unidic-lite]` （MeCab ベースの形態素解析，import 名は `fugashi` ），Matplotlib（可視化），PyYAML（実験設定）．

## Experimental Design

- **対象作家の選定根拠**：コーパス全 30 作品・25 作家のうち 2 作品以上を持つ作家は上記 4 名のみ．全員が明治後期～昭和初期の口語体（言文一致体）であり，文語体（福沢諭吉・幸田露伴等）や翻訳文学（ディケンズ等）を混ぜた場合の「時代・翻訳調」の識別という交絡を回避．
  - 夏目漱石：『坊っちゃん』『こころ』『三四郎』
  - 太宰治：『惜別』『斜陽』
  - 寺田寅彦：『映画芸術』『柿の種』
  - 田山花袋：『蒲団』『田舎教師』
- **データ分割**：擬似反復（Pseudoreplication）を避けるため，同一作品内分割（Intra-work Split，作品単位の層化ランダム分割）と作品単位分割（Inter-work / Leave-One-Work-Out Split，9 fold）の両方で評価．
- **チャンク分割**：1 行 = 1 文の構造を利用し，累積文字数が $L$ に達した文末で区切るグリーディ集約．作品末尾の $L$ 未満の端数チャンクは破棄．
- **「学習」フェーズの定義**：重み更新ではなく，Train チャンクのみからの語彙辞書・IDF・正規化パラメータの算出と，作家ごとの登録ベクトル（重心・登録チャンク一覧）の保存．Test チャンクは `transform` のみを通過．
- **類似度指標**：講義資料（第 2・3・5 回）に登場するコサイン係数・重み付き Jaccard 係数・重み付き Dice 係数に限定．
  - メイン実験：全特徴量 A～D に共通適用できるコサイン類似度で特徴量を比較．
  - サブ実験：非負ベクトルである A・B・C で 3 指標を比較．負値を含む D はコサイン専用．
- **推論方式**：重心法（Centroid）と $k$-NN 法（ $k \in \{1, 3, 5, 10, 20\}$ ）．$k$-NN は作家別上位 $k$ 平均（ `knn_mode="per_author"` ，`k_eff = min(k, len(D_a))` ）を本採用とし，全体上位 $k$ の類似度重み付き投票（ `knn_mode="global"` ）を比較用に保持．
- **主評価指標**：
  - 類似度直接評価：同一著者ペア平均 $\mu_{\text{same}}$ ，異著者ペア平均 $\mu_{\text{diff}}$ ，ギャップ $\Delta S$ ．作家間のチャンク数不均衡を除くため，4×4 作家間平均類似度行列の対角平均と非対角平均の差であるマクロ版 `delta_s_macro` を主指標とし，ペア単位のミクロ版 `delta_s_micro` を併記．
  - 著者判別評価：Accuracy，マクロ F1，混同行列，MRR．Inter-work では fold 単位のマクロ F1 が定義不能（Test が 1 作家のみ）のため，全 fold の混同行列を合算する `aggregate_confusion` （Pooled Out-of-Fold）で算出．
- **既知の交絡リスク**：同一作家でも作品（底本）ごとに漢字・かなの表記方針が異なる例（『坊っちゃん』「大分」と『三四郎』「だいぶ」）．文字表層を見る A は底本の表記方針を識別している可能性があり，考察で必ず検討．

## Technical Context

### Target Computational Stack

- **Language / Environment**: Python 3.13 / `uv` （ `.python-version` を `3.13` に固定）
- **Core Libraries**:
  - `numpy` ：TF-IDF，3 類似度指標，SVD（LSI）のフルスクラッチ行列演算
  - `fugashi-plus[unidic-lite]` ：Python 3.13 対応・型ヒント同梱の `fugashi` ドロップイン代替と同梱軽量 UniDic 辞書（OS 側の MeCab 導入不要）
  - `matplotlib` ：混同行列，チャンク長別推移，同一著者・異著者間の類似度分布の描画
  - `pyyaml` ：実験条件・ハイパーパラメータの外部設定ファイル読み込み
- **Experiment Management / Config Tools**: `configs/default.yaml` （特徴量レジストリ・実験条件），実行ごとの `run_meta.json` （設定・CLI 引数・git 状態・コーパスハッシュ）
- **Prohibited Tools / APIs（課題 2 規約）**: `scikit-learn` 等の機械学習・類似度計算ライブラリ（ `cosine_similarity` ， `TfidfVectorizer` ， `CountVectorizer` ， `TruncatedSVD` ， `KNeighborsClassifier` ， `MinMaxScaler` ， `StandardScaler` 等）の使用禁止．形態素解析（ `fugashi` ）と基本配列演算（ `numpy` ）以外の特徴量抽出・類似度計算・推論アルゴリズムはすべて自作．
- **Computational Resource Constraints**: 標準的な CPU 環境（ノート PC，RAM 8～16GB）．GPU 不使用．Jaccard / Dice は $O(N_{\text{test}} N_{\text{train}} M)$ のため，各特徴量の次元数上限と $L = 100$ でのコサイン限定で計算量を制御．

### Data Handling Guardrails

- 生コーパス `data/raw/corpus20150426_with_titles.txt` （全 30 作品・約 100 万語の単一ファイル）を，エージェントが Read ツールや `cat` 等の全文出力で直接読み込むことは禁止（コンテキスト枯渇の防止）．
- 文字コード確認（配布仕様は Shift-JIS / `cp932` ，UTF-8 変換版の可能性あり），作品タイトル行・著者名行の抽出，仮名遣い検査，チャンク分割は，Python スクリプトによる行単位のストリーム処理または部分抽出で実行．
- 班員間のコーパス同一性は `data/processed/authors_corpus.json` の `meta.source_sha256` で照合．

### Directory Structure & Asset Placement Rules

- `plans/`: this file, `wbs.md`, and `{NNN}-{task-slug}.md` tickets. Written by the Orchestrator.
- `configs/`: 実験条件・ハイパーパラメータ・特徴量レジストリ（YAML）
- `src/`: 再利用可能なコアロジック
  - `dataset.py` ：対象作品抽出，文境界チャンク分割，Intra-work / Inter-work 分割器
  - `similarity.py` ：コサイン・Jaccard・Dice 係数のベクトル化実装
  - `classifier.py` ：重心法・ $k$-NN 法（ `per_author` / `global` ）の著者推定器
  - `evaluator.py` ：類似度直接評価（ $\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， $\Delta S$ ，4×4 作家間行列）と著者判別評価（Accuracy，マクロ F1，混同行列，MRR）
  - `features/` ：特徴量抽出パッケージ（班員 4 名が 1 モジュールずつ担当）
    - `base.py` ：抽象基底クラス `BaseFeatureExtractor` （ `fit` ， `transform` ， `fit_transform` ）
    - `char_unigram.py` ：文字 unigram 相対頻度（ベースライン）
    - `stylometry_char.py` ：担当 A
    - `pos_function.py` ：担当 B
    - `word_tfidf.py` ：担当 C
    - `lsi_embedding.py` ：担当 D
- `scripts/`: 前処理・検証・実験実行・集計のエントリーポイント
- `data/`: `raw/` （読み取り専用の元コーパス）と `processed/` （抽出済みの 4 作家・9 作品）
- `outputs/`: 実験ログ（JSON / CSV），図（PNG / PDF），レポート．git 管理外
- `presentation/`: 発表資料・質疑応答対策ドキュメント，発表に使う図のコピー
- `docs/`: 講義資料・配布資料（読み取り専用の参照資料）

### Domain Glossary & Mathematical Notation

- $A = \{a_1, a_2, a_3, a_4\}$ (`authors`): 対象 4 作家の集合
- $L \in \{100, 500, 1000\}$ (`target_chunk_chars`): チャンク分割の目標累積文字数
- $N$ (`num_train_chunks`): Train チャンク総数
- $M$ (`vocab_size` / `feature_dim`, config では `max_features` が上限): Train データから構築した語彙（特徴量空間）の次元数
- $D_a$ : 作家 $a$ の Train チャンク集合
- $\text{tf}(t, d)$ (`tf_matrix`): チャンク $d$ における特徴 $t$ の相対出現頻度
- $\text{df}(t)$ (`doc_freq`): Train チャンクのうち特徴 $t$ が 1 回以上出現する数（Test は含めない）
- $\text{idf}(t) = \log \frac{N}{\text{df}(t)} + 1$ (`idf_vector`): 平滑化付き逆文書頻度（ `smooth_idf=False` で講義定義の $\log \frac{N}{\text{df}(t)}$ ）
- $K \in \{8, 16, 32, 64\}$ (`n_components`, 既定 32): LSI の潜在次元数
- $\vec{d}_i, \vec{q} \in \mathbb{R}^M$ (`train_vectors`, `query_vector`): 登録チャンクと未知入力チャンクの特徴量ベクトル
- $\cos(\vec{d}_i, \vec{q}) = \frac{\vec{d}_i \cdot \vec{q}}{\|\vec{d}_i\|_2 \|\vec{q}\|_2}$ (`cosine_similarity`)
- $J_w(\vec{d}_i, \vec{q}) = \frac{\sum_{j=1}^M \min(d_{ij}, q_j)}{\sum_{j=1}^M \max(d_{ij}, q_j)}$ (`jaccard_similarity`): 重み付き Jaccard 係数（非負ベクトル専用）
- $D_w(\vec{d}_i, \vec{q}) = \frac{2 \sum_{j=1}^M \min(d_{ij}, q_j)}{\sum_{j=1}^M (d_{ij} + q_j)}$ (`dice_similarity`): 重み付き Dice 係数（非負ベクトル専用）．$D_w = 2 J_w / (1 + J_w)$ の単調変換
- $\mu_{\text{same}}, \mu_{\text{diff}}$ ：同一著者・異著者チャンクペアの平均類似度
- $\Delta S = \mu_{\text{same}} - \mu_{\text{diff}}$ (`delta_s_macro`, `delta_s_micro`): 類似度ギャップ
- $\epsilon = 10^{-12}$ (`similarity.eps`): ゼロ除算防止の微小項
- "Intra-work Split": 各作品のチャンクを Train / Test に分ける分割（トピック漏洩あり）
- "Inter-work Split" / "LOWO": 1 作品を丸ごと Test に回す Leave-One-Work-Out 分割
