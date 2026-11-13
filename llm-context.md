# Research Technical Context

## Target Computational Stack

- **Language / Environment**: Python 3.13 / `uv` による仮想環境・依存関係管理（`.python-version` を `3.13` に固定し OS 非依存の再現性を確保）
- **Core Libraries**:
  - `numpy`: ベクトル空間モデル，TF-IDF，コサイン類似度・Jaccard 係数・Dice 係数，特異値分解（SVD / LSI）のフルスクラッチ行列演算
  - `fugashi-plus[unidic-lite]`: Cython 3.0 ベースの MeCab ラッパー（Python 3.13 対応・型ヒント同梱の `fugashi` ドロップイン代替，インポート名は `import fugashi`）および同梱軽量 UniDic 辞書（OS 側への MeCab 本体インストール不要）
  - `matplotlib`: 混同行列，チャンク長別精度推移，同一著者・異著者間の類似度分布（ヒストグラム・箱ひげ図）の描画
  - `pyyaml`: 実験条件およびハイパーパラメータの外部設定ファイル読み込み
- **Prohibited Tools / APIs（課題 2 規約に基づく禁止事項）**:
  - `scikit-learn` 等の機械学習・類似度計算ライブラリ（`cosine_similarity`, `TfidfVectorizer`, `CountVectorizer`, `TruncatedSVD`, `KNeighborsClassifier`, `SVC`, `MultinomialNB` 等）の使用は一切禁止．前処理の形態素解析（`fugashi-plus` / `import fugashi`）と基本配列演算（`numpy`）のみを使用し，特徴量抽出と類似度計算・推論アルゴリズムはすべて自作すること．
- **Computational Resource Constraints**: 標準的な CPU 環境（ノート PC 等，RAM 8GB〜16GB）で全実験が数分以内に完結する軽量・ベクトル化実装．

## Critical Data Handling Guardrails（巨大コーパスファイルの取り扱い厳守事項）

- **AI エージェントによる直接読み込みの完全禁止**:
  - 生データファイル（1 つのテキストファイルに全 30 作品・約 100 万語が結合されたコーパス）を，AI エージェントがファイル閲覧ツールや全文出力コマンド（`cat` 等）で直接読み込むことは，コンテキストウィンドウの枯渇（トークン爆発）を招くため**絶対に行わないこと**．
  - データの文字コード確認（Shift-JIS / `cp932` または UTF-8），作品タイトル行・著者名行の抽出，新旧仮名遣いの検査，およびチャンク分割処理は，必ず Python スクリプトによる行単位のストリーム処理（イテレータ処理）または部分抽出コマンドを介して実行すること．

## Directory Structure & Asset Placement Rules

- `configs/`: 実験条件・ハイパーパラメータ設定ファイル（YAML 形式）
- `src/`: 再利用可能なコアロジック（システム基盤および 4 名の班員が分担する特徴量モジュール）
  - `dataset.py`: コーパスからの対象作品抽出，文境界チャンク分割，作品内（Intra-work）/ 作品間（Inter-work）データ分割器
  - `similarity.py`: 講義準拠の類似度計算関数（コサイン類似度，Jaccard 係数，Dice 係数）のベクトル化実装
  - `classifier.py`: 類似度に基づく著者推定器（重心法 Centroid および $k$-近傍類似度法 $k$-NN：$k \in \{1, 3, 5, 10, 20\}$，作家別上位 $k$ 平均 `per_author` をデフォルトとし全体上位 $k$ 重み付き投票 `global` も選択可能）
  - `evaluator.py`: 類似度直接評価指標（同一著者ペア平均類似度 $\mu_{\text{same}}$，異著者ペア平均類似度 $\mu_{\text{diff}}$，類似度ギャップ $\Delta S = \mu_{\text{same}} - \mu_{\text{diff}}$，4×4 作家間平均類似度マトリクス）および著者判別指標（Accuracy，マクロ F1，混同行列，MRR）の算出・可視化モジュール
  - `features/`: 班員 4 名が 1 モジュールずつ担当する特徴量抽出パッケージ
    - `base.py`: 抽象基底クラス `BaseFeatureExtractor`（`fit`, `transform`, `fit_transform` のインターフェース定義）
    - `stylometry_char.py`: 【担当 A】文字 N-gram（Bi-gram / Tri-gram）および文体統計量（句読点率・字種比率・平均文長）と Train 統計量に基づく正規化（非負制約を満たす Min-Max 正規化をデフォルトとし，Z-score 標準化・未正規化も切り替え可能）
    - `pos_function.py`: 【担当 B】形態素解析に基づく品詞 N-gram（POS Bi-gram）および機能語（助詞・助動詞・接続詞・連体詞・感動詞・補助記号）の相対頻度（活用形の癖を保持する `surface` 表層形をデフォルトとし，`lemma` 基本形も切り替え可能）
    - `word_tfidf.py`: 【担当 C】内容語（名詞・動詞・形容詞）の語彙素（`lemma`）抽出による表記揺れ吸収とフルスクラッチ TF-IDF 重み計算
    - `lsi_embedding.py`: 【担当 D】文書-単語共起行列に対する特異値分解（SVD / LSI，$K \in \{8, 16, 32, 64\}$，デフォルト $K=32$）による低次元密ベクトル表現（コサイン類似度専用）
- `scripts/`: 実験実行・データ前処理・結果集計用のエントリーポイントスクリプト群
- `data/`: `raw/`（読み取り専用の元コーパス配置場所）および `processed/`（スクリプトで抽出した対象 4 作家・9 作品の整形済みデータ）
- `outputs/`: 実験ログ（JSON / CSV），混同行列や類似度分布のグラフ画像（PNG / PDF）
- `plans/`: 実験計画および実装タスク指示書（WBS）

## Domain Glossary & Mathematical Notation

- $A = \{a_1, a_2, a_3, a_4\}$ (`authors`): 対象とする 4 名の著者集合（夏目漱石，太宰治，寺田寅彦，田山花袋）
- $L \in \{100, 500, 1000\}$ (`target_chunk_chars`): チャンク分割時の目標累積文字数（文境界を保持して $L$ 文字以上に達した文末で区切る）
- $N$ (`num_train_chunks`): 事前登録（Train）データにおける総チャンク数
- $M$ (`vocab_size` / `feature_dim`): 事前登録データから構築された語彙集合（または特徴量空間）の次元数
- $\text{tf}(t, d)$ (`tf_matrix`): チャンク $d$ における特徴（単語・N-gram）$t$ の出現頻度（または総数で割った相対頻度）
- $\text{df}(t)$ (`doc_freq`): 事前登録チャンク群 $N$ のうち，特徴 $t$ が 1 回以上出現するチャンク数（Test データは一切含めない）
- $\text{idf}(t) = \log \frac{N}{\text{df}(t)} + 1$ (`idf_vector`): 逆文書頻度（ゼロ除算および全チャンク出現語が消えるのを防ぐ平滑化付き定義，または講義定義の $\log(N / \text{df}(t))$ を設定で切り替え可能とする）
- $\vec{d}_i, \vec{q} \in \mathbb{R}^M$ (`train_vectors`, `query_vector`): 登録チャンクおよび未知入力チャンクの特徴量ベクトル表現
- $\cos(\vec{d}_i, \vec{q})$ (`cosine_similarity`): コサイン類似度 $\frac{\vec{d}_i \cdot \vec{q}}{\|\vec{d}_i\|_2 \|\vec{q}\|_2}$
- $J_w(\vec{d}_i, \vec{q})$ (`jaccard_similarity`): 重み付き Jaccard 係数 $\frac{\sum_{j=1}^M \min(d_{ij}, q_j)}{\sum_{j=1}^M \max(d_{ij}, q_j)}$（非負ベクトル対象）
- $D_w(\vec{d}_i, \vec{q})$ (`dice_similarity`): 重み付き Dice 係数 $\frac{2 \sum_{j=1}^M \min(d_{ij}, q_j)}{\sum_{j=1}^M (d_{ij} + q_j)}$（非負ベクトル対象）
