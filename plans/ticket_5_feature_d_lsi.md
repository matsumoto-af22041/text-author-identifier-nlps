# Ticket 5: 【担当 D】特異値分解（LSI / SVD）による潜在意味・分散表現モジュールの実装（担当：メンバー D）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- 第 5 回講義資料の「潜在的意味解析（LSI：Latent Semantic Indexing）」および第 3 回講義資料の「分散表現（埋め込み表現）」に基づき，高次元スパースな単語（または機能語＋文体）重み行列 $X \in \mathbb{R}^{N \times M}$ に対して `numpy.linalg.svd` を用いた特異値分解（Truncated SVD）を適用し，$K$ 次元（**デフォルト $K = 32$**，検証レンジ $K \in \{8, 16, 32, 64\}$）の低次元密ベクトル空間へ射影する `LSIEmbeddingExtractor(BaseFeatureExtractor)` の実装．
- 表面的な単語の一致がなくても，共起関係から背後の潜在的な文体・意味因子を抽出できるかを検証する発展的モジュール（出力に負の値を含むため，類似度関数は **コサイン類似度専用** とする）．

## 2. 対象ファイルパス（Target File Paths）

- `src/features/lsi_embedding.py`

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，すべての規約を遵守すること．
- `scikit-learn` の `TruncatedSVD` 等は絶対に使用せず，`numpy.linalg.svd` のみを用いてフルスクラッチ実装すること．
- `fit(train_texts)` において，Train データの重み行列 $X_{\text{train}} \in \mathbb{R}^{N \times M}$ に対して特異値分解 $X_{\text{train}} = U \Sigma V^T$（`np.linalg.svd(X_train, full_matrices=False)`）を実行し，上位 $K$ 個の右特異ベクトル（語彙空間から潜在空間への射影行列 $V_K \in \mathbb{R}^{M \times K}$）を保存すること．
- `transform(texts)` では，入力テキストの重み行列 $X \in \mathbb{R}^{N_{\text{eval}} \times M}$ に対して保存した射影行列を掛ける線形変換 $Z = X V_K \in \mathbb{R}^{N_{\text{eval}} \times K}$ のみを行い，Test データを含めて SVD を再計算するデータ漏洩を絶対に行わないこと．

## 4. 完了検証基準（Verification Criteria）

- 指定した潜在次元数 $K$（例：$K=32$）に対して，`transform` の出力形状が `(N_samples, K)` となること，および射影行列 $V_K$ の列ベクトル同士が直交正規（$V_K^T V_K \approx I_K$）であることを `assert np.allclose(V_k.T @ V_k, np.eye(K), atol=1e-5)` で検証すること．
