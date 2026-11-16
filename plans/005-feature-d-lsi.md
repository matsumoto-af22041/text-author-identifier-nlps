# 005: 特徴量 D（LSI / SVD による潜在意味ベクトル）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー D
- 第 5 回講義資料の潜在的意味解析（LSI）と第 3 回講義資料の分散表現に基づき，高次元スパースな重み行列 $X \in \mathbb{R}^{N \times M}$ に `numpy.linalg.svd` による Truncated SVD を適用し， $K$ 次元（既定 $K = 32$ ，検証レンジ $K \in \{8, 16, 32, 64\}$ ）の密ベクトルへ射影する `LSIEmbeddingExtractor(BaseFeatureExtractor)` ．
- 表層的な単語の一致がなくとも，共起関係から潜在的な文体・意味因子を抽出できるかを検証する発展的モジュール．
- 出力に負値を含むため，クラス属性 `non_negative = False` を明示的に宣言し，類似度はコサイン専用（ `evaluate_fold` が Jaccard / Dice との組合せを拒否）．

## Target File Paths

- `src/features/lsi_embedding.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `TruncatedSVD` 等は禁止． `numpy.linalg.svd` のみで実装．
- `fit` で Train の重み行列 $X_{\text{train}} \in \mathbb{R}^{N \times M}$ に $X_{\text{train}} = U \Sigma V^\top$ （ `np.linalg.svd(X_train, full_matrices=False)` ）を適用し，上位 $K$ 個の右特異ベクトル $V_K \in \mathbb{R}^{M \times K}$ を保存．
- `transform` は保存した射影行列による線形変換 $Z = X V_K \in \mathbb{R}^{N_{\text{eval}} \times K}$ のみ．Test を含めた SVD の再計算（データ漏洩）は禁止．
- 潜在次元数の引数名は `configs/default.yaml` の `features.D.params` / `ablations` に合わせて `n_components` とすること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性を自動検査．

## Verification Criteria

- [ ] $K = 32$ で `transform` の出力形状が `(N_samples, K)` ．
- [ ] 射影行列の列ベクトルが直交正規であることを `assert np.allclose(V_k.T @ V_k, np.eye(K), atol=1e-5)` で確認．
