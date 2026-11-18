# 005: 特徴量 D（LSI / SVD による潜在意味ベクトル）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー D（004 の完了後に着手）
- 第 5 回講義資料 p.36-37 の潜在的意味解析（LSI）に基づき，文書-単語行列を特異値分解して $K$ 次元（既定 $K = 32$ ，検証レンジ $K \in \{8, 16, 32, 64\}$ ）のトピック空間へ射影する `LSIEmbeddingExtractor(BaseFeatureExtractor)` ．
- **入力行列**：第 5 回 p.25 のベクトル空間モデルの重み（tf・idf）による文書-単語行列．004 の `WordTFIDFExtractor` （既定パラメータ）を内部で生成し，Train のみで fit した TF-IDF 行列 $X_{\text{train}} \in \mathbb{R}^{N \times M}$ を用いる．
- **分解（講義 p.37 の図どおり）**： $X_{\text{train}} = U \Sigma V^\top$ ．中心化（平均の引き算）は講義にないため行わない．上位 $K$ 個の特異値 $\Sigma_K$ と右特異ベクトル $V_K \in \mathbb{R}^{M \times K}$ を保存．
- **射影（講義 p.37「検索質問を $V$ によりトピックの重みベクトルに変換し， $U$ を用いて文書と比較する」）**：
  - 既定 `projection="u"` ： $\hat{\vec{q}} = \Sigma_K^{-1} V_K^\top \vec{q}$ ．行列形式では $Z = X V_K \Sigma_K^{-1}$ ．Train 文書を射影すると $U_K$ の行と一致するため，Test の照会ベクトルと Train 文書を講義どおり $U$ の空間で比較できる．この式は講義スライドに明示がないが，講義 p.37 の $X = U \Sigma V^\top$ の両辺に右から $V_K \Sigma_K^{-1}$ を掛けると $U_K = X V_K \Sigma_K^{-1}$ となることから導出でき，検索質問にも同じ変換を適用する．
  - アブレーション `projection="us"` ： $Z = X V_K$ （Train 文書では $U_K \Sigma_K$ ）．特異値で各トピック次元を重み付けした比較（独自設計）．
- 出力に負値を含むため，クラス属性 `non_negative = False` を明示的に宣言（類似度ヒストグラムの範囲が $[-1, 1]$ になる）．
- 講義第 3 回の「分散表現」（word2vec 等のニューラル手法）とは別物．発表で LSI を「分散表現」と呼ばないこと（講義第 5 回 p.41 の「密ベクトル」とは呼んでよい）．

## Target File Paths

- `src/features/lsi_embedding.py`
- `configs/default.yaml` （ `features.D` セクションのみ： `params: {n_components: 32, projection: u}` ， `ablations: {n_components: [8, 16, 64], projection: [us]}` ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `TruncatedSVD` 等は禁止． `numpy.linalg.svd(X_train, full_matrices=False)` のみで実装．
- 004 の `WordTFIDFExtractor` を変更せずに再利用（コピー実装は禁止）．内部の TF-IDF 抽出器も Train のみで fit．
- `transform` は保存した $V_K$ ， $\Sigma_K$ による線形変換のみ．Test を含めた SVD の再計算（データ漏洩）は禁止．
- $K > \min(N, M)$ ，または $\sigma_K \leq 10^{-10} \sigma_1$ （ランク不足で $\Sigma_K^{-1}$ が不安定）の場合は `ValueError` で停止（Fail-Fast）．暗黙に $K$ を切り詰めないこと．
- SVD の符号の不定性（ $u_k, v_k$ の同時反転）はコサイン類似度に影響しないが，再現性確認のため各 $v_k$ の絶対値最大成分が正になるよう符号を正規化．
- 潜在次元数・射影方式の引数名は `configs/default.yaml` の `features.D.params` / `ablations` に合わせて `n_components` ， `projection` とすること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性を自動検査．

## Verification Criteria

- [ ] $K = 32$ で `transform` の出力形状が `(N_samples, K)` ．
- [ ] 射影行列の列ベクトルが直交正規であることを `assert np.allclose(V_k.T @ V_k, np.eye(K), atol=1e-5)` で確認．
- [ ] `projection="u"` で Train 自身を `transform` した結果が $U_K$ と一致（ `np.allclose(Z_train, U_k, atol=1e-8)` ）．講義の「 $U$ を用いて比較」との対応の数値確認．
- [ ] `projection="us"` で Train 自身を `transform` した結果が $U_K \Sigma_K$ と一致．
- [ ] Train チャンク数より大きい $K$ を指定すると `ValueError` で停止．
- [ ] 同一入力で 2 回 fit・transform した出力が完全一致（符号正規化の確認）．
