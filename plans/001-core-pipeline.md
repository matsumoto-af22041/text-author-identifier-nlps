# 001: 共通システム基盤（チャンク分割・類似度計算・推論・評価）

## Expected Deliverables & Design Policy

- 担当：システム基盤（リーダー）
- **文境界保持チャンク分割とデータ分割器**：1 行 = 1 文の構造を利用し，累積文字数が $L \in \{100, 500, 1000\}$ に達した文末で区切るチャンク分割（作品末尾の端数チャンクは破棄）．Intra-work Split と Inter-work（Leave-One-Work-Out）Split の 2 分割器．
- **特徴量抽出器の抽象基底クラス**：全特徴量モジュールが準拠する `fit(train_texts) -> self` と `transform(texts) -> np.ndarray` （形状 `(N_samples, D_features)` ）のインターフェース．
- **3 類似度指標**： `numpy` のブロードキャスト・行列演算によるコサイン類似度，重み付き Jaccard 係数，重み付き Dice 係数（ $\epsilon = 10^{-12}$ と `NaN` / `Inf` アサーションを含む）．
- **推論器**：
  - 重心法（Centroid）と $k$-NN 法（ $k \in \{1, 3, 5, 10, 20\}$ ）．
  - $k$-NN は作家間のチャンク数不均衡（漱石 3 作品 vs 他 2 作品）と 4 クラス多数決の同票問題を避けるため，各作家の Train チャンク群 $D_a$ 内の上位 `k_eff = min(k, len(D_a))` 個の平均類似度で判定する `knn_mode="per_author"` を既定とし，全体上位 $k$ 個の類似度重み付き投票 `knn_mode="global"` も切り替え可能．
- **2 段階評価器（ `src/evaluator.py` ）**：Accuracy・マクロ F1・混同行列に加え，$\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， $\Delta S$ ，4×4 作家間平均類似度行列，同一著者 vs 異著者の類似度分布データの算出．

## Target File Paths

- `src/__init__.py`
- `src/dataset.py`
- `src/features/__init__.py`
- `src/features/base.py`
- `src/similarity.py`
- `src/classifier.py`
- `src/evaluator.py`
- `scripts/verify_core_pipeline.py` （Verification Criteria の検証スクリプト）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `plans/research-context.md` の Prohibited Tools / APIs を遵守し， `numpy` のみでフルスクラッチ実装．
- `src/similarity.py` でサンプル間の 2 重 `for` ループを禁止．行列積（ `X @ Y.T` ）またはメモリ上限付きのバッチブロードキャストでベクトル化．
- データ漏洩防止のため，評価パイプラインは Train チャンクのみを `fit` に渡し，Test チャンクは `transform` のみを通す構造を強制．抽出器は fold ごとに新規生成．

## Verification Criteria

- [ ] 数値的健全性：ランダム非負行列 `X` （形状 `(10, 50)` ）で `cosine_similarity(X, X)` ， `jaccard_similarity(X, X)` ， `dice_similarity(X, X)` の対角がすべて $1.0$ （許容誤差 $10^{-6}$ ），全要素が $[0, 1]$ に収まることを `assert` で検証．
- [ ] 単一バッチ過学習：恒等変換相当の抽出器で Train 自身を Test として推論した際，Centroid と $k = 1$ の $k$-NN の Accuracy が $1.0$ ．
- [ ] 乱数シード再現性： `seed=42` で Intra-work Split を 2 回実行し，Train / Test のチャンクインデックスが完全一致．
