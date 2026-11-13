# Ticket 1: 共通システム基盤（チャンク分割・類似度計算・推論・評価）の実装（担当：システム基盤／リーダー）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- **文境界保持チャンク分割とデータ分割器**：1 行＝1 文の構造を利用し，累積文字数が目標長 $L \in \{100, 500, 1000\}$ に達した文末で区切るチャンク分割処理（作品末尾の端数チャンクは破棄）と，「同一作品内分割（Intra-work Split）」および「作品単位分割（Inter-work / Leave-One-Work-Out Split）」の 2 種類の分割器の実装．
- **特徴量抽出器の抽象基底クラス**：すべての特徴量モジュール（A〜D）が準拠すべき `fit(train_texts) -> self` および `transform(texts) -> np.ndarray`（形状：`(N_samples, D_features)`）のインターフェース定義．
- **講義準拠の 3 種類の類似度計算エンジン**：`numpy` のブロードキャスト・行列演算を用いた **コサイン類似度**，**重み付き Jaccard 係数**，**重み付き Dice 係数** のフルスクラッチ実装（ゼロ除算防止の微小項 $\epsilon = 10^{-12}$ および `NaN`/`Inf` アサーションを含む）．
- **推論器と評価パイプライン**：
  - **重心（Centroid）方式** と **$k$-近傍（$k$-NN）類似度方式**（$k \in \{1, 3, 5, 10, 20\}$）の 2 種類の著者推定器．
  - $k$-NN 方式は，作家間のチャンク数不均衡（漱石 3 作品 vs 他 2 作品）および 4 クラス多数決での同票問題を回避するため，**各作家の登録チャンク群 $D_a$ 内の上位 $k$ 個（ただし `k_eff = min(k, len(D_a))`）の平均類似度で判定する `knn_mode="per_author"` をデフォルト（本採用）** とし，全体上位 $k$ 個の類似度重み付き投票 `knn_mode="global"` も設定で切り替え可能とすること．
  - **類似度直接評価と著者判別評価の 2 段階評価器（`src/evaluator.py`）**：「文章の類似度を測り，著者を判別できるか」という問題意識に直結させるため，単なる分類精度（Accuracy・マクロ F1・混同行列）だけでなく，**①同一著者ペアの平均類似度 $\mu_{\text{same}}$ と異著者ペアの平均類似度 $\mu_{\text{diff}}$ およびそのギャップ $\Delta S = \mu_{\text{same}} - \mu_{\text{diff}}$**，**② 4×4 の作家間平均類似度マトリクス（対角成分が同一著者，非対角成分が異著者）**，**③同一著者 vs 異著者の類似度分布データ** を必ず算出・描画する機構の実装．

## 2. 対象ファイルパス（Target File Paths）

- `src/__init__.py`
- `src/dataset.py`
- `src/features/__init__.py`
- `src/features/base.py`
- `src/similarity.py`
- `src/classifier.py`
- `src/evaluator.py`

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，すべての規約を遵守すること．
- `scikit-learn` 等の外部機械学習・類似度計算ライブラリの使用は一切禁止．`numpy` のみを用いてフルスクラッチで実装すること．
- 類似度計算（`src/similarity.py`）において，サンプル間の 2 重 `for` ループを書くことを禁ずる．必ず行列積（`X @ Y.T`）や 3 次元ブロードキャスト（またはメモリ効率を考慮したバッチブロードキャスト）でベクトル化すること．
- 情報漏洩（Data Leakage）を防ぐため，評価パイプラインでは必ず Train チャンクのみを `feature_extractor.fit(train_texts)` に渡し，Test チャンクは `transform(test_texts)` のみを通す構造を強制すること．

## 4. 完了検証基準（Verification Criteria）

- **数値的健全性チェック（Numerical Sanity Check）**：ランダム生成した非負ダミー行列 `X`（形状 `(10, 50)`）に対して，`cosine_similarity(X, X)`，`jaccard_similarity(X, X)`，`dice_similarity(X, X)` の対角成分がすべて $1.0$（許容誤差 $10^{-6}$）になり，全要素が $[0, 1]$ の範囲内に収まることを `assert` で検証すること．
- **単一バッチ過学習チェック（Single-Batch Overfit Check）**：ダミーの特徴量抽出器（または恒等変換）を用い，Train データ自身を Test データとして推論器（Centroid および $k=1$ の $k$-NN）に入力した際，正解率（Accuracy）が $1.0$（100%）になることを確認すること．
- **乱数シード再現性検証**：同一の乱数シード（例：`seed=42`）で Intra-work Split を 2 回実行した際，分割される Train / Test チャンクのインデックスが完全に一致することを確認すること．
