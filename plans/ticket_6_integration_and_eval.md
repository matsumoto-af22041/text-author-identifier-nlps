# Ticket 6: 全体統合実験スクリプトと可視化・レポート出力の実装（担当：全員／統合はリーダー）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- 特徴量モジュール A〜D，類似度関数（コサイン類似度・Jaccard 係数・Dice 係数），チャンク長（$L \in \{100, 500, 1000\}$），データ分割方式（Intra-work vs Inter-work），推論方式（重心法 Centroid vs $k$-NN 法 $k \in \{1, 3, 5, 10, 20\}$）の全組み合わせ実験，および質疑応答対策用のアブレーション実験（担当 A の正規化有無，担当 B の `surface` vs `lemma`，担当 D の次元数 $K \in \{8, 16, 32, 64\}$）を一括または条件指定で実行し，結果の表（CSV / Markdown）とプレゼン用グラフ（混同行列，チャンク長別精度推移，同一著者 vs 異著者の類似度ヒストグラム）を自動生成する実験スクリプトの実装．

## 2. 対象ファイルパス（Target File Paths）

- `scripts/run_experiments.py`
- `scripts/plot_results.py`
- `outputs/`（実験結果ログおよびグラフ画像の出力先）

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，すべての規約を遵守すること．
- `llm-rules.md` の「5. Experiment Traceability（実験の追跡可能性）」に従い，出力される結果 JSON / CSV には必ず実行時の設定パラメータ（乱数シード，チャンク長，特徴量ハイパーパラメータ）を併記して保存すること．
- 負の値を含む特徴量 D（LSI，`non_negative = False`）に対してはコサイン類似度のみを適用し，非負ベクトルである特徴量 A・B・C に対してはコサイン類似度・Jaccard 係数・Dice 係数を比較できるよう制御すること．
- **計算コスト制御と主指標の統一**：
  1. Jaccard / Dice 係数は計算量が $O(N_{\text{test}} N_{\text{train}} D)$ となるため，**代表チャンク長 $L = 500$（および $L = 1000$）のみに限定して実行**し，チャンク数が膨大になる $L = 100$ では高速なコサイン類似度のみを実行する構成をデフォルトとすること．
  2. 作家間のチャンク数不均衡（漱石 3 作品 vs 他 2 作品）によるバイアスを排除するため，類似度ギャップの主指標には **4×4 作家間平均類似度行列の対角平均と非対角平均の差であるマクロ平均ギャップ `delta_s_macro` を採用**（ペア単位のミクロ版 `delta_s_micro` も併記）し，Inter-work（LOWO 9 fold）の判別精度・マクロ F1 は **全 fold の混同行列を合算した `aggregate_confusion`（Pooled Out-of-Fold）** で算出すること．

## 4. 完了検証基準（Verification Criteria）

- `uv run python scripts/run_experiments.py --config configs/default.yaml` がエラーなく完走し，「文章の類似度を測り，著者を判別できるか」を実証する以下の 5 種類の評価データが `outputs/` に出力されること：
  1. **類似度ギャップと著者判別精度の総合比較表（Intra-work vs Inter-work）**：各特徴量（A〜D）における「同一著者平均類似度 $\mu_{\text{same}}$」「異著者平均類似度 $\mu_{\text{diff}}$」「類似度ギャップ $\Delta S = \mu_{\text{same}} - \mu_{\text{diff}}$」および「著者判別精度（Accuracy / Macro-F1）」の比較表（内容語 C は別作品で $\Delta S$ と精度が急落する一方，文体特徴 A・B は別作品でも $\mu_{\text{same}} \gg \mu_{\text{diff}}$ を維持することの定量証明）．
  2. **同一著者ペア vs 異著者ペアの類似度分布ヒストグラムと 4×4 作家間平均類似度ヒートマップ**（「同じ著者の文章は類似度が高く，別の著者の文章は低くなる」ことの視覚的証明）．
  3. **チャンク長 $L \in \{100, 500, 1000\}$ による類似度ギャップ $\Delta S$ と判別精度の推移グラフ**：先生の質問（「どのくらいの文字数があれば類似度で著者を判別できるか」）に対する定量的な回答データ．
  4. **類似度関数（Cosine vs Jaccard vs Dice）および推論方式（Centroid vs $k$-NN $k \in \{1, 3, 5, 10, 20\}$）の比較表**．
  5. **最良モデルの混同行列（Confusion Matrix）と誤判別チャンクの定性抜粋ログ**．
