# Ticket 2: 【担当 A】表層・文字 N-gram・文体統計モジュールの実装（担当：メンバー A）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- 形態素解析に依存しない計量文献学（Stylometry）的特徴として，以下の 2 系統の特徴量を抽出・結合する `StylometryCharExtractor(BaseFeatureExtractor)` の実装：
  1. **表層文体統計量**：1 文あたりの平均文字数（**注意：入力 `text` は文が改行なしで連結された文字列であるため，必ず句点 `。` または `．` で分割して文数を算出すること**），読点（「、」「，」）の出現率，句点・感嘆符・疑問符・括弧類の使用率，および字種構成比（漢字・ひらがな・カタカナ・その他文字の比率）．
  2. **文字 N-gram 相対頻度**：空白・改行を除いた文字列から抽出した文字 Bi-gram（または Tri-gram）の出現相対頻度．Jaccard / Dice 係数の計算コスト爆発を防ぐため，**上位 $M_{\text{char}} \le 1500$ 件（デフォルト `max_features=1000`）** の語彙を `fit` 時に Train データから構築すること．
- **スケールの正規化（確定仕様）**：平均文長（数十）と文字 N-gram 相対頻度（$0.0\text{x}$）のスケール差によるコサイン類似度の支配を防ぎ，かつ非負ベクトルを前提とする Jaccard 係数・Dice 係数にも共通適用できるよう，`fit` 時に Train データでの各次元の最小値・最大値を計算して $[0, 1]$ に揃える **Min-Max 正規化（`scaling="minmax"`）をデフォルト（本採用）** とする．また，質疑応答でのアブレーション検証用に Z-score 標準化（`"zscore"`，負値を含むため `non_negative = False` 扱い）および未正規化（`"none"`）も引数で切り替え可能とすること．
- **発表・考察時の重要ポイント（底本の表記差の分析）**：同一作家でも作品（底本）によって漢字・かなの表記方針が異なるケース（例：『坊っちゃん』は「大分」，『三四郎』は「だいぶ」）が存在するため，字種比率や文字 N-gram が「作家の文体」だけでなく「底本の表記方針」にどの程度影響されるか（作品内 vs 作品間のばらつき）を考察に含めること．

## 2. 対象ファイルパス（Target File Paths）

- `src/features/stylometry_char.py`

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，すべての規約を遵守すること．
- `scikit-learn` 等の外部機械学習・前処理ライブラリ（`MinMaxScaler`, `StandardScaler`, `CountVectorizer` 等）の使用は一切禁止．`numpy` と Python 標準機能のみでフルスクラッチ実装すること．
- 文字 N-gram の語彙辞書（上位 $M_{\text{char}}$ 個の選択）および正規化パラメータ（$\min, \max$ または $\mu, \sigma$）は，必ず `fit(train_texts)` 内で Train データのみから算出し，`transform(texts)` では未知の N-gram を無視して固定パラメータで変換すること（`scaling="minmax"` 時は Test データの値が $[0, 1]$ を超えた場合に `np.clip(X, 0.0, 1.0)` で非負性を保証すること）．

## 4. 完了検証基準（Verification Criteria）

- サンプルテキスト 5 件を入力して `fit_transform` を実行した際，出力行列の形状が `(5, D_features)` であり，`NaN` や `Inf` が一切含まれないこと（`assert np.all(np.isfinite(X))`）．
- デフォルトの `scaling="minmax"` において，出力行列がすべて $[0, 1]$ の非負範囲に収まること，および未知の文字を含むテスト文字列を `transform` に入力しても Train 時と同一の次元数 `D_features` のベクトルが出力されることを確認すること．
