# 002: 特徴量 A（表層・文字 N-gram・文体統計）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー A
- 形態素解析に依存しない計量文献学的特徴として，以下 2 系統を抽出・結合する `StylometryCharExtractor(BaseFeatureExtractor)` ．
  - **表層文体統計量**：1 文あたりの平均文字数，読点（「、」「，」）の出現率，句点・感嘆符・疑問符・括弧類の使用率，字種構成比（漢字・ひらがな・カタカナ・その他）．
  - **文字 N-gram 相対頻度**：空白・改行を除いた文字列の文字 Bi-gram（または Tri-gram）の相対頻度．Jaccard / Dice の計算量を抑えるため，語彙は `fit` 時に Train データから上位 `max_features` 件（既定 1000，上限 1500）で構築．
- **スケール正規化**：平均文長（数十）と N-gram 相対頻度（ $0.0\text{x}$ ）のスケール差によるコサイン類似度の支配を防ぎ，非負前提の Jaccard / Dice にも共通適用するため，Train データの各次元の最小値・最大値で $[0, 1]$ に揃える Min-Max 正規化（ `scaling="minmax"` ）を既定．アブレーション用に Z-score 標準化（ `"zscore"` ，負値を含むため非負性を満たさない）と未正規化（ `"none"` ）も切り替え可能．
- **考察上の要点**：同一作家でも作品（底本）により漢字・かなの表記方針が異なる例（『坊っちゃん』「大分」，『三四郎』「だいぶ」）が存在．字種比率や文字 N-gram が作家の文体ではなく底本の表記方針を拾っていないか，作品内と作品間のばらつきを比較して考察に含めること．

## Target File Paths

- `src/features/stylometry_char.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `MinMaxScaler` ， `StandardScaler` ， `CountVectorizer` 等の外部ライブラリは禁止． `numpy` と Python 標準機能のみで実装．
- 入力 `text` は文を区切り文字なしで連結した文字列．平均文長は句点（「。」「．」）で分割して文数を算出．
- N-gram 語彙と正規化パラメータ（ $\min, \max$ または $\mu, \sigma$ ）は `fit` 内で Train データのみから算出．`transform` では未知 N-gram を無視して固定パラメータで変換．
- `scaling="minmax"` 時は Test の値が $[0, 1]$ を超えた場合に `np.clip(X, 0.0, 1.0)` で非負性を保証．
- `scaling="zscore"` を選んだインスタンスは非負性を満たさないため，クラス属性 `non_negative` との整合（ `BaseFeatureExtractor` の非負性検査に抵触しない設計）を確保．
- コンストラクタ引数名は `configs/default.yaml` の `features.A.params` / `ablations` （ `scaling` ， `max_features` ）と一致させること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性， `non_negative = True` 時の非負性を自動検査．サブクラスは `fit` と `transform` の実装のみで可．

## Verification Criteria

- [ ] サンプルテキスト 5 件で `fit_transform` を実行し，出力形状が `(5, D_features)` ，かつ `assert np.all(np.isfinite(X))` が通る．
- [ ] 既定の `scaling="minmax"` で出力がすべて $[0, 1]$ に収まり，Train に無い文字を含むテキストを `transform` しても Train 時と同一の `D_features` 次元が出力される．
