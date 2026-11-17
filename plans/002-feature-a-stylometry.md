# 002: 特徴量 A（文字 N-gram・記号の相対頻度）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー A
- 形態素解析に依存せず，文書を「文字の列」として扱う（第 5 回 p.10）特徴量 `StylometryCharExtractor(BaseFeatureExtractor)` ．すべての次元を「出現回数 ÷ チャンクの総文字数」の相対出現頻度（第 5 回 p.12 の tf 正規化を文字単位に適用）で表す．
  - **文字 Bi-gram 相対頻度**：空白・改行を除いた文字列の文字 2-gram（第 2 回 p.25 の N-gram を文字に適用）．語彙は `fit` 時に Train データの総出現回数上位 `max_features` 件（既定 1000，上限 1500，同数は文字列の辞書順で決定的に選択）．
  - **記号の相対頻度**：読点（「、」「，」），文末記号（「。」「．」「？」「！」），括弧類（「「」『』（）」等の開き記号）それぞれの出現回数 ÷ 総文字数．
- **平均文長の扱い**：講義の文の定義（第 2 回 p.8「通常『。』『？』などで終わる」）に従い，文数 = 文末記号の出現数とする．文末記号の相対頻度は $\frac{\text{文数}}{\text{総文字数}} = \frac{1}{\text{平均文長}}$ であり，平均文長の情報を他の次元と同程度の大きさ（ $0.0\text{x}$ ）で保持する．
- **字種構成比は類似度ベクトルに含めない**：漢字・ひらがな等の構成比は値が $0.1$ ～ $0.6$ と他の次元より 1 桁以上大きく，そのまま入れるとコサイン類似度をほぼ単独で支配する．しかも底本の表記方針（「大分」「だいぶ」）に最も直撃される特徴のため，類似度の主成分にすると A が底本の識別器になりかねない．字種構成比は考察用の記述統計として，作品別に算出する関数 `char_type_ratios(texts)` を公開する（評価パイプラインでは使用しない）．
- **スケーリング**：既定は `scaling="none"` （講義準拠の相対頻度のみ）．アブレーションとして，Train の各次元の最小値・最大値で $[0, 1]$ に揃える `scaling="minmax"` を切り替え可能（講義外．出典は `plans/research-context.md` の Han, Kamber & Pei (2011)）．
- **ブロック選択**：引数 `blocks` で使用するブロックを切り替え可能．既定 `"all"` （Bi-gram + 記号），アブレーション `"bigram"` （Bi-gram のみ）と `"symbol"` （記号 3 次元のみ）．「A のどちらのブロックが著者識別に効くか」と「句読点・文長だけでどこまで見分けられるか」（教授フィードバックの観点）に答えるための比較条件．
- **事前調査（010，実施済み）**：既定仕様の二乗ノルム割合は Bi-gram / 記号 = 0.72 / 0.28（ $L = 100$ ），0.57 / 0.43（500），0.52 / 0.48（1000）．字種を含む旧案は字種が約 99%，Min-Max 適用時は Bi-gram が 88～96% を占め，単一ブロックに支配されないのは既定仕様のみ．正規化なしの採用根拠として発表で提示する．文末記号は Bi-gram（「た。」等）と情報が一部重なる点を考察で明記．
- **考察上の要点**：同一作家でも作品（底本）により漢字・かなの表記方針が異なる例（『坊っちゃん』「大分」，『三四郎』「だいぶ」）が存在．`char_type_ratios` による作品別の字種構成比と，文字 Bi-gram の作品内・作品間のばらつきを比較し，A が作家の文体ではなく底本の表記方針を拾っていないかを考察に含めること．

## Target File Paths

- `src/features/stylometry_char.py`
- `configs/default.yaml` （ `features.A` セクションのみ： `params: {scaling: none, blocks: all, max_features: 1000}` ， `ablations: {scaling: [minmax], blocks: [bigram, symbol]}` ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `MinMaxScaler` ， `StandardScaler` ， `CountVectorizer` 等の外部ライブラリは禁止． `numpy` と Python 標準機能のみで実装．
- 記号・字種・Bi-gram の定義は 010 の調査（ `scripts/investigate_a_scale.py` ）と一致させる：開き括弧類は「「『（［【〔〈《(」の 9 種，文末記号は全角「。．？！」，字種は Unicode ブロックで判定し漢字に「々〆〇」，カタカナに長音「ー」を含める，Bi-gram はチャンク内のみで数える（文境界は跨ぐ）．
- 入力 `text` は文を区切り文字なしで連結した文字列（ `src/dataset.py` の `Chunk.text` ）．総文字数は空白・改行を除いた文字数．総文字数が 0 のテキストは `ValueError` で停止（Fail-Fast）．
- N-gram 語彙と（ `minmax` 時の）正規化パラメータは `fit` 内で Train データのみから算出． `transform` では未知 N-gram を無視して固定パラメータで変換．
- `scaling="minmax"` 時，Train で $\max = \min$ となる次元は分母 0 のため，その次元を 0 に固定し，該当次元数を属性に記録．Test の値は `np.clip(X, 0.0, 1.0)` で $[0, 1]$ に収める．
- `scaling` に `"none"` ， `"minmax"` 以外， `blocks` に `"all"` ， `"bigram"` ， `"symbol"` 以外が渡された場合は `ValueError` ．
- コンストラクタ引数名は `configs/default.yaml` の `features.A.params` / `ablations` （ `scaling` ， `blocks` ， `max_features` ）と一致させること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性， `non_negative = True` 時の非負性を自動検査．サブクラスは `fit` と `transform` の実装のみで可．

## Verification Criteria

- [ ] サンプルテキスト 5 件で `fit_transform` を実行し，出力形状が `(5, D_features)` ，かつ `assert np.all(np.isfinite(X))` が通る．
- [ ] 既定の `scaling="none"` で，記号の各次元が手計算の「出現回数 ÷ 総文字数」と小数点以下 6 桁まで一致（例：「吾輩は猫である。名前はまだ無い。」で文末記号率 $2 / 16$ ）．
- [ ] `char_type_ratios` の 4 区分の和が各行で $1.0$ （許容誤差 $10^{-9}$ ）．
- [ ] `blocks="bigram"` で出力次元が Bi-gram 語彙数， `blocks="symbol"` で 3 となり，それぞれ `blocks="all"` の対応する列と一致．
- [ ] `scaling="minmax"` で出力がすべて $[0, 1]$ に収まり，Train に無い文字を含むテキストを `transform` しても Train 時と同一の `D_features` 次元が出力される．
