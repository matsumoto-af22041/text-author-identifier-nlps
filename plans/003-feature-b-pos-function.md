# 003: 特徴量 B（統語・品詞 N-gram・機能語）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー B
- トピックや固有名詞に依存しない統語・文体特徴として， `fugashi` （ `fugashi-plus[unidic-lite]` ）による以下の特徴量を抽出する `POSFunctionExtractor(BaseFeatureExtractor)` ．
  - **品詞 N-gram 相対頻度**：各形態素の品詞大分類（ `pos1` ）と中分類（ `pos2` ）の系列から構築した品詞遷移パターン（POS Bi-gram / Tri-gram）の相対頻度．
  - **機能語の相対頻度**：内容語（名詞・動詞・形容詞・副詞）を除外し，助詞・助動詞・接続詞・連体詞・感動詞・補助記号のみを対象．助動詞の活用形に現れる文末の癖を保持するため，表層形＋品詞（ `token_key="surface"` ）を既定とし，比較用に語彙素・基本形（ `token_key="lemma"` ）も切り替え可能．
- **考察上の要点**：文字表層を見る特徴量 A は底本の漢字・かな表記の揺れに直撃されるのに対し，本モジュールは品詞の並びと機能語のみを見るため表記方針の違いに頑健，という対比を示すこと．ただし `surface` では一部の副詞・接続詞の表記差が残る点に留意．

## Target File Paths

- `src/features/pos_function.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 外部機械学習ライブラリは禁止． `numpy` と `fugashi` のみで実装．
- 名詞・動詞・形容詞・副詞が機能語語彙に混入しないよう，対象品詞を `TARGET_POS1 = ("助詞", "助動詞", "接続詞", "連体詞", "感動詞", "補助記号")` として厳密に定義．
- `fit` で Train データに出現した品詞 N-gram と機能語の語彙（最小文書頻度 `min_df` 以上，最大 `max_features` 件，既定 1500）を固定し， `transform` では各チャンクの総形態素数で割った相対頻度（非負）を出力．
- コンストラクタ引数名は `configs/default.yaml` の `features.B.params` / `ablations` （ `token_key` ， `max_features` ）と一致させること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性， `non_negative = True` 時の非負性を自動検査．

## Verification Criteria

- [ ] 名詞のみを入れ替えた 2 文（「吾輩は猫である。」と「先生は人間である。」）を変換した際，機能語（「は」「で」「ある」「。」等）と品詞系列が一致し，コサイン類似度が $> 0.8$ ．
- [ ] 出力がすべて非負（ `assert np.all(X >= 0)` ）かつ有限値．
