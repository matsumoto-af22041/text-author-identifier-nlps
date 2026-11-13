# Ticket 4: 【担当 C】語彙・内容語 TF-IDF モジュールの実装（担当：メンバー C）

## 1. 期待される成果物と設計方針（Expected Deliverables & Design Policy）

- 第 5 回講義資料のベクトル空間モデル（TF-IDF）に基づき，内容語（名詞・動詞・形容詞）を対象として TF-IDF 重みベクトルをフルスクラッチで算出する `WordTFIDFExtractor(BaseFeatureExtractor)` の実装．
- **表記揺れの吸収（確定仕様）**：近代文学における漢字・仮名の表記揺れ（例：「矢張」と「やはり」等）および活用による語形変化を正規化するため，`unidic-lite` の **語彙素・基本形（`feature.lemma`，未定義時は `surface` にフォールバック）を特徴量キーとして固定採用** する．
- **実験上の役割**：「同一作品内分割（Intra-work）」では固有名詞や題材の一致により高い精度を出す一方，「作品単位分割（Inter-work）」では未知作品の語彙変化により精度が低下することを確認し，文体特徴（A・B）の優位性を示すためのベースライン．

## 2. 対象ファイルパス（Target File Paths）

- `src/features/word_tfidf.py`

## 3. 必須制約事項（Mandatory Constraints）

- 作業開始前に必ずプロジェクトルートの `llm-rules.md` および `llm-context.md` を精読し，すべての規約を遵守すること．
- `scikit-learn` の `TfidfVectorizer` 等は絶対に使用せず，`numpy` と `fugashi-plus[unidic-lite]`（`import fugashi`）のみで数式通り実装すること．
- `fit(train_texts)` にて，Train チャンク総数 $N$ と各単語 $t$ が 1 回以上出現した Train チャンク数 $\text{df}(t)$ を集計し，逆文書頻度 $\text{idf}(t) = \log \frac{N}{\text{df}(t)} + 1$（平滑化ありデフォルト，引数 `smooth_idf=False` で講義スライドそのままの $\log \frac{N}{\text{df}(t)}$ も選択可能）を算出して保持すること．
- `transform(texts)` にて，各チャンク内の相対出現頻度 $\text{tf}(t, d)$ と保持しておいた $\text{idf}(t)$ の積 $\text{tf}(t, d) \times \text{idf}(t)$ を行列演算で算出すること．

## 4. 完了検証基準（Verification Criteria）

- 手計算可能な極小のトイデータ（3 文書・4 単語程度）に対して `fit_transform` を実行し，算出された TF, DF, IDF, TF-IDF の値が数式の手計算結果と小数点以下 6 桁まで一致することを検証すること．
- Test データにのみ出現する未知語（Out-of-Vocabulary）が含まれていてもゼロ除算（$\text{df}(t) = 0$）を起こさず安全に無視されることを確認すること．
