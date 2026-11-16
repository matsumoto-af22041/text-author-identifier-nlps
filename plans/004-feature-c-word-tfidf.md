# 004: 特徴量 C（語彙・内容語 TF-IDF）モジュール

## Expected Deliverables & Design Policy

- 担当：メンバー C
- 第 5 回講義資料のベクトル空間モデルに基づき，内容語（名詞・動詞・形容詞）の TF-IDF 重みベクトルをフルスクラッチで算出する `WordTFIDFExtractor(BaseFeatureExtractor)` ．
- **表記揺れの吸収**：作品（底本）間の漢字・仮名の揺れ（「大分」と「だいぶ」，「矢張」と「やはり」等）と活用による語形変化を正規化するため，語彙素（ `feature.lemma` ，未定義時は `surface` ）を特徴量キーとして固定採用．語彙数は上位 `max_features` 件（既定 2000）に制限．
- **実験上の役割**：Intra-work では固有名詞・題材の一致で高精度，Inter-work では未知作品の語彙変化で精度低下することを示し，文体特徴（A・B）の優位性を際立たせるベースライン．

## Target File Paths

- `src/features/word_tfidf.py`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `TfidfVectorizer` 等は禁止． `numpy` と `fugashi` のみで数式通りに実装．
- `fit` で Train チャンク総数 $N$ と各語 $t$ の文書頻度 $\text{df}(t)$ を集計し， $\text{idf}(t) = \log \frac{N}{\text{df}(t)} + 1$ （既定）を保持． `smooth_idf=False` で講義定義の $\log \frac{N}{\text{df}(t)}$ ．
- `transform` で相対出現頻度 $\text{tf}(t, d)$ と保持した $\text{idf}(t)$ の積を行列演算で算出．
- コンストラクタ引数名は `configs/default.yaml` の `features.C.params` （ `smooth_idf` ， `max_features` ）と一致させること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性， `non_negative = True` 時の非負性を自動検査．

## Verification Criteria

- [ ] 手計算可能なトイデータ（3 文書・4 語程度）で `fit_transform` を実行し，TF，DF，IDF，TF-IDF が手計算結果と小数点以下 6 桁まで一致．
- [ ] Test にのみ出現する未知語を含む入力でもゼロ除算（ $\text{df}(t) = 0$ ）を起こさず無視される．
