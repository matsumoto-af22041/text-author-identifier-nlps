# 004: 特徴量 C（語彙・内容語 TF-IDF）モジュール

## Expected Deliverables & Design Policy

- 担当：リーダー
- 第 5 回講義資料のベクトル空間モデル（p.25）に基づき，内容語の TF-IDF 重みベクトルをフルスクラッチで算出する `WordTFIDFExtractor(BaseFeatureExtractor)` ．
- **索引語の選択**：第 5 回 p.11 の索引語（文書の内容を表す単語）として，名詞・動詞・形容詞（UniDic の `pos1` ）を対象とする．
- **特徴量キー**：第 2 回 p.16 の原形復元に相当する UniDic の語彙素（ `feature.lemma` ，未知語で `None` の場合は `surface` ）を固定採用．活用による語形変化に加え，内容語の表記の揺れ（「云う」と「言う」，「解る」と「分かる」，「事」と「こと」等）も統一される．後者は UniDic 固有の仕様で講義外（出典は `plans/research-context.md` の伝ほか (2007)）．「大分」と「だいぶ」，「矢張」と「やはり」も語彙素は統一されるが副詞のため C の索引語には含まれない（底本の表記差が効くのは文字表層を見る A のみ）．
- **数式（講義定義どおり）**：
  - $\text{tf}(t, d) = \frac{\text{チャンク } d \text{ における語 } t \text{ の出現回数}}{\text{チャンク } d \text{ の総形態素数}}$ （第 5 回 p.12．総形態素数は補助記号・空白を除く全形態素数で，B と同じ定義）
  - $\text{idf}(t) = \log \frac{N}{\text{df}(t)}$ （第 5 回 p.13．自然対数）
  - 重み $w_{dt} = \text{tf}(t, d) \cdot \text{idf}(t)$
  - Train の全チャンクに出現する語（ $\text{df}(t) = N$ ）は $\text{idf}(t) = 0$ となり重みが消える．これは第 5 回 p.11「どの文書にも均質に現れる語」は索引語として不適切，という定義と整合するため，補正（ $+1$ 等）は行わない．
- 語彙数は Train の文書頻度上位 `max_features` 件（既定 2000，同数は辞書順で決定的に選択）に制限．
- **実験上の役割**：Intra-work では固有名詞・題材の一致で高精度，Inter-work では未知作品の語彙変化で精度低下することを示し，文体特徴（A・B）と比較するベースライン．

## Target File Paths

- `src/features/word_tfidf.py`
- `configs/default.yaml` （ `features.C` セクションのみ： `params: {max_features: 2000}` ， `ablations: {}` ． `smooth_idf` は削除）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `TfidfVectorizer` 等は禁止． `numpy` と `fugashi` のみで数式通りに実装．
- `fit` で Train チャンク総数 $N$ と各語 $t$ の文書頻度 $\text{df}(t)$ を集計し， $\text{idf}(t) = \log \frac{N}{\text{df}(t)}$ を保持．語彙は Train に出現した語のみなので $\text{df}(t) \geq 1$ が常に成立することを `assert` ．
- `transform` で相対出現頻度 $\text{tf}(t, d)$ と保持した $\text{idf}(t)$ の積を行列演算で算出．Test のみに出現する語は無視．
- 総形態素数が 0 のテキストは `ValueError` で停止（Fail-Fast）．語彙に一致する語を持たないチャンクはゼロベクトルとなり得るが，これは停止せず，件数を `transform` 呼び出しごとに属性へ記録（評価側の同点処理は 008 で対応）．
- 内部の形態素解析処理（ `fugashi.Tagger` による語の抽出と tf・df 集計）は，005 から再利用できるようメソッドとして公開すること．
- コンストラクタ引数名は `configs/default.yaml` の `features.C.params` （ `max_features` ）と一致させること．
- `BaseFeatureExtractor` は `__init_subclass__` により fit 前 `transform` の停止，出力の 2 次元性・行数一致・有限値，次元数の一貫性， `non_negative = True` 時の非負性を自動検査．

## Verification Criteria

- [ ] 手計算可能なトイデータ（3 文書・4 語程度）で `fit_transform` を実行し，TF，DF，IDF，TF-IDF が手計算結果と小数点以下 6 桁まで一致．全文書に出現する語の IDF と重みが厳密に 0 であることを含む．
- [ ] Test にのみ出現する未知語を含む入力でもゼロ除算を起こさず無視され，次元数が Train 時と一致．
- [ ] 内容語の表記揺れ（「云った」と「言った」，「解らない」と「分からない」，「事」と「こと」）を含む文の対で，両者が同一の語彙次元に集計され行ベクトルが一致することを確認（UniDic 語彙素による表記統一の実測）．
- [ ] 『坊っちゃん』と『三四郎』の全文で「大分」の表層形を持つ形態素の `pos1` と `lemma` の内訳を集計し，地名（固有名詞，lemma = オオイタ）として誤解析され C の語彙に入りうる件数を報告．

## Review Notes（2026-11-17）

- 初版の検証 3 は「大分」と「だいぶ」の統一を C の根拠としていたが，両語は副詞で C の索引語（名詞・動詞・形容詞，第 5 回 p.11）の対象外と判明．索引語に副詞を加える案は講義の定義と B の内容語対照条件から外れるため採らず，根拠の例を内容語の揺れに差し替えた．
- 実測結果：「云った / 言った」→ `言う` ，「解らない / 分からない」→ `分かる` ，「事 / こと」→ `事` で行ベクトル完全一致．『坊っちゃん』の「大分」20 件中，文単位の解析で副詞 15 件・地名（オオイタ）誤解析 5 件（全文連結では 16 / 4 件．文脈依存）．『三四郎』は「大分」0 件，「だいぶ」38 件・「だいぶん」2 件ですべて副詞．
- 実データ $L = 100$ ，LOWO `lowo_soseki_0` で Train (6322, 2000)，Test (725, 2000)，ゼロベクトル 0 件，idf = 0 の語 0 件，df 範囲 11～4664．
