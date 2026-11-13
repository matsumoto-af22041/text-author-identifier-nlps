# Ticket 0・1 完了報告および要検証項目の状況（Orchestrator 向け）

## 1. 概要

- 対象：Ticket 0（環境構築・コーパス検査），Ticket 1（共通システム基盤）
- 要検証項目の出典：`presentation/qa_verification_instructions.md`（Q1〜Q5，Q10，Q11），`presentation/note.md`（当て字・表記揺れの事前定量検査）
- コミット状態
  - Ticket 0：コミット済み（`1f39f5a`）
  - Ticket 1：未コミット（`src/`，`scripts/verify_core_pipeline.py`，`configs/default.yaml` / `pyproject.toml` / `uv.lock` への追記）
- 再現コマンド
  - `uv run python scripts/inspect_and_extract_corpus.py`（Ticket 0）
  - `uv run python scripts/verify_core_pipeline.py`（Ticket 1）

## 2. 完了検証基準の結果

| チケット | 検証基準 | 結果 |
| :--- | :--- | :---: |
| Ticket 0 | スクリプト完走，全 30 作品のメタ情報と作家別作品数の標準出力 | 合格 |
| Ticket 0 | 4 作家・9 作品の欠落なき抽出，`total_chars > 1000` | 合格（最小は『映画芸術』の 26,749 字） |
| Ticket 1 | `(10, 50)` 非負行列で 3 類似度の対角 $= 1$（誤差 $10^{-6}$ 以内），全要素 $[0, 1]$ | 合格（最大誤差 $1.2 \times 10^{-13}$） |
| Ticket 1 | Train = Test で Centroid・$k=1$ の $k$-NN の Accuracy $= 1.0$ | 合格（3 類似度 × Centroid / per_author / global の全組合せ） |
| Ticket 1 | `seed=42` で Intra-work Split を 2 回実行した際のインデックス完全一致 | 合格（`seed=43` では相違することも確認） |

追加で確認した事項

- 素朴な定義式（ペア単位）との数値一致，バッチサイズ非依存性，負値入力時の Jaccard 停止
- fit 前の `transform` 呼び出しの停止（`NotFittedError`）
- 実データ通し実行（ダミー文字 unigram，$L=1000$）：Intra-work（cosine）で Acc 0.91〜0.98，LOWO（Jaccard，全 fold の混同行列を合算）で Acc 0.45〜0.59

## 3. 要検証項目の状況

状態の凡例：**検証済**（数値取得済み），**実装済・実験待ち**（基盤は完成，Ticket 6 の本実験で数値取得），**要修正**（設計文書の記述と実測が不整合）

| 項目 | 出典 | 状態 | 結果・根拠 |
| :--- | :--- | :---: | :--- |
| Q1. 複数作品を持つ作家が 4 名のみであること | QA Q1（Ticket 0 で集計） | 検証済 | 全 30 作品・25 作家を集計．2 作品以上を持つ作家は夏目漱石 3，寺田寅彦 2，田山花袋 2，太宰治 2 の 4 名のみ．他 21 作家は各 1 作品 |
| Q2-a. 対象 9 作品の仮名遣い | QA Q2，note.md | 検証済 | 9 作品すべて新仮名と判定．歴史的仮名遣い特有パターン（ゐ・ゑ，てゐ，でせう，だらう，やう・さう＋助詞，漢字＋ふ＋句読点）の出現率は 8 作品で 0，『田舎教師』のみ 1 万字あたり 1.13（17 件，閾値 5.0 未満，引用部と推定） |
| Q2-b. 当て字の有無 | note.md | 検証済（**交絡の疑いあり，4 章参照**） | 9 語の漢字表記とかな表記を計数．『蒲団』は漢字表記中心（矢張 6 / やはり 0，沢山 3 / たくさん 0），他は概ねかな表記．ただし同一作家内でも不一致（『坊っちゃん』大分 20 / だいぶ 0，『三四郎』大分 0 / だいぶ 40） |
| Q2-c. `lemma` による表記揺れの吸収 | QA Q2 | 検証済 | `unidic-lite` で「矢張り・やはり・矢っ張り・やっぱり」→ `矢張り`，「成程・なるほど」→ `成る程`，「沢山・たくさん」→ `沢山` に統一されることを確認 |
| Q3. チャンク長の実測分布 | QA Q3 | 検証済（**記述は要修正**） | 平均 ± 標準偏差：$L=100$ で $125.8 \pm 33.1$（最大 753，$2L$ 以上 187 個），$L=500$ で $527.3 \pm 36.7$（最大 1,233），$L=1000$ で $1029.1 \pm 42.9$（最大 1,509） |
| Q4. 末尾の端数チャンクの破棄率 | QA Q4 | 検証済（**記述は要修正**） | 全体の破棄率：$L=100$ で 0.04%，$L=500$ で 0.16%，$L=1000$ で 0.53%．作品単位の最大値は 1.66%（$L=500$・$L=1000$） |
| Q5. Intra-work / Inter-work の両分割 | QA Q5 | 実装済・実験待ち | `intra_work_split`（作品単位の層化ランダム分割，`test_ratio=0.2`），`inter_work_splits`（Leave-One-Work-Out，9 fold，全 fold で 4 作家が Train に残ることを検査）を実装 |
| Q10. 重心法・$k=1$ で Jaccard と Dice の予測が一致すること | QA Q10 | 検証済（理論・数値） | $\max \lvert D - 2J/(1+J) \rvert = 2.2 \times 10^{-16}$．非負乱数データ（Train 200，Test 300）で予測の一致率は Centroid・per_author $k=1$・global $k=1$ が 100%，per_author $k=5$ が 99.3%，$k=20$ が 98.0% |
| Q11. `knn_mode="per_author"` のデフォルト化と $k \in \{1, 3, 5, 10, 20\}$ | QA Q11 | 実装済・実験待ち（**記述は要修正**） | `k_eff = min(k, len(D_a))` の per_author と，類似度で重み付けした global 投票を実装．同点の場合は小さいラベルを採用（決定的） |

注：Q2-b，Q2-c，Q3，Q4，Q10 の数値はアドホックな計測によるもの．QA-1 の `scripts/verify_qa_points.py` で再生成し，`presentation/qa_defense_matrix.md` に反映すること．

## 4. 判明した問題点・リスク（判断依頼を含む）

### 4.1 研究設計への影響

1. **表記（漢字 / かな）の作品間不一致による交絡の疑い**
   - 同一作家でも作品ごとに表記方針が異なる（『坊っちゃん』「大分」と『三四郎』「だいぶ」）．底本や校訂の方針の違いが混入している可能性．
   - 担当 A（文字 N-gram・字種比率）は表記の差を直接拾う．Inter-work で A が高精度でも，作家の文体ではなく底本の表記方針を識別しているだけ，という反論が成り立ち得る．
   - 担当 C（`lemma` 固定）では吸収される．担当 B（`surface` デフォルト）では一部の副詞・接続詞の表記差が残る．
   - 提案：Ticket 2 の考察に「字種比率の作品内・作品間のばらつき」を追加．QA に「表記の差は文体か底本か」の想定質問を追加．
2. **Q3 の回答ロジック「相対頻度化と L2 正規化で完全にキャンセル」は過大な主張**
   - スケールの差は消える．しかし $L=100$ では変動係数が約 26%，$2L$ 以上のチャンクが 187 個あり，チャンクの長さに応じて頻度推定の分散（スパース性）が変わる影響は残る．
   - 提案：「スケールの差は正規化で除去，推定の分散の差は残存」に記述を修正．
3. **Q4 の「全体の 1% 未満」は全体値としては正しいが，作品単位では最大 1.66%**
   - 提案：作品単位の最大値を併記．
4. **Q11 の「$k = 10, 20 \to$ Centroid への漸近」は数学的に不正確**
   - per_author で $k \to \lvert D_a \rvert$ とした極限は「作家 $a$ の全チャンクとの平均類似度」であり，重心との類似度 $\text{sim}(\vec{q}, \vec{\mu}_a)$ とは一致しない（コサインの平均 ≠ 平均ベクトルとのコサイン）．
   - 提案：「全チャンク平均類似度への漸近（重心法に近い大域的評価）」に記述を修正．
5. **類似度ギャップ $\Delta S$ の作家不均衡**
   - ペア単位の $\mu_{\text{same}}$，$\mu_{\text{diff}}$ は，チャンク数が約半数を占める漱石に引っ張られる．
   - 対応済み：作家単位のマクロ版 `delta_s_macro`（4×4 行列の対角平均と非対角平均の差）を併せて算出．
   - 判断依頼：Ticket 6 の主要比較表で，どちらを主指標にするか．
6. **LOWO の fold 単位ではマクロ F1 が定義不能**
   - 1 fold の Test には 1 作家しか含まれない．fold 単位のマクロ F1 は `NaN` とし，全 fold の混同行列を合算する `aggregate_confusion` で算出する仕様とした．
   - 2 作品しかない作家（太宰・寅彦・花袋）は，保持された fold で Train が 1 作品のみとなる．

### 4.2 計算コスト（Ticket 6 の実行計画に影響）

- Jaccard / Dice は $\sum \min(x, y)$ が行列積に帰着しないため，計算量は $O(n_{\text{test}} \cdot n_{\text{train}} \cdot D)$．
- 実測（$L=100$，Intra-work，$D = 3{,}464$ の文字 unigram）：1 fold あたり Jaccard 約 39 秒，cosine 約 0.6 秒．
- 文字 bigram・trigram など $D$ が数万規模になる特徴では，全組合せ実験が「数分以内」の制約を大きく超える見込み．
- 対応済み：類似度行列 $S$ は 1 fold につき 1 回だけ計算し，類似度直接評価と全 $k$ で共有（通し実行 11.4 秒 → 4.7 秒）．
- 判断依頼：以下のいずれかの方針
  - 各特徴量モジュールでの次元数の上限（上位 $M$ 個）の設定
  - $L=100$ での Jaccard / Dice の実行条件の限定
  - `float32` 化による半減

### 4.3 データ・環境

1. **配布仕様と手元ファイルの文字コードの不一致**
   - 配布仕様は Shift-JIS．作業中に `data/raw/corpus20150426_with_titles.txt` が UTF-8 変換版（7.4MB）から Shift-JIS 原本（5,186,722 バイト，zip 内と同一）に置き換わった．
   - スクリプトは `cp932` → `utf-8` の順で全行を厳密にデコードして判定するため，どちらでも動作する．
   - 班員間の同一性は `authors_corpus.json` の `meta.source_sha256` で照合可能．
2. **`pyproject.toml` の変更**
   - `uv init` の生成値 `requires-python = ">=3.14"` を `">=3.13,<3.14"` に修正．
   - `src` を import 可能にするため，hatchling の build-system を追加（Ticket 1）．
3. **`outputs/` が `.gitignore` 未登録**
   - 生成物は再現可能．誤判別の抜粋ログには本文の断片が含まれる．
   - 提案：`outputs/` を除外し，発表に使う図のみ `presentation/` にコピーして管理．

## 5. 後続チケットへの申し送り（インタフェース仕様）

- `BaseFeatureExtractor`（`src/features/base.py`）
  - サブクラスは `fit(train_texts) -> self` と `transform(texts) -> np.ndarray` を通常どおり実装するだけでよい．
  - `__init_subclass__` により以下の検査が自動付与される．
    - fit 前の `transform` の停止
    - 出力の 2 次元性・行数一致・有限値
    - 次元数の一貫性
    - `non_negative = True` 時の非負性
  - 担当 D（LSI）は `non_negative = False` を宣言すること．`evaluate_fold` が Jaccard / Dice との組合せを拒否する．
- `Chunk`（`src/dataset.py`）
  - `text` は文を区切り文字なしで連結した文字列．
  - 文境界が必要な特徴（担当 A の平均文長等）は `sentences` を参照するか，句点で再分割すること．
- 評価は `evaluate_fold(make_extractor, chunk_set, split, similarity_name, classifier_specs)` を使用．
  - 抽出器は fold ごとに新規生成し，Train のみで fit する構造を強制．
