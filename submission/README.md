# 青空文庫の文体・表現類似度に基づく著者識別システム（提出用プログラム一式）

自然言語処理 課題 2（文・文章の類似度計算）の提出物．班名・氏名：（提出直前に記入）

## システムの概要

- 対象：講義で配布された青空文庫サンプルコーパスのうち，2 作品以上を持つ 4 作家・計 9 作品．
  - 夏目漱石：『坊っちゃん』『こころ』『三四郎』
  - 太宰治：『惜別』『斜陽』
  - 寺田寅彦：『映画芸術』『柿の種』
  - 田山花袋：『蒲団』『田舎教師』
- 問い：4 つの観点（A～D）で文章の類似度を測り，未知の作品の著者を識別できるのはどの観点か．
- 処理の流れ：各作品を文の境界で区切った長さ $L \in \{100, 500, 1000\}$ 文字のチャンクに分割 → チャンクを特徴量ベクトルに変換 → コサイン類似度 → 重心法（作家ごとの平均ベクトルとの類似度が最大の作家に判定）・ $k$-NN 法で著者を推定 → 作家別 MAP の平均・作家別正解率の平均・作家別 F 値の平均と各基準値で評価．
- 特徴量（4 観点とベースライン）：
  - A：文字 Bi-gram ・記号（読点・文末記号・開き括弧類）の相対頻度
  - B：品詞 2-gram ・付属語（助詞・助動詞）の相対頻度
  - C：内容語（名詞・動詞・形容詞）の TF-IDF
  - D：C の TF-IDF 行列の特異値分解（LSI）による潜在意味ベクトル
  - base：文字 unigram の相対頻度（比較の基準）
- 分割：同一作品内分割（Intra-work，各作品のチャンクを Train / Test に分ける）と作品単位分割（Inter-work，1 作品を丸ごと Test にする 9 fold）．Train チャンク数は作家ごとに最少の作家の数へ間引いた条件を主結果とし，間引きなしを併記．
- 特徴量抽出・ TF-IDF ・ SVD による LSI ・コサイン類似度・推論器・評価指標はすべて自作（NumPy の配列演算と `np.linalg.svd` のみ使用）．

## 使用したツール

- Python 3.13（パッケージ管理は uv ，バージョンは `.python-version` で固定）
- NumPy：ベクトル・行列演算， SVD
- fugashi（パッケージ名 `fugashi-plus[unidic-lite]` ）と同梱の UniDic（unidic-lite）：形態素解析．OS 側の MeCab の導入は不要
- Matplotlib：図の描画（日本語フォントは `configs/default.yaml` の `report.font_candidates` から自動選択）
- PyYAML：実験設定の読み込み
- scikit-learn 等の機械学習・類似度計算のライブラリは不使用．
- `uv.lock` の開発用グループに含まれる python-pptx は発表スライドの作成用で，本プログラム一式では使わない．

## ディレクトリ構成

```text
README.md            本ファイル
pyproject.toml       依存パッケージ
uv.lock              依存パッケージの版の固定
.python-version      Python の版（3.13）
.gitignore           コーパス・実験出力・仮想環境を Git の追跡から外す設定
configs/default.yaml 実験条件・特徴量の設定（チャンク長・シード・各特徴量の引数・比較条件）
src/                 システムの本体
scripts/             前処理・実験・集計・検証の実行スクリプト
```

コーパス（ `data/raw/` ），前処理の出力（ `data/processed/` ），実験の出力（ `outputs/` ）は含まない（下記の手順で生成）．

### src/ の各モジュール

| ファイル | 役割 |
|---|---|
| `src/dataset.py` | 対象作品の読み込み，文の境界を保つチャンク分割， Intra-work / Inter-work の分割， Train チャンク数の間引き |
| `src/morphology.py` | 形態素解析（fugashi）の共通処理とテキスト単位のキャッシュ（B ・ C ・ D が共用） |
| `src/similarity.py` | コサイン類似度（行列積によるベクトル化実装） |
| `src/classifier.py` | 重心法と $k$-NN 法（作家別上位 $k$ 平均・全体上位 $k$ の重み付き投票）の著者推定器 |
| `src/evaluator.py` | 同一著者・異著者の平均類似度とその差，作家別 MAP の平均，作家別正解率の平均，作家別 F 値の平均，混同行列， MRR と各基準値．1 fold の評価（Train のみで `fit` ） |
| `src/features/base.py` | 特徴量抽出の抽象基底クラス（ `fit` ・ `transform` ） |
| `src/features/char_unigram.py` | base：文字 unigram の相対頻度 |
| `src/features/stylometry_char.py` | A：文字 Bi-gram ・記号の相対頻度 |
| `src/features/pos_function.py` | B：品詞 2-gram ・付属語の相対頻度 |
| `src/features/word_tfidf.py` | C：内容語の TF-IDF |
| `src/features/lsi_embedding.py` | D：LSI の潜在意味ベクトル |

### 主な scripts/

| ファイル | 役割 |
|---|---|
| `scripts/inspect_and_extract_corpus.py` | コーパスの文字コード・構造・仮名遣いの検査と，対象 9 作品の抽出（ `data/processed/authors_corpus.json` ） |
| `scripts/run_experiments.py` | 全体の実験（特徴量 × チャンク長 × 分割 × 推論方式）．結果の CSV ・ JSON と，続けて図・ `report.md` を出力（ `--no-report` で図・ `report.md` を省略） |
| `scripts/plot_results.py` | 実験結果のディレクトリから図・表・ `report.md` を再生成 |
| `scripts/sensitivity_core_vocab.py` | 付属語と内容語の比較（B の語ブロック単独）の，内容語の語彙選択に対する感度分析 |
| `scripts/plot_d_k_sweep.py` | D の潜在次元数 $K$ の推移図 |
| `scripts/function_word_profiles.py` | 作家ごとに偏りのある付属語の具体例（記述統計） |
| `scripts/verify_*.py` | 各モジュールの検証（トイデータでの形状・数値の一致，評価指標の手計算との一致等） |
| その他（ `measure_*.py` ， `summarize_*.py` ， `investigate_a_scale.py` ， `probe_unidic_fields.py` ） | 補助実験・記述統計 |

## 動かし方

以下はすべて展開したディレクトリ（ `README.md` のある場所）で実行．

### 準備：Git の管理下に置く

前処理と実験のスクリプトは，再現性の記録のため実行時の git の commit を出力に記録し，git の管理下でないと停止する．展開直後に次を 1 回だけ実行（ `.gitignore` によりコーパス・実験出力・仮想環境は追跡されない）．

```bash
git init
git add -A
git -c user.name=submission -c user.email=submission commit -m "submission"
```

### 環境構築

```bash
uv sync
```

Python 3.13 と依存パッケージ（形態素解析の辞書を含む）が `.venv/` に入る．

### コーパスの配置と前処理

1. 講義で配布された青空文庫サンプルコーパス `corpus20150426_with_titles.txt` を `data/raw/` に置く（配置先は `configs/default.yaml` の `paths.raw_corpus` ）．

   ```bash
   mkdir -p data/raw
   cp <配布ファイルの場所>/corpus20150426_with_titles.txt data/raw/
   ```

2. 前処理（検査と 9 作品の抽出）．出力は `data/processed/authors_corpus.json` ．

   ```bash
   uv run python scripts/inspect_and_extract_corpus.py
   ```

3. コーパスが同じものかは，出力の `meta.source_sha256` が次の値と一致することで確認できる．

   ```text
   90301ebac41bc8a723b1b25fb955c31862911940efa59d31432ce61ff3f07d45
   ```

   ```bash
   uv run python -c "import json; print(json.load(open('data/processed/authors_corpus.json'))['meta']['source_sha256'])"
   ```

### 動作確認（約 2 分）

特徴量 A ～ D を $L = 1000$ だけで実行する小さな実験と，その図・ `report.md` の生成．出力は `outputs/smoke/` ．

```bash
uv run python scripts/run_experiments.py --chunk-chars 1000 --run-name smoke
```

`run_experiments.py` は結果の保存後に `plot_results.py` と同じ処理で図・ `report.md` を作る．下記の表のように `--no-report` で実験だけを行い，図・ `report.md` を `plot_results.py` で後から作っても同じ出力になる．

### 主結果の再現

| 内容 | コマンド | 出力 | 所要時間の目安 |
|---|---|---|---|
| A ～ D の全チャンク長・両分割・比較条件 | `uv run python scripts/run_experiments.py --ablation --run-name full_006 --no-report` | `outputs/full_006/` （ `summary.csv` ， `folds.csv` ， `results.json` 等） | 約 17 分 |
| 上記の図・ `report.md` | `uv run python scripts/plot_results.py outputs/full_006` | `outputs/full_006/` （ `report.md` ， `figures/` ，表の CSV ） | 1 分未満 |
| ベースライン base | `uv run python scripts/run_experiments.py --features base --run-name full_base --no-report` と `uv run python scripts/plot_results.py outputs/full_base` | `outputs/full_base/` | 1 分未満 |
| D の潜在次元数 $K$ の推移 | `uv run python scripts/plot_d_k_sweep.py --run-dir outputs/full_006` | `outputs/full_006/` 以下 | 1 分未満 |
| 付属語と内容語の比較の感度分析 | `uv run python scripts/sensitivity_core_vocab.py --baseline-run full_006 --chunk-chars 500 --out-dir outputs/sensitivity_L500` （ $L = 100, 1000$ も同様） | `outputs/sensitivity_L500/` | $L$ ごとに約 1 ～ 2 分 |
| 作家ごとに偏りのある付属語 | `uv run python scripts/function_word_profiles.py --out-dir outputs/function_word_profiles` | `outputs/function_word_profiles/` | 1 分未満 |

- 所要時間は開発に使ったノート PC（CPU のみ）での実測値．前処理は数秒，全部で 30 分程度．
- 乱数のシード（Intra-work の分割と間引き：42 ～ 46）は `configs/default.yaml` の `split` で固定しており，同じコーパス・同じ設定なら同じ数値になる．
- 各実行の設定・引数・ git の commit ・コーパスのハッシュは出力の `run_meta.json` に記録される．
- `run_experiments.py` ・ `sensitivity_core_vocab.py` ・ `function_word_profiles.py` は，同じ名前の出力ディレクトリがあると上書きせずに停止する．
- `sensitivity_core_vocab.py` は `--baseline-run` の結果との一致を確認してから実行するため， `full_006` の実行後に行う．

## 注記

- コードのコメント・ docstring にある 3 桁の番号（例： `（041）` ， `013～015` ， `plans/029` ， `Ticket 005` ）は，開発時の作業記録の番号．作業記録は提出物に含めない．
- コメントにある「第 5 回 p.26」等は講義資料の回・頁．コーパスの構造の記述（ `references/AozoraCorpus.pdf` 準拠）はコーパスの配布資料を指す．
- 発表スライドの生成・検査と発表用の図の作成（発表資料の内容データが必要），開発時の Git の履歴の版との数値の一致の確認（ `verify_pos_function.py` ）のスクリプトは，本プログラム一式では動かないため含めない．結果の再現には不要．
- `scripts/measure_centroid_norm.py` ・ `scripts/summarize_exclusion.py` ・ `scripts/verify_qa_points.py` は，先に `run_experiments.py` で作った出力（run 名を引数で指定）を読む．
