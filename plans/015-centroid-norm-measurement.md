# 015: 重心の長さと作家内類似度の実測（014 の再実行を含む）

## Expected Deliverables & Design Policy

- **手順の前半（014 の再実行）**：コードを変更する前に，014 の 10 run（`bal_s{42..46}_{base,C}` ）を 014 に記載のコマンドで再実行し，`uv run python scripts/summarize_balance.py` を再実行．014 の run はコード未コミットの状態で実行されたため，コミット `daca831` のコードで再実行して `run_meta.json` の追跡情報を確定させる．
  - `dirty_files` に `src/` ， `scripts/` ， `configs/` 配下のファイルが含まれないことを確認（`docs/` と `plans/` は可）．
- **手順の後半（実測）**：重心法の漱石への予測集中について，次の仮説を検証するための記述統計を出す．
  - 長さ 1 に正規化したベクトル $\hat{d}$ の作家別平均を $\hat{c}_a = \frac{1}{n_a} \sum_{d \in D_a} \hat{d}$ とすると，$\frac{1}{n_a} \sum_{d \in D_a} \cos(q, d) = \|\hat{c}_a\| \cos(q, \hat{c}_a)$ ，$\|\hat{c}_a\|^2 = \frac{1}{n_a} + \left(1 - \frac{1}{n_a}\right) \bar{\rho}_a$ （ $\bar{\rho}_a$ ：作家 $a$ の Train チャンク同士（自分自身を除く）の平均コサイン）．
  - 仮説：重心とのコサインは作家内の平均類似度を $\|\hat{c}_a\|$ で割り増しした量であり，$\bar{\rho}_a$ が小さい疎な特徴（C）ではチャンク数 $n_a$ と作品の多様性が $\|\hat{c}_a\|$ を下げて割り増しを大きくする．
  - 実装の重心（[src/classifier.py](../src/classifier.py) の `CentroidClassifier` ）は正規化前のベクトルの平均のため，上の関係は近似．正規化前の重心 $c_a$ のノルムと，$\cos(q, c_a)$ と $\cos(q, \hat{c}_a)$ の差も併せて報告し，近似の程度を示す．
- **対象**：`base` と C，$L \in \{100, 500, 1000\}$ ，Inter-work（LOWO）の 9 fold．Train 構成は「間引きなし」と「014 の間引き（シード 42～46）」の 2 種類．
- **出力する量**（fold × 作家ごと．間引きありは 5 シードの平均と標準偏差）：
  - $n_a$ ，$\bar{\rho}_a$ ，$\|\hat{c}_a\|$ ，$1 / \|\hat{c}_a\|$ （割り増し係数），$\frac{1}{n_a}$ の項と $\bar{\rho}_a$ の項の大きさの比較
  - 正規化前の重心のノルム $\|c_a\|_2$
  - 恒等式の数値確認：Test チャンクについて $\frac{1}{n_a} \sum_d \cos(q, d)$ と $\|\hat{c}_a\| \cos(q, \hat{c}_a)$ の最大絶対誤差
  - 近似の程度：Test チャンクについて，正規化前の重心による予測と正規化後の重心による予測の一致率
  - 割り増し係数と，その fold の漱石以外のチャンク中の漱石予測割合（重心法，`outputs/c_only/` ・ `outputs/full_base/` ・ `outputs/bal_s*_*/` の `folds.csv` または `results.json` から取得）の対応表
- **実装**：`scripts/measure_centroid_norm.py` （新規）．特徴抽出は `run_experiments.py` と同じく `configs/default.yaml` の本採用パラメータで，各 fold の Train で `fit` ．間引きは `src/dataset.py` の `balance_train_split` と `np.random.default_rng([seed, fold 番号])` を使い，014 と同一の Train を再現する．出力は `outputs/centroid_norm/` （CSV と `report.md` ）．

## Target File Paths

- `scripts/measure_centroid_norm.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 014 の再実行は，コードを 1 行も変更していない状態で最初に行う．
- `src/` ， `configs/` ， `scripts/run_experiments.py` は変更しない．
- $\bar{\rho}_a$ は対角（自己類似度 1）を除いて計算する．
- 数値は実行結果からの転記のみ．解釈（仮説の成否）は観測値と分けて報告．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] 014 の 10 run の再実行で，`summarize_balance.py` の出力が再実行前と数値一致．各 run の `run_meta.json` の `commit` と `dirty_files` を報告．
- [ ] 間引きで再現した Train の作家別件数と Train インデックスが，再実行した `bal_s*` の `run_meta.json` に記録された件数と全 fold・全 $L$ ・全シードで一致．
- [ ] 恒等式の最大絶対誤差が $10^{-10}$ 未満．
- [ ] `uv run python scripts/measure_centroid_norm.py` が 10 分以内に完走し，上記の量を `outputs/centroid_norm/` に出力．所要時間と，$L = 500$ の C と `base` の作家別 $\bar{\rho}_a$ ，$\|\hat{c}_a\|$ ，割り増し係数（間引きなし・ありの両方）を報告．
