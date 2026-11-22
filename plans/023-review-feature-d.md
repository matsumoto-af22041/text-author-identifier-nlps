# 023: 特徴量 D のプルリクエストブランチの動作確認

## Expected Deliverables & Design Policy

- 目的：班員が作成したブランチ `feat/feature-d-lsi` （005 の実装）のレビューのため，検証スクリプトの実行と，スライド案の数値の再現確認を行う．コードは変更しない．
- 作業はブランチ `feat/feature-d-lsi` をチェックアウトした状態（現在の作業ツリー）で行う．ブランチの切り替え・マージ・コミットはしない．
- 確認内容：
  - `uv run python scripts/verify_lsi_embedding.py` の全項目の結果と，実コーパスでの累積寄与率の出力．
  - `uv run python scripts/run_experiments.py --config configs/default.yaml --features D --chunk-chars 500 --run-name review_d_L500` を実行し，`presentation/slides_d.md` の「C と D の比較」表の D の列（間引きあり，5 シード平均，重心法， $L = 500$ ）と比較．
    - Intra-work の作家別正解率の平均 0.957 ± 0.016，作家別 MAP の平均 0.533 ± 0.012
    - Inter-work の作家別正解率の平均 0.296 ± 0.008，作家別 MAP の平均 0.292 ± 0.001
    - あわせて，ヒートマップの値（Inter-work，間引きあり）として slides_d.md に書かれた寺田の対角 0.176，太宰の行の太宰 0.073・漱石 0.111 と，`report.md` のヒートマップ図の値を比較．
  - `run_experiments.py` の実行時間と，Inter-work の $L = 1000$ ， $K = 64$ （ $n_{\min} = 25$ で Train が 100 チャンク）で `n_components` の上限検査に掛からないかを，`LSIEmbeddingExtractor(n_components=64)` を $L = 1000$ の `lowo_terada_1` の間引きあり Train（シード 42）で fit して確認（スクリプトは scratchpad に置く）．

## Target File Paths

- なし（リポジトリのファイルは変更しない．出力は `outputs/review_d_L500/` ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `scripts/` ・ `presentation/` ・ `configs/` のファイルを変更しない．
- 不具合を見つけても修正せず，箇所と再現手順を報告する．
- コミット・ブランチの切り替えをしない．`plans/wbs.md` は変更しない．

## Verification Criteria

- [ ] `verify_lsi_embedding.py` の出力をそのまま報告．
- [ ] `review_d_L500` の該当行の値と slides_d.md の値の対応表（一致・不一致）を報告．
- [ ] $L = 1000$ ， $K = 64$ の fit の結果（成功または例外の内容，$N$ ， $M$ ，$\sigma_{64} / \sigma_1$ ）を報告．
- [ ] `run_meta.json` の `git` の内容と実行時間を報告．
