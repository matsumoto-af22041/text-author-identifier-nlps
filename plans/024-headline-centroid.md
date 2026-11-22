# 024: 図表の代表条件を重心法に変更

## Expected Deliverables & Design Policy

- 目的：`plans/research-context.md` の「推論方式」の代表条件（2026-11-23 決定）に合わせ，`report.md` の図表の代表条件を `knn5_per_author` から重心法に変更．
- `configs/default.yaml` の `report.headline.classifier` を重心法を表す値に変更．値の文字列は `scripts/run_experiments.py` が `summary.csv` の分類器列に書く重心法の名前と一致させる（コードで確認して決める）．
- `report.best_model_metric` と最良モデルの選択方法は変更しない．
- README や `presentation/` に代表条件の推論方式の記述があれば合わせて修正（なければ変更不要）．

## Target File Paths

- `configs/default.yaml`
- `README.md` （該当箇所がある場合のみ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 作業は `main` ブランチで行う（D のブランチのマージ後）．コミットはしない．
- `src/` ・ `scripts/` は変更しない．代表条件の値がコードの都合で設定できない場合は変更せずに報告．

## Verification Criteria

- [ ] `uv run python scripts/run_experiments.py --config configs/default.yaml --features base --chunk-chars 500 --run-name smoke_024` が完走し，`outputs/smoke_024/report.md` の「代表条件（headline）」の行が重心法を示し，「代表条件 … が今回の実行範囲に含まれない」の注意が出ないこと．
- [ ] 同 `report.md` の特徴量比較表の `base` の Inter-work（間引きあり）の値が，`outputs/smoke_024/summary.csv` の重心法の行の値と一致することを報告（値を記載）．
- [ ] `git diff` が `configs/default.yaml` （と該当時の README）の該当行のみであること．
