# 046: 提出前の修正（実験スクリプトの不具合，提出から外すスクリプト）

## Expected Deliverables & Design Policy

- 背景：043 の報告への研究者の決定（2026-11-24）．
- **`scripts/run_experiments.py` の不具合の修正**：721 行目付近の `make_report(run_dir)` が，`scripts/plot_results.py` の現在の関数 `make_report(run_dir, out_dir, config_path)` と引数が合わず，`--no-report` なしで実行すると結果の保存後に TypeError で停止する．呼び出しを現在の関数に合わせる（例：`make_report(run_dir, run_dir, args.config)` ．`plot_results.py` を CLI で実行したときと同じ出力先・設定になるよう確認）．数値の計算には触れない．
- **Git がない環境での停止はコードを変えない**（研究者決定）．043 の README の `git init` の手順で回避．
- **提出から外すスクリプト**：`scripts/make_submission.py` の除外リストに `scripts/plot_slide_figures.py` （提出しない `presentation/` に依存）と `scripts/verify_pos_function.py` （開発時の Git の履歴に依存）を追加．`submission/README.md` の該当の注記を更新．
- **残りのスクリプトの分類（報告のみ，除外は研究者の判断後）**：zip に入る `scripts/` の各ファイルを次の 3 つに分類し，根拠とともに表で報告．
  - (a) 実験の実行・結果の出力に必要（前処理，主実験，図・レポート）．
  - (b) 発表資料・ storyline ・ qa に載っている数値・図を出力する補助実験・集計（`presentation/presentation_storyline.md` ・ `presentation/qa_defense_matrix.md` ・ `presentation/slides_content.yaml` の `runs` ・ノートでスクリプト名や出力を検索して対応を確認）．
  - (c) 自作の実装の正しさの検証（参照実装との一致の確認等）．
  - (d) 開発時の調査のみで，発表・結果に使われていない（例：辞書の項目の調べ）．
  - 各スクリプトについて，発表資料のどこ（スライド番号・補助スライド・ qa の質問番号）の数値に対応するかを書く．(d) と判断したものは除外の候補として提案（除外はしない）．
- コミットしない（研究者の判断の後に，043 ・ 046 をまとめてコミットし，本番の zip を作る）．

## Target File Paths

- `scripts/run_experiments.py` （`make_report` の呼び出しのみ）
- `scripts/make_submission.py` ， `submission/README.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `run_experiments.py` の計算・乱数・出力の数値に関わる部分を変更しない．
- `src/` ・ `configs/` ・ `presentation/` ・ `plans/` を変更しない（wbs の状態欄を除く）．
- 生コーパスを全文出力しない．個人名を出力しない．コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/run_experiments.py --chunk-chars 1000 --run-name smoke_046` （`--no-report` なし，10 分以内）が終了コード 0 で完走し，レポート（report.md と図）が出力される．summary.csv が `outputs/full_006` の同じ条件と一致（最大差 0）．確認後 `outputs/smoke_046` は残してよい（Git 管理外）．
- [ ] `make_submission.py` の除外リストに 2 ファイルが入ったことを，未コミットでも動く試作の方法（043 と同じ scratchpad のクローン）で zip の一覧から確認．
- [ ] スクリプトの分類表（(a)～(d)，根拠，発表資料での対応箇所）．
