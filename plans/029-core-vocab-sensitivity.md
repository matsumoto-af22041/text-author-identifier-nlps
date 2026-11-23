# 029: 主結果の語彙選択に対する感度分析スクリプト

## Expected Deliverables & Design Policy

- 背景：028 の調査（`outputs/investigate_028/` ，git 管理外）で，Inter-work の付属語 − 内容語（B，`blocks=words` ）の作家別 MAP の平均の差が，内容語の語彙選択に依存して +0.006～+0.033（ $L = 500$ ）の範囲で動くと判明．`plans/research-context.md` の「主結果の報告方針」（2026-11-23）により，この感度を発表で示すため，漏洩のない条件のみを再現可能なスクリプトとしてリポジトリに置く．
- 新規スクリプト `scripts/sensitivity_core_vocab.py` ．028 の `recompute.py` ・ `analyze.py` ・ `own_only_test.py` を基に，次の条件だけを扱う．
  - `bal` ：fit・検索対象とも間引き後（主結果と同じ）
  - `nobal` ：fit・検索対象とも全件（間引きなしと同じ）
  - `fitfull_poolbal` ：語彙・重み（fit）は fold の Train 全件，検索対象と重心は間引き後
  - `fitbal_poolfull` ：fit は間引き後，検索対象は全件
  - `drop_single_author` ：間引き後の Train で fit した内容語の語彙から，Train 中で 1 作家にしか出ない語を除く（Test の作家情報は使わない）
  - 028 の「Test 作家の Train 側作品にしか出ない語を除く」診断は Test の情報を使う漏洩を含むため実装しない．
- 出力：$L \in \{100, 500, 1000\}$ ごとに，各条件の付属語・内容語の作家別 MAP の平均，作家別 AP（4 作家），重心法の作家別正解率の平均，差（付属語 − 内容語）．5 シードの平均 ± 標準偏差（`nobal` は 1 回）．`outputs/<--out-dir>/` に CSV と `report.md` ．
- `bal` と `nobal` の値が `--baseline-run` で指定した run（例：`full_006` ）の `summary.csv` の対応する行と一致することをスクリプト内で検査し，不一致なら停止．
- 間引き・分割・抽出器・評価器は `src/` の既存の関数を使う．run 名・出力先・$L$ ・シードは CLI 引数（既定値は `configs/default.yaml` ）．
- `drop_single_author` は付属語にも同じ規則を適用した値を併記（統制のため）．

## Target File Paths

- `scripts/sensitivity_core_vocab.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ・ `configs/` ・既存の `scripts/` を変更しない．
- Test の情報（作家ラベル・作品）を fit・語彙選択に使わない．
- `data/raw/` のコーパスを直接読まない．
- 既存の `--out-dir` を上書きしない．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/sensitivity_core_vocab.py --baseline-run full_006 --chunk-chars 500 --out-dir outputs/sensitivity_029_L500` が 10 分以内に完走し，`bal` ・ `nobal` の作家別 MAP の平均・作家別 AP・作家別正解率の平均が `full_006` と一致（誤差 1e-12 以内）．
- [ ] 同出力の付属語 − 内容語の作家別 MAP の平均の差が，028 の報告値（ $L = 500$ ：`bal` +0.033，`nobal` +0.004，`fitfull_poolbal` +0.006，`fitbal_poolfull` +0.029，1 作家語除外の内容語 0.3610）と一致することを報告．
- [ ] $L = 100$ と $L = 1000$ についても完走するか，10 分を超える場合は Full Experiment にコマンドを記載．
- [ ] 漏洩がないことの説明（各条件で fit に使うデータ）をスクリプトの docstring と `report.md` に記載．

## Full Experiment (optional, run by the researcher)

- **Command**：$L = 100$ が 10 分を超える場合のみ．Executor が用意．
- **Outputs**：`outputs/sensitivity_029_L*/`
