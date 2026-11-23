# 031: 発表用の図の表記を研究の用語に合わせる

## Expected Deliverables & Design Policy

- 背景：`presentation/figures/` の図（`outputs/full_006/figures/` からのコピー）の軸・凡例・タイトルに，使わないと決めた指標名（「Balanced accuracy」「Macro-F1」）やコード上の名前（`intra` ， `inter` ， `bal` ， `centroid` ， `function / words` 等）が描かれている．呼称の方針は `plans/research-context.md` の「主評価指標」（2026-11-18 決定）．
- `scripts/plot_results.py` の図の文字列を日本語の研究用語に変更．
  - 指標：作家別 MAP の平均，作家別正解率の平均，作家別 F 値の平均，MRR，$\Delta S$ 等（README の「実験用語の定義」と一致）．
  - 分割・条件：作品内分割（Intra-work）/ 作品間分割（Inter-work），間引きあり / なし，重心法，作家別上位 $k$ 平均 等．
  - 主結果の凡例：付属語・語ブロック，内容語・語ブロック，付属語・結合，内容語・結合．
  - 作家名：漱石・太宰・寺田・花袋．ヒートマップの軸は「照会（Test）作家」「登録（Train）作家」．
  - 基準値の横線には「基準値」の凡例．
- 日本語フォント：`configs/default.yaml` の `report` に候補リスト（例：`font_candidates` ）を追加し，実行環境で最初に見つかったものを使う（macOS・Windows の班員を想定し，Hiragino Sans，Yu Gothic，Meiryo，Noto Sans CJK JP 等）．候補が 1 つも見つからない場合は停止（豆腐文字の図を黙って出さない）．使ったフォント名を `report.md` に記録．
- 既存 run の出力から図だけを再生成できるようにする（実験の再実行なし）．出力先は新しいディレクトリ（例：`outputs/full_006_figs_ja/` ）とし，`outputs/full_006/` を上書きしない．`report.md` 内の図の参照先も整合させる．
- `presentation/figures/` の対象の図（`chunk_length_curves.png` ， `confusion_best.png` ， `core_chunk_length.png` ， `dist_{A,B,C,D}_L500_inter_bal_cosine.png` ， `heatmap_{A,B,C,D}_L500_inter_bal_cosine.png` ）を再生成した図で置き換える（ファイル名は維持）．
- `scripts/plot_d_k_sweep.py` の出力（`presentation/figures/d_k_sweep.png` ， `d_heatmap_L500_inter_bal.png` ）も同じ基準で確認し，該当する文字列があれば同様に修正・再生成．
- `presentation/qa_verification_instructions.md` の冒頭に 1 行の注記を追加：「本ファイルの数値は 2026-11-18 時点の旧値．最新の数値は `presentation/qa_defense_matrix.md` を参照」（研究者の指示による．007 の「変更しない」の例外，本文は変更しない）．

## Target File Paths

- `scripts/plot_results.py`
- `scripts/plot_d_k_sweep.py` （該当する文字列がある場合のみ）
- `configs/default.yaml` （ `report` セクション）
- `presentation/figures/` （上記の図の置き換え）
- `presentation/qa_verification_instructions.md` （冒頭の 1 行のみ）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 図の数値（描く値・誤差棒・色の対応）は変えない．文字列とフォントのみの変更．
- `outputs/full_006/` を変更しない．
- `src/` を変更しない．
- コミットしない．

## Verification Criteria

- [ ] 図の再生成コマンドが完走し，使用フォントが `report.md` に記録される．存在しないフォントだけを候補にした設定で停止することを確認．
- [ ] 再生成した図と旧図の描画データ（各図の線・棒・セルの値）が一致することを，図を描く前の数値配列の比較で確認（目視ではなく数値で）．
- [ ] 置き換えた全図を画像として確認し，「Balanced」「Macro」「intra」「inter」「bal」「centroid」等の英語のコード名が残っていないこと，日本語が豆腐文字になっていないことを報告（図ごとに一言）．
- [ ] `git grep -n "Balanced\|Macro-F1" -- scripts/plot_results.py scripts/plot_d_k_sweep.py` で図の文字列としての残存が 0 件（コード内の列名 `balanced_accuracy` 等は対象外）．
- [ ] `presentation/qa_verification_instructions.md` の差分が冒頭 1 行の追加のみ．
