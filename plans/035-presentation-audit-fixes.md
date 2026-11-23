# 035: 034 の監査で見つかった発表資料の不足・不一致の修正

## Expected Deliverables & Design Policy

- 034 の監査（`outputs/audit_034/checks.csv` ・ `checks_text.csv` ，`plans/wbs.md` の 034 のメモ）で見つかった項目を修正する．値は出典（`outputs/full_006/` ，`verify_qa_points.py` 等の出力）から丸め前の値を取り，資料の桁で一度だけ丸める（二重丸めをしない）．
- 不一致の修正：
  - 「全条件で同点 0 件」（storyline の評価データの定義・Slide 15・推論方式の比較の注，qa Q12）：「本採用の A～D の全条件で 0 件．アブレーションの一部の $k$-NN（`A[blocks=symbol]` の $L = 500$ ，B の `words` の付属語・内容語の $L = 100$ ）で 0.2～4.0 件（シード平均）」の旨に修正．値は `full_006/summary.csv` から．
  - A の `blocks=symbol` の Intra-work → Inter-work の作家別 MAP の平均の低下（storyline Slide 5-6，qa Q6）：−0.008 → 丸め前の差から再計算（−0.007）．
  - D の割り増し係数の比の範囲（qa 追加質問「漱石への偏り」）：丸め前の値から再計算（0.92～1.43）．
  - `slides_d.md` の図のバイト一致の記述：031 の再生成後の版（`outputs/full_006_figs_ja/figures/` ）を出典とする記述に修正．
  - qa Q10 の $L = 1000$ の上位 10 語の出現率の範囲：0.998～1.000 に修正．
  - storyline Slide 5-6 の字種構成比 2 値（『柿の種』の漢字，『田舎教師』のカタカナ）：丸め前の値から 3 桁で丸め直し．同じ表の他の値も同じ方法で確認．
- 値なしの修正：
  - Slide 2 の表 1 に，コーパス全 30 作品の作家別作品数（`verify_qa_points.py` の Q1 の出力から転記）を追加．
  - Slide 1 の班員一覧・役割分担表，Slide 1 の概念図，Slide 3 の図 1 は資料から埋められないため，storyline の該当箇所に「スライド作成時に作成（研究者・スライド作成担当）」と明記．グラフ A・C・F も「値は本資料の表，図はスライド作成時に作成」と明記．
- 表現の修正：
  - Slide 1a・qa Q14 の講義 p.14 の「再現」：「上位の語の顔ぶれは一致，順位は異なる（例：講義で 2 位の「だ」が本コーパスの付属語では 9 位）」の旨に修正．
  - Slide 9-10・qa Q8 の idf = 0 の語数：間引きなしの Train で数えた値である旨を併記．
  - qa Q2 の『三四郎』の「だいぶ」：表層の出現数（40）と形態素の数（38）の違いを明記．
  - `chunk_length_curves.png` ：作家別 MAP の平均のパネルに，間引きあり・なしそれぞれの基準値の線（`summary.csv` の `map_random` ）を追加して再生成（`scripts/plot_results.py` の該当パネル，031 と同じく `outputs/full_006_figs_ja/` に再生成し `presentation/figures/` を置換）．描画データの他の要素は変えない．図の凡例が線に重なる場合は凡例の位置を調整（`core_chunk_length.png` も同様に調整してよい．値は変えない）．
  - `presentation/README.md` の図の注意を，修正後の状態に合わせて更新．
- `presentation/` の資料の他の記述は変更しない．

## Target File Paths

- `presentation/presentation_storyline.md`
- `presentation/qa_defense_matrix.md`
- `presentation/slides_d.md`
- `presentation/README.md`
- `scripts/plot_results.py` （基準値の線と凡例の位置のみ）
- `presentation/figures/chunk_length_curves.png` ， `presentation/figures/core_chunk_length.png` （再生成で置換）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `outputs/full_006/` ・ `src/` ・ `configs/` を変更しない．
- 新しい主張を加えない．数値は出典から転記．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従う．
- コミットしない．

## Verification Criteria

- [ ] 034 の監査スクリプト（`outputs/audit_034/audit_check.py` ・ `audit_check_text.py` ）を修正後の資料で再実行し，不一致 0・値なしは「スライド作成時に作成」と明記した 3 項目（とグラフ A・C・F）のみであることを報告（スクリプトの期待値の更新が必要なら，その差分を報告）．
- [ ] 再生成した 2 図の描画データが，追加した基準値の線と凡例の位置以外で旧図と一致（031 と同じ方法で数値比較）．基準値の線の値が `summary.csv` の `map_random` と一致．
- [ ] 修正した各箇所の修正前・修正後の対応表を報告．
