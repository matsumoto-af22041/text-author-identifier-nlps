# 012: 特徴量 C の検証スクリプト整備・単独実験・発表資料への反映

## Expected Deliverables & Design Policy

- 004 の完了を受け，C に関する事実を再現可能な形で確定させ，`presentation/` に反映する．
- **検証スクリプトのリポジトリ化**：004 の Executor が scratchpad に置いた検証スクリプト（ `/private/tmp/claude-501/-Users-miharu-ghq-github-com-matsumoto-af22041-text-author-identifier-nlps/9fc55c37-7f31-4f44-9a94-e3fd7169f9f5/scratchpad/verify_004.py` ．存在しなければ 004 の Verification Criteria から再作成）を `scripts/verify_word_tfidf.py` に移し，改訂後の 004 の Verification Criteria（トイデータの手計算一致，未知語，内容語の表記揺れの統一，「大分」の内訳）を再現できるようにする．
- **作品別の表記揺れの集計（追加）**：`scripts/verify_word_tfidf.py` に，9 作品それぞれについて次の表層形の出現数と，それらが統一される `lemma` を集計する処理を追加．底本の表記方針による交絡の具体例として発表に使う．
  - 「云う」系（表層形が「云」で始まる動詞）と「言う」系（「言」で始まる動詞），および仮名書き「いう」系
  - 「解る」系，「分かる」「分る」系，仮名書き「わかる」系
  - 「事」（名詞）と「こと」（名詞）
- **C 単独実験**：`uv run python scripts/run_experiments.py --config configs/default.yaml --features C --run-name c_only` （全チャンク長，Intra-work / Inter-work）を実行し，図表とレポートを生成（ `run_experiments.py` が自動生成しない場合は `scripts/plot_results.py` を続けて実行）．
  - 確定値として扱うのは Inter-work（LOWO 9 fold，決定的分割）の Pooled 指標．
  - Intra-work は現状 `seed=42` の 1 回分割のため「暫定（006 で 5 シード化）」と明記．
- **発表資料への反映**（ `presentation/` ）：上記で得た値のみを転記し，各値に出典（スクリプト名または run 名 `c_only` ）を併記．
  - `presentation/presentation_storyline.md` ：
    - Slide 9-10（担当 C）：表記統一の例を「大分 / だいぶ」から内容語の揺れ（「云う / 言う」等）に差し替え，作品別の集計値を提示．「大分 / だいぶ」は副詞で C の対象外であり，底本の表記差が効くのは文字表層を見る A のみ，と整理．C の Inter-work 確定値（Accuracy，マクロ F1，MRR， $\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， $\Delta S_{\text{macro}}$ ，チャンク長別）と Intra-work 暫定値を記載．
    - Slide 1a（導入）：講義第 5 回 p.15 の『坊っちゃん』上位語「云う」が，C では語彙素 `言う` に統一されることを一言添える．
  - `presentation/qa_verification_instructions.md` ：
    - Q2：「大分 / だいぶ」の統一は語彙素レベルでは成立するが副詞のため C・B の対象外であること，C での統一例（「云う / 言う」等）と作品別集計，「大分」の地名誤解析の件数（『坊っちゃん』で文単位 5 件 / 全文連結 4 件）を反映．
    - Q5：C の Intra-work（暫定）と Inter-work（確定）の比較値を反映．
    - Q8：実データで $\text{idf} = 0$ となる語の件数（ $L$ ごと，Train 全体での fit 時）を反映．
  - `presentation/note.md` ：`lemma` の統一例の節に，副詞（「大分」「矢張り」）と内容語（「言う」「分かる」「事」）の区別と，「大分」の地名誤解析の件数を追記．

## Target File Paths

- `scripts/verify_word_tfidf.py` （新規）
- `presentation/presentation_storyline.md`
- `presentation/qa_verification_instructions.md`
- `presentation/note.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ， `configs/` ， `plans/` （ `plans/wbs.md` の 012 のステータス欄を除く）は変更しない．
- 数値は実行結果からの転記のみ．創作・推測値は禁止．未計測の値は「要計測（006）」のまま残す．
- A・B・D の値や，C と他特徴量の優劣に関する結論は書かない（未実装のため）．
- 生コーパスを直接読まず，`data/processed/authors_corpus.json` または `src/dataset.py` 経由で取得．
- ドキュメントは体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペースの規約に従うこと．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] `uv run python scripts/verify_word_tfidf.py` が完走し，004 の Verification Criteria の 4 項目と作品別の表記揺れ集計を出力．
- [ ] `c_only` の実行が 10 分以内に完走し，`outputs/c_only/` に `summary.csv` ， `report.md` ，図が生成される．所要時間と，Inter-work の Pooled 指標（重心法と `per_author` の $k \in \{1, 3, 5, 10, 20\}$ ，各 $L$ ）を報告．
- [ ] 発表資料に転記した各数値が `outputs/c_only/summary.csv` またはスクリプト出力と一致することを確認し，対応表を報告．
- [ ] 3 文書に「大分 / だいぶ」を C の表記統一の例とする記述が残っていないことを `grep` で確認（A の交絡例としての言及は可）．

## Full Experiment (optional, run by the researcher)

- 該当なし（C 単独実験は 10 分以内の見込みのため Executor が実行．超過する場合は実行を中断し，コマンドと見積もり時間を報告）．
