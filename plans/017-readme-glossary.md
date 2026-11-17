# 017: README の実験用語の定義と現行設計への追従

## Expected Deliverables & Design Policy

- **読者**：班員（担当 A・B・D の実装者を含む）．会話の経緯を知らない班員が README だけで実験の用語と設計を正しく理解できること．
- **用語の定義の節を追加**：README に「実験用語の定義」の節を新設し，このプロジェクトでの定義を記載．定義は `plans/research-context.md` を正とし，矛盾させない．各用語に，意味・具体例（作品名やチャンク数等）・関連する設定キーやコード上の名前を簡潔に記す．対象は少なくとも次の用語．
  - チャンク，チャンク長 $L$ ，端数チャンクの破棄
  - Train（登録）チャンク / Test（判定）チャンク，「学習」（ `fit` ）の意味（重み更新ではなく語彙・IDF・SVD 等の算出）
  - Intra-work（同一作品内分割）：各作品のチャンクを無作為に Train / Test に分ける．トピック漏洩あり．シード 42～46 の 5 回
  - Inter-work（作品単位分割，LOWO）：1 作品を丸ごと Test にし残りを Train にする．9 fold．例として『坊っちゃん』を Test にした fold の Train 構成
  - fold
  - トピック漏洩，擬似反復（Hurlbert (1984)）と，Intra / Inter の両方を評価する理由（C の $L = 500$ 重心法で Intra 0.955 / Inter 0.351 の例．出典 run 名 `c_only` ，Intra は暫定値と明記）
  - Train チャンク数の作家間均等化（間引き）， $n_{\min}$ ，間引きあり・なしの両方を報告すること，その理由（漱石への予測集中，013～015）
  - 重心法，$k$-NN 法（ `per_author` / `global` ）
  - 同点（判別不能）の扱い
  - $\mu_{\text{same}}$ ， $\mu_{\text{diff}}$ ， $\Delta S$ （ `delta_s_macro` / `delta_s_micro` ）
  - Accuracy，マクロ F1，MRR，Pooled Out-of-Fold
  - 本採用（既定）とアブレーション
  - ベースライン `base` （文字 unigram 相対頻度）
- **現行設計と矛盾する既存記述の修正**：README の次の記述を `plans/research-context.md` と `plans/002～007` の現行内容に合わせて修正．
  - 担当 A の比較検証：Min-Max 正規化を改善策とする記述 → 既定は相対頻度のみ（ `scaling=none` ），比較条件は `scaling=minmax` ， `blocks=bigram` ， `blocks=symbol`（002）
  - 類似度計算「3 種」 → コサインのみ
  - 質疑応答の「Jaccard と Dice の単調増加関係」等の旧項目 → 現行の `presentation/qa_verification_instructions.md` の項目に合わせる
  - 「Ticket N」表記 → `plans/00N-*.md` 形式．担当 C はリーダー担当（004 は完了済み）．担当 B の比較条件に内容語対照（ `target_pos=[名詞, 動詞, 形容詞]` ）を追加．担当 D は C の TF-IDF を入力に使う（005）
  - スライド数・部構成 → 現行の `presentation/presentation_storyline.md` に合わせる
  - 評価の説明に，間引きあり・なしの両方を報告することを反映
- 上記以外の記述（環境構築手順等）は変更しない．

## Target File Paths

- `README.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- 数値は `outputs/` の実行結果または `plans/wbs.md` の Review Notes からの転記のみ．出典（run 名またはチケット番号）を併記．
- 班員の実装方法（使うツール等）を制限する記述は書かない．
- 体言止め，全角の「，」「．」，半角英数字・数式の前後に半角スペース．新設する見出しには番号を付けない．
- 個人名・メールアドレス・ローカルの絶対パスを書かない（作家名・作品名は可）．
- `references/` の中身は読まない．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] 上記の全用語が「実験用語の定義」の節に含まれることを，用語ごとの行番号の一覧で報告．
- [ ] `grep -n "Jaccard\|Dice\|Min-Max\|minmax\|3 種\|Ticket" README.md` の出力をそのまま報告し，残った一致がすべて現行設計と矛盾しない記述（例：比較条件としての `minmax` ，不採用理由としての Jaccard）であること．
- [ ] README 中の数値と出典の対応表を報告．
- [ ] README の用語定義と `plans/research-context.md` の記述の対応表（用語，README の行，research-context の該当箇所）を報告．
