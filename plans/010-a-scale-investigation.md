# 010: 特徴量 A のブロック別スケール事前調査

## Expected Deliverables & Design Policy

- 目的：002 の既定仕様（文字 Bi-gram と記号の相対頻度のみ，字種構成比を除外，正規化なし）で，特定のブロックがコサイン類似度を支配しないかを実測し，002 の仕様を確定させる．
- 調査対象の 3 構成（いずれも「出現回数 ÷ 総文字数」の相対頻度）：
  - (i) 案 C（002 の既定）：文字 Bi-gram 上位 1000 + 記号 3 次元（読点・文末記号・括弧類）
  - (ii) 旧案：(i) + 字種構成比 4 次元（漢字・ひらがな・カタカナ・その他）
  - (iii) (ii) に各次元の Min-Max 正規化を適用
- 算出する量（ $L \in \{100, 500, 1000\}$ ごと，全チャンク対象）：
  - 各ブロック（Bi-gram，記号，字種）の二乗ノルムがベクトル全体の二乗ノルムに占める割合 $\frac{\|\vec{x}_{\text{block}}\|^2}{\|\vec{x}\|^2}$ の平均・中央値・最小・最大
  - 各ブロックの次元ごとの値の中央値と最大値
  - 文字 Bi-gram 上位 1000 種が全 Bi-gram 出現に占める被覆率
  - 参考：全チャンク対のコサイン類似度について，(i) と「Bi-gram ブロックのみ」の相関係数，(ii) と「字種ブロックのみ」の相関係数（支配の有無を類似度の水準で確認．計算量が大きい場合は 500 チャンクの無作為抽出， `seed=42` ）
- 調査用の使い捨てスクリプトとして実装し， `src/features/` には何も追加しない（特徴量 A の実装は班員の担当）．

## Target File Paths

- `scripts/investigate_a_scale.py` （新規）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/features/` ， `configs/default.yaml` は変更しない．
- コーパスは `src.dataset.load_corpus` と `build_chunks` 経由で読み込み，生コーパスを直接読まない．
- 本調査は記述統計であり評価ではないため，語彙と Min-Max の範囲は全チャンクから算出してよい（評価パイプラインには使わないことをスクリプト冒頭のコメントに明記）．
- `numpy` のみ．乱数は `seed=42` 固定．

## Verification Criteria

- [ ] `uv run python scripts/investigate_a_scale.py` が 10 分以内に完走し，上記の量を $L$ ・構成ごとの表として標準出力に出す．
- [ ] 報告には観測値そのもの（各ブロックの二乗ノルム割合の平均・中央値・最小・最大）を含めること．
