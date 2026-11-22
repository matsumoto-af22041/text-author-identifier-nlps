# 026: 形態素解析キャッシュの共通化（B・C・D）

## Expected Deliverables & Design Policy

- 背景：025 で B（`src/features/pos_function.py` ）にテキスト単位のプロセス内キャッシュ `morphemes_cached` を追加し，数値不変のまま実行時間を約 1/5 にした（403 秒 / 4 条件 → 125 秒 / 7 条件）．C（`src/features/word_tfidf.py` ）は fit / transform のたびに `fugashi.Tagger` で解析し直しており，C を内部で使う D（`src/features/lsi_embedding.py` ）も同様．研究者の決定（2026-11-23）により，今後は形態素解析を使うすべての特徴量でキャッシュを効かせる．
- 形態素解析とキャッシュを共通モジュール `src/morphology.py` に移す．
  - B の `Morpheme` （ `surface` ， `lemma` ， `pos1` ， `pos2` ，`lemma` が `None` の未知語は表層形）， `_shared_tagger` ， `morphemes_cached` を移動し，B はそれを import する．
  - C の `analyze` は `morphemes_cached` の結果から `(pos1, lemma)` を作るよう変更（`lemma or surface` の扱いは現行と同じ）．
  - D は C を経由するため変更不要の見込み．D が独自に解析している箇所があれば同様に置き換える．
  - 新しい特徴量やスクリプトで形態素解析を使う場合は `src/morphology.py` を使う旨を同モジュールの docstring に記載．
- キャッシュは解析結果のみを保持し，語彙・IDF・SVD 等の Train 依存の量は保持しない（リーク防止）．その旨をモジュールの docstring に明記．
- `README.md` のディレクトリ構成（ `src/` の説明）に `morphology.py` を追記．`plans/research-context.md` は Orchestrator が更新する．

## Target File Paths

- `src/morphology.py` （新規）
- `src/features/pos_function.py`
- `src/features/word_tfidf.py`
- `src/features/lsi_embedding.py` （必要な場合のみ）
- `scripts/verify_pos_function.py` （import 先の変更が必要な場合のみ）
- `README.md`

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- B・C・D の出力は変更前と数値が完全一致すること．
- 外部機械学習ライブラリは禁止．
- `data/raw/` のコーパスを直接読まない．
- コミットしない．

## Verification Criteria

- [ ] `uv run python scripts/verify_pos_function.py` と `uv run python scripts/verify_lsi_embedding.py` が完走し全項目合格．C の検証スクリプトがリポジトリにあれば同様に実行．
- [ ] 数値一致：`uv run python scripts/run_experiments.py --config configs/default.yaml --features C D --chunk-chars 1000 --run-name smoke_026` の `summary.csv` が，変更前のコード（`git stash` 等は使わず，`git worktree` も使わない．変更前の `summary.csv` を先に同じコマンドの別 run 名 `smoke_026_before` で作成しておく）と，`run_meta` 由来の列を除き全行一致．
- [ ] 同 2 run の所要時間（`timings.csv` または実時間）と最大 RSS を報告．
- [ ] `git grep -n "fugashi.Tagger" -- src/` が `src/morphology.py` のみ．
