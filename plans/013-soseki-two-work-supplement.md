# 013: 作品数の偏りの補助実験（漱石 2 作品版）

## Expected Deliverables & Design Policy

- **目的**：作品単位分割（LOWO）では Test 作品の作家だけ Train の作品数が 1 つ少なく，漱石（3 作品）の重心が他作家のチャンクを引き寄せている可能性がある（012 の C 単独実験で，Pooled 混同行列の太宰 211 / 330，花袋 262 / 371 が漱石と予測）．漱石を他作家と同じ 2 作品にしたとき，漱石への予測集中が解消するかを検証する．
- **位置付け**：主実験は 9 作品のまま（「2 作品以上の全作家」という選定基準を維持）．本チケットは補助実験．どの作品を除くかの恣意性を避けるため，漱石から 1 作品ずつ除いた 3 通り（『坊っちゃん』除外，『こころ』除外，『三四郎』除外．各 8 作品）をすべて実行・報告する．
- **実装**：`scripts/run_experiments.py` に作品除外オプション `--exclude-works <作品名> ...` を追加．
  - 除外はコーパス読み込み直後（チャンク分割の前）に適用し，Intra-work / Inter-work の両方に効く．
  - 存在しない作品名は `ValueError` で停止．除外の結果，作品が 2 未満になる作家が生じる場合も停止．
  - 除外した作品名を `run_meta.json` に記録．
  - オプション未指定時の挙動・出力は従来と完全に同一．
- **実行**：実装済みの特徴量（ `base` と C）について，3 通り × 2 特徴量を実行．run 名は `excl_<作品の略称>_<特徴量>` （例：`excl_bocchan_C` ， `excl_kokoro_base` ， `excl_sanshiro_C` ）．比較対象は 9 作品版（C は既存の `c_only` ，`base` は本チケットで `full_base` として実行）．
- **報告する量**（Inter-work の Pooled，各 $L$ ，重心法と `per_author` の $k = 5$ ）：
  - Accuracy，マクロ F1，作家別の再現率
  - 漱石と予測された割合（全 Test チャンク中，および漱石以外の作家のチャンク中）
  - 多数派クラス（全チャンク中で最多の作家）の割合（全予測をその作家とした場合の Accuracy）
  - 9 作品版との差分の一覧表
- A・B・D は班員の実装後に 006 で同じ 3 通りを実行する（本チケットの対象外）．

## Target File Paths

- `scripts/run_experiments.py` （ `--exclude-works` オプションの追加のみ）
- `scripts/summarize_exclusion.py` （新規．上記の報告量を各 run の出力から集計して表にする）

## Mandatory Constraints

- Comply with `.claude/rules/research-coding.md` (loaded automatically).
- `src/` ， `configs/` は変更しない．
- オプション未指定時に既存の出力と数値が一致することを確認（後方互換）．
- 数値は実行結果からの転記のみ．
- コミットしないこと（Orchestrator がレビュー後に別途コミット）．

## Verification Criteria

- [ ] `--exclude-works` 未指定の `--features C --chunk-chars 500` の実行結果の `summary.csv` が `outputs/c_only/summary.csv` の $L = 500$ の行と数値一致．
- [ ] 存在しない作品名，および漱石の 2 作品を同時に除外する指定で `ValueError` ．
- [ ] 6 run（3 通り × 2 特徴量）と `full_base` が各 10 分以内に完走し，所要時間を報告．
- [ ] `uv run python scripts/summarize_exclusion.py` が上記の報告量と 9 作品版との差分表を出力．観測値そのものを報告．
