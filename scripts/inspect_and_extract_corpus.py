"""青空文庫サンプルコーパスの構造検査と対象 4 作家・9 作品の抽出．

生コーパスは行単位のストリーム処理でのみ走査し，全 30 作品のメタ情報（作品名・著者名・行数・文字数），
文字コード，新旧仮名遣いの状態を検査した上で，対象作品の整形済みデータを JSON として保存．

コーパス構造（docs/AozoraCorpus.pdf 準拠）：
    作品名（1 行目）
    著者名（2 行目）
    本文（句点ごとに改行済み）
    空行（作品間に 1 行）
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Work:
    """1 作品分のブロック（ヘッダ 2 行と本文行）．"""

    index: int
    title: str
    author: str
    sentences: tuple[str, ...]

    @property
    def num_lines(self) -> int:
        return len(self.sentences)

    @property
    def num_sentences(self) -> int:
        # 配布仕様で句点改行済みのため，本文 1 行を 1 文として計数
        return len(self.sentences)

    @property
    def num_chars(self) -> int:
        return sum(map(len, self.sentences))


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stream_lines(path: Path, encoding: str) -> Iterator[str]:
    """改行コードと BOM を除去した行をストリームで返すイテレータ．"""
    with path.open(encoding=encoding, errors="strict", newline=None) as f:
        for line in f:
            yield line.rstrip("\n").lstrip("﻿")


def detect_encoding(path: Path, candidates: list[str]) -> str:
    """候補エンコーディングで全行を厳密にストリームデコードし，最初に完走したものを採用．"""

    def decodes(encoding: str) -> bool:
        try:
            collections.deque(stream_lines(path, encoding), maxlen=0)
        except UnicodeDecodeError as e:
            print(f"[encoding] {encoding}: 不可 ({e.reason} at byte {e.start})")
            return False
        print(f"[encoding] {encoding}: 全行デコード可")
        return True

    detected = next((enc for enc in candidates if decodes(enc)), None)
    if detected is None:
        raise ValueError(f"いずれの候補エンコーディングでもデコード不可: {candidates}")
    return detected


def detect_newline(path: Path) -> str:
    with path.open("rb") as f:
        first = f.readline()
    return "CRLF" if first.endswith(b"\r\n") else "LF"


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def parse_works(lines: Iterator[str]) -> Iterator[Work]:
    """空行区切りで作品ブロックを切り出し，ヘッダ 2 行（作品名・著者名）を検証して返す．"""
    blocks = itertools.groupby(lines, key=lambda s: bool(s.strip()))
    text_blocks = (
        tuple(group)
        for is_text, group in blocks
        if is_text or _assert_single_blank(tuple(group))
    )
    return itertools.starmap(_to_work, enumerate(text_blocks))


def _assert_single_blank(block: tuple[str, ...]) -> bool:
    if len(block) != 1:
        raise ValueError(f"作品間の空行が 1 行ではない（{len(block)} 行連続）")
    return False


def _to_work(index: int, block: tuple[str, ...]) -> Work:
    if len(block) < 3:
        raise ValueError(f"作品ブロック #{index} が 3 行未満（ヘッダ 2 行 + 本文が必要）: {block[:2]}")
    title, author, *body = (s.strip() for s in block)
    if not title or not author:
        raise ValueError(f"作品ブロック #{index} のヘッダが空: title={title!r}, author={author!r}")
    return Work(index=index, title=title, author=author, sentences=tuple(body))


def inspect_orthography(work: Work, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """歴史的仮名遣い特有パターンの出現数と per_chars 文字あたり出現率を算出．"""
    text = "\n".join(work.sentences)
    patterns: Mapping[str, str] = cfg["patterns"]
    # shape: (num_patterns,)
    counts = np.array([len(re.findall(p, text)) for p in patterns.values()], dtype=np.int64)
    rates = counts / work.num_chars * cfg["per_chars"]
    assert np.all(np.isfinite(rates)), f"出現率に NaN / Inf: {work.title}"
    total_rate = float(rates.sum())
    return {
        "pattern_counts": dict(zip(patterns, counts.tolist(), strict=True)),
        "historical_rate_per_chars": cfg["per_chars"],
        "historical_rate": round(total_rate, 4),
        "kana_usage": "historical" if total_rate > cfg["historical_rate_threshold"] else "modern",
    }


def print_corpus_summary(works: list[Work]) -> None:
    print(f"\n=== 全 {len(works)} 作品のメタ情報 ===")
    print(f"{'#':>2}  {'行数':>6}  {'文字数':>8}  作品名（著者名）")
    print(
        "\n".join(
            f"{w.index:>2}  {w.num_lines:>6}  {w.num_chars:>8}  {w.title}（{w.author}）" for w in works
        )
    )
    per_author = collections.Counter(w.author for w in works)
    print(f"\n=== 作家別作品数（{len(per_author)} 名）===")
    print("\n".join(f"{n:>2}  {a}" for a, n in per_author.most_common()))


def extract_targets(
    works: list[Work], targets: list[Mapping[str, Any]]
) -> list[tuple[Mapping[str, Any], int, Work]]:
    """設定の (作家, 作品名) と完全一致する作品を (作家設定, 作家内番号, 作品) として抽出．欠落・重複で停止．"""
    occurrences = collections.Counter((w.author, w.title) for w in works)
    by_key = {(w.author, w.title): w for w in works}
    keys = [(t, i, (t["name"], title)) for t in targets for i, title in enumerate(t["works"])]
    invalid = [key for _, _, key in keys if occurrences[key] != 1]
    if invalid:
        raise ValueError(f"対象作品の欠落または重複: {invalid}")
    return [(t, i, by_key[key]) for t, i, key in keys]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fix_seed(cfg["seed"])
    corpus_cfg, ortho_cfg = cfg["corpus"], cfg["orthography"]
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_corpus"]
    out_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    if not raw_path.is_file():
        raise FileNotFoundError(f"生コーパスが存在しない: {raw_path}")

    encoding = detect_encoding(raw_path, corpus_cfg["candidate_encodings"])
    newline = detect_newline(raw_path)
    print(f"[encoding] 採用: {encoding} / 改行コード: {newline}")

    works = list(parse_works(stream_lines(raw_path, encoding)))
    if len(works) != corpus_cfg["expected_num_works"]:
        raise ValueError(f"作品数が想定と不一致: {len(works)} != {corpus_cfg['expected_num_works']}")
    print_corpus_summary(works)

    records = [
        {
            "work_id": f"{t['id']}_{i}",
            "author_id": t["id"],
            "author": w.author,
            "title": w.title,
            "corpus_index": w.index,
            "num_lines": w.num_lines,
            "num_sentences": w.num_sentences,
            "num_chars": w.num_chars,
            "orthography": inspect_orthography(w, ortho_cfg),
            "sentences": list(w.sentences),
        }
        for t, i, w in extract_targets(works, cfg["targets"])
    ]

    min_chars = corpus_cfg["min_work_chars"]
    assert all(r["num_chars"] > min_chars and r["num_sentences"] > 0 for r in records), (
        f"総文字数 {min_chars} 以下または文数 0 の作品が存在"
    )

    print(f"\n=== 対象 {len(cfg['targets'])} 作家・{len(records)} 作品（encoding={encoding}）===")
    print(f"{'work_id':<10}  {'文数':>6}  {'文字数':>8}  {'旧仮名率':>8}  判定        作品名（著者名）")
    print(
        "\n".join(
            f"{r['work_id']:<10}  {r['num_sentences']:>6}  {r['num_chars']:>8}  "
            f"{r['orthography']['historical_rate']:>8.2f}  {r['orthography']['kana_usage']:<10}  "
            f"{r['title']}（{r['author']}）  {r['orthography']['pattern_counts']}"
            for r in records
        )
    )
    historical = [r["title"] for r in records if r["orthography"]["kana_usage"] == "historical"]
    if historical and ortho_cfg["fail_on_historical"]:
        raise ValueError(f"対象作品に旧仮名遣い判定: {historical}")

    output = {
        "meta": {
            "source_path": cfg["paths"]["raw_corpus"],
            "source_sha256": sha256_of(raw_path),
            "source_encoding": encoding,
            "source_newline": newline,
            "git_commit": git_commit(),
            "config_path": str(args.config.resolve().relative_to(PROJECT_ROOT)),
            "config": cfg,
            "sentence_definition": "配布仕様（句点改行）に基づき本文 1 行を 1 文とみなす",
        },
        "corpus_summary": [
            {k: v for k, v in asdict(w).items() if k != "sentences"}
            | {"num_lines": w.num_lines, "num_chars": w.num_chars}
            for w in works
        ],
        "authors": [{"id": t["id"], "name": t["name"], "works": t["works"]} for t in cfg["targets"]],
        "works": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=1)
    print(f"\n[output] {out_path.relative_to(PROJECT_ROOT)} に保存（{len(records)} 作品）")


if __name__ == "__main__":
    main()
