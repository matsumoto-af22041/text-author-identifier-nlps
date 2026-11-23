"""作家ごとに偏りのある付属語の具体例を規則のみで抽出する記述統計（plans/030）．

評価には使わない記述統計：4 作家・9 作品の全文（全チャンク・全作品）を使うため Train / Test の区別がなく，
評価パイプライン（ scripts/run_experiments.py ）の結果には影響しない．発表の Slide 7-8 と質疑
「付属語の何が作家ごとに違うのか」の具体例用．

付属語の定義は B の既定（ configs/default.yaml の features.B.params ）と同じ：
    対象：pos1 が target_pos（既定：助詞・助動詞，第 2 回 p.11）の形態素，キーは token_key（既定：表層形）
    総形態素数：補助記号・空白を除く形態素数（ src.features.pos_function.EXCLUDED_POS ）
    形態素解析は src.morphology 経由で文単位（ Slide 1a の verify_qa_points と同じ単位，キャッシュなし）

記号（作品 j ，作家 a ，付属語 w ）：
    c_j(w), n_j           作品 j の出現回数と総形態素数
    f_j(w) = c_j(w) / n_j                                 作品の相対頻度（第 5 回 p.12）
    f_a(w) = sum_{j in a} c_j(w) / sum_{j in a} n_j       作家の相対頻度（全作品を合算）
    fbar_{¬a}(w) = (1/3) sum_{b != a} f_b(w)              他の 3 作家の相対頻度の単純平均
    r_a(w) = f_a(w) / fbar_{¬a}(w)                        偏りの比（作家の比）
    r_{a,j}(w) = f_j(w) / fbar_{¬a}(w)                    作品の比

抽出規則（手作業の取捨なし）：
    頻度の下限：4 作家合計の出現回数 sum_a sum_{j in a} c_j(w) >= --min-count
    fbar_{¬a}(w) = 0 の語は比を計算せず「他作家に出現なし」として別表（0 で置き換えない）
    一貫（高）：作家 a のすべての作品で r_{a,j}(w) > 1，一貫（低）：すべての作品で r_{a,j}(w) < 1，それ以外は不一貫
    出力：一貫（高）を r_a の降順，一貫（低）を r_a の昇順に上位 --top-n 語（同値は語の辞書順）

使い方：
    uv run python scripts/function_word_profiles.py --out-dir outputs/function_word_profiles [--min-count 100] [--top-n 8]

出力（--out-dir．既存のディレクトリには書き込まず停止）：
    unit_totals.csv          作品・作家ごとの総形態素数と付属語の延べ数
    word_counts.csv          付属語ごとの作品別出現回数と pos1 の内訳（全語，下限の適用前）
    author_ratios.csv        作家 × 語（下限を満たし fbar > 0）の f_a，fbar，r_a，一貫性の区分
    work_ratios.csv          作家 × 語 × 作品の f_j と r_{a,j}
    no_other_author.csv      下限を満たし他の 3 作家に出現しない語
    consistency_counts.csv   作家ごとの一貫（高・低）・不一貫・他作家に出現なしの件数
    top_words.csv            作家ごとの上位・下位 --top-n 語
    report.md                上記の Markdown 表，抽出規則，注記
    run_meta.json            CLI 引数・B の定義・git 状態・コーパスハッシュ
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.run_experiments import PROJECT_ROOT, corpus_meta, display_path, git_state
from src.dataset import Corpus, load_corpus
from src.features.pos_function import EXCLUDED_POS, FUNCTION_POS, validate_target_pos
from src.morphology import TokenKey, morphemes_of, shared_tagger

# 相対頻度の表示単位（report.md のみ．CSV は無次元の相対頻度）
PER_MORPHEMES = 1000


@dataclass(frozen=True)
class WorkCounts:
    """1 作品の付属語の出現回数と総形態素数．"""

    work_id: str
    author_id: str
    title: str
    counts: Counter[str]
    pos1_counts: Counter[tuple[str, str]]
    num_morphemes: int


def count_work(sentences: Sequence[str], work_id: str, author_id: str, title: str, token_key: TokenKey, pos_set: frozenset[str]) -> WorkCounts:
    """文単位で形態素解析し，対象品詞の語の出現回数と補助記号・空白を除く総形態素数を集計．"""
    tagger = shared_tagger()
    morphs = [m for s in sentences for m in morphemes_of(tagger, s)]
    targets = [m for m in morphs if m.pos1 in pos_set]
    num_morphemes = sum(m.pos1 not in EXCLUDED_POS for m in morphs)
    if num_morphemes == 0:
        raise ValueError(f"総形態素数が 0 の作品: {work_id}")
    return WorkCounts(
        work_id=work_id,
        author_id=author_id,
        title=title,
        counts=Counter(m.key(token_key) for m in targets),
        pos1_counts=Counter((m.key(token_key), m.pos1) for m in targets),
        num_morphemes=num_morphemes,
    )


@dataclass(frozen=True)
class Profiles:
    """作品・作家の出現回数行列と相対頻度．V = 付属語の異なり数（辞書順），J = 作品数，A = 作家数．"""

    words: tuple[str, ...]
    work_ids: tuple[str, ...]
    work_author: np.ndarray  # shape: (J,), 作品の作家インデックス
    work_counts: np.ndarray  # shape: (J, V)
    work_totals: np.ndarray  # shape: (J,)
    author_counts: np.ndarray  # shape: (A, V)
    author_totals: np.ndarray  # shape: (A,)

    @property
    def work_freq(self) -> np.ndarray:
        # shape: (J, V)
        return self.work_counts / self.work_totals[:, None]

    @property
    def author_freq(self) -> np.ndarray:
        # shape: (A, V)
        return self.author_counts / self.author_totals[:, None]

    @property
    def other_mean_freq(self) -> np.ndarray:
        """fbar_{¬a}(w)：自作家を除く作家の相対頻度の単純平均．shape: (A, V)"""
        f = self.author_freq
        num_authors = f.shape[0]
        return (f.sum(axis=0, keepdims=True) - f) / (num_authors - 1)


def build_profiles(works: Sequence[WorkCounts], author_ids: Sequence[str]) -> Profiles:
    words = tuple(sorted(set().union(*(w.counts for w in works))))
    index = {a: i for i, a in enumerate(author_ids)}
    # shape: (J, V)
    work_counts = np.array([[w.counts[t] for t in words] for w in works], dtype=np.float64)
    work_totals = np.array([w.num_morphemes for w in works], dtype=np.float64)
    work_author = np.array([index[w.author_id] for w in works], dtype=np.int64)
    # 作品 → 作家の所属行列．shape: (A, J)
    membership = (work_author[None, :] == np.arange(len(author_ids))[:, None]).astype(np.float64)
    author_counts = membership @ work_counts
    author_totals = membership @ work_totals
    assert work_counts.shape == (len(works), len(words)) and author_counts.shape == (len(author_ids), len(words))
    assert np.all(membership.sum(axis=1) >= 2), "各作家に 2 作品以上が必要（作品をまたいだ一貫性の判定）"
    assert np.all(author_totals > 0)
    return Profiles(words, tuple(w.work_id for w in works), work_author, work_counts, work_totals, author_counts, author_totals)


def classify(work_ratios: np.ndarray) -> np.ndarray:
    """作品の比の列から一貫性の区分．shape: (J_a, V) -> (V,)"""
    high, low = np.all(work_ratios > 1.0, axis=0), np.all(work_ratios < 1.0, axis=0)
    return np.where(high, "high", np.where(low, "low", "inconsistent"))


def pos1_breakdown(pos1_counts: Counter[tuple[str, str]], word: str) -> str:
    """'助詞:1234;助動詞:56' の形式（回数降順，同数は品詞名順）．"""
    items = sorted(((p, c) for (w, p), c in pos1_counts.items() if w == word), key=lambda pc: (-pc[1], pc[0]))
    return ";".join(f"{p}:{c}" for p, c in items)


def analyze(profiles: Profiles, works: Sequence[WorkCounts], author_ids: Sequence[str], min_count: int) -> dict[str, list[dict[str, Any]]]:
    """作家ごとの比・一貫性・別表の行を作る．"""
    p = profiles
    total = p.author_counts.sum(axis=0)  # shape: (V,)
    eligible = total >= min_count
    fbar = p.other_mean_freq  # shape: (A, V)
    f_auth, f_work = p.author_freq, p.work_freq
    assert np.all(np.isfinite(fbar)) and np.all(fbar >= 0)
    pos1_by_author = {a: sum((w.pos1_counts for w in works if w.author_id == a), Counter()) for a in author_ids}
    pos1_all = sum((w.pos1_counts for w in works), Counter())
    title_of = {w.work_id: w.title for w in works}

    def author_rows(ai: int, a: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        has_ratio = eligible & (fbar[ai] > 0)
        no_other = eligible & (fbar[ai] == 0)
        cols = np.flatnonzero(has_ratio)
        jw = np.flatnonzero(p.work_author == ai)
        # shape: (|cols|,), (J_a, |cols|)
        r = f_auth[ai, cols] / fbar[ai, cols]
        r_work = f_work[np.ix_(jw, cols)] / fbar[ai, cols][None, :]
        assert np.all(np.isfinite(r)) and np.all(np.isfinite(r_work))
        cls = classify(r_work)
        ratio_rows = [
            {
                "author_id": a,
                "word": p.words[v],
                "total_count": int(total[v]),
                "author_count": int(p.author_counts[ai, v]),
                "author_rel_freq": float(f_auth[ai, v]),
                "other_mean_rel_freq": float(fbar[ai, v]),
                "ratio": float(r[k]),
                "consistency": str(cls[k]),
                "works": ";".join(f"{title_of[p.work_ids[j]]}:{float(f_work[j, v])!r}:{float(r_work[i, k])!r}" for i, j in enumerate(jw)),
                "pos1_author": pos1_breakdown(pos1_by_author[a], p.words[v]),
                "pos1_all": pos1_breakdown(pos1_all, p.words[v]),
            }
            for k, v in enumerate(cols)
        ]
        work_rows = [
            {
                "author_id": a,
                "word": p.words[v],
                "work_id": p.work_ids[j],
                "title": title_of[p.work_ids[j]],
                "count": int(p.work_counts[j, v]),
                "work_morphemes": int(p.work_totals[j]),
                "rel_freq": float(f_work[j, v]),
                "work_ratio": float(r_work[i, k]),
            }
            for k, v in enumerate(cols)
            for i, j in enumerate(jw)
        ]
        no_other_rows = [
            {"author_id": a, "word": p.words[v], "total_count": int(total[v]), "author_count": int(p.author_counts[ai, v]), "pos1_all": pos1_breakdown(pos1_all, p.words[v])}
            for v in np.flatnonzero(no_other)
        ]
        return ratio_rows, work_rows, no_other_rows

    per_author = [author_rows(ai, a) for ai, a in enumerate(author_ids)]
    ratio_rows = [r for rs, _, _ in per_author for r in rs]
    no_other_rows = [r for _, _, ns in per_author for r in ns]
    count_rows = [
        {
            "author_id": a,
            "num_words_total": len(p.words),
            "num_eligible": int(eligible.sum()),
            "num_no_other_author": sum(r["author_id"] == a for r in no_other_rows),
            "num_with_ratio": sum(r["author_id"] == a for r in ratio_rows),
            "num_consistent_high": sum(r["author_id"] == a and r["consistency"] == "high" for r in ratio_rows),
            "num_consistent_low": sum(r["author_id"] == a and r["consistency"] == "low" for r in ratio_rows),
            "num_inconsistent": sum(r["author_id"] == a and r["consistency"] == "inconsistent" for r in ratio_rows),
            # 作家に出現しない語（r_a = 0 で一貫・低の同値．上位の切り捨ては語の辞書順）
            "num_absent_in_author": sum(r["author_id"] == a and r["author_count"] == 0 for r in ratio_rows),
        }
        for a in author_ids
    ]
    assert all(c["num_with_ratio"] + c["num_no_other_author"] == c["num_eligible"] for c in count_rows)
    assert all(c["num_consistent_high"] + c["num_consistent_low"] + c["num_inconsistent"] == c["num_with_ratio"] for c in count_rows)
    return {
        "ratios": ratio_rows,
        "work_ratios": [r for _, ws, _ in per_author for r in ws],
        "no_other": no_other_rows,
        "counts": count_rows,
    }


def top_rows(ratio_rows: Sequence[dict[str, Any]], author_ids: Sequence[str], top_n: int) -> list[dict[str, Any]]:
    """一貫（高）は r_a 降順，一貫（低）は r_a 昇順の上位 top_n 語（同値は語の辞書順）．"""
    order = {"high": lambda r: (-r["ratio"], r["word"]), "low": lambda r: (r["ratio"], r["word"])}
    return [
        {"author_id": a, "direction": d, "rank": rank, **row}
        for a in author_ids
        for d in ("high", "low")
        for rank, row in enumerate(sorted((r for r in ratio_rows if r["author_id"] == a and r["consistency"] == d), key=order[d])[:top_n], start=1)
    ]


def unit_rows(profiles: Profiles, works: Sequence[WorkCounts], corpus: Corpus) -> list[dict[str, Any]]:
    p = profiles
    work_part = [
        {"unit": "work", "author_id": w.author_id, "work_id": w.work_id, "title": w.title, "num_morphemes": w.num_morphemes, "num_function_tokens": sum(w.counts.values())}
        for w in works
    ]
    author_part = [
        {"unit": "author", "author_id": a, "work_id": "", "title": name, "num_morphemes": int(p.author_totals[i]), "num_function_tokens": int(p.author_counts[i].sum())}
        for i, (a, name) in enumerate(zip(corpus.author_ids, corpus.author_names, strict=True))
    ]
    return work_part + author_part


def word_count_rows(profiles: Profiles, works: Sequence[WorkCounts]) -> list[dict[str, Any]]:
    p = profiles
    pos1_all = sum((w.pos1_counts for w in works), Counter())
    return [
        {"word": t, "total_count": int(p.work_counts[:, v].sum()), **{f"count_{j}": int(p.work_counts[k, v]) for k, j in enumerate(p.work_ids)}, "pos1_all": pos1_breakdown(pos1_all, t)}
        for v, t in enumerate(p.words)
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"空の表で列名が未指定: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames or rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def per_k(x: float) -> str:
    return f"{x * PER_MORPHEMES:.2f}"


def report_lines(
    tops: Sequence[dict[str, Any]],
    counts: Sequence[dict[str, Any]],
    no_other: Sequence[dict[str, Any]],
    units: Sequence[dict[str, Any]],
    corpus: Corpus,
    args: argparse.Namespace,
    b_def: dict[str, Any],
) -> list[str]:
    name_of = dict(zip(corpus.author_ids, corpus.author_names, strict=True))
    lines = [
        "# 作家ごとに偏りのある付属語（記述統計，plans/030）",
        "",
        "- **評価に使わない記述統計**．4 作家・9 作品の全文（全チャンク・全作品）を使うため Train / Test の区別がなく，評価結果（ `full_006` 等）とは独立．",
        f"- 付属語の定義：B の既定（ `token_key={b_def['token_key']}` ， `target_pos={b_def['target_pos']}` ）．総形態素数は補助記号・空白を除く形態素数．形態素解析は文単位（Slide 1a の `verify_qa_points` と同じ）．",
        f"- 相対頻度は {PER_MORPHEMES} 形態素あたりで表示（CSV は無次元の相対頻度）．作家の相対頻度は全作品の出現回数と総形態素数を合算した値（長い作品の重みが大きい）．",
        "",
        "## 抽出規則",
        "",
        "- 比： $r_a(w) = f_a(w) / \\bar{f}_{\\neg a}(w)$ （ $\\bar{f}_{\\neg a}$ は他の 3 作家の相対頻度の単純平均）．作品の比は作品の相対頻度 ÷ $\\bar{f}_{\\neg a}$ ．",
        f"- 頻度の下限：4 作家合計の出現回数 {args.min_count} 以上（ `--min-count {args.min_count}` ）．",
        "- $\\bar{f}_{\\neg a}(w) = 0$ の語は比を計算せず「他作家に出現なし」の別表へ．",
        "- 作品をまたいだ一貫性：作家のすべての作品で作品の比が 1 より大（一貫・高）またはすべて 1 より小（一貫・低）の語のみ採用．漱石は 3 作品，他は 2 作品のため漱石の条件が厳しい．",
        f"- 一貫・高を $r_a$ の降順，一貫・低を $r_a$ の昇順に上位 {args.top_n} 語（ `--top-n {args.top_n}` ，同値は語の辞書順）．",
        "- 表層形をキーとするため，同じ表層形で品詞の異なる語（例：「で」の助詞と助動詞）は 1 語に集計．各語の pos1 の内訳（作家の全作品）を列に併記．",
        "",
        "## 総形態素数",
        "",
        "| 単位 | 作家 | 作品 | 総形態素数 | 付属語の延べ数 |",
        "| :--- | :--- | :--- | ---: | ---: |",
        *(f"| {u['unit']} | {name_of[u['author_id']]} | {u['title'] if u['unit'] == 'work' else '（全作品）'} | {u['num_morphemes']} | {u['num_function_tokens']} |" for u in units),
        "",
        "## 一貫性の件数",
        "",
        "| 作家 | 下限を満たす語 | 他作家に出現なし | 比を計算 | 一貫・高 | 一貫・低 | 不一貫（除外） | うち作家に出現なし（ $r_a = 0$ ） |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *(
            f"| {name_of[c['author_id']]} | {c['num_eligible']} | {c['num_no_other_author']} | {c['num_with_ratio']} | {c['num_consistent_high']} | {c['num_consistent_low']} | {c['num_inconsistent']} | {c['num_absent_in_author']} |"
            for c in counts
        ),
        "",
        "- 「うち作家に出現なし」は一貫・低に含まれる $r_a = 0$ の語の件数．同値のため上位 `--top-n` 語への切り捨ては語の辞書順で決まる．",
        "",
    ]

    def table(a: str, d: str) -> list[str]:
        rows = [t for t in tops if t["author_id"] == a and t["direction"] == d]
        head = "多い" if d == "high" else "少ない"
        body = [
            f"| {t['rank']} | {t['word']} | {per_k(t['author_rel_freq'])} | {per_k(t['other_mean_rel_freq'])} | {t['ratio']:.2f} | "
            + " / ".join(f"{s.split(':')[0]} {per_k(float(s.split(':')[1]))} ({float(s.split(':')[2]):.2f})" for s in t["works"].split(";"))
            + f" | {t['author_count']} | {t['pos1_author']} |"
            for t in rows
        ]
        return [
            f"### {name_of[a]}：他の 3 作家より{head}付属語",
            "",
            "| 順位 | 語 | 作家 | 他 3 作家の平均 | $r_a$ | 作品ごとの相対頻度（比） | 出現回数 | pos1 の内訳 |",
            "| ---: | :--- | ---: | ---: | ---: | :--- | ---: | :--- |",
            *(body or ["| － | 該当なし | | | | | | |"]),
            "",
        ]

    lines += ["## 作家ごとの上位・下位", ""] + [ln for a in corpus.author_ids for d in ("high", "low") for ln in table(a, d)]
    lines += [
        "## 他作家に出現なし（比を計算しない語）",
        "",
        "| 作家 | 語 | 作家の出現回数 | pos1 の内訳 |",
        "| :--- | :--- | ---: | :--- |",
        *(f"| {name_of[n['author_id']]} | {n['word']} | {n['author_count']} | {n['pos1_all']} |" for n in no_other),
        *([] if no_other else ["| － | 該当なし | | |"]),
        "",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--min-count", type=int, default=100, help="4 作家合計の出現回数の下限")
    parser.add_argument("--top-n", type=int, default=8, help="作家ごと・方向ごとの出力語数")
    parser.add_argument("--out-dir", type=Path, required=True, help="出力先（相対パスはプロジェクトルート基準．既存なら停止）")
    args = parser.parse_args()
    if args.min_count < 1 or args.top_n < 1:
        raise ValueError(f"--min-count と --top-n は正の整数: {args.min_count}, {args.top_n}")
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    if out_dir.exists():
        raise FileExistsError(f"出力先が既に存在（上書き防止）: {out_dir}")

    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b_params = cfg["features"]["B"]["params"]
    token_key: TokenKey = b_params["token_key"]
    pos_set = validate_target_pos(b_params["target_pos"])
    if pos_set != FUNCTION_POS:
        raise ValueError(f"features.B.params.target_pos が付属語（{sorted(FUNCTION_POS)}）でない: {b_params['target_pos']}")
    b_def = {"token_key": token_key, "target_pos": list(b_params["target_pos"]), "excluded_pos_in_total": sorted(EXCLUDED_POS)}

    corpus_path = PROJECT_ROOT / cfg["paths"]["processed_corpus"]
    corpus = load_corpus(corpus_path)
    works = [count_work(w.sentences, w.work_id, w.author_id, w.title, token_key, pos_set) for w in corpus.works]
    profiles = build_profiles(works, corpus.author_ids)
    assert np.array_equal(profiles.author_totals, np.array([sum(w.num_morphemes for w in works if w.author_id == a) for a in corpus.author_ids], dtype=np.float64))

    result = analyze(profiles, works, corpus.author_ids, args.min_count)
    tops = top_rows(result["ratios"], corpus.author_ids, args.top_n)
    units = unit_rows(profiles, works, corpus)

    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": display_path(args.config),
        "cli": {"min_count": args.min_count, "top_n": args.top_n, "out_dir": display_path(out_dir)},
        "function_word_definition": b_def,
        "analysis_unit": "sentence",
        "per_morphemes_in_report": PER_MORPHEMES,
        "git": git_state(),
        "corpus": corpus_meta(corpus_path),
        "author_ids": list(corpus.author_ids),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    if meta["git"]["dirty"]:
        print(f"[warn] 未コミットの変更あり（run_meta.json に記録）: {meta['git']['dirty_files']}")
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "unit_totals.csv", units)
    write_csv(out_dir / "word_counts.csv", word_count_rows(profiles, works))
    write_csv(out_dir / "author_ratios.csv", result["ratios"])
    write_csv(out_dir / "work_ratios.csv", result["work_ratios"])
    write_csv(out_dir / "no_other_author.csv", result["no_other"], fieldnames=["author_id", "word", "total_count", "author_count", "pos1_all"])
    write_csv(out_dir / "consistency_counts.csv", result["counts"])
    write_csv(out_dir / "top_words.csv", tops)
    (out_dir / "report.md").write_text("\n".join(report_lines(tops, result["counts"], result["no_other"], units, corpus, args, b_def)), encoding="utf-8")
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[info] 出力：{display_path(out_dir)}")


if __name__ == "__main__":
    main()
