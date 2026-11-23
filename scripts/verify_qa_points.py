"""想定質問（presentation/qa_defense_matrix.md）の根拠となる記述統計の補助検証スクリプト（plans/007）．

使い方：
    uv run python scripts/verify_qa_points.py [--config configs/default.yaml] [--top-n 20] [--top-words-per-work 12]

出力項目（標準出力のみ．評価パイプラインの結果には影響しない）：
    Q1      コーパス全 30 作品の作家別作品数と，2 作品以上を持つ作家
    Q2      底本の表記揺れ語（副詞「大分 / だいぶ」「矢張り / やはり / やっぱり」）の作品別の表層出現数と，
            UniDic の語彙素（ lemma ）による表記の統一例（副詞と内容語）
    Q3      チャンクの実測文字数（平均・標準偏差・変動係数・最小・最大，L ごと）
    Q4      端数チャンクの破棄率（全体・作品単位の最大，L ごと）
    Q9      D の累積寄与率（Intra-work の Train，間引きあり・なし，presentation/slides_d.md の表の照合用）
    Q10     B の語ブロック各次元のチャンク出現率（ x_j > 0 のチャンクの割合）の上位語と全次元の中央値
            （付属語は本採用，内容語対照は参考．全チャンクで fit した記述統計で，評価には不使用）
    Slide1a 作品別の付属語の出現回数上位（講義第 5 回 p.14 の頻度表の本コーパスでの再現）
    A       作品別の字種構成比（ char_type_ratios ，底本の表記方針の交絡の考察用）

コーパスは `data/processed/authors_corpus.json` を `src.dataset` 経由で読み込む（生コーパスは読まない）．
形態素解析は `src.morphology` 経由（ fugashi を直接使わない）．
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.run_experiments import intra_balance_rng
from src.dataset import Corpus, balance_train_split, build_chunks, chunk_sentences, intra_work_split, load_corpus
from src.features.lsi_embedding import LSIEmbeddingExtractor
from src.features.pos_function import CONTENT_POS, FUNCTION_POS, POSFunctionExtractor, top_function_words
from src.features.stylometry_char import CHAR_TYPE_NAMES, char_type_ratios
from src.morphology import morphemes_of, shared_tagger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Q2：底本の表記揺れ（同一語の漢字 / かな表記）．各群の先頭が漢字表記
ORTHOGRAPHIC_VARIANTS: dict[str, tuple[str, ...]] = {
    "大分": ("大分", "だいぶ"),
    "矢張り": ("矢張り", "やはり", "やっぱり"),
}
# Q2：語彙素による表記統一の例文（{} に語を入れて解析し，対象語の lemma を表示）
LEMMA_EXAMPLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("今日は{}寒くなった。", ("大分", "だいぶ", "矢張り", "やはり", "やっぱり")),
    ("彼はそう{}た。", ("云っ", "言っ")),
    ("意味が{}ない。", ("解ら", "分から")),
    ("それは大事な{}だ。", ("事", "こと")),
)


def header(title: str) -> None:
    print(f"\n==== {title} ====")


def git_state() -> str:
    rev = subprocess.run(["git", "rev-parse", "--short=10", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    return f"{rev}（dirty: {bool(dirty)}）"


# ---------------------------------------------------------------- Q1


def report_q1(corpus_json: dict[str, Any]) -> None:
    header("Q1：コーパス全作品の作家別作品数")
    summary = corpus_json["corpus_summary"]
    per_author = Counter(w["author"] for w in summary)
    print(f"作品数 {len(summary)}，作家数 {len(per_author)}")
    multi = sorted(((a, n) for a, n in per_author.items() if n >= 2), key=lambda an: (-an[1], an[0]))
    print(f"2 作品以上の作家（{len(multi)} 名）：" + "，".join(f"{a} {n}" for a, n in multi))
    print(f"1 作品の作家：{sum(n == 1 for n in per_author.values())} 名")
    titles = {a: [w["title"] for w in summary if w["author"] == a] for a, _ in multi}
    print("\n".join(f"  {a}：{'・'.join(t)}" for a, t in titles.items()))
    print("全作品（index，作家，作品名，文字数）：")
    print("\n".join(f"  {w['index']:>2} {w['author']} 『{w['title']}』 {w['num_chars']}" for w in summary))


# ---------------------------------------------------------------- Q2


def report_q2(corpus: Corpus) -> None:
    header("Q2：底本の表記揺れ語の作品別出現数（表層の部分文字列の出現数）と語彙素の内訳")
    tagger = shared_tagger()
    for group, variants in ORTHOGRAPHIC_VARIANTS.items():
        print(f"[{group}] " + " / ".join(variants))
        for w in corpus.works:
            text = "".join(w.sentences)
            counts = [text.count(v) for v in variants]
            # 表層形が変種のいずれかと一致する形態素の (表層形, pos1, lemma) 内訳（変種を含む文のみ文単位で解析）
            hits = [s for s in w.sentences if any(v in s for v in variants)]
            morph_counts = Counter(
                (m.surface, m.pos1, m.lemma) for s in hits for m in morphemes_of(tagger, s) if m.surface in variants
            )
            detail = "，".join(f"{s}:{p}:{lem}={n}" for (s, p, lem), n in sorted(morph_counts.items()))
            print(f"  {w.title}（{w.author_id}）：" + " / ".join(map(str, counts)) + (f"  形態素 {detail}" if detail else ""))
    print("語彙素による統一例（例文中の対象語の 表層形 → lemma（pos1））：")
    for template, words in LEMMA_EXAMPLES:
        rows = []
        for word in words:
            morphs = morphemes_of(tagger, template.format(word))
            target = [m for m in morphs if m.surface == word]
            if len(target) != 1:
                raise ValueError(f"例文の対象語が 1 形態素に解析されない: {template.format(word)!r} → {[m.surface for m in morphs]}")
            rows.append(f"{word} → {target[0].lemma}（{target[0].pos1}）")
        print(f"  {template}：" + "，".join(rows))


# ---------------------------------------------------------------- Q3 / Q4


def report_q3_q4(corpus: Corpus, chunk_lengths: Sequence[int]) -> None:
    header("Q3：チャンクの実測文字数（本文の文字数，L ごと）")
    for L in chunk_lengths:
        cs = build_chunks(corpus, L)
        # shape: (num_chunks,)，単位：文字
        n = np.array([len(c.text) for c in cs.chunks], dtype=np.float64)
        mean, std = n.mean(), n.std(ddof=1)
        print(
            f"L = {L:>4}：チャンク数 {n.size}，平均 {mean:.1f}，標準偏差 {std:.1f}（ddof = 1），"
            f"変動係数 {std / mean:.3f}，最小 {int(n.min())}，最大 {int(n.max())}"
        )
    header("Q4：端数チャンクの破棄率（破棄された文字数 / 作品の総文字数）")
    for L in chunk_lengths:
        # shape: (num_works,)
        total = np.array([sum(map(len, w.sentences)) for w in corpus.works], dtype=np.float64)
        kept = np.array([sum(len(s) for ch in chunk_sentences(w.sentences, L) for s in ch) for w in corpus.works], dtype=np.float64)
        dropped = total - kept
        assert np.all(dropped >= 0) and np.all(dropped < total)
        rate = dropped / total
        worst = int(rate.argmax())
        print(
            f"L = {L:>4}：全体 {dropped.sum() / total.sum() * 100:.2f}%（{int(dropped.sum())} / {int(total.sum())} 字），"
            f"作品単位の最大 {rate[worst] * 100:.2f}%（{corpus.works[worst].title}，{int(dropped[worst])} 字）"
        )


# ---------------------------------------------------------------- Q10


def chunk_presence(texts: Sequence[str], params: dict[str, Any], target_pos: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """語ブロックの各次元のチャンク出現率．返り値：(語彙, shape (|V|,) の出現率)．"""
    ex = POSFunctionExtractor(**{**params, "target_pos": list(target_pos), "blocks": "words"}).fit(texts)
    # shape: (num_chunks, |V_word|)
    X = ex.transform(texts)
    presence = (X > 0).mean(axis=0)
    assert presence.shape == (len(ex.word_vocab_),) and np.all(np.isfinite(presence))
    vocab = sorted(ex.word_vocab_, key=ex.word_vocab_.__getitem__)
    return vocab, presence


def report_q10(corpus: Corpus, chunk_lengths: Sequence[int], b_params: dict[str, Any], top_n: int) -> None:
    header("Q10：B の語ブロック各次元のチャンク出現率（ x_j > 0 のチャンクの割合）")
    print(
        "注：語彙は全 9 作品の全チャンクで fit した記述統計（評価の各 fold の Train で fit した語彙とは異なり，評価には不使用）．"
        f"B の params は configs の本採用値 {b_params} から blocks = words と target_pos のみ差し替え．"
        "中央値は語彙の全次元（低頻度の語を含む）についての値"
    )
    conditions = (("付属語（本採用）", sorted(FUNCTION_POS)), ("内容語対照（参考）", sorted(CONTENT_POS)))
    for L in chunk_lengths:
        texts = build_chunks(corpus, L).texts
        for name, pos in conditions:
            vocab, presence = chunk_presence(texts, b_params, pos)
            order = np.argsort(-presence, kind="stable")[:top_n]
            print(
                f"L = {L:>4} {name}：チャンク数 {len(texts)}，次元数 {len(vocab)}，"
                f"出現率の中央値 {np.median(presence):.3f}，出現率 >= 0.5 の次元 {int((presence >= 0.5).sum())}，"
                f">= 0.9 の次元 {int((presence >= 0.9).sum())}"
            )
            print(f"  上位 {top_n}：" + "，".join(f"{vocab[j]} {presence[j]:.3f}" for j in order))


# ---------------------------------------------------------------- Q9


def report_q9(corpus: Corpus, config: dict[str, Any]) -> None:
    """D の累積寄与率（Intra-work の Train，間引きあり・なし，シード平均）．slides_d.md の表の照合用．"""
    header("Q9：D の累積寄与率 Σ_{j≤K} σ_j^2 / ||X||_F^2（Intra-work の Train で fit）")
    L = int(config["report"]["headline"]["chunk_chars"])
    d_cfg = config["features"]["D"]
    k_values = sorted({int(d_cfg["params"]["n_components"]), *map(int, d_cfg["ablations"]["n_components"])})
    chunk_set = build_chunks(corpus, L)
    texts, labels = chunk_set.texts, chunk_set.labels
    seeds: list[int] = list(config["split"]["intra_seeds"])

    def ratios(seed: int, balance: bool) -> np.ndarray:
        split = intra_work_split(chunk_set, config["split"]["intra_test_ratio"], seed)
        split = balance_train_split(split, labels, intra_balance_rng(seed)) if balance else split
        ext = LSIEmbeddingExtractor(n_components=max(k_values)).fit([texts[i] for i in split.train_idx])
        # shape: (|K|,)
        return ext.cumulative_explained_ratio_[np.array(k_values) - 1]

    for balance, name in ((True, "間引きあり"), (False, "間引きなし")):
        # shape: (n_seeds, |K|)
        r = np.stack([ratios(s, balance) for s in seeds])
        print(
            f"L = {L}，{name}（シード {seeds}）：" + "，".join(
                f"K = {k} {m:.4f} ± {sd:.4f}" for k, m, sd in zip(k_values, r.mean(axis=0), r.std(axis=0, ddof=1), strict=True)
            )
        )


# ---------------------------------------------------------------- Slide 1a：作品別の付属語頻度上位


def report_top_function_words(corpus: Corpus, b_params: dict[str, Any], top_n: int) -> None:
    """講義第 5 回 p.14 の頻度表の本コーパスでの再現（作品単位，文単位の解析，B の本採用の token_key・target_pos）．"""
    header(f"Slide 1a：作品別の付属語の出現回数上位 {top_n} 語（ top_function_words ）")
    for w in corpus.works:
        top = top_function_words(w.sentences, n=top_n, token_key=b_params["token_key"], target_pos=b_params["target_pos"])
        print(f"  {w.title}（{w.author_id}）：" + "，".join(f"{t} {c}" for t, c in top))


# ---------------------------------------------------------------- A：字種構成比


def report_char_types(corpus: Corpus) -> None:
    header("A：作品別の字種構成比（作品の全文，空白・改行を除く文字数に対する割合）")
    ratios = char_type_ratios(["".join(w.sentences) for w in corpus.works])
    print("作品（作家）：" + " / ".join(CHAR_TYPE_NAMES))
    for w, r in zip(corpus.works, ratios, strict=True):
        print(f"  {w.title}（{w.author_id}）：" + " / ".join(f"{v:.4f}" for v in r))


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--top-n", type=int, default=20, help="Q10 で表示する出現率上位の語数")
    parser.add_argument("--top-words-per-work", type=int, default=12, help="Slide 1a で表示する作品別の付属語の上位語数")
    args = parser.parse_args()
    if args.top_n < 1 or args.top_words_per_work < 1:
        raise ValueError(f"--top-n・--top-words-per-work は正の整数: {args.top_n}, {args.top_words_per_work}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    corpus_path = PROJECT_ROOT / config["paths"]["processed_corpus"]
    corpus_json = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus = load_corpus(corpus_path)
    chunk_lengths: list[int] = list(config["chunking"]["target_chunk_chars"])
    b_params: dict[str, Any] = dict(config["features"]["B"]["params"])

    print(f"git：{git_state()}")
    print(f"コーパス：{corpus_path.relative_to(PROJECT_ROOT)}（source_sha256 {corpus_json['meta']['source_sha256'][:16]}…，"
          f"processed sha256 {hashlib.sha256(corpus_path.read_bytes()).hexdigest()[:16]}…）")
    print(f"対象：{len(corpus.author_ids)} 作家・{len(corpus.works)} 作品，L = {chunk_lengths}")

    report_q1(corpus_json)
    report_q2(corpus)
    report_q3_q4(corpus, chunk_lengths)
    report_q9(corpus, config)
    report_q10(corpus, chunk_lengths, b_params, args.top_n)
    report_top_function_words(corpus, b_params, args.top_words_per_work)
    report_char_types(corpus)


if __name__ == "__main__":
    main()
