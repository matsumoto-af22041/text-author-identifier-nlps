"""特徴量 C（内容語 TF-IDF，`src/features/word_tfidf.py`）の検証スクリプト（Ticket 004 の Verification Criteria，012 で整備）．

使い方：
    uv run python scripts/verify_word_tfidf.py [--config configs/default.yaml]

検証・集計項目：
    toy         トイデータ（3 文書・4 語）で TF・DF・IDF・TF-IDF が手計算と一致（idf = 0 の語の重みは厳密に 0）
    unseen      Test のみに出現する未知語を無視し次元数が一致，総形態素数 0 のテキストは ValueError
    variants    内容語の表記揺れ（云う / 言う，解る / 分かる，事 / こと）の文対で行ベクトルが一致
    oita        『坊っちゃん』『三四郎』の「大分」の pos1・pos2・lemma の内訳（文単位 / 全文連結）
    per_work    9 作品それぞれの内容語の表記揺れ（表層形の系統）の出現数と，統一先の lemma の内訳（文単位の解析）
    idf_zero    実データで idf = 0（df = N）となる語の件数（L ごと，各分割の Train で fit）

コーパスは `data/processed/authors_corpus.json` を `src.dataset` 経由で読み込む．
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.dataset import Corpus, Split, build_chunks, inter_work_splits, intra_work_split, load_corpus
from src.features.word_tfidf import AnalyzedText, WordTFIDFExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIRAGANA = re.compile(r"[ぁ-ゟ]+")
# 手計算との一致判定の許容誤差（小数点以下 6 桁）
ATOL = 1e-6

# (pos1, pos2, lemma, surface)．fugashi のノードは再利用されるため解析直後にタプル化
Morph = tuple[str, str, str | None, str]


def header(title: str) -> None:
    print(f"\n==== {title} ====")


def morphs_of(ex: WordTFIDFExtractor, text: str) -> list[Morph]:
    return [(n.feature.pos1, n.feature.pos2, n.feature.lemma, n.surface) for n in ex.tagger(text)]


# ---------------------------------------------------------------- toy / unseen / variants


def check_toy(max_features: int) -> WordTFIDFExtractor:
    header("toy：TF・DF・IDF・TF-IDF の手計算との一致")
    docs = ["猫が走る。", "猫が鳴く。", "猫と犬が走る。"]
    ex = WordTFIDFExtractor(max_features=max_features)
    print("analyze:", [ex.analyze(d) for d in docs])
    X = ex.fit_transform(docs)
    # 手計算：総形態素数（補助記号を除く）= 3, 3, 5．語彙は df 降順・辞書順
    exp_vocab = ["猫", "走る", "犬", "鳴く"]
    exp_df = np.array([3, 2, 1, 1], dtype=np.float64)
    exp_idf = np.log(3 / exp_df)
    # shape: (3, 4)
    exp_tf = np.array([[1 / 3, 1 / 3, 0, 0], [1 / 3, 0, 0, 1 / 3], [1 / 5, 1 / 5, 1 / 5, 0]])
    exp_w = exp_tf * exp_idf[None, :]
    tf = ex.tf_matrix(ex.analyze_all(docs))
    print("vocab:", list(ex.vocab_), " N:", ex.num_train_docs_, " df:", ex.df_.tolist())
    print("TF:\n", np.round(tf, 6), "\nIDF:", np.round(ex.idf_, 6), "\nW:\n", np.round(X, 6))
    errs = {"TF": np.abs(tf - exp_tf).max(), "IDF": np.abs(ex.idf_ - exp_idf).max(), "W": np.abs(X - exp_w).max()}
    print("max abs error:", {k: float(v) for k, v in errs.items()})
    assert list(ex.vocab_) == exp_vocab, ex.vocab_
    assert np.array_equal(ex.df_, exp_df)
    assert all(v <= ATOL for v in errs.values()), errs
    assert ex.idf_[0] == 0.0 and np.all(X[:, 0] == 0.0), "df = N の語（猫）の idf と重みは厳密に 0"
    print("idf(猫) == 0.0:", ex.idf_[0] == 0.0, " W[:, 猫]:", X[:, 0].tolist())
    return ex


def check_unseen(ex: WordTFIDFExtractor) -> None:
    header("unseen：Test のみの未知語の無視と Fail-Fast")
    Y = ex.transform(["象が泳ぐ。", "象と猫が走る。"])
    print("Y:", Y.tolist(), " shape:", Y.shape, " n_features:", ex.n_features, " n_zero_rows_:", ex.n_zero_rows_)
    assert Y.shape == (2, len(ex.vocab_)) and np.all(Y[0] == 0.0)
    assert math.isclose(Y[1, ex.vocab_["走る"]], (1 / 5) * math.log(3 / 2), abs_tol=ATOL)
    assert ex.n_zero_rows_ == 1
    for bad in ["。、！", "　"]:
        try:
            ex.transform([bad])
        except ValueError as e:
            print("ValueError（期待どおり）:", e)
        else:
            raise AssertionError(f"総形態素数 0 のテキストで ValueError が出ない: {bad!r}")


def check_variants(max_features: int) -> None:
    header("variants：内容語の表記揺れの統一（文対の行ベクトル一致）")
    pairs = [("彼はそう云った。", "彼はそう言った。"), ("意味が解らない。", "意味が分からない。"), ("そんな事はない。", "そんなことはない。")]
    for pair in pairs:
        ex = WordTFIDFExtractor(max_features=max_features).fit(list(pair))
        # df = N = 2 の語は idf = 0 となるため，重みではなく出現回数行列で語彙次元への集計を比較
        counts = ex.count_matrix(ex.analyze_all(pair))
        Z = ex.transform(list(pair))
        print(pair, " vocab:", list(ex.vocab_), " counts equal:", np.array_equal(counts[0], counts[1]), " rows equal:", np.array_equal(Z[0], Z[1]))
        assert len(ex.vocab_) == len({t for a in ex.analyze_all(pair) for t in a.terms})
        assert np.array_equal(counts[0], counts[1]) and np.array_equal(Z[0], Z[1]), pair
    ex = WordTFIDFExtractor(max_features=max_features)
    adverbs = ["今日は大分寒くなった。", "今日はだいぶ寒くなった。"]
    print("参考（副詞は索引語外）:", [[m for m in morphs_of(ex, s) if m[3] in ("大分", "だいぶ")] for s in adverbs], [ex.analyze(s).terms for s in adverbs])


# ---------------------------------------------------------------- corpus-based


def check_oita(ex: WordTFIDFExtractor, corpus: Corpus, titles: Sequence[str]) -> None:
    header("oita：「大分」の pos1・pos2・lemma の内訳")
    works = {w.title: w for w in corpus.works}
    missing = set(titles) - set(works)
    if missing:
        raise KeyError(f"コーパスに存在しない作品: {missing}")
    for title in titles:
        w = works[title]
        per_sent = Counter((p1, p2, lem) for s in w.sentences for p1, p2, lem, surf in morphs_of(ex, s) if surf == "大分")
        joined = Counter((p1, p2, lem) for p1, p2, lem, surf in morphs_of(ex, "".join(w.sentences)) if surf == "大分")
        kana = Counter(surf for s in w.sentences for _, _, _, surf in morphs_of(ex, s) if surf in ("だいぶ", "だいぶん"))
        n_place = {name: sum(v for k, v in c.items() if k[2] == "オオイタ") for name, c in (("文単位", per_sent), ("全文連結", joined))}
        print(f"{title}：文単位 {dict(per_sent)}")
        print(f"{' ' * len(title)}  全文連結 {dict(joined)}")
        print(f"{' ' * len(title)}  地名（lemma = オオイタ）：文単位 {n_place['文単位']} / {per_sent.total()} 件，全文連結 {n_place['全文連結']} / {joined.total()} 件．かな表記：{dict(kana)}")


@dataclass(frozen=True)
class VariantGroup:
    """表層形の系統（作品別集計の単位）．predicate は (pos1, lemma, surface) を受け取る．"""

    family: str
    label: str
    predicate: Callable[[str, str | None, str], bool]


def is_hiragana(s: str) -> bool:
    return HIRAGANA.fullmatch(s) is not None


VARIANT_GROUPS: tuple[VariantGroup, ...] = (
    VariantGroup("いう", "云-（動詞，表層形が「云」始まり）", lambda p, l, s: p == "動詞" and s.startswith("云")),
    VariantGroup("いう", "言-（動詞，表層形が「言」始まり）", lambda p, l, s: p == "動詞" and s.startswith("言")),
    VariantGroup("いう", "仮名書き（動詞，lemma = 言う，表層形がひらがな）", lambda p, l, s: p == "動詞" and l == "言う" and is_hiragana(s)),
    VariantGroup("わかる", "解-（動詞，表層形が「解」始まり）", lambda p, l, s: p == "動詞" and s.startswith("解")),
    VariantGroup("わかる", "分-（動詞，表層形が「分」始まり）", lambda p, l, s: p == "動詞" and s.startswith("分")),
    VariantGroup("わかる", "仮名書き（動詞，lemma = 分かる，表層形がひらがな）", lambda p, l, s: p == "動詞" and l == "分かる" and is_hiragana(s)),
    VariantGroup("こと", "事（名詞，表層形 = 事）", lambda p, l, s: p == "名詞" and s == "事"),
    VariantGroup("こと", "こと（名詞，表層形 = こと）", lambda p, l, s: p == "名詞" and s == "こと"),
)
# 各系統の統一先として期待する lemma
FAMILY_LEMMA: dict[str, str] = {"いう": "言う", "わかる": "分かる", "こと": "事"}


def check_per_work_variants(ex: WordTFIDFExtractor, corpus: Corpus) -> None:
    header("per_work：作品別の内容語の表記揺れと統一先 lemma（文単位の解析）")
    print("各セル：系統の出現数 / うち統一先 lemma の件数．内訳は lemma ごとの件数（統一先以外の lemma を含む）")
    for w in corpus.works:
        morphs = [(p1, lem, surf) for s in w.sentences for p1, _, lem, surf in morphs_of(ex, s)]
        print(f"\n[{w.author_id}] {w.title}（{len(w.sentences)} 文）")
        for g in VARIANT_GROUPS:
            hits = Counter((lem, surf) for p1, lem, surf in morphs if g.predicate(p1, lem, surf))
            by_lemma = Counter({lem: 0 for lem, _ in hits}) + Counter(lem for lem, _ in hits.elements())
            target = FAMILY_LEMMA[g.family]
            top_surface = ", ".join(f"{s}:{n}" for (_, s), n in hits.most_common(5))
            print(f"  {g.label:<40} {hits.total():>5} / {by_lemma[target]:>5} → lemma {dict(by_lemma.most_common())}  表層形上位 [{top_surface}]")


def check_idf_zero(ex: WordTFIDFExtractor, corpus: Corpus, cfg: dict[str, Any]) -> None:
    header("idf_zero：実データで idf = 0（df = N）となる語の件数")
    max_features = cfg["features"]["C"]["params"]["max_features"]
    for L in cfg["chunking"]["target_chunk_chars"]:
        cs = build_chunks(corpus, L)
        texts = [c.text for c in cs.chunks]
        # 全チャンクを 1 回だけ解析し，各分割の Train 部分集合で df を集計（fit と同一の集計）
        analyzed = ex.analyze_all(texts)
        splits: list[Split] = [intra_work_split(cs, cfg["split"]["intra_test_ratio"], cfg["seed"]), *inter_work_splits(cs)]

        def stats(train: Sequence[AnalyzedText]) -> tuple[int, int, int, int, int, tuple[str, ...]]:
            df = WordTFIDFExtractor.document_frequency(train)
            vocab = WordTFIDFExtractor.select_vocabulary(df, max_features)
            n = len(train)
            # df = N の語は df が最大のため必ず語彙に入る（max_features >= 1）
            zero = tuple(t for t in vocab if df[t] == n)
            return n, len(zero), len(df), min(df[t] for t in vocab), max(df[t] for t in vocab), zero

        rows = [(s.name, *stats([analyzed[i] for i in s.train_idx])) for s in splits]
        print(f"L = {L}：チャンク数 {len(texts)}")
        print("\n".join(f"  {name}: N={n}, idf0={z} {list(zw)}, |V_all|={v}, 語彙内 df∈[{lo}, {hi}]" for name, n, z, v, lo, hi, zw in rows))
        print(f"  idf = 0 の語の件数（全 {len(rows)} 分割の Train）：最小 {min(r[2] for r in rows)}，最大 {max(r[2] for r in rows)}，最大 df / N = {max(r[5] / r[1] for r in rows):.4f}")
        # 実際の fit との突き合わせ（LOWO 先頭 fold）
        s0 = splits[1]
        fitted = WordTFIDFExtractor(max_features=max_features).fit([texts[i] for i in s0.train_idx])
        n0, z0 = rows[1][1], rows[1][2]
        assert fitted.num_train_docs_ == n0 and int(np.sum(fitted.idf_ == 0.0)) == z0, "部分集合での集計と fit の結果が不一致"
        print(f"  cross-check（{s0.name} の fit）：idf == 0 の次元数 {int(np.sum(fitted.idf_ == 0.0))}（一致）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--oita-titles", nargs="+", default=["坊っちゃん", "三四郎"], help="「大分」の内訳を集計する作品")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    max_features = cfg["features"]["C"]["params"]["max_features"]

    ex = check_toy(max_features)
    check_unseen(ex)
    check_variants(max_features)

    corpus = load_corpus(PROJECT_ROOT / cfg["paths"]["processed_corpus"])
    tagger_ex = WordTFIDFExtractor(max_features=max_features)
    check_oita(tagger_ex, corpus, args.oita_titles)
    check_per_work_variants(tagger_ex, corpus)
    check_idf_zero(tagger_ex, corpus, cfg)
    print("\n[done] 全項目の検証が完了")


if __name__ == "__main__":
    main()
