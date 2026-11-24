"""生成した発表スライド（presentation/slides.pptx）を検査し，違反があれば終了コード 1 で停止．

使い方：
    uv run python scripts/check_slides.py [--pptx presentation/slides.pptx] [--content presentation/slides_content.yaml]
                                          [--run-dir outputs/full_006] [--values-dir outputs/slide_figures]
                                          [--sens-dir outputs/sensitivity_029_L{L}] [--digest]

検査項目（基準）：
  構成：本編 24 枚（ページ番号 1 / 24 ～ 24 / 24．1 枚目は表紙，最後は参考文献（role: references））→ 補助 S1～S17 の順（042，044，045 で類似度ギャップを削除）．
        Slide 9～20 の題名が「【担当 X】定義 / 実験の概要 / 結果・考察」の順（A～D の 3 枚構成）．
        補助スライドは想定質問（question）・質問の番号（qa）・対応する本編のスライドの番号（supplements）を持ち，
        supplements の昇順に並ぶ（044）．定義の一覧（deflist）の行は「用語：定義」の 2 要素のみ（042）．
  補助スライドの表示（044）：スライドの本文（ノートを除く，全スライド）に「想定質問」「答え：」「Q」＋数字がない．
        本編の本文に「補助 S」＋数字の参照がない．補助スライドの本文（extra_metrics: true のスライド＝指標の理由・基準値の根拠を
        除く）に「F 値」「MRR」がない．補助スライドの本文・発表者ノートに「ΔS（$\\Delta S$）」「類似度ギャップ」がない（045：ΔS は発表で扱わない）．
  本編の指標（042）：本編のスライド（参考文献を除く）の本文に「作家別 F 値」「MRR」「ΔS（$\\Delta S$）」「事前の仮説」が現れない．
        補助スライドの本文・ノートに「仮説」が現れない．
  品詞の 4 階層の図（Slide 12）：図の各セルの文字列が，例文を src/morphology.py の共有 Tagger で解析した品詞 4 階層と一致．
  照合（指標・条件）：slides_content.yaml の表の refs と各スライドの num_refs で対応付けた数値を，出典の CSV（<run-dir>/summary.csv，
        table_core_per_work.csv，similarity_stats.json，<sens-dir>/diff.csv）の key・列，または storyline の表のセルと照合
        （セルの値の入れ替えを検出）．丸める前の値をスライドに書かれた桁で丸めて比較．対応付けのない数値は下記のトークン照合のみ
        （件数を token_only_numbers に表示）．
  数式：内容データの $ の対応，すべての数式（短い記号を含む，041）が代替テキストに元の LaTeX を持つ画像としてスライドごとに
        置かれていること，本文の文字に平文の _ ^ \\ $ が残っていないこと，ノートに数式の記法がないこと．
  数値：本文・表・図形の文字・発表者ノートの数値トークン（小数，または 2 桁以上の整数．カンマ・符号は除いて比較）が，
        slides_content.yaml の num_sources で指定した storyline・qa・slides_d の範囲に現れること．
        ただし 0～29 の整数・100・500・1000・2000 と 4 桁の年は除外（L・枚数・番号）．出典行は下記の書き方の検査で扱う．
  追加した図：plot_slide_figures.py が書き出した描画値（<values-dir>/*.csv）が出典と完全一致すること．
        summary.csv から描いた図（グラフ A・B-1・C・F，図 B・グラフ E のスライド用）は値を summary.csv と照合し，平均値を 3 桁（ΔS は 4 桁）に
        丸めた値が storyline・qa・slides_d または <run-dir>/report.md に現れること（棒の上に値を書くグラフ A・B-1・C・F は
        storyline・qa・slides_d に現れない値の件数を表示）．
        test_composition は作家別 Test チャンク数を storyline の表 1 の作家小計（L = 500）と照合し，全件を漱石と予測したときの
        2 つの集計を件数から再計算して照合．s8 は作家間平均類似度・μ を similarity_stats.json から再計算して照合．
        スライドの画像は presentation/figures の現行ファイルとバイト一致．
  書き方：
    出典行：各項目に「第 N 回 p.」「構想発表資料 p.」「独自設計」のいずれかを含み，プログラム名・.py・run 名・
            チケット番号（3 桁の 0 始まりの数）・下線を含まない．
    本文：「第 N 回 p.」の直書き，.py，run 名，チケット番号，平文の添字（_ ^）を含まない．
    列挙：1 段落（表のセルの 1 行を含む）の読点「，」が 3 個以上，または 2 個で区切られた 3 片がすべて 8 字以下なら違反．
    段落の長さ：本文の 1 段落は 72 字以下，表のセルの 1 行は 50 字以下（マークアップを除いた文字数）．
  個人名・秘匿情報（見つけた文字列は出力せず，箇所のみ）：
    pptx のメタデータ（作成者・最終更新者・題名・会社名等）が空．
    pptx 内のすべての XML にメールアドレス・ローカルパス（/Users/，/home/，C:\\）・学籍番号の形（英字 2 字＋数字 5 字）がない．
    git の user.name・user.email・リポジトリのログイン名の各部分（4 字以上）と，元のスライド（presentation/orig.pptx，
    読み取りのみ）のメタデータの作成者・最終更新者の文字列がない．
  はみ出し：すべての図形がスライド内．テキスト枠は文字数から見積もった必要な高さが枠の高さ以下．
            本文の主要な図形（題名・主張・本文・表・図・出典）が互いに重ならない．
--digest：生成日時等のメタデータを除いた内容（テキスト・ノート・図と数式の画像の参照とバイト列のハッシュ）の SHA-256 を表示．
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pptx import Presentation
from pptx.util import Emu

from scripts.build_slides import (DIAGRAM_LATEX, MARKUP, NS_A, diagram_blocks, n_lines_em, paragraph_xml_metrics, paragraph_xml_text,
                                  pos4_table, yaml_texts)
from src.evaluator import SimilarityStats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRES = PROJECT_ROOT / "presentation"
EMU_PER_IN = 914400

NUM = re.compile(r"[+\-−]?\d[\d,]*\.\d+|[+\-−]?\d{2,}[\d,]*")
IGNORE_NUM = re.compile(r"^(\d|1\d|2\d|100|500|1000|2000|(19|20)\d\d)$")
RUN_NAME = re.compile(r"full_\d|sensitivity|exclusion|centroid_norm|function_word|verify_|outputs/|scripts/|summary\.csv")
TICKET = re.compile(r"(?<![\d.,])0\d\d(?![\d.,])")
LECTURE_PAGE = re.compile(r"第 ?\d+ ?回 ?p\.")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LOCAL_PATH = re.compile(r"/Users/|/home/|[A-Za-z]:\\")
STUDENT_ID = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{5}(?![0-9])")
MAX_BODY_CHARS, MAX_CELL_CHARS = 72, 50
N_MAIN, N_SUPP, PART_START = 24, 17, 9  # 本編の枚数，補助スライドの枚数，担当 A の最初のスライドの番号（042，044，045 で補助 17 枚）
MAIN_BANNED = ("作家別 F 値", "MRR", "\\Delta S", "ΔS", "事前の仮説")  # 本編の本文に書かない語（042）
# summary.csv から描いた追加図（plot_slide_figures.py の描画値の CSV の名前）
SUMMARY_FIGURES = ("a_ablation", "b_ablation", "c_intra_inter", "classifier_comparison", "core_chunk_length_slide", "chunk_length_slide")
# 補助スライドの表示（044）：全スライドの本文に出さない語，本編の本文に出さない補助スライドの番号の参照，補助スライドの本文に出さない指標
DISPLAY_BANNED = re.compile(r"想定質問|答え：|Q\d")
MAIN_SUPP_REF = re.compile(r"補助 ?S\d")
SUPP_METRIC_BANNED = re.compile(r"F 値|MRR")
SUPP_DELTA_S_BANNED = re.compile(r"ΔS|\\Delta S|類似度ギャップ")  # 045：補助スライドの本文・ノートに書かない
AUTHOR_NAMES = {"soseki": "夏目漱石", "dazai": "太宰治", "terada": "寺田寅彦", "tayama": "田山花袋"}


def norm_num(t: str) -> str:
    return t.replace("−", "-").replace(",", "").lstrip("+-")


def num_tokens(s: str) -> set[str]:
    return {norm_num(t) for t in NUM.findall(s)} - {t for t in map(norm_num, NUM.findall(s)) if IGNORE_NUM.match(t)}


# ---------------------------------------------------------------- 資料の範囲


class Docs:
    def __init__(self) -> None:
        self.text = {n: (PRES / f).read_text(encoding="utf-8") for n, f in
                     (("storyline", "presentation_storyline.md"), ("qa", "qa_defense_matrix.md"), ("slides_d", "slides_d.md"))}

    def segment(self, spec: str) -> str:
        if spec == "slides_d":
            return self.text["slides_d"]
        if spec.startswith("storyline:"):
            label = spec.split(":", 1)[1]
            lines = [ln for ln in self.text["storyline"].splitlines() if ln.startswith(f"| **{label}:")]
            if len(lines) != 1:
                raise ValueError(f"storyline の行 {label!r} が 1 行に定まらない（{len(lines)} 行）")
            return lines[0]
        doc, head = spec.split("#", 1)
        lines = self.text[doc].splitlines()
        idx = [i for i, ln in enumerate(lines) if ln.startswith("#") and head in ln]
        if len(idx) != 1:
            raise ValueError(f"{doc} の見出し {head!r} が 1 つに定まらない（{len(idx)} 件）")
        level = len(lines[idx[0]]) - len(lines[idx[0]].lstrip("#"))
        end = next((j for j in range(idx[0] + 1, len(lines))
                    if lines[j].startswith("#") and len(lines[j]) - len(lines[j].lstrip("#")) <= level), len(lines))
        return "\n".join(lines[idx[0]:end])


# ---------------------------------------------------------------- 指標・条件の対応付け（出典の CSV・storyline の表のセル）

CELL_NUM = re.compile(r"[+\-−]?\d[\d,]*(?:\.\d+)?")


def norm_signed(t: str) -> str:
    return t.replace("−", "-").replace(",", "").lstrip("+")


class RefSources:
    """照合の指定（ref）の値を返す．
    summary:<key>:<metric>                 <run-dir>/summary.csv の key の行の列（± のセルは <metric>_std も照合）
    per_work:<L>:<作品>:<列>               <run-dir>/table_core_per_work.csv の行（chunk_chars と title）の列
    sens:<L>:<条件>:<列>                   <sens-dir>/diff.csv（付属語 − 内容語の差）の cond の行の列（sensitivity_029）
    stats:<key>:<行>,<列>                  <run-dir>/similarity_stats.json の key の作家間平均類似度行列の成分（全シード合算）
    <ref> - <ref>                         2 つの数値の差（丸める前の値の差）
    story:<見出し>#<n>|<行>|<列>           storyline の見出しの下の n 番目（1 始まり）の表の，1 列目が <行>（または「列名=値」）の行・見出しが <列> のセル
                                          （セルの数値の並びとスライドのセルの数値の並びを照合）
    """

    def __init__(self, run_dir: Path, story_text: str, sens_dir: str) -> None:
        with (run_dir / "summary.csv").open(encoding="utf-8") as f:
            self.summary = {r["key"]: r for r in csv.DictReader(f)}
        with (run_dir / "table_core_per_work.csv").open(encoding="utf-8") as f:
            self.per_work = {(r["chunk_chars"], r["title"]): r for r in csv.DictReader(f)}
        self.stats = json.loads((run_dir / "similarity_stats.json").read_text(encoding="utf-8"))
        self.sens_dir = sens_dir
        self.story = story_text.splitlines()

    def number(self, ref: str, std: bool = False) -> float:
        if " - " in ref:
            a, b = ref.split(" - ")
            if std:
                raise ValueError(f"差の照合に標準偏差はない: {ref}")
            return self.number(a) - self.number(b)
        src, *rest = ref.split(":")
        if src == "summary":
            key, metric = rest
            if key not in self.summary:
                raise KeyError(f"summary.csv に key がない: {key}")
            v = self.summary[key][metric + ("_std" if std else "")]
        elif src == "per_work":
            L, title, col = rest
            v = self.per_work[(L, title)][col + ("_std" if std else "")]
        elif src == "sens":
            L, cond, col = rest
            with (PROJECT_ROOT / self.sens_dir.format(L=L) / "diff.csv").open(encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if r["cond"] == cond]
            if len(rows) != 1:
                raise ValueError(f"diff.csv の cond {cond} が 1 行に定まらない: {ref}")
            v = rows[0][col + ("_std" if std else "")]
        elif src == "stats":
            key, ij = ":".join(rest[:-1]), rest[-1]
            if std:
                raise ValueError(f"作家間平均類似度に標準偏差はない: {ref}")
            i, j = map(int, ij.split(","))
            v = str(SimilarityStats.from_dict(self.stats[key]).author_matrix[i, j])
        else:
            raise ValueError(f"未知の照合の指定: {ref}")
        if v == "":
            raise ValueError(f"照合先の値が空: {ref}")
        return float(v)

    def story_cell(self, ref: str) -> str:
        head_tab, row, col = ref.removeprefix("story:").split("|")
        head, n = head_tab.rsplit("#", 1)
        idx = [i for i, ln in enumerate(self.story) if ln.startswith("#") and head in ln]
        if len(idx) != 1:
            raise ValueError(f"storyline の見出し {head!r} が 1 つに定まらない（{len(idx)} 件）")
        end = next((j for j in range(idx[0] + 1, len(self.story)) if self.story[j].startswith("#")), len(self.story))
        tables: list[list[str]] = []
        prev = False
        for ln in self.story[idx[0] + 1:end]:
            is_row = ln.startswith("|")
            if is_row and not prev:
                tables.append([])
            if is_row:
                tables[-1].append(ln)
            prev = is_row
        rows = [[c.strip().replace("**", "").replace("`", "") for c in ln.strip().strip("|").split("|")] for ln in tables[int(n) - 1]]
        header, body = rows[0], rows[2:]
        cols = [j for j, h in enumerate(header) if h == col]
        # 行の指定は 1 列目の値，または「列名=値」
        key_col, key = (header.index(row.split("=", 1)[0]), row.split("=", 1)[1]) if "=" in row else (0, row)
        hits = [r for r in body if r[key_col] == key]
        if len(cols) != 1 or len(hits) != 1:
            raise ValueError(f"storyline の表のセルが 1 つに定まらない: {ref}（列 {len(cols)}，行 {len(hits)}）")
        return hits[0][cols[0]]


def check_number(slide_tok: str, value: float) -> bool:
    """スライドの数値（書かれた桁）が丸める前の値を同じ桁に丸めた値と一致するか（符号を書いた場合は符号も）．"""
    t = norm_signed(slide_tok)
    digits = len(t.split(".")[1]) if "." in t else 0
    return float(t) == float(f"{value:.{digits}f}")


def expand_table_refs(refs: dict[str, Any], n_rows: int, n_cols: int) -> dict[tuple[int, int], str]:
    """表の照合の指定：vars（データ行ごとの置換．null は照合しない），cols（列ごとの雛形．{v} を置換），cells（"行,列" の個別指定）．"""
    out: dict[tuple[int, int], str] = {}
    vars_ = refs.get("vars", [None] * n_rows)
    cols = refs.get("cols", [None] * n_cols)
    if len(vars_) != n_rows or len(cols) != n_cols:
        raise ValueError(f"表の照合の指定の行数・列数が表と一致しない（vars {len(vars_)} / {n_rows}，cols {len(cols)} / {n_cols}）")
    for i, v in enumerate(vars_):
        for j, c in enumerate(cols):
            if v is not None and c is not None:
                out[(i, j)] = c.format(**v) if isinstance(v, dict) else c.format(v=v)
    for k, r in refs.get("cells", {}).items():
        i, j = map(int, k.split(","))
        out[(i, j)] = r
    return out


# ---------------------------------------------------------------- pptx の読み取り


def iter_shapes(shapes: Any) -> Iterator[Any]:
    for s in shapes:
        yield s
        if s.shape_type == 6:
            yield from iter_shapes(s.shapes)


def shape_paragraphs(s: Any) -> list[tuple[str, str]]:
    """[(種類, 段落の文字列)]．種類は text / cell（定義の一覧の表は text）．数式は線形の文字列で含める．"""
    out: list[tuple[str, str]] = []
    if s.has_text_frame:
        out += [("text", t) for p in s.text_frame.paragraphs if (t := paragraph_xml_text(p._p)).strip()]
    if getattr(s, "has_table", False) and s.has_table:
        kind = "text" if s.name.startswith("deflist:") else "cell"
        out += [(kind, t) for r in s.table.rows for c in r.cells for p in c.text_frame.paragraphs
                if (t := paragraph_xml_text(p._p)).strip()]
    return out


def table_cells(s: Any) -> list[list[str]]:
    """表の各セルの文字列（段落は改行で連結）．"""
    return [["\n".join(paragraph_xml_text(p._p) for p in c.text_frame.paragraphs) for c in r.cells] for r in s.table.rows]


class Checker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.pptx_path = args.pptx
        self.prs = Presentation(str(args.pptx))
        self.content = yaml.safe_load(args.content.read_text(encoding="utf-8"))
        self.args = args
        self.violations: list[str] = []
        self.counts: dict[str, int] = {}

    def fail(self, msg: str) -> None:
        self.violations.append(msg)

    def bump(self, k: str, n: int = 1) -> None:
        self.counts[k] = self.counts.get(k, 0) + n

    # ---------- 構成

    def check_structure(self) -> None:
        slides = list(self.prs.slides)
        specs = self.content["slides"]
        if len(slides) != len(specs):
            self.fail(f"構成：pptx {len(slides)} 枚と内容データ {len(specs)} 枚が一致しない")
        titles = [self.named_text(s, "title") for s in slides]
        pages = [self.named_text(s, "pageno") for s in slides]
        expect_pages = [f"{i} / {N_MAIN}" for i in range(1, N_MAIN + 1)] + [f"S{i}" for i in range(1, N_SUPP + 1)]
        if pages != expect_pages:
            self.fail(f"構成：ページ番号の並びが 本編 {N_MAIN} → 補助 S1～S{N_SUPP} でない: {pages}")
        if any(spec["kind"] == "sources" or "出典一覧" in spec["title"] for spec in specs):
            self.fail("構成：出典一覧のスライドがある（041 で削除，042 で参考文献として本編の最後に置く）")
        refs = [i for i, spec in enumerate(specs, 1) if spec.get("role") == "references"]
        if refs != [N_MAIN] or specs[N_MAIN - 1]["title"] != "参考文献":
            self.fail(f"構成：参考文献のスライドが本編の最後（Slide {N_MAIN}）の 1 枚でない: {refs}")
        for spec in specs:
            if spec["kind"] == "supp" and not all(spec.get(k) for k in ("question", "qa", "supplements")):
                self.fail(f"構成：補助スライド {spec['id']} に想定質問・質問の番号・対応する本編のいずれかがない")
        sup = [spec.get("supplements") for spec in specs if spec["kind"] == "supp"]
        if not all(isinstance(n, int) and 2 <= n < N_MAIN for n in sup) or sup != sorted(sup):
            self.fail(f"構成：補助スライドが対応する本編のスライドの番号の昇順でない: {sup}")
        self.supp_order = [(spec["id"], spec.get("supplements")) for spec in specs if spec["kind"] == "supp"]
        for spec in specs:
            for b in walk_blocks([x for k in ("body", "left", "right") for x in spec.get(k, [])]):
                if "deflist" in b and any(len(r) != 2 for r in b["deflist"]["rows"]):
                    self.fail(f"構成：{spec['id']} の定義の一覧に「用語：定義」の形でない行がある")
        if specs and specs[0]["layout"] != "cover":
            self.fail("構成：1 枚目が表紙（layout: cover）でない")
        for i, t in enumerate(titles[N_MAIN:], 1):
            if not t.startswith(f"S{i}："):
                self.fail(f"構成：補助スライド {i} 枚目の題名が S{i}： で始まらない")
        parts = [f"【担当 {p}】{k}" for p in "ABCD" for k in ("定義", "実験の概要", "結果・考察")]
        if titles[PART_START - 1:PART_START + 11] != parts:
            self.fail(f"構成：Slide {PART_START}～{PART_START + 11} が A～D の 定義 / 実験の概要 / 結果・考察 の順でない: "
                      f"{titles[PART_START - 1:PART_START + 11]}")
        for s, spec in zip(slides, specs, strict=False):
            if self.named_text(s, "title") != spec["title"]:
                self.fail(f"構成：題名が内容データと一致しない（{spec['id']}）")
        self.counts["slides"] = len(slides)
        self.counts["main"] = sum(p.endswith(f"/ {N_MAIN}") for p in pages)
        self.counts["supp"] = sum(p.startswith("S") for p in pages)

    @staticmethod
    def named_text(slide: Any, name: str) -> str:
        return next(("\n".join(paragraph_xml_text(p._p) for p in s.text_frame.paragraphs)
                     for s in slide.shapes if s.name == name and s.has_text_frame), "")

    # ---------- 数値

    def check_numbers(self, docs: Docs) -> None:
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            allowed = set().union(*(num_tokens(docs.segment(src)) for src in spec["num_sources"]))
            body = [t for s in iter_shapes(slide.shapes) if s.name not in ("footnote", "pageno") for _, t in shape_paragraphs(s)]
            notes = slide.notes_slide.notes_text_frame.text
            mapped = self.mapped_tokens.get(spec["id"], set())
            for where, texts in (("本文", body), ("ノート", [notes])):
                for tok in sorted(set().union(*(num_tokens(t) for t in texts)) if texts else set()):
                    self.bump("numbers")
                    self.bump("token_only_numbers" if where == "ノート" or tok not in mapped else "token_also_mapped")
                    if tok not in allowed:
                        self.fail(f"数値：{spec['id']} の{where}の {tok} が資料の範囲 {spec['num_sources']} にない")

    def check_refs(self, refs_src: RefSources) -> None:
        """表のセルと本文の数値を，指標・条件（特徴量，L，分割，間引き，推論方式）まで出典の CSV・storyline の表のセルと照合．"""
        self.mapped_tokens: dict[str, set[str]] = {}
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            sid = spec["id"]
            covered: set[str] = set()
            tables = [b["table"] for k in ("body", "left", "right") for b in walk_blocks(spec.get(k, [])) if "table" in b]
            shapes = [s for s in iter_shapes(slide.shapes) if s.name.startswith("table:")]
            if len(tables) != len(shapes):
                self.fail(f"照合：{sid} の表の数が内容データと pptx で一致しない")
                continue
            for t, shp in zip(tables, shapes, strict=True):
                if "refs" not in t:
                    continue
                cells = table_cells(shp)[1:]  # 見出しの行を除く
                for (i, j), ref in expand_table_refs(t["refs"], len(t["rows"]), len(t["header"])).items():
                    covered |= self.verify(sid, f"表 {i + 1} 行 {j + 1} 列", cells[i][j], ref, refs_src)
            body = [t for s in iter_shapes(slide.shapes) if s.name not in ("footnote", "pageno") for _, t in shape_paragraphs(s)]
            for nr in spec.get("num_refs", []):
                at = nr["at"]
                if not any(at in p for p in body):
                    self.fail(f"照合：{sid} の本文に「{at}」がない")
                    continue
                refs = nr["refs"] if "refs" in nr else [nr["ref"]]
                if len(refs) == 1 and "±" in at:  # 「平均 ± 標準偏差」の 1 組
                    covered |= self.verify(sid, f"本文「{at}」", at, refs[0], refs_src)
                    continue
                toks = CELL_NUM.findall(at)
                if len(toks) != len(refs):
                    self.fail(f"照合：{sid} の「{at}」の数値 {len(toks)} 個と照合の指定 {len(refs)} 個が一致しない")
                    continue
                for tok, ref in zip(toks, refs, strict=True):
                    if ref is not None:  # null は照合しない数値（例：k の値）
                        covered |= self.verify(sid, f"本文「{at}」", tok, ref, refs_src)
            self.mapped_tokens[sid] = covered

    def verify(self, sid: str, where: str, text: str, ref: str, refs_src: RefSources) -> set[str]:
        """1 つのセル（または本文の数値）を照合し，照合した数値のトークンを返す．"""
        toks = CELL_NUM.findall(text)
        if ref.startswith("story:"):
            expect = [norm_signed(x) for x in CELL_NUM.findall(refs_src.story_cell(ref))]
            got = [norm_signed(x) for x in toks]
            self.bump("mapped_numbers", len(got))
            if got != expect:
                self.fail(f"照合：{sid} の{where}「{text}」の数値 {got} が {ref} の {expect} と一致しない")
        else:
            if not 1 <= len(toks) <= 2 or (len(toks) == 2 and "±" not in text):
                self.fail(f"照合：{sid} の{where}「{text}」は 1 つの数値か「平均 ± 標準偏差」でない")
                return set()
            self.bump("mapped_numbers", len(toks))
            for tok, std in zip(toks, (False, True)):
                v = refs_src.number(ref, std=std)
                if not check_number(tok, v):
                    self.fail(f"照合：{sid} の{where}の {tok} が {ref}{'（標準偏差）' if std else ''} の値 {v:.6f} と一致しない")
        return {norm_num(t) for t in toks}

    # ---------- 追加した図

    def check_figures(self, docs: Docs) -> None:
        with (self.args.run_dir / "summary.csv").open(encoding="utf-8") as f:
            summary = {r["key"]: r for r in csv.DictReader(f)}
        all_docs = docs.text["storyline"] + docs.text["qa"] + docs.text["slides_d"]
        doc_tokens = num_tokens(all_docs)
        report_tokens = num_tokens((self.args.run_dir / "report.md").read_text(encoding="utf-8"))
        for name in SUMMARY_FIGURES:
            path = self.args.values_dir / f"{name}.csv"
            if not path.exists():
                self.fail(f"図：描画値 {path.name} がない（plot_slide_figures.py を実行）")
                continue
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                self.bump("figure_values")
                src = summary[r["key"]]
                if float(r["value"]) != float(src[r["metric"]]):
                    self.fail(f"図：{name} の {r['key']} {r['metric']} が summary.csv と一致しない")
                if r["std"] != "" and float(r["std"]) != float(src[r["metric"] + "_std"]):
                    self.fail(f"図：{name} の {r['key']} {r['metric']} の標準偏差が summary.csv と一致しない")
                digits = 4 if r["metric"].startswith("delta_s") else 3
                rounded = f"{float(r['value']):.{digits}f}"
                # 基準値 0.25 のように資料が短い桁で書く値は末尾の 0 を除いた形も認める
                in_docs = rounded in doc_tokens or rounded.rstrip("0") in doc_tokens
                if not in_docs:
                    self.bump(f"figure_values_report_only:{name}")
                if not in_docs and rounded not in report_tokens and rounded.rstrip("0") not in report_tokens:
                    self.fail(f"図：{name} の {r['key']} {r['metric']} の丸め値 {rounded} が storyline・qa・slides_d・report.md にない")
        self.check_composition(docs)
        self.check_s8()
        for i, slide in enumerate(self.prs.slides, 1):
            for s in iter_shapes(slide.shapes):
                if s.name.startswith("figure:"):
                    self.bump("figure_refs")
                    f = PRES / s.name.split(":", 1)[1]
                    if not f.exists():
                        self.fail(f"図：スライド {i} の {f.name} が presentation にない")
                    elif hashlib.sha256(s.image.blob).hexdigest() != hashlib.sha256(f.read_bytes()).hexdigest():
                        self.fail(f"図：スライド {i} の {f.name} が presentation/figures の現行ファイルと異なる")

    def check_composition(self, docs: Docs) -> None:
        """指標の定義のスライドの図：作家別 Test チャンク数（L = 500 の作品間分割）を storyline の表 1 の作家小計と照合し，
        全件を最多の作家と予測したときの全 Test チャンクでの正解の割合・作家別正解率の平均を件数から再計算して照合．"""
        path = self.args.values_dir / "test_composition.csv"
        if not path.exists():
            self.fail("図：描画値 test_composition.csv がない（plot_slide_figures.py を実行）")
            return
        with path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        counts = {r["author"]: int(r["value"]) for r in rows if r["item"] == "n_test"}
        story = RefSources(self.args.run_dir, docs.text["storyline"], self.args.sens_dir)
        for author, label in AUTHOR_NAMES.items():
            self.bump("figure_values")
            cell = story.story_cell(f"story:表 1（Slide 6）#2|作家={label} 小計|チャンク数 $L = 500$")
            if int(norm_num(cell)) != counts[author]:
                self.fail(f"図：test_composition の {author} の Test チャンク数 {counts[author]} が storyline の表 1 の小計 {cell} と一致しない")
        n = sum(counts.values())
        expect = {"overall_if_all_soseki": counts["soseki"] / n, "per_author_mean_if_all_soseki": 0.25}
        for r in rows:
            if r["item"] in expect:
                self.bump("figure_values")
                if float(r["value"]) != expect[r["item"]]:
                    self.fail(f"図：test_composition の {r['item']} {r['value']} が再計算 {expect[r['item']]} と一致しない")
        if f"{expect['overall_if_all_soseki']:.3f}" not in num_tokens(docs.text["qa"]):
            self.fail("図：test_composition の全 Test チャンクでの正解の割合（3 桁）が qa にない")

    def check_s8(self) -> None:
        """補助 S16 の図：作家間平均類似度・μ を similarity_stats.json から再計算して照合（完全一致）．"""
        path = self.args.values_dir / "s8.csv"
        if not path.exists():
            self.fail("図：描画値 s8.csv がない（plot_slide_figures.py を実行）")
            return
        raw = json.loads((self.args.run_dir / "similarity_stats.json").read_text(encoding="utf-8"))
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.bump("figure_values")
                st = SimilarityStats.from_dict(raw[f"{r['feature']}|L500|inter|bal|cosine|all"])
                v = (float(st.author_matrix[int(r["i"]), int(r["j"])]) if r["item"] == "author_matrix"
                     else getattr(st, r["item"]))
                if float(r["value"]) != v:
                    self.fail(f"図：s8 の {r['feature']} {r['item']} {r['i']},{r['j']} が similarity_stats.json と一致しない")

    # ---------- 書き方

    def check_writing(self) -> None:
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            sid = spec["id"]
            foot = self.named_text(slide, "footnote")
            items = [it.strip() for it in re.split(r"\[\d+\]", foot) if it.strip()]
            for it in items:
                self.bump("footnote_items")
                if not (LECTURE_PAGE.search(it) or "独自設計" in it or "構想発表資料 p." in it):
                    self.fail(f"書き方：{sid} の出典「{it}」が講義資料の回・頁か独自設計でない")
                if RUN_NAME.search(it) or ".py" in it or TICKET.search(it) or "_" in it:
                    self.fail(f"書き方：{sid} の出典にプログラム名・run 名・チケット番号がある")
            for s in iter_shapes(slide.shapes):
                if s.name in ("footnote", "pageno"):
                    continue
                for kind, p in shape_paragraphs(s):
                    self.bump("paragraphs")
                    if spec.get("role") != "references" and LECTURE_PAGE.search(p):
                        self.fail(f"書き方：{sid} の本文に講義資料の頁の直書きがある（{s.name}）")
                    if RUN_NAME.search(p) or ".py" in p or TICKET.search(p):
                        self.fail(f"書き方：{sid} の本文にプログラム名・run 名・チケット番号がある（{s.name}）")
                    if "_" in p or "^" in p:
                        self.fail(f"書き方：{sid} の本文に平文の添字（_ ^）がある（{s.name}）")
                    segs = p.split("，")
                    if len(segs) >= 4 or (len(segs) == 3 and all(len(x.strip()) <= 8 for x in segs)):
                        self.fail(f"書き方：{sid} に読点区切りの列挙がある（{s.name}）：{p}")
                    limit = MAX_CELL_CHARS if kind == "cell" else MAX_BODY_CHARS
                    if len(p) > limit:
                        self.fail(f"書き方：{sid} の段落が {len(p)} 字で上限 {limit} 字を超える（{s.name}）")
            notes = slide.notes_slide.notes_text_frame.text
            if re.search(r"_\{|\^\{|\^-", notes):
                self.fail(f"書き方：{sid} のノートに平文の添字がある")

    def check_terms(self) -> dict[str, int]:
        """本編（参考文献を除く）の本文に本編で使わない指標・記号と「事前の仮説」がない．補助スライドの本文・ノートに「仮説」がない（042）．
        内容データの文字列（ノートを除く）と pptx の本文の両方を検索し，語ごとの件数を返す．"""
        hits: dict[str, int] = {}
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            texts = yaml_texts(spec) + [t for s in iter_shapes(slide.shapes) if s.name not in ("footnote", "pageno")
                                        for _, t in shape_paragraphs(s)]
            if spec["kind"] == "main" and spec.get("role") != "references":
                for w in MAIN_BANNED:
                    n = sum(t.count(w) for t in texts)
                    hits[f"main:{w}"] = hits.get(f"main:{w}", 0) + n
                    if n:
                        self.fail(f"書き方：本編 {spec['id']} に「{w}」がある（{n} 件）")
            if spec["kind"] == "supp":
                n = sum(t.count("仮説") for t in texts + [slide.notes_slide.notes_text_frame.text])
                hits["supp:仮説"] = hits.get("supp:仮説", 0) + n
                if n:
                    self.fail(f"書き方：補助 {spec['id']} に「仮説」がある（{n} 件）")
        return hits

    def check_display(self) -> dict[str, int]:
        """補助スライドの表示（044）：スライドの本文（ノートを除く）の語の件数を数え，0 でなければ違反．
        内容データの文字列（ノートを除く）と pptx の本文の両方を検索．"""
        hits = {"all:想定質問・答え：・Q＋数字": 0, "main:補助 S＋数字": 0, "supp（指標の例外を除く）:F 値・MRR": 0,
                "supp（ノートを含む）:ΔS・類似度ギャップ": 0}
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            texts = yaml_texts(spec) + [t for s in iter_shapes(slide.shapes) if s.name not in ("footnote", "pageno")
                                        for _, t in shape_paragraphs(s)]
            notes = [slide.notes_slide.notes_text_frame.text]
            checks = [("all:想定質問・答え：・Q＋数字", DISPLAY_BANNED, True),
                      ("main:補助 S＋数字", MAIN_SUPP_REF, spec["kind"] == "main"),
                      ("supp（指標の例外を除く）:F 値・MRR", SUPP_METRIC_BANNED, spec["kind"] == "supp" and not spec.get("extra_metrics")),
                      ("supp（ノートを含む）:ΔS・類似度ギャップ", SUPP_DELTA_S_BANNED, spec["kind"] == "supp")]
            for label, pat, applies in checks:
                scope = texts + notes if label.startswith("supp（ノートを含む）") else texts
                n = sum(len(pat.findall(t)) for t in scope) if applies else 0
                hits[label] += n
                if n:
                    self.fail(f"書き方：{spec['id']} の本文に {label.split(':', 1)[1]} が {n} 件ある")
        return hits

    def check_pos4(self) -> None:
        """品詞の 4 階層の図：セル（diagram:pos4:c<列>_<行>）の文字列が例文の解析結果（build_slides.pos4_table）と一致．"""
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            for b in diagram_blocks(spec):
                if b["diagram"] != "pos4":
                    continue
                expect = {f"diagram:pos4:c{c}_{r}": t for c, (surf, lv) in enumerate(pos4_table(b["sentence"]))
                          for r, t in enumerate([surf, *lv])}
                got = {s.name: s.text_frame.text for s in iter_shapes(slide.shapes) if s.name.startswith("diagram:pos4:c")}
                self.bump("pos4_cells", len(expect))
                if got != expect:
                    self.fail(f"図：{spec['id']} の品詞の 4 階層の図が解析結果と一致しない")

    # ---------- 数式

    def check_math(self) -> None:
        """数式：内容データの $ の対応が取れている．すべての数式がスライドごとに画像として置かれ，
        その代替テキスト（descr）の LaTeX の多重集合が内容データ（と図形の図の DIAGRAM_LATEX）の数式と一致．
        本文の文字（a:t）に平文の _ ^ \\ $ が残っていない．"""
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            sid = spec["id"]
            texts = yaml_texts(spec)
            if any(t.count("$") % 2 for t in texts):
                self.fail(f"数式：{sid} の内容データに対応の取れない $ がある")
            latex = [mt.group("math") for t in texts for mt in MARKUP.finditer(t) if mt.group("math")]
            latex += [t for b in diagram_blocks(spec) for t in DIAGRAM_LATEX[b["diagram"]]]
            expect = sorted(latex)  # すべての数式が画像（短い記号を含む，041）
            pics = [s for s in iter_shapes(slide.shapes) if s.name.startswith("math:")]
            got = sorted(p._element.nvPicPr.cNvPr.get("descr", "") for p in pics)
            self.bump("math_images", len(got))
            if got != expect:
                self.fail(f"数式：{sid} の数式の画像（代替テキスト）{len(got)} 個が内容データの画像にする数式 {len(expect)} 個と一致しない")
            for t in slide._element.iter(f"{{{NS_A}}}t"):
                self.bump("math_residue_checked")
                if re.search(r"[_^\\$]", t.text or ""):
                    self.fail(f"数式：{sid} の本文に平文の _ ^ \\ $ が残る：{t.text}")
            notes = slide.notes_slide.notes_text_frame.text
            if "$" in notes or "\\" in notes:
                self.fail(f"数式：{sid} のノートに数式の記法（$ や \\）がある（ノートは平文）")

    # ---------- 個人名・秘匿情報

    def secrets(self) -> list[str]:
        """照合する文字列（出力しない）：git の user.name・user.email・リモートのログイン名の部分，元のスライドのメタデータの人名．"""
        def git(*a: str) -> str:
            r = subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
            return r.stdout.strip()

        raw = [git("config", "user.name"), git("config", "user.email").split("@")[0], git("remote", "get-url", "origin")]
        parts = {p for s in raw for p in re.split(r"[^0-9A-Za-z]+", s) if len(p) >= 4} - {"https", "github", "git", "com", "nlps", "text",
                                                                                            "author", "identifier"}
        orig = PRES / "orig.pptx"
        if orig.exists():
            cp = Presentation(str(orig)).core_properties
            # 元のスライドのメタデータの文字列のうち，個人名の監査済みの storyline に既に現れる語（講義名等）は除く
            story = (PRES / "presentation_storyline.md").read_text(encoding="utf-8")
            parts |= {p for s in (cp.author, cp.last_modified_by) if s for p in re.split(r"[\s　,，]+", s) if len(p) >= 2 and p not in story}
        return sorted(parts)

    def check_privacy(self) -> None:
        cp = self.prs.core_properties
        for field in ("author", "last_modified_by", "title", "subject", "keywords", "comments", "category"):
            if getattr(cp, field):
                self.fail(f"個人情報：メタデータの {field} が空でない")
        secrets = self.secrets()
        self.counts["secret_terms"] = len(secrets)
        with zipfile.ZipFile(self.pptx_path) as z:
            for name in z.namelist():
                if not name.endswith((".xml", ".rels")):
                    continue
                xml = z.read(name).decode("utf-8", errors="replace")
                self.bump("xml_parts")
                if name == "docProps/app.xml" and re.search(r"<(Company|Manager)>[^<]+</", xml):
                    self.fail("個人情報：docProps/app.xml の会社名・管理者が空でない")
                for label, pat in (("メールアドレス", EMAIL), ("ローカルパス", LOCAL_PATH), ("学籍番号の形", STUDENT_ID)):
                    if pat.search(xml):
                        self.fail(f"個人情報：{name} に{label}の文字列がある")
                low = xml.lower()
                hits = sum(s.lower() in low for s in secrets)
                if hits:
                    self.fail(f"個人情報：{name} に git のユーザー名・ログイン名または元のスライドの人名の一部が {hits} 件ある")

    # ---------- はみ出し・重なり

    def check_layout(self) -> None:
        W, H = self.prs.slide_width, self.prs.slide_height
        for i, slide in enumerate(self.prs.slides, 1):
            boxes = []
            for s in iter_shapes(slide.shapes):
                self.bump("shapes")
                x, y, w, h = (Emu(v or 0) for v in (s.left, s.top, s.width, s.height))
                if x < 0 or y < 0 or x + w > W or y + h > H:
                    self.fail(f"はみ出し：スライド {i} の {s.name} がスライドの外に出る")
                if s.has_text_frame and s.text_frame.text.strip() and not s.name.startswith("diagram:"):
                    need = self.text_height_in(s)
                    if need > h / EMU_PER_IN + 0.05:
                        self.fail(f"はみ出し：スライド {i} の {s.name} の文字（見積もり {need:.2f} in）が枠（{h / EMU_PER_IN:.2f} in）を超える")
                if s.name.split(":")[0] in ("title", "claim", "body", "table", "deflist", "figure", "footnote"):
                    boxes.append((s.name, x, y, x + w, y + h))
            for a in range(len(boxes)):
                for b in range(a + 1, len(boxes)):
                    n1, ax0, ay0, ax1, ay1 = boxes[a]
                    n2, bx0, by0, bx1, by1 = boxes[b]
                    if min(ax1, bx1) - max(ax0, bx0) > 9144 and min(ay1, by1) - max(ay0, by0) > 9144:
                        self.fail(f"重なり：スライド {i} の {n1} と {n2} が重なる")

    @staticmethod
    def text_height_in(s: Any) -> float:
        tf = s.text_frame
        w = (s.width - (tf.margin_left or 0) - (tf.margin_right or 0)) / EMU_PER_IN
        total = 0.0
        for p in tf.paragraphs:
            if not paragraph_xml_text(p._p).strip():
                continue
            ppr = p._p.pPr
            mar = int(ppr.get("marL", 0)) / EMU_PER_IN if ppr is not None else 0.0
            # 出典番号（baseline 属性）は 0.7 倍の幅，数式の画像の場所は文字間隔（spc）を含む幅，行の高さは固定の行間
            em, pitch, size = paragraph_xml_metrics(p._p)
            space_after = ppr.find(f"{{{NS_A}}}spcAft/{{{NS_A}}}spcPts") if ppr is not None else None
            total += (n_lines_em(em, size, w - mar) * pitch + (int(space_after.get("val")) / 100 if space_after is not None else 0)) / 72
        return total + ((tf.margin_top or 0) + (tf.margin_bottom or 0)) / EMU_PER_IN

    # ---------- 内容のハッシュ

    def digest(self) -> str:
        h = hashlib.sha256()
        for slide in self.prs.slides:
            for s in iter_shapes(slide.shapes):
                h.update(f"{s.name}|{s.left}|{s.top}|{s.width}|{s.height}|".encode())
                h.update("\n".join(t for _, t in shape_paragraphs(s)).encode())
                if s.name.startswith(("figure:", "math:")):
                    h.update(hashlib.sha256(s.image.blob).digest())
            h.update(slide.notes_slide.notes_text_frame.text.encode())
        return h.hexdigest()


def walk_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ブロックの列（columns の中を含む，描画順）．"""
    return [x for b in blocks for x in ([b] if "columns" not in b else walk_blocks(b["columns"]["left"]) + walk_blocks(b["columns"]["right"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pptx", type=Path, default=PRES / "slides.pptx")
    parser.add_argument("--content", type=Path, default=PRES / "slides_content.yaml")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "full_006")
    parser.add_argument("--values-dir", type=Path, default=PROJECT_ROOT / "outputs" / "slide_figures")
    parser.add_argument("--sens-dir", default="outputs/sensitivity_029_L{L}", help="語彙選択に対する感度の run（{L} はチャンク長）")
    parser.add_argument("--digest", action="store_true", help="内容のハッシュのみ表示して終了")
    args = parser.parse_args()
    c = Checker(args)
    if args.digest:
        print(c.digest())
        return
    docs = Docs()
    c.check_structure()
    c.check_refs(RefSources(args.run_dir, docs.text["storyline"], args.sens_dir))
    c.check_numbers(docs)
    c.check_figures(docs)
    c.check_writing()
    c.check_math()
    c.check_privacy()
    c.check_layout()
    c.check_pos4()
    terms = c.check_terms()
    print("[terms] " + "，".join(f"{k} = {v}" for k, v in terms.items()))
    display = c.check_display()
    print("[display] " + "，".join(f"{k} = {v}" for k, v in display.items()))
    print("[supp] " + "，".join(f"{sid} → Slide {n}" for sid, n in c.supp_order))
    print("[counts] " + "，".join(f"{k} = {v}" for k, v in c.counts.items()))
    if c.violations:
        print(f"[NG] 違反 {len(c.violations)} 件")
        [print(f"  - {v}") for v in c.violations]
        sys.exit(1)
    print("[OK] 違反 0 件")


if __name__ == "__main__":
    main()
