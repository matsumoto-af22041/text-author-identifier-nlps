"""発表スライド（presentation/slides.pptx）を内容データ（presentation/slides_content.yaml）から生成．

使い方：
    uv run python scripts/build_slides.py [--content presentation/slides_content.yaml] [--out presentation/slides.pptx]

- 内容（文章・表・図の参照・出典・発表者ノート）は YAML に持ち，このスクリプトは配置と書式のみを担う．
  修正は YAML（と本スクリプト）で行い，pptx を直接編集しない（presentation/README.md）．
- 配色・フォント・縦横比は YAML の theme（presentation/orig.pptx のテーマから読み取った値）に従う．
- 配置は各ブロックの高さを文字数から見積もって縦に積む．領域からはみ出す場合は停止（黙って崩れたスライドを出さない）．
- pptx のメタデータ（作成者・最終更新者・会社名等）は空にする（presentation/README.md の「個人名・秘匿情報の扱い」）．
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.parts.image import Image as PptxImage
from pptx.util import Emu, Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = PROJECT_ROOT / "presentation"

# ---------------------------------------------------------------- 配置の定数（インチ）
MARGIN_X = 0.5
SECTION_Y, SECTION_H = 0.18, 0.3
TITLE_Y, TITLE_H = 0.42, 0.62
CLAIM_Y, CLAIM_H = 1.12, 0.62
BODY_Y = 1.92
BODY_BOTTOM = 6.78
FOOTNOTE_Y, FOOTNOTE_H = 6.86, 0.52
COLUMN_GAP = 0.35
BLOCK_GAP = 0.14
# 文字幅の見積もり（em 単位）：全角 1.0，半角 0.55．行の高さは文字サイズの 1.2 倍＋段落間隔
HALF_WIDTH_EM = 0.55
LINE_SPACING = 1.2
WIDTH_SAFETY = 0.9
PARA_SPACE_PT = 4
TEXT_PAD_IN = 0.1  # テキスト枠の内側の余白（上下左右，python-pptx の既定 0.1 / 0.05 より大きめに見積もる）
CELL_PAD_X_IN, CELL_PAD_Y_IN = 0.06, 0.04

MARKUP = re.compile(r"\*\*(?P<bold>.+?)\*\*|_\{(?P<sub>[^}]*)\}|\^\{(?P<sup>[^}]*)\}|\[\^(?P<ref>\d+)\]")
BASELINE = {"sub": "-25000", "sup": "30000", "ref": "30000"}


# ---------------------------------------------------------------- 文字列のマークアップと幅の見積もり


@dataclass(frozen=True)
class Run:
    text: str
    style: str = "normal"  # normal / bold / sub / sup / ref


def parse_markup(s: str) -> list[Run]:
    """`**太字**`，`_{下付き}`，`^{上付き}`，`[^n]`（出典の番号）を書式付きの Run に分解．"""
    runs: list[Run] = []
    pos = 0
    for m in MARKUP.finditer(s):
        if m.start() > pos:
            runs.append(Run(s[pos : m.start()]))
        kind = m.lastgroup
        assert kind is not None
        runs.append(Run(f"[{m.group(kind)}]" if kind == "ref" else m.group(kind), kind))
        pos = m.end()
    if pos < len(s):
        runs.append(Run(s[pos:]))
    return runs


def plain(s: str) -> str:
    return "".join(r.text for r in parse_markup(s))


def refs_in(s: str) -> list[int]:
    return [int(m.group("ref")) for m in MARKUP.finditer(s) if m.group("ref")]


def char_em(c: str) -> float:
    return 1.0 if unicodedata.east_asian_width(c) in ("W", "F", "A") else HALF_WIDTH_EM


def text_width_em(s: str) -> float:
    """表示幅（em）．下付き・上付き・出典番号は 0.7 倍の文字サイズ．"""
    return sum(sum(char_em(c) for c in r.text) * (1.0 if r.style in ("normal", "bold") else 0.7) for r in parse_markup(s))


def n_lines_em(width_em: float, size_pt: float, width_in: float) -> int:
    """表示幅 width_em（em）の文字列が幅 width_in の枠で何行になるか．"""
    per_line = width_in * 72 / size_pt * WIDTH_SAFETY  # 1 行に入る em 数（描画環境のフォント差の余裕を見込む）
    if per_line <= 1:
        raise ValueError(f"枠の幅 {width_in:.2f} in が文字サイズ {size_pt} pt に対して狭すぎる")
    return max(1, math.ceil(width_em / per_line))


def n_lines(s: str, size_pt: float, width_in: float) -> int:
    return n_lines_em(text_width_em(s), size_pt, width_in)


def paragraphs_height_in(paras: list[tuple[str, float, float]]) -> float:
    """[(文字列, 文字サイズ pt, 使える幅 in)] の合計の高さ（in）．"""
    return sum(n_lines(s, size, w) * size * LINE_SPACING / 72 + PARA_SPACE_PT / 72 for s, size, w in paras)


# ---------------------------------------------------------------- 書式の低水準ヘルパー


class Style:
    def __init__(self, theme: dict[str, Any]) -> None:
        self.latin = theme["font_latin"]
        self.ea = theme["font_ea"]
        self.c = {k: RGBColor.from_string(v) for k, v in theme["colors"].items()}

    def font(self, run: Any, size: float, color: str = "text", bold: bool = False) -> None:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = self.c[color]
        rpr = run._r.get_or_add_rPr()
        # a:latin は python-pptx の font.name，東アジアの文字は a:ea で指定
        run.font.name = self.latin
        ea = rpr.find(qn("a:ea"))
        if ea is None:
            ea = etree.SubElement(rpr, qn("a:ea"))
        ea.set("typeface", self.ea)

    def add_runs(self, para: Any, s: str, size: float, color: str = "text", bold: bool = False) -> None:
        for r in parse_markup(s):
            run = para.add_run()
            run.text = r.text
            is_bold = bold or r.style == "bold"
            self.font(run, size, "accent_dark" if r.style == "ref" else color, is_bold)
            if r.style in BASELINE:
                run._r.get_or_add_rPr().set("baseline", BASELINE[r.style])


def set_bullet(para: Any, level: int, size: float) -> None:
    """箇条書きの記号と字下げ（level 0：●，level 1：－）．"""
    ppr = para._p.get_or_add_pPr()
    indent = Pt(size * 1.1)
    ppr.set("marL", str(int(indent * (level + 1))))
    ppr.set("indent", str(-int(indent)))
    [ppr.remove(e) for e in ppr.findall(qn("a:buNone"))]
    bu = etree.SubElement(ppr, qn("a:buChar"))
    bu.set("char", "•" if level == 0 else "－")


def fill(shape: Any, color: RGBColor | None, line: RGBColor | None = None) -> None:
    if color is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)


def textbox(slide: Any, name: str, x: float, y: float, w: float, h: float, *, anchor: Any = MSO_ANCHOR.TOP) -> Any:
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(TEXT_PAD_IN / 2)
    tf.margin_top = tf.margin_bottom = Inches(TEXT_PAD_IN / 2)
    return tb


def cell_border(cell: Any, color: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcpr, qn(tag), w="6350", cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr"), val=color)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")
    # 塗りは境界線の後ろに置く（スキーマの順序：lnL..lnB → fill）
    fills = [e for e in tcpr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    [tcpr.remove(e) or tcpr.append(e) for e in fills]


# ---------------------------------------------------------------- ブロック（高さの見積もりと描画）


class Builder:
    def __init__(self, content: dict[str, Any]) -> None:
        self.theme = content["theme"]
        self.style = Style(self.theme)
        self.slide_w = self.theme["slide_width_in"]
        self.slide_h = self.theme["slide_height_in"]
        self.prs = Presentation()
        self.prs.slide_width = Inches(self.slide_w)
        self.prs.slide_height = Inches(self.slide_h)
        self.blank = self.prs.slide_layouts[6]

    # ---------- 高さの見積もり

    def bullet_paras(self, items: list[Any], size: float, w: float, level: int = 0) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for it in items:
            text, sub = (it, []) if isinstance(it, str) else (it["text"], it.get("sub", []))
            s = size if level == 0 else size - 2
            out.append((text, s, w - TEXT_PAD_IN - size * 1.1 * (level + 1) / 72))
            out += self.bullet_paras(sub, size, w, level + 1)
        return out

    def table_row_heights(self, t: dict[str, Any], w: float) -> list[float]:
        size = t.get("size", 14)
        widths = [w * r for r in t["widths"]]
        rows = [t["header"], *t["rows"]]

        def cell_h(text: str, cw: float) -> float:
            lines = sum(n_lines(p, size, cw - 2 * CELL_PAD_X_IN) for p in str(text).split("\n"))
            return lines * size * LINE_SPACING / 72 + 2 * CELL_PAD_Y_IN + 0.04

        return [max(cell_h(c, cw) for c, cw in zip(r, widths, strict=True)) for r in rows]

    def figure_size(self, path: str, w: float, h_max: float) -> tuple[float, float]:
        img = PptxImage.from_file(str(PRESENTATION_DIR / path))
        px_w, px_h = img.size
        scale = min(w / px_w, h_max / px_h)
        return px_w * scale, px_h * scale

    def block_height(self, b: dict[str, Any], w: float) -> float:
        if "bullets" in b:
            return paragraphs_height_in(self.bullet_paras(b["bullets"], b.get("size", 18), w)) + TEXT_PAD_IN
        if "table" in b:
            t = b["table"]
            if not math.isclose(sum(t["widths"]), 1.0, abs_tol=1e-6):
                raise ValueError(f"表の列幅の比の合計が 1 でない: {t['widths']}")
            if any(len(r) != len(t["header"]) for r in t["rows"]):
                raise ValueError(f"表の列数が見出しと一致しない: {t['header']}")
            return sum(self.table_row_heights(t, w))
        if "figure" in b:
            return self.figure_size(b["figure"], w, b.get("h", 4.0))[1]
        if "figures" in b:
            f = b["figures"]
            return f["h"]
        if "diagram" in b:
            return b.get("h", {"concept": 3.0, "split": 4.6}[b["diagram"]])
        if "box" in b:
            size = b.get("size", 17)
            return paragraphs_height_in([(b["box"], size, w - 0.3)]) + 0.22
        if "text" in b:
            size = b.get("size", 12 if b.get("muted") else 15)
            return paragraphs_height_in([(b["text"], size, w - TEXT_PAD_IN)]) + TEXT_PAD_IN
        if "columns" in b:
            c = b["columns"]
            wl = (w - COLUMN_GAP) * c["ratio"]
            wr = w - COLUMN_GAP - wl
            return max(self.stack_height(c["left"], wl), self.stack_height(c["right"], wr))
        raise ValueError(f"未知のブロック: {list(b)}")

    def stack_height(self, blocks: list[dict[str, Any]], w: float) -> float:
        return sum(self.block_height(b, w) for b in blocks) + BLOCK_GAP * (len(blocks) - 1)

    # ---------- 描画

    def draw_stack(self, slide: Any, sid: str, blocks: list[dict[str, Any]], x: float, y: float, w: float, bottom: float) -> None:
        need = self.stack_height(blocks, w)
        if y + need > bottom + 1e-6:
            raise ValueError(f"スライド {sid}：本文の高さ {need:.2f} in が領域 {bottom - y:.2f} in を超える（はみ出し）")
        for i, b in enumerate(blocks):
            h = self.block_height(b, w)
            self.draw_block(slide, f"{sid}.{i}", b, x, y, w, h)
            y += h + BLOCK_GAP

    def draw_block(self, slide: Any, bid: str, b: dict[str, Any], x: float, y: float, w: float, h: float) -> None:
        st = self.style
        if "bullets" in b:
            size = b.get("size", 18)
            tf = textbox(slide, f"body:{bid}", x, y, w, h).text_frame
            first = True

            def put(items: list[Any], level: int) -> None:
                nonlocal first
                for it in items:
                    text, sub = (it, []) if isinstance(it, str) else (it["text"], it.get("sub", []))
                    para = tf.paragraphs[0] if first else tf.add_paragraph()
                    first = False
                    para.space_after = Pt(PARA_SPACE_PT)
                    set_bullet(para, level, size)
                    st.add_runs(para, text, size if level == 0 else size - 2)
                    put(sub, level + 1)

            put(b["bullets"], 0)
        elif "table" in b:
            self.draw_table(slide, bid, b["table"], x, y, w)
        elif "figure" in b:
            fw, fh = self.figure_size(b["figure"], w, b.get("h", 4.0))
            pic = slide.shapes.add_picture(str(PRESENTATION_DIR / b["figure"]), Inches(x + (w - fw) / 2), Inches(y), Inches(fw), Inches(fh))
            pic.name = f"figure:{b['figure']}"
        elif "figures" in b:
            f = b["figures"]
            cols = f["cols"]
            rows = math.ceil(len(f["paths"]) / cols)
            cw, ch = w / cols, (f["h"] - 0.05 * (rows - 1)) / rows
            for i, p in enumerate(f["paths"]):
                fw, fh = self.figure_size(p, cw - 0.05, ch)
                r, c = divmod(i, cols)
                pic = slide.shapes.add_picture(str(PRESENTATION_DIR / p), Inches(x + c * cw + (cw - fw) / 2),
                                               Inches(y + r * (ch + 0.05) + (ch - fh) / 2), Inches(fw), Inches(fh))
                pic.name = f"figure:{p}"
        elif "diagram" in b:
            {"concept": self.diagram_concept, "split": self.diagram_split}[b["diagram"]](slide, x, y, w, h)
        elif "box" in b:
            shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
            shp.name = f"body:{bid}"
            fill(shp, st.c["highlight"], st.c["accent"])
            tf = shp.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Inches(0.15)
            st.add_runs(tf.paragraphs[0], b["box"], b.get("size", 17), bold=True)
        elif "text" in b:
            size = b.get("size", 12 if b.get("muted") else 15)
            tf = textbox(slide, f"body:{bid}", x, y, w, h).text_frame
            st.add_runs(tf.paragraphs[0], b["text"], size, "muted" if b.get("muted") else "text", bold=not b.get("muted"))
        elif "columns" in b:
            c = b["columns"]
            wl = (w - COLUMN_GAP) * c["ratio"]
            self.draw_stack(slide, f"{bid}L", c["left"], x, y, wl, y + h)
            self.draw_stack(slide, f"{bid}R", c["right"], x + wl + COLUMN_GAP, y, w - COLUMN_GAP - wl, y + h)
        else:
            raise ValueError(f"未知のブロック: {list(b)}")

    def draw_table(self, slide: Any, bid: str, t: dict[str, Any], x: float, y: float, w: float) -> None:
        st = self.style
        size = t.get("size", 14)
        heights = self.table_row_heights(t, w)
        rows = [t["header"], *t["rows"]]
        gf = slide.shapes.add_table(len(rows), len(t["header"]), Inches(x), Inches(y), Inches(w), Inches(sum(heights)))
        gf.name = f"table:{bid}"
        tbl = gf.table
        # 表のスタイルは「スタイルなし・グリッドなし」にし，塗りと罫線をセルごとに指定
        tbl._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
        tbl.first_row = True
        tbl.horz_banding = False
        for j, ratio in enumerate(t["widths"]):
            tbl.columns[j].width = Emu(int(Inches(w * ratio)))
        for i, (row, rh) in enumerate(zip(rows, heights, strict=True)):
            tbl.rows[i].height = Emu(int(Inches(rh)))
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(CELL_PAD_X_IN)
                cell.margin_top = cell.margin_bottom = Inches(CELL_PAD_Y_IN)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = st.c["light"] if i == 0 else st.c["background"]
                tf = cell.text_frame
                tf.word_wrap = True
                for k, line in enumerate(str(val).split("\n")):
                    para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                    para.alignment = PP_ALIGN.LEFT if j == 0 or not is_numeric_cell(line) else PP_ALIGN.RIGHT
                    st.add_runs(para, line, size, bold=(i == 0))
                cell_border(cell, self.theme["colors"]["line"])

    # ---------- 図形で描く図

    def label(self, slide: Any, name: str, s: str, x: float, y: float, w: float, h: float, size: float, *,
              color: str = "text", bold: bool = False, align: Any = PP_ALIGN.CENTER) -> None:
        tf = textbox(slide, name, x, y, w, h, anchor=MSO_ANCHOR.MIDDLE).text_frame
        tf.paragraphs[0].alignment = align
        self.style.add_runs(tf.paragraphs[0], s, size, color, bold)

    def rect(self, slide: Any, name: str, x: float, y: float, w: float, h: float, color: str, text: str = "",
             size: float = 12, line: str | None = None, bold: bool = False, shape: Any = MSO_SHAPE.RECTANGLE) -> Any:
        st = self.style
        shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.name = name
        fill(shp, st.c[color], st.c[line] if line else None)
        if text:
            tf = shp.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Inches(0.04)
            tf.margin_top = tf.margin_bottom = Inches(0.02)
            tf.paragraphs[0].alignment = PP_ALIGN.CENTER
            st.add_runs(tf.paragraphs[0], text, size, bold=bold)
        return shp

    def diagram_concept(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """Slide 1 の概念図：未知の文章 q と 4 作家の文章群のコサイン類似度を比べ，最も似ている作家を推定．"""
        authors = ["夏目漱石の文章群", "太宰治の文章群", "寺田寅彦の文章群", "田山花袋の文章群"]
        qw, qh = 1.6, 1.0
        qx, qy = x + 0.05, y + (h - qh) / 2
        self.rect(slide, "diagram:concept:q", qx, qy, qw, qh, "highlight", "未知の文章 q", 14, line="accent", bold=True)
        aw, ah = 2.1, 0.5
        ax = x + w - aw - 0.05
        top = y + 0.36  # 最上段の上に「最も似ている作家と推定」の注記を置く
        gap = (y + h - top - 4 * ah) / 3
        for i, a in enumerate(authors):
            ay = top + i * (ah + gap)
            self.rect(slide, f"diagram:concept:a{i}", ax, ay, aw, ah, "light" if i else "accent", a, 12, line="line", bold=(i == 0))
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(qx + qw), Inches(qy + qh / 2), Inches(ax), Inches(ay + ah / 2))
            conn.name = f"diagram:concept:edge{i}"
            conn.line.color.rgb = self.style.c["accent_dark" if i == 0 else "line"]
            conn.line.width = Pt(2.5 if i == 0 else 1.25)
        self.label(slide, "diagram:concept:cos", "コサイン類似度で比較[^1]", qx - 0.3, qy + qh + 0.05, qw + 0.6, 0.35, 12, color="muted")
        self.label(slide, "diagram:concept:pick", "最も似ている作家と推定", ax - 0.2, y, aw + 0.4, 0.32, 12, color="accent_dark", bold=True)

    def diagram_split(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """Slide 4 の図 1：作品内分割（各作品の一部が Test）と作品間分割（1 作品が丸ごと Test）．"""
        works = ["坊っちゃん", "こころ", "三四郎", "惜別", "斜陽", "映画芸術", "柿の種", "蒲団", "田舎教師"]
        test_work = 1  # 作品間分割の図で Test に回す作品（例示）
        pw = (w - 0.3) / 2
        head_h, legend_h = 0.62, 0.35
        rh = (h - head_h - legend_h - 0.1) / len(works)
        name_w = 0.95
        for p, (title, note) in enumerate([("作品内分割（Intra-work）", "各作品の一部を Test（5 シード）"),
                                            ("作品間分割（Inter-work）", "1 作品を丸ごと Test（9 fold）")]):
            px = x + p * (pw + 0.3)
            self.label(slide, f"diagram:split:title{p}", title, px, y, pw, 0.32, 13, bold=True)
            self.label(slide, f"diagram:split:note{p}", note, px, y + 0.3, pw, 0.28, 10.5, color="muted")
            bw = pw - name_w - 0.05
            for i, wk in enumerate(works):
                ry = y + head_h + i * rh
                self.label(slide, f"diagram:split:w{p}_{i}", wk, px, ry, name_w, rh, 10, align=PP_ALIGN.RIGHT)
                if p == 0:
                    tr = 0.75
                    self.rect(slide, f"diagram:split:tr{p}_{i}", px + name_w + 0.05, ry + 0.05, bw * tr, rh - 0.1, "line")
                    self.rect(slide, f"diagram:split:te{p}_{i}", px + name_w + 0.05 + bw * tr, ry + 0.05, bw * (1 - tr), rh - 0.1, "accent")
                else:
                    self.rect(slide, f"diagram:split:b{p}_{i}", px + name_w + 0.05, ry + 0.05, bw, rh - 0.1,
                              "accent" if i == test_work else "line")
        ly = y + h - legend_h
        for k, (color, s) in enumerate([("line", "Train"), ("accent", "Test")]):
            lx = x + w / 2 - 1.3 + k * 1.4
            self.rect(slide, f"diagram:split:legend{k}", lx, ly + 0.08, 0.35, 0.2, color)
            self.label(slide, f"diagram:split:legendtext{k}", s, lx + 0.4, ly, 0.8, 0.35, 11, align=PP_ALIGN.LEFT)

    # ---------- スライド

    def add_slide(self, sd: dict[str, Any], index: int, total_main: int) -> None:
        st = self.style
        sid = sd["id"]
        check_refs(sd)
        slide = self.prs.slides.add_slide(self.blank)
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = st.c["background"]
        content_w = self.slide_w - 2 * MARGIN_X

        # 区分の表示・題名・主張
        self.label(slide, "section", sd["section"], MARGIN_X, SECTION_Y, 4.0, SECTION_H, 11, color="muted", align=PP_ALIGN.LEFT)
        if sd.get("question"):
            self.label(slide, "question", f"想定質問：{sd['question']}", self.slide_w - MARGIN_X - 7.5, SECTION_Y, 7.5, SECTION_H, 11,
                       color="muted", align=PP_ALIGN.RIGHT)
        bar = self.rect(slide, "title_bar", MARGIN_X, TITLE_Y + 0.1, 0.09, TITLE_H - 0.2, "accent")
        bar.name = "title_bar"
        title_size = 26 if sd["layout"] == "title" else 24
        n_title = n_lines(sd["title"], title_size, content_w - 0.3)
        if n_title > 1:
            raise ValueError(f"スライド {sid}：題名が 1 行に収まらない（{n_title} 行）")
        self.label(slide, "title", sd["title"], MARGIN_X + 0.2, TITLE_Y, content_w - 0.2, TITLE_H, title_size, bold=True, align=PP_ALIGN.LEFT)
        # 主張は 1 行（17 pt で収まらなければ 15 pt まで下げる）
        claim_size = next((z for z in (17, 16, 15) if n_lines(sd["claim"], z, content_w - 0.3) == 1), None)
        if claim_size is None:
            raise ValueError(f"スライド {sid}：主張が 15 pt でも 1 行に収まらない")
        claim = self.rect(slide, "claim", MARGIN_X, CLAIM_Y, content_w, CLAIM_H, "highlight")
        claim.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        claim.text_frame.margin_left = Inches(0.18)
        st.add_runs(claim.text_frame.paragraphs[0], sd["claim"], claim_size, bold=True)

        # 本文
        if sd["layout"] == "single":
            self.draw_stack(slide, sid, sd["body"], MARGIN_X, BODY_Y, content_w, BODY_BOTTOM)
        elif sd["layout"] in ("split", "title"):
            wl = (content_w - COLUMN_GAP) * sd.get("ratio", 0.5)
            self.draw_stack(slide, f"{sid}L", sd["left"], MARGIN_X, BODY_Y, wl, BODY_BOTTOM)
            self.draw_stack(slide, f"{sid}R", sd["right"], MARGIN_X + wl + COLUMN_GAP, BODY_Y, content_w - COLUMN_GAP - wl, BODY_BOTTOM)
        else:
            raise ValueError(f"スライド {sid}：未知の layout {sd['layout']}")

        # 出典（スライド下部の薄い文字）とページ番号
        notes = sd.get("footnotes", [])
        if notes:
            line = "　".join(f"[{i}] {s}" for i, s in enumerate(notes, 1))
            if n_lines(line, 10, content_w - 1.2) > 2:
                raise ValueError(f"スライド {sid}：出典が 2 行を超える")
            self.label(slide, "footnote", line, MARGIN_X, FOOTNOTE_Y, content_w - 1.0, FOOTNOTE_H, 10, color="muted", align=PP_ALIGN.LEFT)
        page = f"{sid} / {total_main}" if sd["kind"] == "main" else sid
        self.label(slide, "pageno", page, self.slide_w - MARGIN_X - 1.0, FOOTNOTE_Y, 1.0, FOOTNOTE_H, 10, color="muted", align=PP_ALIGN.RIGHT)

        # 発表者ノート
        n = sd["notes"]
        lines = [f"想定時間：{n['time']}", "話すこと："] + [f"・{p}" for p in n["points"]]
        if n.get("runs"):
            lines.append(f"数値の出典（run 名）：{'，'.join(n['runs'])}")
        slide.notes_slide.notes_text_frame.text = "\n".join(lines)

    def clear_metadata(self) -> None:
        """作成者・最終更新者・会社名等を空にする（python-pptx の既定テンプレートは最終更新者に人名を含む）．"""
        cp = self.prs.core_properties
        for field in ("author", "last_modified_by", "title", "subject", "keywords", "comments", "category", "content_status",
                      "identifier", "language", "version"):
            setattr(cp, field, "")
        cp.revision = 1
        for part in self.prs.part.package.iter_parts():
            if str(part.partname) == "/docProps/app.xml":
                xml = part.blob.decode("utf-8")
                for tag in ("Company", "Manager"):
                    xml = re.sub(rf"<{tag}>[^<]*</{tag}>", f"<{tag}></{tag}>", xml)
                part._blob = xml.encode("utf-8")


DIAGRAM_REFS = {"concept": "[^1]", "split": ""}


def is_numeric_cell(s: str) -> bool:
    return bool(re.fullmatch(r"[+−\-]?[\d.,]+( ± [\d.]+)?|－", s.strip()))


def check_refs(sd: dict[str, Any]) -> None:
    """本文の [^n] と footnotes の対応を確認（番号の欠け・未使用の出典は停止）．"""
    texts: list[str] = [sd["title"], sd["claim"]]

    def walk(o: Any) -> None:
        if isinstance(o, str):
            texts.append(o)
        elif isinstance(o, dict):
            [walk(v) for k, v in o.items() if k not in ("figure", "figures", "diagram")]
        elif isinstance(o, list):
            [walk(v) for v in o]

    blocks = [b for k in ("body", "left", "right") for b in sd.get(k, [])]
    [walk(b) for b in blocks]
    # 概念図の「コサイン類似度で比較[^1]」は図形内の文字列
    texts += [DIAGRAM_REFS[b["diagram"]] for b in blocks if "diagram" in b]
    used = {r for t in texts for r in refs_in(t)}
    n = len(sd.get("footnotes", []))
    if used != set(range(1, n + 1)):
        raise ValueError(f"スライド {sd['id']}：出典番号の使用 {sorted(used)} と footnotes の数 {n} が一致しない")


def build(content_path: Path, out_path: Path) -> Path:
    content = yaml.safe_load(content_path.read_text(encoding="utf-8"))
    b = Builder(content)
    slides = content["slides"]
    total_main = sum(s["kind"] == "main" for s in slides)
    [b.add_slide(sd, i, total_main) for i, sd in enumerate(slides, 1)]
    b.clear_metadata()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    b.prs.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", type=Path, default=PRESENTATION_DIR / "slides_content.yaml")
    parser.add_argument("--out", type=Path, default=PRESENTATION_DIR / "slides.pptx")
    args = parser.parse_args()
    out = build(args.content, args.out)
    print(f"[output] {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")


if __name__ == "__main__":
    main()
