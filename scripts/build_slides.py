"""発表スライド（presentation/slides.pptx）を内容データ（presentation/slides_content.yaml）から生成．

使い方：
    uv run python scripts/build_slides.py [--content presentation/slides_content.yaml] [--out presentation/slides.pptx]
                                          [--lualatex <コマンド>] [--pdftocairo <コマンド>] [--math-cache outputs/slides_math]

- 内容（文章・表・図の参照・出典・発表者ノート）は YAML に持ち，このスクリプトは配置と書式のみを担う．
  修正は YAML（と本スクリプト）で行い，pptx を直接編集しない（presentation/README.md）．
- 配色・フォント・縦横比は YAML の theme（presentation/orig.pptx のテーマから読み取った値）に従う．
- 配置は各ブロックの高さを文字数から見積もって縦に積む．領域からはみ出す場合は停止（黙って崩れたスライドを出さない）．
- pptx のメタデータ（作成者・最終更新者・会社名等）は空にする（presentation/README.md の「個人名・秘匿情報の扱い」）．
- 数式：文字列中の `$...$`（LaTeX の記法）を 1 式ずつ standalone（amsmath，Computer Modern）の文書にして lualatex で PDF にし，
  pdftocairo（-png -transp -r <dpi>）で背景透過の PNG に変換して画像として置く（代替テキストに元の LaTeX）．
  表示の大きさは TeX の箱の寸法（幅・高さ・深さ）を本文の文字の大きさとの比で換算し，深さからベースラインをそろえる．
  段落の中では全角スペース 1 字（文字間隔 spc で画像の幅に合わせる）で場所を空け，固定の行間（spcPts）から求めた
  ベースラインの位置に画像を置く．数式と出典番号だけの段落のうち分数・総和等を含む式は別行立ての大きさ（\\displaystyle）で，
  行の上下中央に置く．短い記号（1 文字の変数等）も含め，すべての数式を画像にする（書体を Computer Modern にそろえる，041）．
  画像は本文・表のセル・定義の一覧・主張の帯・図形の中（中央ぞろえ）のいずれにも置ける．
  PNG は式・プリアンブル・解像度のハッシュをキーに outputs/slides_math/ にキャッシュ（Git 管理外）．
  コマンドが無い・コンパイルに失敗したら停止（平文への代替はしない）．発表者ノートは平文のまま．
- 行間：画像の位置を決めるため，すべての段落を固定の行間（文字の大きさ × LINE_SPACING，画像が高ければその高さ）にする．
- 定義の一覧（deflist）：罫線なしの 2 列の表（左列：用語・記号，右列：説明）で各列の行頭をそろえる．
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from lxml import etree
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font, LoadFlags
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
# 補助スライド：題名の直下に想定質問の文，その下に主張（答え）の帯（044）
SUPP_QUESTION_Y, SUPP_QUESTION_H = 1.04, 0.32
SUPP_CLAIM_Y, SUPP_CLAIM_H = 1.38, 0.5
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

MARKUP = re.compile(r"\*\*(?P<bold>.+?)\*\*|\$(?P<math>[^$]+)\$|\[\^(?P<ref>\d+)\]")
BASELINE = {"ref": "30000"}

# 数式：LaTeX を 1 式ずつ lualatex（standalone，amsmath，Computer Modern）で PDF にし，pdftocairo で背景透過の PNG に変換して置く
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
TEX_BASE_PT = 10.0  # standalone の文字の大きさ（pt）．表示の倍率は本文の文字の大きさ ÷ この値
TEX_PT_TO_BP = 72 / 72.27  # TeX の pt → PostScript の pt（スライドの pt）
TEX_PREAMBLE = "\\documentclass[10pt,border=0pt]{standalone}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n"
# PowerPoint の固定の行間（spcPts）での 1 行目のベースラインの位置：行の上端から行間 × 0.74（PowerPoint の PDF の実測，誤差 0.4 pt 以内）
BASELINE_RATIO = 0.74
GAP_CHAR = "\u2003"  # 数式の画像の場所を空ける全角スペース（幅 1 em）．文字間隔 spc で画像の幅に合わせる
# 数式と出典番号だけの段落で別行立て（\displaystyle，行の上下中央）にする式：分数・総和等の縦に大きい構造を含むもの
TALL_MATH = re.compile(r"\\(frac|dfrac|sum|prod|int)\b")


@dataclass(frozen=True)
class MathBox:
    """コンパイル済みの数式（寸法は TeX の pt，文字の大きさ TEX_BASE_PT のとき）．"""
    latex: str
    display: bool
    png: Path
    wd: float
    ht: float
    dp: float

    def scaled(self, size_pt: float) -> tuple[float, float, float]:
        """本文の文字の大きさ size_pt での（幅，ベースラインより上の高さ，深さ）（スライドの pt）．"""
        k = size_pt / TEX_BASE_PT * TEX_PT_TO_BP
        return self.wd * k, self.ht * k, self.dp * k


class TexRenderer:
    """1 式ずつ standalone の文書を lualatex でコンパイルし，pdftocairo で背景透過の PNG に変換．
    PNG と箱の寸法は式・文字の様式・プリアンブル・解像度のハッシュをキーに cache_dir にキャッシュ（同じ式なら同じ画像）．
    コマンドが無い・コンパイルに失敗したら停止（平文への代替はしない）．"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.lualatex = self.resolve(cfg["lualatex"])
        self.pdftocairo = self.resolve(cfg["pdftocairo"])
        self.dpi = int(cfg["dpi"])
        self.cache_dir = Path(cfg["cache_dir"])
        self.cache_dir = self.cache_dir if self.cache_dir.is_absolute() else PROJECT_ROOT / self.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memo: dict[tuple[str, bool], MathBox] = {}

    @staticmethod
    def resolve(cmd: str) -> str:
        found = shutil.which(cmd)
        if found is None:
            raise FileNotFoundError(f"数式の画像化のコマンドが見つからない: {cmd}（theme.math の設定か --lualatex / --pdftocairo で指定）")
        return found

    def render(self, latex: str, display: bool) -> MathBox:
        if (latex, display) in self.memo:
            return self.memo[(latex, display)]
        body = f"$\\displaystyle {latex}$" if display else f"${latex}$"
        doc = (TEX_PREAMBLE + "\\begin{document}%\n\\sbox0{" + body + "}%\n"
               "\\typeout{MATHBOX:\\the\\wd0,\\the\\ht0,\\the\\dp0}%\n\\usebox0%\n\\end{document}\n")
        key = hashlib.sha256(json.dumps({"doc": doc, "dpi": self.dpi}, sort_keys=True).encode("utf-8")).hexdigest()[:24]
        png, meta = self.cache_dir / f"{key}.png", self.cache_dir / f"{key}.json"
        if not (png.exists() and meta.exists()):
            with tempfile.TemporaryDirectory() as tmp:
                t = Path(tmp)
                (t / "eq.tex").write_text(doc, encoding="utf-8")
                r = subprocess.run([self.lualatex, "-interaction=nonstopmode", "-halt-on-error", "eq.tex"], cwd=t,
                                   capture_output=True, text=True, check=False)
                dims = re.search(r"MATHBOX:([\d.]+)pt,([\d.]+)pt,([\d.]+)pt", r.stdout)
                if r.returncode != 0 or dims is None or not (t / "eq.pdf").exists():
                    tail = "\n".join(ln for ln in r.stdout.splitlines() if ln.startswith("!"))[:600]
                    raise RuntimeError(f"数式のコンパイルに失敗: {latex!r}\n{tail}")
                r2 = subprocess.run([self.pdftocairo, "-png", "-transp", "-r", str(self.dpi), "-singlefile", "eq.pdf", "eq"], cwd=t,
                                    capture_output=True, text=True, check=False)
                if r2.returncode != 0 or not (t / "eq.png").exists():
                    raise RuntimeError(f"数式の PNG への変換に失敗: {latex!r}\n{r2.stderr[:600]}")
                shutil.copyfile(t / "eq.png", png)
                wd, ht, dp = map(float, dims.groups())
                meta.write_text(json.dumps({"latex": latex, "display": display, "wd": wd, "ht": ht, "dp": dp}, ensure_ascii=False),
                                encoding="utf-8")
        d = json.loads(meta.read_text(encoding="utf-8"))
        box = MathBox(latex, display, png, d["wd"], d["ht"], d["dp"])
        assert box.wd > 0 and box.ht + box.dp > 0, f"数式の箱の寸法が 0: {latex!r}"
        self.memo[(latex, display)] = box
        return box


class FontMetrics:
    """文字の送り幅（em）．全角の文字は 1 em，半角は本文の書体（Yu Gothic）・数式の文字（セリフ体）の実際の送り幅．"""

    def __init__(self, families: dict[str, tuple[str, str, str]]) -> None:
        self.fonts: dict[str, Any] = {}
        for name, (family, weight, style) in families.items():
            path = font_manager.findfont(font_manager.FontProperties(family=family, weight=weight, style=style), fallback_to_default=False)
            f = FT2Font(path)
            f.set_size(1000, 72)
            self.fonts[name] = f
        self.memo: dict[tuple[str, str], float] = {}

    def em(self, ch: str, face: str) -> float:
        if unicodedata.east_asian_width(ch) in ("W", "F") or ch == GAP_CHAR:
            return 1.0
        if (face, ch) not in self.memo:
            f = self.fonts[face]
            # 書体に字形が無い文字は PowerPoint が東アジアの書体で描くため 1 em とみなす
            self.memo[(face, ch)] = (f.load_char(ord(ch), flags=LoadFlags.NO_HINTING).linearHoriAdvance / 65536 / 1000
                                     if f.get_char_index(ord(ch)) else 1.0)
        return self.memo[(face, ch)]


@dataclass
class Seg:
    """段落の中の 1 片（文字列，または画像の数式）．"""
    text: str = ""
    size: float = 0.0
    face: str = "body"  # body / bold / ref
    math: MathBox | None = None


@dataclass
class ParaInfo:
    """描いた段落の配置の情報（画像の数式の位置を求めるため）．"""
    segs: list[Seg]
    size: float
    pitch: float  # 固定の行間（pt）
    space_after: float  # 段落後の間隔（pt）
    marl: float = 0.0  # 左の字下げ（pt）

    @property
    def has_image(self) -> bool:
        return any(s.math is not None for s in self.segs)


def char_em(c: str) -> float:
    return 1.0 if unicodedata.east_asian_width(c) in ("W", "F", "A") else HALF_WIDTH_EM


def n_lines_em(width_em: float, size_pt: float, width_in: float) -> int:
    """表示幅 width_em（em）の文字列が幅 width_in の枠で何行になるか．"""
    per_line = width_in * 72 / size_pt * WIDTH_SAFETY  # 1 行に入る em 数（描画環境のフォント差の余裕を見込む）
    if per_line <= 1:
        raise ValueError(f"枠の幅 {width_in:.2f} in が文字サイズ {size_pt} pt に対して狭すぎる")
    return max(1, math.ceil(width_em / per_line))


DISPLAY_PAD_PT = 4.0  # 別行立ての数式の上下の余白の合計（pt）


def line_pitch(size: float, boxes: list[tuple[float, float, float]], display: bool = False) -> float:
    """固定の行間（pt）：文字の大きさ × LINE_SPACING と，数式の画像が行に収まる高さの大きい方（0.01 pt 単位）．
    文中の数式はベースラインをそろえるため上下それぞれが収まる高さ，別行立ての数式は行の上下中央に置くため画像の高さ＋余白．"""
    need = [size * LINE_SPACING]
    for _, ht, dp in boxes:
        need += [ht + dp + DISPLAY_PAD_PT] if display else [(ht + 1.0) / BASELINE_RATIO, (dp + 0.5) / (1 - BASELINE_RATIO)]
    return math.ceil(max(need) * 100) / 100


def paragraph_xml_metrics(p: Any) -> tuple[float, float, float]:
    """pptx の段落（a:p）の表示幅（em），固定の行間（pt），文字の大きさ（pt）（check_slides の見積もり用．build と同じ規則）．"""
    sizes = [int(e.get("sz")) / 100 for e in p.iter(f"{{{NS_A}}}rPr") if e.get("sz")]
    size = max(sizes, default=18)
    em = 0.0
    for r in p.findall(f"{{{NS_A}}}r"):
        rpr = r.find(f"{{{NS_A}}}rPr")
        rsize = int(rpr.get("sz")) / 100 if rpr is not None and rpr.get("sz") else size
        scale = 0.7 if rpr is not None and rpr.get("baseline") else 1.0
        text = r.findtext(f"{{{NS_A}}}t") or ""
        spc = int(rpr.get("spc", 0)) / 100 if rpr is not None else 0.0
        em += (sum(map(char_em, text)) * scale * rsize + spc * len(text)) / size
    spc_pts = p.find(f"{{{NS_A}}}pPr/{{{NS_A}}}lnSpc/{{{NS_A}}}spcPts")
    pitch = int(spc_pts.get("val")) / 100 if spc_pts is not None else size * LINE_SPACING
    return em, pitch, size


def paragraph_xml_text(p: Any) -> str:
    """pptx の段落の文字列（数式の画像の場所の全角スペースを含む）．"""
    out: list[str] = []
    for c in p:
        tag = etree.QName(c).localname
        if tag in ("r", "fld"):
            out.append(c.findtext(f"{{{NS_A}}}t") or "")
        elif tag == "br":
            out.append("\n")
    return "".join(out)


# ---------------------------------------------------------------- 文字列のマークアップと幅の見積もり


@dataclass(frozen=True)
class Run:
    text: str
    style: str = "normal"  # normal / bold / math / ref


def parse_markup(s: str) -> list[Run]:
    """`**太字**`，`$数式$`（LaTeX の記法），`[^n]`（出典の番号）を書式付きの Run に分解．`$` の対応が取れなければ停止．"""
    runs: list[Run] = []
    pos = 0
    for mt in MARKUP.finditer(s):
        if mt.start() > pos:
            runs.append(Run(s[pos : mt.start()]))
        kind = mt.lastgroup
        assert kind is not None
        runs.append(Run(f"[{mt.group(kind)}]" if kind == "ref" else mt.group(kind), kind))
        pos = mt.end()
    if pos < len(s):
        runs.append(Run(s[pos:]))
    stray = [r.text for r in runs if r.style != "math" and "$" in r.text]
    if stray:
        raise ValueError(f"数式の $ の対応が取れない: {s!r}")
    return runs


def refs_in(s: str) -> list[int]:
    return [int(mt.group("ref")) for mt in MARKUP.finditer(s) if mt.group("ref")]


# ---------------------------------------------------------------- 書式の低水準ヘルパー


class Style:
    def __init__(self, theme: dict[str, Any], tex: TexRenderer, metrics: FontMetrics) -> None:
        self.latin = theme["font_latin"]
        self.ea = theme["font_ea"]
        self.c = {k: RGBColor.from_string(v) for k, v in theme["colors"].items()}
        self.tex = tex
        self.metrics = metrics

    def font(self, run: Any, size: float, color: str = "text", bold: bool = False, latin: str | None = None,
             italic: bool = False) -> None:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = self.c[color]
        rpr = run._r.get_or_add_rPr()
        # a:latin は python-pptx の font.name，東アジアの文字は a:ea で指定
        run.font.name = latin or self.latin
        ea = rpr.find(qn("a:ea"))
        if ea is None:
            ea = etree.SubElement(rpr, qn("a:ea"))
        ea.set("typeface", self.ea)

    def segments(self, s: str, size: float, bold: bool = False) -> list[Seg]:
        """文字列を描く片に分解．数式はすべて画像（数式と出典番号だけの段落で分数・総和等を含む式は別行立ての大きさ）．"""
        runs = parse_markup(s)
        display = (sum(r.style == "math" for r in runs) == 1 and all(r.style in ("math", "ref") for r in runs)
                   and any(r.style == "math" and TALL_MATH.search(r.text) for r in runs))
        segs: list[Seg] = []
        for r in runs:
            b = bold or r.style == "bold"
            if r.style == "math":
                segs.append(Seg(size=size, math=self.tex.render(r.text, display)))
            elif r.style == "ref":
                segs.append(Seg(r.text, size * 0.7, "ref"))
            else:
                segs.append(Seg(r.text, size, "bold" if b else "body"))
        return segs

    def width_em(self, s: str, size: float) -> float:
        """表示幅（em）の見積もり（枠の大きさ用．文字は char_em，数式の画像は実際の幅）．"""
        return sum(seg.math.scaled(size)[0] / size if seg.math else sum(map(char_em, seg.text)) * seg.size / size
                   for seg in self.segments(s, size))

    def pitch(self, s: str, size: float) -> float:
        segs = self.segments(s, size)
        return line_pitch(size, [seg.math.scaled(size) for seg in segs if seg.math], any(seg.math and seg.math.display for seg in segs))

    def add_runs(self, para: Any, s: str, size: float, color: str = "text", bold: bool = False, *, space_after: float = 0.0,
                 marl: float = 0.0, images: bool = True) -> ParaInfo:
        """段落に文字を追加し，固定の行間を設定．画像の数式の場所は全角スペース（文字間隔で画像の幅に合わせる）で空け，
        画像は呼び出し側が place_math で置く．images=False の場所に画像の数式があれば停止．"""
        segs = self.segments(s, size, bold)
        boxes = [seg.math.scaled(size) for seg in segs if seg.math]
        if boxes and not images:
            raise ValueError(f"この場所には画像の数式を置けない（短い記号にするか文を書き換える）: {s!r}")
        pitch = line_pitch(size, boxes, any(seg.math and seg.math.display for seg in segs))
        para.line_spacing = Pt(pitch)
        para.space_after = Pt(space_after)
        for seg in segs:
            run = para.add_run()
            if seg.math is not None:
                run.text = GAP_CHAR
                self.font(run, size, color, bold)
                run._r.get_or_add_rPr().set("spc", str(round((seg.math.scaled(size)[0] - size) * 100)))
            elif seg.face == "ref":
                run.text = seg.text
                self.font(run, size, "accent_dark", bold)
                run._r.get_or_add_rPr().set("baseline", BASELINE["ref"])
            else:
                run.text = seg.text
                self.font(run, size, color, seg.face == "bold")
        return ParaInfo(segs, size, pitch, space_after, marl)

    def wrap(self, info: ParaInfo, width_pt: float) -> tuple[int, list[tuple[int, float, Seg]], list[float]]:
        """段落を幅 width_pt で折り返したときの行数，画像の数式の（行，行頭からの x（pt），片），各行の幅（pt）．
        全角の文字は 1 字ずつ，半角の語は語ごとに送る（行頭禁則の文字の前では折り返さない）．"""
        face_of = {"body": "body", "bold": "bold", "ref": "body"}
        atoms: list[tuple[float, bool, bool, Seg | None]] = []  # (幅，空白か，前で折り返せるか，数式の片)
        for seg in info.segs:
            if seg.math is not None:
                atoms.append((seg.math.scaled(info.size)[0], False, True, seg))
                continue
            face = face_of.get(seg.face, seg.face)
            for tok in re.findall(r"\s+|[\x21-\x7e\u00a0-\u024f]+|.", seg.text):
                w = sum(self.metrics.em(ch, face) for ch in tok) * seg.size
                atoms.append((w, tok.isspace(), tok[0] not in KINSOKU_HEAD, None))
        line, x, out, widths = 0, 0.0, [], []
        for w, space, breakable, seg in atoms:
            if x > 0 and x + w > width_pt and breakable and not space:
                widths.append(x)
                line, x = line + 1, 0.0
            if space and x == 0 and line > 0:
                continue
            if seg is not None:
                out.append((line, x, seg))
            x += w
        return line + 1, out, widths + [x]


KINSOKU_HEAD = set("，．、。）」』】〕・：；？！ー")


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
    def __init__(self, content: dict[str, Any], tex: TexRenderer) -> None:
        self.theme = content["theme"]
        metrics = FontMetrics({"body": ("YuGothic", "medium", "normal"), "bold": ("YuGothic", "bold", "normal")})
        self.style = Style(self.theme, tex, metrics)
        self.math_log: list[tuple[str, str, bool]] = []  # (スライド，LaTeX，文中か)
        self.slide_w = self.theme["slide_width_in"]
        self.slide_h = self.theme["slide_height_in"]
        self.prs = Presentation()
        self.prs.slide_width = Inches(self.slide_w)
        self.prs.slide_height = Inches(self.slide_h)
        self.blank = self.prs.slide_layouts[6]

    # ---------- 高さの見積もり

    def n_lines(self, s: str, size: float, width_in: float) -> int:
        return n_lines_em(self.style.width_em(s, size), size, width_in)

    def paragraphs_height_in(self, paras: list[tuple[str, float, float]]) -> float:
        """[(文字列, 文字サイズ pt, 使える幅 in)] の合計の高さ（in）．行の高さは固定の行間（画像の数式があれば高くなる）．"""
        return sum(self.n_lines(s, size, w) * self.style.pitch(s, size) / 72 + PARA_SPACE_PT / 72 for s, size, w in paras)

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
            paras = [(p, size, cw - 2 * CELL_PAD_X_IN) for p in str(text).split("\n")]
            return self.paragraphs_height_in(paras) - len(paras) * PARA_SPACE_PT / 72 + 2 * CELL_PAD_Y_IN + 0.04

        return [max(cell_h(c, cw) for c, cw in zip(r, widths, strict=True)) for r in rows]

    def deflist_layout(self, d: dict[str, Any], w: float) -> list[tuple[list[tuple[str, float, float]], list[tuple[str, float, float]]]]:
        """定義の一覧の各行の [(文字列, pt, 幅)]（左列，右列）．1 要素の行は 2 列にまたがる．説明が文字列の列なら箇条書き．"""
        size = d.get("size", 16)
        wl, wr = (w * r - 2 * CELL_PAD_X_IN for r in d.get("widths", DEFLIST_WIDTHS))
        out = []
        for row in d["rows"]:
            if len(row) == 1:
                out.append(([(row[0], size, w - 2 * CELL_PAD_X_IN)], []))
            elif len(row) == 2:
                desc = row[1]
                right = ([(it, size - 2, wr - size * 1.1 / 72) for it in desc] if isinstance(desc, list) else [(desc, size, wr)])
                out.append(([(row[0], size, wl)], right))
            else:
                raise ValueError(f"定義の一覧の行は [用語, 説明] か [文字列]: {row}")
        return out

    def deflist_row_heights(self, d: dict[str, Any], w: float) -> list[float]:
        if not math.isclose(sum(d.get("widths", DEFLIST_WIDTHS)), 1.0, abs_tol=1e-6):
            raise ValueError(f"定義の一覧の列幅の比の合計が 1 でない: {d['widths']}")
        return [max(self.paragraphs_height_in(left), self.paragraphs_height_in(right)) + 2 * CELL_PAD_Y_IN + 0.06
                for left, right in self.deflist_layout(d, w)]

    def figure_size(self, path: str, w: float, h_max: float) -> tuple[float, float]:
        img = PptxImage.from_file(str(PRESENTATION_DIR / path))
        px_w, px_h = img.size
        scale = min(w / px_w, h_max / px_h)
        return px_w * scale, px_h * scale

    def block_height(self, b: dict[str, Any], w: float) -> float:
        if "bullets" in b:
            return self.paragraphs_height_in(self.bullet_paras(b["bullets"], b.get("size", 18), w)) + TEXT_PAD_IN
        if "deflist" in b:
            return sum(self.deflist_row_heights(b["deflist"], w))
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
            return b.get("h", DIAGRAM_H[b["diagram"]])
        if "box" in b:
            size = b.get("size", 17)
            return self.paragraphs_height_in([(b["box"], size, w - 0.3)]) + 0.22
        if "text" in b:
            size = b.get("size", 12 if b.get("muted") else 15)
            return self.paragraphs_height_in([(b["text"], size, w - TEXT_PAD_IN)]) + TEXT_PAD_IN
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
            infos: list[ParaInfo] = []

            def put(items: list[Any], level: int) -> None:
                for it in items:
                    text, sub = (it, []) if isinstance(it, str) else (it["text"], it.get("sub", []))
                    para = tf.paragraphs[0] if not infos else tf.add_paragraph()
                    set_bullet(para, level, size)
                    infos.append(st.add_runs(para, text, size if level == 0 else size - 2, space_after=PARA_SPACE_PT,
                                             marl=size * 1.1 * (level + 1)))
                    put(sub, level + 1)

            put(b["bullets"], 0)
            self.place_math(slide, bid, infos, x * 72 + TEXT_PAD_IN * 36, (w - TEXT_PAD_IN) * 72, y * 72 + TEXT_PAD_IN * 36)
        elif "table" in b:
            self.draw_table(slide, bid, b["table"], x, y, w)
        elif "deflist" in b:
            self.draw_deflist(slide, bid, b["deflist"], x, y, w)
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
            {"concept": self.diagram_concept, "split": self.diagram_split, "a_blocks": self.diagram_a_blocks,
             "svd": self.diagram_svd, "flow": self.diagram_flow, "balance": self.diagram_balance, "b_blocks": self.diagram_b_blocks,
             "pos4": lambda *a: self.diagram_pos4(*a, sentence=b["sentence"])}[b["diagram"]](slide, x, y, w, h)
        elif "box" in b:
            shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
            shp.name = f"body:{bid}"
            fill(shp, st.c["highlight"], st.c["accent"])
            tf = shp.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Inches(0.15)
            info = st.add_runs(tf.paragraphs[0], b["box"], b.get("size", 17), bold=True)
            self.place_math(slide, f"body:{bid}", [info], (x + 0.15) * 72, (w - 0.3) * 72, (y + 0.05) * 72, middle_h=(h - 0.1) * 72)
        elif "text" in b:
            size = b.get("size", 12 if b.get("muted") else 15)
            tf = textbox(slide, f"body:{bid}", x, y, w, h).text_frame
            info = st.add_runs(tf.paragraphs[0], b["text"], size, "muted" if b.get("muted") else "text", bold=not b.get("muted"))
            self.place_math(slide, bid, [info], x * 72 + TEXT_PAD_IN * 36, (w - TEXT_PAD_IN) * 72, y * 72 + TEXT_PAD_IN * 36)
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
        col_x = [x + w * sum(t["widths"][:j]) for j in range(len(t["widths"]))]
        for i, (row, rh) in enumerate(zip(rows, heights, strict=True)):
            tbl.rows[i].height = Emu(int(Inches(rh)))
            row_y = y + sum(heights[:i])
            for j, val in enumerate(row):
                cell = tbl.cell(i, j)
                cell.margin_left = cell.margin_right = Inches(CELL_PAD_X_IN)
                cell.margin_top = cell.margin_bottom = Inches(CELL_PAD_Y_IN)
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.fill.solid()
                cell.fill.fore_color.rgb = st.c["light"] if i == 0 else st.c["background"]
                tf = cell.text_frame
                tf.word_wrap = True
                infos = []
                for k, line in enumerate(str(val).split("\n")):
                    para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                    numeric = j > 0 and is_numeric_cell(line)
                    para.alignment = PP_ALIGN.RIGHT if numeric else PP_ALIGN.LEFT
                    infos.append(st.add_runs(para, line, size, bold=(i == 0), images=not numeric))
                cw = w * t["widths"][j]
                self.place_math(slide, f"{bid}.{i}.{j}", infos, (col_x[j] + CELL_PAD_X_IN) * 72, (cw - 2 * CELL_PAD_X_IN) * 72,
                                (row_y + CELL_PAD_Y_IN) * 72, middle_h=(rh - 2 * CELL_PAD_Y_IN) * 72)
                cell_border(cell, self.theme["colors"]["line"])

    def draw_deflist(self, slide: Any, bid: str, d: dict[str, Any], x: float, y: float, w: float) -> None:
        """定義の一覧：罫線なしの 2 列の表（左列：用語・記号，右列：説明）．行の区切りは薄い横線 1 本．"""
        st = self.style
        size = d.get("size", 16)
        widths = d.get("widths", DEFLIST_WIDTHS)
        heights = self.deflist_row_heights(d, w)
        gf = slide.shapes.add_table(len(d["rows"]), 2, Inches(x), Inches(y), Inches(w), Inches(sum(heights)))
        gf.name = f"deflist:{bid}"
        tbl = gf.table
        tbl._tbl.tblPr.find(qn("a:tableStyleId")).text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
        tbl.first_row = False
        tbl.horz_banding = False
        for j, ratio in enumerate(widths):
            tbl.columns[j].width = Emu(int(Inches(w * ratio)))
        for i, (row, rh) in enumerate(zip(d["rows"], heights, strict=True)):
            tbl.rows[i].height = Emu(int(Inches(rh)))
            row_y = y + sum(heights[:i])
            if len(row) == 1:
                tbl.cell(i, 0).merge(tbl.cell(i, 1))
            cells = [(tbl.cell(i, 0), [row[0]], True)] if len(row) == 1 else \
                [(tbl.cell(i, 0), [row[0]], True), (tbl.cell(i, 1), row[1] if isinstance(row[1], list) else [row[1]], isinstance(row[1], list))]
            for j, (cell, paras, plain_or_list) in enumerate(cells):
                cell.margin_left = cell.margin_right = Inches(CELL_PAD_X_IN)
                cell.margin_top = cell.margin_bottom = Inches(CELL_PAD_Y_IN)
                cell.vertical_anchor = MSO_ANCHOR.TOP
                cell.fill.background()
                tf = cell.text_frame
                tf.word_wrap = True
                is_list = j == 1 and plain_or_list
                infos = []
                for k, text in enumerate(paras):
                    para = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                    para.alignment = PP_ALIGN.LEFT
                    if is_list:
                        set_bullet(para, 0, size - 2)
                    infos.append(st.add_runs(para, text, size - 2 if is_list else size, bold=(j == 0 and len(row) == 2),
                                             space_after=PARA_SPACE_PT, marl=(size - 2) * 1.1 if is_list else 0.0))
                cx, cw = (x, w) if len(row) == 1 else (x + w * sum(widths[:j]), w * widths[j])
                self.place_math(slide, f"{bid}.{i}.{j}", infos, (cx + CELL_PAD_X_IN) * 72, (cw - 2 * CELL_PAD_X_IN) * 72,
                                (row_y + CELL_PAD_Y_IN) * 72)
            line = self.theme["colors"]["line"]
            top = None if i == 0 else line
            bottom = None if i == len(d["rows"]) - 1 else line
            [deflist_border(tbl.cell(i, j), top, bottom) for j in range(2)]

    def place_math(self, slide: Any, bid: str, infos: list[ParaInfo], left_pt: float, width_pt: float, top_pt: float,
                   middle_h: float | None = None, align: str = "left") -> None:
        """段落の列の画像の数式を，空けた場所（行，x）とベースラインの位置に置く．
        top_pt はテキストの領域の上端（枠の余白の内側）．middle_h を与えると上下中央ぞろえ（領域の高さ）．
        align="center" は各行を幅の中央にそろえた位置（行の幅は wrap の見積もり）．"""
        laid = [(info, *self.style.wrap(info, width_pt - info.marl)) for info in infos]
        y = top_pt
        if middle_h is not None:
            text_h = sum(n * info.pitch + info.space_after for info, n, _, _ in laid)
            y = top_pt + (middle_h - text_h) / 2
        for k, (info, n, slots, line_w) in enumerate(laid):
            for line, dx, seg in slots:
                if align == "center":
                    dx += (width_pt - info.marl - line_w[line]) / 2
                assert seg.math is not None
                w, ht, dp = seg.math.scaled(info.size)
                line_top = y + line * info.pitch
                # 文中の数式はベースラインをそろえる．別行立ての数式（数式と出典番号だけの段落）は行の上下中央
                top = line_top + (info.pitch - ht - dp) / 2 if seg.math.display else line_top + BASELINE_RATIO * info.pitch - ht
                pic = slide.shapes.add_picture(str(seg.math.png), Emu(int(Pt(left_pt + info.marl + dx))), Emu(int(Pt(top))),
                                               Emu(int(Pt(w))), Emu(int(Pt(ht + dp))))
                pic.name = f"math:{bid}.{k}"
                pic._element.nvPicPr.cNvPr.set("descr", seg.math.latex)
                at_start = line == 0 and dx == 0
                self.math_log.append((bid, seg.math.latex, not (at_start or seg.math.display)))
            y += n * info.pitch + info.space_after

    def math_at(self, slide: Any, name: str, latex: str, cx: float, cy: float, size: float) -> None:
        """図形の図の数式：画像（文中の大きさ）を中心 (cx, cy)（インチ）に置く．"""
        box = self.style.tex.render(latex, False)
        w, ht, dp = box.scaled(size)
        pic = slide.shapes.add_picture(str(box.png), Emu(int(Inches(cx) - Pt(w / 2))), Emu(int(Inches(cy) - Pt((ht + dp) / 2))),
                                       Emu(int(Pt(w))), Emu(int(Pt(ht + dp))))
        pic.name = f"math:{name}"
        pic._element.nvPicPr.cNvPr.set("descr", latex)
        self.math_log.append((name, latex, False))

    # ---------- 図形で描く図

    def label(self, slide: Any, name: str, s: str, x: float, y: float, w: float, h: float, size: float, *,
              color: str = "text", bold: bool = False, align: Any = PP_ALIGN.CENTER) -> None:
        """1 段落の文字（上下中央）．数式の画像は左・中央ぞろえのみ置ける．"""
        tf = textbox(slide, name, x, y, w, h, anchor=MSO_ANCHOR.MIDDLE).text_frame
        tf.paragraphs[0].alignment = align
        info = self.style.add_runs(tf.paragraphs[0], s, size, color, bold)
        if info.has_image:
            if align not in (PP_ALIGN.CENTER, PP_ALIGN.LEFT):
                raise ValueError(f"右ぞろえの文字に数式の画像は置けない: {s!r}")
            pad = TEXT_PAD_IN / 2
            self.place_math(slide, name, [info], (x + pad) * 72, (w - 2 * pad) * 72, (y + pad) * 72, middle_h=(h - 2 * pad) * 72,
                            align="center" if align == PP_ALIGN.CENTER else "left")

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
            info = st.add_runs(tf.paragraphs[0], text, size, bold=bold)
            if info.has_image:
                self.place_math(slide, name, [info], (x + 0.04) * 72, (w - 0.08) * 72, (y + 0.02) * 72, middle_h=(h - 0.04) * 72,
                                align="center")
        return shp

    def diagram_concept(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """Slide 2 の概念図：未知の文章 q と 4 作家の文章群のコサイン類似度を比べ，最も似ている作家を推定．"""
        authors = ["夏目漱石の文章群", "太宰治の文章群", "寺田寅彦の文章群", "田山花袋の文章群"]
        qw, qh = 1.6, 1.0
        qx, qy = x + 0.05, y + (h - qh) / 2
        self.rect(slide, "diagram:concept:q", qx, qy, qw, qh, "highlight", "未知の文章 $q$", 14, line="accent", bold=True)
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
        """Slide 7 の図 1：作品内分割（各作品の一部が Test）と作品間分割（1 作品が丸ごと Test）．"""
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

    def diagram_a_blocks(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """A の特徴量ベクトルの構成：文字 Bi-gram のブロックと記号のブロックをつなげたもの（字種のブロックは旧案のみ）．
        各ブロックの次元の例を小さな升目で示す（次元数の比は実寸ではない）．"""
        st = self.style
        self.label(slide, "diagram:ablocks:title", "A の特徴量ベクトル $\\vec{x}$：ブロックをつなげたもの（各次元の値は相対頻度）",
                   x, y, w, 0.34, 13, bold=True, align=PP_ALIGN.LEFT)
        top, cell_h = y + 0.42, 0.42
        blocks = [("bigram", "文字 Bi-gram ブロック（上位 1000 次元）", ["「た。」", "「して」", "「いう」", "…"], 0.55, "light", "line"),
                  ("symbol", "記号ブロック（3 次元）", ["読点", "文末記号", "開き括弧類"], 0.29, "highlight", "accent"),
                  ("chartype", "字種ブロック（旧案のみ．既定では使わない）", ["漢字", "ひらがな", "…"], 0.16, "background", "line")]
        bx = x
        for name, caption, cells, ratio, color, line in blocks:
            bw = w * ratio - 0.08
            frame = self.rect(slide, f"diagram:ablocks:{name}", bx, top, bw, cell_h + 0.16, color, line=line)
            if name == "chartype":
                frame.line.dash_style = 4  # 破線（MSO_LINE_DASH_STYLE.DASH）：既定では使わないブロック
            cw = (bw - 0.12) / len(cells)
            for k, c in enumerate(cells):
                self.rect(slide, f"diagram:ablocks:{name}{k}", bx + 0.06 + k * cw, top + 0.08, cw - 0.06, cell_h, "background", c, 12,
                          line="line")
            self.label(slide, f"diagram:ablocks:{name}cap", caption, bx, top + cell_h + 0.2, bw, 0.32, 11.5,
                       color="muted" if name == "chartype" else "text", align=PP_ALIGN.LEFT)
            bx += w * ratio
        del st

    def diagram_svd(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """特異値分解の模式図：X（N × M）= U_K（N × K）Σ_K（K × K）V_K^⊤（K × M）．長方形の縦横の比で行列の形を示す
        （N：Train チャンク数，M：語彙の次元数，K：残す次元数．比は実寸ではない）．"""
        size = 15
        dim_h = 0.34
        gap, eq_w = 0.18, 0.4
        r_m, r_k = 1.35, 0.26  # M・K に当たる長さの N に対する比（模式）
        k_min = 0.62  # K の辺の最小の長さ（下の次元の表記「N × K」等が隣と重ならない幅，042）
        hN = min(h - dim_h - 0.1, (w - eq_w - 2 * gap - 2 * k_min) / (2 * r_m))  # N に当たる高さ（幅に収まるように）
        wM, k = hN * r_m, max(hN * r_k, k_min)
        total = wM + eq_w + k + gap + k + gap + wM
        x0 = x + (w - total) / 2
        y += (h - dim_h - 0.1 - hN) / 2
        shapes = [("X", "X", wM, hN, "light", "N \\times M"), ("eq", None, eq_w, 0, None, None),
                  ("U", "U_K", k, hN, "highlight", "N \\times K"), ("gap", None, gap, 0, None, None),
                  ("S", "\\Sigma_K", k, k, "highlight", "K \\times K"), ("gap", None, gap, 0, None, None),
                  ("V", "V_K^\\top", wM, k, "highlight", "K \\times M")]
        cx = x0
        for name, tex, sw, sh, color, dim in shapes:
            if name == "eq":
                self.math_at(slide, "diagram:svd:eq", "=", cx + sw / 2, y + hN / 2, size + 4)
            elif tex is not None:
                self.rect(slide, f"diagram:svd:{name}", cx, y, sw, sh, color, line="strong" if name == "X" else "accent_dark")
                self.math_at(slide, f"diagram:svd:{name}", tex, cx + sw / 2, y + sh / 2, size + 2)
                self.math_at(slide, f"diagram:svd:{name}dim", dim, cx + sw / 2, y + hN + 0.1 + dim_h / 2, size - 2)
            cx += sw

    def arrow(self, slide: Any, name: str, x1: float, y1: float, x2: float, y2: float, *, color: str = "strong",
              dashed: bool = False, width: float = 1.5) -> None:
        """矢印（直線のコネクタ，終点に三角の矢じり）．座標はインチ．"""
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        conn.name = name
        conn.line.color.rgb = self.style.c[color]
        conn.line.width = Pt(width)
        if dashed:
            conn.line.dash_style = 4  # MSO_LINE_DASH_STYLE.DASH
        etree.SubElement(conn.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle", w="med", len="med")

    def diagram_flow(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """システムの機能と処理の流れ：入力 → 前処理 → 形態素解析 → 特徴抽出 A～D → コサイン類似度 → 重心法 → 出力，
        出力の下に評価，特徴抽出～重心法の下に学習（Train のみ）．"""
        steps = [("in", "入力：著者不明の文章", "accent"), ("pre", "前処理：作品の抽出・チャンク分割", "light"),
                 ("morph", "形態素解析", "light"), ("feat", "特徴抽出 A～D", "light"), ("cos", "コサイン類似度", "light"),
                 ("clf", "重心法で著者を推定", "light"), ("out", "出力：推定した著者と類似度の順位", "accent")]
        gap = 0.28
        bw = (w - gap * (len(steps) - 1)) / len(steps)
        bh = 0.9
        xs = [x + i * (bw + gap) for i in range(len(steps))]
        for (name, text, color), bx in zip(steps, xs, strict=True):
            self.rect(slide, f"diagram:flow:{name}", bx, y, bw, bh, color, text, 11.5, line="line", bold=name in ("in", "out"))
        for i in range(len(steps) - 1):
            self.arrow(slide, f"diagram:flow:edge{i}", xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)
        low_y, low_h = y + bh + 0.55, h - bh - 0.55
        i_feat, i_clf, i_out = 3, 5, 6
        lx, lw = xs[i_feat], xs[i_clf] + bw - xs[i_feat]
        learn = self.rect(slide, "diagram:flow:learn", lx, low_y, lw, low_h, "highlight",
                          "学習（Train チャンクのみ）：語彙・IDF・SVD・作家の重心を算出", 12.5, line="accent_dark")
        learn.line.dash_style = 4
        for i in (i_feat, i_clf):
            self.arrow(slide, f"diagram:flow:learn{i}", xs[i] + bw / 2, low_y, xs[i] + bw / 2, y + bh, color="accent_dark", dashed=True)
        self.rect(slide, "diagram:flow:eval", xs[i_out], low_y, bw, low_h, "background", "評価：正解の著者と比べる", 12.5, line="line")
        self.arrow(slide, "diagram:flow:edge_eval", xs[i_out] + bw / 2, y + bh, xs[i_out] + bw / 2, low_y)

    def diagram_balance(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """Train チャンク数の間引き：作品内分割・作品間分割の各分割で，Train を作家ごとに最少の作家の数 n_min にそろえる．
        薄い棒：間引き前の Train，濃い棒：残す Train（長さは模式），Test は間引かない．"""
        authors = ["漱石", "太宰", "寺田", "花袋"]
        panels = [("作品内分割", [1.0, 0.45, 0.25, 0.5]), ("作品間分割（寺田の作品が Test の例）", [1.0, 0.45, 0.12, 0.5])]
        pw = (w - 0.25) / 2
        head, foot, nmin_h = 0.3, 0.45, 0.28
        rh = (h - head - foot - nmin_h) / len(authors)
        name_w = 0.5
        for p, (title, lengths) in enumerate(panels):
            px = x + p * (pw + 0.25)
            self.label(slide, f"diagram:balance:title{p}", title, px, y, pw, head, 10.5, bold=True, align=PP_ALIGN.LEFT)
            bar_w = pw - name_w - 0.45
            n_min = min(lengths) * bar_w
            for i, (a, ln) in enumerate(zip(authors, lengths, strict=True)):
                ry = y + head + i * rh
                self.label(slide, f"diagram:balance:a{p}_{i}", a, px, ry, name_w, rh, 10.5, align=PP_ALIGN.RIGHT)
                self.rect(slide, f"diagram:balance:full{p}_{i}", px + name_w + 0.05, ry + 0.05, ln * bar_w, rh - 0.1, "light", line="line")
                self.rect(slide, f"diagram:balance:kept{p}_{i}", px + name_w + 0.05, ry + 0.05, n_min, rh - 0.1, "accent")
            lx = px + name_w + 0.05 + n_min
            conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(lx), Inches(y + head), Inches(lx), Inches(y + head + len(authors) * rh))
            conn.name = f"diagram:balance:nmin{p}"
            conn.line.color.rgb = self.style.c["accent_dark"]
            conn.line.width = Pt(1.25)
            conn.line.dash_style = 4
            self.math_at(slide, f"diagram:balance:nminlabel{p}", "n_{\\min}", lx, y + head + len(authors) * rh + nmin_h / 2, 11)
        for k, text in enumerate(["黄：残す Train チャンク（最少の作家の数にそろえる）　灰：間引く分", "Test は間引かない（棒の長さは模式）"]):
            self.label(slide, f"diagram:balance:legend{k}", text, x, y + h - foot + k * foot / 2, w, foot / 2, 10.5, color="muted",
                       align=PP_ALIGN.LEFT)

    def diagram_b_blocks(self, slide: Any, x: float, y: float, w: float, h: float) -> None:
        """B の特徴量ベクトルの構成：品詞 2-gram のブロックと語（付属語）のブロックをつなげたもの（diagram_a_blocks と同じ形式）．"""
        self.label(slide, "diagram:bblocks:title", "B の特徴量ベクトル：ブロックをつなげたもの（各次元の値は相対頻度）",
                   x, y, w, 0.34, 13, bold=True, align=PP_ALIGN.LEFT)
        top, cell_h = y + 0.42, 0.42
        blocks = [("pos", "品詞 2-gram ブロック（隣り合う品詞の組）",
                   ["名詞-普通名詞 → 助詞-格助詞", "助詞-格助詞 → 動詞-一般", "…"], 0.64, "light", "line"),
                  ("words", "語ブロック（付属語 275 次元，表層形）", ["を", "た", "は", "…"], 0.36, "highlight", "accent")]
        bx = x
        for name, caption, cells, ratio, color, line in blocks:
            bw = w * ratio - 0.08
            self.rect(slide, f"diagram:bblocks:{name}", bx, top, bw, cell_h + 0.16, color, line=line)
            weights = [3 if len(c) > 2 else 1 for c in cells]
            unit = (bw - 0.12) / sum(weights)
            cx = bx + 0.06
            for k, (c, wt) in enumerate(zip(cells, weights, strict=True)):
                self.rect(slide, f"diagram:bblocks:{name}{k}", cx, top + 0.08, unit * wt - 0.06, cell_h, "background", c, 11, line="line")
                cx += unit * wt
            self.label(slide, f"diagram:bblocks:{name}cap", caption, bx, top + cell_h + 0.2, bw, 0.32, 11.5, align=PP_ALIGN.LEFT)
            bx += w * ratio

    def diagram_pos4(self, slide: Any, x: float, y: float, w: float, h: float, *, sentence: str) -> None:
        """品詞の 4 階層：例文を本研究の形態素解析（src/morphology.py の共有 Tagger，fugashi + unidic-lite）で解析した
        各形態素の品詞 4 階層（列：形態素，行：表層形・大分類～細分類）．t_i に使う上位 2 段（大分類・中分類）を太枠で強調．"""
        table = pos4_table(sentence)
        cols = len(table)
        label_w = 0.75
        cw = (w - label_w) / cols
        rows = ["表層形", *POS4_LEVELS]
        rh = h / len(rows)
        for r, head in enumerate(rows):
            self.label(slide, f"diagram:pos4:head{r}", head, x, y + r * rh, label_w, rh, 11, bold=True, align=PP_ALIGN.LEFT)
        for c, (surface, levels) in enumerate(table):
            for r, text in enumerate([surface, *levels]):
                self.rect(slide, f"diagram:pos4:c{c}_{r}", x + label_w + c * cw + 0.03, y + r * rh + 0.03, cw - 0.06, rh - 0.06,
                          "light" if r == 0 else "background", text, 11, line="line", bold=r == 0)
        frame = self.rect(slide, "diagram:pos4:frame", x + label_w - 0.02, y + rh - 0.01, w - label_w + 0.04, 2 * rh + 0.02, "background")
        frame.fill.background()
        frame.line.color.rgb = self.style.c["accent_dark"]
        frame.line.width = Pt(2.5)

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
        if sd["layout"] == "cover":
            self.cover(slide, sd, content_w)
        else:
            self.page(slide, sd, sid, content_w)

        # 出典（スライド下部の薄い文字）とページ番号
        notes = sd.get("footnotes", [])
        if notes:
            line = "　".join(f"[{i}] {s}" for i, s in enumerate(notes, 1))
            if self.n_lines(line, 10, content_w - 1.2) > 2:
                raise ValueError(f"スライド {sid}：出典が 2 行を超える")
            self.label(slide, "footnote", line, MARGIN_X, FOOTNOTE_Y, content_w - 1.0, FOOTNOTE_H, 10, color="muted", align=PP_ALIGN.LEFT)
        page = f"{sid} / {total_main}" if sd["kind"] == "main" else sid
        self.label(slide, "pageno", page, self.slide_w - MARGIN_X - 1.0, FOOTNOTE_Y, 1.0, FOOTNOTE_H, 10, color="muted", align=PP_ALIGN.RIGHT)

        # 発表者ノート
        n = sd["notes"]
        lines = [f"想定時間：{n['time']}"]
        if sd["kind"] == "supp":
            # 044：質問の番号（qa_defense_matrix.md）と対応する本編のスライドは発表者ノートにのみ書く
            lines += [f"想定質問（qa_defense_matrix.md）：{sd['qa']}", f"対応する本編：Slide {sd['supplements']}"]
        lines += ["話すこと："] + [f"・{p}" for p in n["points"]]
        if n.get("runs"):
            lines.append(f"数値の出典（run 名）：{'，'.join(n['runs'])}")
        slide.notes_slide.notes_text_frame.text = "\n".join(lines)

    def cover(self, slide: Any, sd: dict[str, Any], content_w: float) -> None:
        """タイトルスライド：題名，班の氏名欄（プレースホルダ）．045 で副題（科目名）を削除（subtitle があれば題名の下に置く）．"""
        n_title = self.n_lines(sd["title"], COVER_TITLE_PT, content_w - 0.4)
        if n_title > 1:
            raise ValueError(f"スライド {sd['id']}：題名が 1 行に収まらない（{n_title} 行）")
        self.label(slide, "title", sd["title"], MARGIN_X, 1.75, content_w, 0.95, COVER_TITLE_PT, bold=True)
        self.rect(slide, "title_bar", self.slide_w / 2 - 2.0, 2.82, 4.0, 0.07, "accent")
        if sd.get("subtitle"):
            self.label(slide, "subtitle", sd["subtitle"], MARGIN_X, 3.05, content_w, 0.5, 20, color="muted")
        members = sd["members"]
        row_h, col_w = 0.46, 2.3
        top = 3.95
        for i, (role, name) in enumerate(members):
            yy = top + i * row_h
            self.label(slide, f"member:{i}", role, self.slide_w / 2 - col_w - 0.1, yy, col_w, row_h, 17, bold=True, align=PP_ALIGN.RIGHT)
            self.label(slide, f"member_name:{i}", name, self.slide_w / 2 + 0.1, yy, col_w + 1.0, row_h, 17, color="muted",
                       align=PP_ALIGN.LEFT)

    def page(self, slide: Any, sd: dict[str, Any], sid: str, content_w: float) -> None:
        """通常のスライド：区分の表示・題名・（補助スライドは質問の文）・主張の帯・本文．"""
        # 区分の表示・題名・主張
        # 補助スライドは「質問 → 答え（主張の帯）→ 根拠（本文）」の形（042）．044：題名の直下に質問の文だけを置き，
        # 「想定質問：」「答え：」の文字列と質問の番号はスライドに出さない（番号と本編との対応は発表者ノートのみ）
        supp = sd["kind"] == "supp"
        self.label(slide, "section", sd["section"], MARGIN_X, SECTION_Y, 4.0, SECTION_H, 11, color="muted", align=PP_ALIGN.LEFT)
        if supp:
            if not sd.get("question"):
                raise ValueError(f"スライド {sid}：補助スライドに想定質問（question）がない")
            if self.n_lines(sd["question"], 14, content_w - 0.3) > 1:
                raise ValueError(f"スライド {sid}：質問の文が 1 行に収まらない")
            self.label(slide, "question", sd["question"], MARGIN_X + 0.2, SUPP_QUESTION_Y, content_w - 0.2, SUPP_QUESTION_H, 14,
                       color="strong", bold=True, align=PP_ALIGN.LEFT)
        bar = self.rect(slide, "title_bar", MARGIN_X, TITLE_Y + 0.1, 0.09, TITLE_H - 0.2, "accent")
        bar.name = "title_bar"
        title_size = 26 if sd["layout"] == "title" else 24
        n_title = self.n_lines(sd["title"], title_size, content_w - 0.3)
        if n_title > 1:
            raise ValueError(f"スライド {sid}：題名が 1 行に収まらない（{n_title} 行）")
        self.label(slide, "title", sd["title"], MARGIN_X + 0.2, TITLE_Y, content_w - 0.2, TITLE_H, title_size, bold=True, align=PP_ALIGN.LEFT)
        # 主張は 1 行（17 pt で収まらなければ 15 pt まで下げる）．参考文献のスライドは主張の帯を置かない（044：文献の一覧だけ）
        if "claim" not in sd:
            if sd.get("role") != "references":
                raise ValueError(f"スライド {sid}：主張（claim）がない")
        else:
            self.claim_band(slide, sid, sd["claim"], content_w, *((SUPP_CLAIM_Y, SUPP_CLAIM_H) if supp else (CLAIM_Y, CLAIM_H)))
        # 本文（主張の帯のない参考文献のスライドは帯の位置から始める，044）
        body_y = BODY_Y if "claim" in sd else CLAIM_Y
        if sd["layout"] == "single":
            self.draw_stack(slide, sid, sd["body"], MARGIN_X, body_y, content_w, BODY_BOTTOM)
        elif sd["layout"] in ("split", "title"):
            wl = (content_w - COLUMN_GAP) * sd.get("ratio", 0.5)
            self.draw_stack(slide, f"{sid}L", sd["left"], MARGIN_X, body_y, wl, BODY_BOTTOM)
            self.draw_stack(slide, f"{sid}R", sd["right"], MARGIN_X + wl + COLUMN_GAP, body_y, content_w - COLUMN_GAP - wl, BODY_BOTTOM)
        else:
            raise ValueError(f"スライド {sid}：未知の layout {sd['layout']}")

    def claim_band(self, slide: Any, sid: str, claim_text: str, content_w: float, y: float, h: float) -> None:
        """主張の帯（1 行．17 pt で収まらなければ 15 pt まで下げる）．"""
        claim_size = next((z for z in (17, 16, 15) if self.n_lines(claim_text, z, content_w - 0.3) == 1), None)
        if claim_size is None:
            raise ValueError(f"スライド {sid}：主張が 15 pt でも 1 行に収まらない")
        claim = self.rect(slide, "claim", MARGIN_X, y, content_w, h, "highlight")
        claim.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        claim.text_frame.margin_left = Inches(0.18)
        info = self.style.add_runs(claim.text_frame.paragraphs[0], claim_text, claim_size, bold=True)
        self.place_math(slide, "claim", [info], (MARGIN_X + 0.18) * 72, (content_w - 0.22) * 72, (y + 0.02) * 72, middle_h=(h - 0.04) * 72)

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


COVER_TITLE_PT = 30

DIAGRAM_REFS = {"concept": "[^1]", "split": "", "a_blocks": "", "svd": "", "flow": "", "balance": "", "b_blocks": "", "pos4": ""}
# 図形で描く図の中の数式（画像の代替テキストと照合）
DIAGRAM_LATEX = {"concept": ["q"], "split": [], "a_blocks": ["\\vec{x}"],
                 "svd": ["=", "X", "N \\times M", "U_K", "N \\times K", "\\Sigma_K", "K \\times K", "V_K^\\top", "K \\times M"],
                 "flow": [], "balance": ["n_{\\min}", "n_{\\min}"], "b_blocks": [], "pos4": []}
DIAGRAM_H = {"concept": 3.0, "split": 4.6, "a_blocks": 1.45, "svd": 1.9, "flow": 2.45, "balance": 2.1, "b_blocks": 1.45, "pos4": 2.3}
POS4_LEVELS = ("大分類", "中分類", "小分類", "細分類")
POS4_EMPTY = "－"  # UniDic の空の階層（*）の表示


DEFLIST_WIDTHS = [0.3, 0.7]


def deflist_border(cell: Any, top: str | None, bottom: str | None) -> None:
    """定義の一覧のセルの罫線：左・右はなし，行の区切りは薄い線 1 本（最初の行の上と最終行の下はなし）．
    区切りの線は上の行の下辺と下の行の上辺の両方に同じ線を指定する（2 列にまたがる行の下辺が隣の行の上辺の指定で消えるのを防ぐ）．"""
    tcpr = cell._tc.get_or_add_tcPr()
    for tag, color in (("a:lnL", None), ("a:lnR", None), ("a:lnT", top), ("a:lnB", bottom)):
        ln = etree.SubElement(tcpr, qn(tag), w="6350", cap="flat", cmpd="sng", algn="ctr")
        if color is not None:
            sf = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(sf, qn("a:srgbClr"), val=color)
        else:
            etree.SubElement(ln, qn("a:noFill"))
    fills = [e for e in tcpr if e.tag in (qn("a:solidFill"), qn("a:noFill"))]
    [tcpr.remove(e) or tcpr.append(e) for e in fills]


def pos4_table(sentence: str) -> list[tuple[str, list[str]]]:
    """例文の各形態素の（表層形，品詞 4 階層）．本研究の形態素解析と同じ共有 Tagger（src/morphology.py）で解析．
    空の階層（UniDic の *）は POS4_EMPTY で表示．check_slides.py が pptx の図の文字列と照合．"""
    from src.morphology import shared_tagger

    nodes = list(shared_tagger()(sentence))
    if not nodes:
        raise ValueError(f"例文の形態素解析の結果が空: {sentence!r}")
    show = lambda v: POS4_EMPTY if v in (None, "", "*") else v  # noqa: E731
    return [(n.surface, [show(n.feature.pos1), show(n.feature.pos2), show(n.feature.pos3), show(n.feature.pos4)]) for n in nodes]


def is_numeric_cell(s: str) -> bool:
    return bool(re.fullmatch(r"[+−\-]?[\d.,]+( ± [\d.]+)?|－", s.strip()))


def is_complex_math(tex: str) -> bool:
    """段落の途中に置くと読みにくい複雑な式：分数・総和等，または添字・上付き・ノルムを含む等式・関数の形・割り算．
    1 文字の変数，添字・上付き付きの記号（$U_K$，$\\Sigma_K^{-1}$ 等），ノルム（$\\lVert \\hat{c}_a \\rVert$），
    数値と演算子の並び（$L = 500$ 等）は複雑でない．"""
    return bool(TALL_MATH.search(tex) or (re.search(r"[=(/]", tex) and re.search(r"[_^]|\\lVert", tex)))


def diagram_blocks(sd: dict[str, Any]) -> list[dict[str, Any]]:
    """スライドの図形の図のブロック（columns の中を含む）．"""
    def walk(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [x for b in blocks for x in ([b] if "columns" not in b else walk(b["columns"]["left"]) + walk(b["columns"]["right"]))]
    return [b for k in ("body", "left", "right") for b in walk(sd.get(k, [])) if "diagram" in b]


def check_refs(sd: dict[str, Any]) -> None:
    """本文の [^n] と footnotes の対応を確認（番号の欠け・未使用の出典は停止）．"""
    texts: list[str] = [sd["title"], sd.get("claim", "")]

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


def yaml_texts(spec: dict[str, Any]) -> list[str]:
    """スライドに描く内容データの文字列（ノート・照合の指定・出典の範囲・図の参照を除く）．"""
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            [walk(v) for k, v in o.items() if k not in ("figure", "figures", "diagram", "refs", "widths", "size", "h", "cols", "ratio")]
        elif isinstance(o, list):
            [walk(v) for v in o]

    [walk(spec.get(k)) for k in ("title", "subtitle", "members", "claim", "question", "body", "left", "right", "footnotes")]
    return out


def build(content_path: Path, out_path: Path, math_overrides: dict[str, Any]) -> Path:
    content = yaml.safe_load(content_path.read_text(encoding="utf-8"))
    cfg = {**content["theme"]["math"], **{k: v for k, v in math_overrides.items() if v is not None}}
    b = Builder(content, TexRenderer(cfg))
    slides = content["slides"]
    total_main = sum(s["kind"] == "main" for s in slides)
    [b.add_slide(sd, i, total_main) for i, sd in enumerate(slides, 1)]
    b.clear_metadata()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    b.prs.save(str(out_path))
    per_slide = {sd["id"]: sum(sh.name.startswith("math:") for sh in sl.shapes) for sd, sl in zip(slides, b.prs.slides, strict=True)}
    latex_all = [mt.group("math") for sd in slides for t in yaml_texts(sd) for mt in MARKUP.finditer(t) if mt.group("math")]
    latex_all += [t for sd in slides for blk in diagram_blocks(sd) for t in DIAGRAM_LATEX[blk["diagram"]]]
    n_img = sum(per_slide.values())
    print(f"[math] 数式 {len(latex_all)} 個：画像 {n_img} 個．"
          f"画像の数式を含むスライド {sum(v > 0 for v in per_slide.values())} 枚：" + "，".join(f"{k}（{v}）" for k, v in per_slide.items() if v))
    complex_inline = [(bid, tex) for bid, tex, mid in b.math_log if mid and is_complex_math(tex)]
    print(f"[math] 段落の途中の複雑な数式 {len(complex_inline)} 個："
          + "；".join(f"{bid} {tex}" for bid, tex in complex_inline))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--content", type=Path, default=PRESENTATION_DIR / "slides_content.yaml")
    parser.add_argument("--out", type=Path, default=PRESENTATION_DIR / "slides.pptx")
    parser.add_argument("--lualatex", default=None, help="lualatex のコマンド（既定：theme.math.lualatex）")
    parser.add_argument("--pdftocairo", default=None, help="pdftocairo のコマンド（既定：theme.math.pdftocairo）")
    parser.add_argument("--math-cache", default=None, help="数式の PNG のキャッシュ（既定：theme.math.cache_dir）")
    args = parser.parse_args()
    out = build(args.content, args.out, {"lualatex": args.lualatex, "pdftocairo": args.pdftocairo, "cache_dir": args.math_cache})
    print(f"[output] {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")


if __name__ == "__main__":
    main()
