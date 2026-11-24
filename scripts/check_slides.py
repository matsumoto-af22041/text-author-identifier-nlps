"""生成した発表スライド（presentation/slides.pptx）を検査し，違反があれば終了コード 1 で停止．

使い方：
    uv run python scripts/check_slides.py [--pptx presentation/slides.pptx] [--content presentation/slides_content.yaml]
                                          [--run-dir outputs/full_006] [--values-dir outputs/slide_figures] [--digest]

検査項目（基準）：
  構成：本編 21 枚（ページ番号 1 / 21 ～ 21 / 21）→ 出典一覧 1 枚 → 補助 S1～S14 の順．Slide 6～17 の題名が
        「【担当 X】定義 / 実験の概要 / 結果・考察」の順（A～D の 3 枚構成）．
  数値：本文・表・図形の文字・発表者ノートの数値トークン（小数，または 2 桁以上の整数．カンマ・符号は除いて比較）が，
        slides_content.yaml の num_sources で指定した storyline・qa・slides_d の範囲に現れること．
        ただし 0～29 の整数・100・500・1000・2000 と 4 桁の年は除外（L・枚数・番号）．出典行は下記の書き方の検査で扱う．
  追加した図：plot_slide_figures.py が書き出した描画値（<values-dir>/*.csv）が summary.csv の値と完全一致し，
        平均値を 3 桁（ΔS は 4 桁）に丸めた値が storyline・qa に現れること．スライドの画像は presentation/figures の現行ファイルとバイト一致．
  書き方：
    出典行：各項目に「第 N 回 p.」「構想発表資料 p.」「独自設計」のいずれかを含み，プログラム名・.py・run 名・
            チケット番号（3 桁の 0 始まりの数）・下線を含まない．
    本文（出典一覧のスライドを除く）：「第 N 回 p.」の直書き，.py，run 名，チケット番号，平文の添字（_ ^）を含まない．
    列挙：1 段落（表のセルの 1 行を含む）の読点「，」が 3 個以上，または 2 個で区切られた 3 片がすべて 8 字以下なら違反．
    段落の長さ：本文の 1 段落は 72 字以下，表のセルの 1 行は 50 字以下（マークアップを除いた文字数）．
  個人名・秘匿情報（見つけた文字列は出力せず，箇所のみ）：
    pptx のメタデータ（作成者・最終更新者・題名・会社名等）が空．
    pptx 内のすべての XML にメールアドレス・ローカルパス（/Users/，/home/，C:\\）・学籍番号の形（英字 2 字＋数字 5 字）がない．
    git の user.name・user.email・リポジトリのログイン名の各部分（4 字以上）と，元のスライド（presentation/orig.pptx，
    読み取りのみ）のメタデータの作成者・最終更新者の文字列がない．
  はみ出し：すべての図形がスライド内．テキスト枠は文字数から見積もった必要な高さが枠の高さ以下．
            本文の主要な図形（題名・主張・本文・表・図・出典）が互いに重ならない．
--digest：生成日時等のメタデータを除いた内容（テキスト・ノート・図の参照とバイト列のハッシュ）の SHA-256 を表示．
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.build_slides import LINE_SPACING, PARA_SPACE_PT, char_em, n_lines_em

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


# ---------------------------------------------------------------- pptx の読み取り


def iter_shapes(shapes: Any) -> Iterator[Any]:
    for s in shapes:
        yield s
        if s.shape_type == 6:
            yield from iter_shapes(s.shapes)


def shape_paragraphs(s: Any) -> list[tuple[str, str]]:
    """[(種類, 段落の文字列)]．種類は text / cell．"""
    out: list[tuple[str, str]] = []
    if s.has_text_frame:
        out += [("text", p.text) for p in s.text_frame.paragraphs if p.text.strip()]
    if getattr(s, "has_table", False) and s.has_table:
        out += [("cell", p.text) for r in s.table.rows for c in r.cells for p in c.text_frame.paragraphs if p.text.strip()]
    return out


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
        expect_pages = [f"{i} / 21" for i in range(1, 22)] + ["出典"] + [f"S{i}" for i in range(1, 15)]
        if pages != expect_pages:
            self.fail(f"構成：ページ番号の並びが 本編 21 → 出典 1 → 補助 S1～S14 でない: {pages}")
        if titles[21] != "出典一覧":
            self.fail("構成：22 枚目が出典一覧でない")
        for i, t in enumerate(titles[22:], 1):
            if not t.startswith(f"S{i}："):
                self.fail(f"構成：補助スライド {i} 枚目の題名が S{i}： で始まらない")
        parts = [f"【担当 {p}】{k}" for p in "ABCD" for k in ("定義", "実験の概要", "結果・考察")]
        if titles[5:17] != parts:
            self.fail(f"構成：Slide 6～17 が A～D の 定義 / 実験の概要 / 結果・考察 の順でない: {titles[5:17]}")
        for s, spec in zip(slides, specs, strict=False):
            if self.named_text(s, "title") != spec["title"]:
                self.fail(f"構成：題名が内容データと一致しない（{spec['id']}）")
        self.counts["slides"] = len(slides)
        self.counts["main"] = sum(p.endswith("/ 21") for p in pages)
        self.counts["supp"] = sum(p.startswith("S") for p in pages)

    @staticmethod
    def named_text(slide: Any, name: str) -> str:
        return next((s.text_frame.text for s in slide.shapes if s.name == name and s.has_text_frame), "")

    # ---------- 数値

    def check_numbers(self, docs: Docs) -> None:
        for slide, spec in zip(self.prs.slides, self.content["slides"], strict=True):
            allowed = set().union(*(num_tokens(docs.segment(src)) for src in spec["num_sources"]))
            body = [t for s in iter_shapes(slide.shapes) if s.name not in ("footnote", "pageno") for _, t in shape_paragraphs(s)]
            notes = slide.notes_slide.notes_text_frame.text
            for where, texts in (("本文", body), ("ノート", [notes])):
                for tok in sorted(set().union(*(num_tokens(t) for t in texts)) if texts else set()):
                    self.bump("numbers")
                    if tok not in allowed:
                        self.fail(f"数値：{spec['id']} の{where}の {tok} が資料の範囲 {spec['num_sources']} にない")

    # ---------- 追加した図

    def check_figures(self, docs: Docs) -> None:
        with (self.args.run_dir / "summary.csv").open(encoding="utf-8") as f:
            summary = {r["key"]: r for r in csv.DictReader(f)}
        all_docs = docs.text["storyline"] + docs.text["qa"] + docs.text["slides_d"]
        doc_tokens = num_tokens(all_docs)
        for name in ("a_ablation", "c_intra_inter", "classifier_comparison"):
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
                digits = 4 if r["metric"].startswith("delta_s") else 3
                rounded = f"{float(r['value']):.{digits}f}"
                # 基準値 0.25 のように資料が短い桁で書く値は末尾の 0 を除いた形も認める
                if rounded not in doc_tokens and rounded.rstrip("0") not in doc_tokens:
                    self.fail(f"図：{name} の {r['key']} {r['metric']} の丸め値 {float(r['value']):.{digits}f} が storyline・qa にない")
        for i, slide in enumerate(self.prs.slides, 1):
            for s in iter_shapes(slide.shapes):
                if s.name.startswith("figure:"):
                    self.bump("figure_refs")
                    f = PRES / s.name.split(":", 1)[1]
                    if not f.exists():
                        self.fail(f"図：スライド {i} の {f.name} が presentation にない")
                    elif hashlib.sha256(s.image.blob).hexdigest() != hashlib.sha256(f.read_bytes()).hexdigest():
                        self.fail(f"図：スライド {i} の {f.name} が presentation/figures の現行ファイルと異なる")

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
                    if spec["kind"] != "sources" and LECTURE_PAGE.search(p):
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
                if s.name.split(":")[0] in ("title", "claim", "body", "table", "figure", "footnote"):
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
            if not p.text:
                continue
            size = max((r.font.size.pt for r in p.runs if r.font.size), default=18)
            ppr = p._p.pPr
            mar = int(ppr.get("marL", 0)) / EMU_PER_IN if ppr is not None else 0.0
            # 下付き・上付き（baseline 属性）の文字は 0.7 倍の幅（build_slides.text_width_em と同じ見積もり）
            em = sum(sum(map(char_em, r.text)) * (0.7 if r._r.get_or_add_rPr().get("baseline") else 1.0) for r in p.runs)
            total += n_lines_em(em, size, w - mar) * size * LINE_SPACING / 72 + PARA_SPACE_PT / 72
        return total + ((tf.margin_top or 0) + (tf.margin_bottom or 0)) / EMU_PER_IN

    # ---------- 内容のハッシュ

    def digest(self) -> str:
        h = hashlib.sha256()
        for slide in self.prs.slides:
            for s in iter_shapes(slide.shapes):
                h.update(f"{s.name}|{s.left}|{s.top}|{s.width}|{s.height}|".encode())
                h.update("\n".join(t for _, t in shape_paragraphs(s)).encode())
                if s.name.startswith("figure:"):
                    h.update(hashlib.sha256(s.image.blob).digest())
            h.update(slide.notes_slide.notes_text_frame.text.encode())
        return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pptx", type=Path, default=PRES / "slides.pptx")
    parser.add_argument("--content", type=Path, default=PRES / "slides_content.yaml")
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "outputs" / "full_006")
    parser.add_argument("--values-dir", type=Path, default=PROJECT_ROOT / "outputs" / "slide_figures")
    parser.add_argument("--digest", action="store_true", help="内容のハッシュのみ表示して終了")
    args = parser.parse_args()
    c = Checker(args)
    if args.digest:
        print(c.digest())
        return
    docs = Docs()
    c.check_structure()
    c.check_numbers(docs)
    c.check_figures(docs)
    c.check_writing()
    c.check_privacy()
    c.check_layout()
    print("[counts] " + "，".join(f"{k} = {v}" for k, v in c.counts.items()))
    if c.violations:
        print(f"[NG] 違反 {len(c.violations)} 件")
        [print(f"  - {v}") for v in c.violations]
        sys.exit(1)
    print("[OK] 違反 0 件")


if __name__ == "__main__":
    main()
