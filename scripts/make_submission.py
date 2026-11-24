"""提出用のプログラム一式の zip を Git の追跡ファイル（コミット済みの内容）から作成．

使い方：
    uv run python scripts/make_submission.py [--ref HEAD] [--out-dir outputs/submission] [--top-dir text-author-identifier-nlps]

処理：
    作業ツリーに未コミットの変更・未追跡のファイル（.gitignore で除外されたものを除く）があれば停止．
    git archive で --ref の内容のうち INCLUDE_PATHS だけを取り出し，EXCLUDE_FILES を除いて，
    提出用の README（submission/README.md）を zip の直下の README.md に置き換えて詰め直す．
    詰め直した zip の全ファイルに次の検査を行い，1 件でも見つかれば zip を残さず停止：
        入れない範囲のパス（data/ ・ plans/ ・ presentation/ ・ references/ ・ outputs/ ・エージェント用のファイル等）
        キャッシュ・環境のファイル（__pycache__ ・ *.pyc ・ .venv ・ .DS_Store 等）と実験の生成物（run_meta.json 等）
        秘匿情報（メールアドレス・ローカルの絶対パス・学籍番号の形・ git のユーザー名やログイン名の一部）．
        見つけた文字列は出力せず，種類とファイル・行番号のみ出力．
    併せて，src/ ・ scripts/ ・ configs/ のコメント・docstring にある開発時の作業記録の番号（3 桁）の件数を集計（情報のみ）．

出力（--out-dir，既定 outputs/submission/．同名の zip があれば停止）：
    <top-dir>_<commit 先頭 10 桁>.zip    提出用の zip（展開すると <top-dir>/ 以下に展開）
    <top-dir>_<commit 先頭 10 桁>.json   commit・ファイル一覧・サイズ・検査結果の件数（zip には入れない）
"""

from __future__ import annotations

import argparse
import ast
import getpass
import io
import json
import re
import subprocess
import tokenize
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# zip に入れる範囲（Git で追跡されているもののみ）．.gitignore は展開先を git 管理下に置く手順で
# コーパス・ .venv ・ outputs/ を追跡しないために入れる
INCLUDE_PATHS: tuple[str, ...] = ("src", "scripts", "configs", "pyproject.toml", "uv.lock", ".python-version", ".gitignore")
# 範囲内でも入れないファイル（発表スライドの生成・検査・発表用の図は presentation/ の内容データが必要，
# verify_pos_function.py は開発時の Git の履歴の版と比べる，本スクリプトは提出物の作成用）
EXCLUDE_FILES: frozenset[str] = frozenset({
    "scripts/build_slides.py", "scripts/check_slides.py", "scripts/plot_slide_figures.py",
    "scripts/verify_pos_function.py", "scripts/make_submission.py",
})
# 提出用の README の生成元と zip 内のパス
README_SRC, README_DST = "submission/README.md", "README.md"

# 入れない範囲（zip 内の相対パスの接頭辞・ファイル名）
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "data/", "plans/", "presentation/", "references/", "outputs/", ".claude/", ".agents/", "submission/", ".git/",
)
FORBIDDEN_NAME = re.compile(r"^(CLAUDE\.md|AGENTS\.md|llm.*\.(md|txt)|run_meta\.json|results\.json|summary\.csv|folds\.csv)$")
CACHE_PATTERN = re.compile(r"(^|/)(__pycache__|\.venv|\.ipynb_checkpoints|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)"
                           r"|\.py[cod]$|(^|/)\.DS_Store$|\.egg-info(/|$)|(^|/)Thumbs\.db$")

# 秘匿情報の形（check_slides.py の定義を，ソースコードと uv.lock での誤検出を避けるよう絞ったもの）
#   ローカルパス：Windows のドライブ名は直後に 2 字以上のパス名が続くものに限る（Python の "W:\n" 等の文字列を除く）
#   学籍番号の形：前後が英数字でないものに限る（uv.lock のハッシュ値の 16 進の一部を除く）
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
LOCAL_PATH = re.compile(r"/Users/|/home/|/private/(?:var|tmp)/|/Volumes/|(?<![A-Za-z])[A-Za-z]:\\[\w .-]{2,}")
STUDENT_ID = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2}\d{5}(?![A-Za-z0-9])")
# git のユーザー名・リモート URL の部分のうち，個人を特定しない一般語
GENERIC_TERMS: frozenset[str] = frozenset({"https", "http", "github", "gitlab", "git", "com", "nlps", "text", "author", "identifier"})
MIN_TERM_LEN = 4

# 開発時の作業記録の番号（plans/NNN，Ticket NNN，000～049 の 3 桁）．小数・英数字列の一部は除く
TICKET_REF = re.compile(r"plans/\d{3}|(?i:ticket)\s*\d{1,3}|(?<![\w.])0[0-4]\d(?![\w])")
TICKET_SCAN_DIRS: tuple[str, ...] = ("src/", "scripts/", "configs/")


@dataclass(frozen=True)
class Hit:
    kind: str
    path: str
    line: int


def git(*args: str, binary: bool = False) -> str | bytes:
    out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
    return out if binary else out.decode("utf-8")


def git_optional(*args: str) -> str:
    """値がないとき（未設定の config 等）は空文字．"""
    r = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def secret_terms() -> list[str]:
    """照合する文字列（出力しない）：git の user.name ・ user.email ・リモート URL ・ OS のログイン名とホームのディレクトリ名の部分．"""
    raw = [git_optional("config", "user.name"), git_optional("config", "user.email"),
           *git_optional("remote", "-v").split(), getpass.getuser(), Path.home().name]
    parts = {p.lower() for s in raw for p in re.split(r"[^0-9A-Za-z]+", s) if len(p) >= MIN_TERM_LEN}
    return sorted(parts - GENERIC_TERMS)


def archive_entries(ref: str) -> dict[str, tuple[zipfile.ZipInfo, bytes]]:
    """git archive の zip から，提出物の相対パス → (ZipInfo, 内容) を作る（ディレクトリ項目は除く）．"""
    blob = git("archive", "--format=zip", ref, "--", *INCLUDE_PATHS, README_SRC, binary=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        files = {i.filename: (i, z.read(i)) for i in z.infolist() if not i.is_dir()}
    missing = [p for p in (README_SRC, *INCLUDE_PATHS) if p not in files and not any(f.startswith(f"{p}/") for f in files)]
    if missing:
        raise SystemExit(f"[error] {ref} に必要なファイルがない: {missing}")
    rename = lambda p: README_DST if p == README_SRC else p  # noqa: E731
    return {rename(p): v for p, v in files.items() if p not in EXCLUDE_FILES}


def check_paths(paths: Iterable[str]) -> list[Hit]:
    return [Hit(kind, p, 0) for p in paths for kind, bad in (
        ("入れない範囲のパス", p.startswith(FORBIDDEN_PREFIXES) or FORBIDDEN_NAME.match(Path(p).name) is not None),
        ("キャッシュ・環境のファイル", CACHE_PATTERN.search(p) is not None),
    ) if bad]


def check_secrets(files: dict[str, bytes], terms: list[str]) -> list[Hit]:
    """1 行ずつ照合し，種類・パス・行番号のみ返す（文字列は返さない）．UTF-8 で読めないファイルは停止．"""
    def scan(path: str, text: str) -> Iterable[Hit]:
        for n, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            yield from (Hit(kind, path, n) for kind, found in (
                ("メールアドレス", EMAIL.search(line) is not None),
                ("ローカルの絶対パス", LOCAL_PATH.search(line) is not None),
                ("学籍番号の形", STUDENT_ID.search(line) is not None),
                ("git のユーザー名・ログイン名の一部", any(t in low for t in terms)),
            ) if found)

    return [h for p, data in files.items() for h in scan(p, data.decode("utf-8"))]


def comment_lines(path: str, text: str) -> set[int]:
    """コメント・docstring の行番号（.py は tokenize と ast，.yaml は # 以降を含む行）．"""
    if path.endswith((".yaml", ".yml")):
        return {n for n, line in enumerate(text.splitlines(), 1) if "#" in line}
    if not path.endswith(".py"):
        return set()
    comments = {t.start[0] for t in tokenize.generate_tokens(io.StringIO(text).readline) if t.type == tokenize.COMMENT}
    tree = ast.parse(text)
    doc_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)]
    docs = {ln for n in doc_nodes if n.body and isinstance(e := n.body[0], ast.Expr)
            and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str)
            for ln in range(e.lineno, (e.end_lineno or e.lineno) + 1)}
    return comments | docs


def ticket_refs(files: dict[str, bytes]) -> dict[str, dict[str, int]]:
    """src/ ・ scripts/ ・ configs/ の作業記録の番号の件数（ファイルごと，コメント・docstring とそれ以外の文字列）．"""
    def count(path: str, text: str) -> dict[str, int]:
        in_comment = comment_lines(path, text)
        c = Counter("comment_docstring" if n in in_comment else "other"
                    for n, line in enumerate(text.splitlines(), 1) for _ in TICKET_REF.finditer(line))
        return dict(c)

    counts = {p: count(p, d.decode("utf-8")) for p, d in sorted(files.items()) if p.startswith(TICKET_SCAN_DIRS)}
    return {p: c for p, c in counts.items() if c}


def write_zip(path: Path, top_dir: str, entries: dict[str, tuple[zipfile.ZipInfo, bytes]]) -> None:
    """git archive の更新時刻（commit 時刻）と属性を保ったまま <top_dir>/ 以下に詰め直す．"""
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for rel, (src, data) in sorted(entries.items()):
            info = zipfile.ZipInfo(f"{top_dir}/{rel}", date_time=src.date_time)
            info.external_attr, info.compress_type = src.external_attr, zipfile.ZIP_DEFLATED
            z.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="HEAD", help="zip にする git のリビジョン（既定：HEAD）")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "submission")
    parser.add_argument("--top-dir", default="text-author-identifier-nlps", help="zip 内の最上位ディレクトリ名（個人名を含めない）")
    args = parser.parse_args()

    dirty = [line for line in git("status", "--porcelain", "--untracked-files=all").splitlines() if line.strip()]
    if dirty:
        print(f"[warn] 未コミットの変更・未追跡のファイルが {len(dirty)} 件あり．コミット後に再実行")
        print("\n".join(f"  {line}" for line in dirty))
        raise SystemExit(1)

    commit = git("rev-parse", "--verify", f"{args.ref}^{{commit}}").strip()
    entries = archive_entries(commit)
    files = {p: data for p, (_, data) in entries.items()}

    terms = secret_terms()
    hits = check_paths(files) + check_secrets(files, terms)
    print(f"[check] 照合した git・ OS のユーザー名等の文字列：{len(terms)} 種（文字列は出力しない）")
    kinds = ("入れない範囲のパス", "キャッシュ・環境のファイル", "メールアドレス", "ローカルの絶対パス", "学籍番号の形",
             "git のユーザー名・ログイン名の一部")
    by_kind = Counter(h.kind for h in hits)
    print("\n".join(f"[check] {k}：{by_kind[k]} 件" for k in kinds))
    if hits:
        print("\n".join(f"  {h.kind}：{h.path}" + (f":{h.line}" if h.line else "") for h in hits))
        raise SystemExit("[error] 検査で見つかった箇所あり．zip は作成しない")

    refs = ticket_refs(files)
    total_refs = Counter({k: sum(c.get(k, 0) for c in refs.values()) for k in ("comment_docstring", "other")})
    print(f"[info] 作業記録の番号：コメント・docstring {total_refs['comment_docstring']} 件，その他の文字列 "
          f"{total_refs['other']} 件（{len(refs)} ファイル）")
    print("\n".join(f"  {p}：{c.get('comment_docstring', 0)} / {c.get('other', 0)}" for p, c in refs.items()))

    stem = f"{args.top_dir}_{commit[:10]}"
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path, meta_path = out_dir / f"{stem}.zip", out_dir / f"{stem}.json"
    if zip_path.exists() or meta_path.exists():
        raise SystemExit(f"[error] 既に存在: {zip_path.name}（同じ commit の zip は作成済み）")
    write_zip(zip_path, args.top_dir, entries)

    sizes = {p: len(d) for p, d in sorted(files.items())}
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit": commit,
        "top_dir": args.top_dir,
        "num_files": len(sizes),
        "total_bytes": sum(sizes.values()),
        "zip_bytes": zip_path.stat().st_size,
        "files": sizes,
        "checks": {k: by_kind[k] for k in kinds} | {"secret_terms": len(terms)},
        "ticket_refs": refs,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(f"  {p}  {n} B" for p, n in sizes.items()))
    print(f"[done] {zip_path.name}：{len(sizes)} ファイル，合計 {meta['total_bytes']} B（zip {meta['zip_bytes']} B）")


if __name__ == "__main__":
    main()
