"""`unidic-lite` の素性フィールド実測調査（Ticket 009，`presentation/note.md` の記述根拠）．

`fugashi`（`fugashi-plus[unidic-lite]`）で同梱辞書を読み込み，以下を標準出力に出す．
    - 辞書の版（`unidic_lite` 同梱の `version` ファイル）とパッケージ版
    - 1 形態素あたりの素性フィールド数とフィールド名一覧（既知語・未知語の別）
    - 語種 `goshu` フィールドの有無と値の例
    - 表記揺れ語の `lemma`（文脈つきで解析し，対象語に一致する形態素を抽出）
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

import fugashi
import unidic_lite

# 既定の調査対象語と，副詞としての用法を引き出す文脈（単独で解析すると「大分」が地名と解析されるため）
DEFAULT_WORDS: tuple[str, ...] = ("大分", "だいぶ", "矢張り", "やはり", "やっぱり")
DEFAULT_TEMPLATE = "今日は{word}寒くなった。"
# 未知語のフィールド数確認用（辞書に存在しない文字列）
DEFAULT_UNKNOWN = "qwzxv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--words", nargs="+", default=list(DEFAULT_WORDS), help="lemma を確認する語")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="語を埋め込む文脈（{word} を含むこと）")
    parser.add_argument("--unknown", default=DEFAULT_UNKNOWN, help="未知語判定の確認に使う文字列")
    args = parser.parse_args()
    if "{word}" not in args.template:
        raise ValueError(f"--template must contain '{{word}}': {args.template!r}")
    return args


@dataclass(frozen=True)
class Morph:
    """1 形態素の素性スナップショット（fugashi のノードは次の解析呼び出しで再利用されるため即時に複製）．"""

    surface: str
    is_unk: bool
    field_names: tuple[str, ...]
    features: tuple[str | None, ...]
    raw_field_count: int  # feature_raw を CSV（引用符つきフィールド対応）として分割した列数

    def get(self, name: str) -> str | None:
        return self.features[self.field_names.index(name)]


def analyze(tagger: fugashi.Tagger, text: str) -> list[Morph]:
    def snapshot(node: fugashi.UnidicNode) -> Morph:
        names = getattr(node.feature, "_fields", None)
        if names is None:
            raise TypeError(f"feature is not a namedtuple: {type(node.feature)!r}")
        return Morph(
            surface=node.surface,
            is_unk=bool(node.is_unk),
            field_names=tuple(names),
            features=tuple(node.feature),
            raw_field_count=len(next(csv.reader([node.feature_raw]))),
        )

    return [snapshot(n) for n in tagger(text)]


def describe(m: Morph) -> str:
    return (
        f"surface={m.surface} lemma={m.get('lemma')} lForm={m.get('lForm')} orthBase={m.get('orthBase')} "
        f"pos={m.get('pos1')}-{m.get('pos2')} goshu={m.get('goshu')}"
    )


def main() -> None:
    args = parse_args()
    dicdir = Path(unidic_lite.DICDIR)
    tagger = fugashi.Tagger()
    sentences = [args.template.format(word=w) for w in args.words]
    parsed = [analyze(tagger, s) for s in sentences]
    known = [m for morphs in parsed for m in morphs if not m.is_unk]
    unknown = [m for m in analyze(tagger, args.unknown) if m.is_unk]

    print("== 辞書・パッケージ ==")
    print(f"unidic-lite dictionary version : {(dicdir / 'version').read_text(encoding='utf-8').strip()}")
    print(f"unidic-lite package            : {importlib.metadata.version('unidic-lite')}")
    print(f"fugashi-plus package           : {importlib.metadata.version('fugashi-plus')}")

    # 既知語のフィールド名集合が単一であることを確認
    schemas = {m.field_names for m in known}
    assert len(schemas) == 1, f"known-word schemas are not unique: {schemas}"
    (schema,) = schemas
    print("\n== 素性フィールド（既知語） ==")
    print(f"field count (namedtuple)       : {len(schema)}")
    print(f"field count (raw CSV)          : {sorted({m.raw_field_count for m in known})}")
    print("field names                    : " + ", ".join(f"[{i}] {name}" for i, name in enumerate(schema)))

    if not unknown:
        raise RuntimeError(f"--unknown {args.unknown!r} produced no unknown-word token")
    print("\n== 素性フィールド（未知語） ==")
    print(f"unknown tokens                 : {[m.surface for m in unknown]}")
    print(f"field count (raw CSV)          : {sorted({m.raw_field_count for m in unknown})}")
    print(f"non-None fields                : {[n for m in unknown[:1] for n, v in zip(m.field_names, m.features) if v is not None]}")
    print(f"lemma                          : {[m.get('lemma') for m in unknown]}")

    print("\n== 語種 goshu ==")
    print(f"'goshu' in fields              : {'goshu' in schema}")
    print("goshu examples                 : " + ", ".join(f"{m.surface}={m.get('goshu')}" for m in parsed[0]))

    print("\n== 表記揺れ語の lemma ==")
    print(f"context template               : {args.template}")
    for word, morphs in zip(args.words, parsed, strict=True):
        matched = [m for m in morphs if m.surface == word]
        if not matched:
            raise RuntimeError(f"'{word}' was not segmented as a single morpheme: {' / '.join(m.surface for m in morphs)}")
        print("  " + " | ".join(describe(m) for m in matched))

    print("\n== 参考：文脈なしで単独解析 ==")
    print("\n".join(f"  {w} -> " + " / ".join(describe(m) for m in analyze(tagger, w)) for w in args.words))


if __name__ == "__main__":
    main()
