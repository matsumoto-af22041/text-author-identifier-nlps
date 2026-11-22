"""形態素解析とテキスト単位のプロセス内キャッシュ（B・C・D 共通，Ticket 26）．

UniDic（ `fugashi-plus[unidic-lite]` ）の解析結果を，1 形態素ごとに表層形・語彙素・品詞の上位 2 階層
（第 2 回 p.16, p.36）へ即時にコピーして返す．fugashi のノードは次の解析で再利用されるため，ノードを
保持してはならない．

形態素解析を使う特徴量・スクリプトは，新規のものも含め `fugashi.Tagger` を直接作らず本モジュールを使う．
    morphemes_cached(text)：評価パイプライン用．同じチャンクを fold・シード・特徴量・アブレーション条件の
                            間で再解析しない（同じ辞書・同じテキストなら解析結果は決定的なため数値は不変）．
    morphemes_of(shared_tagger(), text)：作品全文など，評価で再利用しない一回限りのテキスト用（キャッシュしない）．

リーク防止：キャッシュが保持するのはテキスト単体の解析結果（入力テキストのみの関数）だけで，語彙・df・IDF・
SVD 射影行列など Train の集合に依存する量は保持しない．それらは各抽出器の `fit` で Train のみから毎回算出．
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import fugashi

TokenKey = Literal["surface", "lemma"]

# 型注釈用の別名（抽出器側で fugashi を import せずに済ませる）
Tagger = fugashi.Tagger


@dataclass(frozen=True, slots=True)
class Morpheme:
    """1 形態素の確定済み情報．lemma は UniDic の語彙素で，None の未知語は表層形で代替（第 2 回 p.16 の原形に相当）．"""

    surface: str
    lemma: str
    pos1: str
    pos2: str

    def key(self, token_key: TokenKey) -> str:
        return self.surface if token_key == "surface" else self.lemma


def morphemes_of(tagger: Tagger, text: str) -> tuple[Morpheme, ...]:
    """形態素解析（キャッシュなし）．"""
    return tuple(
        Morpheme(
            surface=node.surface,
            lemma=node.feature.lemma or node.surface,
            pos1=node.feature.pos1,
            pos2=node.feature.pos2,
        )
        for node in tagger(text)
    )


@cache
def shared_tagger() -> Tagger:
    """プロセス内で共有する Tagger（辞書の読み込みは 1 回）．"""
    return fugashi.Tagger()


@cache
def morphemes_cached(text: str) -> tuple[Morpheme, ...]:
    """morphemes_of の結果をテキスト単位でキャッシュ．

    キャッシュはプロセス終了まで保持（ `morphemes_cached.cache_clear()` で解放）．
    """
    return morphemes_of(shared_tagger(), text)
