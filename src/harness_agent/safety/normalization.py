"""文本折叠与药名归一化（M2 硬规则基础）。

硬规则不走向量：所有匹配发生在"折叠空间"——
全角转半角 + 小写 + 去除全部空白，使 ``ＡＭＯＸＩＣＩＬＬＩＮ``、
``Amoxicillin``、``阿莫 西林`` 与 ``amoxicillin`` / ``阿莫西林``
折叠到同一匹配键上，对抗大小写/全角/空格混淆。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from harness_agent.safety.dictionary import DrugDictionary

__all__ = ["DrugMention", "DrugNormalizer", "fold_text"]

#: 全角字符 -> 半角字符映射（U+FF01..U+FF5E -> ASCII 0x21..0x7E）
#: 外加全角空格 U+3000 -> 半角空格（随后被整体去除）。
_FULLWIDTH_TRANSLATION = {0xFF01 + i: 0x21 + i for i in range(0x5E)}
_FULLWIDTH_TRANSLATION[0x3000] = 0x20


def fold_text(text: str) -> str:
    """文本折叠：全角转半角 + 小写 + 去除全部空白。"""
    half = text.translate(_FULLWIDTH_TRANSLATION)
    lowered = half.lower()
    return "".join(lowered.split())


class DrugMention(BaseModel):
    """文本中检出的一次药物提及。"""

    normalized_name: str
    #: 命中的折叠别名（折叠空间中的匹配键，非原文片段）
    matched_text: str


class DrugNormalizer:
    """药名归一化器：别名 / 商品名 / 中英文 -> 归一化标准药名。

    - ``normalize``：单个药名 -> 归一化名（未知返回 None）；
    - ``find_mentions``：自由文本扫描 -> 提及列表
      （最长优先、互不重叠；同一药物多个别名并提会产生多条提及，
      消费方按 ``normalized_name`` 集合去重）。
    """

    def __init__(self, dictionary: DrugDictionary) -> None:
        self._dictionary = dictionary
        self._alias_index = dictionary.alias_index
        self._match_keys = dictionary.match_keys

    def normalize(self, name: str) -> str | None:
        """归一化单个药名；词典未知返回 None。"""
        return self._alias_index.get(fold_text(name))

    def find_mentions(self, text: str) -> list[DrugMention]:
        """扫描文本中的全部药物提及（最长优先、互不重叠）。"""
        folded = fold_text(text)
        if not folded:
            return []
        occupied = bytearray(len(folded))
        mentions: list[DrugMention] = []
        for key, normalized_name in self._match_keys:
            start = folded.find(key)
            while start != -1:
                end = start + len(key)
                if not any(occupied[start:end]):
                    mentions.append(DrugMention(normalized_name=normalized_name, matched_text=key))
                    occupied[start:end] = b"\x01" * len(key)
                start = folded.find(key, end)
        return mentions
