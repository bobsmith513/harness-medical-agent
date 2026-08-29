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

#: 零宽与不可见字符。``str.split()`` 只去 Unicode 空白，这类字符会留在
#: 折叠结果里——``阿莫<ZWSP>西林`` 折叠后仍不是 ``阿莫西林``，``str.find``
#: 返回 -1，输入闸门 / 装配闸门 / 输出闸门三道同时漏检。
_INVISIBLE_CHARS = (
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u2060"  # WORD JOINER
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u00ad"  # SOFT HYPHEN
    "\u2028"  # LINE SEPARATOR
    "\u2029"  # PARAGRAPH SEPARATOR
)

#: 中缀分隔符：把药名从中间拆开是对抗样本的典型手法
#: （``阿莫-西林``、``阿莫·西林``、``AMOXI/CILLIN``）。
_INFIX_SEPARATORS = "-/\\*·・‧‥._~"

#: 折叠前统一删除的字符（``str.translate`` 映射为 None 即删除）。
#: 注意顺序：先转半角，再剥离——全角分隔符（如 U+FF0D）转半角后同样被剥离。
_STRIP_TRANSLATION = {ord(ch): None for ch in _INVISIBLE_CHARS + _INFIX_SEPARATORS}


def fold_text(text: str) -> str:
    """文本折叠：全角转半角 → 剥离零宽/中缀分隔符 → 小写 → 去除空白。

    折叠空间是硬规则唯一的匹配空间：**词典侧**（``dictionary.py`` 建别名
    索引时对每个别名调用本函数）与**文本侧**（``normalize`` / ``find_mentions``）
    共用同一函数，剥离规则对两侧一致生效，不存在"词典折叠了、文本没折叠"
    的漏检缝。
    """
    half = text.translate(_FULLWIDTH_TRANSLATION).translate(_STRIP_TRANSLATION)
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
