"""检索层分词器：中文二元组 + 英文/数字词元（共享于嵌入与 BM25）。

零依赖纯 Python，两路检索共用同一词元空间保证口径一致。
"""

from __future__ import annotations

__all__ = ["tokenize"]


def tokenize(text: str) -> list[str]:
    """分词：ASCII 字母数字串为词元；连续汉字生成二元组（单字成词元）。

    >>> tokenize("患者 penicillin 过敏")
    ['患者', 'penicillin', '过敏']
    """
    tokens: list[str] = []
    ascii_run: list[str] = []
    cjk_run: list[str] = []

    def flush_ascii() -> None:
        if ascii_run:
            tokens.append("".join(ascii_run))
            ascii_run.clear()

    def flush_cjk() -> None:
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        elif len(cjk_run) > 1:
            tokens.extend(a + b for a, b in zip(cjk_run, cjk_run[1:], strict=False))
        cjk_run.clear()

    for ch in text.lower():
        if ch.isascii() and ch.isalnum():
            flush_cjk()
            ascii_run.append(ch)
        elif "\u4e00" <= ch <= "\u9fff":
            flush_ascii()
            cjk_run.append(ch)
        else:
            flush_ascii()
            flush_cjk()
    flush_ascii()
    flush_cjk()
    return tokens
