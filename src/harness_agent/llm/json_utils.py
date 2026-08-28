"""LLM 输出 JSON 提取（M5 基础设施，路由器与 judge 共用）。

此前路由器用 ``\\{[^{}]*\\}``（不支持嵌套）、judge 用贪婪 ``\\{.*\\}``
加 DOTALL（多段 JSON 会跨段误匹配），两处正则在真实模型输出下均易碎。
本模块以括号深度扫描替代正则：

- 剥离 markdown 代码围栏（```json ... ```）；
- 从首个 ``{`` 起做**字符串感知的括号配平**：字符串内的 ``{``/``}``
  与转义字符不改变深度；
- 返回首个平衡的 JSON 对象原文；不存在平衡片段返回 None
  （调用方按解析失败 fail-closed 处理）。

用法::

    fragment = extract_json_object(text)
    if fragment is None:
        ...  # fail-closed
    payload = json.loads(fragment)
"""

from __future__ import annotations

import re

__all__ = ["extract_json_object", "strip_code_fences"]

_CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*")


def strip_code_fences(text: str) -> str:
    """剥离 markdown 代码围栏标记（``` 与 ```json），保留正文。"""
    return _CODE_FENCE_RE.sub("", text).strip()


def extract_json_object(text: str) -> str | None:
    """提取首个平衡的 JSON 对象片段（支持嵌套与字符串内花括号）。

    - 扫描是字符串感知的：``{"k": "v}"}`` 中的 ``}`` 不闭合对象；
    - 多段 JSON 只取第一段（judge / router 都只需要首个裁决对象）；
    - 无 ``{``、或从不闭合、或纯字符串前缀无对象 → None。
    """
    cleaned = strip_code_fences(text)
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : index + 1]
    return None
