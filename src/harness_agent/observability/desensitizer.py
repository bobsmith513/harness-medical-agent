"""出站脱敏中间件（M7）：正则匹配患者标识并替换。

脱敏边界延伸至：
- 外部模型 API 调用（LLM 请求 prompt 中的患者标识）；
- 沙箱检查点 state（写检查点前执行）；
- 日志输出（审计记录的 payload）；
- Redis 缓存 value（缓存前执行）。

支持的 PII 类型：
- 身份证号（18 位 / 15 位）→ ``[REDACTED-ID]``
- 手机号（11 位）→ ``[REDACTED-PHONE]``
- 患者编号（pat-xxx / 患者编号:xxx）→ ``[REDACTED-PATID]``
- 邮箱地址 → ``[REDACTED-EMAIL]``
- 患者姓名标记（姓名：xxx / 患者：xxx —— 显式标记式，避免误伤临床正文）→ ``[REDACTED-NAME]``
"""

from __future__ import annotations

import re

from harness_agent.contracts.observability import DesensitizedText

__all__ = ["PatternDesensitizer"]

#: 脱敏规则：(pattern, entity_type)
#:
#: 边界用环视（lookaround）而非 ``\b`` 词边界：Python 正则里中文也属于
#: ``\w``，中文与数字之间不构成词边界，"身份证310101…" / "电话138…"
#: 这类紧贴中文（无空格）的 PII 会被 ``\b`` 静默漏过。
#: 数字环视同时防止长数字串内部的误报（如 18 位身份证内嵌 11 位手机号）。
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 身份证号（18 位：地区6 + 年月日8 + 顺序3 + 校验1）
    (
        re.compile(
            r"(?<![0-9Xx])[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
            r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
        ),
        "ID",
    ),
    # 身份证号（15 位：旧版）
    (
        re.compile(
            r"(?<![0-9Xx])[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?!\d)"
        ),
        "ID",
    ),
    # 手机号（11 位，1 开头）
    (
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "PHONE",
    ),
    # 患者编号（pat-xxx 或 patient_id=xxx）
    (
        re.compile(r"(?<![A-Za-z0-9])pat-[a-z0-9]+(?![A-Za-z0-9])", re.IGNORECASE),
        "PATID",
    ),
    (
        re.compile(r"patient[_\s]*id\s*[:=]\s*\S+", re.IGNORECASE),
        "PATID",
    ),
    # 邮箱
    (
        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "EMAIL",
    ),
    # 姓名标记（姓名:xxx / 患者：xxx —— 显式标记式，避免误伤临床正文）
    (
        re.compile(r"姓名\s*[:：]\s*[\u4e00-\u9fa5]{2,4}"),
        "NAME",
    ),
    (
        re.compile(r"患者\s*[:：]\s*[\u4e00-\u9fa5]{2,4}"),
        "NAME",
    ),
]


class PatternDesensitizer:
    """正则脱敏中间件：出站调用前去除患者标识。

    实现简单、零外部依赖，适合 demo 与中等合规要求场景；
    生产环境可替换为 NER 模型脱敏器（实现同一 ``Desensitizer`` 接口）。
    """

    def __init__(self, patterns: list[tuple[re.Pattern[str], str]] | None = None) -> None:
        self._patterns = patterns or _PATTERNS

    def desensitize(self, text: str) -> DesensitizedText:
        """脱敏：替换 PII 为占位符，记录被移除的实体。"""
        removed: list[str] = []
        result = text

        for pattern, entity_type in self._patterns:
            matches = pattern.findall(result)
            for match in matches:
                removed.append(f"{entity_type}:{match}")
            result = pattern.sub(f"[REDACTED-{entity_type}]", result)

        return DesensitizedText(text=result, removed_entities=removed)

    def desensitize_dict(self, data: dict) -> dict:
        """递归脱敏字典中的字符串值（检查点 state / 日志 payload 用）。"""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.desensitize(value).text
            elif isinstance(value, dict):
                result[key] = self.desensitize_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.desensitize(item).text if isinstance(item, str) else item for item in value
                ]
            else:
                result[key] = value
        return result
