"""公共值对象与工具（M1 地基）。

被 models 与 contracts 全体引用，保持零依赖（仅标准库 + pydantic）。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

__all__ = [
    "ConfidenceLevel",
    "Provenance",
    "new_id",
    "now_utc",
]

#: 来源置信度：临床提示层转正、摘要审核等环节依赖此分级。
ConfidenceLevel = Literal["high", "medium", "low"]

#: 事实来源类型。
#: - knowledge_base: 知识库已转正条目（脱敏入库，可作为检索事实）
#: - model_inference: 模型推断（未经审核不得固化为可召回事实）
#: - doctor_verified: 医生审定（最高可信度）
Provenance = Literal["knowledge_base", "model_inference", "doctor_verified"]


def now_utc() -> datetime:
    """UTC 时间戳（带时区，避免 naive datetime 歧义）。"""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """生成带前缀的短 id：``ev-3f9c2a1b8d4e`` 形式，便于日志与审计定位。"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
