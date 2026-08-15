"""审计与门禁裁决模型（M1）。

安全闸门与质量门禁的统一裁决载体：三道供给闸门（输入拦截 / 装配复核 /
输出校验）与质量门禁（LLM-judge / 药物安全）共用 ``GateVerdict``；
全链路事件与审计记录供 Langfuse / PostgreSQL 落地（M7 实现存储，
连接串在配置中默认留空）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from harness_agent.models.common import new_id, now_utc

__all__ = [
    "GateName",
    "GateVerdict",
    "AuditRecord",
    "TraceEvent",
]

#: 五道闸门名称：供给三道 + 质量两道。
GateName = Literal["input", "assembly", "output", "quality_judge", "drug_safety"]


class GateVerdict(BaseModel):
    """闸门裁决结果。

    fail-closed 语义：``allowed=False`` 即拦截，调用方必须走澄清或人工，
    绝不允许静默降级放行；拦截原因与被拦截药物实体全量记录，供审计回流。
    """

    gate: GateName
    allowed: bool
    reason: str = ""
    #: 被拦截的过敏/交叉反应药物（归一化后标准药名）
    blocked_drugs: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=now_utc)


class TraceEvent(BaseModel):
    """全链路 trace 事件（Langfuse 事件的最小载体，M7 落地）。"""

    event_id: str = Field(default_factory=lambda: new_id("evt"))
    trace_id: str
    session_id: str | None = None
    #: 事件类型：route / llm_call / gate_check / retrieve / sandbox_exec / ...
    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=now_utc)


class AuditRecord(BaseModel):
    """审计日志记录（PostgreSQL 存审计、全量写入；DSN 留空时降级 SQLite）。"""

    audit_id: str = Field(default_factory=lambda: new_id("aud"))
    trace_id: str
    session_id: str | None = None
    turn_index: int | None = None
    #: 行为主体：orchestrator / reasoning_expert / memory_expert / gate:input / ...
    actor: str
    #: 动作：delegate / retrieve / gate_check / conclude / escalate / ...
    action: str
    verdict: GateVerdict | None = None
    created_at: datetime = Field(default_factory=now_utc)
