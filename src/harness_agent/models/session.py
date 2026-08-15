"""会话与路由模型（M1）。

fail-closed 与上下文压缩的类型级落地：

1. ``RouteDecision`` 只有三个出口：需要临床推理 / 不需要 / 升级人工。
   **没有"直接回答"出口**——主 Agent 纯编排无应答权，从枚举层面锁死。
2. ``SessionContext.add_turn`` 只保留最近 ``keep`` 轮 + 文件指针，
   溢出的旧轮由调用方持久化至 VFS（M6 实现），此为长会话 Token 压缩核心。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from harness_agent.models.common import new_id

__all__ = [
    "RouteDecision",
    "RouteRecord",
    "SessionContext",
    "TurnRecord",
]

#: 路由二值判断 + 升级出口（fail-closed）：
#: - need_reasoning: 需要临床推理 -> 委派推理专家
#: - no_reasoning:   无需推理     -> 记忆专家装配上下文直接响应
#: - escalate:       规则与 LLM 兜底均无法裁决 -> 转澄清或人工
RouteDecision = Literal["need_reasoning", "no_reasoning", "escalate"]


class RouteRecord(BaseModel):
    """一次路由裁决记录。

    ``by_rule=True`` 表示规则前置命中（关键词/正则），未命中才走 LLM 兜底；
    ``attempt`` 为 1 表示首次路由，为 2 表示误判后的二次路由，
    二次仍失败必须 escalate，绝不回退为主 Agent 直接应答。
    """

    decision: RouteDecision
    by_rule: bool = False
    attempt: int = 1
    reason: str = ""


class TurnRecord(BaseModel):
    """单轮会话记录（输入为脱敏后文本，审计可安全落库）。"""

    turn_id: str = Field(default_factory=lambda: new_id("turn"))
    turn_index: int
    user_input: str
    token_count: int = 0
    route: RouteRecord | None = None
    evidence_pack_id: str | None = None
    conclusion_id: str | None = None
    escalated_to_human: bool = False


class SessionContext(BaseModel):
    """会话上下文：最近 N 轮 + VFS 文件指针。

    长会话下证据、推理链、摘要持久化至虚拟目录，上下文只留最近 3 轮
    与文件指针（``file_pointers``：逻辑路径 -> VFS 文件名），
    20 轮以上长会话 Token 降约 50% 的实现载体。
    """

    session_id: str = Field(default_factory=lambda: new_id("sess"))
    patient_id: str
    recent_turns: list[TurnRecord] = Field(default_factory=list)
    file_pointers: dict[str, str] = Field(default_factory=dict)
    token_budget_used: int = 0

    def add_turn(self, turn: TurnRecord, keep: int = 3) -> list[TurnRecord]:
        """压入新一轮，返回被移出上下文的旧轮（按时间序）。

        超出 ``keep`` 的旧轮被移出——M6 将其持久化至 VFS 并登记文件指针；
        M1 仅定义结构语义，调用方负责处理返回值。
        """
        self.recent_turns.append(turn)
        dropped: list[TurnRecord] = []
        while len(self.recent_turns) > keep:
            dropped.append(self.recent_turns.pop(0))
        return dropped
