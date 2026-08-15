"""LLM 客户端统一契约（M1）。

四类模型（编排 / 推理 / judge / 路由）共用同一 ``LLMClient`` 接口：

- Mock 实现：脚本化应答（demo 模式，零依赖）；
- OpenAI 兼容实现：走本地 vLLM 端口或在线 API（端点在 M0 配置中
  **默认留空**，``HARNESS_LLM__*_BASE_URL`` 填写后即生效）。

两种实现可无差别注入，业务逻辑永不分叉——这是本仓库 mock 边界的
基石约定（docs/development-plan.md 第五节）。
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from harness_agent.models.common import new_id

__all__ = ["LLMClient", "LLMMessage", "LLMResult", "LLMRole"]


class LLMMessage(BaseModel):
    """对话消息（system / user / assistant）。"""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    """模型返回：文本 + 用量（供 token 预算与可观测统计）。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    request_id: str = Field(default_factory=lambda: new_id("llm"))


#: LLM 用途角色：与 M0 配置的四组端点一一对应。
LLMRole = Literal["orchestrator", "reasoning", "judge", "router"]


@runtime_checkable
class LLMClient(Protocol):
    """LLM 统一调用接口（结构化契约）。

    实现方负责绑定自身端点与超时（来自 M0 配置）；
    调用方只关心消息进、结果出。
    """

    role: LLMRole

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult: ...
