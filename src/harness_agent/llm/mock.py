"""Mock LLM 客户端（M4）：脚本化应答，零外部依赖。

路由器 LLM 兜底的默认实现：按 ``role`` 绑定应答脚本，
``complete`` 依序弹出应答（或恒定应答），不发起任何网络调用。

脚本注入两种形态（构造参数 / ``set_script``），测试与 demo
用它精确控制"可解析应答 / 误判应答 / 异常"三类路径。
"""

from __future__ import annotations

from collections.abc import Sequence

from harness_agent.contracts.llm import LLMMessage, LLMResult, LLMRole

__all__ = ["MockLLMClient"]


class MockLLMClient:
    """脚本化 LLM 客户端（实现 M1 ``LLMClient`` 契约）。

    应答队列依次弹出；耗尽后重复最后一个应答（demo 会话
    天然幂等）；空脚本默认应答空文本（调用方按解析失败处理）。
    """

    def __init__(
        self,
        role: LLMRole,
        script: Sequence[str] = (),
    ) -> None:
        self.role = role
        self._script: list[str] = list(script)
        self._cursor = 0
        #: 记录收到的全部消息（断言提示词构造用）
        self.calls: list[list[LLMMessage]] = []

    def set_script(self, script: Sequence[str]) -> None:
        """重置应答脚本（测试夹具复用同一实例时使用）。"""
        self._script = list(script)
        self._cursor = 0

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        if not self._script:
            return LLMResult(text="", model="mock")
        index = min(self._cursor, len(self._script) - 1)
        text = self._script[index]
        self._cursor += 1
        # 用量估算：prompt 按字符、completion 按词（演示口径，够审计展示）
        return LLMResult(
            text=text,
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(text.split()) if text.strip() else 0,
            model=f"mock-{self.role}",
        )
