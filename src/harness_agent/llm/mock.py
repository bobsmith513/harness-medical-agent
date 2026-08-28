"""Mock LLM 客户端（M4）：脚本化应答，零外部依赖。

路由器 LLM 兜底的默认实现：按 ``role`` 绑定应答脚本，
``complete`` 依序弹出应答（或恒定应答），不发起任何网络调用。

脚本注入两种形态（构造参数 / ``set_script``），测试与 demo
用它精确控制"可解析应答 / 误判应答 / 异常"三类路径。

空脚本时的角色默认应答（零依赖演示模式保持可用）：

- ``judge``：忠实度 0.95 的合法裁决 JSON（演示模式下门禁有真实裁决可走）；
- ``router``：``need_reasoning`` 二值裁决 JSON；
- ``reasoning``：从提示中的合法 evidence_id 列表构造最小三段式推理链
  （无可用证据 ID 时返回空文本 → 调用方按解析失败 fail-closed 升级）；
- 其他角色：空文本。

设计边界：默认应答让"mock 模式"表现得像一个循规蹈矩的模型；
而**显式注入的垃圾脚本**（非 JSON 文本）仍会触发解析失败——
消费方的 fail-closed 路径照样被测试覆盖，两种行为互不掩盖。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from harness_agent.contracts.llm import LLMMessage, LLMResult, LLMRole

__all__ = ["MockLLMClient"]


def _default_judge_json() -> str:
    """judge 默认应答：忠实度 0.95，无臆测，无因果倒置。"""
    return json.dumps(
        {
            "faithfulness": 0.95,
            "has_hallucination": False,
            "causal_inversion": False,
            "reason": "Mock 默认应答（演示模式）：结论与证据一致",
        },
        ensure_ascii=False,
    )


def _default_router_json() -> str:
    """router 默认应答：需要临床推理。"""
    return json.dumps({"decision": "need_reasoning"}, ensure_ascii=False)


def _default_reasoning_json(messages: list[LLMMessage]) -> str:
    """reasoning 默认应答：按提示中的合法 evidence_id 构造最小推理链。"""
    user_content = "".join(m.content for m in messages if m.role != "system")
    ev_ids = sorted(set(re.findall(r"\bev-[0-9a-f]{6,}\b", user_content)))
    if not ev_ids:
        # 无合法证据可引用：返回空文本，由消费方按解析失败 fail-closed 升级
        return ""
    cited = ev_ids[:1]
    chain = {
        "steps": [
            {
                "kind": "evidence",
                "text": "引用证据：检索层返回的条目支持当前问诊判断",
                "citations": cited,
            },
            {
                "kind": "inference",
                "text": "结合患者主诉与上述证据推断：方案不涉及患者已知过敏药物",
            },
            {
                "kind": "conclusion",
                "text": "综合上述证据给出建议（Mock 默认应答，需临床医生确认）",
                "citations": cited,
            },
        ],
        "statement": "基于检索证据的初步建议（Mock 默认应答，需临床医生确认）",
        "self_check_notes": "自检通过（3/3）：引用真实、因果正向、依据充分",
    }
    return json.dumps(chain, ensure_ascii=False)


class MockLLMClient:
    """脚本化 LLM 客户端（实现 M1 ``LLMClient`` 契约）。

    应答队列依次弹出；耗尽后重复最后一个应答（demo 会话
    天然幂等）；空脚本按角色返回合法默认应答（见模块 docstring）。
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
            text = self._role_default(messages)
            model = f"mock-{self.role}"
        else:
            index = min(self._cursor, len(self._script) - 1)
            text = self._script[index]
            self._cursor += 1
            model = f"mock-{self.role}"
        # 用量估算：prompt 按字符、completion 按词（演示口径，够审计展示）
        return LLMResult(
            text=text,
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(text.split()) if text.strip() else 0,
            model=model,
        )

    def _role_default(self, messages: list[LLMMessage]) -> str:
        """空脚本时的角色默认应答。"""
        if self.role == "judge":
            return _default_judge_json()
        if self.role == "router":
            return _default_router_json()
        if self.role == "reasoning":
            return _default_reasoning_json(messages)
        return ""
