"""路由器（M4）：规则前置 + LLM 兜底，二值输出，误判二次路由。

裁决优先级（严格顺序，永不跳级）::

    规则前置（关键词/正则，100% 确定，零 LLM 开销）
      └─ 命中 → RouteRecord(by_rule=True)
      └─ 未命中 → LLM 兜底（结构化 JSON 输出）
                    └─ 解析失败/非法值 = 误判
                         └─ 二次路由（attempt=2，纠错提示更强）
                              └─ 仍失败 → escalate（转澄清/人工）
                              └─ 成功 → RouteRecord(by_rule=False, attempt=2)

**没有"直接回答"出口**：``RouteDecision`` 三值枚举（M1 冻结）在
类型层面锁死主 Agent 的应答权——路由失败只能升级，绝不回退为
主 Agent 自行应答。
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from harness_agent.contracts.llm import LLMClient, LLMMessage
from harness_agent.models.session import RouteDecision, RouteRecord, SessionContext

__all__ = [
    "BinaryRouter",
    "LLMRouter",
    "RouteRuleSet",
    "default_rule_set",
]

#: 合法二值（不含 escalate：它只由路由器内部失败路径产生，LLM 不允许自报）
_BINARY_DECISIONS: frozenset[str] = frozenset({"need_reasoning", "no_reasoning"})


class RouteRuleSet(BaseModel):
    """规则路由表：关键词集（声明式，可由外部配置覆盖）。

    - ``reasoning_patterns``：命中即需要临床推理（诊断/治疗/用药决策）；
    - ``context_patterns``：命中即无需推理（既往史查询/复诊确认/信息登记）；
    - 正则大小写不敏感，中文直接子串匹配。
    """

    reasoning_patterns: list[str] = Field(default_factory=list)
    context_patterns: list[str] = Field(default_factory=list)

    def match(self, query: str) -> RouteDecision | None:
        """返回命中裁决；两表均未命中返回 None（移交 LLM 兜底）。"""
        folded = query.strip().lower()
        if any(re.search(pattern.lower(), folded) for pattern in self.reasoning_patterns):
            return "need_reasoning"
        if any(re.search(pattern.lower(), folded) for pattern in self.context_patterns):
            return "no_reasoning"
        return None


def default_rule_set() -> RouteRuleSet:
    """内置默认规则表（demo / 测试基线；生产可从 YAML 覆盖）。"""
    return RouteRuleSet(
        reasoning_patterns=[
            # 诊断推理类
            r"诊断",
            r"确诊",
            r"鉴别",
            r"是什么病",
            r"可能.*病因",
            r"严重吗",
            # 治疗决策类
            r"怎么治",
            r"治疗方案",
            r"用药",
            r"剂量",
            r"处方",
            r"能不能吃",
            r"可以.*一起.*吃",
            r"换药",
            r"停药",
            r"调整.*方案",
            # 风险评估类
            r"副作用",
            r"不良反应",
            r"禁忌",
            r"相互作用",
            r"过敏.*反应",
            # 英文（大小写不敏感）
            r"diagnos",
            r"treat",
            r"dosage",
            r"prescrib",
            r"interact",
            r"contraindicat",
            r"side.?effect",
        ],
        context_patterns=[
            # 既往信息查询（事实检索，无需推理）
            r"既往",
            r"病史",
            r"上次",
            r"之前.*(说|讲|提)",
            r"记录.*(是|有)什么",
            r"我.*说过",
            r"记得.*吗",
            r"查.*过敏史",
            # 复诊确认
            r"复诊",
            r"还是.*上次",
            r"继续.*原来",
            # 登记与闲聊
            r"帮我记",
            r"记住",
            r"你好",
            r"谢谢",
            r"再见",
            # 英文
            r"what did i",
            r"my history",
            r"last visit",
            r"remember",
        ],
    )


class RuleRouter:
    """规则前置路由：关键词/正则精确匹配，零 LLM 开销。"""

    def __init__(self, rules: RouteRuleSet | None = None) -> None:
        self._rules = rules if rules is not None else default_rule_set()

    def route(self, query: str) -> RouteRecord | None:
        """命中返回 by_rule 裁决；未命中返回 None（走 LLM 兜底）。"""
        decision = self._rules.match(query)
        if decision is None:
            return None
        return RouteRecord(
            decision=decision,
            by_rule=True,
            attempt=1,
            reason=f"规则前置命中: {decision}",
        )


#: LLM 兜底系统提示词（强约束 JSON 二值输出）
_ROUTER_SYSTEM_PROMPT = """你是医疗多智能体系统的路由器。判断用户查询是否需要临床推理。

需要推理（need_reasoning）：诊断、鉴别、治疗决策、用药选择、剂量调整、
风险评估、副作用/相互作用分析等需要证据与推断链的问题。
无需推理（no_reasoning）：既往史查询、复诊确认、信息登记、闲聊寒暄等
事实检索或事务性问题。

只输出一行 JSON，禁止任何其他文本：
{"decision": "need_reasoning"}
或
{"decision": "no_reasoning"}"""

#: 二次路由纠错提示（首次解析失败后追加）
_RETRY_SUFFIX = """
上一次输出无法解析为合法二值裁决。你必须只输出一行 JSON，
形如 {"decision": "need_reasoning"} 或 {"decision": "no_reasoning"}，
不要输出任何解释、markdown 或多余字符。"""


class LLMRouter:
    """LLM 兜底路由：结构化 JSON 输出解析（非法输出即误判）。"""

    def __init__(self, client: LLMClient) -> None:
        if client.role != "router":
            raise ValueError(f"路由 LLM 客户端 role 必须为 router，收到: {client.role}")
        self._client = client

    def _parse(self, text: str) -> RouteDecision | None:
        """从模型输出提取二值裁决；不可解析返回 None（误判信号）。"""
        # 容错：剥离 markdown 代码块围栏后取首个 JSON 对象
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        match = re.search(r"\{[^{}]*\}", cleaned)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        decision = payload.get("decision") if isinstance(payload, dict) else None
        if not isinstance(decision, str) or decision not in _BINARY_DECISIONS:
            return None
        return decision  # type: ignore[return-value]

    def route(
        self,
        query: str,
        context: SessionContext,
        *,
        attempt: int,
        retry_hint: bool = False,
    ) -> RouteRecord:
        """单次 LLM 兜底裁决（解析失败仍返回 escalate 记录，不抛异常）。"""
        system = _ROUTER_SYSTEM_PROMPT + (_RETRY_SUFFIX if retry_hint else "")
        result = self._client.complete(
            [
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=query),
            ]
        )
        decision = self._parse(result.text)
        if decision is None:
            return RouteRecord(
                decision="escalate",
                by_rule=False,
                attempt=attempt,
                reason=f"LLM 输出不可解析为二值裁决（attempt={attempt}），fail-closed 升级",
            )
        return RouteRecord(
            decision=decision,
            by_rule=False,
            attempt=attempt,
            reason=f"LLM 兜底裁决（attempt={attempt}）",
        )


class BinaryRouter:
    """路由门面：规则前置 → LLM 兜底 → 误判二次路由 → escalate。

    主 Agent 的唯一路由入口；``escalate`` 是唯一失败出口
    （转澄清/人工），不存在"主 Agent 直接应答"路径。
    """

    def __init__(self, rule_router: RuleRouter, llm_router: LLMRouter) -> None:
        self._rule_router = rule_router
        self._llm_router = llm_router

    def route(self, query: str, context: SessionContext) -> RouteRecord:
        """完整路由链：返回最终裁决（含 escalate 出口）。"""
        # 1. 规则前置：可判场景 100% 命中，零 LLM 开销
        ruled = self._rule_router.route(query)
        if ruled is not None:
            return ruled

        # 2. LLM 兜底首次
        first = self._llm_router.route(query, context, attempt=1)
        if first.decision != "escalate":
            return first

        # 3. 误判 → 二次路由（纠错提示更强）
        second = self._llm_router.route(query, context, attempt=2, retry_hint=True)
        if second.decision != "escalate":
            return second

        # 4. 二次仍失败 → escalate（fail-closed，绝不回退为主 Agent 应答）
        return RouteRecord(
            decision="escalate",
            by_rule=False,
            attempt=2,
            reason="两次 LLM 兜底均无法解析为二值裁决，升级澄清/人工",
        )
