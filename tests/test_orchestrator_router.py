"""M4 路由器测试：规则前置 / LLM 兜底 / 误判二次路由 / fail-closed。

验收锁定：
1. 规则可判场景 100% 命中（by_rule=True，零 LLM 调用）；
2. 规则不可判走 LLM 兜底（by_rule=False）；
3. 误判（输出不可解析/非法值）→ 二次路由（attempt=2）→ 仍失败 escalate；
4. 不存在"直接回答"出口（escalate 是唯一失败终点）。
"""

from __future__ import annotations

import pytest

from harness_agent.contracts.llm import LLMClient
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator.router import (
    BinaryRouter,
    LLMRouter,
    RouteRuleSet,
    RuleRouter,
    default_rule_set,
)


def _context(patient_id: str = "pat-003") -> SessionContext:
    return SessionContext(patient_id=patient_id)


def _router(script: list[str] | None = None) -> tuple[BinaryRouter, MockLLMClient]:
    llm = MockLLMClient(role="router", script=script or [])
    return (
        BinaryRouter(rule_router=RuleRouter(), llm_router=LLMRouter(client=llm)),
        llm,
    )


# ---------------------------------------------------------------------------
# Mock LLM 客户端
# ---------------------------------------------------------------------------
class TestMockLLMClient:
    def test_satisfies_contract(self):
        assert isinstance(MockLLMClient(role="router"), LLMClient)

    def test_script_sequence_then_sticky_last(self):
        llm = MockLLMClient(role="router", script=["a", "b"])
        assert llm.complete([]).text == "a"
        assert llm.complete([]).text == "b"
        assert llm.complete([]).text == "b"  # 耗尽后重复最后一个

    def test_empty_script_returns_empty_text(self):
        assert MockLLMClient(role="router").complete([]).text == ""

    def test_calls_recorded(self):
        llm = MockLLMClient(role="router", script=["ok"])
        llm.complete([])
        llm.complete([])
        assert len(llm.calls) == 2

    def test_usage_fields_populated(self):
        result = MockLLMClient(role="router", script=["hello world"]).complete([])
        assert result.completion_tokens == 2
        assert result.model == "mock-router"


# ---------------------------------------------------------------------------
# 规则路由
# ---------------------------------------------------------------------------
class TestRuleRouter:
    @pytest.mark.parametrize(
        "query",
        [
            "这个症状可能是什么诊断",
            "帮我看看治疗方案",
            "这两种药可以一起吃吗",
            "头孢的副作用有哪些",
            "what is the recommended dosage",
            "any contraindications with aspirin",
        ],
    )
    def test_reasoning_queries_hit_rules(self, query):
        record = RuleRouter().route(query)
        assert record is not None
        assert record.decision == "need_reasoning"
        assert record.by_rule is True

    @pytest.mark.parametrize(
        "query",
        [
            "我上次说过什么",
            "查一下我的既往病史",
            "还是上次的药继续吗",
            "复诊",
            "帮我记一下我今天吃了阿莫西林",
            "what did i say last visit",
        ],
    )
    def test_context_queries_hit_rules(self, query):
        record = RuleRouter().route(query)
        assert record is not None
        assert record.decision == "no_reasoning"
        assert record.by_rule is True

    @pytest.mark.parametrize(
        "query",
        [
            "最近老睡不着觉白天没精神",  # 主诉模糊：两表均未命中
            "帮我看下这个报告单",  # 指代不明
            "12345",
        ],
    )
    def test_ambiguous_queries_miss_rules(self, query):
        assert RuleRouter().route(query) is None

    def test_rule_hit_consumes_no_llm(self):
        router, llm = _router()
        record = router.route("帮我看看怎么治疗", _context())
        assert record is not None and record.by_rule is True
        assert len(llm.calls) == 0  # 规则命中零 LLM 开销

    def test_custom_rule_set_overrides_default(self):
        rules = RouteRuleSet(reasoning_patterns=["特殊关键词"], context_patterns=[])
        router = RuleRouter(rules)
        assert router.route("包含特殊关键词的任意话") is not None
        assert router.route("帮我看看诊断") is None  # 默认表不再生效

    def test_default_rule_set_is_fresh_instance(self):
        first = default_rule_set()
        first.reasoning_patterns.clear()
        assert default_rule_set().reasoning_patterns  # 清空不污染后续调用


# ---------------------------------------------------------------------------
# LLM 兜底路由
# ---------------------------------------------------------------------------
class TestLLMRouter:
    def test_role_must_be_router(self):
        with pytest.raises(ValueError, match="router"):
            LLMRouter(client=MockLLMClient(role="reasoning"))

    def test_parses_plain_json(self):
        llm = MockLLMClient(role="router", script=['{"decision": "need_reasoning"}'])
        record = LLMRouter(client=llm).route("模糊问题", _context(), attempt=1)
        assert record.decision == "need_reasoning"
        assert record.by_rule is False
        assert record.attempt == 1

    def test_parses_json_in_code_fence(self):
        llm = MockLLMClient(role="router", script=['```json\n{"decision": "no_reasoning"}\n```'])
        record = LLMRouter(client=llm).route("模糊问题", _context(), attempt=1)
        assert record.decision == "no_reasoning"

    def test_unparsable_output_is_escalate(self):
        llm = MockLLMClient(role="router", script=["我认为这个问题很复杂，无法简单二分"])
        record = LLMRouter(client=llm).route("模糊问题", _context(), attempt=1)
        assert record.decision == "escalate"
        assert "不可解析" in record.reason

    def test_illegal_decision_value_is_escalate(self):
        llm = MockLLMClient(role="router", script=['{"decision": "maybe"}'])
        record = LLMRouter(client=llm).route("模糊问题", _context(), attempt=1)
        assert record.decision == "escalate"

    def test_escalate_value_rejected_from_llm(self):
        """LLM 不得自报 escalate（升级只能由路由器内部失败路径产生）。"""
        llm = MockLLMClient(role="router", script=['{"decision": "escalate"}'])
        record = LLMRouter(client=llm).route("模糊问题", _context(), attempt=1)
        assert record.decision == "escalate"
        assert "不可解析" in record.reason  # 按非法值处理


# ---------------------------------------------------------------------------
# 二值路由门面（完整链路）
# ---------------------------------------------------------------------------
class TestBinaryRouter:
    def test_rule_hit_short_circuits_llm(self):
        router, llm = _router(script=['{"decision": "no_reasoning"}'])
        record = router.route("这个诊断是什么", _context())
        assert record.decision == "need_reasoning"
        assert record.by_rule is True
        assert len(llm.calls) == 0

    def test_miss_falls_back_to_llm(self):
        router, llm = _router(script=['{"decision": "no_reasoning"}'])
        record = router.route("最近老睡不着觉白天没精神", _context())
        assert record.decision == "no_reasoning"
        assert record.by_rule is False
        assert record.attempt == 1
        assert len(llm.calls) == 1

    def test_misjudge_retries_then_succeeds(self):
        """首次输出不可解析 → 二次路由（纠错提示）→ 成功。"""
        router, llm = _router(script=["无法判断这个问题", '{"decision": "need_reasoning"}'])
        record = router.route("帮我看个事", _context())
        assert record.decision == "need_reasoning"
        assert record.by_rule is False
        assert record.attempt == 2
        assert len(llm.calls) == 2
        # 二次调用携带纠错提示
        retry_messages = llm.calls[1][0].content
        assert "上一次输出无法解析" in retry_messages

    def test_double_misjudge_escalates(self):
        """两次均误判 → escalate（fail-closed，绝不主 Agent 应答）。"""
        router, llm = _router(script=["完全无关的回答", "还是不行"])
        record = router.route("帮我看个事", _context())
        assert record.decision == "escalate"
        assert record.attempt == 2
        assert len(llm.calls) == 2
        assert "两次" in record.reason

    def test_escalate_never_returns_direct_answer(self):
        """失败终点唯一：escalate（枚举无直接应答出口）。"""
        router, _ = _router(script=["x", "y"])
        record = router.route("帮我看个事", _context())
        assert record.decision in {"need_reasoning", "no_reasoning", "escalate"}
        if record.decision == "escalate":
            assert record.by_rule is False
