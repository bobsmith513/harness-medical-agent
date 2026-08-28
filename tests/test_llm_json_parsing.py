"""LLM 输出 JSON 提取测试：嵌套 / 围栏 / 多段 / 字符串内花括号。

背景（静态分析整改）：路由器旧正则 ``\\{[^{}]*\\}`` 不支持嵌套对象，
judge 旧贪婪正则 ``\\{.*\\}`` + DOTALL 在多段 JSON 输出下跨段误匹配。
两处已统一改用 ``extract_json_object``（括号深度扫描，字符串感知），
本文件锁定其行为契约。
"""

from __future__ import annotations

import json

from factories import make_conclusion, make_evidence_pack
from harness_agent.gates.quality_judge import LLMJudgeGate
from harness_agent.llm.json_utils import extract_json_object, strip_code_fences
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.session import SessionContext
from harness_agent.orchestrator.router import LLMRouter


# ===========================================================================
# 1. extract_json_object 契约
# ===========================================================================
class TestExtractJsonObject:
    def test_plain_object(self):
        assert extract_json_object('{"decision": "need_reasoning"}') == (
            '{"decision": "need_reasoning"}'
        )

    def test_nested_object(self):
        """嵌套对象：旧正则 ``\\{[^{}]*\\}`` 在此返回 None（不支持嵌套）。"""
        text = '{"decision": "no_reasoning", "meta": {"confidence": 0.8}}'
        fragment = extract_json_object(text)
        assert fragment is not None
        assert json.loads(fragment)["decision"] == "no_reasoning"
        assert json.loads(fragment)["meta"]["confidence"] == 0.8

    def test_braces_inside_strings_do_not_close(self):
        """字符串内的 ``}`` 不闭合对象（字符串感知扫描）。"""
        text = '{"reason": "含 } 花括号", "faithfulness": 0.9}'
        fragment = extract_json_object(text)
        assert fragment is not None
        assert json.loads(fragment)["reason"] == "含 } 花括号"

    def test_escaped_quote_inside_string(self):
        text = '{"reason": "说 \\"你好\\"", "ok": true}'
        fragment = extract_json_object(text)
        assert fragment is not None
        assert json.loads(fragment)["ok"] is True

    def test_code_fence_stripped(self):
        text = '```json\n{"decision": "need_reasoning"}\n```'
        fragment = extract_json_object(text)
        assert fragment is not None
        assert json.loads(fragment)["decision"] == "need_reasoning"

    def test_preamble_text_before_object(self):
        text = '好的，我的裁决如下：\n{"decision": "no_reasoning"}'
        fragment = extract_json_object(text)
        assert fragment is not None
        assert json.loads(fragment)["decision"] == "no_reasoning"

    def test_multiple_objects_takes_first(self):
        """多段 JSON：取首个平衡对象（旧贪婪正则会跨段拼出非法片段）。"""
        text = '{"a": 1} 中间说明文字 {"b": 2}'
        assert extract_json_object(text) == '{"a": 1}'

    def test_no_brace_returns_none(self):
        assert extract_json_object("纯文本，没有对象") is None
        assert extract_json_object("") is None

    def test_unbalanced_returns_none(self):
        """从不闭合的片段 → None（调用方按解析失败 fail-closed）。"""
        assert extract_json_object('{"decision": ') is None
        assert extract_json_object('{"a": {"b": 1}') is None


class TestStripCodeFences:
    def test_strips_json_fence(self):
        assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_plain_fence(self):
        assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_plain_text_untouched(self):
        assert strip_code_fences("无围栏文本") == "无围栏文本"


# ===========================================================================
# 2. 路由器：嵌套 JSON 输出仍可解析
# ===========================================================================
class TestRouterNestedParsing:
    def _llm_router(self, script: list[str]) -> LLMRouter:
        return LLMRouter(client=MockLLMClient(role="router", script=script))

    def test_nested_output_parsed(self):
        """模型附加嵌套 meta 字段时裁决仍可取出（旧正则此场景必 escalate）。"""
        router = self._llm_router(
            ['{"decision": "need_reasoning", "meta": {"lang": "zh", "confidence": 0.9}}']
        )
        record = router.route("随意文本", SessionContext(patient_id="pat-001"), attempt=1)
        assert record.decision == "need_reasoning"
        assert record.by_rule is False

    def test_nested_string_with_brace(self):
        router = self._llm_router(['{"decision": "no_reasoning", "note": "用户说了 } 这个符号"}'])
        record = router.route("随意文本", SessionContext(patient_id="pat-001"), attempt=1)
        assert record.decision == "no_reasoning"

    def test_garbage_still_escalates(self):
        router = self._llm_router(["对不起，我无法输出 JSON"])
        record = router.route("随意文本", SessionContext(patient_id="pat-001"), attempt=1)
        assert record.decision == "escalate"

    def test_first_object_without_decision_escalates(self):
        """首个对象缺 decision 字段 → 解析失败 fail-closed（不跨段找第二个）。"""
        router = self._llm_router(['{"a": 1} 说明 {"decision": "need_reasoning"}'])
        record = router.route("随意文本", SessionContext(patient_id="pat-001"), attempt=1)
        assert record.decision == "escalate"


# ===========================================================================
# 3. judge：多段 / 嵌套输出解析
# ===========================================================================
class TestJudgeParsing:
    def test_multi_segment_takes_first(self):
        verdict = LLMJudgeGate._parse_judge_output('{"noise": 1} 分析说明 {"faithfulness": 0.9}')
        # 首个平衡对象 {"noise": 1} 被取出：合法 dict 但缺 faithfulness，
        # 由 _build_verdict 按字段缺失 fail-closed 拦截（不会跨段取第二个）
        assert verdict == {"noise": 1}
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge"))
        built = gate._build_verdict(verdict)
        assert built.allowed is False
        assert "忠实度字段缺失" in built.reason

    def test_nested_judge_output_parsed(self):
        text = json.dumps(
            {
                "faithfulness": 0.88,
                "has_hallucination": False,
                "causal_inversion": False,
                "details": {"cited": ["ev-1"], "score_breakdown": {"support": 0.9}},
            },
            ensure_ascii=False,
        )
        verdict = LLMJudgeGate._parse_judge_output(text)
        assert verdict["faithfulness"] == 0.88
        assert verdict["details"]["score_breakdown"]["support"] == 0.9

    def test_garbage_fails_closed(self):
        verdict = LLMJudgeGate._parse_judge_output("模型抽风了，没有 JSON")
        assert verdict["faithfulness"] == 0.0
        assert "不可解析" in verdict["reason"]

    def test_evaluate_with_fenced_script(self):
        """端到端：围栏包裹的 judge 应答经完整门禁链路仍可放行。"""
        fenced = (
            "```json\n"
            + json.dumps(
                {
                    "faithfulness": 0.95,
                    "has_hallucination": False,
                    "causal_inversion": False,
                    "reason": "结论有证据支撑",
                }
            )
            + "\n```"
        )
        gate = LLMJudgeGate(llm=MockLLMClient(role="judge", script=[fenced]))
        verdict = gate.evaluate(make_conclusion(), make_evidence_pack())
        assert verdict.allowed is True
        assert "0.95" in verdict.reason
