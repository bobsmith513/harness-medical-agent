"""审计模型测试：裁决、事件、审计记录。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from harness_agent.models.audit import AuditRecord, GateVerdict, TraceEvent


class TestGateVerdict:
    def test_defaults(self):
        verdict = GateVerdict(gate="input", allowed=True)
        assert verdict.reason == ""
        assert verdict.blocked_drugs == []
        assert verdict.checked_at is not None

    def test_blocked_drugs_recorded(self):
        verdict = GateVerdict(
            gate="assembly",
            allowed=False,
            reason="证据含过敏药物实体",
            blocked_drugs=["penicillin"],
        )
        assert verdict.blocked_drugs == ["penicillin"]

    def test_invalid_gate_name_rejected(self):
        with pytest.raises(ValidationError):
            GateVerdict(gate="final", allowed=True)

    def test_all_five_gate_names_valid(self):
        for gate in ("input", "assembly", "output", "quality_judge", "drug_safety"):
            assert GateVerdict(gate=gate, allowed=True).gate == gate


class TestTraceEvent:
    def test_defaults(self):
        event = TraceEvent(trace_id="tr-1", event_type="route")
        assert event.session_id is None
        assert event.payload == {}
        assert event.event_id.startswith("evt-")

    def test_payload_carries_details(self):
        event = TraceEvent(
            trace_id="tr-1",
            session_id="sess-1",
            event_type="llm_call",
            payload={"model": "deepseek-chat", "prompt_tokens": 1200},
        )
        assert event.payload["model"] == "deepseek-chat"


class TestAuditRecord:
    def test_optional_fields(self):
        record = AuditRecord(trace_id="tr-1", actor="gate:input", action="gate_check")
        assert record.session_id is None
        assert record.turn_index is None
        assert record.verdict is None

    def test_with_verdict(self):
        verdict = GateVerdict(gate="quality_judge", allowed=False, reason="忠实度低于阈值")
        record = AuditRecord(
            trace_id="tr-1",
            session_id="sess-1",
            turn_index=5,
            actor="gate:quality_judge",
            action="gate_check",
            verdict=verdict,
        )
        assert record.verdict is not None
        assert record.verdict.allowed is False
