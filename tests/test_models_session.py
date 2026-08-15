"""会话与路由模型测试：fail-closed 出口 + 上下文压缩语义。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from factories import make_turn
from harness_agent.models.session import RouteRecord, SessionContext, TurnRecord


class TestRouteFailClosed:
    """路由只有二值 + 升级出口，没有"直接回答"分支。"""

    def test_valid_decisions_accepted(self):
        for decision in ("need_reasoning", "no_reasoning", "escalate"):
            record = RouteRecord(decision=decision)
            assert record.decision == decision

    def test_direct_answer_exit_does_not_exist(self):
        """主 Agent 无应答权：路由枚举层面不存在 direct_answer 出口。"""
        with pytest.raises(ValidationError):
            RouteRecord(decision="direct_answer")

    def test_second_attempt_recorded(self):
        """误判后的二次路由用 attempt=2 记录（仍失败必须 escalate）。"""
        record = RouteRecord(decision="escalate", by_rule=False, attempt=2, reason="二次路由仍失败")
        assert record.attempt == 2

    def test_route_defaults(self):
        record = RouteRecord(decision="need_reasoning")
        assert record.by_rule is False
        assert record.attempt == 1
        assert record.reason == ""


class TestContextCompaction:
    """上下文只留最近 keep 轮 + 文件指针（长会话 Token 压缩核心）。"""

    def test_add_turn_keeps_recent_three(self):
        session = SessionContext(patient_id="pat-001")
        for i in range(1, 6):
            session.add_turn(make_turn(index=i), keep=3)
        assert [t.turn_index for t in session.recent_turns] == [3, 4, 5]

    def test_add_turn_returns_dropped_in_order(self):
        session = SessionContext(patient_id="pat-001")
        dropped: list[TurnRecord] = []
        for i in range(1, 5):
            dropped.extend(session.add_turn(make_turn(index=i), keep=3))
        assert [t.turn_index for t in dropped] == [1]
        assert [t.turn_index for t in session.recent_turns] == [2, 3, 4]

    def test_file_pointers_default_empty(self):
        session = SessionContext(patient_id="pat-001")
        assert session.file_pointers == {}
        assert session.token_budget_used == 0

    def test_escalated_turn_marked(self):
        turn = TurnRecord(
            turn_index=1,
            user_input="转人工",
            escalated_to_human=True,
        )
        assert turn.escalated_to_human is True
        assert turn.conclusion_id is None
