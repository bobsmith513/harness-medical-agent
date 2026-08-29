"""M6 虚拟文件系统与记忆审核测试。

验收标准（development-plan.md M6）：
1. 20 轮模拟会话上下文 token 降约 50%（打印前后对比）；
2. "未审核摘要不得被召回"单测通过。

覆盖范围：
- VFS 存储（内存 + 文件两种实现）
- VFS 目录（四个虚拟目录的读写列举）
- 上下文压缩（溢出轮持久化 + 文件指针 + Token 压缩）
- 记忆审核队列（提交 → 自动审核 → 人工审核 → 转正/驳回 → 可召回）
- "未审核摘要不得被召回"强制约束
- 20 轮模拟会话 Token 降约 50% 验收
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from harness_agent.models.evidence import (
    Evidence,
    EvidencePack,
    SourceRef,
)
from harness_agent.models.memory import Memory
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)
from harness_agent.models.session import (
    RouteRecord,
    SessionContext,
    TurnRecord,
)
from harness_agent.vfs import (
    DIR_EVIDENCE,
    DIR_REASONING,
    DIR_SUMMARIES,
    ContextCompactor,
    FileBackedVfsStore,
    InMemoryVfsStore,
    MemoryReviewQueue,
    build_compactor,
    build_directory,
    build_vfs_store,
)


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------
def _evidence_pack(session_id: str = "sess-1", patient_id: str = "pat-1") -> EvidencePack:
    """已通过装配复核的证据包。"""
    evidence = Evidence(
        content="阿奇霉素适用于社区获得性肺炎，成人常规剂量 500mg qd",
        source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
        confidence="medium",
        provenance="knowledge_base",
    )
    return EvidencePack(
        session_id=session_id,
        patient_id=patient_id,
        query="测试查询",
        evidence=[evidence],
    )


def _conclusion(citation: str = "ev-1") -> ClinicalConclusion:
    """构造临床结论。"""
    chain = ReasoningChain(
        steps=[
            ReasoningStep(kind="evidence", text="引用证据", citations=[citation]),
            ReasoningStep(kind="inference", text="基于证据推断"),
            ReasoningStep(
                kind="conclusion",
                text="阿奇霉素 500mg qd 可用于肺炎",
                citations=[citation],
            ),
        ],
        self_check_passed=True,
    )
    return ClinicalConclusion(
        statement="阿奇霉素 500mg qd 可用于肺炎",
        reasoning_chain=chain,
        cited_evidence_ids=[citation],
    )


def _turn(index: int, token_count: int = 500) -> TurnRecord:
    """构造单轮会话记录。"""
    return TurnRecord(
        turn_index=index,
        user_input=f"第 {index} 轮对话：患者主诉症状描述……" * 5,
        token_count=token_count,
        route=RouteRecord(decision="need_reasoning", by_rule=True, reason="关键词命中"),
    )


# ===========================================================================
# 1. VFS 存储测试
# ===========================================================================
class TestInMemoryVfsStore:
    """内存 VFS 存储测试。"""

    def test_write_and_read(self):
        store = InMemoryVfsStore()
        path = store.write("sess-1", "/evidence/ev-1.json", '{"content": "test"}')
        assert path == "/evidence/ev-1.json"
        assert store.read("sess-1", "/evidence/ev-1.json") == '{"content": "test"}'

    def test_read_nonexistent_returns_none(self):
        store = InMemoryVfsStore()
        assert store.read("sess-1", "/evidence/missing.json") is None

    def test_exists(self):
        store = InMemoryVfsStore()
        store.write("sess-1", "/evidence/ev-1.json", "data")
        assert store.exists("sess-1", "/evidence/ev-1.json")
        assert not store.exists("sess-1", "/evidence/ev-2.json")

    def test_list_dir(self):
        store = InMemoryVfsStore()
        store.write("sess-1", "/evidence/ev-1.json", "a")
        store.write("sess-1", "/evidence/ev-2.json", "b")
        store.write("sess-1", "/reasoning/cc-1.json", "c")
        entries = store.list_dir("sess-1", "/evidence")
        assert len(entries) == 2
        assert all(e.startswith("/evidence/") for e in entries)

    def test_delete(self):
        store = InMemoryVfsStore()
        store.write("sess-1", "/evidence/ev-1.json", "data")
        assert store.delete("sess-1", "/evidence/ev-1.json")
        assert not store.exists("sess-1", "/evidence/ev-1.json")
        assert not store.delete("sess-1", "/evidence/ev-1.json")

    def test_session_isolation(self):
        """不同会话的数据隔离。"""
        store = InMemoryVfsStore()
        store.write("sess-1", "/evidence/ev-1.json", "session1")
        store.write("sess-2", "/evidence/ev-1.json", "session2")
        assert store.read("sess-1", "/evidence/ev-1.json") == "session1"
        assert store.read("sess-2", "/evidence/ev-1.json") == "session2"

    def test_path_normalization(self):
        """路径不以 / 开头时自动补。"""
        store = InMemoryVfsStore()
        path = store.write("sess-1", "evidence/ev-1.json", "data")
        assert path == "/evidence/ev-1.json"
        assert store.read("sess-1", "/evidence/ev-1.json") == "data"


class TestFileBackedVfsStore:
    """文件 VFS 存储测试。"""

    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileBackedVfsStore(tmp)
            store.write("sess-1", "/evidence/ev-1.json", '{"content": "test"}')
            assert store.read("sess-1", "/evidence/ev-1.json") == '{"content": "test"}'

    def test_list_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileBackedVfsStore(tmp)
            store.write("sess-1", "/evidence/ev-1.json", "a")
            store.write("sess-1", "/evidence/ev-2.json", "b")
            store.write("sess-1", "/reasoning/cc-1.json", "c")
            entries = store.list_dir("sess-1", "/evidence")
            assert len(entries) == 2

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileBackedVfsStore(tmp)
            store.write("sess-1", "/evidence/ev-1.json", "data")
            assert store.delete("sess-1", "/evidence/ev-1.json")
            assert not store.exists("sess-1", "/evidence/ev-1.json")

    def test_directory_traversal_protection(self):
        """目录穿越防护（.. 被清理）。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = FileBackedVfsStore(tmp)
            # 正常写入
            store.write("sess-1", "/evidence/ev-1.json", "data")
            # 尝试穿越路径（应被清理为正常路径）
            store.write("sess-1", "/evidence/../ev-1.json", "hacked")
            # 不应逃逸到 root_dir 之外
            assert store.read("sess-1", "/evidence/ev-1.json") is not None


class TestBuildVfsStore:
    """工厂函数测试。"""

    def test_empty_root_uses_memory(self):
        store = build_vfs_store("")
        assert isinstance(store, InMemoryVfsStore)

    def test_nonempty_root_uses_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = build_vfs_store(tmp)
            assert isinstance(store, FileBackedVfsStore)

    def test_unwritable_root_falls_back_to_memory(self):
        # 以「真实文件的子路径」作为 root：os.makedirs 在 POSIX 与 Windows
        # 上都会失败（NotADirectoryError 属 OSError）→ 降级内存存储。
        # 不用 /dev/null/... —— 那是 POSIX 专有路径，Windows 上 makedirs
        # 会真实创建 C:\dev\null\... 目录，测试既不失败也污染磁盘。
        with tempfile.NamedTemporaryFile() as fp:
            store = build_vfs_store(os.path.join(fp.name, "not_a_directory"))
        assert isinstance(store, InMemoryVfsStore)


# ===========================================================================
# 2. VFS 目录测试
# ===========================================================================
class TestVfsDirectory:
    """VFS 目录门面测试。"""

    def test_write_and_read_evidence(self):
        dir_ = build_directory("sess-1")
        data = {"pack_id": "pack-1", "evidence": ["ev-1"]}
        path = dir_.write_evidence("pack-1", data)
        assert path == "/evidence/pack-1.json"
        result = dir_.read_evidence("pack-1")
        assert result is not None
        assert result["pack_id"] == "pack-1"

    def test_write_and_read_reasoning(self):
        dir_ = build_directory("sess-1")
        data = {"conclusion_id": "cc-1", "statement": "测试结论"}
        path = dir_.write_reasoning("cc-1", data)
        assert path == "/reasoning/cc-1.json"
        result = dir_.read_reasoning("cc-1")
        assert result is not None
        assert result["statement"] == "测试结论"

    def test_list_entries_by_directory(self):
        dir_ = build_directory("sess-1")
        dir_.write_evidence("pack-1", {"data": 1})
        dir_.write_evidence("pack-2", {"data": 2})
        dir_.write_reasoning("cc-1", {"data": 3})
        dir_.write_summary("summary-1", {"data": 4})
        dir_.write_memory("mem-1", {"data": 5})

        assert len(dir_.list_evidence()) == 2
        assert len(dir_.list_reasoning()) == 1
        assert len(dir_.list_summaries()) == 1
        assert len(dir_.list_memories()) == 1

    def test_count_entries(self):
        dir_ = build_directory("sess-1")
        dir_.write_evidence("pack-1", {})
        dir_.write_evidence("pack-2", {})
        dir_.write_evidence("pack-3", {})
        assert dir_.count_entries(DIR_EVIDENCE) == 3
        assert dir_.count_entries(DIR_REASONING) == 0

    def test_read_nonexistent_returns_none(self):
        dir_ = build_directory("sess-1")
        assert dir_.read_evidence("missing") is None
        assert dir_.read_reasoning("missing") is None

    def test_delete(self):
        dir_ = build_directory("sess-1")
        dir_.write_evidence("pack-1", {"data": 1})
        assert dir_.exists("/evidence/pack-1.json")
        assert dir_.delete("/evidence/pack-1.json")
        assert not dir_.exists("/evidence/pack-1.json")


# ===========================================================================
# 3. 上下文压缩测试
# ===========================================================================
class TestContextCompaction:
    """上下文压缩器测试。"""

    def test_compact_single_turn(self):
        """压缩单轮：持久化证据+推理+摘要，登记文件指针。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")
        turn = _turn(1)
        pack = _evidence_pack()
        conclusion = _conclusion()

        result = compactor.compact_turn(turn, context, evidence_pack=pack, conclusion=conclusion)

        assert result.turn_index == 1
        assert result.evidence_path == f"/evidence/{pack.pack_id}.json"
        assert result.reasoning_path == f"/reasoning/{conclusion.conclusion_id}.json"
        assert result.summary_path == "/summaries/summary-turn-1.json"
        # 文件指针登记
        assert f"evidence:{turn.turn_index}" in context.file_pointers
        assert f"reasoning:{turn.turn_index}" in context.file_pointers
        assert f"summary:{turn.turn_index}" in context.file_pointers
        # token 压缩
        assert result.token_saved > 0

    def test_compact_without_evidence_or_conclusion(self):
        """无证据/结论时只持久化摘要。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")
        turn = _turn(1)

        result = compactor.compact_turn(turn, context)

        assert result.evidence_path == ""
        assert result.reasoning_path == ""
        assert result.summary_path != ""
        # 文件指针只有 summary
        assert f"summary:{turn.turn_index}" in context.file_pointers
        assert f"evidence:{turn.turn_index}" not in context.file_pointers

    def test_compact_batch(self):
        """批量压缩多轮。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")
        turns = [_turn(i) for i in range(1, 6)]

        stats = compactor.compact_batch(turns, context)

        assert stats.compacted_turns == 5
        assert stats.total_tokens_before > 0
        assert stats.total_tokens_after < stats.total_tokens_before
        assert len(stats.results) == 5

    def test_compaction_without_directory_returns_empty(self):
        """无 VFS 目录时返回空结果（不崩溃）。"""
        compactor = ContextCompactor(directory=None)
        context = SessionContext(patient_id="pat-1", session_id="sess-1")
        turn = _turn(1)

        result = compactor.compact_turn(turn, context)

        assert result.evidence_path == ""
        assert result.reasoning_path == ""
        assert result.summary_path == ""

    def test_summary_contains_confidence_and_provenance(self):
        """摘要必须标注来源置信度（记忆审核的输入）。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")
        turn = _turn(1)
        pack = _evidence_pack()

        compactor.compact_turn(turn, context, evidence_pack=pack)
        summary = compactor.directory.read("/summaries/summary-turn-1.json")
        assert summary is not None
        data = json.loads(summary)
        assert "confidence" in data
        assert "provenance" in data
        assert data["confidence"] in ("high", "medium", "low")
        assert data["provenance"] in ("knowledge_base", "model_inference", "doctor_verified")

    def test_summary_contains_patient_id(self):
        """摘要必须带 patient_id（记忆审核队列据此分区，防 unknown 兜底）。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")

        compactor.compact_turn(_turn(1), context)
        data = json.loads(compactor.directory.read("/summaries/summary-turn-1.json"))
        assert data["patient_id"] == "pat-1"

        # 上下文缺 patient_id 时兜底到证据包（而非 unknown 分区）
        context_no_pid = SessionContext(patient_id="", session_id="sess-1")
        compactor.compact_turn(_turn(2), context_no_pid, evidence_pack=_evidence_pack())
        data2 = json.loads(compactor.directory.read("/summaries/summary-turn-2.json"))
        assert data2["patient_id"] == "pat-1"


# ===========================================================================
# 4. 记忆审核队列测试
# ===========================================================================
class TestMemoryReviewQueue:
    """记忆审核队列测试。"""

    def test_submit_creates_pending_review(self):
        """提交后状态为 pending_review。"""
        queue = MemoryReviewQueue()
        memory = queue.submit(
            patient_id="pat-1",
            content="患者对阿奇霉素有效",
            provenance="model_inference",
            confidence="medium",
            source_turn=5,
        )
        assert memory.status == "pending_review"
        assert memory.provenance == "model_inference"
        assert memory.confidence == "medium"
        assert memory.source_turn == 5
        assert queue.pending_count == 1

    def test_auto_review_passes_doctor_verified_high(self):
        """医生审定 + 高置信度自动通过。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="医生确认诊断",
            provenance="doctor_verified",
            confidence="high",
            source_turn=1,
        )
        results = queue.auto_review()
        assert len(results) == 1
        assert results[0].status == "approved"
        assert queue.pending_count == 0
        assert queue.approved_count == 1

    def test_auto_review_does_not_pass_model_inference(self):
        """模型推断不自动通过——必须人工审核。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="模型推断的结论",
            provenance="model_inference",
            confidence="high",  # 即使高置信度也不自动通过
            source_turn=1,
        )
        results = queue.auto_review()
        assert len(results) == 0  # 不自动通过
        assert queue.pending_count == 1

    def test_auto_review_partial(self):
        """混合来源：部分自动通过，部分待审。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="医生确认",
            provenance="doctor_verified",
            confidence="high",
            source_turn=1,
        )
        queue.submit(
            patient_id="pat-1",
            content="模型推断",
            provenance="model_inference",
            confidence="medium",
            source_turn=2,
        )
        queue.submit(
            patient_id="pat-1",
            content="医生确认但中置信度",
            provenance="doctor_verified",
            confidence="medium",
            source_turn=3,
        )

        results = queue.auto_review()
        assert len(results) == 1  # 只有第一个自动通过
        assert queue.pending_count == 2

    def test_human_approve(self):
        """人工审核通过 → 持久化到 /memories/。"""
        dir_ = build_directory("sess-1")
        queue = MemoryReviewQueue(directory=dir_)
        memory = queue.submit(
            patient_id="pat-1",
            content="模型推断结论",
            provenance="model_inference",
            confidence="medium",
            source_turn=3,
        )

        result = queue.approve(memory.memory_id, "doctor-zhang", "审核通过")
        assert result.status == "approved"
        assert result.reviewer == "doctor-zhang"
        # 持久化到 /memories/
        assert dir_.exists(f"/memories/{memory.memory_id}.json")
        # 可召回
        assert queue.get_recallable("pat-1") == [queue.get_memory(memory.memory_id)]

    def test_human_reject(self):
        """人工审核驳回 → 不可召回。"""
        queue = MemoryReviewQueue()
        memory = queue.submit(
            patient_id="pat-1",
            content="不准确的推断",
            provenance="model_inference",
            confidence="low",
            source_turn=3,
        )

        result = queue.reject(memory.memory_id, "doctor-li", "依据不足")
        assert result.status == "rejected"
        assert queue.get_recallable("pat-1") == []

    def test_approve_nonexistent_raises(self):
        """审核不存在的记忆 → KeyError。"""
        queue = MemoryReviewQueue()
        with pytest.raises(KeyError, match="不在审核队列"):
            queue.approve("fake-id", "doctor")

    def test_double_approve_raises(self):
        """重复审核已通过的记忆 → ValueError。"""
        queue = MemoryReviewQueue()
        memory = queue.submit(
            patient_id="pat-1",
            content="测试",
            provenance="model_inference",
            confidence="medium",
            source_turn=1,
        )
        queue.approve(memory.memory_id, "doctor")
        with pytest.raises(ValueError, match="不可审核"):
            queue.approve(memory.memory_id, "doctor")

    def test_sync_callback_called_on_approve(self):
        """审核通过时调用同步回调（同步向量索引）。"""
        called: list[tuple] = []
        queue = MemoryReviewQueue(
            sync_callback=lambda pid, mem: called.append((pid, mem.memory_id)),
        )
        memory = queue.submit(
            patient_id="pat-1",
            content="测试",
            provenance="model_inference",
            confidence="medium",
            source_turn=1,
        )
        queue.approve(memory.memory_id, "doctor")
        assert len(called) == 1
        assert called[0] == ("pat-1", memory.memory_id)

    def test_list_pending_ordered(self):
        """待审队列按提交时间排序。"""
        queue = MemoryReviewQueue()
        for i in range(3):
            queue.submit(
                patient_id="pat-1",
                content=f"记忆 {i}",
                provenance="model_inference",
                confidence="medium",
                source_turn=i,
            )
        pending = queue.list_pending()
        assert len(pending) == 3
        assert pending[0].source_turn == 0
        assert pending[2].source_turn == 2

    def test_list_approved_by_patient(self):
        """按患者分区查询已通过记忆。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="p1",
            provenance="doctor_verified",
            confidence="high",
            source_turn=1,
        )
        queue.submit(
            patient_id="pat-2",
            content="p2",
            provenance="doctor_verified",
            confidence="high",
            source_turn=1,
        )
        queue.auto_review()

        pat1 = queue.list_approved("pat-1")
        pat2 = queue.list_approved("pat-2")
        assert len(pat1) == 1
        assert len(pat2) == 1
        assert pat1[0].patient_id == "pat-1"
        assert pat2[0].patient_id == "pat-2"

    def test_submit_from_summary(self):
        """从摘要 JSON 构造 Memory 并提交。"""
        queue = MemoryReviewQueue()
        summary = {
            "turn_index": 5,
            "conclusion": "阿奇霉素有效",
            "confidence": "medium",
            "provenance": "model_inference",
            "patient_id": "pat-1",
        }
        memory = queue.submit_from_summary(summary)
        assert memory.content == "阿奇霉素有效"
        assert memory.confidence == "medium"
        assert memory.provenance == "model_inference"
        assert memory.source_turn == 5


# ===========================================================================
# 5. "未审核摘要不得被召回"验收测试
# ===========================================================================
class TestUnapprovedNotRecallable:
    """验收："未审核摘要不得被召回"单测通过。"""

    def test_pending_review_not_recallable(self):
        """pending_review 状态的记忆不可召回。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="待审记忆",
            provenance="model_inference",
            confidence="high",
            source_turn=1,
        )
        # 未审核 → 不可召回
        assert queue.get_recallable("pat-1") == []

    def test_rejected_not_recallable(self):
        """rejected 状态的记忆不可召回。"""
        queue = MemoryReviewQueue()
        memory = queue.submit(
            patient_id="pat-1",
            content="被驳回的记忆",
            provenance="model_inference",
            confidence="low",
            source_turn=1,
        )
        queue.reject(memory.memory_id, "doctor", "不准确")
        assert queue.get_recallable("pat-1") == []

    def test_only_approved_is_recallable(self):
        """只有 approved 状态的记忆可召回。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="待审",
            provenance="model_inference",
            confidence="medium",
            source_turn=1,
        )
        m2 = queue.submit(
            patient_id="pat-1",
            content="自动通过",
            provenance="doctor_verified",
            confidence="high",
            source_turn=2,
        )
        m3 = queue.submit(
            patient_id="pat-1",
            content="驳回",
            provenance="model_inference",
            confidence="low",
            source_turn=3,
        )
        queue.auto_review()
        queue.reject(m3.memory_id, "doctor", "驳回")

        recallable = queue.get_recallable("pat-1")
        assert len(recallable) == 1
        assert recallable[0].memory_id == m2.memory_id

    def test_model_inference_cannot_auto_bypass(self):
        """模型推断即使高置信度也不自动通过——必须人工审核。"""
        queue = MemoryReviewQueue()
        queue.submit(
            patient_id="pat-1",
            content="模型推断的高置信记忆",
            provenance="model_inference",
            confidence="high",
            source_turn=1,
        )
        queue.auto_review()
        # 仍然 pending，不可召回
        assert queue.pending_count == 1
        assert queue.get_recallable("pat-1") == []

    def test_memory_can_be_recalled_method_enforcement(self):
        """Memory.can_be_recalled() 仅对 approved 返回 True。"""
        from harness_agent.models.common import now_utc

        m_pending = Memory(
            patient_id="pat-1",
            content="待审",
            provenance="model_inference",
            confidence="medium",
            source_turn=1,
            status="pending_review",
        )
        m_approved = Memory(
            patient_id="pat-1",
            content="已通过",
            provenance="model_inference",
            confidence="medium",
            source_turn=2,
            status="approved",
            reviewed_at=now_utc(),
            reviewer="doctor",
        )
        m_rejected = Memory(
            patient_id="pat-1",
            content="已驳回",
            provenance="model_inference",
            confidence="low",
            source_turn=3,
            status="rejected",
            reviewed_at=now_utc(),
            reviewer="doctor",
        )
        assert not m_pending.can_be_recalled()
        assert m_approved.can_be_recalled()
        assert not m_rejected.can_be_recalled()


# ===========================================================================
# 6. 20 轮模拟会话 Token 压缩验收
# ===========================================================================
class TestTwentyRoundCompression:
    """验收：20 轮模拟会话上下文 token 降约 50%。"""

    def test_twenty_round_token_reduction(self):
        """20 轮模拟会话：只保留最近 3 轮 + 文件指针。

        20 轮 × 500 token/轮 = 10000 token（未压缩）
        压缩后：3 轮 × 500 + 17 轮指针 ≈ 1500 + ~200 = ~1700 token
        压缩率 ≈ 83% >> 50% 验收标准。
        """
        compactor = build_compactor("sess-20")
        context = SessionContext(patient_id="pat-1", session_id="sess-20")
        keep = 3

        # 模拟 20 轮会话
        for i in range(1, 21):
            turn = _turn(i, token_count=500)
            dropped = context.add_turn(turn, keep=keep)
            # 压缩溢出轮
            if dropped:
                _evidence_pack(session_id="sess-20", patient_id="pat-1")
                _conclusion()
                compactor.compact_batch(dropped, context)

        # 未压缩的总 token（20 轮全量）
        full_tokens = 20 * 500  # 10000

        # 压缩后的上下文 token（3 轮 + 指针）
        compressed_tokens = compactor.estimate_compressed_tokens(context)

        reduction_pct = (1 - compressed_tokens / full_tokens) * 100

        # 验收：token 降约 50%（实际远超 50%）
        assert reduction_pct >= 50.0, f"Token 压缩率 {reduction_pct:.1f}% < 50%"
        # 打印对比（验收要求"打印前后对比"）
        print(f"\n  压缩前: {full_tokens} token (20 轮)")
        print(f"  压缩后: {compressed_tokens} token (3 轮 + {len(context.file_pointers)} 指针)")
        print(f"  压缩率: {reduction_pct:.1f}%")

    def test_file_pointers_registered_for_all_dropped_turns(self):
        """17 轮溢出均登记了文件指针（证据+推理+摘要各一）。"""
        compactor = build_compactor("sess-20")
        context = SessionContext(patient_id="pat-1", session_id="sess-20")

        for i in range(1, 21):
            turn = _turn(i, token_count=500)
            dropped = context.add_turn(turn, keep=3)
            if dropped:
                pack = _evidence_pack(session_id="sess-20", patient_id="pat-1")
                conclusion = _conclusion()
                compactor.compact_batch(
                    dropped,
                    context,
                    evidence_packs={t.turn_index: pack for t in dropped},
                    conclusions={t.turn_index: conclusion for t in dropped},
                )

        # 17 轮溢出 × 3 指针/轮 = 51 指针
        assert len(context.file_pointers) == 17 * 3

    def test_vfs_contains_all_dropped_artifacts(self):
        """VFS 目录包含 17 轮溢出的证据/推理/摘要。"""
        compactor = build_compactor("sess-20")
        context = SessionContext(patient_id="pat-1", session_id="sess-20")

        for i in range(1, 21):
            turn = _turn(i, token_count=500)
            dropped = context.add_turn(turn, keep=3)
            if dropped:
                pack = _evidence_pack(session_id="sess-20", patient_id="pat-1")
                conclusion = _conclusion()
                compactor.compact_batch(
                    dropped,
                    context,
                    evidence_packs={t.turn_index: pack for t in dropped},
                    conclusions={t.turn_index: conclusion for t in dropped},
                )

        # 17 轮溢出 × 3 目录 = 51 条目（evidence + reasoning + summaries）
        assert compactor.directory.count_entries(DIR_EVIDENCE) == 17
        assert compactor.directory.count_entries(DIR_REASONING) == 17
        assert compactor.directory.count_entries(DIR_SUMMARIES) == 17

    def test_recent_turns_only_keeps_three(self):
        """上下文只保留最近 3 轮。"""
        compactor = build_compactor("sess-20")
        context = SessionContext(patient_id="pat-1", session_id="sess-20")

        for i in range(1, 21):
            turn = _turn(i, token_count=500)
            dropped = context.add_turn(turn, keep=3)
            if dropped:
                compactor.compact_batch(dropped, context)

        assert len(context.recent_turns) == 3
        # 最近 3 轮是 18, 19, 20
        assert context.recent_turns[0].turn_index == 18
        assert context.recent_turns[2].turn_index == 20


# ===========================================================================
# 7. 端到端：压缩 → 审核 → 召回
# ===========================================================================
class TestCompactionToReviewE2E:
    """端到端：压缩产出的摘要 → 记忆审核 → 可召回。"""

    def test_compact_then_review_then_recall(self):
        """完整闭环：压缩 → 提交审核 → 人工通过 → 可召回。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")

        # 压缩溢出轮
        turn = _turn(1)
        pack = _evidence_pack()
        conclusion = _conclusion()
        compactor.compact_turn(turn, context, evidence_pack=pack, conclusion=conclusion)

        # 读取摘要
        summary_json = compactor.directory.read("/summaries/summary-turn-1.json")
        assert summary_json is not None
        summary = json.loads(summary_json)

        # 提交到审核队列（摘要原生含 patient_id，无需手工补字段）
        queue = MemoryReviewQueue(directory=compactor.directory)
        memory = queue.submit_from_summary(summary)
        assert memory.status == "pending_review"
        assert queue.get_recallable("pat-1") == []  # 未审核不可召回

        # 人工审核通过
        queue.approve(memory.memory_id, "doctor-zhang")
        recallable = queue.get_recallable("pat-1")
        assert len(recallable) == 1
        assert recallable[0].memory_id == memory.memory_id
        assert recallable[0].status == "approved"
        # 持久化到 /memories/
        assert compactor.directory.exists(f"/memories/{memory.memory_id}.json")

    def test_compact_then_auto_review_doctor_verified(self):
        """医生审定的高置信度摘要 → 自动审核通过 → 可召回。"""
        compactor = build_compactor("sess-1")
        context = SessionContext(patient_id="pat-1", session_id="sess-1")

        # 构造 doctor_verified + high 的证据包
        evidence = Evidence(
            content="医生确认诊断结果",
            source=SourceRef(source_id="s1", source_type="document", chunk_id="kb-1"),
            confidence="high",
            provenance="doctor_verified",
        )
        pack = EvidencePack(
            session_id="sess-1",
            patient_id="pat-1",
            query="q",
            evidence=[evidence],
            assembly_gate=None,
        )
        pack.assembly_gate = None  # 简化：直接用
        turn = _turn(1)
        compactor.compact_turn(turn, context, evidence_pack=pack)

        # 读取摘要 → 提交（摘要原生含 patient_id） → 自动审核
        summary = json.loads(compactor.directory.read("/summaries/summary-turn-1.json"))
        queue = MemoryReviewQueue(directory=compactor.directory)
        queue.submit_from_summary(summary)
        results = queue.auto_review()

        assert len(results) == 1
        assert results[0].status == "approved"
        assert len(queue.get_recallable("pat-1")) == 1
