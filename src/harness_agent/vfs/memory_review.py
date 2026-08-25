"""记忆审核队列（M6）：摘要 → 抽样审核 → 通过转正 → 同步索引。

审核闭环（对应 development-plan.md 第三节记忆审核流）：

1. **提交**：会话摘要（``/summaries/``）标注来源置信度后提交到审核队列，
   Memory 状态从 ``session_pointer`` → ``pending_review``；
2. **自动审核**：高置信度（``doctor_verified`` + ``high``）自动通过，
   ``model_inference`` 必须人工审核（阻断模型推断固化为检索事实）；
3. **人工审核**：``approve`` → ``approved``（持久化到 ``/memories/``，
   可选同步向量索引）；``reject`` → ``rejected``（不可召回）；
4. **强制约束**：未审核记忆仅作会话内指针，绝不进入检索结果
   （``Memory.can_be_recalled()`` 返回 False）。

**设计语义**：模型推断不得直接固化为可召回事实——必须经审核
转正，这是阻断幻觉记忆的核心安全门。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from harness_agent.models.common import ConfidenceLevel, Provenance, now_utc
from harness_agent.models.memory import Memory, MemoryStatus
from harness_agent.vfs.directory import VfsDirectory

__all__ = [
    "MemoryReviewQueue",
    "ReviewResult",
]

#: 审核通过后同步向量索引的回调签名（patient_id, memory → None）
SyncCallback = Callable[[str, Memory], None]


@dataclass(frozen=True)
class ReviewResult:
    """单次审核结果。"""

    memory_id: str
    status: MemoryStatus
    reviewer: str
    reason: str = ""


@dataclass
class MemoryReviewQueue:
    """记忆审核队列：提交 → 审核 → 转正/驳回。

    参数：
        directory: VFS 目录门面（审核通过后持久化到 ``/memories/``）
        sync_callback: 审核通过后的向量索引同步回调（可选，
            M3 检索层的 ``index_memory`` 方法注入）
    """

    directory: VfsDirectory | None = None
    sync_callback: SyncCallback | None = None
    # memory_id → Memory（全生命周期追踪）
    _memories: dict[str, Memory] = field(default_factory=dict)
    # 按时间序排列的待审 memory_id 队列
    _pending: list[str] = field(default_factory=list)

    # ---- 提交 ----

    def submit(
        self,
        *,
        patient_id: str,
        content: str,
        provenance: Provenance,
        confidence: ConfidenceLevel,
        source_turn: int,
    ) -> Memory:
        """提交摘要到审核队列（``session_pointer`` → ``pending_review``）。

        模型推断（``model_inference``）必须人工审核；
        医生审定（``doctor_verified``）+ 高置信度可自动通过。
        """
        memory = Memory(
            patient_id=patient_id,
            content=content,
            status="pending_review",
            provenance=provenance,
            confidence=confidence,
            source_turn=source_turn,
        )
        self._memories[memory.memory_id] = memory
        self._pending.append(memory.memory_id)
        return memory

    def submit_from_summary(self, summary: dict) -> Memory:
        """从 ``/summaries/`` 的摘要 JSON 构造 Memory 并提交。"""
        return self.submit(
            patient_id=summary.get("patient_id", "unknown"),
            content=summary.get("conclusion") or summary.get("evidence_summary", ""),
            provenance=summary.get("provenance", "model_inference"),
            confidence=summary.get("confidence", "low"),
            source_turn=summary.get("turn_index", 0),
        )

    # ---- 自动审核 ----

    def auto_review(self) -> list[ReviewResult]:
        """自动审核高置信度记忆（``doctor_verified`` + ``high``）。

        模型推断（``model_inference``）不自动通过——必须人工审核，
        这是阻断模型推断固化为检索事实的核心约束。
        """
        results: list[ReviewResult] = []
        still_pending: list[str] = []

        for mid in self._pending:
            memory = self._memories[mid]
            if memory.provenance == "doctor_verified" and memory.confidence == "high":
                result = self._do_approve(mid, "auto-review", "高置信度自动通过")
                results.append(result)
            else:
                still_pending.append(mid)

        self._pending = still_pending
        return results

    # ---- 人工审核 ----

    def approve(self, memory_id: str, reviewer: str, reason: str = "") -> ReviewResult:
        """人工审核通过 → 持久化到 ``/memories/`` + 同步索引。"""
        if memory_id not in self._memories:
            raise KeyError(f"记忆 {memory_id} 不在审核队列中")
        memory = self._memories[memory_id]
        if memory.status not in ("pending_review", "session_pointer"):
            raise ValueError(f"记忆 {memory_id} 状态为 {memory.status}，不可审核")

        result = self._do_approve(memory_id, reviewer, reason or "人工审核通过")
        if memory_id in self._pending:
            self._pending.remove(memory_id)
        return result

    def reject(self, memory_id: str, reviewer: str, reason: str = "") -> ReviewResult:
        """人工审核驳回 → 不可召回。"""
        if memory_id not in self._memories:
            raise KeyError(f"记忆 {memory_id} 不在审核队列中")
        memory = self._memories[memory_id]
        if memory.status not in ("pending_review", "session_pointer"):
            raise ValueError(f"记忆 {memory_id} 状态为 {memory.status}，不可审核")

        rejected = memory.model_copy(
            update={
                "status": "rejected",
                "reviewed_at": now_utc(),
                "reviewer": reviewer,
            }
        )
        self._memories[memory_id] = rejected
        if memory_id in self._pending:
            self._pending.remove(memory_id)
        return ReviewResult(
            memory_id=memory_id,
            status="rejected",
            reviewer=reviewer,
            reason=reason or "人工审核驳回",
        )

    # ---- 查询 ----

    def list_pending(self) -> list[Memory]:
        """列出待审记忆（按提交时间序）。"""
        return [self._memories[mid] for mid in self._pending]

    def list_approved(self, patient_id: str | None = None) -> list[Memory]:
        """列出已通过的记忆（可召回）。"""
        approved = [m for m in self._memories.values() if m.status == "approved"]
        if patient_id is not None:
            approved = [m for m in approved if m.patient_id == patient_id]
        return approved

    def list_rejected(self, patient_id: str | None = None) -> list[Memory]:
        """列出已驳回的记忆。"""
        rejected = [m for m in self._memories.values() if m.status == "rejected"]
        if patient_id is not None:
            rejected = [m for m in rejected if m.patient_id == patient_id]
        return rejected

    def get_recallable(self, patient_id: str) -> list[Memory]:
        """获取可召回的记忆（仅 ``approved``，按 patient_id 分区隔离）。

        未审核记忆仅作会话内指针，绝不进入检索结果。
        """
        return [
            m for m in self._memories.values() if m.can_be_recalled() and m.patient_id == patient_id
        ]

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._memories.get(memory_id)

    @property
    def total_count(self) -> int:
        return len(self._memories)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def approved_count(self) -> int:
        return sum(1 for m in self._memories.values() if m.status == "approved")

    # ---- 内部 ----

    def _do_approve(self, memory_id: str, reviewer: str, reason: str) -> ReviewResult:
        """执行审核通过：更新状态 + 持久化 + 同步索引。"""
        memory = self._memories[memory_id]
        approved = memory.model_copy(
            update={
                "status": "approved",
                "reviewed_at": now_utc(),
                "reviewer": reviewer,
            }
        )
        self._memories[memory_id] = approved

        # 持久化到 /memories/
        if self.directory is not None:
            self.directory.write_memory(memory_id, approved.model_dump())

        # 同步向量索引（注入回调）
        if self.sync_callback is not None:
            self.sync_callback(approved.patient_id, approved)

        return ReviewResult(
            memory_id=memory_id,
            status="approved",
            reviewer=reviewer,
            reason=reason,
        )
