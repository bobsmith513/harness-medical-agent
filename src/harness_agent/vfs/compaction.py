"""上下文压缩（M6）：溢出轮持久化至 VFS，上下文只留最近 3 轮 + 文件指针。

压缩流程（对应 ``SessionContext.add_turn`` 返回的溢出轮）：

1. 证据包快照 → ``/evidence/<evidence_id>.json``
2. 推理链 + 结论快照 → ``/reasoning/<conclusion_id>.json``
3. 旧轮摘要（含来源置信度标注）→ ``/summaries/<summary_id>.json``
4. 上下文 ``file_pointers`` 登记三个文件指针
5. Token 预算更新：移除溢出轮 token，仅留指针引用

**压缩率口径**：验收下限是"20 轮模拟会话上下文 token 降幅 ≥ 50%"，
零依赖 demo 的合成输入下实测约 81%。**它不是基准测试结果**——
每轮 token 数与压缩后的估算都是 demo 内的假设值，未在真实
tokenizer 与真实 LLM 计费场景下实测。口径见
``docs/design-decisions.md`` 第 2 节。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion
from harness_agent.models.session import SessionContext, TurnRecord
from harness_agent.vfs.directory import VfsDirectory
from harness_agent.vfs.store import build_vfs_store

__all__ = [
    "CompactionResult",
    "CompactionStats",
    "ContextCompactor",
]

#: 摘要的最小 token 估算系数（中文按字符数估算，英文按词数 * 1.3）
_TOKEN_CHARS_PER_UNIT = 1.5


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文按字符、英文按词混合口径，够审计展示用）。"""
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars
    # 中文约 1 token/字，英文约 1 token/4 字符（0.25 * other_chars）
    return int(chinese_chars + other_chars * 0.25)


@dataclass(frozen=True)
class CompactionResult:
    """单轮压缩结果：持久化的文件指针与 token 估算。"""

    turn_index: int
    evidence_path: str = ""
    reasoning_path: str = ""
    summary_path: str = ""
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def token_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


@dataclass
class CompactionStats:
    """会话级压缩统计（验收指标用）。"""

    total_turns: int = 0
    compacted_turns: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    results: list[CompactionResult] = field(default_factory=list)

    @property
    def token_reduction_pct(self) -> float:
        """Token 压缩百分比（验收指标：≥ 50%）。"""
        if self.total_tokens_before == 0:
            return 0.0
        return (1 - self.total_tokens_after / self.total_tokens_before) * 100

    @property
    def tokens_saved(self) -> int:
        return max(0, self.total_tokens_before - self.total_tokens_after)


class ContextCompactor:
    """上下文压缩器：溢出轮 → VFS 持久化 + 文件指针登记。

    与 ``SessionContext.add_turn`` 配合使用：
    - ``add_turn`` 返回被移出上下文的旧轮列表；
    - 本类接收旧轮列表，持久化至 VFS 并登记文件指针；
    - 上下文只留最近 ``keep`` 轮 + 文件指针，长会话 Token 降幅
      验收下限 ≥ 50%（压缩率口径见模块 docstring）。
    """

    def __init__(self, directory: VfsDirectory | None = None) -> None:
        self._directory = directory

    @property
    def directory(self) -> VfsDirectory | None:
        return self._directory

    def compact_turn(
        self,
        turn: TurnRecord,
        context: SessionContext,
        evidence_pack: EvidencePack | None = None,
        conclusion: ClinicalConclusion | None = None,
    ) -> CompactionResult:
        """压缩单轮：持久化证据/推理/摘要，登记文件指针。

        参数：
            turn: 被移出上下文的旧轮记录
            context: 当前会话上下文（登记文件指针到此）
            evidence_pack: 该轮的证据包（持久化到 /evidence/）
            conclusion: 该轮的临床结论（持久化到 /reasoning/）

        返回压缩结果（含文件指针与 token 估算）。
        """
        if self._directory is None:
            return CompactionResult(turn_index=turn.turn_index)

        tokens_before = turn.token_count or _estimate_tokens(turn.user_input)

        evidence_path = ""
        reasoning_path = ""
        summary_path = ""

        # 1. 证据包快照
        if evidence_pack is not None:
            pack_id = evidence_pack.pack_id
            evidence_path = self._directory.write_evidence(pack_id, evidence_pack.model_dump())
            context.file_pointers[f"evidence:{turn.turn_index}"] = evidence_path

        # 2. 推理链 + 结论快照
        if conclusion is not None:
            reasoning_path = self._directory.write_reasoning(
                conclusion.conclusion_id, conclusion.model_dump()
            )
            context.file_pointers[f"reasoning:{turn.turn_index}"] = reasoning_path

        # 3. 旧轮摘要（标注患者分区 + 来源置信度，供记忆审核队列消费）
        summary_data = self._build_summary(
            turn, evidence_pack, conclusion, patient_id=context.patient_id
        )
        summary_id = f"summary-turn-{turn.turn_index}"
        summary_path = self._directory.write_summary(summary_id, summary_data)
        context.file_pointers[f"summary:{turn.turn_index}"] = summary_path

        # 4. 更新 token 预算
        # 文件指针约占 50 token（逻辑路径字符串），远小于完整轮
        pointer_tokens = _estimate_tokens(evidence_path + reasoning_path + summary_path)
        tokens_after = min(pointer_tokens, tokens_before)

        context.token_budget_used = context.token_budget_used - tokens_before + tokens_after

        return CompactionResult(
            turn_index=turn.turn_index,
            evidence_path=evidence_path,
            reasoning_path=reasoning_path,
            summary_path=summary_path,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def compact_batch(
        self,
        turns: list[TurnRecord],
        context: SessionContext,
        evidence_packs: dict[int, EvidencePack] | None = None,
        conclusions: dict[int, ClinicalConclusion] | None = None,
    ) -> CompactionStats:
        """批量压缩溢出轮（``SessionContext.add_turn`` 返回值）。"""
        stats = CompactionStats()
        evidence_packs = evidence_packs or {}
        conclusions = conclusions or {}

        for turn in turns:
            result = self.compact_turn(
                turn,
                context,
                evidence_pack=evidence_packs.get(turn.turn_index),
                conclusion=conclusions.get(turn.turn_index),
            )
            stats.results.append(result)
            stats.compacted_turns += 1
            stats.total_tokens_before += result.tokens_before
            stats.total_tokens_after += result.tokens_after

        return stats

    @staticmethod
    def _build_summary(
        turn: TurnRecord,
        evidence_pack: EvidencePack | None,
        conclusion: ClinicalConclusion | None,
        patient_id: str = "",
    ) -> dict:
        """构造旧轮摘要（标注患者分区 + 来源置信度，供记忆审核消费）。

        ``patient_id`` 优先取会话上下文，兜底证据包——摘要缺此字段时
        ``MemoryReviewQueue.submit_from_summary`` 会落到 ``unknown``
        分区，记忆转正后无法按患者隔离召回。
        """
        resolved_patient = patient_id or (
            evidence_pack.patient_id if evidence_pack is not None else ""
        )
        evidence_summary = ""
        confidence = "low"
        provenance = "model_inference"
        if evidence_pack is not None and evidence_pack.evidence:
            evidence_summary = "; ".join(e.content[:100] for e in evidence_pack.evidence[:3])
            # 取最高置信度
            confidences = [e.confidence for e in evidence_pack.evidence]
            if "high" in confidences:
                confidence = "high"
            elif "medium" in confidences:
                confidence = "medium"
            provenance = evidence_pack.evidence[0].provenance

        conclusion_statement = ""
        if conclusion is not None:
            conclusion_statement = conclusion.statement

        return {
            "turn_index": turn.turn_index,
            "patient_id": resolved_patient or "unknown",
            "user_input": turn.user_input[:200],
            "route": turn.route.model_dump() if turn.route else None,
            "evidence_summary": evidence_summary,
            "conclusion": conclusion_statement,
            "confidence": confidence,
            "provenance": provenance,
            "escalated": turn.escalated_to_human,
        }

    @staticmethod
    def estimate_full_context_tokens(context: SessionContext) -> int:
        """估算未压缩时的全量上下文 token（含全部轮 + 证据 + 推理）。

        **调用时机警告**：必须在 ``SessionContext.add_turn`` 截断
        ``recent_turns`` **之前**调用。``add_turn`` 会就地丢弃超出
        keep 的旧轮，压缩之后再调本方法只能看到剩下的 ``keep`` 轮，
        与 ``estimate_compressed_tokens`` 的遍历范围完全相同——
        两者相减会得到约 0% 的压缩率。真实的全量基数要由调用方在
        逐轮 add_turn 时自行累加（见 ``examples/demo_long_conversation.py``）。
        """
        total = 0
        for turn in context.recent_turns:
            total += turn.token_count or _estimate_tokens(turn.user_input)
        # 文件指针本身也占少量 token
        for ptr in context.file_pointers.values():
            total += _estimate_tokens(ptr)
        return total

    @staticmethod
    def estimate_compressed_tokens(context: SessionContext, keep: int = 3) -> int:
        """估算压缩后的上下文 token（仅最近 ``keep`` 轮 + 指针引用）。

        与 ``estimate_full_context_tokens`` 的区别就在轮数截断：
        溢出轮已持久化至 VFS，上下文只保留最近 ``keep`` 轮原文 +
        指向持久化文件的指针（指针开销远小于整轮原文）。
        """
        total = 0
        for turn in context.recent_turns[-keep:]:
            total += turn.token_count or _estimate_tokens(turn.user_input)
        # 文件指针约占完整轮的 5-10%
        for ptr in context.file_pointers.values():
            total += _estimate_tokens(ptr)
        return total


def build_compactor(session_id: str, root_dir: str = "") -> ContextCompactor:
    """装配上下文压缩器（零依赖默认内存 VFS）。"""
    directory = VfsDirectory(store=build_vfs_store(root_dir), session_id=session_id)
    return ContextCompactor(directory=directory)
