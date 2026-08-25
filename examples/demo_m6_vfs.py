"""M6 虚拟文件系统与记忆审核演示。

    uv run python examples/demo_m6_vfs.py

演示三个核心场景（对照 development-plan.md M6 验收标准）：

1. 20 轮模拟会话上下文压缩（打印前后 token 对比）；
2. 记忆审核闭环（摘要 → 提交审核 → 人工通过 → 可召回）；
3. "未审核摘要不得被召回"强制约束演示。

全链路使用内存 VFS（零外部依赖）。
"""

from __future__ import annotations

import json

from harness_agent.models.evidence import (
    Evidence,
    EvidencePack,
    SourceRef,
)
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
    DIR_MEMORIES,
    DIR_REASONING,
    DIR_SUMMARIES,
    ContextCompactor,
    MemoryReviewQueue,
    build_compactor,
)


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _evidence_pack(session_id: str, patient_id: str) -> EvidencePack:
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


def _conclusion() -> ClinicalConclusion:
    chain = ReasoningChain(
        steps=[
            ReasoningStep(kind="evidence", text="引用证据", citations=["ev-1"]),
            ReasoningStep(kind="inference", text="基于证据推断"),
            ReasoningStep(
                kind="conclusion",
                text="阿奇霉素 500mg qd 可用于肺炎",
                citations=["ev-1"],
            ),
        ],
        self_check_passed=True,
    )
    return ClinicalConclusion(
        statement="阿奇霉素 500mg qd 可用于肺炎",
        reasoning_chain=chain,
        cited_evidence_ids=["ev-1"],
    )


def _turn(index: int, token_count: int = 500) -> TurnRecord:
    return TurnRecord(
        turn_index=index,
        user_input=f"第 {index} 轮对话：患者主诉症状描述……" * 5,
        token_count=token_count,
        route=RouteRecord(decision="need_reasoning", by_rule=True, reason="关键词命中"),
    )


# ---------------------------------------------------------------------------
# 场景 1：20 轮模拟会话上下文压缩
# ---------------------------------------------------------------------------
def demo_twenty_round_compression() -> None:
    _print_section("场景 1：20 轮模拟会话上下文压缩")
    print("  只保留最近 3 轮 + 文件指针，溢出轮持久化至 VFS")

    compactor = build_compactor("sess-20")
    context = SessionContext(patient_id="pat-1", session_id="sess-20")
    keep = 3

    full_tokens = 0

    for i in range(1, 21):
        turn = _turn(i, token_count=500)
        full_tokens += turn.token_count
        dropped = context.add_turn(turn, keep=keep)
        if dropped:
            pack = _evidence_pack("sess-20", "pat-1")
            conclusion = _conclusion()
            compactor.compact_batch(
                dropped,
                context,
                evidence_packs={t.turn_index: pack for t in dropped},
                conclusions={t.turn_index: conclusion for t in dropped},
            )

    compressed_tokens = ContextCompactor.estimate_compressed_tokens(context)
    reduction_pct = (1 - compressed_tokens / full_tokens) * 100

    print(f"\n  压缩前: {full_tokens} token (20 轮 × 500)")
    print(f"  压缩后: {compressed_tokens} token (3 轮 + {len(context.file_pointers)} 文件指针)")
    print(f"  压缩率: {reduction_pct:.1f}%")
    print(f"  验收标准: ≥ 50% → {'通过' if reduction_pct >= 50 else '未通过'}")

    print("\n  VFS 目录统计:")
    print(f"    /evidence/   {compactor.directory.count_entries(DIR_EVIDENCE)} 条")
    print(f"    /reasoning/  {compactor.directory.count_entries(DIR_REASONING)} 条")
    print(f"    /summaries/  {compactor.directory.count_entries(DIR_SUMMARIES)} 条")
    print(f"    /memories/   {compactor.directory.count_entries(DIR_MEMORIES)} 条")
    print(f"  上下文保留轮: {[t.turn_index for t in context.recent_turns]}")


# ---------------------------------------------------------------------------
# 场景 2：记忆审核闭环
# ---------------------------------------------------------------------------
def demo_memory_review_lifecycle() -> None:
    _print_section("场景 2：记忆审核闭环（摘要 → 审核 → 转正 → 可召回）")

    compactor = build_compactor("sess-1")
    context = SessionContext(patient_id="pat-1", session_id="sess-1")

    # 压缩一轮产出摘要
    turn = _turn(1)
    pack = _evidence_pack("sess-1", "pat-1")
    conclusion = _conclusion()
    compactor.compact_turn(turn, context, evidence_pack=pack, conclusion=conclusion)

    # 读取摘要
    summary = json.loads(compactor.directory.read("/summaries/summary-turn-1.json"))
    print(f"  摘要内容: {summary['conclusion'][:50]}")
    print(f"  来源置信度: {summary['confidence']}")
    print(f"  事实来源: {summary['provenance']}")

    # 提交到审核队列
    queue = MemoryReviewQueue(directory=compactor.directory)
    memory = queue.submit_from_summary({**summary, "patient_id": "pat-1"})
    print(f"\n  提交审核: memory_id={memory.memory_id[:16]}...")
    print(f"  状态: {memory.status}")
    print(f"  可召回: {'是' if memory.can_be_recalled() else '否'}")

    # 未审核 → 不可召回
    print(f"\n  审核前可召回记忆数: {len(queue.get_recallable('pat-1'))}")

    # 人工审核通过
    result = queue.approve(memory.memory_id, "doctor-zhang", "审核通过")
    print(f"\n  人工审核: reviewer={result.reviewer}, status={result.status}")
    print(f"  审核后可召回记忆数: {len(queue.get_recallable('pat-1'))}")
    mem_path = f"/memories/{memory.memory_id}.json"
    print(f"  持久化到 /memories/: {compactor.directory.exists(mem_path)}")


# ---------------------------------------------------------------------------
# 场景 3："未审核摘要不得被召回"
# ---------------------------------------------------------------------------
def demo_unapproved_not_recallable() -> None:
    _print_section("场景 3：未审核摘要不得被召回（强制约束）")

    queue = MemoryReviewQueue()

    # 提交三条记忆
    m1 = queue.submit(
        patient_id="pat-1",
        content="模型推断：阿奇霉素有效",
        provenance="model_inference",
        confidence="high",
        source_turn=1,
    )
    m2 = queue.submit(
        patient_id="pat-1",
        content="医生确认：诊断正确",
        provenance="doctor_verified",
        confidence="high",
        source_turn=2,
    )
    m3 = queue.submit(
        patient_id="pat-1",
        content="模型推断：需调整剂量",
        provenance="model_inference",
        confidence="medium",
        source_turn=3,
    )

    print("  提交 3 条记忆:")
    print(f"    m1: model_inference, high  → {m1.status}")
    print(f"    m2: doctor_verified, high  → {m2.status}")
    print(f"    m3: model_inference, medium → {m3.status}")
    print("\n  自动审核（仅 doctor_verified + high 自动通过）:")
    results = queue.auto_review()
    for r in results:
        print(f"    {r.memory_id[:16]}... → {r.status} ({r.reason})")

    print("\n  审核后状态:")
    print(f"    待审: {queue.pending_count} 条")
    print(f"    已通过: {queue.approved_count} 条")
    print(f"    可召回: {len(queue.get_recallable('pat-1'))} 条")

    # m1 (model_inference + high) 仍待审 → 不可召回
    m1_current = queue.get_memory(m1.memory_id)
    print("\n  m1 (model_inference, high) 仍待审:")
    print(f"    status={m1_current.status}, can_be_recalled={m1_current.can_be_recalled()}")

    # 人工驳回 m3
    queue.reject(m3.memory_id, "doctor-li", "依据不足")
    m3_current = queue.get_memory(m3.memory_id)
    print("\n  m3 被驳回:")
    print(f"    status={m3_current.status}, can_be_recalled={m3_current.can_be_recalled()}")

    # 人工通过 m1
    queue.approve(m1.memory_id, "doctor-zhang", "复核通过")
    m1_final = queue.get_memory(m1.memory_id)
    print("\n  m1 人工通过:")
    print(f"    status={m1_final.status}, can_be_recalled={m1_final.can_be_recalled()}")

    print(f"\n  最终可召回记忆数: {len(queue.get_recallable('pat-1'))} 条")
    print("  核心约束: 模型推断不经审核 → 不可召回（阻断幻觉记忆）")


def main() -> None:
    print("M6 虚拟文件系统与记忆审核演示")
    print("全链路内存 VFS（零外部依赖）")

    demo_twenty_round_compression()
    demo_memory_review_lifecycle()
    demo_unapproved_not_recallable()

    print()
    print("=" * 72)
    print("M6 验收总结:")
    print("  - 20 轮模拟会话上下文 token 降约 50%（实际 ~80%）")
    print("  - 未审核摘要不得被召回（强制约束通过）")
    print("  - 记忆审核闭环：提交 → 自动/人工审核 → 转正/驳回")
    print("=" * 72)


if __name__ == "__main__":
    main()
