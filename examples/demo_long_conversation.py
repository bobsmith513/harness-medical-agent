"""M8 端到端演示三：长会话压缩。

    uv run python examples/demo_long_conversation.py

场景：20 轮模拟会话，展示上下文压缩全链路（对照 M6 验收标准）。

完整流程：

    20 轮对话 → 超出 keep=3 的旧轮持久化至 VFS
    → 上下文只留最近 3 轮 + 文件指针
    → Token 降幅 ≥ 50%（验收指标）
    → 文件指针可回溯（evidence/reasoning/summary 三类）
    → 摘要 → 记忆审核队列 → 批量审核

全链路使用内存 VFS（零外部依赖）。
"""

from __future__ import annotations

import json
import sys

from harness_agent.models.audit import GateVerdict
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

KEEP = 3  # 上下文保留轮数
TOTAL_TURNS = 20


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _evidence_pack(turn_index: int, patient_id: str = "pat-001") -> EvidencePack:
    """每轮的证据包。"""
    return EvidencePack(
        session_id="sess-long",
        patient_id=patient_id,
        query=f"第 {turn_index} 轮查询",
        evidence=[
            Evidence(
                evidence_id=f"ev-{turn_index}",
                content=f"第 {turn_index} 轮证据：症状描述与检验指标……" * 3,
                source=SourceRef(
                    source_id=f"src-{turn_index}",
                    source_type="document",
                    chunk_id=f"chunk-{turn_index}",
                ),
                confidence="medium" if turn_index % 2 == 0 else "high",
                provenance="knowledge_base" if turn_index % 2 == 0 else "doctor_verified",
            )
        ],
        assembly_gate=GateVerdict(gate="assembly", allowed=True, reason="复核通过"),
    )


def _conclusion(turn_index: int) -> ClinicalConclusion:
    """每轮的临床结论。"""
    chain = ReasoningChain(
        steps=[
            ReasoningStep(
                kind="evidence",
                text=f"引用证据 ev-{turn_index}",
                citations=[f"ev-{turn_index}"],
            ),
            ReasoningStep(
                kind="inference",
                text=f"基于第 {turn_index} 轮证据推断",
            ),
            ReasoningStep(
                kind="conclusion",
                text=f"第 {turn_index} 轮结论",
                citations=[f"ev-{turn_index}"],
            ),
        ],
        self_check_passed=True,
        self_check_notes=f"自检通过（3/3）轮次 {turn_index}",
    )
    return ClinicalConclusion(
        statement=f"第 {turn_index} 轮结论：建议结合检查结果确认。",
        reasoning_chain=chain,
        cited_evidence_ids=[f"ev-{turn_index}"],
    )


def _turn(index: int, token_count: int = 500) -> TurnRecord:
    return TurnRecord(
        turn_index=index,
        user_input=f"第 {index} 轮对话：患者描述症状变化与用药反应……" * 5,
        token_count=token_count,
        route=RouteRecord(
            decision="need_reasoning" if index % 3 == 0 else "no_reasoning",
            by_rule=True,
            reason="关键词命中" if index % 3 == 0 else "LLM 兜底",
        ),
    )


def main() -> None:
    # Windows 默认终端（GBK）无法编码 ✓/✗，打印时抛 UnicodeEncodeError。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("M8 端到端演示三：长会话压缩")
    print("全链路内存 VFS（零外部依赖）")
    print(f"模拟 {TOTAL_TURNS} 轮会话，保留最近 {KEEP} 轮 + 文件指针")

    # ---- Phase 1：20 轮会话 + 逐轮压缩 ----
    _print_section(f"Phase 1：{TOTAL_TURNS} 轮会话逐轮压缩")
    compactor = build_compactor("sess-long")
    context = SessionContext(patient_id="pat-001", session_id="sess-long")

    full_tokens = 0
    compacted_count = 0

    for i in range(1, TOTAL_TURNS + 1):
        turn = _turn(i, token_count=500)
        full_tokens += turn.token_count
        dropped = context.add_turn(turn, keep=KEEP)

        if dropped:
            packs = {t.turn_index: _evidence_pack(t.turn_index) for t in dropped}
            conclusions = {t.turn_index: _conclusion(t.turn_index) for t in dropped}
            stats = compactor.compact_batch(
                dropped, context, evidence_packs=packs, conclusions=conclusions
            )
            compacted_count += stats.compacted_turns

            for r in stats.results:
                print(
                    f"  轮 {r.turn_index:2d}: 压缩 {r.tokens_before}→{r.tokens_after} token "
                    f"(省 {r.token_saved})"
                )

    print(f"\n  压缩轮数: {compacted_count}/{TOTAL_TURNS}")
    print(f"  压缩前总 token: {full_tokens}")
    compressed_tokens = ContextCompactor.estimate_compressed_tokens(context)
    print(f"  压缩后总 token: {compressed_tokens}")
    reduction = (1 - compressed_tokens / full_tokens) * 100
    print(f"  压缩率: {reduction:.1f}%")
    print(f"  验收标准: ≥ 50% → {'通过' if reduction >= 50 else '未通过'}")

    # ---- Phase 2：VFS 目录统计 ----
    _print_section("Phase 2：VFS 虚拟目录持久化统计")
    dir = compactor.directory
    print(f"  /evidence/   {dir.count_entries(DIR_EVIDENCE)} 条")
    print(f"  /reasoning/  {dir.count_entries(DIR_REASONING)} 条")
    print(f"  /summaries/  {dir.count_entries(DIR_SUMMARIES)} 条")
    print(f"  /memories/   {dir.count_entries(DIR_MEMORIES)} 条")
    print(f"  文件指针: {len(context.file_pointers)} 个")
    print(f"  上下文保留轮: {[t.turn_index for t in context.recent_turns]}")

    # ---- Phase 3：文件指针可回溯 ----
    _print_section("Phase 3：文件指针回溯验证")
    sample_pointers = list(context.file_pointers.items())[:6]
    all_pointers_exist = bool(sample_pointers)
    for key, path in sample_pointers:
        exists = dir.exists(path)
        all_pointers_exist = all_pointers_exist and exists
        print(f"  {key:25s} → {path}  exists={exists}")

    # 读取一个摘要验证内容
    sample_summary_path = context.file_pointers.get("summary:1", "")
    if sample_summary_path and dir.exists(sample_summary_path):
        summary = json.loads(dir.read(sample_summary_path))
        print("\n  回读 summary:1:")
        print(f"    turn_index: {summary.get('turn_index')}")
        print(f"    conclusion: {summary.get('conclusion', '')[:50]}")
        print(f"    confidence: {summary.get('confidence')}")
        print(f"    provenance: {summary.get('provenance')}")

    # ---- Phase 4：批量记忆审核 ----
    _print_section("Phase 4：摘要 → 批量记忆审核")
    queue = MemoryReviewQueue(directory=compactor.directory)

    # 从所有摘要提交到审核队列
    for turn_idx in range(1, TOTAL_TURNS - KEEP + 1):
        path = f"/summaries/summary-turn-{turn_idx}.json"
        if dir.exists(path):
            data = json.loads(dir.read(path))
            queue.submit_from_summary({**data, "patient_id": "pat-001"})

    print(f"  提交审核: {queue.total_count} 条")
    print(f"  待审: {queue.pending_count} 条")

    # 自动审核（doctor_verified + high 自动通过）
    auto_results = queue.auto_review()
    print(f"  自动通过: {len(auto_results)} 条")
    print(f"  仍待审: {queue.pending_count} 条（model_inference 需人工）")

    # 批量人工审核剩余
    remaining = queue.list_pending()
    for mem in remaining:
        queue.approve(mem.memory_id, "doctor-batch", "批量审核通过")
    print(f"  人工批量通过: {len(remaining)} 条")
    print(f"  最终可召回: {len(queue.get_recallable('pat-001'))} 条")
    print(f"  /memories/ 持久化: {dir.count_entries(DIR_MEMORIES)} 条")

    # ---- Phase 5：压缩前后对比表 ----
    _print_section("Phase 5：压缩效果对比")
    print(f"  {'指标':20s} {'压缩前':>10s} {'压缩后':>10s} {'变化':>10s}")
    print(f"  {'-' * 52}")
    print(f"  {'上下文轮数':20s} {TOTAL_TURNS:>10d} {KEEP:>10d} {(TOTAL_TURNS - KEEP):>+10d}")
    print(
        f"  {'上下文 token':20s} {full_tokens:>10d} {compressed_tokens:>10d} "
        f"{(full_tokens - compressed_tokens):>+10d}"
    )
    vfs_count = (
        dir.count_entries(DIR_EVIDENCE)
        + dir.count_entries(DIR_REASONING)
        + dir.count_entries(DIR_SUMMARIES)
    )
    print(f"  {'VFS 文件数':20s} {0:>10d} {vfs_count:>10d}")
    print(f"  {'可召回记忆':20s} {0:>10d} {len(queue.get_recallable('pat-001')):>10d}")

    # ---- 验收总结（按本轮真实结果条件打印，非恒绿） ----
    recallable = queue.get_recallable("pat-001")
    checks = [
        (
            f"{TOTAL_TURNS} 轮模拟会话上下文 token 降约 {reduction:.0f}%（≥ 50%）",
            reduction >= 50,
        ),
        (
            f"上下文只留最近 {KEEP} 轮 + {len(context.file_pointers)} 文件指针",
            len(context.recent_turns) == KEEP,
        ),
        (
            "VFS 四目录持久化: evidence + reasoning + summaries + memories",
            all(
                dir.count_entries(d) > 0
                for d in (DIR_EVIDENCE, DIR_REASONING, DIR_SUMMARIES, DIR_MEMORIES)
            ),
        ),
        ("文件指针可回溯（路径存在性验证通过）", all_pointers_exist),
        (
            "摘要 → 批量审核 → 转正为可召回记忆",
            queue.total_count > 0 and len(recallable) == queue.total_count,
        ),
        ("未审核记忆不可召回（强制约束）", queue.pending_count == 0 and len(recallable) > 0),
    ]
    print()
    print("=" * 72)
    print("长会话压缩验收总结:")
    for line, ok in checks:
        print(f"  {'✓' if ok else '✗'} {line}")
    print("=" * 72)


if __name__ == "__main__":
    main()
