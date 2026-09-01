"""M8 端到端演示一：初诊推理全链路。

    uv run python examples/demo_first_diagnosis.py

完整链路（对照 development-plan.md M8 验收标准）：

    用户输入（含身份证 / 手机号等患者标识）
      → 脱敏中间件（去除患者标识，脱敏产物即编排链路的查询）
      → 路由器：需要临床推理（关键词命中"用药"）
      → 检索供给层（输入闸门通过 → 知识库双路召回 → 装配闸门复核）
      → 推理专家（证据引用 → 逐步推断 → 结论 + 自检）
      → 质量门禁（LLM-judge 忠实度校验）
      → 输出闸门（药物安全全文扫描，阿奇霉素不在阻断列表）
      → 临床结论输出

患者 pat-001 已知青霉素过敏（M2 种子数据，安全栈持有），
查询文本（脱敏产物）刻意不提及过敏药名——输入闸门不做意图
识别，检出过敏药名即拦截转人工（fail-closed）；过敏史由安全栈
的硬规则精确匹配保障（非向量检索）。

**哪些是真跑的、哪些是脚本**（避免把演示当成证明）：

- **真实执行**：脱敏、路由、检索供给层（`HybridRetrievalService`：
  输入闸门 → 哈希嵌入 + BM25 双路召回 → RRF 融合 → identity 精排
  → 同父补全 → 装配闸门）、推理专家自检、LLM-judge 解析、输出闸门
  药物扫描、VFS/审计落盘、全链路 trace；
- **脚本提供**：三处 LLM 应答的文本（推理链 / judge 打分 / 路由兜底
  未触发）。门禁拿到的是真实解析结果，不是预置的"通过"常量。
  推理脚本的 citation 在调用时刻从提示里取真实 evidence_id
  （见 ``_DynamicReasoningLLM``），不是写死的 ID。
"""

from __future__ import annotations

import json
import re
import sys

from harness_agent.contracts.llm import LLMMessage, LLMResult
from harness_agent.contracts.retrieval import StoredChunk
from harness_agent.experts.reasoning_expert import ReasoningExpertImpl
from harness_agent.llm.mock import MockLLMClient
from harness_agent.models.session import SessionContext
from harness_agent.observability import PatternDesensitizer, build_observability_stack
from harness_agent.orchestrator import build_orchestrator
from harness_agent.retrieval.wiring import build_retrieval_stack
from harness_agent.safety import build_safety_stack

PAT_PENICILLIN = "pat-001"  # M2 种子：青霉素过敏（阻断 beta_lactam 全组）

# 合成知识条目（来自 synthetic_data.py 的 KNOWLEDGE_ENTRIES）
#
# 两条条目都刻意带上主诉关键词（咳嗽 / 发热），因为零依赖模式用的是
# **哈希嵌入 + BM25**：稠密路无语义能力，召回几乎全落在稀疏路关键词上。
# 条目只写"β-内酰胺过敏"而不写主诉词的话，脱敏后的查询（含
# [REDACTED-xx] 占位符噪声）会召回不到这条本该命中的证据。
#
# 另一处刻意的取舍：kb-cap-01 不写具体过敏药名（只用"β-内酰胺类"类名）。
# 一旦正文出现"青霉素"，**装配闸门会把它整条过滤掉**——这是闸门按设计
# 工作（过滤含患者过敏药物实体的证据），但真实指南条目普遍是
# "青霉素过敏者改用阿奇霉素"这种写法，全量过滤会误伤。这是当前实现
# 的一条真实边界，已列入 README「已知边界（诚实标注）」。
_KB_ENTRIES = [
    StoredChunk(
        chunk_id="kb-cap-01",
        content=(
            "社区获得性肺炎（CAP）：咳嗽、发热患者的经验性治疗可选"
            "大环内酯类（阿奇霉素）。阿奇霉素 500mg qd，疗程 3-5 天。"
            "对 β-内酰胺类过敏者同样适用。"
        ),
    ),
    StoredChunk(
        chunk_id="kb-cap-02",
        content=("社区获得性肺炎评估：CURB-65 评分用于判断严重程度。评分 0-1 分可门诊治疗。"),
    ),
]


def _reasoning_output(citation: str) -> str:
    """推理专家 LLM 合法输出（三段式推理链）。

    注意：推理链文本不直接提及患者过敏的具体药名（如"青霉素"），
    输出闸门会对结论+推理链全文做药物安全扫描，提及过敏药名
    会触发拦截。过敏约束由安全栈硬规则保障（非推理链文本）。
    """
    return json.dumps(
        {
            "steps": [
                {
                    "kind": "evidence",
                    "text": (
                        "引用证据：CAP 患者对 β-内酰胺类过敏时，"
                        "阿奇霉素为安全替代方案，常规剂量 500mg qd"
                    ),
                    "citations": [citation],
                },
                {
                    "kind": "inference",
                    "text": (
                        "患者有 β-内酰胺类过敏史，阿奇霉素与之无交叉反应，"
                        "可安全使用；结合咳嗽发热症状与 CAP 指南，"
                        "经验性治疗合理"
                    ),
                },
                {
                    "kind": "conclusion",
                    "text": ("建议阿奇霉素 500mg qd，疗程 3-5 天，门诊随访观察疗效"),
                    "citations": [citation],
                },
            ],
            "statement": ("CAP 经验性治疗：阿奇霉素 500mg qd × 3-5 天"),
            "self_check_notes": "自检通过（3/3）：引用真实、因果正向、依据充分",
        }
    )


def _judge_output() -> str:
    """质量门禁 judge LLM 输出（通过）。"""
    return json.dumps(
        {
            "faithfulness": 0.92,
            "has_hallucination": False,
            "causal_inversion": False,
            "reason": "结论有充分证据支撑，无臆测，因果顺序正确",
        }
    )


class _DynamicReasoningLLM:
    """推理 LLM 替身：调用时刻才从提示中取真实 evidence_id 填进 citations。

    为什么不用 ``MockLLMClient(role="reasoning", script=[...])``：
    证据 ID 由检索层在 ``retrieve`` 时动态生成（每次调用都是新 ID），
    而 MockLLMClient 的脚本在**构造时**就固定了——写死 "ev-1" 会让
    推理专家的自检（引用真实性：每条 citation 必须存在于证据包）
    失败，demo 会退化成"转人工"。

    本替身把脚本延迟到 ``complete`` 时刻执行，从提示的"合法
    evidence_id 列表"里解析出真实 ID 再套用三段式文本。这等价于真实
    LLM 的行为（读提示 → 引用提示里给出的 ID），因此推理专家的自检、
    LLM-judge、输出闸门走的都是与生产一致的路径。

    它实现 ``LLMClient`` 契约，与 ``MockLLMClient`` / ``OpenAICompatClient``
    可互换——换成在线端点时删掉这个类即可，编排层零改动。
    """

    role = "reasoning"

    def __init__(self) -> None:
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> LLMResult:
        self.calls.append(list(messages))
        user_content = "".join(m.content for m in messages if m.role != "system")
        ev_ids = sorted(set(re.findall(r"\bev-[0-9a-f]{6,}\b", user_content)))
        # 无合法证据可引用时返回空文本，消费方按解析失败 fail-closed 升级
        text = "" if not ev_ids else _reasoning_output(ev_ids[0])
        return LLMResult(
            text=text,
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(text.split()) if text.strip() else 0,
            model="mock-reasoning",
        )


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


class _StubMemoryExpert:
    """记忆专家桩（初诊场景不委派记忆专家，但注册表需绑定）。"""

    name = "memory_expert"

    def assemble(self, query, context):
        from harness_agent.contracts.experts import ContextBundle

        return ContextBundle(patient_id=context.patient_id, stable_facts=["血型 O 型"])


def main() -> None:
    # Windows 默认终端（代码页 936/GBK）无法编码 ✓/✗ 等符号，打印时抛
    # UnicodeEncodeError——demo 会崩在最后的验收总结。统一按 UTF-8 输出，
    # 消除演示环境差异。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("M8 端到端演示一：初诊推理全链路")
    print("全链路 Mock LLM + 内存检索栈（零外部依赖）")
    print("患者档案：pat-001 张明，45 岁男，青霉素过敏，咳嗽三天伴发热")

    # ---- 0. 脱敏中间件前置 ----
    _print_section("步骤 0：脱敏中间件前置")
    desensitizer = PatternDesensitizer()
    # 患者姓名用显式标记（患者：xxx），与脱敏器的姓名规则一致——
    # 裸姓名（无标记）按设计不脱敏，避免误伤临床正文中的人名。
    # 输入刻意不含过敏药名（如"盘尼西林"）：输入闸门不做意图识别，
    # 检出过敏药名即拦截转人工；过敏史由安全栈硬规则精确匹配（见步骤 1）。
    raw_input = (
        "患者：张明（身份证 310101198001011234）咳嗽三天，"
        "发烧 38.5 度，用药方案怎么定？电话 13812345678"
    )
    redacted = desensitizer.desensitize(raw_input)
    print(f"  原始输入: {raw_input}")
    print(f"  脱敏后:   {redacted.text}")
    print(f"  移除标识: {', '.join(redacted.removed_entities)}")
    print("  → 患者标识已替换为 [REDACTED-xx] 占位符")
    print("  → 脱敏产物将作为查询进入编排链路（见步骤 2）")

    # ---- 1. 装配全链路组件 ----
    _print_section("步骤 1：装配全链路组件")
    stack = build_retrieval_stack()
    stack.service.index(_KB_ENTRIES)
    print(f"  知识库入库: {len(_KB_ENTRIES)} 条（CAP 指南）")

    safety = build_safety_stack()
    allergies = safety.allergy_store.get(PAT_PENICILLIN)
    for a in allergies:
        print(f"  过敏史: {a.drug_name_raw} → {a.normalized_drug} (ATC={a.atc_code})")
        cross = sorted(a.cross_reactants)
        print(f"    交叉反应阻断（{len(cross)} 条）: {', '.join(cross) or '（无）'}")

    obs = build_observability_stack(data_dir=".data/demo-m8-first")
    obs.tracer.bind("sess-first", "trace-first")
    print("  可观测栈: NoopTracer + SQLiteAuditStore + MemoryCacheStore")

    # ---- 2. 路由器裁决 ----
    _print_section("步骤 2：路由器裁决")
    # 推理替身在调用时刻从提示中取真实 evidence_id（见类 docstring）。
    reasoning_llm = _DynamicReasoningLLM()
    judge_llm = MockLLMClient(role="judge", script=[_judge_output()])

    agent = build_orchestrator(
        experts={
            "reasoning_expert": ReasoningExpertImpl(llm=reasoning_llm),
            "memory_expert": _StubMemoryExpert(),
        },
        retrieval=stack.service,
        judge_llm=judge_llm,
    )

    context = SessionContext(session_id="sess-first", patient_id=PAT_PENICILLIN)
    # 查询 = 步骤 0 的脱敏产物（含 [REDACTED-xx] 占位符）——脱敏中间件
    # 真实串进编排链路，而非仅打印展示。查询不含任何过敏药名：输入闸门
    # 不做意图识别，检出过敏药名即拦截转人工（患者陈述过敏史的问法
    # "我青霉素过敏，能吃什么"同样会被拦）。这是 fail-closed 的代价，
    # 已列入 README「已知边界（诚实标注）」一节。
    user_query = redacted.text

    result = agent.handle(user_query, context)

    print(f"  查询: {user_query}")
    print(f"  路由: {result.route.decision} (by_rule={result.route.by_rule})")
    print(f"  规则命中: {result.route.reason}")

    # ---- 3. 检索供给 ----
    _print_section("步骤 3：检索供给层")
    if result.evidence_pack is not None:
        pack = result.evidence_pack
        print(f"  证据包: {len(pack.evidence)} 条, is_reviewed={pack.is_reviewed}")
        print(f"  阻断药物: {pack.blocked_drugs or '（无）'}")
        for ev in pack.evidence:
            tag = "[补全]" if ev.is_structural_completion else "[命中]"
            print(f"  {tag} [{ev.evidence_id[:16]}] {ev.content[:80]}")

    # ---- 4. 推理专家 ----
    _print_section("步骤 4：推理专家（三段式推理链 + 自检）")
    if result.conclusion is not None:
        chain = result.conclusion.reasoning_chain
        print(f"  自检: {chain.self_check_passed} — {chain.self_check_notes}")
        print(f"  推理链 {len(chain.steps)} 步:")
        for i, step in enumerate(chain.steps):
            cite = f" (引用: {step.citations})" if step.citations else ""
            print(f"    {i + 1}. [{step.kind}] {step.text[:80]}{cite}")

    # ---- 5. 质量门禁 ----
    _print_section("步骤 5：质量门禁（LLM-judge + 输出闸门）")
    if result.gate_verdicts:
        for v in result.gate_verdicts:
            status = "通过" if v.allowed else "拦截"
            print(f"  {v.gate}: {status} — {v.reason[:60]}")

    # ---- 6. 临床结论 ----
    _print_section("步骤 6：临床结论输出")
    if result.conclusion is not None:
        print(f"  结论: {result.conclusion.statement}")
        print(f"  产出者: {result.conclusion.produced_by}")
        print(f"  引用证据: {result.conclusion.cited_evidence_ids}")
        print(f"  结论ID: {result.conclusion.conclusion_id[:24]}")
    else:
        print("  （结论被门禁拦截，未交付）")

    # ---- 7. 全链路 trace ----
    _print_section("步骤 7：全链路 trace 事件")
    from harness_agent.models.audit import TraceEvent

    # payload 全部取自本轮编排的真实产出（非手写字面量）
    trace_events: list[tuple[str, dict]] = [
        (
            "route",
            {
                "decision": result.route.decision,
                "by_rule": result.route.by_rule,
            },
        ),
    ]
    if result.evidence_pack is not None:
        ev_count = len(result.evidence_pack.evidence)
        trace_events.append(("retrieve", {"query": user_query, "evidence_count": ev_count}))
    if result.conclusion is not None:
        chain = result.conclusion.reasoning_chain
        trace_events.append(
            ("reason", {"chain_steps": len(chain.steps), "self_check": chain.self_check_passed})
        )
    if result.gate_verdicts:
        trace_events.append(
            (
                "gate_check",
                {v.gate: ("pass" if v.allowed else "block") for v in result.gate_verdicts},
            )
        )
        if all(v.allowed for v in result.gate_verdicts) and result.conclusion is not None:
            trace_events.append(("conclude", {"statement": result.conclusion.statement}))
    elif result.escalation is not None:
        trace_events.append(("escalate", {"reason": result.escalation.reason}))

    for event_type, payload in trace_events:
        event = TraceEvent(
            trace_id="trace-first",
            session_id="sess-first",
            event_type=event_type,
            payload=payload,
        )
        obs.tracer.record(event)
        print(f"  [TRACE] {event_type}: {payload}")

    print(f"\n  全链路事件总数: {obs.tracer.event_count}")

    # ---- 验收总结（按本轮真实结果条件打印，非恒绿） ----
    def _verdict(gate: str):
        return next((v for v in result.gate_verdicts if v.gate == gate), None)

    quality_verdict = _verdict("quality_judge")
    output_verdict = _verdict("output")
    checks = [
        ("脱敏中间件前置（患者标识去除）", bool(redacted.removed_entities)),
        (
            "路由器规则命中（need_reasoning，零 LLM 开销）",
            result.route.by_rule and result.route.decision == "need_reasoning",
        ),
        (
            "检索供给（知识库双路召回 + 装配闸门复核）",
            result.evidence_pack is not None and result.evidence_pack.is_reviewed,
        ),
        (
            "推理专家（三段式推理链 + 自检 3/3 通过）",
            result.conclusion is not None and result.conclusion.reasoning_chain.self_check_passed,
        ),
        (
            "质量门禁（LLM-judge 忠实度 0.92 ≥ 0.70）",
            quality_verdict is not None and quality_verdict.allowed,
        ),
        (
            "输出闸门（药物安全全文扫描通过，阿奇霉素不在阻断列表）",
            output_verdict is not None and output_verdict.allowed,
        ),
        ("临床结论交付（含证据溯源与推理链）", result.conclusion is not None),
        ("全链路 trace 事件 5 个可打印", obs.tracer.event_count >= 5),
    ]
    print()
    print("=" * 72)
    print("初诊推理全链路验收总结:")
    for line, ok in checks:
        print(f"  {'✓' if ok else '✗'} {line}")
    print("=" * 72)


if __name__ == "__main__":
    main()
