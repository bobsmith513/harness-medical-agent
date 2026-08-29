"""harness-online 命令行入口——整个项目的统一启动点（main 函数在此）。

    uv run harness-online                                # 交互问诊（选患者 → 提问）
    uv run harness-online --list                         # 列出预置患者与样例问题
    uv run harness-online --patient pat-001 \\
        --query "咳嗽三天伴发热，用药方案怎么定？"         # 单次问诊
    uv run harness-online --no-seed                      # 空库运行（不灌样例数据）

所有模型（含微调推理模型）与数据库地址全部来自 ``.env``：

- 模型：``HARNESS_LLM__PROVIDER`` + ``HARNESS_LLM__API_KEY``（共享），
  四个角色（orchestrator/reasoning/judge/router）均可逐角色覆盖；
- 微调模型：``HARNESS_LLM__REASONING_BASE_URL`` 指向外部 API 或本地 vLLM；
- 数据库：``HARNESS_RETRIEVAL__STORE``（local / milvus）+ ``MILVUS_URI``，
  库内容默认由 ``HARNESS_APP__SEED_SAMPLE_DATA=true`` 自动灌入合成样例。

在线调用失败时 fail-closed：转人工升级，绝不静默降级应答。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 自举：直接以文件路径运行（python src/harness_agent/cli.py）时，
# 把包根目录 src/ 加入搜索路径，避免 ModuleNotFoundError: harness_agent
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from harness_agent.config.settings import Settings, get_settings, reset_settings  # noqa: E402
from harness_agent.contracts.retrieval import StoredChunk  # noqa: E402
from harness_agent.experts.memory_expert import MemoryExpertImpl  # noqa: E402
from harness_agent.experts.reasoning_expert import ReasoningExpertImpl  # noqa: E402
from harness_agent.llm.wiring import build_llm_client, describe_llm_setup  # noqa: E402
from harness_agent.models.session import SessionContext  # noqa: E402
from harness_agent.observability import PatternDesensitizer  # noqa: E402
from harness_agent.orchestrator import build_orchestrator  # noqa: E402
from harness_agent.retrieval.wiring import build_retrieval_stack  # noqa: E402
from harness_agent.safety import build_safety_stack  # noqa: E402
from harness_agent.seed_data import KNOWLEDGE_ENTRIES, PATIENT_PROFILES  # noqa: E402

__all__ = ["main"]


#: 每位患者的样例问诊问题（交互模式回车展示，可直接输入）；
#: 首条为记忆召回类（no_reasoning 路径，验证记忆专家装配），
#: 其余为推理类（need_reasoning 路径，验证三段式推理链 + 双门禁）。
SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "pat-001": [
        "我的高血压病史有多久了？",
        "咳嗽三天伴发热，用药方案怎么定？",
        "社区获得性肺炎怎么治，需要住院吗？",
    ],
    "pat-002": [
        "我之前的病史记录是什么？",
        "关节疼痛加重，类风湿怎么用药？",
        "类风湿关节炎的治疗方案有哪些？",
    ],
    "pat-003": [
        "肺炎复查，之前的记录是什么？",
        "社区获得性肺炎的副作用和禁忌？",
    ],
    "pat-004": [
        "我之前的病史和血糖情况记录是什么？",
        "血糖控制得怎么样，需要调药吗？",
        "二甲双胍的剂量和副作用？",
    ],
}

#: 占位 Key 标记（.env 模板值；检测到即提示替换）
_PLACEHOLDER_MARK = "REPLACE_ME"


# ---------------------------------------------------------------------------
# 启动横幅与配置检查
# ---------------------------------------------------------------------------


def _key_is_placeholder(settings: Settings) -> bool:
    """共享或任一角色 Key 仍是 .env 模板占位值。"""
    llm = settings.llm
    keys = [llm.api_key] + [
        getattr(llm, f"{role}_api_key", "")
        for role in ("orchestrator", "reasoning", "judge", "router")
    ]
    return any(_PLACEHOLDER_MARK in key for key in keys)


def _banner(settings: Settings) -> None:
    print("=" * 72)
    print("harness-medical-agent 在线问诊入口（main）")
    print("=" * 72)
    print(describe_llm_setup(settings))

    if settings.llm.provider == "mock":
        print()
        print("  ⚠ 当前为 mock 模式（.env 未配置在线 API）")
        print("    填写 HARNESS_LLM__PROVIDER=deepseek + HARNESS_LLM__API_KEY")
        print("    后重启，即可切换为真实在线推理。详见 .env.example。")
    elif _key_is_placeholder(settings):
        print()
        print("  ⚠ API Key 仍是占位值 sk-REPLACE_ME——在线调用将返回 401。")
        print("    两个办法：")
        print("    1. 把 .env 中的 sk-REPLACE_ME 换成你的真实 Key（推荐）")
        print("    2. 无 Key 体验完整在线链路：另开终端运行")
        print("       uv run python examples/mock_openai_server.py")
        print("       再把 .env 的各 BASE_URL 指向 http://127.0.0.1:8100/v1")

    seed = settings.app.seed_sample_data
    print()
    print(
        f"  数据库: {settings.retrieval.store}（地址来自 .env）｜"
        f"样例灌入: {'开（8 条知识 + 4 位患者）' if seed else '关（空库运行）'}"
    )
    print()


# ---------------------------------------------------------------------------
# 专家装配
# ---------------------------------------------------------------------------


def _patient_memory_chunks(profiles: list) -> list[StoredChunk]:
    """患者档案 → 分区记忆 chunk（doctor_verified 稳定 / model_inference 易变）。

    记忆专家（MemoryExpertImpl）经检索层召回这些 chunk 并按
    provenance 分类装配——替代早期"直接读静态档案"的演示替身，
    主链路与 M3 分区隔离 / M5 装配逻辑完全一致。
    """
    chunks: list[StoredChunk] = []
    for profile in profiles:
        for i, fact in enumerate(profile.stable_facts):
            chunks.append(
                StoredChunk(
                    chunk_id=f"mem-{profile.patient_id}-s{i}",
                    patient_id=profile.patient_id,
                    content=fact,
                    metadata={"provenance": "doctor_verified", "confidence": "high"},
                )
            )
        for i, fact in enumerate(profile.volatile_facts):
            chunks.append(
                StoredChunk(
                    chunk_id=f"mem-{profile.patient_id}-v{i}",
                    patient_id=profile.patient_id,
                    content=fact,
                    metadata={"provenance": "model_inference", "confidence": "medium"},
                )
            )
    return chunks


def _build_agent(settings: Settings):
    """装配主 Agent（LLM 客户端形态由 .env 决定，代码零分叉）。

    安全栈单例：检索层的输入/装配闸门与编排层的输出闸门共用同一
    ``SafetyStack`` 实例——否则两侧各持一份过敏史，过敏记录一旦来自
    外部 HIS / EMR 就会出现"一端拦、一端放"的口径分叉。
    """
    reasoning_llm = build_llm_client("reasoning", settings)
    judge_llm = build_llm_client("judge", settings)
    router_llm = build_llm_client("router", settings)

    safety = build_safety_stack(settings)
    stack = build_retrieval_stack(settings, safety=safety)

    kb_count = 0
    mem_count = 0
    if settings.app.seed_sample_data:
        chunks = [
            StoredChunk(
                chunk_id=entry.chunk_id,
                content=entry.content,
                metadata={k: str(v) for k, v in entry.metadata.items()},
            )
            for entry in KNOWLEDGE_ENTRIES
        ]
        memories = _patient_memory_chunks(PATIENT_PROFILES)
        stack.service.index(chunks + memories)
        kb_count = len(chunks)
        mem_count = len(memories)

    agent = build_orchestrator(
        experts={
            "reasoning_expert": ReasoningExpertImpl(llm=reasoning_llm, retrieval=stack.service),
            "memory_expert": MemoryExpertImpl(retrieval=stack.service, safety=safety),
        },
        retrieval=stack.service,
        router_llm=router_llm,
        judge_llm=judge_llm,
        settings=settings,
        safety=safety,
    )
    return agent, kb_count, mem_count


# ---------------------------------------------------------------------------
# 单轮编排与白盒输出
# ---------------------------------------------------------------------------


def _print_result(result) -> None:
    """白盒打印一轮编排的全部中间产物。"""
    print("-" * 72)
    print(f"  路由: {result.route.decision} (by_rule={result.route.by_rule})")
    print(f"        {result.route.reason}")

    if result.evidence_pack is not None:
        pack = result.evidence_pack
        print(f"  证据包: {len(pack.evidence)} 条 (复核={'通过' if pack.is_reviewed else '未过'})")
        print(f"  阻断药物: {pack.blocked_drugs or '（无）'}")
        for ev in pack.evidence:
            tag = "[补全]" if ev.is_structural_completion else "[命中]"
            print(f"    {tag} [{ev.evidence_id[:16]}] {ev.content[:60]}")

    if result.conclusion is not None:
        chain = result.conclusion.reasoning_chain
        print(f"  推理链 {len(chain.steps)} 步（自检: {chain.self_check_passed}）:")
        for i, step in enumerate(chain.steps):
            cite = f" ← {step.citations}" if step.citations else ""
            print(f"    {i + 1}. [{step.kind}] {step.text[:70]}{cite}")

    if result.gate_verdicts:
        for v in result.gate_verdicts:
            status = "✓ 通过" if v.allowed else "✗ 拦截"
            print(f"  门禁 {v.gate}: {status} — {v.reason[:70]}")

    print()
    if result.escalation is not None:
        print(f"  ⚠ 转人工（{result.escalation.reason[:70]}）")
        print("  结论未交付（fail-closed：拦截即升级，绝不静默放行）")
    elif result.conclusion is not None:
        print(f"  ★ 临床结论: {result.conclusion.statement}")
        print(f"    产出者: {result.conclusion.produced_by}")
        print(f"    引用证据: {result.conclusion.cited_evidence_ids}")
    elif result.context_bundle is not None:
        print("  ★ 记忆专家装配（no_reasoning 路径）:")
        print(f"    稳定事实: {result.context_bundle.stable_facts}")
        print(f"    过敏史: {len(result.context_bundle.allergies)} 条")
    print("-" * 72)


def _run_once(agent, patient_id: str, query: str, session_id: str) -> None:
    """单轮问诊：脱敏 → 编排 → 白盒输出（含错误友好提示）。"""
    desensitizer = PatternDesensitizer()
    redacted = desensitizer.desensitize(query).text
    if redacted != query:
        print(f"  脱敏后输入: {redacted}")

    context = SessionContext(session_id=session_id, patient_id=patient_id)
    try:
        result = agent.handle(redacted, context)
    except ValueError as exc:
        print("-" * 72)
        print(f"  ✗ 推理链未通过自检，结论不产出: {exc}")
        print("-" * 72)
        return
    except Exception as exc:  # noqa: BLE001 - 在线 API 网络/鉴权错误
        print("-" * 72)
        print(f"  ✗ 在线 API 调用失败: {type(exc).__name__}: {str(exc)[:90]}")
        print("    检查 .env 的 API Key 与端点连通性（Key / 网络 / 额度）")
        print("-" * 72)
        return

    _print_result(result)


# ---------------------------------------------------------------------------
# 交互循环
# ---------------------------------------------------------------------------


def _interactive(agent) -> None:
    """交互问诊循环：选患者 → 提问 → 白盒输出。"""
    print("预置患者（合成数据，输入序号或回车默认 1）:")
    for i, p in enumerate(PATIENT_PROFILES, 1):
        allergy = f"｜过敏: {'、'.join(p.allergies)}" if p.allergies else "｜无已知过敏"
        print(f"  {i}. {p.patient_id} {p.name} {p.age}岁{p.gender}｜{p.chief_complaint}{allergy}")

    current = PATIENT_PROFILES[0]
    turn = 0
    while True:
        print()
        prompt = f"[患者 {current.patient_id} {current.name}] 回车看样例问题 / q 退出 / 1-4 切换 > "
        try:
            choice = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D / Ctrl+C / stdin 关闭：优雅退出，不裸抛堆栈
            print()
            break
        if choice.lower() in ("q", "quit", "exit"):
            break
        if choice.isdigit() and 1 <= int(choice) <= len(PATIENT_PROFILES):
            current = PATIENT_PROFILES[int(choice) - 1]
            print(f"  → 已切换到 {current.patient_id} {current.name}（{current.chief_complaint}）")
            continue
        if not choice:
            for q in SAMPLE_QUESTIONS.get(current.patient_id, []):
                print(f"    · {q}")
            continue

        turn += 1
        _run_once(agent, current.patient_id, choice, f"sess-online-{turn}")


# ---------------------------------------------------------------------------
# main：整个项目的启动函数
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harness-online",
        description="在线问诊入口：模型与数据库全部来自 .env",
    )
    parser.add_argument("--patient", default="pat-001", help="患者 ID（默认 pat-001）")
    parser.add_argument("--query", default=None, help="单次模式：问诊问题")
    parser.add_argument("--list", action="store_true", help="列出预置患者与样例问题")
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="不灌入样例数据（空库运行，等价 HARNESS_APP__SEED_SAMPLE_DATA=false）",
    )
    args = parser.parse_args()

    try:
        reset_settings()  # 确保 .env 变更即时生效（清配置缓存）
        settings = get_settings()
    except Exception as exc:  # 配置校验失败：给出修复指引而非裸栈
        print(f"✗ 配置校验失败: {exc}")
        print("  参考 .env.example 检查 HARNESS_LLM__PROVIDER 取值")
        sys.exit(1)

    if args.no_seed:
        settings.app.seed_sample_data = False

    _banner(settings)

    if args.list:
        for p in PATIENT_PROFILES:
            print(f"{p.patient_id} {p.name} {p.age}岁{p.gender}｜{p.chief_complaint}")
            for q in SAMPLE_QUESTIONS.get(p.patient_id, []):
                print(f"    · {q}")
        return

    try:
        agent, kb_count, mem_count = _build_agent(settings)
    except ValueError as exc:
        print(f"✗ {exc}")
        print()
        print("  修复指引（三选一）：")
        print("  1. 在线调用：.env 填 HARNESS_LLM__PROVIDER=deepseek + API_KEY=sk-xxx")
        print("  2. 混合部署：另填 REASONING_BASE_URL=http://localhost:8001/v1（微调）")
        print("  3. 零依赖演示：.env 留空（provider=mock），交互与门禁逻辑照常")
        sys.exit(1)

    if kb_count:
        print(f"  知识库入库: {kb_count} 条（合成样例，可 index 接口替换真实数据）")
        print(f"  患者记忆分区: {mem_count} 条（记忆专家经检索召回装配）")
        print("  安全栈: pat-001 青霉素过敏 / pat-002 阿司匹林过敏（M2 种子）")
    else:
        print("  知识库: 空库运行（--no-seed；检索不命中时推理链走兜底）")
    print()

    if args.query:
        _run_once(agent, args.patient, args.query, "sess-online-single")
    else:
        _interactive(agent)


if __name__ == "__main__":
    main()
