"""推理专家（M5）：三段式推理链生成 + 自检。

推理链固定结构（M1 模型层面已锁死）::

    证据引用步（citations 必非空）
      → 逐步推断步（可引用证据也可纯推断）
      → 结论步（末步，可携带最终引用）

自检三项（自检不过则推理专家不产出 ``ClinicalConclusion``——
``ClinicalConclusion`` 构造校验要求 ``self_check_passed=True``，
主 Agent 编排角色无法绕过）：

1. **引用真实性**：每条 citation 必须出现在证据包的 evidence_id 集合中；
2. **因果正向**：证据步在推断步之前、结论步在末尾（结构校验已在
   ``ReasoningChain`` 模型层固化，自检复核一层冗余）；
3. **依据充分性**：结论步的引用 ⊆ 推理链引用集合（不得引入
   链中从未出现的证据 ID）。

LLM 调用默认 Mock（脚本化应答，零外部依赖）；端点配置后换
``OpenAICompatClient`` 零改动——同一推理专家注入不同实现，
行为只取决于 LLM 应答的解析结果，业务逻辑不分叉。
"""

from __future__ import annotations

import json

from harness_agent.contracts.experts import ExpertTask
from harness_agent.contracts.llm import LLMClient, LLMMessage
from harness_agent.contracts.retrieval import RetrievalService
from harness_agent.llm.json_utils import extract_json_object
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import (
    ClinicalConclusion,
    ReasoningChain,
    ReasoningStep,
)
from harness_agent.models.session import SessionContext

__all__ = ["ReasoningExpertImpl"]


_REASONING_SYSTEM_PROMPT = """\
你是医疗多智能体系统的临床推理专家。基于已通过装配复核的证据包，
生成"证据引用 → 逐步推断 → 结论"三段式推理链。

输出格式（严格 JSON，禁止其他文本）：
{
  "steps": [
    {"kind": "evidence", "text": "对证据的引用与概括", "citations": ["ev_xxx"]},
    {"kind": "inference", "text": "基于证据的逐步推断", "citations": []},
    {"kind": "conclusion", "text": "最终结论", "citations": ["ev_xxx"]}
  ],
  "statement": "结论陈述（一句话）",
  "self_check_notes": "自检说明"
}

约束：
- evidence 步必须至少引用一条 evidence_id；
- citations 中的 ID 必须来自提供的证据包；
- 结论步必须是最后一步；
- 证据不足时如实声明，不得臆测。"""


class ReasoningExpertImpl:
    """推理专家实现（M1 ``ReasoningExpert`` 契约）。

    默认 Mock LLM（零依赖）：构造时传入脚本化应答，
    端点配置后换 ``OpenAICompatClient`` 零改动。
    """

    def __init__(
        self,
        llm: LLMClient,
        retrieval: RetrievalService | None = None,
    ) -> None:
        if llm.role != "reasoning":
            raise ValueError(f"推理 LLM 客户端 role 必须为 reasoning，收到: {llm.role}")
        self._llm = llm
        self._retrieval = retrieval

    @property
    def name(self) -> str:
        return "reasoning_expert"

    @property
    def description(self) -> str:
        return "临床推理专家：生成三段式推理链并自检，系统内唯一合法临床结论产出方。"

    def reason(
        self,
        task: ExpertTask,
        evidence: EvidencePack,
        context: SessionContext,
    ) -> ClinicalConclusion:
        """证据包 → 推理链 → 自检 → 临床结论（自检不过抛异常）。"""
        if not evidence.is_reviewed:
            raise ValueError(
                "证据包未通过装配复核（is_reviewed=False），推理专家不得基于其产出结论"
            )

        # 构造提示：证据摘要 + 用户问题
        evidence_block = self._format_evidence(evidence)
        user_prompt = self._build_user_prompt(task, evidence_block, evidence)
        result = self._llm.complete(
            [
                LLMMessage(role="system", content=_REASONING_SYSTEM_PROMPT),
                LLMMessage(role="user", content=user_prompt),
            ]
        )

        # 解析 LLM 输出为推理链
        parsed = self._parse_output(result.text, evidence)
        chain = self._build_chain(parsed, evidence)
        # 自检（三项校验，不过则抛异常——不产出结论）
        notes = self._self_check(chain, evidence)
        chain = chain.model_copy(update={"self_check_passed": True, "self_check_notes": notes})

        return ClinicalConclusion(
            statement=parsed.get("statement", chain.steps[-1].text),
            reasoning_chain=chain,
            cited_evidence_ids=self._conclusion_citations(parsed, chain),
        )

    # ---- 内部方法 ----

    @staticmethod
    def _format_evidence(evidence: EvidencePack) -> str:
        if not evidence.evidence:
            return "（无可用证据——证据包为空）"
        lines = []
        for ev in evidence.evidence:
            lines.append(
                f"- [{ev.evidence_id}] (置信度={ev.confidence}, 来源={ev.provenance}) "
                f"{ev.content[:200]}"
            )
        return "\n".join(lines)

    @staticmethod
    def _build_user_prompt(task: ExpertTask, evidence_block: str, evidence: EvidencePack) -> str:
        valid_ids = [e.evidence_id for e in evidence.evidence]
        return (
            f"任务指令：\n{task.instruction}\n\n"
            f"可用证据（citations 必须从这些 ID 中选取）：\n{evidence_block}\n\n"
            f"合法 evidence_id 列表：{valid_ids}"
        )

    @staticmethod
    def _parse_output(text: str, evidence: EvidencePack) -> dict:
        """从 LLM 输出解析 JSON（容错：剥离 markdown 围栏 + 括号深度扫描）。

        fail-closed：无 JSON 或解析失败时抛异常（由编排层捕获转升级），
        绝不构造"兜底结论"——与患者问题无关的罐头结论正是本系统
        承诺绝不交付的东西。
        """
        fragment = extract_json_object(text)
        if fragment is None:
            raise ValueError(
                f"LLM 输出不含 JSON（长度 {len(text)} 字符），无法解析为推理链——fail-closed 转人工"
            )
        try:
            parsed = json.loads(fragment)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM 输出 JSON 解析失败: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("LLM 输出 JSON 顶层必须是对象")
        return parsed

    @staticmethod
    def _build_chain(parsed: dict, evidence: EvidencePack) -> ReasoningChain:
        raw_steps = parsed.get("steps", [])
        if not raw_steps:
            raise ValueError("LLM 输出无 steps 字段")
        steps = []
        for raw in raw_steps:
            steps.append(
                ReasoningStep(
                    kind=raw["kind"],
                    text=raw.get("text", ""),
                    citations=raw.get("citations", []),
                )
            )
        return ReasoningChain(steps=steps, self_check_passed=False)

    @staticmethod
    def _self_check(chain: ReasoningChain, evidence: EvidencePack) -> str:
        """自检三项：引用真实性 / 因果正向 / 依据充分性。"""
        valid_ids = {e.evidence_id for e in evidence.evidence}
        issues: list[str] = []

        # 1. 引用真实性：每条 citation 必须在证据包中
        for step in chain.steps:
            for citation in step.citations:
                if citation not in valid_ids:
                    issues.append(f"步骤 {step.step_id[:8]} 引用了不存在的证据: {citation}")

        # 2. 因果正向：结构校验已在模型层固化，此处复核
        kinds = [s.kind for s in chain.steps]
        if kinds[0] != "evidence":
            issues.append("推理链首步必须是证据引用步")
        if kinds[-1] != "conclusion":
            issues.append("推理链末步必须是结论步")
        if "inference" not in kinds:
            issues.append("推理链必须包含推断步")

        # 3. 依据充分性：结论步引用 ⊆ 结论步之前各步的引用集合
        conclusion_step = chain.steps[-1]
        prior_cited: set[str] = set()
        for step in chain.steps[:-1]:
            prior_cited.update(step.citations)
        extra = set(conclusion_step.citations) - prior_cited
        if extra:
            issues.append(f"结论步引用了推理链未引用的证据: {sorted(extra)}")

        if issues:
            raise ValueError(f"推理链自检失败: {'; '.join(issues)}")
        return f"自检通过（3/3）：引用真实、因果正向、依据充分；证据 {len(valid_ids)} 条"

    @staticmethod
    def _conclusion_citations(parsed: dict, chain: ReasoningChain) -> list[str]:
        """结论层显式引用（取 LLM 声明或链末步引用）。"""
        stated = parsed.get("cited_evidence_ids", [])
        if isinstance(stated, list) and stated:
            return stated
        return chain.steps[-1].citations
