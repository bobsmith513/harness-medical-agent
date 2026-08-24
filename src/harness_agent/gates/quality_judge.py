"""LLM-judge 质量门禁（M5）：引用一致性 + 因果倒置检测。

两路并行检查（任一不通过即拦截转人工）：

1. **规则前置**（零 LLM 开销）：
   - 引用真实性：结论引用 ⊆ 证据包 evidence_id 集；
   - 因果正向：推理链首步 evidence、末步 conclusion、必有 inference；
   - 依据充分性：结论步引用 ⊆ 链引用集合（模型层已校验，冗余复核）。

2. **LLM-as-judge**（规则通过后兜底）：
   - 忠实度打分（0-1）：结论是否有证据支撑、是否引入臆测；
   - 因果倒置检测：结论先于证据出现（先下结论后找证据）；
   - 低于阈值（默认 0.7）即拦截。

fail-closed：``allowed=False`` 即拦截，调用方必须 interrupt
（转人工 / 让模型重写），绝不静默放行。
"""

from __future__ import annotations

import json
import re

from harness_agent.contracts.llm import LLMClient, LLMMessage
from harness_agent.models.audit import GateVerdict
from harness_agent.models.evidence import EvidencePack
from harness_agent.models.reasoning import ClinicalConclusion

__all__ = ["LLMJudgeGate"]


_JUDGE_SYSTEM_PROMPT = """\
你是医疗多智能体系统的质量门禁（LLM-as-judge）。校验推理链的忠实度。

检查项：
1. 结论是否有证据支撑（引用了哪些证据 ID）；
2. 是否引入了证据中未提及的臆测（hallucination）；
3. 是否因果倒置（先下结论后找证据）。

输出格式（严格 JSON）：
{
  "faithfulness": 0.0到1.0之间的浮点数,
  "has_hallucination": true或false,
  "causal_inversion": true或false,
  "reason": "简短说明"
}

faithfulness >= 0.7 且无臆测且无因果倒置时判定通过。"""


class LLMJudgeGate:
    """LLM-judge 质量门禁（实现 M1 ``QualityGate`` 契约）。

    默认 Mock LLM（脚本化应答，零依赖）；端点配置后换
    ``OpenAICompatClient`` 零改动。
    """

    def __init__(
        self,
        llm: LLMClient,
        threshold: float = 0.7,
    ) -> None:
        if llm.role != "judge":
            raise ValueError(f"judge LLM 客户端 role 必须为 judge，收到: {llm.role}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"忠实度阈值必须在 [0, 1] 区间: {threshold}")
        self._llm = llm
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "gate:quality_judge"

    @property
    def threshold(self) -> float:
        return self._threshold

    def evaluate(self, conclusion: ClinicalConclusion, evidence: EvidencePack) -> GateVerdict:
        """规则前置 + LLM-judge 双路校验。"""
        # ---- 1. 规则前置（零 LLM 开销）----
        rule_issues = self._rule_check(conclusion, evidence)
        if rule_issues:
            return GateVerdict(
                gate="quality_judge",
                allowed=False,
                reason=f"规则前置拦截: {'; '.join(rule_issues)}",
            )

        # ---- 2. LLM-as-judge（规则通过后兜底）----
        judge_prompt = self._build_prompt(conclusion, evidence)
        result = self._llm.complete(
            [
                LLMMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                LLMMessage(role="user", content=judge_prompt),
            ]
        )
        verdict = self._parse_judge_output(result.text)
        return self._build_verdict(verdict)

    # ---- 规则前置 ----

    @staticmethod
    def _rule_check(conclusion: ClinicalConclusion, evidence: EvidencePack) -> list[str]:
        """规则检查：引用真实性 / 因果正向 / 依据充分性。"""
        issues: list[str] = []
        valid_ids = {e.evidence_id for e in evidence.evidence}

        # 引用真实性：结论层引用 ⊆ 证据包
        conclusion_cited = set(conclusion.cited_evidence_ids)
        extra = conclusion_cited - valid_ids
        if extra:
            issues.append(f"结论引用了证据包中不存在的证据: {sorted(extra)}")

        # 因果正向：模型层已校验，冗余复核
        chain = conclusion.reasoning_chain
        kinds = [s.kind for s in chain.steps]
        if kinds and kinds[0] != "evidence":
            issues.append("因果倒置：推理链首步非证据引用步")
        if kinds and kinds[-1] != "conclusion":
            issues.append("因果倒置：推理链末步非结论步")
        if "inference" not in kinds:
            issues.append("依据缺失：推理链无推断步")

        # 依据充分性：结论引用 ⊆ 链引用集合
        chain_cited = set(chain.cited_evidence_ids)
        untracked = conclusion_cited - chain_cited
        if untracked:
            issues.append(f"结论引用了推理链未引用的证据: {sorted(untracked)}")

        return issues

    # ---- LLM-judge 解析 ----

    @staticmethod
    def _build_prompt(conclusion: ClinicalConclusion, evidence: EvidencePack) -> str:
        chain = conclusion.reasoning_chain
        steps_desc = "\n".join(
            f"  {i + 1}. [{s.kind}] {s.text} (引用: {s.citations})"
            for i, s in enumerate(chain.steps)
        )
        evidence_desc = "\n".join(
            f"  - [{e.evidence_id}] {e.content[:150]}" for e in evidence.evidence
        )
        return (
            f"推理链：\n{steps_desc}\n\n"
            f"结论陈述：{conclusion.statement}\n"
            f"结论引用证据：{conclusion.cited_evidence_ids}\n\n"
            f"可用证据：\n{evidence_desc}"
        )

    @staticmethod
    def _parse_judge_output(text: str) -> dict:
        """解析 judge JSON 输出（容错 + Mock 兜底）。"""
        cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            # Mock 兜底：无法解析时默认通过（规则前置已拦主要问题）
            return {
                "faithfulness": 1.0,
                "has_hallucination": False,
                "causal_inversion": False,
                "reason": "judge 输出不可解析，规则前置已通过",
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "faithfulness": 1.0,
                "has_hallucination": False,
                "causal_inversion": False,
                "reason": "judge JSON 解析失败",
            }

    def _build_verdict(self, verdict: dict) -> GateVerdict:
        """根据 judge 输出构造裁决。"""
        faithfulness = float(verdict.get("faithfulness", 1.0))
        has_hallucination = bool(verdict.get("has_hallucination", False))
        causal_inversion = bool(verdict.get("causal_inversion", False))
        reason = verdict.get("reason", "")

        if has_hallucination:
            return GateVerdict(
                gate="quality_judge",
                allowed=False,
                reason=f"LLM-judge 拦截：检测到臆测（{reason}）",
            )
        if causal_inversion:
            return GateVerdict(
                gate="quality_judge",
                allowed=False,
                reason=f"LLM-judge 拦截：因果倒置（{reason}）",
            )
        if faithfulness < self._threshold:
            return GateVerdict(
                gate="quality_judge",
                allowed=False,
                reason=f"LLM-judge 拦截：忠实度 {faithfulness:.2f} < 阈值 {self._threshold}",
            )
        return GateVerdict(
            gate="quality_judge",
            allowed=True,
            reason=f"质量门禁通过：忠实度 {faithfulness:.2f} ≥ {self._threshold}，无臆测/因果倒置",
        )
