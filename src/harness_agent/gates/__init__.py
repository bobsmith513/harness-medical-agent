"""质量门禁包（M5）。

- ``quality_judge``: LLM-as-judge 忠实度门禁（引用一致性 + 因果倒置 + 阈值）
- ``pipeline``:     质量门禁 → 输出闸门 串联流水线（fail-closed）
"""

from harness_agent.gates.pipeline import GatePipeline, GatePipelineResult
from harness_agent.gates.quality_judge import LLMJudgeGate

__all__ = ["GatePipeline", "GatePipelineResult", "LLMJudgeGate"]
