"""LLM 客户端装配工厂：配置 → 各角色客户端。

``build_llm_client(role, settings)`` 按 ``llm/providers.py`` 预设表
解析端点，返回 ``OpenAICompatClient``（在线 API）或 ``MockLLMClient``
（零依赖演示模式）——两类实现共用 M1 ``LLMClient`` 契约，调用方
（专家 / 路由器 / 门禁）零改动。

解析优先级（每个字段独立生效）::

    逐角色显式覆盖 > 共享配置 > provider 预设

    base_url: <ROLE>_BASE_URL > 预设 base_url（provider=mock 时无预设）
    api_key:  <ROLE>_API_KEY  > 共享 API_KEY
    model:    <ROLE>_MODEL    > 预设推荐模型名

微调模型旁路：``reasoning_base_url`` 填本地 vLLM 端口后，推理专家
走微调模型，其余角色继续在线 API——典型混合部署形态。
"""

from __future__ import annotations

from harness_agent.config.settings import Settings, get_settings
from harness_agent.contracts.llm import LLMClient, LLMRole
from harness_agent.llm.mock import MockLLMClient
from harness_agent.llm.openai_compat import OpenAICompatClient
from harness_agent.llm.providers import get_preset

__all__ = ["build_llm_client", "build_llm_clients", "describe_llm_setup"]


def build_llm_client(role: LLMRole, settings: Settings | None = None) -> LLMClient:
    """按配置为指定角色装配 LLM 客户端（mock / 在线 API 可切换）。"""
    if settings is None:
        settings = get_settings()
    llm = settings.llm
    prefix = role.upper()

    base_url = getattr(llm, f"{role}_base_url", "")
    api_key = getattr(llm, f"{role}_api_key", "") or llm.api_key
    model = getattr(llm, f"{role}_model", "")
    timeout = getattr(llm, f"{role}_timeout_s", 30.0)

    # provider=mock：恒走 MockLLMClient（零依赖演示模式）
    if llm.provider == "mock":
        return MockLLMClient(role=role)

    # provider 预设：补齐未显式覆盖的字段（全部非 mock 服务商均有预设）
    preset = get_preset(llm.provider)
    base_url = base_url or preset.base_url
    model = model or getattr(preset, f"{role}_model")

    if not base_url:
        raise ValueError(
            f"LLM 端点未配置（role={role}）：填写 HARNESS_LLM__{prefix}_BASE_URL，"
            f"或把 HARNESS_LLM__PROVIDER 设为预设服务商"
            f"（deepseek/qwen/zhipu/moonshot/openai/siliconflow）"
        )
    if not model:
        raise ValueError(
            f"LLM 模型名未指定（role={role}）：填写 HARNESS_LLM__{prefix}_MODEL或使用 provider 预设"
        )
    if not api_key:
        raise ValueError(
            f"LLM API Key 未配置（role={role}）：填写 HARNESS_LLM__API_KEY"
            f"（共享）或 HARNESS_LLM__{prefix}_API_KEY（独立）"
        )

    return OpenAICompatClient(
        role,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout,
    )


def build_llm_clients(settings: Settings | None = None) -> dict[LLMRole, LLMClient]:
    """一次装配四个角色的客户端（orchestrator / reasoning / judge / router）。"""
    return {
        role: build_llm_client(role, settings)
        for role in ("orchestrator", "reasoning", "judge", "router")
    }


def describe_llm_setup(settings: Settings | None = None) -> str:
    """人话描述当前 LLM 接线形态（启动横幅 / 白盒日志用）。"""
    if settings is None:
        settings = get_settings()
    llm = settings.llm
    if llm.provider == "mock":
        return "Mock 模式（脚本化应答，零外部依赖）"
    preset = get_preset(llm.provider)
    lines = [f"在线调用模式：provider={llm.provider}", f"  端点: {preset.base_url}"]
    for role in ("orchestrator", "reasoning", "judge", "router"):
        bypass = getattr(llm, f"{role}_base_url", "")
        base_url = bypass or preset.base_url
        model = getattr(llm, f"{role}_model", "") or getattr(preset, f"{role}_model")
        tag = "（微调模型旁路）" if bypass else ""
        lines.append(f"  {role:13s} → {model}{tag} @ {base_url}")
    return "\n".join(lines)
