"""在线 LLM 服务商预设表（在线调用模式）。

把 ``HARNESS_LLM__PROVIDER`` 设为下表任一服务商名，再填一个
``HARNESS_LLM__API_KEY``，路由 / 推理 / judge 三类模型即全部
切换为在线 API 调用（OpenAI 兼容协议），无需填写任何 base_url。

预设只提供「端点 + 推荐模型名」的默认值，逐角色仍可用
``HARNESS_LLM__<ROLE>_BASE_URL / _MODEL`` 覆盖——例如推理专家
指向本地 vLLM 部署的 SFT+DPO 微调模型，其余角色走在线 API。
推荐模型名是编写时的快照（2026-08 核对），以各服务商官网当期在售
型号为准；下架或更名时用 ``_MODEL`` 覆盖即可，预设只是默认值而非
功能依赖。**新增/修改预设前请核对官网模型列表**——把已下线型号写成
默认值会让"填两行 .env 即可运行"直接变成 404。

各服务商端点均为 OpenAI 兼容协议（``{base_url}/chat/completions``）。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProviderPreset", "PROVIDER_PRESETS", "get_preset"]


@dataclass(frozen=True)
class ProviderPreset:
    """一个在线服务商预设：端点 + 各角色推荐模型。"""

    base_url: str
    #: 推理专家推荐模型（在线替代微调基座时使用）
    reasoning_model: str
    #: 质量门禁 judge 推荐模型（忠实度校验，非推理任务）
    judge_model: str
    #: 路由器兜底推荐模型（二值分类，轻量即可）
    router_model: str
    #: 主 Agent 编排推荐模型（纯编排无应答权）
    orchestrator_model: str
    #: 服务商申请密钥的入口（提示用）
    key_portal: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    # 深度求索：deepseek-v4 系列，性价比高，中文医疗语料表现好
    "deepseek": ProviderPreset(
        base_url="https://api.deepseek.com/v1",
        reasoning_model="deepseek-v4-pro",
        judge_model="deepseek-v4-flash",
        router_model="deepseek-v4-flash",
        orchestrator_model="deepseek-v4-pro",
        key_portal="https://platform.deepseek.com/",
    ),
    # 阿里云百炼（DashScope 兼容模式）：qwen3 系列，同时托管 deepseek/kimi/glm
    "qwen": ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        reasoning_model="qwen3.8-max",
        judge_model="qwen3.8-flash",
        router_model="qwen3.8-flash",
        orchestrator_model="qwen3.7-plus",
        key_portal="https://bailian.console.aliyun.com/",
    ),
    # 智谱 GLM：glm-4.6 / glm-4.5-air
    "zhipu": ProviderPreset(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        reasoning_model="glm-4.6",
        judge_model="glm-4.5-air",
        router_model="glm-4.5-air",
        orchestrator_model="glm-4.6",
        key_portal="https://open.bigmodel.cn/",
    ),
    # 月之暗面 Kimi：k3 旗舰 + k2.6 轻量档，256K 长上下文
    # ⚠ 已下线的型号不要写进预设：kimi-k2.5 与 moonshot-v1 系列于
    # 2026-08-31 全平台下线（此前已停止向新注册用户开放），
    # kimi-k2 系列更早于 2026-05-25 下线。
    "moonshot": ProviderPreset(
        base_url="https://api.moonshot.cn/v1",
        reasoning_model="kimi-k3",
        judge_model="kimi-k2.6",
        router_model="kimi-k2.6",
        orchestrator_model="kimi-k3",
        key_portal="https://platform.moonshot.cn/",
    ),
    # OpenAI 官方：gpt-4o 系列
    "openai": ProviderPreset(
        base_url="https://api.openai.com/v1",
        reasoning_model="gpt-4o",
        judge_model="gpt-4o-mini",
        router_model="gpt-4o-mini",
        orchestrator_model="gpt-4o",
        key_portal="https://platform.openai.com/api-keys",
    ),
    # 硅基流动：模型聚合平台，托管多家开源模型
    "siliconflow": ProviderPreset(
        base_url="https://api.siliconflow.cn/v1",
        reasoning_model="deepseek-ai/DeepSeek-V3",
        judge_model="Qwen/Qwen2.5-72B-Instruct",
        router_model="Qwen/Qwen2.5-7B-Instruct",
        orchestrator_model="deepseek-ai/DeepSeek-V3",
        key_portal="https://cloud.siliconflow.cn/",
    ),
}


def get_preset(provider: str) -> ProviderPreset:
    """按服务商名取预设；未知名称抛 ValueError（列出全部可选项）。"""
    try:
        return PROVIDER_PRESETS[provider]
    except KeyError as exc:
        available = ", ".join(sorted(PROVIDER_PRESETS))
        raise ValueError(f"未知的 LLM provider: {provider!r}（可选: {available}）") from exc
