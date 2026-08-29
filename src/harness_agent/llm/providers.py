"""在线 LLM 服务商预设表（在线调用模式）。

把 ``HARNESS_LLM__PROVIDER`` 设为下表任一服务商名，再填一个
``HARNESS_LLM__API_KEY``，路由 / 推理 / judge 三类模型即全部
切换为在线 API 调用（OpenAI 兼容协议），无需填写任何 base_url。

预设只提供「端点 + 推荐模型名」的默认值，逐角色仍可用
``HARNESS_LLM__<ROLE>_BASE_URL / _MODEL`` 覆盖——例如推理专家
指向本地 vLLM 部署的 SFT+DPO 微调模型，其余角色走在线 API。
推荐模型名只是**默认值**，不是功能依赖：各服务商型号会随版本更迭
上下架，**以官网当期在售列表为准**；型号不符时用
``HARNESS_LLM__<ROLE>_MODEL`` 覆盖即可。**修改预设前请对照官网核对
一遍**——把已下线或不存在的型号写成默认值，会让"填两行 .env 即可
运行"直接变成 404。

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
    # 深度求索：deepseek-reasoner 深度推理 / deepseek-chat 通用对话
    "deepseek": ProviderPreset(
        base_url="https://api.deepseek.com/v1",
        reasoning_model="deepseek-reasoner",
        judge_model="deepseek-chat",
        router_model="deepseek-chat",
        orchestrator_model="deepseek-chat",
        key_portal="https://platform.deepseek.com/",
    ),
    # 阿里云百炼（DashScope 兼容模式）：qwen-max 旗舰 / qwen-plus 均衡 / qwen-turbo 快
    "qwen": ProviderPreset(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        reasoning_model="qwen-max",
        judge_model="qwen-plus",
        router_model="qwen-turbo",
        orchestrator_model="qwen-plus",
        key_portal="https://bailian.console.aliyun.com/",
    ),
    # 智谱 GLM：glm-4-plus 旗舰 / glm-4-air 轻量
    "zhipu": ProviderPreset(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        reasoning_model="glm-4-plus",
        judge_model="glm-4-air",
        router_model="glm-4-air",
        orchestrator_model="glm-4-plus",
        key_portal="https://open.bigmodel.cn/",
    ),
    # 月之暗面 Moonshot：moonshot-v1 系列（8k / 32k / 128k 三档上下文）
    "moonshot": ProviderPreset(
        base_url="https://api.moonshot.cn/v1",
        reasoning_model="moonshot-v1-32k",
        judge_model="moonshot-v1-8k",
        router_model="moonshot-v1-8k",
        orchestrator_model="moonshot-v1-32k",
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
