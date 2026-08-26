"""LLM 客户端实现包。

- ``mock``: 脚本化应答（demo / 测试默认，零外部依赖）
- ``openai_compat``: vLLM / OpenAI 兼容端点客户端（M5 引入，extras=llm）
- ``providers``: 在线服务商预设表（deepseek/qwen/zhipu/moonshot 等）
- ``wiring``: 配置 → 客户端装配工厂（在线调用模式入口）
"""

from harness_agent.llm.mock import MockLLMClient
from harness_agent.llm.openai_compat import OpenAICompatClient
from harness_agent.llm.providers import PROVIDER_PRESETS, get_preset
from harness_agent.llm.wiring import build_llm_client, build_llm_clients, describe_llm_setup

__all__ = [
    "MockLLMClient",
    "OpenAICompatClient",
    "PROVIDER_PRESETS",
    "build_llm_client",
    "build_llm_clients",
    "describe_llm_setup",
    "get_preset",
]
