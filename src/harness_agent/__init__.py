"""harness-agent：Harness Engineering 范式的医疗多智能体系统。

主 Agent 纯编排无应答权；路由 fail-closed；证据与记忆经 MCP 供给；
临床结论全量过质量门禁。详见 README 与 docs/development-plan.md。
"""

from harness_agent.config.settings import Settings, get_settings, reset_settings

__version__ = "0.1.0"

__all__ = ["Settings", "get_settings", "reset_settings", "__version__"]
