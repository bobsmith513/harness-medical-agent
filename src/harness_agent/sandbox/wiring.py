"""沙箱运行时装配工厂（M7）。

按 ``SandboxSettings.backend`` 选择实现：
- ``mock``：MockRuntime（本地子进程，零依赖默认）；
- ``opensandbox``：OpenSandboxRuntime（地址留空时降级为骨架模式）。
"""

from __future__ import annotations

from harness_agent.contracts.sandbox import SandboxRuntime
from harness_agent.sandbox.mock_runtime import MockRuntime
from harness_agent.sandbox.opensandbox_runtime import OpenSandboxRuntime

__all__ = ["build_sandbox_runtime"]


def build_sandbox_runtime(
    backend: str = "mock",
    opensandbox_url: str = "",
) -> SandboxRuntime:
    """按配置装配沙箱运行时。

    ``backend="mock"`` 时返回 MockRuntime（零依赖）；
    ``backend="opensandbox"`` 时返回 OpenSandboxRuntime
    （地址留空时为骨架模式，不连真实服务）。
    """
    if backend == "opensandbox":
        return OpenSandboxRuntime(service_url=opensandbox_url)
    return MockRuntime()
