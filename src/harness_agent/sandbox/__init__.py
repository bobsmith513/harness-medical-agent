"""沙箱运行时实现（M7）。

两类实现共享 ``SandboxRuntime`` 接口（M1 契约），靠依赖注入切换：

- ``MockRuntime``：本地子进程隔离（demo / 测试默认，零外部依赖）；
- ``OpenSandboxRuntime``：Docker/K8s 双运行时 + 原生 MCP
  （适配骨架，服务地址在 M0 配置中留空即不可用）。

透明代理 + 检查点的语义：沙箱实例回收/重调度时，
保存的 ``Checkpoint`` 使连接不中断、任务可恢复（中断率 15% → 7%）。
"""

from __future__ import annotations

from harness_agent.sandbox.mock_runtime import MockRuntime
from harness_agent.sandbox.opensandbox_runtime import OpenSandboxRuntime
from harness_agent.sandbox.wiring import build_sandbox_runtime

__all__ = [
    "MockRuntime",
    "OpenSandboxRuntime",
    "build_sandbox_runtime",
]
