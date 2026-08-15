"""沙箱契约（M1）。

代码执行隔离（检验计算、剂量换算等）与检查点恢复的统一接口：

- MockRuntime：本地子进程隔离（demo 模式默认）；
- OpenSandboxRuntime：Docker/K8s 双运行时 + 原生 MCP（M7 适配骨架，
  服务地址在 M0 配置中留空）。

透明代理 + 检查点的语义：沙箱实例回收/重调度时，
保存的 ``Checkpoint`` 使连接不中断、任务可恢复（中断率 15% -> 7%）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from harness_agent.models.common import new_id, now_utc

__all__ = [
    "Checkpoint",
    "ExecutionResult",
    "SandboxBackend",
    "SandboxRuntime",
]

#: 沙箱后端标识（与 M0 配置 HARNESS_SANDBOX__BACKEND 对应）。
SandboxBackend = Literal["mock", "opensandbox"]


class ExecutionResult(BaseModel):
    """沙箱执行结果（检验计算、剂量换算的产出载体）。"""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    #: 产出文件（文件名 -> 内容），供后续装配引用
    artifacts: dict[str, str] = Field(default_factory=dict)
    executed_at: datetime = Field(default_factory=now_utc)


class Checkpoint(BaseModel):
    """沙箱检查点：任务状态快照（实例回收/重调度后恢复依据）。

    ``state`` 为序列化任务状态（脱敏边界延伸至此——M7 的脱敏中间件
    在写检查点前执行）。
    """

    checkpoint_id: str = Field(default_factory=lambda: new_id("cp"))
    session_id: str
    state: dict[str, str] = Field(default_factory=dict)
    saved_at: datetime = Field(default_factory=now_utc)


@runtime_checkable
class SandboxRuntime(Protocol):
    """沙箱运行时统一接口。"""

    backend: SandboxBackend

    def execute(
        self, code: str, language: str = "python", timeout_s: float = 30.0
    ) -> ExecutionResult: ...

    def save_checkpoint(self, session_id: str, state: dict[str, str]) -> Checkpoint: ...

    def restore(self, checkpoint: Checkpoint) -> bool: ...
