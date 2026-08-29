"""OpenSandboxRuntime：Docker/K8s 双运行时适配骨架（M7）。

服务地址在 M0 配置 ``HARNESS_SANDBOX__OPENSANDBOX_URL`` 中留空即不可用。
本类是**适配骨架**——实现了 ``SandboxRuntime`` 接口的完整方法签名，
但 ``execute`` 的真实 MCP 调用尚未实现（规划中），当前所有路径
均 fail-closed：地址留空返回降级错误，地址填写返回"未实现"错误。
生产部署需补全 MCP 协议调用后移除 fail-closed 占位。
"""

from __future__ import annotations

from typing import Literal

from harness_agent.contracts.sandbox import Checkpoint, ExecutionResult

__all__ = ["OpenSandboxRuntime"]

_Backend = Literal["mock", "opensandbox"]


class OpenSandboxRuntime:
    """OpenSandbox 适配骨架（Docker/K8s + 原生 MCP）。

    ``service_url`` 留空时全部方法返回降级结果（不连真实服务），
    填写后由 MCP 协议透明代理句柄与检查点。
    """

    backend: _Backend = "opensandbox"

    def __init__(self, service_url: str = "") -> None:
        self._service_url = service_url
        self._available = bool(service_url)
        # 句柄表：session_id -> sandbox_handle（透明代理）
        self._handles: dict[str, str] = {}
        # 检查点存储（骨架：进程内；真实部署落持久卷）
        self._checkpoints: dict[str, dict[str, Checkpoint]] = {}

    @property
    def available(self) -> bool:
        """服务是否可用（地址已配置）。"""
        return self._available

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout_s: float = 30.0,
    ) -> ExecutionResult:
        """在沙箱实例中执行代码（透明代理句柄）。

        地址留空时返回降级结果（骨架模式）。
        """
        if not self._available:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=(
                    "OpenSandbox 服务地址未配置"
                    "（HARNESS_SANDBOX__OPENSANDBOX_URL 留空），"
                    "请使用 MockRuntime 或填写服务地址"
                ),
            )
        # 适配器未实现真实调用（规划中）：fail-closed，不假装成功
        return ExecutionResult(
            exit_code=-1,
            stdout="",
            stderr="OpenSandbox 适配器未实现真实 MCP 调用（规划中），请使用 MockRuntime",
        )

    def save_checkpoint(self, session_id: str, state: dict[str, str]) -> Checkpoint:
        """保存检查点到持久卷（骨架：进程内存储）。"""
        cp = Checkpoint(session_id=session_id, state=state)
        self._checkpoints.setdefault(session_id, {})[cp.checkpoint_id] = cp
        return cp

    def restore(self, checkpoint: Checkpoint) -> bool:
        """从持久卷恢复检查点（透明代理句柄不变）。"""
        sid = checkpoint.session_id
        self._checkpoints.setdefault(sid, {})[checkpoint.checkpoint_id] = checkpoint
        return True
