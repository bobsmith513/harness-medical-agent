"""MockRuntime：本地子进程隔离的沙箱运行时（M7）。

demo / 测试默认实现，零外部依赖：
- ``execute``：通过 ``subprocess`` 在隔离进程中执行 Python 代码，
  捕获 stdout/stderr + exit_code，超时杀进程；
- ``save_checkpoint`` / ``restore``：进程内字典存储检查点快照，
  模拟沙箱实例回收后的状态恢复。

与 ``OpenSandboxRuntime`` 共用 ``SandboxRuntime`` 接口，
配置切换零逻辑分叉。
"""

from __future__ import annotations

import subprocess
import sys

from harness_agent.contracts.sandbox import Checkpoint, ExecutionResult

__all__ = ["MockRuntime"]


class MockRuntime:
    """本地子进程沙箱（demo 模式默认）。

    - ``backend`` 固定为 ``"mock"``，与配置 ``HARNESS_SANDBOX__BACKEND=mock`` 对应；
    - ``execute`` 在子进程中执行代码，捕获 stdout/stderr + exit_code；
    - ``save_checkpoint`` / ``restore`` 进程内字典存储，模拟中断恢复。
    """

    backend: str = "mock"

    def __init__(self) -> None:
        # session_id → {checkpoint_id → Checkpoint}
        self._checkpoints: dict[str, dict[str, Checkpoint]] = {}

    def execute(
        self,
        code: str,
        language: str = "python",
        timeout_s: float = 30.0,
    ) -> ExecutionResult:
        """在子进程中执行代码，返回执行结果。

        - 超时 → 杀进程，exit_code=-1，stderr 记录超时；
        - 异常 → exit_code=1，stderr 记录异常信息；
        - 正常 → exit_code=0，stdout/stderr 透传。
        """
        if language != "python":
            return ExecutionResult(
                exit_code=1,
                stderr=f"MockRuntime 仅支持 Python，收到 language={language}",
            )

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return ExecutionResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"子进程超时（{timeout_s}s）",
            )
        except Exception as exc:
            return ExecutionResult(
                exit_code=1,
                stdout="",
                stderr=f"子进程异常: {exc}",
            )

    def save_checkpoint(self, session_id: str, state: dict[str, str]) -> Checkpoint:
        """保存检查点快照（进程内字典存储）。"""
        cp = Checkpoint(session_id=session_id, state=state)
        self._checkpoints.setdefault(session_id, {})[cp.checkpoint_id] = cp
        return cp

    def restore(self, checkpoint: Checkpoint) -> bool:
        """恢复检查点（进程内字典存储，模拟中断恢复）。"""
        sid = checkpoint.session_id
        self._checkpoints.setdefault(sid, {})[checkpoint.checkpoint_id] = checkpoint
        return True

    def get_checkpoint(self, session_id: str, checkpoint_id: str) -> Checkpoint | None:
        """查询检查点（测试与审计用）。"""
        return self._checkpoints.get(session_id, {}).get(checkpoint_id)

    def list_checkpoints(self, session_id: str) -> list[Checkpoint]:
        """列出会话的全部检查点（审计用）。"""
        return list(self._checkpoints.get(session_id, {}).values())
