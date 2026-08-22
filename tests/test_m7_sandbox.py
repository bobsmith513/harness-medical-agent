"""M7 沙箱运行时测试。

覆盖范围：
- MockRuntime：execute（正常/超时/异常）、checkpoint save/restore；
- OpenSandboxRuntime：骨架模式（地址留空降级）、检查点存储；
- 装配工厂：backend 选择、降级路径。
"""

from __future__ import annotations

from harness_agent.contracts.sandbox import Checkpoint
from harness_agent.sandbox import (
    MockRuntime,
    OpenSandboxRuntime,
    build_sandbox_runtime,
)


class TestMockRuntimeExecute:
    """MockRuntime execute 测试。"""

    def test_execute_simple_code(self):
        """执行简单 Python 代码 → exit_code=0 + stdout。"""
        runtime = MockRuntime()
        result = runtime.execute("print('hello world')")
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_execute_with_output(self):
        """执行带计算结果的代码。"""
        runtime = MockRuntime()
        result = runtime.execute("print(2 + 3)")
        assert result.exit_code == 0
        assert "5" in result.stdout

    def test_execute_with_artifacts(self):
        """执行产出文件的代码（stdout 包含产出）。"""
        runtime = MockRuntime()
        code = "print('artifact_content')"
        result = runtime.execute(code)
        assert result.exit_code == 0
        assert "artifact_content" in result.stdout

    def test_execute_error_code(self):
        """执行报错的代码 → exit_code=1 + stderr。"""
        runtime = MockRuntime()
        result = runtime.execute("raise ValueError('test error')")
        assert result.exit_code != 0
        assert "test error" in result.stderr

    def test_execute_timeout(self):
        """超时 → exit_code=-1 + stderr 记录超时。"""
        runtime = MockRuntime()
        result = runtime.execute("import time; time.sleep(10)", timeout_s=0.5)
        assert result.exit_code == -1
        assert "超时" in result.stderr

    def test_execute_non_python_language(self):
        """非 Python 语言 → exit_code=1 + stderr。"""
        runtime = MockRuntime()
        result = runtime.execute("console.log('test')", language="javascript")
        assert result.exit_code == 1
        assert "仅支持 Python" in result.stderr

    def test_execute_captures_stderr(self):
        """stderr 被正确捕获。"""
        runtime = MockRuntime()
        result = runtime.execute("import sys; sys.stderr.write('error_msg')")
        assert result.exit_code == 0
        assert "error_msg" in result.stderr


class TestMockRuntimeCheckpoint:
    """MockRuntime checkpoint save/restore 测试。"""

    def test_save_checkpoint(self):
        """保存检查点 → 返回 Checkpoint。"""
        runtime = MockRuntime()
        cp = runtime.save_checkpoint("sess-1", {"step": "1", "data": "abc"})
        assert cp.session_id == "sess-1"
        assert cp.state["step"] == "1"
        assert cp.state["data"] == "abc"

    def test_restore_checkpoint(self):
        """恢复检查点 → 返回 True。"""
        runtime = MockRuntime()
        cp = runtime.save_checkpoint("sess-1", {"step": "2"})
        assert runtime.restore(cp) is True

    def test_get_checkpoint(self):
        """查询检查点。"""
        runtime = MockRuntime()
        cp = runtime.save_checkpoint("sess-1", {"step": "3"})
        found = runtime.get_checkpoint("sess-1", cp.checkpoint_id)
        assert found is not None
        assert found.state["step"] == "3"

    def test_list_checkpoints(self):
        """列出会话的全部检查点。"""
        runtime = MockRuntime()
        runtime.save_checkpoint("sess-1", {"step": "1"})
        runtime.save_checkpoint("sess-1", {"step": "2"})
        runtime.save_checkpoint("sess-2", {"step": "1"})
        assert len(runtime.list_checkpoints("sess-1")) == 2
        assert len(runtime.list_checkpoints("sess-2")) == 1

    def test_restore_external_checkpoint(self):
        """恢复外部检查点（非本进程创建）→ 模拟 OpenSandbox 重调度。"""
        runtime = MockRuntime()
        external_cp = Checkpoint(session_id="sess-1", state={"step": "external"})
        assert runtime.restore(external_cp) is True

    def test_checkpoint_isolation(self):
        """不同会话的检查点隔离。"""
        runtime = MockRuntime()
        cp1 = runtime.save_checkpoint("sess-1", {"data": "s1"})
        cp2 = runtime.save_checkpoint("sess-2", {"data": "s2"})
        assert runtime.get_checkpoint("sess-1", cp1.checkpoint_id).state["data"] == "s1"
        assert runtime.get_checkpoint("sess-2", cp2.checkpoint_id).state["data"] == "s2"


class TestOpenSandboxRuntime:
    """OpenSandboxRuntime 骨架测试。"""

    def test_skeleton_mode_without_url(self):
        """地址留空 → 降级结果。"""
        runtime = OpenSandboxRuntime(service_url="")
        assert runtime.available is False
        result = runtime.execute("print('test')")
        assert result.exit_code == -1
        assert "未配置" in result.stderr

    def test_skeleton_mode_checkpoint(self):
        """骨架模式检查点存储。"""
        runtime = OpenSandboxRuntime(service_url="")
        cp = runtime.save_checkpoint("sess-1", {"step": "1"})
        assert cp.session_id == "sess-1"
        assert runtime.restore(cp) is True

    def test_backend_identifier(self):
        """backend 标识为 opensandbox。"""
        runtime = OpenSandboxRuntime(service_url="")
        assert runtime.backend == "opensandbox"

    def test_with_url_still_skeleton(self):
        """填写地址但无 SDK → execute 返回骨架结果。"""
        runtime = OpenSandboxRuntime(service_url="http://fake:8080")
        assert runtime.available is True
        # 骨架模式：不真正连接服务
        result = runtime.execute("print('test')")
        assert result.exit_code == 0
        assert "骨架" in result.stdout


class TestBuildSandboxRuntime:
    """装配工厂测试。"""

    def test_default_returns_mock(self):
        runtime = build_sandbox_runtime()
        assert isinstance(runtime, MockRuntime)

    def test_mock_backend(self):
        runtime = build_sandbox_runtime(backend="mock")
        assert isinstance(runtime, MockRuntime)

    def test_opensandbox_backend_without_url(self):
        runtime = build_sandbox_runtime(backend="opensandbox")
        assert isinstance(runtime, OpenSandboxRuntime)
        assert runtime.available is False

    def test_opensandbox_backend_with_url(self):
        runtime = build_sandbox_runtime(
            backend="opensandbox",
            opensandbox_url="http://fake:8080",
        )
        assert isinstance(runtime, OpenSandboxRuntime)
        assert runtime.available is True


# ---------------------------------------------------------------------------
# 验收：沙箱检查点中断恢复 demo
# ---------------------------------------------------------------------------
class TestCheckpointRecoveryDemo:
    """验收：沙箱检查点中断恢复 demo。"""

    def test_checkpoint_save_and_restore_flow(self):
        """完整流程：执行 → 保存检查点 → 恢复检查点 → 继续执行。"""
        runtime = MockRuntime()

        # 1. 执行初始代码
        result1 = runtime.execute("print('step 1 done')")
        assert result1.exit_code == 0

        # 2. 保存检查点（模拟中断前的状态快照）
        cp = runtime.save_checkpoint(
            "sess-1",
            {
                "last_step": "1",
                "intermediate_result": "42",
            },
        )
        assert cp.state["last_step"] == "1"

        # 3. 模拟中断恢复
        assert runtime.restore(cp) is True

        # 4. 恢复后继续执行
        result2 = runtime.execute("print('step 2 done')")
        assert result2.exit_code == 0
        assert "step 2 done" in result2.stdout

    def test_multiple_checkpoints_timeline(self):
        """多检查点时间线（审计回溯）。"""
        runtime = MockRuntime()
        for i in range(3):
            runtime.save_checkpoint("sess-1", {"step": str(i), "state": f"checkpoint-{i}"})

        checkpoints = runtime.list_checkpoints("sess-1")
        assert len(checkpoints) == 3
        # 验证时间线完整性
        for i, cp in enumerate(checkpoints):
            assert cp.state["step"] == str(i)
