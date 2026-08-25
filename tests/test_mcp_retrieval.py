"""M3 MCP 封装测试：工具面注册 / 检索与入库接线 / 闸门语义外显。

用桩 FastMCP 替身验证工具注册与调用链（不依赖真实协议层）；
fastmcp 未安装时另验证 ImportError 安装指引。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from harness_agent.mcp.retrieval import create_retrieval_mcp_server
from harness_agent.models.evidence import EvidencePack

_FASTMCP_INSTALLED = importlib.util.find_spec("fastmcp") is not None


class _StubFastMCP:
    """记录 @tool 注册的函数（验证接线，不跑真实协议层）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable] = {}

    def tool(self, fn: Callable) -> Callable:
        self.tools[fn.__name__] = fn
        return fn

    def run(self) -> None:  # pragma: no cover（真实运行入口，测试不触达）
        ...


def _make_server(monkeypatch: pytest.MonkeyPatch) -> _StubFastMCP:
    """注入桩 fastmcp 模块并创建检索 MCP server。"""
    created: list[_StubFastMCP] = []
    stub = types.ModuleType("fastmcp")

    def _factory(name: str) -> _StubFastMCP:
        server = _StubFastMCP(name)
        created.append(server)
        return server

    stub.FastMCP = _factory
    monkeypatch.setitem(sys.modules, "fastmcp", stub)
    create_retrieval_mcp_server()
    return created[0]


class TestCreateServer:
    def test_missing_dependency_hint(self):
        if _FASTMCP_INSTALLED:
            pytest.skip("fastmcp 已安装")
        with pytest.raises(ImportError, match="mcp"):
            create_retrieval_mcp_server()

    def test_tools_registered(self, monkeypatch: pytest.MonkeyPatch):
        server = _make_server(monkeypatch)
        assert server.name == "harness-retrieval"
        assert set(server.tools) == {"retrieve", "index_chunks"}


class TestRetrieveTool:
    def test_returns_reviewed_evidence_pack_json(self, monkeypatch: pytest.MonkeyPatch):
        server = _make_server(monkeypatch)
        server.tools["index_chunks"](
            [
                {"chunk_id": "kb-1", "content": "阿奇霉素的适应证与用法", "patient_id": None},
                {"chunk_id": "kb-2", "content": "血糖监测的目标范围", "patient_id": None},
            ]
        )
        raw = server.tools["retrieve"]("阿奇霉素 适应证", patient_id="pat-003", top_k=3)
        pack = EvidencePack.model_validate_json(raw)
        assert pack.is_reviewed is True
        assert any("阿奇霉素" in e.content for e in pack.evidence)

    def test_allergy_block_visible_through_tool(self, monkeypatch: pytest.MonkeyPatch):
        """闸门拦截语义经 MCP 工具外显：allowed=false 即 fail-closed。"""
        server = _make_server(monkeypatch)
        raw = server.tools["retrieve"]("青霉素类怎么用", patient_id="pat-001")
        pack = EvidencePack.model_validate_json(raw)
        assert pack.is_reviewed is False
        assert pack.evidence == []
        assert "penicillin" in pack.blocked_drugs


class TestIndexChunksTool:
    def test_indexes_and_returns_count(self, monkeypatch: pytest.MonkeyPatch):
        server = _make_server(monkeypatch)
        result = server.tools["index_chunks"](
            [
                {"chunk_id": "kb-1", "content": "糖尿病饮食控制要点", "patient_id": None},
                {"chunk_id": "kb-2", "content": "糖尿病运动干预建议", "patient_id": None},
            ]
        )
        assert "2" in result
        # 入库后立即可检索（同一栈内存生效）
        raw = server.tools["retrieve"]("糖尿病 饮食", patient_id="pat-003")
        pack = EvidencePack.model_validate_json(raw)
        assert pack.is_reviewed is True
        assert len(pack.evidence) >= 1

    def test_invalid_chunk_rejected_by_model(self, monkeypatch: pytest.MonkeyPatch):
        """缺失必填字段的 chunk 被模型校验拒绝（入库侧 fail-closed）。"""
        server = _make_server(monkeypatch)
        with pytest.raises(ValidationError):
            server.tools["index_chunks"]([{"chunk_id": "kb-1"}])
