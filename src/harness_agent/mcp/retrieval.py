"""检索供给 MCP 封装（M3）：HybridRetrievalService → MCP 工具面。

运行（stdio 传输，供编排层或外部 Agent 挂载）::

    uv run harness-mcp-retrieval

工具面（闸门语义外显，调用方按 ``is_reviewed`` 处理 fail-closed）：

- ``retrieve``:      查询文本 + patient_id → 证据包 JSON
  （``assembly_gate.allowed=false`` 即拦截：输入闸门命中过敏药、
  或装配复核拒绝交付，调用方必须转澄清 / 人工）；
- ``index_chunks``:  chunk 列表 → 双路入库（patient_id=None 为
  共享知识库，非 None 为患者记忆分区）。

fastmcp 为可选依赖（``uv sync --extra mcp``），核心包零依赖不受影响。
"""

from __future__ import annotations

from typing import Any

from harness_agent.config.settings import Settings
from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
from harness_agent.retrieval.wiring import build_retrieval_stack

__all__ = ["create_retrieval_mcp_server", "main"]


def create_retrieval_mcp_server(settings: Settings | None = None) -> Any:
    """构建检索 MCP server（配置驱动的完整供给栈 + 工具注册）。"""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "fastmcp 未安装：uv sync --extra mcp 后重试（核心检索功能不依赖本封装）"
        ) from exc

    stack = build_retrieval_stack(settings)
    mcp: Any = FastMCP("harness-retrieval")

    @mcp.tool
    def retrieve(
        query: str,
        patient_id: str,
        top_k: int = 5,
        session_id: str = "",
    ) -> str:
        """混合检索：查询 + 患者 → 证据包 JSON（含三道闸门裁决）。

        返回 EvidencePack：``assembly_gate.allowed=false`` 表示被闸门
        拦截（输入提及过敏药 / 装配复核拒绝），调用方须 fail-closed。
        """
        pack = stack.service.retrieve(
            RetrievalQuery(text=query, patient_id=patient_id, top_k=top_k, session_id=session_id)
        )
        return pack.model_dump_json()

    @mcp.tool
    def index_chunks(chunks: list[dict[str, Any]]) -> str:
        """双路入库：chunk 列表 → 索引（content 必填，patient_id 分区键）。"""
        items = [StoredChunk.model_validate(chunk) for chunk in chunks]
        stack.service.index(items)
        return f'{{"indexed": {len(items)}}}'

    return mcp


def main() -> None:
    """命令行入口：``harness-mcp-retrieval``（stdio 传输）。"""
    mcp = create_retrieval_mcp_server()
    mcp.run()


if __name__ == "__main__":
    main()
