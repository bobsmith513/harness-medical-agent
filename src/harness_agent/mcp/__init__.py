"""MCP 服务封装包：把 harness 各供给能力暴露为 MCP 工具。

- ``retrieval``: 检索供给层（M3）——查询进、证据包出，闸门裁决外显

fastmcp 为可选依赖（``uv sync --extra mcp``），未安装时构造函数
抛出带安装指引的 ImportError，核心包不受影响。
"""

from harness_agent.mcp.retrieval import create_retrieval_mcp_server

__all__ = ["create_retrieval_mcp_server"]
