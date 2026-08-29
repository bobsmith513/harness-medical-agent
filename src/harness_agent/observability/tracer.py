"""Tracer 实现（M7）：NoopTracer + LangfuseTracer 骨架。

- ``NoopTracer``：Langfuse 密钥留空时自动降级，仅打印事件到 stdout；
- ``LangfuseTracer``：密钥填写后初始化 Langfuse SDK，
  但 v3+ 集成尚未实现（``record`` 降级为 Noop 打印）。

两者共用 ``Tracer`` 接口（M1 契约），配置切换零逻辑分叉。
"""

from __future__ import annotations

import sys

from harness_agent.contracts.observability import Tracer
from harness_agent.models.audit import TraceEvent

__all__ = ["NoopTracer", "LangfuseTracer", "build_tracer"]


class NoopTracer:
    """Noop Tracer：仅打印事件到 stdout（Langfuse 缺省时降级）。

    全链路 trace 事件可打印——验收指标"全链路 trace 事件可打印"的核心实现。
    """

    def __init__(self, *, verbose: bool = True) -> None:
        self._verbose = verbose
        self._session_id: str | None = None
        self._trace_id: str | None = None
        self._events: list[TraceEvent] = []

    def bind(self, session_id: str, trace_id: str) -> None:
        """绑定会话与 trace 标识。"""
        self._session_id = session_id
        self._trace_id = trace_id
        if self._verbose:
            print(
                f"[TRACE] bind session={session_id} trace={trace_id}",
                file=sys.stderr,
            )

    def record(self, event: TraceEvent) -> None:
        """记录 trace 事件（打印到 stderr + 进程内缓存）。"""
        self._events.append(event)
        if self._verbose:
            print(
                f"[TRACE] {event.event_type} "
                f"trace={event.trace_id} "
                f"payload_keys={list(event.payload.keys())}",
                file=sys.stderr,
            )

    @property
    def events(self) -> list[TraceEvent]:
        """已记录的全部事件（测试用）。"""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)


class LangfuseTracer:
    """Langfuse Tracer 骨架（密钥填写后初始化 SDK，v3+ 集成待实现）。

    生产部署需安装 ``langfuse`` 包：
        pip install langfuse

    密钥填写但 langfuse 未安装时启动报错（对齐 design-decisions 降级承诺）。
    SDK 已安装时初始化客户端，但 v3+ ``record`` 适配尚未实现——
    事件降级为 Noop 打印（不丢失 trace 数据，但不写入 Langfuse）。
    """

    def __init__(
        self,
        public_key: str = "",
        secret_key: str = "",
        host: str = "",
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._client = None
        self._session_id: str | None = None
        self._trace_id: str | None = None
        self._fallback = NoopTracer()

        # 密钥填写时尝试初始化 Langfuse 客户端
        if public_key and secret_key:
            try:
                from langfuse import Langfuse  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "langfuse 包未安装：uv sync --extra all 后重试，"
                    "或清空 Langfuse 密钥使用 NoopTracer（零依赖默认）"
                ) from exc
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host or "https://cloud.langfuse.com",
            )

    def bind(self, session_id: str, trace_id: str) -> None:
        self._session_id = session_id
        self._trace_id = trace_id
        # v3+ bind 适配待实现，降级为 Noop 打印
        self._fallback.bind(session_id, trace_id)

    def record(self, event: TraceEvent) -> None:
        # v3+ record 适配待实现（client.trace() 为 v1/v2 已废弃 API），
        # 降级为 Noop 打印——不丢失 trace 事件，但不写入 Langfuse
        self._fallback.record(event)


def build_tracer(
    langfuse_public_key: str = "",
    langfuse_secret_key: str = "",
    langfuse_host: str = "",
) -> Tracer:
    """按配置装配 Tracer。

    Langfuse 密钥留空时返回 NoopTracer（仅打印）；
    填写时返回 LangfuseTracer（需安装 langfuse 包，否则启动报错）。
    """
    if not langfuse_public_key or not langfuse_secret_key:
        return NoopTracer()
    return LangfuseTracer(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host,
    )
