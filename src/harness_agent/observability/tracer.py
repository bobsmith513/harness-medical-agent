"""Tracer 实现（M7）：NoopTracer + LangfuseTracer 骨架。

- ``NoopTracer``：Langfuse 密钥留空时自动降级，仅打印事件到 stdout；
- ``LangfuseTracer``：密钥填写后通过 Langfuse SDK 写入
  （本类为骨架，真实部署需安装 langfuse 包）。

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
    """Langfuse Tracer 骨架（密钥填写后通过 SDK 写入）。

    生产部署需安装 ``langfuse`` 包：
        pip install langfuse

    本类实现 ``Tracer`` 接口完整签名，但在未安装 SDK 时
    自动降级为打印模式（与 NoopTracer 等价）。
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

        # 尝试初始化 Langfuse 客户端
        if public_key and secret_key:
            try:
                from langfuse import Langfuse  # type: ignore[import-untyped]

                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host or "https://cloud.langfuse.com",
                )
            except ImportError:
                # langfuse 包未安装 → 降级为 Noop
                pass

    def bind(self, session_id: str, trace_id: str) -> None:
        self._session_id = session_id
        self._trace_id = trace_id
        if self._client is None:
            self._fallback.bind(session_id, trace_id)

    def record(self, event: TraceEvent) -> None:
        if self._client is not None:
            # 真实部署：通过 Langfuse SDK 写入
            self._client.trace(
                id=event.trace_id,
                session_id=event.session_id,
                name=event.event_type,
                metadata=event.payload,
            )
        else:
            self._fallback.record(event)


def build_tracer(
    langfuse_public_key: str = "",
    langfuse_secret_key: str = "",
    langfuse_host: str = "",
) -> Tracer:
    """按配置装配 Tracer。

    Langfuse 密钥留空时返回 NoopTracer（仅打印）；
    填写时返回 LangfuseTracer（SDK 未安装自动降级）。
    """
    if not langfuse_public_key or not langfuse_secret_key:
        return NoopTracer()
    return LangfuseTracer(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host,
    )
