"""可观测与合规契约（M1）。

全链路可观测 + 审计 + 脱敏的接口切分（M7 实现）：

- ``Tracer``:      事件写入（Langfuse 缺省时 NoopTracer，密钥留空自动降级）
- ``AuditStore``:  审计持久化（PostgreSQL 缺省时降级 SQLite，DSN 留空）
- ``CacheStore`` / ``DistLock``: 缓存与分布式锁（Redis 缺省时降级进程内存）
- ``Desensitizer``: 出站脱敏（调用外部模型 API 前去除患者标识，
                    边界延伸至沙箱检查点与 Redis 缓存）

所有降级实现与真实实现共用本文件接口，配置切换零逻辑分叉。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from harness_agent.models.audit import AuditRecord, TraceEvent

__all__ = [
    "AuditStore",
    "CacheStore",
    "DesensitizedText",
    "Desensitizer",
    "DistLock",
    "Tracer",
]


class DesensitizedText(BaseModel):
    """脱敏结果：净化文本 + 被移除的实体（供审计与召回对账）。"""

    text: str
    removed_entities: list[str] = []


@runtime_checkable
class Tracer(Protocol):
    """全链路 trace 写入（关联 session_id / trace_id）。"""

    def bind(self, session_id: str, trace_id: str) -> None: ...

    def record(self, event: TraceEvent) -> None: ...


@runtime_checkable
class AuditStore(Protocol):
    """审计存储：append-only 写入 + 按会话查询。"""

    def append(self, record: AuditRecord) -> None: ...

    def query(self, session_id: str) -> list[AuditRecord]: ...


@runtime_checkable
class CacheStore(Protocol):
    """缓存（会话缓存等；Redis 或进程内存实现）。"""

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...


@runtime_checkable
class DistLock(Protocol):
    """分布式锁（Redis 或进程内互斥实现）。"""

    def acquire(self, key: str, ttl_s: float) -> bool: ...

    def release(self, key: str) -> None: ...


@runtime_checkable
class Desensitizer(Protocol):
    """出站脱敏中间件：外部调用前的患者标识去除。"""

    def desensitize(self, text: str) -> DesensitizedText: ...
