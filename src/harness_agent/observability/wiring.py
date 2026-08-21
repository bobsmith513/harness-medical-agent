"""可观测组件装配工厂（M7）。

一站式装配 Tracer + Desensitizer + AuditStore + CacheStore + DistLock，
按 ``ObservabilitySettings`` 配置自动选择实现（留空降级）。
"""

from __future__ import annotations

from dataclasses import dataclass

from harness_agent.contracts.observability import (
    AuditStore,
    CacheStore,
    Desensitizer,
    DistLock,
    Tracer,
)
from harness_agent.observability.audit_store import build_audit_store
from harness_agent.observability.cache_store import build_cache_store
from harness_agent.observability.desensitizer import PatternDesensitizer
from harness_agent.observability.dist_lock import build_dist_lock
from harness_agent.observability.tracer import build_tracer

__all__ = ["ObservabilityStack", "build_observability_stack"]


@dataclass(frozen=True)
class ObservabilityStack:
    """可观测组件栈（统一装配入口）。"""

    tracer: Tracer
    desensitizer: Desensitizer
    audit_store: AuditStore
    cache_store: CacheStore
    dist_lock: DistLock


def build_observability_stack(
    *,
    langfuse_public_key: str = "",
    langfuse_secret_key: str = "",
    langfuse_host: str = "",
    audit_dsn: str = "",
    redis_url: str = "",
    data_dir: str = ".data",
) -> ObservabilityStack:
    """按配置装配可观测组件栈。

    全部留空时降级为：
    - NoopTracer（仅打印事件）
    - PatternDesensitizer（正则脱敏）
    - SQLiteAuditStore（落 data_dir）
    - MemoryCacheStore（进程内）
    - MemoryLock（进程内互斥）
    """
    return ObservabilityStack(
        tracer=build_tracer(langfuse_public_key, langfuse_secret_key, langfuse_host),
        desensitizer=PatternDesensitizer(),
        audit_store=build_audit_store(audit_dsn, data_dir),
        cache_store=build_cache_store(redis_url),
        dist_lock=build_dist_lock(redis_url),
    )
