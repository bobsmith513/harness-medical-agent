"""可观测与合规实现（M7）。

全部实现与 ``contracts/observability.py`` 接口对齐，靠依赖注入切换：

- ``NoopTracer`` / ``LangfuseTracer``：全链路 trace 写入
  （Langfuse 密钥留空时自动降级 Noop，仅打印事件）；
- ``PatternDesensitizer``：出站脱敏中间件
  （正则匹配患者标识：身份证 / 手机号 / 患者编号 / 邮箱，
  替换为 ``[REDACTED-类型]`` 占位符）；
- ``SQLiteAuditStore`` / ``PostgresAuditStore``：审计持久化
  （PostgreSQL DSN 留空时降级 SQLite，落 ``app.data_dir``）；
- ``MemoryCacheStore`` / ``RedisCacheStore``：缓存
  （Redis URL 留空时降级进程内存）；
- ``MemoryLock`` / ``RedisLock``：分布式锁
  （Redis URL 留空时降级进程内互斥）。

所有降级实现与真实实现共用同一接口，配置切换零逻辑分叉。
"""

from __future__ import annotations

from harness_agent.observability.audit_store import (
    PostgresAuditStore,
    SQLiteAuditStore,
    build_audit_store,
)
from harness_agent.observability.cache_store import (
    MemoryCacheStore,
    RedisCacheStore,
    build_cache_store,
)
from harness_agent.observability.desensitizer import PatternDesensitizer
from harness_agent.observability.dist_lock import (
    MemoryLock,
    RedisLock,
    build_dist_lock,
)
from harness_agent.observability.tracer import (
    LangfuseTracer,
    NoopTracer,
    build_tracer,
)
from harness_agent.observability.wiring import (
    ObservabilityStack,
    build_observability_stack,
)

__all__ = [
    # tracer
    "NoopTracer",
    "LangfuseTracer",
    "build_tracer",
    # desensitizer
    "PatternDesensitizer",
    # audit
    "SQLiteAuditStore",
    "PostgresAuditStore",
    "build_audit_store",
    # cache
    "MemoryCacheStore",
    "RedisCacheStore",
    "build_cache_store",
    # lock
    "MemoryLock",
    "RedisLock",
    "build_dist_lock",
    # wiring
    "ObservabilityStack",
    "build_observability_stack",
]
