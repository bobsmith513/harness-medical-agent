"""Redis 可选依赖统一入口（M7 基础设施）。

此前 ``cache_store.py`` 与 ``dist_lock.py`` 各自重复实现同一种
「import redis → ImportError → 降级进程内实现」模式；本模块收口为
单一 ``try_redis_client``，供两者共用（行为不变，来源唯一）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["try_redis_client"]

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查用
    from redis import Redis


def try_redis_client(url: str) -> Redis | None:
    """尝试按 URL 构建 redis 客户端；redis 未安装返回 None。

    调用方约定：返回 None 即降级到进程内实现（零依赖默认路径），
    绝不静默假装已连接。
    """
    if not url:
        return None
    try:
        import redis  # type: ignore[import-untyped]
    except ImportError:
        return None
    return redis.from_url(url)
