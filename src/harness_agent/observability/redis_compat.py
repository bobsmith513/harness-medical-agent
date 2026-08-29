"""Redis 可选依赖统一入口（M7 基础设施）。

此前 ``cache_store.py`` 与 ``dist_lock.py`` 各自重复实现同一种
「import redis → ImportError → 降级进程内实现」模式；本模块收口为
单一 ``try_redis_client``，供两者共用。

降级语义（对齐 design-decisions）：
- URL 留空 → 返回 None（调用方降级到进程内实现，零依赖默认路径）；
- URL 填写但 redis 未安装 → **启动报错**（不静默降级）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["try_redis_client"]

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查用
    from redis import Redis


def try_redis_client(url: str) -> Redis | None:
    """尝试按 URL 构建 redis 客户端。

    - URL 留空 → 返回 None（调用方降级到进程内实现，零依赖默认路径）；
    - URL 填写但 redis 未安装 → raise ImportError（对齐 design-decisions 降级承诺）。
    """
    if not url:
        return None
    try:
        import redis  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "redis 包未安装：pip install redis 后重试，"
            "或清空 Redis URL 使用进程内实现（零依赖默认）"
        ) from exc
    return redis.from_url(url)
