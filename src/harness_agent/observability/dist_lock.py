"""分布式锁实现（M7）：MemoryLock + RedisLock。

- ``MemoryLock``：Redis URL 留空时降级，进程内互斥锁；
- ``RedisLock``：URL 填写后通过 redis-py SET NX EX 实现。

两者共用 ``DistLock`` 接口（M1 契约）。redis 客户端的构建统一走
``redis_compat.try_redis_client``（与 ``cache_store`` 共用同一降级模式）。
"""

from __future__ import annotations

import threading
import time

from harness_agent.contracts.observability import DistLock
from harness_agent.observability.redis_compat import try_redis_client

__all__ = ["MemoryLock", "RedisLock", "build_dist_lock"]


class MemoryLock:
    """进程内分布式锁（Redis 留空时降级，零依赖）。

    使用 ``threading.Lock`` 实现进程内互斥，
    TTL 通过时间戳软超时（锁持有超过 TTL 自动释放）。
    """

    def __init__(self) -> None:
        self._locks: dict[str, float] = {}  # key -> expires_at
        self._mutex = threading.Lock()

    def acquire(self, key: str, ttl_s: float) -> bool:
        """尝试获取锁（成功返回 True，已被持有返回 False）。"""
        with self._mutex:
            now = time.time()
            expires_at = self._locks.get(key)
            if expires_at is not None and now < expires_at:
                return False  # 锁仍被持有
            self._locks[key] = now + ttl_s
            return True

    def release(self, key: str) -> None:
        """释放锁。"""
        with self._mutex:
            self._locks.pop(key, None)

    def is_locked(self, key: str) -> bool:
        """检查锁是否被持有（测试用）。"""
        with self._mutex:
            expires_at = self._locks.get(key)
            if expires_at is None:
                return False
            if time.time() >= expires_at:
                del self._locks[key]
                return False
            return True


class RedisLock:
    """Redis 分布式锁（URL 填写后通过 SET NX EX 实现）。

    真实部署需安装 redis：
        pip install redis

    未安装时自动降级为进程内 MemoryLock。
    """

    def __init__(self, url: str = "") -> None:
        self._url = url
        self._client = try_redis_client(url)
        self._fallback = MemoryLock()

    def acquire(self, key: str, ttl_s: float) -> bool:
        if self._client is not None:
            # SET NX EX：不存在才设置 + 过期时间
            result = self._client.set(key, "locked", nx=True, ex=int(ttl_s))
            return bool(result)
        return self._fallback.acquire(key, ttl_s)

    def release(self, key: str) -> None:
        if self._client is not None:
            self._client.delete(key)
        else:
            self._fallback.release(key)


def build_dist_lock(redis_url: str = "") -> DistLock:
    """按配置装配分布式锁。

    Redis URL 留空时返回 MemoryLock（零依赖默认）；
    填写时返回 RedisLock（redis 包未安装自动降级）。
    """
    if not redis_url:
        return MemoryLock()
    return RedisLock(url=redis_url)
