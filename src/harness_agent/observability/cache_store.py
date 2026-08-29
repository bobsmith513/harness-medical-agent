"""缓存存储实现（M7）：MemoryCacheStore + RedisCacheStore。

- ``MemoryCacheStore``：Redis URL 留空时降级，进程内字典 + TTL；
- ``RedisCacheStore``：URL 填写后通过 redis-py 连接（get/set/delete/
  clear/size 与内存版接口对齐）。

两者共用 ``CacheStore`` 接口（M1 契约）。redis 客户端的构建统一走
``redis_compat.try_redis_client``（与 ``dist_lock`` 共用同一降级模式）。
"""

from __future__ import annotations

import time

from harness_agent.contracts.observability import CacheStore
from harness_agent.observability.redis_compat import try_redis_client

__all__ = ["MemoryCacheStore", "RedisCacheStore", "build_cache_store"]


class MemoryCacheStore:
    """进程内缓存（Redis 留空时降级，零依赖）。

    支持 TTL 过期：``set(key, value, ttl_s=60)`` 60 秒后自动失效。
    """

    def __init__(self) -> None:
        # key -> (value, expires_at_timestamp | None)
        self._store: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        """读取缓存（过期返回 None）。"""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        """写入缓存（可选 TTL）。"""
        expires_at = time.time() + ttl_s if ttl_s is not None else None
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """删除缓存条目。"""
        return self._store.pop(key, None) is not None

    def clear(self) -> None:
        """清空缓存（测试用）。"""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class RedisCacheStore:
    """Redis 缓存（URL 填写后通过 redis-py 连接）。

    真实部署需安装 redis：
        pip install redis

    接口与 ``MemoryCacheStore`` 完全对齐（get/set/delete/clear/size）；
    URL 填写但 redis 包未安装时启动报错（对齐 design-decisions 降级承诺）。
    """

    def __init__(self, url: str = "") -> None:
        self._url = url
        self._client = try_redis_client(url)
        self._fallback = MemoryCacheStore()

    def get(self, key: str) -> str | None:
        if self._client is not None:
            val = self._client.get(key)
            return val.decode("utf-8") if val else None
        return self._fallback.get(key)

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        if self._client is not None:
            if ttl_s is not None:
                self._client.setex(key, ttl_s, value)
            else:
                self._client.set(key, value)
        else:
            self._fallback.set(key, value, ttl_s)

    def delete(self, key: str) -> bool:
        """删除缓存条目（返回是否存在；Redis DEL 返回删除数）。"""
        if self._client is not None:
            return bool(self._client.delete(key))
        return self._fallback.delete(key)

    def clear(self) -> None:
        """清空缓存（SCAN + 批量删除，不用 FLUSHDB 防误伤共享库）。"""
        if self._client is not None:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor=cursor)
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        else:
            self._fallback.clear()

    @property
    def size(self) -> int:
        """缓存条目数（Redis DBSIZE；降级时为内存字典长度）。"""
        if self._client is not None:
            return int(self._client.dbsize())
        return self._fallback.size


def build_cache_store(redis_url: str = "") -> CacheStore:
    """按配置装配缓存存储。

    Redis URL 留空时返回 MemoryCacheStore（零依赖默认）；
    填写时返回 RedisCacheStore（需安装 redis，否则启动报错）。
    """
    if not redis_url:
        return MemoryCacheStore()
    return RedisCacheStore(url=redis_url)
