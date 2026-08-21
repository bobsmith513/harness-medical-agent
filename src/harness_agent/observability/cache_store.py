"""缓存存储实现（M7）：MemoryCacheStore + RedisCacheStore 骨架。

- ``MemoryCacheStore``：Redis URL 留空时降级，进程内字典 + TTL；
- ``RedisCacheStore``：URL 填写后通过 redis-py 连接
  （骨架，真实部署需安装 redis 包）。

两者共用 ``CacheStore`` 接口（M1 契约）。
"""

from __future__ import annotations

import time

from harness_agent.contracts.observability import CacheStore

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
    """Redis 缓存骨架（URL 填写后通过 redis-py 连接）。

    真实部署需安装 redis：
        pip install redis

    未安装时自动降级为进程内 MemoryCacheStore。
    """

    def __init__(self, url: str = "") -> None:
        self._url = url
        self._client = None
        self._fallback = MemoryCacheStore()

        if url:
            try:
                import redis  # type: ignore[import-untyped]

                self._client = redis.from_url(url)
            except ImportError:
                pass

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


def build_cache_store(redis_url: str = "") -> CacheStore:
    """按配置装配缓存存储。

    Redis URL 留空时返回 MemoryCacheStore（零依赖默认）；
    填写时返回 RedisCacheStore（redis 包未安装自动降级）。
    """
    if not redis_url:
        return MemoryCacheStore()
    return RedisCacheStore(url=redis_url)
