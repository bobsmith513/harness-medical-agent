"""VFS 存储抽象与实现（M6）。

两类实现共享同一 ``VfsStore`` 接口，靠依赖注入切换，业务逻辑不分叉：

- ``InMemoryVfsStore``：进程内字典存储（demo / 测试默认，零依赖）；
- ``FileBackedVfsStore``：文件系统存储（落 ``root_dir/<session_id>/<path>``，
  生产环境持久化用；目录结构即逻辑路径的物理映射）。

两种实现按 ``VfsSettings.root_dir`` 是否非空且可写自动选择——
留空（默认 ``.data/vfs``）时用文件存储，不可写时降级内存。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Protocol, runtime_checkable

from harness_agent.models.common import new_id, now_utc

__all__ = [
    "VfsEntry",
    "VfsStore",
    "InMemoryVfsStore",
    "FileBackedVfsStore",
    "build_vfs_store",
]


class VfsEntry:
    """VFS 条目：逻辑路径 + 内容 + 元数据。

    逻辑路径形如 ``/evidence/ev-abc123.json``——前缀即目录
    （evidence / reasoning / summaries / memories），调用方
    按前缀查询同目录下全部条目。
    """

    __slots__ = ("entry_id", "session_id", "path", "content", "created_at", "size_bytes")

    def __init__(
        self,
        *,
        session_id: str,
        path: str,
        content: str,
        entry_id: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.entry_id = entry_id or new_id("vfs")
        self.session_id = session_id
        self.path = path if path.startswith("/") else f"/{path}"
        self.content = content
        self.created_at = created_at or now_utc()
        self.size_bytes = len(content.encode("utf-8"))

    def __repr__(self) -> str:
        return f"VfsEntry(path={self.path!r}, size={self.size_bytes}B)"


@runtime_checkable
class VfsStore(Protocol):
    """VFS 统一存储接口。

    逻辑路径前缀对应四个虚拟目录：
    - ``/evidence/``：证据包持久化（溢出轮的证据快照）
    - ``/reasoning/``：推理链持久化（溢出轮的结论快照）
    - ``/summaries/``：会话摘要（压缩后的旧轮摘要）
    - ``/memories/``：长期记忆（审核通过后转正的记忆条目）
    """

    def write(self, session_id: str, path: str, content: str) -> str:
        """写入条目，返回 VFS 文件名（逻辑路径）。"""
        ...

    def read(self, session_id: str, path: str) -> str | None:
        """读取条目内容（不存在返回 None）。"""
        ...

    def list_dir(self, session_id: str, prefix: str) -> list[str]:
        """列出指定前缀下的全部条目路径。"""
        ...

    def exists(self, session_id: str, path: str) -> bool:
        """条目是否存在。"""
        ...

    def delete(self, session_id: str, path: str) -> bool:
        """删除条目（返回是否删除成功）。"""
        ...


class InMemoryVfsStore:
    """进程内 VFS 存储（demo / 测试默认）。

    按 ``session_id`` 分区，按逻辑路径索引；
    会话间隔离与患者分区语义一致（patient_id 即 session 的归属键）。
    """

    def __init__(self) -> None:
        # session_id -> {path -> VfsEntry}
        self._entries: dict[str, dict[str, VfsEntry]] = {}

    def write(self, session_id: str, path: str, content: str) -> str:
        normalized = path if path.startswith("/") else f"/{path}"
        entry = VfsEntry(session_id=session_id, path=normalized, content=content)
        self._entries.setdefault(session_id, {})[normalized] = entry
        return normalized

    def read(self, session_id: str, path: str) -> str | None:
        normalized = path if path.startswith("/") else f"/{path}"
        entry = self._entries.get(session_id, {}).get(normalized)
        return entry.content if entry else None

    def list_dir(self, session_id: str, prefix: str) -> list[str]:
        normalized = prefix if prefix.startswith("/") else f"/{prefix}"
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        paths = self._entries.get(session_id, {})
        return sorted(p for p in paths if p.startswith(normalized))

    def exists(self, session_id: str, path: str) -> bool:
        normalized = path if path.startswith("/") else f"/{path}"
        return normalized in self._entries.get(session_id, {})

    def delete(self, session_id: str, path: str) -> bool:
        normalized = path if path.startswith("/") else f"/{path}"
        return self._entries.get(session_id, {}).pop(normalized, None) is not None


class FileBackedVfsStore:
    """文件系统 VFS 存储（生产环境持久化）。

    物理映射：``<root_dir>/<session_id>/<path>``
    逻辑路径 ``/evidence/ev-abc.json`` → 物理路径 ``root_dir/sess/evidence/ev-abc.json``
    """

    def __init__(self, root_dir: str) -> None:
        self._root = os.path.abspath(root_dir)
        os.makedirs(self._root, exist_ok=True)

    def _physical(self, session_id: str, path: str) -> str:
        """逻辑路径 → 物理路径（安全拼接，防目录穿越）。"""
        normalized = path.lstrip("/")
        # 禁止 .. 穿越
        safe = normalized.replace("..", "").replace("//", "/")
        return os.path.join(self._root, session_id, safe)

    def write(self, session_id: str, path: str, content: str) -> str:
        normalized = path if path.startswith("/") else f"/{path}"
        phys = self._physical(session_id, normalized)
        os.makedirs(os.path.dirname(phys), exist_ok=True)
        with open(phys, "w", encoding="utf-8") as f:
            f.write(content)
        return normalized

    def read(self, session_id: str, path: str) -> str | None:
        normalized = path if path.startswith("/") else f"/{path}"
        phys = self._physical(session_id, normalized)
        if not os.path.isfile(phys):
            return None
        with open(phys, encoding="utf-8") as f:
            return f.read()

    def list_dir(self, session_id: str, prefix: str) -> list[str]:
        normalized = prefix if prefix.startswith("/") else f"/{prefix}"
        if not normalized.endswith("/"):
            normalized = f"{normalized}/"
        base = self._physical(session_id, normalized)
        if not os.path.isdir(base):
            return []
        results: list[str] = []
        for root, _, files in os.walk(base):
            for fname in files:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self._physical(session_id, ""))
                results.append(f"/{rel}")
        return sorted(results)

    def exists(self, session_id: str, path: str) -> bool:
        normalized = path if path.startswith("/") else f"/{path}"
        return os.path.isfile(self._physical(session_id, normalized))

    def delete(self, session_id: str, path: str) -> bool:
        normalized = path if path.startswith("/") else f"/{path}"
        phys = self._physical(session_id, normalized)
        if os.path.isfile(phys):
            os.remove(phys)
            return True
        return False


def build_vfs_store(root_dir: str = "") -> VfsStore:
    """按配置装配 VFS 存储。

    ``root_dir`` 留空时用内存存储（demo / 测试默认）；
    指定路径时用文件存储（生产环境持久化）。
    """
    if not root_dir:
        return InMemoryVfsStore()
    try:
        return FileBackedVfsStore(root_dir)
    except OSError:
        # 不可写目录 → 降级内存存储
        return InMemoryVfsStore()
