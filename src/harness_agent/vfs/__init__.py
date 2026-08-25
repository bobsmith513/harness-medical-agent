"""虚拟文件系统与记忆审核（M6）。

四个虚拟目录 + 上下文压缩 + 记忆审核队列：

- ``VfsStore``：存储抽象（内存默认 / 文件可选，零依赖）；
- ``VfsDirectory``：目录门面（``/evidence/ /reasoning/ /summaries/ /memories/``）；
- ``ContextCompactor``：溢出轮持久化 + 文件指针登记（Token 压缩核心）；
- ``MemoryReviewQueue``：摘要 → 抽样审核 → 通过转正 → 同步索引
  （未审核仅会话内指针，绝不进入检索结果）。

验收指标：
- 20 轮模拟会话上下文 token 降约 50%；
- "未审核摘要不得被召回"单测通过。
"""

from __future__ import annotations

from harness_agent.vfs.compaction import (
    CompactionResult,
    CompactionStats,
    ContextCompactor,
    build_compactor,
)
from harness_agent.vfs.directory import (
    DIR_EVIDENCE,
    DIR_MEMORIES,
    DIR_REASONING,
    DIR_SUMMARIES,
    VfsDirectory,
    build_directory,
)
from harness_agent.vfs.memory_review import (
    MemoryReviewQueue,
    ReviewResult,
)
from harness_agent.vfs.store import (
    FileBackedVfsStore,
    InMemoryVfsStore,
    VfsEntry,
    VfsStore,
    build_vfs_store,
)

__all__ = [
    # store
    "VfsEntry",
    "VfsStore",
    "InMemoryVfsStore",
    "FileBackedVfsStore",
    "build_vfs_store",
    # directory
    "VfsDirectory",
    "DIR_EVIDENCE",
    "DIR_REASONING",
    "DIR_SUMMARIES",
    "DIR_MEMORIES",
    "build_directory",
    # compaction
    "ContextCompactor",
    "CompactionResult",
    "CompactionStats",
    "build_compactor",
    # memory review
    "MemoryReviewQueue",
    "ReviewResult",
]
