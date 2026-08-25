"""VFS 目录抽象（M6）：四个虚拟目录的统一管理门面。

目录语义（对应 development-plan.md 第三节长会话数据流）：

- ``/evidence/``：证据包快照——溢出上下文的旧轮证据包持久化于此，
  上下文只留文件指针（``evidence_pack_id → /evidence/xxx.json``）；
- ``/reasoning/``：推理链快照——溢出轮的临床结论 + 推理链全文，
  审计可回溯，上下文不留全文只留 ``conclusion_id`` 指针；
- ``/summaries/``：会话摘要——压缩后的旧轮摘要，标注来源置信度，
  记忆审核的输入源（``/summaries/ → 审核队列 → /memories/``）；
- ``/memories/``：长期记忆——审核通过后转正的记忆条目，
  同步向量索引（M3 检索可召回），未审核仅作会话内指针。

``VfsDirectory`` 是门面：封装存储操作 + 路径规范 + 目录语义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from harness_agent.vfs.store import VfsStore, build_vfs_store

__all__ = [
    "VfsDirectory",
    "DIR_EVIDENCE",
    "DIR_REASONING",
    "DIR_SUMMARIES",
    "DIR_MEMORIES",
]

#: 四个虚拟目录前缀（逻辑路径固定语义，不可改）
DIR_EVIDENCE = "/evidence"
DIR_REASONING = "/reasoning"
DIR_SUMMARIES = "/summaries"
DIR_MEMORIES = "/memories"


def _json_default(obj):
    """JSON 序列化 fallback：datetime → ISO 格式字符串。"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _dumps(data: dict) -> str:
    """安全 JSON 序列化（datetime 兼容）。"""
    return json.dumps(data, ensure_ascii=False, default=_json_default)


@dataclass(frozen=True)
class VfsDirectory:
    """VFS 目录门面：四个虚拟目录的统一操作入口。

    封装 ``VfsStore`` 的存储操作，提供语义化的目录路径。
    调用方只关心"写证据"、"读推理链"、"列摘要"，不需要拼路径。
    """

    store: VfsStore
    session_id: str

    # ---- 通用操作 ----

    def write(self, directory: str, filename: str, content: str) -> str:
        """写入条目，返回完整逻辑路径。"""
        path = f"{directory}/{filename}"
        return self.store.write(self.session_id, path, content)

    def read(self, path: str) -> str | None:
        """按完整逻辑路径读取。"""
        return self.store.read(self.session_id, path)

    def list_entries(self, directory: str) -> list[str]:
        """列出目录下全部条目路径。"""
        return self.store.list_dir(self.session_id, directory)

    def exists(self, path: str) -> bool:
        return self.store.exists(self.session_id, path)

    def delete(self, path: str) -> bool:
        return self.store.delete(self.session_id, path)

    # ---- 语义化快捷方法 ----

    def write_evidence(self, evidence_id: str, data: dict) -> str:
        """持久化证据包快照到 ``/evidence/``。"""
        return self.write(DIR_EVIDENCE, f"{evidence_id}.json", _dumps(data))

    def write_reasoning(self, conclusion_id: str, data: dict) -> str:
        """持久化推理链 + 结论到 ``/reasoning/``。"""
        return self.write(DIR_REASONING, f"{conclusion_id}.json", _dumps(data))

    def write_summary(self, summary_id: str, data: dict) -> str:
        """持久化会话摘要到 ``/summaries/``（审核输入源）。"""
        return self.write(DIR_SUMMARIES, f"{summary_id}.json", _dumps(data))

    def write_memory(self, memory_id: str, data: dict) -> str:
        """持久化审核通过的记忆到 ``/memories/``（可召回）。"""
        return self.write(DIR_MEMORIES, f"{memory_id}.json", _dumps(data))

    def read_evidence(self, evidence_id: str) -> dict | None:
        content = self.read(f"{DIR_EVIDENCE}/{evidence_id}.json")
        return json.loads(content) if content else None

    def read_reasoning(self, conclusion_id: str) -> dict | None:
        content = self.read(f"{DIR_REASONING}/{conclusion_id}.json")
        return json.loads(content) if content else None

    def list_evidence(self) -> list[str]:
        return self.list_entries(DIR_EVIDENCE)

    def list_reasoning(self) -> list[str]:
        return self.list_entries(DIR_REASONING)

    def list_summaries(self) -> list[str]:
        return self.list_entries(DIR_SUMMARIES)

    def list_memories(self) -> list[str]:
        return self.list_entries(DIR_MEMORIES)

    def count_entries(self, directory: str) -> int:
        return len(self.list_entries(directory))


def build_directory(session_id: str, root_dir: str = "") -> VfsDirectory:
    """装配 VFS 目录门面（零依赖默认内存存储）。"""
    store = build_vfs_store(root_dir)
    return VfsDirectory(store=store, session_id=session_id)
