"""审计存储实现（M7）：SQLiteAuditStore + PostgresAuditStore 骨架。

- ``SQLiteAuditStore``：PostgreSQL DSN 留空时降级，落 ``app.data_dir``
  下的 SQLite 文件（零外部依赖）；
- ``PostgresAuditStore``：DSN 填写后通过 psycopg 写入
  （骨架，真实部署需安装 psycopg 包）。

两者共用 ``AuditStore`` 接口（M1 契约），配置切换零逻辑分叉。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

from harness_agent.contracts.observability import AuditStore
from harness_agent.models.audit import AuditRecord
from harness_agent.models.common import now_utc

__all__ = ["SQLiteAuditStore", "PostgresAuditStore", "build_audit_store"]

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_records (
    audit_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    session_id TEXT,
    turn_index INTEGER,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    verdict_json TEXT,
    created_at TEXT NOT NULL
)
"""


class SQLiteAuditStore:
    """SQLite 审计存储（DSN 留空时降级，零依赖）。

    审计记录 append-only 写入，按 session_id 查询；
    SQLite 文件落在 ``data_dir/audit.db``。
    """

    def __init__(self, data_dir: str = ".data") -> None:
        os.makedirs(data_dir, exist_ok=True)
        self._db_path = os.path.join(data_dir, "audit.db")
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def append(self, record: AuditRecord) -> None:
        """追加审计记录（append-only）。"""
        verdict_json = ""
        if record.verdict is not None:
            verdict_json = record.verdict.model_dump_json()

        self._conn.execute(
            "INSERT INTO audit_records "
            "(audit_id, trace_id, session_id, turn_index, actor, action, "
            "verdict_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.audit_id,
                record.trace_id,
                record.session_id,
                record.turn_index,
                record.actor,
                record.action,
                verdict_json,
                record.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def query(self, session_id: str) -> list[AuditRecord]:
        """按会话查询审计记录。"""
        cursor = self._conn.execute(
            "SELECT audit_id, trace_id, session_id, turn_index, actor, action, "
            "verdict_json, created_at FROM audit_records "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        rows = cursor.fetchall()
        records: list[AuditRecord] = []
        for row in rows:
            verdict = None
            if row[6]:
                from harness_agent.models.audit import GateVerdict

                verdict = GateVerdict.model_validate_json(row[6])
            dt = datetime.fromisoformat(row[7]) if row[7] else now_utc()
            records.append(
                AuditRecord(
                    audit_id=row[0],
                    trace_id=row[1],
                    session_id=row[2],
                    turn_index=row[3],
                    actor=row[4],
                    action=row[5],
                    verdict=verdict,
                    created_at=dt,
                )
            )
        return records

    def count(self) -> int:
        """总记录数（测试用）。"""
        cursor = self._conn.execute("SELECT COUNT(*) FROM audit_records")
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class PostgresAuditStore:
    """PostgreSQL 审计存储骨架（DSN 填写后通过 psycopg 写入）。

    真实部署需安装 psycopg：
        pip install "psycopg[binary]"

    DSN 填写但 psycopg 未安装时启动报错（对齐 design-decisions 降级承诺）——
    审计记录不可静默降级为内存存储（进程退出全丢，合规风险）。
    """

    def __init__(self, dsn: str = "") -> None:
        self._dsn = dsn
        self._conn = None

        if dsn:
            try:
                import psycopg  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    'psycopg 未安装：pip install "psycopg[binary]" 后重试，'
                    "或清空 DSN 使用 SQLiteAuditStore（零依赖默认）"
                ) from exc
            self._conn = psycopg.connect(dsn)
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()

    def append(self, record: AuditRecord) -> None:
        verdict_json = ""
        if record.verdict is not None:
            verdict_json = record.verdict.model_dump_json()
        self._conn.execute(
            "INSERT INTO audit_records "
            "(audit_id, trace_id, session_id, turn_index, actor, action, "
            "verdict_json, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                record.audit_id,
                record.trace_id,
                record.session_id,
                record.turn_index,
                record.actor,
                record.action,
                verdict_json,
                record.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def query(self, session_id: str) -> list[AuditRecord]:
        cursor = self._conn.execute(
            "SELECT audit_id, trace_id, session_id, turn_index, actor, "
            "action, verdict_json, created_at FROM audit_records "
            "WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        rows = cursor.fetchall()
        records: list[AuditRecord] = []
        for row in rows:
            verdict = None
            if row[6]:
                from harness_agent.models.audit import GateVerdict

                verdict = GateVerdict.model_validate_json(row[6])
            records.append(
                AuditRecord(
                    audit_id=row[0],
                    trace_id=row[1],
                    session_id=row[2],
                    turn_index=row[3],
                    actor=row[4],
                    action=row[5],
                    verdict=verdict,
                    created_at=datetime.fromisoformat(row[7]),
                )
            )
        return records


def build_audit_store(dsn: str = "", data_dir: str = ".data") -> AuditStore:
    """按配置装配审计存储。

    DSN 留空时返回 SQLiteAuditStore（零依赖默认）；
    填写时返回 PostgresAuditStore（需安装 psycopg，否则启动报错）。
    """
    if not dsn:
        return SQLiteAuditStore(data_dir=data_dir)
    return PostgresAuditStore(dsn=dsn)
