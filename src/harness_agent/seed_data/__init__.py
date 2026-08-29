"""合成示例数据包（全部为虚构数据，不含任何真实病历与患者信息）。

- ``synthetic_data``：患者档案 / 知识条目 / 多轮会话脚本
"""

from harness_agent.seed_data.synthetic_data import (
    KNOWLEDGE_ENTRIES,
    PATIENT_PROFILES,
    SESSION_SCRIPTS,
    KnowledgeEntry,
    PatientProfile,
    ScriptTurn,
)

__all__ = [
    "KNOWLEDGE_ENTRIES",
    "PATIENT_PROFILES",
    "SESSION_SCRIPTS",
    "KnowledgeEntry",
    "PatientProfile",
    "ScriptTurn",
]
