"""向后兼容垫片：合成数据已迁移至 ``harness_agent.seed_data.synthetic_data``。"""

from harness_agent.seed_data.synthetic_data import (  # noqa: F401
    KNOWLEDGE_ENTRIES,
    PATIENT_PROFILES,
    SESSION_SCRIPTS,
    KnowledgeEntry,
    PatientProfile,
    ScriptTurn,
)
