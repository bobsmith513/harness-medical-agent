"""M8 合成示例数据（全部为虚构数据，不含任何真实病历与患者信息）。

数据组成：
- ``PATIENT_PROFILES``：患者档案（含过敏史、稳定/易变事实）；
- ``KNOWLEDGE_ENTRIES``：知识库条目（供检索层入库与双路召回）；
- ``SESSION_SCRIPTS``：多轮会话脚本（初诊/复诊/长会话/门禁拦截）。

与 M2 的 ``adversarial_samples.py`` 互补：
- adversarial_samples 聚焦安全闸门对抗维度（每类 1-2 条）；
- 本文件聚焦端到端 demo 的完整业务叙事（患者画像 + 知识 + 对话）。

位于包内（``harness_agent.seed_data``）：CLI 启动时按 ``HARNESS_APP__SEED_SAMPLE_DATA``
开关把这些数据灌入检索库——数据库地址来自 ``.env``，内容来自这里。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "KNOWLEDGE_ENTRIES",
    "PATIENT_PROFILES",
    "SESSION_SCRIPTS",
    "KnowledgeEntry",
    "PatientProfile",
    "ScriptTurn",
]

#: 已有过敏种子（M2 ``SEED_ALLERGIES`` 对应的患者 ID）：
#: - pat-001: 青霉素过敏（阻断 beta_lactam 全组）
#: - pat-002: 阿司匹林过敏（阻断 nsaid 全组）
#: - pat-003: 无已知过敏
#: M8 新增 pat-004（糖尿病患者，无过敏，复诊场景用）


class PatientProfile(BaseModel):
    """合成患者档案（虚构数据）。"""

    patient_id: str
    name: str  # 虚构姓名
    age: int
    gender: str
    chief_complaint: str
    allergies: list[str] = Field(default_factory=list)
    stable_facts: list[str] = Field(default_factory=list)
    volatile_facts: list[str] = Field(default_factory=list)
    visit_type: str = "first"  # first | followup


class KnowledgeEntry(BaseModel):
    """合成知识库条目（供检索层入库）。"""

    chunk_id: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ScriptTurn(BaseModel):
    """多轮会话脚本的单轮。"""

    turn_index: int
    user_input: str
    expected_route: str = "need_reasoning"
    notes: str = ""


# ---------------------------------------------------------------------------
# 患者档案（4 位虚构患者）
# ---------------------------------------------------------------------------
PATIENT_PROFILES: list[PatientProfile] = [
    PatientProfile(
        patient_id="pat-001",
        name="张明",
        age=45,
        gender="男",
        chief_complaint="咳嗽三天伴发热",
        allergies=["盘尼西林"],
        stable_facts=["血型 O 型", "高血压病史 5 年"],
        volatile_facts=["近期服用氨氯地平 5mg qd"],
        visit_type="first",
    ),
    PatientProfile(
        patient_id="pat-002",
        name="李芳",
        age=38,
        gender="女",
        chief_complaint="关节疼痛加重",
        allergies=["拜阿司匹灵"],
        stable_facts=["血型 A 型", "类风湿性关节炎 3 年"],
        volatile_facts=["近期服用甲氨蝶呤 7.5mg qw"],
        visit_type="followup",
    ),
    PatientProfile(
        patient_id="pat-003",
        name="王强",
        age=52,
        gender="男",
        chief_complaint="社区获得性肺炎复查",
        allergies=[],
        stable_facts=["血型 B 型", "无慢性病史"],
        volatile_facts=[],
        visit_type="followup",
    ),
    PatientProfile(
        patient_id="pat-004",
        name="赵雪",
        age=60,
        gender="女",
        chief_complaint="血糖控制情况咨询",
        allergies=[],
        stable_facts=["血型 AB 型", "2 型糖尿病 8 年", "糖尿病肾病 II 期"],
        volatile_facts=["近期服用二甲双胍 500mg bid", "空腹血糖 7.2 mmol/L"],
        visit_type="followup",
    ),
]


# ---------------------------------------------------------------------------
# 知识库条目（合成医学知识，供检索入库）
# ---------------------------------------------------------------------------
KNOWLEDGE_ENTRIES: list[KnowledgeEntry] = [
    KnowledgeEntry(
        chunk_id="kb-cap-01",
        content=(
            "社区获得性肺炎（CAP）常见病原体为肺炎链球菌、流感嗜血杆菌等。"
            "阿奇霉素属于大环内酯类抗生素，适用于 CAP 经验性治疗，"
            "成人常规剂量 500mg qd，疗程 3-5 天。"
        ),
        metadata={"doc_id": "cap-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-cap-02",
        content=(
            "社区获得性肺炎评估：CURB-65 评分（意识、尿素氮、呼吸频率、"
            "血压、年龄 ≥65）用于判断严重程度。评分 0-1 分可门诊治疗，"
            "2 分需住院，≥3 分考虑 ICU。"
        ),
        metadata={"doc_id": "cap-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-cap-03",
        content=(
            "CAP 患者若对 β-内酰胺类过敏，可选大环内酯类（阿奇霉素、"
            "克拉霉素）或多西环素替代。阿奇霉素与青霉素无交叉反应，"
            "青霉素过敏患者可安全使用。"
        ),
        metadata={"doc_id": "cap-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-dm-01",
        content=(
            "2 型糖尿病血糖控制目标：空腹血糖 4.4-7.0 mmol/L，"
            "餐后 2 小时 < 10.0 mmol/L，HbA1c < 7.0%。"
            "二甲双胍为一线用药，成人常规剂量 500mg bid 至 1000mg bid。"
        ),
        metadata={"doc_id": "dm-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-dm-02",
        content=(
            "糖尿病肾病分期：I 期肾小球高滤过、II 期微量白蛋白尿、"
            "III 期临床蛋白尿、IV 期肾功能下降、V 期尿毒症。"
            "II 期需优化血糖血压控制，ACEI/ARB 类药物可减少蛋白尿。"
        ),
        metadata={"doc_id": "dm-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-ra-01",
        content=(
            "类风湿性关节炎（RA）治疗以改善病情抗风湿药（DMARDs）为主，"
            "甲氨蝶呤为首选。起始剂量 7.5mg qw，可逐步增至 15-20mg qw。"
            "需监测肝功能与血常规。"
        ),
        metadata={"doc_id": "ra-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-ra-02",
        content=(
            "RA 急性期关节疼痛可短期使用 NSAIDs 或低剂量糖皮质激素。"
            "对阿司匹林过敏者，其他 NSAIDs（布洛芬、塞来昔布）存在"
            "交叉不耐受风险，应避免使用 NSAID 类药物。"
        ),
        metadata={"doc_id": "ra-guideline", "provenance": "knowledge_base"},
    ),
    KnowledgeEntry(
        chunk_id="kb-allergy-01",
        content=(
            "药物过敏史是临床用药安全的核心约束。青霉素过敏患者"
            "应避免使用所有 β-内酰胺类抗生素（含头孢类），"
            "交叉反应率约 1-10%。替代方案可选大环内酯类或氟喹诺酮类。"
        ),
        metadata={"doc_id": "allergy-safety", "provenance": "knowledge_base"},
    ),
]


# ---------------------------------------------------------------------------
# 多轮会话脚本（初诊/复诊/长会话/门禁拦截）
# ---------------------------------------------------------------------------
SESSION_SCRIPTS: dict[str, list[ScriptTurn]] = {
    # 初诊推理全链路：咳嗽三天 + 青霉素过敏 → 检索 CAP 指南 → 推理 → 门禁 → 结论
    "first_diagnosis": [
        ScriptTurn(
            turn_index=1,
            user_input="咳嗽三天，发烧 38.5 度，之前打盘尼西林过敏，该怎么用药？",
            expected_route="need_reasoning",
            notes="触发过敏硬规则 + 需要推理",
        ),
    ],
    # 复诊记忆命中：糖尿病复查，已审核记忆可召回
    "followup_memory": [
        ScriptTurn(
            turn_index=1,
            user_input="上次查的血糖结果怎么样，需要调药吗？",
            expected_route="no_reasoning",
            notes="记忆专家装配上下文，免重复问询",
        ),
    ],
    # 长会话压缩：20 轮模拟对话
    "long_conversation": [
        ScriptTurn(
            turn_index=i,
            user_input=f"第 {i} 轮：患者描述症状变化与用药反应……" * 5,
            expected_route="need_reasoning" if i % 3 == 0 else "no_reasoning",
            notes=f"长会话第 {i} 轮",
        )
        for i in range(1, 21)
    ],
    # 门禁拦截：推理产出过敏药物结论 → 输出闸门拦截 → 转人工
    "gate_interception": [
        ScriptTurn(
            turn_index=1,
            user_input="我的肺炎能用青霉素治疗吗？",
            expected_route="need_reasoning",
            notes="推理专家产出含过敏药物结论 → 门禁拦截",
        ),
    ],
}
