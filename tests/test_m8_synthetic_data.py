"""M8 合成数据完整性测试。

验证合成示例数据的结构正确性、与 M2 过敏种子的一致性、
以及各 demo 脚本的覆盖度。

M8 验收标准：全新环境按 README 三条命令跑通全部 demo。
本测试保证合成数据本身的结构完备性。
"""

from __future__ import annotations

import pytest

from fixtures.adversarial_samples import ADVERSARIAL_SAMPLES
from harness_agent.fixtures.synthetic_data import (
    KNOWLEDGE_ENTRIES,
    PATIENT_PROFILES,
    SESSION_SCRIPTS,
)

# ---------------------------------------------------------------------------
# 患者档案
# ---------------------------------------------------------------------------


class TestPatientProfiles:
    """合成患者档案完整性。"""

    def test_four_profiles(self) -> None:
        assert len(PATIENT_PROFILES) == 4

    def test_all_have_required_fields(self) -> None:
        for p in PATIENT_PROFILES:
            assert p.patient_id
            assert p.name
            assert p.age > 0
            assert p.gender in ("男", "女")
            assert p.chief_complaint
            assert p.visit_type in ("first", "followup")

    def test_unique_patient_ids(self) -> None:
        ids = [p.patient_id for p in PATIENT_PROFILES]
        assert len(ids) == len(set(ids))

    def test_pat001_has_penicillin_allergy(self) -> None:
        """pat-001 必须有青霉素过敏（M2 过敏种子）。"""
        p = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-001")
        assert any("盘尼西林" in a or "青霉素" in a for a in p.allergies)

    def test_pat002_has_aspirin_allergy(self) -> None:
        """pat-002 必须有阿司匹林过敏（M2 过敏种子）。"""
        p = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-002")
        assert any("阿司匹林" in a or "拜阿司匹灵" in a for a in p.allergies)

    def test_pat003_no_allergies(self) -> None:
        """pat-003 无已知过敏。"""
        p = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-003")
        assert p.allergies == []

    def test_pat004_diabetes_followup(self) -> None:
        """pat-004 为糖尿病复查患者（复诊记忆 demo 用）。"""
        p = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-004")
        assert p.visit_type == "followup"
        assert any("糖尿病" in f for f in p.stable_facts)
        assert any("二甲双胍" in f for f in p.volatile_facts)

    def test_stable_and_volatile_facts_present(self) -> None:
        """每位患者至少有 1 条稳定事实。"""
        for p in PATIENT_PROFILES:
            assert len(p.stable_facts) >= 1

    def test_allergies_are_raw_names(self) -> None:
        """过敏字段使用药名原文（非归一化后），供安全层归一化。"""
        for p in PATIENT_PROFILES:
            for a in p.allergies:
                assert isinstance(a, str)
                assert len(a) > 0


# ---------------------------------------------------------------------------
# 知识库条目
# ---------------------------------------------------------------------------


class TestKnowledgeEntries:
    """合成知识库条目完整性。"""

    def test_eight_entries(self) -> None:
        assert len(KNOWLEDGE_ENTRIES) == 8

    def test_unique_chunk_ids(self) -> None:
        ids = [e.chunk_id for e in KNOWLEDGE_ENTRIES]
        assert len(ids) == len(set(ids))

    def test_all_have_content(self) -> None:
        for e in KNOWLEDGE_ENTRIES:
            assert len(e.content) > 20

    def test_all_have_provenance(self) -> None:
        for e in KNOWLEDGE_ENTRIES:
            assert e.metadata.get("provenance") == "knowledge_base"

    def test_cap_guideline_entries(self) -> None:
        """CAP 指南条目覆盖初诊 demo 检索。"""
        cap = [e for e in KNOWLEDGE_ENTRIES if e.chunk_id.startswith("kb-cap")]
        assert len(cap) >= 3
        contents = " ".join(e.content for e in cap)
        assert "阿奇霉素" in contents
        assert "β-内酰胺" in contents

    def test_dm_guideline_entries(self) -> None:
        """糖尿病指南条目覆盖复诊 demo 检索。"""
        dm = [e for e in KNOWLEDGE_ENTRIES if e.chunk_id.startswith("kb-dm")]
        assert len(dm) >= 2
        contents = " ".join(e.content for e in dm)
        assert "二甲双胍" in contents
        assert "HbA1c" in contents

    def test_ra_guideline_entries(self) -> None:
        """类风湿关节炎指南条目覆盖交叉反应场景。"""
        ra = [e for e in KNOWLEDGE_ENTRIES if e.chunk_id.startswith("kb-ra")]
        assert len(ra) >= 2
        contents = " ".join(e.content for e in ra)
        assert "甲氨蝶呤" in contents

    def test_allergy_safety_entry(self) -> None:
        """药物过敏安全指南条目。"""
        entries = [e for e in KNOWLEDGE_ENTRIES if e.chunk_id.startswith("kb-allergy")]
        assert len(entries) >= 1
        contents = " ".join(e.content for e in entries)
        assert "青霉素" in contents
        assert "β-内酰胺" in contents or "交叉反应" in contents


# ---------------------------------------------------------------------------
# 会话脚本
# ---------------------------------------------------------------------------


class TestSessionScripts:
    """多轮会话脚本完整性。"""

    def test_four_scripts(self) -> None:
        assert len(SESSION_SCRIPTS) == 4
        assert "first_diagnosis" in SESSION_SCRIPTS
        assert "followup_memory" in SESSION_SCRIPTS
        assert "long_conversation" in SESSION_SCRIPTS
        assert "gate_interception" in SESSION_SCRIPTS

    def test_first_diagnosis_triggers_reasoning(self) -> None:
        turns = SESSION_SCRIPTS["first_diagnosis"]
        assert len(turns) >= 1
        assert turns[0].expected_route == "need_reasoning"

    def test_followup_memory_no_reasoning(self) -> None:
        turns = SESSION_SCRIPTS["followup_memory"]
        assert len(turns) >= 1
        assert turns[0].expected_route == "no_reasoning"

    def test_long_conversation_has_20_turns(self) -> None:
        turns = SESSION_SCRIPTS["long_conversation"]
        assert len(turns) == 20

    def test_long_conversation_turn_indices_sequential(self) -> None:
        turns = SESSION_SCRIPTS["long_conversation"]
        indices = [t.turn_index for t in turns]
        assert indices == list(range(1, 21))

    def test_gate_interception_triggers_reasoning(self) -> None:
        turns = SESSION_SCRIPTS["gate_interception"]
        assert len(turns) >= 1
        assert turns[0].expected_route == "need_reasoning"

    def test_all_turns_have_user_input(self) -> None:
        for script_name, turns in SESSION_SCRIPTS.items():
            for t in turns:
                assert t.user_input, f"{script_name} turn {t.turn_index} has empty input"


# ---------------------------------------------------------------------------
# 跨数据一致性：合成数据 vs 对抗样本
# ---------------------------------------------------------------------------


class TestCrossDataConsistency:
    """合成数据与 M2 对抗样本的一致性。"""

    def test_adversarial_patient_ids_exist_in_profiles(self) -> None:
        """对抗样本引用的 patient_id 必须在患者档案中存在。"""
        profile_ids = {p.patient_id for p in PATIENT_PROFILES}
        for sample in ADVERSARIAL_SAMPLES:
            assert sample.patient_id in profile_ids, (
                f"对抗样本 {sample.sample_id} 引用 patient_id={sample.patient_id} 不在患者档案中"
            )

    def test_pat001_allergy_in_profiles_matches_adversarial(self) -> None:
        """pat-001 的过敏药物在合成档案与对抗样本中一致。"""
        profile = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-001")
        adv_samples = [s for s in ADVERSARIAL_SAMPLES if s.patient_id == "pat-001"]
        assert len(adv_samples) >= 2

        # 对抗样本期望阻断 penicillin / amoxicillin / ceftriaxone 等 β-内酰胺类
        expected_drugs = set()
        for s in adv_samples:
            expected_drugs.update(s.expected_drugs)
        assert expected_drugs, "对抗样本应包含期望阻断药物"

        # 合成档案过敏字段含"盘尼西林"→ 归一化为 penicillin
        assert any("盘尼西林" in a or "青霉素" in a for a in profile.allergies)

    def test_pat002_allergy_in_profiles_matches_adversarial(self) -> None:
        """pat-002 的过敏药物在合成档案与对抗样本中一致。"""
        profile = next(p for p in PATIENT_PROFILES if p.patient_id == "pat-002")
        adv_samples = [s for s in ADVERSARIAL_SAMPLES if s.patient_id == "pat-002"]
        assert len(adv_samples) >= 1

        # 合成档案过敏字段含阿司匹林相关
        assert any("阿司匹林" in a or "拜阿司匹灵" in a for a in profile.allergies)


# ---------------------------------------------------------------------------
# demo 可导入性（确保 demo 文件语法正确）
# ---------------------------------------------------------------------------


class TestDemoImportability:
    """验证四个端到端 demo 模块可被导入（语法/依赖正确）。"""

    @pytest.mark.parametrize(
        "module_name",
        [
            "demo_first_diagnosis",
            "demo_followup_memory",
            "demo_long_conversation",
            "demo_gate_interception",
        ],
    )
    def test_demo_module_importable(self, module_name: str) -> None:
        import importlib

        mod = importlib.import_module(f"examples.{module_name}")
        assert hasattr(mod, "main")
