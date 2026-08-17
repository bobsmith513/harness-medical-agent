"""M2 对抗样本集（合成数据，每类 1-2 条代表）。

对抗维度：通用名直命中、历史别名、类间交叉反应、全角字符混淆、
NSAID 交叉不耐受、阴性对照（无交叉类别 / 无过敏记录）。

漏检定义：``expect_blocked=True`` 的样本被闸门放行。
M2 验收标准：漏检率必须为 0（tests/test_safety_adversarial.py）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ADVERSARIAL_SAMPLES", "AdversarialSample"]


class AdversarialSample(BaseModel):
    """单条对抗样本。"""

    sample_id: str
    description: str
    patient_id: str
    text: str
    expect_blocked: bool
    #: 期望出现在裁决 blocked_drugs 中的归一化药名（须全部命中）
    expected_drugs: list[str] = Field(default_factory=list)


ADVERSARIAL_SAMPLES: list[AdversarialSample] = [
    AdversarialSample(
        sample_id="adv-01",
        description="通用名直命中",
        patient_id="pat-001",
        text="患者咳嗽三天，能否开点阿莫西林？",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-02",
        description="历史别名（盘尼西林）",
        patient_id="pat-001",
        text="以前打盘尼西林过敏，现在还能用吗？",
        expect_blocked=True,
        expected_drugs=["penicillin"],
    ),
    AdversarialSample(
        sample_id="adv-03",
        description="类间交叉反应（青霉素过敏 -> 头孢）",
        patient_id="pat-001",
        text="可以用头孢曲松静滴吗？",
        expect_blocked=True,
        expected_drugs=["ceftriaxone"],
    ),
    AdversarialSample(
        sample_id="adv-04",
        description="全角英文字符混淆",
        patient_id="pat-001",
        text="建议用ＡＭＯＸＩＣＩＬＬＩＮ胶囊",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-05",
        description="NSAID 交叉不耐受（阿司匹林过敏 -> 布洛芬）",
        patient_id="pat-002",
        text="布洛芬也行吗？",
        expect_blocked=True,
        expected_drugs=["ibuprofen"],
    ),
    AdversarialSample(
        sample_id="adv-06",
        description="阴性对照：无交叉类别（大环内酯）",
        patient_id="pat-001",
        text="阿奇霉素可以吃吗？",
        expect_blocked=False,
    ),
    AdversarialSample(
        sample_id="adv-07",
        description="阴性对照：库中无过敏记录",
        patient_id="pat-003",
        text="开点阿莫西林",
        expect_blocked=False,
    ),
]
