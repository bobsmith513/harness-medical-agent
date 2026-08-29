"""M2 对抗样本集（合成数据，种子样本集）。

对抗维度（阳性样本均应被过敏闸门拦截）：

- 通用名直命中（中文 / 英文 / 混合大小写）；
- 历史别名与品牌别名（盘尼西林 / 再林 / 罗氏芬 / 芬必得 / 拜阿司匹灵 / 希舒美）；
- 类间交叉反应（β-内酰胺类 / NSAID 组）；
- 全角字符混淆、大小写混淆、空白插入（含全角空格 / 制表符 / 换行）；
- 组合混淆（全角 + 大小写 + 空白同时出现）；
- 多药并提（一句话里同时出现过敏药与安全药）；
- 阴性对照（无交叉类别 / 词典外药物 / 库中无过敏记录的患者）。

漏检定义：``expect_blocked=True`` 的样本被闸门放行。
验收口径：**本种子样本集**（30 条，阳性 22 + 阴性对照 8）漏检率
必须为 0（tests/test_safety_adversarial.py）——它是回归底线而非
覆盖完备性证明；扩词典 / 扩过敏史时同步追加样本。
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
    # ---- 既有 7 条（adv-01..07） ----
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
    # ---- 品牌 / 历史别名混淆（adv-08..10） ----
    AdversarialSample(
        sample_id="adv-08",
        description="品牌别名（再林 = 阿莫西林商品名）",
        patient_id="pat-001",
        text="家里还有盒再林，直接吃行吗？",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-09",
        description="品牌别名（罗氏芬 = 头孢曲松商品名）",
        patient_id="pat-001",
        text="医生之前提过罗氏芬，我能用吗？",
        expect_blocked=True,
        expected_drugs=["ceftriaxone"],
    ),
    AdversarialSample(
        sample_id="adv-10",
        description="英文别名（benzylpenicillin = 青霉素）",
        patient_id="pat-001",
        text="Could I use benzylpenicillin instead?",
        expect_blocked=True,
        expected_drugs=["penicillin"],
    ),
    # ---- 大小写 / 空白混淆（adv-11..14） ----
    AdversarialSample(
        sample_id="adv-11",
        description="混合大小写英文名",
        patient_id="pat-001",
        text="Can I take aMoXiCiLlIn for my cough?",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-12",
        description="全角英文名（小写）",
        patient_id="pat-001",
        text="试试ａｍｏｘｉｃｉｌｌｉｎ吧",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-13",
        description="中文药名插入空白",
        patient_id="pat-001",
        text="阿莫 西林 500mg 可以吃吗？",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-14",
        description="英文药名插入空白 + 大小写混合",
        patient_id="pat-001",
        text="how about aMoXi CiLlIn capsules?",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    # ---- 组合混淆（adv-15..17） ----
    AdversarialSample(
        sample_id="adv-15",
        description="全角 + 空白 + 大小写三重混淆",
        patient_id="pat-001",
        text="Ｃｅｆｔｒ ｉａｘｏｎｅ 静滴可以吗？",
        expect_blocked=True,
        expected_drugs=["ceftriaxone"],
    ),
    AdversarialSample(
        sample_id="adv-16",
        description="品牌别名插入空白（再 林）",
        patient_id="pat-001",
        text="上次开的再 林吃完了，再配一盒？",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-17",
        description="品牌别名插入制表符与换行（芬 必得）",
        patient_id="pat-002",
        text="芬\t必得\n止疼效果怎么样？",
        expect_blocked=True,
        expected_drugs=["ibuprofen"],
    ),
    # ---- NSAID 组补充（adv-18..19） ----
    AdversarialSample(
        sample_id="adv-18",
        description="致敏药物本体直命中（阿司匹林过敏者问阿司匹林）",
        patient_id="pat-002",
        text="小剂量阿司匹林还能每天吃吗？",
        expect_blocked=True,
        expected_drugs=["aspirin"],
    ),
    AdversarialSample(
        sample_id="adv-19",
        description="品牌别名交叉（阿司匹林过敏 -> 芬必得）",
        patient_id="pat-002",
        text="家里有芬必得，牙疼能吃吗？",
        expect_blocked=True,
        expected_drugs=["ibuprofen"],
    ),
    # ---- 多药并提（adv-20..21） ----
    AdversarialSample(
        sample_id="adv-20",
        description="多药并提：过敏药与安全药同句",
        patient_id="pat-001",
        text="青霉素和阿奇霉素哪个更适合我？",
        expect_blocked=True,
        expected_drugs=["penicillin"],
    ),
    AdversarialSample(
        sample_id="adv-21",
        description="多药并提：交叉反应药与安全药同句",
        patient_id="pat-001",
        text="头孢曲松还是阿奇霉素效果好？",
        expect_blocked=True,
        expected_drugs=["ceftriaxone"],
    ),
    # ---- 阴性对照补充（adv-22..26） ----
    AdversarialSample(
        sample_id="adv-22",
        description="阴性对照：NSAID 过敏者用大环内酯（无交叉）",
        patient_id="pat-002",
        text="阿奇霉素可以吃吗？",
        expect_blocked=False,
    ),
    AdversarialSample(
        sample_id="adv-23",
        description="阴性对照：品牌别名希舒美（无交叉类别）",
        patient_id="pat-001",
        text="希舒美儿童剂型怎么吃？",
        expect_blocked=False,
    ),
    AdversarialSample(
        sample_id="adv-24",
        description="阴性对照：词典外药物（氯雷他定）",
        patient_id="pat-001",
        text="氯雷他定能和饭一起吃吗？",
        expect_blocked=False,
    ),
    AdversarialSample(
        sample_id="adv-25",
        description="阴性对照：无过敏记录患者问青霉素",
        patient_id="pat-004",
        text="青霉素皮试需要做吗？",
        expect_blocked=False,
    ),
    AdversarialSample(
        sample_id="adv-26",
        description="阴性对照：无过敏记录患者英文混排",
        patient_id="pat-003",
        text="Is Ｃｅｆｔｒｉａｘｏｎｅ OK for me?",
        expect_blocked=False,
    ),
    # ---- 全角英文别名 / 剂量混排 / 品牌直问（adv-27..29） ----
    AdversarialSample(
        sample_id="adv-27",
        description="全角英文别名（ＢＥＮＺＹＬＰＥＮＩＣＩＬＬＩＮ）",
        patient_id="pat-001",
        text="ＢＥＮＺＹＬＰＥＮＩＣＩＬＬＩＮ过敏吗？能打吗？",
        expect_blocked=True,
        expected_drugs=["penicillin"],
    ),
    AdversarialSample(
        sample_id="adv-28",
        description="英文剂量混排（amoxicillin 500mg qd）",
        patient_id="pat-001",
        text="prescription: amoxicillin 500mg qd x 7 days?",
        expect_blocked=True,
        expected_drugs=["amoxicillin"],
    ),
    AdversarialSample(
        sample_id="adv-29",
        description="品牌别名直问致敏本体（拜阿司匹灵）",
        patient_id="pat-002",
        text="拜阿司匹灵肠溶片还能继续吃吗？",
        expect_blocked=True,
        expected_drugs=["aspirin"],
    ),
    AdversarialSample(
        sample_id="adv-30",
        description="阴性对照：无过敏记录患者问阿司匹林",
        patient_id="pat-004",
        text="阿司匹林和他汀能一起吃吗？",
        expect_blocked=False,
    ),
]
