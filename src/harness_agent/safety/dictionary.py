"""药名词典：硬规则的数据源（M2）。

**数据落位说明（生产环境）**：
- 内置 ``SEED_DRUG_DICTIONARY`` 是演示 / 测试用的**最小合成种子**：
  每个模式只保留 1-2 条代表药物（青霉素类直命中、头孢类交叉反应、
  NSAID 交叉不耐受、无交叉类别阴性对照）；
- 真实部署通过 ``DrugDictionary.from_json()`` 加载完整词典，
  路径由 ``HARNESS_SAFETY__DICTIONARY_PATH`` 配置（留空 = 内置种子）；
  示例格式见仓库根目录 ``data/drug_dictionary.json``（内容与种子一致，
  即"生产数据从这里填"的空位标记）。

构造期校验 fail-closed：重复归一化名、别名冲突、空别名直接抛错——
带歧义的词典会让硬规则静默失配，必须在加载时暴露。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from harness_agent.safety.normalization import fold_text

__all__ = ["DrugDictionary", "DrugEntry", "SEED_DRUG_DICTIONARY"]


class DrugEntry(BaseModel):
    """单个药物词条：归一化名 + ATC 编码 + 别名集合 + 交叉反应组。"""

    normalized_name: str
    atc_code: str
    #: 匹配别名（中文通用名 / 商品名 / 英文名 / 历史名，全部等价）
    aliases: list[str] = Field(default_factory=list)
    #: 交叉反应组标识（beta_lactam / nsaid）：组内药物存在交叉过敏风险，
    #: 过敏阻断集合按组扩展
    cross_group: str | None = None


#: 内置合成词典（零依赖默认，32 条）。
#:
#: 覆盖口径：``beta_lactam`` 13 条（青霉素类 + 一至四代头孢 + 碳青霉烯）、
#: ``nsaid`` 8 条、``sulfonamide`` 3 条、单环 β-内酰胺（氨曲南，刻意不并入
#: ``beta_lactam``）1 条，以及 7 条无交叉组的阴性对照
#: （大环内酯 / 喹诺酮 / 对乙酰氨基酚 / 二甲双胍）。合计 13+1+8+3+7=32。
#:
#: 别名重叠的说明：``DrugNormalizer.find_mentions`` 按最长优先 + 占位扫描，
#: 长别名会吃掉短别名的区间（``阿莫西林克拉维酸钾`` 吃掉 ``阿莫西林``、
#: ``复方新诺明`` 吃掉 ``新诺明``、``氨苄青霉素`` 吃掉 ``青霉素``）。
#: 这些重叠对**同组内**发生，阻断集合一致，因此不构成漏检缝——
#: 扩词典时新增的长别名必须与它吃掉的短别名同组，否则会真的漏检。
SEED_DRUG_DICTIONARY: list[DrugEntry] = [
    # ==== beta_lactam：青霉素类 + 头孢菌素类 + 碳青霉烯类（13 条）====
    # 结构共性为 β-内酰胺环，青霉素过敏者使用头孢存在交叉过敏风险（fail-closed 全组阻断）。
    DrugEntry(
        normalized_name="penicillin",
        atc_code="J01CE01",
        cross_group="beta_lactam",
        aliases=["青霉素", "盘尼西林", "苄青霉素", "benzylpenicillin"],
    ),
    DrugEntry(
        normalized_name="amoxicillin",
        atc_code="J01CA04",
        cross_group="beta_lactam",
        aliases=["阿莫西林", "再林", "阿莫仙"],
    ),
    DrugEntry(
        normalized_name="ampicillin",
        atc_code="J01CA01",
        cross_group="beta_lactam",
        aliases=["氨苄西林", "氨苄青霉素"],
    ),
    DrugEntry(
        normalized_name="piperacillin",
        atc_code="J01CA12",
        cross_group="beta_lactam",
        aliases=["哌拉西林", "氧哌嗪青霉素"],
    ),
    DrugEntry(
        normalized_name="amoxicillin_clavulanate",
        atc_code="J01CR02",
        cross_group="beta_lactam",
        aliases=["阿莫西林克拉维酸钾", "安灭菌", "奥格门汀"],
    ),
    DrugEntry(
        normalized_name="cefazolin",
        atc_code="J01DB04",
        cross_group="beta_lactam",
        aliases=["头孢唑林", "先锋五号"],
    ),
    DrugEntry(
        normalized_name="cefuroxime",
        atc_code="J01DC02",
        cross_group="beta_lactam",
        aliases=["头孢呋辛", "西力欣"],
    ),
    DrugEntry(
        normalized_name="ceftriaxone",
        atc_code="J01DD04",
        cross_group="beta_lactam",
        aliases=["头孢曲松", "罗氏芬"],
    ),
    DrugEntry(
        normalized_name="ceftazidime",
        atc_code="J01DD02",
        cross_group="beta_lactam",
        aliases=["头孢他啶", "复达欣"],
    ),
    DrugEntry(
        normalized_name="cefepime",
        atc_code="J01DE01",
        cross_group="beta_lactam",
        aliases=["头孢吡肟", "马斯平"],
    ),
    DrugEntry(
        normalized_name="meropenem",
        atc_code="J01DH02",
        cross_group="beta_lactam",
        aliases=["美罗培南", "美平"],
    ),
    DrugEntry(
        normalized_name="imipenem_cilastatin",
        atc_code="J01DH51",
        cross_group="beta_lactam",
        aliases=["亚胺培南西司他丁", "泰能"],
    ),
    DrugEntry(
        normalized_name="ertapenem",
        atc_code="J01DH03",
        cross_group="beta_lactam",
        aliases=["厄他培南", "怡万之"],
    ),
    # ==== 单环 β-内酰胺：刻意不并入 beta_lactam（1 条）====
    # 氨曲南与青霉素类交叉反应极低，临床视为青霉素过敏者的可安全替代；
    # 并入 beta_lactam 会无谓误拦。ATC 前缀兜底同样排除 J01DF（见 atc.py）。
    DrugEntry(
        normalized_name="aztreonam",
        atc_code="J01DF01",
        aliases=["氨曲南", "君刻单"],
    ),
    # ==== nsaid：解热镇痛抗炎药（8 条）====
    # 阿司匹林过敏 / 阿司匹林哮喘患者对其他 NSAID 存在交叉不耐受风险。
    DrugEntry(
        normalized_name="aspirin",
        atc_code="N02BA01",
        cross_group="nsaid",
        aliases=["阿司匹林", "拜阿司匹灵", "乙酰水杨酸"],
    ),
    DrugEntry(
        normalized_name="ibuprofen",
        atc_code="M01AE01",
        cross_group="nsaid",
        aliases=["布洛芬", "芬必得"],
    ),
    DrugEntry(
        normalized_name="naproxen",
        atc_code="M01AE02",
        cross_group="nsaid",
        aliases=["萘普生", "甲氧萘丙酸"],
    ),
    DrugEntry(
        normalized_name="diclofenac",
        atc_code="M01AB05",
        cross_group="nsaid",
        aliases=["双氯芬酸", "扶他林"],
    ),
    DrugEntry(
        normalized_name="indomethacin",
        atc_code="M01AB01",
        cross_group="nsaid",
        aliases=["吲哚美辛", "消炎痛"],
    ),
    DrugEntry(
        normalized_name="ketorolac",
        atc_code="M01AB15",
        cross_group="nsaid",
        aliases=["酮咯酸", "痛力克"],
    ),
    DrugEntry(
        normalized_name="meloxicam",
        atc_code="M01AC06",
        cross_group="nsaid",
        aliases=["美洛昔康", "莫比可"],
    ),
    DrugEntry(
        normalized_name="celecoxib",
        atc_code="M01AH01",
        cross_group="nsaid",
        aliases=["塞来昔布", "西乐葆"],
    ),
    # ==== sulfonamide：磺胺类（3 条）====
    DrugEntry(
        normalized_name="sulfamethoxazole",
        atc_code="J01EC01",
        cross_group="sulfonamide",
        aliases=["磺胺甲噁唑", "磺胺甲基异噁唑", "新诺明"],
    ),
    DrugEntry(
        normalized_name="co_trimoxazole",
        atc_code="J01EE01",
        cross_group="sulfonamide",
        aliases=["复方新诺明", "甲氧苄啶磺胺甲噁唑", "cotrimoxazole"],
    ),
    DrugEntry(
        normalized_name="sulfasalazine",
        atc_code="A07EC01",
        cross_group="sulfonamide",
        aliases=["柳氮磺吡啶", "柳氮磺胺吡啶"],
    ),
    # ==== 无交叉组：阴性对照（7 条）====
    # 大环内酯：β-内酰胺过敏患者的标准替代方案。
    DrugEntry(
        normalized_name="azithromycin",
        atc_code="J01FA10",
        aliases=["阿奇霉素", "希舒美"],
    ),
    DrugEntry(
        normalized_name="clarithromycin",
        atc_code="J01FA09",
        aliases=["克拉霉素", "克拉仙"],
    ),
    DrugEntry(
        normalized_name="erythromycin",
        atc_code="J01FA01",
        aliases=["红霉素", "利君沙"],
    ),
    # 喹诺酮：与 β-内酰胺无交叉，同为常用替代。
    DrugEntry(
        normalized_name="levofloxacin",
        atc_code="J01MA12",
        aliases=["左氧氟沙星", "可乐必妥"],
    ),
    DrugEntry(
        normalized_name="moxifloxacin",
        atc_code="J01MA14",
        aliases=["莫西沙星", "拜复乐"],
    ),
    # 对乙酰氨基酚：非 NSAID（无明显抗炎作用），NSAID 不耐受者通常可用。
    DrugEntry(
        normalized_name="paracetamol",
        atc_code="N02BE01",
        aliases=["对乙酰氨基酚", "扑热息痛", "泰诺林", "acetaminophen"],
    ),
    # 双胍类降糖药：demo 复诊场景的稳定事实。
    DrugEntry(
        normalized_name="metformin",
        atc_code="A10BA02",
        aliases=["二甲双胍", "格华止"],
    ),
]


class DrugDictionary:
    """药名词典：归一化名 -> 词条，折叠别名 -> 归一化名。"""

    def __init__(self, entries: Sequence[DrugEntry]) -> None:
        self._entries: dict[str, DrugEntry] = {}
        alias_index: dict[str, str] = {}
        for entry in entries:
            name = entry.normalized_name
            if name in self._entries:
                raise ValueError(f"词典中存在重复的归一化药名: {name}")
            self._entries[name] = entry
            for alias in [name, *entry.aliases]:
                key = fold_text(alias)
                if not key:
                    raise ValueError(f"药物 {name} 存在折叠后为空的别名: {alias!r}")
                existing = alias_index.get(key)
                if existing is not None and existing != name:
                    raise ValueError(
                        f"别名 {alias!r} 同时映射到 {existing} 与 {name}，词典存在歧义"
                    )
                alias_index[key] = name
        # 最长优先的匹配键（同长度按字典序，保证确定性）
        self.alias_index: dict[str, str] = alias_index
        self.match_keys: list[tuple[str, str]] = sorted(
            alias_index.items(), key=lambda kv: (-len(kv[0]), kv[0])
        )

    @property
    def entries(self) -> list[DrugEntry]:
        """全部词条（按录入顺序）。"""
        return list(self._entries.values())

    def get(self, normalized_name: str) -> DrugEntry | None:
        """按归一化名查词条；未知返回 None。"""
        return self._entries.get(normalized_name)

    def __len__(self) -> int:
        return len(self._entries)

    @classmethod
    def from_json(cls, path: str | Path) -> DrugDictionary:
        """从 JSON 文件加载词典（生产数据的扩展入口）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([DrugEntry(**item) for item in data])

    def to_json(self, path: str | Path) -> None:
        """导出词典为 JSON（种子导出 / 生产词典模板生成）。"""
        payload = [entry.model_dump() for entry in self.entries]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
