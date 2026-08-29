"""ATC 药理类别交叉反应服务（M2）。

交叉反应组（种子数据内置，生产词典经 ``data/drug_dictionary.json`` 扩展）：

- ``beta_lactam``：青霉素类 + 头孢菌素类（β-内酰胺环结构共性，
  青霉素过敏患者使用头孢存在交叉过敏风险）；
- ``nsaid``：解热镇痛抗炎药（阿司匹林过敏 / 阿司匹林哮喘患者
  对其他 NSAID 存在交叉不耐受风险）；
- ``sulfonamide``：磺胺类。

硬规则语义：患者对组内任一药物过敏，阻断集合扩展到组内全部药物
（宁可误拦追问，不可漏放——fail-closed）。
"""

from __future__ import annotations

from harness_agent.safety.dictionary import DrugDictionary

__all__ = ["ATCService"]


#: ATC 前缀 -> 交叉反应组（**兜底**成组：词条漏打 ``cross_group`` 时生效）。
#:
#: 为什么需要兜底：成组若只靠人工标签，新增一个 β-内酰胺类药物时漏打
#: ``cross_group`` 就会**静默 fail-open**——青霉素过敏患者查它查不到阻断，
#: 系统也不告警。ATC 编码本身已编码了药理类别，按前缀推断可以让漏标
#: 退化为"仍被阻断"，而不是"放行"。
#:
#: 刻意排除的前缀（与词典中显式为 ``None`` 的词条保持一致）：
#: - ``J01DF`` 单环 β-内酰胺（氨曲南）：与青霉素类交叉反应极低，临床视为
#:   可安全替代，并入 beta_lactam 会无谓误拦；
#: - ``N02BE`` 对乙酰氨基酚：非 NSAID，不属于 nsaid 交叉不耐受范畴。
#:
#: 顺序无关（按前缀长度降序匹配，长前缀优先）。
_ATC_GROUP_PREFIXES: tuple[tuple[str, str], ...] = (
    # β-内酰胺：青霉素类（J01CA/CE/CR）+ 头孢菌素类（J01DB/DC/DD/DE）+ 碳青霉烯（J01DH）
    ("J01CA", "beta_lactam"),
    ("J01CE", "beta_lactam"),
    ("J01CR", "beta_lactam"),
    ("J01DB", "beta_lactam"),
    ("J01DC", "beta_lactam"),
    ("J01DD", "beta_lactam"),
    ("J01DE", "beta_lactam"),
    ("J01DH", "beta_lactam"),
    # 解热镇痛抗炎药
    ("M01A", "nsaid"),
    ("N02BA", "nsaid"),
    # 磺胺类与甲氧苄啶复方
    ("J01E", "sulfonamide"),
)

#: 按前缀长度降序，保证更具体的前缀先命中
_SORTED_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(_ATC_GROUP_PREFIXES, key=lambda kv: len(kv[0]), reverse=True)
)


class ATCService:
    """ATC 编码查询与交叉反应组解析。

    成组口径（两级）：

    1. 词条显式 ``cross_group``（权威来源，生产词典应显式标注）；
    2. 缺失时按 ATC 前缀推断（兜底，防漏标静默 fail-open）。

    ``cross_group`` 显式为 ``None`` 与"未标注"在此不可区分，因此兜底
    前缀表刻意排除了 J01DF / N02BE 等已知例外——新增例外类别时，
    **既要**在词典里留空 ``cross_group``，**也要**在 ``_ATC_GROUP_PREFIXES``
    里排除对应前缀，两处必须同步。
    """

    def __init__(self, dictionary: DrugDictionary) -> None:
        self._dictionary = dictionary
        groups: dict[str, list[str]] = {}
        for entry in dictionary.entries:
            group = entry.cross_group or self.infer_group(entry.atc_code)
            if group:
                groups.setdefault(group, []).append(entry.normalized_name)
        self._groups = groups

    @staticmethod
    def infer_group(atc_code: str) -> str | None:
        """按 ATC 编码前缀推断交叉反应组；无匹配前缀返回 None。"""
        for prefix, group in _SORTED_PREFIXES:
            if atc_code.startswith(prefix):
                return group
        return None

    def atc_code(self, drug: str) -> str | None:
        """归一化药名 -> ATC 编码；未知药物返回 None。"""
        entry = self._dictionary.get(drug)
        return entry.atc_code if entry else None

    def cross_group(self, drug: str) -> str | None:
        """归一化药名 -> 交叉反应组标识；无组返回 None。"""
        entry = self._dictionary.get(drug)
        if entry is None:
            return None
        return entry.cross_group or self.infer_group(entry.atc_code)

    def cross_reactants(self, drug: str) -> list[str]:
        """同交叉反应组的其他药物（归一化药名，不含自身）。"""
        group = self.cross_group(drug)
        if group is None:
            return []
        return [name for name in self._groups[group] if name != drug]
