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


class ATCService:
    """ATC 编码查询与交叉反应组解析。"""

    def __init__(self, dictionary: DrugDictionary) -> None:
        self._dictionary = dictionary
        groups: dict[str, list[str]] = {}
        for entry in dictionary.entries:
            if entry.cross_group:
                groups.setdefault(entry.cross_group, []).append(entry.normalized_name)
        self._groups = groups

    def atc_code(self, drug: str) -> str | None:
        """归一化药名 -> ATC 编码；未知药物返回 None。"""
        entry = self._dictionary.get(drug)
        return entry.atc_code if entry else None

    def cross_group(self, drug: str) -> str | None:
        """归一化药名 -> 交叉反应组标识；无组返回 None。"""
        entry = self._dictionary.get(drug)
        return entry.cross_group if entry else None

    def cross_reactants(self, drug: str) -> list[str]:
        """同交叉反应组的其他药物（归一化药名，不含自身）。"""
        group = self.cross_group(drug)
        if group is None:
            return []
        return [name for name in self._groups[group] if name != drug]
