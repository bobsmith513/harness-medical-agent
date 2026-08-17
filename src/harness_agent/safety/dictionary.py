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


SEED_DRUG_DICTIONARY: list[DrugEntry] = [
    # --- beta_lactam 组：青霉素类（2）+ 头孢类（1），演示直命中与类间交叉 ---
    DrugEntry(
        normalized_name="penicillin",
        atc_code="J01CE01",
        cross_group="beta_lactam",
        aliases=["青霉素", "盘尼西林", "benzylpenicillin"],
    ),
    DrugEntry(
        normalized_name="amoxicillin",
        atc_code="J01CA04",
        cross_group="beta_lactam",
        aliases=["阿莫西林", "再林"],
    ),
    DrugEntry(
        normalized_name="ceftriaxone",
        atc_code="J01DD04",
        cross_group="beta_lactam",
        aliases=["头孢曲松", "罗氏芬"],
    ),
    # --- nsaid 组：2 条，演示交叉不耐受 ---
    DrugEntry(
        normalized_name="aspirin",
        atc_code="N02BA01",
        cross_group="nsaid",
        aliases=["阿司匹林", "拜阿司匹灵"],
    ),
    DrugEntry(
        normalized_name="ibuprofen",
        atc_code="M01AE01",
        cross_group="nsaid",
        aliases=["布洛芬", "芬必得"],
    ),
    # --- 无交叉组：阴性对照（青霉素过敏患者通常可用） ---
    DrugEntry(
        normalized_name="azithromycin",
        atc_code="J01FA10",
        aliases=["阿奇霉素", "希舒美"],
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
