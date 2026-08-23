"""YAML 专家声明式配置加载器（M4）。

``configs/experts.yaml`` → ``list[ExpertSpec]``：

- 新增专家零改动主流程（声明即注册）；
- ``kind`` 决定主 Agent 的委派语义（reasoning / memory / generic）；
- 重名校验、未知 kind 拒绝、inputs 键与指令模板占位符交叉校验
  （fail-closed：配置错误在装配期暴露，不进运行时）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from harness_agent.config.paths import anchor_path

__all__ = ["ExpertKind", "ExpertSpec", "ExpertRegistry", "load_experts"]

#: 专家委派语义（与 contracts.experts 的特化协议对应）
ExpertKind = Literal["reasoning", "memory", "generic"]

#: 默认配置路径：锚定仓库根（见 config/paths.py）。
#: 从任意 CWD 运行均能命中；生产环境用 settings 指定绝对路径。
DEFAULT_EXPERTS_CONFIG = str(anchor_path("configs/experts.yaml"))


class ExpertSpec(BaseModel):
    """单个专家的声明式定义（YAML 条目的运行时形态）。"""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    kind: ExpertKind
    description: str = Field(default="", min_length=0)
    instruction: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_no_whitespace(cls, value: str) -> str:
        if any(ch.isspace() for ch in value):
            raise ValueError(f"专家名不得含空白字符: {value!r}")
        return value

    @property
    def placeholders(self) -> set[str]:
        """指令模板中的 {placeholder} 集合（主 Agent 委派时填充）。"""
        import re

        return set(re.findall(r"\{(\w+)\}", self.instruction))


class ExpertRegistry(BaseModel):
    """专家注册表：name → spec 唯一映射（委派目标解析用）。"""

    model_config = ConfigDict(frozen=True)

    experts: list[ExpertSpec] = Field(min_length=1)

    def specs(self) -> list[ExpertSpec]:
        return list(self.experts)

    def get(self, name: str) -> ExpertSpec:
        """按名取专家；未注册抛 KeyError（委派目标必须存在）。"""
        for spec in self.experts:
            if spec.name == name:
                return spec
        registered = ", ".join(s.name for s in self.experts)
        raise KeyError(f"专家未注册: {name!r}（已注册: {registered}）")

    def has(self, name: str) -> bool:
        return any(spec.name == name for spec in self.experts)

    def by_kind(self, kind: ExpertKind) -> list[ExpertSpec]:
        return [spec for spec in self.experts if spec.kind == kind]


def load_experts(path: str | Path = DEFAULT_EXPERTS_CONFIG) -> ExpertRegistry:
    """加载并校验专家声明式配置。

    校验项（fail-closed）：
    - 文件存在且为合法 YAML；
    - ``experts`` 非空列表，条目字段类型正确；
    - 专家名全局唯一、无空白；
    - kind 属于已知枚举；
    - 指令模板的占位符 ⊆ inputs 声明（占位符无输入来源即装配期报错）。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"专家配置不存在: {config_path}"
            f"（默认已锚定仓库根 configs/experts.yaml；"
            f"自定义路径用 HARNESS_ORCHESTRATOR__EXPERTS_CONFIG_PATH 指定）"
        )
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict) or "experts" not in payload:
        raise ValueError(f"专家配置必须为含 'experts' 列表的 YAML: {config_path}")
    registry = ExpertRegistry(experts=payload["experts"])

    names = [spec.name for spec in registry.experts]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(f"专家名重复: {sorted(duplicates)}")

    for spec in registry.experts:
        unbound = spec.placeholders - set(spec.inputs)
        if unbound:
            raise ValueError(
                f"专家 {spec.name} 指令模板占位符 {sorted(unbound)} "
                f"未在 inputs 中声明（无法填充，fail-closed）"
            )
    return registry
