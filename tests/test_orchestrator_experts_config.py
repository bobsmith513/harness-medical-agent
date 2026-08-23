"""M4 专家配置加载器测试：声明式 YAML → 注册表（fail-closed 校验）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_agent.orchestrator.experts_config import (
    DEFAULT_EXPERTS_CONFIG,
    ExpertRegistry,
    ExpertSpec,
    load_experts,
)


class TestLoadRealConfig:
    def test_shipped_config_loads(self):
        registry = load_experts(DEFAULT_EXPERTS_CONFIG)
        names = {spec.name for spec in registry.specs()}
        assert {"reasoning_expert", "memory_expert", "intake_expert"} <= names

    def test_kinds_declared(self):
        registry = load_experts(DEFAULT_EXPERTS_CONFIG)
        assert len(registry.by_kind("reasoning")) >= 1
        assert len(registry.by_kind("memory")) >= 1
        assert len(registry.by_kind("generic")) >= 1

    def test_instruction_placeholders_bound_to_inputs(self):
        registry = load_experts(DEFAULT_EXPERTS_CONFIG)
        spec = registry.get("reasoning_expert")
        assert spec.placeholders <= set(spec.inputs)


class TestRegistryLookup:
    def test_get_unknown_expert_raises(self):
        registry = load_experts(DEFAULT_EXPERTS_CONFIG)
        with pytest.raises(KeyError, match="未注册"):
            registry.get("nonexistent")

    def test_has(self):
        registry = load_experts(DEFAULT_EXPERTS_CONFIG)
        assert registry.has("reasoning_expert")
        assert not registry.has("ghost")


def _write(tmp_path: Path, content: str) -> Path:
    config = tmp_path / "experts.yaml"
    config.write_text(content, encoding="utf-8")
    return config


class TestBedrockValidation:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="专家配置不存在"):
            load_experts(tmp_path / "ghost.yaml")

    def test_empty_yaml_raises(self, tmp_path: Path):
        config = _write(tmp_path, "")
        with pytest.raises(ValueError, match="experts"):
            load_experts(config)

    def test_duplicate_names_rejected(self, tmp_path: Path):
        config = _write(
            tmp_path,
            """
experts:
  - name: dup
    kind: generic
    instruction: "a {q}"
    inputs: [q]
  - name: dup
    kind: generic
    instruction: "b {q}"
    inputs: [q]
""",
        )
        with pytest.raises(ValueError, match="重复"):
            load_experts(config)

    def test_unknown_kind_rejected(self, tmp_path: Path):
        with pytest.raises(Exception, match="kind|literal"):
            ExpertSpec(name="weird", kind="alchemist", instruction="x")  # type: ignore[arg-type]

    def test_placeholder_without_input_rejected(self, tmp_path: Path):
        config = _write(
            tmp_path,
            """
experts:
  - name: broken
    kind: generic
    instruction: "处理 {utterance}"
    inputs: [question]
""",
        )
        with pytest.raises(ValueError, match="占位符"):
            load_experts(config)

    def test_name_with_whitespace_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError, match="空白"):
            ExpertSpec(name="bad name", kind="generic", instruction="x")

    def test_blank_instruction_rejected(self):
        with pytest.raises(ValueError):
            ExpertSpec(name="ok", kind="generic", instruction="")

    def test_empty_experts_list_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ExpertRegistry(experts=[])
