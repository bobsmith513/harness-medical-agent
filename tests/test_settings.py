"""M0 配置系统测试。

锁定三类事实：
1. 默认零依赖：外部端点全部留空，mock / local / noop 模式可跑通 demo；
2. 环境变量机制：前缀 HARNESS_ + 嵌套分隔符 __ 生效，.env 文件生效；
3. 架构语义默认值：recent_turns=3、记忆审核开启、patient_id 分区键等
   ——这些默认值即系统设计决策，后续里程碑依赖它们不被意外改动。
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from harness_agent.config import Settings, get_settings, reset_settings


@pytest.fixture()
def clean_env(tmp_path: os.PathLike[str], monkeypatch: pytest.MonkeyPatch):
    """隔离环境变量与 .env 文件，保证测试不受外部环境干扰。"""
    monkeypatch.chdir(tmp_path)
    for key in [k for k in os.environ if k.startswith("HARNESS_")]:
        monkeypatch.delenv(key, raising=False)
    reset_settings()
    yield
    reset_settings()


class TestZeroDependencyDefaults:
    """默认配置必须是零依赖可运行的 mock 模式。"""

    def test_llm_defaults_to_mock(self, clean_env):
        assert Settings(_env_file=None).llm.provider == "mock"

    def test_all_external_endpoints_left_blank(self, clean_env):
        """外部模型端点默认全部留空（待使用者填写本地 vLLM 端口或在线 URL）。"""
        llm = Settings(_env_file=None).llm
        for attr in (
            "orchestrator_base_url",
            "orchestrator_api_key",
            "reasoning_base_url",
            "reasoning_api_key",
            "judge_base_url",
            "judge_api_key",
            "router_base_url",
            "router_api_key",
        ):
            assert getattr(llm, attr) == "", f"llm.{attr} 应默认留空"

    def test_retrieval_defaults_to_local_store(self, clean_env):
        retrieval = Settings(_env_file=None).retrieval
        assert retrieval.store == "local"
        assert retrieval.milvus_uri == ""

    def test_reranker_disabled_by_default(self, clean_env):
        assert Settings(_env_file=None).retrieval.reranker_enabled is False

    def test_sandbox_defaults_to_mock(self, clean_env):
        sandbox = Settings(_env_file=None).sandbox
        assert sandbox.backend == "mock"
        assert sandbox.opensandbox_url == ""

    def test_observability_endpoints_blank_mean_degraded(self, clean_env):
        obs = Settings(_env_file=None).observability
        assert obs.langfuse_public_key == ""
        assert obs.langfuse_secret_key == ""
        assert obs.langfuse_host == ""
        assert obs.audit_dsn == ""
        assert obs.redis_url == ""

    def test_safety_dictionary_path_blank_by_default(self, clean_env):
        """生产药名词典路径留空 = 使用内置合成种子词典。"""
        assert Settings(_env_file=None).safety.dictionary_path == ""


class TestArchitectureSemantics:
    """配置默认值必须体现系统设计语义（M0 冻结，后续里程碑依赖）。"""

    def test_vfs_keeps_only_recent_three_turns(self, clean_env):
        assert Settings(_env_file=None).vfs.recent_turns == 3

    def test_memory_review_required_by_default(self, clean_env):
        assert Settings(_env_file=None).vfs.memory_review_required is True

    def test_partition_key_is_patient_id(self, clean_env):
        assert Settings(_env_file=None).retrieval.partition_key == "patient_id"

    def test_embedded_model_is_bge_large_zh(self, clean_env):
        assert Settings(_env_file=None).retrieval.embedding_model == "BAAI/bge-large-zh"


class TestEnvOverride:
    """环境变量机制：HARNESS_ 前缀 + __ 嵌套分隔符。"""

    def test_nested_env_var_overrides_provider(self, clean_env, monkeypatch):
        monkeypatch.setenv("HARNESS_LLM__PROVIDER", "openai")
        monkeypatch.setenv("HARNESS_LLM__ORCHESTRATOR_BASE_URL", "http://localhost:8000/v1")
        settings = Settings(_env_file=None)
        assert settings.llm.provider == "openai"
        assert settings.llm.orchestrator_base_url == "http://localhost:8000/v1"

    def test_store_switches_to_milvus(self, clean_env, monkeypatch):
        monkeypatch.setenv("HARNESS_RETRIEVAL__STORE", "milvus")
        assert Settings(_env_file=None).retrieval.store == "milvus"

    def test_invalid_provider_is_rejected(self, clean_env, monkeypatch):
        monkeypatch.setenv("HARNESS_LLM__PROVIDER", "anthropic")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_dotenv_file_is_loaded(self, clean_env, tmp_path):
        (tmp_path / ".env").write_text(
            "HARNESS_RETRIEVAL__STORE=milvus\nHARNESS_VFS__RECENT_TURNS=5\n",
            encoding="utf-8",
        )
        settings = Settings()
        assert settings.retrieval.store == "milvus"
        assert settings.vfs.recent_turns == 5

    def test_unknown_env_vars_are_ignored(self, clean_env, monkeypatch):
        monkeypatch.setenv("HARNESS_UNKNOWN__FIELD", "whatever")
        settings = Settings(_env_file=None)  # extra="ignore"，不应抛错
        assert settings.llm.provider == "mock"


class TestSettingsCache:
    """get_settings 单例缓存行为。"""

    def test_get_settings_is_cached(self, clean_env):
        assert get_settings() is get_settings()

    def test_reset_settings_reloads_from_env(self, clean_env, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("HARNESS_VFS__RECENT_TURNS", "7")
        reset_settings()
        second = get_settings()
        assert second is not first
        assert second.vfs.recent_turns == 7
