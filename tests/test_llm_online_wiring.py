"""在线调用模式装配测试：providers 预设 + wiring 工厂。

覆盖（对照 M-online 验收标准）：
- 预设表完整性与未知 provider 报错；
- mock / 在线模式切换（同一工厂，配置驱动）；
- 预设端点与模型名解析（共享 api_key 优先级）；
- 逐角色覆盖（微调模型旁路：reasoning_base_url 独立指向 vLLM）；
- 配置缺失的 fail-fast 报错（含修复指引语义）。
"""

from __future__ import annotations

import os

import pytest

from harness_agent.config import Settings
from harness_agent.llm.mock import MockLLMClient
from harness_agent.llm.openai_compat import OpenAICompatClient
from harness_agent.llm.providers import PROVIDER_PRESETS, get_preset
from harness_agent.llm.wiring import build_llm_client, build_llm_clients, describe_llm_setup

#: 全部预设服务商（mock / openai 不在预设表——前者零依赖，后者端点自填）
_ALL_PROVIDERS = {"deepseek", "qwen", "zhipu", "moonshot", "openai", "siliconflow"}


@pytest.fixture()
def clean_env(tmp_path: os.PathLike[str], monkeypatch: pytest.MonkeyPatch):
    """隔离环境变量与 .env 文件（与 test_settings.py 同模式）。"""
    monkeypatch.chdir(tmp_path)
    for key in [k for k in os.environ if k.startswith("HARNESS_")]:
        monkeypatch.delenv(key, raising=False)
    yield


class TestProviderPresets:
    """预设表：端点 + 各角色推荐模型名。"""

    def test_all_presets_present(self):
        assert set(PROVIDER_PRESETS) == {
            "deepseek",
            "qwen",
            "zhipu",
            "moonshot",
            "openai",
            "siliconflow",
        }

    def test_preset_fields_complete(self):
        """每个预设必须给出四类角色模型与端点（留空角色默认值即断链）。"""
        for name, preset in PROVIDER_PRESETS.items():
            assert preset.base_url.startswith("https://"), name
            for role in ("reasoning_model", "judge_model", "router_model", "orchestrator_model"):
                assert getattr(preset, role), f"{name}.{role} 不能为空"

    def test_unknown_provider_raises_with_options(self):
        with pytest.raises(ValueError, match="可选"):
            get_preset("nonexistent")

    def test_endpoints_are_openai_compatible(self):
        """预设端点全部走 OpenAI 兼容协议（/chat/completions 拼接约定）。"""
        for preset in PROVIDER_PRESETS.values():
            assert "/v1" in preset.base_url or "/api/paas/v4" in preset.base_url


class TestBuildLLMClient:
    """装配工厂：配置 → 客户端（mock / 在线可切换）。"""

    def test_mock_mode_returns_mock_client(self, clean_env):
        settings = Settings(_env_file=None)
        client = build_llm_client("reasoning", settings)
        assert isinstance(client, MockLLMClient)
        assert client.role == "reasoning"

    def test_preset_resolves_endpoint_and_model(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "deepseek"
        settings.llm.api_key = "sk-test"
        client = build_llm_client("judge", settings)
        assert isinstance(client, OpenAICompatClient)
        preset = get_preset("deepseek")
        assert client._base_url == preset.base_url
        assert client._model == preset.judge_model

    def test_role_override_beats_preset(self, clean_env):
        """微调模型旁路：reasoning_base_url 独立指向本地 vLLM。"""
        settings = Settings(_env_file=None)
        settings.llm.provider = "deepseek"
        settings.llm.api_key = "sk-test"
        settings.llm.reasoning_base_url = "http://localhost:8001/v1"
        settings.llm.reasoning_model = "sft-dpo-aligned"

        reasoning = build_llm_client("reasoning", settings)
        judge = build_llm_client("judge", settings)
        assert reasoning._base_url == "http://localhost:8001/v1"
        assert reasoning._model == "sft-dpo-aligned"
        # 其余角色继续走在线预设（混合部署形态）
        assert judge._base_url == get_preset("deepseek").base_url

    def test_shared_api_key_inherited_by_all_roles(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "qwen"
        settings.llm.api_key = "sk-shared"
        clients = build_llm_clients(settings)
        for role, client in clients.items():
            assert client._api_key == "sk-shared", role

    def test_role_api_key_overrides_shared(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "deepseek"
        settings.llm.api_key = "sk-shared"
        settings.llm.judge_api_key = "sk-judge-only"
        client = build_llm_client("judge", settings)
        assert client._api_key == "sk-judge-only"

    def test_missing_key_raises_with_guidance(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "deepseek"
        with pytest.raises(ValueError, match="API Key"):
            build_llm_client("reasoning", settings)

    def test_openai_provider_uses_official_endpoint(self, clean_env):
        """openai 走官方预设端点（api.openai.com + gpt-4o 系列）。"""
        settings = Settings(_env_file=None)
        settings.llm.provider = "openai"
        settings.llm.api_key = "sk-x"
        client = build_llm_client("router", settings)
        assert client._base_url == "https://api.openai.com/v1"
        assert client._model == get_preset("openai").router_model

    def test_all_four_roles_assembled(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "deepseek"
        settings.llm.api_key = "sk-test"
        clients = build_llm_clients(settings)
        assert set(clients) == {"orchestrator", "reasoning", "judge", "router"}
        for client in clients.values():
            assert isinstance(client, OpenAICompatClient)


class TestDescribeSetup:
    """启动横幅描述：接线形态人话输出。"""

    def test_mock_description(self, clean_env):
        settings = Settings(_env_file=None)
        assert "Mock" in describe_llm_setup(settings)

    def test_online_description_lists_roles(self, clean_env):
        settings = Settings(_env_file=None)
        settings.llm.provider = "zhipu"
        settings.llm.api_key = "sk-x"
        text = describe_llm_setup(settings)
        assert "provider=zhipu" in text
        for role in ("orchestrator", "reasoning", "judge", "router"):
            assert role in text
        assert get_preset("zhipu").base_url in text


class TestSettingsOnlineMode:
    """LLMSettings 在线模式字段语义。"""

    def test_provider_accepts_all_preset_names(self, clean_env):
        for provider in _ALL_PROVIDERS | {"mock"}:
            settings = Settings(_env_file=None)
            settings.llm.provider = provider
            assert settings.llm.provider == provider

    def test_model_defaults_empty_means_inherit_preset(self, clean_env):
        """模型名默认留空 = 继承预设推荐值（装配层解析）。"""
        llm = Settings(_env_file=None).llm
        for role in ("orchestrator", "reasoning", "judge", "router"):
            assert getattr(llm, f"{role}_model") == ""
            assert getattr(llm, f"{role}_base_url") == ""

    def test_env_provider_switch(self, clean_env, monkeypatch):
        monkeypatch.setenv("HARNESS_LLM__PROVIDER", "deepseek")
        monkeypatch.setenv("HARNESS_LLM__API_KEY", "sk-env")
        settings = Settings(_env_file=None)
        assert settings.llm.provider == "deepseek"
        assert settings.llm.api_key == "sk-env"
