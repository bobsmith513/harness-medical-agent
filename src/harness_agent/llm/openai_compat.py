"""OpenAI 兼容 LLM 客户端（M5）：vLLM / 在线 API 端点。

端点配置来自 M0 的四组 ``LLMSettings``（orchestrator / reasoning /
judge / router），端点留空时调用方应注入 ``MockLLMClient``——本类
不做静默降级，配置与实现的绑定由装配工厂显式控制（接口注入，
业务逻辑不分叉）。

安装：``uv sync --extra llm``（httpx）；vLLM 部署的端点形如
``http://localhost:8001/v1``（填写 ``HARNESS_LLM__REASONING_BASE_URL``
等环境变量后即生效）。
"""

from __future__ import annotations

from harness_agent.contracts.llm import LLMMessage, LLMResult, LLMRole

__all__ = ["OpenAICompatClient"]


class OpenAICompatClient:
    """OpenAI 兼容 chat completions 客户端（vLLM / 在线 API）。

    实现方负责绑定自身端点、超时、模型名（来自 M0 配置）；
    调用方只关心 ``LLMClient`` 契约（消息进、结果出）。

    与 ``MockLLMClient`` 完全可互换——同一专家 / 路由器 / 门禁
    注入哪个实现，行为只取决于实现本身的应答逻辑，业务代码零分叉。
    """

    def __init__(
        self,
        role: LLMRole,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError(
                f"LLM 端点未配置（role={role}）：填写 HARNESS_LLM__"
                f"{role.upper()}_BASE_URL 或改用 MockLLMClient"
            )
        if not model:
            raise ValueError(f"LLM 模型名未指定（role={role}）")
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx 未安装：uv sync --extra llm 后重试，或改用 MockLLMClient（零依赖默认）"
            ) from exc
        self.role = role
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._client = httpx.Client(timeout=timeout_s)
        self.calls: list[list[LLMMessage]] = []

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMResult:
        self.calls.append(list(messages))
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        response = self._client.post(
            f"{self._base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]["message"]
        usage = data.get("usage", {})
        return LLMResult(
            text=choice["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self._model,
        )
