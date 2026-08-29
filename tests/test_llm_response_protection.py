"""OpenAI 兼容客户端响应解析保护测试（静态分析整改项）。

背景：``complete()`` 旧实现对 ``response.json()`` 与 ``choices[0]``
取值无任何保护——非 200 由 raise_for_status 兜住，但 200 + 非 JSON
结构异常会裸抛 KeyError/TypeError 栈。整改后统一抛
``OpenAICompatResponseError``（含 role / 端点 / 缺失字段上下文），
上层编排的 fail-closed 捕获可拿到明确错误分类。

测试用 ``httpx.MockTransport`` 注入离线响应，不发起真实网络请求。
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness_agent.contracts.llm import LLMMessage
from harness_agent.llm.openai_compat import OpenAICompatClient, OpenAICompatResponseError


def _client_with(handler) -> OpenAICompatClient:
    """构建注入 MockTransport 的客户端（离线，零网络）。"""
    client = OpenAICompatClient(
        role="reasoning",
        base_url="http://test-endpoint/v1",
        api_key="sk-test",
        model="test-model",
    )
    client._client = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def _ok_body(content: str = "测试应答") -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _json_response(payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="咳嗽三天怎么处理？")]


class TestHappyPath:
    def test_valid_response(self):
        client = _client_with(lambda request: _json_response(_ok_body("阿奇霉素 500mg")))
        result = client.complete(_messages())
        assert result.text == "阿奇霉素 500mg"
        assert result.model == "test-model"
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5

    def test_calls_recorded(self):
        client = _client_with(lambda request: _json_response(_ok_body()))
        client.complete(_messages())
        client.complete(_messages())
        assert len(client.calls) == 2

    def test_null_content_becomes_empty_text(self):
        """OpenAI 协议 content 可为 null（工具调用场景）→ 空文本（下游 fail-closed）。"""
        body = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        client = _client_with(lambda request: _json_response(body))
        result = client.complete(_messages())
        assert result.text == ""

    def test_missing_usage_defaults_to_zero(self):
        body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        client = _client_with(lambda request: _json_response(body))
        result = client.complete(_messages())
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0


class TestMalformedResponses:
    def test_non_json_body(self):
        def handler(request):
            return httpx.Response(200, text="<html>502 Bad Gateway</html>")

        client = _client_with(handler)
        with pytest.raises(OpenAICompatResponseError, match="非 JSON"):
            client.complete(_messages())

    def test_top_level_not_object(self):
        client = _client_with(lambda request: _json_response(["not", "an", "object"]))
        with pytest.raises(OpenAICompatResponseError, match="顶层非 JSON 对象"):
            client.complete(_messages())

    def test_missing_choices(self):
        client = _client_with(lambda request: _json_response({"error": "quota"}))
        with pytest.raises(OpenAICompatResponseError, match="缺少 choices"):
            client.complete(_messages())

    def test_empty_choices_list(self):
        client = _client_with(lambda request: _json_response({"choices": []}))
        with pytest.raises(OpenAICompatResponseError, match="缺少 choices"):
            client.complete(_messages())

    def test_choice_missing_message(self):
        client = _client_with(
            lambda request: _json_response({"choices": [{"finish_reason": "stop"}]})
        )
        with pytest.raises(OpenAICompatResponseError, match="message.content"):
            client.complete(_messages())

    def test_content_not_text(self):
        body = {"choices": [{"message": {"role": "assistant", "content": ["数组"]}}]}
        client = _client_with(lambda request: _json_response(body))
        with pytest.raises(OpenAICompatResponseError, match="content 非文本"):
            client.complete(_messages())

    def test_error_context_in_message(self):
        """错误信息包含 role 与端点上下文（可观测性要求）。"""
        client = _client_with(lambda request: _json_response({}))
        with pytest.raises(OpenAICompatResponseError) as excinfo:
            client.complete(_messages())
        assert "reasoning" in str(excinfo.value)
        assert "http://test-endpoint/v1" in str(excinfo.value)


class TestProtocolCompliance:
    def test_request_payload_shape(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            captured["auth"] = request.headers.get("Authorization")
            return _json_response(_ok_body())

        client = _client_with(handler)
        client.complete(_messages(), temperature=0.3, max_tokens=128)
        assert captured["payload"]["model"] == "test-model"
        assert captured["payload"]["temperature"] == 0.3
        assert captured["payload"]["max_tokens"] == 128
        assert captured["payload"]["messages"][0]["content"] == "咳嗽三天怎么处理？"
        assert captured["auth"] == "Bearer sk-test"

    def test_reasoner_model_omits_temperature(self):
        """deepseek-reasoner 类推理模型已弃用 temperature → 不发送（其余参数照常）。"""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["payload"] = json.loads(request.content)
            return _json_response(_ok_body())

        client = OpenAICompatClient(
            role="reasoning",
            base_url="http://test-endpoint/v1",
            api_key="sk-test",
            model="deepseek-reasoner",
        )
        client._client = httpx.Client(transport=httpx.MockTransport(handler))
        client.complete(_messages(), temperature=0.3, max_tokens=128)
        assert "temperature" not in captured["payload"]
        assert captured["payload"]["max_tokens"] == 128

    def test_close_and_context_manager_release_pool(self):
        """close() 关闭底层连接池；上下文管理器退出等价于 close()（防泄漏）。"""
        client = _client_with(lambda request: _json_response(_ok_body()))
        client.complete(_messages())
        client.close()
        assert client._client.is_closed

        with _client_with(lambda request: _json_response(_ok_body())) as scoped:
            assert not scoped._client.is_closed
        assert scoped._client.is_closed

    def test_concurrent_calls_record_all(self):
        """calls 记录并发安全（锁保护下的交错追加不丢条目）。"""
        import threading

        client = _client_with(lambda request: _json_response(_ok_body()))
        barrier = threading.Barrier(8)

        def call():
            barrier.wait()
            client.complete(_messages())

        threads = [threading.Thread(target=call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(client.calls) == 8
