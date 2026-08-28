"""本地 OpenAI 兼容验证服务器：无 Key 时走完整在线调用链路。

    uv run python examples/mock_openai_server.py            # 默认 127.0.0.1:8100
    uv run python examples/mock_openai_server.py --port 8200

然后把 `.env` 中各角色的 BASE_URL 指向 ``http://127.0.0.1:8100/v1``：
`harness-online` 的全部 LLM 调用（推理 / judge / 路由兜底）就会通过
真实的 httpx → HTTP → JSON 在线链路完成——与真实服务商的代码路径
完全一致，换回真实 base_url + Key 即是生产形态。

应答逻辑（按 system 提示词区分角色，输出严格符合各契约的 JSON）：
- 推理专家：三段式推理链，引用 ID 从请求中的"合法 evidence_id 列表"提取；
- 质量门禁：忠实度 0.92、无臆测、无因果倒置；
- 路由器：need_reasoning。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = ["build_completion_response", "main"]


def _extract_evidence_ids(user_content: str) -> list[str]:
    """从推理专家的用户提示中提取合法 evidence_id（排除自编 ID）。"""
    return sorted(set(re.findall(r"\bev-[0-9a-f]{6,}\b", user_content)))


def _reasoning_chain_json(system: str, user_content: str) -> str:
    """推理专家应答：三段式推理链（引用真实 evidence_id，规避过敏药名）。"""
    ev_ids = _extract_evidence_ids(user_content)
    cited = ev_ids[:1] or ["ev-unknown"]
    chain = {
        "steps": [
            {
                "kind": "evidence",
                "text": "引用证据：检索层返回的指南条目支持当前问诊的诊疗判断",
                "citations": cited,
            },
            {
                "kind": "inference",
                "text": "结合患者主诉与上述证据逐步推断：本例方案不涉及患者已知过敏药物",
            },
            {
                "kind": "conclusion",
                "text": "综合上述证据，按检索到的指南方案给出建议（需临床医生确认）",
                "citations": cited,
            },
        ],
        "statement": "基于检索证据的初步建议（需临床医生确认）",
        "self_check_notes": "自检通过（3/3）：引用真实、因果正向、依据充分",
    }
    return json.dumps(chain, ensure_ascii=False)


def _judge_json() -> str:
    """质量门禁应答：忠实度 0.92，无臆测，无因果倒置。"""
    return json.dumps(
        {
            "faithfulness": 0.92,
            "has_hallucination": False,
            "causal_inversion": False,
            "reason": "结论有证据支撑，无臆测，因果顺序正确",
        },
        ensure_ascii=False,
    )


def _router_json() -> str:
    """路由器应答：需要临床推理。"""
    return json.dumps({"decision": "need_reasoning"}, ensure_ascii=False)


def build_completion_response(messages: list[dict]) -> str:
    """按 system 提示词路由到对应角色的合法应答。

    匹配顺序与标记按各角色系统提示词的首句自述（"质量门禁"/"路由器"/
    "推理专家"）——旧版按"临床推理"关键词匹配会把路由请求（其提示词含
    "是否需要临床推理"）误判为推理角色，导致 LLM 兜底路由恒失败。
    """
    system = ""
    user_content = ""
    for msg in messages:
        if msg.get("role") == "system":
            system += str(msg.get("content", ""))
        else:
            user_content += str(msg.get("content", ""))

    if "质量门禁" in system:
        return _judge_json()
    if "路由器" in system:
        return _router_json()
    if "推理专家" in system or "临床推理" in system:
        return _reasoning_chain_json(system, user_content)
    # 未识别角色：按路由器处理（最宽松的兜底，输出仍是合法二值裁决）
    return _router_json()


class _Handler(BaseHTTPRequestHandler):
    """OpenAI 兼容端点：POST {base}/chat/completions。"""

    def do_POST(self) -> None:  # noqa: N802 - http.server 命名约定
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "only /chat/completions is served")
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages", [])
        model = payload.get("model", "mock-server")

        content = build_completion_response(messages)
        body = json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": sum(len(str(m.get("content", ""))) for m in messages),
                    "completion_tokens": len(content),
                    "total_tokens": 0,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

        # 白盒日志：角色 / 模型 / 应答类型（stderr，不污染管道输出）
        role = "router"
        system = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        if "推理专家" in system or "临床推理" in system:
            role = "reasoning"
        elif "质量门禁" in system or "judge" in system.lower():
            role = "judge"
        print(
            f"[mock-openai] role={role:9s} model={model} resp={len(content)} chars",
            file=sys.stderr,
        )

    def log_message(self, fmt: str, *args) -> None:  # 静默默认访问日志
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 OpenAI 兼容验证服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"本地 OpenAI 兼容服务器已启动: http://{args.host}:{args.port}/v1")
    print("把 .env 各角色 BASE_URL 指向该地址即可走完整在线链路（Ctrl+C 退出）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
        server.server_close()


if __name__ == "__main__":
    main()
