"""M3 供给层演示：双路召回 → RRF 融合 → 精排 → 三道闸门。

零依赖运行（mock 栈：哈希嵌入 + 内存向量 + 本地 BM25）：

    uv run python examples/demo_retrieval.py

演示三个场景：
1. 正常检索：知识库 + 患者记忆双路召回，患者分区隔离；
2. 输入闸门拦截：青霉素过敏患者查询含过敏药物 → fail-closed；
3. 装配闸门过滤：查询干净但召回证据含过敏药物 → 过滤后放行。
"""

from __future__ import annotations

from harness_agent.contracts.retrieval import RetrievalQuery, StoredChunk
from harness_agent.models.evidence import EvidencePack
from harness_agent.retrieval.wiring import build_retrieval_stack

PAT_PENICILLIN = "pat-001"  # M2 种子数据：青霉素过敏（阻断 beta_lactam 全组）
PAT_CLEAN = "pat-003"  # M2 种子数据：无已知过敏


def _chunk(content: str, chunk_id: str, patient_id: str | None = None) -> StoredChunk:
    return StoredChunk(chunk_id=chunk_id, patient_id=patient_id, content=content)


def _describe(pack: EvidencePack) -> None:
    verdict = pack.assembly_gate
    print(f"    is_reviewed={pack.is_reviewed}  gate={verdict.gate}  allowed={verdict.allowed}")
    print(f"    blocked_drugs={pack.blocked_drugs or '（无）'}")
    print(f"    reason: {verdict.reason}")
    for evidence in pack.evidence:
        marker = "[补全]" if evidence.is_structural_completion else "[命中]"
        print(f"    {marker} {evidence.source.chunk_id}: {evidence.content[:40]}")


def main() -> None:
    stack = build_retrieval_stack()

    # ---- 入库：共享知识库 + 两位患者的隔离记忆 ----
    stack.service.index(
        [
            _chunk("阿奇霉素的适应证：社区获得性肺炎等感染", "kb-1"),
            _chunk("青霉素类药物的皮试要求与用法", "kb-2"),
            _chunk("血糖监测的目标范围与频率建议", "kb-3"),
            _chunk("患者甲的血糖随访记录：控制平稳", "mem-a", patient_id="pat-A"),
        ]
    )

    print("=" * 72)
    print("场景 1：正常检索（知识库 + 患者记忆双路，分区隔离）")
    print("=" * 72)
    pack = stack.service.retrieve(
        RetrievalQuery(text="血糖 随访 记录", patient_id="pat-A", top_k=3)
    )
    _describe(pack)
    print("    -- 同一查询换成 pat-B：患者甲的记忆不可见 --")
    pack_b = stack.service.retrieve(
        RetrievalQuery(text="血糖 随访 记录", patient_id="pat-B", top_k=3)
    )
    _describe(pack_b)

    print()
    print("=" * 72)
    print("场景 2：输入闸门拦截（青霉素过敏患者查询过敏药物）")
    print("=" * 72)
    pack = stack.service.retrieve(
        RetrievalQuery(text="青霉素类抗生素怎么用", patient_id=PAT_PENICILLIN, top_k=3)
    )
    _describe(pack)

    print()
    print("=" * 72)
    print("场景 3：装配闸门过滤（查询干净，召回证据含过敏药物）")
    print("=" * 72)
    stack.service.index([_chunk("阿莫西林胶囊的用法用量说明", "kb-4")])
    pack = stack.service.retrieve(
        RetrievalQuery(text="胶囊 用法 用量", patient_id=PAT_PENICILLIN, top_k=3)
    )
    _describe(pack)

    print()
    print("=" * 72)
    print("场景 4：MCP 工具面（可选：uv sync --extra mcp 后可用）")
    print("=" * 72)
    try:
        from harness_agent.mcp.retrieval import create_retrieval_mcp_server

        server = create_retrieval_mcp_server()
        print(f"    FastMCP server 就绪: {server.name}（tools: retrieve / index_chunks）")
    except ImportError as exc:
        print(f"    未安装 fastmcp（{exc}）——核心检索不依赖它，属预期")


if __name__ == "__main__":
    main()
