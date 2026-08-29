"""M3 供给层门面端到端测试：闸门串联 + fail-closed + 分区隔离 + 同父补全。

场景基于 M2 种子安全栈：
- pat-001 青霉素过敏 → 阻断 beta_lactam 全组（penicillin/amoxicillin/ceftriaxone）
- pat-003 无过敏记录 → 不阻断任何药物

锁定语义：
1. 输入闸门拦截 → 空证据包 + gate=input 裁决（is_reviewed=False）；
2. 装配闸门过滤 → 含过敏药物实体的证据被移除后放行；
3. 全部证据被过滤 → 拒绝交付空证据包（fail-closed）；
4. 患者分区隔离贯穿门面（查询只携带本患者分区 + 共享知识库）；
5. 同父补全：sibling_ids 取回相邻 chunk，标记 is_structural_completion；
   跨患者 chunk 永不补全（``get_chunk`` 无 patient_id 条件，校验在门面层）。
"""

from __future__ import annotations

import pytest

from harness_agent.contracts.retrieval import (
    EmbeddingProvider,
    RetrievalQuery,
    RetrievalService,
    StoredChunk,
)
from harness_agent.retrieval.bm25 import BM25SparseRetriever
from harness_agent.retrieval.embeddings import HashingEmbeddingProvider
from harness_agent.retrieval.fusion import IdentityReranker
from harness_agent.retrieval.service import HybridRetrievalService
from harness_agent.retrieval.vector_store import InMemoryVectorStore
from harness_agent.retrieval.wiring import build_retrieval_stack
from harness_agent.safety import build_safety_stack

PAT_PENICILLIN = "pat-001"  # 青霉素过敏（阻断 beta_lactam 全组）
PAT_CLEAN = "pat-003"  # 无已知过敏


def _chunk(
    content: str,
    chunk_id: str,
    *,
    patient_id: str | None = None,
    sibling_ids: list[str] | None = None,
    parent_id: str | None = None,
    metadata: dict[str, str] | None = None,
) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        patient_id=patient_id,
        content=content,
        sibling_ids=sibling_ids or [],
        parent_id=parent_id,
        metadata=metadata or {},
    )


def _service() -> HybridRetrievalService:
    """零依赖本地栈：哈希嵌入 + 内存存储 + BM25 + identity 精排 + 种子安全栈。"""
    safety = build_safety_stack()
    return HybridRetrievalService(
        embedding_provider=HashingEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        sparse=BM25SparseRetriever(),
        reranker=IdentityReranker(),
        input_gate=safety.input_gate,
        assembly_gate=safety.assembly_gate,
        resolver=safety.resolver,
    )


def _query(text: str, patient_id: str = PAT_CLEAN, *, top_k: int = 5) -> RetrievalQuery:
    return RetrievalQuery(text=text, patient_id=patient_id, top_k=top_k)


class TestFacadeContract:
    def test_satisfies_retrieval_service_contract(self):
        assert isinstance(_service(), RetrievalService)

    def test_missing_chunk_lookup_capability_rejected(self):
        safety = build_safety_stack()

        class _NoGetChunkStore:
            def upsert(self, items, embeddings) -> None: ...

            def search(self, query, embedding, top_k):
                return []

        with pytest.raises(TypeError, match="get_chunk"):
            HybridRetrievalService(
                embedding_provider=HashingEmbeddingProvider(),
                vector_store=_NoGetChunkStore(),
                sparse=BM25SparseRetriever(),
                reranker=IdentityReranker(),
                input_gate=safety.input_gate,
                assembly_gate=safety.assembly_gate,
                resolver=safety.resolver,
            )

    def test_missing_sparse_upsert_capability_rejected(self):
        safety = build_safety_stack()

        class _NoUpsertSparse:
            def search(self, query, top_k):
                return []

        with pytest.raises(TypeError, match="upsert"):
            HybridRetrievalService(
                embedding_provider=HashingEmbeddingProvider(),
                vector_store=InMemoryVectorStore(),
                sparse=_NoUpsertSparse(),
                reranker=IdentityReranker(),
                input_gate=safety.input_gate,
                assembly_gate=safety.assembly_gate,
                resolver=safety.resolver,
            )


class TestHappyPath:
    def test_retrieve_returns_reviewed_pack_with_traceable_evidence(self):
        service = _service()
        service.index(
            [
                _chunk("阿奇霉素的适应证与大环内酯类作用机制", "kb-1"),
                _chunk("血糖监测的目标范围与频率建议", "kb-2"),
            ]
        )
        pack = service.retrieve(_query("阿奇霉素 适应证"))

        assert pack.is_reviewed is True
        assert pack.patient_id == PAT_CLEAN
        assert pack.blocked_drugs == []  # 无过敏患者：阻断集为空
        assert len(pack.evidence) >= 1
        top = pack.evidence[0]
        assert "阿奇霉素" in top.content
        # 来源可回溯：chunk_id / doc_id / 精排分数
        assert top.source.chunk_id == "kb-1"
        assert top.source.source_type == "document"
        assert top.score is not None and top.score > 0
        # 共享知识库条目默认 provenance
        assert top.provenance == "knowledge_base"
        assert top.confidence == "medium"

    def test_patient_memory_provenance_and_partition(self):
        service = _service()
        service.index(
            [
                _chunk("患者甲的血糖随访记录", "mem-a1", patient_id="pat-A"),
                _chunk("血糖管理通用指南", "kb-1"),
            ]
        )
        # pat-A 查询：自己记忆 + 共享知识库
        pack_a = service.retrieve(_query("血糖 随访", patient_id="pat-A"))
        assert pack_a.is_reviewed is True
        ids_a = {e.source.chunk_id for e in pack_a.evidence}
        assert "mem-a1" in ids_a and "kb-1" in ids_a
        # 患者记忆默认 provenance=doctor_verified（审核通过才可入库，M6 强制）
        memory = next(e for e in pack_a.evidence if e.source.chunk_id == "mem-a1")
        assert memory.provenance == "doctor_verified"

        # pat-B 查询同一文本：只见共享知识库（隔离贯穿门面）
        pack_b = service.retrieve(_query("血糖 随访", patient_id="pat-B"))
        ids_b = {e.source.chunk_id for e in pack_b.evidence}
        assert "mem-a1" not in ids_b
        assert "kb-1" in ids_b

    def test_session_id_and_query_echoed(self):
        service = _service()
        service.index([_chunk("降糖药物的分类与选择", "kb-1")])
        query = RetrievalQuery(text="降糖药物", patient_id=PAT_CLEAN, session_id="sess-42", top_k=3)
        pack = service.retrieve(query)
        assert pack.session_id == "sess-42"
        assert pack.query == "降糖药物"

    def test_metadata_overrides_provenance_and_confidence(self):
        service = _service()
        service.index(
            [
                _chunk(
                    "经医生复核的用药方案",
                    "kb-1",
                    metadata={"provenance": "doctor_verified", "confidence": "high"},
                )
            ]
        )
        pack = service.retrieve(_query("用药方案"))
        assert pack.evidence[0].provenance == "doctor_verified"
        assert pack.evidence[0].confidence == "high"


class TestInputGateIntegration:
    def test_allergy_query_blocked_before_retrieval(self):
        """查询命中患者过敏药物：检索不发生，fail-closed 交付。"""
        service = _service()
        service.index([_chunk("青霉素的皮试要求与用法", "kb-1")])

        pack = service.retrieve(_query("青霉素类抗生素怎么用", patient_id=PAT_PENICILLIN))
        assert pack.is_reviewed is False
        assert pack.evidence == []  # 输入闸门拦截：无召回发生
        assert pack.assembly_gate.gate == "input"
        assert pack.assembly_gate.allowed is False
        assert "penicillin" in pack.blocked_drugs
        # 阻断集为全集（直接过敏 + ATC 交叉反应组）
        assert set(pack.blocked_drugs) >= {"penicillin", "amoxicillin", "ceftriaxone"}

    def test_cross_reactant_mention_also_blocked(self):
        """青霉素过敏患者查询头孢曲松（交叉反应）：同样在入口拦截。"""
        service = _service()
        pack = service.retrieve(_query("头孢曲松怎么用", patient_id=PAT_PENICILLIN))
        assert pack.is_reviewed is False
        assert "ceftriaxone" in pack.assembly_gate.blocked_drugs


class TestAssemblyGateIntegration:
    def test_offending_evidence_filtered_but_pack_delivered(self):
        """查询干净但召回证据含过敏药物：过滤后放行，裁决记录过滤明细。"""
        service = _service()
        service.index(
            [
                _chunk("青霉素类抗生素的用法与皮试要求", "kb-dirty"),
                _chunk("阿奇霉素的适应证与用法", "kb-clean"),
            ]
        )
        # 查询不提药名（不触发输入闸门），但召回证据含青霉素
        pack = service.retrieve(_query("抗生素的用法", patient_id=PAT_PENICILLIN))
        assert pack.is_reviewed is True
        kept_ids = {e.source.chunk_id for e in pack.evidence}
        assert "kb-dirty" not in kept_ids  # 过敏证据被移除
        assert pack.assembly_gate.allowed is True
        assert "penicillin" in pack.assembly_gate.reason or pack.assembly_gate.blocked_drugs

    def test_all_evidence_offending_fails_closed(self):
        """全部召回证据均含过敏药物：拒绝交付空证据包。"""
        service = _service()
        service.index(
            [
                _chunk("阿莫西林胶囊的用法用量说明", "kb-1"),
                _chunk("青霉素注射剂的配伍与皮试", "kb-2"),
            ]
        )
        pack = service.retrieve(_query("胶囊 用法 用量", patient_id=PAT_PENICILLIN))
        assert pack.is_reviewed is False
        assert pack.evidence == []
        assert pack.assembly_gate.allowed is False

    def test_clean_patient_gets_full_evidence(self):
        """无过敏患者：同一批证据全部放行（闸门不误伤）。"""
        service = _service()
        service.index(
            [
                _chunk("阿莫西林胶囊的用法用量说明", "kb-1"),
                _chunk("阿奇霉素的适应证与用法", "kb-2"),
            ]
        )
        pack = service.retrieve(_query("阿莫西林 胶囊 用法", patient_id=PAT_CLEAN))
        assert pack.is_reviewed is True
        assert {e.source.chunk_id for e in pack.evidence} == {"kb-1", "kb-2"}


class TestSiblingCompletion:
    def test_sibling_fetched_as_structural_completion(self):
        """命中 chunk 的相邻 sibling 被补全为结构性证据（低置信、不计分）。"""
        service = _service()
        service.index(
            [
                _chunk(
                    "血糖监测的目标范围",
                    "c1",
                    sibling_ids=["c2"],
                    parent_id="doc-1",
                ),
                _chunk("饮食运动配合建议", "c2", sibling_ids=["c1"], parent_id="doc-1"),
            ]
        )
        # top_k=1：只保留最强命中 c1，c2 经 sibling 链补全
        pack = service.retrieve(_query("血糖 监测 目标", top_k=1))
        assert pack.is_reviewed is True

        hit = next(e for e in pack.evidence if e.source.chunk_id == "c1")
        completion = next(e for e in pack.evidence if e.source.chunk_id == "c2")
        assert hit.is_structural_completion is False
        assert hit.score is not None
        assert completion.is_structural_completion is True
        assert completion.confidence == "low"
        assert completion.score is None
        assert completion.source.parent_id == "doc-1"
        # 高置信证据排除结构性补全
        assert hit in pack.high_confidence_evidence
        assert completion not in pack.high_confidence_evidence

    def test_sibling_already_hit_not_duplicated(self):
        """sibling 本身已在命中集：作为常规证据出现，不重复补全。"""
        service = _service()
        service.index(
            [
                _chunk("血糖监测目标", "c1", sibling_ids=["c2"], parent_id="doc-1"),
                _chunk("血糖监测频率", "c2", sibling_ids=["c1"], parent_id="doc-1"),
            ]
        )
        pack = service.retrieve(_query("血糖 监测", top_k=5))
        ids = [e.source.chunk_id for e in pack.evidence]
        assert sorted(ids) == ["c1", "c2"]
        # 两个都是常规命中（非结构补全）
        assert all(not e.is_structural_completion for e in pack.evidence)

    def test_same_patient_sibling_is_completed(self):
        """本患者分区的 sibling 正常补全（隔离校验不得误伤）。"""
        service = _service()
        service.index(
            [
                _chunk("血糖监测目标范围", "kb-1", sibling_ids=["mem-a"], parent_id="doc-1"),
                _chunk("本患者的既往病史摘要", "mem-a", patient_id=PAT_CLEAN, parent_id="doc-1"),
            ]
        )
        pack = service.retrieve(_query("血糖 监测 目标", patient_id=PAT_CLEAN, top_k=1))
        ids = [e.source.chunk_id for e in pack.evidence]
        assert "kb-1" in ids
        assert "mem-a" in ids

    def test_cross_patient_sibling_never_enters_evidence(self):
        """跨患者 sibling 永不进入证据包（分区隔离兜底）。

        ``vector_store.get_chunk`` 是按 chunk_id 的全局直查，三处实现
        （内存 / Milvus / BM25）均无 patient_id 条件，因此归属校验只能在
        门面层完成——否则患者记忆可经 sibling 链泄漏给其他患者，绕过
        "跨患者内容不是过滤后丢弃，而是从不读起"的隔离语义。
        """
        service = _service()
        service.index(
            [
                _chunk("血糖监测目标范围", "kb-1", sibling_ids=["mem-b"], parent_id="doc-1"),
                _chunk(
                    "另一位患者的既往病史摘要",
                    "mem-b",
                    patient_id="pat-002",
                    parent_id="doc-1",
                ),
            ]
        )
        pack = service.retrieve(_query("血糖 监测 目标", patient_id=PAT_CLEAN, top_k=1))
        ids = [e.source.chunk_id for e in pack.evidence]
        assert "kb-1" in ids  # 命中项本身照常交付
        assert "mem-b" not in ids  # 跨患者内容不泄漏


class TestRecallDepthFloor:
    """召回深度下限回归：query.top_k 大于配置深度时不得截断目标条目。

    背景：患者记忆与共享知识库竞争固定名次。若双路召回深度恒为
    配置值（8），排在深度之外的患者记忆永远进不了融合窗口——
    深度下限应取 ``max(配置深度, query.top_k)``。
    """

    def test_top_k_beyond_configured_depth_still_recalls(self):
        """top_k=12 > 配置深度 8：第 9-12 名候选仍可被召回。"""
        service = _service()
        # 10 条知识库条目共享查询词面（占据融合头部），1 条患者记忆垫底
        service.index(
            [_chunk(f"高血压治疗方案说明 {i}", f"kb-{i}") for i in range(10)]
            + [_chunk("血型 O 型", "mem-1", patient_id=PAT_CLEAN)]
        )
        pack = service.retrieve(_query("高血压治疗方案", PAT_CLEAN, top_k=12))
        ids = [e.source.chunk_id for e in pack.evidence]
        # 垫底的患者记忆没有被固定深度窗口截断
        assert "mem-1" in ids

    def test_default_top_k_unaffected_by_floor(self):
        """top_k=5 ≤ 配置深度：行为与旧实现一致（下限不放大窗口）。"""
        service = _service()
        service.index(
            [_chunk(f"高血压治疗方案说明 {i}", f"kb-{i}") for i in range(10)]
            + [_chunk("血型 O 型", "mem-1", patient_id=PAT_CLEAN)]
        )
        pack = service.retrieve(_query("高血压治疗方案", PAT_CLEAN, top_k=5))
        assert len(pack.evidence) == 5
        assert "mem-1" not in [e.source.chunk_id for e in pack.evidence]


class TestEdgeCases:
    def test_empty_store_returns_allowed_empty_pack(self):
        """无召回：诚实空包（裁决=无需过滤），供推理专家如实回应无证据。"""
        service = _service()
        pack = service.retrieve(_query("任意查询"))
        assert pack.is_reviewed is True
        assert pack.evidence == []

    def test_index_empty_is_noop(self):
        service = _service()
        service.index([])
        assert service.retrieve(_query("任意")).evidence == []


class TestWiringFactory:
    def test_default_stack_is_zero_dependency_and_functional(self):
        """默认配置装配的栈：哈希嵌入 + 内存存储 + BM25 + identity 精排。"""
        stack = build_retrieval_stack()
        assert isinstance(stack.service, RetrievalService)
        assert isinstance(stack.embedding_provider, HashingEmbeddingProvider)
        assert isinstance(stack.embedding_provider, EmbeddingProvider)
        assert isinstance(stack.vector_store, InMemoryVectorStore)
        assert isinstance(stack.reranker, IdentityReranker)

        stack.service.index([_chunk("糖尿病饮食控制的要点", "kb-1")])
        pack = stack.service.retrieve(_query("糖尿病 饮食"))
        assert pack.is_reviewed is True
        assert len(pack.evidence) >= 1

    def test_service_and_stack_share_components(self):
        stack = build_retrieval_stack()
        # 门面持有的正是栈暴露的组件（替换即全局生效）
        assert stack.service._embedding_provider is stack.embedding_provider  # noqa: SLF001

    def test_injected_safety_stack_is_reused(self):
        """注入的安全栈必须被检索层复用——两端阻断口径一致的前提。"""
        safety = build_safety_stack()
        stack = build_retrieval_stack(safety=safety)
        assert stack.safety is safety
        assert stack.service._input_gate is safety.input_gate  # noqa: SLF001
        assert stack.service._assembly_gate is safety.assembly_gate  # noqa: SLF001
        assert stack.service._resolver is safety.resolver  # noqa: SLF001

    def test_without_injection_each_stack_builds_its_own(self):
        """未注入时各自新建（向后兼容，但两端口径可能分叉——故不推荐）。"""
        assert build_retrieval_stack().safety is not build_retrieval_stack().safety
