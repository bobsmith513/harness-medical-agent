"""接口契约测试（M1）。

用最小 mock 实现验证：所有 Protocol 契约可被普通类满足
（结构化 isinstance 检查）——这保证后续 M2-M7 的 mock 实现与
真实实现（外部端点留空）在同一契约下无差别注入。
"""

from __future__ import annotations

from factories import make_conclusion, make_evidence_pack, make_session
from harness_agent.contracts.experts import (
    ContextBundle,
    Expert,
    ExpertTask,
    MemoryExpert,
    ReasoningExpert,
)
from harness_agent.contracts.gates import (
    AssemblyGate,
    ClinicalQuery,
    Gate,
    InputGate,
    QualityGate,
)
from harness_agent.contracts.llm import LLMClient, LLMMessage, LLMResult
from harness_agent.contracts.observability import (
    AuditStore,
    CacheStore,
    Desensitizer,
    DistLock,
    Tracer,
)
from harness_agent.contracts.retrieval import (
    EmbeddingProvider,
    Reranker,
    RetrievalQuery,
    RetrievalService,
    SparseRetriever,
    VectorStore,
)
from harness_agent.contracts.sandbox import Checkpoint, SandboxRuntime
from harness_agent.models.audit import AuditRecord, GateVerdict, TraceEvent
from harness_agent.models.common import new_id
from harness_agent.models.session import SessionContext


# ---------------------------------------------------------------------------
# 最小 mock 实现（仅供契约验证；正式 mock 实现属于后续里程碑）
# ---------------------------------------------------------------------------
class FakeLLM:
    role = "router"

    def complete(self, messages, *, temperature=0.2, max_tokens=None):
        return LLMResult(text="ok", model="fake")


class FakeEmbedding:
    def embed(self, texts):
        return [[0.0, 1.0] for _ in texts]


class FakeVectorStore:
    def upsert(self, items, embeddings):
        pass

    def search(self, query, embedding, top_k):
        return []


class FakeSparse:
    def search(self, query, top_k):
        return []


class FakeReranker:
    def rerank(self, query, items, top_k):
        return items[:top_k]


class FakeRetrieval:
    def retrieve(self, query):
        return make_evidence_pack()


class FakeExpert:
    name = "lab_expert"
    description = "检验解读专家"

    def run(self, task, context):
        return {"summary": "ok"}


class FakeReasoningExpert:
    name = "reasoning_expert"

    def reason(self, task, evidence, context):
        return make_conclusion()


class FakeMemoryExpert:
    name = "memory_expert"

    def assemble(self, query, context):
        return ContextBundle(patient_id=query.patient_id)


class FakeGate:
    name = "gate:input"

    def check(self, payload, context=None):
        return GateVerdict(gate="input", allowed=True)


class FakeQualityGate:
    name = "gate:quality_judge"
    threshold = 0.7

    def evaluate(self, conclusion, evidence):
        return GateVerdict(gate="quality_judge", allowed=True)


class FakeSandbox:
    backend = "mock"

    def execute(self, code, language="python", timeout_s=30.0):
        from harness_agent.contracts.sandbox import ExecutionResult

        return ExecutionResult(exit_code=0, stdout="done")

    def save_checkpoint(self, session_id, state):
        return Checkpoint(session_id=session_id, state=state)

    def restore(self, checkpoint):
        return True


class FakeTracer:
    def bind(self, session_id, trace_id):
        pass

    def record(self, event):
        pass


class FakeAuditStore:
    def append(self, record):
        pass

    def query(self, session_id):
        return []


class FakeCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ttl_s=None):
        self._store[key] = value


class FakeLock:
    def acquire(self, key, ttl_s):
        return True

    def release(self, key):
        pass


class FakeDesensitizer:
    def desensitize(self, text):
        from harness_agent.contracts.observability import DesensitizedText

        return DesensitizedText(text="***", removed_entities=["患者姓名"])


class NotAGate:
    """缺 check 方法：不应满足 Gate 契约。"""

    name = "broken"


# ---------------------------------------------------------------------------
# 契约满足性验证
# ---------------------------------------------------------------------------
class TestLLMContract:
    def test_fake_llm_satisfies(self):
        assert isinstance(FakeLLM(), LLMClient)

    def test_complete_returns_result(self):
        result = FakeLLM().complete([LLMMessage(role="user", content="你好")])
        assert isinstance(result, LLMResult)
        assert result.text == "ok"


class TestRetrievalContract:
    def test_embedding_provider(self):
        assert isinstance(FakeEmbedding(), EmbeddingProvider)

    def test_vector_store(self):
        assert isinstance(FakeVectorStore(), VectorStore)

    def test_sparse_retriever(self):
        assert isinstance(FakeSparse(), SparseRetriever)

    def test_reranker(self):
        assert isinstance(FakeReranker(), Reranker)

    def test_retrieval_service(self):
        assert isinstance(FakeRetrieval(), RetrievalService)

    def test_retrieval_query_satisfies_clinical_query(self):
        """RetrievalQuery 天然满足 ClinicalQuery 载荷形态（patient_id + text）。"""
        query = RetrievalQuery(text="查询", patient_id="pat-001")
        assert isinstance(query, ClinicalQuery)


class TestExpertContract:
    def test_generic_expert(self):
        assert isinstance(FakeExpert(), Expert)

    def test_reasoning_expert(self):
        assert isinstance(FakeReasoningExpert(), ReasoningExpert)

    def test_reasoning_expert_produces_conclusion(self):
        conclusion = FakeReasoningExpert().reason(
            ExpertTask(expert="reasoning_expert", instruction="推理"),
            make_evidence_pack(),
            make_session(),
        )
        assert conclusion.produced_by == "reasoning_expert"

    def test_memory_expert(self):
        assert isinstance(FakeMemoryExpert(), MemoryExpert)


class TestGateContract:
    def test_generic_gate(self):
        assert isinstance(FakeGate(), Gate)

    def test_gate_returns_verdict(self):
        verdict = FakeGate().check({"text": "查询"})
        assert isinstance(verdict, GateVerdict)
        assert verdict.allowed is True

    def test_input_gate_contract(self):
        assert isinstance(FakeGate(), InputGate)

    def test_assembly_gate_contract(self):
        assert isinstance(FakeGate(), AssemblyGate)

    def test_quality_gate_contract(self):
        assert isinstance(FakeQualityGate(), QualityGate)

    def test_broken_class_not_a_gate(self):
        assert not isinstance(NotAGate(), Gate)


class TestSandboxContract:
    def test_sandbox_runtime(self):
        assert isinstance(FakeSandbox(), SandboxRuntime)

    def test_checkpoint_roundtrip_shape(self):
        sandbox = FakeSandbox()
        cp = sandbox.save_checkpoint("sess-1", {"task": "dose_calc"})
        assert sandbox.restore(cp) is True
        assert cp.state == {"task": "dose_calc"}


class TestObservabilityContract:
    def test_tracer(self):
        assert isinstance(FakeTracer(), Tracer)

    def test_audit_store(self):
        assert isinstance(FakeAuditStore(), AuditStore)

    def test_cache_store(self):
        assert isinstance(FakeCache(), CacheStore)

    def test_dist_lock(self):
        assert isinstance(FakeLock(), DistLock)

    def test_desensitizer(self):
        assert isinstance(FakeDesensitizer(), Desensitizer)

    def test_desensitize_returns_model(self):
        result = FakeDesensitizer().desensitize("张三的检验结果")
        assert result.text == "***"
        assert result.removed_entities == ["患者姓名"]


class TestContractWireTogether:
    """契约模型可以端到端串起一次门禁裁决（类型互操作性冒烟）。"""

    def test_gate_verdict_into_audit_record(self):
        verdict = FakeQualityGate().evaluate(make_conclusion(), make_evidence_pack())
        record = AuditRecord(
            trace_id=new_id("tr"),
            session_id="sess-1",
            turn_index=1,
            actor="gate:quality_judge",
            action="gate_check",
            verdict=verdict,
        )
        assert record.verdict is not None
        assert record.verdict.gate == "quality_judge"

    def test_trace_event_from_session(self):
        session: SessionContext = make_session()
        event = TraceEvent(
            trace_id=new_id("tr"),
            session_id=session.session_id,
            event_type="route",
            payload={"decision": "need_reasoning"},
        )
        assert event.payload["decision"] == "need_reasoning"
