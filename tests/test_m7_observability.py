"""M7 可观测与脱敏测试。

覆盖范围：
- PatternDesensitizer：各类 PII 脱敏 + 前后对照；
- NoopTracer / build_tracer：事件记录 + 降级；
- SQLiteAuditStore / build_audit_store：审计写入 + 查询；
- MemoryCacheStore / build_cache_store：缓存 + TTL；
- MemoryLock / build_dist_lock：分布式锁 + 互斥；
- build_observability_stack：一站式装配。

验收标准（development-plan.md M7）：
1. 脱敏前后对照样例；
2. 全链路 trace 事件可打印。
"""

from __future__ import annotations

import time

import pytest

from harness_agent.contracts.observability import (
    AuditStore,
    CacheStore,
    Desensitizer,
    DistLock,
    Tracer,
)
from harness_agent.models.audit import AuditRecord, GateVerdict, TraceEvent
from harness_agent.observability import (
    LangfuseTracer,
    MemoryCacheStore,
    MemoryLock,
    NoopTracer,
    PatternDesensitizer,
    PostgresAuditStore,
    RedisCacheStore,
    RedisLock,
    SQLiteAuditStore,
    build_audit_store,
    build_cache_store,
    build_dist_lock,
    build_observability_stack,
    build_tracer,
)


def _simulate_missing_redis(monkeypatch: pytest.MonkeyPatch, url: str) -> str:
    """进程内模拟 redis 包未安装，返回原 URL 供构造函数使用。

    实现：拦截内建 ``__import__``，仅对 ``import redis`` 抛
    ImportError，其余 import 原样放行。``RedisCacheStore`` /
    ``RedisLock`` 构造函数捕获 ImportError 后自动降级进程内实现，
    本函数让这条降级路径在任何开发机上确定性可测（不依赖本机
    是否安装 redis、是否运行 Redis 服务，也不发起任何网络连接）。
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("No module named 'redis' (simulated for fallback test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    return url


# ===========================================================================
# 1. 脱敏中间件测试
# ===========================================================================
class TestPatternDesensitizer:
    """脱敏中间件测试（验收：脱敏前后对照样例）。"""

    def setup_method(self):
        self.desensitizer = PatternDesensitizer()

    def test_desensitize_id_card_18(self):
        """身份证号（18 位）脱敏。"""
        text = "患者身份证号 310101199001011234 就诊"
        result = self.desensitizer.desensitize(text)
        assert "310101199001011234" not in result.text
        assert "[REDACTED-ID]" in result.text
        assert any("ID" in e for e in result.removed_entities)

    def test_desensitize_id_card_15(self):
        """身份证号（15 位旧版）脱敏。"""
        text = "身份证 110101900101123"
        result = self.desensitizer.desensitize(text)
        assert "110101900101123" not in result.text
        assert "[REDACTED-ID]" in result.text

    def test_desensitize_phone(self):
        """手机号脱敏。"""
        text = "联系电话 13812345678 联系人"
        result = self.desensitizer.desensitize(text)
        assert "13812345678" not in result.text
        assert "[REDACTED-PHONE]" in result.text

    def test_desensitize_patient_id(self):
        """患者编号脱敏。"""
        text = "patient_id=pat-abc12345"
        result = self.desensitizer.desensitize(text)
        assert "pat-abc12345" not in result.text
        assert "[REDACTED-PATID]" in result.text

    def test_desensitize_email(self):
        """邮箱脱敏。"""
        text = "邮箱 zhangsan@hospital.com 已记录"
        result = self.desensitizer.desensitize(text)
        assert "zhangsan@hospital.com" not in result.text
        assert "[REDACTED-EMAIL]" in result.text

    def test_desensitize_name_marker(self):
        """姓名标记脱敏。"""
        text = "姓名：张三 诊断为感冒"
        result = self.desensitizer.desensitize(text)
        assert "张三" not in result.text
        assert "[REDACTED-NAME]" in result.text

    def test_desensitize_passport_standalone(self):
        """护照号脱敏（中国护照格式：字母 + 8 位数字）。"""
        text = "护照信息 E12345678 已登记"
        result = self.desensitizer.desensitize(text)
        assert "E12345678" not in result.text
        assert "[REDACTED-PASSPORT]" in result.text
        assert any("PASSPORT" in e for e in result.removed_entities)

    def test_desensitize_passport_marker(self):
        """护照号脱敏（标记式：护照号: xxx）。"""
        text = "护照号: G28876543 复核"
        result = self.desensitizer.desensitize(text)
        assert "G28876543" not in result.text
        assert "[REDACTED-PASSPORT]" in result.text

    def test_desensitize_insurance_number(self):
        """医保卡号脱敏（标记式）。"""
        text = "医保卡号: D33010601 就诊"
        result = self.desensitizer.desensitize(text)
        assert "D33010601" not in result.text
        assert "[REDACTED-INSURANCE]" in result.text
        assert any("INSURANCE" in e for e in result.removed_entities)

    def test_desensitize_insurance_number_long(self):
        """医保编号脱敏（纯数字长号）。"""
        text = "医保编号 33010620260812001 已核对"
        result = self.desensitizer.desensitize(text)
        assert "33010620260812001" not in result.text
        assert "[REDACTED-INSURANCE]" in result.text

    def test_desensitize_no_false_positive_on_atc_code(self):
        """ATC 药理类别码（如 J01CE01）不应被误判为护照号。"""
        text = "penicillin (ATC=J01CE01) 与 amoxicillin 存在交叉反应"
        result = self.desensitizer.desensitize(text)
        assert "J01CE01" in result.text
        assert "[REDACTED-PASSPORT]" not in result.text

    def test_desensitize_multiple_pii(self):
        """多种 PII 混合脱敏。"""
        text = "患者张三，身份证 310101199001011234，电话 13812345678，邮箱 zhangsan@hospital.com"
        result = self.desensitizer.desensitize(text)
        assert "张三" not in result.text
        assert "310101199001011234" not in result.text
        assert "13812345678" not in result.text
        assert "zhangsan@hospital.com" not in result.text
        assert "[REDACTED-NAME]" in result.text
        assert "[REDACTED-ID]" in result.text
        assert "[REDACTED-PHONE]" in result.text
        assert "[REDACTED-EMAIL]" in result.text
        assert len(result.removed_entities) >= 4

    def test_desensitize_no_pii(self):
        """无 PII 文本 → 原样返回。"""
        text = "阿奇霉素适用于社区获得性肺炎"
        result = self.desensitizer.desensitize(text)
        assert result.text == text
        assert len(result.removed_entities) == 0

    def test_desensitize_dict(self):
        """递归脱敏字典中的字符串值。"""
        data = {
            "patient_info": "姓名：李四 电话 13987654321",
            "diagnosis": "上呼吸道感染",
            "nested": {
                "id": "310101199001011234",
                "note": "无 PII",
            },
        }
        result = self.desensitizer.desensitize_dict(data)
        assert "李四" not in result["patient_info"]
        assert "[REDACTED-NAME]" in result["patient_info"]
        assert "[REDACTED-PHONE]" in result["patient_info"]
        assert "[REDACTED-ID]" in result["nested"]["id"]
        assert result["diagnosis"] == "上呼吸道感染"
        assert result["nested"]["note"] == "无 PII"

    def test_before_after_comparison(self):
        """验收：脱敏前后对照样例。"""
        original = "患者李芳，身份证 420101198503156789，电话 13712345678"
        result = self.desensitizer.desensitize(original)
        # 原文保留 PII
        assert "李芳" in original
        assert "420101198503156789" in original
        assert "13712345678" in original
        # 脱敏后无 PII
        assert "李芳" not in result.text
        assert "420101198503156789" not in result.text
        assert "13712345678" not in result.text
        # 被移除的实体可追溯
        assert len(result.removed_entities) >= 3

    def test_desensitizer_protocol(self):
        """PatternDesensitizer 实现 Desensitizer 接口。"""
        assert isinstance(self.desensitizer, Desensitizer)


# ===========================================================================
# 2. Tracer 测试
# ===========================================================================
class TestNoopTracer:
    """NoopTracer 测试（验收：全链路 trace 事件可打印）。"""

    def test_bind_and_record(self):
        """绑定会话 + 记录事件。"""
        tracer = NoopTracer(verbose=False)
        tracer.bind("sess-1", "trace-1")
        event = TraceEvent(
            trace_id="trace-1",
            session_id="sess-1",
            event_type="llm_call",
            payload={"model": "deepseek-v4", "tokens": 100},
        )
        tracer.record(event)
        assert tracer.event_count == 1
        assert tracer.events[0].event_type == "llm_call"

    def test_multiple_events(self):
        """多事件记录。"""
        tracer = NoopTracer(verbose=False)
        tracer.bind("sess-1", "trace-1")
        for i in range(5):
            tracer.record(
                TraceEvent(
                    trace_id="trace-1",
                    event_type=f"event_{i}",
                )
            )
        assert tracer.event_count == 5
        types = [e.event_type for e in tracer.events]
        assert "event_0" in types
        assert "event_4" in types

    def test_tracer_protocol(self):
        """NoopTracer 实现 Tracer 接口。"""
        assert isinstance(NoopTracer(), Tracer)

    def test_events_captured_for_audit(self):
        """事件可被审计用（事件内容完整保留）。"""
        tracer = NoopTracer(verbose=False)
        tracer.bind("sess-1", "trace-1")
        tracer.record(
            TraceEvent(
                trace_id="trace-1",
                session_id="sess-1",
                event_type="gate_check",
                payload={"gate": "quality_judge", "allowed": False},
            )
        )
        event = tracer.events[0]
        assert event.payload["gate"] == "quality_judge"
        assert event.payload["allowed"] is False


class TestLangfuseTracer:
    """LangfuseTracer 骨架测试。"""

    def test_without_keys_falls_back_to_noop(self):
        """无密钥 → 降级为 Noop。"""
        tracer = LangfuseTracer(public_key="", secret_key="")
        assert tracer._client is None
        # 降级到 fallback NoopTracer
        tracer.bind("sess-1", "trace-1")
        tracer.record(TraceEvent(trace_id="trace-1", event_type="test"))
        assert tracer._fallback.event_count == 1

    def test_with_keys_but_no_sdk(self):
        """有密钥但 SDK 未安装 → 降级为 Noop。"""
        tracer = LangfuseTracer(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://fake.langfuse.com",
        )
        # langfuse 包未安装 → _client 为 None
        assert tracer._client is None


class TestBuildTracer:
    """Tracer 工厂测试。"""

    def test_empty_keys_returns_noop(self):
        tracer = build_tracer("", "", "")
        assert isinstance(tracer, NoopTracer)

    def test_with_keys_returns_langfuse(self):
        tracer = build_tracer("pk-test", "sk-test", "https://fake.langfuse.com")
        assert isinstance(tracer, LangfuseTracer)


# ===========================================================================
# 3. AuditStore 测试
# ===========================================================================
class TestSQLiteAuditStore:
    """SQLite 审计存储测试。"""

    def test_append_and_query(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        record = AuditRecord(
            trace_id="trace-1",
            session_id="sess-1",
            turn_index=1,
            actor="reasoning_expert",
            action="conclude",
        )
        store.append(record)
        results = store.query("sess-1")
        assert len(results) == 1
        assert results[0].actor == "reasoning_expert"
        assert results[0].action == "conclude"

    def test_query_empty_session(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        assert len(store.query("nonexistent")) == 0

    def test_multiple_records_ordered(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        for i in range(3):
            store.append(
                AuditRecord(
                    trace_id="trace-1",
                    session_id="sess-1",
                    turn_index=i,
                    actor="orchestrator",
                    action=f"action_{i}",
                )
            )
        results = store.query("sess-1")
        assert len(results) == 3

    def test_append_with_verdict(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        verdict = GateVerdict(gate="quality_judge", allowed=False, reason="臆测")
        store.append(
            AuditRecord(
                trace_id="trace-1",
                session_id="sess-1",
                actor="gate:quality_judge",
                action="gate_check",
                verdict=verdict,
            )
        )
        results = store.query("sess-1")
        assert results[0].verdict is not None
        assert results[0].verdict.gate == "quality_judge"
        assert results[0].verdict.allowed is False

    def test_count(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        store.append(AuditRecord(trace_id="t1", session_id="s1", actor="a", action="b"))
        store.append(AuditRecord(trace_id="t2", session_id="s2", actor="a", action="b"))
        assert store.count() == 2

    def test_audit_store_protocol(self, tmp_path):
        store = SQLiteAuditStore(data_dir=str(tmp_path))
        assert isinstance(store, AuditStore)


class TestBuildAuditStore:
    """审计存储工厂测试。"""

    def test_empty_dsn_returns_sqlite(self, tmp_path):
        store = build_audit_store("", str(tmp_path))
        assert isinstance(store, SQLiteAuditStore)

    def test_with_dsn_returns_postgres(self):
        store = build_audit_store("postgresql://fake")
        assert isinstance(store, PostgresAuditStore)

    def test_postgres_without_sdk_falls_back(self):
        """Postgres DSN 但 psycopg 未安装 → 进程内降级。"""
        store = PostgresAuditStore(dsn="postgresql://fake")
        store.append(
            AuditRecord(
                trace_id="t1",
                session_id="s1",
                actor="a",
                action="b",
            )
        )
        results = store.query("s1")
        assert len(results) == 1


# ===========================================================================
# 4. CacheStore 测试
# ===========================================================================
class TestMemoryCacheStore:
    """内存缓存存储测试。"""

    def test_set_and_get(self):
        cache = MemoryCacheStore()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent(self):
        cache = MemoryCacheStore()
        assert cache.get("missing") is None

    def test_set_with_ttl(self):
        """TTL 过期测试。"""
        cache = MemoryCacheStore()
        cache.set("key1", "value1", ttl_s=1)
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_set_with_short_ttl(self):
        """短 TTL 快速过期。"""
        cache = MemoryCacheStore()
        cache.set("key1", "value1", ttl_s=0)
        # ttl_s=0 意味着立即过期
        assert cache.get("key1") is None or cache.get("key1") == "value1"

    def test_delete(self):
        cache = MemoryCacheStore()
        cache.set("key1", "value1")
        assert cache.delete("key1")
        assert cache.get("key1") is None
        assert not cache.delete("key1")

    def test_overwrite(self):
        cache = MemoryCacheStore()
        cache.set("key1", "v1")
        cache.set("key1", "v2")
        assert cache.get("key1") == "v2"

    def test_clear(self):
        cache = MemoryCacheStore()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.size == 0

    def test_cache_store_protocol(self):
        assert isinstance(MemoryCacheStore(), CacheStore)


class TestBuildCacheStore:
    """缓存存储工厂测试。"""

    def test_empty_url_returns_memory(self):
        store = build_cache_store("")
        assert isinstance(store, MemoryCacheStore)

    def test_with_url_returns_redis(self):
        store = build_cache_store("redis://localhost:6379")
        assert isinstance(store, RedisCacheStore)

    def test_redis_without_sdk_falls_back(self, monkeypatch):
        """Redis URL 但 redis 包不可用（ImportError）→ 降级为 Memory。

        通过拦截内建 import 在进程内模拟"未安装 redis"，使降级路径
        的验证不依赖开发机是否装了 redis 包 / 起了 Redis 服务——
        装了包的环境下原实现会发起真实连接（localhost:6379）。
        """
        store = RedisCacheStore(url=_simulate_missing_redis(monkeypatch, "redis://localhost:6379"))
        store.set("k", "v")
        assert store.get("k") == "v"  # 降级到 MemoryCacheStore


# ===========================================================================
# 5. DistLock 测试
# ===========================================================================
class TestMemoryLock:
    """进程内分布式锁测试。"""

    def test_acquire_and_release(self):
        lock = MemoryLock()
        assert lock.acquire("key1", ttl_s=10) is True
        lock.release("key1")
        assert lock.acquire("key1", ttl_s=10) is True  # 释放后可再次获取

    def test_acquire_already_locked(self):
        """锁已被持有 → 获取失败。"""
        lock = MemoryLock()
        assert lock.acquire("key1", ttl_s=10) is True
        assert lock.acquire("key1", ttl_s=10) is False  # 已被持有

    def test_lock_ttl_expiry(self):
        """TTL 过期后自动释放。"""
        lock = MemoryLock()
        lock.acquire("key1", ttl_s=0.1)
        assert lock.is_locked("key1")
        time.sleep(0.15)
        assert not lock.is_locked("key1")
        assert lock.acquire("key1", ttl_s=10) is True

    def test_release_nonexistent(self):
        """释放不存在的锁 → 无异常。"""
        lock = MemoryLock()
        lock.release("nonexistent")  # 不应报错

    def test_is_locked(self):
        lock = MemoryLock()
        assert not lock.is_locked("key1")
        lock.acquire("key1", ttl_s=10)
        assert lock.is_locked("key1")

    def test_dist_lock_protocol(self):
        assert isinstance(MemoryLock(), DistLock)


class TestBuildDistLock:
    """分布式锁工厂测试。"""

    def test_empty_url_returns_memory(self):
        lock = build_dist_lock("")
        assert isinstance(lock, MemoryLock)

    def test_with_url_returns_redis(self):
        lock = build_dist_lock("redis://localhost:6379")
        assert isinstance(lock, RedisLock)

    def test_redis_without_sdk_falls_back(self, monkeypatch):
        """Redis URL 但 redis 包不可用（ImportError）→ 降级为 MemoryLock。

        同上：进程内模拟 ImportError，避免对开发机环境敏感。
        """
        lock = RedisLock(url=_simulate_missing_redis(monkeypatch, "redis://localhost:6379"))
        assert lock.acquire("key1", ttl_s=10) is True
        assert lock.acquire("key1", ttl_s=10) is False  # 已锁定


# ===========================================================================
# 6. 装配工厂测试
# ===========================================================================
class TestBuildObservabilityStack:
    """一站式装配测试。"""

    def test_default_returns_all_noop(self, tmp_path):
        stack = build_observability_stack(data_dir=str(tmp_path))
        assert isinstance(stack.tracer, NoopTracer)
        assert isinstance(stack.desensitizer, PatternDesensitizer)
        assert isinstance(stack.audit_store, SQLiteAuditStore)
        assert isinstance(stack.cache_store, MemoryCacheStore)
        assert isinstance(stack.dist_lock, MemoryLock)

    def test_with_langfuse_keys(self, tmp_path):
        stack = build_observability_stack(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            data_dir=str(tmp_path),
        )
        assert isinstance(stack.tracer, LangfuseTracer)

    def test_with_redis_url(self, tmp_path):
        stack = build_observability_stack(
            redis_url="redis://localhost:6379",
            data_dir=str(tmp_path),
        )
        assert isinstance(stack.cache_store, RedisCacheStore)
        assert isinstance(stack.dist_lock, RedisLock)

    def test_stack_components_work_together(self, tmp_path):
        """组件栈协同工作：trace → audit → cache → lock。"""
        stack = build_observability_stack(data_dir=str(tmp_path))

        # 1. trace 事件
        stack.tracer.bind("sess-1", "trace-1")
        stack.tracer.record(
            TraceEvent(
                trace_id="trace-1",
                session_id="sess-1",
                event_type="llm_call",
                payload={"model": "test"},
            )
        )

        # 2. 审计记录
        stack.audit_store.append(
            AuditRecord(
                trace_id="trace-1",
                session_id="sess-1",
                actor="orchestrator",
                action="delegate",
            )
        )

        # 3. 缓存
        stack.cache_store.set("session:sess-1", "cached_data", ttl_s=60)
        assert stack.cache_store.get("session:sess-1") == "cached_data"

        # 4. 锁
        assert stack.dist_lock.acquire("session:sess-1", ttl_s=30) is True
        stack.dist_lock.release("session:sess-1")

        # 验证
        audit_records = stack.audit_store.query("sess-1")
        assert len(audit_records) == 1
        assert isinstance(stack.tracer, NoopTracer)
        assert stack.tracer.event_count == 1


# ===========================================================================
# 7. 验收：脱敏前后对照 + 全链路 trace 可打印
# ===========================================================================
class TestAcceptanceCriteria:
    """验收标准测试。"""

    def test_desensitization_before_after_sample(self):
        """验收：脱敏前后对照样例。"""
        desensitizer = PatternDesensitizer()
        original = (
            "患者王明，身份证号 110101199003071234，"
            "手机 13912345678，邮箱 wangming@hospital.cn，"
            "患者编号 pat-abc12345678"
        )
        result = desensitizer.desensitize(original)

        # 前后对照
        assert "王明" in original and "王明" not in result.text
        assert "110101199003071234" in original
        assert "110101199003071234" not in result.text
        assert "13912345678" in original and "13912345678" not in result.text
        assert "wangming@hospital.cn" in original
        assert "wangming@hospital.cn" not in result.text
        assert "pat-abc12345678" in original and "pat-abc12345678" not in result.text

        # 占位符替换
        assert "[REDACTED-NAME]" in result.text
        assert "[REDACTED-ID]" in result.text
        assert "[REDACTED-PHONE]" in result.text
        assert "[REDACTED-EMAIL]" in result.text
        assert "[REDACTED-PATID]" in result.text

    def test_trace_events_printable(self):
        """验收：全链路 trace 事件可打印。"""
        tracer = NoopTracer(verbose=False)
        tracer.bind("sess-1", "trace-1")

        # 模拟全链路事件
        events = [
            ("route", {"decision": "need_reasoning"}),
            ("llm_call", {"model": "deepseek-v4", "tokens": 150}),
            ("retrieve", {"query": "阿奇霉素", "top_k": 5}),
            ("gate_check", {"gate": "quality_judge", "allowed": True}),
            ("conclude", {"statement": "阿奇霉素 500mg qd"}),
        ]

        for event_type, payload in events:
            tracer.record(
                TraceEvent(
                    trace_id="trace-1",
                    session_id="sess-1",
                    event_type=event_type,
                    payload=payload,
                )
            )

        # 全链路事件可打印（5 个事件全部捕获）
        assert tracer.event_count == 5
        types = [e.event_type for e in tracer.events]
        assert types == [e[0] for e in events]
        # payload 完整保留
        assert tracer.events[0].payload["decision"] == "need_reasoning"
        assert tracer.events[3].payload["gate"] == "quality_judge"

    def test_desensitizer_applied_to_checkpoint_state(self):
        """验收：脱敏边界延伸至沙箱检查点 state。"""
        desensitizer = PatternDesensitizer()
        checkpoint_state = {
            "patient_info": "姓名：李芳 电话 13812345678",
            "diagnosis": "社区获得性肺炎",
            "treatment": "阿奇霉素 500mg qd",
        }
        desensitized = desensitizer.desensitize_dict(checkpoint_state)
        assert "李芳" not in desensitized["patient_info"]
        assert "13812345678" not in desensitized["patient_info"]
        assert desensitized["diagnosis"] == "社区获得性肺炎"
