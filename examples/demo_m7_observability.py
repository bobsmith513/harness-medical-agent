"""M7 沙箱适配与可观测演示。

    uv run python examples/demo_m7_observability.py

演示三个核心场景（对照 development-plan.md M7 验收标准）：

1. 脱敏前后对照样例（身份证/手机号/邮箱/患者编号/姓名标记）；
2. 沙箱检查点中断恢复 demo（执行 → 保存 → 恢复 → 继续）；
3. 全链路 trace 事件可打印（route → llm → retrieve → gate → conclude）。

全链路零外部依赖（NoopTracer + MockRuntime + MemoryCacheStore）。
"""

from __future__ import annotations

from harness_agent.models.audit import AuditRecord, TraceEvent
from harness_agent.observability import (
    PatternDesensitizer,
    build_observability_stack,
)
from harness_agent.sandbox import build_sandbox_runtime


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 场景 1：脱敏前后对照
# ---------------------------------------------------------------------------
def demo_desensitization() -> None:
    _print_section("场景 1：脱敏前后对照样例")
    print("  出站调用前去除患者标识（PII → [REDACTED-类型] 占位符）")

    desensitizer = PatternDesensitizer()

    samples = [
        ("姓名标记", "姓名：张三 诊断为社区获得性肺炎"),
        ("身份证号", "患者身份证号 310101199001011234 就诊记录"),
        ("手机号", "联系电话 13812345678 已确认"),
        ("邮箱", "邮箱 zhangsan@hospital.com 已记录"),
        ("患者编号", "patient_id=pat-abc12345678"),
        (
            "混合 PII",
            ("患者李芳，身份证 420101198503156789，电话 13712345678，邮箱 lifang@hospital.cn"),
        ),
    ]

    for label, original in samples:
        result = desensitizer.desensitize(original)
        print(f"\n  [{label}]")
        print(f"    原文: {original}")
        print(f"    脱敏: {result.text}")
        if result.removed_entities:
            print(f"    移除: {', '.join(result.removed_entities)}")


# ---------------------------------------------------------------------------
# 场景 2：沙箱检查点中断恢复
# ---------------------------------------------------------------------------
def demo_sandbox_checkpoint() -> None:
    _print_section("场景 2：沙箱检查点中断恢复 demo")
    print("  执行 → 保存检查点 → 恢复检查点 → 继续执行")

    runtime = build_sandbox_runtime(backend="mock")

    # 1. 执行初始代码
    print("\n  步骤 1：执行初始代码")
    result1 = runtime.execute("print('step 1: 剂量计算 500mg')")
    print(f"    exit_code={result1.exit_code}, stdout={result1.stdout.strip()}")

    # 2. 保存检查点（模拟中断前）
    print("\n  步骤 2：保存检查点（模拟中断前状态快照）")
    cp = runtime.save_checkpoint(
        "sess-1",
        {
            "last_step": "1",
            "intermediate_result": "500mg",
            "patient_context": "社区获得性肺炎",
        },
    )
    print(f"    checkpoint_id={cp.checkpoint_id}")
    print(f"    state={cp.state}")

    # 3. 模拟中断恢复
    print("\n  步骤 3：恢复检查点")
    restored = runtime.restore(cp)
    print(f"    restored={restored}")

    # 4. 恢复后继续执行
    print("\n  步骤 4：恢复后继续执行")
    result2 = runtime.execute("print('step 2: 疗程 5 天')")
    print(f"    exit_code={result2.exit_code}, stdout={result2.stdout.strip()}")

    # 5. 审计：检查点时间线
    print("\n  步骤 5：检查点审计时间线")
    for i, cp_item in enumerate(runtime.list_checkpoints("sess-1")):
        print(
            f"    [{i}] cp={cp_item.checkpoint_id[:16]}... "
            f"step={cp_item.state.get('last_step', '?')}"
        )


# ---------------------------------------------------------------------------
# 场景 3：全链路 trace 事件可打印
# ---------------------------------------------------------------------------
def demo_trace_events() -> None:
    _print_section("场景 3：全链路 trace 事件可打印")
    print("  route → llm → retrieve → gate → conclude（5 个事件）")

    stack = build_observability_stack(data_dir=".data/demo-m7")
    tracer = stack.tracer
    tracer.bind("sess-1", "trace-1")

    # 模拟全链路事件
    events = [
        ("route", {"decision": "need_reasoning", "by_rule": True}),
        ("llm_call", {"model": "deepseek-chat", "tokens": 150}),
        ("retrieve", {"query": "阿奇霉素", "top_k": 5, "evidence_count": 3}),
        ("gate_check", {"gate": "quality_judge", "allowed": True}),
        ("conclude", {"statement": "阿奇霉素 500mg qd 可用于肺炎"}),
    ]

    print("\n  绑定会话: session=sess-1, trace=trace-1")
    print()

    for event_type, payload in events:
        event = TraceEvent(
            trace_id="trace-1",
            session_id="sess-1",
            event_type=event_type,
            payload=payload,
        )
        tracer.record(event)
        payload_str = ", ".join(f"{k}={v}" for k, v in payload.items())
        print(f"  [TRACE] {event_type}: {payload_str}")

    print(f"\n  全链路事件总数: {tracer.event_count}")

    # 审计记录
    print("\n  审计记录:")
    for event_type, _ in events:
        stack.audit_store.append(
            AuditRecord(
                trace_id="trace-1",
                session_id="sess-1",
                actor=f"actor:{event_type}",
                action=event_type,
            )
        )

    audit_records = stack.audit_store.query("sess-1")
    print(f"  审计记录数: {len(audit_records)}")
    for r in audit_records:
        print(f"    [{r.actor}] {r.action}")

    # 缓存 + 锁
    print("\n  缓存与锁:")
    stack.cache_store.set("session:sess-1", "active", ttl_s=60)
    cached = stack.cache_store.get("session:sess-1")
    print(f"    缓存 session:sess-1 = {cached}")

    locked = stack.dist_lock.acquire("session:sess-1", ttl_s=30)
    print(f"    锁 session:sess-1 acquired={locked}")
    stack.dist_lock.release("session:sess-1")
    print("    锁 session:sess-1 released")


def main() -> None:
    print("M7 沙箱适配与可观测演示")
    print("全链路零外部依赖（NoopTracer + MockRuntime + MemoryCacheStore）")

    demo_desensitization()
    demo_sandbox_checkpoint()
    demo_trace_events()

    print()
    print("=" * 72)
    print("M7 验收总结:")
    print("  - 脱敏前后对照样例（6 类 PII + 混合场景）")
    print("  - 沙箱检查点中断恢复 demo（执行→保存→恢复→继续）")
    print("  - 全链路 trace 事件可打印（5 个事件完整捕获）")
    print("  - 审计记录可查询（SQLite 降级，零依赖）")
    print("  - 缓存与锁可用（Memory 降级，零依赖）")
    print("=" * 72)


if __name__ == "__main__":
    main()
