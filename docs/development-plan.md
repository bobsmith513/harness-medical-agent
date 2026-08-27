# 开发路线与整体任务逻辑

> 本文档是仓库构建蓝本：系统模块逻辑（要建什么）与里程碑执行逻辑（怎么建）。
> 状态标注：✅ 已完成 · 🔨 进行中 · 空白 待开工。

## 一、系统定位（一句话）

以 Harness Engineering 范式构建覆盖全病程的医疗多智能体系统：工程脚手架做厚而非换更强模型；主 Agent 纯编排无应答权，临床结论均出自带核查的推理管线；误路由与门禁未达标一律 fail-closed 转澄清或人工。

## 二、系统架构：六层模块逻辑

| 层 | 模块 | 核心职责 | 关键设计 |
|----|------|---------|---------|
| ① 入口层 | 会话管理 | 用户会话、多轮上下文 | 脱敏中间件前置 |
| ② 编排层 | 主 Agent | 规划＋虚拟文件系统＋子代理委派 | 纯编排无应答权（结构层面禁止产出临床结论） |
| ② 编排层 | 路由器 | 二值判断"是否需要临床推理" | 规则前置＋LLM 兜底；误判二次路由，仍失败 fail-closed |
| ③ 专家层 | 推理专家 | 生成"证据引用→逐步推断→结论"推理链并自检 | 运行于 SFT+DPO 对齐基座 |
| ③ 专家层 | 记忆专家 | 供给编排：硬规则精确匹配＋软记忆召回 | 过敏史不走向量，走药名归一化＋ATC 交叉反应 |
| ④ 供给层 | MCP 检索服务 | 以 MCP 统一供给证据与记忆 | 内嵌三道安全闸门；patient_id 分区隔离 |
| ⑤ 质量层 | 质量门禁 | LLM-as-judge 忠实度＋药物安全 API 全量把关 | 未达标触发 interrupt 转人工 |
| ⑥ 基础设施 | 沙箱 / 可观测 | 代码执行隔离、全链路 trace、审计 | OpenSandbox；Langfuse＋PostgreSQL＋Redis；脱敏延伸至沙箱检查点 |

## 三、一次会话的完整数据流

```
用户输入
  → 脱敏中间件（去除患者标识）
  → 路由器：是否需要临床推理？
      ├─ 否 → 记忆专家装配上下文 → 稳定事实免问询 / 易变事实确认式追问 → 响应
      └─ 是 → 委派推理专家
              → 经 MCP 调供给层取证据
                  （输入闸门：过敏硬规则拦截 → BGE+BM25 双路召回 → RRF 融合 → 精排 → 装配闸门复核）
              → 生成推理链（证据引用→逐步推断→结论）+ 自检
              → 质量门禁
                  （LLM-judge 忠实度校验 + 药物安全 API 校验）
                  ├─ 通过 → 临床结论输出
                  └─ 未达标 → interrupt → LLM 生成参考提问 / 转人工（约 10%）
长会话伴随：证据/推理链/摘要 → 虚拟目录持久化 → 上下文只留最近 3 轮 + 文件指针
            → 摘要标注来源置信度 → 抽样审核 → 通过才写入 /memories/ 转正为可召回记忆
```

## 四、执行里程碑

> 依赖关系：M0 → M1 →（M2 ∥ M3 ∥ M7 可并行）→ M4 → M5 → M6 → M8 → M9

### M0 工程基座 ✅
- 任务：目录骨架、`pyproject.toml`（uv 管理）、`.gitignore`、`LICENSE`、`.env.example`、配置系统（pydantic-settings）、CI（ruff + pytest）
- 产出：可安装的包，pytest 全绿
- 验收：目录结构完整、CI 工作流就位、git 分批提交

### M1 领域模型与接口契约 ✅
- 任务：核心类型定义（`Evidence` 含来源/置信度/原图引用、`Memory` 绑定 patient_id、`ReasoningChain`、`ClinicalConclusion`、`AuditRecord`、`SessionContext`）；模块接口（`RetrievalService`、`ExpertTool`、`Gate`、`SandboxRuntime`、`Tracer` 各为 Protocol）
- 产出：`src/harness_agent/models/` + `src/harness_agent/contracts/`
- 验收：类型与接口被后续所有模块引用不改动；类型单测通过
- 实现落点：
  - `src/harness_agent/models/` 六文件：common / audit / evidence / memory / reasoning / session
  - `src/harness_agent/contracts/` 六文件：llm / retrieval / experts / gates / sandbox / observability
  - **主 Agent 无应答权的类型级落地**：`ClinicalConclusion` 必须携带自检通过的
    `ReasoningChain`，结论引用证据必须 ⊆ 推理链引用集合
  - **fail-closed 路由**：`RouteDecision` 枚举无"直接回答"出口
  - **记忆审核闭环**：`Memory.can_be_recalled()` 仅 approved 为真
  - **分区隔离进接口签名**：`RetrievalService.retrieve` 永远携带 patient_id
  - 全部 Protocol `@runtime_checkable`，测试以最小 mock 实现验证契约可满足

### M2 供给层：硬规则与三道闸门 ✅
- 任务：药名归一化（别名/商品名/中英文词典）；ATC 交叉反应映射（内置种子数据）；三道闸门（输入拦截 → 装配复核 → 输出校验 API）；对抗样本集（合成）与漏检率监控
- 产出：`src/harness_agent/safety/` + 对抗样本 fixture
- 验收：对抗样本集漏检率必须为 0（硬规则全拦截）；单测覆盖归一化边界情况
- 实现落点：
  - `src/harness_agent/safety/`：dictionary（最小种子词典 + JSON 可插拔落位）/ normalization
    （折叠空间：全角转半角+小写+去空白；最长优先提及扫描）/ atc（交叉反应组）/
    allergy_store（AllergyStore 供给接口，HIS/EMR 对接留空）/ resolver / 三道闸门
  - 三道闸门：输入拦截（查询命中过敏/交叉药物即拒绝，fail-closed）、
    装配复核（`apply` 过滤含过敏药物实体证据；全过滤则拒绝交付空证据包）、
    输出校验（结论陈述 + 推理链全文扫描）
  - 对抗样本 7 条（每类 1-2 条代表：直命中/历史别名/交叉反应/全角混淆/
    NSAID 交叉/阴性对照），输入与输出闸门漏检率 = 0 锁定于
    `tests/test_safety_adversarial.py`
  - 可插拔数据位：`data/drug_dictionary.json` +
    `HARNESS_SAFETY__DICTIONARY_PATH`（留空用内置种子）
  - M1 契约唯一细化：`OutputGate.check` 携带 `SessionContext`（裁决需患者过敏史）

### M3 供给层：软记忆与混合检索 ✅
- 任务：BGE-large-zh 封装（sentence-transformers，CPU 可跑）；BM25 双路；RRF 融合；reranker（可关）；patient_id 分区隔离；MCP 服务封装（FastMCP）
- 产出：`src/harness_agent/retrieval/` + `src/harness_agent/mcp/` + `docker-compose.milvus.yaml`（Milvus 可选启动）
- 验收：示例记忆库上"双路召回 → 融合 → 精排"跑通；跨患者隔离单测通过
- 实现落点：
  - `src/harness_agent/retrieval/` 七文件：tokenizer（中文二元组+ASCII 词元，双路共用）
    / embeddings（哈希嵌入默认 + BGE 可插拔）/ vector_store（内存分区
    隔离默认 + Milvus HNSW 骨架）/ bm25（本地稀疏路，IDF 统计量同分区
    隔离）/ fusion（RRF 数学 + identity/bge-reranker 两级精排）/
    service（HybridRetrievalService 门面）/ wiring（配置→全组件接线）
  - **分区隔离双保险**：查询分区 + 共享分区两个分组物理隔离（非过滤
    后丢弃），BM25 统计量（IDF/avgdl）也只来自可见分区
  - **门面闸门串联**：输入闸门拦截 → 空包 + gate=input 裁决；
    装配闸门过滤含过敏药物证据、全过滤拒绝交付（fail-closed 语义
    经 `is_reviewed` 外显）；同父补全走 sibling_ids 标记 structural
  - `src/harness_agent/mcp/retrieval.py`：FastMCP 封装（retrieve/index_chunks 工具），
    `uv run harness-mcp-retrieval` 启动；fastmcp/milvus/bge 均为可选
    extras，零依赖默认全链路可用
  - `harness_agent.safety.build_safety_stack`：安全层全家桶装配工厂（M4+ 共用）
  - 可插拔开关：`EMBEDDING_PROVIDER`（hashing|bge）、`STORE`（local|milvus）、
    `RERANKER_ENABLED`（identity|bge），配置选择实现，逻辑零分叉

### M4 主 Agent 编排与路由 ✅
- 任务：路由器（规则前置 + LLM 兜底，二值输出；误判 → 二次路由 → 仍失败转澄清）；主 Agent（任务清单规划、task 委派；类型层面禁止直接产出临床结论）；YAML 专家声明式配置加载器
- 产出：`src/harness_agent/orchestrator/` + `configs/experts.yaml`
- 验收：路由单测（规则可判场景 100% 命中，不可判走兜底）；委派 demo 跑通
- 实现落点：
  - `src/harness_agent/orchestrator/` 六文件：router（规则前置 + LLM 兜底 +
    二次路由门面）/ experts_config（YAML 声明式加载器，新增专家
    零改动主流程）/ planner（路由裁决 → 委派任务清单）/ agent
    （HarnessOrchestrator，langgraph StateGraph 编排）/ state
    （OrchestrationResult 等）/ wiring（配置 → 主 Agent 装配工厂）
  - `configs/experts.yaml`：三专家声明（reasoning / memory / intake），
    重名、未知 kind、占位符未绑定 inputs 均装配期 fail-closed
  - **langgraph 图编排**：START→route→plan→（retrieve→reason |
    memory | escalate_node）→finalize→END，条件边按裁决分流；
    编排图结构即"无应答权"的结构级落地——不存在生成结论的节点
  - **二值路由链**：规则前置（关键词/正则，零 LLM 开销）→ LLM
    兜底（JSON 结构化解析）→ 误判二次路由（纠错提示）→ 仍失败
    escalate；LLM 不得自报 escalate（升级只由内部失败路径产生）
  - **fail-closed 全分支**：证据包 is_reviewed=False 不进推理；
    专家未绑定实现 / 检索异常 / 注册表缺 reasoning 专家均升级人工
  - `src/harness_agent/llm/mock.py`：脚本化 LLM 客户端（路由兜底默认实现），
    M5 换 OpenAI 兼容客户端零改动
  - 62 项新测试 + `examples/demo_orchestrator.py` 五场景演示
    （含规则路由零 LLM、误判二次路由、装配闸门拦截转人工）

### M5 推理专家与质量门禁 ✅
- 任务：推理专家（OpenAI 兼容接口调用，默认 mock；生成三段式推理链 + 自检）；LLM-judge 门禁（校验引用与证据一致性，专查依据缺失与因果倒置，阈值可配）；药物安全 API 门禁（复用 M2）；LangGraph interrupt 机制（参考提问 / 转人工）
- 产出：`src/harness_agent/experts/reasoning_expert.py` + `src/harness_agent/gates/`
- 验收：端到端一次"取证据 → 推理 → 门禁 → 输出"跑通（mock 模型）；门禁拦截 badcase 样例可演示

### M6 虚拟文件系统与记忆审核 ✅
- 任务：VFS 目录抽象（`/evidence/ /reasoning/ /summaries/ /memories/`）；上下文压缩（只保留最近 3 轮 + 文件指针）；记忆审核（摘要标注来源置信度 → 抽样审核队列 → 通过转正并同步索引，未审核仅作会话内指针）
- 产出：`src/harness_agent/vfs/`
- 验收：20 轮模拟会话上下文 token 降约 50%（打印前后对比）；"未审核摘要不得被召回"单测通过

### M7 沙箱适配与可观测 ✅
- 任务：`SandboxRuntime` 接口 + MockRuntime（本地进程）+ OpenSandbox 适配骨架（透明代理句柄、检查点保存/恢复）；Tracer 接口（Langfuse 缺省时 Noop）；脱敏中间件（应用于出站调用与日志）；PostgreSQL 审计写入 + Redis 缓存锁（缺省降级 SQLite/内存）
- 产出：`src/harness_agent/sandbox/` + `src/harness_agent/observability/`
- 验收：脱敏前后对照样例；沙箱检查点中断恢复 demo；全链路 trace 事件可打印

### M8 端到端演示与文档 ✅
- 任务：合成示例数据（患者档案含过敏对抗样例、知识条目、多轮会话脚本）；四个 demo（初诊推理全链路 / 复诊记忆命中免问询 / 长会话压缩 / 门禁拦截转人工）；README 完善；测试补齐
- 产出：`examples/` + 完整 `README.md` + 完整 `tests/`
- 验收：全新环境按 README 三条命令跑通全部 demo；CI 全绿
- 实现落点：
  - `tests/fixtures/synthetic_data.py`：4 位虚构患者档案（含 M2 过敏种子
    pat-001/002/003 + 新增 pat-004 糖尿病复查）、8 条知识库条目
    （CAP/DM/RA/过敏安全指南）、4 套多轮会话脚本
  - `examples/demo_first_diagnosis.py`：初诊推理全链路
    （脱敏 → 路由 → 检索 → 推理 → 门禁 → 结论 → trace，7 步）
  - `examples/demo_followup_memory.py`：复诊记忆命中免问询
    （初诊摘要 → 审核 → 转正 → 复诊召回 → 编排验证，4 Phase）
  - `examples/demo_long_conversation.py`：20 轮长会话压缩
    （逐轮压缩 → VFS 统计 → 指针回溯 → 批量审核 → 对比表，5 Phase）
  - `examples/demo_gate_interception.py`：门禁拦截转人工
    （忠实度不足 / 臆测 / 过敏药物 3 拦截 + 1 正常对照）
  - README 路线图 M3-M8 全部 ✅，新增 demo 导航与快速命令
  - `tests/test_m8_synthetic_data.py`：合成数据完整性测试

### M-online 在线调用模式 ✅
- 任务：除微调模型外全部切换为在线 API 调用；.env 填两行（PROVIDER + API_KEY）
  即可运行交互式问诊；样本数据自动装配
- 产出：`llm/providers.py` + `llm/wiring.py` + `examples/run_online.py` + `.env.example`
- 验收：预设服务商切换零代码改动；微调模型旁路独立生效；在线失败 fail-closed
- 实现落点：
  - `llm/providers.py`：6 家服务商预设表（deepseek/qwen/zhipu/moonshot/openai/
    siliconflow），端点 + 四角色推荐模型名 + 密钥申请入口
  - `llm/wiring.py`：`build_llm_client` 装配工厂（解析优先级：逐角色覆盖 >
    共享配置 > provider 预设）；mock / 在线同一契约零分叉
  - `settings.py`：`provider` 扩展预设名单 + 共享 `api_key`；模型名默认留空
    = 继承预设推荐值
  - `examples/run_online.py`：交互式问诊入口薄封装（真正实现见
    `src/harness_agent/main.py`，与 `uv run harness-online` 同一 main；
    患者选择 → 提问 → 白盒输出路由/证据/推理链/门禁/结论；
    单次模式 `--patient --query`；启动错误给修复指引而非裸栈）
  - 微调模型旁路：`REASONING_BASE_URL` 指向本地 vLLM 的 SFT+DPO 模型，
    其余角色继续在线 API（混合部署形态）
  - `tests/test_llm_online_wiring.py`：17 项（预设完整性 / 解析优先级 /
    Key 继承 / fail-fast 报错 / 描述输出）

### M9 GitHub 发布（本地就绪，待 push）
- 任务：提交历史整理；建仓与 push；仓库描述 + topic（`llm` `multi-agent` `mcp` `langgraph` `deepagents` `medical-ai`）
- 验收：仓库公开可访问、README 渲染正常、CI 绿
- 当前状态：提交历史按里程碑（M0 → M-online）分批组织，`v0.1.0` 标签已打，全部测试（496 项）与 demo 通过
- 待执行：`git remote add origin` → `git push -u origin main` → `git push --tags`

## 五、Mock 与真实依赖边界

| 外部依赖 | Demo 模式 | 真实模式 | 切换开关 |
|---------|----------|---------|---------|
| 编排 / 路由 / judge 模型 | MockLLM 脚本化应答 | 在线 API（deepseek/qwen/zhipu/moonshot/openai/siliconflow 预设） | `HARNESS_LLM__PROVIDER` + `API_KEY` |
| 对齐推理模型（微调） | MockReasoner 模板推理链 | 本地 vLLM / 在线 API | `REASONING_BASE_URL`（旁路口） |
| BGE-large-zh | sentence-transformers CPU | GPU | 自动检测 |
| Milvus | 本地 numpy 向量 + BM25 | docker-compose | `HARNESS_RETRIEVAL__STORE` |
| bge-reranker | 跳过（identity） | 真实加载 | `HARNESS_RETRIEVAL__RERANKER_ENABLED` |
| Langfuse | NoopTracer | 自托管实例 | Langfuse 密钥是否配置 |
| OpenSandbox | MockRuntime 本地进程 | Docker/K8s | `HARNESS_SANDBOX__BACKEND` |
| PostgreSQL / Redis | SQLite / 内存字典 | docker-compose | DSN / URL 是否配置 |

原则：mock 与真实实现共用同一接口（M1 契约），靠依赖注入切换，逻辑永不分叉。所有真实端点默认留空，见 `.env.example`。

## 六、Git 提交策略

| 阶段 | 提交信息风格 |
|------|------------|
| M0 | `chore: scaffold repository...` / `feat(config): settings system...` / `ci: add ruff and pytest workflow` |
| M1 | `feat: domain models and module contracts` |
| M2–M3 | `feat(safety): drug normalization, atc gates` / `feat(retrieval): hybrid recall with rrf fusion` |
| M4–M5 | `feat(orchestrator): binary router with fail-closed` / `feat(gates): llm-judge and drug safety` |
| M6–M7 | `feat(vfs): context compaction and memory review` / `feat(observability): tracing and desensitize` |
| M8–M9 | `docs: readme with demos and metrics` / `release: v0.1.0` |

## 七、风险与对策

- **依赖版本**：DeepAgents/LangGraph 迭代快，锁定版本并在 CI 验证
- **Milvus 本地重**：local 降级模式保证零 Docker 也能跑 demo
- **mock 与真实分叉**：接口 + 依赖注入，禁止 if-else 散落业务代码
- **数据合规**：示例数据全部合成，README 显著声明；脱敏中间件默认开启
- **指标口径**：README 以"项目实测结果"表格呈现，代码提供评估框架，注明复现需完整数据与算力
