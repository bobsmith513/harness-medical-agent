# 设计决策：关键实现参数的理由与数据口径

> 本文档记录 harness-medical-agent 关键实现参数的选择理由，
> 以及所有"数字声明"的口径来源。原则：**每个数字要么有代码出处，
> 要么有文献依据，要么明确标注为演示假设**。

## 1. 检索层参数

| 参数 | 默认值 | 出处 | 理由 |
|------|--------|------|------|
| `dense_top_k` | 8 | `config/settings.py` | 双路各取 8，融合后截 5，给 RRF 留融合空间 |
| `sparse_top_k` | 8 | `config/settings.py` | 同上，双路对称 |
| `rrf_k` | 60 | `retrieval/fusion.py` | 文献常用常数：k 越大名次差异边际影响越平缓 |
| `rerank_top_k` | 5 | `config/settings.py` | 推理专家输入的证据包上限，控制 Token 预算 |
| `embedding_model` | BAAI/bge-large-zh | `config/settings.py` | 中文语义检索基线；零依赖时降级哈希嵌入 |
| `reranker_model` | BAAI/bge-reranker-v2-m3 | `retrieval/fusion.py` | 交叉编码器精排基线；零依赖时降级 identity |

**RRF 融合数学**（`retrieval/fusion.py::rrf_fuse`）：

```
score(d) = Σ_{路径 p} 1 / (k + rank_p(d))    rank 从 1 计
```

未进某路的候选该路贡献为 0（等价 rank→∞），双路命中的候选天然
高于单路头部——这是"混合检索"的数学意义，非经验调参。

**口径声明**：零依赖默认（哈希嵌入 + identity 精排）**无语义能力**，
README 架构图与分层表均诚实标注。安装 `--extra bge` 后才是
语义检索形态。

## 2. 会话与记忆参数

| 参数 | 默认值 | 出处 | 理由 |
|------|--------|------|------|
| `recent_turns`（keep） | 3 | `models/session.py::add_turn` | 见下文"50% Token 降幅"口径 |
| `checkpoint_every_turns` | 5 | `config/settings.py` | 沙箱检查点频率：够密可恢复，够疏不拖慢 |

**"长会话 Token 降约 50%"的口径**：这是**结构设计估算**而非
实测数字。计算方式——`SessionContext` 只保留最近 3 轮 +
文件指针，溢出旧轮（证据、推理链、摘要）持久化至 VFS 不进
上下文；对 20 轮以上长会话，上下文内容量近似从"全部轮次"降为
"3 轮 + 指针开销"，故约降一半。**它不是基准测试结果**，
未在真实 LLM 计费场景实测——README 相应表述已按此口径改写。

## 3. 质量门禁参数

| 参数 | 默认值 | 出处 | 理由 |
|------|--------|------|------|
| `threshold`（faithfulness） | 0.7 | `gates/quality_judge.py` | 行业 judge 打分的常用下限；可经构造参数调 |
| 低于阈值行为 | 拦截转人工 | 同上 | fail-closed：judge 打分低于 0.7 不放行 |

**fail-closed 语义清单**（`gates/quality_judge.py`）：

- judge 输出不可解析 → faithfulness 视为 0.0 → 拦截；
- faithfulness 字段缺失 / 非数值 / 布尔 → 拦截；
- `has_hallucination` / `causal_inversion` 任一为真 → 拦截；
- 字符串 `"false"` 显式按假处理（规避 `bool("false")` 为真的陷阱），
  但字段缺失按假处理的前提是 faithfulness 已单独校验。

## 4. 记忆审核规则

**自动通过的唯一组合**（`vfs/memory_review.py::auto_review`）：

```
provenance == "doctor_verified" AND confidence == "high"
```

`model_inference` 来源的记忆**永不自动通过**——必须人工审核。
理由：模型推断直接固化为可召回事实 = 幻觉记忆进入检索层，
这是本系统定义里最危险的一类数据污染。

**未审核记忆的可见性**：`Memory.can_be_recalled()` 仅在
`status == "approved"` 时为真——`pending_review` /
`session_pointer` / `rejected` 一律不可召回（检索层无从绕过）。

## 5. 安全闸门口径

- **过敏拦截**：输入闸门（词典硬匹配）先于一切 LLM 调用。
  词典归一化（品牌名→通用名，如"再林"→"amoxicillin"）在
  匹配前完成。
- **折叠空间的两类漏检**（`safety/normalization.py::fold_text`）：
  折叠必须同时剥离**零宽字符**（U+200B/C/D、U+FEFF、U+00AD 等）
  与**中缀分隔符**（`- / · * . _`）。`str.split()` 只去 Unicode 空白，
  这两类字符会留在折叠结果里使 `str.find` 返回 -1，
  **三道闸门同时漏检**——这是单条规则失效放大到全链路的典型。
  剥离规则对词典侧（建别名索引）与文本侧（扫描）同时生效，
  两侧共用 `fold_text`，不存在漏检缝。
- **ATC 两级成组**（`safety/atc.py`）：先取词条显式 `cross_group`，
  缺失时按 ATC 前缀推断（J01CA/CE/CR/DB/DC/DD/DE/DH、
  M01A/N02BA、J01E）。兜底的目的是让"新药漏打标签"退化为
  "仍被阻断"而非"静默放行"。**例外前缀刻意不入表**：
  `J01DF`（氨曲南，单环 β-内酰胺，与青霉素交叉反应极低）、
  `N02BE`（对乙酰氨基酚，非 NSAID）。新增例外时词典与前缀表
  必须同步修改。
- **对抗样本集**：37 条，阳性 28 + 阴性对照 9
  （`tests/fixtures/adversarial_samples.py`）。
  **范围声明**：这是词典与规则闸门的回归测试集（品牌别名、
  剂型绕过、混合请求、编码变体、零宽/中缀插入），**不是**
  LLM 越狱鲁棒性基准。后者需要真实 LLM 端点 + 独立评测流程，
  本仓库不覆盖。新增样本须验证它在修复前确实漏检（变异检查），
  否则会退化成永真测试。
- **Milvus 过滤注入防护**：`retrieval/vector_store.py` 的
  `_filter_literal` 白名单转义，输入含引号/反斜杠/特殊字符
  直接拒绝（ValueError）而非转义拼接。

## 6. 测试数字口径

**唯一口径**：以零依赖环境（不装任何 extras）运行
`uv run pytest` 的结果为准。

- 总用例数：**631**（2026-08-29 本地实测收集数）。以 `uv run pytest`
  的收集统计为准（README、requirements-dev.txt、CI 说明均引用同一数字）；
- `--all-extras` 场景（装齐可选依赖）为 **627 通过 + 4 跳过**，总收集数
  不变——4 项验证的是"依赖缺失时的降级路径"，装齐后无需重复验证；
  两个数字符合 627 + 4 = 631，文档引用时不要写成"零依赖也是 627"；
- 覆盖率：零依赖实测 **84.88%**（`pytest --cov`），CI 阈值 80%
  （阈值定义见 `pyproject.toml`，留约 5 个点防环境抖动）。

> 历史教训：这一节此前写的是"以收集统计为准"却没写数字，README 里的
> 581 是凭印象填的，与真实收集数差 10。数字一律实测回填，不靠记忆。

**为什么不标"passing"徽章**：静态徽章无法随真实运行刷新，
数字过期即成误导。改为"可本地复跑"的可复核口径。

## 7. 降级表口径（从 Mock 到生产）

README"从 Mock 到生产：字段一览"一节的降级表与
`observability/redis_compat.py::try_redis_client` 等装配函数
一一对应。统一语义：

- 配置为空 / 依赖未安装 → 内存实现，**接口不变**；
- 依赖未安装**不抛异常**（返回 None → 装配层降级），
  显式配置了 URI 但依赖缺失 → 启动时明确报错。

区分这两种失败：前者是"没要求生产形态"，后者是"要求了但
环境不满足"——报错语义必须不同。

## 8. 提交历史口径

**提交时间戳不构成开发过程的证据**。评估工作量的正确路径：

1. [development-plan.md](development-plan.md)（每个里程碑的设计取舍）；
2. 本文档（关键实现参数的理由与数据口径）；
3. `tests/`（每项安全与降级语义的断言）。

开发方式：AI 辅助开发 + 人工设计决策与验证——架构分层、
fail-closed 语义、安全闸门取舍由作者制定，代码实现与测试编写
有 AI 工具深度参与，作者负责逐模块审读、运行验证并承担
全部正确性责任（README"开发方式说明"一节同口径）。
