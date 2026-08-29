"""集中式配置系统（M0）。

设计原则（对应 docs/development-plan.md 第六节 Mock 边界）：

1. **外部端点一律留空**：模型端点（本地 vLLM 端口或在线 API）、Milvus、
   Langfuse、OpenSandbox、PostgreSQL、Redis 的地址与密钥默认全部为空字符串，
   由使用者按需填写（参考 ``.env.example``）。
2. **留空即降级**：真实地址未填写时，系统自动使用 mock / 本地实现，
   保证零依赖跑通全部 demo；mock 与真实实现共用同一接口（M1 契约），
   配置只负责选择实现，业务逻辑永不分叉。
3. **语义默认值冻结架构决策**：如上下文只保留最近 3 轮、记忆审核默认开启、
   patient_id 作为分区键等，这些默认值就是系统设计语义，测试会锁定它们。

环境变量约定：前缀 ``HARNESS_`` + 嵌套分隔符 ``__``（双下划线），
例如 ``HARNESS_LLM__PROVIDER=openai``、``HARNESS_RETRIEVAL__STORE=milvus``。
同时支持项目根目录 ``.env`` 文件（参考 ``.env.example``）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from harness_agent.config.paths import anchor_path, env_file_candidates

__all__ = [
    "AppSettings",
    "LLMSettings",
    "ObservabilitySettings",
    "RetrievalSettings",
    "SafetySettings",
    "SandboxSettings",
    "Settings",
    "VfsSettings",
    "get_settings",
    "reset_settings",
]


class AppSettings(BaseModel):
    """应用级全局设置。"""

    debug: bool = False
    # 本地运行时数据根目录：mock 向量存储 / VFS / SQLite 审计降级均落在此处（gitignore）
    data_dir: str = ".data"
    # 启动时把合成样例（4 位患者 + 8 条知识）灌入检索库：
    # 数据库地址来自 .env，内容来自 harness_agent.seed_data（真实数据走 index 接口）
    seed_sample_data: bool = True


class LLMSettings(BaseModel):
    """外部模型连接配置。

    **在线调用模式（推荐起步）**：``provider`` 填预设服务商名
    （deepseek / qwen / zhipu / moonshot / openai / siliconflow），
    再填一个共享 ``api_key``，路由 / 推理 / judge 三类模型即全部
    在线调用——端点与推荐模型名由 ``llm/providers.py`` 预设表给出。

    **微调模型旁路**：推理专家可单独用 ``reasoning_base_url`` 指向
    本地 vLLM 部署的 SFT+DPO 微调模型，其余角色继续走在线 API
    （逐角色字段覆盖共享配置）。

    ``provider="mock"`` 时全部端点不生效（MockLLM 脚本化应答，
    零外部依赖）。``provider="openai"`` 走通用 OpenAI 兼容协议，
    端点需自行填写。

    四类模型角色（均可独立覆盖，留空继承共享 / 预设值）：
    - orchestrator：主 Agent 编排模型（纯编排无应答权）
    - reasoning：推理专家（SFT+DPO 对齐基座，专职临床结论）
    - judge：质量门禁 LLM-as-judge（忠实度校验）
    - router：路由器 LLM 兜底（规则前置，仅兜不可判场景）
    """

    provider: Literal[
        "mock",
        "openai",
        "deepseek",
        "qwen",
        "zhipu",
        "moonshot",
        "siliconflow",
    ] = "mock"

    # ---- 共享 API Key（全部角色默认复用；逐角色可用 <ROLE>_API_KEY 覆盖）----
    api_key: str = ""

    # ---- 主 Agent 编排模型 ----
    # 覆盖预设端点：本地 vLLM（如 http://localhost:8000/v1）或在线 API URL
    orchestrator_base_url: str = ""
    orchestrator_api_key: str = ""
    # 留空 = 使用 provider 预设推荐模型名
    orchestrator_model: str = ""
    orchestrator_timeout_s: float = 30.0

    # ---- 推理专家（微调模型旁路口）----
    # 留空 = 与其他角色共用 provider 预设；填本地 vLLM 端口 = 微调模型独立部署
    reasoning_base_url: str = ""
    reasoning_api_key: str = ""
    reasoning_model: str = ""
    reasoning_timeout_s: float = 120.0

    # ---- 质量门禁 judge 模型 ----
    judge_base_url: str = ""
    judge_api_key: str = ""
    judge_model: str = ""
    judge_timeout_s: float = 60.0

    # ---- 路由器 LLM 兜底 ----
    router_base_url: str = ""
    router_api_key: str = ""
    router_model: str = ""
    router_timeout_s: float = 15.0


class RetrievalSettings(BaseModel):
    """检索与记忆供给层设置。"""

    # local：本地 numpy 向量 + BM25（零 Docker）；milvus：docker-compose 启动
    store: Literal["local", "milvus"] = "local"
    # 待填：Milvus 连接串（如 http://localhost:19530），store=local 时不生效
    milvus_uri: str = ""
    # 患者记忆库分区隔离字段（固定语义，一般不改）
    partition_key: str = "patient_id"
    # BGE-large-zh 嵌入模型（M3 引入，sentence-transformers CPU/GPU 自动检测）
    # 嵌入实现选择：hashing（零依赖默认）/ bge（extras=bge 安装 sentence-transformers）
    embedding_provider: Literal["hashing", "bge"] = "hashing"
    embedding_model: str = "BAAI/bge-large-zh"
    embedding_dim: int = 1024
    # 双路召回
    dense_top_k: int = 8
    sparse_top_k: int = 8
    # RRF 融合常数（k 越大，排名差异对融合分影响越平缓）
    rrf_k: int = 60
    # bge-reranker-v2-m3 精排：本地重模型，默认关闭走 identity（M3 引入）
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 5


class VfsSettings(BaseModel):
    """虚拟文件系统与上下文压缩设置。"""

    # 虚拟目录根：<root_dir>/<session_id>/{evidence,reasoning,summaries,memories}
    root_dir: str = ".data/vfs"
    # 上下文只保留最近 N 轮 + 文件指针（长会话 Token 压缩核心参数）
    recent_turns: int = 3
    # 摘要写入 /memories/ 并同步向量索引前，强制走审核队列（合规默认开）
    memory_review_required: bool = True


class SandboxSettings(BaseModel):
    """代码执行沙箱设置（检验计算、剂量换算等）。"""

    # mock：本地子进程隔离；opensandbox：Docker/K8s 双运行时 + 原生 MCP
    backend: Literal["mock", "opensandbox"] = "mock"
    # 待填：OpenSandbox 服务地址，backend=mock 时不生效
    opensandbox_url: str = ""
    # 检查点保存间隔（会话轮数）：沙箱实例回收/重调度后任务可恢复
    checkpoint_every_turns: int = 5


class SafetySettings(BaseModel):
    """硬规则安全层设置（M2）。"""

    # 生产药名词典 JSON 路径（归一化名/别名/ATC/交叉反应组）。
    # 留空 = 使用内置合成种子词典；示例格式见 data/drug_dictionary.json
    dictionary_path: str = ""


class OrchestratorSettings(BaseModel):
    """主 Agent 编排层设置（M4）。

    - ``experts_config_path``：专家声明式 YAML（留空 = 仓库根
      configs/experts.yaml，开发态默认；安装态部署需指定绝对路径）；
    - 路由 LLM 端点复用 ``llm.router_base_url``（四组端点之一）。
    """

    experts_config_path: str = ""


class ObservabilitySettings(BaseModel):
    """可观测与审计设置。全部留空时自动降级：Noop tracer / SQLite / 内存实现。"""

    # Langfuse（留空 -> NoopTracer，仅打印事件）
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""
    # PostgreSQL 审计库 DSN（留空 -> 降级 SQLite，落 app.data_dir）
    audit_dsn: str = ""
    # Redis 缓存与分布式锁（留空 -> 降级进程内存实现）
    redis_url: str = ""


class Settings(BaseSettings):
    """harness-medical-agent 全局配置聚合根。

    用法：``from harness_agent.config import get_settings``（进程内单例）。
    """

    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_nested_delimiter="__",
        # 直接构造 Settings() 时按 CWD 解析（延迟到实例化时，行为同旧版）；
        # 生产路径 get_settings() 会显式传入仓库根锚定的候选链（见下）
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    vfs: VfsSettings = Field(default_factory=VfsSettings)
    sandbox: SandboxSettings = Field(default_factory=SandboxSettings)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    def model_post_init(self, __context: object) -> None:  # noqa: ARG002
        """运行时目录锚定：相对路径落仓库根而非 CWD（防止父目录被 .data 污染）。"""
        if not Path(self.app.data_dir).is_absolute():
            self.app.data_dir = str(anchor_path(self.app.data_dir))
        if not Path(self.vfs.root_dir).is_absolute():
            self.vfs.root_dir = str(anchor_path(self.vfs.root_dir))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局配置单例（进程内缓存）。

    ``_env_file`` 显式传入候选链（调用时动态解析）：
    仓库根 ``.env`` 打底 + CWD ``.env`` 覆盖——从任意目录运行
    （如直接 ``python src/harness_agent/cli.py``、父目录调用）都能
    读到项目配置，修复"父目录运行 → 意外 mock 模式"。
    """
    return Settings(_env_file=env_file_candidates())


def reset_settings() -> None:
    """清除配置缓存。测试中修改环境变量后调用，使下次 get_settings 重新加载。"""
    get_settings.cache_clear()
