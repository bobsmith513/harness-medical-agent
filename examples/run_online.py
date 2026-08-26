"""在线问诊入口（薄封装）——真正的 main 在 ``harness_agent.main``。

    uv run harness-online                     # 等价命令（pyproject [project.scripts]）
    uv run python examples/run_online.py      # 本文件

所有模型（含微调推理模型）与数据库地址全部来自 `.env`。
"""

from __future__ import annotations

from harness_agent.main import main

if __name__ == "__main__":
    main()
