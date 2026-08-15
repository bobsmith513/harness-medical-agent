"""pytest 安全网：未安装包时也能直接从 src 导入。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
# src：harness_agent 包本体；根目录：examples/ 命名空间包（demo 可导入性测试）
for _path in (str(_SRC), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
