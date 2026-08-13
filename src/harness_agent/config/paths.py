"""路径锚定：把项目资源路径从"当前工作目录"解耦到"仓库根"。

## 为何存在

此前 ``.env`` 与 ``configs/experts.yaml`` 都按 **CWD 相对路径** 加载。
在仓库根运行没问题，但从任意目录运行（如直接 ``python src/harness_agent/cli.py``
或从父目录调用）时会得到：

- ``FileNotFoundError: 专家配置不存在: configs\\experts.yaml``；
- ``.env`` 静默加载失败 → 配置回退默认值 → 意外落入 mock 模式。

## 解析规则（``project_root()``）

1. 从 **本文件所在位置** 向上找带 ``pyproject.toml`` 的目录（src 布局恒命中）；
2. 从 **CWD** 向上找（覆盖安装态包 + 在仓库任意子目录运行的场景）；
3. 都失败时回退到包位置的 src 布局推断（``src/`` 的上一级）。

``anchor_path()`` 把相对路径锚定到仓库根；绝对路径原样返回。
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["project_root", "anchor_path", "env_file_candidates"]

#: 本文件绝对路径（src/harness_agent/config/paths.py）
_SELF = Path(__file__).resolve()


def _looks_like_repo(base: Path) -> bool:
    """目录是否为项目仓库根（pyproject.toml 存在即算）。"""
    return (base / "pyproject.toml").is_file()


def project_root() -> Path:
    """返回项目仓库根目录（见模块 docstring 的三级解析规则）。"""
    # 1) 从包位置向上：src 布局开发态 / 可编辑安装态恒命中
    for base in _SELF.parents:
        if _looks_like_repo(base):
            return base
    # 2) 从 CWD 向上：安装态运行、仓库子目录运行
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if _looks_like_repo(base):
            return base
    # 3) 兜底：src/harness_agent/config → parents[2] 是 src/，再上一级是仓库根
    return _SELF.parents[2].parent


def anchor_path(path: str | os.PathLike[str]) -> Path:
    """相对路径 → 锚定仓库根的绝对路径；绝对路径原样返回。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def env_file_candidates() -> list[str]:
    """.env 加载链：仓库根打底、CWD 覆盖（靠后者优先级高）。

    - 仓库根 ``.env`` 保证从任意目录运行都能读到项目配置
      （修复"父目录运行 → .env 加载失败 → 意外 mock 模式"）；
    - CWD ``.env`` 保持标准 dotenv 惯例：就近覆盖仓库默认值。
    """
    root_env = project_root() / ".env"
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env == root_env:
        return [str(root_env)]
    return [str(root_env), str(cwd_env)]
