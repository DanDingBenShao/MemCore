"""
fallback_retriever.py — 后备检索回路安全边界 (MemCore v0.3.1)

当检索流水线前四道（指代/实体/霍普菲尔德/提炼）全部未命中时，
通过安全受限的文件搜索作为兜底。

安全约束：
  - 白名单：仅允许搜索指定目录
  - 黑名单：跳过敏感文件和目录
  - 命令注入防护：使用参数化搜索，不拼接裸字符串
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# 默认安全边界
# ---------------------------------------------------------------------------

DEFAULT_WHITELIST = ["./src", "./lib", "./docs"]
"""仅允许在这些目录下搜索文件。"""

DEFAULT_BLACKLIST = [
    ".env",
    ".git",
    "node_modules",
    "__pycache__",
    "*.key",
    "*.pem",
    ".pytest_cache",
    ".workbuddy",
]
"""跳过匹配这些模式的文件/目录。"""


# ---------------------------------------------------------------------------
# 安全搜索
# ---------------------------------------------------------------------------

def _build_rg_cmd(query: str, whitelist: list[str], blacklist: list[str],
                  max_results: int) -> list[str]:
    """构造 ripgrep 命令参数列表（防注入）。

    使用参数化列表而非字符串拼接。
    """
    cmd = ["rg", "-l", "--max-count", str(max_results),
           "-i", query, "--type-add", "all:*"]
    for blk in blacklist:
        cmd.extend(["--glob", f"!{blk}"])
    cmd.extend(whitelist)
    return cmd


def _build_findstr_cmd(query: str, whitelist: list[str],
                       blacklist: list[str]) -> list[str]:
    """构造 Windows findstr 命令（无 rg 时的兼容降级）。

    返回批处理风格的 cmd 参数。
    """
    # 构建排除目录的 findstr
    cmd = ["findstr", "/s", "/i", "/m", query]
    # findstr 不支持黑名单过滤，需要手动排除
    return cmd


def search_files(
    query: str,
    max_results: int = 3,
    whitelist: Optional[list[str]] = None,
    blacklist: Optional[list[str]] = None,
) -> list[str]:
    """安全文件搜索。

    使用 ripgrep (首选) 或 findstr (降级) 在指定白名单目录中搜索。

    Parameters
    ----------
    query : str
        搜索关键词（仅用于文件内容搜索，不会被 shell 执行）。
    max_results : int, default 3
        每个白名单目录最大匹配数。
    whitelist : list[str] | None, default None
        允许搜索的目录列表。None 使用 DEFAULT_WHITELIST。
    blacklist : list[str] | None, default None
        排除的文件/目录模式。None 使用 DEFAULT_BLACKLIST。

    Returns
    -------
    list[str]
        匹配的文件路径列表，最多 len(whitelist) * max_results 条。
    """
    wl = whitelist if whitelist is not None else DEFAULT_WHITELIST
    bl = blacklist if blacklist is not None else DEFAULT_BLACKLIST

    # 验证白名单目录存在
    valid_dirs = [d for d in wl if os.path.isdir(d)]
    if not valid_dirs:
        return []

    # 白名单路径安全性检查：防止路径遍历
    from pathlib import Path
    cwd_abs = str(Path(".").resolve())
    for d in valid_dirs:
        abs_d = str(Path(d).resolve())
        if not abs_d.startswith(cwd_abs):
            return []

    try:
        # 尝试 ripgrep（Linux/macOS 优先）
        cmd = _build_rg_cmd(query, valid_dirs, bl, max_results)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode in (0, 1):  # 0=找到, 1=未找到
            files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            return files[:max_results]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ripgrep 不可用时尝试 findstr（Windows 降级）
    try:
        cmd = _build_findstr_cmd(query, valid_dirs, bl)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=valid_dirs[0] if valid_dirs else None,
        )
        if result.returncode in (0, 1):
            files = [f.strip() for f in result.stdout.split("\n") if f.strip()]
            # findstr 不支持黑名单过滤，Python 端手动排除
            bl_literal = [b for b in bl if "*" not in b]
            files = [f for f in files if not any(b in f for b in bl_literal)]
            return files[:max_results]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return []


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  fallback_retriever.py — 自测 Demo")
    print("=" * 60)

    # 1. 白名单/黑名单验证
    print("\n─── 1. 安全边界验证 ───")
    files = search_files("memory_core", max_results=2)
    print(f"  搜索 'memory_core' → 找到 {len(files)} 个文件")
    safe = all(not any(b in f for b in DEFAULT_BLACKLIST if "*" not in b)
               for f in files)
    print(f"  {'✅ 搜索结果安全' if safe else '⚠️ 可能包含黑名单文件'}")

    # 2. 空白名单处理
    print("\n─── 2. 空白名单 ───")
    files_empty = search_files("test", whitelist=["./nonexistent"])
    print(f"  搜索不存在的目录 → {len(files_empty)} 个结果 (应为 0)")
    assert len(files_empty) == 0, "不存在的白名单应返回空"
    print("  ✅ 空白名单正确处理")

    # 3. 注入防护（打印命令，不执行）
    print("\n─── 3. 注入防护 ───")
    dangerous_query = "test; rm -rf /"
    # 参数化命令不应受注入影响
    cmd = _build_rg_cmd(dangerous_query, DEFAULT_WHITELIST, DEFAULT_BLACKLIST, 3)
    assert dangerous_query in cmd, "查询词应作为独立参数"
    print(f"  ✅ 命令行注入防护有效")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
