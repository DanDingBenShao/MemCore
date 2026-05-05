"""
code_parser.py — 代码解析器 (MemCore v0.4)

使用 AST 模块从源代码中提取结构化信息：
函数签名、docstring、参数列表、返回类型、模块依赖等。

支持语言：
- Python (ast 标准库)
- 回退：正则表达式兜底

输出格式：
    {
        "module": "auth",
        "functions": [
            {
                "name": "login",
                "params": ["user", "pw"],
                "return_type": "str | None",
                "docstring": "验证用户凭据。",
                "signature": "def login(user, pw) -> str | None"
            }
        ],
        "classes": [...],
        "imports": ["os", "jwt"]
    }
"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Python AST 解析器
# ---------------------------------------------------------------------------


def _parse_python(source: str, file_path: str = "") -> dict[str, Any]:
    """使用 AST 解析 Python 源代码。

    Parameters
    ----------
    source : str
        Python 源代码文本。
    file_path : str, default ""
        文件路径（用于提取模块名）。

    Returns
    -------
    dict
        结构化代码信息。
    """
    result: dict[str, Any] = {
        "module": _infer_module_name(file_path),
        "functions": [],
        "classes": [],
        "imports": [],
        "errors": [],
    }

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        result["errors"].append(str(e))
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = _extract_function(node, source)
            result["functions"].append(func_info)
        elif isinstance(node, ast.ClassDef):
            class_info = _extract_class(node, source)
            result["classes"].append(class_info)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                result["imports"].append(alias.name)

    return result


def _extract_function(node: ast.FunctionDef, source: str) -> dict[str, Any]:
    """从 AST 函数节点提取结构化信息。"""
    # 参数
    params = [arg.arg for arg in node.args.args]

    # 返回类型
    return_annotation = ""
    if node.returns:
        return_annotation = ast.unparse(node.returns)

    # docstring
    docstring = ast.get_docstring(node) or ""

    # 构建签名
    sig_parts = [f"def {node.name}("]
    sig_parts.append(", ".join(params))
    sig_parts.append(")")
    if return_annotation:
        sig_parts.append(f" -> {return_annotation}")
    signature = "".join(sig_parts)

    return {
        "name": node.name,
        "params": params,
        "return_type": return_annotation,
        "docstring": docstring[:200],
        "signature": signature,
        "lineno": node.lineno,
    }


def _extract_class(node: ast.ClassDef, source: str) -> dict[str, Any]:
    """从 AST 类节点提取结构化信息。"""
    bases = [ast.unparse(b) for b in node.bases]
    methods = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            methods.append(_extract_function(item, source))

    docstring = ast.get_docstring(node) or ""

    return {
        "name": node.name,
        "bases": bases,
        "docstring": docstring[:200],
        "methods": methods,
        "lineno": node.lineno,
    }


def _infer_module_name(file_path: str) -> str:
    """从文件路径推断模块名。"""
    if not file_path:
        return ""
    name = file_path.replace("\\", "/").split("/")[-1]
    name = re.sub(r'\.py$|\.go$|\.js$|\.ts$', '', name)
    return name


def _summarize_function(func: dict[str, Any]) -> str:
    """生成单行函数摘要，供记忆系统存储。

    格式："{module}.{name}：{docstring首句}。参数：{params}。返回：{return_type}。"
    """
    parts = [func["name"]]
    if func["docstring"]:
        first_sent = func["docstring"].split("。")[0].split("\n")[0]
        parts.append(f"：{first_sent}。")
    if func["params"]:
        parts.append(f"参数：{'、'.join(func['params'])}。")
    if func["return_type"]:
        parts.append(f"返回：{func['return_type']}。")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 正则兜底解析器（当 AST 不可用或语言不支持时）
# ---------------------------------------------------------------------------


def _parse_regex_fallback(source: str) -> dict[str, Any]:
    """使用正则表达式从源码中提取粗略的结构化信息。

    兜底方案：仅提取函数定义和导入语句。
    """
    result: dict[str, Any] = {
        "module": "",
        "functions": [],
        "classes": [],
        "imports": [],
        "errors": ["AST parse failed, using regex fallback"],
    }

    # 函数定义
    func_pattern = re.compile(
        r'(?:async\s+)?def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^:]+))?\s*:',
        re.DOTALL,
    )
    for m in func_pattern.finditer(source):
        name = m.group(1)
        params_raw = m.group(2)
        params = [p.strip().split(":")[0].strip()
                  for p in params_raw.split(",") if p.strip()]
        return_type = m.group(3).strip() if m.group(3) else ""

        # 提取紧随的 docstring
        body_start = m.end()
        doc_match = re.match(r'\s*"""(.+?)"""', source[body_start:], re.DOTALL)
        docstring = doc_match.group(1)[:200] if doc_match else ""

        result["functions"].append({
            "name": name,
            "params": params,
            "return_type": return_type,
            "docstring": docstring,
            "signature": f"def {name}({', '.join(params)})",
        })

    # 导入语句
    for m in re.finditer(r'^import\s+(\S+)', source, re.MULTILINE):
        result["imports"].append(m.group(1))
    for m in re.finditer(r'^from\s+(\S+)\s+import', source, re.MULTILINE):
        result["imports"].append(m.group(1))

    return result


# ---------------------------------------------------------------------------
# 主接口
# ---------------------------------------------------------------------------


def parse_code(
    source: str,
    language: str = "python",
    file_path: str = "",
) -> dict[str, Any]:
    """解析源代码，提取结构化信息。

    Parameters
    ----------
    source : str
        源代码文本。
    language : str, default "python"
        编程语言（当前支持 "python"）。
    file_path : str, default ""
        文件路径（用于推断模块名）。

    Returns
    -------
    dict
        结构化代码信息。
    """
    if language == "python":
        try:
            return _parse_python(source, file_path)
        except Exception:
            return _parse_regex_fallback(source)
    else:
        return _parse_regex_fallback(source)


def summarize_code(
    source: str,
    language: str = "python",
    file_path: str = "",
) -> list[str]:
    """解析代码并生成适合存入记忆系统的摘要列表。

    每条摘要对应一个函数/方法，格式：
        "{module}.{name}：{docstring首句}。参数：{params}。返回：{return_type}。"

    Parameters
    ----------
    source : str
        源代码文本。
    language : str, default "python"
        编程语言。
    file_path : str, default ""
        文件路径。

    Returns
    -------
    list[str]
        函数摘要列表。
    """
    parsed = parse_code(source, language, file_path)
    summaries = []

    module_name = parsed.get("module", "")

    for func in parsed.get("functions", []):
        summary = _summarize_function(func)
        if module_name:
            summary = f"{module_name}.{summary}"
        summaries.append(summary)

    return summaries


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  code_parser.py — 自测 Demo")
    print("=" * 60)

    # 1. Python AST 解析
    print("\n─── 1. Python AST 解析 ───")
    py_source = '''
def login(user, pw) -> str | None:
    \"\"\"验证用户凭据。
    
    Args:
        user: 用户名
        pw: 密码
    Returns:
        token 或 None
    \"\"\"
    return verify_user(user, pw)

class AuthHandler:
    \"\"\"认证处理器。\"\"\"
    def logout(self, session_id: str) -> bool:
        \"\"\"退出登录。\"\"\"
        return True
'''
    parsed = parse_code(py_source, file_path="auth.py")
    print(f"  模块: {parsed['module']}")
    print(f"  函数数: {len(parsed['functions'])}")
    print(f"  类数: {len(parsed['classes'])}")
    assert len(parsed["functions"]) >= 1
    assert parsed["functions"][0]["name"] == "login"
    assert "user" in parsed["functions"][0]["params"]
    print("  ✅ Python AST 解析通过")

    # 2. 记忆摘要生成
    print("\n─── 2. 记忆摘要生成 ───")
    summaries = summarize_code(py_source, file_path="auth.py")
    for s in summaries:
        print(f"  → {s[:80]}...")
    assert any("login" in s for s in summaries)
    print("  ✅ 摘要生成通过")

    # 3. 正则兜底
    print("\n─── 3. 正则兜底解析 ───")
    bad_source = "def bad(x, y): pass"
    fallback = _parse_regex_fallback(bad_source)
    assert len(fallback["functions"]) >= 1
    assert fallback["functions"][0]["name"] == "bad"
    print(f"  兜底解析: {fallback['functions'][0]['name']}")
    print("  ✅ 正则兜底通过")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
