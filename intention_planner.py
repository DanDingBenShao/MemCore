"""
intention_planner.py — 意图规划器

将用户的原始自然语言输入翻译成标准化的记忆查询文本，
供下游 MemoryCore.recall() 使用。

核心函数:
    plan_memory_query(user_input, model, temperature) -> str

LLM 调用统一由 llm_client 提供，纯本地规则引擎，零外部依赖。
"""

from __future__ import annotations

from typing import Optional

from llm_client import call_llm, PLAN_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def plan_memory_query(
    user_input: str,
    model: str = "deepseek-v3.2",
    temperature: float = 0.1,
) -> str:
    """将用户的原始自然语言输入翻译成标准化的记忆查询文本。

    该函数会：
      1. 替换模糊指代（那个、上次、之前）为具体主题。
      2. 保持输出简洁，不超过 15 个字。
      3. 输出纯文本，不含引号和 JSON。

    Parameters
    ----------
    user_input : str
        用户的原始自然语言输入。
    model : str, default "deepseek-v3.2"
        使用的模型标识符（目前仅用于记录，实际模型调用由 call_llm 控制）。
    temperature : float, default 0.1
        生成温度，控制输出的确定性。

    Returns
    -------
    str
        标准化后的记忆查询文本。
    """
    response = call_llm(
        prompt=user_input,
        system_prompt=PLAN_SYSTEM_PROMPT,
        temperature=temperature,
    )
    return response.strip()


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("intention_planner.py — 自测 Demo")
    print("=" * 55)

    test_cases = [
        "上次说的那个bug还在吗",
        "我们用的向量是多少维的",
        "今天天气真好",
        "我想吃香菜",
        "系统性能太慢了",
        "部署方案确定了吗",
        "单元测试覆盖率如何",
        "那个异常日志是怎么回事",
        "顺便问下架构方案 还有测试进展",
    ]

    for i, case in enumerate(test_cases, 1):
        result = plan_memory_query(case)
        print(f"[{i}] 输入: {case}")
        print(f"    输出: {result}")

    # 向后兼容验证
    assert plan_memory_query("上次说的那个bug还在吗") == "bug 状态"
    assert plan_memory_query("我们用的向量是多少维的") == "向量维度"
    assert plan_memory_query("今天天气真好") == "今日天气"
    assert plan_memory_query("我想吃香菜") == "用户饮食偏好"
    print("\n[兼容性] 全部向后兼容 ✅")

    print("\n" + "=" * 55)
    print("Demo 结束")
    print("=" * 55)
