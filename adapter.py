"""
adapter.py — 在线推理适配器

桥接 intention_planner 与 MemoryCore API，自动完成：
  1. 用户输入 → 标准化查询（plan_memory_query）
  2. 标准化查询 → 记忆召回（POST /api/recall）
  3. 记忆上下文 → 拼装最终 System Prompt

核心函数:
    build_prompt_with_memory(user_input, system_prompt, memory_api_url) -> dict
"""

from __future__ import annotations

from typing import Optional

import requests

from intention_planner import plan_memory_query

# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def build_prompt_with_memory(
    user_input: str,
    system_prompt: str = "你是一个有用的助手。",
    memory_api_url: str = "http://localhost:8000/api/recall",
) -> dict:
    """接收用户原始输入，自动规划查询、召回记忆、拼装最终 Prompt。

    Parameters
    ----------
    user_input : str
        用户的原始自然语言输入。
    system_prompt : str, default "你是一个有用的助手。"
        原始系统提示词，记忆上下文将追加到其后。
    memory_api_url : str, default "http://localhost:8000/api/recall"
        记忆 API 的召回端点地址。

    Returns
    -------
    dict
        包含以下键：
        - "final_system_prompt" : str — 拼装了记忆上下文的最终 System Prompt
        - "user_input"          : str — 用户原始输入（原样保留）
        - "memories_used"      : list[str] — 本次使用的记忆内容列表（调试用）
    """
    # 默认返回值：API 不可用或查询无结果时回退到此
    memories_used: list[str] = []

    # ---- a. 意图规划 ----
    try:
        query_text: str = plan_memory_query(user_input)
    except Exception:
        query_text = user_input  # 规划失败时直接用原始输入

    # ---- b. 调用记忆 API ----
    try:
        resp = requests.post(
            memory_api_url,
            json={"query_text": query_text},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            memories_used = data.get("memories", [])
    except (requests.RequestException, ValueError):
        # API 不可用、超时或返回非 JSON：静默降级
        pass

    # ---- c. 拼装最终 System Prompt ----
    if memories_used:
        memory_block = "\n".join(f"- {m}" for m in memories_used)
        final_system_prompt = f"{system_prompt}\n\n[相关记忆]\n{memory_block}"
    else:
        final_system_prompt = system_prompt

    return {
        "final_system_prompt": final_system_prompt,
        "user_input": user_input,
        "memories_used": memories_used,
    }


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import json

    print("=" * 55)
    print("adapter.py — 自测 Demo")
    print("=" * 55)

    DEMO_API_URL = "http://127.0.0.1:18127/api/recall"
    DEMO_ADMIN_URL = "http://127.0.0.1:18127/admin/remember"

    # ---- 检查记忆服务是否在线 ----
    server_online = False
    try:
        r = requests.get(f"http://127.0.0.1:18127/stats", timeout=3)
        server_online = r.status_code == 200
        print(f"\n[检查] 记忆服务: {'✅ 在线' if server_online else '❌ 离线'}")
    except requests.RequestException:
        print(f"\n[检查] 记忆服务: ❌ 离线")

    # ---- 如果服务在线，预存一条 bug 相关的记忆 ----
    if server_online:
        try:
            resp = requests.post(
                DEMO_ADMIN_URL,
                json={
                    "conversation_text": (
                        "用户：深色模式切换按钮点了没反应。\n"
                        "助手：收到，这是一个已知bug，正在修复夜间模式的切换逻辑。"
                    )
                },
                timeout=5,
            )
            print(f"[预存] 存入深色模式bug记忆: {resp.json()}")
        except Exception as e:
            print(f"[预存] 失败: {e}")

    # ---- 测试 1：正常流程（含记忆上下文） ----
    print(f"\n─── 测试 1：用户查询 bug ───")

    # 使用明确的系统提示词
    my_system_prompt = "你是一个资深前端工程师，擅长调试和修复 UI 问题。"

    result = build_prompt_with_memory(
        user_input="上次说的那个bug还在吗",
        system_prompt=my_system_prompt,
        memory_api_url=DEMO_API_URL,
    )

    print(f"用户输入: {result['user_input']}")
    print(f"使用记忆: {result['memories_used']}")
    print(f"最终 System Prompt:\n{result['final_system_prompt']}")

    # ---- 测试 2：API 不可用时的降级行为 ----
    print(f"\n─── 测试 2：API 不可用降级 ───")

    result_fallback = build_prompt_with_memory(
        user_input="上次说的那个bug还在吗",
        system_prompt=my_system_prompt,
        memory_api_url="http://127.0.0.1:19999/api/recall",  # 故意写错的端口
    )

    print(f"API 地址: http://127.0.0.1:19999/api/recall (不可用)")
    print(f"使用记忆: {result_fallback['memories_used']}")
    print(f"最终 System Prompt 与原 Prompt 一致: "
          f"{result_fallback['final_system_prompt'] == my_system_prompt}")
    print(f"    → {result_fallback['final_system_prompt']}")

    # ---- 测试 3：不相关查询（无记忆返回） ----
    print(f"\n─── 测试 3：不相关查询 ───")

    result_empty = build_prompt_with_memory(
        user_input="今天天气真好",
        system_prompt=my_system_prompt,
        memory_api_url=DEMO_API_URL,
    )

    print(f"用户输入: {result_empty['user_input']}")
    print(f"使用记忆: {result_empty['memories_used']}")
    print(f"最终 System Prompt 与原 Prompt 一致: "
          f"{result_empty['final_system_prompt'] == my_system_prompt}")

    print("\n" + "=" * 55)
    print("Demo 结束")
    print("=" * 55)
