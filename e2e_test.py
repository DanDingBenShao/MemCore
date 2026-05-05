"""
e2e_test.py — 端到端集成测试

模拟一个多轮对话场景，验证记忆的存入、召回、遗忘全流程。

使用 FastAPI TestClient 进行纯进程内测试，无需外部 HTTP 服务。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

from fastapi.testclient import TestClient

import server as server_module
from intention_planner import plan_memory_query
from memory_core import MemoryCore

# ===========================================================================
# 工具函数
# ===========================================================================

PASS = 0
FAIL = 0


def ok(label: str, condition: bool, detail: str = "") -> None:
    """断言辅助：记录通过/失败。"""
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"    ✅ {label}")
    else:
        FAIL += 1
        msg = f"    ❌ {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def build_prompt_with_memory(
    user_input: str,
    system_prompt: str = "你是一个有用的助手。",
    api_client: TestClient | None = None,
) -> dict:
    """复现 adapter.build_prompt_with_memory 的完整逻辑（使用 TestClient 替代 requests）。"""
    client = api_client or TestClient(server_module.app)

    # 1. 意图规划
    query = plan_memory_query(user_input)

    # 2. 调用记忆 API
    try:
        resp = client.post("/api/recall", json={"query_text": query})
        memories = resp.json().get("memories", [])
    except Exception:
        memories = []

    # 3. 拼装 Prompt
    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        final_sp = f"{system_prompt}\n\n[相关记忆]\n{memory_block}"
    else:
        final_sp = system_prompt

    return {
        "final_system_prompt": final_sp,
        "user_input": user_input,
        "memories_used": memories,
    }


# ===========================================================================
# 主测试
# ===========================================================================


def main() -> int:
    global PASS, FAIL
    PASS = 0
    FAIL = 0

    print("=" * 62)
    print("  MemoryCore 端到端集成测试")
    print("  5 轮对话 + FIFO 遗忘验证")
    print("=" * 62)

    # 使用临时目录，避免跨运行数据残留
    tmp_dir = tempfile.mkdtemp(prefix="e2e_test_")

    # 清理可能存在的默认数据库，防止预存数据干扰
    cleanup()

    # 在 TestClient 进入前设置 DualMemory（lifespan 会检测到已设置，不再覆盖）
    from dual_memory import DualMemory
    server_module.memory_system = DualMemory(
        short_max=200,
        long_max=5000,
        short_threshold=0.12,
        long_threshold=0.3,
        db_dir=tmp_dir,
        consolidation_count=3,
    )

    # TestClient 上下文：进入时触发 startup，退出时触发 shutdown
    with TestClient(server_module.app) as client:

        # ---------- 第 1 轮 ----------
        print(f"\n{'─'*62}")
        print("【第 1 轮】用户：我不喜欢吃香菜，喜欢喝咖啡")
        print(f"{'─'*62}")

        # 第一轮跳过 remember
        result = build_prompt_with_memory(
            "我不喜欢吃香菜，喜欢喝咖啡",
            system_prompt="你是一个有用的助手。",
            api_client=client,
        )
        query1 = plan_memory_query('我不喜欢吃香菜，喜欢喝咖啡')
        print(f"  标准化查询: {query1!r}")
        print(f"  召回记忆数: {len(result['memories_used'])}")
        ok("初始阶段无记忆可召回", len(result["memories_used"]) == 0)
        sp_short = result['final_system_prompt'][:200]
        print(f"  System Prompt (前200字符): {sp_short}")
        ok("Prompt 未被追加记忆块",
           result["final_system_prompt"] == "你是一个有用的助手。")
        print(f"  💬 模拟回复: {query1}")

        # ---------- 第 2 轮 ----------
        print(f"\n{'─'*62}")
        print("【第 2 轮】用户：我们项目决定用384维向量做记忆矩阵")
        print(f"{'─'*62}")

        # 存入上一轮对话（食物偏好）
        resp = client.post(
            "/admin/remember",
            json={"conversation_text": "我不喜欢吃香菜，喜欢喝咖啡"},
        )
        print(f"  存入第1轮对话 → {resp.json()}")
        ok("/admin/remember 返回 success", resp.json()["status"] == "success")

        result = build_prompt_with_memory(
            "我们项目决定用384维向量做记忆矩阵",
            api_client=client,
        )
        query2 = plan_memory_query('我们项目决定用384维向量做记忆矩阵')
        print(f"  标准化查询: {query2!r}")
        print(f"  召回记忆数: {len(result['memories_used'])}")
        stats = client.get("/stats").json()
        print(f"  当前统计: {stats}")
        ok("记忆条数增长至 1", stats["short_term"]["total_memories"] == 1)
        print(f"  💬 模拟回复: {query2}")

        # ---------- 第 3 轮（关键验证） ----------
        print(f"\n{'─'*62}")
        print("【第 3 轮】用户：上次说的那个饮食偏好的事，你还记得吗")
        print(f"{'─'*62}")

        # 存入第 2 轮对话（向量选择）
        resp = client.post(
            "/admin/remember",
            json={"conversation_text": "我们项目决定用384维向量做记忆矩阵"},
        )
        print(f"  存入第2轮对话 → {resp.json()}")
        ok("向量对话存入成功", resp.json()["status"] == "success")

        result = build_prompt_with_memory(
            "上次说的那个饮食偏好的事，你还记得吗",
            api_client=client,
        )
        query3 = plan_memory_query('上次说的那个饮食偏好的事，你还记得吗')
        print(f"  标准化查询: {query3!r}")
        print(f"  召回记忆: {result['memories_used']}")
        sp_short = result['final_system_prompt'][:200]
        print(f"  System Prompt (前200字符):")
        print(f"    {sp_short}")

        # ★ 验证点 1：召回内容包含"香菜"或"咖啡"
        ok("召回记忆非空", len(result["memories_used"]) > 0)
        food_recalled = any(
            kw in m for m in result["memories_used"] for kw in ("香菜", "咖啡")
        )
        ok("召回内容包含「香菜」或「咖啡」", food_recalled,
           f"召回: {result['memories_used']}")

        stats = client.get("/stats").json()
        print(f"  当前统计: {stats}")
        ok("记忆条数增长至 2", stats["short_term"]["total_memories"] == 2)
        print(f"  💬 模拟回复: {query3}")

        # ---------- 第 4 轮（不相关查询） ----------
        print(f"\n{'─'*62}")
        print("【第 4 轮】用户：今天天气真好")
        print(f"{'─'*62}")

        # 存入第 3 轮对话（查询文本 → mock 返回 skipped）
        resp = client.post(
            "/admin/remember",
            json={"conversation_text": "上次说的那个饮食偏好的事，你还记得吗"},
        )
        print(f"  存入第3轮对话 → {resp.json()}")

        result = build_prompt_with_memory(
            "今天天气真好",
            api_client=client,
        )
        query4 = plan_memory_query('今天天气真好')
        print(f"  标准化查询: {query4!r}")
        print(f"  召回记忆: {result['memories_used']}")

        # ★ 验证点 2：不相关查询 memories_used 为空
        ok("不相关查询召回为空", len(result["memories_used"]) == 0,
           f"召回: {result['memories_used']}")

        stats = client.get("/stats").json()
        ok("记忆条数未变（skipped）", stats["short_term"]["total_memories"] == 2)
        print(f"  💬 模拟回复: {query4}")

        # ---------- 第 5 轮（验证持久性） ----------
        print(f"\n{'─'*62}")
        print("【第 5 轮】用户：我之前说不喜欢吃什么来着")
        print(f"{'─'*62}")

        # 存入第 4 轮对话（天气 → skipped）
        resp = client.post(
            "/admin/remember",
            json={"conversation_text": "今天天气真好"},
        )
        print(f"  存入第4轮对话 → {resp.json()}")

        result = build_prompt_with_memory(
            "我之前说不喜欢吃什么来着",
            api_client=client,
        )
        query5 = plan_memory_query('我之前说不喜欢吃什么来着')
        print(f"  标准化查询: {query5!r}")
        print(f"  召回记忆: {result['memories_used']}")

        # ★ 验证点 3：召回内容仍包含"香菜"
        ok("召回记忆非空", len(result["memories_used"]) > 0)
        food_still_recalled = any(
            kw in m for m in result["memories_used"] for kw in ("香菜", "咖啡")
        )
        ok("召回内容仍包含「香菜」或「咖啡」", food_still_recalled,
           f"召回: {result['memories_used']}")

        stats = client.get("/stats").json()
        ok("记忆条数仍为 2", stats["short_term"]["total_memories"] == 2)
        print(f"  💬 模拟回复: {query5}")

        # ==================================================================
        # 扩展测试：FIFO 遗忘（独立 MemoryCore 实例）
        # ==================================================================
        print(f"\n\n{'='*62}")
        print("  [扩展] FIFO 遗忘验证")
        print(f"{'='*62}")

        mc_fifo = MemoryCore(max_memories=3, similarity_threshold=0.3)
        mc_fifo.add_memory("A", "早期记忆A", "记忆A 标签")
        mc_fifo.add_memory("B", "早期记忆B", "记忆B 标签")
        mc_fifo.add_memory("C", "中期记忆C", "记忆C 标签")
        print(f"  存入 3 条后: {mc_fifo.get_stats()}")

        mc_fifo.add_memory("D", "最新记忆D", "记忆D 标签")
        print(f"  存入第 4 条（触发遗忘）: {mc_fifo.get_stats()}")
        print(f"  当前 memories 列表:")
        for m in mc_fifo.memories:
            print(f"    └ {m['content']}")

        ok("触发遗忘后条数 = max_memories", len(mc_fifo.memories) == 3)
        ok("早期记忆A 已被遗忘",
           all("早期记忆A" not in m["content"] for m in mc_fifo.memories))
        ok("最新记忆D 已存入",
           any("最新记忆D" in m["content"] for m in mc_fifo.memories))

        res_forgotten = mc_fifo.recall("关于记忆A的内容")
        ok("被遗忘的记忆无法被召回",
           all("早期记忆A" not in r for r in res_forgotten),
           f"召回: {res_forgotten}")

        # ==================================================================
        # 汇总
        # ==================================================================
        print(f"\n{'='*62}")
        print(f"  测试完成: {PASS}/{PASS + FAIL} 通过", end="")
        print("  ✅" if FAIL == 0 else f"  ❌ {FAIL} 失败")
        print(f"{'='*62}")

    # 清理临时数据库
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return FAIL


def cleanup():
    """清理临时测试数据。"""
    db_dir = "./memory_db"
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        cleanup()
    sys.exit(exit_code)
