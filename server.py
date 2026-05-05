"""
server.py — DualMemory 记忆后端 API 服务

基于 FastAPI 构建，暴露四个端点：
  - POST /admin/remember  将对话提炼为记忆并存储到短期
  - POST /api/recall      从短期+长期双层召回记忆
  - POST /admin/consolidate  手动巩固短期记忆到长期
  - GET  /stats           获取双层记忆系统统计信息

启动方式：
    uvicorn server:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from dual_memory import DualMemory
from memory_core import refine_conversation
from profile_config import get_profile, set_profile, ProfileName, ProfileConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic 请求模型
# ---------------------------------------------------------------------------


class RememberRequest(BaseModel):
    """POST /admin/remember 的请求体。"""
    conversation_text: str


class RecallRequest(BaseModel):
    """POST /api/recall 的请求体。"""
    query_text: str
    context_budget: int | None = None
    """可选 token 预算。None 时从配置读取 memory_token_budget。"""


# ---------------------------------------------------------------------------
# 应用实例与全局 DualMemory
# ---------------------------------------------------------------------------

# 在应用生命周期内保持的 DualMemory 单例
memory_system: DualMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时实例化 DualMemory，关闭时清理。

    仅在 memory_system 尚未设置时创建默认实例，
    允许测试在进入 TestClient 前注入自定义实例。
    配置来源：当前 ProfileConfig（默认 dev 模式）。
    """
    global memory_system
    if memory_system is None:
        cfg = get_profile()
        memory_system = DualMemory(profile=cfg)
    yield
    # 关闭时清理（如有需要可扩展）
    memory_system = None


app = FastAPI(
    title="DualMemory API",
    description="双记忆架构后端，支持短期+长期双层记忆、巩固与召回",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 端点 1：POST /admin/remember
# ---------------------------------------------------------------------------


@app.post("/admin/remember")
async def remember(req: RememberRequest) -> dict[str, str]:
    """将一段对话提炼为结构化记忆并存入短期记忆层。

    输入 JSON: {"conversation_text": "原始对话文本"}

    处理流程:
      1. 调用 refine_conversation 获取结构化记忆。
      2. 若返回有效记忆，调用 memory_system.add_memory(...) 存入短期。
      3. 返回 {"status": "success"} 或 {"status": "skipped"}。
    """
    try:
        result = refine_conversation(req.conversation_text)
        if result is not None:
            memory_system.add_memory(
                query=result["query"],
                content=result["content"],
                fingerprint=result["fingerprint"],
            )
            return {"status": "success"}
        return {"status": "skipped"}
    except Exception as e:
        logger.exception("remember 端点异常")
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# 端点 2：POST /api/recall
# ---------------------------------------------------------------------------


@app.post("/api/recall")
async def recall(req: RecallRequest) -> dict[str, list[str]]:
    """从短期+长期双层记忆系统召回最相关的记忆。

    输入 JSON: {"query_text": "当前查询的自然语言描述", "context_budget": 10000}

    返回 JSON: {"memories": ["内容1", "内容2"]}
    """
    try:
        results = memory_system.recall(req.query_text, context_budget=req.context_budget)
        return {"memories": results}
    except Exception as e:
        logger.exception("recall 端点异常")
        return {"memories": [], "error": str(e)}


# ---------------------------------------------------------------------------
# 端点 3：POST /admin/consolidate
# ---------------------------------------------------------------------------


@app.post("/admin/consolidate")
async def consolidate() -> dict[str, int]:
    """手动巩固：将短期中被频繁召回的条目提升到长期记忆。

    返回 JSON: {"consolidated_count": N}
    """
    try:
        n = memory_system.consolidate(force=True)
        return {"consolidated_count": n}
    except Exception as e:
        logger.exception("consolidate 端点异常")
        return {"consolidated_count": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# 端点 4：GET /stats
# ---------------------------------------------------------------------------


@app.get("/stats")
async def stats() -> dict:
    """返回双层记忆系统的统计信息。

    返回 JSON:
    {
        "short_term": {"total_memories": N, "index_size": M},
        "long_term": {"total_memories": N, "index_size": M},
        "pending_consolidation": K
    }
    """
    try:
        return memory_system.get_stats()
    except Exception as e:
        logger.exception("stats 端点异常")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 端点 5：GET /admin/profile
# ---------------------------------------------------------------------------


@app.get("/admin/profile")
async def get_current_profile() -> dict:
    """获取当前运行模式及完整配置。

    返回 JSON: {"profile": "dev", "short_max": 500, ...}
    """
    from profile_config import profile_to_dict, available_profiles
    return {
        "current": profile_to_dict(),
        "available": available_profiles(),
    }


# ---------------------------------------------------------------------------
# 端点 6：POST /admin/profile
# ---------------------------------------------------------------------------


class SetProfileRequest(BaseModel):
    """POST /admin/profile 的请求体。"""
    profile: ProfileName
    overrides: dict | None = None


@app.post("/admin/profile")
async def set_current_profile(req: SetProfileRequest) -> dict:
    """切换运行模式并重建 DualMemory 实例。

    输入 JSON: {"profile": "dev", "overrides": {"short_max": 800}}
    """
    global memory_system
    try:
        cfg = set_profile(req.profile, req.overrides)
        # 重建 DualMemory 实例，从已有数据库加载数据
        memory_system = DualMemory(profile=cfg)
        return {
            "status": "success",
            "profile": req.profile,
            "message": f"已切换到 {req.profile} 模式，DualMemory 实例已重建。",
        }
    except ValueError as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# 端点 7：POST /admin/report_negative (v0.6 新增)
# ---------------------------------------------------------------------------


class ReportNegativeRequest(BaseModel):
    """POST /admin/report_negative 的请求体。"""
    memory_id: int | None = None
    content: str | None = None


@app.post("/admin/report_negative")
async def report_negative(req: ReportNegativeRequest) -> dict:
    """负反馈：降低指定记忆的保护分和访问计数。

    输入 JSON: {"memory_id": 123} 或 {"content": "具体内容"}

    用于 Agent 纠错场景：发现某条记忆检索错误或有误，
    通过此端点降低其权重，减少未来被召回的概率。
    """
    if req.memory_id is None and req.content is None:
        return {"success": False, "error": "需要 memory_id 或 content"}

    target = req.memory_id if req.memory_id is not None else req.content
    try:
        success = memory_system.short_term.report_negative(target)
        if not success:
            # 尝试在长期记忆中查找
            success = memory_system.long_term.report_negative(target)
        return {"success": success}
    except Exception as e:
        logger.exception("report_negative 端点异常")
        return {"success": False, "error": str(e)}
