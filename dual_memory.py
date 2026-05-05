"""
dual_memory.py — 双记忆架构 + 事件索引三层漏斗 (v0.6)

模拟人类记忆的分层巩固机制：
  - 短期记忆：容量小、召回阈值低、写入快
  - 长期记忆：容量大、召回阈值高、通过巩固从短期提升
  - 事件索引（新增）：LLM/规则引擎做梦后产生的结构化知识块

检索优先级：
  Layer 1 (首选): 事件索引 — 主题聚合知识，高精度
  Layer 2 (回退): 原始记忆（短期+长期）— 始终可用

依赖：memory_core, event_index, dream_processor, profile_config
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from memory_core import MemoryCore
from profile_config import ProfileConfig, get_profile
from llm_client import is_noise  # 噪音检测

# v0.6 事件索引 & 做梦进程
from event_index import EventIndex
from dream_processor import DreamScheduler


# ===========================================================================
# 辅助函数
# ===========================================================================

def _ensure_dir(path: str) -> None:
    """确保目录存在。"""
    if path and path != ":memory:":
        os.makedirs(path, exist_ok=True)


def _db_path(base_dir: str, name: str) -> str:
    """计算数据库文件路径。":memory:" 模式直接返回。"""
    if base_dir == ":memory:":
        return ":memory:"
    return os.path.join(base_dir, name)


def _deduplicate_ordered(seq: list[str]) -> list[str]:
    """保持顺序去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _estimate_token_count(text: str) -> int:
    """估算文本 token 数（简单估算）。"""
    # 中文约 1.5 token/字，英文约 1 token/词
    chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english_words = len([w for w in text.split() if w.strip() and all('\u0000' <= c <= '\u007f' for c in w)])
    return int(chinese_chars * 1.5 + english_words)


def _assess_query_complexity(query: str, intent: str) -> str:
    """评估查询复杂度：simple | medium | complex。"""
    # 复杂模式关键词
    complex_patterns = [
        "所有", "全部", "完整", "总结", "梳理", "回顾",
        "历史", "讨论过", "聊过", "涉及", "相关",
    ]
    # 简单模式关键词
    simple_patterns = [
        "叫什么", "是什么", "多少", "哪个", "谁",
    ]
    
    if any(p in query for p in complex_patterns):
        return "complex"
    if intent == "reference":
        return "simple"
    if any(p in query for p in simple_patterns):
        return "simple"
    return "medium"


def _is_covered_by_event(short_content: str, event_text: str, threshold: float = 0.7) -> bool:
    """判断短期记忆条目是否被事件文本覆盖。"""
    if len(short_content) < 3 or len(event_text) < 3:
        return False
    # trigram Jaccard 相似度
    tri1 = {short_content[i:i+3] for i in range(len(short_content) - 2)}
    tri2 = {event_text[i:i+3] for i in range(len(event_text) - 2)}
    if not tri1 or not tri2:
        return False
    jaccard = len(tri1 & tri2) / len(tri1 | tri2)
    return jaccard >= threshold


# ===========================================================================
# DualMemory 类
# ===========================================================================


class DualMemory:
    """双记忆架构：短期 + 长期双层记忆系统。

    Parameters
    ----------
    short_max : int | None
        短期记忆容量上限。None 时从 ProfileConfig 读取。
    long_max : int | None
        长期记忆容量上限。None 时从 ProfileConfig 读取。
    short_threshold : float | None
        短期召回阈值。None 时从 ProfileConfig 读取。
    long_threshold : float | None
        长期召回阈值。None 时从 ProfileConfig 读取。
    db_dir : str | None
        数据库文件存放目录。None 时从 ProfileConfig 读取。
    consolidation_count : int | None
        巩固阈值。None 时从 ProfileConfig 读取。
    top_k : int | None
        MemoryCore 每次 recall 返回数。None 时从 ProfileConfig 读取。
    recall_merge_max : int | None
        合并去重后最大返回数。None 时从 ProfileConfig 读取。
    profile : ProfileConfig | None
        显式传入的配置实例。若提供，忽略其他 None 参数直接从中读取。
    """

    def __init__(
        self,
        short_max: Optional[int] = None,
        long_max: Optional[int] = None,
        short_threshold: Optional[float] = None,
        long_threshold: Optional[float] = None,
        db_dir: Optional[str] = None,
        consolidation_count: Optional[int] = None,
        top_k: Optional[int] = None,
        recall_merge_max: Optional[int] = None,
        profile: Optional[ProfileConfig] = None,
        use_disk_store: Optional[bool] = None,  # v0.7: 磁盘模式
    ) -> None:
        # 从 ProfileConfig 读取未指定的参数
        cfg = profile or get_profile()
        self.short_max = short_max if short_max is not None else cfg.short_max
        self.long_max = long_max if long_max is not None else cfg.long_max
        self.short_threshold = short_threshold if short_threshold is not None else cfg.short_threshold
        self.long_threshold = long_threshold if long_threshold is not None else cfg.long_threshold
        self.db_dir = db_dir if db_dir is not None else cfg.db_dir
        self.consolidation_count = consolidation_count if consolidation_count is not None else cfg.consolidation_count
        _top_k = top_k if top_k is not None else cfg.top_k
        self.recall_merge_max = recall_merge_max if recall_merge_max is not None else cfg.recall_merge_max
        
        # Token 预算参数（v0.7）
        self.memory_token_budget = getattr(cfg, "memory_token_budget", 10000)
        self.token_per_memory = getattr(cfg, "token_per_memory", 60)
        self.event_budget_ratio = getattr(cfg, "event_budget_ratio", 0.30)
        self.short_budget_ratio = getattr(cfg, "short_budget_ratio", 0.50)
        self.long_budget_ratio = getattr(cfg, "long_budget_ratio", 0.20)
        # 意图感知预算分配（v0.7）
        self.intent_budget_overrides = getattr(cfg, "intent_budget_overrides", {})
        # 时间窗口（v0.7）
        self.short_term_window_hours = getattr(cfg, "short_term_window_hours", 24)

        # v0.7 磁盘模式
        self.use_disk_store = use_disk_store if use_disk_store is not None else getattr(cfg, "use_disk_store", False)

        # 确保数据库目录存在（使用已从 profile 解析的 self.db_dir）
        _ensure_dir(self.db_dir)

        # 初始化短期和长期记忆
        self.short_term = MemoryCore(
            max_memories=self.short_max,
            similarity_threshold=self.short_threshold,
            db_path=_db_path(self.db_dir, "short.db"),
            top_k=_top_k,
            use_disk_store=self.use_disk_store,
        )
        self.long_term = MemoryCore(
            max_memories=self.long_max,
            similarity_threshold=self.long_threshold,
            db_path=_db_path(self.db_dir, "long.db"),
            top_k=_top_k,
            use_disk_store=self.use_disk_store,
        )

        # 访问计数：(query, content) -> 被召回次数
        # 使用元组作 key 避免 content 相同但 query 不同的碰撞
        self.access_count: dict[tuple[str, str], int] = {}

        # #5 写入锁（单 worker 场景下防 _auto_consolidate_on_forget 竞争）
        self._add_memory_lock = threading.Lock()

        # ── v0.6 事件索引 & 做梦 ──
        # 事件索引：三层漏斗的 Layer 1（高精度检索路径，回退到原始记忆）
        self.event_index = EventIndex(
            db_dir=self.db_dir,
            top_k=_top_k,
            recall_merge_max=self.recall_merge_max,
        )
        # 做梦调度器：从 profile 读取配置参数
        self.dream_scheduler = DreamScheduler(
            event_index=self.event_index,
            idle_threshold=getattr(cfg, "dream_min_interval", 180) // 3,  # idle = interval/3
            min_interval=getattr(cfg, "dream_min_interval", 180),
            min_new_memories=getattr(cfg, "dream_min_new_memories", 15),
        )

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭数据库连接，释放文件锁。"""
        import db
        if self.short_term.conn:
            self.short_term.conn.close()
            self.short_term.conn = None
        if self.long_term.conn:
            self.long_term.conn.close()
            self.long_term.conn = None
        if self.event_index.conn:
            self.event_index.conn.close()
            self.event_index.conn = None

    def __del__(self) -> None:
        """析构时确保关闭连接。"""
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 公开方法
    # ------------------------------------------------------------------

    def add_memory(self, query: str, content: str, fingerprint: str) -> bool:
        """将记忆写入短期记忆层（写入锁保护）。
        
        写入后检查时间窗口：超出窗口的短期记忆自动巩固到长期。
        
        Parameters
        ----------
        query : str
            标准化查询主题。
        content : str
            核心事实陈述。
        fingerprint : str
            关键词串。
        
        Returns
        -------
        bool
            True 表示成功写入，False 表示被噪音过滤跳过。
        """
        # 噪音过滤：纯确认/指令句跳过写入
        if is_noise(content) or is_noise(query):
            return False
        
        with self._add_memory_lock:
            self.short_term.add_memory(query, content, fingerprint)
            self.dream_scheduler.record_new_memory()
            self._auto_consolidate_on_forget()
            self._expire_short_term_window()
            self.dream_scheduler.try_dream(self.short_term, self.long_term)
        return True

    def _expire_short_term_window(self) -> None:
        """时间窗口过期清理：超出窗口的短期记忆自动巩固到长期。"""
        import time
        now = time.time()
        window_seconds = self.short_term_window_hours * 3600
        
        # 收集超出时间窗口的条目索引
        expired_indices: list[int] = []
        for i, ts in enumerate(self.short_term.timestamps):
            if now - ts > window_seconds:
                expired_indices.append(i)
        
        # 从后往前处理，避免索引偏移
        for i in reversed(expired_indices):
            entry = self.short_term.memories[i]
            # 巩固到长期（移动语义）
            self.long_term.add_memory(
                query=entry["query"],
                content=entry["content"],
                fingerprint=entry["fingerprint"],
            )
            # 清除 DualMemory 层的访问记录
            self.access_count.pop((entry["query"], entry["content"]), None)
            # 从短期记忆中移除
            self.short_term._remove_at(i)

    def recall(self, query_text: str, context_budget: int | None = None) -> list[str]:
        """Token 预算制的三层串行召回。

        流程：
          1. 评估查询复杂度，计算 token 预算
          2. 第一层：事件记忆（骨架），填充 30% 预算
          3. 第二层：短期记忆（细节），填充 50% 预算，跳过已被事件覆盖的
          4. 第三层：长期记忆（深度），填充 20% 预算
          5. 累计 token 直到预算用完，不硬塞

        Parameters
        ----------
        query_text : str
            用户的自然语言查询。
        context_budget : int | None
            可选的 token 预算上限。None 时从配置读取。

        Returns
        -------
        list[str]
            最相关的记忆内容列表，总量不超过 token 预算。
        """
        self.dream_scheduler.record_operation()

        # ── 时间窗口清理 ──
        # 每次 recall 时也检查超时记忆（防止长期不写入时超时记忆不被清理）
        self._expire_short_term_window()

        # ── Token 预算计算 ──
        # 意图分类（简化版）
        import re
        if re.search(r'刚才|上次|之前|第\\d+轮|前几轮|那个', query_text):
            intent = "reference"
        elif re.search(r'\\.py$|\\.go$|\\.md$|\\.txt$|文档|文件', query_text):
            intent = "document"
        elif re.search(r'[a-z_][a-zA-Z0-9_]*(\\.[a-z_][a-zA-Z0-9_]*)+', query_text):
            intent = "entity"
        else:
            intent = "semantic"
        
        complexity = _assess_query_complexity(query_text, intent)
        
        # Token 预算（可从外部传入）
        base_budget = context_budget or self.memory_token_budget
        
        # 根据复杂度调整预算
        complexity_multipliers = {"simple": 0.5, "medium": 1.0, "complex": 1.5}
        budget = int(base_budget * complexity_multipliers.get(complexity, 1.0))
        
        # 各层预算分配（意图感知）
        overrides = self.intent_budget_overrides.get(intent, {})
        event_ratio = overrides.get("event", self.event_budget_ratio)
        short_ratio = overrides.get("short", self.short_budget_ratio)
        long_ratio = overrides.get("long", self.long_budget_ratio)
        
        event_budget = int(budget * event_ratio)
        short_budget = int(budget * short_ratio)
        long_budget = int(budget * long_ratio)

        results: list[str] = []
        used_tokens = 0
        event_texts: set[str] = set()  # 已返回的事件文本（用于去重）

        # ── 第一层：事件记忆（骨架）──
        # 动态填充：命中多少用多少，不强求填满 event_budget
        if len(self.event_index.event_list) > 0:
            event_raw = self.event_index.recall(query_text)
            for event_text in event_raw:
                tokens = _estimate_token_count(event_text)
                # 用总预算的 80% 作为事件上限（防止单层占满）
                if used_tokens + tokens <= int(budget * 0.5):
                    results.append(event_text)
                    event_texts.add(event_text)
                    used_tokens += tokens
                else:
                    continue  # 超过上限，跳过这条，继续尝试更小的

        # ── 第二层：短期记忆（细节）──
        # 跳过已被事件覆盖的条目
        short_raw = self.short_term.recall(query_text)
        for content in short_raw:
            # 检查是否被任一事件覆盖（合并所有事件文本，但阈值更严格）
            is_covered = False
            for et in event_texts:
                if _is_covered_by_event(content, et, threshold=0.7):  # 单事件判断
                    is_covered = True
                    break
            if is_covered:
                continue
            
            tokens = _estimate_token_count(content)
            # 用剩余预算填充（不超过 80% 总预算）
            if used_tokens + tokens <= int(budget * 0.8):
                if content not in results:  # 去重
                    results.append(content)
                    used_tokens += tokens
                    # 更新访问计数
                    content_to_query = {m["content"]: m["query"] for m in self.short_term.memories}
                    q = content_to_query.get(content, "")
                    key = (q, content)
                    self.access_count[key] = self.access_count.get(key, 0) + 1
            else:
                continue  # 超过预算，跳过这条，继续尝试更小的

        # ── 第三层：长期记忆（深度）──
        long_raw = self.long_term.recall(query_text)
        for content in long_raw:
            # 也检查事件覆盖
            is_covered = False
            for et in event_texts:
                if _is_covered_by_event(content, et, threshold=0.7):
                    is_covered = True
                    break
            if is_covered:
                continue
            
            tokens = _estimate_token_count(content)
            if used_tokens + tokens <= budget:  # 用总预算控制
                if content not in results:  # 去重
                    results.append(content)
                    used_tokens += tokens
                    # 更新访问计数
                    content_to_query = {m["content"]: m["query"] for m in self.long_term.memories}
                    q = content_to_query.get(content, "")
                    key = (q, content)
                    self.access_count[key] = self.access_count.get(key, 0) + 1
            else:
                break  # 预算用完，停止

        return results

    def consolidate(self, force: bool = False) -> int:
        """巩固：将短期中被频繁召回的条目提升到长期记忆（移动语义）。

        巩固后的条目从短期记忆中移除，释放短期空间。
        使用 MemoryCore 内部的 access_counts 作为访问计数来源
        （比 DualMemory.access_count 更准确，因为 recall() 已同步更新）。

        Parameters
        ----------
        force : bool, default False
            如果为 True，巩固所有短期记忆条目（忽略 access_count）。

        Returns
        -------
        int
            本次巩固的条目数量。
        """
        count = 0
        # 收集需要巩固的条目索引（从后往前遍历，避免删除时索引偏移）
        indices_to_consolidate: list[int] = []

        for i, entry in enumerate(self.short_term.memories):
            # 优先使用 MemoryCore 内部的 access_counts
            internal_acc = (
                self.short_term.access_counts[i]
                if i < len(self.short_term.access_counts) else 0
            )
            key = (entry["query"], entry["content"])
            external_acc = self.access_count.get(key, 0)
            # 取两者中较大的值，避免遗漏
            acc = max(internal_acc, external_acc)

            if force or acc >= self.consolidation_count:
                indices_to_consolidate.append(i)

        # 从后往前移除，避免索引偏移
        for i in reversed(indices_to_consolidate):
            entry = self.short_term.memories[i]
            # 写入长期记忆
            self.long_term.add_memory(
                query=entry["query"],
                content=entry["content"],
                fingerprint=entry["fingerprint"],
            )
            # 清除 DualMemory 层的访问记录
            self.access_count.pop((entry["query"], entry["content"]), None)
            # 从短期记忆中移除（移动语义）
            self.short_term._remove_at(i)
            count += 1

        # ── 保护分衰减（v0.7）──
        # 未被巩固的记忆保护分统一衰减，促进遗忘不活跃记忆
        self._decay_protection_scores()

        return count

    def _decay_protection_scores(self) -> None:
        """保护分衰减：基于时间衰减，非每次 consolidate 都触发。
        
        衰减规则：
        - 距离上次 consolidate 超过 1 小时才触发衰减
        - 衰减因子 = 0.98^(经过的小时数)
        - 最低保持 1.0（防止新记忆过快遗忘）
        """
        import time
        import db
        
        # 距离上次 consolidate 的时间
        if not hasattr(self, '_last_consolidate_time'):
            self._last_consolidate_time = time.time()
            return  # 首次 consolidate，不衰减
        
        elapsed_hours = (time.time() - self._last_consolidate_time) / 3600
        self._last_consolidate_time = time.time()
        
        if elapsed_hours < 1.0:
            return  # 不到 1 小时，不衰减
        
        # 按经过的小时数计算衰减因子
        decay_factor = 0.98 ** elapsed_hours
        pending_updates: list[tuple[int, int, float]] = []
        
        for i, entry in enumerate(self.short_term.memories):
            old_score = self.short_term.protection_scores[i]
            new_score = max(1.0, old_score * decay_factor)  # 不低于 1.0
            self.short_term.protection_scores[i] = new_score
            
            # 持久化到数据库
            db_id = entry.get("db_id")
            if db_id is not None:
                pending_updates.append(
                    (db_id, self.short_term.access_counts[i], new_score)
                )
        
        # 批量持久化
        if pending_updates and self.short_term.conn:
            db.batch_update_memory_stats(self.short_term.conn, pending_updates)

    def _auto_consolidate_on_forget(self) -> None:
        """在短期记忆即将遗忘前，自动巩固被频繁访问的条目。

        使用 MemoryCore._forget_oldest() 的返回值获取将被遗忘的条目，
        若其访问计数 >= consolidation_count，则先巩固到长期再遗忘。

        访问计数来源：优先使用 MemoryCore 内部的 access_counts
        （recall() 时已同步更新），回退到 DualMemory.access_count。
        """
        if len(self.short_term.memories) < self.short_term.max_memories:
            return

        evicted = self.short_term._forget_oldest()
        if evicted is None:
            return

        content = evicted["content"]

        # 访问计数：取 MemoryCore 内部计数 和 DualMemory 层的最大值
        internal_acc = evicted.get("access_count", 0)
        key = (evicted["query"], content)
        external_acc = self.access_count.get(key, 0)
        acc = max(internal_acc, external_acc)

        if acc >= self.consolidation_count:
            self.long_term.add_memory(
                query=evicted["query"],
                content=content,
                fingerprint=evicted["fingerprint"],
            )
            self.access_count.pop(key, None)

    def get_stats(self) -> dict[str, Any]:
        """返回记忆系统的完整统计信息。

        包含 short_term, long_term, event_index, dream_scheduler 四个维度。

        Returns
        -------
        dict
            包含四个子字段的统计字典。
        """
        stats = {
            "short_term": self.short_term.get_stats(),
            "long_term": self.long_term.get_stats(),
            "pending_consolidation": len(self.access_count),
            "event_index": self.event_index.get_stats(),
            "dream_scheduler": self.dream_scheduler.get_stats(),
        }
        return stats


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("DualMemory 双记忆架构 — 自测 Demo")
    print("=" * 60)

    # 1. 创建实例（使用 :memory: 避免文件残留）
    dm = DualMemory(
        short_max=5,
        long_max=20,
        short_threshold=0.12,
        long_threshold=0.4,
        db_dir=":memory:",
        consolidation_count=2,  # 为演示方便，2 次就巩固
    )
    print(f"\n[1] 创建 DualMemory")
    print(f"    短期容量: {dm.short_max}, 长期容量: {dm.long_max}")
    print(f"    巩固阈值: {dm.consolidation_count} 次")

    # 2. 先存入 3 条记忆（不触发遗忘）
    print(f"\n[2] 存入 3 条记忆（短期未满）...")
    for i in range(3):
        dm.add_memory(
            query=f"记忆{i}",
            content=f"记忆内容_{i}",
            fingerprint=f"记忆{i}",
        )
    stats = dm.get_stats()
    print(f"    短期: {stats['short_term']['total_memories']} 条 (上限 {dm.short_max})")
    print(f"    长期: {stats['long_term']['total_memories']} 条")

    # 3. 模拟 3 次对「记忆内容_0」的召回
    print(f"\n[3] 模拟 3 次对「记忆内容_0」的召回...")
    for i in range(3):
        r = dm.recall("记忆0")
        print(f"    第 {i+1} 次召回结果: {r}")
    acc = dm.access_count.get(("记忆0", "记忆内容_0"), 0)
    print(f"    access_count[('记忆0', '记忆内容_0')] = {acc}")
    assert acc == 3, f"access_count 应为 3，实际 {acc}"
    print("    ✓ access_count 累加正确")

    # 4. 手动巩固
    print(f"\n[4] 手动巩固 (consolidate)...")
    short_before = dm.get_stats()['short_term']['total_memories']
    n = dm.consolidate()
    print(f"    巩固了 {n} 条记忆到长期")
    stats = dm.get_stats()
    print(f"    巩固前短期: {short_before} 条")
    print(f"    巩固后短期: {stats['short_term']['total_memories']} 条")
    print(f"    巩固后长期: {stats['long_term']['total_memories']} 条")
    print(f"    待巩固计数: {stats['pending_consolidation']}")
    assert stats["long_term"]["total_memories"] > 0, "长期应有记忆"
    # 移动语义验证：巩固后短期应减少
    assert stats["short_term"]["total_memories"] < short_before, (
        f"巩固后短期应减少: {short_before} -> {stats['short_term']['total_memories']}"
    )
    print("    ✓ 巩固逻辑正确（移动语义）")

    # 5. 再存入更多记忆，触发短期遗忘（记忆内容_0 应被遗忘）
    print(f"\n[5] 再存入 4 条触发遗忘...")
    for i in range(3, 7):
        dm.add_memory(
            query=f"记忆{i}",
            content=f"记忆内容_{i}",
            fingerprint=f"记忆{i}",
        )
    stats = dm.get_stats()
    print(f"    短期: {stats['short_term']['total_memories']} 条")
    print(f"    长期: {stats['long_term']['total_memories']} 条")
    short_contents = [m["content"] for m in dm.short_term.memories]
    print(f"    短期内容: {short_contents}")
    assert "记忆内容_0" not in short_contents, "记忆0 应已被短期遗忘"

    # 6. 验证巩固后的记忆仍可从长期召回
    print(f"\n[6] 验证双层召回（短期遗忘后仍能从长期定位）...")
    r_merged = dm.recall("记忆0")
    print(f"    查询「记忆0」召回: {r_merged}")
    assert any("记忆内容_0" in m for m in r_merged), f"应能召回记忆内容_0: {r_merged}"
    print("    ✓ 长期巩固后仍能召回")

    # 7. 验证短期记忆单独召回
    r_short = dm.recall("记忆5")
    print(f"\n[7] 查询仅短期存在的「记忆5」: {r_short}")
    assert any("记忆内容_5" in m for m in r_short), f"短期召回失败: {r_short}"
    print("    ✓ 短期独立召回正确")

    # 8. 打印完整统计
    print(f"\n[8] 最终统计:")
    stats = dm.get_stats()
    print(f"    {json.dumps(stats, ensure_ascii=False, indent=4)}")
    print(f"    短期记忆列表:")
    for m in dm.short_term.memories:
        print(f"      └ {m['content']}")
    print(f"    长期记忆列表:")
    for m in dm.long_term.memories:
        print(f"      └ {m['content']}")

    print(f"\n{'='*60}")
    print("Demo 完成")
    print("=" * 60)
