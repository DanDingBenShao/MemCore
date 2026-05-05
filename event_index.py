"""
event_index.py — MemCore v0.6 事件索引系统

三层漏斗检索架构：
  Layer 1: 时间分区路由 → 缩小到时间相关的事件桶
  Layer 2: 事件向量匹配 → 找到语义最相关的事件
  Layer 3: 事件内精排   → 在事件内 3-15 条知识中做最终排序

事件结构：
  - theme: 事件主题摘要（Phase1 用 query 拼接，Phase2 由 LLM 生成）
  - facts: 该事件的原子知识列表（3-15 条）
  - vector: 事件级 BGE 向量（theme + 关键内容拼接编码）
  - timestamp + time_bucket: 时间分区定位

依赖：memory_core（用于事件内精排），embedder（BGE 向量）
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

import db
from embedder import create_embedder, get_embedder_info
from memory_core import MemoryCore


# ===========================================================================
# 常量
# ===========================================================================

# 事件内知识条数约束
# 调大无副作用：事件向量只取前5条拼接编码，精排用事件向量非逐条
# 唯一风险是开放事件变垃圾桶 → 缩短自动关闭时间应对
EVENT_MIN_FACTS = 3
EVENT_MAX_FACTS = 50

# 时间分区粒度（秒）
DAY_SECONDS = 86400
WEEK_SECONDS = 7 * DAY_SECONDS
HOUR_SECONDS = 3600

# 事件向量编码长度（拼接主题 + 关键内容前 N 字）
EVENT_ENCODE_CONTENT_PREFIX = 200  # 取前 200 字拼接编码

# 事件重开阈值：新记忆与已有事件的 BGE 余弦 > 此值则追加
EVENT_REOPEN_THRESHOLD = 0.70

# 事件自动关闭：超过此秒数无更新则标记 closed
# 注意：是用 last_updated 判断（最近一次追加的时间），不是创建时间
# 每天都有新知识的项目事件永远不会自动关闭
EVENT_CLOSE_AFTER_SECONDS = 2 * DAY_SECONDS  # 连续2天无更新则关

# 事件链展开最大深度
MAX_EVENT_CHAIN_DEPTH = 2


# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass
class AtomicFact:
    """事件内的一条原子知识。"""
    query: str
    content: str
    fingerprint: str
    source_dbs: list[tuple[str, int]] = field(default_factory=list)
    # source_dbs = [("short.db", db_id), ...] 溯源引用

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "content": self.content,
            "fingerprint": self.fingerprint,
            "source_dbs": self.source_dbs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AtomicFact:
        return cls(
            query=d.get("query", ""),
            content=d.get("content", ""),
            fingerprint=d.get("fingerprint", ""),
            source_dbs=d.get("source_dbs", []),
        )


@dataclass
class EventBlock:
    """事件块：同一话题的知识聚合单元。"""
    event_id: str
    theme: str
    timestamp: float
    status: str = "closed"           # "open" | "closed"
    facts: list[AtomicFact] = field(default_factory=list)
    fingerprint: str = ""            # 合并后的关键词串
    vector: Optional[np.ndarray] = None  # BGE 事件向量 (1024 维)
    access_count: int = 0
    last_accessed: float = 0.0
    db_id: Optional[int] = None      # events 表的自增 ID
    parent_event_id: Optional[str] = None  # 父事件ID，用于事件链
    chain_depth: int = 0             # 事件链深度，根事件=0
    last_updated: float = 0.0        # 最近一次追加时间
    doc_mode: bool = False            # True=整份文档模式，不精炼不切分

    def add_fact(self, fact: AtomicFact, fact_vector: np.ndarray) -> None:
        """添加一条知识到事件，增量更新事件向量。"""
        self.facts.append(fact)
        self.last_updated = time.time()
        # 增量更新 fingerprint
        existing_tokens = set(self.fingerprint.split()) if self.fingerprint else set()
        new_tokens = set(fact.fingerprint.split())
        combined = existing_tokens | new_tokens
        self.fingerprint = " ".join(sorted(combined)) if combined else ""

        # 增量更新事件向量（均值聚合）
        if self.vector is None:
            self.vector = fact_vector.copy()
        else:
            n = len(self.facts)
            self.vector = self.vector * (n - 1) / n + fact_vector / n

    def recall_within(
        self,
        query_vector: np.ndarray,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """事件内精排：余弦相似度排序后返回 (content, score)。"""
        if not self.facts or self.vector is None:
            return []

        q_norm = float(np.linalg.norm(query_vector))
        if q_norm < 1e-10:
            return []

        scored: list[tuple[str, float]] = []
        if self.doc_mode:
            # 文档模式：直接返回完整原文，不精排不切分
            full_text = "\n".join(f.content for f in self.facts if f.content.strip())
            if full_text:
                return [(full_text, 1.0)]

        for fact in self.facts:
            # 事件内知识没有单独存储向量，直接用事件向量做降级评分
            # Phase2 后可为每条事件内知识单独存储向量
            sim = float(np.dot(self.vector, query_vector) / (
                np.linalg.norm(self.vector) * q_norm
            ))
            scored.append((fact.content, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored[:top_k]]

    def to_event_text(self) -> str:
        """用于 BGE 编码的拼接文本。"""
        if self.doc_mode:
            # 文档模式：用整份文档的前 2000 字编码
            full = " ".join(f.content for f in self.facts if f.content.strip())
            return self.theme + "。" + full[:2000]
        contents = [f.content[:100] for f in self.facts[:5]]
        return self.theme + "。" + "。".join(contents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "theme": self.theme,
            "timestamp": self.timestamp,
            "status": self.status,
            "facts": [f.to_dict() for f in self.facts],
            "fingerprint": self.fingerprint,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "parent_event_id": self.parent_event_id,
            "doc_mode": self.doc_mode,
        }

    def should_reopen(self, new_fingerprint: str, threshold: float = EVENT_REOPEN_THRESHOLD) -> bool:
        """判断新记忆是否应归入此事件（基于 BGE 余弦）。
        Phase2 由 LLM 替代此判断。"""
        if self.vector is None:
            return False
        new_vec = self._encode_text(f"{self.theme} {new_fingerprint}")
        sim = float(np.dot(self.vector, new_vec) / (
            np.linalg.norm(self.vector) * np.linalg.norm(new_vec)
        ))
        return sim > threshold

    @staticmethod
    def _encode_text(text: str) -> np.ndarray:
        encoder = create_embedder()
        return encoder.encode(text).astype(np.float32)


# ===========================================================================
# 时间分区索引
# ===========================================================================

class TimeBucketIndex:
    """轻量级时间分区索引：支持按日/周/自定义粒度过滤事件。"""

    def __init__(self) -> None:
        # "2026-05-04" → [event_id, ...] （按时间顺序）
        self.daily: dict[str, list[str]] = {}
        # "2026-W19" → [event_id, ...]
        self.weekly: dict[str, list[str]] = {}

    def add_event(self, event: EventBlock) -> None:
        """将事件加入所有适用的时间桶。"""
        day_key = self._day_key(event.timestamp)
        self.daily.setdefault(day_key, []).append(event.event_id)

        week_key = self._week_key(event.timestamp)
        self.weekly.setdefault(week_key, []).append(event.event_id)

    def remove_event(self, event: EventBlock) -> None:
        """从时间桶移除事件。"""
        day_key = self._day_key(event.timestamp)
        if day_key in self.daily:
            try:
                self.daily[day_key].remove(event.event_id)
            except ValueError:
                pass
            if not self.daily[day_key]:
                del self.daily[day_key]

        week_key = self._week_key(event.timestamp)
        if week_key in self.weekly:
            try:
                self.weekly[week_key].remove(event.event_id)
            except ValueError:
                pass
            if not self.weekly[week_key]:
                del self.weekly[week_key]

    def query_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[str]:
        """返回时间范围内所有事件 ID（按天合并）。"""
        event_ids: list[str] = []
        day = self._day_key(start_time)
        end_day = self._day_key(end_time)
        while day <= end_day:
            ids = self.daily.get(day, [])
            event_ids.extend(ids)
            day = self._next_day(day)
        return event_ids

    def get_recent_days(self, days: int = 3) -> list[str]:
        """返回最近 N 天的所有事件 ID，用于时间路由默认回退。"""
        now = time.time()
        start = now - days * DAY_SECONDS
        return self.query_time_range(start, now)

    def resolve_time_hint(self, query_text: str) -> Optional[list[str]]:
        """从查询文本中解析时间暗示，返回匹配的事件 ID 列表。

        支持的暗示：
          - "今天"  → 今天桶
          - "昨天"  → 昨天桶
          - "这周"  → 本周桶
          - "最近"  → 近 3 天
          - 无暗示  → None（调用方决定默认范围）
        """
        now = time.time()
        today_key = self._day_key(now)
        yesterday_key = self._day_key(now - DAY_SECONDS)
        this_week_key = self._week_key(now)

        if "今天" in query_text:
            return self.daily.get(today_key, [])
        elif "昨天" in query_text:
            return self.daily.get(yesterday_key, [])
        elif "这周" in query_text or "本周" in query_text:
            return self.weekly.get(this_week_key, [])
        elif "最近" in query_text:
            return self.get_recent_days(3)
        return None

    @staticmethod
    def _day_key(ts: float) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(ts))

    @staticmethod
    def _week_key(ts: float) -> str:
        return time.strftime("%Y-W%W", time.localtime(ts))

    @staticmethod
    def _next_day(day_key: str) -> str:
        """返回 day_key 的下一天。day_key 格式 '2026-05-04'。"""
        t = time.mktime(time.strptime(day_key, "%Y-%m-%d"))
        return time.strftime("%Y-%m-%d", time.localtime(t + DAY_SECONDS))


# ===========================================================================
# 事件索引管理器
# ===========================================================================

class EventIndex:
    """事件索引管理器：存储、检索、持久化事件块。

    Phase1 架构（无 LLM）：
      - 事件由 dream_processor 按规则引擎创建
      - theme 用 query 词频统计
      - facts 直接来自原始记忆（无精炼）

    Phase2 架构（有 LLM）：
      - 事件由 dream_processor 的 LLM 做梦进程创建
      - theme 由 LLM 生成摘要
      - facts 经过 LLM 去重合并精炼
    """

    def __init__(
        self,
        db_dir: str = "./memory_db",
        top_k: int = 5,
        recall_merge_max: int = 5,
    ) -> None:
        self.db_dir = db_dir
        self.top_k = top_k
        self.recall_merge_max = recall_merge_max

        # 事件内存存储
        self.events: dict[str, EventBlock] = {}  # event_id → EventBlock
        self.event_list: list[EventBlock] = []    # 保持插入顺序

        # 事件向量矩阵（用于 Layer2 批量余弦匹配）
        self.event_vectors: np.ndarray = np.zeros((0, self._get_dim()), dtype=np.float32)

        # 时间分区索引
        self.time_buckets = TimeBucketIndex()

        # 数据库连接
        self.conn = db.init_event_db(db_dir)

        # 加载已有事件
        self._load_from_db()

    @staticmethod
    def _get_dim() -> int:
        info = get_embedder_info()
        return info.get("dimension", 1024)

    # ------------------------------------------------------------------
    # 事件生命周期
    # ------------------------------------------------------------------

    def create_event(
        self,
        theme: str,
        timestamp: float,
        facts: list[AtomicFact],
        fingerprint: str = "",
        status: str = "open",
        parent_event_id: str | None = None,
        doc_mode: bool = False,
    ) -> EventBlock:
        """创建新事件并加入索引。

        Parameters
        ----------
        parent_event_id : str | None
            父事件ID，用于事件链（检索时展开深度 ≤ MAX_EVENT_CHAIN_DEPTH）。
        doc_mode : bool, default False
            文档模式：True 时不精炼不切分，存储全文。适合蓝图/设计文档。
        """
        event_id = f"evt_{time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))}_{uuid.uuid4().hex[:4]}"

        # 计算链深度
        chain_depth = 0
        if parent_event_id and parent_event_id in self.events:
            parent = self.events[parent_event_id]
            chain_depth = parent.chain_depth + 1

        event = EventBlock(
            event_id=event_id,
            theme=theme,
            timestamp=timestamp,
            status=status,
            facts=facts,
            fingerprint=fingerprint,
            access_count=0,
            last_accessed=0.0,
            last_updated=timestamp,
            parent_event_id=parent_event_id,
            chain_depth=chain_depth,
            doc_mode=doc_mode,
        )

        # 生成事件 BGE 向量
        encoder = create_embedder()
        event_text = event.to_event_text()
        vec = encoder.encode(event_text).astype(np.float32)
        event.vector = vec

        # 写入内存索引
        self.events[event_id] = event
        self.event_list.append(event)
        self.event_vectors = np.vstack([self.event_vectors, vec.reshape(1, -1)])
        self.time_buckets.add_event(event)

        # 写入数据库
        db_id = db.insert_event(
            self.conn,
            event_id=event_id,
            theme=theme,
            timestamp=timestamp,
            status=status,
            facts_json=json.dumps([f.to_dict() for f in facts], ensure_ascii=False),
            vector_blob=vec.tobytes(),
            fingerprint=fingerprint,
            access_count=0,
            last_accessed=0.0,
            parent_event_id=parent_event_id,
            doc_mode=doc_mode,
        )
        event.db_id = db_id

        return event

    def add_to_event(
        self,
        event_id: str,
        fact: AtomicFact,
        fact_vector: np.ndarray,
    ) -> bool:
        """向已有事件追加一条知识。返回 True 表示成功追加。"""
        event = self.events.get(event_id)
        if event is None:
            return False

        # 禁止超出最大条目
        if len(event.facts) >= EVENT_MAX_FACTS:
            return False

        old_n = len(event.facts)
        event.add_fact(fact, fact_vector)

        # 仅数据库内容变更时更新
        if old_n != len(event.facts):
            db.update_event_facts(
                self.conn,
                event_id=event_id,
                facts_json=json.dumps([f.to_dict() for f in event.facts], ensure_ascii=False),
                fingerprint=event.fingerprint,
                vector_blob=event.vector.tobytes() if event.vector is not None else None,
                last_updated=event.last_updated,
            )
        return True

    def close_event(self, event_id: str) -> None:
        """关闭事件（标记为 closed，不再接受新知识追加）。"""
        event = self.events.get(event_id)
        if event is None:
            return
        event.status = "closed"
        db.update_event_status(self.conn, event_id=event_id, status="closed")

    def reopen_event(self, event_id: str) -> None:
        """重开事件（标记为 open，允许新知识追加）。"""
        event = self.events.get(event_id)
        if event is None:
            return
        event.status = "open"
        db.update_event_status(self.conn, event_id=event_id, status="open")

    def delete_event(self, event_id: str) -> None:
        """删除事件。"""
        event = self.events.pop(event_id, None)
        if event is None:
            return
        self.event_list = [e for e in self.event_list if e.event_id != event_id]
        self.time_buckets.remove_event(event)
        db.delete_event(self.conn, event_id=event_id)

    # ------------------------------------------------------------------
    # 检索（三层漏斗）
    # ------------------------------------------------------------------

    def recall(self, query_text: str) -> list[str]:
        """三层漏斗检索事件记忆。

        Layer 1: 时间路由 → 缩小搜索范围
        Layer 2: 事件向量匹配 → top_k
        Layer 3: 事件内精排 → 返回最终结果

        如果事件索引为空，返回空列表（调用方回退到原始记忆）。
        """
        if not self.event_list:
            return []

        encoder = create_embedder()
        query_vector = encoder.encode(query_text).astype(np.float32)

        # ── Layer 1: 时间分区路由 ──
        time_hit_ids = self.time_buckets.resolve_time_hint(query_text)
        if time_hit_ids is not None and not time_hit_ids:
            # 时间暗示有明确指向但无结果（如查"今天"但今天无事件）
            return []

        if time_hit_ids is not None:
            # 有时间暗示 → 只在限定事件中搜索
            candidates = [e for e in self.event_list if e.event_id in time_hit_ids]
            if not candidates:
                return []
        elif len(self.event_list) >= 50:
            # 无时间暗示但事件较多 → 默认搜索近 3 天
            recent_ids = self.time_buckets.get_recent_days(3)
            candidates = [e for e in self.event_list if e.event_id in recent_ids]
            if not candidates:
                # 近 3 天无事件，回退到全部
                candidates = self.event_list
        else:
            # 事件较少 → 全量搜索
            candidates = self.event_list

        # ── Layer 2: 事件向量余弦匹配 ──
        q_norm = float(np.linalg.norm(query_vector))
        if q_norm < 1e-10:
            return []

        candidate_vectors = np.array([c.vector for c in candidates if c.vector is not None])
        if len(candidate_vectors) == 0:
            return []

        sims = np.dot(candidate_vectors, query_vector) / (
            np.linalg.norm(candidate_vectors, axis=1) * q_norm
        )
        sims = np.nan_to_num(sims)
        sims = np.maximum(sims, 0.0)

        top_n = min(self.top_k, len(sims))
        if top_n == 0:
            return []

        top_indices = np.argsort(sims)[-top_n:][::-1]

        # ── Layer 3: 事件内精排 ──
        results: list[str] = []
        seen: set[str] = set()

        for idx in top_indices:
            event = candidates[idx]
            event.access_count += 1
            event.last_accessed = time.time()

            # 更新时间分区（access 刷新）
            db.update_event_access(
                self.conn,
                event_id=event.event_id,
                access_count=event.access_count,
                last_accessed=event.last_accessed,
            )

            # 事件内精排
            event_results = event.recall_within(query_vector, top_k=3)
            for r in event_results:
                if r not in seen:
                    results.append(r)
                    seen.add(r)
                    if len(results) >= self.recall_merge_max:
                        break
            if len(results) >= self.recall_merge_max:
                break

        # ── 事件链展开：命中父事件时，展开子事件的知识 ──
        # 展开深度 ≤ MAX_EVENT_CHAIN_DEPTH，防止递归爆炸
        if results and MAX_EVENT_CHAIN_DEPTH > 0:
            self._expand_event_chain(results, seen)

        return results

    def _expand_event_chain(self, results: list[str], seen: set[str]) -> None:
        """展开命中事件的子事件链，追加子事件内的事实到 results。"""
        if not results:
            return
        # 找哪些命中事件有子事件
        hit_event_ids = set()
        for result in results:
            for e in self.event_list:
                for f in e.facts:
                    if f.content == result:
                        hit_event_ids.add(e.event_id)
                        break

        # 展开子事件（深度 ≤ MAX_EVENT_CHAIN_DEPTH）
        for eid in hit_event_ids:
            parent = self.events.get(eid)
            if parent is None:
                continue
            self._collect_child_facts(parent, results, seen, depth=1)

    def _collect_child_facts(
        self,
        parent: EventBlock,
        results: list[str],
        seen: set[str],
        depth: int,
        visited: set[str] | None = None,
    ) -> None:
        """递归收集子事件的知识（visited 防止循环）。"""
        if depth > MAX_EVENT_CHAIN_DEPTH:
            return
        if visited is None:
            visited = {parent.event_id}
        for child in self.event_list:
            if child.parent_event_id != parent.event_id:
                continue
            if child.event_id in visited:
                continue  # 循环检测
            visited.add(child.event_id)
            # 追加子事件的事实
            for f in child.facts:
                if f.content not in seen:
                    results.append(f.content)
                    seen.add(f.content)
                    if len(results) >= self.recall_merge_max * 2:
                        return
            # 递归展开
            self._collect_child_facts(child, results, seen, depth + 1, visited)

    # ------------------------------------------------------------------
    # 管理接口
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """返回事件索引统计信息。"""
        total = len(self.event_list)
        open_count = sum(1 for e in self.event_list if e.status == "open")
        total_facts = sum(len(e.facts) for e in self.event_list)
        bucket_count = len(self.time_buckets.daily)
        doc_mode_count = sum(1 for e in self.event_list if e.doc_mode)
        avg_facts_per_event = total_facts / total if total > 0 else 0
        return {
            "total_events": total,
            "open_events": open_count,
            "doc_mode_events": doc_mode_count,
            "total_facts": total_facts,
            "avg_facts_per_event": round(avg_facts_per_event, 1),
            "time_buckets": bucket_count,
            "event_vector_dim": self.event_vectors.shape[1] if len(self.event_vectors) > 0 else 0,
        }

    def find_best_event_for(
        self,
        fingerprint: str,
        threshold: float = EVENT_REOPEN_THRESHOLD,
    ) -> Optional[EventBlock]:
        """为新记忆找最匹配的开放事件。余弦 > threshold 则命中。"""
        if not self.event_list:
            return None
        encoder = create_embedder()
        vec = encoder.encode(fingerprint).astype(np.float32)
        best_sim = threshold
        best_event = None
        for event in self.event_list:
            if event.status != "open":
                continue
            if event.vector is None:
                continue
            sim = float(np.dot(vec, event.vector) / (
                np.linalg.norm(vec) * np.linalg.norm(event.vector)
            ))
            if sim > best_sim:
                best_sim = sim
                best_event = event
        return best_event

    def reopen_closed_event_for(
        self,
        theme_hint: str,
        threshold: float = 0.75,
    ) -> Optional[EventBlock]:
        """为新内容找最匹配的已关闭事件并重开。

        当没有开放事件匹配新记忆时，检查已关闭事件。
        余弦 > threshold 则自动重开事件。

        Parameters
        ----------
        theme_hint : str
            新内容的主题提示（query 或 fingerprint）。
        threshold : float, default 0.75
            余弦阈值（比开放事件高，避免误重开）。

        Returns
        -------
        EventBlock | None
            如果找到匹配的已关闭事件并重开返回之，否则 None。
        """
        if not self.event_list:
            return None
        encoder = create_embedder()
        vec = encoder.encode(theme_hint).astype(np.float32)
        best_sim = threshold
        best_event = None
        for event in self.event_list:
            if event.status != "closed":
                continue
            if event.vector is None:
                continue
            sim = float(np.dot(vec, event.vector) / (
                np.linalg.norm(vec) * np.linalg.norm(event.vector)
            ))
            if sim > best_sim:
                best_sim = sim
                best_event = event
        if best_event is not None:
            self.reopen_event(best_event.event_id)
        return best_event

    # ------------------------------------------------------------------
    # 数据库持久化
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """从数据库加载所有事件到内存。"""
        rows = db.get_all_events(self.conn)
        for row in rows:
            event_id = row["event_id"]
            theme = row["theme"]
            timestamp = row["timestamp"]
            status = row["status"]
            fingerprint = row.get("fingerprint", "")
            access_count = row.get("access_count", 0)
            last_accessed = row.get("last_accessed", 0.0)
            db_id = row["id"]

            facts_json = row.get("facts_json", "[]")
            try:
                fact_dicts = json.loads(facts_json)
            except (json.JSONDecodeError, TypeError):
                fact_dicts = []
            facts = [AtomicFact.from_dict(fd) for fd in fact_dicts]

            vector_blob = row.get("vector_blob")
            vector = None
            if vector_blob is not None:
                dim = self._get_dim()
                vec = np.frombuffer(vector_blob, dtype=np.float32).copy()
                if len(vec) == dim:
                    vector = vec

            event = EventBlock(
                event_id=event_id,
                theme=theme,
                timestamp=timestamp,
                status=status,
                facts=facts,
                fingerprint=fingerprint,
                vector=vector,
                access_count=access_count,
                last_accessed=last_accessed,
                db_id=db_id,
                parent_event_id=row.get("parent_event_id"),
                last_updated=row.get("last_updated", timestamp),
                doc_mode=bool(row.get("doc_mode", 0)),
            )

            self.events[event_id] = event
            self.event_list.append(event)
            if vector is not None:
                self.event_vectors = np.vstack([self.event_vectors, vector.reshape(1, -1)])
            self.time_buckets.add_event(event)

    def auto_close_stale_events(self) -> int:
        """自动关闭超过 EVENT_CLOSE_AFTER_SECONDS 未更新的开放事件。"""
        now = time.time()
        closed = 0
        for event in self.event_list:
            last_ts = max(event.last_updated, event.timestamp)
            if event.status == "open" and (now - last_ts) > EVENT_CLOSE_AFTER_SECONDS:
                event.status = "closed"
                db.update_event_status(self.conn, event_id=event.event_id, status="closed")
                closed += 1
        return closed
