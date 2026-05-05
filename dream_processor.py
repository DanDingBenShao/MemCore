"""
dream_processor.py — MemCore v0.6 做梦进程

模拟人类睡眠时海马体→新皮层的记忆巩固机制：

  Phase1（无 LLM）：规则引擎，按时间+关键词分组
  Phase2（有 LLM）：通过 Ollama 调用 LLM，精确识别事件边界

架构：
  - DreamScheduler: 触发策略 + 异步执行 + 批量处理
  - RuleBasedDream: Phase1 规则引擎
  - DreamWorker: Phase2 LLM 接口
"""
from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import requests

from event_index import (
    EVENT_CLOSE_AFTER_SECONDS,
    EVENT_MAX_FACTS,
    EVENT_MIN_FACTS,
    AtomicFact,
    EventBlock,
    EventIndex,
)
from memory_core import MemoryCore


# ===========================================================================
# 常量
# ===========================================================================

DREAM_MIN_NEW_MEMORIES = 15        # 积累至少 15 条新记忆
DREAM_IDLE_SECONDS = 60            # 空闲至少 60 秒
DREAM_MIN_INTERVAL = 180           # 两次做梦间隔至少 180 秒
SAME_EVENT_TIME_GAP = 300          # 5 分钟内为候选
FINGERPRINT_OVERLAP_THRESHOLD = 0.25
ORPHAN_EVENT_THEME = "__孤立事实__"
DREAM_BATCH_SIZE = 20              # 每批最多 20 条记忆给 LLM
DREAM_PROCESSED_SOURCE_HINT_LEN = 30  # 溯源片段长度

# Ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3.5:4b"
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 2


# ===========================================================================
# Phase1: 规则引擎
# ===========================================================================

class RuleBasedDream:
    """Phase1 规则引擎。"""

    def __init__(self, event_index: EventIndex) -> None:
        self.event_index = event_index

    def dream(self, short_term: MemoryCore, long_term: MemoryCore, since_timestamp: float = 0.0) -> int:
        raw_entries = self._collect_raw_entries(short_term, long_term, since_timestamp)
        if not raw_entries:
            return 0
        groups = self._group_by_rule(raw_entries)
        events_created = 0
        for group in groups:
            if len(group) < EVENT_MIN_FACTS:
                continue
            if len(group) > EVENT_MAX_FACTS:
                for g in self._split_oversized_group(group):
                    if len(g) >= EVENT_MIN_FACTS:
                        self._create_event_from_group(g)
                        events_created += 1
            else:
                self._create_event_from_group(group)
                events_created += 1
        return events_created

    def _collect_raw_entries(
        self, short_term: MemoryCore, long_term: MemoryCore,
        since_timestamp: float = 0.0,
    ) -> list[dict]:
        """收集原始记忆，#1 增量 + #3 增量过滤。"""
        entries: list[dict] = []
        now = time.time()
        for i, mem in enumerate(short_term.memories):
            ts = short_term.timestamps[i] if i < len(short_term.timestamps) else now
            if ts <= since_timestamp:
                continue
            db_id = mem.get("db_id")
            entries.append({
                "query": mem.get("query", ""),
                "content": mem.get("content", ""),
                "fingerprint": mem.get("fingerprint", ""),
                "source": ("short", db_id) if db_id is not None else None,
                "timestamp": ts,
            })
        if len(entries) < DREAM_MIN_NEW_MEMORIES:
            for i, mem in enumerate(long_term.memories):
                ts = long_term.timestamps[i] if i < len(long_term.timestamps) else now
                if ts <= since_timestamp:
                    continue
                db_id = mem.get("db_id")
                entries.append({
                    "query": mem.get("query", ""),
                    "content": mem.get("content", ""),
                    "fingerprint": mem.get("fingerprint", ""),
                    "source": ("long", db_id) if db_id is not None else None,
                    "timestamp": ts,
                })
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        return entries[:100]

    def _group_by_rule(self, entries: list[dict]) -> list[list[dict]]:
        if not entries:
            return []
        groups: list[list[dict]] = [[entries[0]]]
        for entry in entries[1:]:
            last_entry = groups[-1][0]
            if last_entry["timestamp"] - entry["timestamp"] > SAME_EVENT_TIME_GAP:
                groups.append([entry])
            elif self._fingerprint_overlap_ok(last_entry, entry):
                groups[-1].append(entry)
            else:
                groups.append([entry])
        return groups

    @staticmethod
    def _fingerprint_overlap_ok(a: dict, b: dict) -> bool:
        fp_a = set(a.get("fingerprint", "").split())
        fp_b = set(b.get("fingerprint", "").split())
        if not fp_a or not fp_b:
            q_a = set(a.get("query", "").split())
            q_b = set(b.get("query", "").split())
            return len(q_a & q_b) / max(len(q_a | q_b), 1) >= FINGERPRINT_OVERLAP_THRESHOLD if q_a and q_b else False
        return len(fp_a & fp_b) / max(len(fp_a | fp_b), 1) >= FINGERPRINT_OVERLAP_THRESHOLD

    @staticmethod
    def _split_oversized_group(group: list[dict]) -> list[list[dict]]:
        n = len(group)
        parts = (n + EVENT_MAX_FACTS - 1) // EVENT_MAX_FACTS
        chunk = (n + parts - 1) // parts
        return [group[i:i + chunk] for i in range(0, n, chunk)]

    def _create_event_from_group(self, group: list[dict]) -> None:
        oldest = min(group, key=lambda e: e["timestamp"])
        all_queries = [e.get("query", "") for e in group]
        tokens = []
        for q in all_queries:
            tokens.extend(q.split())
        token_counts = Counter(tokens)
        theme = " ".join(t for t, _ in token_counts.most_common(3)) or group[0].get("query", "未分类事件")
        fp_set: set[str] = set()
        for e in group:
            fp_set |= set(e.get("fingerprint", "").split())
        facts = [AtomicFact(query=e.get("query",""), content=e.get("content",""),
                 fingerprint=e.get("fingerprint",""),
                 source_dbs=[(e["source"][0]+".db", e["source"][1])] if e.get("source") and e["source"][1] else [])
                 for e in group]
        seen: set[str] = set()
        deduped = []
        for f in facts:
            if f.content not in seen:
                seen.add(f.content)
                deduped.append(f)
        self.event_index.create_event(theme=theme, timestamp=oldest["timestamp"],
            facts=deduped, fingerprint=" ".join(sorted(fp_set)) if fp_set else "",
            status="closed")


# ===========================================================================
# Phase2: LLM 做梦
# ===========================================================================

class DreamWorker:
    """Phase2 LLM 做梦——通过 Ollama 调用本地 LLM。"""

    def __init__(self, event_index: EventIndex) -> None:
        self.event_index = event_index
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def call_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                resp = self.session.post(OLLAMA_API_URL, json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1024},
                }, timeout=LLM_TIMEOUT)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()
            except Exception:
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(1)
        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 回复中提取 JSON 字符串。"""
        if "```json" in text:
            return text.split("```json")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()

    EVENT_BOUNDARY_PROMPT = """你是一个记忆整理助手。将以下记忆片段按话题分组为"事件"。

每组事件的特征：同一技术话题、同一设计流程、讨论同一项配置或bug。
不同事件的判断：话题明显不同、讨论对象不同。

约束：每事件3-50条；超过50条拆分为子事件；少于3条合并到最近事件或作孤立标记。

输出JSON: {"events":[{"event_theme":"主题","memory_indices":[0,1,2]},...]}
不要多余的解释。"""

    def identify_event_boundaries(self, raw_entries: list[dict]) -> list[list[dict]] | None:
        lines = [f"[{i}] {e.get('query','')} | {e.get('content','')[:120]} | {e.get('fingerprint','')}"
                 for i, e in enumerate(raw_entries)]
        user_prompt = "记忆片段：\n" + "\n".join(lines)
        result = self.call_llm(self.EVENT_BOUNDARY_PROMPT, user_prompt)
        if not result:
            return None
        try:
            json_str = self._extract_json(result)
            data = json.loads(json_str)
            events = data.get("events", [])
        except Exception:
            return None
        groups, seen = [], set()
        for evt in events:
            indices = [i for i in evt.get("memory_indices", [])
                       if 0 <= i < len(raw_entries) and i not in seen]
            if not indices:
                continue
            seen.update(indices)
            group = [raw_entries[i] for i in indices]
            if len(group) >= 3:
                groups.append(group)
        orphans = set(range(len(raw_entries))) - seen
        if len(orphans) >= 3:
            groups.append([raw_entries[i] for i in sorted(orphans)])
        return groups or None

    THEME_PROMPT = "用10-20字总结该事件的核心主题，包含关键技术术语，不要多余文字。"

    def generate_event_theme(self, entries: list[dict]) -> str | None:
        lines = [f"{e.get('query','')} | {e.get('content','')[:100]}" for e in entries]
        return self.call_llm(self.THEME_PROMPT, "以下属于同一事件，请总结主题：\n" + "\n".join(lines))

    REFINE_PROMPT = """你是一个知识精炼助手。给定原始记忆片段，请：1)去重 2)合并描述同一件事的记忆 3)保留技术细节。

输出JSON: [{"query":"主题","content":"精炼事实(≤80字)","fingerprint":"关键词1 关键词2 关键词3"}]
每事件输出3-15条精炼事实。"""

    def refine_facts(self, entries: list[dict]) -> list[dict] | None:
        lines = [f"[{i}] {e.get('query','')} | {e.get('content','')} | {e.get('fingerprint','')}"
                 for i, e in enumerate(entries)]
        result = self.call_llm(self.REFINE_PROMPT, "请精炼合并：\n" + "\n".join(lines))
        if not result:
            return None
        try:
            refined = json.loads(self._extract_json(result))
            if isinstance(refined, list):
                # #2 溯源：每条精炼事实加 source_hint
                for fact in refined:
                    source = next((e.get("content", "")[:DREAM_PROCESSED_SOURCE_HINT_LEN]
                                   for e in entries), "")
                    fact["source_hint"] = source
                return refined
        except Exception:
            pass
        return None

    APPEND_PROMPT = """给定已有事件的主题和新记忆，判断这些新记忆是否应归入该事件。

已有事件主题：{theme}

输出JSON: {"should_append": true/false, "reason": "一句话原因"}"""

    def should_append_to_event(self, theme: str, entries: list[dict]) -> bool:
        """#4 事件演化：判断新记忆是否应追加到已有事件。"""
        lines = [f"{e.get('query','')} | {e.get('content','')[:100]}" for e in entries[:5]]
        user_prompt = "新记忆：\n" + "\n".join(lines)
        result = self.call_llm(self.APPEND_PROMPT.format(theme=theme), user_prompt)
        if not result:
            return False
        try:
            return json.loads(self._extract_json(result)).get("should_append", False)
        except Exception:
            return False

    DOC_DETECT_PROMPT = """判断以下内容是否为一份完整的设计文档/蓝图/系统架构方案。

特征：
- 包含多个章节或组成部分（如"API层"、"数据库"、"架构"等）
- 描述的是一个完整的系统、方案或产品设计
- 长度超过200字

输出JSON: {"is_document": true/false, "doc_theme": "文档主题摘要（15字内）"}"""

    def detect_document(self, text: str) -> tuple[bool, str]:
        """检测是否为完整文档/蓝图，返回 (is_doc, theme)。"""
        if len(text) < 200:
            return False, ""
        result = self.call_llm(self.DOC_DETECT_PROMPT, "内容：\n" + text[:2000])
        if not result:
            return False, ""
        try:
            data = json.loads(self._extract_json(result))
            return bool(data.get("is_document", False)), data.get("doc_theme", "")
        except Exception:
            return False, ""


# ===========================================================================
# 做梦调度器
# ===========================================================================

class DreamScheduler:
    """做梦调度器：触发策略 + 异步执行 + 分批处理。"""

    def __init__(
        self,
        event_index: EventIndex,
        idle_threshold: int = DREAM_IDLE_SECONDS,
        min_interval: int = DREAM_MIN_INTERVAL,
        min_new_memories: int = DREAM_MIN_NEW_MEMORIES,
    ) -> None:
        self.event_index = event_index
        self.idle_threshold = idle_threshold
        self.min_interval = min_interval
        self.min_new_memories = min_new_memories

        self.last_dream_time: float = 0.0
        self.last_operation_time: float = time.time()
        self.new_memories_since_last_dream: int = 0
        self.total_dreams: int = 0
        self.total_events_created: int = 0

        # #1 增量：记录上次做梦的时间戳用于过滤
        self.last_dream_timestamp: float = 0.0

        # #5 异步：防并发锁
        self._dream_lock = threading.Lock()
        self.is_dreaming: bool = False

        self.rule_dream = RuleBasedDream(event_index)
        self.llm_worker = DreamWorker(event_index)
        self.use_llm = False

    def record_operation(self) -> None:
        self.last_operation_time = time.time()

    def record_new_memory(self) -> None:
        self.new_memories_since_last_dream += 1
        self.last_operation_time = time.time()

    def should_dream(self) -> bool:
        if self.new_memories_since_last_dream < self.min_new_memories:
            return False
        now = time.time()
        if now - self.last_dream_time < self.min_interval:
            return False
        if now - self.last_operation_time < self.idle_threshold:
            return False
        return True

    def try_dream(self, short_term, long_term) -> int:
        """#5 异步：检测条件后启动线程立即返回，不阻塞调用方。"""
        with self._dream_lock:
            if not self.should_dream() or self.is_dreaming:
                return 0
            self.last_dream_time = time.time()
            self.new_memories_since_last_dream = 0
            self.is_dreaming = True

        thread = threading.Thread(
            target=self._dream_worker,
            args=(short_term, long_term),
            daemon=True,
        )
        thread.start()
        return 0

    def _dream_worker(self, short_term, long_term) -> None:
        """实际做梦逻辑，在后台线程中执行。"""
        try:
            # #3 增量：只取上次做梦后的新记忆
            raw_entries = self.rule_dream._collect_raw_entries(
                short_term, long_term, since_timestamp=self.last_dream_timestamp,
            )
            self.last_dream_timestamp = time.time()

            if not raw_entries:
                return

            events_created = 0

            if self.use_llm:
                # #6 分批：LLM 分批处理
                events_created = self._llm_dream_batched(raw_entries)
            else:
                events_created = self.rule_dream.dream(
                    short_term, long_term, since_timestamp=self.last_dream_timestamp)

            self.total_dreams += 1
            self.total_events_created += events_created

            # 做梦后自动清理过期开放事件
            try:
                closed = self.event_index.auto_close_stale_events()
                if closed > 0:
                    self.total_events_created += 0  # 不计入创建数，仅日志
            except Exception:
                pass

        finally:
            self.is_dreaming = False

    def _llm_dream_batched(self, raw_entries: list[dict]) -> int:
        """#6 分批：将原始记忆分批送 LLM，防止上下文溢出。"""
        events_created = 0
        # 提取 embedder 到循环外，避免重复创建（BGE 模型加载开销大）
        encoder = None
        try:
            from embedder import create_embedder
            encoder = create_embedder()
        except Exception:
            encoder = None

        for start in range(0, len(raw_entries), DREAM_BATCH_SIZE):
            batch = raw_entries[start:start + DREAM_BATCH_SIZE]

            # #4 事件演化：先查是否有开放事件可以追加
            appended = False
            for evt in self.event_index.event_list:
                if evt.status != "open":
                    continue
                if self.llm_worker.should_append_to_event(evt.theme, batch):
                    # 追加到已有事件
                    for entry in batch:
                        fact = AtomicFact(
                            query=entry.get("query", ""),
                            content=entry.get("content", ""),
                            fingerprint=entry.get("fingerprint", ""),
                            source_dbs=[(entry["source"][0]+".db", entry["source"][1])]
                            if entry.get("source") and entry["source"][1] else [],
                        )
                        vec = None
                        if encoder is not None:
                            try:
                                vec = encoder.encode(fact.content).astype("float32")
                            except Exception:
                                vec = None
                        if vec is not None:
                            self.event_index.add_to_event(evt.event_id, fact, vec)
                    appended = True
                    events_created += 1
                    break

            if appended:
                continue

            # 无开放事件匹配→尝试重开已关闭事件
            if len(self.event_index.event_list) > 0:
                batch_text = " ".join(e.get("content", "")[:100] for e in batch[:3])
                reopened = self.event_index.reopen_closed_event_for(batch_text)
                if reopened:
                    # 追加到重开的事件
                    for entry in batch:
                        fact = AtomicFact(query=entry.get("query",""), content=entry.get("content",""),
                            fingerprint=entry.get("fingerprint",""),
                            source_dbs=[(entry["source"][0]+".db", entry["source"][1])] if entry.get("source") and entry["source"][1] else [])
                        vec = None
                        if encoder is not None:
                            try:
                                vec = encoder.encode(fact.content).astype("float32")
                            except Exception:
                                vec = None
                        if vec is not None:
                            self.event_index.add_to_event(reopened.event_id, fact, vec)
                    events_created += 1
                    continue

            # LLM 识别事件边界
            groups = self.llm_worker.identify_event_boundaries(batch)
            if not groups:
                continue

            for group in groups:
                # 检测是否为文档/蓝图
                group_text = " ".join(e.get("content", "") for e in group)
                is_doc, doc_theme = self.llm_worker.detect_document(group_text)

                if is_doc:
                    # 文档模式：整份存入，不精炼不切分
                    theme = doc_theme or "文档蓝图"
                    doc_fact = AtomicFact(
                        query=theme,
                        content=group_text,
                        fingerprint="文档 蓝图 设计 架构",
                        source_dbs=[],
                    )
                    self.event_index.create_event(
                        theme=theme,
                        timestamp=min(e["timestamp"] for e in group),
                        facts=[doc_fact],
                        fingerprint="文档 蓝图 设计 架构",
                        status="closed",
                        doc_mode=True,
                    )
                    events_created += 1
                else:
                    # 普通事件：精炼后存入
                    theme = self.llm_worker.generate_event_theme(group)
                    if not theme:
                        theme = group[0].get("query", "未分类事件")

                    refined = self.llm_worker.refine_facts(group)
                    if not refined:
                        continue

                    # #1 幻觉校验：精炼事实与原始记忆的 BGE 余弦 < 0.5 则丢弃
                    validated = []
                    for f in refined:
                        fc = f.get("content", "")
                        if not fc:
                            continue
                        # 用 source_hint 或任意一条来源记忆校验
                        source = f.get("source_hint", "")
                        if source and encoder is not None:
                            try:
                                v1 = encoder.encode(fc).astype("float32")
                                v2 = encoder.encode(source).astype("float32")
                                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
                                if cos < 0.5:
                                    continue  # 幻觉，丢弃
                            except Exception:
                                pass
                        validated.append(f)

                    if not validated:
                        continue

                    self.event_index.create_event(
                        theme=theme,
                        timestamp=min(e["timestamp"] for e in group),
                        facts=[AtomicFact.from_dict(f) for f in validated],
                        fingerprint=" ".join(set(f.get("fingerprint", "") for f in refined)),
                        status="closed",
                    )
                    events_created += 1

        return events_created

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_dreams": self.total_dreams,
            "total_events_created": self.total_events_created,
            "new_memories_pending": self.new_memories_since_last_dream,
            "last_dream_time": self.last_dream_time,
            "is_idle": (time.time() - self.last_operation_time) > self.idle_threshold,
            "is_dreaming": self.is_dreaming,
        }
