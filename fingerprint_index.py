"""
fingerprint_index.py — 对话指纹索引 (MemCore v0.4)

内存常驻的对话指纹链，用于：
- 显式指代 O(1) 定位（"第N轮"、"刚才"、"上一轮"）
- 隐式指代语义匹配（"那个报错"、"之前讨论的方案"）
- 第二道实体倒排索引

每条对话轮次对应一个指纹：
    DialogueFingerprint:
        round_id: int
        timestamp: float
        summary_phrase: str       # 3-8 字摘要
        entities: list[str]       # 提到的实体
        memory_fiber_ids: list[int]  # 该轮关联的记忆 ID
        previous_round_id: int    # 上一轮 ID（形成链）

500 轮约 150KB，1000 轮约 300KB。
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional


# ---------------------------------------------------------------------------
# 指纹结构
# ---------------------------------------------------------------------------


class DialogueFingerprint:
    """一条对话轮次的指纹。"""

    __slots__ = (
        "round_id", "timestamp", "summary_phrase",
        "entities", "memory_fiber_ids", "previous_round_id",
    )

    def __init__(
        self,
        round_id: int,
        summary_phrase: str = "",
        entities: Optional[list[str]] = None,
        memory_fiber_ids: Optional[list[int]] = None,
        previous_round_id: int = -1,
    ):
        self.round_id = round_id
        self.timestamp = time.time()
        self.summary_phrase = summary_phrase
        self.entities = entities or []
        self.memory_fiber_ids = memory_fiber_ids or []
        self.previous_round_id = previous_round_id


# ---------------------------------------------------------------------------
# 指纹索引
# ---------------------------------------------------------------------------


class FingerprintIndex:
    """对话指纹索引。

    维护两条数据结构：
    - round_map: round_id → DialogueFingerprint（O(1) 定位）
    - entity_index: entity → set[round_id]（实体倒排）
    """

    def __init__(self):
        self.round_map: dict[int, DialogueFingerprint] = {}
        self.entity_index: dict[str, set[int]] = {}
        self._next_round: int = 0

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_round(
        self,
        summary_phrase: str,
        entities: Optional[list[str]] = None,
        memory_fiber_ids: Optional[list[int]] = None,
    ) -> int:
        """添加一轮新指纹。

        Parameters
        ----------
        summary_phrase : str
            该轮对话的 3-8 字摘要。
        entities : list[str] | None
            该轮提到的实体列表。
        memory_fiber_ids : list[int] | None
            该轮关联的记忆 ID 列表。

        Returns
        -------
        int
            新分配的 round_id。
        """
        round_id = self._next_round
        prev_id = round_id - 1 if round_id > 0 else -1

        fp = DialogueFingerprint(
            round_id=round_id,
            summary_phrase=summary_phrase,
            entities=entities,
            memory_fiber_ids=memory_fiber_ids,
            previous_round_id=prev_id,
        )
        self.round_map[round_id] = fp
        self._next_round += 1

        # 更新实体倒排
        if entities:
            for ent in entities:
                if ent not in self.entity_index:
                    self.entity_index[ent] = set()
                self.entity_index[ent].add(round_id)

        return round_id

    def set_memory_fibers(self, round_id: int, fiber_ids: list[int]) -> None:
        """为指定轮次设置关联的记忆 ID。"""
        if round_id in self.round_map:
            self.round_map[round_id].memory_fiber_ids = fiber_ids

    # ------------------------------------------------------------------
    # 显式指代检索（第一道 a：O(1) 定位）
    # ------------------------------------------------------------------

    def resolve_explicit_reference(self, query: str) -> Optional[int]:
        """解析显式指代，返回匹配的 round_id。

        支持：
        - "第N轮" → round_id = N
        - "上一轮" / "前一轮" → current - 1
        - "刚才" → 最新一轮
        - "最后" → 最新一轮

        Parameters
        ----------
        query : str
            用户查询文本。

        Returns
        -------
        int | None
            匹配的 round_id，未找到时返回 None。
        """
        if not self.round_map:
            return None

        max_round = max(self.round_map.keys())

        # 第N轮（第1轮 → round_id=0）
        m = re.search(r'第(\d+)轮', query)
        if m:
            rid = int(m.group(1)) - 1
            if rid in self.round_map:
                return rid
            return None

        # 上一轮 / 前一轮
        if re.search(r'上一轮|前一轮|前几轮', query):
            return max_round - 1 if max_round > 0 else None

        # 刚才 / 最后
        if re.search(r'刚才|最后|最近', query):
            return max_round

        return None

    # ------------------------------------------------------------------
    # 隐式指代检索（第一道 b：语义匹配）
    # ------------------------------------------------------------------

    def resolve_implicit_reference(self, query: str) -> list[int]:
        """解析隐式指代，返回按相关度排序的 round_id 列表。

        在指纹的 summary_phrase 和 entities 中做关键词匹配。
        支持字符级和子串级匹配以兼容中文。

        Parameters
        ----------
        query : str
            用户查询文本（如"那个报错"、"之前讨论的方案"）。

        Returns
        -------
        list[int]
            按匹配度降序排列的 round_id 列表（最多 3 个）。
        """
        if not self.round_map:
            return []

        # 提取查询中的双字滑动窗口
        chars = list(query)
        query_bigrams = set()
        for i in range(len(chars) - 1):
            bigram = chars[i] + chars[i + 1]
            query_bigrams.add(bigram)

        # 提取英文/数字 token
        query_tokens = set(re.findall(r'[A-Za-z0-9_.]{2,}', query))

        scored: list[tuple[int, int]] = []
        for rid, fp in self.round_map.items():
            score = 0
            fp_text = f"{fp.summary_phrase} {' '.join(fp.entities)}"

            # 双字窗口匹配
            for bg in query_bigrams:
                if bg in fp_text:
                    score += 1

            # 英文 token 匹配
            for t in query_tokens:
                if t.lower() in fp_text.lower():
                    score += 3
                for ent in fp.entities:
                    if t.lower() in ent.lower() or ent.lower() in t.lower():
                        score += 2

            if score > 0:
                scored.append((score, rid))

        scored.sort(reverse=True)
        return [rid for _, rid in scored[:3]]

    # ------------------------------------------------------------------
    # 实体倒排检索（第二道）
    # ------------------------------------------------------------------

    def resolve_entity(self, entity: str) -> list[int]:
        """通过实体倒排索引查找关联的 round_id。

        Parameters
        ----------
        entity : str
            实体名称。

        Returns
        -------
        list[int]
            包含该实体的 round_id 列表（按时间降序）。
        """
        if entity in self.entity_index:
            return sorted(self.entity_index[entity], reverse=True)
        return []

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_fingerprint(self, round_id: int) -> Optional[DialogueFingerprint]:
        """获取指定轮次的指纹。"""
        return self.round_map.get(round_id)

    def get_chain(self, round_id: int) -> list[int]:
        """从指定轮次追溯到第一条，返回完整轮次 ID 链。"""
        chain = []
        current = round_id
        while current in self.round_map and current >= 0:
            chain.append(current)
            current = self.round_map[current].previous_round_id
        return chain

    def size(self) -> int:
        """返回当前指纹数。"""
        return len(self.round_map)

    def estimated_memory_kb(self) -> float:
        """估算内存占用（KB）。"""
        return len(self.round_map) * 0.3  # 约 300 字节/条


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  fingerprint_index.py — 自测 Demo")
    print("=" * 60)

    idx = FingerprintIndex()

    # 1. 添加轮次
    print("\n─── 1. 添加对话轮次 ───")
    idx.add_round("饮食偏好", entities=["香菜", "咖啡"], memory_fiber_ids=[1])
    idx.add_round("384维向量", entities=["384", "向量", "矩阵"], memory_fiber_ids=[2])
    idx.add_round("登录bug修复", entities=["LoginHandler", "bug"], memory_fiber_ids=[3, 4])
    print(f"  共 {idx.size()} 轮")
    assert idx.size() == 3
    print("  ✅ 添加轮次通过")

    # 2. 显式指代定位
    print("\n─── 2. 显式指代 (O(1)) ───")
    assert idx.resolve_explicit_reference("第2轮") == 1
    assert idx.resolve_explicit_reference("上一轮") == 1  # max=2, -1 = 1
    assert idx.resolve_explicit_reference("刚才") == 2
    print("  ✅ O(1) 显式指代通过")

    # 3. 隐式指代
    print("\n─── 3. 隐式指代 ───")
    implicit = idx.resolve_implicit_reference("那个登录报错")
    print(f"  '那个登录报错' → round_ids={implicit}")
    assert 2 in implicit, f"应匹配登录bug轮次: {implicit}"
    print("  ✅ 隐式指代通过")

    # 4. 实体倒排
    print("\n─── 4. 实体倒排 ───")
    entity_rounds = idx.resolve_entity("384")
    print(f"  entity '384' → round_ids={entity_rounds}")
    assert 1 in entity_rounds
    print("  ✅ 实体倒排通过")

    # 5. 链追踪
    print("\n─── 5. 链追踪 ───")
    chain = idx.get_chain(2)
    print(f"  从第2轮追溯: {chain}")
    assert chain == [2, 1, 0]
    print("  ✅ 链追溯通过")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
