"""
followup_navigator.py — 追问导航 (MemCore v0.5)

当召回结果为空时，分析查询与现有记忆模式，生成追问建议。
帮助用户精化查询，缩小或调整记忆检索范围。

策略：
1. 关键词差距分析：查询中的哪些词在记忆指纹中完全不存在
2. 相邻概念推荐：从指纹索引中找到语义最近的轮次，建议用户参考
3. 实体边界提示：检测到潜在实体但未命中时，建议使用精确名称
"""

from __future__ import annotations

import re
from typing import Optional

from fingerprint_index import FingerprintIndex


# ---------------------------------------------------------------------------
# 追问生成
# ---------------------------------------------------------------------------


class FollowupNavigator:
    """追问导航器。

    Parameters
    ----------
    fingerprint_index : FingerprintIndex | None, default None
        对话指纹索引（可选，用于生成更精准的追问）。
    """

    def __init__(self, fingerprint_index: Optional[FingerprintIndex] = None):
        self.fp_index = fingerprint_index

    def analyze(
        self,
        query: str,
        memories: list[str],
        memory_queries: Optional[list[str]] = None,
    ) -> list[str]:
        """分析查询与记忆的差距，生成追问建议。

        Parameters
        ----------
        query : str
            用户的原始查询。
        memories : list[str]
            召回的记忆内容列表（可能为空）。
        memory_queries : list[str] | None, default None
            与 memories 对应的查询文本列表。

        Returns
        -------
        list[str]
            追问建议列表（最多 3 条），空列表表示无需追问。
        """
        suggestions: list[str] = []
        if memories:
            return suggestions  # 有召回结果，不需要追问

        # 提取查询中的名词性关键词
        query_tokens = self._extract_key_terms(query)

        if not query_tokens:
            suggestions.append("请提供更具体的描述，例如涉及的功能、模块或问题现象。")
            return suggestions

        # 1. 关键词差距分析
        if memory_queries:
            unseen = self._find_unseen_terms(query_tokens, memory_queries)
            if unseen:
                terms = "、".join(unseen[:3])
                suggestions.append(
                    f"当前记忆中没有包含「{terms}」的条目。是否需要调整关键词？"
                )

        # 2. 指纹索引推荐
        if self.fp_index and self.fp_index.size() > 0:
            nearest = self.fp_index.resolve_implicit_reference(query)
            if nearest:
                fp = self.fp_index.get_fingerprint(nearest[0])
                if fp and fp.summary_phrase:
                    suggestions.append(
                        f"您是否在查询关于「{fp.summary_phrase}」的信息？"
                    )

        # 3. 实体边界提示
        entity_hints = self._detect_entity_boundary(query)
        if entity_hints:
            suggestions.append(f"检测到可能的实体「{entity_hints}」，请尝试使用完整名称。")

        # 4. 默认兜底
        if not suggestions:
            suggestions.append("未找到匹配的记忆。请尝试换个说法，或提供更多上下文。")

        return suggestions[:3]

    @staticmethod
    def _extract_key_terms(text: str) -> set[str]:
        """提取文本中的关键词（中文词 + 代码标识符）。"""
        terms: set[str] = set()

        # 英文代码标识符：点分或下划线连接（避免匹配普通英文单词）
        for m in re.finditer(
            r'[a-z_][a-zA-Z0-9_]*(\.[a-z_][a-zA-Z0-9_]*)+'
            r'|(?<![A-Za-z0-9])[a-z_][a-zA-Z0-9_]*(?:_+[a-zA-Z0-9_]+)+',
            text,
        ):
            terms.add(m.group())

        # 中文双字及以上
        chars = list(text)
        for i in range(len(chars) - 1):
            if '\u4e00' <= chars[i] <= '\u9fff':
                bigram = chars[i] + chars[i + 1]
                if '\u4e00' <= chars[i + 1] <= '\u9fff':
                    terms.add(bigram)

        return terms

    @staticmethod
    def _find_unseen_terms(
        query_terms: set[str],
        memory_queries: list[str],
    ) -> list[str]:
        """找出查询中有但所有记忆查询中都没有的关键词。"""
        all_mem_text = " ".join(memory_queries)
        unseen = []
        for term in query_terms:
            if term not in all_mem_text:
                unseen.append(term)
        return unseen

    @staticmethod
    def _detect_entity_boundary(text: str) -> str:
        """检测可能被截断或不完整的实体名。"""
        # 检测以 . 结尾的词（可能被截断的代码标识符）
        m = re.search(r'([A-Za-z_][A-Za-z0-9_]{1,})\.$', text)
        if m:
            return m.group(1) + "XXX"
        # 检测孤立的单字符（可能是中文省略）
        m = re.search(r'(?:^|\s)([a-zA-Z])$', text)
        if m:
            return m.group(1)
        return ""


def suggest_followup(
    query: str,
    memories: list[str],
    memory_queries: Optional[list[str]] = None,
    fingerprint_index: Optional[FingerprintIndex] = None,
) -> list[str]:
    """便捷函数：一步完成追问分析。"""
    navigator = FollowupNavigator(fingerprint_index=fingerprint_index)
    return navigator.analyze(query, memories, memory_queries)


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  followup_navigator.py — 自测 Demo")
    print("=" * 60)

    nav = FollowupNavigator()

    # 1. 有召回结果 → 无需追问
    print("\n─── 1. 有召回无需追问 ───")
    sugg = nav.analyze("咖啡偏好", ["用户喜欢喝咖啡"])
    print(f"  建议: {sugg}")
    assert len(sugg) == 0
    print("  ✅ 有结果时不追问")

    # 2. 无召回 → 生成追问
    print("\n─── 2. 无召回生成追问 ───")
    sugg = nav.analyze("LoginHandler 超时bug", [], ["咖啡偏好", "384向量"])
    print(f"  建议: {sugg}")
    assert len(sugg) > 0
    print("  ✅ 无召回时生成追问")

    # 3. 带指纹索引的追问
    print("\n─── 3. 指纹索引辅助追问题 ───")
    from fingerprint_index import FingerprintIndex
    fpidx = FingerprintIndex()
    fpidx.add_round("登录超时修复", entities=["LoginHandler", "超时"], memory_fiber_ids=[5])
    nav2 = FollowupNavigator(fingerprint_index=fpidx)
    sugg2 = nav2.analyze("登录超时", [])
    print(f"  建议: {sugg2}")
    assert len(sugg2) > 0
    print("  ✅ 指纹索引辅助追问通过")

    # 4. 边界
    print("\n─── 4. 边界处理 ───")
    sugg3 = nav.analyze("", [])
    print(f"  空查询: {sugg3}")
    assert len(sugg3) > 0
    print("  ✅ 边界处理通过")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
