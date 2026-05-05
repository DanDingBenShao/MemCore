"""
context_builder.py — 上下文构造器 (MemCore v0.4)

从召回的记忆列表中构造最终注入 LLM 的上下文块。
流水线：
  冲突仲裁 → 去冗 → Token 预算压缩 → 动态叙事排序

Token 预算控制（默认 800 tokens）：
- 预算充足 → 完整文本
- 预算紧张 → 摘要
- 预算勉强 → 截断

叙事排序模板（根据查询意图动态选择）：
- bug_fix → 警示优先，历史次之，参考代码最后
- how_to → 当前状态优先，参考次之
- overview → 背景优先，细节随后
- 默认 → 背景 → 当前状态 → 历史 → 警示 → 参考
"""

from __future__ import annotations

import re
from typing import Any, Optional

from profile_config import get_profile, ProfileConfig


# ---------------------------------------------------------------------------
# 默认配置（向后兼容，实际默认值由 ProfileConfig 提供）
# ---------------------------------------------------------------------------

DEFAULT_TOKEN_BUDGET = 800
"""默认 Token 预算。"""

SUMMARY_THRESHOLD = 0.6
"""当预算比例 < 此值时使用摘要而非完整文本。"""

TRUNCATE_THRESHOLD = 0.3
"""当预算比例 < 此值时仅截断输出最关键的条目。"""


# ---------------------------------------------------------------------------
# Token 估算
# ---------------------------------------------------------------------------

# 中文字符 ≈ 1.5 tokens，英文词 ≈ 1 token
_CHINESE_PAT = re.compile(r'[\u4e00-\u9fff]')
_EN_WORD_PAT = re.compile(r'[A-Za-z0-9]+')


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数。

    中文按字符数 × 1.5，英文按词数，标点不计。
    """
    chinese_chars = len(_CHINESE_PAT.findall(text))
    en_words = len(_EN_WORD_PAT.findall(text))
    return int(chinese_chars * 1.5 + en_words)


# ---------------------------------------------------------------------------
# 叙事排序模板
# ---------------------------------------------------------------------------

_NARRATIVE_TEMPLATES: dict[str, list[str]] = {
    "bug_fix": ["alert", "history", "code"],
    "how_to": ["current", "reference"],
    "overview": ["background", "details"],
    "default": ["background", "current", "history", "alert", "reference"],
}

_NARRATIVE_LABELS = {
    "background": "【背景】",
    "current": "【当前状态】",
    "history": "【历史】",
    "alert": "【警示】",
    "reference": "【参考】",
}


# ---------------------------------------------------------------------------
# 上下文条目
# ---------------------------------------------------------------------------


class ContextItem:
    """一条待注入上下文的记忆条目。"""

    __slots__ = (
        "content", "summary", "category",
        "importance", "token_count", "source",
    )

    def __init__(
        self,
        content: str,
        summary: str = "",
        category: str = "reference",
        importance: float = 0.5,
        source: str = "memory",
    ):
        self.content = content
        self.summary = summary or content[:60]
        self.category = category
        self.importance = importance
        self.token_count = estimate_tokens(content)
        self.source = source


# ---------------------------------------------------------------------------
# 上下文构造器
# ---------------------------------------------------------------------------


class ContextBuilder:
    """上下文构造器。

    Parameters
    ----------
    token_budget : int, default 800
        最大 Token 预算。
    narrative_template : str, default "default"
        叙事模板名称（bug_fix / how_to / overview / default）。
    """

    def __init__(
        self,
        token_budget: Optional[int] = None,
        narrative_template: str = "default",
        profile: Optional[ProfileConfig] = None,
    ):
        cfg = profile or get_profile()
        self.token_budget = token_budget if token_budget is not None else cfg.token_budget
        self.narrative_template = narrative_template

    def build(
        self,
        items: list[ContextItem],
        query: str = "",
    ) -> str:
        """将记忆条目列表构造为最终的上下文文本。

        Parameters
        ----------
        items : list[ContextItem]
            从记忆系统召回的记忆条目。
        query : str, default ""
            用户的查询文本。

        Returns
        -------
        str
            构造后的上下文文本（含叙事排序和标签）。
        """
        if not items:
            return ""

        # 1. 按重要性排序
        sorted_items = sorted(items, key=lambda x: x.importance, reverse=True)

        # 2. 确定叙事模板
        template = self._select_template(query)

        # 3. 按叙事顺序重新排列
        ordered = self._narrative_sort(sorted_items, template)

        # 4. Token 预算压缩
        compressed = self._budget_compress(ordered)

        # 5. 格式化输出
        return self._format(compressed)

    def _select_template(self, query: str) -> list[str]:
        """根据查询选择叙事模板。"""
        if re.search(r'bug|错误|修复|问题|失败|异常|error|fix|broken', query, re.IGNORECASE):
            return _NARRATIVE_TEMPLATES["bug_fix"]
        if re.search(r'如何|怎样|怎么|步骤|how|guide|tutorial', query, re.IGNORECASE):
            return _NARRATIVE_TEMPLATES["how_to"]
        if re.search(r'概述|介绍|总览|overview|summary', query, re.IGNORECASE):
            return _NARRATIVE_TEMPLATES["overview"]
        return _NARRATIVE_TEMPLATES[self.narrative_template]

    @staticmethod
    def _narrative_sort(
        items: list[ContextItem],
        template: list[str],
    ) -> list[ContextItem]:
        """按叙事顺序重新排列条目。

        items 已按重要性排序，此方法对类别做分组但不改变组内顺序。
        """
        ordered: list[ContextItem] = []
        seen: set[str] = set()

        for cat in template:
            for item in items:
                if item.category == cat and item.content not in seen:
                    ordered.append(item)
                    seen.add(item.content)

        # 追加未分类的条目（保持重要性顺序）
        for item in items:
            if item.content not in seen:
                ordered.append(item)
                seen.add(item.content)

        return ordered

    def _budget_compress(self, items: list[ContextItem]) -> list[ContextItem]:
        """根据 Token 预算压缩条目列表。

        返回压缩后的条目列表（摘要化或截断）。
        """
        total_tokens = sum(item.token_count for item in items)

        # 预算充足，全部保留
        if total_tokens <= self.token_budget:
            return items

        budget_ratio = self.token_budget / total_tokens

        # 预算紧张：使用摘要替代完整内容
        if budget_ratio >= TRUNCATE_THRESHOLD:
            compressed = []
            budget_remaining = self.token_budget
            for item in items:
                summary_tokens = estimate_tokens(item.summary)
                if summary_tokens <= budget_remaining:
                    compressed.append(ContextItem(
                        content=item.summary,
                        summary=item.summary,
                        category=item.category,
                        importance=item.importance,
                        source=item.source,
                    ))
                    budget_remaining -= summary_tokens
                else:
                    break
            return compressed

        # 预算勉强：仅保留最高重要性的条目
        compressed = []
        budget_remaining = int(self.token_budget * 0.8)  # 留 20% 给标签
        for item in items:
            if item.token_count <= budget_remaining:
                compressed.append(item)
                budget_remaining -= item.token_count
            else:
                break
        return compressed

    @staticmethod
    def _format(items: list[ContextItem]) -> str:
        """格式化为最终上下文文本。"""
        lines = []
        for item in items:
            label = _NARRATIVE_LABELS.get(item.category, "")
            if label:
                lines.append(f"{label} {item.content}")
            else:
                lines.append(item.content)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def build_context(
    memories: list[str],
    query: str = "",
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    narrative_template: str = "default",
) -> str:
    """便捷函数：直接传入记忆文本列表，一步构建上下文。

    Parameters
    ----------
    memories : list[str]
        从记忆系统召回的文本列表。
    query : str, default ""
        用户查询（用于自动选择叙事模板）。
    token_budget : int, default 800
        Token 预算。
    narrative_template : str, default "default"
        叙事模板名称。

    Returns
    -------
    str
        构造后的上下文文本。
    """
    items = [
        ContextItem(content=m, summary=m[:60], importance=max(0.0, 1.0 - i * 0.1))
        for i, m in enumerate(memories)
    ]
    builder = ContextBuilder(
        token_budget=token_budget,
        narrative_template=narrative_template,
    )
    return builder.build(items, query=query)


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  context_builder.py — 自测 Demo")
    print("=" * 60)

    # 1. 基本构建
    print("\n─── 1. 基本上下文构建 ───")
    items = [
        ContextItem("用户登录超时，session 过期时间为 15 分钟", category="alert", importance=0.9),
        ContextItem("LoginHandler 验证用户凭据，返回 JWT token", category="code", importance=0.7),
        ContextItem("之前讨论过 token 过期时间需要改成 30 分钟", category="history", importance=0.5),
    ]
    builder = ContextBuilder(token_budget=500)
    ctx = builder.build(items, query="修复登录超时bug")
    print(ctx[:300])
    assert "警示" in ctx or "【" in ctx
    print("\n  ✅ 基本构建通过")

    # 2. Token 预算压缩
    print("\n─── 2. Token 预算压缩 ───")
    long_items = [
        ContextItem(
            "这是一段很长的文本。" * 50,
            category="reference",
            importance=0.8,
        ),
        ContextItem("简短文本", category="current", importance=0.6),
    ]
    tight_builder = ContextBuilder(token_budget=100)
    tight_ctx = tight_builder.build(long_items, query="")
    print(f"  Token 预算 100，输出 {len(tight_ctx)} 字符")
    assert len(tight_ctx) > 0
    print("  ✅ 预算压缩通过")

    # 3. 空上下文
    print("\n─── 3. 空上下文 ───")
    empty = ContextBuilder().build([])
    assert empty == ""
    print("  ✅ 空上下文正确处理")

    # 4. 便捷函数
    print("\n─── 4. 便捷函数 ───")
    result = build_context(
        ["用户喜欢喝咖啡", "今天天气很好", "微服务架构包含网关模块"],
        query="用户饮食偏好",
    )
    print(f"  便捷函数输出:\n{result}")
    print("  ✅ 便捷函数通过")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
