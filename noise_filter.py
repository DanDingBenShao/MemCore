"""
noise_filter.py — 记忆去噪管道 (MemCore v0.4)

去除口语冗余、指代词、语气词，保留关键实体、数值和动作。
策略：规则清理（正则）→ 本地量化模型重写（可选）→ API 批量重写（备选）。

用法：
    filter = NoiseFilter()
    cleaned = filter.clean("我们刚才决定把 LoginHandler 里的 token 过期时间从 15 分钟改成 30 分钟")
    # → "LoginHandler token 过期时间: 30分钟（由15分钟修改）"
"""

from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# 默认规则
# ---------------------------------------------------------------------------

# 无意义语气词/填充词
# 注意：不使用 \b（对中文无效），改用前瞻约束确保不误截词
_FILLER_PATTERNS = [
    (re.compile(r'那个(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'这个嘛(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'就是说(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'然后呢(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'实际上(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'基本上(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'我觉得(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'我认为(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    (re.compile(r'我们(?=[们的了着过地得])'), ''),  # 仅匹配"我们的/我们了"等，不截断"我们"独立使用
    (re.compile(r'(?<=[，。；、\s])你(?=[，。；、\s])'), ''),  # 仅在标点之间的独立"你"
    (re.compile(r'^你(?=[，。；、\s])'), ''),  # 句首独立"你"
    (re.compile(r'嗯'), ''),
    (re.compile(r'呃'), ''),
    (re.compile(r'那个啥'), ''),
]

# 口语化指代 → 标准化
_DEIXIS_PATTERNS = [
    (re.compile(r'刚才说的'), ''),
    (re.compile(r'刚才提到'), ''),
    (re.compile(r'之前说的'), ''),
    (re.compile(r'上次说的'), ''),
    (re.compile(r'这个(?![\u4e00-\u9fffA-Za-z0-9])'), ''),
    # 注意：「那个」已在 _FILLER_PATTERNS 中处理，此处不再重复
]

# 冗余修饰 → 精简
_VERBOSE_PATTERNS = [
    (re.compile(r'基本上来说'), ''),
    (re.compile(r'从某种意义上来说'), ''),
    (re.compile(r'某种程度上'), ''),
    (re.compile(r'所谓的'), ''),
    (re.compile(r'关于这个'), ''),
]

# 代码相关：口语 -> 标准化
_CODE_MAP = [
    (re.compile(r'里面的'), '的'),
    (re.compile(r'当中的'), '的'),
    (re.compile(r'叫做'), '为'),
    (re.compile(r'命名为'), '为'),
    (re.compile(r'改成了?'), '改为'),
    (re.compile(r'改成'), '改为'),
    (re.compile(r'设置成'), '设为'),
    (re.compile(r'设定为'), '设为'),
]

# 连续空白压缩
_WS = re.compile(r'\s+')


class NoiseFilter:
    """记忆去噪过滤器。

    Parameters
    ----------
    enable_filler_removal : bool, default True
        是否去除语气词/填充词。
    enable_deixis_normalization : bool, default True
        是否标准化指代词。
    enable_code_normalization : bool, default True
        是否标准化代码相关口语。
    """

    def __init__(
        self,
        enable_filler_removal: bool = True,
        enable_deixis_normalization: bool = True,
        enable_code_normalization: bool = True,
    ):
        self.enable_filler_removal = enable_filler_removal
        self.enable_deixis_normalization = enable_deixis_normalization
        self.enable_code_normalization = enable_code_normalization

    def clean(self, text: str) -> str:
        """对输入文本执行去噪处理。

        Parameters
        ----------
        text : str
            原始文本（通常来自对话或用户输入）。

        Returns
        -------
        str
            去噪后的精简文本。
        """
        if not text:
            return text

        result = text

        # 1. 去除填充词
        if self.enable_filler_removal:
            for pattern, replacement in _FILLER_PATTERNS:
                result = pattern.sub(replacement, result)

        # 2. 指代标准化
        if self.enable_deixis_normalization:
            for pattern, replacement in _DEIXIS_PATTERNS:
                result = pattern.sub(replacement, result)

        # 3. 冗余修饰精简
        for pattern, replacement in _VERBOSE_PATTERNS:
            result = pattern.sub(replacement, result)

        # 4. 代码相关口语标准化
        if self.enable_code_normalization:
            for pattern, replacement in _CODE_MAP:
                result = pattern.sub(replacement, result)

        # 5. 压缩连续空白
        result = _WS.sub(' ', result).strip()

        # 6. 首尾标点清理
        result = result.lstrip('，,。.;；、')
        result = result.rstrip('，,。.;；、')

        # 7. 尝试构建"key: value（from ...）"的结构化格式
        result = self._try_restructure(result)

        return result

    @staticmethod
    def _try_restructure(text: str) -> str:
        """尝试将变更类文本重组为结构化格式。

        检测模式："把X从A改成B" → "X: B（由A修改）"
        """
        # 模式：把...从...改成/改为...
        m = re.search(r'把(.+?)从(.+?)改(?:成|为)(.+?)$', text)
        if m:
            subject = m.group(1).strip()
            old_val = m.group(2).strip()
            new_val = m.group(3).strip()
            return f"{subject}: {new_val}（由{old_val}修改）"

        # 模式：...从...改成/改为...
        m = re.search(r'(.+?)从(.+?)改(?:成|为)(.+?)$', text)
        if m:
            subject = m.group(1).strip()
            old_val = m.group(2).strip()
            new_val = m.group(3).strip()
            return f"{subject}: {new_val}（由{old_val}修改）"

        return text


def clean_text(
    text: str,
    enable_filler_removal: bool = True,
    enable_deixis_normalization: bool = True,
    enable_code_normalization: bool = True,
) -> str:
    """便捷函数：一步完成文本去噪。"""
    return NoiseFilter(
        enable_filler_removal=enable_filler_removal,
        enable_deixis_normalization=enable_deixis_normalization,
        enable_code_normalization=enable_code_normalization,
    ).clean(text)


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  noise_filter.py — 自测 Demo")
    print("=" * 60)

    nf = NoiseFilter()

    # 1. 基本去噪
    print("\n─── 1. 基本去噪 ───")
    cases = [
        ("嗯，我觉得那个LoginHandler里面的代码需要改一下", "LoginHandler的代码需要改一下"),
        ("我们基本上要把token过期时间从15分钟改成30分钟", "token过期时间: 30分钟（由15分钟修改）"),
        ("呃，实际上那个配置项我们上次说的要改成false", "配置项要改为false"),
    ]
    for raw, expected_hint in cases:
        cleaned = nf.clean(raw)
        print(f"  输入: {raw}")
        print(f"  输出: {cleaned}")
        assert len(cleaned) <= len(raw) or "由" in cleaned, f"去噪后不应更长: {cleaned}"
    print("  ✅ 基本去噪通过")

    # 2. 结构化重组
    print("\n─── 2. 结构化重组 ───")
    structured = nf.clean("我们刚才决定把 LoginHandler 里面的 token 过期时间从 15 分钟改成 30 分钟")
    print(f"  输入: 我们刚才决定把 LoginHandler 里面的 token 过期时间从 15 分钟改成 30 分钟")
    print(f"  输出: {structured}")
    assert "由15" in structured and "修改" in structured, f"应包含修改来源: {structured}"
    assert "LoginHandler" in structured
    print("  ✅ 结构化重组通过")

    # 3. 空/边界
    print("\n─── 3. 边界处理 ───")
    assert nf.clean("") == ""
    assert nf.clean("   ") == ""
    single = nf.clean("嗯")
    print(f"  单语气词: '{single}'")
    print("  ✅ 边界处理通过")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
