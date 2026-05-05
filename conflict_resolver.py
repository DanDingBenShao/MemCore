"""
conflict_resolver.py — 冲突仲裁与年龄惩罚 (MemCore v0.3.1)

当多条记忆对同一事物提供矛盾信息时，通过有效置信度进行仲裁。
引入时间衰减：超过 7 天的记忆置信度按日衰减。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Optional

from profile_config import get_profile, ProfileConfig


# ---------------------------------------------------------------------------
# 默认参数（向后兼容，实际默认值由 ProfileConfig 提供）
# ---------------------------------------------------------------------------

DEFAULT_BASE_CONFIDENCE = 0.7
"""AI 生成记忆的默认置信度。"""

HUMAN_CONFIRMED_CONFIDENCE = 1.0
"""人工确认记忆的置信度。"""

DISPUTE_THRESHOLD = 0.1
"""有效置信度差距 > 此值时决出胜负，否则标注'存在争议'。"""

AGE_DECAY_DAYS = 7
"""多少天后开始衰减。"""

AGE_DECAY_RATE = 0.01
"""每日衰减率（超过 AGE_DECAY_DAYS 后）。"""

AGE_DECAY_FLOOR = 0.5
"""衰减下限。"""


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------


def effective_confidence(
    mem: dict[str, Any],
    now: float | None = None,
    profile: Optional[ProfileConfig] = None,
) -> float:
    """计算记忆的有效置信度（基础置信度 × 时间衰减）。

    Parameters
    ----------
    mem : dict
        记忆条目，至少包含 "timestamp" 和 "confidence" 字段。
        若无 "confidence"，使用配置中的 base_confidence。
    now : float | None, default None
        当前时间戳（time.time()）。None 时自动获取。
    profile : ProfileConfig | None
        配置实例。None 时从全局 ProfileConfig 读取。

    Returns
    -------
    float
        有效置信度，范围 [age_decay_floor * base_confidence, 1.0]。
    """
    if now is None:
        now = time.time()

    cfg = profile or get_profile()
    base_conf = cfg.base_confidence
    decay_days = cfg.age_decay_days
    decay_rate = cfg.age_decay_rate
    decay_floor = cfg.age_decay_floor

    base = mem.get("confidence", base_conf)

    # 获取记忆时间戳
    mem_ts = mem.get("timestamp")
    if mem_ts is None:
        return base  # 无时间戳，不衰减

    if isinstance(mem_ts, datetime):
        age_days = (datetime.fromtimestamp(now) - mem_ts).days
    elif isinstance(mem_ts, (int, float)):
        age_days = (now - mem_ts) / 86400.0
    elif isinstance(mem_ts, str):
        # 尝试解析 ISO 8601 格式的时间戳字符串
        try:
            parsed_dt = datetime.fromisoformat(mem_ts)
            age_days = (datetime.fromtimestamp(now) - parsed_dt).days
        except (ValueError, TypeError):
            return base
    else:
        return base

    if age_days <= decay_days:
        return base

    decay = max(decay_floor, 1.0 - decay_rate * (age_days - decay_days))
    return base * decay


def resolve_conflict(
    mem_a: dict[str, Any],
    mem_b: dict[str, Any],
    now: float | None = None,
) -> dict[str, Any]:
    """仲裁两条矛盾记忆，返回胜出的记忆。

    规则：
    - 计算各自的 effective_confidence
    - 差距 > DISPUTE_THRESHOLD → 胜出者
    - 否则标注 "存在争议"

    Parameters
    ----------
    mem_a, mem_b : dict
        两条矛盾的记忆条目。
    now : float | None, default None
        当前时间戳。

    Returns
    -------
    dict
        胜出的记忆，包含额外字段 "resolution" 说明仲裁结果。
    """
    conf_a = effective_confidence(mem_a, now)
    conf_b = effective_confidence(mem_b, now)

    if conf_a - conf_b > DISPUTE_THRESHOLD:
        winner = dict(mem_a)
        winner["resolution"] = "won"
        winner["resolution_reason"] = (
            f"effective_confidence {conf_a:.3f} > {conf_b:.3f}"
        )
        return winner

    if conf_b - conf_a > DISPUTE_THRESHOLD:
        winner = dict(mem_b)
        winner["resolution"] = "won"
        winner["resolution_reason"] = (
            f"effective_confidence {conf_b:.3f} > {conf_a:.3f}"
        )
        return winner

    # 差距太小，标注争议
    result = dict(mem_a)
    result["resolution"] = "disputed"
    result["resolution_reason"] = (
        f"effective_confidence gap {abs(conf_a - conf_b):.3f} <= {DISPUTE_THRESHOLD}"
    )
    result["confidence_a"] = conf_a
    result["confidence_b"] = conf_b
    return result


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import time
    from datetime import datetime
    from profile_config import create_profile

    print("=" * 60)
    print("  conflict_resolver.py — 自测 Demo")
    print("=" * 60)

    now = time.time()
    # 使用独立的 chat 配置实例来测试（与旧参数一致：days=7, rate=0.01, floor=0.5）
    test_profile = create_profile("chat")

    # 1. 有效置信度测试
    print("\n─── 1. 有效置信度 ───")
    fresh_mem = {
        "timestamp": datetime.fromtimestamp(now - 3600),  # 1小时前
        "confidence": 0.7,
    }
    old_mem = {
        "timestamp": datetime.fromtimestamp(now - 60 * 86400),  # 60天前
        "confidence": 0.7,
    }
    fresh_conf = effective_confidence(fresh_mem, now, profile=test_profile)
    old_conf = effective_confidence(old_mem, now, profile=test_profile)
    # chat 模式：days=7, rate=0.01, floor=0.5
    # 60天：(60-7)*0.01=0.53, 1-0.53=0.47, floor=0.5, 0.7*0.5=0.35
    print(f"  新记忆(1小时): {fresh_conf:.3f} (应为 0.700)")
    print(f"  旧记忆(60天): {old_conf:.3f} (应为 0.350)")
    assert abs(fresh_conf - 0.7) < 0.01
    assert abs(old_conf - 0.35) < 0.01
    print("  ✓ 年龄衰减计算正确 (chat 模式)")

    # 2. 冲突仲裁
    print("\n─── 2. 冲突仲裁 ───")
    winner = resolve_conflict(fresh_mem, old_mem, now)
    print(f"  仲裁结果: {winner['resolution']}")
    assert winner["resolution"] == "won"
    print("  ✓ 新鲜记忆胜出")

    # 3. 标注争议
    both_old = {
        "timestamp": datetime.fromtimestamp(now - 30 * 86400),
        "confidence": 0.7,
    }
    both_old2 = {
        "timestamp": datetime.fromtimestamp(now - 31 * 86400),
        "confidence": 0.7,
    }
    dispute = resolve_conflict(both_old, both_old2, now)
    print(f"  两条相近旧记忆仲裁: {dispute['resolution']}")
    assert dispute["resolution"] == "disputed"
    print("  ✓ 差距过小时标注争议")

    # 4. 不同模式参数对比
    print("\n─── 4. 三模式衰减参数对比 ───")
    for pname in ["chat", "dev", "unlimited"]:
        p = create_profile(pname)
        print(f"  {pname}: decay_days={p.age_decay_days}, "
              f"rate={p.age_decay_rate}, floor={p.age_decay_floor}")

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
