"""
profile_config.py — 三模式配置中心 (MemCore v0.5)

提供三种运行模式，适配不同规模的使用场景：
  - Chat   : 日常对话，轻量快速，资源占用小
  - Dev    : 大型项目开发，平衡容量与性能
  - Unlimited : 无限制模式，在安全范围内尽量放宽

设计原则：
  1. 算法核心参数（beta, lambda_decay, alpha）跨模式不变——
     这些是 Hopfield 内核的收敛保证参数，改了会影响召回质量
  2. 容量/阈值/频率等外围参数按场景调优
  3. 所有模块通过 get_profile() 获取当前配置，不直接硬编码
  4. 支持运行时切换模式（通过 Server API）

使用方式：
    from profile_config import get_profile, set_profile
    cfg = get_profile()           # 获取当前配置
    set_profile("dev")            # 切换到开发模式
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Literal


# ===========================================================================
# Profile 名称类型
# ===========================================================================

ProfileName = Literal["chat", "dev", "unlimited"]


# ===========================================================================
# ProfileConfig 数据类
# ===========================================================================

@dataclass
class ProfileConfig:
    """三模式配置数据类。

    所有字段均有默认值（dev 模式），可直接实例化后按需覆盖。
    """

    # ---- 模式标识 ----
    profile: ProfileName = "dev"

    # ---- 容量类 ----
    short_max: int = 500
    """短期记忆容量硬上限（防止内存溢出）。实际容量由时间窗口控制。"""
    
    short_term_window_hours: int = 24
    """短期记忆时间窗口（小时）。窗口内的记忆保留，窗口外的交给做梦/巩固处理。"""

    long_max: int = 10000
    """长期记忆容量上限。"""

    consolidation_count: int = 3
    """巩固阈值：被召回次数达到该值后巩固到长期。
    Chat 模式设高（5），减少不必要的巩固开销；
    Unlimited 设低（2），快速巩固以防遗忘。"""

    max_retries: int = 3
    """UUID 冲突重试深度上限。"""

    # ---- 召回类 ----
    short_threshold: float = 0.10
    """短期召回阈值（宽松，容易召回）。"""

    long_threshold: float = 0.30
    """长期召回阈值（严格，精确才召回）。"""

    top_k: int = 5
    """MemoryCore 每次 recall 返回的最大结果数（已废弃，改用 token 预算）。"""

    recall_merge_max: int = 5
    """DualMemory 双层合并后返回的最大结果数（已废弃，改用 token 预算）。"""
    
    # ---- Token 预算类 ----
    memory_token_budget: int = 10000
    """记忆召回的 token 预算上限。实际返回量由匹配数决定，不硬塞。"""
    
    token_per_memory: int = 60
    """每条记忆的平均 token 数（用于条数估算）。"""
    
    event_budget_ratio: float = 0.30
    """事件记忆占预算的比例（骨架优先）。"""
    
    short_budget_ratio: float = 0.50
    """短期记忆占预算的比例（细节填充）。"""
    
    long_budget_ratio: float = 0.20
    """长期记忆占预算的比例（深度补充）。"""
    
    # ---- 意图感知预算分配 ----
    intent_budget_overrides: dict[str, dict[str, float]] = field(default_factory=lambda: {
        "reference": {"event": 0.50, "short": 0.30, "long": 0.20},  # 指代→事件优先
        "entity":    {"event": 0.20, "short": 0.30, "long": 0.50},  # 实体→长期优先
        "semantic":  {"event": 0.30, "short": 0.50, "long": 0.20},  # 语义→短期优先
        "document":  {"event": 0.40, "short": 0.30, "long": 0.30},  # 文档→事件优先
    })
    """意图感知的预算比例覆盖。按查询意图调整三层预算分配。"""

    # ---- Hopfield 算法参数（跨模式不变）----
    beta: float = 8.0
    """霍普菲尔德迭代的相似度放大系数。"""

    lambda_decay: float = 5.73e-7
    """时间衰减系数（14 天半衰期）。"""

    alpha: float = 0.75
    """双向量融合系数：semantic * alpha + keyword * (1-alpha)。"""

    # ---- Context 类 ----
    token_budget: int = 2000
    """上下文窗口 Token 预算。"""

    summary_threshold: float = 0.5
    """当预算比例 < 此值时使用摘要而非完整文本。"""

    truncate_threshold: float = 0.2
    """当预算比例 < 此值时仅截断输出最关键的条目。"""

    # ---- Digest 类 ----
    poll_interval: int = 10
    """DigestPipeline 轮询间隔（秒）。"""

    consolidate_interval: int = 300
    """DigestPipeline 巩固间隔（秒）。"""

    batch_size: int = 10
    """DigestPipeline 批次大小。"""

    # ---- Conflict 类 ----
    base_confidence: float = 0.7
    """冲突解决基础置信度。"""

    age_decay_days: int = 14
    """置信度衰减起始天数。"""

    age_decay_rate: float = 0.005
    """每日衰减率。"""

    age_decay_floor: float = 0.6
    """衰减下限。"""

    # ---- Embedder 类 ----
    embedding_dim: int = 1024
    """嵌入向量维度。当前 BGE-m3 为 1024 维（多语言支持英文/中文/代码）。"""

    # ---- 事件索引类 (v0.6) ----
    event_top_k: int = 5
    """事件索引 Layer 2 返回的候选事件数。"""

    dream_enabled: bool = True
    """是否启用后台做梦进程（Phase1 规则引擎 / Phase2 LLM）。"""

    dream_min_interval: int = 180
    """两次做梦之间的最短间隔（秒）。"""

    dream_min_new_memories: int = 15
    """触发做梦所需的最小新记忆数。"""

    event_reopen_threshold: float = 0.70
    """新记忆与已有开放事件的 BGE 余弦阈值，超过则追加而非新建事件。"""

    # ---- 后备检索 (v0.6) ----
    fallback_enabled: bool = True
    """是否启用后备文本检索（向量检索失败时兜底）。"""
    
    fallback_whitelist: list[str] = field(default_factory=lambda: ["./src", "./lib", "./docs"])
    """后备检索白名单目录。"""
    
    fallback_blacklist: list[str] = field(default_factory=lambda: [".env", ".git", "*.key", "*.pem"])
    """后备检索黑名单模式。"""
    
    # ---- 通用 ----
    db_dir: str = "./memory_db"
    """数据库文件存放目录。"""


# ===========================================================================
# 三模式预设
# ===========================================================================

_PROFILE_PRESETS: dict[ProfileName, dict] = {
    "chat": {
        "profile": "chat",
        # 容量
        "short_max": 50,
        "short_term_window_hours": 12,  # Chat 模式窗口短
        "long_max": 500,
        "consolidation_count": 5,
        "max_retries": 2,
        # 召回
        "short_threshold": 0.15,
        "long_threshold": 0.40,
        "top_k": 3,
        "recall_merge_max": 3,
        # Token 预算
        "memory_token_budget": 3000,
        "token_per_memory": 60,
        "event_budget_ratio": 0.30,
        "short_budget_ratio": 0.50,
        "long_budget_ratio": 0.20,
        "intent_budget_overrides": {
            "reference": {"event": 0.50, "short": 0.30, "long": 0.20},
            "entity":    {"event": 0.20, "short": 0.30, "long": 0.50},
            "semantic":  {"event": 0.30, "short": 0.50, "long": 0.20},
            "document":  {"event": 0.40, "short": 0.30, "long": 0.30},
        },
        # Hopfield（不变）
        "beta": 8.0,
        "lambda_decay": 5.73e-7,
        "alpha": 0.75,
        # Context
        "token_budget": 400,
        "summary_threshold": 0.6,
        "truncate_threshold": 0.3,
        # Digest
        "poll_interval": 30,
        "consolidate_interval": 600,
        "batch_size": 3,
        # Conflict
        "base_confidence": 0.7,
        "age_decay_days": 7,
        "age_decay_rate": 0.01,
        "age_decay_floor": 0.5,
        # Embedder
        "embedding_dim": 1024,
        # 事件索引 (v0.6)
        "event_top_k": 3,
        "dream_enabled": True,
        "dream_min_interval": 120,
        "dream_min_new_memories": 10,
        "event_reopen_threshold": 0.75,
        # 通用
        "db_dir": "./memory_db",
    },
    "dev": {
        "profile": "dev",
        # 容量
        "short_max": 500,
        "short_term_window_hours": 24,  # Dev 模式窗口适中
        "long_max": 10000,
        "consolidation_count": 3,
        "max_retries": 3,
        # 召回
        "short_threshold": 0.10,
        "long_threshold": 0.30,
        "top_k": 5,
        "recall_merge_max": 5,
        # Token 预算
        "memory_token_budget": 10000,
        "token_per_memory": 60,
        "event_budget_ratio": 0.30,
        "short_budget_ratio": 0.50,
        "long_budget_ratio": 0.20,
        "intent_budget_overrides": {
            "reference": {"event": 0.50, "short": 0.30, "long": 0.20},
            "entity":    {"event": 0.20, "short": 0.30, "long": 0.50},
            "semantic":  {"event": 0.30, "short": 0.50, "long": 0.20},
            "document":  {"event": 0.40, "short": 0.30, "long": 0.30},
        },
        # Hopfield（不变）
        "beta": 8.0,
        "lambda_decay": 5.73e-7,
        "alpha": 0.75,
        # Context
        "token_budget": 2000,
        "summary_threshold": 0.5,
        "truncate_threshold": 0.2,
        # Digest
        "poll_interval": 10,
        "consolidate_interval": 300,
        "batch_size": 10,
        # Conflict
        "base_confidence": 0.7,
        "age_decay_days": 14,
        "age_decay_rate": 0.005,
        "age_decay_floor": 0.6,
        # Embedder
        "embedding_dim": 1024,
        # 事件索引 (v0.6)
        "event_top_k": 5,
        "dream_enabled": True,
        "dream_min_interval": 180,
        "dream_min_new_memories": 15,
        "event_reopen_threshold": 0.70,
        # 通用
        "db_dir": "./memory_db",
    },
    "unlimited": {
        "profile": "unlimited",
        # 容量
        "short_max": 2000,
        "short_term_window_hours": 48,  # Unlimited 模式窗口宽
        "long_max": 50000,
        "consolidation_count": 2,
        "max_retries": 5,
        # 召回
        "short_threshold": 0.05,
        "long_threshold": 0.20,
        "top_k": 10,
        "recall_merge_max": 10,
        # Token 预算
        "memory_token_budget": 30000,
        "token_per_memory": 60,
        "event_budget_ratio": 0.30,
        "short_budget_ratio": 0.50,
        "long_budget_ratio": 0.20,
        "intent_budget_overrides": {
            "reference": {"event": 0.50, "short": 0.30, "long": 0.20},
            "entity":    {"event": 0.20, "short": 0.30, "long": 0.50},
            "semantic":  {"event": 0.30, "short": 0.50, "long": 0.20},
            "document":  {"event": 0.40, "short": 0.30, "long": 0.30},
        },
        # Hopfield（不变）
        "beta": 8.0,
        "lambda_decay": 5.73e-7,
        "alpha": 0.75,
        # Context
        "token_budget": 4000,
        "summary_threshold": 0.4,
        "truncate_threshold": 0.1,
        # Digest
        "poll_interval": 5,
        "consolidate_interval": 120,
        "batch_size": 20,
        # Conflict
        "base_confidence": 0.7,
        "age_decay_days": 30,
        "age_decay_rate": 0.002,
        "age_decay_floor": 0.7,
        # Embedder
        "embedding_dim": 1024,
        # 事件索引 (v0.6)
        "event_top_k": 10,
        "dream_enabled": True,
        "dream_min_interval": 300,
        "dream_min_new_memories": 20,
        "event_reopen_threshold": 0.65,
        # 通用
        "db_dir": "./memory_db",
    },
}


# ===========================================================================
# 全局配置实例
# ===========================================================================

_current_profile: ProfileConfig = ProfileConfig()  # 默认 dev


def get_profile() -> ProfileConfig:
    """获取当前全局配置实例。"""
    return _current_profile


def set_profile(name: ProfileName, overrides: dict | None = None) -> ProfileConfig:
    """切换到指定模式，可选覆盖部分参数。

    Parameters
    ----------
    name : ProfileName
        目标模式名称："chat", "dev", "unlimited"。
    overrides : dict | None
        可选参数覆盖，如 {"short_max": 1000}。

    Returns
    -------
    ProfileConfig
        切换后的配置实例。

    Raises
    ------
    ValueError
        无效的模式名称。
    """
    global _current_profile

    if name not in _PROFILE_PRESETS:
        valid = ", ".join(_PROFILE_PRESETS.keys())
        raise ValueError(f"无效的模式 '{name}'，可选: {valid}")

    preset = _PROFILE_PRESETS[name].copy()
    if overrides:
        # 只允许覆盖 ProfileConfig 中存在的字段
        valid_fields = {f.name for f in fields(ProfileConfig)}
        for k, v in overrides.items():
            if k not in valid_fields:
                raise ValueError(
                    f"无效的配置项 '{k}'，可选: {', '.join(sorted(valid_fields))}"
                )
            preset[k] = v

    _current_profile = ProfileConfig(**preset)
    return _current_profile


def create_profile(name: ProfileName, overrides: dict | None = None) -> ProfileConfig:
    """创建一个新的 ProfileConfig 实例（不影响全局配置）。

    用于需要多个独立配置实例的场景，如测试。
    """
    if name not in _PROFILE_PRESETS:
        valid = ", ".join(_PROFILE_PRESETS.keys())
        raise ValueError(f"无效的模式 '{name}'，可选: {valid}")

    preset = _PROFILE_PRESETS[name].copy()
    if overrides:
        valid_fields = {f.name for f in fields(ProfileConfig)}
        for k, v in overrides.items():
            if k not in valid_fields:
                raise ValueError(
                    f"无效的配置项 '{k}'，可选: {', '.join(sorted(valid_fields))}"
                )
            preset[k] = v

    return ProfileConfig(**preset)


def profile_to_dict() -> dict:
    """将当前全局配置导出为字典。"""
    return asdict(_current_profile)


def available_profiles() -> list[ProfileName]:
    """返回所有可用的模式名称。"""
    return list(_PROFILE_PRESETS.keys())


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("profile_config.py — 三模式配置中心 自测 Demo")
    print("=" * 60)

    # 1. 默认配置（dev 模式）
    cfg = get_profile()
    print(f"\n[1] 默认配置: profile={cfg.profile}")
    print(f"    short_max={cfg.short_max}, long_max={cfg.long_max}")
    print(f"    top_k={cfg.top_k}, recall_merge_max={cfg.recall_merge_max}")
    print(f"    token_budget={cfg.token_budget}")
    assert cfg.profile == "dev"
    print("    ✓ 默认为 dev 模式")

    # 2. 切换到 chat 模式
    cfg = set_profile("chat")
    print(f"\n[2] 切换到 chat 模式:")
    print(f"    short_max={cfg.short_max}, long_max={cfg.long_max}")
    print(f"    consolidation_count={cfg.consolidation_count}")
    print(f"    poll_interval={cfg.poll_interval}s")
    assert cfg.profile == "chat"
    assert cfg.short_max == 50
    assert cfg.long_max == 500
    print("    ✓ chat 模式参数正确")

    # 3. 切换到 unlimited 模式
    cfg = set_profile("unlimited")
    print(f"\n[3] 切换到 unlimited 模式:")
    print(f"    short_max={cfg.short_max}, long_max={cfg.long_max}")
    print(f"    top_k={cfg.top_k}, recall_merge_max={cfg.recall_merge_max}")
    assert cfg.profile == "unlimited"
    assert cfg.short_max == 2000
    assert cfg.long_max == 50000
    print("    ✓ unlimited 模式参数正确")

    # 4. 切换到 dev 模式并覆盖部分参数
    cfg = set_profile("dev", overrides={"short_max": 800, "top_k": 8})
    print(f"\n[4] dev 模式 + 自定义覆盖:")
    print(f"    short_max={cfg.short_max} (覆盖为 800)")
    print(f"    top_k={cfg.top_k} (覆盖为 8)")
    print(f"    long_max={cfg.long_max} (保持默认)")
    assert cfg.short_max == 800
    assert cfg.top_k == 8
    assert cfg.long_max == 10000  # 未覆盖，保持 dev 默认
    print("    ✓ 参数覆盖正确")

    # 5. 创建独立配置实例（不影响全局）
    independent = create_profile("chat")
    print(f"\n[5] 独立配置实例:")
    print(f"    independent.profile={independent.profile}")
    print(f"    全局 profile={get_profile().profile}")
    assert independent.profile == "chat"
    assert get_profile().profile == "dev"  # 全局仍为 dev
    print("    ✓ 独立实例不影响全局")

    # 6. 错误处理
    print(f"\n[6] 错误处理:")
    try:
        set_profile("invalid")
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"    无效模式: {e}")
        print("    ✓ 正确拒绝无效模式")

    try:
        set_profile("dev", overrides={"nonexistent_param": 42})
        assert False, "应抛出 ValueError"
    except ValueError as e:
        print(f"    无效配置项: {e}")
        print("    ✓ 正确拒绝无效配置项")

    # 7. 导出为字典
    cfg = set_profile("dev")
    d = profile_to_dict()
    print(f"\n[7] 导出为字典:")
    print(f"    字段数: {len(d)}")
    assert d["profile"] == "dev"
    assert d["short_max"] == 500
    print("    ✓ 字典导出正确")

    # 8. 三模式参数对比表
    print(f"\n[8] 三模式参数对比表:")
    header = f"{'参数':<25} {'Chat':>10} {'Dev':>10} {'Unlimited':>10}"
    print(f"    {header}")
    print(f"    {'-'*55}")
    key_params = [
        "short_max", "long_max", "consolidation_count",
        "short_threshold", "long_threshold", "top_k", "recall_merge_max",
        "token_budget", "poll_interval", "consolidate_interval", "batch_size",
    ]
    for pname in _PROFILE_PRESETS:
        pass  # 以下直接用预设值
    for param in key_params:
        chat_val = _PROFILE_PRESETS["chat"][param]
        dev_val = _PROFILE_PRESETS["dev"][param]
        unlim_val = _PROFILE_PRESETS["unlimited"][param]
        print(f"    {param:<25} {str(chat_val):>10} {str(dev_val):>10} {str(unlim_val):>10}")

    # 9. 可用模式列表
    print(f"\n[9] 可用模式: {available_profiles()}")

    print(f"\n{'='*60}")
    print("Demo 完成")
    print("=" * 60)
