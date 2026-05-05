"""
cognitive_snapshot.py — 认知快照 (MemCore v0.5)

定期压缩工作记忆状态，形成可恢复的快照。
快照包含：活跃记忆的完整数据、热索引片段、当前焦点实体。

用于：
- 快速重启恢复（不需要从数据库重新加载全部数据）
- 跨对话工作记忆保存
- 多流融合的基础交换单元
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import numpy as np

from memory_core import MemoryCore


# ---------------------------------------------------------------------------
# 快照结构
# ---------------------------------------------------------------------------

SNAPSHOT_VERSION = 1
"""快照格式版本号。"""

SNAPSHOT_SUFFIX = ".snapshot.json"
"""快照文件后缀。"""


class CognitiveSnapshot:
    """认知快照：MemoryCore 工作记忆的紧凑序列化表示。

    包含：
    - 元数据（创建时间、记忆数、版本）
    - memories 列表（id, uuid, query, content, fingerprint）
    - 热点数据（access_counts, protection_scores, timestamps）
    - 向量矩阵（X 和 X_kw 的序列化）
    """

    def __init__(self, metadata: Optional[dict] = None):
        self.metadata: dict[str, Any] = metadata or {
            "version": SNAPSHOT_VERSION,
            "created_at": time.time(),
            "total_memories": 0,
        }
        self.memories: list[dict[str, Any]] = []
        self.access_counts: list[int] = []
        self.protection_scores: list[float] = []
        self.timestamps: list[float] = []
        self.X_bytes: Optional[bytes] = None
        self.X_kw_bytes: Optional[bytes] = None
        self.vec_dim: int = 0
        self.kw_dim: int = 0


# ---------------------------------------------------------------------------
# 快照器
# ---------------------------------------------------------------------------


class Snapshooter:
    """认知快照器。

    从 MemoryCore 实例创建快照，或将快照恢复到 MemoryCore。
    """

    @staticmethod
    def take(mc: MemoryCore) -> CognitiveSnapshot:
        """从 MemoryCore 创建当前状态的快照。

        Parameters
        ----------
        mc : MemoryCore
            要快照的记忆核心实例。

        Returns
        -------
        CognitiveSnapshot
            包含当前工作记忆的紧凑快照。
        """
        snap = CognitiveSnapshot(metadata={
            "version": SNAPSHOT_VERSION,
            "created_at": time.time(),
            "total_memories": len(mc.memories),
            "vec_dim": mc.vec_dim,
            "kw_dim": mc.KW_DIM,
        })

        # 序列化 memories（不含 np.ndarray）
        snap.memories = [
            {
                "db_id": m.get("db_id"),
                "uuid": m.get("uuid", ""),
                "query": m["query"],
                "content": m["content"],
                "fingerprint": m["fingerprint"],
            }
            for m in mc.memories
        ]

        snap.access_counts = list(mc.access_counts)
        snap.protection_scores = list(mc.protection_scores)
        snap.timestamps = list(mc.timestamps)

        # 序列化向量矩阵
        if len(mc.X) > 0:
            snap.X_bytes = mc.X.tobytes()
            snap.X_kw_bytes = mc.X_kw.tobytes()

        snap.vec_dim = mc.vec_dim
        snap.kw_dim = mc.KW_DIM

        return snap

    @staticmethod
    def restore(mc: MemoryCore, snap: CognitiveSnapshot) -> None:
        """从快照恢复 MemoryCore 状态。

        恢复内存数据后，同步写入数据库以确保进程重启后数据不丢失。

        Parameters
        ----------
        mc : MemoryCore
            目标记忆核心实例（应为空或将被覆盖）。
        snap : CognitiveSnapshot
            要恢复的快照。
        """
        # 重建 memories
        mc.memories.clear()
        mc.memories.extend(snap.memories)

        # 辅助数组
        mc.access_counts = list(snap.access_counts)
        mc.protection_scores = list(snap.protection_scores)
        mc.timestamps = list(snap.timestamps)

        # 向量矩阵
        if snap.X_bytes and snap.vec_dim == mc.vec_dim:
            n = len(snap.memories)
            # .copy() 将只读的 frombuffer 数组转为可写，避免后续 add_memory 修改时崩溃
            mc.X = np.frombuffer(snap.X_bytes, dtype=np.float32).reshape(n, -1).copy()
            mc.X_kw = np.frombuffer(snap.X_kw_bytes, dtype=np.float32).reshape(n, -1).copy()
        else:
            mc.X = np.zeros((0, mc.vec_dim), dtype=np.float32)
            mc.X_kw = np.zeros((0, mc.KW_DIM), dtype=np.float32)

        # 同步数据库：在事务中清空旧数据 → 逐条写入快照记忆
        if hasattr(mc, 'conn') and mc.conn is not None:
            import db as _db
            try:
                mc.conn.execute("BEGIN")
                mc.conn.execute("DELETE FROM memories")
                for i, m in enumerate(snap.memories):
                    vec_bytes = mc.X[i].tobytes() if i < len(mc.X) else None
                    kw_bytes = mc.X_kw[i].tobytes() if i < len(mc.X_kw) else None
                    mc.conn.execute(
                        "INSERT INTO memories (query, content, fingerprint, query_vector, "
                        "keyword_vector, uuid, access_count, protection_score) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (m.get("query", ""), m.get("content", ""), m.get("fingerprint", ""),
                         vec_bytes, kw_bytes, m.get("uuid", ""),
                         snap.access_counts[i] if i < len(snap.access_counts) else 0,
                         snap.protection_scores[i] if i < len(snap.protection_scores) else 1.0),
                    )
                mc.conn.execute("COMMIT")
            except Exception:
                mc.conn.execute("ROLLBACK")
                raise


# ---------------------------------------------------------------------------
# 文件持久化
# ---------------------------------------------------------------------------


def save_snapshot(mc: MemoryCore, file_path: str) -> str:
    """将快照保存到文件。

    Parameters
    ----------
    mc : MemoryCore
        要快照的记忆核心。
    file_path : str
        目标文件路径（不含后缀，自动添加 .snapshot.json）。

    Returns
    -------
    str
        实际保存的文件路径。
    """
    if not file_path.endswith(SNAPSHOT_SUFFIX):
        file_path += SNAPSHOT_SUFFIX

    snap = Snapshooter.take(mc)

    # 序列化
    data = {
        "metadata": snap.metadata,
        "memories": snap.memories,
        "access_counts": snap.access_counts,
        "protection_scores": snap.protection_scores,
        "timestamps": snap.timestamps,
        "vec_dim": snap.vec_dim,
        "kw_dim": snap.kw_dim,
    }

    # 向量矩阵 base64 编码
    if snap.X_bytes is not None:
        import base64
        data["X_b64"] = base64.b64encode(snap.X_bytes).decode("ascii")
        data["X_kw_b64"] = base64.b64encode(snap.X_kw_bytes).decode("ascii")
        data["X_shape"] = [len(snap.memories), snap.vec_dim]
        data["X_kw_shape"] = [len(snap.memories), snap.kw_dim]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return file_path


def load_snapshot(file_path: str) -> CognitiveSnapshot:
    """从文件加载快照。

    Parameters
    ----------
    file_path : str
        快照文件路径。

    Returns
    -------
    CognitiveSnapshot
        反序列化的快照。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    snap = CognitiveSnapshot(metadata=data.get("metadata", {}))
    snap.memories = data.get("memories", [])
    snap.access_counts = data.get("access_counts", [])
    snap.protection_scores = data.get("protection_scores", [])
    snap.timestamps = data.get("timestamps", [])
    snap.vec_dim = data.get("vec_dim", 0)
    snap.kw_dim = data.get("kw_dim", 0)

    # 解码向量矩阵
    import base64
    x_b64 = data.get("X_b64")
    if x_b64:
        snap.X_bytes = base64.b64decode(x_b64)
    x_kw_b64 = data.get("X_kw_b64")
    if x_kw_b64:
        snap.X_kw_bytes = base64.b64decode(x_kw_b64)

    return snap


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  cognitive_snapshot.py — 自测 Demo")
    print("=" * 60)

    # 1. 创建快照
    print("\n─── 1. 创建快照 ───")
    mc = MemoryCore(max_memories=10, similarity_threshold=0.30, db_path=":memory:")
    mc.add_memory("咖啡偏好", "用户喜欢喝咖啡", "咖啡 用户")
    mc.add_memory("天气话题", "今天天气很好", "天气 今天")
    mc.add_memory("系统架构", "微服务架构", "架构 微服务")

    # 执行几次 recall 提升 access_count
    mc.recall("咖啡相关")
    mc.recall("咖啡相关")
    mc.recall("天气")

    snap = Snapshooter.take(mc)
    print(f"  快照记忆数: {snap.metadata['total_memories']}")
    print(f"  向量维度: {snap.vec_dim}")
    assert snap.metadata["total_memories"] == 3
    assert snap.X_bytes is not None
    print("  ✅ 快照创建通过")

    # 2. 快照恢复
    print("\n─── 2. 快照恢复 ───")
    mc2 = MemoryCore(max_memories=10, similarity_threshold=0.30, db_path=":memory:")
    Snapshooter.restore(mc2, snap)
    print(f"  恢复后记忆数: {len(mc2.memories)}")
    print(f"  恢复后 access_counts: {mc2.access_counts}")
    assert len(mc2.memories) == 3
    assert mc2.access_counts == [2, 1, 0]
    r = mc2.recall("咖啡相关")
    assert len(r) > 0
    print("  ✅ 快照恢复通过")

    # 3. 文件持久化
    print("\n─── 3. 文件持久化 ───")
    tmp_dir = tempfile.mkdtemp()
    snap_file = os.path.join(tmp_dir, "test_snapshot")
    saved = save_snapshot(mc, snap_file)
    print(f"  保存到: {saved}")
    assert os.path.exists(saved)

    loaded_snap = load_snapshot(saved)
    mc3 = MemoryCore(max_memories=10, similarity_threshold=0.30, db_path=":memory:")
    Snapshooter.restore(mc3, loaded_snap)
    print(f"  文件恢复后记忆数: {len(mc3.memories)}")
    assert len(mc3.memories) == 3
    r3 = mc3.recall("天气")
    assert len(r3) > 0
    print("  ✅ 文件持久化恢复通过")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
