"""
memory_bridge.py — 多流融合 + 跨项目透传 (MemCore v0.5)

通过共享 SQLite 数据库实现多个 MemoryCore 实例之间的记忆共享和传递。

功能：
1. 多流融合：多个 MemoryCore（同一项目的不同对话流）共享热点记忆
2. 跨项目透传：当本地召回结果不足时，从共享池中查询补充
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

import numpy as np

from memory_core import MemoryCore
from embedder import create_embedder
from profile_config import get_profile, ProfileConfig


# ---------------------------------------------------------------------------
# 共享记忆池
# ---------------------------------------------------------------------------


DEFAULT_POOL_PATH = "./memory_pool.db"
"""默认共享池数据库路径。"""


class MemoryPool:
    """跨实例/跨项目共享记忆池。

    使用独立的 SQLite 数据库存储共享记忆的元数据 + 向量。
    每个 MemoryCore 实例可以 push（导出）自己的热点记忆到池，
    或 pull（导入）池中与查询相关的记忆。
    """

    def __init__(self, db_path: str = DEFAULT_POOL_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.encoder = create_embedder()

    def _init_db(self) -> None:
        """初始化共享池表结构。"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pool (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid        TEXT UNIQUE NOT NULL,
                source_tag  TEXT NOT NULL DEFAULT '',
                query       TEXT NOT NULL,
                content     TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                query_vector BLOB,
                keyword_vector BLOB,
                access_count INTEGER DEFAULT 0,
                protection_score REAL DEFAULT 1.0,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_pulled TIMESTAMP
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Push：将记忆导出到共享池
    # ------------------------------------------------------------------

    def push_memory(
        self,
        query: str,
        content: str,
        fingerprint: str,
        query_vector: Optional[np.ndarray] = None,
        source_tag: str = "",
        _retry_depth: int = 0,
        profile: Optional[ProfileConfig] = None,
    ) -> int:
        """将一条记忆推入共享池。

        Parameters
        ----------
        query, content, fingerprint : str
            记忆字段。
        query_vector : np.ndarray | None
            语义向量。None 时自动编码。
        source_tag : str
            来源标识（如 "project:memcore", "session:abc"）。
        _retry_depth : int
            UUID 冲突重试深度（内部参数，外部无需传递）。
        profile : ProfileConfig | None
            配置实例。None 时从全局 ProfileConfig 读取。

        Returns
        -------
        int
            共享池中记录的 ID。
        """
        if query_vector is None:
            query_vector = self.encoder.encode(query).astype(np.float32)
        vec_bytes = query_vector.tobytes()

        # 生成 keyword_vector
        from memory_core import MemoryCore
        kw_vec = MemoryCore._generate_keyword_vector(
            f"{query} {content} {fingerprint}", 128
        )
        kw_bytes = kw_vec.tobytes()

        import uuid as _uuid
        uid = str(_uuid.uuid4())

        try:
            cursor = self.conn.execute(
                "INSERT INTO pool (uuid, source_tag, query, content, fingerprint, "
                "query_vector, keyword_vector) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (uid, source_tag, query, content, fingerprint, vec_bytes, kw_bytes),
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # UUID 冲突（极低概率），重试次数由 ProfileConfig.max_retries 控制
            cfg = profile or get_profile()
            max_retries = cfg.max_retries
            if _retry_depth >= max_retries:
                raise RuntimeError(
                    f"UUID 冲突重试 {_retry_depth} 次仍失败，放弃插入"
                )
            return self.push_memory(
                query, content, fingerprint, query_vector, source_tag,
                _retry_depth=_retry_depth + 1,
            )

    def push_hot_memories(
        self,
        mc: MemoryCore,
        source_tag: str = "",
        top_n: int = 5,
    ) -> int:
        """将 MemoryCore 中最热门的 N 条记忆推入共享池。

        按 protection_score 排序，取最高的 top_n 条。

        Parameters
        ----------
        mc : MemoryCore
            源记忆核心。
        source_tag : str
            来源标识。
        top_n : int, default 5
            推送条数。

        Returns
        -------
        int
            实际推送条数。
        """
        if not mc.memories:
            return 0

        # 按 protection_score 排序
        indexed = list(enumerate(mc.protection_scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        hot_indices = [idx for idx, _ in indexed[:top_n]]

        pushed = 0
        for idx in hot_indices:
            mem = mc.memories[idx]
            self.push_memory(
                query=mem["query"],
                content=mem["content"],
                fingerprint=mem["fingerprint"],
                query_vector=mc.X[idx] if idx < len(mc.X) else None,
                source_tag=source_tag,
            )
            pushed += 1

        return pushed

    # ------------------------------------------------------------------
    # Pull：从共享池中召回相关记忆
    # ------------------------------------------------------------------

    def pull_memories(
        self,
        query_text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.25,
        source_filter: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """从共享池中召回与查询相关的记忆。

        Parameters
        ----------
        query_text : str
            查询文本。
        top_k : int, default 3
            最大返回条数。
        similarity_threshold : float, default 0.25
            余弦相似度最低阈值。
        source_filter : str | None, default None
            来源过滤（如 "project:memcore"）。None 表示不过滤。

        Returns
        -------
        list[dict]
            每条含 {query, content, fingerprint, source_tag, similarity}。
        """
        rows = self.conn.execute(
            "SELECT id, uuid, source_tag, query, content, fingerprint, query_vector "
            "FROM pool ORDER BY last_pulled IS NULL, last_pulled DESC"
        ).fetchall()

        if not rows:
            return []

        q_vec = self.encoder.encode(query_text).astype(np.float32)
        q_norm = float(np.linalg.norm(q_vec))

        scored: list[tuple[float, dict[str, Any]]] = []
        pulled_ids: list[int] = []  # 记录实际被拉取的行 ID
        for r in rows:
            if source_filter and r["source_tag"] != source_filter:
                continue
            vec_bytes = r["query_vector"]
            if not vec_bytes:
                continue
            mem_vec = np.frombuffer(vec_bytes, dtype=np.float32)
            m_norm = float(np.linalg.norm(mem_vec))
            if q_norm > 0 and m_norm > 0:
                cos = float(np.dot(q_vec, mem_vec)) / (q_norm * m_norm)
                if cos >= similarity_threshold:
                    scored.append((cos, {
                        "query": r["query"],
                        "content": r["content"],
                        "fingerprint": r["fingerprint"],
                        "source_tag": r["source_tag"],
                        "similarity": cos,
                    }))
                    pulled_ids.append(r["id"])

        # 仅更新实际被拉取的行的 last_pulled 时间
        if pulled_ids:
            now_ts = time.time()
            for pid in pulled_ids:
                self.conn.execute(
                    "UPDATE pool SET last_pulled = ? WHERE id = ?",
                    (now_ts, pid),
                )
            self.conn.commit()

        scored.sort(reverse=True, key=lambda x: x[0])
        return [item for _, item in scored[:top_k]]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    def size(self) -> int:
        """返回共享池中的记忆总数。"""
        row = self.conn.execute("SELECT COUNT(*) AS cnt FROM pool").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """关闭数据库连接。"""
        self.conn.close()

    def __enter__(self) -> "MemoryPool":
        """支持 with 语句，自动管理连接生命周期。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出 with 块时自动关闭连接。"""
        self.close()


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import tempfile, os

    print("=" * 60)
    print("  memory_bridge.py — 自测 Demo")
    print("=" * 60)

    pool_file = os.path.join(tempfile.mkdtemp(), "pool.db")
    pool = MemoryPool(pool_file)

    # 1. Push / Pull
    print("\n─── 1. Push + Pull ───")
    pool.push_memory("咖啡偏好", "用户喜欢喝咖啡", "咖啡 用户", source_tag="session:food")
    pool.push_memory("天气话题", "今天天气很好", "天气 今天", source_tag="session:weather")
    pool.push_memory("系统架构", "微服务架构", "架构 微服务", source_tag="session:arch")
    print(f"  共享池: {pool.size()} 条")
    assert pool.size() == 3
    print("  ✅ Push 通过")

    pulled = pool.pull_memories("咖啡相关", top_k=2)
    print(f"  Pull「咖啡相关」: {[p['content'] for p in pulled]}")
    assert len(pulled) > 0
    print("  ✅ Pull 通过")

    # 2. 从 MemoryCore 推送热点
    print("\n─── 2. 热点推送 ───")
    mc = MemoryCore(max_memories=10, similarity_threshold=0.30, db_path=":memory:")
    mc.add_memory("记忆A", "记忆A内容", "A")
    mc.add_memory("记忆B", "记忆B内容", "B")
    mc.add_memory("记忆C", "记忆C内容", "C")
    for _ in range(5):
        mc.recall("记忆A")

    pushed = pool.push_hot_memories(mc, source_tag="project:test", top_n=2)
    print(f"  推送了 {pushed} 条热点记忆到共享池")
    assert pushed > 0
    print("  ✅ 热点推送通过")

    # 3. 来源过滤
    print("\n─── 3. 来源过滤 ───")
    filtered = pool.pull_memories("内容", source_filter="project:test")
    print(f"  过滤 project:test: {len(filtered)} 条")
    assert all(p["source_tag"] == "project:test" for p in filtered)
    print("  ✅ 来源过滤通过")

    pool.close()
    print(f"\n{'='*60}")
    print("  Demo 完成")
    print("=" * 60)
