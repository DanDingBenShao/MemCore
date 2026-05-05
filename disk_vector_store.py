# disk_vector_store.py — 磁盘向量存储组件
#
# 为长期记忆提供可扩展的向量存储方案：
# - FAISS IVF 索引（推荐，需要安装 faiss-cpu 或 faiss-gpu）
# - SQLite 暴力搜索（fallback，无需额外依赖）
#
# 设计原则：
# - 磁盘索引做粗筛（top_k=200）
# - SQLite 存储完整向量（用于精排加载）
# - 支持标记删除 + 定期 compact

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# FAISS 可选导入
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[DiskVectorStore] FAISS 未安装，使用 SQLite 暴力搜索模式")


class DiskVectorStore:
    """磁盘向量存储 — 支持大规模向量的持久化和检索。
    
    两种模式：
    1. FAISS 模式：使用 IVF 索引，训练阈值默认 10000
    2. SQLite 模式：暴力余弦搜索，无需训练
    
    接口设计遵循"磁盘粗筛 + Hopfield精排"的两阶段召回架构。
    """
    
    # 训练阈值：N >= 此值时训练 IVF 索引
    TRAIN_THRESHOLD = 10000
    # 默认粗筛候选数
    DEFAULT_TOP_K = 200
    # IVF nlist 参数（聚类中心数）
    IVF_NLIST = 100
    
    def __init__(
        self,
        db_path: str,
        dim: int = 1024,
        use_faiss: bool = True,
    ):
        """初始化磁盘向量存储。
        
        Parameters
        ----------
        db_path : str
            SQLite 数据库路径（存储向量和元数据）
        dim : int, default 1024
            向量维度
        use_faiss : bool, default True
            是否尝试使用 FAISS。若 FAISS 不可用则自动降级到 SQLite 模式
        """
        self.db_path = db_path
        self.dim = dim
        self._lock = threading.Lock()
        
        # 模式选择
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self._index = None  # FAISS 索引
        self._index_trained = False
        
        # 初始化 SQLite
        self._init_sqlite()
        
        # 若使用 FAISS 且已有数据，尝试加载/重建索引
        if self.use_faiss:
            self._load_or_build_index()
    
    def _init_sqlite(self) -> None:
        """初始化 SQLite 表结构。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS disk_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector BLOB NOT NULL,
                deleted INTEGER DEFAULT 0,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON disk_vectors(deleted)")
        conn.commit()
        conn.close()
    
    def _load_or_build_index(self) -> None:
        """加载已有 FAISS 索引，或从 SQLite 重建。"""
        index_path = self._get_index_path()
        
        if os.path.exists(index_path):
            try:
                self._index = faiss.read_index(index_path)
                self._index_trained = True
                return
            except Exception as e:
                print(f"[DiskVectorStore] 加载索引失败: {e}，将重建")
        
        # 从 SQLite 重建索引
        self._rebuild_index()
    
    def _get_index_path(self) -> str:
        """获取 FAISS 索引文件路径。"""
        return self.db_path.replace(".db", "_faiss.index")
    
    def _rebuild_index(self) -> None:
        """从 SQLite 重建 FAISS 索引。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT id, vector FROM disk_vectors WHERE deleted = 0"
        )
        
        vectors = []
        ids = []
        for row in cursor:
            vec = np.frombuffer(row[1], dtype=np.float32)
            if len(vec) == self.dim:
                vectors.append(vec)
                ids.append(row[0])
        
        conn.close()
        
        if len(vectors) < self.TRAIN_THRESHOLD:
            # 数据量不足，不训练索引
            self._index_trained = False
            return
        
        # 训练 IVF 索引
        X = np.vstack(vectors).astype(np.float32)
        nlist = min(self.IVF_NLIST, len(vectors) // 10)
        
        quantizer = faiss.IndexFlatL2(self.dim)
        self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
        self._index.train(X)
        self._index.add(X)
        
        self._id_map = ids  # FAISS 内部 ID 与 SQLite ID 的映射
        self._index_trained = True
        
        # 保存索引
        faiss.write_index(self._index, self._get_index_path())
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> List[int]:
        """批量写入向量。
        
        Parameters
        ----------
        vectors : np.ndarray
            向量矩阵，shape (n, dim)
        ids : list[int] | None
            可选的外部 ID。若不提供则使用 SQLite 自增 ID
        
        Returns
        -------
        list[int]
            写入的 ID 列表
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n = vectors.shape[0]
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配: 期望 {self.dim}，实际 {vectors.shape[1]}")
        
        written_ids = []
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            
            for i in range(n):
                vec_bytes = vectors[i].astype(np.float32).tobytes()
                cursor = conn.execute(
                    "INSERT INTO disk_vectors (vector) VALUES (?)",
                    (vec_bytes,)
                )
                written_ids.append(cursor.lastrowid)
            
            conn.commit()
            conn.close()
        
        # 若使用 FAISS 且已训练，追加到索引
        if self.use_faiss and self._index_trained and self._index is not None:
            self._index.add(vectors.astype(np.float32))
            if not hasattr(self, '_id_map'):
                self._id_map = []
            self._id_map.extend(written_ids)
        
        return written_ids
    
    def search(self, query: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[Tuple[int, float]]:
        """粗筛：返回最相似的 top_k 个候选 ID。
        
        Parameters
        ----------
        query : np.ndarray
            查询向量，shape (dim,) 或 (1, dim)
        top_k : int, default 200
            返回候选数
        
        Returns
        -------
        list[tuple[int, float]]
            [(id, 相似度分数), ...]，按相似度降序
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        
        # FAISS 模式
        if self.use_faiss and self._index_trained and self._index is not None:
            return self._search_faiss(query, top_k)
        
        # SQLite 暴力搜索模式
        return self._search_sqlite(query, top_k)
    
    def _search_faiss(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """FAISS 索引搜索。"""
        # FAISS 返回 L2 距离，转换为余弦相似度
        distances, indices = self._index.search(query, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and hasattr(self, '_id_map') and idx < len(self._id_map):
                # L2 距离 → 相似度（近似）
                sim = 1.0 / (1.0 + distances[0][i])
                results.append((self._id_map[idx], sim))
        
        return results
    
    def _search_sqlite(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """SQLite 暴力余弦搜索。"""
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []
        
        query_normalized = query.flatten() / query_norm
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT id, vector FROM disk_vectors WHERE deleted = 0"
        )
        
        candidates = []
        for row in cursor:
            vec_id = row[0]
            vec = np.frombuffer(row[1], dtype=np.float32)
            
            if len(vec) != self.dim:
                continue
            
            # 余弦相似度
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-10:
                continue
            
            sim = float(np.dot(vec, query_normalized) / vec_norm)
            candidates.append((vec_id, sim))
        
        conn.close()
        
        # 按相似度降序排序，取 top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def get_vectors_by_ids(self, ids: List[int]) -> np.ndarray:
        """根据 ID 列表加载向量。
        
        Parameters
        ----------
        ids : list[int]
            向量 ID 列表
        
        Returns
        -------
        np.ndarray
            向量矩阵，shape (len(ids), dim)
        """
        if not ids:
            return np.zeros((0, self.dim), dtype=np.float32)
        
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join("?" * len(ids))
        cursor = conn.execute(
            f"SELECT id, vector FROM disk_vectors WHERE id IN ({placeholders}) AND deleted = 0",
            ids
        )
        
        # 保持顺序
        id_to_vec = {}
        for row in cursor:
            vec = np.frombuffer(row[1], dtype=np.float32)
            if len(vec) == self.dim:
                id_to_vec[row[0]] = vec
        
        conn.close()
        
        # 按 ids 顺序组装矩阵
        vectors = []
        for vid in ids:
            if vid in id_to_vec:
                vectors.append(id_to_vec[vid])
        
        if vectors:
            return np.vstack(vectors).astype(np.float32)
        else:
            return np.zeros((0, self.dim), dtype=np.float32)
    
    def update_vector(self, vector_id: int, new_vector: np.ndarray) -> bool:
        """更新指定ID的向量。"""
        if new_vector is None or len(new_vector) != self.vec_dim:
            return False
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "UPDATE disk_vectors SET vector = ? WHERE id = ? AND deleted = 0",
                (new_vector.tobytes(), vector_id)
            )
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
        
        return success
    
    def remove(self, ids: List[int]) -> int:
        """标记删除向量（不立即物理删除）。
        
        Parameters
        ----------
        ids : list[int]
            要删除的向量 ID 列表
        
        Returns
        -------
        int
            实际标记删除的数量
        """
        if not ids:
            return 0
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                f"UPDATE disk_vectors SET deleted = 1 WHERE id IN ({','.join('?' * len(ids))})",
                ids
            )
            count = cursor.rowcount
            conn.commit()
            conn.close()
        
        return count
    
    def compact(self) -> int:
        """物理删除已标记删除的向量，重建索引。
        
        Returns
        -------
        int
            清理的向量数量
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("DELETE FROM disk_vectors WHERE deleted = 1")
            count = cursor.rowcount
            conn.commit()
            conn.close()
        
        # 重建 FAISS 索引
        if self.use_faiss:
            self._rebuild_index()
        
        return count
    
    def __len__(self) -> int:
        """返回有效向量数量。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM disk_vectors WHERE deleted = 0")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_stats(self) -> dict:
        """返回存储统计信息。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM disk_vectors WHERE deleted = 0")
        active_count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT COUNT(*) FROM disk_vectors WHERE deleted = 1")
        deleted_count = cursor.fetchone()[0]
        conn.close()
        
        return {
            "active_count": active_count,
            "deleted_count": deleted_count,
            "total_count": active_count + deleted_count,
            "dimension": self.dim,
            "mode": "faiss" if (self.use_faiss and self._index_trained) else "sqlite",
            "faiss_available": FAISS_AVAILABLE,
        }


# ===========================================================================
# 自测 Demo
# ===========================================================================
if __name__ == "__main__":
    import tempfile
    
    print("=" * 60)
    print("  DiskVectorStore 自测 Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_vectors.db")
        
        # 初始化
        store = DiskVectorStore(db_path, dim=128, use_faiss=False)
        print(f"\n1. 初始化完成: {store.get_stats()}")
        
        # 写入测试向量
        np.random.seed(42)
        vectors = np.random.randn(100, 128).astype(np.float32)
        ids = store.add(vectors)
        print(f"\n2. 写入 100 条向量，ID 范围: {min(ids)} - {max(ids)}")
        
        # 搜索测试
        query = vectors[0]  # 用第一条向量查询
        results = store.search(query, top_k=5)
        print(f"\n3. 搜索 top-5 结果:")
        for i, (vid, sim) in enumerate(results):
            print(f"   [{i+1}] ID={vid}, sim={sim:.4f}")
        
        # 验证第一条应该是最相似的
        assert results[0][0] == ids[0], f"第一条应是最相似的"
        print("   [OK] search correct")
        
        # 加载向量测试
        loaded = store.get_vectors_by_ids([ids[0], ids[1]])
        print(f"\n4. 加载 2 条向量: shape={loaded.shape}")
        assert loaded.shape == (2, 128)
        print("   [OK] vector load correct")
        
        # 标记删除测试
        removed = store.remove([ids[0]])
        print(f"\n5. 标记删除 ID={ids[0]}: {removed} 条")
        assert len(store) == 99
        print("   [OK] count after delete correct")
        
        # 统计
        print(f"\n6. 最终统计: {store.get_stats()}")
        
        print("\n" + "=" * 60)
        print("  [OK] All tests passed")
        print("=" * 60)
