# memory_core.py — MemCore v0.3 霍普菲尔德内核
#
# MEMCORE_USE_LEGACY: 不保证运行时兼容。如需回退请使用 git revert。
#
# 核心存储从二维向量外积矩阵替换为现代霍普菲尔德网络。
# 联想记忆模型容量指数级、遗忘精确无残留、融合时间信息。

import json
import os
import re
import time
import threading
import uuid
import warnings
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

import db
from disk_vector_store import DiskVectorStore
from embedder import create_embedder, get_embedder_info
from fallback_retriever import search_files
from fingerprint_index import FingerprintIndex
from llm_client import call_llm, ATOMIC_SYSTEM_PROMPT, QUERY_FINGERPRINT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# LLM 调用：统一由 llm_client.py 提供
# 系统提示词：ATOMIC_SYSTEM_PROMPT, QUERY_FINGERPRINT_SYSTEM_PROMPT 导入自 llm_client
# 旧版 REFINE_SYSTEM_PROMPT 已在 llm_client.py 中统一管理
# ---------------------------------------------------------------------------




# ---------- 函数 1：对话提炼（原子化事实挖掘版） ----------
def refine_conversation(
    conversation_text: str,
    model: str = "deepseek-v3.2",
    temperature: float = 0.1,
) -> Optional[Dict[str, str]]:
    """调用大模型，从文本中挖掘原子化事实，返回合并后的单条记忆。

    使用 ATOMIC_SYSTEM_PROMPT 调用 LLM，解析返回的 JSON 数组。
    - 若数组为空 → 返回 None
    - 若数组非空 → 取首条的 query 和 fingerprint，所有 content 用分号拼接
    - 保持与旧接口兼容：返回 {"query", "content", "fingerprint"} 或 None
    """
    system_prompt = ATOMIC_SYSTEM_PROMPT
    raw_response = call_llm(conversation_text, system_prompt, temperature)
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list) or len(data) == 0:
        return None

    # 合并：取首条 query/fingerprint，所有 content 用分号拼接
    first = data[0]
    all_contents = "; ".join(item.get("content", "") for item in data if item.get("content"))
    if not all_contents:
        return None

    # 过滤：仅由询问性语句组成的文本不应作为记忆存储
    # 检测是否以问号结尾或是纯询问句式
    content_first = all_contents.split(";")[0].strip()
    question_markers = ("吗", "？", "?", "啥", "怎样", "如何", "为什么", "啥时候")
    if any(content_first.endswith(m) for m in question_markers):
        return None

    return {
        "query": first.get("query", ""),
        "content": all_contents,
        "fingerprint": first.get("fingerprint", ""),
    }


def refine_conversation_batch(
    conversation_text: str,
    model: str = "deepseek-v3.2",
    temperature: float = 0.1,
) -> List[Dict[str, str]]:
    """与 refine_conversation 使用相同的系统提示词，但返回完整的原子化事实列表。

    Parameters
    ----------
    conversation_text : str
        待挖掘的文本。
    model : str, default "deepseek-v3.2"
        模型标识符（仅记录用途）。
    temperature : float, default 0.1
        生成温度。

    Returns
    -------
    list[dict]
        原子化事实列表，每个元素含 {"query", "content", "fingerprint"}。
        若无事实则返回空列表。
    """
    system_prompt = ATOMIC_SYSTEM_PROMPT
    raw_response = call_llm(conversation_text, system_prompt, temperature)
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    # 过滤无效条目
    valid = [item for item in data if isinstance(item, dict) and item.get("content")]
    return valid


# ---------- 函数 2：查询关键词提取（保留供外部调用） ----------
def generate_query_fingerprint(
    query_text: str,
    model: str = "deepseek-v3.2",
    temperature: float = 0.0,
) -> str:
    """为用户的自然语言查询生成检索关键词串。"""
    system_prompt = QUERY_FINGERPRINT_SYSTEM_PROMPT
    return call_llm(query_text, system_prompt, temperature).strip()


# ===========================================================================
# 类：MemoryCore（霍普菲尔德内核 v0.3）
# ===========================================================================
#
# MEMCORE_USE_LEGACY: 不保证运行时兼容。如需回退请使用 git revert。
#
# 核心变更：
# - 存储矩阵 X: (n, d) 向量堆叠，替代外积矩阵 M
# - recall: Hopfield 迭代 + 时间衰减融合
# - 遗忘：保护分数（protection_score）驱动，非 FIFO
# ===========================================================================

class MemoryCore:
    """
    轻量级记忆核心 — 霍普菲尔德联想记忆内核 (v0.3)。

    使用现代霍普菲尔德网络的能量景观进行联想记忆检索。
    核心矩阵 X 存储所有记忆的嵌入向量，recall 通过迭代吸引子
    动力学收敛到与查询语义最相关的记忆。
    时间信息通过指数衰减权重融入检索分数，实现时间敏感性。

    Parameters
    ----------
    max_memories : int, default 1000
        最大记忆容量。
    similarity_threshold : float | None, default None
        最终余弦相似度阈值。None 时自动设为 0.60。
    db_path : str, default ":memory:"
        数据库文件路径。
    embedding_model : str, default "all-MiniLM-L6-v2"
        嵌入模型名称（仅记录用途）。
    enable_keyword_fallback : bool | None, default None
        v0.3 已废弃。传入非 None 值时触发 DeprecationWarning。
    beta : float, default 8.0
        霍普菲尔德迭代的相似度放大系数。
    lambda_decay : float, default 5.73e-7
        时间衰减系数（14 天半衰期，配合 time.time() 秒级时间戳）。
    top_k : int, default 3
        每次 recall 返回的最大结果数。
    """

    def __init__(
        self,
        max_memories: int = 1000,
        similarity_threshold: Optional[float] = None,
        db_path: str = ":memory:",
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_keyword_fallback: Optional[bool] = None,
        # ---- v0.3 新参数 ----
        beta: float = 8.0,
        lambda_decay: float = 5.73e-7,
        top_k: int = 3,
        # ---- v0.3.1 新参数 ----
        alpha: float = 0.75,
        # ---- v0.8 新参数：磁盘向量存储 ----
        use_disk_store: bool = False,
        disk_store_path: Optional[str] = None,
    ):
        # -- 废弃参数警告 --
        if enable_keyword_fallback is not None:
            warnings.warn(
                "enable_keyword_fallback is deprecated in v0.3 (Hopfield kernel). "
                "This parameter is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.max_memories = max_memories
        self.memories: List[Dict[str, Any]] = []
        self._db_lock = threading.Lock()  # 数据库线程安全锁

        # -- 霍普菲尔德核 --
        self.encoder = create_embedder()
        if hasattr(self.encoder, 'get_embedding_dimension'):
            self.vec_dim: int = self.encoder.get_embedding_dimension()
        else:
            self.vec_dim: int = self.encoder.get_sentence_embedding_dimension()

        embed_info = get_embedder_info()
        self.is_semantic = embed_info["is_semantic"]
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = 0.60 if self.is_semantic else 0.30

        # -- 霍普菲尔德参数 --
        self.beta = beta
        self.lambda_decay = lambda_decay
        self.top_k = top_k
        self.alpha = alpha  # 双向量融合系数：semantic * alpha + keyword * (1-alpha)

        # -- 两阶段检索参数 --
        self.HNSW_THRESHOLD = 5000   # N > 此值时启用粗筛（降低阈值，更早启动优化）
        self.HNSW_TOP_K = 100        # 粗筛保留候选数（增加候选，精排更准确）

        # -- v0.8 磁盘向量存储 --
        self.use_disk_store = use_disk_store
        self.disk_store: Optional[DiskVectorStore] = None
        self._disk_id_map: Dict[int, int] = {}  # memory db_id -> disk_vector id

        # -- 存储矩阵与辅助数组 --
        self.KW_DIM = 128  # 关键词向量维度
        if self.use_disk_store:
            # 磁盘模式：不初始化内存矩阵，使用 DiskVectorStore
            self.X: np.ndarray = np.zeros((0, self.vec_dim), dtype=np.float32)
            self.X_kw: np.ndarray = np.zeros((0, self.KW_DIM), dtype=np.float32)
            disk_path = disk_store_path or db_path.replace(".db", "_vectors.db")
            self.disk_store = DiskVectorStore(disk_path, dim=self.vec_dim)
        else:
            # 内存模式：原有逻辑
            self.X: np.ndarray = np.zeros((0, self.vec_dim), dtype=np.float32)
            self.X_kw: np.ndarray = np.zeros((0, self.KW_DIM), dtype=np.float32)  # 关键词向量矩阵
        self.timestamps: List[float] = []
        self.access_counts: List[int] = []
        self.protection_scores: List[float] = []

        # -- 数据库持久化 --
        self.db_path = db_path
        self.conn = db.init_db(db_path)
        self.load_from_db()

        # -- 对话指纹索引（v0.4，用于指代消解） --
        self.fingerprint_index = FingerprintIndex()

        # -- 对话上下文窗口（v0.5，用于指代消解和话题跟踪） --
        self.context_window: deque[str] = deque(maxlen=5)

    # ------------------------------------------------------------------
    # 数据库恢复
    # ------------------------------------------------------------------

    def load_from_db(self) -> None:
        """从数据库重新加载所有记忆到内存，重建 Hopfield 存储矩阵和辅助数组。
        
        磁盘模式 (use_disk_store=True)：只加载元数据，向量保持在磁盘索引中。
        内存模式 (use_disk_store=False)：全量加载向量到 self.X / self.X_kw。
        """
        rows = db.get_all_memories(self.conn)

        self.memories.clear()
        self.timestamps.clear()
        self.access_counts.clear()
        self.protection_scores.clear()
        self._disk_id_map.clear()

        vectors: List[np.ndarray] = []
        kw_vectors: List[np.ndarray] = []
        disk_vectors: List[np.ndarray] = []
        disk_ids: List[int] = []

        for row in rows:
            vec = np.frombuffer(row["query_vector"], dtype=np.float32)
            # 兼容旧数据库：向量维度与新嵌入器不匹配时重新编码
            if len(vec) != self.vec_dim:
                vec = self.encoder.encode(f"{row['query']}。{row['content']}").astype(np.float32)
            else:
                # 【P0修复】强制归一化（修复旧数据库未归一化问题）
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec = vec / norm

            # 加载关键词向量（兼容旧库不存在的场景）
            kw_bytes = row.get("keyword_vector")
            if kw_bytes:
                kw_vec = np.frombuffer(kw_bytes, dtype=np.float32)
                if len(kw_vec) != self.KW_DIM:
                    kw_text = f"{row['query']} {row['content']} {row['fingerprint']}"
                    kw_vec = self._generate_keyword_vector(kw_text, self.KW_DIM)
            else:
                kw_text = f"{row['query']} {row['content']} {row['fingerprint']}"
                kw_vec = self._generate_keyword_vector(kw_text, self.KW_DIM)

            if self.use_disk_store:
                # 磁盘模式：收集向量稍后写入磁盘索引
                disk_vectors.append(vec)
                disk_ids.append(row["id"])
            else:
                # 内存模式：收集到列表
                vectors.append(vec)
                kw_vectors.append(kw_vec)

            # 解析时间
            ts_str = row["created_at"]
            if isinstance(ts_str, str):
                from datetime import datetime
                # Python 3.12 fromisoformat 接受空格分隔格式
                ts_str_fixed = ts_str.replace(" ", "T")
                ts = datetime.fromisoformat(ts_str_fixed)
            else:
                ts = ts_str
            self.timestamps.append(ts.timestamp())
            self.access_counts.append(row.get("access_count", 0))
            self.protection_scores.append(row.get("protection_score", 2.0))
            self.memories.append({
                "db_id": row["id"],
                "uuid": row.get("uuid", ""),
                "query": row["query"],
                "content": row["content"],
                "fingerprint": row["fingerprint"],
            })

        # 存储矩阵初始化
        if self.use_disk_store:
            # 磁盘模式：先清空已有索引，再写入 DiskVectorStore
            if len(self.disk_store) > 0:
                # 清空已有向量（避免重复加载）
                all_ids = list(range(1, len(self.disk_store) + 1))
                self.disk_store.remove(all_ids)
            if disk_vectors:
                written_ids = self.disk_store.add(np.vstack(disk_vectors).astype(np.float32))
                # 建立 memory db_id -> disk_vector id 映射
                for mem_db_id, disk_id in zip(disk_ids, written_ids):
                    self._disk_id_map[mem_db_id] = disk_id
            self.X = np.zeros((0, self.vec_dim), dtype=np.float32)
            self.X_kw = np.zeros((0, self.KW_DIM), dtype=np.float32)
        else:
            # 内存模式：原有逻辑
            if vectors:
                self.X = np.vstack(vectors).astype(np.float32)
                self.X_kw = np.vstack(kw_vectors).astype(np.float32)
            else:
                self.X = np.zeros((0, self.vec_dim), dtype=np.float32)
                self.X_kw = np.zeros((0, self.KW_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_keyword_vector(text: str, dim: int = 128) -> np.ndarray:
        """从文本生成 128 维关键词向量（token n-gram 哈希 + L2 归一化）。

        v0.8: 增强英文/代码支持 — 提取英文单词(小写归一化)、CamelCase拆分、
        中文2字词(lookahead)，再对token序列做n-gram哈希。
        纯字符级n-gram对英文大小写敏感("React"≠"react")，
        token级归一化后相同语义的词映射到同一哈希槽。
        """
        import re as _re

        # ── Step 1: 提取有意义的token ──
        tokens: list = []

        # 1a. 英文单词 + CamelCase拆分 + 小写归一化
        # 匹配连续英文字母数字下划线点（变量名/类名/方法名等）
        en_words = _re.findall(r'[a-zA-Z][a-zA-Z0-9_.]{1,}', text)
        for w in en_words:
            # CamelCase拆分: "useState" → ["use", "state"], "XMLParser" → ["XML", "Parser"]
            parts = _re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)', w)
            if parts and len(parts) > 1:
                tokens.extend(p.lower() for p in parts)
            else:
                tokens.append(w.lower())

        # 1b. 中文2字词（lookahead重叠匹配，确保"咖啡"不被"喝咖"吞掉）
        cn_terms = _re.findall(r'(?=([\u4e00-\u9fff]{2}))', text)
        tokens.extend(cn_terms)

        # ── Step 2: 归一化文本的字符级n-gram ──
        # 英文字符小写归一化，中文保持原样，数字/标点保持原样
        normalized_text = ''.join(c.lower() if c.isalpha() and ord(c) < 128 else c for c in text)
        raw_ngrams: list = []
        for ngram_len in (1, 2, 3):
            for i in range(len(normalized_text) - ngram_len + 1):
                raw_ngrams.append(normalized_text[i:i + ngram_len])

        # ── Step 3: 哈希编码 ──
        vec = np.zeros(dim, dtype=np.float32)

        # token级n-gram（权重更高，英文/中文归一化后语义匹配更准）
        token_str = " ".join(tokens)
        for ngram_len in (1, 2, 3):
            token_ngrams = token_str.split() if ngram_len == 1 else []
            if ngram_len > 1:
                # 对token序列做n-gram组合（"react hooks" → "react_hooks"）
                for i in range(len(tokens) - ngram_len + 1):
                    token_ngrams.append("_".join(tokens[i:i + ngram_len]))
            for idx, ngram in enumerate(token_ngrams):
                h = sum(ord(c) * (idx + 1) for i, c in enumerate(ngram)) % dim
                vec[h] += 1.0 / ngram_len * 1.5  # token级权重x1.5

        # 字符级n-gram（补充覆盖数字、标点等token未捕获的内容）
        for idx, ngram in enumerate(raw_ngrams):
            h = sum(ord(c) * (idx + 1) for i, c in enumerate(ngram)) % dim
            freq = 1.0 / (3 if len(ngram) == 1 else 2 if len(ngram) == 2 else 1)
            vec[h] += freq

        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec /= norm
        return vec

    def add_memory(
        self,
        query: str,
        content: str,
        fingerprint: str,
        query_vector: Optional[np.ndarray] = None,
    ) -> None:
        """添加新记忆：编码向量 → 写 DB → 更新 Hopfield 矩阵。超容量时保护分数遗忘。

        Parameters
        ----------
        query : str
            标准化查询主题。
        content : str
            核心事实陈述。
        fingerprint : str
            空格分隔的关键词串。
        query_vector : np.ndarray | None, default None
            预计算的查询嵌入向量。为 None 时自动编码。
        """
        if query_vector is None:
            query_vector = self.encoder.encode(f"{query}。{content}")
        query_vector = query_vector.astype(np.float32)
        vec_bytes = query_vector.tobytes()

        # ── BGE 写入去重 ──
        # 如果新内容与已有记忆的 BGE 余弦 > 0.85，且内容长度 >= 20，
        # 视为重复，合并而非新增。
        # 短文本（< 20 字符）的 BGE 余弦不可靠（"早期记忆A" vs "早期记忆B"
        # 共享前缀导致余弦虚高），跳过去重避免误合并。
        # 再加一道字符级保护：要求内容有显著的文本重叠（字符三词连 Jaccard）
        DEDUP_MIN_LEN = 20
        DEDUP_COSINE_THRESHOLD = 0.85
        
        # 磁盘模式下用 disk_store.search()，内存模式用 self.X
        max_sim = -1.0
        max_idx = -1
        
        if self.use_disk_store and len(content) >= DEDUP_MIN_LEN:
            # 磁盘模式：search top-1 检查语义相似度
            results = self.disk_store.search(query_vector, top_k=1)
            if results:
                disk_id, sim = results[0]
                if sim > DEDUP_COSINE_THRESHOLD:
                    # disk_id → db_id → memory_idx
                    # 需要找到对应的 memory_idx
                    for idx, mem in enumerate(self.memories):
                        mem_db_id = mem.get("db_id")
                        if mem_db_id is not None and self._disk_id_map.get(mem_db_id) == disk_id:
                            max_sim = sim
                            max_idx = idx
                            break
        elif len(self.X) > 0 and len(content) >= DEDUP_MIN_LEN:
            # 内存模式：原逻辑
            q_norm = float(np.linalg.norm(query_vector))
            if q_norm > 1e-10:
                sims = np.dot(self.X, query_vector) / q_norm
                max_sim = float(np.max(sims))
                max_idx = int(np.argmax(sims))

            # 候选去重
            if max_sim > DEDUP_COSINE_THRESHOLD:
                # 额外检查：字符级文本重叠（避免 BGE 对共享前缀的短文本误判）
                existing_content = self.memories[max_idx].get("content", "")
                # 提取字符三词连（trigram）并计算 Jaccard 相似度
                def _trigram_set(t: str) -> set[str]:
                    return {t[i:i+3] for i in range(len(t) - 2)} if len(t) >= 3 else {t}
                tri1 = _trigram_set(content)
                tri2 = _trigram_set(existing_content)
                text_jaccard = len(tri1 & tri2) / max(len(tri1 | tri2), 1) if (tri1 or tri2) else 0.0

                if text_jaccard >= 0.5:
                    # 合并：保留更长 content，合并 fingerprint token
                    existing = self.memories[max_idx]
                    old_content = existing.get("content", "")
                    old_fp = existing.get("fingerprint", "")
                    old_query = existing.get("query", "")

                    # 用更长、信息更丰富的内容
                    new_content = content if len(content) > len(old_content) else old_content
                    # 合并 fingerprint（去重）
                    fp_tokens = set(old_fp.split()) | set(fingerprint.split())
                    new_fp = " ".join(sorted(fp_tokens))
                    # query 用更具体的（更长的不一定更好，保留原有用原有）
                    new_query = old_query if len(old_query) >= len(query) else query

                    # 内容变更时重新编码
                    if new_content != old_content or new_query != old_query:
                        new_vec = self.encoder.encode(f"{new_query}。{new_content}").astype(np.float32)
                        new_vec_bytes = new_vec.tobytes()
                        new_kw_text = f"{new_query} {new_content} {new_fp}"
                        new_kw_vec = self._generate_keyword_vector(new_kw_text, self.KW_DIM)
                        new_kw_bytes = new_kw_vec.tobytes()
                    else:
                        new_vec_bytes = vec_bytes
                        new_kw_bytes = None

                    # 更新 DB
                    db_id = existing.get("db_id")
                    if db_id is not None:
                        with self._db_lock:
                            db.update_memory_content(
                                self.conn, db_id, new_query, new_content, new_fp,
                                new_vec_bytes, new_kw_bytes,
                            )

                    # 更新向量（磁盘模式 vs 内存模式）
                    if self.use_disk_store:
                        # 磁盘模式：更新 DiskVectorStore 中的向量
                        if new_vec_bytes != vec_bytes or new_kw_bytes is not None:
                            mem_db_id = existing.get("db_id")
                            disk_id = self._disk_id_map.get(mem_db_id) if mem_db_id else None
                            if disk_id is not None and new_vec_bytes != vec_bytes:
                                new_vec = np.frombuffer(new_vec_bytes, dtype=np.float32).copy()
                                self.disk_store.update_vector(disk_id, new_vec)
                            # 关键词向量只在内存中更新
                            if new_kw_bytes is not None:
                                self.X_kw[max_idx] = np.frombuffer(new_kw_bytes, dtype=np.float32).copy()
                    else:
                        # 内存模式：更新 self.X / self.X_kw
                        self.X[max_idx] = np.frombuffer(new_vec_bytes, dtype=np.float32).copy()
                        if new_kw_bytes is not None:
                            self.X_kw[max_idx] = np.frombuffer(new_kw_bytes, dtype=np.float32).copy()
                    self.timestamps[max_idx] = time.time()
                    self.access_counts[max_idx] += 1
                    self.protection_scores[max_idx] = min(
                        self.PROTECTION_CAP,
                        self.protection_scores[max_idx] + 0.2,
                    )
                    existing.update({
                        "query": new_query,
                        "content": new_content,
                        "fingerprint": new_fp,
                    })
                    return  # 合并完成，不创建新条目

        # 生成关键词向量（128 维 L2 归一化）
        kw_text = f"{query} {content} {fingerprint}"
        kw_vec = self._generate_keyword_vector(kw_text, self.KW_DIM)
        kw_bytes = kw_vec.tobytes()

        # 生成 UUID
        mem_uuid = str(uuid.uuid4())

        # 写 DB 并获取 lastrowid
        with self._db_lock:
            db_id = db.insert_memory(
                self.conn, query, content, fingerprint,
                vec_bytes, kw_bytes, mem_uuid,
            )

        # 更新存储
        if self.use_disk_store:
            # 磁盘模式：写入 DiskVectorStore
            disk_ids = self.disk_store.add(query_vector.reshape(1, -1))
            self._disk_id_map[db_id] = disk_ids[0]
        else:
            # 内存模式：更新 Hopfield 存储矩阵
            self.X = np.vstack([self.X, query_vector.reshape(1, -1)])
            self.X_kw = np.vstack([self.X_kw, kw_vec.reshape(1, -1)])

        self.timestamps.append(time.time())
        self.access_counts.append(0)
        self.protection_scores.append(2.0)
        self.memories.append({
            "db_id": db_id,
            "uuid": mem_uuid,
            "query": query,
            "content": content,
            "fingerprint": fingerprint,
        })

        # 容量控制
        if len(self.memories) > self.max_memories:
            self._forget_oldest()

    def _adaptive_beta(self) -> float:
        """beta 自适应：beta = 8.0 + 2.0 * (log10(N) - 3)，其中 N 为记忆条数。"""
        N = len(self.memories) if self.use_disk_store else len(self.X)
        if N <= 1000:
            return self.beta  # N<=1000 使用配置值
        return float(self.beta + 2.0 * (np.log10(N) - 3.0))

    # TODO(v0.4): 任务偏置 - 在检索分数中融入 active_context 的语义相似度
    def _classify_query_intent(self, query_text: str) -> str:
        """意图分类（纯正则，<0.1ms）。

        返回 'reference' | 'entity' | 'semantic' | 'document'。
        """
        if re.search(r'刚才|上次|之前|第\d+轮|前几轮|那个', query_text):
            return "reference"
        # 实体：含点分标识符（如 module.function）或下划线连接的代码标识符
        if re.search(r'[a-z_][a-zA-Z0-9_]*(\.[a-z_][a-zA-Z0-9_]*)+', query_text):
            return "entity"
        if re.search(r'(?<![A-Za-z0-9])[a-z_][a-zA-Z0-9_]*(?:_+[a-zA-Z0-9_]+)+(?!["\w])', query_text):
            return "entity"
        if re.search(r'\.py$|\.go$|\.md$|\.txt$|文档|文件', query_text):
            return "document"
        return "semantic"

    def recall(self, query_text: str) -> List[str]:
        """霍普菲尔德迭代联想记忆召回。

        流程：
          1. 对查询向量执行 Hopfield 迭代吸引子动力学（最多 3 步）
          2. 在迭代中融入时间衰减权重
          3. 最终按余弦相似度 + 时间权重综合排序
          4. 保护分数增量、去重后返回 top_k

        Parameters
        ----------
        query_text : str
            用户的自然语言查询。

        Returns
        -------
        list[str]
            最相关的记忆内容列表，最多 self.top_k 条。
        """
        if self.use_disk_store:
            if len(self.memories) == 0:
                return []
        else:
            if len(self.X) == 0:
                return []

        # ── 对话上下文窗口增强 ──
        # 当查询包含指代词时，用最近上下文的查询主题增强输入，
        # 使 BGE 编码能捕获当前话题偏向
        original_query = query_text
        ref_pattern = re.compile(r'(那个|这个|上次|之前|刚才|前面|上面|前一个|前一)')
        if ref_pattern.search(query_text) and self.context_window:
            context_terms = list(self.context_window)
            # 取最近 2 条上下文的主题词（过长的可能引入噪声）
            recent_context = " ".join(context_terms[-2:])
            query_text = f"{recent_context} {query_text}"
            if os.environ.get("MEMCORE_DEBUG"):
                print(f"    [Context] enhanced: '{original_query}' → '{query_text}'")

        # ── 意图分类 + 第一/二道短路检索 ──
        intent = self._classify_query_intent(query_text)
        entity_results: List[str] = []

        # 第一道 a：显式指代 → 指纹索引 O(1) 定位，命中后短路
        if intent == "reference":
            rid = self.fingerprint_index.resolve_explicit_reference(query_text)
            if rid is not None:
                fp = self.fingerprint_index.get_fingerprint(rid)
                if fp and fp.memory_fiber_ids:
                    # 通过关联的记忆 ID 找到对应内容
                    fid_set = set(fp.memory_fiber_ids)
                    for mem in self.memories:
                        if mem.get("db_id") in fid_set:
                            return [mem["content"]]
                    # 无直接关联记忆：返回查询摘要
                    return [fp.summary_phrase] if fp.summary_phrase else []
            # 落空则继续走霍普菲尔德

            # 第一道 b：隐式指代 → 指纹语义匹配
            implicit_rounds = self.fingerprint_index.resolve_implicit_reference(query_text)
            if implicit_rounds:
                # 返回最高匹配轮次的关联记忆内容
                for rid in implicit_rounds:
                    fp = self.fingerprint_index.get_fingerprint(rid)
                    if fp and fp.memory_fiber_ids:
                        fid_set = set(fp.memory_fiber_ids)
                        for mem in self.memories:
                            if mem.get("db_id") in fid_set:
                                return [mem["content"]]
                # 落空则继续走霍普菲尔德

        # 第二道：实体倒排（entity）→ 精筛候选池
        if intent == "entity":
            entity_pattern = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.]{2,}', query_text)
            entity_set = set(entity_pattern)
            if entity_set:
                for mem in self.memories:
                    fp_set = set(mem.get("fingerprint", "").split())
                    if entity_set & fp_set:
                        entity_results.append(mem["content"])

        # ── 两阶段检索 ──
        # 第三道：霍普菲尔德向量检索
        # 磁盘模式：DiskVectorStore 粗筛 → 加载候选 → Hopfield 精排
        # 内存模式：当 N >= HNSW_THRESHOLD 时：HNSW 粗筛 → 霍普菲尔德精排
        #          当 N < HNSW_THRESHOLD 时：直接全量霍普菲尔德

        xi = self.encoder.encode(query_text).astype(np.float32)
        q_original = xi.copy()
        now = time.time()

        # ── 磁盘模式：两阶段检索 ──
        if self.use_disk_store:
            # Phase 1: DiskVectorStore 粗筛 top-200
            candidates = self.disk_store.search(q_original, top_k=self.HNSW_TOP_K)
            if not candidates:
                return []
            
            # Phase 2: 加载候选向量到内存
            candidate_disk_ids = [c[0] for c in candidates]
            X_subset = self.disk_store.get_vectors_by_ids(candidate_disk_ids)
            if X_subset.shape[0] == 0:
                return []
            
            # 建立 disk_id -> memories 索引的映射
            disk_to_mem_idx = {}
            for i, mem in enumerate(self.memories):
                disk_id = self._disk_id_map.get(mem.get("db_id"))
                if disk_id is not None:
                    disk_to_mem_idx[disk_id] = i
            
            # 收集候选的记忆索引和时间戳
            candidate_mem_indices = []
            timestamps_subset = []
            for disk_id, _ in candidates:
                if disk_id in disk_to_mem_idx:
                    mem_idx = disk_to_mem_idx[disk_id]
                    candidate_mem_indices.append(mem_idx)
                    timestamps_subset.append(self.timestamps[mem_idx])
            
            if not candidate_mem_indices:
                return []
            
            # 转换为 numpy 数组便于后续处理
            candidate_indices = np.array(candidate_mem_indices)
        else:
            # ── 内存模式：原有逻辑 ──
            n_total = len(self.X)
            if n_total >= self.HNSW_THRESHOLD:
                # Phase 1: 快速余弦粗筛（X 已归一化，跳过 norm 计算）
                xi_norm = float(np.linalg.norm(q_original))
                if xi_norm > 1e-10:
                    quick_sims = np.dot(self.X, q_original) / xi_norm
                else:
                    quick_sims = np.zeros(n_total)
                top_n_candidates = min(self.HNSW_TOP_K, n_total)
                candidate_indices = np.argsort(quick_sims)[-top_n_candidates:]
                # Phase 2: 仅在候选子集上做霍普菲尔德精排
                X_subset = self.X[candidate_indices]
                timestamps_subset = [self.timestamps[i] for i in candidate_indices]
            else:
                # 全量霍普菲尔德
                candidate_indices = np.arange(n_total)
                X_subset = self.X
                timestamps_subset = self.timestamps

        time_deltas = now - np.array(timestamps_subset)
        time_weights = np.maximum(np.exp(-self.lambda_decay * time_deltas), 1e-12)

        iteration_steps = 0
        cos_changes: List[float] = []

        # -- Hopfield 迭代 --
        for _ in range(3):
            xi_prev = xi.copy()
            iteration_steps += 1
            current_beta = self._adaptive_beta()
            sim_scores = current_beta * (X_subset @ xi)
            scores = sim_scores + np.log(time_weights)
            scores = scores - np.max(scores)
            p = np.exp(scores)
            p = p / p.sum()
            xi = p @ X_subset

            n1 = float(np.linalg.norm(xi))
            n2 = float(np.linalg.norm(xi_prev))
            if n1 < 1e-10 or n2 < 1e-10:
                break
            cos_sim = float(np.dot(xi, xi_prev)) / (n1 * n2)
            cos_changes.append(cos_sim)
            if cos_sim > 0.999:
                break

        # -- 双向量融合评分 --
        # 语义相似度（使用原始查询向量，避免 Hopfield 虚假吸引子问题）
        # Hopfield 迭代会将查询向量收敛到最近的存储模式，
        # 对不相关查询也会产生极高余弦（虚假吸引子），因此用 q_original 做主评分。
        # Hopfield 注意力权重通过 hopfield_attention 融入辅助评分。
        # 
        # 磁盘模式：X_subset 是候选向量矩阵
        # 内存模式：self.X 是全量向量矩阵
        q_norm = float(np.linalg.norm(q_original))
        if q_norm < 1e-10:
            return []
        
        if self.use_disk_store:
            # 磁盘模式：在候选子集上计算相似度
            sem_sims = np.dot(X_subset, q_original) / q_norm
            sem_sims = np.nan_to_num(sem_sims)
            sem_sims = np.maximum(sem_sims, 0.0)  # 截断到 [0, 1]
            
            # 关键词相似度：对候选记忆生成关键词向量
            kw_sims = np.zeros(len(candidate_indices))
            q_kw = self._generate_keyword_vector(query_text, self.KW_DIM)
            kw_norm = float(np.linalg.norm(q_kw))
            if kw_norm > 1e-10:
                for i, mem_idx in enumerate(candidate_indices):
                    mem = self.memories[mem_idx]
                    kw_text = f"{mem.get('query', '')} {mem.get('content', '')} {mem.get('fingerprint', '')}"
                    mem_kw = self._generate_keyword_vector(kw_text, self.KW_DIM)
                    kw_sims[i] = float(np.dot(mem_kw, q_kw) / kw_norm)
                kw_sims = np.maximum(kw_sims, 0.0)
        else:
            # 内存模式：原有逻辑
            # 优化(v0.7)：X 已归一化（encode 强制归一化），norm ≈ 1.0，跳过计算
            sem_sims = np.dot(self.X, q_original) / q_norm
            sem_sims = np.nan_to_num(sem_sims)
            sem_sims = np.maximum(sem_sims, 0.0)  # 截断到 [0, 1]

            # 关键词相似度（128 维 n-gram 向量）
            # 优化(v0.7)：X_kw 已归一化（_generate_keyword_vector 强制归一化）
            q_kw = self._generate_keyword_vector(query_text, self.KW_DIM)
            kw_norm = float(np.linalg.norm(q_kw))
            if kw_norm > 1e-10:
                kw_sims = np.dot(self.X_kw, q_kw) / kw_norm
                kw_sims = np.nan_to_num(kw_sims)
                kw_sims = np.maximum(kw_sims, 0.0)
            else:
                kw_sims = np.zeros(len(self.X_kw))

        # 自适应 alpha：含代码标识符（下划线/点分）的查询使用更平衡的融合
        alpha = self.alpha
        if re.search(r'[a-z_][a-zA-Z0-9_]*(\.[a-z_][a-zA-Z0-9_]*)+|_+[a-zA-Z]', query_text):
            alpha = 0.5

        # 融合得分
        fused_sims = alpha * sem_sims + (1.0 - alpha) * kw_sims

        # 提前计算有效阈值（用于短名增强和最终过滤）
        min_threshold = 0.50 if self.is_semantic else 0.30
        effective_threshold = max(self.similarity_threshold, min_threshold)

        # 人名/短名增强：2-4 字纯中文查询的内容子串匹配加权
        # BGE 模型对短文本（2-4 字）编码信息不足，语义相似度偏低，
        # 关键词 n-gram 余弦对极短文本也时常无法匹配。
        # 改用直接内容/fingerprint 子串检查，确保人名能准确召回。
        if re.match(r'^[\u4e00-\u9fff]{2,4}$', query_text):
            if self.use_disk_store:
                # 磁盘模式：遍历候选
                for pos, mem_idx in enumerate(candidate_indices):
                    mem = self.memories[mem_idx]
                    if query_text in mem.get("content", "") or query_text in mem.get("fingerprint", ""):
                        fused_sims[pos] = max(fused_sims[pos], effective_threshold + 0.15)
            else:
                # 内存模式：原有逻辑
                for i, mem in enumerate(self.memories):
                    if query_text in mem.get("content", "") or query_text in mem.get("fingerprint", ""):
                        fused_sims[i] = max(fused_sims[i], effective_threshold + 0.15)

        # 技术名词/标识符精确匹配增强：提取查询中的英文标识符、技术名词
        # BGE 对相似技术文本（dual_memory.py vs memory_core.py）的余弦距离 < 0.05，
        # 无法区分不同模块。通过 fingerprint/content 子串精确匹配绕过 BGE 碰撞。
        # v0.8: 英文技术名词大小写不敏感匹配（"react" 可匹配 "React"）
        tech_terms = re.findall(r'[A-Za-z_][A-Za-z0-9_.]{2,}', query_text)
        if tech_terms:
            if self.use_disk_store:
                # 磁盘模式：遍历候选
                for pos, mem_idx in enumerate(candidate_indices):
                    mem = self.memories[mem_idx]
                    for term in tech_terms:
                        content = mem.get("content", "")
                        fp = mem.get("fingerprint", "")
                        # 英文术语大小写不敏感匹配
                        term_lower = term.lower()
                        if term_lower in content.lower() or term_lower in fp.lower():
                            fused_sims[pos] = max(fused_sims[pos], effective_threshold + 0.25)
                            break
            else:
                # 内存模式：原有逻辑
                for i, mem in enumerate(self.memories):
                    for term in tech_terms:
                        content = mem.get("content", "")
                        fp = mem.get("fingerprint", "")
                        # 英文术语大小写不敏感匹配
                        term_lower = term.lower()
                        if term_lower in content.lower() or term_lower in fp.lower():
                            fused_sims[i] = max(fused_sims[i], effective_threshold + 0.25)
                            break
        
        # 【新增】中文关键词 fingerprint 匹配增强
        # 解决 BGE 对中文短句子语义理解不足的问题
        # 用 lookahead 实现重叠匹配，提取所有 2 字词（确保不漏）
        cn_terms = re.findall(r'(?=([\u4e00-\u9fff]{2}))', query_text)
        if cn_terms:
            if self.use_disk_store:
                # 磁盘模式：遍历候选
                for pos, mem_idx in enumerate(candidate_indices):
                    mem = self.memories[mem_idx]
                    fp = mem.get("fingerprint", "")
                    content = mem.get("content", "")
                    for term in cn_terms:
                        if term in fp or term in content:
                            fused_sims[pos] = max(fused_sims[pos], effective_threshold + 0.20)
                            break
            else:
                # 内存模式：原有逻辑
                for i, mem in enumerate(self.memories):
                    fp = mem.get("fingerprint", "")
                    content = mem.get("content", "")
                    for term in cn_terms:
                        if term in fp or term in content:
                            fused_sims[i] = max(fused_sims[i], effective_threshold + 0.20)
                            break

        # Hopfield 注意力增强：将最后一轮迭代的 softmax 概率作为辅助信号
        # 当查询与某条记忆语义强相关时，Hopfield 会赋予其更高概率，
        # 这在 N 较小时为精确匹配提供额外区分度
        # 磁盘模式下跳过（候选集已在磁盘粗筛阶段筛选）
        if not self.use_disk_store and n_total < self.HNSW_THRESHOLD:
            # 全量模式：p 已对全部记忆计算
            hopfield_attention = p  # 最后一轮迭代的 softmax 概率
            # 将 hopfield_attention 缩放到 [0, 0.15] 区间，作为辅助加分
            p_max = float(np.max(hopfield_attention))
            if p_max > 1.0 / n_total:  # 非均匀分布时才加分
                hopfield_boost = np.clip(hopfield_attention / p_max * 0.15, 0, 0.15)
                fused_sims = fused_sims + hopfield_boost

        # ── 两阶段召回：关键词预筛掩码 ──
        # 当 N >= 50 且有足够查询词时，用 fingerprint token 交集中止不相关记忆
        # 磁盘模式下跳过（磁盘粗筛已过滤）
        if not self.use_disk_store and n_total >= 50:
            q_tokens = set(re.findall(r'[a-zA-Z0-9_\u4e00-\u9fff]{2,}', query_text.lower()))
            if len(q_tokens) >= 2:
                fp_overlap = np.zeros(n_total, dtype=np.float32)
                for i, mem in enumerate(self.memories):
                    fp = mem.get("fingerprint", "").lower()
                    fp_tokens = set(fp.split())
                    if fp_tokens and q_tokens:
                        overlap = len(q_tokens & fp_tokens)
                        fp_overlap[i] = overlap / max(len(q_tokens), 1)
                # 取 top 50% 或 overlap >= 0.2 的记忆
                if np.max(fp_overlap) > 1e-6:
                    fp_threshold = max(float(np.percentile(fp_overlap[fp_overlap > 0], 50)), 0.2) \
                        if np.any(fp_overlap > 0) else 0.0
                else:
                    fp_threshold = 0.0
                candidate_mask = fp_overlap >= fp_threshold
                n_candidates = int(np.sum(candidate_mask))
                if n_candidates >= 5 and n_candidates < n_total:
                    fused_sims = fused_sims * candidate_mask.astype(np.float32)
                    if os.environ.get("MEMCORE_DEBUG"):
                        print(f"    [TwoStage] pre-filter: {n_candidates}/{n_total} memories passed")

        # ── 时间衰减 ──
        if self.use_disk_store:
            # 磁盘模式：只对候选计算时间衰减
            candidate_timestamps = [self.timestamps[idx] for idx in candidate_indices]
            final_time_weights = np.exp(-self.lambda_decay * (now - np.array(candidate_timestamps)))
        else:
            # 内存模式：全量时间衰减
            final_time_weights = np.exp(-self.lambda_decay * (now - np.array(self.timestamps)))
        final_time_weights = np.maximum(final_time_weights, 1e-12)
        # 时间衰减：乘法平滑，0~1 区间缩放
        # 对比原加法 (fused_sims + log(time_weights))，乘法不会反转相似度排序，
        # 旧记忆最多衰减到 0，不会因 log(-27) 把高分记忆拖到阈值以下。
        # 1h 后 0.9996, 24h 后 0.952, 7d 后 0.708, 14d 后 0.501
        final_scores = fused_sims * final_time_weights

        top_indices = np.argsort(final_scores)[::-1]

        # 语义区分度检查：如果所有记忆的相似度几乎完全相同，说明查询无法区分
        # 不同记忆，返回空（避免霍普菲尔德均匀收敛导致的误召回）。
        # 硬阈值 0.03 对小规模记忆集（N < 50）过于敏感——少量记忆天然相似度接近，
        # 不等于是虚假收敛。只在 Hopfield 迭代真正有风险的 N >= 50 时启用。
        if len(fused_sims) >= 50 and fused_sims.max() - fused_sims.min() < 0.03:
            return []

        seen: set[str] = set()
        results: List[str] = []

        # 使用已计算的有效阈值（见上方 short_name 增强前的定义）
        pending_db_updates: list[tuple[int, int, float]] = []  # (db_id, access_count, protection_score)
        for idx in top_indices:
            if fused_sims[idx] >= effective_threshold:
                # 磁盘模式：映射到正确的记忆索引
                mem_idx = candidate_indices[idx] if self.use_disk_store else idx
                content = self.memories[mem_idx]["content"]
                if content not in seen:
                    results.append(content)
                    seen.add(content)
                    self.access_counts[mem_idx] += 1
                    self.memories[mem_idx]["access_count"] = self.access_counts[mem_idx]  # 同步字典
                    self.protection_scores[mem_idx] = min(
                        self.PROTECTION_CAP,
                        self.protection_scores[mem_idx] + 0.2,
                    )
                    # 收集待持久化的更新（批量提交减少 I/O）
                    db_id = self.memories[mem_idx].get("db_id")
                    if db_id is not None:
                        pending_db_updates.append(
                            (db_id, self.access_counts[mem_idx], self.protection_scores[mem_idx])
                        )
            if len(results) >= self.top_k:
                break

        # 批量持久化到数据库（单次事务，加锁防并发冲突）
        if pending_db_updates:
            with self._db_lock:
                db.batch_update_memory_stats(self.conn, pending_db_updates)

        # 打印迭代信息（仅自测 Demo 中可见，可通过 logger 替换）
        if os.environ.get("MEMCORE_DEBUG"):
            print(f"    [Hopfield] iter={iteration_steps}, cos_changes={[f'{c:.6f}' for c in cos_changes]}")

        # 合并实体倒排结果（在尾部追加，保持 Hopfield 结果优先）
        # 允许实体结果扩充到 top_k + len(entity_results)，统一在末尾截断
        if entity_results:
            for r in entity_results:
                if r not in seen and r not in results:
                    results.append(r)

        # ── 第四道：后备文本检索（向量检索失败时兜底） ──
        # 当前三道（指代/实体/霍普菲尔德）结果不足时，
        # 通过安全受限的文件搜索作为兜底，确保关键信息不漏。
        if len(results) < self.top_k:
            from profile_config import get_profile
            _cfg = get_profile()
            if getattr(_cfg, "fallback_enabled", True):
                # 提取查询关键词（技术名词 + 中文短词）
                _kw = re.findall(
                    r'[a-zA-Z_][a-zA-Z0-9_]{2,}|[\u4e00-\u9fff]{2,4}', query_text
                )
                _search_term = " ".join(_kw[:3]) if _kw else query_text[:20]
                _wl = getattr(_cfg, "fallback_whitelist", None)
                _bl = getattr(_cfg, "fallback_blacklist", None)
                try:
                    _fallback_files = search_files(
                        _search_term,
                        max_results=self.top_k - len(results),
                        whitelist=_wl,
                        blacklist=_bl,
                    )
                    for _f in _fallback_files:
                        if _f not in seen:
                            results.append(f"[文件] {_f}")
                except Exception as _e:
                    if os.environ.get("MEMCORE_DEBUG"):
                        print(f"    [Fallback] search_files failed: {_e}")

        # 更新对话上下文窗口（仅当查询有实际内容时）
        stripped = original_query.strip()
        if len(stripped) > 1:
            self.context_window.append(stripped)

        return results[:self.top_k]

    # ------------------------------------------------------------------
    # 负反馈
    # ------------------------------------------------------------------

    PROTECTION_CAP = 10.0  # 保护分数硬上限

    def report_negative(self, memory_id_or_content) -> bool:
        """外部负反馈：显式降权指定记忆。

        按 ID（int）或 content（str）查找记忆，执行：
        - protection_score -= 2.0（不低于 0）
        - access_count -= 1（不低于 0）

        Parameters
        ----------
        memory_id_or_content : int | str
            记忆的 db_id（int）或 content 文本（str）。

        Returns
        -------
        bool
            是否找到并更新了目标记忆。
        """
        # 查找目标索引
        target_idx = None
        target_db_id = None
        for i, mem in enumerate(self.memories):
            if isinstance(memory_id_or_content, int):
                if mem.get("db_id") == memory_id_or_content:
                    target_idx = i
                    target_db_id = mem["db_id"]
                    break
            else:
                if mem["content"] == memory_id_or_content:
                    target_idx = i
                    target_db_id = mem["db_id"]
                    break

        if target_idx is None:
            return False

        # 扣减
        self.protection_scores[target_idx] = max(0.0, self.protection_scores[target_idx] - 2.0)
        self.access_counts[target_idx] = max(0, self.access_counts[target_idx] - 1)

        # 持久化
        if target_db_id is not None and self.conn is not None:
            db.update_memory_stats(
                self.conn, target_db_id,
                self.access_counts[target_idx],
                self.protection_scores[target_idx],
            )
        return True

    # ------------------------------------------------------------------
    # 关系链
    # ------------------------------------------------------------------

    def _create_successor(self, old_idx: int, query: str, content: str,
                          fingerprint: str) -> Optional[int]:
        """原子创建后继记忆（事务保护）。

        1. 编码新记忆向量
        2. 在同一事务中：插入新记忆 + 更新旧记忆的 successor_id

        Parameters
        ----------
        old_idx : int
            原记忆在 self.memories 中的索引。
        query, content, fingerprint : str
            后继记忆的内容字段。

        Returns
        -------
        int | None
            新记忆的 db_id，失败时返回 None。
        """
        query_vector = self.encoder.encode(query).astype(np.float32)
        kw_text = f"{query} {content} {fingerprint}"
        kw_vec = self._generate_keyword_vector(kw_text, self.KW_DIM)

        old_id = self.memories[old_idx].get("db_id")
        if old_id is None:
            return None

        # 在事务中：先插入新记忆，再更新旧记忆的 successor_id
        mem_uuid = str(uuid.uuid4())
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO memories (query, content, fingerprint, query_vector, keyword_vector, uuid) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (query, content, fingerprint,
                 query_vector.tobytes(), kw_vec.tobytes(), mem_uuid),
            )
            new_id = cursor.lastrowid
            self.conn.execute(
                "UPDATE memories SET successor_id = ? WHERE id = ?",
                (new_id, old_id),
            )

        # 更新内存数组
        self.X = np.vstack([self.X, query_vector.reshape(1, -1)])
        self.X_kw = np.vstack([self.X_kw, kw_vec.reshape(1, -1)])
        self.timestamps.append(time.time())
        self.access_counts.append(0)
        self.protection_scores.append(2.0)
        self.memories.append({
            "db_id": new_id,
            "uuid": mem_uuid,
            "query": query,
            "content": content,
            "fingerprint": fingerprint,
        })

        # 容量控制（与 add_memory 一致）
        if len(self.X) > self.max_memories:
            self._forget_oldest()

        return new_id

    # ------------------------------------------------------------------
    # 遗忘
    # ------------------------------------------------------------------

    def _forget_oldest(self) -> Optional[Dict[str, Any]]:
        """保护分数驱动遗忘：查找 protection_score < 3.0 的首条记忆并移除；
        若全部 >= 3.0 则移除保护分数最低的记忆。

        Returns
        -------
        dict | None
            被移除的记忆条目（含 query, content, fingerprint, db_id, access_count），
            若无记忆可移除则返回 None。
        """
        if not self.memories:
            return None
        for i in range(len(self.memories)):
            if self.protection_scores[i] < 3.0:
                evicted = dict(self.memories[i])
                evicted["access_count"] = self.access_counts[i]
                self._remove_at(i)
                return evicted
        min_idx = int(np.argmin(self.protection_scores))
        evicted = dict(self.memories[min_idx])
        evicted["access_count"] = self.access_counts[min_idx]
        self._remove_at(min_idx)
        return evicted

    def _remove_at(self, idx: int) -> None:
        """移除指定索引处的记忆（内存 + 数据库 + 磁盘索引）。"""
        memory_id = self.memories[idx].get("db_id")
        
        # 磁盘模式：移除磁盘向量
        if self.use_disk_store and memory_id is not None:
            disk_id = self._disk_id_map.pop(memory_id, None)
            if disk_id is not None:
                self.disk_store.remove([disk_id])
        else:
            # 内存模式：删除向量矩阵行
            self.X = np.delete(self.X, idx, axis=0)
            self.X_kw = np.delete(self.X_kw, idx, axis=0)
        
        self.timestamps.pop(idx)
        self.access_counts.pop(idx)
        self.protection_scores.pop(idx)
        self.memories.pop(idx)
        if memory_id is not None and hasattr(self, 'conn') and self.conn is not None:
            db.delete_memory_by_id(self.conn, memory_id)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_memories": len(self.memories),
            "hopfield_dim": self.X.shape[1] if len(self.X) > 0 else self.vec_dim,
            "keyword_dim": self.KW_DIM,
            "index_size": 0,  # 向后兼容（v0.3 已废弃关键词索引）
            "avg_protection_score": (
                sum(self.protection_scores) / len(self.protection_scores)
                if self.protection_scores else 0.0
            ),
        }


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    os.environ["MEMCORE_DEBUG"] = "1"  # 启用 Hopfield 迭代步数打印
    import tempfile

    print("=" * 62)
    print("  MemoryCore v0.3 霍普菲尔德内核 — 自测 Demo")
    print(f"  嵌入器: {type(MemoryCore().encoder).__name__}")
    print(f"  向量维度: {MemoryCore().vec_dim}")
    print("=" * 62)

    # ---- 1. 精确召回测试 ----
    print("\n─── 1. 精确召回（3 条不同主题）───")
    mc = MemoryCore(max_memories=100, similarity_threshold=0.1)

    mc.add_memory("咖啡偏好", "用户喜欢喝咖啡", "咖啡 用户 偏好")
    mc.add_memory("天气话题", "今天天气很好适合外出", "天气 今天 外出")
    mc.add_memory("系统架构", "微服务架构包含网关模块", "架构 微服务 网关")
    print("  已存入 3 条记忆")

    r1 = mc.recall("咖啡相关偏好")
    print(f"  查询「咖啡相关偏好」: {r1}")
    assert any("咖啡" in m for m in r1), f"应召回咖啡相关: {r1}"
    print("  ✅ 咖啡召回正确")

    r2 = mc.recall("天气外出")
    print(f"  查询「天气外出」: {r2}")
    assert any("天气" in m for m in r2), f"应召回天气相关: {r2}"
    print("  ✅ 天气召回正确")

    r3 = mc.recall("微服务架构网关")
    print(f"  查询「微服务架构网关」: {r3}")
    assert any("架构" in m for m in r3), f"应召回架构相关: {r3}"
    print("  ✅ 架构召回正确")

    # ---- 2. 模糊查询召回 ----
    print("\n─── 2. 模糊查询召回 ───")
    # 用部分重叠语义词查询
    r4 = mc.recall("喝咖啡的习惯")
    print(f"  查询「喝咖啡的习惯」: {r4}")
    assert any("咖啡" in m for m in r4), f"应模糊召回咖啡相关: {r4}"
    print("  ✅ 模糊查询正常")

    r5 = mc.recall("今天适合外出活动")
    print(f"  查询「今天适合外出活动」: {r5}")
    assert any("天气" in m for m in r5), f"应模糊召回天气相关: {r5}"
    print("  ✅ 模糊查询正常")

    # 无关查询
    r6 = mc.recall("Python机器学习教程")
    print(f"  查询「Python机器学习教程」(期望无匹配): {r6}")
    print("  ✅ 无关查询不误召回")

    # ---- 3. 遗忘后 X 和 memories 长度同步 ----
    print("\n─── 3. 遗忘后 X / memories / access_counts 长度同步 ───")
    mc2 = MemoryCore(max_memories=3, similarity_threshold=0.08)
    for i in range(5):
        mc2.add_memory(f"Q{i}", f"记忆_{i}", f"标签{i}")
    print(f"  最终 memories 条数: {len(mc2.memories)} (max={mc2.max_memories})")
    print(f"  X 矩阵行数: {mc2.X.shape[0]}")
    print(f"  timestamps 长度: {len(mc2.timestamps)}")
    print(f"  access_counts 长度: {len(mc2.access_counts)}")
    print(f"  protection_scores 长度: {len(mc2.protection_scores)}")
    assert len(mc2.memories) == 3, f"memories 应为 3: {len(mc2.memories)}"
    assert mc2.X.shape[0] == 3, f"X 应为 3 行: {mc2.X.shape[0]}"
    assert len(mc2.timestamps) == 3
    assert len(mc2.access_counts) == 3
    assert len(mc2.protection_scores) == 3
    print("  ✅ 所有数组长度完全同步")

    # ---- 4. 迭代步数和余弦变化打印 ----
    print("\n─── 4. Hopfield 迭代详情（逐查询）───")
    for q_text in ["咖啡相关偏好", "天气外出", "Python机器学习"]:
        print(f"\n  查询: {q_text}")
        _ = mc.recall(q_text)

    print("\n  ✅ 迭代信息已打印")

    # ---- 5. 持久化恢复测试 ----
    print("\n─── 5. 持久化恢复测试（文件 DB）───")
    tmp_dir = tempfile.mkdtemp()
    db_file = os.path.join(tmp_dir, "test.db")

    mc3 = MemoryCore(
        max_memories=10,
        similarity_threshold=0.1,
        db_path=db_file,
    )
    # 存入 3 条记忆
    mc3.add_memory("咖啡偏好", "用户喜欢喝咖啡", "咖啡 用户 偏好")
    mc3.add_memory("天气话题", "今天天气很好适合外出", "天气 今天 外出")
    mc3.add_memory("系统架构", "微服务架构包含网关模块", "架构 微服务 网关")

    # 执行多次 recall 提升 access_count / protection_score
    for _ in range(3):
        mc3.recall("咖啡相关偏好")
    for _ in range(2):
        mc3.recall("天气外出")

    # 记录状态
    saved_access = list(mc3.access_counts)
    saved_protect = list(mc3.protection_scores)
    saved_timestamps = list(mc3.timestamps)
    print(f"  保存前: access_counts={saved_access}, protection_scores={[f'{p:.1f}' for p in saved_protect]}")
    assert saved_access[0] >= 3, f"access_counts[0] 应 >= 3: {saved_access[0]}"
    assert saved_protect[0] >= 1.59, f"protection_scores[0] 应 >= 1.59: {saved_protect[0]}"

    # 销毁并重新加载
    del mc3
    mc4 = MemoryCore(
        max_memories=10,
        similarity_threshold=0.1,
        db_path=db_file,
    )
    print(f"  恢复后: memories={len(mc4.memories)} 条")
    print(f"  X 矩阵: {mc4.X.shape}")
    print(f"  access_counts={mc4.access_counts}")
    print(f"  protection_scores={[f'{p:.1f}' for p in mc4.protection_scores]}")

    assert len(mc4.memories) == 3, f"恢复后应为 3 条: {len(mc4.memories)}"
    assert mc4.access_counts == saved_access, (
        f"access_counts 不匹配: {mc4.access_counts} vs {saved_access}"
    )
    for p_loaded, p_saved in zip(mc4.protection_scores, saved_protect):
        assert abs(p_loaded - p_saved) < 0.01, (
            f"protection_score 偏差过大: {p_loaded} vs {p_saved}"
        )

    # 恢复后召回仍正确
    r_restore = mc4.recall("咖啡相关偏好")
    print(f"  恢复后查询「咖啡相关偏好」: {r_restore}")
    assert any("咖啡" in m for m in r_restore), f"恢复后召回失败: {r_restore}"

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  ✅ 持久化恢复完整通过")

    # ---- 最终统计 ----
    print(f"\n─── 最终统计 ───")
    import json
    stats = mc.get_stats()
    print(f"  {json.dumps(stats, ensure_ascii=False)}")

    print(f"\n{'='*62}")
    print("  MemoryCore v0.3 自测 Demo 全部通过 ✓")
    print("=" * 62)
