"""
embedder.py — 语义级文本嵌入器

自动降级链：
  1. BAAI/bge-m3 (1024维, 语义, 多语言支持英文/中文/代码, 优先加载)
  2. BAAI/bge-base-zh-v1.5 (768维, 语义, 中文优化)
  3. google/embeddinggemma-300m (768维, 语义)
  4. intfloat/e5-small-v2 (384维, 语义)
  5. LocalEmbedder (384维, 字符哈希, 兜底)

API 统一：
  - encode(text) -> np.ndarray
  - get_sentence_embedding_dimension() -> int
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("embedder")

# ===========================================================================
# 模型配置
# ===========================================================================

MODEL_CHAIN = [
    {
        "name": "BAAI/bge-m3",
        "local_path": "models/bge-m3",
        "dim": 1024,
        "semantic": True,
    },
    {
        "name": "BAAI/bge-base-zh-v1.5",
        "local_path": "models/bge-base-zh-v1.5",
        "dim": 768,
        "semantic": True,
    },
    {
        "name": "google/embeddinggemma-300m",
        "local_path": "models/embeddinggemma-300m",
        "dim": 768,
        "semantic": True,
    },
    {
        "name": "intfloat/e5-small-v2",
        "local_path": "models/e5-small-v2",
        "dim": 384,
        "semantic": True,
    },
]

# 基于embedder.py所在目录解析模型路径
_EMBEDDER_DIR = Path(__file__).resolve().parent

# 当前激活的嵌入器信息（全局单例）
_ACTIVE_EMBEDDER: Any = None
_ACTIVE_INFO: dict[str, Any] = {}


# ===========================================================================
# LocalEmbedder（兜底）
# ===========================================================================

class LocalEmbedder:
    """基于字符词袋哈希的稠密本地嵌入器。
    全离线、零下载、零依赖。
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        text = text.lower().strip()
        tokens = re.findall(r'[\w]+', text)

        for word in set(tokens):
            if len(word) < 2:
                continue
            idx1 = self._hash(word + "_w1")
            idx2 = self._hash(word + "_w2")
            count = tokens.count(word)
            vec[idx1] += count * 1.0
            vec[idx2] += count * 0.5

        joined = "".join(tokens)
        for i in range(len(joined) - 1):
            idx = self._hash(joined[i:i+2] + "_bg")
            vec[idx] += 0.5
        for i in range(len(joined) - 2):
            idx = self._hash(joined[i:i+3] + "_tg")
            vec[idx] += 0.3

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        return vec

    def _hash(self, key: str) -> int:
        h = hashlib.md5(key.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little") % self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


# ===========================================================================
# 模型加载器
# ===========================================================================

def _try_load_sentence_transformer(config: dict) -> Any | None:
    """尝试加载一个 sentence-transformer 模型。"""
    model_name = config["name"]
    local_path = config["local_path"]
    abs_path = str(Path(local_path).resolve())

    # 修复 Windows 上 torch 2.x + pwd 兼容性问题
    _fix_pwd_on_windows()

    # 先检查本地路径
    if os.path.isdir(abs_path):
        logger.info(f"  从本地加载: {local_path}")
        try:
            from sentence_transformers import SentenceTransformer
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            model = SentenceTransformer(abs_path)
            return model
        except Exception as e:
            logger.warning(f"  本地加载失败: {e}")
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # 尝试 HuggingFace 下载
    logger.info(f"  尝试在线加载: {model_name}")
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3.0)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        logger.warning(f"  在线加载失败: {e}")
        return None
    finally:
        socket.setdefaulttimeout(old_timeout)


def _fix_pwd_on_windows() -> None:
    """修复 Windows 上 torch 2.x 的 pwd 模块缺失问题。"""
    import getpass as _gp
    import os as _os
    if hasattr(_gp, "getuser") and callable(_gp.getuser):
        try:
            _gp.getuser()
        except Exception:
            _gp.getuser = lambda: _os.environ.get("USERNAME", "user")
            logger.warning("  ⚠ 已修复 pwd 兼容性问题")


# ===========================================================================
# 对外接口
# ===========================================================================

def create_embedder() -> Any:
    """创建嵌入器实例（自动降级链）。"""
    global _ACTIVE_EMBEDDER, _ACTIVE_INFO

    if _ACTIVE_EMBEDDER is not None:
        return _ACTIVE_EMBEDDER

    # 修复 Windows pwd 兼容性
    _fix_pwd_on_windows()

    # 先检查所有本地路径（避免触发在线下载）
    for cfg in MODEL_CHAIN:
        local_path = str((_EMBEDDER_DIR / cfg["local_path"]).resolve())
        if os.path.isdir(local_path):
            logger.info(f"  尝试本地加载: {cfg['name']}")
            try:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(local_path)
                _ACTIVE_EMBEDDER = model
                _ACTIVE_INFO = {
                    "model_name": cfg["name"],
                    "dimension": cfg["dim"],
                    "is_semantic": True,
                }
                logger.info(f"  ✅ 激活: {cfg['name']}")
                return _ACTIVE_EMBEDDER
            except Exception as e:
                logger.warning(f"  本地加载失败: {e}")
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

    # 兜底：LocalEmbedder
    local = LocalEmbedder(dim=384)
    _ACTIVE_EMBEDDER = local
    _ACTIVE_INFO = {
        "model_name": "LocalEmbedder",
        "dimension": 384,
        "is_semantic": False,
    }
    logger.info("  ⚠ 所有语义模型不可用，回退到 LocalEmbedder")
    return _ACTIVE_EMBEDDER


def get_embedder_info() -> dict[str, Any]:
    """返回当前使用的嵌入器信息。

    Returns
    -------
    dict
        {"model_name": str, "dimension": int, "is_semantic": bool}
    """
    if not _ACTIVE_INFO:
        create_embedder()
    return dict(_ACTIVE_INFO)


def encode(text: str) -> np.ndarray:
    """统一编码接口（确保归一化输出）。"""
    emb = create_embedder()
    vec = emb.encode(text)
    # 强制归一化：确保所有模型（BGE/LocalEmbedder/其他）输出统一格式
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-10:
        vec = vec / norm
    return vec
