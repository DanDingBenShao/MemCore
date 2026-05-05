"""
download_models.py — 离线下载语义嵌入模型

在联网环境中运行一次，将模型缓存到 ./models/ 目录，
之后可离线使用。

用法：
    python download_models.py

支持的模型（按优先级）：
  1. BAAI/bge-base-zh-v1.5 (768维, 推荐)
  2. google/embeddinggemma-300m (768维)
  3. intfloat/e5-small-v2 (384维)
  4. BAAI/bge-small-zh-v1.5 (512维, 旧版)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 模型配置
MODELS = [
    {
        "name": "BAAI/bge-m3",
        "dir": "bge-m3",
        "dim": 1024,
    },
    {
        "name": "BAAI/bge-base-zh-v1.5",
        "dir": "bge-base-zh-v1.5",
        "dim": 768,
    },
    {
        "name": "google/embeddinggemma-300m",
        "dir": "embeddinggemma-300m",
        "dim": 768,
    },
    {
        "name": "intfloat/e5-small-v2",
        "dir": "e5-small-v2",
        "dim": 384,
    },
    {
        "name": "BAAI/bge-small-zh-v1.5",
        "dir": "bge-small-zh-v1.5",
        "dim": 512,
    },
]

LOCAL_DIR = Path("./models")


def download_model(cfg: dict) -> bool:
    """下载单个模型到本地目录。

    使用 sentence-transformers 的缓存机制，
    然后将缓存拷贝到 ./models/{dir}/ 下。
    """
    model_name = cfg["name"]
    target_dir = LOCAL_DIR / cfg["dir"]

    if target_dir.exists():
        size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        print(f"  [SKIP] {model_name} exists ({size / 1024 / 1024:.1f}MB)")
        return True

    print(f"  [+] Downloading {model_name}...")
    print(f"     Target: {target_dir}")

    try:
        from sentence_transformers import SentenceTransformer

        # 下载模型（会缓存到 HuggingFace 默认缓存目录）
        model = SentenceTransformer(model_name)

        # 找到缓存路径
        cache_dir = None
        if hasattr(model, "_modules"):
            # 尝试定位实际文件
            for name, module in model._modules.items():
                if hasattr(module, "_model_card_text"):
                    continue
        # 通用方法：从模型对象获取
        if hasattr(model, "save"):
            # 保存到目标目录
            os.makedirs(target_dir, exist_ok=True)
            model.save(str(target_dir))
            size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
            print(f"  [OK] {model_name} downloaded ({size / 1024 / 1024:.1f}MB)")
            return True

    except Exception as e:
        print(f"  [FAIL] Download failed: {e}")
        return False


def main() -> int:
    print("=" * 55)
    print("  Embedding Model Downloader")
    print("=" * 55)
    print()

    os.makedirs(LOCAL_DIR, exist_ok=True)

    success = 0
    for cfg in MODELS:
        if download_model(cfg):
            success += 1
        print()

    print(f"Result: {success}/{len(MODELS)} models downloaded")
    if success > 0:
        print(f"Model directory: {LOCAL_DIR.resolve()}")
        print("Copy ./models/ directory to offline environment for offline use.")
    else:
        print("Please check network connectivity and retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
