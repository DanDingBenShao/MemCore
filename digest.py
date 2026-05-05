"""
digest.py — 离线消化管道

作为独立后台进程运行，从对话日志文件中自动读取新对话，
提炼为记忆，写入记忆服务，并周期性触发巩固。

不依赖 Web 框架，依赖项目内模块和 requests。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

from memory_core import refine_conversation_batch
from profile_config import get_profile, ProfileConfig

# ===========================================================================
# 默认配置（向后兼容，实际默认值由 ProfileConfig 提供）
# ===========================================================================

LOG_FILE: str = "./conversation_logs.txt"
MEMORY_API_URL: str = "http://localhost:8000"
POLL_INTERVAL: int = 10
CONSOLIDATE_INTERVAL: int = 300
BATCH_SIZE: int = 5


# ===========================================================================
# DigestPipeline 类
# ===========================================================================


class DigestPipeline:
    """离线对话消化管道。

    轮询日志文件 → 读取新行 → 本地预检 → 发送到记忆服务 → 周期性巩固。

    Parameters
    ----------
    log_file : str, default "./conversation_logs.txt"
        对话日志文件路径。
    memory_api_url : str, default "http://localhost:8000"
        记忆服务 API 基础地址。
    poll_interval : int, default 10
        轮询间隔（秒）。
    consolidate_interval : int, default 300
        巩固间隔（秒，从上次巩固算起）。
    batch_size : int, default 5
        每次轮询最多处理的行数（从最新行取）。
    """

    def __init__(
        self,
        log_file: str = LOG_FILE,
        memory_api_url: str = MEMORY_API_URL,
        poll_interval: Optional[int] = None,
        consolidate_interval: Optional[int] = None,
        batch_size: Optional[int] = None,
        profile: Optional[ProfileConfig] = None,
    ) -> None:
        # 从 ProfileConfig 读取未指定的参数
        cfg = profile or get_profile()
        self.log_file = log_file
        self.memory_api_url = memory_api_url.rstrip("/")
        self.poll_interval = poll_interval if poll_interval is not None else cfg.poll_interval
        self.consolidate_interval = consolidate_interval if consolidate_interval is not None else cfg.consolidate_interval
        self.batch_size = batch_size if batch_size is not None else cfg.batch_size

        # 文件位置指针（已读取的字节偏移）
        self._file_pos: int = 0
        # 上次巩固时间
        self._last_consolidate: float = time.monotonic()
        # 运行状态
        self._running = False
        # 统计
        self.stats: dict[str, int] = {
            "lines_read": 0,
            "lines_digested": 0,
            "lines_skipped": 0,
            "consolidations": 0,
        }

        # 确保日志文件存在
        self._ensure_log_file()

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def read_new_lines(self) -> list[str]:
        """读取自上次调用后新增的日志行。

        返回新增的行列表。若文件不存在则创建空文件。
        维护内部字节偏移指针，不依赖文件修改时间。

        Returns
        -------
        list[str]
            新增的行。空行和纯空白行被过滤。
        """
        self._ensure_log_file()

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                lines = f.readlines()
                self._file_pos = f.tell()
        except (OSError, IOError):
            return []

        # 过滤空行和纯空白，去除行尾换行符
        cleaned = [line.rstrip("\n\r") for line in lines if line.strip()]
        self.stats["lines_read"] += len(cleaned)
        return cleaned

    def process_line(self, line: str) -> list[dict[str, str]]:
        """处理单行对话文本：解析 → 批量提炼 → 返回原子化事实列表。

        Parameters
        ----------
        line : str
            一行对话文本。可以是：
              - 纯文本，直接作为对话文本
              - JSON 对象，自动提取 user/ai 字段拼接

        Returns
        -------
        list[dict]
            原子化事实列表，每个含 {"query", "content", "fingerprint"}。
            若无事实则返回空列表。
        """
        # 尝试解析 JSON
        conversation_text = self._extract_text(line)

        if not conversation_text or not conversation_text.strip():
            return []

        # 调用 refine_conversation_batch 批量提炼
        results = refine_conversation_batch(conversation_text)

        if results:
            self.stats["lines_digested"] += len(results)
            return results

        self.stats["lines_skipped"] += 1
        return []

    def run(self) -> None:
        """主循环：轮询日志 → 处理 → 发送 → 巩固。"""
        self._running = True
        print(f"[digest] 管道启动")
        print(f"  日志文件: {self.log_file}")
        print(f"  记忆服务: {self.memory_api_url}")
        print(f"  轮询间隔: {self.poll_interval}s")
        print(f"  巩固间隔: {self.consolidate_interval}s")
        print(f"  批次大小: {self.batch_size}")
        print(f"[digest] 开始轮询...")

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                print(f"\n[digest] 收到中断信号，停止")
                break
            except Exception as e:
                print(f"[digest] 轮询异常: {e}")

            time.sleep(self.poll_interval)

    def stop(self) -> None:
        """安全停止主循环。"""
        self._running = False

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """单次轮询逻辑。"""
        # 1. 读取新行
        lines = self.read_new_lines()
        if not lines:
            return

        # 2. 取最新 N 行处理
        batch = lines[-self.batch_size:] if len(lines) > self.batch_size else lines
        for line in batch:
            self._handle_line(line)

        # 3. 检查是否需要巩固
        elapsed = time.monotonic() - self._last_consolidate
        if elapsed >= self.consolidate_interval:
            print(f"[digest] 触发巩固 (距上次 {elapsed:.0f}s)")
            self._trigger_consolidate()

    def _handle_line(self, line: str) -> None:
        """处理单行：批量预检 → 逐条发送到服务端。

        注意：每条原子化事实独立发送，使用 conversation_text 字段
        （与服务端 RememberRequest 匹配）。服务端会对每条事实
        再次调用 refine_conversation，若已结构化则通常会原样存入。
        """
        # 批量预检（同时也积累了 stats）
        facts = self.process_line(line)

        if not facts:
            return  # 无价值内容，跳过

        # 逐条发送到记忆服务
        for fact in facts:
            # 将已提炼的原子化事实拼为结构化对话文本
            # 格式：query: content (fingerprint: fp)
            structured_text = (
                f"{fact.get('query', '')}: {fact.get('content', '')} "
                f"(fingerprint: {fact.get('fingerprint', '')})"
            )
            try:
                resp = requests.post(
                    f"{self.memory_api_url}/admin/remember",
                    json={"conversation_text": structured_text},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "?")
                    if status == "success":
                        print(f"  ✓ 已存入: {fact.get('content', '')[:40]}...")
                    else:
                        print(f"  - 跳过 (服务端返回 {status}): {fact.get('content', '')[:30]}")
                else:
                    print(f"  ⚠ 服务端返回 {resp.status_code}: {resp.text[:60]}")
            except requests.RequestException as e:
                print(f"  ✗ 请求失败: {e}")

    def _trigger_consolidate(self) -> None:
        """调用服务端 /admin/consolidate 触发巩固。

        仅在请求成功时重置巩固定时器，失败时保留以便下轮重试。
        """
        try:
            resp = requests.post(
                f"{self.memory_api_url}/admin/consolidate",
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                n = data.get("consolidated_count", 0)
                self.stats["consolidations"] += 1
                self._last_consolidate = time.monotonic()  # 仅成功时重置
                print(f"  ✓ 巩固完成: {n} 条提升到长期")
            else:
                print(f"  ⚠ 巩固请求失败: {resp.status_code}")
        except requests.RequestException as e:
            print(f"  ✗ 巩固请求异常: {e}")

    def _extract_text(self, line: str) -> str:
        """从一行中提取对话文本。

        尝试解析 JSON，提取 user/ai 字段拼接；
        若解析失败，直接返回原文本。
        """
        # 尝试 JSON 解析
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                parts = []
                for key in ("user", "ai", "human", "assistant", "text", "content"):
                    val = obj.get(key)
                    if val and isinstance(val, str) and val.strip():
                        label = {"user": "用户", "ai": "AI", "human": "用户",
                                 "assistant": "AI", "text": "", "content": ""}.get(key, key)
                        if label:
                            parts.append(f"{label}：{val}")
                        else:
                            parts.append(val)
                if parts:
                    return "\n".join(parts)
            return line
        except (json.JSONDecodeError, TypeError):
            pass

        return line

    def _ensure_log_file(self) -> None:
        """确保日志文件存在，不存在则创建。"""
        if not os.path.exists(self.log_file):
            try:
                Path(self.log_file).touch()
            except OSError:
                pass


# ===========================================================================
# 自测 Demo
# ===========================================================================

if __name__ == "__main__":
    import shutil

    print("=" * 55)
    print("digest.py — 离线消化管道自测")
    print("=" * 55)

    # 需要 server.py 在后台运行
    # 启动一个测试服务器
    SERVER_PORT = 18128
    SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"

    print(f"\n[setup] 启动记忆服务 (端口 {SERVER_PORT})...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app",
         "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
        cwd=str(Path(__file__).resolve().parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待服务就绪
    for _ in range(20):
        try:
            r = requests.get(f"{SERVER_URL}/stats", timeout=2)
            if r.status_code == 200:
                print(f"  ✓ 服务已就绪: {SERVER_URL}")
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        print("  ❌ 服务未能启动")
        server_proc.kill()
        sys.exit(1)

    # 创建临时日志文件
    demo_log = Path("_digest_test_log.txt")
    if demo_log.exists():
        demo_log.unlink()

    try:
        # 写入 3 条模拟对话（含 1 条无价值）
        demo_lines = [
            '{"user": "我在用384维向量做记忆系统", "ai": "好的，这个方案很合理"}',
            "今天天气真好，适合出去散步",
            '{"user": "深色模式有个显示bug，切换按钮没反应", "ai": "我来排查这个问题"}',
        ]
        with open(demo_log, "w", encoding="utf-8") as f:
            for line in demo_lines:
                f.write(line + "\n")
        print(f"\n[setup] 已写入 {len(demo_lines)} 行到 {demo_log}")

        # 初始化管道（指向测试服务器和临时日志）
        pipeline = DigestPipeline(
            log_file=str(demo_log),
            memory_api_url=SERVER_URL,
            poll_interval=2,         # 短间隔便于测试
            consolidate_interval=60,  # 轮次内不会触发
            batch_size=10,
        )

        # 首次读取前重置文件指针到开头
        pipeline._file_pos = 0

        # 手动执行一轮处理（不进入 while 循环）
        print(f"\n[run] 手动执行一轮处理...")
        lines = pipeline.read_new_lines()
        print(f"  读取到 {len(lines)} 行新内容")

        for line in lines:
            pipeline._handle_line(line)

        # 打印统计
        print(f"\n[stats] 管道统计:")
        print(f"  读取行数: {pipeline.stats['lines_read']}")
        print(f"  提炼成功: {pipeline.stats['lines_digested']}")
        print(f"  跳过:     {pipeline.stats['lines_skipped']}")

        # 通过 API 验证
        print(f"\n[verify] 通过 API 验证...")
        try:
            r = requests.get(f"{SERVER_URL}/stats", timeout=5)
            stats = r.json()
            print(f"  GET /stats: {stats}")
            short_total = stats.get("short_term", {}).get("total_memories", 0)
            print(f"  短期记忆条数: {short_total}")
            if short_total >= 1:
                print(f"  ✅ 测试通过: 记忆已成功存入")
            else:
                print(f"  ⚠ 零条记忆被存入，检查 refine_conversation 的 mock 覆盖率")
        except Exception as e:
            print(f"  ⚠ 验证请求异常: {e}")

        # 验证巩固端点
        print(f"\n[consolidate] 测试巩固端点...")
        try:
            r = requests.post(f"{SERVER_URL}/admin/consolidate", timeout=5)
            print(f"  POST /admin/consolidate: {r.json()}")
            print(f"  ✅ 巩固端点可用")
        except Exception as e:
            print(f"  ⚠ 巩固请求异常: {e}")

    finally:
        # 清理
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        if demo_log.exists():
            demo_log.unlink()
            print(f"\n[cleanup] 已清理临时日志")

    print(f"\n{'='*55}")
    print("自测完成")
    print("=" * 55)
