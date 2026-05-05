"""
stress_test.py — MemCore 三模式压力测试 (v0.5)

压测维度：
  1. 写入吞吐 — 三模式 × 5 规模
  2. 召回延迟 — 三模式 × 5 规模
  3. 巩固性能
  4. 持久化写入
"""

from __future__ import annotations

import random
import sys
import time
from typing import Any

from dual_memory import DualMemory
from profile_config import create_profile, ProfileConfig

random.seed(42)

# ===========================================================================
# 词池 & 生成器
# ===========================================================================

WORD_POOL = [
    "性能", "基准", "测试", "数据", "系统", "架构", "模块", "接口",
    "配置", "部署", "缓存", "队列", "索引", "查询", "写入", "读取",
    "删除", "备份", "恢复", "迁移", "版本", "发布", "回滚", "监控",
    "报警", "日志", "追踪", "熔断", "降级", "限流", "协议", "认证",
    "授权", "加密", "签名", "凭证", "令牌", "会话", "持久", "事务",
    "异步", "同步", "阻塞", "回调", "事件", "消息", "管道", "路由",
    "过滤", "映射", "内存", "磁盘", "网络", "负载", "均衡", "容错",
    "超时", "重试", "补偿", "校验", "权限", "角色", "审计",
    "通知", "推送", "订阅", "发布", "模板", "渲染", "编译", "构建",
]


def pick_words(n: int) -> list[str]:
    return random.sample(WORD_POOL, k=min(n, len(WORD_POOL)))


def gen_mems(count: int) -> list[dict[str, str]]:
    """生成 count 条记忆模板（query, content, fingerprint）。"""
    mems: list[dict[str, str]] = []
    for i in range(count):
        words = pick_words(random.randint(3, 6))
        mems.append({
            "query": f"主题_{i:04d}",
            "content": f"记忆_{i:06d}_" + "".join(words[:3]),
            "fingerprint": " ".join(words),
        })
    return mems


def gen_queries(mems: list[dict[str, str]], n: int = 20) -> list[str]:
    """从记忆的前 n 条中生成召回查询（取指纹前 2~3 个词）。"""
    queries: list[str] = []
    for mem in mems[:n]:
        words = mem["fingerprint"].split()
        k = min(len(words), random.randint(2, 3))
        queries.append("，".join(words[:k]))
    return queries


# ===========================================================================
# 格式化
# ===========================================================================

def print_sep(c: str = "=", w: int = 72):
    print(c * w)


def print_table(headers: list[str], rows: list[list[str]]):
    cw = [max(len(h), max((len(r[i]) for r in rows), default=0))
          for i, h in enumerate(headers)]
    print("  " + " │ ".join(h.ljust(cw[i]) for i, h in enumerate(headers)))
    print("  " + "─┼─".join("─" * w for w in cw))
    for row in rows:
        print("  " + " │ ".join(str(row[i]).ljust(cw[i]) for i in range(len(row))))


# ===========================================================================
# 测试 1：写入吞吐
# ===========================================================================

def test_write(scales: list[int]):
    print("\n" + "=" * 72)
    print("  1. 写入吞吐测试（三模式 × 5 规模）")
    print("=" * 72)

    profiles = ["chat", "dev", "unlimited"]
    header = ["规模(条)"] + [f"{p.capitalize()}" for p in profiles]
    rows: list[list[str]] = []

    for scale in scales:
        mems = gen_mems(scale)
        row = [str(scale)]
        for pname in profiles:
            cfg = create_profile(pname)
            dm = DualMemory(profile=cfg, db_dir=":memory:")
            t0 = time.perf_counter()
            for m in mems:
                dm.add_memory(m["query"], m["content"], m["fingerprint"])
            t = time.perf_counter() - t0
            s = dm.get_stats()
            sc = s["short_term"]["total_memories"]
            row.append(f"{t:.2f}s ({scale/t:.0f}/s) [{sc}/{cfg.short_max}]")
        rows.append(row)

    print()
    print_table(header, rows)


# ===========================================================================
# 测试 2：召回延迟
# ===========================================================================

def test_recall(scales: list[int]):
    print("\n" + "=" * 72)
    print("  2. 召回延迟测试（三模式 × 5 规模）")
    print("=" * 72)

    nq = 10  # 每模式每规模 10 个查询

    for scale in scales:
        mems = gen_mems(scale)
        queries = gen_queries(mems, n=nq)

        print(f"\n  规模 {scale:>5} 条 | 查询 {nq} 次")
        header = ["模式", "总耗时", "平均", "最慢", "最快", "结果数"]
        rows: list[list[str]] = []

        for pname in ["chat", "dev", "unlimited"]:
            cfg = create_profile(pname)
            dm = DualMemory(profile=cfg, db_dir=":memory:")
            for m in mems:
                dm.add_memory(m["query"], m["content"], m["fingerprint"])

            times: list[float] = []
            n_results = 0
            for q in queries:
                t0 = time.perf_counter()
                r = dm.recall(q)
                times.append((time.perf_counter() - t0) * 1000)
                n_results = max(n_results, len(r))

            avg = sum(times) / len(times)
            mx = max(times)
            mn = min(times)
            total = sum(times)
            rows.append([
                pname,
                f"{total:.0f}ms",
                f"{avg:.1f}ms",
                f"{mx:.1f}ms",
                f"{mn:.1f}ms",
                str(n_results),
            ])

        print_table(header, rows)
        sys.stdout.flush()


# ===========================================================================
# 测试 3：巩固性能
# ===========================================================================

def test_consolidation():
    print("\n" + "=" * 72)
    print("  3. 巩固性能测试")
    print("=" * 72)

    total = 1000
    mems = gen_mems(total)

    for pname in ["chat", "dev", "unlimited"]:
        cfg = create_profile(pname)
        dm = DualMemory(profile=cfg, db_dir=":memory:")
        for m in mems:
            dm.add_memory(m["query"], m["content"], m["fingerprint"])

        # 对前 100 条频繁召回 → 触发巩固
        queries = gen_queries(mems, n=100)
        for _ in range(cfg.consolidation_count + 2):
            for q in queries:
                dm.recall(q)

        t0 = time.perf_counter()
        n = dm.consolidate()
        t = time.perf_counter() - t0
        s = dm.get_stats()
        print(f"  {pname:>10}: 巩固 {n:>3} 条 | 耗时 {t:.3f}s | "
              f"{n/t:.0f} 条/s | 短期 {s['short_term']['total_memories']:>4} | "
              f"长期 {s['long_term']['total_memories']:>4}")


# ===========================================================================
# 测试 4：持久化
# ===========================================================================

def test_persistent():
    print("\n" + "=" * 72)
    print("  4. 持久化写入（磁盘 DB）")
    print("=" * 72)

    import tempfile, os, shutil
    tmpdir = tempfile.mkdtemp(prefix="memcore_stress_")
    scale = 1000

    for pname in ["chat", "dev", "unlimited"]:
        cfg = create_profile(pname)
        db_dir = os.path.join(tmpdir, pname)
        dm = DualMemory(profile=cfg, db_dir=db_dir)
        mems = gen_mems(scale)

        t0 = time.perf_counter()
        for m in mems:
            dm.add_memory(m["query"], m["content"], m["fingerprint"])
        t = time.perf_counter() - t0
        s = dm.get_stats()
        print(f"  {pname:>10}: {scale} 条 | {t:.2f}s | {scale/t:.0f} 条/s | "
              f"短期 {s['short_term']['total_memories']} | 长期 {s['long_term']['total_memories']}")

    # 重启持久化验证
    print("\n  --- 重启验证（持久化恢复）---")
    for pname in ["dev", "unlimited"]:
        cfg = create_profile(pname)
        db_dir = os.path.join(tmpdir, pname)
        dm2 = DualMemory(profile=cfg, db_dir=db_dir)
        s2 = dm2.get_stats()
        print(f"  {pname:>10}: 恢复 | 短期 {s2['short_term']['total_memories']} | "
              f"长期 {s2['long_term']['total_memories']}")

    shutil.rmtree(tmpdir, ignore_errors=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 72)
    print("  MemCore v0.5 三模式压力测试")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  词池: {len(WORD_POOL)} 词")
    print("=" * 72)

    scales = [100, 500, 1000, 2000, 5000]

    test_write(scales)
    test_recall(scales)
    test_consolidation()
    test_persistent()

    print("\n" + "=" * 72)
    print("  压力测试完成 ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
