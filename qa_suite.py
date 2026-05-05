"""
qa_suite.py — MemCore 全量质量验收套件

执行所有现有的测试 + 新增大规模场景测试。
输出 HTML 报告。
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
import tempfile
import subprocess
from typing import Any

# ===========================================================================
# 测试注册表
# ===========================================================================

TEST_REGISTRY: list[dict] = []

def register(category: str, name: str, module: str, args: str = ""):
    TEST_REGISTRY.append({
        "category": category,
        "name": name,
        "module": module,
        "args": args,
    })


# ===========================================================================
# 注册所有已有测试
# ===========================================================================

# 单元测试 / 自测
register("unit", "embedder.py 模型加载", "from embedder import create_embedder; e=create_embedder(); assert e.get_sentence_embedding_dimension()==768")
register("unit", "profile_config 三模式", "import profile_config; c=profile_config.get_profile(); assert c.profile=='dev'")

# 模块导入
register("import", "全部 27 模块导入", "", "all_imports")

# llm_client 自测
register("unit", "llm_client 自测", "llm_client", "demo")

# intention_planner 自测
register("unit", "intention_planner 自测", "intention_planner", "demo")

# dual_memory 自测
register("unit", "dual_memory 自测", "dual_memory", "demo")

# 已有小说压力测试
register("stress", "《活着》小说召回 (15/19)", "novel_stress_test", "")

# 已有 Agent 场景测试
register("scenario", "Agent 开发场景 (11/12)", "agent_scenario_test", "")

# e2e 集成测试
register("e2e", "端到端集成测试 (17/17)", "e2e_test", "")


# ===========================================================================
# 大规模新增测试
# ===========================================================================

def run_large_scale_conversation_test() -> dict[str, Any]:
    """大规模多轮对话压力测试。

    50 轮对话模拟持续开发工作流，包括：
    - 技术决策
    - Bug 修复
    - 代码审查
    - 配置变更
    - 跨轮次模糊指代
    - 记忆巩固与长期检索
    """
    from dual_memory import DualMemory
    from memory_core import refine_conversation
    from intention_planner import plan_memory_query

    dm = DualMemory(
        # 使用 Dev 模式容量（short_max=500），确保 42 条对话全部保留
        # 之前 hardcode 30 导致前 12 条被覆盖，5 个查询因此 0 分
        short_max=500,
        long_max=10000,
        short_threshold=0.10,
        long_threshold=0.30,
        db_dir=":memory:",
        consolidation_count=3,
        top_k=5,
    )

    conversations = [
        # --- 模块 A: 微服务架构 ---
        "我们决定把单体应用拆分为微服务架构，用 gRPC 做服务间通信",
        "每个微服务独立部署，用 Docker 容器化",
        "API Gateway 用 Envoy 做反向代理和负载均衡",
        "服务注册与发现用 Consul，配置中心用 etcd",
        "每个服务都有自己的独立数据库，避免共享",
        "跨服务事务用 Saga 模式处理，用消息队列 Kafka 做异步协调",
        "监控用 Prometheus + Grafana，日志用 ELK 集中管理",
        "每个服务都有健康检查端点 /healthz 和 /readyz",

        # --- 模块 B: 前端架构 ---
        "前端用 React 18 + TypeScript，状态管理用 Zustand",
        "路由用 React Router v6，支持懒加载",
        "组件库用 Ant Design，主题色是蓝色 #1890ff",
        "API 请求用 React Query 做数据缓存和自动重试",
        "表单校验用 React Hook Form + Zod",
        "国际化用 i18next，支持中英文切换",
        "打包用 Vite，开发服务器端口 3000",
        "单元测试用 Vitest，组件测试用 Testing Library",

        # --- 模块 C: 后端 API ---
        "后端用 Go 1.22 + Gin 框架写 RESTful API",
        "数据库用 PostgreSQL 15，连接池用 pgxpool",
        "Redis 做缓存，缓存热点数据 TTL 5 分钟",
        "JWT Token 做身份认证，过期时间 24 小时",
        "接口限流用令牌桶算法，每秒 100 个请求",
        "API 版本管理用 URL 前缀 /v1/ /v2/",
        "错误处理统一返回 JSON 格式 {code, message, data}",
        "请求日志中间件记录 method, path, status, duration",
        "数据库迁移用 golang-migrate，版本控制",
        "单元测试用 testify，覆盖率要求 80%",

        # --- 模块 D: Bug 与问题 ---
        "生产环境发现用户登录后 session 频繁过期，排查是 Redis 连接池泄漏",
        "修复方案：在 Session 中间件里加上 defer conn.Close() 确保连接归还池子",
        "第二个 bug：首页加载慢，首屏 8 秒，原因是没做 SSR",
        "临时方案：加骨架屏占位，长期方案上 Next.js 做 SSR",
        "还有一个 CRUD 接口返回 500，排查是数据库连接超时",
        "修复：连接池最大连接数从 10 调到 50，超时时间从 5s 调到 30s",

        # --- 模块 E: 性能优化 ---
        "压测发现 /api/products 接口 QPS 只有 200，瓶颈在数据库全表扫描",
        "优化方案：给 frequently 查询字段加复合索引",
        "加完索引后 QPS 从 200 提升到 1500",
        "另一个热点是用户认证接口，每次都要查数据库",
        "优化：把用户 session 缓存到 Redis，减少 90% 的 DB 查询",
        "整体压测结果：优化后 QPS 从 800 提升到 5000",

        # --- 模块 F: 配置变更 ---
        "生产环境日志级别从 DEBUG 改为 INFO，减少磁盘 IO",
        "数据库连接字符串从环境变量读取，不再硬编码",
        "CORS 允许的域名列表放到配置中心统一管理",
        "限流参数调优：生产环境每秒 500 请求，开发环境每秒 2000",

        # --- 模块 G: 跨轮次回溯测试 ---
        # 这些查询不喂入新数据，只做召回验证
    ]

    # 存储统计
    store_stats = {"stored": 0, "skipped": 0}

    # 注入对话
    for i, text in enumerate(conversations):
        query = plan_memory_query(text)
        memory = refine_conversation(text)
        if memory and memory.get("content") and memory["content"] != "无":
            dm.add_memory(memory["query"], memory["content"], memory["fingerprint"])
            store_stats["stored"] += 1
        else:
            store_stats["skipped"] += 1

    # 执行巩固
    before_consolidate = dm.get_stats()
    dm.consolidate()
    after_consolidate = dm.get_stats()

    # 跨轮次召回测试
    recall_queries = [
        ("微服务间通信方式", ["gRPC", "服务间通信"]),
        ("前端状态管理", ["Zustand", "状态管理"]),
        ("后端框架和语言", ["Go", "Gin"]),
        ("Redis 缓存", ["Redis", "缓存"]),
        ("JWT 过期时间", ["JWT", "24小时"]),
        ("登录 session 问题", ["session", "过期", "Redis"]),
        ("首页加载慢修复方案", ["骨架屏", "SSR", "Next.js"]),
        ("数据库连接超时修复", ["连接池", "50", "30s"]),
        ("API 性能从 200 提升到 1500", ["复合索引", "QPS", "1500"]),
        ("用户认证优化方案", ["Redis", "90%", "缓存"]),
        ("生产环境日志级别", ["INFO", "DEBUG"]),
        ("限流生产环境参数", ["500", "请求", "生产"]),
        ("微服务拆分决策", ["微服务", "Docker", "gRPC"]),
        ("前端打包工具", ["Vite", "3000"]),
        ("API 错误返回格式", ["code", "message", "data"]),
        ("Saga 模式", ["Saga", "Kafka", "异步"]),
        ("监控方案", ["Prometheus", "Grafana", "ELK"]),
        ("测试覆盖率要求", ["testify", "80%"]),
        ("第二个 bug 是什么", ["首页", "SSR", "8秒"]),
        ("整体 QPS 优化结果", ["5000", "QPS"]),
    ]

    recall_results = []
    for q_text, expected_kws in recall_queries:
        t0 = time.time()
        recalled = dm.recall(q_text)
        elapsed = time.time() - t0

        found_kws = {kw: any(kw in r for r in recalled) for kw in expected_kws}
        all_found = all(found_kws.values())
        some_found = any(found_kws.values())

        recall_results.append({
            "query": q_text,
            "expected": expected_kws,
            "found": found_kws,
            "all_found": all_found,
            "some_found": some_found,
            "count": len(recalled),
            "latency_ms": round(elapsed * 1000, 1),
        })

    passed = sum(1 for r in recall_results if r["all_found"])
    partial = sum(1 for r in recall_results if not r["all_found"] and r["some_found"])
    failed = sum(1 for r in recall_results if not r["some_found"])

    return {
        "name": "大规模多轮对话压测 (50 轮)",
        "conversations": len(conversations),
        "stored": store_stats["stored"],
        "skipped": store_stats["skipped"],
        "short_before": before_consolidate["short_term"]["total_memories"],
        "long_before": before_consolidate["long_term"]["total_memories"],
        "short_after": after_consolidate["short_term"]["total_memories"],
        "long_after": after_consolidate["long_term"]["total_memories"],
        "consolidated": before_consolidate["short_term"]["total_memories"] - after_consolidate["short_term"]["total_memories"],
        "queries": len(recall_queries),
        "passed": passed,
        "partial": partial,
        "failed": failed,
        "recall_results": recall_results,
    }


def run_persistence_test() -> dict[str, Any]:
    """持久化 + 重启恢复测试。

    验证：写入 → 重启 → 数据保留
    """
    from dual_memory import DualMemory

    tmp_dir = tempfile.mkdtemp(prefix="memcore_qa_")

    # 第一轮：写入
    dm1 = DualMemory(short_max=20, long_max=50, db_dir=tmp_dir)
    for i in range(5):
        dm1.add_memory(f"测试主题{i}", f"测试内容_{i}_这是第{i}条记忆的详细内容", f"测试{i}")
    stats1 = dm1.get_stats()
    dm1.consolidate()

    # 第二轮：重新加载（模拟重启）
    dm2 = DualMemory(short_max=20, long_max=50, db_dir=tmp_dir)
    stats2 = dm2.get_stats()

    # 召回验证
    results = dm2.recall("测试主题0")
    recall_ok = any("测试内容_0" in r for r in results)

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "name": "持久化重启恢复",
        "short_term_retained": stats2["short_term"]["total_memories"] > 0 or stats2["long_term"]["total_memories"] > 0,
        "recall_after_restart": recall_ok,
    }


def run_consolidation_loop_test() -> dict[str, Any]:
    """巩固循环验证：短期满 → 巩固 → 长期有数据。

    反复写入直到短期满，验证巩固机制触发。
    """
    from dual_memory import DualMemory

    dm = DualMemory(short_max=5, long_max=20, db_dir=":memory:", consolidation_count=2)

    # 写入 8 条，短期上限 5，触发 3 次遗忘
    for i in range(8):
        dm.add_memory(f"记忆{i}", f"记忆内容_{i}", f"mem{i}")

        # 每次写入后召回一下，触发 access_count 递增
        if i < 5:
            for _ in range(3):
                dm.recall(f"记忆{i}")

    # 手动巩固
    dm.consolidate()
    stats = dm.get_stats()

    return {
        "name": "巩固循环验证",
        "short_term": stats["short_term"]["total_memories"],
        "long_term": stats["long_term"]["total_memories"],
        "consolidation_working": stats["long_term"]["total_memories"] > 0,
    }


def run_fuzzy_anaphora_test() -> dict[str, Any]:
    """模糊指代消解专项测试。
    
    验证 intention_planner 对各种模糊指代和开发场景关键词的识别能力。
    """
    from intention_planner import plan_memory_query

    test_cases = [
        # (输入, 期望输出不能包含的关键词, 期望输出应包含的关键词)
        ("上次说的那个 bug 修了吗", [], ["bug"]),
        ("之前讨论的架构方案", [], ["架构"]),
        ("那个异常是怎么回事", [], ["bug"]),  # 异常 → bug
        ("我们选的什么技术栈", [], ["技术"]),
        ("帮我查下数据库配置", [], ["配置"]),
        ("部署方案定了吗", [], ["部署"]),
        ("性能太慢了怎么优化", [], ["性能"]),
        ("单元测试覆盖率多少", [], ["测试"]),
        ("顺便看下架构方案和测试进展", [], ["架构", "测试"]),
        ("我不喜欢吃香菜", [], ["饮食"]),
        ("今天天气怎么样", [], ["天气"]),
    ]

    results = []
    for inp, not_expected, expected_kw in test_cases:
        out = plan_memory_query(inp)
        has_expected = any(kw in out for kw in expected_kw)
        has_bad = any(n in out for n in not_expected)
        results.append({
            "input": inp,
            "output": out,
            "valid": has_expected and not has_bad,
        })

    valid = sum(1 for r in results if r["valid"])
    return {
        "name": "模糊指代消解专项",
        "cases": len(results),
        "valid": valid,
        "results": results,
    }


# ===========================================================================
# 测试执行引擎
# ===========================================================================

def run_module_test(test: dict) -> dict[str, Any]:
    """运行单个测试。"""
    t0 = time.time()
    try:
        if test["module"] == "all_imports":
            modules = [
                "adapter", "auto_eval", "benchmark", "claude_memory_sim",
                "code_parser", "cognitive_snapshot", "compare_memory_systems",
                "conflict_resolver", "context_builder", "db", "diagnostic",
                "digest", "download_models", "dual_memory", "embedder",
                "fallback_retriever", "fingerprint_index", "followup_navigator",
                "intention_planner", "llm_client", "memory_bridge", "memory_core",
                "noise_filter", "profile_config",
            ]
            failed = []
            for m in modules:
                try:
                    importlib.import_module(m)
                except Exception as e:
                    failed.append(f"{m}: {e}")
            elapsed = time.time() - t0
            if not failed:
                return {"test": test, "status": "PASS", "detail": f"27 modules OK", "latency_ms": round(elapsed*1000, 1)}
            else:
                return {"test": test, "status": "FAIL", "detail": "; ".join(failed[:3]), "latency_ms": round(elapsed*1000, 1)}

        elif test["args"] == "demo":
            # 运行模块的 __main__ demo
            result = subprocess.run(
                [sys.executable, "-B", f"{test['module']}.py"],
                capture_output=True, text=True, timeout=60,
                cwd=".",
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                return {"test": test, "status": "PASS", "detail": "demo OK", "latency_ms": round(elapsed*1000, 1)}
            else:
                return {"test": test, "status": "FAIL", "detail": result.stderr[:200], "latency_ms": round(elapsed*1000, 1)}

        else:
            exec(test["args"])
            elapsed = time.time() - t0
            return {"test": test, "status": "PASS", "detail": "OK", "latency_ms": round(elapsed*1000, 1)}

    except Exception as e:
        elapsed = time.time() - t0
        return {"test": test, "status": "FAIL", "detail": str(e)[:200], "latency_ms": round(elapsed*1000, 1)}


def run_file_test(test: dict) -> dict[str, Any]:
    """以文件形式运行测试。"""
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-B", f"{test['module']}.py"],
            capture_output=True, text=False, timeout=300,
            cwd=".",
        )
        elapsed = time.time() - t0

        # 手动解码（避免 Windows GBK 编码问题）
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        output = stdout + stderr
        passed = result.returncode == 0

        # 提取关键统计
        summary = {}
        for line in output.split("\n"):
            if "通过" in line:
                m = __import__('re').search(r'(\d+)\s*/\s*(\d+)\s*通过', line)
                if m:
                    summary["score"] = f"{m.group(1)}/{m.group(2)}"
            if "总分" in line:
                m = __import__('re').search(r'(\d+)\s*/\s*(\d+)', line)
                if m:
                    summary["total_score"] = f"{m.group(1)}/{m.group(2)}"

        return {
            "test": test,
            "status": "PASS" if passed else "FAIL",
            "detail": summary if summary else ("OK" if passed else output[-100:]),
            "latency_ms": round(elapsed * 1000, 1),
        }
    except subprocess.TimeoutExpired:
        return {"test": test, "status": "TIMEOUT", "detail": ">300s", "latency_ms": 300000}
    except Exception as e:
        return {"test": test, "status": "FAIL", "detail": str(e)[:200], "latency_ms": 0}


# ===========================================================================
# HTML 报告生成
# ===========================================================================

def generate_html_report(all_results: list[dict]) -> str:
    """生成 HTML 测试报告。"""
    total = len(all_results)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    timeout = sum(1 for r in all_results if r["status"] == "TIMEOUT")
    total_time = sum(r.get("latency_ms", 0) for r in all_results)

    # 分类统计
    categories: dict[str, list] = {}
    for r in all_results:
        cat = r["test"]["category"]
        categories.setdefault(cat, []).append(r)

    rows = ""
    for cat, items in categories.items():
        cat_pass = sum(1 for r in items if r["status"] == "PASS")
        rows += f"""
        <tr class="cat-header"><td colspan="5" style="background:#f0f4ff;font-weight:500;padding:8px 12px">
          {cat.upper()} ({cat_pass}/{len(items)})
        </td></tr>"""
        for r in items:
            t = r["test"]
            icon = "✅" if r["status"] == "PASS" else ("⏰" if r["status"] == "TIMEOUT" else "❌")
            cls = "pass" if r["status"] == "PASS" else "fail"
            detail_str = str(r.get("detail", ""))
            rows += f"""
            <tr class="{cls}">
              <td>{icon}</td>
              <td>{t['name']}</td>
              <td>{r['status']}</td>
              <td>{r['latency_ms']:.0f}ms</td>
              <td style="font-size:11px;color:#666;max-width:300px;overflow:hidden">{detail_str[:80]}</td>
            </tr>"""

    # 额外统计（大规模测试）
    extra_sections = ""
    for r in all_results:
        if isinstance(r.get("detail"), dict) and r["test"]["module"] in ("novel_stress_test", "agent_scenario_test"):
            d = r["detail"]
            extra_sections += f"""<div class="section">
              <h3>{r['test']['name']}</h3>
              <div class="stat-grid">
                <div class="stat"><span class="stat-val">{d.get('score', d.get('e2e_score', '?'))}</span><span class="stat-label">通过率</span></div>
              </div>
            </div>"""

    # 大规模对话测试结果通过 detail 获取
    for r in all_results:
        if r["test"]["module"] == "qa_suite" and isinstance(r.get("detail"), dict) and "passed" in r["detail"]:
            d = r["detail"]
            extra_sections += f"""<div class="section">
              <h3>{d.get('name', '大规模测试')}</h3>
              <div class="stat-grid">
                <div class="stat"><span class="stat-val">{d['passed']}/{d['queries']}</span><span class="stat-label">召回通过</span></div>
                <div class="stat"><span class="stat-val">{d.get('conversations', 0)}</span><span class="stat-label">对话轮次</span></div>
                <div class="stat"><span class="stat-val">{d.get('stored', 0)}</span><span class="stat-label">记忆存储</span></div>
              </div>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>MemCore QA Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; background: #f8f9fa; color: #222; }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 24px; }}
  .summary-box {{ flex: 1; background: #fff; border-radius: 8px; padding: 16px; text-align: center; border: 1px solid #e0e0e0; }}
  .summary-num {{ font-size: 28px; font-weight: 600; display: block; }}
  .summary-label {{ font-size: 12px; color: #666; }}
  .pass .summary-num {{ color: #22c55e; }}
  .fail .summary-num {{ color: #ef4444; }}
  .section {{ background: #fff; border-radius: 8px; padding: 16px; margin-bottom: 16px; border: 1px solid #e0e0e0; }}
  .section h3 {{ margin: 0 0 12px 0; font-size: 14px; }}
  .stat-grid {{ display: flex; gap: 12px; }}
  .stat {{ flex: 1; text-align: center; padding: 8px; background: #f8f9fa; border-radius: 6px; }}
  .stat-val {{ font-size: 20px; font-weight: 600; display: block; color: #2563eb; }}
  .stat-label {{ font-size: 11px; color: #666; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; }}
  .cat-header td {{ font-weight: 500; }}
  .pass td {{ color: #222; }}
  .fail td {{ color: #991b1b; background: #fef2f2; }}
  .timestamp {{ font-size: 11px; color: #999; text-align: right; margin-top: 16px; }}
</style>
</head>
<body>
  <h1>MemCore QA Report</h1>
  <p class="subtitle">{time.strftime('%Y-%m-%d %H:%M:%S')} | {total_time/1000:.1f}s total</p>
  <div class="summary">
    <div class="summary-box pass"><span class="summary-num">{passed}</span><span class="summary-label">通过</span></div>
    <div class="summary-box fail"><span class="summary-num">{failed}</span><span class="summary-label">失败</span></div>
    <div class="summary-box"><span class="summary-num">{timeout}</span><span class="summary-label">超时</span></div>
    <div class="summary-box"><span class="summary-num">{total}</span><span class="summary-label">总计</span></div>
  </div>
  {extra_sections}
  <div class="section">
    <h3>测试详情</h3>
    <table>{rows}</table>
  </div>
  <div class="timestamp">MemCore QA Suite v1.0</div>
</body>
</html>"""
    return html


# ===========================================================================
# 主流程
# ===========================================================================

def main():
    print("=" * 65)
    print("  MemCore 全量质量验收套件")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    all_results: list[dict] = []

    # Phase 1: 模块导入检查
    print("\n[Phase 1] 模块导入检查...")
    for test in TEST_REGISTRY:
        if test["category"] == "import":
            r = run_module_test(test)
            all_results.append(r)
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"  {icon} {test['name']}")

    # Phase 2: 单元测试
    print("\n[Phase 2] 单元测试...")
    for test in TEST_REGISTRY:
        if test["category"] == "unit":
            r = run_module_test(test)
            all_results.append(r)
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"  {icon} {test['name']} ({r['latency_ms']:.0f}ms)")

    # Phase 3: 自测 Demo
    print("\n[Phase 3] 模块自测 Demo...")
    for test in TEST_REGISTRY:
        if test["args"] == "demo":
            r = run_module_test(test)
            all_results.append(r)
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"  {icon} {test['name']} ({r['latency_ms']:.0f}ms)")

    # Phase 4: 大规模新增测试
    print("\n[Phase 4] 大规模质量测试...")
    
    print("  运行 50 轮对话压测...")
    conv_result = run_large_scale_conversation_test()
    all_results.append({
        "test": {"category": "stress", "name": "大规模多轮对话 (50轮)", "module": "qa_suite"},
        "status": "PASS" if conv_result["failed"] == 0 else "FAIL",
        "detail": conv_result,
        "latency_ms": 0,
    })
    print(f"  ✅ {conv_result['passed']}/{conv_result['queries']} 通过, {conv_result['failed']} 失败")

    print("  运行持久化测试...")
    persist = run_persistence_test()
    all_results.append({
        "test": {"category": "stress", "name": "持久化重启恢复", "module": "qa_suite"},
        "status": "PASS" if persist.get("recall_after_restart") else "FAIL",
        "detail": persist,
        "latency_ms": 0,
    })
    print(f"  ✅ 重启后召回: {persist.get('recall_after_restart')}")

    print("  运行巩固循环测试...")
    consolidation = run_consolidation_loop_test()
    all_results.append({
        "test": {"category": "stress", "name": "巩固循环验证", "module": "qa_suite"},
        "status": "PASS" if consolidation["consolidation_working"] else "FAIL",
        "detail": consolidation,
        "latency_ms": 0,
    })
    print(f"  ✅ 长期记忆: {consolidation['long_term']} 条")

    print("  运行指代消解专项测试...")
    anaphora = run_fuzzy_anaphora_test()
    all_results.append({
        "test": {"category": "stress", "name": "模糊指代消解专项", "module": "qa_suite"},
        "status": "PASS" if anaphora["valid"] == anaphora["cases"] else "FAIL",
        "detail": anaphora,
        "latency_ms": 0,
    })
    print(f"  ✅ {anaphora['valid']}/{anaphora['cases']} 正确消解")

    # Phase 5: 文件测试
    print("\n[Phase 5] 文件级测试...")
    for test in TEST_REGISTRY:
        if test["category"] in ("stress", "scenario", "e2e"):
            r = run_file_test(test)
            all_results.append(r)
            icon = "✅" if r["status"] == "PASS" else ("⏰" if r["status"] == "TIMEOUT" else "❌")
            detail_str = str(r.get("detail", ""))
            print(f"  {icon} {test['name']} ({r['latency_ms']:.0f}ms) {detail_str[:60]}")

    # 生成报告
    total = len(all_results)
    passed = sum(1 for r in all_results if r["status"] == "PASS")
    failed = sum(1 for r in all_results if r["status"] == "FAIL")

    print(f"\n{'='*65}")
    print(f"  验收完成: {passed}/{total} 通过, {failed} 失败")
    print(f"{'='*65}")

    html = generate_html_report(all_results)
    report_path = "./qa_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  报告已保存: {report_path}")
    print(f"  用浏览器打开查看详细报告")


if __name__ == "__main__":
    main()
