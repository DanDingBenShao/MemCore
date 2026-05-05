# MemCore 认知记忆系统 — 项目交接文档

**项目版本**: v0.6  
**文档生成**: 2026-05-05  
**项目路径**: `C:\Users\16535\WorkBuddy\20260410092750\`  

---

## 一、项目概述

MemCore 是一个类人脑分层的认知记忆系统，基于霍普菲尔德联想网络 + 双向量融合召回，为 LLM 提供结构化记忆服务。

**设计哲学**：
- 零神经网络训练（代数运算 + 确定性规则）
- 纯本地可运行（个人电脑即可）
- 类人脑分层（底层→上层逐层构建）
- 精确遗忘（集合移除，非概率性降权）

---

## 二、文件清单

### 2.1 核心模块（13 个文件）

| 文件 | 路径 | 行数 | 职责 |
|------|------|------|------|
| **memory_core.py** | `./memory_core.py` | 1113 | 霍普菲尔德内核、向量召回、双向量融合、意图分类、原子化提炼 |
| **dual_memory.py** | `./dual_memory.py` | 428 | 双记忆架构（短期+长期）、巩固机制、事件索引集成、噪音过滤 |
| **event_index.py** | `./event_index.py` | 784 | 事件索引三层漏斗检索、事件生命周期管理 |
| **dream_processor.py** | `./dream_processor.py` | 582 | 做梦进程（后台LLM事件聚类）、规则引擎+LLM双模式 |
| **llm_client.py** | `./llm_client.py` | ~580 | 统一LLM调用客户端、规则引擎fallback、噪音检测 |
| **embedder.py** | `./embedder.py` | 237 | BGE语义嵌入器、自动降级链（bge-base → bge-small → LocalEmbedder） |
| **db.py** | `./db.py` | 565 | SQLite数据库层、记忆表+事件表CRUD |
| **server.py** | `./server.py` | 216 | FastAPI服务端点（/admin/remember, /api/recall 等） |
| **profile_config.py** | `./profile_config.py` | 467 | 三模式配置中心（Chat/Dev/Unlimited） |
| **intention_planner.py** | `./intention_planner.py` | 94 | 意图规划器、查询标准化 |
| **adapter.py** | `./adapter.py` | 174 | 在线推理适配器 |
| **digest.py** | `./digest.py` | 421 | 离线消化管道 |
| **fingerprint_index.py** | `./fingerprint_index.py` | 328 | 对话指纹索引、指代消解 |

### 2.2 辅助模块（12 个文件）

| 文件 | 路径 | 行数 | 职责 |
|------|------|------|------|
| **noise_filter.py** | `./noise_filter.py` | 232 | 记忆去噪（口语冗余、填充词、结构化重组） |
| **context_builder.py** | `./context_builder.py` | 360 | 上下文构造层（Token预算、叙事排序、冲突仲裁） |
| **conflict_resolver.py** | `./conflict_resolver.py` | 225 | 冲突仲裁（有效置信度、年龄惩罚） |
| **fallback_retriever.py** | `./fallback_retriever.py` | 184 | 后备提炼回路（白名单安全搜索） |
| **cognitive_snapshot.py** | `./cognitive_snapshot.py` | 327 | 认知快照（内存状态序列化/恢复） |
| **code_parser.py** | `./code_parser.py` | 340 | 代码解析器（AST提取） |
| **memory_bridge.py** | `./memory_bridge.py` | 332 | 跨项目记忆桥接 |
| **followup_navigator.py** | `./followup_navigator.py` | 207 | 追问导航 |
| **diagnostic.py** | `./diagnostic.py` | 198 | 诊断工具 |
| **benchmark.py** | `./benchmark.py` | 528 | 性能基准测试 |
| **qa_suite.py** | `./qa_suite.py` | 662 | QA自动化套件 |
| **compare_memory_systems.py** | `./compare_memory_systems.py` | 461 | 记忆系统对比评测 |

### 2.3 测试与工具（10 个文件）

| 文件 | 路径 | 职责 |
|------|------|------|
| **e2e_test.py** | `./e2e_test.py` | 端到端集成测试（17/17 通过） |
| **test_server.py** | `./test_server.py` | Server API 测试 |
| **stress_test.py** | `./stress_test.py` | 三模式压力测试 |
| **novel_stress_test.py** | `./novel_stress_test.py` | 小说真实场景压力测试 |
| **multi_dimension_stress_test.py** | `./multi_dimension_stress_test.py` | 多维度压力测试（245条/8场景） |
| **agent_scenario_test.py** | `./agent_scenario_test.py` | Agent场景模拟测试 |
| **auto_eval.py** | `./auto_eval.py` | 自动化评估 |
| **test_embedder_upgrade.py** | `./test_embedder_upgrade.py` | 嵌入器升级测试 |
| **claude_memory_sim.py** | `./claude_memory_sim.py` | Claude 记忆模拟器 |
| **download_models.py** | `./download_models.py` | 模型下载脚本 |

### 2.4 配置文件与文档（6 个文件）

| 文件 | 路径 | 职责 |
|------|------|------|
| **profile_config.py** | `./profile_config.py` | 三模式配置（Chat/Dev/Unlimited） |
| **code_review_report.md** | `./code_review_report.md` | 四轮代码审查报告 |
| **auto_eval_report.md** | `./auto_eval_report.md` | 自动化评估报告 |
| **qa_report.html** | `./qa_report.html` | QA报告 |
| **多维度压力测试报告.md** | `./多维度压力测试报告_MemCore_v0.6.md` | 压力测试详细报告 |
| **.workbuddy/memory/MEMORY.md** | `./.workbuddy/memory/MEMORY.md` | 项目长期记忆 |

### 2.5 模型文件

| 模型 | 路径 | 大小 | 用途 |
|------|------|------|------|
| **bge-small-zh-v1.5** | `./models/bge-small-zh-v1.5/` | **91.4MB** | 语义嵌入模型（写入+召回） |
| **bge-base-zh-v1.5** | `./models/bge-base-zh-v1.5/` | 390MB | 优先级更高（语义嵌入） |
| **qwen3.5:4b** | Ollama | 2.7GB | 做梦进程LLM推理 |
| **deepseek-r1:8b** | Ollama | 5.2GB | 备用LLM模型 |

---

## 三、架构数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     输入处理（Agent/Server 层）               │
│                                                             │
│  用户输入 → server.py(/admin/remember)                       │
│            → refine_conversation()                           │
│            → llm_client.call_llm() → 规则引擎                │
│            → dual_memory.add_memory() → 噪音过滤 → 短期存储   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     检索流水线（五道回退）                    │
│                                                             │
│  用户查询 → /api/recall                                      │
│           → dual_memory.recall()                             │
│             ├─ 0. 意图分类（reference/entity/semantic）      │
│             ├─ 1a. 显式指代 O(1) → 短路返回                  │
│             ├─ 1b. 隐式指代 <1ms                             │
│             ├─ 2. 实体倒排 + 3. 霍普菲尔德（合并去重）       │
│             ├─ 4. 后备提炼回路（fallback_retriever）          │
│             └─ 5. 原始上下文回退                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     后台维护进程                              │
│                                                             │
│  巩固：短期→长期（access_count ≥ consolidation_count 时）    │
│  做梦：EventIndex → DreamScheduler → DreamWorker → Ollama   │
│  代谢：保护分衰减 + 冷热迁移                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、关键接口

### 4.1 Server API 端点

| 端点 | 方法 | 输入 | 输出 |
|------|------|------|------|
| `/admin/remember` | POST | `{"conversation_text": "..."}` | `{"status": "success"|"skipped"}` |
| `/api/recall` | POST | `{"query_text": "..."}` | `{"memories": [内容1, 内容2]}` |
| `/admin/consolidate` | POST | 无 | `{"consolidated_count": N}` |
| `/stats` | GET | 无 | 完整统计信息 |
| `/admin/profile` | GET | 无 | 当前配置 |
| `/admin/profile` | POST | `{"profile": "dev"}` | 切换模式 |

### 4.2 核心调用关系

```
memory_core.MemoryCore          ← 基础记忆单元
    ↓ 继承/聚合
dual_memory.DualMemory          ← 双记忆架构
    ├── event_index.EventIndex  ← 事件索引（v0.6）
    └── dream_processor.DreamScheduler ← 做梦调度器（v0.6）
```

---

## 五、配置说明

### 5.1 三模式速查

| 参数 | Chat | Dev | Unlimited |
|------|------|-----|-----------|
| short_max | 50 | 500 | 2000 |
| long_max | 500 | 10000 | 50000 |
| top_k | 3 | 5 | 10 |
| token_budget | 400 | 2000 | 4000 |
| dream_min_interval | 120s | 180s | 300s |
| 适用场景 | 日常对话 | 项目开发 | 无限制 |

### 5.2 环境变量

```bash
# LLM 配置（做梦进程）
LLM_USE_OLLAMA=1              # 启用Ollama（默认开启）
LLM_OLLAMA_MODEL=qwen3.5:4b   # 做梦模型
LLM_OLLAMA_URL=http://localhost:11434/api/chat

# 日志调试
MEMCORE_DEBUG=1               # 打印Hopfield迭代信息
```

---

## 六、当前版本状态

### ✅ 已完成
- [x] 霍普菲尔德联想网络内核（v0.3）
- [x] 双记忆架构（短期+长期 + 巩固）（v0.3）
- [x] BGE 语义向量 + 关键词向量双融合（v0.4）
- [x] 意图分类 + 五道检索流水线（v0.4）
- [x] 三模式配置系统（v0.5）
- [x] 事件索引 + 做梦进程（v0.6）
- [x] 文档模式（蓝图整份存储）（v0.6）
- [x] 噪音过滤集成（v0.6）
- [x] 13 项代码审查+风险修复

### ⏳ 未完成/待修复
- [ ] **写入管道需 bge-small-zh 语义辅助** — 当前 `_extract_atomic_facts()` 是纯规则引擎，需要配合 bge-small-zh 做语义去重合并、多查询锚点生成
- [ ] **保护分缺少自然衰减** — 蓝图规定 consolidate 后 `*= 0.98`，当前只增不减
- [ ] **检索流水线缺第 4/5 道** — 后备提炼回路和原始上下文回退未实现
- [ ] **多查询锚点** — 当前只有 1 个 query，蓝图要求 3-5 个自然语言问法
- [ ] **负反馈通道** — `report_negative()` 蓝图已设计但未实现

### 🧪 性能基线

| 指标 | 实测值 |
|------|--------|
| 写入吞吐量 | 28.8 条/秒 |
| 写入平均延迟 | 34.7ms |
| 召回平均延迟 | 20.9ms |
| 6场景召回率 | 100% |
| 指代消解成功率 | 100% |
| 巩固吞吐量 | 28.6 条/秒 |

---

## 七、快速开始

```bash
# 1. 启动 Server
cd C:\Users\16535\WorkBuddy\20260410092750
uvicorn server:app --reload

# 2. 写入记忆
curl -X POST http://localhost:8000/admin/remember \
  -H "Content-Type: application/json" \
  -d '{"conversation_text": "登录超时时间设为30分钟"}'

# 3. 召回记忆
curl -X POST http://localhost:8000/api/recall \
  -H "Content-Type: application/json" \
  -d '{"query_text": "登录超时"}'

# 4. 运行测试
python e2e_test.py
python multi_dimension_stress_test.py
```

---

*本交接文档于 2026-05-05 生成，对应 MemCore v0.6。*
