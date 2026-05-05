# -*- coding: utf-8 -*-
"""
MemCore v0.7 全仿真测试 - 真实场景模拟
基于真实文本数据验证召回精度
"""
import sys
import os
import time
import gc
import psutil
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore
from dual_memory import DualMemory
from profile_config import ProfileConfig

# ==================== 真实文本数据 ====================

REACT_TEXT = """
React可以改变你对设计和构建应用的思考方式。用React构建用户界面时，首先将其分解为称为组件的片段。然后描述每个组件的不同视觉状态。最后将组件连接在一起，使数据流经它们。

步骤1：将UI分解为组件层次结构
从在模型中为每个组件和子组件画框并命名开始。如果你与设计师合作，他们可能已在设计工具中命名了这些组件。

这个屏幕上有五个组件：
1. FilterableProductTable（灰色）包含整个应用
2. SearchBar（蓝色）接收用户输入
3. ProductTable（紫色）根据用户输入显示和过滤列表
4. ProductCategoryRow（绿色）显示每个类别的标题
5. ProductRow（黄色）显示每个产品的一行

步骤2：在React中构建静态版本
现在你有了组件层次结构，是时候实现你的应用了。最直接的方法是构建一个从数据模型渲染UI而不添加任何交互性的版本。

构建静态版本需要大量输入而不需要思考，但添加交互性需要大量思考而不需要大量输入。

步骤3：找到UI状态的最小但完整的表示
要使UI具有交互性，你需要让用户能够更改底层数据模型。你将使用state。

为了正确构建应用，首先需要考虑应用需要的可变状态的最小集合。关键点是DRY原则（Don't Repeat Yourself）。

步骤4：验证状态应该存在于哪里
确定了应用的最小状态集合后，你需要确定哪个组件可变，或哪个组件拥有此状态。

React的单向数据流意味着所有组件都必须通过共同的祖先组件进行通信。

步骤5：添加反向数据流
React的单向数据流在渲染与数据流匹配的组件层次结构时效果很好，但现在具有不同状态的表单输入将无法工作。React使这变得明确，以帮助你理解应用的工作方式。
"""

VUE_TEXT = """
计算属性是Vue.js的核心概念之一。模板中的表达式虽然方便，但也只能用来做简单的操作。如果在模板中写太多逻辑，会让模板变得臃肿，难以维护。

基础示例：
假设我们有一个包含嵌套数组的对象：
author: {
  name: 'John Doe',
  books: ['Vue 2 - Advanced Guide', 'Vue 3 - Basic Guide', 'Vue 4 - The Mystery']
}

我们可以定义一个计算属性publishedBooksMessage，根据author是否已有一些书籍来展示不同的信息。

计算属性缓存 vs 方法：
计算属性值会基于其响应式依赖被缓存。一个计算属性仅会在其响应式依赖更新时才重新计算。这意味着只要author.books不改变，无论多少次访问publishedBooksMessage都会立即返回先前的计算结果。

可写计算属性：
计算属性默认是只读的。你可以通过同时提供getter和setter来创建可写计算属性。例如fullName可以拆分为firstName和lastName。

最佳实践：
1. Getter不应有副作用 - 不要改变其他状态、做异步请求或更改DOM
2. 避免直接修改计算属性值 - 应该更新它所依赖的源状态
"""

ES_TEXT = """
全文搜索属于最常见的需求，开源的Elasticsearch是目前全文搜索引擎的首选。它可以快速地储存、搜索和分析海量数据。维基百科、Stack Overflow、Github都采用它。

基本概念：
1. Node与Cluster - 单个Elastic实例称为节点，一组节点构成集群
2. Index - Elastic数据管理的顶层单位，相当于数据库
3. Document - Index里面单条的记录
4. Type - Document的逻辑分组（6.x版本后逐步废弃）

安装要求：
- Java 8环境
- 设置JAVA_HOME环境变量
- 默认端口9200

中文分词设置：
需要安装ik分词插件。在需要搜索的中文字段上，要指定analyzer为ik_max_word。

数据操作：
- 新增：PUT请求到/Index/Type/Id
- 查看：GET请求到/Index/Type/Id
- 删除：DELETE请求
- 更新：PUT请求重新发送数据

查询语法：
- match查询：全文搜索，会对搜索词分词
- term查询：精确匹配，不分词
- range查询：范围查询，如时间范围
"""

# ==================== 测试框架 ====================

class SimulationTest:
    def __init__(self):
        self.results = {}
        self.test_dir = None
        
    def setup(self, test_name):
        """创建测试目录"""
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sim_workspace")
        os.makedirs(base_dir, exist_ok=True)
        self.test_dir = os.path.join(base_dir, f"{test_name}_{int(time.time()*1000)}")
        os.makedirs(self.test_dir, exist_ok=True)
        gc.collect()
        
    def get_memory_mb(self):
        """获取当前进程内存使用(MB)"""
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
    def measure_recall(self, mc, queries, expected_keywords, name):
        """测量召回率"""
        hits = 0
        total_time = 0
        total = len(queries)
        
        for query, expected in zip(queries, expected_keywords):
            start = time.perf_counter()
            results = mc.recall(query)
            elapsed = (time.perf_counter() - start) * 1000
            total_time += elapsed
            
            # 检查是否命中
            if results and len(results) > 0:
                for r in results:
                    if any(kw in r for kw in expected):
                        hits += 1
                        break
        
        recall_rate = hits / total if total > 0 else 0
        avg_time = total_time / total if total > 0 else 0
        
        return {
            "name": name,
            "total": total,
            "hits": hits,
            "recall_rate": f"{recall_rate*100:.1f}%",
            "avg_time_ms": f"{avg_time:.1f}",
            "status": "PASS" if recall_rate >= 0.75 else "FAIL"
        }

# ==================== 场景A：技术文档学习 ====================

def scenario_tech_docs():
    """场景B：技术文档学习 - React/Vue/ES真实文档"""
    sim = SimulationTest()
    sim.setup("tech_docs")
    
    db_path = os.path.join(sim.test_dir, "tech.db")
    mc = MemoryCore(db_path=db_path)
    
    # 分段存储React文档
    react_chunks = [
        ("React组件思想", "React构建UI三步骤：分解组件、描述状态、连接数据流", "tech"),
        ("React组件层次", "FilterableProductTable包含整个应用，SearchBar接收输入，ProductTable显示列表", "tech"),
        ("React静态版本", "构建静态版本需要大量输入不思考，添加交互性需要大量思考", "tech"),
        ("React状态原则", "DRY原则：确定最小可变状态集合，避免重复计算", "tech"),
        ("React数据流", "单向数据流：组件通过共同祖先通信，反向数据流处理表单状态", "tech"),
    ]
    
    # 分段存储Vue文档
    vue_chunks = [
        ("Vue计算属性基础", "计算属性用于复杂逻辑，模板中不应写太多计算表达式", "tech"),
        ("Vue计算属性缓存", "计算属性基于响应式依赖缓存，依赖不变就不重新计算", "tech"),
        ("Vue可写计算属性", "默认只读，提供getter/setter可创建可写计算属性如fullName", "tech"),
        ("Vue最佳实践", "Getter无副作用，不修改计算属性值，更新源状态触发重新计算", "tech"),
    ]
    
    # 分段存储ES文档
    es_chunks = [
        ("Elasticsearch基础概念", "Node节点、Cluster集群、Index索引、Document文档、Type分组", "tech"),
        ("Elasticsearch安装", "需要Java 8环境，设置JAVA_HOME，默认端口9200", "tech"),
        ("Elasticsearch中文分词", "安装ik插件，中文字段指定analyzer为ik_max_word", "tech"),
        ("Elasticsearch数据操作", "PUT新增、GET查看、DELETE删除、PUT更新重新发送数据", "tech"),
        ("Elasticsearch查询语法", "match全文搜索分词、term精确匹配、range范围查询", "tech"),
    ]
    
    # 添加所有记忆
    all_chunks = react_chunks + vue_chunks + es_chunks
    for query, content, category in all_chunks:
        mc.add_memory(query, content, category)
    
    # 测试召回
    queries = [
        "React怎么构建UI",
        "React组件有哪些",
        "React静态版本和交互区别",
        "React数据流怎么做",
        "Vue计算属性是什么",
        "Vue计算属性缓存原理",
        "Vue计算属性可以写吗",
        "Vue开发最佳实践",
        "Elasticsearch基本概念",
        "Elasticsearch怎么安装",
        "Elasticsearch中文怎么分词",
        "Elasticsearch怎么查询",
    ]
    
    expected_keywords = [
        ["组件", "分解", "状态"],
        ["FilterableProductTable", "SearchBar", "ProductTable"],
        ["静态", "交互", "思考"],
        ["单向", "数据流", "祖先"],
        ["计算属性", "缓存", "响应式"],
        ["缓存", "依赖", "重新计算"],
        ["可写", "getter", "setter"],
        ["副作用", "源状态", "不要修改"],
        ["Node", "Cluster", "Index", "Document"],
        ["Java", "9200", "JAVA_HOME"],
        ["ik", "analyzer", "ik_max_word"],
        ["match", "term", "range"],
    ]
    
    result = sim.measure_recall(mc, queries, expected_keywords, "技术文档学习")
    
    # 记录内存
    result["memory_mb"] = f"{sim.get_memory_mb():.1f}"
    result["total_memories"] = len(all_chunks)
    
    return result

# ==================== 场景C：日常对话记忆 ====================

def scenario_daily_chat():
    """场景C：日常对话记忆 - 模拟7天对话"""
    sim = SimulationTest()
    sim.setup("daily_chat")
    
    db_path = os.path.join(sim.test_dir, "daily.db")
    mc = MemoryCore(db_path=db_path)
    
    # 模拟7天的对话记录（时间戳模拟）
    daily_memories = [
        # Day 1
        ("咖啡偏好", "我喜欢喝美式咖啡，不加糖", "preference"),
        ("早餐习惯", "平时早餐吃面包和牛奶", "habit"),
        # Day 2
        ("周末计划", "周末打算去电影院看新电影", "plan"),
        ("电影口味", "喜欢科幻片和动作片", "preference"),
        # Day 3
        ("朋友聚会", "下周三和张三李四聚餐", "schedule"),
        ("餐厅选择", "聚餐选了川菜馆", "event"),
        # Day 4
        ("运动习惯", "每周跑步3次，每次5公里", "habit"),
        ("睡眠时间", "晚上11点睡觉，早上7点起床", "habit"),
        # Day 5
        ("读书进度", "正在读三体第二部，进度50%", "activity"),
        ("书籍评价", "三体写得很精彩，黑暗森林法则印象深刻", "opinion"),
        # Day 6
        ("工作安排", "明天下午3点有项目评审会议", "schedule"),
        ("会议准备", "评审会需要准备PPT和技术文档", "todo"),
        # Day 7
        ("健身计划", "办了健身卡，打算每周去两次", "plan"),
        ("体脂目标", "目标体脂率降到15%", "goal"),
    ]
    
    # 添加记忆（模拟时间衰减）
    for query, content, category in daily_memories:
        mc.add_memory(query, content, category)
    
    # 测试召回（涵盖不同时间尺度的记忆）
    queries = [
        "我喜欢喝什么咖啡",
        "周末有什么安排",
        "聚餐是什么时候",
        "平时怎么运动",
        "最近读什么书",
        "明天有什么会议",
        "健身目标是什么",
    ]
    
    expected_keywords = [
        ["美式", "不加糖"],
        ["电影", "电影院"],
        ["周三", "张三", "李四"],
        ["跑步", "5公里", "每周"],
        ["三体", "第二部", "黑暗森林"],
        ["评审", "下午3点", "PPT"],
        ["体脂", "15%", "健身卡"],
    ]
    
    result = sim.measure_recall(mc, queries, expected_keywords, "日常对话记忆")
    result["memory_mb"] = f"{sim.get_memory_mb():.1f}"
    result["total_memories"] = len(daily_memories)
    
    return result

# ==================== 场景D：开发工作日志 ====================

def scenario_dev_logs():
    """场景D：开发工作日志 - Bug修复和配置记录"""
    sim = SimulationTest()
    sim.setup("dev_logs")
    
    db_path = os.path.join(sim.test_dir, "dev.db")
    mc = MemoryCore(db_path=db_path)
    
    dev_memories = [
        # Bug记录
        ("Bug-登录超时", "登录超时问题：JWT过期时间设置为24小时，Redis缓存失效导致", "bug"),
        ("Bug-内存泄漏", "内存泄漏原因：WebSocket连接未正确关闭，修复：添加close事件监听", "bug"),
        ("Bug-中文乱码", "PowerShell输出乱码：编码GBK，修复：使用CHCP 65001设置UTF-8", "bug"),
        ("Bug-跨域问题", "跨域错误CORS：后端添加Access-Control-Allow-Origin头部", "bug"),
        
        # 配置记录
        ("配置-API密钥", "火山引擎API Key: sk-xxx123，Endpoint: https://api.volcengine.com", "config"),
        ("配置-数据库", "MySQL配置：host=192.168.1.100, port=3306, db=neuralflow", "config"),
        ("配置-Redis", "Redis配置：localhost:6379, password=abc123, db=0", "config"),
        ("配置-前端端口", "React开发服务器端口3000，后端端口8080", "config"),
        
        # 技术决策
        ("决策-框架选择", "选择Electron+React：跨平台需求，团队熟悉度，生态成熟", "decision"),
        ("决策-数据库选型", "SQLite用于本地存储：轻量级，无服务器，足够应对10万级数据", "decision"),
        ("决策-向量模型", "BGE-base-zh-v1.5：768维，中文效果好，开源免费", "decision"),
        
        # 代码片段
        ("代码-Hopfield更新", "Hopfield更新公式: x_new = softmax(beta * X @ x_old.T) @ X", "code"),
        ("代码-余弦相似度", "余弦相似度: sim = dot(a,b) / (norm(a) * norm(b))", "code"),
        ("代码-时间衰减", "时间衰减: weight * exp(-lambda * hours_since_access)", "code"),
    ]
    
    for query, content, category in dev_memories:
        mc.add_memory(query, content, category)
    
    queries = [
        "登录超时怎么解决",
        "内存泄漏原因",
        "PowerShell乱码怎么修",
        "跨域错误怎么处理",
        "API密钥是多少",
        "数据库配置",
        "Redis配置",
        "为什么选Electron",
        "向量模型用的什么",
        "Hopfield更新公式",
    ]
    
    expected_keywords = [
        ["JWT", "24小时", "Redis"],
        ["WebSocket", "close", "事件"],
        ["GBK", "UTF-8", "CHCP"],
        ["CORS", "Access-Control"],
        ["sk-xxx123", "volcengine"],
        ["192.168.1.100", "3306", "neuralflow"],
        ["6379", "abc123"],
        ["跨平台", "React", "团队"],
        ["BGE", "768", "中文"],
        ["softmax", "beta", "X"],
    ]
    
    result = sim.measure_recall(mc, queries, expected_keywords, "开发工作日志")
    result["memory_mb"] = f"{sim.get_memory_mb():.1f}"
    result["total_memories"] = len(dev_memories)
    
    return result

# ==================== 场景E：压力测试 ====================

def scenario_pressure_test():
    """场景E：压力测试 - 1000条数据性能测试"""
    sim = SimulationTest()
    sim.setup("pressure")
    
    db_path = os.path.join(sim.test_dir, "pressure.db")
    mc = MemoryCore(db_path=db_path)
    
    # 生成1000条有语义差异的模拟数据
    print("生成1000条测试数据...")
    categories = ["user", "tech", "config", "event", "note"]
    topics = [
        ("用户偏好", "用户喜欢{}，偏好{}风格"),
        ("技术配置", "{}配置：参数{}，需要重启生效"),
        ("会议记录", "{}会议：时间{}，参会人员10人"),
        ("Bug修复", "{}Bug：原因{}，修复方式代码调整"),
        ("文档笔记", "{}文档：要点{}，重要程度高"),
    ]
    
    import random
    random.seed(42)  # 固定种子保证可复现
    
    keywords_a = ["Python", "Java", "Go", "Rust", "TypeScript", "Vue", "React", "Node"]
    keywords_b = ["简单", "复杂", "高效", "稳定", "灵活", "安全", "快速", "可靠"]
    
    stored_queries = []
    for i in range(1000):
        topic_name, template = topics[i % len(topics)]
        kw_a = keywords_a[i % len(keywords_a)]
        kw_b = keywords_b[i % len(keywords_b)]
        query = f"{topic_name}{i:04d}"
        content = template.format(kw_a, kw_b)
        category = categories[i % len(categories)]
        mc.add_memory(query, content, category)
        stored_queries.append((query, kw_a, kw_b))
    
    memory_after_load = sim.get_memory_mb()
    print(f"数据加载完成，内存: {memory_after_load:.1f}MB")
    
    # 测试召回性能（随机选择20条，用关键词查询）
    test_indices = [100, 250, 500, 750, 900, 123, 456, 789, 234, 567,
                    11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    
    queries = []
    expected_keywords = []
    for idx in test_indices:
        # 用存储时的关键词构造查询
        kw_a = keywords_a[idx % len(keywords_a)]
        queries.append(f"{kw_a}相关记忆")
        expected_keywords.append([kw_a])
    
    # 测量召回时间
    start_time = time.perf_counter()
    hits = 0
    for query, expected in zip(queries, expected_keywords):
        results = mc.recall(query)
        if results and len(results) > 0:
            for r in results:
                if any(kw in r for kw in expected):
                    hits += 1
                    break
    total_time = (time.perf_counter() - start_time) * 1000
    avg_time = total_time / len(queries)
    
    recall_rate = hits / len(queries)
    memory_after_test = sim.get_memory_mb()
    
    return {
        "name": "压力测试",
        "total_memories": 1000,
        "test_queries": len(queries),
        "hits": hits,
        "recall_rate": f"{recall_rate*100:.1f}%",
        "avg_time_ms": f"{avg_time:.1f}",
        "total_time_ms": f"{total_time:.1f}",
        "memory_load_mb": f"{memory_after_load:.1f}",
        "memory_test_mb": f"{memory_after_test:.1f}",
        "status": "PASS" if recall_rate >= 0.90 and avg_time < 100 else "FAIL"
    }

# ==================== 主测试函数 ====================

def main():
    print("="*60)
    print("MemCore v0.7 全仿真测试")
    print("="*60)
    
    results = []
    
    # 运行各场景测试
    print("\n[场景A] 技术文档学习...")
    results.append(scenario_tech_docs())
    
    print("\n[场景C] 日常对话记忆...")
    results.append(scenario_daily_chat())
    
    print("\n[场景D] 开发工作日志...")
    results.append(scenario_dev_logs())
    
    print("\n[场景E] 压力测试...")
    results.append(scenario_pressure_test())
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for r in results:
        icon = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        print(f"\n{icon} {r['name']}")
        print(f"   记忆数: {r.get('total_memories', 'N/A')}")
        print(f"   召回率: {r['recall_rate']}")
        print(f"   平均耗时: {r['avg_time_ms']}ms")
        if 'memory_mb' in r:
            print(f"   内存占用: {r['memory_mb']}MB")
    
    # 总体评估
    total_pass = sum(1 for r in results if r["status"] == "PASS")
    total_tests = len(results)
    
    print("\n" + "="*60)
    print(f"总评: {total_pass}/{total_tests} 通过")
    
    if total_pass == total_tests:
        print("评级: *** 优秀")
    elif total_pass >= total_tests * 0.75:
        print("评级: ** 合格")
    else:
        print("评级: * 需改进")
    print("="*60)
    
    # 保存详细报告
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "_sim_workspace", 
                               f"report_{int(time.time()*1000)}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存: {report_path}")

if __name__ == "__main__":
    main()
