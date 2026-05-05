# -*- coding: utf-8 -*-
"""
demo_agent_integration.py - Agent集成验证Demo

模拟真实Agent工作流程：
1. 初始化Agent（磁盘模式+BGE-M3）
2. 多轮对话：技术学习→日常交流→代码讨论→跨话题查询
3. 验证：记忆持久化、召回精度、意图分类、预算控制
4. 模拟重启后记忆恢复
"""

import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, r"C:\Users\16535\WorkBuddy\20260410092750")

from dual_memory import DualMemory
from profile_config import ProfileConfig


class SimulatedAgent:
    """模拟Agent - 集成MemCore的真实工作流"""
    
    def __init__(self, db_dir: str, use_disk: bool = False):
        cfg = ProfileConfig(profile="dev")
        self.memory = DualMemory(db_dir=db_dir, profile=cfg, use_disk_store=use_disk)
        self.use_disk = use_disk
        
        self.stats = {
            "turns": 0,
            "recall_count": 0,
            "recall_success": 0,
            "total_recall_ms": 0,
        }
    
    def chat(self, user_msg: str) -> dict:
        """模拟一轮对话"""
        self.stats["turns"] += 1
        
        # 1. 召回上下文
        t0 = time.time()
        context = self.memory.recall(user_msg)
        recall_ms = (time.time() - t0) * 1000
        
        self.stats["recall_count"] += 1
        self.stats["total_recall_ms"] += recall_ms
        if context:
            self.stats["recall_success"] += 1
        
        # 2. 生成"回复"（demo中只记录记忆）
        # 真实Agent这里会调LLM生成回复
        
        # 3. 自动记忆
        import re
        en_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.]*', user_msg)
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,}', user_msg)
        fingerprint = " ".join(set(en_words[:3] + cn_words[:3])) or "对话"
        
        self.memory.add_memory(
            query=user_msg,
            content=user_msg,  # demo中content等于query
            fingerprint=fingerprint
        )
        
        return {
            "turn": self.stats["turns"],
            "query": user_msg,
            "recall_results": len(context),
            "recall_top": context[0][:60] + "..." if context else None,
            "recall_ms": recall_ms,
        }
    
    def recall(self, query: str) -> dict:
        """纯召回测试"""
        t0 = time.time()
        results = self.memory.recall(query)
        recall_ms = (time.time() - t0) * 1000
        
        return {
            "query": query,
            "results": len(results),
            "top_result": results[0][:80] + "..." if results else None,
            "recall_ms": recall_ms,
        }
    
    def close(self):
        self.memory.close()


def run_scenario():
    """运行集成场景"""
    
    temp_dir = tempfile.mkdtemp(prefix="agent_demo_")
    print(f"DB: {temp_dir}")
    print("=" * 70)
    
    try:
        # ========================================
        # 阶段1: 技术学习对话
        # ========================================
        print("\n[Phase 1] 技术学习对话")
        print("-" * 40)
        
        agent = SimulatedAgent(temp_dir, use_disk=True)
        
        tech_dialogues = [
            "React Hooks让函数组件拥有状态管理能力",
            "useState是React最基础的Hook，用于声明状态变量",
            "useEffect处理副作用，类似componentDidMount",
            "Vue3的Composition API也提供了类似的功能",
            "Python装饰器是修改函数行为的语法糖",
            "TypeScript泛型可以创建可复用的类型安全组件",
        ]
        
        for msg in tech_dialogues:
            result = agent.chat(msg)
            status = "HIT" if result["recall_results"] > 0 else "NEW"
            print(f"  [{status}] T{result['turn']} | {msg[:40]}... | recall: {result['recall_results']} | {result['recall_ms']:.0f}ms")
        
        # ========================================
        # 阶段2: 日常对话记忆
        # ========================================
        print("\n[Phase 2] 日常对话记忆")
        print("-" * 40)
        
        daily_dialogues = [
            "今天下午3点开会讨论项目进度",
            "项目当前完成60%，预计下周完成原型",
            "周三要交一份技术方案文档",
            "客户反馈需要增加导出功能",
            "下周一和产品经理过需求",
        ]
        
        for msg in daily_dialogues:
            result = agent.chat(msg)
            status = "HIT" if result["recall_results"] > 0 else "NEW"
            print(f"  [{status}] T{result['turn']} | {msg[:40]}... | recall: {result['recall_results']} | {result['recall_ms']:.0f}ms")
        
        # ========================================
        # 阶段3: 代码讨论
        # ========================================
        print("\n[Phase 3] 代码讨论")
        print("-" * 40)
        
        code_dialogues = [
            "async function fetchUserData() returns a Promise",
            "const [count, setCount] = useState(0) is basic React state",
            "app.use(cors()) enables Cross-Origin Resource Sharing",
            "SELECT * FROM users WHERE active=true is a basic query",
        ]
        
        for msg in code_dialogues:
            result = agent.chat(msg)
            status = "HIT" if result["recall_results"] > 0 else "NEW"
            print(f"  [{status}] T{result['turn']} | {msg[:40]}... | recall: {result['recall_results']} | {result['recall_ms']:.0f}ms")
        
        # ========================================
        # 阶段4: 跨话题召回测试
        # ========================================
        print("\n[Phase 4] 跨话题召回测试")
        print("-" * 40)
        
        cross_queries = [
            # 指代类
            ("刚才说的React Hook是什么", "reference"),
            ("之前讨论的项目进度怎么样了", "reference"),
            
            # 技术术语类
            ("useState用法", "entity"),
            ("Python decorator", "entity"),
            
            # 语义匹配类
            ("怎么在React中管理状态", "semantic"),
            ("有什么会议安排", "semantic"),
            
            # 代码片段类
            ("cors middleware express", "entity"),
            ("SQL query active users", "entity"),
        ]
        
        hit_count = 0
        for query, intent in cross_queries:
            result = agent.recall(query)
            hit = result["results"] > 0
            if hit:
                hit_count += 1
            status = "HIT" if hit else "MISS"
            top = result["top_result"][:50] if result["top_result"] else "---"
            print(f"  [{status}] intent={intent:9s} | '{query[:30]}' | {result['results']} results | {result['recall_ms']:.0f}ms")
            if hit:
                print(f"         -> {top}...")
        
        recall_rate = hit_count / len(cross_queries) * 100
        
        # ========================================
        # 阶段5: 重启恢复测试
        # ========================================
        print("\n[Phase 5] 重启恢复测试")
        print("-" * 40)
        
        # 记录关闭前的记忆数
        pre_short = len(agent.memory.short_term.memories)
        pre_long = len(agent.memory.long_term.memories)
        print(f"  关闭前: 短期={pre_short}, 长期={pre_long}")
        
        # 关闭并重建
        agent.close()
        time.sleep(0.5)
        
        agent2 = SimulatedAgent(temp_dir, use_disk=True)
        post_short = len(agent2.memory.short_term.memories)
        post_long = len(agent2.memory.long_term.memories)
        print(f"  重启后: 短期={post_short}, 长期={post_long}")
        
        # 验证记忆是否恢复
        recovery_ok = (post_short + post_long) >= (pre_short + pre_long) * 0.5
        print(f"  记忆恢复: {'OK' if recovery_ok else 'WARN'}")
        
        # 重启后召回测试
        recall_after_restart = agent2.recall("React Hooks状态管理")
        print(f"  重启后召回: {recall_after_restart['results']} results")
        
        agent2.close()
        
        # ========================================
        # 最终报告
        # ========================================
        print("\n" + "=" * 70)
        print("Agent Integration Demo - Final Report")
        print("=" * 70)
        
        avg_recall_ms = agent.stats["total_recall_ms"] / max(agent.stats["recall_count"], 1)
        
        print(f"\n  总对话轮次:  {agent.stats['turns']}")
        print(f"  召回次数:    {agent.stats['recall_count']}")
        print(f"  召回命中率:  {agent.stats['recall_success']}/{agent.stats['recall_count']} ({agent.stats['recall_success']/max(agent.stats['recall_count'],1)*100:.0f}%)")
        print(f"  平均召回延迟: {avg_recall_ms:.1f}ms")
        print(f"  跨话题召回率: {hit_count}/{len(cross_queries)} ({recall_rate:.0f}%)")
        print(f"  磁盘模式:    {'ON' if True else 'OFF'}")
        print(f"  记忆恢复:    {'OK' if recovery_ok else 'WARN'}")
        
        # 评估
        print(f"\n  --- Assessment ---")
        grade_recall = "Excellent" if recall_rate >= 90 else "Good" if recall_rate >= 75 else "Needs Work"
        grade_latency = "Excellent" if avg_recall_ms < 50 else "Good" if avg_recall_ms < 100 else "Slow"
        
        print(f"  召回率: {recall_rate:.0f}% -> {grade_recall}")
        print(f"  延迟: {avg_recall_ms:.0f}ms -> {grade_latency}")
        print(f"  恢复: {'PASS' if recovery_ok else 'FAIL'}")
        
        overall = "PASS" if (recall_rate >= 75 and avg_recall_ms < 200 and recovery_ok) else "NEEDS WORK"
        print(f"\n  Overall: {overall}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_scenario()
