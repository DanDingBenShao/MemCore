# -*- coding: utf-8 -*-
"""
demo_tech_assistant.py - 技术笔记助手 Agent Demo

功能：
1. 自然对话交互
2. 自动记忆存储（每轮结束）
3. 上下文召回（每轮开始）
4. 英文/代码支持验证
5. 磁盘模式验证

使用方法：
  python demo_tech_assistant.py

命令：
  /recall <query>  - 手动召回记忆
  /list           - 列出所有记忆
  /stats          - 显示统计信息
  /clear          - 清空屏幕
  /quit           - 退出
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, r"C:\Users\16535\WorkBuddy\20260410092750")

from dual_memory import DualMemory
from profile_config import ProfileConfig


class TechAssistant:
    """技术笔记助手 Agent"""
    
    def __init__(self, db_dir: str = None):
        if db_dir is None:
            import tempfile
            db_dir = tempfile.mkdtemp(prefix="tech_assistant_")
        
        print(f"\n{'='*60}")
        print("TechAssistant Agent Demo")
        print(f"{'='*60}")
        print(f"Memory DB: {db_dir}")
        
        # 配置：dev模式 + 磁盘存储开启
        cfg = ProfileConfig(profile="dev")
        self.memory = DualMemory(db_dir=db_dir, profile=cfg)
        self.db_dir = db_dir
        
        # 强制开启磁盘模式（绕过ProfileConfig缺失字段）
        self.memory.short_term.use_disk_store = True
        self.memory.long_term.use_disk_store = True
        
        print(f"disk_mode: short={self.memory.short_term.use_disk_store}, long={self.memory.long_term.use_disk_store}")
        print(f"embedding: {self.memory.short_term.vec_dim}d")
        print(f"{'='*60}\n")
        
        self.conversation_count = 0
    
    def process_message(self, user_input: str) -> str:
        """处理用户消息"""
        self.conversation_count += 1
        
        # 1. 召回相关记忆
        context = self.memory.recall(user_input)
        
        # 2. 生成回复（简单规则，真实场景可接LLM）
        response = self._generate_response(user_input, context)
        
        # 3. 存储本次对话
        self._remember_conversation(user_input, response)
        
        return response
    
    def _generate_response(self, query: str, context: list) -> str:
        """生成回复（demo用简单规则，生产环境接LLM）"""
        
        # 如果有召回结果
        if context:
            context_str = "\n".join(f"  - {c[:80]}..." if len(c) > 80 else f"  - {c}" for c in context[:3])
            recall_info = f"\n[召回记忆]\n{context_str}"
        else:
            recall_info = "\n[召回] 未找到相关记忆"
        
        # 简单规则回复（demo目的）
        response = f"[Agent] 收到消息：'{query}'{recall_info}"
        
        # 特殊查询处理
        if any(kw in query.lower() for kw in ["react", "hooks", "vue", "node", "python"]):
            response += "\n\n💡 检测到技术问题，已激活技术记忆召回。"
        
        return response
    
    def _remember_conversation(self, user_msg: str, agent_msg: str):
        """记忆本次对话"""
        # 提取关键词作为fingerprint
        keywords = self._extract_keywords(user_msg)
        self.memory.add_memory(
            query=user_msg,
            content=f"用户问：{user_msg}\n助理答：{agent_msg}",
            fingerprint=keywords
        )
    
    def _extract_keywords(self, text: str) -> str:
        """简单关键词提取"""
        import re
        # 英文单词
        en_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_.]*', text)
        # 中文词汇（简化）
        cn_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
        
        keywords = list(set(en_words[:5] + cn_words[:5]))
        return " ".join(keywords) if keywords else "对话"
    
    def manual_recall(self, query: str) -> list:
        """手动召回"""
        return self.memory.recall(query)
    
    def stats(self) -> dict:
        """统计信息"""
        return {
            "短记忆数": len(self.memory.short_term.memories),
            "长记忆数": len(self.memory.long_term.memories),
            "对话轮数": self.conversation_count,
            "磁盘模式": {
                "短期": self.memory.short_term.use_disk_store,
                "长期": self.memory.long_term.use_disk_store
            },
            "向量维度": self.memory.short_term.vec_dim
        }


def main():
    """交互式主循环"""
    assistant = TechAssistant()
    
    print("欢迎使用技术笔记助手！")
    print("\n命令：")
    print("  /recall <query>  - 手动召回")
    print("  /list           - 列出记忆")
    print("  /stats          - 统计信息")
    print("  /clear          - 清屏")
    print("  /quit           - 退出")
    print("\n开始对话（直接输入消息）...\n")
    
    while True:
        try:
            user_input = input("[You] ").strip()
            
            if not user_input:
                continue
            
            # 命令处理
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                
                if cmd == "/quit":
                    print("\n再见！")
                    assistant.memory.close()
                    break
                
                elif cmd == "/clear":
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                
                elif cmd == "/stats":
                    stats = assistant.stats()
                    print("\n📊 统计信息：")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                    print()
                    continue
                
                elif cmd == "/list":
                    print("\n📝 短期记忆：")
                    for i, mem in enumerate(assistant.memory.short_term.memories[:10], 1):
                        query = mem.get("query", "")
                        print(f"  {i}. {query[:50]}{'...' if len(query) > 50 else ''}")
                    
                    print("\n📝 长期记忆：")
                    for i, mem in enumerate(assistant.memory.long_term.memories[:10], 1):
                        query = mem.get("query", "")
                        print(f"  {i}. {query[:50]}{'...' if len(query) > 50 else ''}")
                    print()
                    continue
                
                elif cmd == "/recall":
                    if len(cmd_parts) < 2:
                        print("[ERROR] 用法: /recall <query>")
                        continue
                    query = cmd_parts[1]
                    results = assistant.manual_recall(query)
                    print(f"\n[召回结果] '{query}':")
                    if results:
                        for i, r in enumerate(results, 1):
                            print(f"  {i}. {r[:80]}{'...' if len(r) > 80 else ''}")
                    else:
                        print("  无结果")
                    print()
                    continue
                
                else:
                    print(f"[ERROR] 未知命令: {cmd}")
                    continue
            
            # 正常对话
            start_time = time.time()
            response = assistant.process_message(user_input)
            elapsed = (time.time() - start_time) * 1000
            
            print(response)
            print(f"\n[耗时: {elapsed:.1f}ms]\n")
            
        except KeyboardInterrupt:
            print("\n\n中断退出")
            assistant.memory.close()
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
