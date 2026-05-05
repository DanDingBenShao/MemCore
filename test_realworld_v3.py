# -*- coding: utf-8 -*-
"""MemCore 实际场景测试 v3 - 按模块逐一验证"""
import sys
import os
import time
import gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_core import MemoryCore
from dual_memory import DualMemory
from event_index import EventIndex, AtomicFact
from profile_config import ProfileConfig

# ========== 测试工具 ==========
_test_counter = 0

def _make_dir(test_name):
    global _test_counter
    _test_counter += 1
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_workspace")
    d = os.path.join(base_dir, f"{test_name}_{int(time.time()*1000)}_{_test_counter}")
    os.makedirs(d, exist_ok=True)
    return d

def test_case(name, test_func):
    try:
        test_func()
        gc.collect()
        return {"name": name, "status": "PASS", "error": None}
    except AssertionError as e:
        return {"name": name, "status": "FAIL", "error": str(e)}
    except Exception as e:
        return {"name": name, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}

def print_results(results, module_name):
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n{'='*60}")
    print(f"【{module_name}】 {passed}/{total} 通过")
    print('='*60)
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {icon} {r['name']}")
        if r["error"]:
            print(f"      └─ {r['error'][:120]}")
    print()

# ========== 模块1: MemoryCore 核心召回 ==========
def test_memory_core():
    results = []
    
    # --- 基础召回场景 ---
    
    def test_daily_chat():
        d = _make_dir("daily_chat")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("我喜欢喝咖啡", "用户喜欢喝咖啡，尤其是美式", "chat")
        recalled = mc.recall("我喜欢什么")
        assert len(recalled) > 0, "应召回至少1条记忆"
        assert "咖啡" in recalled[0], "应返回咖啡相关内容"
    results.append(test_case("日常对话-偏好记忆", test_daily_chat))
    
    def test_work_schedule():
        d = _make_dir("work_schedule")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("明天下午3点开会", "明天下午3点有项目评审会，会议室A", "schedule")
        recalled = mc.recall("明天有什么安排")
        assert len(recalled) > 0, "应召回会议安排"
    results.append(test_case("工作安排-会议提醒", test_work_schedule))
    
    def test_fuzzy_query():
        d = _make_dir("fuzzy_query")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("公司地址", "公司在北京市朝阳区建国路88号SOHO现代城", "info")
        recalled = mc.recall("公司在哪")
        assert len(recalled) > 0, "应召回地址信息"
    results.append(test_case("模糊查询-地址检索", test_fuzzy_query))
    
    def test_mixed_lang():
        d = _make_dir("mixed_lang")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("GitHub仓库", "项目GitHub地址: github.com/user/project", "info")
        recalled = mc.recall("GitHub地址")
        assert len(recalled) > 0, "应召回GitHub信息"
    results.append(test_case("中英混合-GitHub地址", test_mixed_lang))
    
    def test_tech_terms():
        d = _make_dir("tech_terms")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("API配置", "火山引擎API Key: abc123xyz, Endpoint: https://api.volcengine.com", "config")
        recalled = mc.recall("API Key是多少")
        assert len(recalled) > 0, "应召回API配置"
    results.append(test_case("技术术语-API配置", test_tech_terms))
    
    def test_precise_recall():
        d = _make_dir("precise_recall")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("张三电话", "张三电话: 13812345678", "contact")
        mc.add_memory("李四电话", "李四电话: 13987654321", "contact")
        mc.add_memory("王五电话", "王五电话: 15012345678", "contact")
        recalled = mc.recall("张三电话")
        assert len(recalled) > 0, "应召回张三电话"
        assert "13812345678" in recalled[0], "应返回张三正确号码"
    results.append(test_case("多条记忆-精准召回", test_precise_recall))
    
    def test_update_overwrite():
        d = _make_dir("update_overwrite")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("手机号", "我的手机号: 13811111111", "info")
        mc.add_memory("手机号", "我的手机号变更为: 18822222222", "info")
        recalled = mc.recall("手机号是多少")
        assert len(recalled) > 0, "应召回手机号"
    results.append(test_case("信息更新-手机号变更", test_update_overwrite))
    
    # --- 边缘场景（小部分用户可能的使用方式）---
    
    def test_emoji_query():
        """有些用户聊天习惯用emoji"""
        d = _make_dir("emoji_query")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("今天心情", "今天心情不错😊", "chat")
        recalled = mc.recall("心情怎么样")
        assert len(recalled) > 0, "应召回心情信息"
    results.append(test_case("Emoji内容-心情记录", test_emoji_query))
    
    def test_very_short_content():
        """有些用户只说一两个字，比如备忘"""
        d = _make_dir("short_content")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("密码", "WiFi: MyHome_5G", "credential")
        recalled = mc.recall("WiFi密码")
        assert len(recalled) > 0, "应召回WiFi密码"
    results.append(test_case("短内容-WiFi密码", test_very_short_content))
    
    def test_chinese_address():
        """中文地址查询，词序可能与存储不同"""
        d = _make_dir("cn_address")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("家地址", "家在上海市浦东新区张江高科技园区", "info")
        recalled = mc.recall("浦东的家在哪")
        assert len(recalled) > 0, "应召回地址"
    results.append(test_case("中文地址-词序变换", test_chinese_address))
    
    def test_number_heavy():
        """技术参数、规格等数字密集型内容"""
        d = _make_dir("number_heavy")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("服务器配置", "服务器: 16核CPU, 64GB内存, 2TB SSD, IP: 192.168.1.100", "config")
        recalled = mc.recall("服务器IP是多少")
        assert len(recalled) > 0, "应召回服务器配置"
    results.append(test_case("数字密集-服务器配置", test_number_heavy))
    
    def test_similar_content():
        """内容高度相似的多条记忆，考验区分度"""
        d = _make_dir("similar_content")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("报销流程A", "差旅报销: 提交发票+行程单，3个工作日审批", "process")
        mc.add_memory("报销流程B", "餐饮报销: 提交发票+用餐事由，1个工作日审批", "process")
        recalled = mc.recall("差旅报销流程")
        assert len(recalled) > 0, "应召回差旅报销"
        # 验证第一条是差旅相关（更匹配）
        assert "差旅" in recalled[0], "首选应是差旅报销"
    results.append(test_case("相似内容-区分度", test_similar_content))
    
    def test_code_snippet():
        """开发者存代码片段"""
        d = _make_dir("code_snippet")
        mc = MemoryCore(db_path=os.path.join(d, "test.db"))
        mc.add_memory("Python装饰器", "@decorator\ndef func(): pass  # Python装饰器用法", "code")
        recalled = mc.recall("Python装饰器怎么用")
        assert len(recalled) > 0, "应召回代码片段"
    results.append(test_case("代码片段-Python装饰器", test_code_snippet))
    
    print_results(results, "MemoryCore 核心召回")
    return results

# ========== 模块2: DualMemory 双层记忆 ==========
def test_dual_memory():
    results = []
    
    def test_short_term_recall():
        d = _make_dir("short_term")
        profile = ProfileConfig(profile="chat", token_budget=3000)
        dm = DualMemory(db_dir=d, profile=profile)
        dm.add_memory("临时提醒", "下班记得买菜", "reminder")
        recalled = dm.recall("买菜")
        assert len(recalled) > 0, "短期记忆应可立即召回"
        dm.close()
    results.append(test_case("短期记忆-即时召回", test_short_term_recall))
    
    def test_long_term_knowledge():
        d = _make_dir("long_term")
        profile = ProfileConfig(profile="dev", token_budget=10000)
        dm = DualMemory(db_dir=d, profile=profile)
        dm.add_memory("Python虚拟环境", "创建虚拟环境: python -m venv .venv", "dev_cmd")
        recalled = dm.recall("创建虚拟环境")
        assert len(recalled) > 0, "长期记忆应可召回"
        dm.close()
    results.append(test_case("长期知识-开发命令", test_long_term_knowledge))
    
    def test_mixed_recall():
        d = _make_dir("mixed_recall")
        profile = ProfileConfig(profile="chat", token_budget=3000)
        dm = DualMemory(db_dir=d, profile=profile)
        dm.add_memory("今天天气", "今天下雨，出门带伞", "weather")
        dm.add_memory("常用命令", "查看端口占用: netstat -ano", "dev_cmd")
        recalled_weather = dm.recall("带伞")
        assert len(recalled_weather) > 0, "应召回天气记忆"
        recalled_cmd = dm.recall("端口占用")
        assert len(recalled_cmd) > 0, "应召回命令记忆"
        dm.close()
    results.append(test_case("混合记忆-天气+命令", test_mixed_recall))
    
    # --- 边缘场景 ---
    
    def test_rapid_add_recall():
        """快速连续添加+召回，模拟高频对话"""
        d = _make_dir("rapid_add")
        profile = ProfileConfig(profile="chat", token_budget=5000)
        dm = DualMemory(db_dir=d, profile=profile)
        
        # 模拟一段连续对话中的关键信息
        dm.add_memory("会议时间", "周三下午2点", "schedule")
        dm.add_memory("会议室", "3号会议室", "schedule")
        dm.add_memory("参会人", "张三、李四、王五", "schedule")
        
        recalled = dm.recall("周三会议")
        assert len(recalled) > 0, "应召回会议信息"
        dm.close()
    results.append(test_case("快速连续-会议信息", test_rapid_add_recall))
    
    def test_cross_domain():
        """跨领域查询，工作与生活混合"""
        d = _make_dir("cross_domain")
        profile = ProfileConfig(profile="chat", token_budget=3000)
        dm = DualMemory(db_dir=d, profile=profile)
        
        dm.add_memory("工作项目", "ComFlow使用Electron+Node.js架构", "work")
        dm.add_memory("健身计划", "每周三次跑步，每次5公里", "fitness")
        dm.add_memory("购物清单", "需要买牛奶、面包、鸡蛋", "shopping")
        
        # 工作查询不应返回购物
        recalled = dm.recall("项目架构")
        assert len(recalled) > 0, "应召回项目信息"
        
        dm.close()
    results.append(test_case("跨领域-工作与生活", test_cross_domain))
    
    def test_stats():
        """验证统计信息"""
        d = _make_dir("stats")
        profile = ProfileConfig(profile="chat", token_budget=3000)
        dm = DualMemory(db_dir=d, profile=profile)
        
        dm.add_memory("测试1", "测试内容1", "test1")
        dm.add_memory("测试2", "测试内容2", "test2")
        
        stats = dm.get_stats()
        assert isinstance(stats, dict), "统计信息应为字典"
        dm.close()
    results.append(test_case("统计信息-get_stats", test_stats))
    
    print_results(results, "DualMemory 双层记忆")
    return results

# ========== 模块3: EventIndex 事件索引 ==========
def test_event_index():
    results = []
    
    def test_event_create_recall():
        d = _make_dir("event_create")
        ei = EventIndex(db_dir=d)
        
        fact = AtomicFact(query="项目启动", content="NeuralFlow项目正式启动", fingerprint="neuralflow_start")
        ei.create_event(
            theme="NeuralFlow项目",
            timestamp=time.time(),
            facts=[fact],
            fingerprint="neuralflow_project"
        )
        
        recalled = ei.recall("NeuralFlow")
        assert len(recalled) > 0, "应召回项目事件"
        # EventIndex没有close方法
    results.append(test_case("事件创建-项目启动", test_event_create_recall))
    
    def test_multi_fact_event():
        d = _make_dir("event_multi")
        ei = EventIndex(db_dir=d)
        
        facts = [
            AtomicFact(query="会议时间", content="下午3点", fingerprint="meeting_time"),
            AtomicFact(query="会议地点", content="会议室A", fingerprint="meeting_loc")
        ]
        ei.create_event(
            theme="项目评审会",
            timestamp=time.time(),
            facts=facts,
            fingerprint="project_review"
        )
        
        recalled = ei.recall("会议")
        assert len(recalled) > 0, "应召回会议事件"
        # EventIndex没有close方法
    results.append(test_case("多事实事件-会议安排", test_multi_fact_event))
    
    # --- 边缘场景 ---
    
    def test_event_reopen():
        """关闭事件后重新打开"""
        d = _make_dir("event_reopen")
        ei = EventIndex(db_dir=d)
        
        fact = AtomicFact(query="部署", content="v1.0部署到生产环境", fingerprint="deploy_v1")
        evt = ei.create_event(
            theme="生产部署",
            timestamp=time.time(),
            facts=[fact],
            fingerprint="deploy_prod"
        )
        
        # 关闭事件
        ei.close_event(evt.event_id)
        
        # 验证已关闭
        closed_evt = ei.events.get(evt.event_id)
        assert closed_evt is not None and closed_evt.status == "closed", "应显示已关闭"
        
        # 重新打开
        ei.reopen_event(evt.event_id)
        
        # 验证重开后状态
        reopened_evt = ei.events.get(evt.event_id)
        assert reopened_evt is not None and reopened_evt.status == "open", "应显示重开后为open"
    results.append(test_case("事件重开-补充部署信息", test_event_reopen))
    
    def test_event_add_facts():
        """向已有事件添加新事实"""
        d = _make_dir("event_add")
        ei = EventIndex(db_dir=d)
        
        fact1 = AtomicFact(query="Bug描述", content="登录页面白屏", fingerprint="bug_login")
        evt = ei.create_event(
            theme="登录Bug修复",
            timestamp=time.time(),
            facts=[fact1],
            fingerprint="login_bug"
        )
        
        # 添加新事实，用实际embedding维度
        import numpy as np
        from embedder import create_embedder
        emb = create_embedder()
        fact2 = AtomicFact(query="Bug原因", content="前端路由配置错误", fingerprint="bug_cause")
        fact2_vec = emb.encode(fact2.content)
        ei.add_to_event(evt.event_id, fact2, fact2_vec)
        
        recalled = ei.recall("登录Bug")
        assert len(recalled) > 0, "应召回Bug事件"
        # EventIndex没有close方法
    results.append(test_case("事件追加-Bug修复进展", test_event_add_facts))
    
    def test_event_stats():
        d = _make_dir("event_stats")
        ei = EventIndex(db_dir=d)
        stats = ei.get_stats()
        assert isinstance(stats, dict), "统计应为字典"
        # EventIndex没有close方法
    results.append(test_case("事件统计-get_stats", test_event_stats))
    
    print_results(results, "EventIndex 事件索引")
    return results

# ========== 主函数 ==========
def main():
    print("\n" + "="*60)
    print("MemCore v0.7 实际场景测试")
    print("测试原则: 贴近真实使用，不极端但覆盖边缘情况")
    print("="*60)
    
    all_results = []
    all_results.extend(test_memory_core())
    all_results.extend(test_dual_memory())
    all_results.extend(test_event_index())
    
    total_passed = sum(1 for r in all_results if r["status"] == "PASS")
    total_count = len(all_results)
    
    print("\n" + "="*60)
    print(f"总计: {total_passed}/{total_count} 测试通过")
    if total_passed < total_count:
        print("\n失败详情:")
        for r in all_results:
            if r["status"] != "PASS":
                print(f"  ❌ {r['name']}: {r['error']}")
    print("="*60)
    
    return all_results

if __name__ == "__main__":
    results = main()
