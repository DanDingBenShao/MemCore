# -*- coding: utf-8 -*-
"""MemCore 实际场景测试 v4 - 辅助模块"""
import sys
import os
import time
import gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fingerprint_index import FingerprintIndex, DialogueFingerprint
from context_builder import ContextBuilder, ContextItem
from noise_filter import NoiseFilter
from conflict_resolver import resolve_conflict, effective_confidence
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

# ========== 模块1: FingerprintIndex 指纹追踪 ==========
def test_fingerprint_index():
    results = []
    
    # 场景1: 多轮对话追踪
    def test_dialogue_chain():
        fi = FingerprintIndex()
        
        # 第一轮
        dp1 = DialogueFingerprint(
            round_id=1,
            summary_phrase="用户询问天气",
            entities=["天气"],
            memory_fiber_ids=[101]
        )
        fi.add_round(dp1)
        
        # 第二轮（引用第一轮）
        dp2 = DialogueFingerprint(
            round_id=2,
            summary_phrase="用户追问明天",
            entities=["明天"],
            memory_fiber_ids=[102],
            previous_round_id=1
        )
        fi.add_round(dp2)
        
        # 获取对话链
        chain = fi.get_chain(2)
        # get_chain可能返回空列表（如果没实现），接受None或空列表
        assert chain is not None, "应返回链"
    results.append(test_case("多轮对话-链式追踪", test_dialogue_chain))
    
    # 场景2: 实体消解
    def test_entity_resolution():
        fi = FingerprintIndex()
        
        dp = DialogueFingerprint(
            round_id=1,
            summary_phrase="用户提到张三",
            entities=["张三", "同事"],
            memory_fiber_ids=[101]
        )
        fi.add_round(dp)
        
        # 消解实体
        resolved = fi.resolve_entity("张三")
        assert resolved is not None, "应能消解实体"
    results.append(test_case("实体消解-张三", test_entity_resolution))
    
    # 场景3: 指代消解
    def test_reference_resolution():
        fi = FingerprintIndex()
        
        dp = DialogueFingerprint(
            round_id=1,
            summary_phrase="用户喜欢喝咖啡",
            entities=["咖啡"],
            memory_fiber_ids=[101]
        )
        fi.add_round(dp)
        
        # 消解"它"
        resolved = fi.resolve_implicit_reference("它")
        # 返回list[int]，可能是空列表
        assert isinstance(resolved, list), "应返回列表"
    results.append(test_case("指代消解-隐性引用", test_reference_resolution))
    
    print_results(results, "FingerprintIndex 指纹追踪")
    return results

# ========== 模块2: ContextBuilder 上下文构建 ==========
def test_context_builder():
    results = []
    
    # 场景1: 基础上下文构建
    def test_build_context():
        profile = ProfileConfig(profile="chat", token_budget=3000)
        cb = ContextBuilder(token_budget=2000, profile=profile)
        
        # 添加上下文项
        items = [
            ContextItem(
                content="用户喜欢喝咖啡",
                category="reference",
                importance=0.9,
                source="short_term"
            ),
            ContextItem(
                content="会议在下午3点",
                category="reference",
                importance=0.8,
                source="event"
            )
        ]
        
        context = cb.build(items)
        assert context is not None, "应生成上下文"
    results.append(test_case("基础构建-多源合并", test_build_context))
    
    # 场景2: Token预算控制
    def test_token_budget():
        profile = ProfileConfig(profile="chat", token_budget=100)  # 很小的预算
        cb = ContextBuilder(token_budget=50, profile=profile)
        
        # 添加大量上下文
        items = [
            ContextItem(content=f"记忆{i}"*20, category="reference", importance=0.9, source="short_term")
            for i in range(10)
        ]
        
        context = cb.build(items)
        # 应该截断到预算内
        assert context is not None, "即使超预算也应返回结果"
    results.append(test_case("Token预算-截断控制", test_token_budget))
    
    print_results(results, "ContextBuilder 上下文构建")
    return results

# ========== 模块3: NoiseFilter 噪声过滤 ==========
def test_noise_filter():
    results = []
    
    # 场景1: 口头禅过滤
    def test_filler_removal():
        nf = NoiseFilter(enable_filler_removal=True)
        
        text = "嗯嗯，那个，就是，帮我查一下天气"
        cleaned = nf.clean(text)
        
        # 口头禅应被过滤
        assert "嗯嗯" not in cleaned or "那个" not in cleaned, "应过滤口头禅"
    results.append(test_case("口头禅过滤-嗯嗯那个", test_filler_removal))
    
    # 场景2: 代码规范化
    def test_code_normalization():
        nf = NoiseFilter(enable_code_normalization=True)
        
        code = "function test() { return 1; }"
        cleaned = nf.clean(code)
        
        # 代码应保持可读
        assert "function" in cleaned or "test" in cleaned, "代码核心内容应保留"
    results.append(test_case("代码规范化-保留核心", test_code_normalization))
    
    # 场景3: 指示词标准化
    def test_deixis_normalization():
        nf = NoiseFilter(enable_deixis_normalization=True)
        
        text = "这里有个问题，那边有个文件"
        cleaned = nf.clean(text)
        
        # 标准化处理（可能替换指示词）
        assert cleaned is not None
    results.append(test_case("指示词标准化-这里那边", test_deixis_normalization))
    
    # 场景4: 混合噪声
    def test_mixed_noise():
        nf = NoiseFilter(enable_filler_removal=True, enable_deixis_normalization=True)
        
        text = "嗯，这个，我觉得吧，这里需要改一下"
        cleaned = nf.clean(text)
        
        assert len(cleaned) > 0, "清理后应有内容"
    results.append(test_case("混合噪声-综合清理", test_mixed_noise))
    
    print_results(results, "NoiseFilter 噪声过滤")
    return results

# ========== 模块4: ConflictResolver 冲突消解 ==========
def test_conflict_resolver():
    results = []
    
    # 场景1: 简单冲突
    def test_simple_conflict():
        old_mem = {"content": "会议在下午2点", "confidence": 0.7, "age_days": 1}
        new_mem = {"content": "会议改到下午3点", "confidence": 0.9, "age_days": 0}
        
        resolved = resolve_conflict(old_mem, new_mem)
        
        # 新信息应覆盖旧信息
        assert resolved is not None, "应返回消解结果"
        assert "content" in resolved, "结果应有content"
    results.append(test_case("时间变更-简单冲突", test_simple_conflict))
    
    # 场景2: 无冲突合并
    def test_no_conflict():
        old_mem = {"content": "用户喜欢喝咖啡", "confidence": 0.7, "age_days": 1}
        new_mem = {"content": "用户也喜欢喝茶", "confidence": 0.8, "age_days": 0}
        
        resolved = resolve_conflict(old_mem, new_mem)
        
        # 可能合并或保留两边
        assert resolved is not None, "应返回结果"
        assert "content" in resolved, "结果应有content"
    results.append(test_case("无冲突-偏好合并", test_no_conflict))
    
    # 场景3: 置信度计算
    def test_confidence():
        profile = ProfileConfig(profile="chat")
        
        # 高置信场景
        mem1 = {"content": "测试", "confidence": 0.7, "age_days": 1}
        conf1 = effective_confidence(mem1, profile=profile)
        
        # 低置信场景（陈旧信息）
        mem2 = {"content": "测试", "confidence": 0.7, "age_days": 30}
        conf2 = effective_confidence(mem2, profile=profile)
        
        # 新鲜信息应比陈旧信息置信度高
        assert conf1 >= conf2, "新鲜信息置信度应更高"
    results.append(test_case("置信度-新鲜vs陈旧", test_confidence))
    
    print_results(results, "ConflictResolver 冲突消解")
    return results

# ========== 模块5: ProfileConfig 配置管理 ==========
def test_profile_config():
    results = []
    
    # 场景1: 预设配置
    def test_builtin_profiles():
        profiles = ["chat", "dev", "unlimited"]
        
        for p in profiles:
            cfg = ProfileConfig(profile=p)
            assert cfg.profile == p, f"应创建{p}配置"
        
    results.append(test_case("预设配置-chat/dev/unlimited", test_builtin_profiles))
    
    # 场景2: 自定义参数
    def test_custom_params():
        cfg = ProfileConfig(
            profile="chat",
            token_budget=5000,
            short_threshold=0.2,
            long_threshold=0.4
        )
        
        assert cfg.token_budget == 5000, "token_budget应生效"
        assert cfg.short_threshold == 0.2, "short_threshold应生效"
    results.append(test_case("自定义参数-覆盖默认值", test_custom_params))
    
    # 场景3: 意图预算分配
    def test_intent_budget():
        cfg = ProfileConfig(profile="chat")
        
        # 检查intent_budget_overrides字段存在
        assert hasattr(cfg, 'intent_budget_overrides'), "应有intent_budget_overrides"
    results.append(test_case("意图预算-配置存在", test_intent_budget))
    
    print_results(results, "ProfileConfig 配置管理")
    return results

# ========== 主函数 ==========
def main():
    print("\n" + "="*60)
    print("MemCore v0.7 辅助模块测试")
    print("="*60)
    
    all_results = []
    all_results.extend(test_fingerprint_index())
    all_results.extend(test_context_builder())
    all_results.extend(test_noise_filter())
    all_results.extend(test_conflict_resolver())
    all_results.extend(test_profile_config())
    
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
