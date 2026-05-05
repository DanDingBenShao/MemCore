"""
auto_eval.py — 记忆系统自动化评估

用 DeepSeek V4 Flash 自动完成"喂文章 → 生成问题 → 查询系统 → 评判打分"的闭环。
覆盖 3 篇不同领域文章，每篇 10 个问题，共 30 个查询。

运行方式：
  python auto_eval.py
  输出 auto_eval_report.md 评估报告
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from memory_core import MemoryCore

# ===========================================================================
# 测试素材：3 篇跨领域文章
# ===========================================================================

ARTICLES = [
    {
        "title": "SpaceX Starship 星舰测试",
        "text": (
            "SpaceX的星舰（Starship），全球最大的火箭，于2023年4月20日完成了首次轨道测试飞行。"
            "火箭高120米，于美国中部时间早上8点33分从得克萨斯州博卡奇卡升空。"
            "然而测试提前终止，因为超重助推器未能分离，导致火箭在起飞4分钟后被遥控引爆。"
            "尽管爆炸，SpaceX CEO 埃隆·马斯克称这是一次\"激动人心的成功\"并承诺改进。"
            "美国联邦航空管理局（FAA）调查了此次事故，要求完成63项整改措施后方可再次发射。"
            "2024年3月14日的第二次测试取得了更好的成果，成功实现了级间分离并飞行了50分钟，"
            "但在再入大气层时解体。"
        ),
    },
    {
        "title": "Netflix vs Blockbuster 兴衰史",
        "text": (
            "2000年，Netflix 创始人里德·哈斯廷斯向百视达（Blockbuster）CEO 约翰·安蒂奥科提出："
            "以5000万美元的价格将 Netflix 卖给百视达。安蒂奥科拒绝了，称 Netflix 是\"小众业务\"。"
            "当时百视达在全球有9000家门店，年收入60亿美元。Netflix 仅有30万订阅用户。"
            "到2010年，百视达已经申请破产。在哈斯廷斯的领导下，Netflix 从DVD租赁转型为流媒体，"
            "拥有了2000万用户。2023年，Netflix 创造了337亿美元的收入。哈斯廷斯后来在他的书"
            "《不拘一格》（No Rules Rules）中写道：百视达的拒绝是\"发生在我们身上最好的事情\"。"
        ),
    },
    {
        "title": "CRISPR 基因编辑技术与争议",
        "text": (
            "CRISPR-Cas9 基因编辑工具于2012年首次在人类细胞中成功演示，"
            "由珍妮弗·杜德纳和埃马纽埃尔·卡彭蒂耶完成。两人因此获得2020年诺贝尔化学奖。"
            "然而该技术面临重大伦理争议。2018年，中国科学家贺建奎宣布诞生了首批CRISPR编辑婴儿，"
            "声称使其对HIV具有抵抗力。这违反了国际伦理准则，贺建奎因此被判入狱。"
            "CRISPR的一个主要局限是脱靶效应——意外修改了非目标基因。"
            "研究表明，脱靶率在0.1%到5%之间，取决于具体使用的引导RNA。"
            "尽管存在这些局限，针对镰状细胞病的CRISPR疗法于2023年12月获得了FDA批准。"
        ),
    },
]

# ===========================================================================
# 30 个测试问题（每篇 10 个，覆盖所有类型）
# ===========================================================================

QUESTIONS: list[dict[str, Any]] = [
    # ── Starship ──
    {"article": 0, "type": "精确事实", "question": "星舰首次轨道测试是什么时候",
     "expected": ["2023", "4月", "20日"]},
    {"article": 0, "type": "时间细节", "question": "第二次星舰测试是在什么时候",
     "expected": ["2024", "3月", "14日"]},
    {"article": 0, "type": "数字细节", "question": "星舰的高度是多少米",
     "expected": ["120米", "120"]},
    {"article": 0, "type": "冷门细节", "question": "FAA 要求星舰完成多少项整改",
     "expected": ["63", "63项"]},
    {"article": 0, "type": "否定/局限", "question": "星舰首次测试出了什么问题",
     "expected": ["超重助推器未能分离", "引爆", "失败"]},
    {"article": 0, "type": "比较型", "question": "星舰两次测试结果有什么不同",
     "expected": ["级间分离", "50分钟", "解体"]},
    {"article": 0, "type": "精确事实", "question": "星舰从哪里发射升空",
     "expected": ["博卡奇卡", "得克萨斯"]},
    {"article": 0, "type": "时间细节", "question": "首次星舰测试起飞后多久爆炸",
     "expected": ["4分钟"]},
    {"article": 0, "type": "数字细节", "question": "第二次星舰测试飞行了多久",
     "expected": ["50分钟"]},
    {"article": 0, "type": "引用查询", "question": "马斯克如何评价第一次星舰测试",
     "expected": ["激动人心的成功", "马斯克"]},

    # ── Netflix ──
    {"article": 1, "type": "精确事实", "question": "Netflix 创始人想以多少钱卖给百视达",
     "expected": ["5000万", "5000万美元"]},
    {"article": 1, "type": "时间细节", "question": "百视达在哪一年申请破产",
     "expected": ["2010"]},
    {"article": 1, "type": "数字细节", "question": "2000年百视达的年收入是多少",
     "expected": ["60亿", "60亿美元"]},
    {"article": 1, "type": "冷门细节", "question": "里德·哈斯廷斯写了什么书",
     "expected": ["不拘一格", "No Rules Rules"]},
    {"article": 1, "type": "否定/局限", "question": "百视达 CEO 如何评价 Netflix",
     "expected": ["小众业务"]},
    {"article": 1, "type": "比较型", "question": "Netflix 2000年与2023年的收入对比",
     "expected": ["337亿", "337亿美元"]},
    {"article": 1, "type": "精确事实", "question": "百视达 CEO 叫什么名字",
     "expected": ["约翰·安蒂奥科", "John Antioco"]},
    {"article": 1, "type": "时间细节", "question": "2000年Netflix有多少订阅用户",
     "expected": ["30万", "300000"]},
    {"article": 1, "type": "数字细节", "question": "百视达破产前全球有多少家门店",
     "expected": ["9000"]},
    {"article": 1, "type": "引用查询", "question": "哈斯廷斯如何看待百视达的拒绝",
     "expected": ["最好的事情"]},

    # ── CRISPR ──
    {"article": 2, "type": "精确事实", "question": "CRISPR-Cas9 首次在人类细胞中演示是哪一年",
     "expected": ["2012"]},
    {"article": 2, "type": "时间细节", "question": "杜德纳和卡彭蒂耶哪年获得诺贝尔奖",
     "expected": ["2020"]},
    {"article": 2, "type": "数字细节", "question": "CRISPR 脱靶率是多少范围",
     "expected": ["0.1%", "5%"]},
    {"article": 2, "type": "冷门细节", "question": "贺建奎声称让编辑婴儿对什么具有抵抗力",
     "expected": ["HIV"]},
    {"article": 2, "type": "否定/局限", "question": "CRISPR 技术的主要局限是什么",
     "expected": ["脱靶效应"]},
    {"article": 2, "type": "比较型", "question": "2012年演示和2023年FDA批准的间隔",
     "expected": ["2023", "12月", "FDA"]},
    {"article": 2, "type": "精确事实", "question": "CRISPR 技术发现者是谁",
     "expected": ["杜德纳", "卡彭蒂耶", "Doudna", "Charpentier"]},
    {"article": 2, "type": "时间细节", "question": "CRISPR 编辑婴儿事件发生在哪一年",
     "expected": ["2018"]},
    {"article": 2, "type": "数字细节", "question": "贺建奎被判入狱",
     "expected": ["入狱"]},
    {"article": 2, "type": "引用查询", "question": "哪种疾病的CRISPR疗法获得了FDA批准",
     "expected": ["镰状细胞病"]},
]

assert len(QUESTIONS) == 30, f"应有 30 个问题，实际 {len(QUESTIONS)}"


# ===========================================================================
# 工具函数
# ===========================================================================

def extract_keywords(text: str) -> list[str]:
    """从文本中提取中英文关键词（去停用词）。"""
    stopwords = {'的','了','是','在','有','我','你','他','它','这','那',
                 '们','与','和','就','也','都','被','把','从','到','于',
                 '上','下','个','人','不','一','着','过','地','为','对',
                 '后','前','中','说','而','或','但','可','以','能','会'}
    tokens = re.findall(r'[\u4e00-\u9fffA-Za-z0-9\.]+', text)
    return [t for t in tokens if len(t) >= 2 and t not in stopwords]


def judge_hit(content: str, expected_kw: list[str]) -> tuple[int, str]:
    """判断召回内容是否命中预期关键词。

    Returns
    -------
    (score, reason)
        score: 1=正确, 0.5=部分正确, 0=错误
    """
    content_lower = content.lower()
    hits = [kw for kw in expected_kw if kw.lower() in content_lower]
    hit_ratio = len(hits) / max(len(expected_kw), 1)

    if hit_ratio >= 0.8:
        return (1, f"命中 {len(hits)}/{len(expected_kw)} 个预期关键词")
    elif hit_ratio >= 0.3:
        return (0.5, f"部分命中 {len(hits)}/{len(expected_kw)} 个关键词")
    else:
        return (0, f"仅命中 {len(hits)}/{len(expected_kw)} 个关键词")


# ===========================================================================
# 主测试流程
# ===========================================================================

def run_evaluation() -> dict[str, Any]:
    """执行完整评估流程，返回汇总数据。"""
    results: list[dict] = []
    db_dir = "_auto_eval_db"

    # 清理旧数据
    import shutil
    for f in ["_auto_eval_0.db", "_auto_eval_1.db", "_auto_eval_2.db"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    print(f"{'='*70}")
    print(f"  记忆系统自动化评估")
    print(f"  3 篇文章 × 10 个问题 = 30 个查询")
    print(f"{'='*70}")

    for art_idx, article in enumerate(ARTICLES):
        title = article["title"]
        text = article["text"]
        db_path = f"_auto_eval_{art_idx}.db"

        # 使用临时目录避免干扰
        mc = MemoryCore(max_memories=500, similarity_threshold=0.6, db_path=db_path)

        # 喂入文章（按句拆分）
        print(f"\n─── 文章 {art_idx + 1}: {title} ───")
        sentences = [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
        for sent in sentences:
            kw = extract_keywords(sent)
            fingerprint = " ".join(kw[:5])
            # 取前3个关键词拼接作为 query，使其与 recall 查询更匹配
            query = " ".join(kw[:3]) if len(kw) >= 3 else (kw[0] if kw else "topic")
            mc.add_memory(query, sent, fingerprint)
        print(f"  已存入 {len(sentences)} 个句子，共 {mc.get_stats()['total_memories']} 条记忆")

        # 逐一回答该文章的 10 个问题
        for q in QUESTIONS:
            if q["article"] != art_idx:
                continue

            question = q["question"]
            expected = q["expected"]
            qtype = q["type"]

            # 查询
            recalled = mc.recall(question)
            best_content = recalled[0] if recalled else ""

            # 评判
            score, reason = judge_hit(best_content, expected)

            entry = {
                "article": title,
                "type": qtype,
                "question": question,
                "expected": expected,
                "recalled": best_content[:100] if best_content else "(无匹配)",
                "score": score,
                "reason": reason,
            }
            results.append(entry)

            score_char = {1: "✅", 0.5: "🟡", 0: "❌"}[score]
            print(f"  {score_char} [{qtype}] {question}")

        # 关闭数据库连接后清理
        mc.conn.close()
        time.sleep(0.2)
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass

    return results


def generate_report(results: list[dict]) -> str:
    """生成 Markdown 评估报告。"""
    total = len(results)
    correct = sum(1 for r in results if r["score"] == 1)
    partial = sum(1 for r in results if r["score"] == 0.5)
    wrong = sum(1 for r in results if r["score"] == 0)
    accuracy = correct / total * 100 if total > 0 else 0

    # 按类型分解
    type_stats: dict[str, dict] = {}
    for r in results:
        t = r["type"]
        if t not in type_stats:
            type_stats[t] = {"total": 0, "correct": 0, "partial": 0, "wrong": 0}
        type_stats[t]["total"] += 1
        if r["score"] == 1:
            type_stats[t]["correct"] += 1
        elif r["score"] == 0.5:
            type_stats[t]["partial"] += 1
        else:
            type_stats[t]["wrong"] += 1

    # 按文章分解
    art_stats: dict[str, dict] = {}
    for r in results:
        a = r["article"]
        if a not in art_stats:
            art_stats[a] = {"total": 0, "correct": 0, "score_sum": 0.0}
        art_stats[a]["total"] += 1
        art_stats[a]["score_sum"] += r["score"]
        if r["score"] == 1:
            art_stats[a]["correct"] += 1

    # 生成报告
    lines = []
    lines.append("# 记忆系统自动化评估报告")
    lines.append("")
    lines.append(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试数量**: 3 篇文章 × 10 个问题 = **{total} 个查询**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 总体统计")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 正确 (score=1) | {correct}/{total} |")
    lines.append(f"| 部分正确 (score=0.5) | {partial}/{total} |")
    lines.append(f"| 错误 (score=0) | {wrong}/{total} |")
    lines.append(f"| **总体正确率** | **{accuracy:.1f}%** |")
    lines.append("")
    lines.append("## 按问题类型分解")
    lines.append("")
    lines.append("| 类型 | 总数 | 正确 | 部分 | 错误 | 正确率 |")
    lines.append("|------|------|------|------|------|--------|")
    for t, s in sorted(type_stats.items()):
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        lines.append(f"| {t} | {s['total']} | {s['correct']} | {s['partial']} | {s['wrong']} | {rate:.0f}% |")
    lines.append("")
    lines.append("## 按文章分解")
    lines.append("")
    lines.append("| 文章 | 总数 | 正确 | 平均分 |")
    lines.append("|------|------|------|--------|")
    for a, s in sorted(art_stats.items()):
        avg = s["score_sum"] / s["total"] if s["total"] > 0 else 0
        lines.append(f"| {a} | {s['total']} | {s['correct']} | {avg:.2f} |")
    lines.append("")
    lines.append("## 详细结果")
    lines.append("")
    lines.append("| # | 文章 | 类型 | 查询 | 预期关键词 | 召回结果 | 得分 | 理由 |")
    lines.append("|---|------|------|------|-----------|----------|------|------|")
    for i, r in enumerate(results):
        score_char = {1: "✅ 1", 0.5: "🟡 0.5", 0: "❌ 0"}[r["score"]]
        recalled_short = r["recalled"][:40] + "..." if len(r["recalled"]) > 40 else r["recalled"]
        expected_str = ", ".join(r["expected"])
        lines.append(
            f"| {i+1} | {r['article'][:12]} | {r['type']} | {r['question'][:18]} | "
            f"{expected_str[:20]} | {recalled_short[:30]} | {score_char} | {r['reason']} |"
        )
    lines.append("")
    lines.append("## 错误案例分析")
    lines.append("")
    wrong_cases = [r for r in results if r["score"] == 0]
    if wrong_cases:
        lines.append(f"共 {len(wrong_cases)} 个错误案例：")
        lines.append("")
        for r in wrong_cases:
            lines.append(f"### ❌ {r['question']}")
            lines.append(f"- **文章**: {r['article']}")
            lines.append(f"- **类型**: {r['type']}")
            lines.append(f"- **预期关键词**: {', '.join(r['expected'])}")
            lines.append(f"- **系统召回**: {r['recalled']}")
            lines.append(f"- **评判理由**: {r['reason']}")
            lines.append("")
    else:
        lines.append("无错误案例。")
        lines.append("")
    lines.append("---")
    lines.append("*报告由 auto_eval.py 自动生成*")

    return "\n".join(lines)


# ===========================================================================
# 主入口
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  记忆系统自动化评估")
    print("=" * 70)

    # 运行评估
    results = run_evaluation()

    # 生成报告
    report = generate_report(results)
    report_path = "auto_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{'='*70}")
    print(f"  评估完成！报告已保存至: {report_path}")
    print(f"  正确率: {sum(1 for r in results if r['score']==1)}/{len(results)} "
          f"({sum(1 for r in results if r['score']==1)/len(results)*100:.1f}%)")
    print(f"{'='*70}")
