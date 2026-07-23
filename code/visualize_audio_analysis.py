"""可视化分析：展示每个音频的详细信息、ground truth、评分标准"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
import os
code_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(code_dir)  # 上级目录是项目根目录
with open(os.path.join(code_dir, "out_neg_final/result.json"), encoding="utf-8") as f:
    neg_with_llm = json.load(f)
with open(os.path.join(code_dir, "out_neg_noLLM/result.json"), encoding="utf-8") as f:
    neg_no_llm = json.load(f)
with open(os.path.join(project_dir, "datasetA/neg.jsonl"), encoding="utf-8") as f:
    neg_gt = [json.loads(line) for line in f]

# 建立映射
neg_with_map = {r["utt_id"]: r for r in neg_with_llm["results"]}
neg_no_map = {r["utt_id"]: r for r in neg_no_llm["results"]}
neg_gt_map = {f"utt{str(item['id']).zfill(4)}_cmd_{item['id']}": item for item in neg_gt}

# 那 11 条 LLM 放过的样本
target_utts = [
    'utt0036_cmd_1036', 'utt0070_cmd_1070', 'utt0099_cmd_5015',
    'utt0131_cmd_5054', 'utt0133_cmd_5056', 'utt0170_cmd_5097',
    'utt0248_cmd_5180', 'utt0391_cmd_5332', 'utt0443_cmd_5386',
    'utt0444_cmd_5387', 'utt0468_cmd_5249'
]

# 创建可视化
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)

# ========== 图1: 评分标准说明 ==========
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
title_text = """
【任务定义与评分标准】

任务：目标说话人 ASR —— 只转写目标说话人（唤醒者）的指令，拒识非目标说话人

输入：
  • 唤醒音频 (enrollment): 唤醒者说唤醒词（如"你好科慕"、"hi colmo"），约 1.8 秒
  • 识别音频 (recognition): 带噪 + 多说话人重叠的音频，需要识别其中的指令

输出：
  • text: 转写文本（如果是目标说话人的指令）
  • rejected: 是否拒识（True=拒识，False=接受）

评分标准（100分）：
  ┌─────────────────────────────────────────────────────────────┐
  │ CER（字符错误率）40分:  越低越好，衡量转写准确度              │
  │ RR（拒识率）    40分:  越高越好，衡量是否正确拒识非目标       │
  │ 效率（RTF+显存）20分:  越快越好，衡量推理速度                 │
  └─────────────────────────────────────────────────────────────┘

关键逻辑：
  • POS 集（正样本）: 识别文本 = ground truth，计算 CER
  • NEG 集（负样本）: 识别文本 = null，应该拒识，计算 RR
  • 声纹匹配 (sim): 判断是否是目标说话人，sim < threshold 则拒
  • LLM 拒识: 只看文本内容，无法判断说话人身份（根本局限）
"""
ax1.text(0.05, 0.95, title_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax1.set_title('任务定义与评分标准', fontsize=14, fontweight='bold', pad=20)

# ========== 图2: 11条 LLM 放过的样本 - sim 分布 ==========
ax2 = fig.add_subplot(gs[1, 0])
sims = []
texts_short = []
for utt_id in target_utts:
    r = neg_with_map[utt_id]
    sims.append(r["max_sim"])
    texts_short.append(r["text"][:8] + "...")

colors = ['red' if s < 0.1 else 'orange' if s < 0.2 else 'green' for s in sims]
bars = ax2.barh(range(len(sims)), sims, color=colors, alpha=0.7)
ax2.set_yticks(range(len(sims)))
ax2.set_yticklabels(texts_short, fontsize=9)
ax2.set_xlabel('声纹相似度 (sim)')
ax2.set_title('11条 LLM 放过的 neg 样本 - sim 分布', fontsize=12, fontweight='bold')
ax2.axvline(x=0.27, color='red', linestyle='--', label='拒识阈值 (0.27)')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# 添加 sim 值标签
for i, (bar, sim) in enumerate(zip(bars, sims)):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
             f'{sim:.3f}', ha='left', va='center', fontsize=8)

# ========== 图3: 开关 LLM 对比 - 决策差异 ==========
ax3 = fig.add_subplot(gs[1, 1])

# 统计决策差异
decisions = []
for utt_id in target_utts:
    with_r = neg_with_map[utt_id]
    no_r = neg_no_map[utt_id]
    with_rej = with_r["rejected"]
    no_rej = no_r["rejected"]

    if no_rej and not with_rej:
        decisions.append("LLM 放过\n(应拒→实接)")
    elif not no_rej and with_rej:
        decisions.append("LLM 抓住\n(应拒→实拒)")
    elif no_rej and with_rej:
        decisions.append("都拒\n(正确)")
    else:
        decisions.append("都接\n(错误)")

from collections import Counter
decision_counts = Counter(decisions)
labels = list(decision_counts.keys())
values = list(decision_counts.values())
colors_pie = ['#ff6b6b', '#51cf66', '#339af0', '#ffd43b']

wedges, texts, autotexts = ax3.pie(values, labels=labels, autopct='%1.1f%%',
                                     colors=colors_pie[:len(labels)], startangle=90)
ax3.set_title('开关 LLM 决策对比', fontsize=12, fontweight='bold')

# ========== 图4: 每个音频的详细信息表格 ==========
ax4 = fig.add_subplot(gs[2:, :])
ax4.axis('off')

# 准备表格数据
table_data = []
for i, utt_id in enumerate(target_utts, 1):
    with_r = neg_with_map[utt_id]
    no_r = neg_no_map[utt_id]
    gt = neg_gt_map.get(utt_id, {})

    # 决策状态
    with_rej = with_r["rejected"]
    no_rej = no_r["rejected"]

    if no_rej and not with_rej:
        status = "⚠️ LLM 放过"
        status_color = 'red'
    elif not no_rej and with_rej:
        status = "✅ LLM 抓住"
        status_color = 'green'
    elif no_rej and with_rej:
        status = "✅ 都拒(正确)"
        status_color = 'green'
    else:
        status = "❌ 都接(错误)"
        status_color = 'red'

    # ground truth
    gt_text = gt.get("识别文本", "null")
    gt_display = "null (应拒识)" if gt_text is None or gt_text == "null" else gt_text

    table_data.append([
        str(i),
        utt_id,
        f"{with_r['max_sim']:.3f}",
        with_r["text"][:20] + ("..." if len(with_r["text"]) > 20 else ""),
        "拒" if no_rej else "接",
        "拒" if with_rej else "接",
        status,
        gt_display[:25] + ("..." if len(gt_display) > 25 else "")
    ])

columns = ['#', 'utt_id', 'sim', 'ASR转写', '关LLM', '开LLM', '状态', 'Ground Truth']

table = ax4.table(cellText=table_data, colLabels=columns, loc='center',
                  cellLoc='center', colWidths=[0.03, 0.12, 0.06, 0.15, 0.06, 0.06, 0.12, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

# 设置表头样式
for j, col in enumerate(columns):
    table[0, j].set_facecolor('#4CAF50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# 设置状态列颜色
for i in range(len(table_data)):
    status = table_data[i][6]
    if "放过" in status:
        table[i+1, 6].set_facecolor('#ffcccc')
    elif "抓住" in status or "都拒" in status:
        table[i+1, 6].set_facecolor('#ccffcc')

ax4.set_title('11条 LLM 放过的 neg 样本详细分析', fontsize=14, fontweight='bold', pad=20)

# ========== 图5: 关键发现总结 ==========
fig.text(0.5, 0.02, """
【关键发现】
• 这 11 条样本的共同特点：非目标说话人说了家居指令（如"打开洗碗机"、"打开智能灯"）
• sim 值都很低（0.01-0.27），声纹模块正确判断为非目标
• LLM 只看文本内容，无法区分说话人身份，所以判为 accept
• 这是 LLM 在"目标说话人 ASR"任务中的根本局限：它只能判断"文本是不是指令"，无法判断"这是不是目标说的指令"
""", ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('目标说话人 ASR - LLM 拒识影响分析可视化', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('llm_impact_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 可视化已保存到: llm_impact_analysis.png")

# ========== 额外：创建详细的音频信息表格 ==========
print("\n" + "=" * 100)
print("【11条 LLM 放过的 neg 样本 - 详细信息】")
print("=" * 100)

for i, utt_id in enumerate(target_utts, 1):
    with_r = neg_with_map[utt_id]
    no_r = neg_no_map[utt_id]
    gt = neg_gt_map.get(utt_id, {})

    print(f"\n{'='*80}")
    print(f"#{i} {utt_id}")
    print(f"{'='*80}")
    print(f"📍 音频路径:")
    print(f"   唤醒音频: {with_r['enrollment']}")
    print(f"   识别音频: {with_r['recognition']}")
    print(f"")
    print(f"🎤 Ground Truth (官方答案):")
    print(f"   识别文本: {gt.get('识别文本', 'null')}")
    print(f"   唤醒文本: {gt.get('唤醒文本', 'N/A')}")
    print(f"   → 应该: {'拒识 (识别文本=null)' if gt.get('识别文本') is None else '接受并转写'}")
    print(f"")
    print(f"📊 声纹匹配:")
    print(f"   sim = {with_r['max_sim']:.4f} {'< 0.27 (应拒)' if with_r['max_sim'] < 0.27 else '>= 0.27 (可接)'}")
    print(f"")
    print(f"🤖 ASR 转写结果:")
    print(f"   text = '{with_r['text']}'")
    print(f"")
    print(f"⚖️ 决策对比:")
    print(f"   关 LLM (sim_only): {'拒识' if no_r['rejected'] else '接受'}")
    print(f"   开 LLM (llm_or_sim): {'拒识' if with_r['rejected'] else '接受'}")
    print(f"   LLM 判定: {with_r.get('llm_verdict', 'N/A')}")

    # 问题分析
    print(f"")
    print(f"❓ 问题分析:")
    if gt.get('识别文本') is None:
        print(f"   • 官方标注: 这是非目标说话人，应该拒识")
        print(f"   • 声纹判断: sim={with_r['max_sim']:.3f} < 0.27，正确判断为非目标")
        print(f"   • LLM 判断: 看到 '{with_r['text']}' 是家居指令，判为 accept")
        print(f"   • 根本原因: LLM 只看文本，不知道谁在说话")
        print(f"   • 结论: LLM 放过了非目标，RR 受损")
    else:
        print(f"   • 官方标注: 这是目标说话人，应该接受")
        print(f"   • 声纹判断: sim={with_r['max_sim']:.3f}，判断为目标/非目标")
        print(f"   • LLM 判断: {with_r.get('llm_verdict', 'N/A')}")
