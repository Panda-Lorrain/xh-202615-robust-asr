#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""家居指令语料生成 (扩充版, 不碰 Dataset-A)。

基于人工设计的家居指令词表 (题目公开规格 + 通用智能家居常识) 组合生成
大量家居指令文本, 供 CosyVoice TTS 批量合成 target 干净音频。同时生成
"干扰闲话" 文本池 (新闻/财经/闲聊, 非家居指令) 供 interferer 合成。

输出:
  code/_home_cmd_corpus.jsonl   (target 指令: {id, text, category})
  code/_chitchat_corpus.jsonl   (干扰闲话: {id, text, category})

红线: 不读 Dataset-A 任何音频/标注 (lessons-pitfalls §14); 仅用人工词表组合。
"""
import argparse
import json
import random
from collections import Counter

# ============ 扩充家居指令模板 (题目家居指令域, 20 类) ============
TEMPLATES = {
    "空调": [
        "空调开到制热模式", "空调开到制冷模式", "空调调到除湿模式",
        "空调调到自动模式", "空调调到{t}度", "空调温度调高一点",
        "空调温度调低一点", "空调风速调到{p}", "空调关闭", "打开空调",
        "空调开到睡眠模式", "空调开到防直吹模式", "空调开到无风感",
        "空调开到自清洁", "空调定时{m}小时关闭",
    ],
    "灯": [
        "打开客厅灯", "关闭客厅灯", "打开卧室灯", "关闭卧室灯",
        "打开餐厅灯", "打开书房灯", "所有灯光打开", "所有灯光关闭",
        "灯光亮度调到{p}", "把灯调亮一点", "把灯调暗一点",
        "把灯调成暖色", "把灯调成冷色", "打开夜灯", "关闭夜灯",
    ],
    "洗衣机": [
        "洗衣机开始工作", "洗衣机暂停", "洗衣机继续", "洗衣机开始漂洗",
        "洗衣机开始脱水", "洗衣机调到轻柔模式", "洗衣机调到标准模式",
        "洗衣机定时{m}分钟", "洗衣机关闭",
    ],
    "电视": [
        "打开电视", "关闭电视", "电视调到{ch}频道", "电视音量调到{p}",
        "电视音量调大一点", "电视音量调小一点", "切换到HDMI一号",
        "切换到HDMI二号", "电视调到CCTV{ch}", "电视暂停", "电视继续播放",
    ],
    "窗帘": [
        "打开窗帘", "关上窗帘", "拉起窗帘", "拉下窗帘",
        "窗帘打开一半", "窗帘关到一半", "打开客厅窗帘", "打开卧室窗帘",
    ],
    "音乐": [
        "播放{artist}的歌", "播放{artist}的歌曲", "下一首", "上一首",
        "暂停音乐", "继续播放音乐", "音量调到{p}", "音量调大一点",
        "音量调小一点", "播放轻音乐", "播放流行音乐", "停止播放",
    ],
    "温度": [
        "温度调到{t}度", "把温度设到{t}度", "温度调高一点", "温度调低一点",
        "把温度调到{t}度",
    ],
    "风速": [
        "风量调到{p}", "风速调到最大", "风速调到最小", "风速调大一点",
        "风速调小一点",
    ],
    "模式": [
        "打开回家模式", "打开离家模式", "启动睡眠模式", "退出睡眠模式",
        "打开观影模式", "退出观影模式", "打开阅读模式", "打开用餐模式",
        "启动起床模式", "打开会客模式",
    ],
    "热水器": [
        "打开热水器", "关闭热水器", "热水器调到{t}度", "热水器定时{m}分钟",
        "热水器开始加热",
    ],
    "空气净化器": [
        "打开空气净化器", "关闭空气净化器", "空气净化器调到自动模式",
        "空气净化器调到睡眠模式", "空气净化器调到高速模式",
    ],
    "扫地机器人": [
        "扫地机器人开始清扫", "扫地机器人回充", "扫地机器人暂停",
        "扫地机器人清扫客厅", "扫地机器人清扫卧室", "扫地机器人停止",
    ],
    "电风扇": [
        "打开电风扇", "关闭电风扇", "电风扇开摇头", "电风扇关摇头",
        "电风扇风速调到{p}", "电风扇开自然风",
    ],
    "智能门锁": [
        "把门锁上", "打开门锁", "反锁大门",
    ],
    "加湿器": [
        "打开加湿器", "关闭加湿器", "加湿器湿度调到{p}",
    ],
    "烤箱": [
        "烤箱预热到{t}度", "微波炉加热{m}分钟", "微波炉定时{m}分钟",
        "烤箱定时{m}分钟",
    ],
    "晾衣架": [
        "晾衣架升起来", "晾衣架降下来", "晾衣架停止",
    ],
    "地暖": [
        "打开地暖", "关闭地暖", "地暖调到{t}度",
    ],
    "通用": [
        "我要出门了", "我回来了", "现在几点了", "明天天气怎么样",
        "今天的日程是什么", "帮我定个{m}分钟的闹钟",
        "提醒我{m}分钟后关火", "关闭所有设备",
    ],
}

TEMP_POOL = list(range(18, 31))            # 18-30 度
PCT_POOL = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ARTIST_POOL = ["周杰伦", "邓紫棋", "林俊杰", "王菲", "陈奕迅", "五月天",
               "张学友", "蔡依林", "薛之谦", "李荣浩"]
MIN_POOL = [5, 10, 15, 20, 30, 45, 60, 90]  # 分钟
CH_POOL = list(range(1, 16))               # 频道号


def _fill(tpl: str, rng: random.Random) -> str:
    return tpl.format(
        t=rng.choice(TEMP_POOL),
        p=f"百分之{rng.choice(PCT_POOL)}",
        artist=rng.choice(ARTIST_POOL),
        m=rng.choice(MIN_POOL),
        ch=rng.choice(CH_POOL),
    )


def generate_home_cmds(n: int, seed: int = 42):
    rng = random.Random(seed)
    seen, out, tries = set(), [], 0
    cats = list(TEMPLATES.keys())
    while len(out) < n and tries < n * 20:
        tries += 1
        cat = rng.choice(cats)
        tpl = rng.choice(TEMPLATES[cat])
        text = _fill(tpl, rng)
        key = (cat, text)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": f"home_{len(out):05d}", "text": text, "category": cat})
    return out


# ============ 干扰闲话 (非家居指令, 供 interferer) ============
CHITCHAT = [
    "今天的股市行情怎么样", "你觉得这个新闻靠谱吗", "最近天气真不错啊",
    "昨天那部电影你看了吗", "下周开会准备得怎么样", "这个项目进展如何",
    "听说公司要组织旅游", "孩子学校的事情办好了吗", "周末有什么安排",
    "这道菜怎么做才好吃", "现在油价又涨了吧", "隔壁家装修真吵",
    "上个月的工资发了吗", "健身房的卡快到期了", "你怎么看这个方案",
    "今天地铁人特别多", "那个客户回复了吗", "晚饭想吃点什么",
    "最近睡眠质量不太好", "这本书写得真不错", "比赛规则你清楚吗",
    "我国GDP增长保持了稳定态势", "今年粮食产量再创新高",
    "科技创新是发展的核心动力", "国际贸易形势复杂多变",
    "这家餐厅的菜很有特色", "外卖到了下楼拿一下",
    "你看到那个热搜了吗", "短视频刷到半夜两点",
]


def generate_chitchat(n: int, seed: int = 43):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append({
            "id": f"chat_{i:05d}",
            "text": rng.choice(CHITCHAT),
            "category": "chitchat",
        })
    return out


def main():
    p = argparse.ArgumentParser(description="生成家居指令 + 干扰闲话语料")
    p.add_argument("--n-home", type=int, default=2000)
    p.add_argument("--n-chat", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="E:/midea_target_asr/code")
    args = p.parse_args()

    homes = generate_home_cmds(args.n_home, args.seed)
    chats = generate_chitchat(args.n_chat, args.seed + 1)

    home_path = f"{args.out_dir}/_home_cmd_corpus.jsonl"
    chat_path = f"{args.out_dir}/_chitchat_corpus.jsonl"
    with open(home_path, "w", encoding="utf-8") as f:
        for r in homes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(chat_path, "w", encoding="utf-8") as f:
        for r in chats:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cat_count = Counter(r["category"] for r in homes)
    print(f"home: {len(homes)} 条 -> {home_path}")
    print(f"chitchat: {len(chats)} 条 -> {chat_path}")
    print("home 类别分布:")
    for cat, c in cat_count.most_common():
        print(f"  {cat}: {c}")
    print("home 样例:")
    for r in homes[:8]:
        print(f"  [{r['category']}] {r['text']}")


if __name__ == "__main__":
    main()
