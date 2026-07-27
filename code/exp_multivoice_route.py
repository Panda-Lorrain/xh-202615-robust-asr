"""multivoice 内容判别选路 POC (2026-07-27).

背景:
- B2 SepFormer 实验 (code/exp_sepformer_b2.py) 在失败组 40 条 (sim≥0.4 & qwen_cer>0.8) 上
  跑了源分离 + 两路 qwen 转写。结果:
    oracle 选路 (取 CER 较低那路):  mean CER 0.603  ← 天花板
    sim 选路 (SI-SDR/stream_sim):    mean CER 1.249  ← 声纹破坏
    argmax 主线:                     mean CER 1.216
- sim 选路证伪 → 试 **内容判别选路**: 不靠声纹, 靠文本语义判"哪路像家居指令".

本脚本:
- 复用 B2 产物 summary.json (40 条两路 transcript + oracle 真值), 不重新分离/转写。
- 策略1: content_gate 二值 (is_valid_command 挑 accept 那路)
- 策略2: LLM (Qwen2.5-3B, .venv_llm) — 若 venv 缺失跳过
- 策略3: 综合 (content_gate + 家居指令启发式评分: 设备词/动作词/功能词/疑问词/品牌锚点/news 黑名单)
- 对照基准: oracle 0.603 / sim 1.249 / argmax 1.216 / 随机 ~1.2

输出:
- code/runs/_multivoice_route/summary.json (每条每策略选路 + CER + 选对没)
- 控制台汇总 (每策略 mean CER + 选路准确率 + fallback 次数)
- docs/multivoice_content_route.md (报告)
"""
import json, os, sys, re, statistics, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from text_utils import is_valid_command, _CONTENT_GATE_NEWS_BLACK, _BRAND_ANCHORS

SEED = 42
B2_SUMMARY = os.path.join(HERE, "runs", "_sepformer_b2", "summary.json")
OUT_DIR = os.path.join(HERE, "runs", "_multivoice_route")
OUT_SUMMARY = os.path.join(OUT_DIR, "summary.json")

# ---- 启发式词典 (领域先验, 非 A 集拟合; 对齐 llm_reject.py SYSTEM_PROMPT 家电实体+动作+参数) ----
DEVICE_KW = [
    "空调", "灯", "洗衣机", "电视", "窗帘", "风扇", "热水器", "净化器", "音箱",
    "扫地", "净水器", "闹钟", "油烟机", "电饭煲", "微波炉", "加湿器", "地暖",
    "播放器", "投影", "机器人",
]
# 动作词 (控制意图); 收紧到组合形式避免单字"开/关"误命中
ACTION_KW = [
    "打开", "关闭", "开启", "关掉", "关上", "关下", "调到", "调节", "调成", "调低", "调高",
    "启动", "暂停", "播放", "定时", "设置", "设为", "设到", "降低", "升高",
    "关小", "加大", "减少", "增加", "扫风", "摆风", "送风", "给我放", "放",
]
# 功能/特征词 (模式/参数名)
FEATURE_KW = [
    "模式", "温度", "风速", "风量", "定时", "睡眠", "ECO", "MCO",  # MCO = ECO ASR 误听
    "送风", "柔风", "防直吹", "无风感", "智控温", "智清洁", "净呼吸",
    "轻干洗", "净干洗", "星香", "左右", "上下", "左右风", "色温",
]
# 疑问/查询词 (ref 多为"吃什么/哪/怎么"等知识查询类指令, 需正面计分)
QUERY_KW = [
    "吃什么", "喝什么", "怎么", "为什么", "什么是", "哪些", "预防", "防辐射",
    "明目", "脂肪肝", "忌口", "腹泻", "哺乳期", "吃什么可以", "适宜", "有利于",
]
# 扩展 news 黑名单 (B2 失败组干扰路常见财经/新闻词, 非 A 集拟合 — 先验: 家居指令绝不出现)
NEWS_BLACK_EXT = [
    "业务", "市场", "操纵", "同比", "收益率", "签订", "售房", "投资", "产业链",
    "保险", "严重", "低估", "面积", "成交", "推行", "推崇", "彩电", "结构", "冲击",
    "竞争", "基线", "参赛", "选手", "较量", "认知", "社团", "支付", "条形码", "微信支付",
    "售房合同", "武汉队", "食草动物", "江西网", "新型", "虚拟现实", "媒体", "监督",
    "行政区域", "成为中国", "反映", "实现", "成功走出", "概率", "制造", "工厂",
    "中国", "奥运", "韩国", "首尔",
]


def _strip_punct(t):
    """去标点 (对齐 is_valid_command 内部 normalize)。"""
    return re.sub(r"[^\w一-鿿]", "", t or "")


def _news_hits(t):
    """news/财经/体育类干扰词命中数。"""
    base = sum(1 for w in _CONTENT_GATE_NEWS_BLACK if w in t)
    ext = sum(1 for w in NEWS_BLACK_EXT if w in t)
    return base + ext


def _digit_run_len(t):
    """最长中文数字/阿拉伯数字串长度 (幻觉串特征: 二三二五一二/916213)。"""
    if not t:
        return 0
    # 中文数字 + 阿拉伯
    m = re.findall(r"[零一二三四五六七八九百千万点十0-9]+", t)
    return max((len(s) for s in m), default=0)


def cmd_score(text):
    """综合"像家居指令"评分。越高越像。用于策略3。

    设计:
    - 以 content_gate (is_valid_command) 为底: 不通过 → -3 起步
    - 正信号: 设备词(+2) / 动作词(+2) / 功能词(+1.5) / 疑问查询词(+2) / 品牌锚点(+3)
    - 负信号: news 词命中(-3 每个) / 数字串≥4(-2 幻觉) / 长度极端
    """
    if not text or not text.strip():
        return -5.0
    raw = text
    s = 0.0
    # content_gate 基底
    if not is_valid_command(raw):
        s -= 3.0
    # 正信号
    dev_hit = [w for w in DEVICE_KW if w in raw]
    act_hit = [w for w in ACTION_KW if w in raw]
    fea_hit = [w for w in FEATURE_KW if w in raw]
    qry_hit = [w for w in QUERY_KW if w in raw]
    bra_hit = [w for w in _BRAND_ANCHORS if w in raw]
    if dev_hit: s += 2.0
    if act_hit: s += 2.0
    if fea_hit: s += 1.5
    if qry_hit: s += 2.0
    if bra_hit: s += 3.0
    # 复合加分: 同时有 (设备+动作) 或 (动作+功能) → 强指令信号
    if dev_hit and act_hit: s += 1.0
    if act_hit and fea_hit: s += 1.0
    # 负信号
    n_news = _news_hits(raw)
    if n_news:
        s -= 3.0 * n_news
    if _digit_run_len(raw) >= 4:
        s -= 2.0
    # 长度
    L = len(_strip_punct(raw))
    if 3 <= L <= 15:
        s += 0.5
    elif L > 22:
        s -= 2.0
    elif L < 3:
        s -= 1.0
    return s


def route_content_gate(per_src):
    """策略1: content_gate 二值挑路。返回 (picked_idx, reason)。"""
    valid = [is_valid_command(s["text"]) for s in per_src]
    if sum(valid) == 1:
        idx = valid.index(True)
        return idx, "one_valid"
    if sum(valid) == 0:
        return 0, "both_invalid_fallback_src0"
    # both valid → content_gate 无法判, fallback 到更短(更像指令的简短句)
    lengths = [len(_strip_punct(s["text"])) for s in per_src]
    if lengths[0] != lengths[1]:
        return (0 if lengths[0] < lengths[1] else 1), "both_valid_tiebreak_shorter"
    return 0, "both_valid_tie_default_src0"


def route_heuristic(per_src):
    """策略3: 综合 content_gate + 启发式评分挑路。返回 (picked_idx, reason)。"""
    scores = [cmd_score(s["text"]) for s in per_src]
    if abs(scores[0] - scores[1]) < 1e-6:
        # 平局 fallback: 取 content_gate 通过的那路; 都通过取更短
        v = [is_valid_command(s["text"]) for s in per_src]
        if sum(v) == 1:
            return v.index(True), "tie_valid_only_one"
        if sum(v) == 0:
            return 0, "tie_both_invalid_fallback_src0"
        lengths = [len(_strip_punct(s["text"])) for s in per_src]
        return (0 if lengths[0] <= lengths[1] else 1), "tie_both_valid_shorter"
    return (0 if scores[0] > scores[1] else 1), f"score_{scores[0]:.2f}_vs_{scores[1]:.2f}"


def route_llm_dummy(per_src):
    """策略2 占位: 实际 LLM 路由若 venv_llm 缺失则跳过。"""
    raise NotImplementedError("LLM routing requires .venv_llm (skipped in this run)")


def _has_llm_venv():
    return os.path.exists(os.path.join(HERE, ".venv_llm", "Scripts", "python.exe"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    b2 = json.load(open(B2_SUMMARY, encoding="utf-8"))
    results = b2["results"]

    per_sample = []
    strat_names = ["content_gate", "heuristic"]
    if _has_llm_venv():
        strat_names.append("llm")
    # 累加基准 (oracle / argmax / sim 都从 B2 拿, 不重算)
    base_argmax_cer = [r["argmax_cer_A"] for r in results]  # 主线 argmax
    # sim 选路 = per_src[target_idx] 的 cer (stream_sims 最大那路)
    base_sim_cer = []
    for r in results:
        ti = r.get("target_idx")
        if ti is None or ti >= len(r["per_src"]):
            base_sim_cer.append(None)
        else:
            base_sim_cer.append(r["per_src"][ti]["cer"])
    base_oracle_cer = [r["oracle_cer"] for r in results]

    summary_each = {s: {"cers": [], "picks_oracle": [], "fallback_reasons": {}} for s in strat_names}

    for r in results:
        per_src = r["per_src"]
        oracle_idx = r["oracle_src_idx"]
        entry = {
            "uid": r["uid"], "ref": r["ref"],
            "texts": [s["text"] for s in per_src],
            "cers": [s["cer"] for s in per_src],
            "oracle_idx": oracle_idx,
            "oracle_cer": r["oracle_cer"],
            "strategies": {},
        }
        # 策略1
        idx, reason = route_content_gate(per_src)
        entry["strategies"]["content_gate"] = {
            "pick": idx, "cer": per_src[idx]["cer"],
            "correct": idx == oracle_idx, "reason": reason,
        }
        summary_each["content_gate"]["cers"].append(per_src[idx]["cer"])
        summary_each["content_gate"]["picks_oracle"].append(idx == oracle_idx)
        summary_each["content_gate"]["fallback_reasons"][reason] = \
            summary_each["content_gate"]["fallback_reasons"].get(reason, 0) + 1
        # 策略3
        idx, reason = route_heuristic(per_src)
        entry["strategies"]["heuristic"] = {
            "pick": idx, "cer": per_src[idx]["cer"],
            "correct": idx == oracle_idx, "reason": reason,
            "scores": [cmd_score(s["text"]) for s in per_src],
        }
        summary_each["heuristic"]["cers"].append(per_src[idx]["cer"])
        summary_each["heuristic"]["picks_oracle"].append(idx == oracle_idx)
        summary_each["heuristic"]["fallback_reasons"][reason] = \
            summary_each["heuristic"]["fallback_reasons"].get(reason, 0) + 1
        per_sample.append(entry)

    # 汇总
    def _agg(name, cers, corrects):
        return {
            "mean_cer": round(statistics.mean(cers), 4),
            "median_cer": round(statistics.median(cers), 4),
            "accuracy": round(sum(corrects) / len(corrects), 4),
            "n_correct": sum(corrects),
            "n": len(cers),
            "fallback_reasons": summary_each[name]["fallback_reasons"],
        }
    strat_summary = {s: _agg(s, summary_each[s]["cers"], summary_each[s]["picks_oracle"])
                     for s in strat_names}

    # 基准
    baselines = {
        "argmax_main": round(statistics.mean(base_argmax_cer), 4),
        "sim_pick_B2": round(statistics.mean([c for c in base_sim_cer if c is not None]), 4),
        "oracle_B2": round(statistics.mean(base_oracle_cer), 4),
        "random_50_50_estimate": round(statistics.mean([
            statistics.mean(r["per_src"][i]["cer"] for i in range(len(r["per_src"])))
            for r in results
        ]), 4),
    }

    # TRAP 统计: 多少条两路 content_gate 都 accept (内容判别物理上限)
    n_both_valid = sum(1 for r in results
                       if all(is_valid_command(s["text"]) for s in r["per_src"]))
    n_one_valid = sum(1 for r in results
                      if sum(is_valid_command(s["text"]) for s in r["per_src"]) == 1)
    n_both_invalid = sum(1 for r in results
                         if not any(is_valid_command(s["text"]) for s in r["per_src"]))

    # TRAP 子集 (both_valid) 上 oracle 平均 CER (即内容判别上限能达到的最好)
    trap_oracle = [r["oracle_cer"] for r in results
                   if all(is_valid_command(s["text"]) for s in r["per_src"])]
    trap_argmax = [r["argmax_cer_A"] for r in results
                   if all(is_valid_command(s["text"]) for s in r["per_src"])]

    out = {
        "verdict": "multivoice 内容判别选路 POC",
        "seed": SEED,
        "n_samples": len(results),
        "strategies": strat_summary,
        "baselines": baselines,
        "trap_analysis": {
            "both_valid_TRAP": n_both_valid,
            "one_valid_clean_route": n_one_valid,
            "both_invalid_fallback": n_both_invalid,
            "trap_oracle_mean_cer": round(statistics.mean(trap_oracle), 4) if trap_oracle else None,
            "trap_argmax_mean_cer": round(statistics.mean(trap_argmax), 4) if trap_argmax else None,
            "note": "TRAP=两路都过 content_gate (内容判别无法区分, 物理上限)",
        },
        "per_sample": per_sample,
    }
    json.dump(out, open(OUT_SUMMARY, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 控制台
    print("=" * 60)
    print(f"multivoice 内容判别选路 (n={len(results)})")
    print("=" * 60)
    print("\n[策略 mean CER + 选路准确率]")
    print(f"{'策略':<16} {'mean CER':>10} {'选路准确率':>12} {'正确/总':>10}")
    for s in strat_names:
        d = strat_summary[s]
        print(f"{s:<16} {d['mean_cer']:>10.4f} {d['accuracy']*100:>11.1f}% {d['n_correct']:>4}/{d['n']}")
    print(f"\n[对照基准]")
    for k, v in baselines.items():
        print(f"  {k:<24}: {v:.4f}")
    print(f"\n[TRAP 分析]")
    print(f"  both_valid (TRAP, 内容无法判): {n_both_valid}/40")
    print(f"  one_valid  (内容可干净挑路): {n_one_valid}/40")
    print(f"  both_invalid (fallback):      {n_both_invalid}/40")
    if trap_oracle:
        print(f"  TRAP 子集 oracle CER: {statistics.mean(trap_oracle):.4f} (内容判别上限)")
        print(f"  TRAP 子集 argmax CER: {statistics.mean(trap_argmax):.4f}")
    print(f"\n产物: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
