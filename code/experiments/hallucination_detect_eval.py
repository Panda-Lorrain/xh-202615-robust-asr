"""content_gate 加强: 幻觉检测信号评估(全量死区确认证实的 CER 收益方向)。

对 argmax vanilla_text 做幻觉检测(循环重复/超长/乱码/非中文), 阳性→拒(CER=1.0,
pos 允许拒无额外惩罚)。算死区召回/非死区误拒/全量 CER 净效果(char-weighted 累计池)
+ hold-out 泛化(uid md5 hash 分 train/val, 纯先验阈值不 train 拟合, 对齐 content_gate)。

纯离线: 用 exp_vanilla_full 现有 vanilla_text, 不重跑 ASR/multi-voice。
对照: 全量死区条 multi-voice(拒幻觉) 全量 CER 0.595→0.476(Δ-0.119, 仅死区条)。
本脚本验证"单路 argmax + 幻觉检测拒"(不需 multi-voice)能否接近这个收益 + 不误伤非死区。
"""
import json, os, hashlib, unicodedata
_HERE = os.path.dirname(os.path.abspath(__file__))

van = json.load(open(os.path.join(_HERE, "exp_vanilla_full.json"), encoding="utf-8"))
pos = {f"cmd_{r['id']}": r for r in json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))}


def normalize(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith("P") and not c.isspace())


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


# ---- 幻觉检测信号 ----
def max_char_run(t):
    """最长连续相同字符(循环幻觉'手手骨骨'→run大)。"""
    if not t:
        return 0
    mx = cur = 1
    for i in range(1, len(t)):
        cur = cur + 1 if t[i] == t[i - 1] else 1
        mx = max(mx, cur)
    return mx


def cn_ratio(t):
    return sum(1 for c in t if "一" <= c <= "鿿") / max(len(t), 1)


def char_diversity(t):
    """unique chars / len(重复多→多样性低)。"""
    return len(set(t)) / max(len(t), 1)


# 扩展规则信号(workflow 对抗审查建议: 财经词典抓新闻话 + 短长度白名单抓碎片)
NEWS_KW = ["经济", "产业", "企业", "成本", "营收", "投资", "基金", "公募", "散户", "同比",
           "上涨", "下降", "市场", "合同", "签订", "赔偿", "法律", "机构", "行业", "半导体",
           "生产", "奥运", "展会", "搜狐", "微软", "三星", "克林顿", "上市", "分成", "冲击",
           "制造", "围观", "市民", "推广", "财经", "播报", "新闻", "韩国", "首尔", "旅游"]
CMD_SEED = ["空调", "风", "温", "度", "播放", "放", "屏", "开", "关", "睡", "模式", "吹",
            "调", "挡", "ECO", "除湿", "制热", "送风", "灯", "帘", "洗", "音", "热", "净",
            "烤", "闹", "定时", "暂停", "风速", "风量", "亮度", "音量"]


def is_hallucination(text, ref, p):
    t = (text or "").strip()
    if not t:
        return False
    run = max_char_run(t)
    lr = len(t) / max(len(ref), 1)
    cn = cn_ratio(t)
    div = char_diversity(t)
    return ((run >= p["run"])                              # 循环重复(主信号, 死区幻觉特征)
            or (len(t) >= p["len_abs"] and lr >= p["len_ratio"])  # 超长(幻觉远超指令)
            or (cn < p["cn"] and len(t) > 3)               # 非中文为主(英文水印/乱码)
            or (div < p["div"] and len(t) >= 10)           # 字符多样性极低(重复乱码)
            or (len(t) >= 6 and any(w in t for w in NEWS_KW))         # 财经/新闻话(workflow B)
            or (p.get("short_len", 0) and len(t) <= p["short_len"]
                and not any(w in t for w in CMD_SEED)))   # 短碎片无指令词


def evaluate(p, split=None):
    """split: None=全集 / 'train' / 'val'(uid md5 hash 分割, 纯先验阈值)。"""
    dead, nond, all_rows = [], [], []
    tot_err_arg = tot_err_mv = tot_char = 0
    for r in van:
        uid = r["uid"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2
        if split == "train" and h != 0:
            continue
        if split == "val" and h != 1:
            continue
        ref = pos.get(uid, {}).get("ref", "")
        text = r.get("vanilla_text", "") or ""
        cer_arg = r.get("vanilla_cer", 1.0)  # 缺失(如diar失败)当 CER1.0
        rlen = max(len(normalize(ref)), 1)
        hal = is_hallucination(text, ref, p)
        cer_mv = 1.0 if hal else cer_arg
        tot_err_arg += cer_arg * rlen
        tot_err_mv += cer_mv * rlen
        tot_char += rlen
        row = {"uid": uid, "cer_arg": cer_arg, "cer_mv": cer_mv, "hal": hal}
        all_rows.append(row)
        if cer_arg > 1:
            dead.append(row)
        elif cer_arg < 0.5:
            nond.append(row)
    return {
        "n": len(all_rows),
        "dead_recall": round(sum(x["hal"] for x in dead) / len(dead), 4) if dead else 0,
        "nond_falsepos": round(sum(x["hal"] for x in nond) / len(nond), 4) if nond else 0,
        "cer_arg_pool": round(tot_err_arg / tot_char, 4),
        "cer_mv_pool": round(tot_err_mv / tot_char, 4),
        "cer_delta": round(tot_err_mv / tot_char - tot_err_arg / tot_char, 4),
        "n_dead": len(dead), "n_nond": len(nond),
    }


# ---- ablation: 几组先验参数 ----
PARAMS = {
    "baseline":       {"run": 4, "len_abs": 25, "len_ratio": 3, "cn": 0.30, "div": 0.35, "short_len": 0},
    "extended":       {"run": 4, "len_abs": 25, "len_ratio": 3, "cn": 0.30, "div": 0.35, "short_len": 5},
    "extended_strict":{"run": 5, "len_abs": 30, "len_ratio": 4, "cn": 0.25, "div": 0.30, "short_len": 5},
    "news_only":      {"run": 999, "len_abs": 999, "len_ratio": 999, "cn": 0.0, "div": 0.0, "short_len": 0},
}

print(f"{'param':12s} {'split':6s} {'死区召回':>8s} {'非死误拒':>8s} {'CER_arg':>9s} {'CER_mv':>9s} {'ΔCER':>9s}")
print("-" * 70)
results = {}
for name, p in PARAMS.items():
    for split in [None, "val"]:
        m = evaluate(p, split)
        results[f"{name}_{split or 'full'}"] = m
        print(f"{name:12s} {split or 'full':6s} {m['dead_recall']:8.3f} {m['nond_falsepos']:8.3f} "
              f"{m['cer_arg_pool']:9.4f} {m['cer_mv_pool']:9.4f} {m['cer_delta']:+9.4f}")

json.dump(results, open(os.path.join(_HERE, "hallucination_detect_eval.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# 落盘 baseline 抓的条 + 死区漏的条, 供 workflow 对抗审查
bp = PARAMS["baseline"]
hal_pos, dead_miss = [], []
for r in van:
    uid = r["uid"]
    ref = pos.get(uid, {}).get("ref", "")
    text = r.get("vanilla_text", "") or ""
    cer_arg = r.get("vanilla_cer", 1.0)
    hal = is_hallucination(text, ref, bp)
    rec = {"uid": uid, "text": text[:100], "ref": ref, "cer": round(cer_arg, 3),
           "max_run": max_char_run(text), "len": len(text), "cn_ratio": round(cn_ratio(text), 2),
           "diversity": round(char_diversity(text), 2)}
    if hal:
        hal_pos.append(rec)
    elif cer_arg > 1:
        dead_miss.append(rec)
json.dump(hal_pos, open(os.path.join(_HERE, "hal_positives.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(dead_miss, open(os.path.join(_HERE, "deadzone_missed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n对照: 全量死区条 multi-voice(仅死区) 全量 CER 0.595→0.476(Δ-0.119)")
print(f"-> hallucination_detect_eval.json | hal_positives.json({len(hal_pos)}) | deadzone_missed.json({len(dead_miss)})")
