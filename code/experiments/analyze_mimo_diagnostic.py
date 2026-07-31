#!/usr/bin/env python3
"""MiMo 诊断标尺:用 SOTA MiMo-V2.5-ASR 当尺子, 定位 pipeline 瓶颈在"切片"还是"转写"。

【原理】exp_mimo_asr.py 已用【同一条 target 切片】喂 MiMo 和 vanilla(源码 line 196-207
注释明说"与 vanilla 实验完全一致, 唯一变量=转写后端")。故 mimo_cer vs vanilla_cer 的差异
= 纯转写器能力差异, 切片输入恒定。
  - 桶内 MiMo 能转好(mimo_cer 低)但 vanilla 差 →【转写器瓶颈】: 切片 OK, vanilla 转写不行,
    换更强转写器能救(但 MiMo 不能直接进提交, 作答辩"上限分析"弹药)。
  - 桶内 MiMo 也翻车(mimo_cer 高) →【切片瓶颈】: diar+声纹切错了 target, 再强转写器也救不回,
    → 攻声纹/diar(声纹强化复评方向)。
据此集中火力, 避免"全转写器派"或"全切片派"的盲打。

【数据】code/exp_mimo_asr_result.json (1364 条 pos; 字段 uid/ref/max_sim/mimo_text/
vanilla_text/mimo_cer/vanilla_cer/target_dur)。mimo_cer 与 vanilla_cer 同 cer_of 口径
(繁简+去标点归一), 公平对比。max_sim 来自 diar+声纹选 target 的最高余弦相似度。

【不进提交】纯分析, 不调 API/模型。诊断结论作答辩"诚实归因 + 业界对标"硬弹药。

用法: code/.venv/Scripts/python.exe code/analyze_mimo_diagnostic.py
"""
import json, re, os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(_HERE, "exp_mimo_asr_result.json")
OUT = os.path.join(_HERE, "_mimo_diagnostic.json")  # _ 前缀 gitignored

BUCKETS = [(0.0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.01)]
LABELS = ["[0.0,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,1.0)"]
# ref 含数字(中文/阿拉伯)的句, vanilla 易因输出阿拉伯数字失分 → 纯文字句剥离该口径红利看真实转写能力
DIGIT = re.compile(r"[0-9一二三四五六七八九十百千万两零]")


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def verdict(m_cer, v_cer, n):
    """桶瓶颈判定。m_cer=MiMo 均值, v_cer=vanilla 均值, n=条数。"""
    if n < 10:
        return "样本少"
    if m_cer > 0.5:
        return "切片瓶颈"  # MiMo 都翻车 → 切片/音频质量是瓶颈, 转写器再强也没用
    if m_cer < 0.3 and (v_cer - m_cer) > 0.15:
        return "转写器瓶颈"  # MiMo 转好 vanilla 差 → 切片 OK, vanilla 转写能力不够
    if (v_cer - m_cer) > 0.15:
        return "偏转写器"
    return "混合/两难"


def analyze(rows, label):
    print(f"\n===== {label}（N={len(rows)}）按 max_sim 分桶 =====")
    print(f"{'sim桶':<11}{'n':>5}{'mimo':>8}{'vanilla':>9}{'Δ(v-m)':>9}{'mimo_c%':>9}{'van_c%':>8}{'更优/差/平':>13}{'判定':<14}")
    out = []
    for (lo, hi), lb in zip(BUCKETS, LABELS):
        b = [r for r in rows if lo <= r["max_sim"] < hi]
        if not b:
            print(f"{lb:<11}{0:>5}{'-':>8}{'-':>9}{'-':>9}{'-':>9}{'-':>8}{'-':>13}{'(空)':<14}")
            continue
        n = len(b)
        m = mean([r["mimo_cer"] for r in b])
        v = mean([r["vanilla_cer"] for r in b])
        mc = sum(1 for r in b if r["mimo_cer"] < 0.5) / n
        vc = sum(1 for r in b if r["vanilla_cer"] < 0.5) / n
        better = sum(1 for r in b if r["mimo_cer"] < r["vanilla_cer"] - 0.01)
        worse = sum(1 for r in b if r["mimo_cer"] > r["vanilla_cer"] + 0.01)
        tie = n - better - worse
        vd = verdict(m, v, n)
        print(f"{lb:<11}{n:>5}{m:>8.3f}{v:>9.3f}{v - m:>+9.3f}{mc:>8.0%}{vc:>8.0%}{f'{better}/{worse}/{tie}':>13}{vd:<14}")
        out.append({"bucket": lb, "n": n, "mimo_cer": round(m, 4), "vanilla_cer": round(v, 4),
                    "delta_v_minus_m": round(v - m, 4), "mimo_correct": round(mc, 3),
                    "van_correct": round(vc, 3), "mimo_better": better, "vanilla_better": worse,
                    "tie": tie, "verdict": vd})
    n = len(rows)
    m = mean([r["mimo_cer"] for r in rows]) if rows else 0
    v = mean([r["vanilla_cer"] for r in rows]) if rows else 0
    print(f"{'合计':<11}{n:>5}{m:>8.3f}{v:>9.3f}{v - m:>+9.3f}")
    return {"label": label, "n": n, "buckets": out,
            "total": {"n": n, "mimo_cer": round(m, 4), "vanilla_cer": round(v, 4),
                      "delta": round(v - m, 4)}}


def main():
    data = json.load(open(RESULT, encoding="utf-8"))
    valid = [r for r in data if "mimo_cer" in r and r.get("vanilla_cer") is not None]
    print(f"[load] {RESULT}: {len(data)} 条, 有效(有 mimo+vanilla cer) {len(valid)} 条")

    full = analyze(valid, "全量 pos")
    pure = analyze([r for r in valid if not DIGIT.search(r["ref"])], "纯文字句(剥离数字口径红利)")
    digit = analyze([r for r in valid if DIGIT.search(r["ref"])], "含数字句")

    # 瓶颈分布(以纯文字句为准, 剥离数字干扰最干净)
    by_verdict = defaultdict(int)
    for b in pure["buckets"]:
        by_verdict[b["verdict"]] += b["n"]
    tot = sum(by_verdict.values()) or 1
    print(f"\n===== 瓶颈分布汇总（纯文字句, N={tot}）=====")
    for vd, n in sorted(by_verdict.items(), key=lambda x: -x[1]):
        print(f"  {vd:<14}: {n:>5} 条 ({n / tot:.1%})")

    print("\n[结论模板]")
    print("  转写器瓶颈桶(sim>=0.3 且 MiMo 转好): MiMo 证明 CER 天花板可压, 但 MiMo 不能直接进")
    print("    提交(云端红线) → 答辩'上限分析'弹药, 未来攻更强转写器或蒸馏(注: 交接已证蒸馏不可行)")
    print("  切片瓶颈桶(sim<0.2 且 MiMo 也翻车): diar+声纹切错 target → 攻声纹强化(CAM++/US-PVAD)")

    json.dump({"full": full, "pure_text": pure, "digit": digit,
               "verdict_dist_pure": dict(by_verdict)},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[done] → {OUT}")


if __name__ == "__main__":
    main()
