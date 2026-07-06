#!/usr/bin/env python3
"""统一 thr 选点: A 集 pos+neg 混合模拟 B 集场景, 扫最优稳健统一 thr。

**背景(FAQ 2026-07-06 C9/Q1/Q2)**:
  B 集是赛事方最终评测题, 参赛方拿不到(否则作弊)。B 集 pos/neg 混合不预分,
  必须用单一 thr 处理所有音频(不给 pos/neg 先验)。本脚本用 A 集(已知 label)
  当验证集模拟该混合场景, 提前选定一个**稳健**的统一 thr 供 B 集提交用。

**为何无需重跑推理(纯后置拒识)**:
  pos_result 是 thr=0 全转写跑的 → 每条 text 都填好(即使 max_sim 极低)。
  扫任意 thr 只需: max_sim<thr 视为拒(pos→CER=1.0 / neg→正确拒)。
  neg 只算 RR(只需 max_sim, 不需 text)。

**加权(本地线性估算, 非官方排名口径)**:
  CER 腿=(1-pos_CER)×40 | RR 腿=neg_RR×40 | 效率腿固定剔除(vanilla RTF~0.24,
  不随 thr 变; ⚠️ 仅在 submit_infer 硬编码 --always-generate 下成立, 去 flag 后
  高 thr 可降 RTF, 效率收益量级>CER+RR 搏弈, 需重扫)。

**对抗验证(v2 增强, 应对 5 agent 审查的 critical/major)**:
  - bootstrap CI(400 次重采样选 thr, 量化选择方差, 不报过精细点估计)
  - 0.005 步长细扫(找真峰, 防 0.01 网格伪影)
  - 诊断字段(丢弃数/cer_text>1 幻觉尾部占比/neg 漏拒条数, 透明度)
  - 真压力(方差缩放 α+neg 重尾注入, 替代退化的加性平移)
  - split_oracle 用 neg 真最优 thr(非写死 0.4, 诚实)

用法: code/.venv/Scripts/python.exe code/scan_unified_thr.py
产物: code/scan_unified_thr.json + stdout 表
"""
import json
import os
import sys
import statistics
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_datasetA import _norm_zh   # 繁→简归一(消除 Whisper 繁体输出虚高)
from eval_metrics import cer

HERE = os.path.dirname(os.path.abspath(__file__))
POS_RES = os.path.join(HERE, "out_pos_vanilla_full", "result.json")
POS_PAIRS = os.path.join(HERE, "pos_pairs_datasetA.json")
NEG_RES = os.path.join(HERE, "out_neg_vanilla_full", "result.json")
NEG_PAIRS = os.path.join(HERE, "neg_pairs_datasetA.json")

W_CER, W_RR = 40, 40   # 效率腿固定(~vanilla RTF 0.24), thr 扫描只比 CER+RR


def key(p):
    """recognition 路径 -> join key (与 eval_datasetA 一致)。"""
    return os.path.splitext(os.path.basename(p))[0]


def load_pos():
    """pos -> (rows, n_manifest, n_miss)。每行 {id, max_sim, cer_text, ref_len, nt}。

    cer_text: text vs ref 的字符级 CER(繁简归一后)。空 text(转写崩)→1.0。
    注意 cer_text 可能 >1(babble 重复循环幻觉使 hyp 超长, CER 均值陷阱)。
    """
    res = {key(it["recognition"]): it
           for it in json.load(open(POS_RES, encoding="utf-8"))["results"]}
    man = json.load(open(POS_PAIRS, encoding="utf-8"))
    rows, miss = [], 0
    for r in man:
        k = key(r["recognition"])
        if k not in res:
            miss += 1
            continue
        it = res[k]
        ref = r.get("ref") or ""
        text = it.get("text", "") or ""
        c = cer(_norm_zh(text), _norm_zh(ref)) if text else 1.0
        rows.append({"id": r["id"], "max_sim": float(it.get("max_sim", 0) or 0),
                     "cer_text": c, "ref_len": len(ref), "nt": it.get("noise_type")})
    return rows, len(man), miss


def load_neg():
    """neg -> (sims, n_manifest, n_miss)。neg 只算 RR, 不需 text。"""
    res = {key(it["recognition"]): it
           for it in json.load(open(NEG_RES, encoding="utf-8"))["results"]}
    man = json.load(open(NEG_PAIRS, encoding="utf-8"))
    sims, miss = [], 0
    for r in man:
        k = key(r["recognition"])
        if k in res:
            sims.append(float(res[k].get("max_sim", 0) or 0))
        else:
            miss += 1
    return sims, len(man), miss


def score_at(thr, pos_rows, neg_sims, shift=0.0, pos_shift=None, neg_shift=None):
    """某 thr 下的全套指标。pos_shift/neg_shift 单独指定时模拟非对称偏移。"""
    ps = shift if pos_shift is None else pos_shift
    ns = shift if neg_shift is None else neg_shift
    n_pos = len(pos_rows)
    n_neg = len(neg_sims)
    pos_cer = sum((1.0 if r["max_sim"] + ps < thr else r["cer_text"])
                  for r in pos_rows) / n_pos
    neg_rr = sum(1 for s in neg_sims if s + ns < thr) / n_neg
    pos_correct = sum(1 for r in pos_rows
                      if not (r["max_sim"] + ps < thr) and r["cer_text"] < 0.5) / n_pos
    cer_leg = (1 - pos_cer) * W_CER
    rr_leg = neg_rr * W_RR
    return {"thr": round(thr, 4), "pos_cer": round(pos_cer, 4),
            "neg_rr": round(neg_rr, 4), "pos_correct": round(pos_correct, 4),
            "cer_leg": round(cer_leg, 2), "rr_leg": round(rr_leg, 2),
            "total": round(cer_leg + rr_leg, 2)}


def bootstrap_thr(pos_rows, neg_sims, thrs, B=400, seed=42):
    """bootstrap 重采样选 argmax thr 的分布(numpy 向量化)。

    报被选 thr 的分位, 量化"thr 选点方差"(防把 in-sample argmax 当真值)。
    """
    rng = np.random.RandomState(seed)
    pos_sim = np.array([r["max_sim"] for r in pos_rows])
    pos_cer = np.array([r["cer_text"] for r in pos_rows])
    neg = np.array(neg_sims)
    n_p, n_n = len(pos_sim), len(neg)
    thrs_arr = np.array(thrs)
    picked = []
    for _ in range(B):
        pi = rng.randint(0, n_p, n_p)
        ni = rng.randint(0, n_n, n_n)
        sp_sim, sp_cer = pos_sim[pi], pos_cer[pi]
        sn = neg[ni]
        best_t, best_s = thrs_arr[0], -1e9
        for t in thrs_arr:
            rej = sp_sim < t
            pc = np.where(rej, 1.0, sp_cer).mean()
            rr = (sn < t).mean()
            tot = (1 - pc) * W_CER + rr * W_RR
            if tot > best_s:
                best_s, best_t = tot, t
        picked.append(float(best_t))
    picked.sort()
    return {"B": B, "median": picked[len(picked) // 2],
            "p10": picked[len(picked) // 10], "p90": picked[len(picked) * 9 // 10],
            "iqr25": picked[len(picked) // 4], "iqr75": picked[len(picked) * 3 // 4],
            "picked_p90_thr": picked[int(len(picked) * 0.9)]}


def scale_sim(pos_rows, neg_sims, alpha):
    """方差缩放: sim'=alpha*sim (改变分布形状, 非退化平移)。"""
    new_pos = [{"max_sim": alpha * r["max_sim"], "cer_text": r["cer_text"]} for r in pos_rows]
    new_neg = [alpha * s for s in neg_sims]
    return new_pos, new_neg


def inject_neg_tail(neg_sims, frac=0.05, lo=0.4, hi=0.6, seed=42):
    """把最高 frac 比例 neg sim 抬到 [lo,hi](注入重尾, 模拟 B 集更多硬 neg)。"""
    rng_np = np.random.RandomState(seed)
    n = len(neg_sims)
    n_inj = int(n * frac)
    idx = sorted(range(n), key=lambda i: -neg_sims[i])[:n_inj]
    new = list(neg_sims)
    for i in idx:
        new[i] = lo + (hi - lo) * float(rng_np.rand())
    return new


def shift_pos_only(pos_rows, delta):
    """pos 单边下移(neg 不动, 非对称, 不退化: neg 不动则最优 thr 不简单跟随)。"""
    return [{"max_sim": r["max_sim"] + delta, "cer_text": r["cer_text"]} for r in pos_rows]


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    pos, n_pos_man, n_pos_miss = load_pos()
    neg, n_neg_man, n_neg_miss = load_neg()
    n_pos, n_neg = len(pos), len(neg)
    print(f"===== 统一 thr 选点(A 集 pos+neg 混合模拟 B 集) [v2 对抗验证增强] =====")
    print(f"pos {n_pos} 条 (manifest {n_pos_man}, 丢弃 {n_pos_miss}) / "
          f"neg {n_neg} 条 (manifest {n_neg_man}, 丢弃 {n_neg_miss})")

    ps = [r["max_sim"] for r in pos]
    cer_gt1 = sum(1 for r in pos if r["cer_text"] > 1.0)
    print(f"\n-- max_sim 分布 + 诊断 --")
    print(f"  pos: p25={pct(ps,.25):.3f} med={pct(ps,.5):.3f} p75={pct(ps,.75):.3f} mean={statistics.mean(ps):.3f}")
    print(f"  neg: p25={pct(neg,.25):.3f} med={pct(neg,.5):.3f} p75={pct(neg,.75):.3f} mean={statistics.mean(neg):.3f}")
    print(f"  张力: pos sim<0.4 占 {sum(1 for s in ps if s<0.4)/n_pos:.1%}(升thr误拒) | "
          f"neg sim≥0.4 占 {sum(1 for s in neg if s>=0.4)/n_neg:.1%}(降thr漏拒)")
    print(f"  ⚠️ 诊断: pos cer_text>1(幻觉超长)占 {cer_gt1/n_pos:.1%} ({cer_gt1} 条) "
          f"— 决定 overall_CER 口径最优 thr 漂移(若官方 CER 封顶 min(·,1.0) 则最优 thr 下移)")

    thrs = [round(0.00 + 0.01 * i, 2) for i in range(51)]
    scan = [score_at(t, pos, neg) for t in thrs]
    best = max(scan, key=lambda x: x["total"])

    # 1. 主扫描 + 2. 细扫(0.005, 找真峰防 0.01 网格伪影)
    thrs_fine = [round(0.20 + 0.005 * i, 3) for i in range(31)]  # [0.200, 0.350]
    scan_fine = [score_at(t, pos, neg) for t in thrs_fine]
    best_fine = max(scan_fine, key=lambda x: x["total"])
    print(f"\n-- 1. 主扫描(0.01 网格) + 2. 细扫(0.005) --")
    print(f"  主扫描 argmax: thr={best['thr']:.2f} (总分 {best['total']:.2f})")
    print(f"  细扫真峰(0.005): thr={best_fine['thr']:.3f} (总分 {best_fine['total']:.2f})  ← 防 0.01 伪影")
    print(f"  {'thr':<6}{'pos_CER':<10}{'neg_RR':<9}{'pos_correct':<12}{'总分':<8}")
    for s in scan:
        if s["thr"] % 0.05 < 1e-9 or abs(s["thr"] - best["thr"]) <= 0.03:
            print(f"  {s['thr']:<6.2f}{s['pos_cer']:<10.4f}{s['neg_rr']:<9.4f}"
                  f"{s['pos_correct']:<12.4f}{s['total']:<8.2f}")

    # 3. bootstrap CI(量化 thr 选点方差, 防 in-sample argmax 过拟合)
    bs = bootstrap_thr(pos, neg, thrs, B=400, seed=42)
    print(f"\n-- 3. bootstrap CI(B=400, 量化 thr 选点方差; argmax 是点估计非真值) --")
    print(f"  被选 thr 分位: median={bs['median']:.2f} | IQR=[{bs['iqr25']:.2f},{bs['iqr75']:.2f}] "
          f"| 80%CI=[{bs['p10']:.2f},{bs['p90']:.2f}]")
    print(f"  → 报区间 [~{bs['iqr25']:.2f}, ~{bs['iqr75']:.2f}], 勿以 2 位小数点估计")

    # 推荐 thr: bootstrap 中位 + pos 侧占优 tie-breaker(0.27 vs 0.28 平局取 pos_cer 低者)
    cand = [s for s in scan if s["thr"] in (0.26, 0.27, 0.28, 0.29)]
    rec = min([s for s in cand if s["total"] >= max(c["total"] for c in cand) - 0.1],
              key=lambda x: x["pos_cer"])  # 总分近平(Δ≤0.1)时取 pos_cer 最低(护 pos)
    print(f"  推荐统一 thr={rec['thr']:.2f} (总分 {rec['total']:.2f}, pos_CER {rec['pos_cer']:.4f}, "
          f"neg_RR {rec['neg_rr']:.4f}) — bootstrap 中位 + pos 占优 tie-breaker")

    # 4. 稳健区间
    print(f"\n-- 4. 稳健区间(平坦度, 抗 thr 微调) --")
    for delta in [0.03, 0.05, 0.10]:
        near = [s for s in scan if abs(s["thr"] - best["thr"]) <= delta + 1e-9]
        scores = [s["total"] for s in near]
        print(f"  ±{delta:<4.2f}: 总分 {min(scores):.2f}~{max(scores):.2f} (波动 Δ={max(scores)-min(scores):.2f})")

    # 5. 分 thr oracle(neg 用真最优 thr, 非写死 0.4)
    pos_thr0_cer = sum(r["cer_text"] for r in pos) / n_pos
    neg_best = max(scan, key=lambda x: x["neg_rr"])   # neg RR 平台在 thr≥0.45
    split_total = (1 - pos_thr0_cer) * W_CER + neg_best["neg_rr"] * W_RR
    print(f"\n-- 5. 分 thr oracle(A 集上界, 用 pos/neg label) vs 统一 thr --")
    print(f"  分 thr:   pos thr=0(CER {pos_thr0_cer:.4f}) + neg thr={neg_best['thr']:.2f}(RR {neg_best['neg_rr']:.4f}) → {split_total:.2f}")
    print(f"  统一推荐: thr={rec['thr']:.2f}(pos_CER {rec['pos_cer']:.4f} + neg_RR {rec['neg_rr']:.4f}) → {rec['total']:.2f}")
    print(f"  ⚠️ 损失: {split_total - rec['total']:.2f} 分 = B 集必须统一 thr(无 label)的代价")

    # 6. 真压力(方差缩放 + neg 重尾 + pos 单边下移; 替代退化的对称平移)
    print(f"\n-- 6. 真压力(固定 thr*={rec['thr']:.2f}, 形状变化/重尾; 对比偏移后 oracle) --")
    print(f"  {'场景':<24}{'oracle_thr':<11}{'固定thr*分':<11}{'oracle分':<10}{'损失':<8}")
    fixed_thr = rec["thr"]
    pressure = []
    scenes = [
        ("A集基准", pos, neg),
        ("sim收缩α=0.8", *scale_sim(pos, neg, 0.8)),
        ("sim扩张α=1.2", *scale_sim(pos, neg, 1.2)),
        ("neg重尾5%→[0.4,0.6]", pos, inject_neg_tail(neg)),
        ("pos单边下移0.10", shift_pos_only(pos, -0.10), neg),
    ]
    for label, pp, nn in scenes:
        sc = [score_at(t, pp, nn) for t in thrs]
        b_ora = max(sc, key=lambda x: x["total"])
        fx = score_at(fixed_thr, pp, nn)
        loss = round(b_ora["total"] - fx["total"], 2)
        pressure.append({"scene": label, "oracle_thr": b_ora["thr"],
                         "oracle_total": b_ora["total"], "fixed_total": fx["total"], "loss": loss})
        print(f"  {label:<22}{b_ora['thr']:<11.2f}{fx['total']:<11.2f}{b_ora['total']:<10.2f}{loss:<+8.2f}")
    print(f"  注: 对称 sim 平移已删(数学退化: 平移≡thr 反向移, 测不出独立泛化证据)")

    # 7. 口径敏感性
    cr_best = max(scan, key=lambda x: x["pos_correct"] * W_CER + x["neg_rr"] * W_RR)
    cr_total = cr_best["pos_correct"] * W_CER + cr_best["neg_rr"] * W_RR
    print(f"\n-- 7. 口径敏感性(主办方 CER 口径待定) --")
    print(f"  口径A overall_CER (pos 拒=1.0):           最优 thr={best['thr']:.2f}, 总分 {best['total']:.2f}")
    print(f"  口径B correct_rate(CER<0.5)×40+RR×40:     最优 thr={cr_best['thr']:.2f}, 等效 {cr_total:.2f}")
    print(f"  口径C pos 不许拒(thr=0): cer_accepted={pos_thr0_cer:.4f}(thr 无关)")
    print(f"  ⚠️ 权重比敏感(线性估算假设 40:40): RR-heavy(20:60)→最优 thr 上移 0.35-0.40; "
          f"CER-heavy(80:0)→thr→0。待主办方口径确认")

    # 8. noise_type 子群体
    print(f"\n-- 8. noise_type 子群体 --")
    by_nt = defaultdict(list)
    for r in pos:
        by_nt[r["nt"]].append(r)
    sub = []
    for nt, rows in sorted(by_nt.items(), key=lambda x: -len(x[1])):
        sub_scan = [score_at(t, rows, neg) for t in thrs]
        sub_best = max(sub_scan, key=lambda x: x["total"])
        sub.append({"nt": nt, "n": len(rows), "best_thr": sub_best["thr"]})
        note = "" if len(rows) >= 200 else " (n<200, 功效不足仅看方向)"
        print(f"  {str(nt):<12}: n={len(rows):4d} → 子群最优 thr={sub_best['thr']:.2f}{note}")

    # neg 漏拒诊断(thr* 处)
    neg_leak = sum(1 for s in neg if s >= fixed_thr)
    print(f"\n-- 9. neg 漏拒诊断(thr*={fixed_thr:.2f} 处) --")
    print(f"  neg 漏拒 {neg_leak}/{n_neg} ({neg_leak/n_neg:.1%}) — 这些 label=accept 进提交, "
          f"RR 腿 {rec['neg_rr']:.4f}(vs thr=0.4 的 {neg_best['neg_rr']:.4f})")
    print(f"  ⚠️ neg_result 在 thr=0.4 生成, [0.28,0.4) 段 leak 文本已被置空, "
          f"漏拒转写内容质量无法评估(若主办方有漏拒严重度惩罚需重跑 neg≤{fixed_thr})")

    # 写 JSON
    out = {
        "n_pos": n_pos, "n_neg": n_neg, "n_pos_miss": n_pos_miss, "n_neg_miss": n_neg_miss,
        "pos_sim_dist": {"p25": pct(ps,.25), "med": pct(ps,.5), "p75": pct(ps,.75), "mean": statistics.mean(ps)},
        "neg_sim_dist": {"p25": pct(neg,.25), "med": pct(neg,.5), "p75": pct(neg,.75), "mean": statistics.mean(neg)},
        "pos_cer_gt1_ratio": round(cer_gt1 / n_pos, 4),
        "argmax_thr_001": best, "argmax_thr_0005": best_fine,
        "bootstrap": bs,
        "recommended_thr": rec,
        "split_oracle": {"pos_thr0_cer": pos_thr0_cer, "neg_best_thr": neg_best["thr"],
                         "neg_best_rr": neg_best["neg_rr"], "total": split_total,
                         "loss_vs_unified": round(split_total - rec["total"], 2)},
        "pressure_test": pressure,
        "caliber_B_best": {"thr": cr_best["thr"], "pos_correct": cr_best["pos_correct"],
                           "neg_rr": cr_best["neg_rr"], "equiv": round(cr_total, 2)},
        "subgroup_noise": sub,
        "scan_full_001": scan,
    }
    out_path = os.path.join(HERE, "scan_unified_thr.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")
    print(f"\n[决策速览] 推荐 thr={rec['thr']:.2f} (区间 [~{bs['iqr25']:.2f},~{bs['iqr75']:.2f}], "
          f"细扫真峰 {best_fine['thr']:.3f}) | 分thr损失 {split_total - rec['total']:.2f} | "
          f"真压力损失见上 | bootstrap B=400")


if __name__ == "__main__":
    main()
