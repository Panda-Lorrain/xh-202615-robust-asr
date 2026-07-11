#!/usr/bin/env python3
"""权威重算: Qwen3-ASR 官方口径数字 —— 坐实 0.3436(transcribe) + 含拒 thr 工作点 + 完整提交数字。

⚠️ 缘起(2026-07-11 核实): CLAUDE.md/handoff/RESULTS 头条「CER 腿 +10.1(16.2→26.3)」是基于
【不含拒 transcribe CER】0.3436 算的; 但提交进排名公式用的是【含拒 overall】(pos 允许拒,
2026-07-08 主办方确认)。两口径差距大, 必须分清防答辩穿帮。且 0.3436 此前未入库(poc json 只有
per-sample 均值 0.3848, 非官方累计池), 本脚本补齐落盘。

数据源: poc_qwen_asr_full_result.json(1350 条 qwen/vanilla 逐条, sim=max_sim 同源 wespeaker)。
口径: 对 qwen/vanilla text apply 提交链路归一(to_simplified → digit_postproc, 同 enroll_infer:384),
      再用 eval_metrics.CERMetric(主办方累计池 total_err/total_char)算。
对比: 同集合 1350 条 qwen vs vanilla(公平, 不混 1350/1362 集合差)。

用法: code/.venv/Scripts/python.exe code/recompute_qwen_official.py
输出: code/qwen_official_cer_workpoints.json + 打印。
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc


def submit_norm(text):
    """复刻 enroll_infer 提交链路归一(顺序: to_simplified → digit_postproc)。"""
    return digit_postproc(to_simplified(text or ""))


def is_eng(t):
    L = [c for c in (t or "") if c.isalpha()]
    return sum(c.isascii() for c in L) / len(L) > 0.5 if len(L) >= 4 else False


def pool_with_reject(hyps, refs, sims, thr):
    """thr 工作点(含拒): sim<thr 拒 → pred 置空 → 官方累计池。
    拒识条 pred='' → editdistance('', norm_ref)=len(ref) → CER=1.0 天然。"""
    m = CERMetric()
    for h, r, s in zip(hyps, refs, sims):
        m.update(["" if s < thr else h], [r])
    res = m.compute()
    per = res["per_sample"]
    n = len(per)
    correct = sum(1 for x in per if x["cer"] < 0.5) / n if n else 0.0
    return res["cer"], correct


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "poc_qwen_asr_full_result.json")
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    n = len(rows)
    refs = [r["ref"] for r in rows]
    sims = [r["sim"] for r in rows]
    qw_sub = [submit_norm(r["qwen"]) for r in rows]
    va_sub = [submit_norm(r["vanilla"]) for r in rows]  # 同集合 vanilla 对照(归一后)

    print(f"===== Qwen3-ASR 官方口径重算（{n} 条 pos target 切片, 提交侧归一后）=====")
    out = {"n": n, "source": os.path.basename(path)}

    # ---- 1. 转写 CER(不拒, pos 全转写诊断口径) ----
    print("\n-- 转写 CER（不拒, 官方累计池 = pos 全转写诊断口径）--")
    out["transcribe"] = {}
    for label, hyps in [("qwen", qw_sub), ("vanilla", va_sub)]:
        m = CERMetric(); m.update(hyps, refs); res = m.compute()
        per = res["per_sample"]
        correct = sum(1 for x in per if x["cer"] < 0.5) / n
        near = sum(1 for x in per if x["cer"] < 0.1) / n
        eng = sum(1 for x in hyps if is_eng(x)) / n
        out["transcribe"][label] = {"overall": round(res["cer"], 4), "correct": round(correct, 4),
                                    "near": round(near, 4), "eng_halluc": round(eng, 4)}
        print(f"  {label:8}: overall={res['cer']:.4f} correct={correct:.1%} near={near:.1%} 英文幻觉={eng:.1%}")

    # ---- 2. 含拒 thr 工作点(提交 overall CER, 排名公式实际用) ----
    print("\n-- 含拒 thr 工作点（提交 overall CER, 官方累计池）【提交口径】--")
    print(f"  {'thr':<6}{'qwen':<9}{'q_corr':<8}{'vanilla':<9}{'v_corr':<8}{'Δ(q-v)':<9}{'qwen腿':<8}{'van腿':<8}")
    out["thr_workpoints"] = {}
    RR = 0.9051  # neg RR@0.27(从 out_neg_vanilla_full, A4 核实 n=474; 与转写器无关, qwen 不转写 neg)
    for thr in [0.2, 0.27, 0.3, 0.35, 0.4, 0.45]:
        qo, qc = pool_with_reject(qw_sub, refs, sims, thr)
        vo, vc = pool_with_reject(va_sub, refs, sims, thr)
        qleg = (1 - qo) * 40
        vleg = (1 - vo) * 40
        out["thr_workpoints"][str(thr)] = {
            "qwen": round(qo, 4), "qwen_correct": round(qc, 4),
            "vanilla": round(vo, 4), "vanilla_correct": round(vc, 4),
            "delta_q_minus_v": round(qo - vo, 4),
            "qwen_cer_leg_40": round(qleg, 2), "vanilla_cer_leg_40": round(vleg, 2)}
        print(f"  {thr:<6.2f}{qo:<9.4f}{qc:<8.1%}{vo:<9.4f}{vc:<8.1%}{qo-vo:<+9.4f}{qleg:<8.2f}{vleg:<8.2f}")

    # ---- 3. 提交数字汇总(thr0.27) ----
    qo27 = out["thr_workpoints"]["0.27"]["qwen"]
    vo27 = out["thr_workpoints"]["0.27"]["vanilla"]
    out["submission_thr027"] = {
        "qwen":   {"pos_CER_containing_reject": qo27, "neg_RR": RR,
                   "CER_leg_40": round((1-qo27)*40, 2), "RR_leg_40": round(RR*40, 2)},
        "vanilla": {"pos_CER_containing_reject": vo27, "neg_RR": RR,
                    "CER_leg_40": round((1-vo27)*40, 2), "RR_leg_40": round(RR*40, 2)},
        "delta_CER_leg_qwen_minus_vanilla": round((1-qo27)*40 - (1-vo27)*40, 2),
        "neg_RR_source": "out_neg_vanilla_full/result.json (n=474, @0.27, 与转写器无关)",
        "efficiency_leg_20": "待 L20 RTF 真测(Qwen3 RTF0.289@4060, L20 待测)"}
    print("\n-- 提交数字汇总（thr=0.27, w1=w2=0.4 估算, 效率腿 20 待 L20）--")
    print(f"  qwen:    pos含拒CER={qo27:.4f} → CER腿={(1-qo27)*40:.2f} | neg RR={RR} → RR腿={RR*40:.2f}")
    print(f"  vanilla: pos含拒CER={vo27:.4f} → CER腿={(1-vo27)*40:.2f} | neg RR={RR} → RR腿={RR*40:.2f}")

    # ---- 4. 双口径诚实标注(防答辩穿帮) ----
    qt = out["transcribe"]["qwen"]["overall"]
    vt = out["transcribe"]["vanilla"]["overall"]
    out["dual_caliber_honest"] = {
        "transcribe_no_reject (pos全转写, 诊断/能力上限)": {
            "qwen": qt, "vanilla": vt,
            "delta_CER_leg_40": round((1-qt)*40 - (1-vt)*40, 2)},
        "containing_reject thr0.27 (提交口径, 排名公式用)": {
            "qwen": qo27, "vanilla": vo27,
            "delta_CER_leg_40": round((1-qo27)*40 - (1-vo27)*40, 2)},
        "note": "提交用含拒口径(pos 隐含允许拒, 主办方 2026-07-08 确认); 答辩须报含拒, 勿用 transcribe 虚高"}
    print("\n-- ⚠️ 双口径诚实标注(防答辩穿帮) --")
    print(f"  transcribe 不拒(pos全转写诊断): qwen {qt:.4f} vs vanilla {vt:.4f} → CER腿Δ={(1-qt)*40-(1-vt)*40:+.2f}分")
    print(f"  含拒 thr0.27(提交, 排名公式用): qwen {qo27:.4f} vs vanilla {vo27:.4f} → CER腿Δ={(1-qo27)*40-(1-vo27)*40:+.2f}分")

    # ---- 5. sim 分桶(qwen/vanilla 官方池, 对照死区突破) ----
    print("\n-- sim 分桶（转写 CER, 官方池）--")
    out["sim_buckets"] = {"qwen": {}, "vanilla": {}}
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]:
        idx = [i for i, s in enumerate(sims) if lo <= s < hi]
        if not idx:
            continue
        for label, hyps in [("qwen", qw_sub), ("vanilla", va_sub)]:
            m = CERMetric(); m.update([hyps[i] for i in idx], [refs[i] for i in idx])
            v = m.compute()["cer"]
            out["sim_buckets"][label][f"[{lo:.1f},{hi:.1f})"] = {"n": len(idx), "cer": round(v, 4)}
            print(f"  {label} sim[{lo:.1f},{hi:.1f}): n={len(idx)} cer={v:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "qwen_official_cer_workpoints.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")


if __name__ == "__main__":
    main()
