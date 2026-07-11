#!/usr/bin/env python3
"""B1 横评: FireRedASR-AED-L vs Qwen3-ASR vs vanilla 官方口径 CER 对比。

数据源: poc_qwen_asr_full_result.json(ref/sim/qwen/vanilla) + _firered_uid2text_full.json(FireRedASR)。
口径: 提交归一(to_simplified+digit_postproc) + eval_metrics.CERMetric 官方累计池。
对比: transcribe 不拒 / 含拒 thr 工作点 / sim 分桶 / 逐条谁更优。选型 + 效率腿(RTF)。

用法(主 venv): code/.venv/Scripts/python.exe code/exp_firered_eval.py
前提: code/_firered_uid2text_full.json 已由 firered_asr_backend.py 全量生成。
"""
import json, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric, cer_official
from text_utils import to_simplified, digit_postproc


def sn(t):
    return digit_postproc(to_simplified(t or ""))


def pool(hyps, refs):
    m = CERMetric(); m.update(hyps, refs); return m.compute()["cer"]


def pool_with_reject(hyps, refs, sims, thr):
    m = CERMetric()
    for h, r, s in zip(hyps, refs, sims):
        m.update(["" if s < thr else h], [r])
    res = m.compute()
    per = res["per_sample"]
    n = len(per)
    correct = sum(1 for x in per if x["cer"] < 0.5) / n if n else 0.0
    return res["cer"], correct


def main():
    poc = json.load(open(os.path.join(os.path.dirname(__file__), "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    fr_path = os.path.join(os.path.dirname(__file__), "_firered_uid2text_full.json")
    fr = json.load(open(fr_path, encoding="utf-8"))

    rows = []
    miss = 0
    for r in poc["rows"]:
        if r["uid"] in fr:
            rows.append({**r, "firered": fr[r["uid"]]})
        else:
            miss += 1
    n = len(rows)
    print(f"===== FireRedASR vs Qwen3-ASR vs vanilla 横评（对齐 {n} 条, miss {miss}）=====")

    refs = [r["ref"] for r in rows]
    sims = [r["sim"] for r in rows]
    qh = [sn(r["qwen"]) for r in rows]
    vh = [sn(r["vanilla"]) for r in rows]
    fh = [sn(r["firered"]) for r in rows]

    # per-row firered cer
    for r in rows:
        r["firered_cer"] = cer_official(sn(r["firered"]), r["ref"])

    # 1. transcribe 不拒(诊断口径)
    print("\n-- transcribe 官方池 CER(pos 全转写, 诊断口径) --")
    out = {"n": n}
    out["transcribe"] = {}
    for label, hyps in [("qwen", qh), ("vanilla", vh), ("firered", fh)]:
        m = CERMetric(); m.update(hyps, refs); res = m.compute()
        per = res["per_sample"]
        correct = sum(1 for x in per if x["cer"] < 0.5) / n
        near = sum(1 for x in per if x["cer"] < 0.1) / n
        out["transcribe"][label] = {"overall": round(res["cer"], 4), "correct": round(correct, 4), "near": round(near, 4)}
        print(f"  {label:8}: overall={res['cer']:.4f} correct={correct:.1%} near={near:.1%}")

    # 2. 含拒 thr 工作点(提交口径)
    print("\n-- 含拒 thr 工作点(提交 overall CER, 官方池) --")
    print(f"  {'thr':<6}{'qwen':<9}{'vanilla':<9}{'firered':<9}{'f-q Δ':<9}")
    out["thr_workpoints"] = {}
    for thr in [0.2, 0.27, 0.3, 0.35, 0.4]:
        qo, _ = pool_with_reject(qh, refs, sims, thr)
        vo, _ = pool_with_reject(vh, refs, sims, thr)
        fo, _ = pool_with_reject(fh, refs, sims, thr)
        out["thr_workpoints"][str(thr)] = {"qwen": round(qo, 4), "vanilla": round(vo, 4), "firered": round(fo, 4)}
        print(f"  {thr:<6.2f}{qo:<9.4f}{vo:<9.4f}{fo:<9.4f}{fo-qo:<+9.4f}")

    # 3. sim 分桶
    print("\n-- sim 分桶(官方池) --")
    out["sim_buckets"] = {}
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 1.0)]:
        idx = [i for i, s in enumerate(sims) if lo <= s < hi]
        if not idx:
            continue
        out["sim_buckets"][f"[{lo},{hi})"] = {"n": len(idx)}
        print(f"  sim[{lo},{hi}) n={len(idx)}:")
        for label, hyps in [("qwen", qh), ("vanilla", vh), ("firered", fh)]:
            v = pool([hyps[i] for i in idx], [refs[i] for i in idx])
            out["sim_buckets"][f"[{lo},{hi})"][label] = round(v, 4)
            print(f"    {label:8}: {v:.4f}")

    # 4. 逐条谁更优
    fr_better = sum(1 for r in rows if r["firered_cer"] < r["qwen_cer"] - 0.01)
    qw_better = sum(1 for r in rows if r["qwen_cer"] < r["firered_cer"] - 0.01)
    tie = n - fr_better - qw_better
    print(f"\n-- 逐条 firered vs qwen --")
    print(f"  firered 更优 {fr_better} / qwen 更优 {qw_better} / 持平 {tie} ({n})")
    out["pairwise"] = {"firered_better": fr_better, "qwen_better": qw_better, "tie": tie}

    out_path = os.path.join(os.path.dirname(__file__), "exp_firered_eval.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[done] → {out_path}")


if __name__ == "__main__":
    main()
