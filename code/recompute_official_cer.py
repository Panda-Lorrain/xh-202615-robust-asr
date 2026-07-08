#!/usr/bin/env python3
"""权威重算: 官方 CER 口径下 vanilla/dicow 全量数字(提交侧归一后)。

数据源 exp_vanilla_full.json(1364 条原始转写, 2026-07-06; 与当前 enroll_infer 转写逻辑
一致 —— 07-06→07-07 间 enroll_infer 仅加 to_simplified/digit_postproc 后处理, 未动 diar/
cut_timeline/vanilla decode, 且 cmd_0/cmd_1 逐字一致, git 证)。

口径: 对原始 text apply 提交链路归一(to_simplified → digit_postproc, 同 enroll_infer:317-319
顺序), 再用 eval_metrics.CERMetric(主办方累计池 total_err/total_char)算。

用法: code/.venv/Scripts/python.exe code/recompute_official_cer.py [exp_vanilla_full.json]
输出: code/recompute_official_cer.json + 打印。
"""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc


def submit_norm(text):
    """复刻 enroll_infer 提交链路归一(顺序: to_simplified → digit_postproc)。"""
    return digit_postproc(to_simplified(text or ""))


def is_eng(t):
    L = [c for c in (t or "") if c.isalpha()]
    return sum(c.isascii() for c in L) / len(L) > 0.5 if len(L) >= 4 else False


def has_digit(t):
    return bool(re.search(r"\d", t or ""))


def pool_with_reject(hyps_submit, refs, sims, thr):
    """thr 工作点(含拒识): sim<thr 拒 → pred 置空(=不输出) → 官方池。
    拒识条 pred='' → editdistance('', norm_ref)=len(ref), char=len(ref) → CER=1.0 天然。"""
    m = CERMetric()
    for h, r, s in zip(hyps_submit, refs, sims):
        m.update(["" if s < thr else h], [r])
    res = m.compute()
    per = res["per_sample"]
    n = len(per)
    correct = sum(1 for x in per if x["cer"] < 0.5) / n if n else 0.0
    return res["cer"], correct


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "exp_vanilla_full.json")
    r = json.load(open(path, encoding="utf-8"))
    # valid 对齐 analyze_vanilla_full.py: 要求 max_sim 存在(排除 2 条 diar 失败 cmd_2220/2227) → 1362 条
    valid = [x for x in r if "vanilla_text" in x and "ref" in x and "max_sim" in x]
    n = len(valid)
    print(f"(total {len(r)} 条, diar 失败 {len(r)-n} 条排除 → {n} 条)")
    refs = [x["ref"] for x in valid]
    sims = [x["max_sim"] for x in valid]
    van_sub = [submit_norm(x["vanilla_text"]) for x in valid]
    dic_sub = [submit_norm(x["dicow_text"]) for x in valid]
    print(f"===== 官方 CER 口径重算（{n} 条, 提交侧归一后）=====")

    out = {"n": n}

    # 1. 转写 CER(不拒, always_generate)
    print("\n-- 转写 CER（不拒, 官方累计池）--")
    out["transcribe"] = {}
    for label, hyps in [("vanilla", van_sub), ("dicow", dic_sub)]:
        m = CERMetric(); m.update(hyps, refs); res = m.compute()
        per = res["per_sample"]
        correct = sum(1 for x in per if x["cer"] < 0.5) / n
        near = sum(1 for x in per if x["cer"] < 0.1) / n
        eng = sum(1 for x in hyps if is_eng(x)) / n
        out["transcribe"][label] = {"overall": round(res["cer"], 4), "correct": round(correct, 4),
                                     "near": round(near, 4), "eng_halluc": round(eng, 4)}
        print(f"  {label:8}: overall={res['cer']:.4f} correct={correct:.1%} near={near:.1%} 英文幻觉={eng:.1%}")

    # 2. thr 工作点(含拒识, 拒=1.0 进累计池 = 提交 overall CER, 最关键)
    print("\n-- thr 工作点（含拒识, 官方累计池 = 提交 overall CER）【最关键】--")
    print(f"  {'thr':<6} {'vanilla':<10} {'v_correct':<10} {'dicow':<10} {'d_correct':<10} {'Δ(v-d)':<10}")
    out["thr_workpoints"] = {}
    for thr in [0.2, 0.27, 0.3, 0.35, 0.4, 0.45]:
        vo, vc = pool_with_reject(van_sub, refs, sims, thr)
        do, dc = pool_with_reject(dic_sub, refs, sims, thr)
        out["thr_workpoints"][str(thr)] = {"vanilla": round(vo, 4), "vanilla_correct": round(vc, 4),
                                            "dicow": round(do, 4), "dicow_correct": round(dc, 4),
                                            "delta": round(vo - do, 4)}
        print(f"  {thr:<6.2f} {vo:<10.4f} {vc:<10.1%} {do:<10.4f} {dc:<10.1%} {vo-do:<+10.4f}")

    # 3. sim 分桶(转写 CER, 官方池)
    print("\n-- sim 分桶（转写 CER, 官方池）--")
    out["sim_buckets"] = {}
    for label, hyps in [("vanilla", van_sub), ("dicow", dic_sub)]:
        out["sim_buckets"][label] = {}
        for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]:
            idx = [i for i, s in enumerate(sims) if lo <= s < hi]
            if not idx:
                continue
            m = CERMetric(); m.update([hyps[i] for i in idx], [refs[i] for i in idx])
            v = m.compute()["cer"]
            out["sim_buckets"][label][f"[{lo:.1f},{hi:.1f})"] = {"n": len(idx), "cer": round(v, 4)}
            print(f"  {label} sim[{lo:.1f},{hi:.1f}): n={len(idx)} cer={v:.3f}")

    # 4. 数字格式影响(vanilla 后处理收益实证)
    print("\n-- 数字格式: vanilla 原始 vs 提交归一(含 digit_postproc) --")
    van_raw = [to_simplified(x["vanilla_text"]) for x in valid]  # 只繁简, 不转数字
    for label, hyps in [("vanilla 仅繁简(不转数字)", van_raw), ("vanilla 提交归一(+转数字)", van_sub)]:
        m = CERMetric(); m.update(hyps, refs); v = m.compute()["cer"]
        print(f"  {label:32}: overall={v:.4f}")

    out_path = os.path.join(os.path.dirname(__file__), "recompute_official_cer.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")


if __name__ == "__main__":
    main()
