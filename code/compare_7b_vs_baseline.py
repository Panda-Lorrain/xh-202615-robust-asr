#!/usr/bin/env python3
"""对比 Qwen3-ASR-7B(量化) vs 1.7B(baseline) 的 CER。

对齐口径(避免"目录不对齐"教训, 2026-07-24):
  baseline refs/sims 来自 code/runs/poc_qwen_asr_full_result.json(1350 条, 1.7B 全量产物)
  7B 输出 qwen7b_quant_backend.py 的 uid2text(同 cmd_X 命名)
  → 同一份 refs/sims, 只换转写 text, CER 差异 = 纯转写器差异(公平对比)

输出: overall transcribe CER + 分桶(死区/主战场/接近解决) + 含拒 thr0.27 提交口径 + RTF/mem。

用法: code/.venv/Scripts/python.exe code/compare_7b_vs_baseline.py \
        --q7b code/runs/_qwen7b_int4_uid2text.json
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc

BASELINE = os.path.join(os.path.dirname(__file__), "runs", "poc_qwen_asr_full_result.json")


def submit_norm(text):
    """复刻 enroll_infer 提交链路归一(to_simplified → digit_postproc)。"""
    return digit_postproc(to_simplified(text or ""))


def pool_cer(hyps, refs):
    """官方累计池 transcribe CER(不拒)。"""
    m = CERMetric(); m.update(hyps, refs); res = m.compute()
    per = res["per_sample"]
    n = len(per) if per else 1
    correct = sum(1 for x in per if x["cer"] < 0.5) / n if per else 0
    near = sum(1 for x in per if x["cer"] < 0.1) / n if per else 0
    return res["cer"], correct, near


def pool_with_reject(hyps, refs, sims, thr):
    m = CERMetric()
    for h, r, s in zip(hyps, refs, sims):
        m.update(["" if s < thr else h], [r])
    return m.compute()["cer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q7b", required=True, help="7B uid2text json(qwen7b_quant_backend 产物)")
    ap.add_argument("--baseline", default=BASELINE, help="baseline json(1.7B 全量)")
    args = ap.parse_args()

    base = json.load(open(args.baseline, encoding="utf-8"))
    q7 = json.load(open(args.q7b, encoding="utf-8"))
    q7_text = q7["uid2text"] if "uid2text" in q7 else q7

    rows = base["rows"]
    # 对齐: 7B 必须覆盖所有 baseline uid
    missing = [r["uid"] for r in rows if r["uid"] not in q7_text]
    if missing:
        print(f"⚠️ 7B 缺 {len(missing)} 条 baseline uid(前5: {missing[:5]}), 仅对齐共同子集")

    common = [r for r in rows if r["uid"] in q7_text]
    refs = [r["ref"] for r in common]
    sims = [r["sim"] for r in common]
    buckets = [r["bucket"] for r in common]
    q17_sub = [submit_norm(r["qwen"]) for r in common]   # 1.7B baseline(同集合重算)
    q7_sub = [submit_norm(q7_text[r["uid"]]) for r in common]

    n = len(common)
    print(f"===== Qwen3-ASR 7B vs 1.7B CER 对比（{n} 条共同 uid, 提交归一后）=====")
    print(f"  7B 模型: {q7.get('model','?')}  quant={q7.get('quant','?')}")
    print(f"  7B 推理: {q7.get('avg_infer_s_per_utt','?')}s/utt  peak_mem={q7.get('peak_gpu_mem_mb','?')}MB")
    print(f"  1.7B 基线: RTF=0.095s/utt  mem=3893MB")

    # ---- overall transcribe ----
    print("\n-- overall transcribe CER（不拒, 官方累计池）--")
    o17, c17, ne17 = pool_cer(q17_sub, refs)
    o7, c7, ne7 = pool_cer(q7_sub, refs)
    print(f"  1.7B : overall={o17:.4f}  correct={c17:.1%}  near={ne17:.1%}")
    print(f"  7B   : overall={o7:.4f}  correct={c7:.1%}  near={ne7:.1%}")
    print(f"  Δ    : {o7-o17:+.4f} ({'7B更优✓' if o7<o17 else '7B更差✗'})  CER腿40分: 1.7B={40*(1-o17):.1f} → 7B={40*(1-o7):.1f}")

    # ---- 分桶 ----
    print("\n-- 分桶（死区/主战场/接近解决）--")
    bset = sorted(set(buckets))
    for bk in bset:
        idx = [i for i, b in enumerate(buckets) if b == bk]
        if not idx:
            continue
        rb = [refs[i] for i in idx]; q17b = [q17_sub[i] for i in idx]; q7b = [q7_sub[i] for i in idx]
        ob17, _, _ = pool_cer(q17b, rb); ob7, _, _ = pool_cer(q7b, rb)
        print(f"  {bk:20} n={len(idx):4}: 1.7B={ob17:.4f} → 7B={ob7:.4f}  Δ={ob7-ob17:+.4f}")

    # ---- 含拒 thr0.27 提交口径 ----
    print("\n-- 含拒 thr=0.27（提交 overall CER, 排名公式实际用）--")
    r17 = pool_with_reject(q17_sub, refs, sims, 0.27)
    r7 = pool_with_reject(q7_sub, refs, sims, 0.27)
    print(f"  1.7B : {r17:.4f}  (baseline 坐实 0.5934)")
    print(f"  7B   : {r7:.4f}")
    print(f"  Δ    : {r7-r17:+.4f}  CER腿(含拒): 1.7B={40*(1-r17):.1f} → 7B={40*(1-r7):.1f}")

    # ---- 逐条 win/tie/loss ----
    m17 = CERMetric(); m17.update(q17_sub, refs); p17 = m17.compute()["per_sample"]
    m7 = CERMetric(); m7.update(q7_sub, refs); p7 = m7.compute()["per_sample"]
    win = sum(1 for a, b in zip(p7, p17) if a["cer"] < b["cer"] - 1e-6)
    tie = sum(1 for a, b in zip(p7, p17) if abs(a["cer"] - b["cer"]) <= 1e-6)
    loss = sum(1 for a, b in zip(p7, p17) if a["cer"] > b["cer"] + 1e-6)
    print(f"\n-- 逐条: 7B更优 {win} / 持平 {tie} / 7B更差 {loss}（共{n}）--")

    out = {"n": n, "model_7b": q7.get("model"), "quant": q7.get("quant"),
           "rtf_7b": q7.get("avg_infer_s_per_utt"), "mem_7b": q7.get("peak_gpu_mem_mb"),
           "transcribe": {"qwen17b": round(o17, 4), "qwen7b": round(o7, 4), "delta": round(o7 - o17, 4)},
           "thr027": {"qwen17b": round(r17, 4), "qwen7b": round(r7, 4), "delta": round(r7 - r17, 4)},
           "win_tie_loss": [win, tie, loss]}
    outp = args.q7b.replace(".json", "_vs_baseline.json")
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {outp}")


if __name__ == "__main__":
    main()
