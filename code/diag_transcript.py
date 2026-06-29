"""诊断转写质量分布 + 干净条件下是否选对 speaker(排查 target_idx/STNO bug)。"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import cer

enroll = json.load(open("code/se_denoised.json", encoding="utf-8"))
mani = {it["file"]: it for it in json.load(
    open("test_wav/dataset/final/final_manifest.json", encoding="utf-8"))["items"]}

buckets = collections.Counter()
by_cond = collections.defaultdict(list)
for r in enroll:
    f = os.path.basename(r.get("recognition", ""))
    m = mani.get(f)
    if not m:
        continue
    ref = m["target_ref"]
    hyp = r.get("transcript", "") or ""
    c = cer(hyp, ref)
    if c < 0.5:
        buckets["good(<0.5)"] += 1
    elif c < 1.0:
        buckets["ok(0.5-1)"] += 1
    elif c < 2.0:
        buckets["bad(1-2)"] += 1
    else:
        buckets["garbage(>=2)"] += 1
    by_cond[(m["overlap_ratio"], m["snr_db"], m["noise_type"])].append(
        (c, hyp, r.get("max_sim", 0), r.get("target_idx"), r.get("speakers")))

print("=== CER 分布(450) ===")
for k in ["good(<0.5)", "ok(0.5-1)", "bad(1-2)", "garbage(>=2)"]:
    print(f"  {k}: {buckets[k]} ({buckets[k]/450:.1%})")

for label, ov, snr, nt in [("干净 ov0/snr+5/white", 0.0, 5, "white"),
                           ("中等 ov0/snr0/white", 0.0, 0, "white"),
                           ("最差 ov0/snr-5/babble", 0.0, -5, "babble"),
                           ("重叠 ov100/snr0/babble", 1.0, 0, "babble")]:
    lst = by_cond.get((ov, snr, nt), [])
    print(f"\n=== {label} (n={len(lst)}) ===")
    for c, hyp, sim, tidx, spk in lst[:4]:
        print(f"  CER={c:.2f} sim={sim:.3f} target_idx={tidx} speakers={spk}")
        print(f"    hyp: {hyp[:55]}")
