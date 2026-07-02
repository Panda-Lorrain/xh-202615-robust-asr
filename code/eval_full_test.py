"""一次性评测：submit_infer 全量 result.json + final_manifest gt -> 真实 CER 全貌。
复用 eval_metrics.cer()。按 overlap/SNR/noise 分解 + correct_rate + 拒识率。
用法: code/.venv/Scripts/python.exe code/eval_full_test.py [result.json] [manifest.json]
"""
import json, re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import cer

RESULT = sys.argv[1] if len(sys.argv) > 1 else "code/submit_out_full/result.json"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else "test_wav/dataset/final/final_manifest.json"
OUT = sys.argv[3] if len(sys.argv) > 3 else "code/submit_out_full/eval.json"

results = json.load(open(RESULT, encoding="utf-8"))["results"]
man = {it["file"].replace(".wav", ""): it
       for it in json.load(open(MANIFEST, encoding="utf-8"))["items"]}

NAME = re.compile(r"ov(\d+)_snr([+-]?\d+)_([a-z]+)")

def parse(uid):
    base = re.sub(r"^utt\d+_", "", uid)
    m = NAME.search(base)
    return base, int(m.group(1)) / 100, int(m.group(2)), m.group(3)

rows = []
miss = 0
for r in results:
    base, ov, snr, noise = parse(r["utt_id"])
    if base not in man:
        miss += 1
        continue
    ref = man[base]["target_ref"]
    if r["rejected"] or not r["text"]:
        c, text = 1.0, ""
    else:
        text = r["text"]
        c = cer(text, ref)
    rows.append({"base": base, "ov": ov, "snr": snr, "noise": noise,
                 "cer": c, "rejected": r["rejected"], "text": text, "ref": ref,
                 "sim": r.get("max_sim")})

n = len(rows)
overall = sum(r["cer"] for r in rows) / n
correct = sum(1 for r in rows if r["cer"] < 0.5) / n
good_strict = sum(1 for r in rows if r["cer"] < 0.1) / n
rej_rate = sum(1 for r in rows if r["rejected"]) / n
acc = [r for r in rows if not r["rejected"]]
cer_acc = sum(r["cer"] for r in acc) / len(acc) if acc else 0.0

def grp(key):
    buckets = {}
    for r in rows:
        buckets.setdefault(r[key], []).append(r["cer"])
    return {k: {"mean_cer": round(sum(v)/len(v), 3), "n": len(v),
                "correct_rate": round(sum(1 for x in v if x < 0.5)/len(v), 3)}
            for k, v in sorted(buckets.items())}

summary = {
    "n": n, "manifest_miss": miss,
    "overall_cer": round(overall, 3),
    "correct_rate(CER<0.5)": round(correct, 3),
    "near_perfect(CER<0.1)": round(good_strict, 3),
    "reject_rate": round(rej_rate, 3),
    "n_rejected": sum(1 for r in rows if r["rejected"]),
    "cer_accepted_only": round(cer_acc, 3),
    "by_overlap": grp("ov"),
    "by_snr": grp("snr"),
    "by_noise": grp("noise"),
}
json.dump(summary, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print(f"\n===== 本次实测 ({n} 条, manifest 未匹配 {miss}) =====")
print(f"overall CER           : {overall:.3f}")
print(f"correct_rate (CER<0.5): {correct:.1%} ({sum(1 for r in rows if r['cer']<0.5)}/{n})")
print(f"near_perfect(CER<0.1) : {good_strict:.1%}")
print(f"reject_rate           : {rej_rate:.1%} ({summary['n_rejected']}/{n})  [450全target在场→均为误拒]")
print(f"cer (仅 accepted)     : {cer_acc:.3f}")
print("\n-- by overlap --")
for k, v in summary["by_overlap"].items():
    print(f"  ov{k:<5}: CER {v['mean_cer']:<6} correct {v['correct_rate']:<6} n={v['n']}")
print("\n-- by SNR --")
for k, v in summary["by_snr"].items():
    print(f"  snr{k:<+3}: CER {v['mean_cer']:<6} correct {v['correct_rate']:<6} n={v['n']}")
print("\n-- by noise --")
for k, v in summary["by_noise"].items():
    print(f"  {k:<7}: CER {v['mean_cer']:<6} correct {v['correct_rate']:<6} n={v['n']}")
print(f"\n[已写] {OUT}")
