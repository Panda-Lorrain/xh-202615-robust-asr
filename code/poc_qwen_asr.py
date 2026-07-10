#!/usr/bin/env python
"""POC: Qwen3-ASR-1.7B 全量 pos target 切片 CER vs vanilla（候选2 全量确认）
===========================================================
60 条探针已证主战场 Δ-0.31。本脚本全量 1362 条 pos 切片 + 分桶(死区/主战场/接近解决)
量化 overall Qwen3-ASR CER vs vanilla（官方口径近似）。

用法: code/.venv_qwen/Scripts/python.exe code/poc_qwen_asr.py
"""
import os, json, glob, time, unicodedata
import torch

QWEN_PATH = "E:/hf_cache/Qwen3-ASR-1.7B"
SLICE_DIR = "E:/target_slices_full"      # 全量切片(enroll_infer --save-target-audio)
EXP = "code/exp_vanilla_full.json"       # uid → {ref, vanilla_text, vanilla_cer, max_sim}


def norm(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(c for c in s if not unicodedata.category(c).startswith("P") and not c.isspace())


def cer(hyp, ref):
    h, r = norm(hyp), norm(ref or "")
    if not r:
        return 0.0
    m, n = len(h), len(r)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if h[i - 1] == r[j - 1] else 1))
        prev = cur
    return prev[n] / n


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    exp_rows = json.load(open(EXP, encoding="utf-8"))
    exp = {x["uid"]: x for x in (exp_rows if isinstance(exp_rows, list) else exp_rows.get("results", []))}

    from qwen_asr import Qwen3ASRModel
    print(f"[load] Qwen3-ASR {QWEN_PATH} bf16 ...")
    model = Qwen3ASRModel.from_pretrained(QWEN_PATH, dtype=torch.bfloat16, device_map="cuda:0")

    slices = sorted(glob.glob(SLICE_DIR + "/*.wav"))
    if not slices:
        print(f"⚠️ 无切片 {SLICE_DIR}（enroll_infer --save-target-audio 未跑完）"); return
    print(f"{len(slices)} 切片\n")

    buckets = {"<0.2 死区": [], "[0.2,0.4) 主战场": [], ">=0.4 接近解决": []}
    rows = []
    t0 = time.time()
    for i, sf in enumerate(slices):
        uid = os.path.splitext(os.path.basename(sf))[0]
        e = exp.get(uid, {})
        ref = e.get("ref", "")
        van_text = e.get("vanilla_text", "")
        sim = float(e.get("max_sim", 0) or 0)
        try:
            res = model.transcribe(audio=sf, language="Chinese")
            text = res[0].text.strip()
        except Exception as ex:
            print(f"  {uid} FAIL {type(ex).__name__}: {str(ex)[:50]}"); continue
        q = cer(text, ref)
        v = cer(van_text, ref)
        b = "<0.2 死区" if sim < 0.2 else ("[0.2,0.4) 主战场" if sim < 0.4 else ">=0.4 接近解决")
        buckets[b].append((q, v))
        rows.append({"uid": uid, "sim": sim, "bucket": b, "ref": ref,
                     "vanilla": van_text, "qwen": text, "qwen_cer": q, "vanilla_cer": v})
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(slices)}] {uid} qwen {q:.3f} van {v:.3f} ({(i+1)/(time.time()-t0):.1f} 条/s)")

    allq = [r["qwen_cer"] for r in rows]
    allv = [r["vanilla_cer"] for r in rows]
    print(f"\n{'='*60}\n全量 pos target 切片 CER (n={len(rows)}):")
    print(f"  overall:  Qwen3-ASR {mean(allq):.4f} vs vanilla {mean(allv):.4f}  Δ={mean(allq)-mean(allv):+.4f}")
    for b in ["<0.2 死区", "[0.2,0.4) 主战场", ">=0.4 接近解决"]:
        lst = buckets[b]
        if lst:
            qs = [q for q, _ in lst]; vs = [v for _, v in lst]
            print(f"  {b:18s} (n={len(lst):4d}): Qwen {mean(qs):.4f} vs vanilla {mean(vs):.4f}  Δ={mean(qs)-mean(vs):+.4f}")
    win = sum(1 for r in rows if r["qwen_cer"] < r["vanilla_cer"])
    print(f"  Qwen 更优: {win}/{len(rows)} = {100*win/len(rows):.0f}%")
    print(f"  耗时 {time.time()-t0:.0f}s, RTF≈{(time.time()-t0)/len(rows):.3f}s/条 (4060)")

    json.dump({"n": len(rows), "overall_qwen": mean(allq), "overall_vanilla": mean(allv),
               "delta": mean(allq) - mean(allv), "win_rate": win/len(rows),
               "buckets": {b: {"qwen": mean([q for q, _ in buckets[b]]),
                               "vanilla": mean([v for _, v in buckets[b]]),
                               "n": len(buckets[b])} for b in buckets},
               "rows": rows},
              open("code/poc_qwen_asr_full_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n存 code/poc_qwen_asr_full_result.json")


if __name__ == "__main__":
    main()
