"""评估 enrollment→target 能力: 读 enroll_infer JSON + final_manifest, 算分档指标。

用法:
  python code/eval_enrollment.py --enroll-json code/enroll_wespeaker_full.json

指标(按 overlap / SNR 分档 + 总体):
  - n: 条数
  - 拒识率: rejected / n
  - diar 失败: 触发 diarization crash 的条数
  - 均值 max_sim
  - 不拒识条的 target CER(转写 vs ground truth, char-level Levenshtein)
  - 近似锁定+转写正确率(CER < lock_cer_thresh): 越高越好

注: final 音频里 diarization 的 speaker 0/1 顺序不定, 无法直接判"选对 target"。
间接判断: 锁定对+转写好 → CER 低; 锁定错(转成 nontarget) → CER 高。
故 CER<0.5 比例 ≈ 锁定+转写正确率。
"""
import json, argparse, os
from collections import defaultdict


def levenshtein(a, b):
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


def cer(hyp, ref):
    hyp, ref = hyp or "", ref or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(hyp, ref) / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll-json", required=True)
    ap.add_argument("--manifest", default="E:/midea_target_asr/test_wav/dataset/final/final_manifest.json")
    ap.add_argument("--lock-cer-thresh", type=float, default=0.5)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    enroll = json.load(open(args.enroll_json, encoding="utf-8"))
    mani = json.load(open(args.manifest, encoding="utf-8"))
    mani_map = {os.path.basename(it["file"]): it for it in mani["items"]}

    stats = defaultdict(lambda: {"n": 0, "reject": 0, "diar_fail": 0, "sims": [], "cers": []})
    for r in enroll:
        fname = os.path.basename(r.get("recognition", ""))
        m = mani_map.get(fname)
        key = (round(m["overlap_ratio"], 2), m["snr_db"]) if m else (None, None)
        s = stats[key]
        s["n"] += 1
        if r.get("error"):
            s["diar_fail"] += 1
            continue
        s["sims"].append(r.get("max_sim", 0))
        if r.get("rejected"):
            s["reject"] += 1
        else:
            c = cer(r.get("transcript", ""), m["target_ref"] if m else "")
            s["cers"].append(c)

    def agg(by_key_func):
        d = defaultdict(lambda: {"n": 0, "reject": 0, "diar_fail": 0, "sims": [], "cers": []})
        for (ov, snr), s in stats.items():
            k = by_key_func(ov, snr)
            dd = d[k]
            dd["n"] += s["n"]; dd["reject"] += s["reject"]; dd["diar_fail"] += s["diar_fail"]
            dd["sims"] += s["sims"]; dd["cers"] += s["cers"]
        return d

    by_ov = agg(lambda ov, snr: ov)
    by_snr = agg(lambda ov, snr: snr)
    tot = agg(lambda ov, snr: "ALL")["ALL"]

    def show(title, d):
        print(f"\n--- 按 {title} {'('+args.label+')' if args.label else ''} ---")
        print(f"{'值':>8} {'n':>4} {'拒识率':>7} {'diar败':>6} {'均sim':>7} {'不拒CER':>8} {'正确率':>7}")
        for k in sorted(d.keys(), key=lambda x: (x is None, x)):
            s = d[k]
            if s["n"] == 0:
                continue
            rej = s["reject"] / s["n"]
            sim = sum(s["sims"]) / len(s["sims"]) if s["sims"] else 0
            nonrej_cer = sum(s["cers"]) / len(s["cers"]) if s["cers"] else float('nan')
            lock = sum(1 for c in s["cers"] if c < args.lock_cer_thresh) / s["n"]
            ks = "ALL" if k == "ALL" else str(k)
            print(f"{ks:>8} {s['n']:>4} {rej:>7.2f} {s['diar_fail']:>6} {sim:>7.3f} {nonrej_cer:>8.3f} {lock:>7.2f}")

    show("overlap", by_ov)
    show("SNR", by_snr)
    show("总体", {"ALL": tot})


if __name__ == "__main__":
    main()
