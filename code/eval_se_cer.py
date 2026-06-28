"""线A — SE 降噪后 CER 对比评测(对 enroll_infer_result.json 算 CER)。

配合 enroll_infer.py 产出:
  code/.venv/Scripts/python.exe code/enroll_infer.py \
      --enrollment .../raw/enrollment/target_long_01.wav \
      --recognition-folder E:/.../final(or se_denoised) \
      --always-generate --out-json code/se_baseline.json(or se_denoised.json)
然后用本脚本:
  code/.venv/Scripts/python.exe code/eval_se_cer.py \
      --result code/se_baseline.json \
      --result-denoised code/se_denoised.json \
      --manifest E:/midea_target_asr/test_wav/dataset/final/final_manifest.json

输出: 全集 CER(baseline vs denoised) + 按 SNR/overlap/noise 分档 CER 对比 + 哪些条改善/变差。
CER 用 eval_metrics.cer(字符级)。
"""
import os
import sys
import json
import argparse
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_metrics import cer  # noqa


def _manifest_map(manifest_path):
    d = json.load(open(manifest_path, encoding="utf-8"))
    return {item["file"]: item for item in d["items"]}


def _basename(p):
    return os.path.basename(p)


def _cer_of(rec, m_map):
    f = _basename(rec["recognition"])
    ref = m_map.get(f, {}).get("target_ref", "")
    hyp = rec.get("transcript", "") or ""
    if rec.get("rejected"):
        # 拒识条: 若本应转(非空ref) -> 空hyp -> CER=1.0(目标漏转, 伤CER)
        pass
    return cer(hyp, ref), ref, hyp


def summarize(result_path, m_map):
    results = json.load(open(result_path, encoding="utf-8"))
    by_sn = defaultdict(list)      # snr_db -> list[(cer, item)]
    by_ov = defaultdict(list)
    by_nt = defaultdict(list)
    all_c = []
    for rec in results:
        f = _basename(rec["recognition"])
        it = m_map.get(f)
        if it is None:
            continue
        c, _, _ = _cer_of(rec, m_map)
        all_c.append((c, it, rec))
        by_sn[it["snr_db"]].append((c, it))
        by_ov[it["overlap_ratio"]].append((c, it))
        by_nt[it["noise_type"]].append((c, it))
    overall = sum(c for c, _, _ in all_c) / len(all_c) if all_c else float("nan")
    return {
        "overall_cer": overall, "n": len(all_c),
        "by_snr": {k: sum(c for c, _ in v) / len(v) for k, v in by_sn.items()},
        "by_overlap": {k: sum(c for c, _ in v) / len(v) for k, v in by_ov.items()},
        "by_noise": {k: sum(c for c, _ in v) / len(v) for k, v in by_nt.items()},
        "all": all_c,
    }


def main():
    ap = argparse.ArgumentParser(description="SE 降噪前后 CER 对比")
    ap.add_argument("--result", required=True, help="baseline enroll_infer JSON")
    ap.add_argument("--result-denoised", help="降噪后 enroll_infer JSON(不传则只报 baseline)")
    ap.add_argument("--manifest", default="E:/midea_target_asr/test_wav/dataset/final/final_manifest.json")
    args = ap.parse_args()

    m_map = _manifest_map(args.manifest)
    base = summarize(args.result, m_map)
    print(f"=== BASELINE ({args.result}) ===")
    print(f"  overall CER = {base['overall_cer']:.4f}  (n={base['n']})")
    print(f"  by_snr    = {base['by_snr']}")
    print(f"  by_overlap= {base['by_overlap']}")
    print(f"  by_noise  = {base['by_noise']}")

    if args.result_denoised:
        den = summarize(args.result_denoised, m_map)
        print(f"\n=== DENOISED ({args.result_denoised}) ===")
        print(f"  overall CER = {den['overall_cer']:.4f}  (n={den['n']})")
        print(f"  by_snr    = {den['by_snr']}")
        print(f"  by_overlap= {den['by_overlap']}")
        print(f"  by_noise  = {den['by_noise']}")
        d_overall = den['overall_cer'] - base['overall_cer']
        print(f"\n=== DELTA (denoised - baseline) ===")
        print(f"  overall ΔCER = {d_overall:+.4f}  ({'改善(↓)' if d_overall < 0 else '变差(↑)' if d_overall > 0 else '持平'})")
        for dim, label in [("by_snr", "SNR"), ("by_overlap", "overlap"), ("by_noise", "noise")]:
            print(f"  [{label}]")
            for k in sorted(base[dim]):
                if k in den[dim]:
                    print(f"    {k}: {base[dim][k]:.4f} -> {den[dim][k]:.4f} (Δ{den[dim][k]-base[dim][k]:+.4f})")

        # 逐条改善/变差 Top
        base_map = {_basename(r["recognition"]): _cer_of(r, m_map)[0] for r in json.load(open(args.result, encoding="utf-8"))}
        diffs = []
        for r in json.load(open(args.result_denoised, encoding="utf-8")):
            f = _basename(r["recognition"])
            if f in base_map:
                dc, _, _ = _cer_of(r, m_map)
                diffs.append((dc - base_map[f], f))
        diffs.sort()
        print("\n  改善最多 Top5:", [(round(d,3),f) for d,f in diffs[:5]])
        print("  变差最多 Top5:", [(round(d,3),f) for d,f in diffs[-5:]])


if __name__ == "__main__":
    main()
