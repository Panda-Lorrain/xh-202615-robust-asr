"""SE 条件化合并: 按 noise_type 从 denoised(atten-lim=0)/a6(atten-lim=6) 挑条,
合成"按噪声类型分流"的最优条件化 json, 供 eval_se_cer 对比。

背景(T18 线A): 全局 atten-lim=6 比 =0 差(overall 3.95 vs 3.65), 但分档揭示——
  - pink(稳态): =6 解过消除(pink Δ +2.20→-0.29), =6 更优
  - babble(人声)/white/SNR-5: =0 全力降噪更优(babble Δ -4.20 vs -0.98)
故最优条件化 = 按 noise_type 分流: babble/white→=0, pink→=6。
预期 overall ~2.82(vs 单一=0 的 3.65, baseline 4.27)。

用法:
  code/.venv/Scripts/python.exe code/merge_se_conditional.py \
      --result-0 code/se_denoised.json \
      --result-6 code/se_denoised_a6.json \
      --out code/se_conditional.json \
      [--rule babble:0,white:0,pink:6]
  code/.venv/Scripts/python.exe code/eval_se_cer.py \
      --result code/se_baseline.json --result-denoised code/se_conditional.json
"""
import os
import json
import argparse


def main():
    ap = argparse.ArgumentParser(description="SE 条件化: 按 noise_type 分流合并 =0/=6")
    ap.add_argument("--result-0", required=True, help="denoised atten-lim=0 json")
    ap.add_argument("--result-6", required=True, help="denoised atten-lim=6 json")
    ap.add_argument("--manifest", default="E:/midea_target_asr/test_wav/dataset/final/final_manifest.json")
    ap.add_argument("--out", required=True, help="输出条件化 json")
    ap.add_argument("--rule", default="babble:0,white:0,pink:6",
                    help="noise_type:版本(0或6), 逗号分; 未列的 noise_type 默认用 0")
    args = ap.parse_args()

    rule = {}
    for pair in args.rule.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            rule[k.strip()] = v.strip()

    m = json.load(open(args.manifest, encoding="utf-8"))
    noise_map = {it["file"]: it["noise_type"] for it in m["items"]}

    r0 = {os.path.basename(r["recognition"]): r for r in json.load(open(args.result_0, encoding="utf-8"))}
    r6 = {os.path.basename(r["recognition"]): r for r in json.load(open(args.result_6, encoding="utf-8"))}

    merged = []
    used = {"0": 0, "6": 0}
    miss = 0
    for f, nt in noise_map.items():
        ver = rule.get(nt, "0")
        src = r0 if ver == "0" else r6
        rec = src.get(f)
        if rec is None:
            rec = r0.get(f) or r6.get(f)  # 兜底: 任一有就用
            miss += 1 if rec is None else 0
        if rec is not None:
            merged.append(rec)
            used[ver if rec in (r0.get(f), r6.get(f)) and rec is src.get(f) else "0"] += 1

    json.dump(merged, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"条件化合并: {len(merged)} 条 | 规则={args.rule}")
    print(f"  (babble/white→=0, pink→=6; 未匹配 manifest 的兜底用 =0)")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
