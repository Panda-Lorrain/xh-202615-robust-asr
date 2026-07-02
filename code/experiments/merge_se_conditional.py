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
    ap.add_argument("--noise-est", help="估计噪声类型 JSON(noise_classify 输出); 传则用估计值(可部署), 否则用 manifest(oracle)")
    ap.add_argument("--out", required=True, help="输出条件化 json")
    ap.add_argument("--rule", default="babble:0,white:0,pink:6",
                    help="noise_type:版本(0或6), 逗号分; 未列的 noise_type 默认用 0")
    args = ap.parse_args()

    rule = {}
    for pair in args.rule.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            rule[k.strip()] = v.strip()

    if args.noise_est:
        est = json.load(open(args.noise_est, encoding="utf-8"))
        noise_map = {r["file"]: r["est_noise"] for r in est}
        src_label = "估计(可部署)"
    else:
        m = json.load(open(args.manifest, encoding="utf-8"))
        noise_map = {it["file"]: it["noise_type"] for it in m["items"]}
        src_label = "manifest(oracle)"

    r0 = {os.path.basename(r["recognition"]): r for r in json.load(open(args.result_0, encoding="utf-8"))}
    r6 = {os.path.basename(r["recognition"]): r for r in json.load(open(args.result_6, encoding="utf-8"))}

    merged = []
    used = {"0": 0, "6": 0}
    miss = 0
    for f, nt in noise_map.items():
        ver = rule.get(nt, "0")
        rec = (r0 if ver == "0" else r6).get(f)
        if rec is None:                       # 该版本无此条, 兜底用另一版
            rec = (r6 if ver == "0" else r0).get(f)
            if rec is None:
                miss += 1
                continue
        merged.append(rec)
        used[ver] += 1

    json.dump(merged, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"条件化合并[{src_label}]: {len(merged)} 条 | 规则={args.rule} | 用 =0:{used['0']} =6:{used['6']} | 缺失跳过:{miss}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
