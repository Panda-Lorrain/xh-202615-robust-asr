#!/usr/bin/env python3
"""FireRedASR-AED-L 批量转写 target 切片（code/.venv_firered, venv 隔离）。

B1 横评: 复用 enroll_infer 切的 E:/target_slices_full/(与 Qwen3-ASR 同源切片),
喂 FireRedASR-AED-L 转写 → 算官方 CER 对比 Qwen3-ASR 0.3436, 选型 + 效率腿(RTF)。

用法: code/.venv_firered/Scripts/python.exe code/firered_asr_backend.py \
        --slice-dir E:/target_slices_full --out code/_firered_uid2text.json [--limit N]
"""
import os, sys, json, glob, argparse, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FireRedASR"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True, help="target 切片 wav 目录(uid 命名)")
    ap.add_argument("--model", default="E:/hf_cache/FireRedASR-AED-L")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(可复现性)")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from fireredasr.models.fireredasr import FireRedAsr
    print(f"[load] FireRedASR-AED-L {args.model} ...")
    model = FireRedAsr.from_pretrained("aed", args.model)

    slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")))
    if args.limit:
        slices = slices[: args.limit]
    print(f"{len(slices)} 切片")

    uid2text = {}
    t0 = time.time()
    for i, sf in enumerate(slices):
        uid = os.path.splitext(os.path.basename(sf))[0]
        try:
            res = model.transcribe([uid], [sf], {"use_gpu": True})
            uid2text[uid] = res[0]["text"].strip()
        except Exception as e:
            print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:80]}")
            uid2text[uid] = ""
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(slices)}] ({(i+1)/(time.time()-t0):.1f}/s)")
    json.dump(uid2text, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"转写 {len(uid2text)} 条 → {args.out} (耗时 {time.time()-t0:.0f}s, {(time.time()-t0)/max(len(uid2text),1):.2f}s/条)")


if __name__ == "__main__":
    main()
