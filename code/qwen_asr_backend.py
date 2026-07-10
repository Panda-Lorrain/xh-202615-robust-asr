#!/usr/bin/env python
"""Qwen3-ASR 批量转写 target 切片（code/.venv_qwen，venv 隔离）。
供 enroll_infer --asr-backend qwen 末尾 subprocess 调用，填 result text。

用法: code/.venv_qwen/Scripts/python.exe code/qwen_asr_backend.py \
        --slice-dir E:/target_slices_qwen --out code/_qwen_uid2text.json [--limit N]
"""
import os, json, glob, argparse, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True, help="target 切片 wav 目录(uid 命名)")
    ap.add_argument("--model", default="E:/hf_cache/Qwen3-ASR-1.7B")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    args = ap.parse_args()

    from qwen_asr import Qwen3ASRModel
    print(f"[load] Qwen3-ASR {args.model} bf16 ...")
    model = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")

    slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")))
    if args.limit:
        slices = slices[:args.limit]
    print(f"{len(slices)} 切片")

    uid2text = {}
    t0 = time.time()
    for i, sf in enumerate(slices):
        uid = os.path.splitext(os.path.basename(sf))[0]
        try:
            res = model.transcribe(audio=sf, language="Chinese")
            uid2text[uid] = res[0].text.strip()
        except Exception as e:
            print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:50]}")
            uid2text[uid] = ""
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(slices)}] ({(i+1)/(time.time()-t0):.1f}/s)")
    json.dump(uid2text, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"转写 {len(uid2text)} 条 → {args.out} (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
