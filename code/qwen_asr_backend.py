#!/usr/bin/env python
"""Qwen3-ASR 批量转写 target 切片（code/.venv_qwen，venv 隔离）。
供 enroll_infer --asr-backend qwen 末尾 subprocess 调用，填 result text。

用法: code/.venv_qwen/Scripts/python.exe code/qwen_asr_backend.py \
        --slice-dir E:/target_slices_qwen --out code/_qwen_uid2text.json [--limit N]
"""
import os, json, glob, argparse, time
import torch

# 跨平台默认模型路径(原 E:/ 硬编码在 Linux 阻塞): env MODEL_QWEN3_ASR 可覆盖
_DEFAULT_QWEN3 = (r"E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt"
                  else "/root/hf_cache/Qwen3-ASR-1.7B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True, help="target 切片 wav 目录(uid 命名)")
    ap.add_argument("--model", default=os.environ.get("MODEL_QWEN3_ASR", _DEFAULT_QWEN3),
                    help="Qwen3-ASR 权重目录(env MODEL_QWEN3_ASR 覆盖)")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(可复现性, 透传自 enroll_infer)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="batch 推理大小(0=逐条, -1=全部一次; 默认 16, 4060 实测 5× 加速, 8GB 安全)")
    args = ap.parse_args()

    # 可复现性: 固定 seed(独立 venv 不依赖 repro.py, 内联 set_seed; Qwen3-ASR generate 默认
    # greedy do_sample=False, seed 主要锁 cudnn 算子 + 初始化, 对贪心 argmax 鲁棒)
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

    from qwen_asr import Qwen3ASRModel
    print(f"[load] Qwen3-ASR {args.model} bf16 ...")
    model = Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0")

    slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")))
    if args.limit:
        slices = slices[:args.limit]
    uids = [os.path.splitext(os.path.basename(sf))[0] for sf in slices]
    print(f"{len(slices)} 切片, batch_size={args.batch_size}")

    uid2text = {}
    t0 = time.time()

    if args.batch_size == 0:
        # 逐条模式(原逻辑, 兼容)
        for i, sf in enumerate(slices):
            uid = uids[i]
            try:
                res = model.transcribe(audio=sf, language="Chinese")
                uid2text[uid] = res[0].text.strip()
            except Exception as e:
                print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:50]}")
                uid2text[uid] = ""
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(slices)}] ({(i+1)/(time.time()-t0):.1f}/s)")
    else:
        # batch 模式: 利用 Qwen3ASRModel.transcribe 内置 batch 推理
        bs = args.batch_size if args.batch_size > 0 else len(slices)
        n_batches = (len(slices) + bs - 1) // bs
        for bi in range(0, len(slices), bs):
            batch_paths = slices[bi : bi + bs]
            batch_uids = uids[bi : bi + bs]
            batch_idx = bi // bs + 1
            try:
                results = model.transcribe(audio=batch_paths, language="Chinese")
                for uid, res in zip(batch_uids, results):
                    uid2text[uid] = res.text.strip()
            except torch.cuda.OutOfMemoryError:
                # OOM fallback: 逐条重试本 batch
                torch.cuda.empty_cache()
                print(f"  batch {batch_idx}/{n_batches} OOM, fallback 逐条")
                for uid, sp in zip(batch_uids, batch_paths):
                    try:
                        res = model.transcribe(audio=sp, language="Chinese")
                        uid2text[uid] = res[0].text.strip()
                    except Exception as e:
                        print(f"    {uid} FAIL {type(e).__name__}: {str(e)[:50]}")
                        uid2text[uid] = ""
            except Exception as e:
                print(f"  batch {batch_idx}/{n_batches} FAIL {type(e).__name__}: {str(e)[:80]}")
                for uid in batch_uids:
                    uid2text[uid] = ""
            elapsed = time.time() - t0
            done = min(bi + bs, len(slices))
            print(f"  [{done}/{len(slices)}] batch {batch_idx}/{n_batches} ({done/elapsed:.1f}/s)")

    json.dump(uid2text, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"转写 {len(uid2text)} 条 → {args.out} (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
