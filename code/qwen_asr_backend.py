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
    ap.add_argument(
        "--paired-slice-dir",
        help="optional second uid-named WAV directory transcribed in the same process",
    )
    ap.add_argument(
        "--paired-out",
        help="uid→text JSON for --paired-slice-dir",
    )
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(可复现性, 透传自 enroll_infer)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="batch 推理大小(0=逐条, -1=全部一次; 默认 16, 4060 实测 5× 加速, 8GB 安全)")
    ap.add_argument("--context", default="",
                    help="transcribe context(家居场景引导, system message 注入; Qwen3-ASR 原生支持 qwen3_asr.py:303). 默认空=不引导(等价原行为); hold-out 验证后再定启用.")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="repetition_penalty 透传 generate(直击循环英文幻觉, 近零RTF; 1.0=关=原greedy; 建议1.05-1.2)")
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0,
                    help="no_repeat_ngram_size 透传 generate(硬ban n-gram重复=循环幻觉克星, 近零RTF; 0=关; 中文建议>=3避免误杀叠字)")
    args = ap.parse_args()
    if bool(args.paired_slice_dir) != bool(args.paired_out):
        ap.error("--paired-slice-dir and --paired-out must be provided together")

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

    # 零RTF解码参数: monkey-patch model.model.generate 注入 rep_penalty/no_repeat_ngram_size
    # (transcribe→_infer_asr_transformers→self.model.generate 不暴露这些 kwarg, 包一层注入;
    #  默认值=原greedy行为, 向后兼容; C1/C2 context 实验不受影响)
    if args.rep_penalty != 1.0 or args.no_repeat_ngram_size > 0:
        _orig_generate = model.model.generate
        def _patched_generate(*a, **kw):
            kw.setdefault("repetition_penalty", args.rep_penalty)
            kw.setdefault("no_repeat_ngram_size", args.no_repeat_ngram_size)
            return _orig_generate(*a, **kw)
        model.model.generate = _patched_generate
        print(f"[patch] generate rep_penalty={args.rep_penalty} no_repeat_ngram_size={args.no_repeat_ngram_size}")

    if args.paired_slice_dir:
        paired_slices = sorted(
            glob.glob(os.path.join(args.paired_slice_dir, "*.wav"))
        )
        if args.limit:
            paired_slices = paired_slices[:args.limit]
        uids = [
            os.path.splitext(os.path.basename(path))[0]
            for path in paired_slices
        ]
        slices = [
            os.path.join(args.slice_dir, uid + ".wav") for uid in uids
        ]
        for uid, primary_path in zip(uids, slices):
            if not os.path.exists(primary_path):
                raise FileNotFoundError(
                    f"primary WAV missing for paired uid {uid}: {primary_path}"
                )
        items = []
        for uid, primary_path, paired_path in zip(
            uids, slices, paired_slices
        ):
            items.extend(
                [
                    (uid, "primary", primary_path),
                    (uid, "paired", paired_path),
                ]
            )
        # Keep each raw/enhanced pair adjacent in the same inference stream.
    else:
        slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")))
        if args.limit:
            slices = slices[:args.limit]
        uids = [os.path.splitext(os.path.basename(sf))[0] for sf in slices]
        items = [(uid, "primary", path) for uid, path in zip(uids, slices)]
    print(
        f"{len(uids)} uid / {len(items)} 切片, "
        f"paired={bool(args.paired_slice_dir)}, batch_size={args.batch_size}"
    )

    uid2text = {}
    paired_uid2text = {}
    t0 = time.time()

    if args.batch_size == 0:
        # 逐条模式(原逻辑, 兼容)
        for i, (uid, stream, sf) in enumerate(items):
            try:
                res = model.transcribe(audio=sf, language="Chinese", context=args.context)
                target = paired_uid2text if stream == "paired" else uid2text
                target[uid] = res[0].text.strip()
            except Exception as e:
                print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:50]}")
                target = paired_uid2text if stream == "paired" else uid2text
                target[uid] = ""
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(items)}] ({(i+1)/(time.time()-t0):.1f}/s)")
    else:
        # batch 模式: 利用 Qwen3ASRModel.transcribe 内置 batch 推理
        bs = args.batch_size if args.batch_size > 0 else len(items)
        n_batches = (len(items) + bs - 1) // bs
        for bi in range(0, len(items), bs):
            batch_items = items[bi : bi + bs]
            batch_paths = [item[2] for item in batch_items]
            batch_idx = bi // bs + 1
            try:
                results = model.transcribe(audio=batch_paths, language="Chinese", context=args.context)
                for (uid, stream, _), res in zip(batch_items, results):
                    target = paired_uid2text if stream == "paired" else uid2text
                    target[uid] = res.text.strip()
            except torch.cuda.OutOfMemoryError:
                # OOM fallback: 逐条重试本 batch
                torch.cuda.empty_cache()
                print(f"  batch {batch_idx}/{n_batches} OOM, fallback 逐条")
                for (uid, stream, sp) in batch_items:
                    try:
                        res = model.transcribe(audio=sp, language="Chinese", context=args.context)
                        target = paired_uid2text if stream == "paired" else uid2text
                        target[uid] = res[0].text.strip()
                    except Exception as e:
                        print(f"    {uid} FAIL {type(e).__name__}: {str(e)[:50]}")
                        target = paired_uid2text if stream == "paired" else uid2text
                        target[uid] = ""
            except Exception as e:
                print(f"  batch {batch_idx}/{n_batches} FAIL {type(e).__name__}: {str(e)[:80]}")
                for uid, stream, _ in batch_items:
                    target = paired_uid2text if stream == "paired" else uid2text
                    target[uid] = ""
            elapsed = time.time() - t0
            done = min(bi + bs, len(items))
            print(f"  [{done}/{len(items)}] batch {batch_idx}/{n_batches} ({done/elapsed:.1f}/s)")

    json.dump(uid2text, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"转写 {len(uid2text)} 条 → {args.out} (耗时 {time.time()-t0:.0f}s)")
    if args.paired_out:
        json.dump(
            paired_uid2text,
            open(args.paired_out, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        print(f"配对转写 {len(paired_uid2text)} 条 → {args.paired_out}")


if __name__ == "__main__":
    main()
