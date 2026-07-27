#!/usr/bin/env python
"""Qwen3-ASR 7B 量化转写 target 切片（code/.venv_qwen，venv 隔离）。

目的: 对比 1.7B(bf16, CER 0.3436 / RTF 0.095 per-utt) 看 7B 量化能否降 CER, 代价 RTF 慢多少。
背景: 4060 8GB, 7B bf16=14GB 跑不了, 必须 int4 量化(NF4, 权重~3.5GB+激活, 8GB 临界能跑)。
已知坑: 历史 bnb int8 对 1.7B 慢 4×(0.095→0.379)且伤精度(防直吹→风直吹); int4 显存更省但 RTF 代价待实测。

量化注入路径(源码核实 qwen_asr/inference/qwen3_asr.py:206):
  Qwen3ASRModel.from_pretrained(model, **kwargs) → AutoModel.from_pretrained(model, **kwargs)
  → 透传 quantization_config=BitsAndBytesConfig(load_in_4bit/nf4/bf16 compute/double_quant) 标准方式。

用法(7B int4 全量):
  code/.venv_qwen/Scripts/python.exe code/qwen7b_quant_backend.py \
    --model E:/hf_cache/Qwen3-ASR-7B --quant int4 \
    --slice-dir E:/target_slices_full --out code/runs/_qwen7b_int4_uid2text.json
抽样先验(60 条主战场桶避免空跑):
  ... --limit 60  (切片已按 cmd_X 排序, 非按桶; 用 separate 抽样脚本控制)
"""
import os, json, glob, argparse, time, random
import torch


def build_quant_config(quant: str):
    """bf16=None(原行为); int4=bnb NF4 double-quant bf16 compute."""
    if quant == "bf16":
        return None
    if quant == "int4":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NF4: 针对正态分布权重最优(Qwen/LLaMA 系标准)
            bnb_4bit_compute_dtype=torch.bfloat16,  # 反量化计算用 bf16(4060 Ada 原生支持)
            bnb_4bit_use_double_quant=True,     # 二次量化再省 ~0.4bit/param 显存
            llm_int8_skip_modules=None,         # 不跳层(全量化; audio encoder 也量, 显存优先)
        )
    raise ValueError(f"unknown quant={quant}, expect bf16|int4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="E:/target_slices_full",
                    help="target 切片 wav 目录(cmd_X 命名, 与 baseline poc_qwen_asr_full_result 对齐)")
    ap.add_argument("--model", default=os.environ.get("MODEL_QWEN3_ASR_7B",
                    r"E:/hf_cache/Qwen3-ASR-7B"),
                    help="7B 权重目录(env MODEL_QWEN3_ASR_7B 覆盖)")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument("--quant", choices=["bf16", "int4"], default="int4",
                    help="量化模式: int4(NF4, 8GB 能跑) / bf16(14GB, 仅 L20)")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="batch 推理(量化+7B 显存敏感, 默认 1 稳妥; L20 可调大)")
    ap.add_argument("--context", default="", help="transcribe context(默认空=不引导)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    try:
        import numpy as np; np.random.seed(args.seed)
    except ImportError:
        pass
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from qwen_asr import Qwen3ASRModel

    quant_cfg = build_quant_config(args.quant)
    load_kwargs = dict(device_map="cuda:0", dtype=torch.bfloat16)
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
        print(f"[load] {args.model} quant=int4(NF4/dbl/bf16-compute) ...")
    else:
        print(f"[load] {args.model} bf16 ...")
    t_load0 = time.time()
    model = Qwen3ASRModel.from_pretrained(args.model, **load_kwargs)
    load_time = time.time() - t_load0
    torch.cuda.synchronize()
    mem_after_load = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"[load] done {load_time:.1f}s, peak_mem={mem_after_load:.0f}MB")

    slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
    if args.limit:
        slices = slices[:args.limit]
    uids = [os.path.splitext(os.path.basename(sf))[0] for sf in slices]
    print(f"{len(slices)} 切片, quant={args.quant}, batch={args.batch_size}")

    uid2text = {}
    per_utt_sec = []   # 每条纯推理秒数(不含加载)
    t0 = time.time()

    def transcribe_one(sf):
        # 逐条: 用 soundfile 读 wav → model.transcribe(audio=np_array)
        import soundfile as sf_mod
        wav, sr = sf_mod.read(sf)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)   # 多通道→单通道
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav.astype("float32"), orig_sr=sr, target_sr=16000)
        return model.transcribe(audio=wav, language="Chinese", context=args.context)

    if args.batch_size <= 1:
        for i, sf in enumerate(slices):
            uid = uids[i]
            ts = time.time()
            try:
                res = transcribe_one(sf)
                uid2text[uid] = res[0].text.strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {uid} OOM, skip → ''")
                uid2text[uid] = ""
            except Exception as e:
                print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:60]}")
                uid2text[uid] = ""
            torch.cuda.synchronize()
            per_utt_sec.append(time.time() - ts)
            if (i + 1) % 50 == 0 or (i + 1) == len(slices):
                el = time.time() - t0
                avg = sum(per_utt_sec) / len(per_utt_sec)
                print(f"  [{i+1}/{len(slices)}] avg_infer={avg:.3f}s/utt peak_mem={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB")
    else:
        # batch 模式(复用 qwen_asr_backend 逻辑, 量化时显存风险高, 仅 L20 用)
        bs = args.batch_size
        for bi in range(0, len(slices), bs):
            batch_paths = slices[bi:bi+bs]; batch_uids = uids[bi:bi+bs]
            ts = time.time()
            try:
                results = model.transcribe(audio=batch_paths, language="Chinese", context=args.context)
                for uid, res in zip(batch_uids, results):
                    uid2text[uid] = res.text.strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  batch OOM @{bi}, fallback 逐条")
                for uid, sp in zip(batch_uids, batch_paths):
                    try:
                        res = transcribe_one(sp); uid2text[uid] = res[0].text.strip()
                    except Exception as e:
                        uid2text[uid] = ""
            except Exception as e:
                print(f"  batch @{bi} FAIL {type(e).__name__}: {str(e)[:60]}")
                for uid in batch_uids:
                    uid2text[uid] = ""
            torch.cuda.synchronize()
            per_utt_sec.append((time.time() - ts) / len(batch_paths))
            done = min(bi + bs, len(slices))
            print(f"  [{done}/{len(slices)}] ({done/(time.time()-t0):.1f}/s)")

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_infer = time.time() - t0
    avg_utt = sum(per_utt_sec) / len(per_utt_sec) if per_utt_sec else 0.0

    out = {
        "model": args.model, "quant": args.quant, "n": len(uid2text),
        "load_time_s": round(load_time, 2),
        "total_infer_s": round(total_infer, 1),
        "avg_infer_s_per_utt": round(avg_utt, 4),
        "peak_gpu_mem_mb": round(peak_mem, 0),
        "batch_size": args.batch_size,
        "uid2text": uid2text,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[DONE] {len(uid2text)} 条 → {args.out}")
    print(f"  quant={args.quant} avg_infer={avg_utt:.3f}s/utt peak_mem={peak_mem:.0f}MB "
          f"load={load_time:.1f}s total_infer={total_infer:.0f}s")
    print(f"  vs baseline 1.7B bf16: RTF=0.095s/utt mem=3893MB CER=0.3436")


if __name__ == "__main__":
    main()
