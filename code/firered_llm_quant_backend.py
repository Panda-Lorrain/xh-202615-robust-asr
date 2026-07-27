#!/usr/bin/env python
"""FireRedASR-LLM-L(1代,8.3B = ASR encoder + Qwen2-7B) int4 量化转写 target 切片。

目的: 对比 Qwen3-ASR-1.7B(CER 0.3436/RTF 0.095) 看 7B 量级中文SOTA能否降CER。
背景: 4060 8GB, FireRedASR-LLM-L=8.3B(Qwen2-7B bf16=14GB 跑不了), 必须 int4 量化 Qwen2-7B decoder。
      ASR encoder/projector 保持 bf16(小), 只量化 LLM(Qwen2-7B)。

量化注入(源码核实, 不改第三方代码, 复用 qwen_asr_backend monkey-patch 模式):
  fireredasr/models/fireredasr_llm.py:56  llm = AutoModelForCausalLM.from_pretrained(llm_dir, ...)
  → patch 该引用, 注入 quantization_config=BitsAndBytesConfig(NF4) + torch_dtype=bf16 + attn=eager(bnb兼容)

用法(int4 全量):
  code/.venv_firered/Scripts/python.exe code/firered_llm_quant_backend.py \
    --model E:/hf_cache/FireRedASR-LLM-L --quant int4 \
    --slice-dir E:/target_slices_full --out code/runs/_firered_llm_int4_uid2text.json
抽样先验(60条验显存+CER趋势):
  ... --limit 60
"""
import os, sys, json, glob, argparse, time, random
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "FireRedASR"))

_DEFAULT = r"E:/hf_cache/FireRedASR-LLM-L"


def patch_llm_quant(quant: str):
    """monkey-patch fireredasr_llm 模块的 AutoModelForCausalLM.from_pretrained 注入量化。
    from_args 调用它加载 Qwen2-7B(line 56), patch 后自动带 quantization_config。"""
    if quant == "bf16":
        return
    import fireredasr.models.fireredasr_llm as llm_mod
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    _orig_fp = llm_mod.AutoModelForCausalLM.from_pretrained

    def _patched_fp(*a, **kw):
        kw["quantization_config"] = bnb_cfg
        kw["torch_dtype"] = torch.bfloat16            # 覆盖 from_args 的 fp16/fp32, bnb compute 用 bf16
        kw["attn_implementation"] = "eager"           # 覆盖 from_args 的 flash2, bnb+eager 最稳
        kw.setdefault("device_map", {"": "cuda:0"})   # 全放 cuda:0(单卡), 避免 auto 跨设备
        return _orig_fp(*a, **kw)

    llm_mod.AutoModelForCausalLM.from_pretrained = _patched_fp
    print(f"[patch] Qwen2-7B → int4 NF4 bf16-compute eager (单卡 cuda:0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", default="E:/target_slices_full",
                    help="target 切片 wav 目录(cmd_X 命名, 与 baseline 对齐)")
    ap.add_argument("--model", default=os.environ.get("MODEL_FIRERED_LLM", _DEFAULT),
                    help="FireRedASR-LLM-L 权重目录(下含 model.pth.tar/asr_encoder.pth.tar/cmvn.ark/Qwen2-7B-Instruct)")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument("--quant", choices=["bf16", "int4"], default="int4",
                    help="量化: int4(NF4, 8GB 能跑) / bf16(8.3B=14GB 跑不了, 仅L20)")
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 条(0=全部)")
    ap.add_argument("--beam-size", type=int, default=1, help="beam(1=greedy 对齐qwen, 3=官方sh)")
    ap.add_argument("--rep-penalty", type=float, default=1.0,
                    help="repetition_penalty(官方sh 3.0; greedy 下 1.0=关)")
    args = ap.parse_args()

    random.seed(42)
    try:
        import numpy as np; np.random.seed(42)
    except ImportError:
        pass
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    patch_llm_quant(args.quant)

    from fireredasr.models.fireredasr import FireRedAsr
    print(f"[load] FireRedASR-LLM-L {args.model} quant={args.quant} ...")
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.time()
    model = FireRedAsr.from_pretrained("llm", args.model)
    load_time = time.time() - t_load0
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"[load] done {load_time:.1f}s peak_mem={mem:.0f}MB")

    slices = sorted(glob.glob(os.path.join(args.slice_dir, "*.wav")),
                    key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]))
    if args.limit:
        slices = slices[:args.limit]
    print(f"{len(slices)} 切片, quant={args.quant}, beam={args.beam_size}, rep={args.rep_penalty}")

    uid2text = {}
    per_utt = []
    t0 = time.time()

    for i, sf in enumerate(slices):
        uid = os.path.splitext(os.path.basename(sf))[0]
        ts = time.time()
        try:
            res = model.transcribe([uid], [sf], {
                "use_gpu": True,
                "beam_size": args.beam_size,
                "decode_max_len": 0, "decode_min_len": 0,
                "repetition_penalty": args.rep_penalty,
                "llm_length_penalty": 0.0, "temperature": 1.0,
            })
            uid2text[uid] = res[0]["text"].strip()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {uid} OOM → ''")
            uid2text[uid] = ""
        except Exception as e:
            print(f"  {uid} FAIL {type(e).__name__}: {str(e)[:70]}")
            uid2text[uid] = ""
        torch.cuda.synchronize()
        per_utt.append(time.time() - ts)
        if (i + 1) % 25 == 0 or (i + 1) == len(slices):
            avg = sum(per_utt) / len(per_utt)
            print(f"  [{i+1}/{len(slices)}] avg_infer={avg:.3f}s/utt peak_mem={torch.cuda.max_memory_allocated()/1024/1024:.0f}MB")

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    total = time.time() - t0
    avg_utt = sum(per_utt) / len(per_utt) if per_utt else 0.0

    out = {
        "model": "FireRedASR-LLM-L", "quant": args.quant, "n": len(uid2text),
        "beam_size": args.beam_size, "rep_penalty": args.rep_penalty,
        "load_time_s": round(load_time, 2),
        "total_infer_s": round(total, 1),
        "avg_infer_s_per_utt": round(avg_utt, 4),
        "peak_gpu_mem_mb": round(peak_mem, 0),
        "uid2text": uid2text,
    }
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[DONE] {len(uid2text)} 条 → {args.out}")
    print(f"  quant={args.quant} avg_infer={avg_utt:.3f}s/utt peak_mem={peak_mem:.0f}MB load={load_time:.1f}s")
    print(f"  vs baseline 1.7B bf16: RTF=0.095s/utt mem=3893MB CER=0.3436")
    print(f"  vs FireRedASR-AED-L(之前测): CER=0.3501(更差)")


if __name__ == "__main__":
    main()
