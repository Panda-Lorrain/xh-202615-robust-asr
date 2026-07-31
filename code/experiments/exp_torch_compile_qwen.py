#!/usr/bin/env python
"""Qwen3-ASR torch.compile 加速效果测试 (在 .venv_qwen 中运行)。

用法:
  code/.venv_qwen/Scripts/python.exe code/exp_torch_compile_qwen.py --num-samples 5
"""
import os, sys, json, time, gc, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def set_seed(seed):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_test_audio(pairs_path, num_samples, seed=42):
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)
    return [(pairs[i]["recognition"], pairs[i].get("ref", "")) for i in sorted(indices)]

def compute_cer(ref, hyp):
    try:
        import editdistance
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
        return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)
    except ImportError:
        return -1.0

def measure_rtf(model, audio_paths, language="Chinese"):
    import torch, librosa
    results = []
    texts = []
    for wav_path in audio_paths:
        audio, sr = librosa.load(wav_path, sr=16000)
        audio_dur = len(audio) / sr
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.transcribe(audio=wav_path, language=language)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        infer_time = t1 - t0
        rtf = infer_time / audio_dur
        text = res[0].text.strip() if res else ""
        results.append({"rtf": rtf, "infer_time": infer_time, "audio_dur": audio_dur})
        texts.append(text)
        print(f"    {os.path.basename(wav_path)}: {audio_dur:.1f}s, {infer_time:.2f}s, RTF={rtf:.4f}")
    return results, texts

def summarize(label, results, texts, refs, compile_time=0.0):
    rtfs = [r["rtf"] for r in results]
    avg_rtf = np.mean(rtfs)
    med_rtf = np.median(rtfs)
    cers = [compute_cer(r, h) for r, h in zip(refs, texts)]
    avg_cer = np.mean(cers)
    print(f"\n  [{label}] avg_RTF={avg_rtf:.4f} median_RTF={med_rtf:.4f} avg_CER={avg_cer:.4f}")
    if compile_time > 0:
        print(f"    首次编译耗时: {compile_time:.1f}s")
    return {"label": label, "avg_rtf": avg_rtf, "median_rtf": med_rtf,
            "avg_cer": avg_cer, "compile_time": compile_time,
            "texts": texts}

def main():
    args = parse_args()
    set_seed(args.seed)
    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = os.environ.get("MODEL_QWEN3_ASR",
                                 "E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt"
                                 else "/root/hf_cache/Qwen3-ASR-1.7B")

    pairs_path = os.path.join(_HERE, "pos_pairs_datasetA.json")
    test_data = load_test_audio(pairs_path, args.num_samples, args.seed)
    audio_paths = [d[0] for d in test_data]
    refs = [d[1] for d in test_data]
    print(f"测试音频: {len(audio_paths)} 条")
    for i, (p, r) in enumerate(test_data):
        print(f"  [{i}] {os.path.basename(p)} ref={r[:30]}...")

    all_results = []

    # ---- A. Baseline ----
    print("\n=== [Qwen3-ASR] A. Baseline ===")
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    # Check attention type
    for name, module in model.model.named_modules():
        if "self_attn" in name:
            print(f"  attention class: {module.__class__.__name__}")
            break
    # Warmup
    print("  CUDA warmup 1 条...")
    model.transcribe(audio=audio_paths[0], language="Chinese")
    torch.cuda.synchronize()
    print("  warmup 完成, 开始测量...")
    results, texts = measure_rtf(model, audio_paths)
    all_results.append(summarize("qwen_baseline", results, texts, refs))
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B. torch.compile (内部 model) ----
    print("\n=== [Qwen3-ASR] B. torch.compile(model.model, mode='reduce-overhead') ===")
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    # Warmup before compile
    print("  预热 1 条...")
    model.transcribe(audio=audio_paths[0], language="Chinese")
    torch.cuda.synchronize()
    print("  编译 model.model 中...")
    t_compile = time.perf_counter()
    try:
        model.model = torch.compile(model.model, mode="reduce-overhead")
        # Trigger compilation
        model.transcribe(audio=audio_paths[0], language="Chinese")
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t_compile
        print(f"  编译完成: {compile_time:.1f}s")
        results, texts = measure_rtf(model, audio_paths)
        all_results.append(summarize("qwen_compile_reduce", results, texts, refs, compile_time))
    except Exception as e:
        print(f"  torch.compile 失败: {type(e).__name__}: {e}")
        all_results.append({"label": "qwen_compile_reduce", "error": str(e)})
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B2. torch.compile max-autotune ----
    print("\n=== [Qwen3-ASR] B2. torch.compile(mode='max-autotune') ===")
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    print("  预热 1 条...")
    model.transcribe(audio=audio_paths[0], language="Chinese")
    torch.cuda.synchronize()
    print("  编译 model.model 中 (max-autotune)...")
    t_compile = time.perf_counter()
    try:
        model.model = torch.compile(model.model, mode="max-autotune")
        model.transcribe(audio=audio_paths[0], language="Chinese")
        torch.cuda.synchronize()
        compile_time = time.perf_counter() - t_compile
        print(f"  编译完成: {compile_time:.1f}s")
        results, texts = measure_rtf(model, audio_paths)
        all_results.append(summarize("qwen_compile_max", results, texts, refs, compile_time))
    except Exception as e:
        print(f"  torch.compile(max-autotune) 失败: {type(e).__name__}: {e}")
        all_results.append({"label": "qwen_compile_max", "error": str(e)})
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- C. Flash Attention 2 ----
    print("\n=== [Qwen3-ASR] C. Flash Attention 2 ===")
    has_flash = False
    try:
        import flash_attn
        has_flash = True
        print(f"  flash-attn 版本: {flash_attn.__version__}")
    except ImportError:
        print("  flash-attn 未安装")
        all_results.append({"label": "qwen_flash_attn2", "error": "flash-attn not installed"})

    if has_flash:
        try:
            from transformers import AutoModel, AutoProcessor, AutoConfig
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = "flash_attention_2"
            raw_model = AutoModel.from_pretrained(
                model_path, config=config, torch_dtype=torch.bfloat16,
                device_map="cuda:0", trust_remote_code=True)
            for name, module in raw_model.named_modules():
                if "self_attn" in name:
                    print(f"  attention class: {module.__class__.__name__}")
                    break
            print("  flash-attn 模型加载成功, 但需配合 qwen_asr pipeline 测试")
            # Wrap in Qwen3ASRModel
            processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
            from qwen_asr import Qwen3ASRModel
            fa_model = Qwen3ASRModel(backend="transformers", model=raw_model, processor=processor)
            # Warmup
            fa_model.transcribe(audio=audio_paths[0], language="Chinese")
            torch.cuda.synchronize()
            results, texts = measure_rtf(fa_model, audio_paths)
            all_results.append(summarize("qwen_flash_attn2", results, texts, refs))
            del fa_model; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            print(f"  flash_attention_2 失败: {type(e).__name__}: {e}")
            all_results.append({"label": "qwen_flash_attn2", "error": str(e)})

    # ---- D. SDPA 显式 ----
    print("\n=== [Qwen3-ASR] D. SDPA (显式指定, 对比) ===")
    try:
        from transformers import AutoModel, AutoProcessor, AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        raw_model = AutoModel.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16,
            device_map="cuda:0", trust_remote_code=True)
        for name, module in raw_model.named_modules():
            if "self_attn" in name:
                print(f"  attention class: {module.__class__.__name__}")
                break
        processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        from qwen_asr import Qwen3ASRModel
        sdpa_model = Qwen3ASRModel(backend="transformers", model=raw_model, processor=processor)
        sdpa_model.transcribe(audio=audio_paths[0], language="Chinese")
        torch.cuda.synchronize()
        results, texts = measure_rtf(sdpa_model, audio_paths)
        all_results.append(summarize("qwen_sdpa_explicit", results, texts, refs))
        del sdpa_model; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"  SDPA 显式指定失败: {type(e).__name__}: {e}")
        all_results.append({"label": "qwen_sdpa_explicit", "error": str(e)})

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("=== QWEN3-ASR 优化效果汇总 ===")
    print("=" * 60)
    baseline_rtf = None
    for r in all_results:
        label = r.get("label", "?")
        if "error" in r:
            print(f"  {label}: 跳过 ({r['error'][:80]})")
            continue
        avg_rtf = r.get("avg_rtf", 0)
        avg_cer = r.get("avg_cer", -1)
        compile_time = r.get("compile_time", 0)
        if baseline_rtf is None:
            baseline_rtf = avg_rtf
            print(f"  {label}: RTF={avg_rtf:.4f} CER={avg_cer:.4f} (baseline)")
        else:
            delta_pct = (avg_rtf - baseline_rtf) / baseline_rtf * 100
            speedup = baseline_rtf / avg_rtf if avg_rtf > 0 else float("inf")
            compile_note = f" compile={compile_time:.0f}s" if compile_time > 0 else ""
            print(f"  {label}: RTF={avg_rtf:.4f} (Δ{delta_pct:+.1f}%, {speedup:.2f}x) CER={avg_cer:.4f}{compile_note}")

    out_path = os.path.join(_HERE, "exp_torch_compile_qwen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"num_samples": args.num_samples, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
