#!/usr/bin/env python
"""int8 量化 (bitsandbytes) 加速效果测试。

测试 Whisper-large-v3-turbo 和 Qwen3-ASR-1.7B 的 int8 量化:
  - 加载速度
  - 推理 RTF
  - CER 是否退化
  - 显存占用

用法:
  code/.venv/Scripts/python.exe code/exp_int8_quant.py --backend vanilla --num-samples 5
  code/.venv_qwen/Scripts/python.exe code/exp_int8_quant.py --backend qwen --num-samples 5
"""
import os, sys, json, time, gc, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vanilla", choices=["vanilla", "qwen"])
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

def gpu_mem():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0

def test_vanilla_int8(args, audio_paths, refs):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

    model_path = os.environ.get("MODEL_VANILLA", "openai/whisper-large-v3-turbo")
    device = torch.device(args.device)
    results_all = []

    # ---- A. Baseline fp16 ----
    print("\n=== [Whisper] A. Baseline fp16 ===")
    gc.collect(); torch.cuda.empty_cache()
    mem_before = gpu_mem()
    t0 = time.perf_counter()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float16).to(device).eval()
    load_time = time.perf_counter() - t0
    mem_after = gpu_mem()
    print(f"  加载耗时: {load_time:.1f}s, 显存: {mem_after:.0f}MB (Δ{mem_after-mem_before:.0f}MB)")
    tok = AutoTokenizer.from_pretrained(model_path)
    fe = AutoFeatureExtractor.from_pretrained(model_path)
    # Warmup
    audio, _ = __import__("librosa").load(audio_paths[0], sr=16000)
    ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
    am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
    with torch.no_grad():
        model.generate(input_features=ifp, attention_mask=am, language="zh", task="transcribe", max_new_tokens=200)
    torch.cuda.synchronize()
    # Measure
    rtfs, texts = [], []
    for wp in audio_paths:
        audio, sr = __import__("librosa").load(wp, sr=16000)
        dur = len(audio) / sr
        ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_features=ifp, attention_mask=am, language="zh", task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize()
        rtf = (time.perf_counter() - t0) / dur
        seqs = out["sequences"] if isinstance(out, dict) else out
        text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        rtfs.append(rtf)
        texts.append(text)
        print(f"    {os.path.basename(wp)}: RTF={rtf:.4f}")
    avg_rtf = np.mean(rtfs)
    cers = [compute_cer(r, h) for r, h in zip(refs, texts)]
    print(f"  avg_RTF={avg_rtf:.4f} avg_CER={np.mean(cers):.4f}")
    results_all.append({"label": "whisper_fp16", "avg_rtf": avg_rtf, "avg_cer": np.mean(cers),
                         "load_time": load_time, "gpu_mem_mb": mem_after, "texts": texts})
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B. int8 量化 ----
    print("\n=== [Whisper] B. int8 量化 (bitsandbytes) ===")
    gc.collect(); torch.cuda.empty_cache()
    mem_before = gpu_mem()
    model_int8 = None
    t0 = time.perf_counter()
    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model_int8 = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, quantization_config=quant_config, device_map="auto")
        load_time = time.perf_counter() - t0
        mem_after = gpu_mem()
        print(f"  加载耗时: {load_time:.1f}s, 显存: {mem_after:.0f}MB (Δ{mem_after-mem_before:.0f}MB)")
        tok = AutoTokenizer.from_pretrained(model_path)
        fe = AutoFeatureExtractor.from_pretrained(model_path)
        # int8 model 已经在 device 上, 通过 device_map 分配, input 也送上同一 device
        model_device = next(model_int8.parameters()).device
        # Warmup
        audio, _ = __import__("librosa").load(audio_paths[0], sr=16000)
        ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model_device, torch.float16)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=model_device)
        with torch.no_grad():
            model_int8.generate(input_features=ifp, attention_mask=am, language="zh", task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize()
        # Measure
        rtfs, texts = [], []
        for wp in audio_paths:
            audio, sr = __import__("librosa").load(wp, sr=16000)
            dur = len(audio) / sr
            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(model_device, torch.float16)
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=model_device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model_int8.generate(input_features=ifp, attention_mask=am, language="zh", task="transcribe", max_new_tokens=200)
            torch.cuda.synchronize()
            rtf = (time.perf_counter() - t0) / dur
            seqs = out["sequences"] if isinstance(out, dict) else out
            text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
            rtfs.append(rtf)
            texts.append(text)
            print(f"    {os.path.basename(wp)}: RTF={rtf:.4f}")
        avg_rtf = np.mean(rtfs)
        cers = [compute_cer(r, h) for r, h in zip(refs, texts)]
        print(f"  avg_RTF={avg_rtf:.4f} avg_CER={np.mean(cers):.4f}")
        results_all.append({"label": "whisper_int8", "avg_rtf": avg_rtf, "avg_cer": np.mean(cers),
                             "load_time": load_time, "gpu_mem_mb": mem_after, "texts": texts})
    except Exception as e:
        print(f"  int8 加载失败: {type(e).__name__}: {e}")
        results_all.append({"label": "whisper_int8", "error": str(e)})
    finally:
        if model_int8 is not None:
            del model_int8
        gc.collect(); torch.cuda.empty_cache()

    return results_all


def test_qwen_int8(args, audio_paths, refs):
    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = os.environ.get("MODEL_QWEN3_ASR",
                                 "E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt"
                                 else "/root/hf_cache/Qwen3-ASR-1.7B")
    results_all = []

    # ---- A. Baseline bf16 ----
    print("\n=== [Qwen3-ASR] A. Baseline bf16 ===")
    gc.collect(); torch.cuda.empty_cache()
    mem_before = gpu_mem()
    t0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    load_time = time.perf_counter() - t0
    mem_after = gpu_mem()
    print(f"  加载耗时: {load_time:.1f}s, 显存: {mem_after:.0f}MB (Δ{mem_after-mem_before:.0f}MB)")
    # Warmup
    model.transcribe(audio=audio_paths[0], language="Chinese")
    torch.cuda.synchronize()
    # Measure
    rtfs, texts = [], []
    for wp in audio_paths:
        audio, sr = __import__("librosa").load(wp, sr=16000)
        dur = len(audio) / sr
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = model.transcribe(audio=wp, language="Chinese")
        torch.cuda.synchronize()
        rtf = (time.perf_counter() - t0) / dur
        text = res[0].text.strip() if res else ""
        rtfs.append(rtf)
        texts.append(text)
        print(f"    {os.path.basename(wp)}: RTF={rtf:.4f}")
    avg_rtf = np.mean(rtfs)
    cers = [compute_cer(r, h) for r, h in zip(refs, texts)]
    print(f"  avg_RTF={avg_rtf:.4f} avg_CER={np.mean(cers):.4f}")
    results_all.append({"label": "qwen_bf16", "avg_rtf": avg_rtf, "avg_cer": np.mean(cers),
                         "load_time": load_time, "gpu_mem_mb": mem_after, "texts": texts})
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B. int8 量化 ----
    print("\n=== [Qwen3-ASR] B. int8 量化 (bitsandbytes) ===")
    gc.collect(); torch.cuda.empty_cache()
    mem_before = gpu_mem()
    model_int8 = None
    t0 = time.perf_counter()
    try:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model_int8 = Qwen3ASRModel.from_pretrained(model_path, quantization_config=quant_config, device_map="auto")
        load_time = time.perf_counter() - t0
        mem_after = gpu_mem()
        print(f"  加载耗时: {load_time:.1f}s, 显存: {mem_after:.0f}MB (Δ{mem_after-mem_before:.0f}MB)")
        # Warmup
        model_int8.transcribe(audio=audio_paths[0], language="Chinese")
        torch.cuda.synchronize()
        # Measure
        rtfs, texts = [], []
        for wp in audio_paths:
            audio, sr = __import__("librosa").load(wp, sr=16000)
            dur = len(audio) / sr
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = model_int8.transcribe(audio=wp, language="Chinese")
            torch.cuda.synchronize()
            rtf = (time.perf_counter() - t0) / dur
            text = res[0].text.strip() if res else ""
            rtfs.append(rtf)
            texts.append(text)
            print(f"    {os.path.basename(wp)}: RTF={rtf:.4f}")
        avg_rtf = np.mean(rtfs)
        cers = [compute_cer(r, h) for r, h in zip(refs, texts)]
        print(f"  avg_RTF={avg_rtf:.4f} avg_CER={np.mean(cers):.4f}")
        results_all.append({"label": "qwen_int8", "avg_rtf": avg_rtf, "avg_cer": np.mean(cers),
                             "load_time": load_time, "gpu_mem_mb": mem_after, "texts": texts})
    except Exception as e:
        print(f"  int8 加载失败: {type(e).__name__}: {e}")
        results_all.append({"label": "qwen_int8", "error": str(e)})
    finally:
        if model_int8 is not None:
            del model_int8
        gc.collect(); torch.cuda.empty_cache()

    return results_all


def main():
    args = parse_args()
    set_seed(args.seed)

    pairs_path = os.path.join(_HERE, "pos_pairs_datasetA.json")
    test_data = load_test_audio(pairs_path, args.num_samples, args.seed)
    audio_paths = [d[0] for d in test_data]
    refs = [d[1] for d in test_data]
    print(f"测试音频: {len(audio_paths)} 条")

    if args.backend == "vanilla":
        all_results = test_vanilla_int8(args, audio_paths, refs)
    else:
        all_results = test_qwen_int8(args, audio_paths, refs)

    # Summary
    print("\n" + "=" * 60)
    print(f"=== {args.backend.upper()} int8 量化效果汇总 ===")
    print("=" * 60)
    baseline = None
    for r in all_results:
        label = r.get("label", "?")
        if "error" in r:
            print(f"  {label}: 失败 ({r['error'][:80]})")
            continue
        avg_rtf = r["avg_rtf"]
        avg_cer = r["avg_cer"]
        load_t = r["load_time"]
        mem = r["gpu_mem_mb"]
        if baseline is None:
            baseline = avg_rtf
            print(f"  {label}: RTF={avg_rtf:.4f} CER={avg_cer:.4f} load={load_t:.1f}s mem={mem:.0f}MB (baseline)")
        else:
            delta = (avg_rtf - baseline) / baseline * 100
            speedup = baseline / avg_rtf if avg_rtf > 0 else float("inf")
            print(f"  {label}: RTF={avg_rtf:.4f} (Δ{delta:+.1f}%, {speedup:.2f}x) CER={avg_cer:.4f} load={load_t:.1f}s mem={mem:.0f}MB")

    out_path = os.path.join(_HERE, f"exp_int8_{args.backend}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

if __name__ == "__main__":
    main()
