#!/usr/bin/env python
"""torch.compile / Flash Attention / 量化 加速效果 POC 测试。

测试对象:
  1. Whisper-large-v3-turbo (vanilla 后端, HF transformers)
  2. Qwen3-ASR-1.7B (qwen 后端, qwen_asr 包)

测试项目:
  A. Baseline (当前默认, SDPA attention)
  B. torch.compile(mode="reduce-overhead")
  C. Flash Attention 2 (需 flash-attn 包)
  D. int8 量化 (需 bitsandbytes)

每个测试: 跑 N 条音频, 记录 RTF + 首次编译耗时 + CER(与 baseline 对比)

用法:
  uv run code/exp_torch_compile.py --backend vanilla --num-samples 5
  uv run code/exp_torch_compile.py --backend qwen --num-samples 5
"""
import os, sys, json, time, argparse, gc
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---- 解析参数 ----
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="vanilla", choices=["vanilla", "qwen"])
    ap.add_argument("--num-samples", type=int, default=5, help="每项测试跑几条音频")
    ap.add_argument("--warmup", type=int, default=1, help="torch.compile 预热条数")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-compile", action="store_true", help="跳过 torch.compile 测试")
    ap.add_argument("--skip-flash", action="store_true", help="跳过 flash attention 测试")
    ap.add_argument("--skip-int8", action="store_true", help="跳过 int8 量化测试")
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
    """从 pairs json 加载测试音频路径和参考文本。"""
    pairs = json.load(open(pairs_path, encoding="utf-8"))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)
    return [(pairs[i]["recognition"], pairs[i].get("ref", "")) for i in sorted(indices)]


def measure_rtf_vanilla(model, fe, tok, audio_paths, device, dtype, language="zh"):
    """测量 vanilla Whisper 的 RTF。返回 (rtf_list, texts_list, compile_time)。"""
    import torch, librosa
    results = []
    texts = []
    for wav_path in audio_paths:
        audio, sr = librosa.load(wav_path, sr=16000)
        audio_dur = len(audio) / sr
        ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(input_features=ifp, attention_mask=am,
                                 language=language, task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        infer_time = t1 - t0
        rtf = infer_time / audio_dur
        seqs = out["sequences"] if isinstance(out, dict) else out
        text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        results.append({"rtf": rtf, "infer_time": infer_time, "audio_dur": audio_dur})
        texts.append(text)
        print(f"    {os.path.basename(wav_path)}: {audio_dur:.1f}s, {infer_time:.2f}s, RTF={rtf:.4f}")
    return results, texts, 0.0


def measure_rtf_qwen(model, audio_paths, language="Chinese"):
    """测量 Qwen3-ASR 的 RTF。"""
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
    return results, texts, 0.0


def compute_cer(ref, hyp):
    """简单 CER (editdistance)。"""
    try:
        import editdistance
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
        return editdistance.eval(ref_chars, hyp_chars) / len(ref_chars)
    except ImportError:
        return -1.0


def summarize(label, results, texts, refs, compile_time=0.0):
    """汇总一组测试结果。"""
    rtfs = [r["rtf"] for r in results]
    avg_rtf = np.mean(rtfs)
    med_rtf = np.median(rtfs)
    cers = [compute_cer(r, h) for r, h in zip(refs, texts)] if refs else []
    avg_cer = np.mean(cers) if cers else -1.0
    print(f"\n  [{label}] avg_RTF={avg_rtf:.4f} median_RTF={med_rtf:.4f}")
    if cers:
        print(f"    avg_CER={avg_cer:.4f} (与 baseline 对比文本差异)")
    if compile_time > 0:
        print(f"    首次编译耗时: {compile_time:.1f}s")
    return {"label": label, "avg_rtf": avg_rtf, "median_rtf": med_rtf,
            "avg_cer": avg_cer, "compile_time": compile_time,
            "details": results, "texts": texts}


# ---- Vanilla Whisper 测试 ----
def test_vanilla(args, audio_paths, refs):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

    model_path = os.environ.get("MODEL_VANILLA", "openai/whisper-large-v3-turbo")
    device = torch.device(args.device)
    dtype = torch.float16
    all_results = []

    # ---- A. Baseline (SDPA, 当前默认) ----
    print("\n=== [Vanilla] A. Baseline (SDPA attention) ===")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    fe = AutoFeatureExtractor.from_pretrained(model_path)
    # 打印实际 attention 类型
    for name, module in model.named_modules():
        if "self_attn" in name:
            print(f"  attention class: {module.__class__.__name__}")
            break
    # CUDA warmup: 避免首次推理的 CUDA 初始化开销污染 baseline
    print("  CUDA warmup 1 条...")
    _wp = audio_paths[0]
    _wa, _ = __import__("librosa").load(_wp, sr=16000)
    _wifp = fe(_wa, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    _wam = torch.ones(1, _wifp.shape[-1], dtype=torch.bool, device=device)
    with torch.no_grad():
        model.generate(input_features=_wifp, attention_mask=_wam,
                       language="zh", task="transcribe", max_new_tokens=200)
    torch.cuda.synchronize()
    print("  warmup 完成, 开始测量...")
    results, texts, _ = measure_rtf_vanilla(model, fe, tok, audio_paths, device, dtype)
    all_results.append(summarize("baseline_SDPA", results, texts, refs))
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B. torch.compile ----
    if not args.skip_compile:
        print("\n=== [Vanilla] B. torch.compile(mode='reduce-overhead') ===")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_path)
        fe = AutoFeatureExtractor.from_pretrained(model_path)
        # 预热
        print(f"  预热 {args.warmup} 条...")
        warmup_paths = audio_paths[:args.warmup]
        for wp in warmup_paths:
            audio, _ = __import__("librosa").load(wp, sr=16000)
            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                model.generate(input_features=ifp, attention_mask=am,
                               language="zh", task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize()
        print("  编译中 (首次可能很慢)...")
        t_compile_start = time.perf_counter()
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            # 触发编译: 跑一条
            audio, _ = __import__("librosa").load(audio_paths[0], sr=16000)
            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                compiled_model.generate(input_features=ifp, attention_mask=am,
                                        language="zh", task="transcribe", max_new_tokens=200)
            torch.cuda.synchronize()
            compile_time = time.perf_counter() - t_compile_start
            print(f"  编译完成: {compile_time:.1f}s")
            results, texts, _ = measure_rtf_vanilla(compiled_model, fe, tok, audio_paths, device, dtype)
            all_results.append(summarize("torch_compile_reduce_overhead", results, texts, refs, compile_time))
        except Exception as e:
            print(f"  torch.compile 失败: {type(e).__name__}: {e}")
            all_results.append({"label": "torch_compile_reduce_overhead", "error": str(e)})
        del model; gc.collect(); torch.cuda.empty_cache()

        # 也测试 mode="max-autotune" (更激进)
        print("\n=== [Vanilla] B2. torch.compile(mode='max-autotune') ===")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_path)
        fe = AutoFeatureExtractor.from_pretrained(model_path)
        print(f"  预热 {args.warmup} 条...")
        for wp in audio_paths[:args.warmup]:
            audio, _ = __import__("librosa").load(wp, sr=16000)
            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                model.generate(input_features=ifp, attention_mask=am,
                               language="zh", task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize()
        t_compile_start = time.perf_counter()
        try:
            compiled_model = torch.compile(model, mode="max-autotune")
            audio, _ = __import__("librosa").load(audio_paths[0], sr=16000)
            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                compiled_model.generate(input_features=ifp, attention_mask=am,
                                        language="zh", task="transcribe", max_new_tokens=200)
            torch.cuda.synchronize()
            compile_time = time.perf_counter() - t_compile_start
            print(f"  编译完成: {compile_time:.1f}s")
            results, texts, _ = measure_rtf_vanilla(compiled_model, fe, tok, audio_paths, device, dtype)
            all_results.append(summarize("torch_compile_max_autotune", results, texts, refs, compile_time))
        except Exception as e:
            print(f"  torch.compile(max-autotune) 失败: {type(e).__name__}: {e}")
            all_results.append({"label": "torch_compile_max_autotune", "error": str(e)})
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- C. Flash Attention 2 ----
    if not args.skip_flash:
        print("\n=== [Vanilla] C. Flash Attention 2 / SDPA 显式指定 ===")
        # 检查 flash-attn
        has_flash = False
        try:
            import flash_attn
            has_flash = True
            print(f"  flash-attn 版本: {flash_attn.__version__}")
        except ImportError:
            print("  flash-attn 未安装, 跳过 flash_attention_2 测试")

        if has_flash:
            # transformers 4.42 不支持 attn_implementation 参数, 尝试手动替换
            print("  尝试手动加载 whisper + flash_attention_2...")
            try:
                from transformers import WhisperForConditionalGeneration, WhisperConfig
                config = WhisperConfig.from_pretrained(model_path)
                config._attn_implementation = "flash_attention_2"
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_path, config=config, torch_dtype=dtype).to(device).eval()
                tok = AutoTokenizer.from_pretrained(model_path)
                fe = AutoFeatureExtractor.from_pretrained(model_path)
                for name, module in model.named_modules():
                    if "self_attn" in name:
                        print(f"  attention class: {module.__class__.__name__}")
                        break
                results, texts, _ = measure_rtf_vanilla(model, fe, tok, audio_paths, device, dtype)
                all_results.append(summarize("flash_attention_2", results, texts, refs))
            except Exception as e:
                print(f"  flash_attention_2 加载失败: {type(e).__name__}: {e}")
                all_results.append({"label": "flash_attention_2", "error": str(e)})
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
        else:
            all_results.append({"label": "flash_attention_2", "error": "flash-attn not installed"})

    # ---- D. int8 量化 ----
    if not args.skip_int8:
        print("\n=== [Vanilla] D. int8 量化 (bitsandbytes) ===")
        has_bnb = False
        try:
            import bitsandbytes
            has_bnb = True
            print(f"  bitsandbytes 版本: {bitsandbytes.__version__}")
        except ImportError:
            print("  bitsandbytes 未安装, 跳过 int8 测试")

        if has_bnb:
            try:
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_path, load_in_8bit=True, device_map="auto")
                tok = AutoTokenizer.from_pretrained(model_path)
                fe = AutoFeatureExtractor.from_pretrained(model_path)
                print(f"  模型加载完成 (int8)")
                # 注意: int8 模型不能直接 .to(device), 用 device_map
                results, texts, _ = measure_rtf_vanilla(model, fe, tok, audio_paths, device, dtype)
                all_results.append(summarize("int8_bnb", results, texts, refs))
            except Exception as e:
                print(f"  int8 量化失败: {type(e).__name__}: {e}")
                all_results.append({"label": "int8_bnb", "error": str(e)})
            finally:
                del model; gc.collect(); torch.cuda.empty_cache()
        else:
            all_results.append({"label": "int8_bnb", "error": "bitsandbytes not installed"})

    return all_results


# ---- Qwen3-ASR 测试 ----
def test_qwen(args, audio_paths, refs):
    """Qwen3-ASR 测试。注意: qwen_asr 包在 .venv_qwen, 需要 sys.path 操作。"""
    import torch

    # qwen_asr 在 .venv_qwen 的 site-packages 里
    venv_qwen = os.path.join(_HERE, ".venv_qwen")
    qwen_site = os.path.join(venv_qwen, "Lib", "site-packages")
    if os.path.isdir(qwen_site) and qwen_site not in sys.path:
        sys.path.insert(0, qwen_site)

    model_path = os.environ.get("MODEL_QWEN3_ASR",
                                 "E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt"
                                 else "/root/hf_cache/Qwen3-ASR-1.7B")
    device = torch.device(args.device)
    all_results = []

    # ---- A. Baseline ----
    print("\n=== [Qwen3-ASR] A. Baseline ===")
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as e:
        print(f"  qwen_asr 包不可用: {e}")
        print(f"  请确认 .venv_qwen 存在且包含 qwen_asr 包")
        return [{"label": "qwen_baseline", "error": f"qwen_asr not available: {e}"}]

    model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map=str(device))
    # 确认 attention 类型
    for name, module in model.model.named_modules():
        if "self_attn" in name:
            print(f"  attention class: {module.__class__.__name__}")
            break
    results, texts, _ = measure_rtf_qwen(model, audio_paths)
    all_results.append(summarize("qwen_baseline", results, texts, refs))
    del model; gc.collect(); torch.cuda.empty_cache()

    # ---- B. torch.compile ----
    if not args.skip_compile:
        print("\n=== [Qwen3-ASR] B. torch.compile(mode='reduce-overhead') ===")
        model = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.bfloat16, device_map=str(device))
        # 预热
        print(f"  预热 {args.warmup} 条...")
        for wp in audio_paths[:args.warmup]:
            model.transcribe(audio=wp, language="Chinese")
        torch.cuda.synchronize()
        t_compile_start = time.perf_counter()
        try:
            # Qwen3ASRModel 内部有 model 属性, 尝试编译内部模型
            if hasattr(model, 'model'):
                model.model = torch.compile(model.model, mode="reduce-overhead")
            elif hasattr(model, '_model'):
                model._model = torch.compile(model._model, mode="reduce-overhead")
            else:
                # 尝试编译整个对象
                model = torch.compile(model, mode="reduce-overhead")
            # 触发编译
            model.transcribe(audio=audio_paths[0], language="Chinese")
            torch.cuda.synchronize()
            compile_time = time.perf_counter() - t_compile_start
            print(f"  编译完成: {compile_time:.1f}s")
            results, texts, _ = measure_rtf_qwen(model, audio_paths)
            all_results.append(summarize("qwen_torch_compile", results, texts, refs, compile_time))
        except Exception as e:
            print(f"  torch.compile 失败: {type(e).__name__}: {e}")
            all_results.append({"label": "qwen_torch_compile", "error": str(e)})
        del model; gc.collect(); torch.cuda.empty_cache()

    # ---- C. Flash Attention ----
    if not args.skip_flash:
        print("\n=== [Qwen3-ASR] C. Flash Attention 2 ===")
        has_flash = False
        try:
            import flash_attn
            has_flash = True
            print(f"  flash-attn 版本: {flash_attn.__version__}")
        except ImportError:
            print("  flash-attn 未安装, 跳过")
            all_results.append({"label": "qwen_flash_attention_2", "error": "flash-attn not installed"})

        if has_flash:
            try:
                from transformers import AutoModelForCausalLM, AutoConfig
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                config._attn_implementation = "flash_attention_2"
                # 直接加载 transformers 模型(不用 qwen_asr 包装)
                raw_model = AutoModelForCausalLM.from_pretrained(
                    model_path, config=config, torch_dtype=torch.bfloat16,
                    device_map=str(device), trust_remote_code=True)
                for name, module in raw_model.named_modules():
                    if "self_attn" in name:
                        print(f"  attention class: {module.__class__.__name__}")
                        break
                # 需要用 raw model 做推理, 但这需要完整的 qwen_asr pipeline
                # 简单起见, 检查 qwen_asr 包是否支持 attn_implementation
                print("  注意: qwen_asr 包封装了完整 pipeline, flash-attn 需要底层模型支持")
                print("  需要修改 qwen_asr 包或直接用 transformers 推理")
                all_results.append({"label": "qwen_flash_attention_2",
                                    "error": "需要修改 qwen_asr 包传入 attn_implementation"})
                del raw_model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  flash_attention_2 失败: {type(e).__name__}: {e}")
                all_results.append({"label": "qwen_flash_attention_2", "error": str(e)})

    # ---- D. int8 量化 ----
    if not args.skip_int8:
        print("\n=== [Qwen3-ASR] D. int8 量化 ===")
        has_bnb = False
        try:
            import bitsandbytes
            has_bnb = True
        except ImportError:
            print("  bitsandbytes 未安装, 跳过")
            all_results.append({"label": "qwen_int8", "error": "bitsandbytes not installed"})

        if has_bnb:
            try:
                from transformers import AutoModelForCausalLM
                raw_model = AutoModelForCausalLM.from_pretrained(
                    model_path, load_in_8bit=True, device_map="auto",
                    trust_remote_code=True)
                print("  int8 加载成功, 但需要 qwen_asr pipeline 配合")
                all_results.append({"label": "qwen_int8",
                                    "error": "需要修改 qwen_asr 包支持 int8"})
                del raw_model; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"  int8 失败: {type(e).__name__}: {e}")
                all_results.append({"label": "qwen_int8", "error": str(e)})

    return all_results


def main():
    args = parse_args()
    set_seed(args.seed)

    # 加载测试音频
    pairs_path = os.path.join(_HERE, "pos_pairs_datasetA.json")
    if not os.path.exists(pairs_path):
        print(f"错误: {pairs_path} 不存在")
        sys.exit(1)

    test_data = load_test_audio(pairs_path, args.num_samples, args.seed)
    audio_paths = [d[0] for d in test_data]
    refs = [d[1] for d in test_data]
    print(f"测试音频: {len(audio_paths)} 条")
    for i, (p, r) in enumerate(test_data):
        print(f"  [{i}] {os.path.basename(p)} ref={r[:30]}...")

    if args.backend == "vanilla":
        all_results = test_vanilla(args, audio_paths, refs)
    else:
        all_results = test_qwen(args, audio_paths, refs)

    # 汇总
    print("\n" + "=" * 60)
    print(f"=== {args.backend.upper()} 优化效果汇总 ===")
    print("=" * 60)
    baseline_rtf = None
    for r in all_results:
        label = r.get("label", "?")
        if "error" in r:
            print(f"  {label}: 跳过 ({r['error'][:60]})")
            continue
        avg_rtf = r.get("avg_rtf", 0)
        avg_cer = r.get("avg_cer", -1)
        compile_time = r.get("compile_time", 0)
        if baseline_rtf is None:
            baseline_rtf = avg_rtf
            print(f"  {label}: RTF={avg_rtf:.4f} (baseline)")
        else:
            delta_pct = (avg_rtf - baseline_rtf) / baseline_rtf * 100
            speedup = baseline_rtf / avg_rtf if avg_rtf > 0 else float("inf")
            cer_note = f" CER={avg_cer:.4f}" if avg_cer >= 0 else ""
            compile_note = f" compile={compile_time:.0f}s" if compile_time > 0 else ""
            print(f"  {label}: RTF={avg_rtf:.4f} (Δ{delta_pct:+.1f}%, {speedup:.2f}x){cer_note}{compile_note}")

    # 保存结果
    out_path = os.path.join(_HERE, f"exp_torch_compile_{args.backend}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "num_samples": args.num_samples,
                    "results": all_results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
