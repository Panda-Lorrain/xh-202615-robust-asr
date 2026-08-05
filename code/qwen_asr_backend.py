#!/usr/bin/env python
"""Qwen3-ASR 批量转写 target 切片（code/.venv_qwen，venv 隔离）。
供 enroll_infer --asr-backend qwen 末尾 subprocess 调用，填 result text。

用法: code/.venv_qwen/Scripts/python.exe code/qwen_asr_backend.py \
        --slice-dir E:/target_slices_qwen --out code/_qwen_uid2text.json [--limit N]
"""
import os, json, glob, argparse, time, sys
import torch

# 跨平台默认模型路径(原 E:/ 硬编码在 Linux 阻塞): env MODEL_QWEN3_ASR 可覆盖
_DEFAULT_QWEN3 = (r"E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt"
                  else "/root/hf_cache/Qwen3-ASR-1.7B")


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True, help="target 切片 wav 目录(uid 命名)")
    ap.add_argument("--model", default=os.environ.get("MODEL_QWEN3_ASR", _DEFAULT_QWEN3),
                    help="Qwen3-ASR 权重目录(env MODEL_QWEN3_ASR 覆盖)")
    ap.add_argument("--engine", choices=("transformers", "vllm"),
                    default=os.environ.get("QWEN_ENGINE", "transformers"),
                    help="推理引擎；默认 transformers，vLLM 需独立 venv")
    ap.add_argument("--vllm-max-model-len", type=int,
                    default=int(os.environ.get("QWEN_VLLM_MAX_MODEL_LEN", "2048")),
                    help="vLLM max_model_len（仅 --engine vllm）")
    ap.add_argument("--vllm-max-num-seqs", type=int,
                    default=int(os.environ.get("QWEN_VLLM_MAX_NUM_SEQS", "1")),
                    help="vLLM 并发序列上限（仅 --engine vllm）")
    ap.add_argument("--vllm-gpu-memory-utilization", type=float,
                    default=float(os.environ.get("QWEN_VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
                    help="vLLM GPU 显存比例（仅 --engine vllm）")
    ap.add_argument("--vllm-enforce-eager", action=argparse.BooleanOptionalAction,
                    default=_env_bool("QWEN_VLLM_ENFORCE_EAGER", False),
                    help="关闭 vLLM compile/cudagraph（默认启用优化路径）")
    ap.add_argument("--out", required=True, help="uid→text json 输出")
    ap.add_argument(
        "--meta-out",
        help="可选 uid→解码置信度元数据 JSON；仅 Transformers 后端，默认不启用",
    )
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
    ap.add_argument("--bias-phrases-file", default="",
                    help="解码期短语偏置文件(JSON数组或逐行短语)；默认空=关闭")
    ap.add_argument("--bias-strength", type=float, default=0.8,
                    help="top-K 内候选 token 的 logits 加分；仅 --bias-phrases-file 生效")
    ap.add_argument("--bias-top-k", type=int, default=20,
                    help="只偏置当前 logits top-K 内的短语候选，防止从噪声凭空回吐热词")
    args = ap.parse_args()
    if bool(args.paired_slice_dir) != bool(args.paired_out):
        ap.error("--paired-slice-dir and --paired-out must be provided together")
    if args.meta_out and args.engine != "transformers":
        ap.error("--meta-out 当前只支持 Transformers 后端；vLLM 不暴露同等 token scores")
    if args.engine == "vllm" and (
        args.rep_penalty != 1.0
        or args.no_repeat_ngram_size > 0
        or args.bias_phrases_file
    ):
        ap.error("vLLM 引擎暂不支持 Transformers generate 的重复抑制/短语 bias 参数")

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

    _t_load0 = time.time()  # [profile] 桶C'(Qwen3 load): 包 import 冷启动 + from_pretrained(口径注明)
    from qwen_asr import Qwen3ASRModel
    if args.engine == "vllm":
        print(
            f"[load] Qwen3-ASR {args.model} via vLLM "
            f"(max_len={args.vllm_max_model_len}, max_num_seqs={args.vllm_max_num_seqs}, "
            f"gpu_mem={args.vllm_gpu_memory_utilization}, "
            f"eager={args.vllm_enforce_eager}) ..."
        )
        model = Qwen3ASRModel.LLM(
            model=args.model,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
        )
    else:
        print(f"[load] Qwen3-ASR {args.model} bf16 ...")
        model = Qwen3ASRModel.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda:0"
        )
    load_sec = time.time() - _t_load0

    # 解码参数: monkey-patch model.model.generate 注入重复抑制/声学 top-K 锚定短语偏置
    # (transcribe→_infer_asr_transformers→self.model.generate 不暴露这些 kwarg, 包一层注入;
    #  默认值=原greedy行为, 向后兼容; C1/C2 context 实验不受影响)
    _phrase_processor = None
    if args.bias_phrases_file:
        from qwen_phrase_bias import (
            AcousticTopKPhraseBias,
            load_phrases,
            tokenize_phrases,
        )
        phrases = load_phrases(args.bias_phrases_file)
        tokenized = tokenize_phrases(model.processor.tokenizer, phrases)
        _phrase_processor = AcousticTopKPhraseBias(
            tokenized, bias=args.bias_strength, top_k=args.bias_top_k
        )
        print(
            f"[bias] phrases={len(phrases)} tokenized={len(tokenized)} "
            f"strength={args.bias_strength} top_k={args.bias_top_k}"
        )

    if (args.rep_penalty != 1.0 or args.no_repeat_ngram_size > 0
            or _phrase_processor is not None):
        _orig_generate = model.model.generate
        def _patched_generate(*a, **kw):
            kw.setdefault("repetition_penalty", args.rep_penalty)
            kw.setdefault("no_repeat_ngram_size", args.no_repeat_ngram_size)
            if _phrase_processor is not None:
                from transformers import LogitsProcessorList
                current = list(kw.get("logits_processor") or [])
                kw["logits_processor"] = LogitsProcessorList(
                    current + [_phrase_processor]
                )
            return _orig_generate(*a, **kw)
        model.model.generate = _patched_generate
        print(
            f"[patch] generate rep_penalty={args.rep_penalty} "
            f"no_repeat_ngram_size={args.no_repeat_ngram_size} "
            f"phrase_bias={_phrase_processor is not None}"
        )

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
    uid2meta = {}
    paired_uid2meta = {}

    # Qwen3ASRModel.transcribe() 只返回 text。Transformers 的底层 generate()
    # 支持 output_scores；这里用可选 wrapper 捕获，不改变默认推理行为和输出 schema。
    _generate_captures = []
    if args.meta_out:
        _orig_generate_for_meta = model.model.generate

        def _capture_generate(*gen_args, **gen_kwargs):
            gen_kwargs["output_scores"] = True
            output = _orig_generate_for_meta(*gen_args, **gen_kwargs)
            input_ids = gen_kwargs.get("input_ids")
            if input_ids is None and gen_args:
                input_ids = gen_args[0]
            prompt_len = int(input_ids.shape[1]) if input_ids is not None else None
            _generate_captures.append((output, prompt_len))
            return output

        model.model.generate = _capture_generate

    def _pop_capture_meta(texts):
        """把最近一次 batch generate 的 scores 转成轻量、可序列化特征。"""
        if not args.meta_out or not _generate_captures:
            return [{} for _ in texts]
        output, prompt_len = _generate_captures.pop(0)
        scores = getattr(output, "scores", None)
        sequences = getattr(output, "sequences", None)
        if scores is None or sequences is None or prompt_len is None:
            return [{"available": False} for _ in texts]
        if len(texts) != int(sequences.shape[0]):
            return [{"available": False} for _ in texts]

        eos_ids = {151645, 151643}
        token_ids = sequences[:, prompt_len:]
        n_steps = min(len(scores), int(token_ids.shape[1]))
        if n_steps <= 0:
            return [{"available": False, "n_generated_tokens": 0} for _ in texts]

        import torch.nn.functional as F
        step_stats = []
        for step in range(n_steps):
            logits = scores[step].float()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            chosen = token_ids[:, step].unsqueeze(1)
            chosen_logprob = log_probs.gather(1, chosen).squeeze(1)
            top1_prob = probs.max(dim=-1).values
            entropy = -(probs * log_probs).sum(dim=-1)
            step_stats.append((chosen_logprob, top1_prob, entropy))

        metas = []
        for row_idx, text in enumerate(texts):
            values = []
            top1_values = []
            entropy_values = []
            saw_eos = False
            for step in range(n_steps):
                token_id = int(token_ids[row_idx, step])
                chosen_lp, top1_prob, entropy = step_stats[step]
                if token_id in eos_ids:
                    saw_eos = True
                    break
                values.append(float(chosen_lp[row_idx]))
                top1_values.append(float(top1_prob[row_idx]))
                entropy_values.append(float(entropy[row_idx]))
            if not values:
                metas.append({
                    "available": True,
                    "n_generated_tokens": 0,
                    "eos_reached": saw_eos,
                    "text_chars": len(text or ""),
                })
                continue
            values.sort()
            top1_values.sort()
            entropy_values.sort()
            p10_index = max(0, int(round(0.10 * (len(values) - 1))))
            metas.append({
                "available": True,
                "n_generated_tokens": len(values),
                "eos_reached": saw_eos,
                "text_chars": len(text or ""),
                "mean_token_logprob": sum(values) / len(values),
                "p10_token_logprob": values[p10_index],
                "mean_top1_prob": sum(top1_values) / len(top1_values),
                "p10_top1_prob": top1_values[p10_index],
                "mean_token_entropy": sum(entropy_values) / len(entropy_values),
                "max_token_entropy": max(entropy_values),
            })
        return metas

    def _store_meta(target, uid, meta):
        if args.meta_out:
            target[uid] = meta

    t0 = time.time()

    if args.batch_size == 0:
        # 逐条模式(原逻辑, 兼容)
        for i, (uid, stream, sf) in enumerate(items):
            try:
                res = model.transcribe(audio=sf, language="Chinese", context=args.context)
                target = paired_uid2text if stream == "paired" else uid2text
                target[uid] = res[0].text.strip()
                meta_target = paired_uid2meta if stream == "paired" else uid2meta
                _store_meta(meta_target, uid, _pop_capture_meta([target[uid]])[0])
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
                texts = [res.text.strip() for res in results]
                metas = _pop_capture_meta(texts)
                for (uid, stream, _), res, meta in zip(batch_items, results, metas):
                    target = paired_uid2text if stream == "paired" else uid2text
                    target[uid] = res.text.strip()
                    meta_target = paired_uid2meta if stream == "paired" else uid2meta
                    _store_meta(meta_target, uid, meta)
            except torch.cuda.OutOfMemoryError:
                # OOM fallback: 逐条重试本 batch
                torch.cuda.empty_cache()
                print(f"  batch {batch_idx}/{n_batches} OOM, fallback 逐条")
                for (uid, stream, sp) in batch_items:
                    try:
                        res = model.transcribe(audio=sp, language="Chinese", context=args.context)
                        target = paired_uid2text if stream == "paired" else uid2text
                        target[uid] = res[0].text.strip()
                        meta_target = paired_uid2meta if stream == "paired" else uid2meta
                        _store_meta(meta_target, uid, _pop_capture_meta([target[uid]])[0])
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

    _write_json(args.out, uid2text)
    transcribe_sec = time.time() - t0  # [profile] 桶C(纯转写, t0 已在 items 构造后)
    # sidecar: load/transcribe 拆解 → enroll_infer.py 读回算 overhead(桶D)
    _timing_path = os.path.join(os.path.dirname(args.out), "_qwen_timing.json")
    _write_json(_timing_path, {
        "load_sec": round(load_sec, 3),
        "transcribe_sec": round(transcribe_sec, 3),
        "n_items": len(items),
        "n_uid": len(uid2text),
        "batch_size": args.batch_size,
        "engine": args.engine,
        "vllm_max_model_len": args.vllm_max_model_len if args.engine == "vllm" else None,
        "vllm_max_num_seqs": args.vllm_max_num_seqs if args.engine == "vllm" else None,
        "vllm_gpu_memory_utilization": (
            args.vllm_gpu_memory_utilization if args.engine == "vllm" else None
        ),
        "vllm_enforce_eager": args.vllm_enforce_eager if args.engine == "vllm" else None,
    })
    print(f"转写 {len(uid2text)} 条 → {args.out} (transcribe {transcribe_sec:.0f}s + load {load_sec:.0f}s)")
    if args.paired_out:
        _write_json(args.paired_out, paired_uid2text)
        print(f"配对转写 {len(paired_uid2text)} 条 → {args.paired_out}")
    if args.meta_out:
        _write_json(args.meta_out, uid2meta)
        print(f"解码元数据 {len(uid2meta)} 条 → {args.meta_out}")

    # vLLM 0.14.0 在 WSL 的 worker 析构阶段可能先于父进程退出，导致正常
    # return 时打印 Engine core died 并返回非零；输出已用上下文管理器落盘，
    # 这里显式以成功码结束后端子进程，避免污染主流程的 check_call。
    if args.engine == "vllm":
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
