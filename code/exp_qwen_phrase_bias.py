#!/usr/bin/env python
"""非 Dataset-A 的 Qwen 声学锚定短语偏置配对 POC。

在同一模型进程、同一批音频上依次跑 baseline 与 candidate。默认数据是项目早期
生成的 450 条智能家居合成混合音频，只用于筛掉明显有害的偏置方案；该数据规模
和域真实性不足以直接支持提交集成。
"""
import argparse
import contextlib
import json
import os
import random
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODEL = (
    r"E:/hf_cache/Qwen3-ASR-1.7B"
    if os.name == "nt"
    else "/root/hf_cache/Qwen3-ASR-1.7B"
)


def select_stratified(rows, limit):
    """按 target_ref×overlap 交错抽样，避免小 limit 偏向单一指令/重叠率。"""
    if limit <= 0 or limit >= len(rows):
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(row["target_ref"], float(row["overlap_ratio"]))].append(row)
    refs = sorted({key[0] for key in groups})
    overlaps = sorted({key[1] for key in groups})
    selected = []
    cursors = defaultdict(int)
    round_index = 0
    while len(selected) < limit:
        added = False
        for ref_index, ref in enumerate(refs):
            overlap = overlaps[(ref_index + round_index) % len(overlaps)]
            key = (ref, overlap)
            cursor = cursors[key]
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                cursors[key] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    return selected


@contextlib.contextmanager
def apply_phrase_bias(model, processor):
    from transformers import LogitsProcessorList

    original = model.model.generate

    def patched_generate(*args, **kwargs):
        current = list(kwargs.get("logits_processor") or [])
        kwargs["logits_processor"] = LogitsProcessorList(current + [processor])
        return original(*args, **kwargs)

    model.model.generate = patched_generate
    try:
        yield
    finally:
        model.model.generate = original


def normalize_output(text):
    from text_utils import brand_homophone_fix, digit_postproc, to_simplified

    return brand_homophone_fix(digit_postproc(to_simplified(text or "")))


def score_rows(rows, key):
    from eval_metrics import CERMetric

    metric = CERMetric()
    metric.update([row[key] for row in rows], [row["ref"] for row in rows])
    return metric.compute()["cer"]


def main():
    ap = argparse.ArgumentParser(description="Qwen acoustic-topK phrase bias paired POC")
    ap.add_argument(
        "--manifest",
        default=os.path.join(ROOT, "test_wav", "dataset", "final", "final_manifest.json"),
    )
    ap.add_argument(
        "--audio-dir",
        default=os.path.join(ROOT, "test_wav", "dataset", "final"),
    )
    ap.add_argument(
        "--phrases",
        default=os.path.join(HERE, "home_bias_phrases.txt"),
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL_QWEN3_ASR", DEFAULT_MODEL),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(HERE, "runs", "_qwen_phrase_bias_synth", "result.json"),
    )
    ap.add_argument("--limit", type=int, default=30, help="0=全量；默认30条烟测")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--bias-strength", type=float, default=0.8)
    ap.add_argument("--bias-top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    source_rows = manifest["items"] if isinstance(manifest, dict) else manifest
    source_rows = select_stratified(source_rows, args.limit)
    eval_rows = [
        {
            **row,
            "audio": os.path.join(args.audio_dir, row["file"]),
            "ref": row["target_ref"],
        }
        for row in source_rows
    ]
    missing = [row["audio"] for row in eval_rows if not os.path.isfile(row["audio"])]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio files missing; first={missing[0]}")

    from qwen_asr import Qwen3ASRModel
    from qwen_phrase_bias import (
        AcousticTopKPhraseBias,
        load_phrases,
        tokenize_phrases,
    )

    print(f"[load] Qwen3-ASR {args.model}; n={len(eval_rows)}")
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=args.batch_size,
    )
    phrases = load_phrases(args.phrases)
    tokenized = tokenize_phrases(model.processor.tokenizer, phrases)
    bias_processor = AcousticTopKPhraseBias(
        tokenized, bias=args.bias_strength, top_k=args.bias_top_k
    )

    t0 = time.perf_counter()
    for start in range(0, len(eval_rows), args.batch_size):
        batch = eval_rows[start : start + args.batch_size]
        paths = [row["audio"] for row in batch]
        baseline = model.transcribe(audio=paths, language="Chinese", context="")
        with apply_phrase_bias(model, bias_processor):
            candidate = model.transcribe(audio=paths, language="Chinese", context="")
        for row, base_result, cand_result in zip(batch, baseline, candidate):
            row["baseline"] = normalize_output(base_result.text)
            row["candidate"] = normalize_output(cand_result.text)
        print(f"  [{min(start + args.batch_size, len(eval_rows))}/{len(eval_rows)}]")

    baseline_cer = score_rows(eval_rows, "baseline")
    candidate_cer = score_rows(eval_rows, "candidate")
    bucket_summary = {}
    for overlap in sorted({float(row["overlap_ratio"]) for row in eval_rows}):
        bucket = [row for row in eval_rows if float(row["overlap_ratio"]) == overlap]
        b = score_rows(bucket, "baseline")
        c = score_rows(bucket, "candidate")
        bucket_summary[str(overlap)] = {
            "n": len(bucket),
            "baseline_cer": b,
            "candidate_cer": c,
            "delta": c - b,
        }

    deltas = [bucket["delta"] for bucket in bucket_summary.values()]
    gate = (
        candidate_cer - baseline_cer <= -0.01
        and max(deltas, default=0.0) <= 0.005
    )
    payload = {
        "dataset": "non-A synthetic home-command mixtures",
        "limitations": [
            "single synthetic target voice",
            "only ten command references",
            "mixture is transcribed directly without enrollment-conditioned target slicing",
            "positive result is only a smoke gate and cannot justify submission integration",
        ],
        "config": {
            "n": len(eval_rows),
            "batch_size": args.batch_size,
            "bias_strength": args.bias_strength,
            "bias_top_k": args.bias_top_k,
            "phrases": phrases,
            "seed": args.seed,
        },
        "overall": {
            "baseline_cer": baseline_cer,
            "candidate_cer": candidate_cer,
            "delta": candidate_cer - baseline_cer,
        },
        "overlap_buckets": bucket_summary,
        "smoke_gate": {
            "pass": gate,
            "criteria": "overall delta <= -0.01 and every overlap bucket delta <= +0.005",
        },
        "wall_sec": round(time.perf_counter() - t0, 3),
        "rows": eval_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(
        f"[result] CER {baseline_cer:.4f} -> {candidate_cer:.4f} "
        f"(delta={candidate_cer-baseline_cer:+.4f}); gate={'PASS' if gate else 'FAIL'}"
    )
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
