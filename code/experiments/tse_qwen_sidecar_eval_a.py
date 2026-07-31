#!/usr/bin/env python3
"""One-shot paired Dataset-A CER evaluation for a locked Sidecar checkpoint.

Every batch is decoded twice in the same model process: zero-init Sidecar
(exact Qwen baseline) and the locked trained Sidecar.  This script has no
training, optimizer, threshold fitting, or checkpoint-selection code.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from eval_metrics_official_ref import CERMetric
from tse_qwen_sidecar import FrozenQwenSidecar, SidecarConfig


def _load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(row.get("dataset_role") != "A_TEST_ONLY" for row in rows):
        raise ValueError("manifest must be explicitly marked A_TEST_ONLY")
    required = {
        "id", "recognition_audio", "ref", "enrollment_embedding"
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('id')}: missing {sorted(missing)}")
    return rows


def _prompt(processor) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    return (
        processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        + "language Chinese<asr_text>"
    )


@torch.no_grad()
def _decode(base, processor, sidecar, inputs, embeddings) -> list[str]:
    with sidecar.inference_context(embeddings):
        generated = base.generate(**inputs, max_new_tokens=256)
    return processor.batch_decode(
        generated.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model", default="E:/hf_cache/Qwen3-ASR-1.7B"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = _load_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]

    from qwen_asr.core.transformers_backend import (
        Qwen3ASRForConditionalGeneration,
    )
    from transformers import AutoProcessor

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    stored = checkpoint["config"]
    config = SidecarConfig(
        layers=int(stored["layers"]),
        rank=int(stored["rank"]),
        activity_loss_weight=float(stored["activity_loss_weight"]),
    )
    base = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(
        args.model, fix_mistral_regex=True
    )
    sidecar = FrozenQwenSidecar(base.thinker, config)
    sidecar.eval()
    zero_state = sidecar.trainable_state_dict()
    trained_state = checkpoint["sidecar"]
    device = next(base.thinker.parameters()).device
    prompt = _prompt(processor)
    baseline_texts = []
    trained_texts = []
    started = time.time()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        wavs = [
            librosa.load(row["recognition_audio"], sr=16000, mono=True)[0]
            for row in batch
        ]
        embeddings = torch.from_numpy(
            np.stack(
                [
                    np.load(row["enrollment_embedding"]).astype(np.float32)
                    for row in batch
                ]
            )
        ).to(device)
        inputs = processor(
            text=[prompt] * len(batch),
            audio=wavs,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(device).to(base.dtype)
        sidecar.load_trainable_state_dict(zero_state)
        baseline_texts.extend(
            _decode(base, processor, sidecar, inputs, embeddings)
        )
        sidecar.load_trainable_state_dict(trained_state)
        trained_texts.extend(
            _decode(base, processor, sidecar, inputs, embeddings)
        )
        done = offset + len(batch)
        print(f"[paired] {done}/{len(rows)}")

    refs = [row["ref"] for row in rows]
    baseline_metric = CERMetric()
    trained_metric = CERMetric()
    baseline_metric.update(baseline_texts, refs)
    trained_metric.update(trained_texts, refs)
    baseline = baseline_metric.compute()
    trained = trained_metric.compute()
    result_rows = []
    for index, row in enumerate(rows):
        result_rows.append(
            {
                "id": row["id"],
                "ref": row["ref"],
                "baseline": baseline_texts[index],
                "trained": trained_texts[index],
                "baseline_cer": baseline["per_sample"][index]["cer"],
                "trained_cer": trained["per_sample"][index]["cer"],
            }
        )
    result = {
        "n": len(rows),
        "checkpoint": args.checkpoint,
        "baseline_cer": baseline["cer"],
        "trained_cer": trained["cer"],
        "delta_cer": trained["cer"] - baseline["cer"],
        "elapsed_sec": time.time() - started,
        "rows": result_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
