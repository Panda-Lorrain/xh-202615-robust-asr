#!/usr/bin/env python3
"""Compare zero-init and trained Sidecar losses on synthetic held-out data.

This diagnostic is intentionally restricted to the guarded AISHELL-train
manifest format.  It separates ASR loss from target-activity BCE before any
one-shot Dataset-A CER evaluation is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tse_qwen_sidecar import FrozenQwenSidecar, SidecarConfig
from tse_qwen_sidecar_train import SidecarCollator, SidecarDataset


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int) -> dict:
    model.eval()
    total = []
    asr = []
    activity = []
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch = {
            name: value.to(device, non_blocking=True)
            for name, value in batch.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
        total.append(float(output.loss.detach().cpu()))
        asr.append(float(output.sidecar_asr_loss.detach().cpu()))
        activity.append(
            float(output.sidecar_activity_loss.detach().cpu())
        )
    if not total:
        raise RuntimeError("no validation batches")
    return {
        "batches": len(total),
        "total_loss": float(np.mean(total)),
        "asr_loss": float(np.mean(asr)),
        "activity_loss": float(np.mean(activity)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model", default="E:/hf_cache/Qwen3-ASR-1.7B"
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
    processor = AutoProcessor.from_pretrained(
        args.model, fix_mistral_regex=True
    )
    dataset = SidecarDataset(
        args.manifest,
        processor,
        require_scale=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=SidecarCollator(processor.tokenizer),
    )
    base = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model = FrozenQwenSidecar(base.thinker, config)
    device = next(base.thinker.parameters()).device
    zero = evaluate(model, loader, device, args.max_batches)
    model.load_trainable_state_dict(checkpoint["sidecar"])
    trained = evaluate(model, loader, device, args.max_batches)
    result = {
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "zero": zero,
        "trained": trained,
        "delta": {
            key: trained[key] - zero[key]
            for key in ("total_loss", "asr_loss", "activity_loss")
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
