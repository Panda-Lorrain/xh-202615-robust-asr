#!/usr/bin/env python3
"""Train the Phase-3 frozen-Qwen Sidecar with direct ASR supervision.

Dataset A is forbidden here.  Training and validation manifests must reference
AISHELL's official ``wav/train`` split and precomputed CAM++/activity arrays.
The script intentionally exposes no Dataset-A evaluation option so checkpoint
selection cannot accidentally leak the test set.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tse_qwen_sidecar import FrozenQwenSidecar, SidecarConfig


SR = 16000
BANNED_PATH_TOKENS = ("dataseta", "/dataset_a/", "\\dataset_a\\")


def _load_rows(
    manifest: str,
    *,
    limit: int = 0,
    require_scale: bool = True,
) -> List[dict]:
    with open(manifest, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"empty manifest: {manifest}")
    _validate_rows(rows, manifest=manifest, require_scale=require_scale)
    return rows


def _validate_rows(
    rows: List[dict],
    *,
    manifest: str = "<memory>",
    require_scale: bool = True,
) -> None:
    required = {
        "id",
        "recognition_audio",
        "target_src",
        "target_spk",
        "ref",
        "enrollment_embedding",
        "target_activity",
    }
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{manifest} row {index} missing {missing}")
        serialized = json.dumps(row, ensure_ascii=False).lower()
        if any(token in serialized for token in BANNED_PATH_TOKENS):
            raise ValueError(f"Dataset A leakage in {manifest} row {index}")
        source = row["target_src"].replace("\\", "/").lower()
        if "/wav/train/" not in source:
            raise ValueError(
                f"non-AISHELL-train target source in {manifest} row {index}: "
                f"{row['target_src']}"
            )
    if require_scale:
        speakers = {row["target_spk"] for row in rows}
        refs = {row["ref"] for row in rows}
        if len(speakers) < 100 or len(refs) < 1000:
            raise ValueError(
                "training manifest is too small for ASR-aware Sidecar: "
                f"{len(speakers)} target speakers / {len(refs)} unique refs; "
                "require >=100 / >=1000 (use --allow-small-smoke only for one-step tests)"
            )


def _build_prompt(processor, language: str) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return prompt + f"language {language}<asr_text>"


class SidecarDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        processor,
        *,
        limit: int = 0,
        require_scale: bool = True,
        language: str = "Chinese",
    ):
        self.rows = _load_rows(
            manifest, limit=limit, require_scale=require_scale
        )
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt = _build_prompt(processor, language)
        self.asr_text_token = self.tokenizer.convert_tokens_to_ids("<asr_text>")
        if self.asr_text_token is None or self.asr_text_token < 0:
            raise ValueError("Qwen tokenizer does not expose <asr_text>")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        wav, _ = librosa.load(row["recognition_audio"], sr=SR, mono=True)
        eos = self.tokenizer.eos_token or "<|endoftext|>"
        inputs = self.processor(
            text=self.prompt + row["ref"] + eos,
            audio=wav,
            return_tensors="pt",
            padding=False,
        )
        item = {
            name: value.squeeze(0)
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        labels = item["input_ids"].clone()
        positions = (labels == self.asr_text_token).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError(f"{row['id']}: <asr_text> missing after processing")
        labels[: positions[0].item() + 1] = -100
        item["labels"] = labels
        embedding = np.load(row["enrollment_embedding"]).astype(np.float32)
        activity = np.load(row["target_activity"]).astype(np.float32)
        if embedding.shape != (512,) or activity.ndim != 1:
            raise ValueError(
                f"{row['id']}: embedding={embedding.shape}, activity={activity.shape}"
            )
        item["enrollment_embeddings"] = torch.from_numpy(embedding)
        item["target_activity"] = torch.from_numpy(activity)
        return item


class SidecarCollator:
    def __init__(self, tokenizer):
        self.pad_token = tokenizer.pad_token_id
        if self.pad_token is None:
            self.pad_token = tokenizer.eos_token_id

    @staticmethod
    def _pad_1d(
        values: List[torch.Tensor], fill: int | float
    ) -> torch.Tensor:
        width = max(value.numel() for value in values)
        output = torch.full(
            (len(values), width),
            fill,
            dtype=values[0].dtype,
        )
        for index, value in enumerate(values):
            output[index, : value.numel()] = value
        return output

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = self._pad_1d(
            [item["input_ids"] for item in batch], self.pad_token
        )
        attention = self._pad_1d(
            [item["attention_mask"].long() for item in batch], 0
        )
        labels = self._pad_1d([item["labels"] for item in batch], -100)
        feature_lengths = [
            item["input_features"].shape[-1] for item in batch
        ]
        max_features = max(feature_lengths)
        mel_bins = batch[0]["input_features"].shape[0]
        features = torch.zeros(
            len(batch),
            mel_bins,
            max_features,
            dtype=batch[0]["input_features"].dtype,
        )
        feature_mask = torch.zeros(
            len(batch), max_features, dtype=torch.long
        )
        for index, item in enumerate(batch):
            length = feature_lengths[index]
            features[index, :, :length] = item["input_features"]
            feature_mask[index, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": features,
            "feature_attention_mask": feature_mask,
            "enrollment_embeddings": torch.stack(
                [item["enrollment_embeddings"] for item in batch]
            ),
            "target_activity": self._pad_1d(
                [item["target_activity"] for item in batch], 0.0
            ),
        }


def _validate_split(train_rows: List[dict], val_rows: List[dict]) -> None:
    train_speakers = {row["target_spk"] for row in train_rows}
    val_speakers = {row["target_spk"] for row in val_rows}
    train_sources = {row["target_src"] for row in train_rows}
    val_sources = {row["target_src"] for row in val_rows}
    if train_speakers & val_speakers:
        raise ValueError("train/val target speakers overlap")
    if train_sources & val_sources:
        raise ValueError("train/val target utterances overlap")


def _move_batch(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
    }


def _assert_gradients(model: FrozenQwenSidecar) -> None:
    trainable = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not any(
        grad is not None
        and torch.isfinite(grad).all()
        and grad.abs().sum().item() > 0
        for grad in trainable
    ):
        raise RuntimeError("Sidecar received no finite non-zero gradient")
    leaked = [
        name
        for name, parameter in model.thinker.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if leaked:
        raise RuntimeError(f"frozen Qwen gradients leaked: {leaked[:3]}")


@torch.no_grad()
def _evaluate(
    model: FrozenQwenSidecar,
    loader: DataLoader,
    device,
    *,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        batch = _move_batch(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    if not losses:
        raise RuntimeError("validation loader produced no batches")
    return float(np.mean(losses))


def _save_sidecar(
    path: Path,
    model: FrozenQwenSidecar,
    args: argparse.Namespace,
    *,
    train_rows: int,
    val_rows: int,
    updates: int,
    train_loss: Optional[float],
    val_loss: Optional[float],
    peak_vram_gib: float,
) -> None:
    torch.save(
        {
            "sidecar": model.trainable_state_dict(),
            "config": vars(args),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "updates": updates,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "peak_vram_gib": peak_vram_gib,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument(
        "--model", default="E:/hf_cache/Qwen3-ASR-1.7B"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--activity-loss-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=100,
        help="fixed speaker-disjoint validation batches per checkpoint (0=all)",
    )
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-small-smoke",
        action="store_true",
        help="disable the >=100 speaker / >=1000 ref guard for one-step smoke only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-3 backward requires CUDA")
    if args.allow_small_smoke and args.max_steps != 1:
        raise ValueError("--allow-small-smoke is restricted to --max-steps 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from qwen_asr.core.transformers_backend import (
        Qwen3ASRForConditionalGeneration,
    )
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model, fix_mistral_regex=True
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    require_scale = not args.allow_small_smoke
    train_set = SidecarDataset(
        args.train_manifest,
        processor,
        limit=args.train_limit,
        require_scale=require_scale,
    )
    val_set = SidecarDataset(
        args.val_manifest,
        processor,
        limit=args.val_limit,
        require_scale=False,
    )
    _validate_split(train_set.rows, val_set.rows)
    collator = SidecarCollator(processor.tokenizer)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    base = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    base.thinker.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    config = SidecarConfig(
        layers=args.layers,
        rank=args.rank,
        activity_loss_weight=args.activity_loss_weight,
    )
    model = FrozenQwenSidecar(base.thinker, config)
    model.train()
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    device = next(base.thinker.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    start = time.time()
    updates = 0
    micro_steps = 0
    losses: List[float] = []
    initial_val_loss = _evaluate(
        model,
        val_loader,
        device,
        max_batches=args.val_batches,
    )
    validation_history = [{"updates": 0, "loss": initial_val_loss}]
    best_val_loss = initial_val_loss
    last_validated_update = 0
    print(f"[val 0] loss={initial_val_loss:.4f}")
    _save_sidecar(
        output_dir / "sidecar_best.pt",
        model,
        args,
        train_rows=len(train_set),
        val_rows=len(val_set),
        updates=0,
        train_loss=None,
        val_loss=initial_val_loss,
        peak_vram_gib=torch.cuda.max_memory_allocated() / (1024 ** 3),
    )
    first_gradient_checked = False
    stop = False
    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = _move_batch(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch)
                loss = output.loss / args.grad_accum
            loss.backward()
            micro_steps += 1
            losses.append(float(output.loss.detach().cpu()))
            if not first_gradient_checked:
                _assert_gradients(model)
                first_gradient_checked = True
            if micro_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates == 1 or updates % args.log_every == 0:
                    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    print(
                        f"[step {updates}] loss={np.mean(losses[-args.grad_accum:]):.4f} "
                        f"peak_vram={peak:.2f}GiB elapsed={time.time()-start:.0f}s"
                    )
                if args.val_every and updates % args.val_every == 0:
                    val_loss = _evaluate(
                        model,
                        val_loader,
                        device,
                        max_batches=args.val_batches,
                    )
                    last_validated_update = updates
                    validation_history.append(
                        {"updates": updates, "loss": val_loss}
                    )
                    print(f"[val {updates}] loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_sidecar(
                            output_dir / "sidecar_best.pt",
                            model,
                            args,
                            train_rows=len(train_set),
                            val_rows=len(val_set),
                            updates=updates,
                            train_loss=losses[-1],
                            val_loss=val_loss,
                            peak_vram_gib=(
                                torch.cuda.max_memory_allocated()
                                / (1024 ** 3)
                            ),
                        )
            if args.max_steps and updates >= args.max_steps:
                stop = True
                break
        if stop:
            break
    if micro_steps % args.grad_accum:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    if last_validated_update != updates:
        val_loss = _evaluate(
            model,
            val_loader,
            device,
            max_batches=args.val_batches,
        )
        validation_history.append({"updates": updates, "loss": val_loss})
        print(f"[val {updates}] loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_sidecar(
                output_dir / "sidecar_best.pt",
                model,
                args,
                train_rows=len(train_set),
                val_rows=len(val_set),
                updates=updates,
                train_loss=losses[-1] if losses else None,
                val_loss=val_loss,
                peak_vram_gib=(
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                ),
            )
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    _save_sidecar(
        output_dir / "sidecar.pt",
        model,
        args,
        train_rows=len(train_set),
        val_rows=len(val_set),
        updates=updates,
        train_loss=losses[-1] if losses else None,
        val_loss=validation_history[-1]["loss"],
        peak_vram_gib=peak_vram_gib,
    )
    summary = {
        "config": vars(args),
        "train_rows": len(train_set),
        "val_rows": len(val_set),
        "updates": updates,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "best_val_loss": best_val_loss,
        "validation_history": validation_history,
        "peak_vram_gib": peak_vram_gib,
    }
    with (output_dir / "train_summary.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(
        f"[done] updates={updates} trainable={model.trainable_parameter_count} "
        f"checkpoint={output_dir / 'sidecar.pt'}"
    )


if __name__ == "__main__":
    main()
