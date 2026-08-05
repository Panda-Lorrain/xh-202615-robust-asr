"""Small speaker-disjoint DACF query-control probe.

This is the gate immediately after the fixed-mixture overfit smoke.  It trains
the compact DACF front-end on several non-Dataset-A A/B/C groups and evaluates
one fixed operating point on speaker/source-disjoint groups.  It deliberately
does not run Qwen, report CER, tune a threshold, or claim that AISHELL read
speech is a verified home-command hard negative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from dacf_frontend import DACFFrontend
from smoke_dacf_overfit import (
    DACFBatch,
    ROLE_ORDER,
    _assert_not_dataset_a,
    _compute_smoke_loss,
    _read_jsonl,
    evaluate_permutation_negative_control,
    load_dacf_group,
)


DEFAULT_SEED = 20260806
DEFAULT_UPDATES = 240
MAX_UPDATES = 600
MAX_GROUPS_PER_SPLIT = 64
FIXED_PRESENCE_THRESHOLD = 0.5

# Preregistered mini-G2 mechanism gate.  Passing it is only conditional-GO to
# collect real home-command data; it is not a CER or integration gate.
MIN_VAL_AUC = 0.80
MIN_VAL_PRESENT_RECALL = 0.75
MIN_VAL_ABSENT_RR = 0.75
MIN_QUERY_RESPONSE = 0.20


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _group_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    groups = sorted({str(row.get("base_mixture_id", "")).strip() for row in rows})
    if not groups or not groups[0]:
        raise ValueError("every row requires a non-empty base_mixture_id")
    if len(groups) > MAX_GROUPS_PER_SPLIT:
        raise ValueError(
            f"small probe is limited to {MAX_GROUPS_PER_SPLIT} groups per split"
        )
    return groups


def _row_speakers(row: Mapping[str, Any]) -> set[str]:
    speakers = {
        str(value)
        for value in (row.get("query_speaker_id"), row.get("target_spk"))
        if value
    }
    mixture = row.get("mixture_speakers")
    if isinstance(mixture, Mapping):
        speakers.update(str(value) for value in mixture.values() if value)
    return speakers


def _row_sources(row: Mapping[str, Any]) -> set[str]:
    sources = {
        str(value)
        for value in (row.get("enrollment_src"), row.get("target_src"))
        if value
    }
    mixture = row.get("mixture_sources")
    if isinstance(mixture, Mapping):
        sources.update(str(value) for value in mixture.values() if value)
    return sources


def validate_probe_manifests(
    train_manifest: str | Path, val_manifest: str | Path
) -> tuple[Path, Path, list[str], list[str], dict[str, int]]:
    """Fail before audio reads if either split leaks identities or Dataset-A."""

    train_path = Path(train_manifest).resolve(strict=True)
    val_path = Path(val_manifest).resolve(strict=True)
    _assert_not_dataset_a(train_path, field="train-manifest")
    _assert_not_dataset_a(val_path, field="val-manifest")
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for index, row in enumerate(rows):
            if bool(row.get("dataset_a_used", False)):
                raise ValueError(f"{split} row {index} declares dataset_a_used=true")
            if str(row.get("split", "")) != split:
                raise ValueError(
                    f"{split} manifest contains row with split={row.get('split')!r}"
                )

    train_speakers = set().union(*(_row_speakers(row) for row in train_rows))
    val_speakers = set().union(*(_row_speakers(row) for row in val_rows))
    speaker_overlap = train_speakers & val_speakers
    if speaker_overlap:
        raise ValueError(f"train/val speaker overlap: {sorted(speaker_overlap)[:3]}")
    train_sources = set().union(*(_row_sources(row) for row in train_rows))
    val_sources = set().union(*(_row_sources(row) for row in val_rows))
    source_overlap = train_sources & val_sources
    if source_overlap:
        raise ValueError(f"train/val source overlap: {sorted(source_overlap)[:3]}")

    train_groups = _group_ids(train_rows)
    val_groups = _group_ids(val_rows)
    return train_path, val_path, train_groups, val_groups, {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_speakers": len(train_speakers),
        "val_speakers": len(val_speakers),
        "speaker_overlap": 0,
        "source_overlap": 0,
    }


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Tie-aware probability that a random positive outranks a negative."""

    if not positive or not negative:
        raise ValueError("ROC AUC requires positive and negative scores")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return wins / (len(positive) * len(negative))


def _model_forward(model: DACFFrontend, batch: DACFBatch) -> Mapping[str, torch.Tensor]:
    device = next(model.parameters()).device
    return model(batch.mixture.to(device), batch.enrollment.to(device))


def _train_loss(model: DACFFrontend, batch: DACFBatch) -> torch.Tensor:
    device = next(model.parameters()).device
    outputs = _model_forward(model, batch)
    targets = batch.targets(include_ctc=False, device=device)
    return _compute_smoke_loss(outputs, targets)["total"]


def evaluate_groups(
    model: DACFFrontend, batches: Sequence[DACFBatch]
) -> dict[str, Any]:
    """Evaluate fixed-threshold presence and enrollment intervention effects."""

    present: list[float] = []
    absent: list[float] = []
    margins: list[float] = []
    query_effects: list[float] = []
    unaffected_drift: list[float] = []
    permutation_presence_loss_delta: list[float] = []
    absent_rms: list[float] = []
    loss_values: dict[str, list[float]] = {}
    target_l1_values: dict[str, list[float]] = {role: [] for role in ROLE_ORDER}

    was_training = model.training
    model.eval()
    try:
        for batch in batches:
            if batch.view_count != 2:
                raise ValueError("generalization probe requires two enrollment views")
            swap_ac = evaluate_permutation_negative_control(
                model,
                batch,
                include_ctc=False,
                permutation=(4, 5, 2, 3, 0, 1),
            )
            swap_bc = evaluate_permutation_negative_control(
                model,
                batch,
                include_ctc=False,
                permutation=(0, 1, 4, 5, 2, 3),
            )
            original = swap_ac["original"]
            for name, value in original["losses"].items():
                if value is not None:
                    loss_values.setdefault(name, []).append(float(value))
            for role, value in original["target_l1"].items():
                target_l1_values[role].append(float(value))
            probs = original["presence_prob"]
            p_a = float(probs["present_A"])
            p_b = float(probs["present_B"])
            p_c = float(probs["absent_C"])
            present.extend((p_a, p_b))
            absent.append(p_c)
            margins.append(0.5 * (p_a + p_b) - p_c)
            query_effects.extend(
                (
                    p_a - float(swap_ac["permuted"]["presence_prob"]["present_A"]),
                    float(swap_ac["permuted"]["presence_prob"]["absent_C"]) - p_c,
                    p_b - float(swap_bc["permuted"]["presence_prob"]["present_B"]),
                    float(swap_bc["permuted"]["presence_prob"]["absent_C"]) - p_c,
                )
            )
            unaffected_drift.extend(
                (
                    abs(
                        float(swap_ac["permuted"]["presence_prob"]["present_B"])
                        - p_b
                    ),
                    abs(
                        float(swap_bc["permuted"]["presence_prob"]["present_A"])
                        - p_a
                    ),
                )
            )
            permutation_presence_loss_delta.extend(
                (
                    float(swap_ac["loss_change"]["presence"]),
                    float(swap_bc["loss_change"]["presence"]),
                )
            )
            absent_rms.append(float(original["absent_output_rms"]))
    finally:
        model.train(was_training)

    threshold = FIXED_PRESENCE_THRESHOLD
    return {
        "groups": len(batches),
        "positive_queries": len(present),
        "absent_queries": len(absent),
        "fixed_threshold": threshold,
        "roc_auc": _roc_auc(present, absent),
        "present_recall": float(np.mean([score >= threshold for score in present])),
        "absent_rr": float(np.mean([score < threshold for score in absent])),
        "present_probability_mean": float(np.mean(present)),
        "absent_probability_mean": float(np.mean(absent)),
        "presence_margin_mean": float(np.mean(margins)),
        "query_permutation_response_mean": float(np.mean(query_effects)),
        "query_permutation_response_min": float(np.min(query_effects)),
        "unaffected_query_drift_mean": float(np.mean(unaffected_drift)),
        "permutation_presence_loss_delta_mean": float(
            np.mean(permutation_presence_loss_delta)
        ),
        "absent_output_rms_mean": float(np.mean(absent_rms)),
        "loss_mean": {
            name: float(np.mean(values)) for name, values in sorted(loss_values.items())
        },
        "target_l1_mean": {
            role: float(np.mean(values))
            for role, values in target_l1_values.items()
        },
    }


def _conditional_gate(metrics: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    checks = {
        "roc_auc": float(metrics["roc_auc"]) >= MIN_VAL_AUC,
        "present_recall": float(metrics["present_recall"])
        >= MIN_VAL_PRESENT_RECALL,
        "absent_rr": float(metrics["absent_rr"]) >= MIN_VAL_ABSENT_RR,
        "query_response": float(metrics["query_permutation_response_mean"])
        >= MIN_QUERY_RESPONSE,
    }
    return all(checks.values()), checks


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_probe(
    train_manifest: str | Path,
    val_manifest: str | Path,
    *,
    updates: int = DEFAULT_UPDATES,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    learning_rate: float = 3e-3,
    checkpoint: Optional[str | Path] = None,
) -> dict[str, Any]:
    if isinstance(updates, bool) or int(updates) != updates:
        raise ValueError(f"updates must be an integer in [1, {MAX_UPDATES}]")
    updates = int(updates)
    if updates < 1 or updates > MAX_UPDATES:
        raise ValueError(f"updates must be in [1, {MAX_UPDATES}], got {updates}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    train_path, val_path, train_ids, val_ids, audit = validate_probe_manifests(
        train_manifest, val_manifest
    )
    _seed_everything(seed)
    train_batches = [load_dacf_group(train_path, group_id) for group_id in train_ids]
    val_batches = [load_dacf_group(val_path, group_id) for group_id in val_ids]
    model = DACFFrontend(
        n_fft=400,
        hop_length=160,
        win_length=400,
        d_model=16,
        n_heads=4,
        vocab_size=2,
        dropout=0.0,
    ).to(resolved)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)

    before = evaluate_groups(model, val_batches)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    order = list(range(len(train_batches)))
    losses: list[float] = []
    model.train()
    for update in range(updates):
        if update % len(order) == 0:
            random.shuffle(order)
        batch = train_batches[order[update % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        loss = _train_loss(model, batch)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    after_train = evaluate_groups(model, train_batches)
    after_val = evaluate_groups(model, val_batches)
    passed, checks = _conditional_gate(after_val)
    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_file = Path(checkpoint)
        _assert_not_dataset_a(checkpoint_file, field="checkpoint")
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": seed,
                "updates": updates,
                "train_manifest_sha256": _manifest_sha256(train_path),
                "val_manifest_sha256": _manifest_sha256(val_path),
            },
            checkpoint_file,
        )
        checkpoint_path = checkpoint_file.resolve(strict=False).as_posix()

    peak_mib = (
        float(torch.cuda.max_memory_allocated(resolved) / (1024 * 1024))
        if resolved.type == "cuda"
        else None
    )
    return {
        "status": "conditional-GO" if passed else "implementation-NO-GO",
        "scope": "mini-G2 speaker-disjoint query-control probe",
        "dataset_a_used": False,
        "hard_negative_verified": False,
        "cer_measured": False,
        "threshold_tuned": False,
        "fixed_presence_threshold": FIXED_PRESENCE_THRESHOLD,
        "limitations": [
            "AISHELL read speech is not a verified complete-command hard negative.",
            "This small probe does not establish CER, RR, RTF, or integration value.",
            "One seed and one fixed update budget are a mechanism gate, not a final estimate.",
            "A/C and B/C permutation response is the paired query intervention form of the presence margin, not an independent metric.",
        ],
        "audit": audit,
        "train_manifest": train_path.as_posix(),
        "val_manifest": val_path.as_posix(),
        "train_manifest_sha256": _manifest_sha256(train_path),
        "val_manifest_sha256": _manifest_sha256(val_path),
        "train_groups": len(train_batches),
        "val_groups": len(val_batches),
        "seed": seed,
        "updates": updates,
        "learning_rate": learning_rate,
        "device": str(resolved),
        "cuda_peak_memory_mib": peak_mib,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_loss_tail_mean": float(np.mean(losses[-min(24, len(losses)) :])),
        "before_val": before,
        "after_train": after_train,
        "after_val": after_val,
        "gate_thresholds": {
            "min_val_auc": MIN_VAL_AUC,
            "min_val_present_recall": MIN_VAL_PRESENT_RECALL,
            "min_val_absent_rr": MIN_VAL_ABSENT_RR,
            "min_query_response": MIN_QUERY_RESPONSE,
        },
        "gate_checks": checks,
        "checkpoint": checkpoint_path,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_probe(
        args.train_manifest,
        args.val_manifest,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        checkpoint=args.checkpoint,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output_json:
        output = Path(args.output_json)
        _assert_not_dataset_a(output, field="output-json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()


__all__ = [
    "FIXED_PRESENCE_THRESHOLD",
    "MAX_UPDATES",
    "_conditional_gate",
    "_roc_auc",
    "run_probe",
    "validate_probe_manifests",
]
