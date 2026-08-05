"""Formal all-pairs trainer for the anchor-preserving DACF-v4 model."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dacf_v4_anchor_relation import DACFV4AnchorRelation
from dacf_v4_objective import DACFV4LossWeights, compute_dacf_v4_loss
from train_dacf_v3_mechanism import MechanismContractError, load_feature_cache
from train_dacf_v4_allpairs import (
    DACFV4TrainingConfig,
    MECHANISM_GATE,
    THRESHOLD,
    _cache_contract,
    _device_for,
    _evaluate_split,
    _json_write,
    _set_determinism,
    _sha256_file,
    build_allpairs_batch,
    epoch_group_batches,
)


PROTOCOL_VERSION = "dacf-v4-anchor-allpairs-mechanism-v0.1"
PREREGISTRATION_SCHEMA = "dacf-v4-anchor-allpairs-preregistration-v0.1"
REPORT_SCHEMA = "dacf-v4-anchor-allpairs-report-v0.1"


def _source_hashes() -> Mapping[str, str]:
    root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "code/experiments/dacf_v4_relation.py",
        "code/experiments/dacf_v4_anchor_relation.py",
        "code/experiments/dacf_v4_objective.py",
        "code/experiments/train_dacf_v3_mechanism.py",
        "code/experiments/train_dacf_v4_allpairs.py",
        "code/experiments/train_dacf_v4_anchor_allpairs.py",
    )
    return {
        relative: _sha256_file(root / relative)
        for relative in relative_paths
    }


def preregistration_payload(cache_root: str | Path) -> Mapping[str, Any]:
    cache = load_feature_cache(cache_root)
    contract = _cache_contract(cache)
    config = DACFV4TrainingConfig()
    config.validate(int(contract["train_groups"]))
    model = DACFV4AnchorRelation()
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_a_used": False,
        "cache": {
            "root": cache.root.as_posix(),
            "report_sha256": _sha256_file(cache.root / "cache_report.json"),
            "cache_sha256": str(cache.report.get("cache_sha256", "")),
            "manifest_sha256": str(cache.report.get("manifest_sha256", "")),
            "contract": contract,
        },
        "failure_mode_removed": {
            "prior_model": "DACFV4Relation",
            "train_only_observation": (
                "learned query projection changed raw CAM++ off-diagonal cosine "
                "from 0.436 to 0.921"
            ),
            "query_projection": "absent",
            "query_embedding": "raw CAM++ L2 normalisation only",
            "pair_logit": "scaled cosine plus one global calibration bias",
            "relation_mlp": "absent",
        },
        "pairing_contract": {
            "query_rows": "present A/B only",
            "same_query_both_labels_in_update": True,
            "query_positive_mixtures_per_batch": 1,
            "query_foreign_mixtures_per_batch": config.groups_per_batch - 1,
            "membership_label_recomputed_for_destination_mixture": True,
            "c_only_queries_optimised": 0,
            "final_deferred": True,
        },
        "training": asdict(config),
        "loss_weights": asdict(DACFV4LossWeights()),
        "threshold": THRESHOLD,
        "mechanism_gate": dict(MECHANISM_GATE),
        "model": {
            "class": "DACFV4AnchorRelation",
            "trainable_parameters": model.trainable_parameter_count(),
            "parameter_limit": 2_000_000,
        },
        "selection_policy": {
            "checkpoint": "epoch_30_final_only",
            "scheduler": "none",
            "early_stop": False,
            "hyperparameter_scan": False,
            "dev_evaluation_count": 1,
            "final_opened": False,
            "qwen_integration": False,
        },
        "source_sha256": _source_hashes(),
    }


def write_preregistration(
    cache_root: str | Path,
    output_path: str | Path,
) -> Mapping[str, Any]:
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"preregistration output already exists: {path}")
    payload = preregistration_payload(cache_root)
    _json_write(path, payload)
    return {**payload, "preregistration_sha256": _sha256_file(path)}


def validate_preregistration(
    cache_root: str | Path,
    preregistration_path: str | Path,
) -> Mapping[str, Any]:
    path = Path(preregistration_path).resolve(strict=True)
    actual = json.loads(path.read_text(encoding="utf-8"))
    expected = preregistration_payload(cache_root)
    if actual != expected:
        raise MechanismContractError(
            "anchor-allpairs preregistration does not match source/cache contract"
        )
    return {"path": path.as_posix(), "sha256": _sha256_file(path), "validated": True}


def run_training(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    preregistration_path: str | Path,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    cache = load_feature_cache(cache_root)
    contract = _cache_contract(cache)
    config = DACFV4TrainingConfig()
    config.validate(int(contract["train_groups"]))
    prereg = validate_preregistration(cache_root, preregistration_path)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    _set_determinism(config.seed)
    device = _device_for(config)
    model = DACFV4AnchorRelation().to(device)
    if model.trainable_parameter_count() >= 2_000_000:
        raise MechanismContractError("anchor relation exceeds the parameter cap")
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    weights = DACFV4LossWeights()
    train_groups = cache.groups["train"]
    loss_trace: list[float] = []
    negative_pair_coverage: set[tuple[str, str]] = set()

    model.train()
    for epoch in range(config.epochs):
        for batch_groups in epoch_group_batches(
            train_groups,
            groups_per_batch=config.groups_per_batch,
            seed=config.seed,
            epoch=epoch,
        ):
            pair_batch = build_allpairs_batch(batch_groups)
            for source_group in set(pair_batch.source_group_ids):
                for destination in (group.group_id for group in batch_groups):
                    if destination != source_group:
                        negative_pair_coverage.add((source_group, destination))
            query1 = torch.from_numpy(pair_batch.embeddings_view1).to(
                device=device, dtype=torch.float32
            )
            query2 = torch.from_numpy(pair_batch.embeddings_view2).to(
                device=device, dtype=torch.float32
            )
            optimiser.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for mixture_index, group in enumerate(batch_groups):
                mixture = torch.from_numpy(group.mixture_features).unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                labels = torch.from_numpy(
                    pair_batch.presence_labels[mixture_index : mixture_index + 1]
                ).to(device=device, dtype=torch.float32)
                activity = torch.from_numpy(
                    pair_batch.activity_targets[mixture_index][None]
                ).to(device=device, dtype=torch.float32)
                state = model.encode_mixture(mixture)
                loss = compute_dacf_v4_loss(
                    model.score_queries(state, query1),
                    model.score_queries(state, query2),
                    presence_labels=labels,
                    activity_targets=activity,
                    margin=config.hard_foreign_margin,
                    weights=weights,
                )
                (loss.total / len(batch_groups)).backward()
                batch_loss += float(loss.total.detach().cpu()) / len(batch_groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimiser.step()
            loss_trace.append(batch_loss)

    checkpoint = output / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "DACFV4AnchorRelation",
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(config),
        },
        checkpoint,
    )
    checkpoint_before = _sha256_file(checkpoint)
    train_metrics = _evaluate_split(
        model,
        train_groups,
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 30_000,
    )
    dev_metrics = _evaluate_split(
        model,
        cache.groups["dev"],
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 40_000,
    )
    checkpoint_after = _sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise MechanismContractError("checkpoint changed during frozen evaluation")

    total_pairs = len(train_groups) * (len(train_groups) - 1)
    report: Mapping[str, Any] = {
        "schema": REPORT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "verdict": (
            "conditional-GO" if dev_metrics["gate"]["passed"] else "implementation-NO-GO"
        ),
        "verdict_scope": {
            "scope": "anchor-preserving all-pairs identity mechanism only",
            "final_opened": False,
            "qwen_integration": "not run",
            "cer": "not measured",
            "official_negative_rr": "not measured",
            "rtf": "not measured",
        },
        "dataset_a_used": False,
        "cache_contract": contract,
        "pairing_contract": {
            "same_query_both_labels_per_update": True,
            "c_only_queries_optimised": 0,
            "negative_source_destination_pairs_seen": len(negative_pair_coverage),
            "negative_source_destination_pairs_total": total_pairs,
            "negative_pair_coverage": len(negative_pair_coverage) / total_pairs,
        },
        "training": {
            "config": asdict(config),
            "loss_weights": asdict(weights),
            "optimizer": "AdamW",
            "scheduler": "none",
            "early_stop": False,
            "hyperparameter_scan": False,
            "precision": "FP32",
            "device": str(device),
            "updates": len(loss_trace),
            "loss_first": loss_trace[0],
            "loss_last": loss_trace[-1],
        },
        "checkpoint": {
            "path": checkpoint.resolve().as_posix(),
            "sha256_before_dev": checkpoint_before,
            "sha256_after_dev": checkpoint_after,
            "frozen_before_dev": True,
        },
        "preregistration": prereg,
        "train_diagnostic": train_metrics,
        "dev": dev_metrics,
        "dev_evaluation_count": 1,
        "same_text_different_speaker": {
            "status": "deferred",
            "required_before_qwen": True,
        },
        "runtime_sec": float(time.perf_counter() - started),
    }
    _json_write(output / "mechanism_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--write-preregistration")
    parser.add_argument("--output-dir")
    parser.add_argument("--preregistration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_preregistration:
        if args.output_dir or args.preregistration:
            raise SystemExit("preregistration mode does not accept training outputs")
        report = write_preregistration(args.cache_root, args.write_preregistration)
    else:
        if not args.output_dir or not args.preregistration:
            raise SystemExit("training requires --output-dir and --preregistration")
        report = run_training(
            args.cache_root,
            args.output_dir,
            preregistration_path=args.preregistration,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
