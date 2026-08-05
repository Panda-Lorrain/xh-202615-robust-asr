"""Preregister and train the DACF-v4 all-pairs relational mechanism.

The input is the provenance-locked DACF-v3 train/dev feature cache.  V4 does
not reuse the original C-only enrollment rows for optimisation.  Instead, the
two speakers present in every mixture form a shared bank inside each batch:
each query is positive for its own mixture and a foreign negative for every
other mixture.  Thus every optimised query receives both labels in the same
update and the v3 speaker-to-role shortcut is unavailable.

This script never reads a final split and never invokes Qwen decoding or CER.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dacf_v4_objective import DACFV4LossWeights, compute_dacf_v4_loss
from dacf_v4_relation import DACFV4Relation
from train_dacf_v3_mechanism import (
    CounterfactualGroup,
    FeatureCache,
    MechanismContractError,
    QueryRecord,
    load_feature_cache,
)


PROTOCOL_VERSION = "dacf-v4-allpairs-mechanism-v0.1"
PREREGISTRATION_SCHEMA = "dacf-v4-allpairs-preregistration-v0.1"
REPORT_SCHEMA = "dacf-v4-allpairs-report-v0.1"
THRESHOLD = 0.5

MECHANISM_GATE: Mapping[str, float] = {
    "presence_auc": 0.85,
    "activity_frame_auc": 0.80,
    "present_recall": 0.80,
    "foreign_rr": 0.95,
    "mixture_top2_exact": 0.75,
    "query_to_mixture_top1": 0.75,
    "query_response_mean": 0.20,
    "query_response_ci_lower": 0.05,
    "mixture_permutation_auc_drop": 0.20,
    "present_A_recall": 0.75,
    "present_B_recall": 0.75,
    "collapsed_query_auc_max": 0.55,
    "collapsed_mixture_auc_max": 0.55,
}


@dataclass(frozen=True)
class DACFV4TrainingConfig:
    seed: int = 2026080617
    epochs: int = 30
    groups_per_batch: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    hard_foreign_margin: float = 0.20
    bootstrap_replicates: int = 2000
    device: str = "auto"

    def validate(self, train_group_count: int) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("epochs", "groups_per_batch", "bootstrap_replicates"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.groups_per_batch < 2:
            raise ValueError("groups_per_batch must be at least two")
        if train_group_count % self.groups_per_batch:
            raise ValueError("train group count must divide evenly into fixed batches")
        for name in (
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
            "hard_foreign_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate <= 0 or self.grad_clip_norm <= 0:
            raise ValueError("learning rate and gradient clip norm must be positive")

    @property
    def optimiser_updates(self) -> int:
        # The formal cache is fixed to 96 train groups.
        return self.epochs * (96 // self.groups_per_batch)


@dataclass(frozen=True)
class AllPairsBatch:
    groups: tuple[CounterfactualGroup, ...]
    speaker_ids: tuple[str, ...]
    query_roles: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    embeddings_view1: np.ndarray
    embeddings_view2: np.ndarray
    presence_labels: np.ndarray
    activity_targets: tuple[np.ndarray, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _present_rows(group: CounterfactualGroup) -> tuple[QueryRecord, QueryRecord]:
    rows = tuple(row for row in group.rows if row.target_present)
    if len(rows) != 2:
        raise MechanismContractError(
            f"{group.split}/{group.group_id} must have exactly two present rows"
        )
    speakers = {row.speaker_id for row in rows}
    if speakers != set(group.mixture_speaker_ids):
        raise MechanismContractError(
            f"{group.split}/{group.group_id} present speakers disagree with mixture"
        )
    return rows


def build_allpairs_batch(groups: Sequence[CounterfactualGroup]) -> AllPairsBatch:
    """Create a shared A/B query bank and prove every query gets both labels."""

    selected = tuple(groups)
    if len(selected) < 2:
        raise MechanismContractError("all-pairs batch needs at least two mixtures")
    split_names = {group.split for group in selected}
    if len(split_names) != 1:
        raise MechanismContractError("all-pairs batch cannot cross train/dev splits")
    group_ids = [group.group_id for group in selected]
    if len(group_ids) != len(set(group_ids)):
        raise MechanismContractError("all-pairs batch repeats a mixture group")

    query_rows: list[QueryRecord] = []
    source_group_ids: list[str] = []
    for group in selected:
        for row in _present_rows(group):
            query_rows.append(row)
            source_group_ids.append(group.group_id)
    speaker_ids = [row.speaker_id for row in query_rows]
    query_roles = [row.role for row in query_rows]
    if len(speaker_ids) != len(set(speaker_ids)):
        raise MechanismContractError(
            "v4 v0.1 cache contract requires unique A/B speakers across groups"
        )

    labels = np.zeros((len(selected), len(query_rows)), dtype=np.float32)
    activity_targets: list[np.ndarray] = []
    for group_index, group in enumerate(selected):
        activity_by_speaker = {
            row.speaker_id: np.asarray(row.target_activity, dtype=np.float32)
            for row in _present_rows(group)
        }
        frames = int(group.mixture_features.shape[1])
        target = np.zeros((len(query_rows), frames), dtype=np.float32)
        for query_index, speaker in enumerate(speaker_ids):
            if speaker in activity_by_speaker:
                labels[group_index, query_index] = 1.0
                source_activity = activity_by_speaker[speaker]
                if source_activity.shape != (frames,):
                    raise MechanismContractError("activity timeline disagrees with mixture")
                target[query_index] = source_activity
        if int(labels[group_index].sum()) != 2:
            raise MechanismContractError("each mixture must have exactly two positive queries")
        activity_targets.append(target)

    # With unique speakers, every query must be positive once and foreign in
    # all remaining mixtures.  This is the central anti-q-only invariant.
    positive_counts = labels.sum(axis=0)
    negative_counts = len(selected) - positive_counts
    if not np.all(positive_counts == 1):
        raise MechanismContractError("every optimised query must be positive exactly once")
    if not np.all(negative_counts == len(selected) - 1):
        raise MechanismContractError("every optimised query must be foreign elsewhere")

    return AllPairsBatch(
        groups=selected,
        speaker_ids=tuple(speaker_ids),
        query_roles=tuple(query_roles),
        source_group_ids=tuple(source_group_ids),
        embeddings_view1=np.stack([row.embedding for row in query_rows]).astype(
            np.float32, copy=False
        ),
        embeddings_view2=np.stack(
            [row.embedding_view2 for row in query_rows]
        ).astype(np.float32, copy=False),
        presence_labels=labels,
        activity_targets=tuple(activity_targets),
    )


def epoch_group_batches(
    groups: Sequence[CounterfactualGroup],
    *,
    groups_per_batch: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[CounterfactualGroup, ...], ...]:
    ordered = list(groups)
    random.Random(seed + 1_000_003 * epoch).shuffle(ordered)
    if len(ordered) % groups_per_batch:
        raise MechanismContractError("fixed all-pairs schedule cannot emit a short batch")
    return tuple(
        tuple(ordered[index : index + groups_per_batch])
        for index in range(0, len(ordered), groups_per_batch)
    )


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.bool_).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive < 1 or negative < 1:
        raise MechanismContractError("AUC requires both classes")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        stop = cursor + 1
        while stop < values.size and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = 0.5 * ((cursor + 1) + stop)
        ranks[order[cursor:stop]] = rank
        cursor = stop
    rank_sum = float(ranks[truth].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise MechanismContractError("group bootstrap needs at least two finite values")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, data.size, size=data.size)
        draws[index] = float(data[sample].mean())
    return {
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
    }


def _device_for(config: DACFV4TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cache_contract(cache: FeatureCache) -> Mapping[str, Any]:
    report = cache.report
    if bool(report.get("dataset_a_used", True)):
        raise MechanismContractError("Dataset-A is forbidden")
    split_contract = report.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise MechanismContractError("cache report has no split_contract")
    if not bool(split_contract.get("final_deferred", False)):
        raise MechanismContractError("v4 requires a train/dev cache with final deferred")
    if tuple(split_contract.get("splits", ())) != ("train", "dev"):
        raise MechanismContractError("cache split_contract must bind train/dev only")
    if set(cache.groups) != {"train", "dev"}:
        raise MechanismContractError("v4 cache must expose only train and dev")
    if len(cache.groups["train"]) != 96 or len(cache.groups["dev"]) != 12:
        raise MechanismContractError("formal v4 cache must contain 96 train/12 dev groups")
    # Full-split construction proves the central label-rotation invariant
    # before either preregistration or training.
    train_allpairs = build_allpairs_batch(cache.groups["train"])
    dev_allpairs = build_allpairs_batch(cache.groups["dev"])
    return {
        "train_groups": 96,
        "dev_groups": 12,
        "train_queries": len(train_allpairs.speaker_ids),
        "dev_queries": len(dev_allpairs.speaker_ids),
        "train_query_positive_count": 1,
        "train_query_foreign_count": 95,
        "dev_query_positive_count": 1,
        "dev_query_foreign_count": 11,
        "c_only_queries_optimised": 0,
        "final_deferred": True,
        "dataset_a_used": False,
    }


def _source_hashes() -> Mapping[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "code" / "experiments" / "dacf_v4_relation.py",
        root / "code" / "experiments" / "dacf_v4_objective.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in paths
    }


def preregistration_payload(
    cache: FeatureCache,
    config: DACFV4TrainingConfig,
) -> Mapping[str, Any]:
    contract = _cache_contract(cache)
    config.validate(contract["train_groups"])
    model = DACFV4Relation()
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
        "pairing_contract": {
            "query_rows": "present A/B only",
            "same_query_both_labels_in_update": True,
            "query_positive_mixtures_per_batch": 1,
            "query_foreign_mixtures_per_batch": config.groups_per_batch - 1,
            "mixture_encoded_without_query": True,
            "activity_presence_input": "normalised key-query product and cosine only",
            "q_only_additive_path": False,
        "mixture_only_pair_head": False,
        },
        "training": asdict(config),
        "loss_weights": asdict(DACFV4LossWeights()),
        "threshold": THRESHOLD,
        "mechanism_gate": dict(MECHANISM_GATE),
        "model": {
            "class": "DACFV4Relation",
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
    cache = load_feature_cache(cache_root)
    payload = preregistration_payload(cache, DACFV4TrainingConfig())
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"preregistration output already exists: {path}")
    _json_write(path, payload)
    return {**payload, "preregistration_sha256": _sha256_file(path)}


def validate_preregistration(
    cache: FeatureCache,
    preregistration_path: str | Path,
    config: DACFV4TrainingConfig,
) -> Mapping[str, Any]:
    path = Path(preregistration_path).resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = preregistration_payload(cache, config)
    if value != expected:
        raise MechanismContractError(
            "external DACF-v4 preregistration does not match code/cache contract"
        )
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "validated": True,
    }


def _evaluate_view(
    model: DACFV4Relation,
    pair_batch: AllPairsBatch,
    *,
    embeddings: np.ndarray,
    device: torch.device,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    model.eval()
    query_bank = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    collapsed_bank = query_bank.mean(dim=0, keepdim=True).expand_as(query_bank)
    mixture_count = len(pair_batch.groups)
    query_count = len(pair_batch.speaker_ids)
    probabilities = np.empty((mixture_count, query_count), dtype=np.float64)
    collapsed_probabilities = np.empty_like(probabilities)
    activity_scores: list[np.ndarray] = []
    activity_labels: list[np.ndarray] = []

    with torch.no_grad():
        for mixture_index, group in enumerate(pair_batch.groups):
            mixture = torch.from_numpy(group.mixture_features).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            state = model.encode_mixture(mixture)
            output = model.score_queries(state, query_bank)
            collapsed = model.score_queries(state, collapsed_bank)
            probabilities[mixture_index] = torch.sigmoid(
                output.presence_logits[0]
            ).cpu().numpy()
            collapsed_probabilities[mixture_index] = torch.sigmoid(
                collapsed.presence_logits[0]
            ).cpu().numpy()
            positive = pair_batch.presence_labels[mixture_index] > 0.5
            activity_scores.append(
                output.activity_probability[0, positive].cpu().numpy().reshape(-1)
            )
            activity_labels.append(
                pair_batch.activity_targets[mixture_index][positive].reshape(-1)
            )

    labels = pair_batch.presence_labels.astype(bool)
    presence_auc = _binary_auc(labels, probabilities)
    collapsed_auc = _binary_auc(labels, collapsed_probabilities)
    # Collapse the mixture dimension while retaining one scalar per query.
    # Any q-only shortcut survives this operation; the role-rotation contract
    # requires its AUC to remain random because every query has both labels.
    collapsed_mixture_probabilities = np.broadcast_to(
        probabilities.mean(axis=0, keepdims=True), probabilities.shape
    )
    collapsed_mixture_auc = _binary_auc(labels, collapsed_mixture_probabilities)
    permuted_probabilities = np.roll(probabilities, shift=1, axis=0)
    permutation_auc = _binary_auc(labels, permuted_probabilities)
    predictions = probabilities >= THRESHOLD
    present_recall = float(predictions[labels].mean())
    foreign_rr = float((~predictions[~labels]).mean())
    responses = np.asarray(
        [
            float(probabilities[index][labels[index]].mean())
            - float(probabilities[index][~labels[index]].max())
            for index in range(mixture_count)
        ],
        dtype=np.float64,
    )
    top2_exact = np.mean(
        [
            set(np.argsort(probabilities[index])[-2:].tolist())
            == set(np.flatnonzero(labels[index]).tolist())
            for index in range(mixture_count)
        ]
    )
    source_group_index = {
        group.group_id: index for index, group in enumerate(pair_batch.groups)
    }
    expected_mixture = np.asarray(
        [source_group_index[group_id] for group_id in pair_batch.source_group_ids]
    )
    query_top1 = float((probabilities.argmax(axis=0) == expected_mixture).mean())
    role_array = np.asarray(pair_batch.query_roles)
    query_predictions = probabilities[expected_mixture, np.arange(query_count)] >= THRESHOLD
    role_recalls = {
        role: float(query_predictions[role_array == role].mean())
        for role in ("present_A", "present_B")
    }
    metrics: Mapping[str, Any] = {
        "pair_count": int(labels.size),
        "positive_pairs": int(labels.sum()),
        "foreign_pairs": int((~labels).sum()),
        "presence_auc": float(presence_auc),
        "activity_frame_auc": float(
            _binary_auc(
                np.concatenate(activity_labels) > 0.5,
                np.concatenate(activity_scores),
            )
        ),
        "present_recall": present_recall,
        "foreign_rr": foreign_rr,
        "mixture_top2_exact": float(top2_exact),
        "query_to_mixture_top1": query_top1,
        "query_response_mean": float(responses.mean()),
        "query_response_ci_95": _bootstrap_mean_ci(
            responses,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "mixture_permutation_auc": float(permutation_auc),
        "mixture_permutation_auc_drop": float(presence_auc - permutation_auc),
        "present_A_recall": role_recalls["present_A"],
        "present_B_recall": role_recalls["present_B"],
        "collapsed_query_auc": float(collapsed_auc),
        "collapsed_mixture_auc": float(collapsed_mixture_auc),
        "threshold": THRESHOLD,
    }
    raw = {
        "probabilities": probabilities,
        "collapsed_probabilities": collapsed_probabilities,
        "collapsed_mixture_probabilities": collapsed_mixture_probabilities,
        "labels": labels,
        "activity_scores": np.concatenate(activity_scores),
        "activity_labels": np.concatenate(activity_labels),
    }
    return metrics, raw


def _average_view_metrics(
    pair_batch: AllPairsBatch,
    left_raw: Mapping[str, Any],
    right_raw: Mapping[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Mapping[str, Any]:
    probabilities = 0.5 * (left_raw["probabilities"] + right_raw["probabilities"])
    collapsed = 0.5 * (
        left_raw["collapsed_probabilities"] + right_raw["collapsed_probabilities"]
    )
    collapsed_mixture = 0.5 * (
        left_raw["collapsed_mixture_probabilities"]
        + right_raw["collapsed_mixture_probabilities"]
    )
    labels = left_raw["labels"]
    presence_auc = _binary_auc(labels, probabilities)
    permutation_auc = _binary_auc(labels, np.roll(probabilities, 1, axis=0))
    predictions = probabilities >= THRESHOLD
    responses = np.asarray(
        [
            float(probabilities[index][labels[index]].mean())
            - float(probabilities[index][~labels[index]].max())
            for index in range(len(pair_batch.groups))
        ]
    )
    top2 = np.mean(
        [
            set(np.argsort(probabilities[index])[-2:].tolist())
            == set(np.flatnonzero(labels[index]).tolist())
            for index in range(len(pair_batch.groups))
        ]
    )
    group_index = {group.group_id: i for i, group in enumerate(pair_batch.groups)}
    expected = np.asarray([group_index[value] for value in pair_batch.source_group_ids])
    role_array = np.asarray(pair_batch.query_roles)
    query_predictions = probabilities[expected, np.arange(probabilities.shape[1])] >= THRESHOLD
    activity_scores = 0.5 * (
        left_raw["activity_scores"] + right_raw["activity_scores"]
    )
    return {
        "pair_count": int(labels.size),
        "positive_pairs": int(labels.sum()),
        "foreign_pairs": int((~labels).sum()),
        "presence_auc": float(presence_auc),
        "activity_frame_auc": float(
            _binary_auc(left_raw["activity_labels"] > 0.5, activity_scores)
        ),
        "present_recall": float(predictions[labels].mean()),
        "foreign_rr": float((~predictions[~labels]).mean()),
        "mixture_top2_exact": float(top2),
        "query_to_mixture_top1": float(
            (probabilities.argmax(axis=0) == expected).mean()
        ),
        "query_response_mean": float(responses.mean()),
        "query_response_ci_95": _bootstrap_mean_ci(
            responses,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "mixture_permutation_auc": float(permutation_auc),
        "mixture_permutation_auc_drop": float(presence_auc - permutation_auc),
        "present_A_recall": float(query_predictions[role_array == "present_A"].mean()),
        "present_B_recall": float(query_predictions[role_array == "present_B"].mean()),
        "collapsed_query_auc": float(_binary_auc(labels, collapsed)),
        "collapsed_mixture_auc": float(_binary_auc(labels, collapsed_mixture)),
        "threshold": THRESHOLD,
    }


def _gate(metrics_by_view: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for view_name in ("view1", "view2"):
        metrics = metrics_by_view[view_name]
        view_checks: dict[str, Any] = {}
        for name, threshold in MECHANISM_GATE.items():
            if name == "query_response_ci_lower":
                value = float(metrics["query_response_ci_95"]["lower"])
                relation = ">"
                ok = value > threshold
            elif name in {"collapsed_query_auc_max", "collapsed_mixture_auc_max"}:
                metric_name = (
                    "collapsed_query_auc"
                    if name == "collapsed_query_auc_max"
                    else "collapsed_mixture_auc"
                )
                value = float(metrics[metric_name])
                relation = "<="
                ok = value <= threshold
            else:
                value = float(metrics[name])
                relation = ">="
                ok = value >= threshold
            view_checks[name] = {
                "value": value,
                "threshold": threshold,
                "relation": relation,
                "passed": bool(ok),
            }
            passed = passed and bool(ok)
        checks[view_name] = view_checks
    return {
        "passed": passed,
        "gated_views": ["view1", "view2"],
        "fixed_gate": dict(MECHANISM_GATE),
        "checks": checks,
    }


def _evaluate_split(
    model: DACFV4Relation,
    groups: Sequence[CounterfactualGroup],
    *,
    device: torch.device,
    bootstrap_replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    pair_batch = build_allpairs_batch(groups)
    view1, raw1 = _evaluate_view(
        model,
        pair_batch,
        embeddings=pair_batch.embeddings_view1,
        device=device,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 11,
    )
    view2, raw2 = _evaluate_view(
        model,
        pair_batch,
        embeddings=pair_batch.embeddings_view2,
        device=device,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 29,
    )
    averaged = _average_view_metrics(
        pair_batch,
        raw1,
        raw2,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 47,
    )
    return {
        "view1": view1,
        "view2": view2,
        "view_averaged": averaged,
        "gate": _gate({"view1": view1, "view2": view2}),
    }


def run_training(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    preregistration_path: str | Path,
    config: DACFV4TrainingConfig | None = None,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    config = config or DACFV4TrainingConfig()
    cache = load_feature_cache(cache_root)
    contract = _cache_contract(cache)
    config.validate(int(contract["train_groups"]))
    prereg = validate_preregistration(cache, preregistration_path, config)

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _set_determinism(config.seed)
    device = _device_for(config)
    model = DACFV4Relation().to(device)
    if model.trainable_parameter_count() >= 2_000_000:
        raise MechanismContractError("DACF-v4 exceeds the preregistered parameter cap")
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_weights = DACFV4LossWeights()
    loss_trace: list[float] = []
    negative_pair_coverage: set[tuple[str, str]] = set()

    train_groups = cache.groups["train"]
    model.train()
    for epoch in range(config.epochs):
        batches = epoch_group_batches(
            train_groups,
            groups_per_batch=config.groups_per_batch,
            seed=config.seed,
            epoch=epoch,
        )
        for batch_groups in batches:
            pair_batch = build_allpairs_batch(batch_groups)
            for source_group in pair_batch.source_group_ids[::2]:
                for destination_group in (group.group_id for group in batch_groups):
                    if destination_group != source_group:
                        negative_pair_coverage.add((source_group, destination_group))
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
                out1 = model.score_queries(state, query1)
                out2 = model.score_queries(state, query2)
                loss = compute_dacf_v4_loss(
                    out1,
                    out2,
                    presence_labels=labels,
                    activity_targets=activity,
                    margin=config.hard_foreign_margin,
                    weights=loss_weights,
                )
                scaled = loss.total / len(batch_groups)
                scaled.backward()
                batch_loss += float(loss.total.detach().cpu()) / len(batch_groups)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimiser.step()
            loss_trace.append(batch_loss)

    checkpoint = output / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "DACFV4Relation",
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(config),
        },
        checkpoint,
    )
    checkpoint_before = _sha256_file(checkpoint)

    # Train is diagnostic only.  Dev remains a single observation after the
    # final checkpoint is frozen; neither result selects a checkpoint.
    train_metrics = _evaluate_split(
        model,
        train_groups,
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 10_000,
    )
    dev_evaluation_count = 1
    dev_metrics = _evaluate_split(
        model,
        cache.groups["dev"],
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 20_000,
    )
    checkpoint_after = _sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise MechanismContractError("checkpoint changed during frozen evaluation")

    total_directed_pairs = len(train_groups) * (len(train_groups) - 1)
    report: Mapping[str, Any] = {
        "schema": REPORT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "verdict": (
            "conditional-GO" if dev_metrics["gate"]["passed"] else "implementation-NO-GO"
        ),
        "verdict_scope": {
            "scope": "all-pairs identity mechanism only",
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
            "negative_source_destination_pairs_total": total_directed_pairs,
            "negative_pair_coverage": len(negative_pair_coverage) / total_directed_pairs,
        },
        "training": {
            "config": asdict(config),
            "loss_weights": asdict(loss_weights),
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
        "dev_evaluation_count": dev_evaluation_count,
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


__all__ = [
    "AllPairsBatch",
    "DACFV4TrainingConfig",
    "MECHANISM_GATE",
    "PROTOCOL_VERSION",
    "build_allpairs_batch",
    "epoch_group_batches",
    "preregistration_payload",
    "run_training",
    "validate_preregistration",
    "write_preregistration",
]


if __name__ == "__main__":
    raise SystemExit(main())
