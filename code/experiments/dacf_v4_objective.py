"""All-pairs mechanism objective for DACF-v4.

Each optimisation batch contains several byte-identical counterfactual
mixtures.  The two present enrollments from every mixture form one shared
query bank.  A query is positive for the mixture containing its speaker and a
foreign negative for every other mixture in the batch.  Consequently the
same enrollment identity receives both labels within the same update and a
query-only speaker-to-label lookup cannot minimise the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from dacf_v4_relation import DACFV4RelationOutput


@dataclass(frozen=True)
class DACFV4LossWeights:
    presence: float = 1.0
    positive_activity: float = 1.0
    foreign_activity: float = 0.25
    hard_foreign_margin: float = 0.5
    view_consistency: float = 0.1
    absent_latent_energy: float = 0.02

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"loss weight {name} must be non-negative")


@dataclass(frozen=True)
class DACFV4Loss:
    total: Tensor
    components: Mapping[str, Tensor]


def _validate_output(output: DACFV4RelationOutput, name: str) -> tuple[int, int, int]:
    if not isinstance(output, DACFV4RelationOutput):
        raise TypeError(f"{name} must be DACFV4RelationOutput")
    if output.activity_logits.ndim != 3:
        raise ValueError(f"{name}.activity_logits must be [B,Q,T]")
    batch, queries, frames = output.activity_logits.shape
    expected = {
        "presence_logits": (batch, queries),
        "activity_probability": (batch, queries, frames),
        "query_conditioned_frames": (batch, queries, frames, 128),
    }
    for field, shape in expected.items():
        value = getattr(output, field)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"{name}.{field} must have shape {shape}, got {tuple(value.shape)}"
            )
    return batch, queries, frames


def _balanced_pair_bce(logits: Tensor, labels: Tensor) -> Tensor:
    """Give equal positive and foreign-negative mass to every mixture."""

    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    positive = labels > 0.5
    negative = ~positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    if bool((positive_count < 1).any().item()) or bool((negative_count < 1).any().item()):
        raise ValueError("every mixture must have both positive and foreign queries")
    positive_loss = (raw * positive).sum(dim=1) / positive_count
    negative_loss = (raw * negative).sum(dim=1) / negative_count
    return (0.5 * positive_loss + 0.5 * negative_loss).mean()


def _balanced_positive_activity(logits: Tensor, targets: Tensor) -> Tensor:
    """Balance active and inactive frames independently for every positive pair."""

    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    active = targets > 0.5
    inactive = ~active
    active_count = active.sum(dim=1)
    inactive_count = inactive.sum(dim=1)
    if bool((active_count < 1).any().item()) or bool((inactive_count < 1).any().item()):
        raise ValueError("positive activity targets need active and inactive frames")
    active_loss = (raw * active).sum(dim=1) / active_count
    inactive_loss = (raw * inactive).sum(dim=1) / inactive_count
    return (0.5 * active_loss + 0.5 * inactive_loss).mean()


def _hard_foreign_margin(
    presence_logits: Tensor,
    presence_labels: Tensor,
    margin: float,
) -> Tensor:
    losses: list[Tensor] = []
    for batch_index in range(presence_logits.shape[0]):
        labels = presence_labels[batch_index] > 0.5
        positives = presence_logits[batch_index][labels]
        negatives = presence_logits[batch_index][~labels]
        if positives.numel() < 1 or negatives.numel() < 1:
            raise ValueError("hard-foreign margin requires positives and negatives")
        hardest_foreign = negatives.max()
        losses.append(F.relu(margin - positives + hardest_foreign).mean())
    return torch.stack(losses).mean()


def compute_dacf_v4_loss(
    view1: DACFV4RelationOutput,
    view2: DACFV4RelationOutput,
    *,
    presence_labels: Tensor,
    activity_targets: Tensor,
    margin: float = 0.20,
    weights: DACFV4LossWeights | None = None,
) -> DACFV4Loss:
    """Compute the fixed two-view all-pairs relation objective."""

    if margin < 0:
        raise ValueError("margin must be non-negative")
    weights = weights or DACFV4LossWeights()
    shape1 = _validate_output(view1, "view1")
    shape2 = _validate_output(view2, "view2")
    if shape1 != shape2:
        raise ValueError("view outputs must have matching B,Q,T shapes")
    batch, queries, frames = shape1
    if tuple(presence_labels.shape) != (batch, queries):
        raise ValueError("presence_labels must have shape [B,Q]")
    if tuple(activity_targets.shape) != (batch, queries, frames):
        raise ValueError("activity_targets must have shape [B,Q,T]")

    device = view1.activity_logits.device
    dtype = view1.activity_logits.dtype
    labels = presence_labels.to(device=device, dtype=dtype)
    targets = activity_targets.to(device=device, dtype=dtype)
    if bool(((labels < 0) | (labels > 1)).any().item()):
        raise ValueError("presence labels must be binary probabilities")
    if bool(((targets < 0) | (targets > 1)).any().item()):
        raise ValueError("activity targets must be in [0,1]")
    present = labels > 0.5
    absent = ~present
    if bool((targets[absent] != 0).any().item()):
        raise ValueError("foreign-query activity targets must be exactly zero")

    presence = 0.5 * (
        _balanced_pair_bce(view1.presence_logits, labels)
        + _balanced_pair_bce(view2.presence_logits, labels)
    )
    positive_activity = 0.5 * (
        _balanced_positive_activity(
            view1.activity_logits[present], targets[present]
        )
        + _balanced_positive_activity(
            view2.activity_logits[present], targets[present]
        )
    )
    foreign_activity = 0.5 * (
        F.binary_cross_entropy_with_logits(
            view1.activity_logits[absent], targets[absent]
        )
        + F.binary_cross_entropy_with_logits(
            view2.activity_logits[absent], targets[absent]
        )
    )
    hard_foreign_margin = 0.5 * (
        _hard_foreign_margin(view1.presence_logits, labels, margin)
        + _hard_foreign_margin(view2.presence_logits, labels, margin)
    )
    view_consistency = 0.5 * (
        F.mse_loss(view1.activity_logits, view2.activity_logits)
        + F.mse_loss(view1.presence_logits, view2.presence_logits)
    )
    absent_latent_energy = 0.5 * (
        view1.query_conditioned_frames[absent].square().mean()
        + view2.query_conditioned_frames[absent].square().mean()
    )

    components = {
        "presence": presence,
        "positive_activity": positive_activity,
        "foreign_activity": foreign_activity,
        "hard_foreign_margin": hard_foreign_margin,
        "view_consistency": view_consistency,
        "absent_latent_energy": absent_latent_energy,
    }
    total = (
        weights.presence * presence
        + weights.positive_activity * positive_activity
        + weights.foreign_activity * foreign_activity
        + weights.hard_foreign_margin * hard_foreign_margin
        + weights.view_consistency * view_consistency
        + weights.absent_latent_energy * absent_latent_energy
    )
    return DACFV4Loss(total=total, components=components)


__all__ = [
    "DACFV4Loss",
    "DACFV4LossWeights",
    "compute_dacf_v4_loss",
]
