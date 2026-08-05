"""Mechanism-stage objective for DACF-v3 ECST.

The objective consumes model outputs and audit-only counterfactual group ids;
group ids never enter the model.  It deliberately excludes ASR/CER and the
diagnostic spectral mask.  A mechanism probe must first learn identity-aware
presence/activity on byte-identical A/B/C mixtures before any Qwen loss is
allowed to participate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from dacf_v3_ecst import DACFV3ECSTOutput


@dataclass(frozen=True)
class DACFV3LossWeights:
    presence: float = 1.0
    activity: float = 1.0
    counterfactual_margin: float = 0.5
    view_consistency: float = 0.1
    absent_latent_energy: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"loss weight {name} must be a non-negative number")


@dataclass(frozen=True)
class DACFV3Loss:
    total: Tensor
    components: Mapping[str, Tensor]


def resize_activity_targets(target: Tensor, frames: int) -> Tensor:
    """Resize binary activity to an ECST timeline with nearest semantics."""

    if not isinstance(target, Tensor) or target.ndim != 2:
        raise ValueError(
            "activity target must have shape [B,T], "
            f"got {getattr(target, 'shape', None)}"
        )
    if target.shape[0] < 1 or target.shape[1] < 1 or frames < 1:
        raise ValueError("activity target and output timeline must be non-empty")
    if not torch.is_floating_point(target):
        target = target.float()
    if not bool(torch.isfinite(target).all().item()):
        raise ValueError("activity target contains NaN or Inf")
    if bool(((target < 0) | (target > 1)).any().item()):
        raise ValueError("activity target must be in [0,1]")
    if target.shape[1] == frames:
        return target
    return F.interpolate(
        target.unsqueeze(1), size=frames, mode="nearest"
    ).squeeze(1)


def balanced_binary_bce(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary BCE with equal positive/negative mass per row or vector.

    Rows containing only one class remain well-defined: the observed class
    receives unit mean weight instead of manufacturing an absent class.
    """

    if logits.shape != labels.shape:
        raise ValueError(
            f"binary logits/labels shape mismatch: {tuple(logits.shape)} != {tuple(labels.shape)}"
        )
    if logits.ndim not in (1, 2):
        raise ValueError("balanced binary BCE accepts rank-1 or rank-2 tensors")
    labels = labels.to(device=logits.device, dtype=logits.dtype)
    if bool(((labels < 0) | (labels > 1)).any().item()):
        raise ValueError("binary labels must be in [0,1]")
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        labels = labels.unsqueeze(0)
        loss = loss.unsqueeze(0)
    positive = labels
    negative = 1.0 - labels
    positive_count = positive.sum(dim=1, keepdim=True)
    negative_count = negative.sum(dim=1, keepdim=True)
    has_both = (positive_count > 0) & (negative_count > 0)
    balanced_weight = positive / positive_count.clamp_min(1.0)
    balanced_weight = balanced_weight + negative / negative_count.clamp_min(1.0)
    single_weight = torch.full_like(loss, 1.0 / loss.shape[1])
    weight = torch.where(has_both, 0.5 * balanced_weight, single_weight)
    return (loss * weight).sum(dim=1).mean()


def _validate_output(output: DACFV3ECSTOutput, name: str) -> tuple[int, int]:
    if output.activity_logits.ndim != 2:
        raise ValueError(f"{name}.activity_logits must be [B,T]")
    batch, frames = output.activity_logits.shape
    expected = {
        "presence_logits": (batch,),
        "activity_probability": (batch, frames),
        "query_conditioned_frames": (batch, frames, 128),
    }
    for field, shape in expected.items():
        value = getattr(output, field)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name}.{field} must have shape {shape}, got {tuple(value.shape)}")
    return batch, frames


def _counterfactual_margin_loss(
    presence_logits: Tensor,
    presence_labels: Tensor,
    group_index: Tensor,
    margin: float,
) -> Tensor:
    probabilities = torch.sigmoid(presence_logits)
    losses: list[Tensor] = []
    for group in torch.unique(group_index, sorted=True):
        selected = group_index == group
        labels = presence_labels[selected]
        present = probabilities[selected][labels > 0.5]
        absent = probabilities[selected][labels <= 0.5]
        if present.numel() != 2 or absent.numel() != 1:
            raise ValueError(
                "each counterfactual group must contain exactly two present and one absent row"
            )
        losses.append(F.relu(margin - present + absent[0]).mean())
    if not losses:
        raise ValueError("at least one counterfactual group is required")
    return torch.stack(losses).mean()


def compute_dacf_v3_loss(
    main: DACFV3ECSTOutput,
    view2: DACFV3ECSTOutput,
    *,
    presence_labels: Tensor,
    activity_targets: Tensor,
    group_index: Tensor,
    margin: float = 0.20,
    weights: DACFV3LossWeights | None = None,
) -> DACFV3Loss:
    """Compute the fixed two-view counterfactual mechanism objective."""

    if margin < 0:
        raise ValueError("counterfactual margin must be non-negative")
    weights = weights or DACFV3LossWeights()
    batch, frames = _validate_output(main, "main")
    view_batch, view_frames = _validate_output(view2, "view2")
    if (view_batch, view_frames) != (batch, frames):
        raise ValueError("main and view2 output shapes must match")
    if tuple(presence_labels.shape) != (batch,):
        raise ValueError(f"presence_labels must have shape {(batch,)}")
    if tuple(group_index.shape) != (batch,):
        raise ValueError(f"group_index must have shape {(batch,)}")
    if group_index.dtype == torch.bool or torch.is_floating_point(group_index):
        raise TypeError("group_index must use an integer dtype")
    presence_labels = presence_labels.to(
        device=main.presence_logits.device, dtype=main.presence_logits.dtype
    )
    group_index = group_index.to(device=main.presence_logits.device)
    activity_targets = resize_activity_targets(
        activity_targets.to(device=main.activity_logits.device), frames
    ).to(dtype=main.activity_logits.dtype)

    presence = 0.5 * (
        balanced_binary_bce(main.presence_logits, presence_labels)
        + balanced_binary_bce(view2.presence_logits, presence_labels)
    )
    activity = 0.5 * (
        balanced_binary_bce(main.activity_logits, activity_targets)
        + balanced_binary_bce(view2.activity_logits, activity_targets)
    )
    counterfactual_margin = 0.5 * (
        _counterfactual_margin_loss(
            main.presence_logits, presence_labels, group_index, margin
        )
        + _counterfactual_margin_loss(
            view2.presence_logits, presence_labels, group_index, margin
        )
    )
    view_consistency = 0.5 * (
        F.mse_loss(main.activity_logits, view2.activity_logits)
        + F.mse_loss(main.presence_logits, view2.presence_logits)
    )
    absent = presence_labels <= 0.5
    if not bool(absent.any().item()):
        raise ValueError("counterfactual batch must include at least one absent row")
    absent_latent_energy = 0.5 * (
        main.query_conditioned_frames[absent].square().mean()
        + view2.query_conditioned_frames[absent].square().mean()
    )

    components = {
        "presence": presence,
        "activity": activity,
        "counterfactual_margin": counterfactual_margin,
        "view_consistency": view_consistency,
        "absent_latent_energy": absent_latent_energy,
    }
    total = (
        weights.presence * presence
        + weights.activity * activity
        + weights.counterfactual_margin * counterfactual_margin
        + weights.view_consistency * view_consistency
        + weights.absent_latent_energy * absent_latent_energy
    )
    return DACFV3Loss(total=total, components=components)


__all__ = [
    "DACFV3Loss",
    "DACFV3LossWeights",
    "balanced_binary_bce",
    "compute_dacf_v3_loss",
    "resize_activity_targets",
]
