"""Low-rank CAM++ cross-query matcher for DACF.

This module is deliberately smaller than a target-speaker extractor.  A
frozen CAM++ model provides two representations:

* one utterance-level enrollment embedding, where speaker identity is known
  to be discriminative; and
* mixture tokens immediately before CAM++ statistics pooling, where temporal
  information is still available.

The matcher learns only a low-rank bilinear alignment between those two
spaces.  It never receives an A/B/C role id.  Consequently, for a byte-identical
mixture, changing the enrollment embedding is the only way to change target
presence or activity predictions.

This is a mechanism component, not an integration claim.  It must pass the
speaker-disjoint A/B/C gate before its activity weights or query-aware tokens
are connected to Qwen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CrossQueryOutput:
    """Outputs of one enrollment query against one frozen mixture feature set."""

    frame_logits: Tensor
    presence_logits: Tensor
    activity_probability: Tensor
    query_aware_tokens: Tensor
    mixture_projection: Tensor
    query_projection: Tensor


def _validate_feature_tensor(value: Tensor, *, name: str, feature_dim: int) -> None:
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B, T, D], got {tuple(value.shape)}")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one batch item and frame")
    if value.shape[2] != feature_dim:
        raise ValueError(
            f"{name} feature dim must be {feature_dim}, got {value.shape[2]}"
        )


def resize_activity_targets(activity: Tensor, target_frames: int) -> Tensor:
    """Resize 10 ms activity targets to CAM++ token rate without hard thresholding."""

    if activity.ndim != 2:
        raise ValueError(
            f"activity must have shape [B, source_frames], got {tuple(activity.shape)}"
        )
    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    if activity.shape[1] < 1:
        raise ValueError("activity must contain at least one source frame")
    resized = F.interpolate(
        activity.float().unsqueeze(1),
        size=target_frames,
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return resized.clamp_(0.0, 1.0)


def class_balanced_bce(logits: Tensor, targets: Tensor) -> Tensor:
    """Binary cross entropy with equal aggregate mass for present/absent labels."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits/targets shape mismatch: {tuple(logits.shape)} != {tuple(targets.shape)}"
        )
    targets = targets.float()
    positive = targets.ge(0.5)
    negative = ~positive
    weights = torch.ones_like(targets)
    if positive.any() and negative.any():
        weights[positive] = 0.5 / positive.sum().float()
        weights[negative] = 0.5 / negative.sum().float()
        weights = weights * targets.numel()
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def counterfactual_margin_loss(
    presence_logits_by_role: Tensor, *, margin: float = 0.20
) -> Tensor:
    """Require present A/B queries to outrank absent C within each fixed mixture.

    ``presence_logits_by_role`` must be ordered ``[present_A, present_B,
    absent_C]`` along its last axis.  The function consumes predictions only;
    role ids are never model inputs.
    """

    if presence_logits_by_role.ndim != 2 or presence_logits_by_role.shape[1] != 3:
        raise ValueError(
            "presence_logits_by_role must have shape [groups, 3] in A/B/C order"
        )
    if margin <= 0:
        raise ValueError("margin must be positive")
    present_a, present_b, absent_c = presence_logits_by_role.unbind(dim=1)
    return 0.5 * (
        F.relu(margin - (present_a - absent_c)).mean()
        + F.relu(margin - (present_b - absent_c)).mean()
    )


class DACFCAMPPQueryMatcher(nn.Module):
    """Align frozen CAM++ enrollment identity with pre-pooling mixture tokens.

    The two CAM++ representations live in different spaces, so the matcher uses
    separate bias-free projections into one small metric space.  All temporal
    predictions are functions of the projected enrollment and mixture only.
    ``query_aware_tokens`` retains the original 512-D CAM++ token shape for a
    later zero-initialized ASR bridge.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 512,
        query_dim: int = 32,
        logit_scale: float = 10.0,
        top_fraction: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be at least 2")
        if query_dim < 2 or query_dim > feature_dim:
            raise ValueError("query_dim must be in [2, feature_dim]")
        if logit_scale <= 0:
            raise ValueError("logit_scale must be positive")
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")

        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.top_fraction = float(top_fraction)

        # Non-affine normalization removes the large common positive direction
        # introduced by the pre-pool ReLU without adding trainable shortcuts.
        self.mixture_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.enrollment_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.mixture_projection = nn.Linear(feature_dim, query_dim, bias=False)
        self.enrollment_projection = nn.Linear(feature_dim, query_dim, bias=False)
        self.register_buffer(
            "logit_scale", torch.tensor(float(logit_scale)), persistent=True
        )

        # One learned calibration offset is required because cosine zero is not
        # an operating threshold.  It is fitted on train only; evaluation keeps
        # the probability threshold fixed at 0.5.
        self.frame_bias = nn.Parameter(torch.zeros(()))

        # This path exposes target-conditioned tokens to the later Qwen bridge.
        # Zero initialization keeps the initial residual exactly zero while the
        # presence/activity matcher can already learn through the logits.
        self.query_residual = nn.Linear(query_dim, feature_dim, bias=False)
        nn.init.zeros_(self.query_residual.weight)

    def forward(
        self, mixture_tokens: Tensor, enrollment_embedding: Tensor
    ) -> CrossQueryOutput:
        _validate_feature_tensor(
            mixture_tokens, name="mixture_tokens", feature_dim=self.feature_dim
        )
        if enrollment_embedding.ndim != 2:
            raise ValueError(
                "enrollment_embedding must have shape [B, D], got "
                f"{tuple(enrollment_embedding.shape)}"
            )
        if enrollment_embedding.shape != (
            mixture_tokens.shape[0],
            self.feature_dim,
        ):
            raise ValueError(
                "enrollment_embedding shape must match mixture batch/feature dims: "
                f"expected {(mixture_tokens.shape[0], self.feature_dim)}, "
                f"got {tuple(enrollment_embedding.shape)}"
            )

        mixture = self.mixture_norm(mixture_tokens.float())
        enrollment = self.enrollment_norm(enrollment_embedding.float())
        mixture_projection = F.normalize(
            self.mixture_projection(mixture), dim=-1, eps=1e-6
        )
        query_projection = F.normalize(
            self.enrollment_projection(enrollment), dim=-1, eps=1e-6
        )

        similarity = torch.einsum("btd,bd->bt", mixture_projection, query_projection)
        frame_logits = self.logit_scale * similarity + self.frame_bias
        top_count = max(1, math.ceil(frame_logits.shape[1] * self.top_fraction))
        presence_logits = frame_logits.topk(top_count, dim=1).values.mean(dim=1)
        activity_probability = torch.sigmoid(frame_logits)

        residual_anchor = self.query_residual(query_projection).unsqueeze(1)
        query_aware_tokens = mixture_tokens.float() + (
            activity_probability.unsqueeze(-1) * residual_anchor
        )
        return CrossQueryOutput(
            frame_logits=frame_logits,
            presence_logits=presence_logits,
            activity_probability=activity_probability,
            query_aware_tokens=query_aware_tokens,
            mixture_projection=mixture_projection,
            query_projection=query_projection,
        )
