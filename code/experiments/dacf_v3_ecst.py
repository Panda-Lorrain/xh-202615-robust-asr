"""DACF-v3 ECST pure-model skeleton.

ECST (Early Conditional Spectro-Temporal) is a mechanism-only front end.  It
accepts the already computed 128-bin log-mel features used by the Qwen
processor and a frozen CAM++ enrollment embedding.  It does not compute a
waveform frontend, load a checkpoint, or connect to Qwen.

The enrollment query is injected immediately after the first time-frequency
convolution and again in every temporal residual block.  The mixture encoder
is trainable, while the enrollment embedding is treated as a frozen input
contract.  This is a pure model skeleton for a ``direction-unresolved``
research route; it makes no CER, integration, or GO claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


MIXTURE_MEL_BINS = 128
CAMPP_EMBEDDING_DIM = 512
QUERY_CONDITIONED_DIM = 128


@dataclass(frozen=True)
class DACFV3ECSTOutput:
    """Outputs consumed by future mechanism probes or a separate Qwen bridge."""

    activity_logits: Tensor
    presence_logits: Tensor
    activity_probability: Tensor
    query_conditioned_frames: Tensor
    diagnostic_spectral_mask: Tensor | None = None


def fixed_top25_presence(activity_logits: Tensor) -> Tensor:
    """Return the fixed top-25% activity-logit mean used for presence.

    ``ceil`` semantics keep the definition valid for very short timelines and
    ensure a one-frame input still produces one presence logit.
    """

    if not isinstance(activity_logits, Tensor) or activity_logits.ndim != 2:
        raise ValueError(
            "activity_logits must have shape [B, T], "
            f"got {getattr(activity_logits, 'shape', None)}"
        )
    if activity_logits.shape[0] < 1 or activity_logits.shape[1] < 1:
        raise ValueError("activity_logits must contain at least one batch item and frame")
    top_count = max(1, math.ceil(activity_logits.shape[1] * 0.25))
    return activity_logits.topk(top_count, dim=1).values.mean(dim=1)


def _validate_input_tensor(
    value: Tensor,
    *,
    name: str,
    ndim: int,
    device: torch.device,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be rank-{ndim}, got shape {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype, got {value.dtype}")
    if value.device != device:
        raise ValueError(
            f"{name} must be on the model device {device}, got {value.device}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


class _QueryTemporalResidualBlock(nn.Module):
    """Depthwise temporal residual block with enrollment FiLM conditioning."""

    def __init__(self, hidden_dim: int, query_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.query_film = nn.Linear(query_dim, hidden_dim * 2)

    def forward(self, frames: Tensor, query: Tensor) -> Tensor:
        residual = self.norm(frames).transpose(1, 2)
        residual = self.depthwise(residual)
        residual = self.pointwise(residual).transpose(1, 2)
        gamma, beta = self.query_film(query).unsqueeze(1).chunk(2, dim=-1)
        residual = residual * (1.0 + 0.5 * torch.tanh(gamma))
        residual = residual + 0.5 * torch.tanh(beta)
        return frames + residual


class DACFV3ECST(nn.Module):
    """Early query-conditioned spectro-temporal DACF-v3 model skeleton.

    Parameters are intentionally modest and all learnable.  The forward
    contract is exactly ``mixture_logmel [B,128,T]`` plus
    ``enrollment_embedding [B,512]``; there is no role, path, ordering, or
    environment input.  ``query_conditioned_frames`` is a separate projected
    latent source whose activity gate is multiplicative but not its only
    query-dependent path.
    """

    def __init__(
        self,
        *,
        stem_channels: int = 48,
        stem_out_channels: int = 64,
        hidden_dim: int = 128,
        query_dim: int = 64,
    ) -> None:
        super().__init__()
        for name, value in (
            ("stem_channels", stem_channels),
            ("stem_out_channels", stem_out_channels),
            ("hidden_dim", hidden_dim),
            ("query_dim", query_dim),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if stem_channels % 8 != 0 or stem_out_channels % 8 != 0:
            raise ValueError(
                "stem_channels and stem_out_channels must be divisible by 8 "
                "for the fixed GroupNorm contract"
            )

        self.stem_channels = stem_channels
        self.stem_out_channels = stem_out_channels
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim

        self.query_projection = nn.Sequential(
            nn.LayerNorm(CAMPP_EMBEDDING_DIM),
            nn.Linear(CAMPP_EMBEDDING_DIM, query_dim),
            nn.GELU(),
        )

        # Time stride is always one.  Frequency stride is two in both convs.
        # The first FiLM is deliberately the first operation after stem_conv.
        self.stem_conv = nn.Conv2d(
            1,
            stem_channels,
            kernel_size=(5, 3),
            stride=(2, 1),
            padding=(2, 1),
        )
        self.stem_query_film = nn.Linear(query_dim, stem_channels * 2)
        self.stem_norm = nn.GroupNorm(8, stem_channels)
        self.stem_out_conv = nn.Conv2d(
            stem_channels,
            stem_out_channels,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding=(1, 1),
        )
        self.stem_out_norm = nn.GroupNorm(8, stem_out_channels)
        self.diagnostic_mask_head = nn.Conv2d(stem_out_channels, 1, kernel_size=1)

        self.temporal_input = nn.Linear(stem_out_channels, hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            _QueryTemporalResidualBlock(hidden_dim, query_dim, dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.activity_head = nn.Linear(hidden_dim, 1)
        self.frame_projection = nn.Linear(hidden_dim, QUERY_CONDITIONED_DIM)

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters updated by an optimizer."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _validate_inputs(
        self, mixture_logmel: Tensor, enrollment_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        reference = next(self.parameters())
        _validate_input_tensor(
            mixture_logmel,
            name="mixture_logmel",
            ndim=3,
            device=reference.device,
        )
        _validate_input_tensor(
            enrollment_embedding,
            name="enrollment_embedding",
            ndim=2,
            device=reference.device,
        )
        if mixture_logmel.shape[0] < 1 or mixture_logmel.shape[2] < 1:
            raise ValueError(
                "mixture_logmel must have shape [B,128,T] with B,T >= 1, "
                f"got {tuple(mixture_logmel.shape)}"
            )
        if mixture_logmel.shape[1] != MIXTURE_MEL_BINS:
            raise ValueError(
                f"mixture_logmel must have 128 mel bins, got {mixture_logmel.shape[1]}"
            )
        if enrollment_embedding.shape[0] != mixture_logmel.shape[0]:
            raise ValueError(
                "mixture_logmel and enrollment_embedding batch sizes must match: "
                f"{mixture_logmel.shape[0]} != {enrollment_embedding.shape[0]}"
            )
        if enrollment_embedding.shape[1] != CAMPP_EMBEDDING_DIM:
            raise ValueError(
                "enrollment_embedding must have 512 features, "
                f"got {enrollment_embedding.shape[1]}"
            )
        return (
            mixture_logmel.to(dtype=reference.dtype),
            enrollment_embedding.to(dtype=reference.dtype),
        )

    @staticmethod
    def _apply_stem_film(features: Tensor, query: Tensor, film: nn.Linear) -> Tensor:
        gamma, beta = film(query).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return features * (1.0 + 0.5 * torch.tanh(gamma)) + 0.5 * torch.tanh(beta)

    def forward(
        self, mixture_logmel: Tensor, enrollment_embedding: Tensor
    ) -> DACFV3ECSTOutput:
        """Compute target-conditioned frames without waveform or Qwen calls."""

        mixture_logmel, enrollment_embedding = self._validate_inputs(
            mixture_logmel, enrollment_embedding
        )
        query = self.query_projection(enrollment_embedding)

        # [B,128,T] -> [B,1,128,T].  Time remains unstrided throughout the stem.
        features = self.stem_conv(mixture_logmel.unsqueeze(1))
        features = self._apply_stem_film(features, query, self.stem_query_film)
        features = F.gelu(self.stem_norm(features))
        features = F.gelu(self.stem_out_norm(self.stem_out_conv(features)))

        mask_logits = self.diagnostic_mask_head(features)
        diagnostic_spectral_mask = torch.sigmoid(
            F.interpolate(
                mask_logits,
                size=(MIXTURE_MEL_BINS, mixture_logmel.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        )

        # Frequency is pooled only after the frequency-downsampling stem;
        # temporal length is unchanged and remains variable.
        frames = features.mean(dim=2).transpose(1, 2)
        frames = self.temporal_input(frames)
        for block in self.temporal_blocks:
            frames = block(frames, query)

        activity_logits = self.activity_head(frames).squeeze(-1)
        presence_logits = fixed_top25_presence(activity_logits)
        activity_probability = torch.sigmoid(activity_logits)

        # This is an independent latent projection.  Activity gates it, but
        # the frame content still depends on the query-conditioned encoder.
        query_conditioned_frames = self.frame_projection(frames)
        query_conditioned_frames = (
            query_conditioned_frames * activity_probability.unsqueeze(-1)
        )
        return DACFV3ECSTOutput(
            activity_logits=activity_logits,
            presence_logits=presence_logits,
            activity_probability=activity_probability,
            query_conditioned_frames=query_conditioned_frames,
            diagnostic_spectral_mask=diagnostic_spectral_mask,
        )


# Keep both common spellings available while retaining one implementation.
DACFv3ECST = DACFV3ECST
ECSTOutput = DACFV3ECSTOutput


__all__ = [
    "CAMPP_EMBEDDING_DIM",
    "DACFV3ECST",
    "DACFV3ECSTOutput",
    "DACFv3ECST",
    "ECSTOutput",
    "MIXTURE_MEL_BINS",
    "QUERY_CONDITIONED_DIM",
    "fixed_top25_presence",
]
