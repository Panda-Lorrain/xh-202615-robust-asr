"""DACF-v4 relational identity front end.

The v3 ECST probe allowed the enrollment query to add FiLM bias directly to
the mixture representation.  Combined with a one-speaker/one-role protocol,
that path learned a speaker-to-label lookup table and failed on unseen
speakers.  V4 makes the identity decision relational by construction:

* the mixture is encoded once without seeing an enrollment query;
* the encoder emits L2-normalised frame keys;
* the enrollment encoder emits L2-normalised queries;
* activity and presence are functions only of key/query products and cosine;
* query-conditioned content is the mixture content multiplied by the
  relational activity probability, with no additive query-only path.

This module is mechanism-only.  It does not load Qwen, compute CER, or claim
that the overall DACF direction is validated.
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
class DACFV4MixtureState:
    """Query-independent mixture representation."""

    frame_hidden: Tensor
    frame_keys: Tensor
    content_frames: Tensor


@dataclass(frozen=True)
class DACFV4RelationOutput:
    """All-pairs query/mixture scores and gated content."""

    activity_logits: Tensor
    presence_logits: Tensor
    activity_probability: Tensor
    query_conditioned_frames: Tensor
    mixture_keys: Tensor
    query_embeddings: Tensor


def fixed_top25_presence(activity_logits: Tensor) -> Tensor:
    """Pool the highest 25 percent of frame logits for every pair."""

    if not isinstance(activity_logits, Tensor) or activity_logits.ndim != 3:
        raise ValueError(
            "activity_logits must have shape [B,Q,T], "
            f"got {getattr(activity_logits, 'shape', None)}"
        )
    if any(size < 1 for size in activity_logits.shape):
        raise ValueError("activity_logits must have non-empty B,Q,T dimensions")
    top_count = max(1, math.ceil(activity_logits.shape[-1] * 0.25))
    return activity_logits.topk(top_count, dim=-1).values.mean(dim=-1)


def _validate_float_tensor(
    value: Tensor,
    *,
    name: str,
    ranks: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim not in ranks:
        expected = "/".join(str(rank) for rank in ranks)
        raise ValueError(f"{name} must be rank {expected}, got {tuple(value.shape)}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating-point dtype")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


class _MixtureTemporalBlock(nn.Module):
    """Query-free depthwise temporal residual block."""

    def __init__(self, hidden_dim: int, dilation: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=hidden_dim,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )

    def forward(self, frames: Tensor) -> Tensor:
        residual = self.norm(frames).transpose(1, 2)
        residual = self.depthwise(residual)
        residual = self.pointwise(residual).transpose(1, 2)
        return frames + residual


class DACFV4Relation(nn.Module):
    """Encode a mixture once and score a bank of enrollment queries.

    ``enrollment_embeddings`` may be a shared bank ``[Q,512]`` or a
    per-mixture bank ``[B,Q,512]``.  No role, path, order, environment, or
    label input exists in this forward contract.
    """

    def __init__(
        self,
        *,
        stem_channels: int = 48,
        stem_out_channels: int = 64,
        hidden_dim: int = 128,
        relation_dim: int = 64,
        relation_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        for name, value in (
            ("stem_channels", stem_channels),
            ("stem_out_channels", stem_out_channels),
            ("hidden_dim", hidden_dim),
            ("relation_dim", relation_dim),
            ("relation_hidden_dim", relation_hidden_dim),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if stem_channels % 8 or stem_out_channels % 8:
            raise ValueError("stem channel counts must be divisible by 8")

        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                stem_channels,
                kernel_size=(5, 3),
                stride=(2, 1),
                padding=(2, 1),
                bias=False,
            ),
            nn.GroupNorm(8, stem_channels),
            nn.GELU(),
            nn.Conv2d(
                stem_channels,
                stem_out_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, stem_out_channels),
            nn.GELU(),
        )
        self.temporal_input = nn.Linear(stem_out_channels, hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            _MixtureTemporalBlock(hidden_dim, dilation)
            for dilation in (1, 2, 4, 8)
        )
        self.key_projection = nn.Linear(hidden_dim, relation_dim, bias=False)
        self.content_projection = nn.Linear(
            hidden_dim, QUERY_CONDITIONED_DIM, bias=False
        )

        self.query_projection = nn.Sequential(
            nn.LayerNorm(CAMPP_EMBEDDING_DIM),
            nn.Linear(CAMPP_EMBEDDING_DIM, relation_dim),
            nn.GELU(),
            nn.Linear(relation_dim, relation_dim, bias=False),
        )

        # Every input to this MLP is a key-query product.  Biases are disabled
        # so it cannot manufacture a query-independent pair score.  A single
        # global scalar bias calibrates the fixed 0.5 threshold.
        self.relation_mlp = nn.Sequential(
            nn.Linear(relation_dim, relation_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1, bias=False),
        )
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(6.0)))
        self.global_logit_bias = nn.Parameter(torch.zeros(()))

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def encode_mixture(self, mixture_logmel: Tensor) -> DACFV4MixtureState:
        """Create keys/content without accepting or consulting a query."""

        reference = next(self.parameters())
        _validate_float_tensor(
            mixture_logmel,
            name="mixture_logmel",
            ranks=(3,),
            device=reference.device,
        )
        if mixture_logmel.shape[0] < 1 or mixture_logmel.shape[2] < 1:
            raise ValueError("mixture_logmel B and T dimensions must be non-empty")
        if mixture_logmel.shape[1] != MIXTURE_MEL_BINS:
            raise ValueError("mixture_logmel must contain exactly 128 mel bins")

        features = self.stem(
            mixture_logmel.to(dtype=reference.dtype).unsqueeze(1)
        )
        frames = self.temporal_input(features.mean(dim=2).transpose(1, 2))
        for block in self.temporal_blocks:
            frames = block(frames)
        keys = F.normalize(self.key_projection(frames), dim=-1, eps=1.0e-6)
        content = self.content_projection(frames)
        return DACFV4MixtureState(
            frame_hidden=frames,
            frame_keys=keys,
            content_frames=content,
        )

    def encode_queries(
        self,
        enrollment_embeddings: Tensor,
        *,
        batch_size: int,
    ) -> Tensor:
        reference = next(self.parameters())
        _validate_float_tensor(
            enrollment_embeddings,
            name="enrollment_embeddings",
            ranks=(2, 3),
            device=reference.device,
        )
        if enrollment_embeddings.shape[-1] != CAMPP_EMBEDDING_DIM:
            raise ValueError("enrollment_embeddings must end with 512 features")
        if enrollment_embeddings.ndim == 2:
            if enrollment_embeddings.shape[0] < 1:
                raise ValueError("the shared enrollment query bank is empty")
            projected = self.query_projection(
                enrollment_embeddings.to(dtype=reference.dtype)
            )
            projected = projected.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            if enrollment_embeddings.shape[0] != batch_size:
                raise ValueError(
                    "per-mixture query bank batch does not match mixture batch"
                )
            if enrollment_embeddings.shape[1] < 1:
                raise ValueError("the per-mixture enrollment query bank is empty")
            projected = self.query_projection(
                enrollment_embeddings.to(dtype=reference.dtype)
            )
        return F.normalize(projected, dim=-1, eps=1.0e-6)

    def score_queries(
        self,
        mixture_state: DACFV4MixtureState,
        enrollment_embeddings: Tensor,
    ) -> DACFV4RelationOutput:
        """Score queries against an already encoded, query-free mixture."""

        if not isinstance(mixture_state, DACFV4MixtureState):
            raise TypeError("mixture_state must be DACFV4MixtureState")
        keys = mixture_state.frame_keys
        content = mixture_state.content_frames
        if keys.ndim != 3 or content.ndim != 3:
            raise ValueError("mixture state tensors must have shape [B,T,D]")
        if keys.shape[:2] != content.shape[:2]:
            raise ValueError("mixture key/content timelines disagree")
        if keys.shape[-1] != self.relation_dim:
            raise ValueError("mixture key dimension disagrees with model")
        if content.shape[-1] != QUERY_CONDITIONED_DIM:
            raise ValueError("mixture content dimension disagrees with model")

        queries = self.encode_queries(
            enrollment_embeddings, batch_size=keys.shape[0]
        )
        products = keys.unsqueeze(1) * queries.unsqueeze(2)
        cosine = products.sum(dim=-1)
        relation_residual = self.relation_mlp(products).squeeze(-1)
        logit_scale = self.logit_scale_log.exp().clamp(1.0, 30.0)
        activity_logits = (
            logit_scale * cosine + relation_residual + self.global_logit_bias
        )
        presence_logits = fixed_top25_presence(activity_logits)
        activity_probability = torch.sigmoid(activity_logits)
        query_conditioned_frames = (
            content.unsqueeze(1) * activity_probability.unsqueeze(-1)
        )
        return DACFV4RelationOutput(
            activity_logits=activity_logits,
            presence_logits=presence_logits,
            activity_probability=activity_probability,
            query_conditioned_frames=query_conditioned_frames,
            mixture_keys=keys,
            query_embeddings=queries,
        )

    def forward(
        self,
        mixture_logmel: Tensor,
        enrollment_embeddings: Tensor,
    ) -> DACFV4RelationOutput:
        state = self.encode_mixture(mixture_logmel)
        return self.score_queries(state, enrollment_embeddings)


__all__ = [
    "CAMPP_EMBEDDING_DIM",
    "DACFV4MixtureState",
    "DACFV4Relation",
    "DACFV4RelationOutput",
    "MIXTURE_MEL_BINS",
    "QUERY_CONDITIONED_DIM",
    "fixed_top25_presence",
]
