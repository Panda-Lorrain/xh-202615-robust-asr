"""DACF-v4 anchor-preserving relational front end.

The first all-pairs implementation removed the q-only label shortcut but its
learnable 512->64 query projection collapsed the healthy CAM++ geometry:
cross-view same-speaker cosine stayed high while different speakers were also
pushed close to one another.  This implementation removes that failure mode.

Enrollment queries are the raw, frozen 512-dimensional CAM++ embeddings after
L2 normalisation.  Only the mixture encoder is trained to emit frame keys in
that fixed coordinate system.  Pair logits are scaled cosine plus one global
calibration bias; there is no query projection, relation MLP, q-only branch,
or mixture-only pair head.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dacf_v4_relation import (
    CAMPP_EMBEDDING_DIM,
    DACFV4MixtureState,
    DACFV4RelationOutput,
    MIXTURE_MEL_BINS,
    QUERY_CONDITIONED_DIM,
    _MixtureTemporalBlock,
    _validate_float_tensor,
    fixed_top25_presence,
)


class DACFV4AnchorRelation(nn.Module):
    """Train mixture keys against the unmodified CAM++ enrollment manifold."""

    relation_dim = CAMPP_EMBEDDING_DIM

    def __init__(
        self,
        *,
        stem_channels: int = 48,
        stem_out_channels: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        for name, value in (
            ("stem_channels", stem_channels),
            ("stem_out_channels", stem_out_channels),
            ("hidden_dim", hidden_dim),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if stem_channels % 8 or stem_out_channels % 8:
            raise ValueError("stem channel counts must be divisible by 8")

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
        self.key_projection = nn.Linear(
            hidden_dim, CAMPP_EMBEDDING_DIM, bias=False
        )
        self.content_projection = nn.Linear(
            hidden_dim, QUERY_CONDITIONED_DIM, bias=False
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
        values = enrollment_embeddings.to(dtype=reference.dtype)
        if values.ndim == 2:
            if values.shape[0] < 1:
                raise ValueError("the shared enrollment query bank is empty")
            values = values.unsqueeze(0).expand(batch_size, -1, -1)
        elif values.shape[0] != batch_size:
            raise ValueError("per-mixture query bank batch does not match mixture batch")
        return F.normalize(values, dim=-1, eps=1.0e-6)

    def score_queries(
        self,
        mixture_state: DACFV4MixtureState,
        enrollment_embeddings: Tensor,
    ) -> DACFV4RelationOutput:
        if not isinstance(mixture_state, DACFV4MixtureState):
            raise TypeError("mixture_state must be DACFV4MixtureState")
        keys = mixture_state.frame_keys
        content = mixture_state.content_frames
        if keys.ndim != 3 or keys.shape[-1] != CAMPP_EMBEDDING_DIM:
            raise ValueError("mixture keys must have shape [B,T,512]")
        if content.shape != (*keys.shape[:2], QUERY_CONDITIONED_DIM):
            raise ValueError("mixture content must have shape [B,T,128]")
        queries = self.encode_queries(
            enrollment_embeddings, batch_size=keys.shape[0]
        )
        cosine = torch.einsum("btd,bqd->bqt", keys, queries)
        scale = self.logit_scale_log.exp().clamp(1.0, 30.0)
        activity_logits = scale * cosine + self.global_logit_bias
        presence_logits = fixed_top25_presence(activity_logits)
        probability = torch.sigmoid(activity_logits)
        conditioned = content.unsqueeze(1) * probability.unsqueeze(-1)
        return DACFV4RelationOutput(
            activity_logits=activity_logits,
            presence_logits=presence_logits,
            activity_probability=probability,
            query_conditioned_frames=conditioned,
            mixture_keys=keys,
            query_embeddings=queries,
        )

    def forward(
        self,
        mixture_logmel: Tensor,
        enrollment_embeddings: Tensor,
    ) -> DACFV4RelationOutput:
        return self.score_queries(
            self.encode_mixture(mixture_logmel), enrollment_embeddings
        )


__all__ = ["DACFV4AnchorRelation"]
