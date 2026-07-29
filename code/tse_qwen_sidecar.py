#!/usr/bin/env python3
"""Frozen-Qwen encoder Sidecar used by the Phase-3 ASR-aware TSE POC.

The module deliberately does *not* alter Qwen's checkpointed weights.  It
inserts zero-initialised residual adapters after the first N audio encoder
layers and trains only those adapters plus a target-activity head.  The public
``FrozenQwenSidecar`` wrapper accepts the normal ``thinker`` inputs and adds
two training-only arguments:

``enrollment_embeddings``: CAM++ embeddings, ``[num_audios, 512]``.
``target_activity``: target-speech activity at any frame rate, one vector per
audio.  It is resampled inside the audio tower and is only used for BCE loss.

Qwen processes its audio tower one utterance at a time, even when the text
side is batched.  This wrapper mirrors that contract so enrollment conditions
remain paired with the correct audio.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SidecarConfig:
    # Qwen3-ASR-1.7B's checkpoint audio tower has 24 layers; use its first
    # third by default.  The wrapper still validates the live model layout.
    layers: int = 8
    rank: int = 64
    enrollment_dim: int = 512
    activity_loss_weight: float = 0.1


class _SidecarLayer(nn.Module):
    """A gated low-rank residual bias for one frozen Qwen encoder layer."""

    def __init__(self, hidden_size: int, config: SidecarConfig):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, config.rank, bias=False)
        self.enrollment = nn.Linear(config.enrollment_dim, config.rank, bias=False)
        self.up = nn.Linear(config.rank, hidden_size, bias=False)
        # Starting as an exact identity avoids an untrained adapter changing
        # baseline Qwen decoding.
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: Tensor, enrollment: Tensor, gate: Tensor) -> Tensor:
        condition = self.enrollment(enrollment).to(hidden.dtype)
        residual = self.up(F.silu(self.down(self.norm(hidden)) + condition))
        return hidden + gate.to(hidden.dtype) * residual


class _InjectedEncoderLayer(nn.Module):
    """Keeps a frozen Qwen layer intact and applies its Sidecar afterwards."""

    def __init__(self, base: nn.Module, sidecar: _SidecarLayer, owner: "FrozenQwenSidecar"):
        super().__init__()
        self.base = base
        self.sidecar = sidecar
        # Avoid registering the wrapper as a child module (which would create
        # a circular module graph through audio_tower.layers).
        object.__setattr__(self, "_owner", owner)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        output = self.base(hidden_states, *args, **kwargs)
        hidden = output[0]
        enrollment, gate = self._owner._condition_for(hidden)
        adapted = self.sidecar(hidden, enrollment, gate)
        return (adapted,) + tuple(output[1:])


class FrozenQwenSidecar(nn.Module):
    """Attach trainable Sidecar modules to a frozen Qwen3-ASR ``thinker``.

    ``thinker`` must expose ``audio_tower.layers`` as in qwen-asr 0.0.5.
    All pre-existing parameters are frozen here; construction fails early if
    the installed Qwen layout differs rather than silently training nothing.
    """

    def __init__(self, thinker: nn.Module, config: SidecarConfig = SidecarConfig()):
        super().__init__()
        if not hasattr(thinker, "audio_tower") or not hasattr(thinker.audio_tower, "layers"):
            raise TypeError("expected Qwen3-ASR thinker.audio_tower.layers")
        if config.layers < 1 or config.layers > len(thinker.audio_tower.layers):
            raise ValueError(f"layers must be in [1, {len(thinker.audio_tower.layers)}]")

        self.thinker = thinker
        self.config = config
        self.hidden_size = int(thinker.audio_tower.config.d_model)
        for parameter in self.thinker.parameters():
            parameter.requires_grad_(False)

        self.activity_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, max(32, config.rank)),
            nn.SiLU(),
            nn.Linear(max(32, config.rank), 1),
        )
        self.adapters = nn.ModuleList(
            [_SidecarLayer(self.hidden_size, config) for _ in range(config.layers)]
        )
        self._enrollment: Optional[Tensor] = None
        self._activity: Optional[Tensor] = None
        self._activity_logits: List[Tensor] = []
        self._audio_index = 0
        self._install_layers()
        self._install_audio_tower_context()

    def _install_layers(self) -> None:
        layers = self.thinker.audio_tower.layers
        for index in range(self.config.layers):
            layers[index] = _InjectedEncoderLayer(layers[index], self.adapters[index], self)

    def _install_audio_tower_context(self) -> None:
        tower = self.thinker.audio_tower
        original_forward = tower.forward

        def wrapped_forward(tower_self, *args, **kwargs):
            # get_audio_features invokes audio_tower once per utterance.
            try:
                return original_forward(*args, **kwargs)
            finally:
                self._audio_index += 1

        tower.forward = MethodType(wrapped_forward, tower)

    def _condition_for(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if self._enrollment is None:
            raise RuntimeError("Sidecar context missing; call FrozenQwenSidecar.forward")
        index = min(self._audio_index, self._enrollment.size(0) - 1)
        enrollment = self._enrollment[index : index + 1].to(hidden.device)
        logits = self.activity_head(hidden).squeeze(-1)
        gate = torch.sigmoid(logits).unsqueeze(-1)
        self._activity_logits.append(logits)
        return enrollment, gate

    @staticmethod
    def _activity_loss(logits: List[Tensor], activity: Optional[Tensor]) -> Tensor:
        if activity is None or not logits:
            return logits[0].new_zeros(()) if logits else torch.zeros(())
        # One activity label sequence is paired with every injected layer; the
        # average prevents the auxiliary loss scale growing with layer count.
        losses = []
        for index, logit in enumerate(logits):
            source = activity[
                min(index // max(1, len(logits) // activity.size(0)), activity.size(0) - 1)
            ].to(logit.device)
            target = F.interpolate(source.float()[None, None], size=logit.numel(), mode="nearest")[0, 0]
            losses.append(F.binary_cross_entropy_with_logits(logit.float(), target))
        return torch.stack(losses).mean()

    def forward(
        self,
        *args,
        enrollment_embeddings: Tensor,
        target_activity: Optional[Tensor] = None,
        **kwargs,
    ):
        if enrollment_embeddings.ndim != 2 or enrollment_embeddings.size(1) != self.config.enrollment_dim:
            raise ValueError(
                f"enrollment_embeddings must be [N,{self.config.enrollment_dim}], got {tuple(enrollment_embeddings.shape)}"
            )
        self._enrollment = enrollment_embeddings
        self._activity = target_activity
        self._activity_logits = []
        self._audio_index = 0
        output = self.thinker(*args, **kwargs)
        activity_loss = self._activity_loss(self._activity_logits, target_activity)
        total_loss = output.loss
        if total_loss is not None:
            total_loss = total_loss + self.config.activity_loss_weight * activity_loss.to(total_loss.device)
        output.loss = total_loss
        output.sidecar_activity_loss = activity_loss
        return output

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """A compact checkpoint containing only newly introduced parameters."""
        state = {
            f"activity_head.{name}": value.detach().cpu()
            for name, value in self.activity_head.state_dict().items()
        }
        state.update(
            {
                f"adapters.{name}": value.detach().cpu()
                for name, value in self.adapters.state_dict().items()
            }
        )
        return state

    @property
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
