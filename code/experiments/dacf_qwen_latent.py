"""D7 DACF -> frozen-Qwen hidden-state latent bridge.

This file only validates the interface and gradient contract.  It does not
load Qwen weights, run a processor, read Dataset-A, or connect to the default
submission chain.  The direction remains ``direction-unresolved`` and this
prototype must not be labelled ``integrate-GO``.

The bridge deliberately calls ``thinker.forward`` with autograd enabled.  A
DACF waveform branch is evaluated before the frozen thinker, and its
``speaker_query_frames`` are interpolated to each audio-tower hidden-state
timeline before a zero-initialised low-rank residual is added after the first
N audio layers.  The real Qwen batch=1 smoke is still a separate gate.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DACFBridgeConfig:
    """Configuration for the latent bridge.

    ``audio_layout='auto'`` accepts the layouts observed in local Qwen-style
    towers: one utterance as ``[T, H]``, ``[1, T, H]`` or ``[T, 1, H]``; a
    genuinely batched tower may use ``[B, T, H]`` or ``[T, B, H]``.  Ambiguous
    layouts and call-count mismatches are errors rather than an opportunity to
    reuse the wrong speaker query.
    """

    layers: int = 8
    rank: int = 64
    dacf_dim: Optional[int] = None
    presence_loss_weight: float = 0.1
    activity_loss_weight: float = 0.1
    audio_layout: str = "auto"


def _balanced_binary_cross_entropy_with_logits(
    logits: Tensor, target: Tensor
) -> Tensor:
    """Give present/absent (or active/inactive) labels equal aggregate weight."""

    target = target.to(device=logits.device, dtype=logits.dtype)
    flat = target.reshape(-1)
    positives = flat.sum()
    negatives = flat.numel() - positives
    per_item = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none"
    )
    if bool((positives > 0).item()) and bool((negatives > 0).item()):
        total = float(flat.numel())
        positive_weight = total / (2.0 * positives)
        negative_weight = total / (2.0 * negatives)
        weights = torch.where(target > 0.5, positive_weight, negative_weight)
        return (per_item * weights).mean()
    return per_item.mean()


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _require_tensor(value: Any, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    return value


def _waveform_batch(value: Tensor, name: str) -> int:
    value = _require_tensor(value, name)
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError(
                f"{name} must be mono when 3-D, got {tuple(value.shape)}"
            )
        return int(value.shape[0])
    if value.ndim != 2 or value.shape[0] < 1 or value.shape[1] < 2:
        raise ValueError(
            f"{name} must have shape [B,T] or [B,1,T], got {tuple(value.shape)}"
        )
    return int(value.shape[0])


def _infer_hidden_size(thinker: nn.Module) -> int:
    tower = thinker.audio_tower
    config = getattr(tower, "config", None)
    for owner in (config, tower):
        for name in ("d_model", "hidden_size", "embed_dim"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return int(value)

    for layer in tower.layers:
        for module in layer.modules():
            for name in ("out_features", "normalized_shape"):
                value = getattr(module, name, None)
                if name == "normalized_shape" and isinstance(value, tuple):
                    value = value[-1]
                if isinstance(value, int) and value > 0:
                    return int(value)
    raise TypeError(
        "cannot infer audio hidden size; expose thinker.audio_tower.config.d_model "
        "or pass a fake tower with an equivalent hidden-size attribute"
    )


def _infer_dacf_dim(dacf: nn.Module, config: DACFBridgeConfig) -> int:
    if config.dacf_dim is not None:
        if config.dacf_dim < 1:
            raise ValueError("dacf_dim must be positive")
        return int(config.dacf_dim)
    for owner in (dacf, getattr(dacf, "config", None)):
        for name in ("d_model", "speaker_dim", "hidden_size"):
            value = getattr(owner, name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    raise TypeError(
        "cannot infer DACF speaker_query_frames width; pass config.dacf_dim"
    )


def _interpolate_time(frames: Tensor, target_length: int, name: str) -> Tensor:
    """Interpolate ``[B,T,D]`` explicitly while preserving autograd."""

    if frames.ndim != 3:
        raise ValueError(f"{name} must be [B,T,D], got {tuple(frames.shape)}")
    if frames.shape[1] < 1 or target_length < 1:
        raise ValueError(
            f"{name} and target timeline must be non-empty: "
            f"source={tuple(frames.shape)}, target={target_length}"
        )
    # F.interpolate works over the final temporal axis for [B,C,T].  No
    # detach/NumPy conversion is allowed in this bridge.
    return F.interpolate(
        frames.transpose(1, 2),
        size=int(target_length),
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


class _LowRankLatentResidual(nn.Module):
    """Map DACF speaker frames to a Qwen hidden residual."""

    def __init__(self, dacf_dim: int, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dacf_dim)
        self.down = nn.Linear(dacf_dim, rank)
        self.up = nn.Linear(rank, hidden_size)
        # This scalar is the identity-preserving gate.  The projection itself
        # stays non-zero so the gate can receive a useful first-step gradient.
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, speaker_frames: Tensor) -> Tensor:
        reference = self.norm.weight
        speaker_frames = speaker_frames.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        latent = F.silu(self.down(self.norm(speaker_frames)))
        residual = self.up(latent)
        return residual * self.gate.to(dtype=residual.dtype, device=residual.device)


def _layer_hidden(output: Any) -> tuple[Tensor, str]:
    if isinstance(output, Tensor):
        return output, "tensor"
    if isinstance(output, (tuple, list)):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("audio layer output must put hidden states at index 0")
        return output[0], "sequence"
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if isinstance(value, Tensor):
            return value, name
    raise TypeError(
        "unsupported audio layer output; expected Tensor, tuple/list, "
        "or an object with last_hidden_state"
    )


def _replace_layer_hidden(output: Any, hidden: Tensor, kind: str) -> Any:
    if kind == "tensor":
        return hidden
    if kind == "sequence":
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return [hidden] + list(output[1:])
    try:
        setattr(output, kind, hidden)
        return output
    except (AttributeError, TypeError):
        if hasattr(output, "_replace"):
            return output._replace(**{kind: hidden})
        raise TypeError(f"cannot replace hidden states in {type(output).__name__}")


class _InjectedAudioLayer(nn.Module):
    """Run a frozen layer, then inject one DACF residual."""

    def __init__(
        self,
        base: nn.Module,
        bridge: _LowRankLatentResidual,
        owner: "FrozenQwenDACFLatent",
    ) -> None:
        super().__init__()
        self.base = base
        # Bridge modules are owned by the top-level wrapper.  Keeping only an
        # unregistered reference here avoids presenting trainable bridge
        # weights as frozen thinker weights through thinker.named_parameters().
        object.__setattr__(self, "_bridge", bridge)
        object.__setattr__(self, "_owner", owner)

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        output = self.base(hidden_states, *args, **kwargs)
        hidden, kind = _layer_hidden(output)
        adapted = hidden + self._owner._residual_for_hidden(hidden, self._bridge)
        return _replace_layer_hidden(output, adapted, kind)


class _OutputProxy:
    """Fallback for immutable thinker outputs such as plain tuples."""

    def __init__(self, base: Any, fields: Mapping[str, Any]) -> None:
        self.base_output = base
        self._fields = dict(fields)
        for name, value in fields.items():
            setattr(self, name, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_output, name)

    def __getitem__(self, key: Any) -> Any:
        if key in self._fields:
            return self._fields[key]
        return self.base_output[key]


class FrozenQwenDACFLatent(nn.Module):
    """Attach a trainable DACF latent bridge to a frozen Qwen thinker.

    ``thinker`` must expose ``audio_tower.layers`` and a callable
    ``audio_tower.forward``.  The normal thinker inputs remain untouched and
    are passed through as ``*thinker_args``/``**thinker_inputs``.  The two raw
    waveforms are intentionally separate arguments because the processor path
    is outside this interface and must not turn the DACF branch into NumPy.
    """

    def __init__(
        self,
        thinker: nn.Module,
        dacf: nn.Module,
        config: DACFBridgeConfig = DACFBridgeConfig(),
    ) -> None:
        super().__init__()
        tower = getattr(thinker, "audio_tower", None)
        layers = getattr(tower, "layers", None)
        if tower is None or layers is None:
            raise TypeError("expected thinker.audio_tower.layers")
        if not isinstance(layers, (nn.ModuleList, list, tuple)):
            raise TypeError("thinker.audio_tower.layers must be indexable")
        if config.layers < 1 or config.layers > len(layers):
            raise ValueError(
                f"layers must be in [1, {len(layers)}], got {config.layers}"
            )
        if config.rank < 1:
            raise ValueError("rank must be positive")
        if config.audio_layout not in {"auto", "utterance", "batch_first", "time_first"}:
            raise ValueError(
                "audio_layout must be one of auto, utterance, batch_first, time_first"
            )
        if config.presence_loss_weight < 0 or config.activity_loss_weight < 0:
            raise ValueError("auxiliary loss weights must be non-negative")

        self.thinker = thinker
        self.dacf = dacf
        self.config = config
        self.hidden_size = _infer_hidden_size(thinker)
        self.dacf_dim = _infer_dacf_dim(dacf, config)

        self._frozen_thinker_parameters = tuple(thinker.parameters())
        for parameter in self._frozen_thinker_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None

        self.bridges = nn.ModuleList(
            [
                _LowRankLatentResidual(self.dacf_dim, self.hidden_size, config.rank)
                for _ in range(config.layers)
            ]
        )
        reference = next(thinker.parameters(), None)
        if reference is not None:
            self.bridges.to(device=reference.device, dtype=reference.dtype)

        self._dacf_output: Any = None
        self._speaker_query_frames: Optional[Tensor] = None
        self._active_batch_size = 0
        self._audio_call_count = 0
        self._active_audio_index: Optional[int] = None
        self._resolved_audio_layout: Optional[str] = None

        # Construction mutates the live thinker just like the existing
        # Sidecar prototype.  Preserve exact originals so same-process paired
        # baselines and repeated experiments can restore the backbone safely.
        object.__setattr__(
            self,
            "_original_audio_layers",
            tuple(tower.layers[index] for index in range(config.layers)),
        )
        self._restored = False

        self._install_layers()
        self._original_audio_tower_forward = tower.forward
        self._install_audio_tower_context()
        # Keep the thinker frozen/eval while allowing DACF and the bridge to
        # train.  This also avoids dropout changing the baseline comparison.
        self.train(True)

    def _install_layers(self) -> None:
        layers = self.thinker.audio_tower.layers
        for index in range(self.config.layers):
            base = layers[index]
            layers[index] = _InjectedAudioLayer(base, self.bridges[index], self)

    def _install_audio_tower_context(self) -> None:
        tower = self.thinker.audio_tower
        original_forward = self._original_audio_tower_forward

        def wrapped_forward(tower_self: nn.Module, *args: Any, **kwargs: Any) -> Any:
            if self._speaker_query_frames is None:
                raise RuntimeError(
                    "audio tower called without DACF context; use the wrapper forward"
                )
            if self._active_audio_index is not None:
                raise RuntimeError("nested audio-tower calls are not supported")
            if self._audio_call_count >= self._active_batch_size:
                raise ValueError(
                    "Qwen audio-tower invoked more times than DACF waveform batch; "
                    f"calls already={self._audio_call_count}, "
                    f"dacf_batch={self._active_batch_size}"
                )
            self._active_audio_index = self._audio_call_count
            try:
                return original_forward(*args, **kwargs)
            finally:
                self._audio_call_count += 1
                self._active_audio_index = None

        object.__setattr__(self, "_wrapped_audio_tower_forward_func", wrapped_forward)
        tower.forward = MethodType(wrapped_forward, tower)

    def restore(self) -> None:
        """Remove this bridge without clobbering a newer wrapper.

        This is intentionally explicit rather than relying on ``__del__`` so
        paired experiments can deterministically return to the exact frozen
        thinker in the same process.
        """

        if self._restored:
            return
        tower = self.thinker.audio_tower
        current_forward = tower.forward
        if (
            getattr(current_forward, "__self__", None) is not tower
            or getattr(current_forward, "__func__", None)
            is not self._wrapped_audio_tower_forward_func
        ):
            raise RuntimeError(
                "cannot restore DACF bridge: audio_tower.forward was replaced "
                "by another owner"
            )
        for index, original in enumerate(self._original_audio_layers):
            current = tower.layers[index]
            if not isinstance(current, _InjectedAudioLayer) or current._owner is not self:
                raise RuntimeError(
                    "cannot restore DACF bridge: an injected audio layer was "
                    f"replaced at index {index}"
                )
            tower.layers[index] = original
        tower.forward = self._original_audio_tower_forward
        self._restored = True

    def _current_query(self) -> Tensor:
        if self._speaker_query_frames is None:
            raise RuntimeError("DACF speaker-query context is not active")
        if self._active_audio_index is None:
            raise RuntimeError("audio layer called outside an audio-tower invocation")
        index = self._active_audio_index
        if index >= self._speaker_query_frames.shape[0]:
            raise ValueError(
                "audio call index exceeds DACF speaker-query batch: "
                f"index={index}, batch={self._speaker_query_frames.shape[0]}"
            )
        return self._speaker_query_frames[index : index + 1]

    def _resolve_layout(self, hidden: Tensor) -> str:
        if hidden.ndim not in (2, 3):
            raise ValueError(
                "audio hidden states must be [T,H], [B,T,H], or [T,B,H]; "
                f"got {tuple(hidden.shape)}"
            )
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "audio hidden width mismatch: "
                f"hidden={tuple(hidden.shape)}, expected H={self.hidden_size}"
            )
        if hidden.ndim == 2:
            candidate = "utterance"
        elif self.config.audio_layout != "auto":
            if self.config.audio_layout == "utterance":
                raise ValueError(
                    "audio_layout='utterance' requires rank-2 [T,H] hidden states; "
                    f"got {tuple(hidden.shape)}"
                )
            candidate = self.config.audio_layout
        else:
            batch = self._active_batch_size
            if batch > 1 and (
                (hidden.shape[0] == batch and hidden.shape[1] in {1, batch})
                or (hidden.shape[1] == batch and hidden.shape[0] in {1, batch})
            ):
                raise ValueError(
                    "ambiguous audio hidden layout; set audio_layout explicitly "
                    f"for hidden={tuple(hidden.shape)}, DACF batch={batch}"
                )
            if hidden.shape[0] == 1 and hidden.shape[1] > 1:
                candidate = "utterance_batch_first"
            elif hidden.shape[1] == 1 and hidden.shape[0] > 1:
                candidate = "utterance_time_first"
            elif (
                self._audio_call_count == 0
                and self._active_audio_index == 0
                and hidden.shape[0] == batch
                and hidden.shape[1] > 1
            ):
                candidate = "batch_first"
            elif (
                self._audio_call_count == 0
                and self._active_audio_index == 0
                and hidden.shape[1] == batch
                and hidden.shape[0] > 1
            ):
                candidate = "time_first"
            else:
                raise ValueError(
                    "ambiguous/unsupported audio hidden layout; expected one "
                    "utterance per call or an explicit batched [B,T,H]/[T,B,H], "
                    f"got {tuple(hidden.shape)} for DACF batch={batch}"
                )

        if self._resolved_audio_layout is None:
            self._resolved_audio_layout = candidate
        elif self._resolved_audio_layout != candidate:
            raise ValueError(
                "audio hidden layout changed within one thinker forward: "
                f"{self._resolved_audio_layout} -> {candidate}"
            )
        return candidate

    def _residual_for_hidden(
        self, hidden: Tensor, bridge: _LowRankLatentResidual
    ) -> Tensor:
        layout = self._resolve_layout(hidden)
        if layout == "utterance":
            source = self._current_query()
            aligned = _interpolate_time(source, hidden.shape[0], "speaker_query_frames")
            residual = bridge(aligned).squeeze(0)
        elif layout == "utterance_batch_first":
            source = self._current_query()
            aligned = _interpolate_time(source, hidden.shape[1], "speaker_query_frames")
            residual = bridge(aligned)
        elif layout == "utterance_time_first":
            source = self._current_query()
            aligned = _interpolate_time(source, hidden.shape[0], "speaker_query_frames")
            residual = bridge(aligned).transpose(0, 1)
        elif layout == "batch_first":
            if self._active_audio_index != 0 or self._audio_call_count != 0:
                raise ValueError("batched audio layout must be used in the first tower call")
            if hidden.shape[0] != self._active_batch_size:
                raise ValueError(
                    "batch-first audio hidden batch mismatch: "
                    f"hidden={hidden.shape[0]}, dacf={self._active_batch_size}"
                )
            aligned = _interpolate_time(
                self._speaker_query_frames,
                hidden.shape[1],
                "speaker_query_frames",
            )
            residual = bridge(aligned)
        elif layout == "time_first":
            if self._active_audio_index != 0 or self._audio_call_count != 0:
                raise ValueError("batched audio layout must be used in the first tower call")
            if hidden.shape[1] != self._active_batch_size:
                raise ValueError(
                    "time-first audio hidden batch mismatch: "
                    f"hidden={hidden.shape[1]}, dacf={self._active_batch_size}"
                )
            aligned = _interpolate_time(
                self._speaker_query_frames,
                hidden.shape[0],
                "speaker_query_frames",
            )
            residual = bridge(aligned).transpose(0, 1)
        else:  # pragma: no cover - guarded by _resolve_layout
            raise AssertionError(f"unknown audio layout {layout}")

        if residual.shape != hidden.shape:
            raise RuntimeError(
                "latent residual shape mismatch after explicit alignment: "
                f"residual={tuple(residual.shape)}, hidden={tuple(hidden.shape)}"
            )
        return residual.to(device=hidden.device, dtype=hidden.dtype)

    def _validate_dacf_output(self, output: Any, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        query = _require_tensor(_field(output, "speaker_query_frames"), "DACF speaker_query_frames")
        present = _require_tensor(
            _field(output, "target_present_logits"),
            "DACF target_present_logits",
        )
        activity = _require_tensor(
            _field(output, "target_activity_logits"),
            "DACF target_activity_logits",
        )
        if query.ndim != 3 or query.shape[0] != batch_size or query.shape[2] != self.dacf_dim:
            raise ValueError(
                "DACF speaker_query_frames must be [B,T,D] with "
                f"B={batch_size}, D={self.dacf_dim}; got {tuple(query.shape)}"
            )
        if present.ndim == 2 and present.shape[1] == 1:
            present = present.squeeze(1)
        if present.ndim != 1 or present.shape[0] != batch_size:
            raise ValueError(
                "DACF target_present_logits must be [B] or [B,1], got "
                f"{tuple(present.shape)}"
            )
        if activity.ndim != 2 or activity.shape[0] != batch_size:
            raise ValueError(
                "DACF target_activity_logits must be [B,T], got "
                f"{tuple(activity.shape)}"
            )
        if not torch.is_grad_enabled() and self.training:
            raise RuntimeError(
                "training FrozenQwenDACFLatent.forward requires grad-enabled "
                "thinker.forward; do not wrap it in torch.no_grad()"
            )
        if self.training and not query.requires_grad:
            raise RuntimeError(
                "DACF speaker_query_frames is detached or was produced under "
                "torch.no_grad(); waveform autograd contract is broken"
            )
        return query, present, activity

    @staticmethod
    def _presence_loss(logits: Tensor, target: Optional[Tensor]) -> Tensor:
        if target is None:
            return logits.sum() * 0.0
        target = _require_tensor(target, "target_present")
        if target.ndim == 2 and target.shape[1] == 1:
            target = target.squeeze(1)
        if target.shape != logits.shape:
            raise ValueError(
                "target_present shape must match DACF presence logits: "
                f"target={tuple(target.shape)}, logits={tuple(logits.shape)}"
            )
        return _balanced_binary_cross_entropy_with_logits(logits, target)

    @staticmethod
    def _activity_loss(logits: Tensor, target: Optional[Tensor]) -> Tensor:
        if target is None:
            return logits.sum() * 0.0
        target = _require_tensor(target, "target_activity")
        if target.ndim == 3 and target.shape[1] == 1:
            target = target[:, 0]
        if target.ndim != 2 or target.shape[0] != logits.shape[0] or target.shape[1] < 1:
            raise ValueError(
                "target_activity must be [B,T] or [B,1,T], got "
                f"{tuple(target.shape)}"
            )
        aligned = _interpolate_time(
            target.float().to(logits.device).unsqueeze(-1),
            logits.shape[1],
            "target_activity",
        ).squeeze(-1)
        return _balanced_binary_cross_entropy_with_logits(logits, aligned)

    @staticmethod
    def _attach_output(output: Any, fields: Mapping[str, Any]) -> Any:
        attached = True
        for name, value in fields.items():
            try:
                setattr(output, name, value)
            except (AttributeError, TypeError):
                attached = False
            if isinstance(output, MutableMapping):
                output[name] = value
        return output if attached or isinstance(output, MutableMapping) else _OutputProxy(output, fields)

    def forward(
        self,
        mixture_waveform: Tensor,
        enrollment_waveform: Tensor,
        *thinker_args: Any,
        target_present: Optional[Tensor] = None,
        target_activity: Optional[Tensor] = None,
        **thinker_inputs: Any,
    ) -> Any:
        """Run DACF and a normal differentiable frozen-thinker forward."""

        if self._restored:
            raise RuntimeError("DACF bridge was restored and can no longer run")

        mixture_batch = _waveform_batch(mixture_waveform, "mixture_waveform")
        enrollment_batch = _waveform_batch(enrollment_waveform, "enrollment_waveform")
        if mixture_batch != enrollment_batch:
            raise ValueError(
                "mixture_waveform and enrollment_waveform batch sizes must match: "
                f"{mixture_batch} != {enrollment_batch}"
            )

        # Never put this call under no_grad or through a processor/NumPy path.
        dacf_output = self.dacf(mixture_waveform, enrollment_waveform)
        query, present_logits, activity_logits = self._validate_dacf_output(
            dacf_output, mixture_batch
        )
        self._dacf_output = dacf_output
        self._speaker_query_frames = query
        self._active_batch_size = mixture_batch
        self._audio_call_count = 0
        self._active_audio_index = None
        self._resolved_audio_layout = None

        try:
            # This is intentionally a direct thinker call with autograd
            # enabled.  The thinker parameters are frozen, but its hidden
            # activations must remain differentiable through the residual.
            output = self.thinker(*thinker_args, **thinker_inputs)
            if self._audio_call_count == 0:
                raise RuntimeError(
                    "thinker.forward did not invoke audio_tower; normal Qwen "
                    "audio inputs/layout are missing"
                )
            if self._resolved_audio_layout in {"batch_first", "time_first"}:
                expected_calls = 1
            else:
                expected_calls = mixture_batch
            if self._audio_call_count != expected_calls:
                raise ValueError(
                    "Qwen audio-tower call count does not match DACF waveform "
                    f"batch: calls={self._audio_call_count}, expected={expected_calls}, "
                    f"layout={self._resolved_audio_layout}"
                )
        finally:
            self._speaker_query_frames = None
            self._active_batch_size = 0
            self._audio_call_count = 0
            self._active_audio_index = None
            self._resolved_audio_layout = None

        asr_loss = _field(output, "loss")
        has_aux_target = target_present is not None or target_activity is not None
        if asr_loss is None and has_aux_target:
            raise RuntimeError(
                "thinker output.loss is None while auxiliary training targets were "
                "provided; pass normal thinker labels or use an inference-only call"
            )

        presence_loss = self._presence_loss(present_logits, target_present)
        activity_loss = self._activity_loss(activity_logits, target_activity)
        aux_loss = (
            self.config.presence_loss_weight * presence_loss
            + self.config.activity_loss_weight * activity_loss
        )
        total_loss = None if asr_loss is None else asr_loss + aux_loss.to(asr_loss.device)

        fields = {
            "loss": total_loss,
            "asr_loss": asr_loss,
            "presence_loss": presence_loss,
            "activity_loss": activity_loss,
            "aux_loss": aux_loss,
            "presence_probs": torch.sigmoid(present_logits),
            "activity_probs": torch.sigmoid(activity_logits),
            "dacf_output": dacf_output,
            "speaker_query_frames": query,
        }
        return self._attach_output(output, fields)

    @contextmanager
    def inference_context(self, mixture_waveform: Tensor, enrollment_waveform: Tensor):
        """Supply DACF waveform context for a direct ``thinker`` call.

        This is intentionally small and does not implement generation or a
        processor.  It is useful only for a future interface smoke; training
        should call :meth:`forward` so its auxiliary losses are explicit.
        """

        mixture_batch = _waveform_batch(mixture_waveform, "mixture_waveform")
        enrollment_batch = _waveform_batch(enrollment_waveform, "enrollment_waveform")
        if mixture_batch != enrollment_batch:
            raise ValueError("inference waveform batches must match")
        dacf_output = self.dacf(mixture_waveform, enrollment_waveform)
        query, _, _ = self._validate_dacf_output(dacf_output, mixture_batch)
        self._dacf_output = dacf_output
        self._speaker_query_frames = query
        self._active_batch_size = mixture_batch
        self._audio_call_count = 0
        self._active_audio_index = None
        self._resolved_audio_layout = None
        try:
            yield dacf_output
        finally:
            self._speaker_query_frames = None
            self._active_batch_size = 0
            self._audio_call_count = 0
            self._active_audio_index = None
            self._resolved_audio_layout = None

    def assert_gradient_contract(self, loss: Optional[Tensor] = None) -> None:
        """Fail unless DACF/bridge gradients are live and thinker grads stay off.

        At exact zero gate an ASR-only first step can update only the scalar
        gates; DACF/projection gradients are mathematically zero.  Call this
        strict check after auxiliary warm-up or after a gate becomes non-zero.
        """

        if loss is not None:
            if not isinstance(loss, Tensor) or not loss.requires_grad or loss.grad_fn is None:
                raise RuntimeError(
                    "loss is detached/no-grad; run thinker.forward(labels) with "
                    "autograd enabled before checking the bridge"
                )

        if self._speaker_query_frames is not None and not self._speaker_query_frames.requires_grad:
            raise RuntimeError("DACF speaker_query_frames is detached")

        def has_live_gradient(parameters: Any) -> bool:
            return any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and bool(parameter.grad.abs().sum() > 0)
                for parameter in parameters
                if parameter.requires_grad
            )

        dacf_parameters = tuple(
            parameter for parameter in self.dacf.parameters() if parameter.requires_grad
        )
        bridge_parameters = tuple(
            parameter for parameter in self.bridges.parameters() if parameter.requires_grad
        )
        if not dacf_parameters and not bridge_parameters:
            raise RuntimeError("no trainable DACF/latent-bridge parameters exist")
        if dacf_parameters and not has_live_gradient(dacf_parameters):
            raise RuntimeError(
                "no finite non-zero DACF gradient; speaker waveform branch may be "
                "detached or hidden behind torch.no_grad()"
            )
        if bridge_parameters and not has_live_gradient(bridge_parameters):
            raise RuntimeError("no finite non-zero latent-bridge gradient")

        leaked = [
            index
            for index, parameter in enumerate(self._frozen_thinker_parameters)
            if parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(f"frozen thinker gradients leaked at indices {leaked[:3]}")
        if any(parameter.requires_grad for parameter in self._frozen_thinker_parameters):
            raise RuntimeError("a frozen thinker parameter was made trainable")

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def train(self, mode: bool = True):
        """Train DACF/bridge while keeping the frozen thinker in eval mode."""

        super().train(False)
        self.training = mode
        self.dacf.train(mode)
        self.bridges.train(mode)
        self.thinker.eval()
        return self
