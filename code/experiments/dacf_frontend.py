"""Small DACF (Dual-Anchor Counterfactual Front-end) PyTorch skeleton.

This file is an architecture POC for the D2 role.  It deliberately does not
load a checkpoint, read Dataset-A, or connect to the submission chain.

The intended training query is counterfactual within one fixed mixture::

    mix(A+B) + enrollment(A) -> A
    mix(A+B) + enrollment(B) -> B
    mix(A+B) + enrollment(C absent) -> blank

The implementation exposes the query-dependent representation and all heads,
but it is not a trained model and makes no CER claim.  In particular, the
mask/STFT path and the counterfactual residual are architectural placeholders
until target-present/absent recall, official CER, paired bootstrap, RTF, and
listening checks are run on a held-out non-Dataset-A development set.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_BLANK_INDEX = 0


def _as_mono_waveform(waveform: Tensor, name: str) -> Tensor:
    """Normalize a waveform input to ``[batch, samples]`` without resampling."""

    if waveform.ndim == 3:
        if waveform.shape[1] != 1:
            raise ValueError(f"{name} must be mono when 3-D, got {tuple(waveform.shape)}")
        waveform = waveform[:, 0]
    if waveform.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B, 1, T], got {tuple(waveform.shape)}")
    if waveform.shape[0] < 1 or waveform.shape[-1] < 2:
        raise ValueError(f"{name} must have a non-empty batch and at least two samples")
    return waveform.float()


def _zero_linear(layer: nn.Linear) -> None:
    """Make a conditioning branch conservative at initialization."""

    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class DACFFrontend(nn.Module):
    """A small, query-aware dual-anchor front-end.

    Inputs are raw mono waveforms.  The model uses a compact magnitude-STFT
    encoder so that the output mask can be applied to the original complex
    mixture STFT and reconstructed with ``torch.istft``.

    ``speaker_cross`` and ``environment_cross`` are intentionally separate
    cross-attention modules.  The two anchors therefore condition mixture
    frames through different key/value paths; they are not concatenated into
    one vector followed by a linear layer.
    """

    def __init__(
        self,
        *,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        d_model: int = 64,
        n_heads: int = 4,
        vocab_size: int = 64,
        max_mask_gain: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_fft < 4:
            raise ValueError("n_fft must be at least 4")
        if hop_length < 1 or hop_length > n_fft:
            raise ValueError("hop_length must be in [1, n_fft]")
        win_length = n_fft if win_length is None else win_length
        if win_length < 2 or win_length > n_fft:
            raise ValueError("win_length must be in [2, n_fft]")
        if d_model < 4 or d_model % n_heads != 0:
            raise ValueError("d_model must be >= 4 and divisible by n_heads")
        if vocab_size < 2:
            raise ValueError("vocab_size must include a blank and at least one token")
        if max_mask_gain <= 1.0:
            raise ValueError("max_mask_gain must be greater than 1")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_mask_gain = float(max_mask_gain)
        initial_probability = 1.0 / self.max_mask_gain
        self.initial_mask_logit = math.log(
            initial_probability / (1.0 - initial_probability)
        )

        self.register_buffer("stft_window", torch.hann_window(win_length), persistent=False)

        freq_bins = n_fft // 2 + 1
        self.mixture_encoder = nn.Sequential(
            nn.Conv1d(freq_bins, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.enrollment_encoder = nn.Sequential(
            nn.Conv1d(freq_bins, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Separate enrollment branches preserve a speaker anchor and an
        # environment anchor instead of collapsing enrollment to one vector.
        self.speaker_branch = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.environment_branch = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

        self.mixture_norm = nn.LayerNorm(d_model)
        self.speaker_norm = nn.LayerNorm(d_model)
        self.environment_norm = nn.LayerNorm(d_model)

        # These are the core dual-anchor paths.  They deliberately remain
        # separate all the way through conditioning.
        self.speaker_cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.environment_cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.speaker_gate = nn.Linear(d_model, d_model)
        self.environment_gate = nn.Linear(d_model, d_model)
        self.speaker_film_scale = nn.Linear(d_model, d_model)
        self.speaker_film_bias = nn.Linear(d_model, d_model)
        self.environment_film_scale = nn.Linear(d_model, d_model)
        self.environment_film_bias = nn.Linear(d_model, d_model)
        self.speaker_condition_norm = nn.LayerNorm(d_model)
        self.condition_norm = nn.LayerNorm(d_model)

        # A signed difference of two independently projected cross contexts is
        # an explicit counterfactual/query term.  It is intentionally small at
        # initialization and must earn useful semantics from query-swap data.
        self.counterfactual_speaker = nn.Linear(d_model, d_model)
        self.counterfactual_mixture = nn.Linear(d_model, d_model)
        self.counterfactual_gate = nn.Linear(d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)

        self.presence_head = nn.Linear(d_model, 1)
        self.activity_head = nn.Linear(d_model, 1)
        self.mask_head = nn.Linear(d_model, freq_bins)
        self.ctc_head = nn.Linear(d_model, vocab_size)

        # A zero residual mask starts as identity (target audio = mixture).
        # The scaled sigmoid spans (0, max_mask_gain), retaining a non-zero
        # derivative at identity while allowing absent queries to approach
        # silence and difficult bins to receive modest amplification.
        _zero_linear(self.mask_head)
        for layer in (
            self.speaker_film_scale,
            self.speaker_film_bias,
            self.environment_film_scale,
            self.environment_film_bias,
        ):
            _zero_linear(layer)

    def _stft(self, waveform: Tensor) -> Tensor:
        window = self.stft_window.to(device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

    @staticmethod
    def _log_magnitude(stft: Tensor) -> Tensor:
        # [B, F, frames], suitable for Conv1d over time.
        return torch.log1p(stft.abs())

    @staticmethod
    def _anchor(tokens: Tensor, weights: Tensor) -> Tensor:
        weighted = tokens * weights.unsqueeze(-1)
        pooled = weighted.sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return F.normalize(pooled, dim=-1, eps=1e-6)

    @staticmethod
    def _frame_partition(enrollment_features: Tensor) -> tuple[Tensor, Tensor]:
        """Softly split speech-dominant and background-dominant frames.

        Both branches still see the same enrollment recording, but they no
        longer receive indistinguishable token pools.  Relative frame energy
        is used only as an architectural prior; the environment 2x2 swap test
        remains necessary before interpreting the second anchor as room or
        device information.
        """

        frame_energy = enrollment_features.mean(dim=1)
        center = frame_energy.median(dim=1, keepdim=True).values
        scale = frame_energy.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        speaker_weights = torch.sigmoid((frame_energy - center) / scale)
        environment_weights = 1.0 - speaker_weights
        return speaker_weights, environment_weights

    def _encode_enrollment(self, enrollment: Tensor) -> Dict[str, Tensor]:
        enrollment_stft = self._stft(enrollment)
        enrollment_features = self._log_magnitude(enrollment_stft)
        speaker_weights, environment_weights = self._frame_partition(enrollment_features)
        base = self.enrollment_encoder(enrollment_features)
        speaker_tokens = self.speaker_branch(base).transpose(1, 2)
        environment_tokens = self.environment_branch(base).transpose(1, 2)
        speaker_tokens = self.speaker_norm(speaker_tokens)
        environment_tokens = self.environment_norm(environment_tokens)
        return {
            "speaker_tokens": speaker_tokens,
            "environment_tokens": environment_tokens,
            "speaker_weights": speaker_weights,
            "environment_weights": environment_weights,
            "speaker_anchor": self._anchor(speaker_tokens, speaker_weights),
            "environment_anchor": self._anchor(environment_tokens, environment_weights),
        }

    def forward(self, mixture: Tensor, enrollment: Tensor) -> Dict[str, Tensor]:
        """Run one query against one mixture batch.

        Args:
            mixture: ``[B, T]`` or mono ``[B, 1, T]`` waveform.
            enrollment: ``[B, T_e]`` or mono ``[B, 1, T_e]`` waveform.  It may
                be the waveform of an absent speaker; absence is represented
                by the targets returned from :func:`make_absent_targets`.

        Returns:
            A dictionary containing anchors, query-aware frames, target
            presence/activity heads, CTC logits, a conservative complex-STFT
            mask, and reconstructed target audio.
        """

        mixture = _as_mono_waveform(mixture, "mixture")
        enrollment = _as_mono_waveform(enrollment, "enrollment")
        if mixture.shape[0] != enrollment.shape[0]:
            raise ValueError(
                "mixture and enrollment batch sizes must match: "
                f"{mixture.shape[0]} != {enrollment.shape[0]}"
            )

        mixture_stft = self._stft(mixture)
        mixture_features = self._log_magnitude(mixture_stft)
        mixture_frames = self.mixture_encoder(mixture_features).transpose(1, 2)
        mixture_frames = self.mixture_norm(mixture_frames)

        enrollment_state = self._encode_enrollment(enrollment)
        speaker_tokens = enrollment_state["speaker_tokens"]
        environment_tokens = enrollment_state["environment_tokens"]
        speaker_weights = enrollment_state["speaker_weights"]
        environment_weights = enrollment_state["environment_weights"]
        speaker_anchor = enrollment_state["speaker_anchor"]
        environment_anchor = enrollment_state["environment_anchor"]

        # Separate cross-attention calls are the dual-anchor conditioning
        # mechanism.  No concatenated anchor vector is used here.
        speaker_context, _ = self.speaker_cross(
            query=mixture_frames,
            key=speaker_tokens,
            value=speaker_tokens,
            key_padding_mask=speaker_weights.lt(0.5),
            need_weights=False,
        )
        environment_context, _ = self.environment_cross(
            query=mixture_frames,
            key=environment_tokens,
            value=environment_tokens,
            key_padding_mask=environment_weights.lt(0.5),
            need_weights=False,
        )

        speaker_gate = torch.sigmoid(self.speaker_gate(speaker_anchor)).unsqueeze(1)
        environment_gate = torch.sigmoid(self.environment_gate(environment_anchor)).unsqueeze(1)
        speaker_scale = 0.1 * torch.tanh(self.speaker_film_scale(speaker_anchor)).unsqueeze(1)
        speaker_bias = 0.1 * torch.tanh(self.speaker_film_bias(speaker_anchor)).unsqueeze(1)
        environment_scale = 0.1 * torch.tanh(
            self.environment_film_scale(environment_anchor)
        ).unsqueeze(1)
        environment_bias = 0.1 * torch.tanh(
            self.environment_film_bias(environment_anchor)
        ).unsqueeze(1)

        speaker_conditioned = mixture_frames * (1.0 + speaker_scale)
        speaker_conditioned = speaker_conditioned + speaker_bias
        speaker_conditioned = speaker_conditioned + 0.1 * speaker_gate * speaker_context
        speaker_conditioned = self.speaker_condition_norm(speaker_conditioned)

        signed_query_delta = self.counterfactual_speaker(speaker_context)
        signed_query_delta = signed_query_delta - self.counterfactual_mixture(
            mixture_frames
        )
        query_gate = torch.sigmoid(
            self.counterfactual_gate(speaker_context + mixture_frames)
        )
        speaker_query_frames = self.query_norm(
            speaker_conditioned + 0.1 * query_gate * signed_query_delta
        )

        # Presence/activity are intentionally insulated from the environment
        # branch.  Environment can adapt reconstruction/ASR acoustics, but it
        # cannot directly vote that a same-room wrong speaker is present.
        target_present_logits = self.presence_head(speaker_query_frames.mean(dim=1)).squeeze(-1)
        target_activity_logits = self.activity_head(speaker_query_frames).squeeze(-1)

        query_aware_frames = speaker_query_frames * (1.0 + environment_scale)
        query_aware_frames = query_aware_frames + environment_bias
        query_aware_frames = query_aware_frames + 0.1 * environment_gate * environment_context
        query_aware_frames = self.condition_norm(query_aware_frames)
        ctc_logits = self.ctc_head(query_aware_frames)

        # [B, frames, F] -> [B, F, frames] to match the complex mixture STFT.
        mask_logits = self.mask_head(query_aware_frames).transpose(1, 2)
        target_mask = self.max_mask_gain * torch.sigmoid(
            mask_logits + self.initial_mask_logit
        )
        target_stft = mixture_stft * target_mask.to(dtype=mixture_stft.real.dtype)
        window = self.stft_window.to(device=mixture.device, dtype=mixture.dtype)
        target_audio = torch.istft(
            target_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            length=mixture.shape[-1],
        )

        return {
            "mixture_audio": mixture,
            "mixture_stft": mixture_stft,
            "target_stft": target_stft,
            "target_mask": target_mask,
            "target_audio": target_audio,
            "speaker_tokens": speaker_tokens,
            "environment_tokens": environment_tokens,
            "speaker_frame_weights": speaker_weights,
            "environment_frame_weights": environment_weights,
            "speaker_anchor": speaker_anchor,
            "environment_anchor": environment_anchor,
            "mixture_frames": mixture_frames,
            "speaker_query_frames": speaker_query_frames,
            "query_aware_frames": query_aware_frames,
            "target_present_logits": target_present_logits,
            "target_present_probs": torch.sigmoid(target_present_logits),
            "target_activity_logits": target_activity_logits,
            "target_activity_probs": torch.sigmoid(target_activity_logits),
            "ctc_logits": ctc_logits,
        }


def make_absent_targets(
    *,
    batch_size: int,
    num_frames: int,
    num_samples: int,
    blank_index: int = DEFAULT_BLANK_INDEX,
    device: Optional[torch.device | str] = None,
) -> Dict[str, Tensor]:
    """Create explicit target-absent labels for a query batch.

    For CTC, a blank transcript is represented by a zero-length target, not a
    target containing the blank ID.  ``blank_index`` is returned as metadata so
    callers can keep the same vocabulary contract when constructing batches.
    """

    if batch_size < 1 or num_frames < 1 or num_samples < 1:
        raise ValueError("batch_size, num_frames, and num_samples must be positive")
    if blank_index < 0:
        raise ValueError("blank_index must be non-negative")
    return {
        "target_present": torch.zeros(batch_size, device=device),
        "target_activity": torch.zeros(batch_size, num_frames, device=device),
        "target_audio": torch.zeros(batch_size, num_samples, device=device),
        "transcript": torch.empty(batch_size, 0, dtype=torch.long, device=device),
        "transcript_lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
        "blank_index": torch.tensor(blank_index, dtype=torch.long, device=device),
    }


def identity_contrastive_loss(
    anchors: Tensor,
    query_ids: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Supervised identity contrastive loss for swapped queries.

    Rows with no same-identity partner contribute zero.  This makes the loss
    safe for an all-unique query-swap smoke batch while still training when the
    same identity occurs across mixtures or augmentations.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if anchors.ndim != 2 or query_ids.ndim != 1 or anchors.shape[0] != query_ids.shape[0]:
        raise ValueError("anchors must be [B, D] and query_ids must be [B]")
    if anchors.shape[0] < 2:
        return anchors.sum() * 0.0
    # With only one identity/environment class there are no negatives.  The
    # supervised contrastive objective would otherwise become a constant
    # log(B-1), inflating total loss without providing a useful gradient.
    if torch.unique(query_ids).numel() < 2:
        return anchors.sum() * 0.0

    normalized = F.normalize(anchors, dim=-1, eps=1e-6)
    similarity = normalized @ normalized.transpose(0, 1)
    similarity = similarity / temperature
    not_self = ~torch.eye(
        anchors.shape[0], dtype=torch.bool, device=anchors.device
    )
    positive = query_ids[:, None].eq(query_ids[None, :]) & not_self
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return anchors.sum() * 0.0

    logits = similarity.masked_fill(~not_self, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_row = -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1) / positive_count
    )
    return per_row[valid].mean()


def anchor_disentanglement_loss(
    speaker_anchors: Tensor,
    environment_anchors: Tensor,
) -> Tensor:
    """Penalize batch-level cross-covariance between the two anchor spaces.

    Separate branches alone do not prove that one branch represents identity
    and the other represents room/device/background.  This term is only a
    lightweight anti-leakage constraint; speaker and environment contrastive
    labels remain the primary supervision.
    """

    if speaker_anchors.shape != environment_anchors.shape or speaker_anchors.ndim != 2:
        raise ValueError("speaker_anchors and environment_anchors must share shape [B, D]")
    if speaker_anchors.shape[0] < 2:
        return speaker_anchors.sum() * 0.0

    speaker = F.normalize(speaker_anchors, dim=-1, eps=1e-6)
    environment = F.normalize(environment_anchors, dim=-1, eps=1e-6)
    speaker = speaker - speaker.mean(dim=0, keepdim=True)
    environment = environment - environment.mean(dim=0, keepdim=True)
    cross_covariance = speaker.transpose(0, 1) @ environment
    cross_covariance = cross_covariance / float(speaker.shape[0] - 1)
    return cross_covariance.square().mean()


def counterfactual_delta_loss(
    predicted_audio: Tensor,
    target_audio: Tensor,
    mixture_ids: Tensor,
    query_ids: Tensor,
    reference_audio: Optional[Tensor] = None,
) -> Tensor:
    """Couple different identity queries made against one fixed mixture.

    Individual reconstruction can be minimized without proving that changing
    only the enrollment caused the output change.  For every same-mixture,
    different-query pair, this term matches the predicted audio delta to the
    supervised target delta.  A/B/C rows must therefore reuse the exact same
    mixture waveform; the data builder and tests enforce that contract.
    """

    if predicted_audio.shape != target_audio.shape or predicted_audio.ndim != 2:
        raise ValueError("predicted_audio and target_audio must share shape [B, T]")
    batch_size = predicted_audio.shape[0]
    mixture_ids = mixture_ids.to(device=predicted_audio.device).view(-1)
    query_ids = query_ids.to(device=predicted_audio.device).view(-1)
    if mixture_ids.shape[0] != batch_size or query_ids.shape[0] != batch_size:
        raise ValueError("mixture_ids and query_ids must have shape [B]")

    same_mixture = mixture_ids[:, None].eq(mixture_ids[None, :])
    different_query = ~query_ids[:, None].eq(query_ids[None, :])
    upper_triangle = torch.triu(
        torch.ones(batch_size, batch_size, dtype=torch.bool, device=predicted_audio.device),
        diagonal=1,
    )
    pairs = (same_mixture & different_query & upper_triangle).nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return predicted_audio.sum() * 0.0

    left, right = pairs[:, 0], pairs[:, 1]
    predicted_delta = predicted_audio[left] - predicted_audio[right]
    target_delta = target_audio[left] - target_audio[right]
    per_pair = (predicted_delta - target_delta).abs().mean(dim=1)
    if reference_audio is not None:
        if reference_audio.shape != predicted_audio.shape:
            raise ValueError("reference_audio must match predicted_audio shape [B, T]")
        scale = reference_audio[left].abs().mean(dim=1).clamp_min(1e-4)
        per_pair = per_pair / scale
    return per_pair.mean()


def _balanced_binary_cross_entropy_with_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """Give positive and negative labels equal aggregate weight when both exist."""

    targets = targets.to(device=logits.device, dtype=logits.dtype)
    flat_targets = targets.reshape(-1)
    positives = flat_targets.sum()
    negatives = flat_targets.numel() - positives
    per_item = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if bool((positives > 0).item()) and bool((negatives > 0).item()):
        total = float(flat_targets.numel())
        positive_weight = total / (2.0 * positives)
        negative_weight = total / (2.0 * negatives)
        weights = torch.where(targets > 0.5, positive_weight, negative_weight)
        return (per_item * weights).mean()
    return per_item.mean()


def _ctc_targets(
    transcript: Tensor,
    transcript_lengths: Optional[Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Flatten padded ``[B, L]`` CTC targets without admitting blank padding."""

    if transcript.ndim != 2 or transcript.shape[0] != batch_size:
        raise ValueError("transcript must have padded shape [B, L]")
    transcript = transcript.to(device=device, dtype=torch.long)
    if transcript_lengths is None:
        lengths = torch.full(
            (batch_size,), transcript.shape[1], dtype=torch.long, device=device
        )
    else:
        lengths = transcript_lengths.to(device=device, dtype=torch.long).view(-1)
        if lengths.shape[0] != batch_size:
            raise ValueError("transcript_lengths must have shape [B]")
    if bool((lengths < 0).any()) or bool((lengths > transcript.shape[1]).any()):
        raise ValueError("transcript_lengths must be within the padded transcript width")

    pieces = [transcript[row, : int(lengths[row].item())] for row in range(batch_size)]
    flat = (
        torch.cat(pieces, dim=0)
        if pieces
        else torch.empty(0, dtype=torch.long, device=device)
    )
    return flat, lengths


def compute_dacf_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    *,
    weights: Optional[Mapping[str, float]] = None,
    blank_index: int = DEFAULT_BLANK_INDEX,
) -> Dict[str, Tensor]:
    """Compute the POC's multi-head training objective.

    Supported terms are ``ctc``, ``activity``, ``presence``,
    ``reconstruction``, ``identity``, ``environment``, ``disentangle``, and
    ``counterfactual``.  The returned dictionary contains every active
    component and ``total``.  Missing optional target fields simply disable
    that component; the standard absent-target helper supplies the four task
    targets and therefore exercises blank CTC correctly.
    """

    default_weights = {
        "ctc": 1.0,
        "activity": 1.0,
        "presence": 1.0,
        "reconstruction": 1.0,
        "identity": 0.1,
        "environment": 0.1,
        "disentangle": 0.01,
        "counterfactual": 1.0,
    }
    if weights is not None:
        default_weights.update(weights)

    logits = outputs["ctc_logits"]
    batch_size, time_steps, vocab_size = logits.shape
    total = logits.sum() * 0.0
    losses: Dict[str, Tensor] = {}

    if "target_present" in targets:
        presence_target = targets["target_present"].to(
            device=logits.device, dtype=logits.dtype
        ).view_as(outputs["target_present_logits"])
        losses["presence"] = _balanced_binary_cross_entropy_with_logits(
            outputs["target_present_logits"], presence_target
        )

    if "target_activity" in targets:
        activity_target = targets["target_activity"].to(
            device=logits.device, dtype=logits.dtype
        )
        if activity_target.ndim != 2 or activity_target.shape[0] != batch_size:
            raise ValueError("target_activity must have shape [B, frames]")
        if activity_target.shape[1] != time_steps:
            activity_target = F.interpolate(
                activity_target.unsqueeze(1),
                size=time_steps,
                mode="nearest",
            ).squeeze(1)
        losses["activity"] = _balanced_binary_cross_entropy_with_logits(
            outputs["target_activity_logits"], activity_target
        )

    if "target_audio" in targets:
        audio_target = targets["target_audio"].to(
            device=outputs["target_audio"].device,
            dtype=outputs["target_audio"].dtype,
        )
        if audio_target.shape != outputs["target_audio"].shape:
            raise ValueError(
                "target_audio must match reconstructed output shape: "
                f"{tuple(audio_target.shape)} != {tuple(outputs['target_audio'].shape)}"
            )
        # Normalize by mixture magnitude so classification losses cannot drown
        # a numerically small waveform loss merely because PCM amplitudes are
        # around 1e-2.  This is not an SI-SNR claim and remains only one term.
        reference_audio = outputs.get("mixture_audio")
        if reference_audio is None:
            reference_audio = outputs["target_audio"].detach()
        scale = reference_audio.abs().mean(dim=1).clamp_min(1e-4)
        losses["reconstruction"] = (
            (outputs["target_audio"] - audio_target).abs().mean(dim=1) / scale
        ).mean()

        counterfactual_query = targets.get("query_role_id", targets.get("query_id"))
        if "mixture_id" in targets and counterfactual_query is not None:
            losses["counterfactual"] = counterfactual_delta_loss(
                outputs["target_audio"],
                audio_target,
                targets["mixture_id"],
                counterfactual_query,
                reference_audio=reference_audio,
            )

    if "transcript" in targets:
        flat_targets, target_lengths = _ctc_targets(
            targets["transcript"],
            targets.get("transcript_lengths"),
            batch_size=batch_size,
            device=logits.device,
        )
        if not 0 <= blank_index < vocab_size:
            raise ValueError("blank_index must be within ctc_logits vocabulary")
        input_lengths = torch.full(
            (batch_size,), time_steps, dtype=torch.long, device=logits.device
        )
        losses["ctc"] = F.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            flat_targets,
            input_lengths,
            target_lengths,
            blank=blank_index,
            reduction="mean",
            zero_infinity=True,
        )

    # query_role_id (A/B/C) is not a speaker identity.  Training loaders must
    # map the manifest's query_speaker_id to an integer query_speaker_label;
    # query_id remains a backwards-compatible fallback for synthetic tests.
    speaker_label = targets.get(
        "query_speaker_label", targets.get("speaker_id", targets.get("query_id"))
    )
    if speaker_label is not None:
        losses["identity"] = identity_contrastive_loss(
            outputs["speaker_anchor"],
            speaker_label.to(device=logits.device, dtype=torch.long).view(-1),
        )

    if "environment_id" in targets:
        losses["environment"] = identity_contrastive_loss(
            outputs["environment_anchor"],
            targets["environment_id"].to(device=logits.device, dtype=torch.long).view(-1),
        )
        losses["disentangle"] = anchor_disentanglement_loss(
            outputs["speaker_anchor"], outputs["environment_anchor"]
        )

    for name, value in losses.items():
        total = total + float(default_weights.get(name, 1.0)) * value
    losses["total"] = total
    return losses


__all__ = [
    "DACFFrontend",
    "DEFAULT_BLANK_INDEX",
    "anchor_disentanglement_loss",
    "compute_dacf_loss",
    "counterfactual_delta_loss",
    "identity_contrastive_loss",
    "make_absent_targets",
]
