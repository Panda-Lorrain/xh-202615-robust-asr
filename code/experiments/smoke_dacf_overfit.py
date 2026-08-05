"""Fixed-mixture DACF causal smoke test.

This experiment reads one three-query DACF group, validates the immutable
mixture contract, and runs a deliberately small overfit loop against
``DACFFrontend``.  It is a query-control/optimization smoke test only: it
does not measure generalization, held-out performance, CER, or submission
score, and it never reads Dataset-A or calls the submission chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf
import torch
from torch import Tensor


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from dacf_frontend import (  # noqa: E402
    DACFFrontend,
    compute_dacf_loss,
    counterfactual_delta_loss,
    identity_contrastive_loss,
)


SAMPLE_RATE = 16_000
ACTIVITY_HOP_SAMPLES = 160
DEFAULT_STEPS = 100
MAX_STEPS = 300
DEFAULT_SEED = 20260806
ROLE_ORDER = ("present_A", "present_B", "absent_C")
ROLE_TO_ID = {role: index for index, role in enumerate(ROLE_ORDER)}
LOSS_NAMES = (
    "ctc",
    "activity",
    "presence",
    "reconstruction",
    "identity",
    "environment",
    "disentangle",
    "counterfactual",
    "total",
)
_DATASET_A_MARKERS = ("dataset-a", "dataset_a", "dataseta")
_PATH_KEY_PARTS = ("audio", "activity", "path", "src", "wav", "manifest")


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _assert_not_dataset_a(value: Any, *, field: str = "path") -> None:
    """Reject forbidden paths before any file is opened."""

    text = _path_text(value).casefold()
    if any(marker in text for marker in _DATASET_A_MARKERS):
        raise ValueError(f"{field} contains forbidden Dataset-A path: {value}")


def _path_key(key: str) -> bool:
    lowered = key.casefold()
    return any(part in lowered for part in _PATH_KEY_PARTS)


def _guard_path_values(value: Any, field: str, *, path_context: bool = False) -> None:
    """Recursively guard path-like manifest fields without opening them."""

    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            child_field = f"{field}.{child_key}"
            _guard_path_values(
                child_value,
                child_field,
                path_context=path_context or _path_key(str(child_key)),
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            _guard_path_values(
                child_value,
                f"{field}[{index}]",
                path_context=path_context,
            )
        return
    if path_context and isinstance(value, (str, Path)):
        _assert_not_dataset_a(value, field=field)


def _read_jsonl(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(manifest_path)
    _assert_not_dataset_a(manifest, field="manifest")
    rows: list[dict[str, Any]] = []
    try:
        handle = manifest.open("r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open DACF JSONL manifest: {manifest}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {manifest}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest row is not an object at {manifest}:{line_number}")
            _guard_path_values(row, f"row {line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"DACF JSONL manifest is empty: {manifest}")
    return rows


def _normalize_role(row: Mapping[str, Any]) -> tuple[str, int]:
    raw_role_id = row.get("query_role_id")
    if isinstance(raw_role_id, bool) or raw_role_id is None:
        raise ValueError("each DACF row requires integer query_role_id 0/1/2")
    try:
        role_id = int(raw_role_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid query_role_id: {raw_role_id!r}") from exc
    if role_id not in (0, 1, 2):
        raise ValueError(f"query_role_id must be 0, 1, or 2, got {role_id}")

    role_text = str(row.get("query_role", "")).strip()
    aliases = {"A": "present_A", "B": "present_B", "C": "absent_C"}
    role = aliases.get(role_text, role_text)
    expected_role = ROLE_ORDER[role_id]
    if role and role != expected_role:
        raise ValueError(
            f"query role mismatch: query_role={role_text!r}, query_role_id={role_id}"
        )
    return expected_role, role_id


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _require_text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"DACF row requires non-empty {key}")
    return str(value)


def _resolve_path(raw: Any, manifest_path: Path, *, field: str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError(f"DACF row requires non-empty path field {field}")
    # Guard the manifest value before normalizing or probing any candidate.
    _assert_not_dataset_a(raw, field=field)
    path = Path(str(raw))
    if path.is_absolute():
        candidates = (path,)
    else:
        # The builder emits paths relative to the repository process cwd;
        # movable fixtures commonly emit paths relative to their manifest.
        candidates = (Path.cwd() / path, manifest_path.parent / path)

    existing: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        _assert_not_dataset_a(candidate, field=field)
        if candidate.exists():
            existing.append(candidate)

    unique_existing = list(dict.fromkeys(existing))
    if len(unique_existing) > 1:
        raise ValueError(
            f"ambiguous {field} path {raw!r}; existing candidates resolve differently: "
            + ", ".join(str(candidate) for candidate in unique_existing)
        )
    if not unique_existing:
        candidate_text = ", ".join(str(candidate.resolve(strict=False)) for candidate in candidates)
        raise ValueError(
            f"cannot resolve {field} path {raw!r}; candidates: {candidate_text}"
        )
    return unique_existing[0]


def _clean_audio_key(row: Mapping[str, Any]) -> str:
    if row.get("clean_target_audio"):
        return "clean_target_audio"
    if row.get("target_audio"):
        return "target_audio"
    raise ValueError("DACF row requires clean_target_audio (or target_audio)")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest_path: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[Path, Path, Path], str]:
    if len(rows) != 3:
        raise ValueError(
            "one base_mixture_id must contain exactly three rows: present_A, "
            f"present_B, absent_C; got {len(rows)}"
        )

    by_role: dict[str, dict[str, Any]] = {}
    for original_row in rows:
        row = dict(original_row)
        role, role_id = _normalize_role(row)
        if role in by_role:
            raise ValueError(f"duplicate DACF query role: {role}")
        row["_smoke_role"] = role
        row["_smoke_role_id"] = role_id
        expected_present = role != "absent_C"
        if "target_present" in row and _as_bool(row["target_present"]) != expected_present:
            raise ValueError(f"target_present disagrees with role {role}")
        by_role[role] = row

    missing = [role for role in ROLE_ORDER if role not in by_role]
    if missing:
        raise ValueError(f"DACF query group is missing roles: {missing}")
    ordered_rows = tuple(by_role[role] for role in ROLE_ORDER)

    mixture_sha_values = []
    mixture_paths = []
    for row in ordered_rows:
        sha = row.get("mixture_sha256")
        if not isinstance(sha, str) or not sha.strip():
            raise ValueError("each DACF query row requires mixture_sha256")
        mixture_sha_values.append(sha.strip().casefold())
        mixture_paths.append(
            _resolve_path(row.get("recognition_audio"), manifest_path, field="recognition_audio")
        )
    if len(set(mixture_sha_values)) != 1:
        raise ValueError(
            "mixture_sha256 mismatch across present_A/present_B/absent_C rows"
        )
    if len(set(mixture_paths)) != 1:
        raise ValueError(
            "recognition_audio must be the same path across all three query rows"
        )

    try:
        reference_bytes = mixture_paths[0].read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read recognition_audio bytes: {mixture_paths[0]}") from exc
    for path in mixture_paths[1:]:
        try:
            candidate_bytes = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read recognition_audio bytes: {path}") from exc
        if candidate_bytes != reference_bytes:
            raise ValueError("recognition_audio file bytes differ across the query rows")
    actual_sha = _sha256_bytes(reference_bytes)
    if actual_sha != mixture_sha_values[0]:
        raise ValueError(
            "mixture_sha256 does not match recognition_audio file bytes: "
            f"declared={mixture_sha_values[0]}, actual={actual_sha}"
        )

    speaker_ids = []
    environment_ids = []
    view2_flags = []
    for row in ordered_rows:
        speaker_ids.append(_require_text(row, "query_speaker_id"))
        environment_ids.append(_require_text(row, "environment_id"))
        _require_text(row, "query_speaker_label")
        _require_text(row, "enrollment_audio")
        _require_text(row, "target_activity")
        _clean_audio_key(row)
        view2_flags.append("enrollment_audio_view2" in row)
        if row.get("target_activity_hop_samples") is not None:
            try:
                activity_hop = int(row["target_activity_hop_samples"])
            except (TypeError, ValueError) as exc:
                raise ValueError("target_activity_hop_samples must be an integer") from exc
            if activity_hop != ACTIVITY_HOP_SAMPLES:
                raise ValueError(
                    "target_activity_hop_samples must be 160 for the 16 kHz/10 ms smoke test"
                )
    if any(view2_flags) and not all(view2_flags):
        raise ValueError(
            "enrollment_audio_view2 must be present for all three rows or none"
        )
    declared_two_view = [
        row.get("enrollment_view_count") == 2 for row in ordered_rows
    ]
    if any(declared_two_view) and not all(view2_flags):
        raise ValueError(
            "enrollment_view_count=2 requires enrollment_audio_view2 on all three rows"
        )
    if any(view2_flags):
        for row in ordered_rows:
            _require_text(row, "enrollment_audio_view2")
            if row.get("enrollment_view_count") is not None:
                try:
                    view_count = int(row["enrollment_view_count"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("enrollment_view_count must be an integer") from exc
                if view_count != 2:
                    raise ValueError(
                        "enrollment_audio_view2 is present but enrollment_view_count is not 2"
                    )
    if len(set(speaker_ids)) != 3:
        raise ValueError(
            "query_speaker_id values must be three global speaker identities; "
            "do not substitute A/B/C role IDs"
        )
    if len(set(environment_ids)) != 1:
        raise ValueError("environment_id must be shared by all three query rows")

    return ordered_rows, (mixture_paths[0], mixture_paths[1], mixture_paths[2]), actual_sha


def _read_mono_audio(path: Path, *, field: str) -> np.ndarray:
    try:
        audio, sample_rate = sf.read(
            str(path), dtype="float32", always_2d=False
        )
    except Exception as exc:  # noqa: BLE001 - expose the manifest field in the error
        raise ValueError(f"cannot read {field} audio: {path}") from exc
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(
            f"{field} must be {SAMPLE_RATE} Hz, got {sample_rate} for {path}"
        )
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"{field} must be mono, got shape {audio.shape} for {path}")
    if audio.size < 2:
        raise ValueError(f"{field} must contain at least two samples: {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{field} contains non-finite samples: {path}")
    return np.ascontiguousarray(audio)


def _read_activity(path: Path, *, field: str) -> np.ndarray:
    try:
        activity = np.load(path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - expose the manifest field in the error
        raise ValueError(f"cannot read {field}: {path}") from exc
    activity = np.asarray(activity)
    if activity.ndim != 1 or activity.size < 1:
        raise ValueError(f"{field} must be a non-empty 1-D array: {path}")
    activity = activity.astype(np.float32, copy=False)
    if not np.all(np.isfinite(activity)) or np.any(activity < 0.0) or np.any(activity > 1.0):
        raise ValueError(f"{field} must contain finite values in [0, 1]: {path}")
    return np.ascontiguousarray(activity)


def _pad_waveforms(waveforms: Sequence[np.ndarray]) -> Tensor:
    width = max(int(waveform.size) for waveform in waveforms)
    output = torch.zeros(len(waveforms), width, dtype=torch.float32)
    for row, waveform in enumerate(waveforms):
        output[row, : waveform.size] = torch.from_numpy(waveform)
    return output


def _dynamic_ctc_targets(texts: Sequence[str]) -> tuple[dict[str, int], list[str], Tensor, Tensor]:
    characters = sorted({character for text in texts for character in text})
    char_to_id = {character: index + 1 for index, character in enumerate(characters)}
    lengths = torch.tensor(
        [len(text) for text in texts], dtype=torch.long
    )
    width = max(1, max((len(text) for text in texts), default=0))
    transcript = torch.zeros(len(texts), width, dtype=torch.long)
    for row, text in enumerate(texts):
        if text:
            transcript[row, : len(text)] = torch.tensor(
                [char_to_id[character] for character in text], dtype=torch.long
            )
    return char_to_id, ["<blank>", *characters], transcript, lengths


@dataclass
class DACFBatch:
    """One fixed mixture with ordered A/B/C query rows and optional views."""

    base_mixture_id: str
    rows: tuple[dict[str, Any], ...]
    role_names: tuple[str, str, str]
    sample_role_names: tuple[str, ...]
    view_count: int
    role_ids: Tensor
    speaker_ids: Tensor
    query_speaker_labels: Tensor
    environment_ids: Tensor
    mixture_ids: Tensor
    mixture: Tensor
    enrollment: Tensor
    target_audio: Tensor
    target_activity: Tensor
    target_present: Tensor
    transcript: Tensor
    transcript_lengths: Tensor
    char_to_id: dict[str, int]
    vocab: list[str]
    mixture_sha256: str
    mixture_path: Path
    speaker_id_map: dict[str, int]
    query_speaker_label_map: dict[str, int]
    environment_id_map: dict[str, int]

    @property
    def effective_batch_size(self) -> int:
        return int(self.mixture.shape[0])

    @property
    def identity_positive_pairs(self) -> int:
        """Count unordered same-label pairs in the effective batch."""

        values = self.query_speaker_labels.detach().cpu().tolist()
        counts: dict[int, int] = {}
        for value in values:
            counts[int(value)] = counts.get(int(value), 0) + 1
        return sum(count * (count - 1) // 2 for count in counts.values())

    def targets(
        self,
        *,
        include_ctc: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> dict[str, Tensor]:
        """Build explicit identity and counterfactual target fields."""

        targets: dict[str, Tensor] = {
            "target_present": self.target_present.to(device=device),
            "target_activity": self.target_activity.to(device=device),
            "target_audio": self.target_audio.to(device=device),
            # These two fields intentionally remain separate.  Identity uses
            # the repeated global query_speaker_label; counterfactual pairing
            # uses A/B/C query_role_id.  No speaker ID is hidden in query_id.
            "query_speaker_label": self.query_speaker_labels.to(device=device),
            "query_role_id": self.role_ids.to(device=device),
            "environment_id": self.environment_ids.to(device=device),
            "mixture_id": self.mixture_ids.to(device=device),
        }
        if include_ctc:
            targets["transcript"] = self.transcript.to(device=device)
            targets["transcript_lengths"] = self.transcript_lengths.to(device=device)
        return targets


def load_dacf_group(
    manifest_path: str | Path,
    base_mixture_id: Optional[str] = None,
) -> DACFBatch:
    """Read and validate exactly one DACF base-mixture query group."""

    manifest = Path(manifest_path).resolve(strict=False)
    rows = _read_jsonl(manifest)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get("base_mixture_id")
        if value is None or not str(value).strip():
            raise ValueError("each DACF row requires base_mixture_id")
        groups.setdefault(str(value), []).append(row)
    selected_id = str(base_mixture_id) if base_mixture_id is not None else next(iter(groups))
    if selected_id not in groups:
        raise ValueError(f"base_mixture_id not found in manifest: {selected_id}")
    ordered_rows, mixture_paths, mixture_sha256 = _validate_group_rows(
        groups[selected_id], manifest_path=manifest
    )

    has_view2 = all("enrollment_audio_view2" in row for row in ordered_rows)
    enrollment_view2_paths: tuple[Optional[Path], ...]
    if has_view2:
        enrollment_view2_paths = tuple(
            _resolve_path(
                row["enrollment_audio_view2"],
                manifest,
                field="enrollment_audio_view2",
            )
            for row in ordered_rows
        )
    else:
        enrollment_view2_paths = (None, None, None)
    view_count = 2 if has_view2 else 1
    effective_batch_size = len(ROLE_ORDER) * view_count

    mixture_np = _read_mono_audio(mixture_paths[0], field="recognition_audio")
    mixture_one = torch.from_numpy(mixture_np.copy())
    # Load the recognition waveform once, then make all three rows from this
    # one source tensor.  No query is allowed to re-read or alter the mixture.
    mixture = mixture_one.unsqueeze(0).expand(effective_batch_size, -1).clone()
    if not all(
        torch.equal(mixture[row], mixture_one)
        for row in range(effective_batch_size)
    ):
        raise RuntimeError("internal error: A/B/C mixture copies are not identical")

    enrollment_paths = tuple(
        _resolve_path(row["enrollment_audio"], manifest, field="enrollment_audio")
        for row in ordered_rows
    )
    target_paths = tuple(
        _resolve_path(row[_clean_audio_key(row)], manifest, field="clean_target_audio")
        for row in ordered_rows
    )
    activity_paths = tuple(
        _resolve_path(row["target_activity"], manifest, field="target_activity")
        for row in ordered_rows
    )

    enrollment_np = tuple(
        _read_mono_audio(path, field="enrollment_audio") for path in enrollment_paths
    )
    enrollment_view2_np = (
        tuple(
            _read_mono_audio(path, field="enrollment_audio_view2")
            for path in enrollment_view2_paths
            if path is not None
        )
        if has_view2
        else ()
    )
    target_np = tuple(
        _read_mono_audio(path, field="clean_target_audio") for path in target_paths
    )
    activity_np = tuple(
        _read_activity(path, field="target_activity") for path in activity_paths
    )
    if any(target.size != mixture_np.size for target in target_np):
        raise ValueError("each clean target waveform must match recognition_audio length")

    for index, role in enumerate(ROLE_ORDER):
        expected_present = role != "absent_C"
        if not expected_present:
            if not np.allclose(target_np[index], 0.0, atol=1e-7):
                raise ValueError("absent_C clean target must be silent")
            if not np.allclose(activity_np[index], 0.0, atol=1e-7):
                raise ValueError("absent_C target activity must be all zero")

    speaker_text_base = tuple(
        _require_text(row, "query_speaker_id") for row in ordered_rows
    )
    speaker_label_text_base = tuple(
        _require_text(row, "query_speaker_label") for row in ordered_rows
    )
    environment_text_base = tuple(
        _require_text(row, "environment_id") for row in ordered_rows
    )
    speaker_values = sorted(set(speaker_text_base))
    speaker_label_values = sorted(set(speaker_label_text_base))
    environment_values = sorted(set(environment_text_base))
    speaker_id_map = {value: index for index, value in enumerate(speaker_values)}
    query_speaker_label_map = {
        value: index for index, value in enumerate(speaker_label_values)
    }
    environment_id_map = {value: index for index, value in enumerate(environment_values)}

    sample_roles: list[str] = []
    sample_role_ids: list[int] = []
    sample_speaker_text: list[str] = []
    sample_speaker_label_text: list[str] = []
    sample_environment_text: list[str] = []
    sample_enrollment_np: list[np.ndarray] = []
    sample_target_np: list[np.ndarray] = []
    sample_activity_np: list[np.ndarray] = []
    for index, role in enumerate(ROLE_ORDER):
        view_enrollments = [enrollment_np[index]]
        if has_view2:
            view_enrollments.append(enrollment_view2_np[index])
        for enrollment in view_enrollments:
            sample_roles.append(role)
            sample_role_ids.append(ROLE_TO_ID[role])
            sample_speaker_text.append(speaker_text_base[index])
            sample_speaker_label_text.append(speaker_label_text_base[index])
            sample_environment_text.append(environment_text_base[index])
            sample_enrollment_np.append(enrollment)
            sample_target_np.append(target_np[index])
            sample_activity_np.append(activity_np[index])

    speaker_ids = torch.tensor(
        [speaker_id_map[value] for value in sample_speaker_text], dtype=torch.long
    )
    query_speaker_labels = torch.tensor(
        [query_speaker_label_map[value] for value in sample_speaker_label_text],
        dtype=torch.long,
    )
    environment_ids = torch.tensor(
        [environment_id_map[value] for value in sample_environment_text],
        dtype=torch.long,
    )
    transcripts_base = tuple(
        str(row.get("target_transcript", row.get("ref", "")))
        if role != "absent_C"
        else ""
        for row, role in zip(ordered_rows, ROLE_ORDER)
    )
    transcripts = tuple(
        text for text in transcripts_base for _ in range(view_count)
    )
    char_to_id, vocab, transcript, transcript_lengths = _dynamic_ctc_targets(transcripts)

    return DACFBatch(
        base_mixture_id=selected_id,
        rows=ordered_rows,
        role_names=ROLE_ORDER,
        sample_role_names=tuple(sample_roles),
        view_count=view_count,
        role_ids=torch.tensor(sample_role_ids, dtype=torch.long),
        speaker_ids=speaker_ids,
        query_speaker_labels=query_speaker_labels,
        environment_ids=environment_ids,
        mixture_ids=torch.zeros(effective_batch_size, dtype=torch.long),
        mixture=mixture,
        enrollment=_pad_waveforms(sample_enrollment_np),
        target_audio=torch.stack(
            [torch.from_numpy(audio) for audio in sample_target_np]
        ),
        target_activity=_pad_waveforms(sample_activity_np),
        target_present=torch.tensor(
            [role != "absent_C" for role in sample_roles], dtype=torch.float32
        ),
        transcript=transcript,
        transcript_lengths=transcript_lengths,
        char_to_id=char_to_id,
        vocab=vocab,
        mixture_sha256=mixture_sha256,
        mixture_path=mixture_paths[0],
        speaker_id_map=speaker_id_map,
        query_speaker_label_map=query_speaker_label_map,
        environment_id_map=environment_id_map,
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model has no parameters") from exc


def _loss_value(value: Optional[Tensor]) -> Optional[float]:
    if value is None:
        return None
    scalar = float(value.detach().cpu().item())
    if not np.isfinite(scalar):
        raise RuntimeError(f"non-finite DACF loss: {scalar}")
    return scalar


def _compute_smoke_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Use explicit speaker-label identity and role-id counterfactual terms."""

    base_targets = {
        key: value
        for key, value in targets.items()
        if key not in {"query_speaker_label", "query_role_id", "mixture_id"}
    }
    losses = compute_dacf_loss(outputs, base_targets)
    identity = identity_contrastive_loss(
        outputs["speaker_anchor"], targets["query_speaker_label"]
    )
    counterfactual = counterfactual_delta_loss(
        outputs["target_audio"],
        targets["target_audio"],
        targets["mixture_id"],
        targets["query_role_id"],
    )
    losses["identity"] = identity
    losses["counterfactual"] = counterfactual
    # dacf_frontend's default weights are identity=0.1 and
    # counterfactual=1.0; the other terms are already in losses["total"].
    losses["total"] = losses["total"] + 0.1 * identity + counterfactual
    return losses


def _snapshot(
    outputs: Mapping[str, Tensor],
    losses: Mapping[str, Tensor],
    batch: DACFBatch,
) -> dict[str, Any]:
    loss_report = {name: _loss_value(losses.get(name)) for name in LOSS_NAMES}
    predicted_audio = outputs["target_audio"].detach()
    target_audio = batch.target_audio.to(device=predicted_audio.device)
    presence = outputs["target_present_probs"].detach().cpu().tolist()
    target_l1 = torch.mean(torch.abs(predicted_audio - target_audio), dim=1)
    role_indices = {
        role: [
            index
            for index, sample_role in enumerate(batch.sample_role_names)
            if sample_role == role
        ]
        for role in ROLE_ORDER
    }
    absent_indices = role_indices["absent_C"]
    absent_rms = torch.sqrt(
        torch.mean(predicted_audio[absent_indices].square(), dim=1)
    ).mean()
    role_audio = {
        role: predicted_audio[indices].mean(dim=0)
        for role, indices in role_indices.items()
    }
    deltas: dict[str, float] = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        key = f"{ROLE_ORDER[left]}_vs_{ROLE_ORDER[right]}"
        deltas[key] = float(
            torch.mean(
                torch.abs(role_audio[ROLE_ORDER[left]] - role_audio[ROLE_ORDER[right]])
            )
            .detach()
            .cpu()
            .item()
        )
    deltas["mean_pairwise"] = float(np.mean(list(deltas.values())))
    return {
        "losses": loss_report,
        "presence_prob": {
            role: float(np.mean([presence[index] for index in indices]))
            for role, indices in role_indices.items()
        },
        "target_l1": {
            role: float(target_l1[indices].mean().detach().cpu().item())
            for role, indices in role_indices.items()
        },
        "absent_output_rms": float(absent_rms.detach().cpu().item()),
        "query_swap_audio_delta": deltas,
    }


def _forward_snapshot(
    model: DACFFrontend,
    batch: DACFBatch,
    *,
    include_ctc: bool,
    no_grad: bool,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    device = _model_device(model)
    mixture = batch.mixture.to(device=device)
    enrollment = batch.enrollment.to(device=device)
    targets = batch.targets(include_ctc=include_ctc, device=device)
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        outputs = model(mixture, enrollment)
        losses = _compute_smoke_loss(outputs, targets)
        report = _snapshot(outputs, losses, batch)
    return report, losses


def evaluate_permutation_negative_control(
    model: DACFFrontend,
    batch: DACFBatch,
    *,
    include_ctc: bool = True,
    permutation: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    """Evaluate enrollment permutation while keeping all labels unchanged.

    The default swaps the complete A and B enrollment-view blocks while
    keeping each block's two views together.  This is a deterministic negative
    control on one fixed group, not a statistical test or a generalization
    result.
    """

    if permutation is None:
        source_roles = (1, 0, 2)
        permutation = tuple(
            source_role * batch.view_count + view
            for source_role in source_roles
            for view in range(batch.view_count)
        )
    permutation = tuple(int(index) for index in permutation)
    expected_indices = list(range(batch.effective_batch_size))
    if sorted(permutation) != expected_indices:
        raise ValueError(
            "permutation must contain every effective-batch index exactly once: "
            f"expected={expected_indices}, got={permutation}"
        )
    device = _model_device(model)
    mixture = batch.mixture.to(device=device)
    enrollment = batch.enrollment.to(device=device)
    permutation_tensor = torch.tensor(permutation, dtype=torch.long, device=device)
    targets = batch.targets(include_ctc=include_ctc, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            original_outputs = model(mixture, enrollment)
            permuted_outputs = model(
                mixture, enrollment.index_select(0, permutation_tensor)
            )
            original_losses = _compute_smoke_loss(original_outputs, targets)
            permuted_losses = _compute_smoke_loss(permuted_outputs, targets)
            original = _snapshot(original_outputs, original_losses, batch)
            permuted = _snapshot(permuted_outputs, permuted_losses, batch)
    finally:
        model.train(was_training)

    loss_change = {
        name: (
            None
            if original["losses"][name] is None or permuted["losses"][name] is None
            else permuted["losses"][name] - original["losses"][name]
        )
        for name in LOSS_NAMES
    }
    probability_change = {
        role: permuted["presence_prob"][role] - original["presence_prob"][role]
        for role in ROLE_ORDER
    }
    return {
        "permutation": list(permutation),
        "labels_unchanged": True,
        "original": original,
        "permuted": permuted,
        "loss_change": loss_change,
        "presence_prob_change": probability_change,
        "note": (
            "One deterministic enrollment permutation on one fixed three-query "
            "group; this is not a statistical or generalization conclusion."
        ),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_overfit(
    manifest_path: str | Path,
    *,
    base_mixture_id: Optional[str] = None,
    steps: int = DEFAULT_STEPS,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    disable_ctc: bool = False,
) -> dict[str, Any]:
    """Run the bounded fixed-mixture overfit and return JSON-ready metrics."""

    if isinstance(steps, bool) or int(steps) != steps:
        raise ValueError(f"steps must be an integer in [0, {MAX_STEPS}]")
    steps = int(steps)
    if steps < 0 or steps > MAX_STEPS:
        raise ValueError(f"steps must be in [0, {MAX_STEPS}], got {steps}")
    resolved_device = _resolve_device(device)
    _seed_everything(int(seed))
    batch = load_dacf_group(manifest_path, base_mixture_id)
    model = DACFFrontend(
        n_fft=400,
        hop_length=ACTIVITY_HOP_SAMPLES,
        win_length=400,
        d_model=16,
        n_heads=4,
        vocab_size=max(2, len(batch.vocab)),
        dropout=0.0,
    ).to(resolved_device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    include_ctc = not disable_ctc

    model.eval()
    before, _ = _forward_snapshot(
        model, batch, include_ctc=include_ctc, no_grad=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    backward_ok = False
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, losses = _forward_snapshot(
            model, batch, include_ctc=include_ctc, no_grad=False
        )
        total = losses["total"]
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError("non-finite DACF total loss before optimizer step")
        total.backward()
        backward_ok = True
        optimizer.step()

    model.eval()
    after, _ = _forward_snapshot(
        model, batch, include_ctc=include_ctc, no_grad=True
    )
    negative_control = evaluate_permutation_negative_control(
        model, batch, include_ctc=include_ctc
    )
    absent_control = evaluate_permutation_negative_control(
        model,
        batch,
        include_ctc=include_ctc,
        permutation=(4, 5, 2, 3, 0, 1),
    )
    cuda_peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(resolved_device) / (1024.0 * 1024.0))
        if resolved_device.type == "cuda"
        else None
    )

    return {
        "status": "smoke_only",
        "scope": "fixed-mixture query-control and constrained overfit",
        "limitations": [
            "Does not answer generalization or held-out performance.",
            "Does not answer CER or submission score.",
            "The permutation control is one deterministic group, not a statistical conclusion.",
        ],
        "dataset_a_read": False,
        "submission_chain_used": False,
        "manifest": _path_text(Path(manifest_path).resolve(strict=False)),
        "base_mixture_id": batch.base_mixture_id,
        "mixture_sha256": batch.mixture_sha256,
        "effective_batch_size": batch.effective_batch_size,
        "identity_positive_pairs": batch.identity_positive_pairs,
        "view_count": batch.view_count,
        "activity_hop_samples": ACTIVITY_HOP_SAMPLES,
        "identity_source": "query_speaker_label",
        "counterfactual_source": "query_role_id",
        "vocab": {
            "blank_index": 0,
            "tokens": batch.vocab,
            "char_to_id": batch.char_to_id,
            "size": len(batch.vocab),
        },
        "speaker_id_map": batch.speaker_id_map,
        "query_speaker_label_map": batch.query_speaker_label_map,
        "environment_id_map": batch.environment_id_map,
        "ctc_enabled": include_ctc,
        "before": before,
        "after": after,
        "permutation_negative_control": negative_control,
        "permutation_absent_control": absent_control,
        "steps": steps,
        "backward_ok": backward_ok,
        "device": str(resolved_device),
        "cuda_peak_memory_mib": cuda_peak_memory_mib,
        "seed": int(seed),
        "model": {
            "n_fft": 400,
            "hop_length": ACTIVITY_HOP_SAMPLES,
            "win_length": 400,
            "sample_rate": SAMPLE_RATE,
            "d_model": 16,
            "n_heads": 4,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a bounded DACF fixed-mixture overfit smoke test."
    )
    parser.add_argument("--manifest", required=True, help="DACF JSONL manifest")
    parser.add_argument("--base-mixture-id", default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-json",
        default=None,
        help="optionally save the same UTF-8 JSON result to this path",
    )
    parser.add_argument(
        "--disable-ctc",
        action="store_true",
        help="omit CTC targets/loss to isolate the front-end heads",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.steps > MAX_STEPS:
        parser.error(f"--steps hard limit is {MAX_STEPS}")
    result = run_overfit(
        args.manifest,
        base_mixture_id=args.base_mixture_id,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        disable_ctc=args.disable_ctc,
    )
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output_json:
        output_path = Path(args.output_json)
        _assert_not_dataset_a(output_path, field="output-json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()


__all__ = [
    "DACFBatch",
    "DEFAULT_STEPS",
    "MAX_STEPS",
    "ROLE_ORDER",
    "evaluate_permutation_negative_control",
    "load_dacf_group",
    "run_overfit",
]
