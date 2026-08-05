#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a small DACF D6 presence/environment probe from REAL-PS4.

This is deliberately a *probe*, not an ASR evaluation set.  REAL-PS4's
``enrolment_speakers`` directory contains short real-room speaker fragments,
but not the complete recognition mixtures or transcripts needed for CER.  The
builder therefore creates a deterministic A+B mixture and asks the same
recognition bytes with three enrollment identities in two conditions:

    same_env:  A, B, and absent C enrolled from the mixture session
    cross_env: A, B, and absent C enrolled from another session

The source WAVs are read only.  No model, Dataset-A file, resampler, or large
dataset is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
DEFAULT_SOURCE_ROOT = Path(
    r"E:\midea_datasets\REAL-PS4\AISHELL-4\enrolment_speakers"
)
DEFAULT_BASE_MIXTURES = 4
MAX_BASE_MIXTURES = 16
SCHEMA_VERSION = "dacf-realps4-presence-v0.1"
QUERY_ROLE_IDS = {"present_A": 0, "present_B": 1, "absent_C": 2}
CONDITIONS = ("same_env", "cross_env")

_DATASET_A_MARKERS = ("dataseta", "dataset-a", "dataset_a", "test_wav/dataset")
_FILENAME_RE = re.compile(
    r"^(?P<recording>.+?_R(?P<room>\d+)S(?P<session>\d{2})C(?P<channel>\d+))_"
    r"(?P<speaker>\d{3}-[A-Za-z0-9]+)_"
    r"(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)\.wav$",
    re.IGNORECASE,
)


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def assert_not_dataset_a(value: Any) -> None:
    """Reject Dataset-A-looking paths before metadata or audio access."""
    text = _path_text(value).casefold()
    if any(marker in text for marker in _DATASET_A_MARKERS):
        raise ValueError(f"Dataset-A path is forbidden for REAL-PS4 probe: {value}")


def _canonical_path_key(value: str | Path) -> str:
    """Return one Windows-safe identity key for source-path comparisons."""
    assert_not_dataset_a(value)
    return str(Path(value).resolve()).casefold()


def parse_segment_filename(path: str | Path) -> dict[str, Any]:
    """Parse session, speaker, and source time bounds from one REAL-PS4 name."""
    assert_not_dataset_a(path)
    path_obj = Path(path)
    match = _FILENAME_RE.match(path_obj.name)
    if match is None:
        raise ValueError(
            "REAL-PS4 enrollment filename does not match the expected pattern: "
            f"{path_obj.name}"
        )
    start_sec = float(match.group("start"))
    end_sec = float(match.group("end"))
    if end_sec <= start_sec:
        raise ValueError(f"segment end must be after start: {path_obj.name}")
    session = f"S{int(match.group('session')):02d}"
    speaker_id = match.group("speaker")
    return {
        "path": _path_text(path_obj),
        "filename": path_obj.name,
        "segment_id": path_obj.stem,
        "recording_id": match.group("recording"),
        "room_id": f"R{int(match.group('room')):03d}",
        "session": session,
        "speaker_id": speaker_id,
        "channel_id": f"C{int(match.group('channel')):02d}",
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": end_sec - start_sec,
    }


def validate_wav_16k_mono(path: str | Path) -> dict[str, Any]:
    """Fail fast unless ``path`` is a non-empty 16 kHz mono WAV."""
    assert_not_dataset_a(path)
    path_obj = Path(path)
    try:
        info = sf.info(str(path_obj))
    except Exception as exc:  # noqa: BLE001 - turn backend errors into a source error
        raise ValueError(f"cannot inspect source WAV: {path_obj}") from exc
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise ValueError(
            f"source must be 16 kHz mono, got {info.samplerate} Hz/{info.channels} ch: "
            f"{path_obj}"
        )
    if info.frames <= 0:
        raise ValueError(f"source WAV is empty: {path_obj}")
    return {
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "subtype": str(info.subtype or ""),
    }


def discover_segments(source_root: str | Path) -> list[dict[str, Any]]:
    """Parse and validate every source WAV before selecting any mixture."""
    assert_not_dataset_a(source_root)
    root = Path(source_root)
    if not root.is_dir():
        raise ValueError(f"REAL-PS4 enrollment directory does not exist: {root}")
    paths = sorted(
        (path for path in root.rglob("*.wav") if path.is_file()),
        key=lambda path: _path_text(path).casefold(),
    )
    if not paths:
        raise ValueError(f"no WAV files found under REAL-PS4 enrollment directory: {root}")

    segments: list[dict[str, Any]] = []
    for path in paths:
        metadata = parse_segment_filename(path)
        metadata["audio_format"] = validate_wav_16k_mono(path)
        segments.append(metadata)
    return segments


def _read_audio(path: str | Path) -> np.ndarray:
    """Read one already-validated source without resampling or channel mixing."""
    assert_not_dataset_a(path)
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE or np.asarray(audio).ndim != 1:
        raise ValueError(f"source changed or is not 16 kHz mono: {path}")
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError(f"source audio is empty or non-finite: {path}")
    return audio


def _group_segments(
    segments: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for segment in segments:
        session = str(segment["session"])
        speaker = str(segment["speaker_id"])
        grouped.setdefault(session, {}).setdefault(speaker, []).append(dict(segment))
    for speakers in grouped.values():
        for rows in speakers.values():
            rows.sort(key=lambda row: (float(row["start_sec"]), str(row["path"])))
    return grouped


def _eligible_session_pairs(
    grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[tuple[str, str, list[str]]]:
    sessions = sorted(grouped)
    pairs: list[tuple[str, str, list[str]]] = []
    for mixture_session in sessions:
        for enrollment_session in sessions:
            if mixture_session == enrollment_session:
                continue
            common = sorted(
                set(grouped[mixture_session]) & set(grouped[enrollment_session])
            )
            if len(common) < 3:
                continue
            if any(len(grouped[mixture_session][speaker]) < 2 for speaker in common):
                continue
            if any(len(grouped[enrollment_session][speaker]) < 1 for speaker in common):
                continue
            pairs.append((mixture_session, enrollment_session, common))
    if not pairs:
        raise ValueError(
            "need two sessions sharing at least three speakers; mixture-session "
            "speakers also need two distinct fragments"
        )
    return pairs


def _select_session_pairs(
    session_pairs: Sequence[tuple[str, str, list[str]]],
    base_mixtures: int,
    seed: int,
) -> list[tuple[str, str, list[str]]]:
    """Round-robin mixture sessions, with deterministic order and cross choice."""
    by_mixture_session: dict[str, list[tuple[str, str, list[str]]]] = {}
    for pair in session_pairs:
        by_mixture_session.setdefault(pair[0], []).append(pair)
    for pairs in by_mixture_session.values():
        pairs.sort(key=lambda pair: pair[1])

    mixture_sessions = sorted(by_mixture_session)
    random.Random(int(seed)).shuffle(mixture_sessions)
    selected: list[tuple[str, str, list[str]]] = []
    for index in range(base_mixtures):
        mixture_session = mixture_sessions[index % len(mixture_sessions)]
        cross_options = by_mixture_session[mixture_session]
        cross_round = index // len(mixture_sessions)
        cross_index = (int(seed) + cross_round) % len(cross_options)
        selected.append(cross_options[cross_index])
    return selected


def _segment_copy(segment: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only JSON-safe source metadata in an emitted manifest."""
    return {
        "path": _path_text(segment["path"]),
        "segment_id": str(segment["segment_id"]),
        "session": str(segment["session"]),
        "speaker_id": str(segment["speaker_id"]),
        "recording_id": str(segment["recording_id"]),
        "start_sec": float(segment["start_sec"]),
        "end_sec": float(segment["end_sec"]),
        "duration_sec": float(segment["duration_sec"]),
    }


def _source_environment_id(segment: Mapping[str, Any]) -> str:
    """Build an auditable source context ID without pretending it is a label."""
    return (
        f"session={segment['session']}|recording={segment['recording_id']}|"
        f"channel={segment['channel_id']}"
    )


def _synthetic_mixture_environment_id(
    source_a: Mapping[str, Any], source_b: Mapping[str, Any]
) -> str:
    contexts = sorted({_source_environment_id(source_a), _source_environment_id(source_b)})
    return "synthetic_sum|" + "+".join(contexts)


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _make_ab_mixture(
    target: np.ndarray,
    interferer: np.ndarray,
    *,
    overlap_ratio: float,
    sir_db: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Place A at zero and B at a deterministic overlap offset, then set SIR."""
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError("overlap_ratio must be between 0 and 1")
    if target.size == 0 or interferer.size == 0:
        raise ValueError("cannot mix an empty source segment")

    overlap_samples = int(round(min(target.size, interferer.size) * overlap_ratio))
    b_start = int(target.size - overlap_samples)
    output_size = max(target.size, b_start + interferer.size)
    clean_a = np.zeros(output_size, dtype=np.float32)
    clean_b = np.zeros(output_size, dtype=np.float32)
    clean_a[: target.size] = target

    interferer_gain = 1.0
    target_rms = _rms(target)
    interferer_rms = _rms(interferer)
    if target_rms > 1e-12 and interferer_rms > 1e-12:
        interferer_gain = target_rms / (
            interferer_rms * (10.0 ** (float(sir_db) / 20.0))
        )
    clean_b[b_start : b_start + interferer.size] = (
        interferer * np.float32(interferer_gain)
    )
    mixture = clean_a + clean_b

    peak = float(
        max(
            np.max(np.abs(mixture)) if mixture.size else 0.0,
            np.max(np.abs(clean_a)) if clean_a.size else 0.0,
            np.max(np.abs(clean_b)) if clean_b.size else 0.0,
        )
    )
    normalization_scale = 1.0 if peak <= 0.98 else 0.98 / peak
    mixture *= np.float32(normalization_scale)
    clean_a *= np.float32(normalization_scale)
    clean_b *= np.float32(normalization_scale)
    timing = {
        "target_start_samples": 0,
        "interferer_start_samples": b_start,
        "overlap_ratio_requested": float(overlap_ratio),
        "overlap_samples": overlap_samples,
        "output_samples": int(output_size),
        "sample_rate": SAMPLE_RATE,
        "sir_db_requested": float(sir_db),
        "sir_db_measured": float(
            20.0 * np.log10(_rms(clean_a) / max(_rms(clean_b), 1e-12))
        )
        if _rms(clean_a) > 1e-12 and _rms(clean_b) > 1e-12
        else None,
        "peak_normalization_scale": float(normalization_scale),
    }
    return mixture.astype(np.float32), clean_a, clean_b, timing


def _activity_from_clean(clean: np.ndarray, hop_samples: int) -> np.ndarray:
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    frame_count = max(1, int(np.ceil(clean.size / hop_samples)))
    padded = np.pad(
        np.asarray(clean, dtype=np.float32),
        (0, frame_count * hop_samples - clean.size),
    )
    frames = padded.reshape(frame_count, hop_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return (rms > 1e-8).astype(np.uint8)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_build_args(base_mixtures: int, overlap_ratio: float, hop_samples: int) -> None:
    if base_mixtures <= 0:
        raise ValueError("base_mixtures must be positive")
    if base_mixtures > MAX_BASE_MIXTURES:
        raise ValueError(
            f"REAL-PS4 probe is capped at {MAX_BASE_MIXTURES} base mixtures"
        )
    if not 0.0 <= overlap_ratio <= 1.0:
        raise ValueError("overlap_ratio must be between 0 and 1")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")


def _assert_output_is_not_source(source_root: Path, out_dir: Path) -> None:
    source_resolved = source_root.resolve()
    output_resolved = out_dir.resolve()
    if output_resolved == source_resolved or source_resolved in output_resolved.parents:
        raise ValueError("output directory must not be inside the source directory")


def build_dacf_realps4_probe(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    out_dir: str | Path = "realps4_probe",
    *,
    base_mixtures: int = DEFAULT_BASE_MIXTURES,
    seed: int = 20260806,
    overlap_ratio: float = 0.5,
    sir_db: float = 0.0,
    hop_samples: int = 160,
) -> dict[str, Any]:
    """Build deterministic same/cross-session presence queries.

    Each base mixture contributes six rows: present A/B and absent C under
    both conditions.  A and B recognition sources are distinct fragments, and
    each present enrollment is another fragment of the same speaker.  C is
    enrollment-only and never enters the mixture.
    """
    _validate_build_args(base_mixtures, overlap_ratio, hop_samples)
    assert_not_dataset_a(source_root)
    assert_not_dataset_a(out_dir)
    source_root_path = Path(source_root)
    out_path = Path(out_dir)
    _assert_output_is_not_source(source_root_path, out_path)

    segments = discover_segments(source_root_path)
    grouped = _group_segments(segments)
    speaker_label_map = {
        speaker_id: label
        for label, speaker_id in enumerate(
            sorted({str(segment["speaker_id"]) for segment in segments})
        )
    }
    session_pairs = _eligible_session_pairs(grouped)
    if base_mixtures > len(session_pairs):
        raise ValueError(
            f"requested {base_mixtures} base mixtures but only "
            f"{len(session_pairs)} eligible ordered session pairs exist"
        )
    selected_session_pairs = _select_session_pairs(
        session_pairs, base_mixtures, seed
    )

    out_path.mkdir(parents=True, exist_ok=True)
    audio_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    base_summaries: list[dict[str, Any]] = []

    for mixture_index, pair in enumerate(selected_session_pairs):
        mixture_session, enrollment_session, common_speakers = pair
        speaker_start = (int(seed) + mixture_index) % len(common_speakers)
        speaker_ids = [
            common_speakers[(speaker_start + offset) % len(common_speakers)]
            for offset in range(3)
        ]
        speaker_a, speaker_b, speaker_c = speaker_ids

        mixture_groups = grouped[mixture_session]
        cross_groups = grouped[enrollment_session]
        rec_a_index = (int(seed) + mixture_index * 3) % len(
            mixture_groups[speaker_a]
        )
        rec_b_index = (int(seed) + mixture_index * 5 + 1) % len(
            mixture_groups[speaker_b]
        )
        rec_a = mixture_groups[speaker_a][rec_a_index]
        rec_b = mixture_groups[speaker_b][rec_b_index]
        same_enroll = {
            speaker_a: mixture_groups[speaker_a][(rec_a_index + 1) % len(mixture_groups[speaker_a])],
            speaker_b: mixture_groups[speaker_b][(rec_b_index + 1) % len(mixture_groups[speaker_b])],
            speaker_c: mixture_groups[speaker_c][
                (int(seed) + mixture_index * 7) % len(mixture_groups[speaker_c])
            ],
        }
        cross_enroll = {
            speaker: cross_groups[speaker][
                (int(seed) + mixture_index * 11 + speaker_offset) % len(cross_groups[speaker])
            ]
            for speaker_offset, speaker in enumerate(speaker_ids)
        }
        recognition_a = audio_cache.setdefault(str(rec_a["path"]), _read_audio(rec_a["path"]))
        recognition_b = audio_cache.setdefault(str(rec_b["path"]), _read_audio(rec_b["path"]))
        mixture, clean_a, clean_b, timing = _make_ab_mixture(
            recognition_a,
            recognition_b,
            overlap_ratio=overlap_ratio,
            sir_db=sir_db,
        )

        base_id = f"realps4_probe_{mixture_index:04d}"
        recognition_path = out_path / "recognition" / f"{base_id}.wav"
        _write_wav(recognition_path, mixture)
        mixture_sha256 = _sha256_file(recognition_path)

        clean_by_role = {
            "present_A": clean_a,
            "present_B": clean_b,
            "absent_C": np.zeros_like(mixture, dtype=np.float32),
        }
        clean_paths: dict[str, Path] = {}
        activity_paths: dict[str, Path] = {}
        for role, clean in clean_by_role.items():
            clean_path = out_path / "clean_target" / f"{base_id}__{role}.wav"
            activity_path = out_path / "activity" / f"{base_id}__{role}.npy"
            _write_wav(clean_path, clean)
            activity_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(activity_path, _activity_from_clean(clean, hop_samples))
            clean_paths[role] = clean_path
            activity_paths[role] = activity_path

        recognition_source_paths = [
            _path_text(rec_a["path"]),
            _path_text(rec_b["path"]),
        ]
        recognition_source_keys = [
            _canonical_path_key(rec_a["path"]),
            _canonical_path_key(rec_b["path"]),
        ]
        mixture_environment_id = _synthetic_mixture_environment_id(rec_a, rec_b)
        condition_enrollments = {
            "same_env": same_enroll,
            "cross_env": cross_enroll,
        }
        for condition in CONDITIONS:
            for role, speaker_id in (
                ("present_A", speaker_a),
                ("present_B", speaker_b),
                ("absent_C", speaker_c),
            ):
                enrollment_source = condition_enrollments[condition][speaker_id]
                enrollment_path = (
                    out_path
                    / "enrollment"
                    / f"{base_id}__{condition}__{role}.wav"
                )
                enrollment_path_key = str(enrollment_source["path"])
                enrollment_audio = audio_cache.setdefault(
                    enrollment_path_key, _read_audio(enrollment_source["path"])
                )
                _write_wav(enrollment_path, enrollment_audio)
                enrollment_source_key = _canonical_path_key(enrollment_source["path"])
                if enrollment_source_key in recognition_source_keys:
                    raise AssertionError(
                        "enrollment source must differ from every recognition source"
                    )
                enrollment_environment_id = _source_environment_id(enrollment_source)
                target_present = role != "absent_C"
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "record_kind": "dacf_realps4_query",
                    "id": f"{base_id}__{condition}__{role}",
                    "base_mixture_id": base_id,
                    "condition": condition,
                    "query_role": role,
                    "query_role_id": QUERY_ROLE_IDS[role],
                    "query_speaker_id": speaker_id,
                    "query_speaker_label": int(speaker_label_map[speaker_id]),
                    "query_speaker_label_contract": (
                        "zero-based label from sorted unique source speaker_id"
                    ),
                    "target_present": target_present,
                    "mixture_session": mixture_session,
                    "enrollment_session": str(enrollment_source["session"]),
                    "mixture_speakers": {
                        "A": speaker_a,
                        "B": speaker_b,
                    },
                    "recognition_audio": _path_text(recognition_path),
                    "recognition_source_audio": recognition_source_paths,
                    "recognition_source_keys": recognition_source_keys,
                    "recognition_source_segments": [
                        _segment_copy(rec_a),
                        _segment_copy(rec_b),
                    ],
                    "enrollment_audio": _path_text(enrollment_path),
                    "enrollment_source_audio": enrollment_path_key,
                    "enrollment_source_key": enrollment_source_key,
                    "enrollment_source_segment": _segment_copy(enrollment_source),
                    "clean_target_audio": _path_text(clean_paths[role]),
                    "target_activity": _path_text(activity_paths[role]),
                    "target_activity_hop_samples": int(hop_samples),
                    "target_activity_source": "clean_target_audio",
                    "clean_target_is_empty": not target_present,
                    "mixture_sha256": mixture_sha256,
                    "mixture_environment_id": mixture_environment_id,
                    "enrollment_environment_id": enrollment_environment_id,
                    "dataset_a_used": False,
                    "transcript_unavailable": True,
                    "source_corpus": "REAL-PS4/AISHELL-4/enrolment_speakers",
                    "source_audio_unchanged": True,
                    "timing": timing,
                    "enrollment_is_recognition_source": False,
                    "mixture_is_native_real_overlap": False,
                    "background_is_superposed": True,
                    "speaker_disjoint": False,
                }
                rows.append(row)

        base_summaries.append(
            {
                "base_mixture_id": base_id,
                "mixture_session": mixture_session,
                "enrollment_session": enrollment_session,
                "mixture_speakers": {"A": speaker_a, "B": speaker_b},
                "absent_speaker": speaker_c,
                "recognition_source_audio": recognition_source_paths,
                "mixture_environment_id": mixture_environment_id,
                "mixture_sha256": mixture_sha256,
                "record_count": 6,
                "timing": timing,
            }
        )

    manifest_path = out_path / "manifest.jsonl"
    _write_jsonl(manifest_path, rows)
    condition_counts = {
        condition: sum(row["condition"] == condition for row in rows)
        for condition in CONDITIONS
    }
    role_counts = {
        role: sum(row["query_role"] == role for row in rows)
        for role in QUERY_ROLE_IDS
    }
    grid_counts = {
        f"{condition}__{'correct' if present else 'wrong'}": sum(
            row["condition"] == condition and row["target_present"] is present
            for row in rows
        )
        for condition in CONDITIONS
        for present in (True, False)
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "readme": {
            "title": "DACF D6 REAL-PS4 same/cross-environment presence probe",
            "purpose": "Mechanism audit of speaker presence and environment transfer.",
            "design": (
                "The recognition WAV is a deterministic synthetic sum of two different "
                "real meeting-room fragments (A+B). The second fragment is additive "
                "background; this is not a native REAL-PS4 overlap recording. It is "
                "queried with present A, present B, and absent C enrollments from the "
                "mixture session and from another session."
            ),
            "comparison_grid": {
                "same_env_correct": "present A/B enrollment from mixture_session",
                "cross_env_correct": "present A/B enrollment from enrollment_session",
                "same_env_wrong": "absent C enrollment from mixture_session",
                "cross_env_wrong": "absent C enrollment from enrollment_session",
            },
            "not_for": [
                "CER: REAL-PS4 enrollment_speakers has no complete recognition transcript here",
                "threshold selection or Dataset-A score claims",
                "speaker-disjoint generalization claims",
            ],
            "freeze_gate": (
                "Run only after the model is frozen from external speaker-disjoint data "
                "such as AISHELL-1; use this set for mechanism audit, not calibration."
            ),
            "training_allowed": False,
            "threshold_selection_allowed": False,
        },
        "source": {
            "root": _path_text(source_root_path),
            "wav_count": len(segments),
            "sessions": sorted(grouped),
            "speakers_by_session": {
                session: sorted(speakers) for session, speakers in sorted(grouped.items())
            },
            "speaker_label_map": speaker_label_map,
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "transcript_unavailable": True,
            "dataset_a_used": False,
            "source_audio_unchanged": True,
        },
        "probe": {
            "base_mixtures": base_mixtures,
            "record_count": len(rows),
            "records_per_base": 6,
            "default_base_mixtures": DEFAULT_BASE_MIXTURES,
            "max_base_mixtures": MAX_BASE_MIXTURES,
            "seed": int(seed),
            "overlap_ratio": float(overlap_ratio),
            "sir_db": float(sir_db),
            "condition_counts": condition_counts,
            "role_counts": role_counts,
            "comparison_grid_counts": grid_counts,
            "recognition_audio_reused_across_conditions": True,
            "synthetic_sum_of_real_fragments": True,
            "mixture_is_native_real_overlap": False,
            "background_is_superposed": True,
        },
        "policy": {
            "training_allowed": False,
            "threshold_selection_allowed": False,
            "cer_evaluation_available": False,
            "dataset_a_used": False,
        },
        "limitations": [
            "The same four speaker IDs repeat across sessions, so this is not speaker-disjoint.",
            "There is no complete mixture/transcript package in this input; this is not a CER set.",
            "The A+B recognition WAV is a synthetic sum of two real fragments, not native overlap; background is superposed.",
            "Do not train a model, select a rejection threshold, or claim Dataset-A generalization from this probe.",
        ],
        "artifacts": {
            "manifest_jsonl": _path_text(manifest_path),
            "base_mixtures": base_summaries,
        },
    }
    _write_json(out_path / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="REAL-PS4/AISHELL-4/enrolment_speakers directory",
    )
    parser.add_argument("--out", required=True, help="small probe output directory")
    parser.add_argument(
        "--base-mixtures",
        type=int,
        default=DEFAULT_BASE_MIXTURES,
        help=f"number of bases (default {DEFAULT_BASE_MIXTURES}, hard cap {MAX_BASE_MIXTURES})",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--sir-db", type=float, default=0.0)
    parser.add_argument("--hop-samples", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_dacf_realps4_probe(
        args.source_root,
        args.out,
        base_mixtures=args.base_mixtures,
        seed=args.seed,
        overlap_ratio=args.overlap_ratio,
        sir_db=args.sir_db,
        hop_samples=args.hop_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
