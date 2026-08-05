"""Build an auditable CAM++ feature layer for the DACF A/B/C manifests.

The builder is intentionally a feature-only experiment.  It never trains a
model, chooses an operating threshold, reads Dataset-A, or passes
``query_role_id`` to the extractor.  The immutable mixture is keyed by its
verified byte SHA256, so the three counterfactual rows can only point at the
same mixture feature artifact.

The real CLI is meant for ``code/.venv_realt`` where ``sherpa_onnx`` and the
local CAM++ ONNX file are already available.  Unit tests inject a fake
extractor and therefore do not load the ONNX model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
FEATURE_DIM = 512
DEFAULT_WINDOW_SEC = 1.5
DEFAULT_HOP_SEC = 0.5
DEFAULT_THREADS = 2
ROLE_IDS = (0, 1, 2)
PRESENT_ROLE_IDS = (0, 1)
ABSENT_ROLE_ID = 2

_DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)
_MIXTURE_KEYS = ("recognition_audio", "mixture_audio", "mixture_path")
_ENROLLMENT_KEYS = ("enrollment_audio", "enrollment_path")
_VIEW2_KEYS = (
    "enrollment_audio_view2",
    "enrollment_view2_audio",
    "view2_audio",
    "enrollment_audio2",
)


class ManifestContractError(ValueError):
    """Raised when an input manifest violates the DACF audit contract."""


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _looks_like_dataset_a(value: Any) -> bool:
    text = _path_text(value).casefold()
    if any(marker in text for marker in _DATASET_A_MARKERS):
        return True
    # Also catch spelling variants such as ``Dataset A`` and mixed separators.
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "dataseta" in compact


def _assert_not_dataset_a(value: Any, *, field: str = "path") -> None:
    """Hard reject Dataset-A paths/markers before opening any audio."""

    if _looks_like_dataset_a(value):
        raise ManifestContractError(
            f"{field} contains forbidden Dataset-A path/marker: {value}"
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n", ""}:
            return False
    return bool(value)


def _guard_dataset_a(value: Any, field: str) -> None:
    """Recursively guard all manifest values and explicit flag fields."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}"
            lowered = str(key).casefold()
            if (
                lowered in {"dataset_a_used", "dataset_a", "used_dataset_a"}
                and _as_bool(child)
            ):
                raise ManifestContractError(
                    f"{child_field}=true is forbidden: Dataset-A may not enter DACF features"
                )
            _guard_dataset_a(child, child_field)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _guard_dataset_a(child, f"{field}[{index}]")
        return
    if isinstance(value, (str, Path)):
        _assert_not_dataset_a(value, field=field)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).resolve(strict=True)
    _assert_not_dataset_a(manifest, field="manifest")
    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestContractError(
                    f"invalid JSONL at {manifest}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ManifestContractError(
                    f"manifest row is not an object at {manifest}:{line_number}"
                )
            _guard_dataset_a(row, f"{manifest}:{line_number}")
            rows.append(row)
    if not rows:
        raise ManifestContractError(f"manifest is empty: {manifest}")
    return rows


def _resolve_path(raw: Any, manifest_path: Path, *, field: str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ManifestContractError(f"missing path field {field}")
    _assert_not_dataset_a(raw, field=field)
    raw_path = Path(str(raw))
    if raw_path.is_absolute():
        candidates = (raw_path,)
    else:
        candidates = (Path.cwd() / raw_path, manifest_path.parent / raw_path)
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        _assert_not_dataset_a(resolved, field=field)
        if resolved.exists() and resolved.is_file():
            existing.append(resolved)
    unique = list(dict.fromkeys(existing))
    if len(unique) > 1:
        raise ManifestContractError(
            f"ambiguous {field}={raw!r}; multiple existing paths: {unique}"
        )
    if not unique:
        shown = ", ".join(str(p.resolve(strict=False)) for p in candidates)
        raise ManifestContractError(
            f"cannot resolve {field}={raw!r}; candidates: {shown}"
        )
    return unique[0]


def _first_path(row: Mapping[str, Any], keys: Sequence[str], *, field: str) -> Any:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (str, Path)) and str(value).strip():
            return value
    raise ManifestContractError(f"row {row.get('id', '<unknown>')} requires {field}")


def _role_id(row: Mapping[str, Any]) -> int:
    raw = row.get("query_role_id")
    if isinstance(raw, bool) or raw is None:
        raise ManifestContractError(
            f"row {row.get('id', '<unknown>')} requires integer query_role_id"
        )
    try:
        role = int(raw)
    except (TypeError, ValueError) as exc:
        raise ManifestContractError(f"invalid query_role_id={raw!r}") from exc
    if role not in ROLE_IDS:
        raise ManifestContractError(f"query_role_id must be 0/1/2, got {role}")
    expected = {0: "present_A", 1: "present_B", 2: "absent_C"}[role]
    declared = str(row.get("query_role", "")).strip()
    aliases = {"A": "present_A", "B": "present_B", "C": "absent_C"}
    declared = aliases.get(declared, declared)
    if declared and declared != expected:
        raise ManifestContractError(
            f"query_role/query_role_id mismatch: {declared!r} vs {expected!r}"
        )
    return role


def _group_id(row: Mapping[str, Any]) -> str:
    value = row.get("base_mixture_id")
    if value is None or not str(value).strip():
        value = row.get("counterfactual_group_key")
        if value is not None:
            value = str(value).split(":", 1)[0]
    if value is None or not str(value).strip():
        raise ManifestContractError(
            f"row {row.get('id', '<unknown>')} requires base_mixture_id"
        )
    return str(value)


def _iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_values(child)
    else:
        yield value


def _speaker_ids(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "query_speaker_id",
        "enrollment_spk",
        "target_spk",
        "interferer_spks",
        "hard_negative_interferer_spks",
        "mixture_speakers",
    ):
        for value in _iter_values(row.get(key)):
            if value is not None and str(value).strip():
                values.add(str(value).strip())
    return values


def _source_ids(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "enrollment_src",
        "target_src",
        "interferer_srcs",
        "mixture_sources",
    ):
        for value in _iter_values(row.get(key)):
            if value is None or not str(value).strip():
                continue
            # A canonical resolved string makes relative/absolute spellings
            # comparable while keeping the audit independent of audio reads.
            values.add(str(Path(str(value)).resolve(strict=False)).casefold())
    return values


def _validate_split_rows(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> tuple[dict[str, list[dict[str, Any]]], set[str], set[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    speakers: set[str] = set()
    sources: set[str] = set()
    for index, row in enumerate(rows):
        if str(row.get("split", "")) != split:
            raise ManifestContractError(
                f"{split} manifest row {index} declares split={row.get('split')!r}"
            )
        if _as_bool(row.get("dataset_a_used", False)):
            raise ManifestContractError(f"{split} row {index} has dataset_a_used=true")
        if str(row.get("dataset_a_policy", "forbidden")).casefold() not in {
            "forbidden",
            "never",
            "not_used",
            "",
        }:
            raise ManifestContractError(
                f"{split} row {index} has non-forbidden dataset_a_policy"
            )
        _role_id(row)
        _first_path(row, _MIXTURE_KEYS, field="recognition_audio/mixture_audio")
        _first_path(row, _ENROLLMENT_KEYS, field="enrollment_audio")
        _first_path(row, _VIEW2_KEYS, field="enrollment_audio_view2")
        gid = _group_id(row)
        groups[gid].append(dict(row))
        speakers.update(_speaker_ids(row))
        sources.update(_source_ids(row))

    for gid, group in groups.items():
        if len(group) != 3:
            raise ManifestContractError(
                f"{split} group {gid!r} must contain exactly A/B/C rows, got {len(group)}"
            )
        roles = [_role_id(row) for row in group]
        if sorted(roles) != [0, 1, 2]:
            raise ManifestContractError(
                f"{split} group {gid!r} must contain role ids 0,1,2, got {roles}"
            )
        declared_hashes = {
            str(row.get("mixture_sha256", "")).strip().casefold() for row in group
        }
        if "" in declared_hashes or len(declared_hashes) != 1:
            raise ManifestContractError(
                f"{split} group {gid!r} has inconsistent/missing mixture_sha256"
            )
    return groups, speakers, sources


def validate_manifests(
    train_manifest: str | Path, val_manifest: str | Path
) -> dict[str, Any]:
    """Read and audit both manifests before opening any audio."""

    train_path = Path(train_manifest).resolve(strict=True)
    val_path = Path(val_manifest).resolve(strict=True)
    _assert_not_dataset_a(train_path, field="train-manifest")
    _assert_not_dataset_a(val_path, field="val-manifest")
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    train_groups, train_speakers, train_sources = _validate_split_rows(
        train_rows, split="train"
    )
    val_groups, val_speakers, val_sources = _validate_split_rows(val_rows, split="val")
    speaker_overlap = sorted(train_speakers & val_speakers)
    source_overlap = sorted(train_sources & val_sources)
    if speaker_overlap:
        raise ManifestContractError(
            f"train/val speaker overlap is forbidden: {speaker_overlap[:8]}"
        )
    if source_overlap:
        raise ManifestContractError(
            f"train/val source overlap is forbidden: {source_overlap[:8]}"
        )
    return {
        "train": {
            "path": train_path,
            "rows": train_rows,
            "groups": train_groups,
            "speakers": train_speakers,
            "sources": train_sources,
        },
        "val": {
            "path": val_path,
            "rows": val_rows,
            "groups": val_groups,
            "speakers": val_speakers,
            "sources": val_sources,
        },
        "audit": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "train_speakers": len(train_speakers),
            "val_speakers": len(val_speakers),
            "train_sources": len(train_sources),
            "val_sources": len(val_sources),
            "speaker_overlap": speaker_overlap,
            "source_overlap": source_overlap,
            "dataset_a_used": False,
        },
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_audio_bytes(data: bytes, *, source: Path) -> tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:
        raise RuntimeError(f"cannot decode audio bytes from {source}") from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"audio is empty or not mono-decodable: {source}")
    waveform = np.ascontiguousarray(waveform)
    if int(sample_rate) == SAMPLE_RATE:
        return waveform, SAMPLE_RATE
    try:
        import sherpa_onnx

        waveform = sherpa_onnx.resample(waveform, int(sample_rate), SAMPLE_RATE)
    except Exception:
        try:
            import librosa

            waveform = librosa.resample(
                waveform, orig_sr=int(sample_rate), target_sr=SAMPLE_RATE
            )
        except Exception as exc:
            raise RuntimeError(
                f"audio sample rate {sample_rate} requires sherpa_onnx or librosa: {source}"
            ) from exc
    return np.ascontiguousarray(np.asarray(waveform, dtype=np.float32)), SAMPLE_RATE


def _pad_signal(waveform: np.ndarray, target_samples: int) -> tuple[np.ndarray, int]:
    """Pad with reflected audio, falling back to tiling for very short audio."""

    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size >= target_samples:
        return np.ascontiguousarray(signal[:target_samples]), 0
    if signal.size == 0:
        padded = np.zeros(target_samples, dtype=np.float32)
        return padded, target_samples
    missing = target_samples - signal.size
    try:
        padded = np.pad(signal, (0, missing), mode="reflect")
    except (ValueError, RuntimeError):
        repeats = int(math.ceil(target_samples / signal.size))
        padded = np.tile(signal, repeats)[:target_samples]
    return np.ascontiguousarray(padded.astype(np.float32)), int(missing)


def window_schedule(
    sample_count: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
) -> list[tuple[int, int]]:
    """Return windows with a guaranteed final window covering the tail."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    window_samples = int(round(float(window_sec) * sample_rate))
    hop_samples = int(round(float(hop_sec) * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window and hop must be positive")
    if sample_count <= window_samples:
        return [(0, sample_count)]
    last_start = sample_count - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, min(start + window_samples, sample_count)) for start in starts]


def _normalize_embedding(
    value: Any, *, expected_dim: int = FEATURE_DIM
) -> tuple[np.ndarray, float, float]:
    raw = np.asarray(value, dtype=np.float32).reshape(-1)
    if raw.size != expected_dim:
        raise RuntimeError(f"CAM++ embedding dimension {raw.size}, expected {expected_dim}")
    if not np.isfinite(raw).all():
        raise RuntimeError("CAM++ embedding contains non-finite values")
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm <= 1e-8:
        raise RuntimeError("CAM++ returned a near-zero embedding")
    normalized = np.ascontiguousarray(raw / raw_norm, dtype=np.float32)
    return normalized, raw_norm, abs(float(np.linalg.norm(normalized)) - 1.0)


def extract_embedding(
    extractor: Any,
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    expected_dim: int = FEATURE_DIM,
    min_samples: int = 1_600,
    stats: Optional[MutableMapping[str, int]] = None,
) -> np.ndarray:
    """Extract one L2-normalized 512d vector through sherpa-onnx's stream API."""

    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise RuntimeError("cannot extract CAM++ embedding from empty audio")
    if signal.size < min_samples:
        signal, _ = _pad_signal(signal, min_samples)
    stream = extractor.create_stream()
    stream.accept_waveform(int(sample_rate), np.ascontiguousarray(signal))
    stream.input_finished()
    embedding = extractor.compute(stream)
    normalized, _, _ = _normalize_embedding(embedding, expected_dim=expected_dim)
    if stats is not None:
        stats["extractor_calls"] = int(stats.get("extractor_calls", 0)) + 1
    return normalized


def _extract_with_audit(
    extractor: Any,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    expected_dim: int,
    stats: MutableMapping[str, int],
) -> tuple[np.ndarray, float, float]:
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise RuntimeError("cannot extract CAM++ embedding from empty audio")
    if signal.size < 1_600:
        signal, _ = _pad_signal(signal, 1_600)
    stream = extractor.create_stream()
    stream.accept_waveform(int(sample_rate), np.ascontiguousarray(signal))
    stream.input_finished()
    raw = extractor.compute(stream)
    normalized, raw_norm, norm_error = _normalize_embedding(raw, expected_dim=expected_dim)
    stats["extractor_calls"] = int(stats.get("extractor_calls", 0)) + 1
    return normalized, raw_norm, norm_error


def extract_mixture_features(
    extractor: Any,
    waveform: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
    expected_dim: int = FEATURE_DIM,
    stats: Optional[MutableMapping[str, int]] = None,
) -> dict[str, Any]:
    """Extract whole-mixture and tail-covered sliding-window features."""

    local_stats: MutableMapping[str, int] = stats if stats is not None else {}
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise RuntimeError("mixture audio is empty")
    whole, whole_raw_norm, whole_norm_error = _extract_with_audit(
        extractor,
        signal,
        sample_rate,
        expected_dim=expected_dim,
        stats=local_stats,
    )
    window_samples = int(round(window_sec * sample_rate))
    windows: list[np.ndarray] = []
    starts: list[int] = []
    ends: list[int] = []
    padded_samples: list[int] = []
    raw_norms: list[float] = []
    norm_errors: list[float] = []
    for start, end in window_schedule(
        signal.size,
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    ):
        window, padding = _pad_signal(signal[start:end], window_samples)
        embedding, raw_norm, norm_error = _extract_with_audit(
            extractor,
            window,
            sample_rate,
            expected_dim=expected_dim,
            stats=local_stats,
        )
        windows.append(embedding)
        starts.append(int(start))
        ends.append(int(end))
        padded_samples.append(int(padding))
        raw_norms.append(float(raw_norm))
        norm_errors.append(float(norm_error))
    return {
        "whole_embedding": whole,
        "whole_raw_norm": float(whole_raw_norm),
        "whole_normalization_error": float(whole_norm_error),
        "window_embeddings": np.ascontiguousarray(np.stack(windows), dtype=np.float32),
        "window_start_samples": np.asarray(starts, dtype=np.int64),
        "window_end_samples": np.asarray(ends, dtype=np.int64),
        "window_padded_samples": np.asarray(padded_samples, dtype=np.int64),
        "window_raw_norms": np.asarray(raw_norms, dtype=np.float32),
        "window_normalization_errors": np.asarray(norm_errors, dtype=np.float32),
        "sample_rate": int(sample_rate),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "feature_dim": int(expected_dim),
    }


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("._") or "item"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _save_mixture_npz(path: Path, features: Mapping[str, Any], mixture_sha256: str) -> None:
    np.savez_compressed(
        path,
        mixture_embedding=np.asarray(features["whole_embedding"], dtype=np.float32),
        window_embeddings=np.asarray(features["window_embeddings"], dtype=np.float32),
        window_start_samples=np.asarray(features["window_start_samples"], dtype=np.int64),
        window_end_samples=np.asarray(features["window_end_samples"], dtype=np.int64),
        window_padded_samples=np.asarray(features["window_padded_samples"], dtype=np.int64),
        window_raw_norms=np.asarray(features["window_raw_norms"], dtype=np.float32),
        window_normalization_errors=np.asarray(
            features["window_normalization_errors"], dtype=np.float32
        ),
        mixture_sha256=np.asarray(mixture_sha256),
        sample_rate=np.asarray(features["sample_rate"], dtype=np.int32),
        window_sec=np.asarray(features["window_sec"], dtype=np.float32),
        hop_sec=np.asarray(features["hop_sec"], dtype=np.float32),
        feature_dim=np.asarray(features["feature_dim"], dtype=np.int32),
        whole_raw_norm=np.asarray(features["whole_raw_norm"], dtype=np.float32),
        whole_normalization_error=np.asarray(
            features["whole_normalization_error"], dtype=np.float32
        ),
    )


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        raise ValueError("present-vs-absent AUC requires both classes")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return float(wins / (len(positive) * len(negative)))


def _capacity_score(query: np.ndarray, mixture: Mapping[str, Any]) -> dict[str, float]:
    whole = float(np.dot(query, np.asarray(mixture["whole_embedding"], dtype=np.float32)))
    window_scores = np.asarray(mixture["window_embeddings"], dtype=np.float32) @ query
    top_k = max(1, int(math.ceil(window_scores.size * 0.25)))
    top25 = float(np.sort(window_scores)[-top_k:].mean())
    return {
        "whole_cosine": whole,
        "window_max": float(window_scores.max()),
        "top25_mean": top25,
    }


def _capacity_metrics(records: Sequence[Mapping[str, Any]], *, query_key: str) -> dict[str, Any]:
    positive: dict[str, list[float]] = defaultdict(list)
    negative: dict[str, list[float]] = defaultdict(list)
    for record in records:
        query = np.asarray(record[query_key], dtype=np.float32)
        scores = _capacity_score(query, record["mixture"])
        target = positive if int(record["query_role_id"]) in PRESENT_ROLE_IDS else negative
        for key, score in scores.items():
            target[key].append(float(score))
    return {
        "query_view": query_key,
        "positive_count": len(positive["whole_cosine"]),
        "absent_count": len(negative["whole_cosine"]),
        "whole_cosine_auc": _roc_auc(positive["whole_cosine"], negative["whole_cosine"]),
        "window_max_auc": _roc_auc(positive["window_max"], negative["window_max"]),
        "top25_mean_auc": _roc_auc(positive["top25_mean"], negative["top25_mean"]),
        "interpretation": (
            "untrained cosine capacity only; not a threshold, model-selection, "
            "home-command hard-negative, CER, or submission result"
        ),
    }


def _query_view_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit whether two independent enrollment views retain speaker identity."""

    scores = np.asarray(
        [
            float(
                np.dot(
                    np.asarray(record["enrollment"], dtype=np.float32),
                    np.asarray(record["view2"], dtype=np.float32),
                )
            )
            for record in records
        ],
        dtype=np.float64,
    )
    if scores.size == 0:
        raise ValueError("query-view agreement requires at least one record")
    return {
        "count": int(scores.size),
        "cosine_mean": float(scores.mean()),
        "cosine_median": float(np.median(scores)),
        "cosine_p10": float(np.percentile(scores, 10)),
        "cosine_min": float(scores.min()),
        "interpretation": (
            "same-speaker enrollment-view stability only; not a target-presence "
            "threshold or mixture capacity result"
        ),
    }


def _new_feature_dirs(output_dir: Path) -> None:
    (output_dir / "mixture").mkdir(parents=True, exist_ok=True)
    (output_dir / "query").mkdir(parents=True, exist_ok=True)


def _verify_mixture_group(
    group: Sequence[Mapping[str, Any]], manifest_path: Path
) -> tuple[str, Path, bytes]:
    actual_hashes: list[str] = []
    paths: list[Path] = []
    for row in group:
        path = _resolve_path(
            _first_path(row, _MIXTURE_KEYS, field="recognition_audio/mixture_audio"),
            manifest_path,
            field="mixture_audio",
        )
        raw = path.read_bytes()
        actual = _sha256_bytes(raw)
        declared = str(row.get("mixture_sha256", "")).strip().casefold()
        if actual != declared:
            raise ManifestContractError(
                f"mixture SHA mismatch for {path}: declared={declared}, actual={actual}"
            )
        actual_hashes.append(actual)
        paths.append(path)
    if len(set(actual_hashes)) != 1:
        raise ManifestContractError(
            f"three DACF rows do not share byte-identical mixture: {actual_hashes}"
        )
    # The first path is only a decoding source.  Every row's bytes were read
    # and checked above; no label can select a different mixture feature.
    return actual_hashes[0], paths[0], paths[0].read_bytes()


def _verify_audio_sha(row: Mapping[str, Any], field: str, actual: str) -> None:
    declared = row.get(field)
    if declared is not None and str(declared).strip():
        if str(declared).strip().casefold() != actual:
            raise ManifestContractError(
                f"{field} mismatch for row {row.get('id')}: "
                f"declared={declared}, actual={actual}"
            )


def _make_sherpa_extractor(model_path: Path, num_threads: int) -> Any:
    try:
        import sherpa_onnx
    except Exception as exc:
        raise RuntimeError(
            "sherpa_onnx is required for the real CAM++ run; use code/.venv_realt"
        ) from exc
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path), num_threads=int(num_threads), debug=False
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    dim = int(getattr(extractor, "dim", FEATURE_DIM))
    if dim != FEATURE_DIM:
        raise RuntimeError(f"CAM++ model reports dim={dim}, expected {FEATURE_DIM}")
    return extractor


def build_feature_dataset(
    train_manifest: str | Path,
    val_manifest: str | Path,
    *,
    model_path: str | Path,
    output_dir: str | Path,
    extractor: Any | None = None,
    extractor_factory: Optional[Callable[[Path], Any]] = None,
    num_threads: int = DEFAULT_THREADS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float = DEFAULT_HOP_SEC,
) -> dict[str, Any]:
    """Build train/val feature artifacts and return the JSON-serializable report."""

    started = time.perf_counter()
    model = Path(model_path).resolve(strict=True)
    _assert_not_dataset_a(model, field="CAM++ model")
    model_sha = sha256_file(model)
    output = Path(output_dir).resolve()
    _assert_not_dataset_a(output, field="feature output")
    output.mkdir(parents=True, exist_ok=True)
    _new_feature_dirs(output)

    bundle = validate_manifests(train_manifest, val_manifest)
    if extractor is None:
        if extractor_factory is not None:
            extractor = extractor_factory(model)
        else:
            extractor = _make_sherpa_extractor(model, num_threads)
    extractor_dim = int(getattr(extractor, "dim", FEATURE_DIM))
    if extractor_dim != FEATURE_DIM:
        raise RuntimeError(f"extractor dim={extractor_dim}, expected {FEATURE_DIM}")

    stats: MutableMapping[str, int] = {
        "extractor_calls": 0,
        "mixture_feature_count": 0,
        "mixture_window_embedding_count": 0,
        "enrollment_embedding_count": 0,
        "enrollment_view2_embedding_count": 0,
    }
    mixture_cache: dict[str, dict[str, Any]] = {}
    mixture_paths: dict[str, str] = {}
    mixture_owner: dict[str, str] = {}
    metric_records: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    enhanced_rows: list[dict[str, Any]] = []
    split_runtime: dict[str, float] = {}

    for split in ("train", "val"):
        split_started = time.perf_counter()
        split_info = bundle[split]
        manifest_path: Path = split_info["path"]
        for group_id in sorted(split_info["groups"]):
            group = split_info["groups"][group_id]
            mixture_sha, mixture_path, mixture_bytes = _verify_mixture_group(
                group, manifest_path
            )
            if mixture_sha not in mixture_cache:
                waveform, sample_rate = _read_audio_bytes(
                    mixture_bytes, source=mixture_path
                )
                features = extract_mixture_features(
                    extractor,
                    waveform,
                    sample_rate=sample_rate,
                    window_sec=window_sec,
                    hop_sec=hop_sec,
                    expected_dim=FEATURE_DIM,
                    stats=stats,
                )
                safe_key = _safe_name(f"{split}_{group_id}_{mixture_sha[:12]}")
                mixture_npz = output / "mixture" / f"{safe_key}.npz"
                _save_mixture_npz(mixture_npz, features, mixture_sha)
                mixture_cache[mixture_sha] = {
                    **features,
                    "path": mixture_npz,
                    "owner_split": split,
                    "owner_group_id": group_id,
                    "mixture_audio_sha256": mixture_sha,
                    "mixture_audio_path": mixture_path,
                }
                mixture_paths[mixture_sha] = _relative(mixture_npz, output)
                mixture_owner[mixture_sha] = f"{split}:{group_id}"
                stats["mixture_feature_count"] += 1
                stats["mixture_window_embedding_count"] += int(
                    features["window_embeddings"].shape[0]
                )
            mixture = mixture_cache[mixture_sha]

            for row in sorted(group, key=_role_id):
                row_id = _safe_name(row.get("id", f"{split}_{group_id}_{_role_id(row)}"))
                enrollment_path = _resolve_path(
                    _first_path(row, _ENROLLMENT_KEYS, field="enrollment_audio"),
                    manifest_path,
                    field="enrollment_audio",
                )
                view2_path = _resolve_path(
                    _first_path(row, _VIEW2_KEYS, field="enrollment_audio_view2"),
                    manifest_path,
                    field="enrollment_audio_view2",
                )
                enrollment_bytes = enrollment_path.read_bytes()
                view2_bytes = view2_path.read_bytes()
                enrollment_sha = _sha256_bytes(enrollment_bytes)
                view2_sha = _sha256_bytes(view2_bytes)
                _verify_audio_sha(row, "enrollment_sha256", enrollment_sha)
                _verify_audio_sha(row, "enrollment_view2_sha256", view2_sha)
                enrollment_wave, enrollment_sr = _read_audio_bytes(
                    enrollment_bytes, source=enrollment_path
                )
                view2_wave, view2_sr = _read_audio_bytes(view2_bytes, source=view2_path)
                enrollment_embedding, enrollment_raw_norm, enrollment_norm_error = (
                    _extract_with_audit(
                        extractor,
                        enrollment_wave,
                        enrollment_sr,
                        expected_dim=FEATURE_DIM,
                        stats=stats,
                    )
                )
                view2_embedding, view2_raw_norm, view2_norm_error = _extract_with_audit(
                    extractor,
                    view2_wave,
                    view2_sr,
                    expected_dim=FEATURE_DIM,
                    stats=stats,
                )
                stats["enrollment_embedding_count"] += 1
                stats["enrollment_view2_embedding_count"] += 1

                enrollment_npy = output / "query" / f"{split}_{row_id}__enrollment.npy"
                view2_npy = output / "query" / f"{split}_{row_id}__view2.npy"
                np.save(enrollment_npy, enrollment_embedding)
                np.save(view2_npy, view2_embedding)

                role_id = _role_id(row)
                scores = _capacity_score(enrollment_embedding, mixture)
                metric_records[split].append(
                    {
                        "query_role_id": role_id,
                        "enrollment": enrollment_embedding,
                        "view2": view2_embedding,
                        "mixture": mixture,
                    }
                )
                enhanced = dict(row)
                enhanced["campp_features"] = {
                    "schema": "dacf-campp-features-v0.1",
                    "model_path": str(model),
                    "model_sha256": model_sha,
                    "feature_dim": FEATURE_DIM,
                    "mixture_audio_sha256_actual": mixture_sha,
                    "mixture_feature_npz": mixture_paths[mixture_sha],
                    "mixture_feature_owner": mixture_owner[mixture_sha],
                    "mixture_feature_reused_across_counterfactual_rows": True,
                    "mixture_window_count": int(mixture["window_embeddings"].shape[0]),
                    "mixture_window_start_samples": [
                        int(value) for value in mixture["window_start_samples"]
                    ],
                    "mixture_window_end_samples": [
                        int(value) for value in mixture["window_end_samples"]
                    ],
                    "mixture_window_padded_samples": [
                        int(value) for value in mixture["window_padded_samples"]
                    ],
                    "mixture_whole_normalization_error": float(
                        mixture["whole_normalization_error"]
                    ),
                    "mixture_window_max_normalization_error": float(
                        np.max(mixture["window_normalization_errors"])
                    ),
                    "enrollment_embedding_npy": _relative(enrollment_npy, output),
                    "enrollment_view2_embedding_npy": _relative(view2_npy, output),
                    "enrollment_audio_sha256_actual": enrollment_sha,
                    "enrollment_view2_audio_sha256_actual": view2_sha,
                    "enrollment_raw_norm": float(enrollment_raw_norm),
                    "enrollment_normalization_error": float(enrollment_norm_error),
                    "enrollment_view2_raw_norm": float(view2_raw_norm),
                    "enrollment_view2_normalization_error": float(view2_norm_error),
                    "query_role_id_used_as_feature_input": False,
                    "query_role_id_is_audit_label_only": True,
                    "cosine_capacity_scores_enrollment": scores,
                }
                enhanced_rows.append(enhanced)
        split_runtime[split] = time.perf_counter() - split_started

    enhanced_manifest = output / "features_manifest.jsonl"
    with enhanced_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in enhanced_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metrics: dict[str, Any] = {}
    for split in ("train", "val"):
        metrics[split] = {
            "enrollment_audio": _capacity_metrics(
                metric_records[split], query_key="enrollment"
            ),
            "enrollment_audio_view2": _capacity_metrics(
                metric_records[split], query_key="view2"
            ),
            "enrollment_view_agreement": _query_view_agreement(
                metric_records[split]
            ),
        }

    report: dict[str, Any] = {
        "schema": "dacf-campp-feature-report-v0.1",
        "verdict": "conditional-GO",
        "verdict_scope": "feature-layer audit only; not CER/integration GO",
        "dataset_a_policy": "hard reject path/flag; not read",
        "dataset_a_used": False,
        "model": {
            "path": str(model),
            "sha256": model_sha,
            "feature_dim": FEATURE_DIM,
        },
        "inputs": {
            "train_manifest": str(bundle["train"]["path"]),
            "val_manifest": str(bundle["val"]["path"]),
        },
        "audit": bundle["audit"],
        "counts": {
            **{key: int(value) for key, value in stats.items()},
            "input_rows": len(enhanced_rows),
            "groups": len(bundle["train"]["groups"]) + len(bundle["val"]["groups"]),
            "unique_mixtures": len(mixture_cache),
            "feature_manifest_rows": len(enhanced_rows),
        },
        "windowing": {
            "sample_rate": SAMPLE_RATE,
            "window_sec": float(window_sec),
            "hop_sec": float(hop_sec),
            "tail_coverage": True,
            "short_audio_padding": "reflect_then_tile",
        },
        "runtime_sec": float(time.perf_counter() - started),
        "split_runtime_sec": {key: float(value) for key, value in split_runtime.items()},
        "capacity_metrics": metrics,
        "limitations": [
            "Cosine AUC is an untrained capacity probe, not threshold/model selection.",
            "No home-command hard-negative or Dataset-A data was used.",
            "No CER, rejection-rate, optimizer, or submission result is measured here.",
            "Mixture features are keyed by verified byte SHA256 and reused across A/B/C.",
        ],
        "artifacts": {
            "enhanced_manifest": _relative(enhanced_manifest, output),
            "report": "report.json",
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--model", default="E:/hf_cache/campplus/campplus.onnx")
    parser.add_argument(
        "--output",
        default="code/runs/dacf_counterfactual_probe32_campp_20260806",
    )
    parser.add_argument("--num-threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--hop-sec", type=float, default=DEFAULT_HOP_SEC)
    args = parser.parse_args(argv)
    report = build_feature_dataset(
        args.train_manifest,
        args.val_manifest,
        model_path=args.model,
        output_dir=args.output,
        num_threads=args.num_threads,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
