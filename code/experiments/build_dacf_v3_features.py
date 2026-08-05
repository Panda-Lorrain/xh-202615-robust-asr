"""Build the provenance-locked DACF-v3 feature cache.

The cache is deliberately a feature layer, not a trainer or an evaluator.  A
real run uses the local ``WhisperFeatureExtractor`` shipped with the Qwen
configuration and the existing sherpa-onnx CAM++ extractor.  Tests inject
small fakes for both extractors, so importing this module never loads a model
and never contacts the network.

The input contract is one A/B/C counterfactual group per ``base_mixture_id``:

    mixture(A+B) + enrollment(A) -> target A
    mixture(A+B) + enrollment(B) -> target B
    mixture(A+B) + enrollment(C) -> absent C

The mixture feature is keyed by its verified audio SHA256 and is extracted
once per unique mixture.  ``query_role_id`` remains an audit label only and
is intentionally absent from every query NPZ/model input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import soundfile as sf


SCHEMA = "dacf-v3-feature-cache-v0.1"
REPORT_SCHEMA = "dacf-v3-feature-cache-report-v0.1"
SOURCE_CORPUS = "AISHELL-1"
DEFAULT_ALLOWED_SOURCE_ROOT = Path("E:/midea_datasets/data_aishell")
SAMPLE_RATE = 16_000
FEATURE_SIZE = 128
CAMPP_DIM = 512
REQUIRED_FEATURE_SPEC = {
    "feature_size": 128,
    "n_fft": 400,
    "hop_length": 160,
    "dither": 0.0,
}
ROLES = ("present_A", "present_B", "absent_C")
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}
ManifestInput = str | Path | Sequence[str | Path]
DECLARED_SPLIT_ALIASES = {
    "train": {"train"},
    "dev": {"dev", "val"},
    "final": {"final"},
}
SOURCE_FIELDS = (
    "mixture_sources",
    "enrollment_src",
    "target_src",
    "interferer_srcs",
    "hard_negative_interferer_srcs",
    "noise_src",
    "rir_src",
)
MIXTURE_KEYS = ("recognition_audio", "mixture_audio", "mixture_path")
ENROLLMENT_KEYS = ("enrollment_audio", "enrollment_path")
VIEW2_KEYS = (
    "enrollment_audio_view2",
    "enrollment_view2_audio",
    "view2_audio",
    "enrollment_audio2",
)
CLEAN_TARGET_KEYS = ("clean_target_audio", "target_audio", "clean_target_path")
ARTIFACT_ROOT_FIELDS = (
    "recognition_audio",
    "mixture_audio",
    "mixture_path",
    "enrollment_audio",
    "enrollment_audio_view2",
    "enrollment_view2_audio",
    "view2_audio",
    "clean_target_audio",
    "target_audio",
    "target_activity",
)
DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)


class FeatureContractError(ValueError):
    """Raised when an input or generated cache violates the audit contract."""


# A shorter name is convenient for callers that share the v0.2 cache tests.
CacheContractError = FeatureContractError


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _looks_like_dataset_a(value: Any) -> bool:
    text = _path_text(value).casefold()
    if any(marker in text for marker in DATASET_A_MARKERS):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "dataseta" in compact


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_parent_segment(value: Any) -> bool:
    text = _path_text(value)
    return any(part == ".." for part in text.split("/"))


def _guard_value(value: Any, *, field: str) -> None:
    """Recursively reject Dataset-A markers, true flags, and path traversal."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}"
            key_text = str(key).casefold()
            if key_text in {"dataset_a_used", "dataset_a", "used_dataset_a"} and _truthy(child):
                raise FeatureContractError(f"{child_field}=true is forbidden: Dataset-A")
            _guard_value(child, field=child_field)
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _guard_value(child, field=f"{field}[{index}]")
        return
    if isinstance(value, (str, Path)):
        if _looks_like_dataset_a(value):
            raise FeatureContractError(
                f"{field} contains forbidden Dataset-A marker: {value}"
            )
        if _has_parent_segment(value):
            raise FeatureContractError(
                f"{field} contains a forbidden '..' path segment: {value}"
            )


def _assert_no_symlink_components(path: Path, *, field: str) -> None:
    """Reject symlinks in every component, including the final file."""

    candidate = Path(os.path.normpath(str(path)))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    anchor = Path(candidate.anchor) if candidate.anchor else Path.cwd()
    current = anchor
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                raise FeatureContractError(f"{field} traverses a symlink: {candidate}")
        except OSError as exc:
            raise FeatureContractError(
                f"cannot inspect path component {current}: {exc}"
            ) from exc


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_under_any(path: Path, roots: Sequence[Path], *, field: str) -> None:
    if not any(_is_under(path, root) for root in roots):
        shown = ", ".join(str(root) for root in roots)
        raise FeatureContractError(f"{field} is outside allowed roots ({shown}): {path}")


def _resolve_dir(raw: str | Path, *, field: str) -> Path:
    _guard_value(raw, field=field)
    raw_path = Path(raw)
    _assert_no_symlink_components(raw_path, field=field)
    try:
        candidate = raw_path.resolve(strict=True)
    except OSError as exc:
        raise FeatureContractError(f"cannot resolve {field}: {raw}") from exc
    _assert_no_symlink_components(candidate, field=field)
    if not candidate.is_dir():
        raise FeatureContractError(f"{field} must be a directory: {candidate}")
    return candidate


def _resolve_file_any(raw: str | Path, *, field: str) -> Path:
    _guard_value(raw, field=field)
    raw_path = Path(raw)
    _assert_no_symlink_components(raw_path, field=field)
    try:
        candidate = raw_path.resolve(strict=True)
    except OSError as exc:
        raise FeatureContractError(f"cannot resolve {field}: {raw}") from exc
    _assert_no_symlink_components(candidate, field=field)
    if not candidate.is_file():
        raise FeatureContractError(f"{field} must be a file: {candidate}")
    return candidate


def _resolve_existing_file(
    raw: Any,
    *,
    field: str,
    relative_roots: Sequence[Path],
    allowed_roots: Sequence[Path],
) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise FeatureContractError(f"missing path field {field}")
    _guard_value(raw, field=field)
    raw_path = Path(str(raw))
    candidates = (
        [raw_path]
        if raw_path.is_absolute()
        else [root / raw_path for root in relative_roots]
    )
    existing: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        _assert_no_symlink_components(candidate, field=field)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise FeatureContractError(f"cannot resolve {field}={raw!r}") from exc
        if not resolved.is_file():
            raise FeatureContractError(f"{field} is not a file: {resolved}")
        _assert_under_any(resolved, allowed_roots, field=field)
        if resolved not in existing:
            existing.append(resolved)
    if not existing:
        tried = ", ".join(str(path.resolve(strict=False)) for path in candidates)
        raise FeatureContractError(f"cannot resolve {field}={raw!r}; tried {tried}")
    if len(existing) > 1:
        raise FeatureContractError(f"ambiguous {field}={raw!r}: {existing}")
    return existing[0]


def _first_path(row: Mapping[str, Any], keys: Sequence[str], *, field: str) -> Any:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (str, Path)) and str(value).strip():
            return value
    raise FeatureContractError(f"row {row.get('id', '<unknown>')} requires {field}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Public spelling used by small callers and tests."""

    return _sha256_file(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("._") or "item"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise FeatureContractError(f"path is outside cache root: {path}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FeatureContractError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise FeatureContractError(f"manifest row is not an object at {path}:{line_number}")
            _guard_value(value, field=f"{path}:{line_number}")
            rows.append(value)
    if not rows:
        raise FeatureContractError(f"manifest is empty: {path}")
    return rows


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        waveform, sample_rate = sf.read(str(path), dtype="float32")
    except Exception as exc:  # pragma: no cover - soundfile decoder detail
        raise FeatureContractError(f"cannot decode audio: {path}") from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise FeatureContractError(f"audio is empty or not mono: {path}")
    if not np.isfinite(waveform).all():
        raise FeatureContractError(f"audio contains non-finite samples: {path}")
    sample_rate = int(sample_rate)
    if sample_rate != SAMPLE_RATE:
        try:
            import librosa

            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )
            sample_rate = SAMPLE_RATE
        except Exception as exc:  # pragma: no cover - only non-16k inputs need it
            raise FeatureContractError(
                f"audio {path} is {sample_rate} Hz and cannot be resampled locally"
            ) from exc
    return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate


def _load_activity(path: Path) -> np.ndarray:
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except Exception as exc:
        raise FeatureContractError(f"cannot load target_activity: {path}") from exc
    if value.ndim != 1 or value.size < 1:
        raise FeatureContractError(f"target_activity must be non-empty 1-D: {path}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise FeatureContractError(f"target_activity must be finite and in [0,1]: {path}")
    return np.ascontiguousarray(value, dtype=np.float32)


def _align_tail(value: np.ndarray, target_length: int, *, fill: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    """Align only by explicit tail zero-padding or tail cropping."""

    array = np.asarray(value)
    if array.ndim < 1:
        raise FeatureContractError("tail alignment requires a rank >= 1 array")
    source_length = int(array.shape[-1])
    target_length = int(target_length)
    if target_length < 1:
        raise FeatureContractError(f"target alignment length must be positive, got {target_length}")
    if source_length == target_length:
        aligned = np.ascontiguousarray(array)
        policy = "exact"
        padded = cropped = 0
    elif source_length < target_length:
        pad_shape = array.shape[:-1] + (target_length - source_length,)
        padding = np.full(pad_shape, fill, dtype=array.dtype)
        aligned = np.concatenate((array, padding), axis=-1)
        policy = "tail_zero_pad"
        padded, cropped = target_length - source_length, 0
    else:
        aligned = np.ascontiguousarray(array[..., :target_length])
        policy = "tail_crop"
        padded, cropped = 0, source_length - target_length
    return np.ascontiguousarray(aligned), {
        "source_length": source_length,
        "target_length": target_length,
        "difference_source_minus_target": source_length - target_length,
        "tail_pad_frames": int(padded),
        "tail_crop_frames": int(cropped),
        "policy": policy,
    }


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _flatten_strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _flatten_strings(child)
    elif isinstance(value, (str, Path)) and str(value).strip():
        yield str(value)


def _collect_speakers(row: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in (
        "query_speaker_id",
        "target_spk",
        "enrollment_spk",
        "interferer_spks",
        "hard_negative_interferer_spks",
        "mixture_speakers",
    ):
        for value in _flatten_strings(row.get(key)):
            result.add(value.strip().casefold())
    return {value for value in result if value}


def _canonical_path(path: Path) -> str:
    return path.as_posix().casefold()


def _resolve_source_tree(
    value: Any,
    *,
    source_root: Path,
    field: str,
    source_sha_cache: MutableMapping[Path, str],
    path_set: set[str],
) -> tuple[Any, Any]:
    """Resolve and hash source-lineage paths without copying their audio."""

    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        hashes: dict[str, Any] = {}
        for key, child in value.items():
            child_path, child_hash = _resolve_source_tree(
                child,
                source_root=source_root,
                field=f"{field}.{key}",
                source_sha_cache=source_sha_cache,
                path_set=path_set,
            )
            resolved[str(key)] = child_path
            hashes[str(key)] = child_hash
        return resolved, hashes
    if isinstance(value, (list, tuple)):
        resolved_list: list[Any] = []
        hash_list: list[Any] = []
        for index, child in enumerate(value):
            child_path, child_hash = _resolve_source_tree(
                child,
                source_root=source_root,
                field=f"{field}[{index}]",
                source_sha_cache=source_sha_cache,
                path_set=path_set,
            )
            resolved_list.append(child_path)
            hash_list.append(child_hash)
        return resolved_list, hash_list
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None
    path = _resolve_existing_file(
        value,
        field=field,
        relative_roots=(source_root,),
        allowed_roots=(source_root,),
    )
    canonical = _canonical_path(path)
    path_set.add(canonical)
    if path not in source_sha_cache:
        source_sha_cache[path] = _sha256_file(path)
    return path.as_posix(), source_sha_cache[path]


def _flatten_path_set(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result: set[str] = set()
        for child in value.values():
            result.update(_flatten_path_set(child))
        return result
    if isinstance(value, (list, tuple, set)):
        result = set()
        for child in value:
            result.update(_flatten_path_set(child))
        return result
    if isinstance(value, (str, Path)) and str(value).strip():
        return {str(value).casefold()}
    return set()


def _load_preprocessor_config(config_dir: Path) -> tuple[Path, dict[str, Any]]:
    config_path = config_dir / "preprocessor_config.json"
    if not config_path.is_file():
        raise FeatureContractError(f"missing local Qwen preprocessor_config.json: {config_path}")
    _assert_no_symlink_components(config_path, field="Qwen preprocessor config")
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureContractError(f"cannot read Qwen preprocessor config: {config_path}") from exc
    if not isinstance(value, dict):
        raise FeatureContractError("Qwen preprocessor_config.json must contain an object")
    _guard_value(value, field="qwen_preprocessor_config")
    return config_path, value


def _coerce_spec_value(key: str, value: Any, *, field: str) -> int | float:
    if value is None:
        raise FeatureContractError(f"{field} lacks required {key}")
    try:
        if key == "dither":
            return float(value)
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FeatureContractError(f"{field}.{key} has invalid value {value!r}") from exc


def validate_feature_extractor_spec(
    extractor: Any,
    *,
    config_values: Optional[Mapping[str, Any]] = None,
) -> dict[str, int | float]:
    """Validate the exact Qwen/Whisper front-end contract."""

    observed: dict[str, int | float] = {}
    for key, expected in REQUIRED_FEATURE_SPEC.items():
        value = _coerce_spec_value(
            key, getattr(extractor, key, None), field="feature_extractor"
        )
        if value != expected:
            raise FeatureContractError(
                f"feature_extractor.{key}={value!r}, expected {expected!r}"
            )
        observed[key] = value
    if config_values is not None:
        for key, expected in REQUIRED_FEATURE_SPEC.items():
            value = _coerce_spec_value(key, config_values.get(key), field="preprocessor_config")
            if value != expected:
                raise FeatureContractError(
                    f"preprocessor_config.{key}={value!r}, expected {expected!r}"
                )
            if value != observed[key]:
                raise FeatureContractError(
                    f"Qwen extractor/config mismatch for {key}: {observed[key]!r} != {value!r}"
                )
    sample_rate = getattr(extractor, "sampling_rate", SAMPLE_RATE)
    if int(sample_rate) != SAMPLE_RATE:
        raise FeatureContractError(
            f"feature_extractor.sampling_rate={sample_rate!r}, expected {SAMPLE_RATE}"
        )
    observed["sampling_rate"] = SAMPLE_RATE
    return observed


def make_qwen_feature_extractor(config_dir: str | Path, *, local_files_only: bool = True) -> Any:
    """Load only a local WhisperFeatureExtractor with the exact Qwen spec."""

    if not local_files_only:
        raise FeatureContractError(
            "DACF-v3 real CLI forbids network model resolution; local_files_only must be true"
        )
    directory = _resolve_dir(config_dir, field="Qwen config dir")
    _, config_values = _load_preprocessor_config(directory)
    try:
        from transformers import WhisperFeatureExtractor
    except Exception as exc:  # pragma: no cover - real CLI dependency diagnostic
        raise FeatureContractError(
            "transformers with WhisperFeatureExtractor is required in code/.venv_realt"
        ) from exc
    try:
        extractor = WhisperFeatureExtractor.from_pretrained(
            str(directory), local_files_only=True
        )
    except Exception as exc:  # pragma: no cover - transformers diagnostic
        raise FeatureContractError(
            f"cannot load local WhisperFeatureExtractor from {directory}"
        ) from exc
    validate_feature_extractor_spec(extractor, config_values=config_values)
    return extractor


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _output_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def extract_qwen_features(
    extractor: Any,
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Qwen-exact ``input_features [128,T]`` and a frame mask."""

    if int(sample_rate) != SAMPLE_RATE:
        raise FeatureContractError(
            f"Qwen feature extraction requires {SAMPLE_RATE} Hz, got {sample_rate}"
        )
    signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise FeatureContractError("Qwen feature input must be finite and non-empty")
    try:
        # Explicit padding=False is part of the Qwen feature contract: the
        # cache stores the real [128,T] timeline, not a 30-second [128,3000]
        # right-padded batch tensor.
        result = extractor(
            signal,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding=False,
        )
    except TypeError as exc:
        # This fallback is only for a minimal injected fake.  The real
        # WhisperFeatureExtractor path accepts both keyword arguments above.
        message = str(exc)
        if "padding" in message:
            try:
                result = extractor(signal, sampling_rate=SAMPLE_RATE, return_tensors="np")
            except TypeError as nested:
                if "return_tensors" not in str(nested):
                    raise
                result = extractor(signal, sampling_rate=SAMPLE_RATE)
        elif "return_tensors" in message:
            result = extractor(signal, sampling_rate=SAMPLE_RATE, padding=False)
        else:
            raise
    raw_features = result if isinstance(result, np.ndarray) else _output_field(result, "input_features")
    if raw_features is None:
        raise FeatureContractError("Qwen feature extractor output lacks input_features")
    features = _as_numpy(raw_features).astype(np.float32, copy=False)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2 or features.shape[0] != FEATURE_SIZE or features.shape[1] < 1:
        raise FeatureContractError(
            f"Qwen input_features must be [128,T], got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise FeatureContractError("Qwen input_features contain NaN or Inf")
    raw_mask = _output_field(result, "feature_attention_mask")
    if raw_mask is None:
        raw_mask = _output_field(result, "attention_mask")
    if raw_mask is None:
        mask = np.ones(features.shape[1], dtype=np.int8)
    else:
        mask = _as_numpy(raw_mask)
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        mask = np.asarray(mask).reshape(-1)
        # Standard WhisperFeatureExtractor returns an audio-sample mask under
        # ``attention_mask``.  It is not a feature-frame mask, so do not
        # mistake its 640/480000 length for T; retain an all-valid frame mask.
        if mask.shape != (features.shape[1],):
            mask = np.ones(features.shape[1], dtype=np.int8)
        else:
            mask = np.ascontiguousarray(mask.astype(np.int8, copy=False))
    return np.ascontiguousarray(features, dtype=np.float32), mask


def _normalize_camp_embedding(value: Any) -> np.ndarray:
    embedding = _as_numpy(value).astype(np.float32, copy=False).reshape(-1)
    if embedding.shape != (CAMPP_DIM,):
        raise FeatureContractError(
            f"CAM++ enrollment embedding must be [512], got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise FeatureContractError("CAM++ enrollment embedding contains NaN or Inf")
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        raise FeatureContractError("CAM++ enrollment embedding is near zero")
    normalized = np.ascontiguousarray(embedding / norm, dtype=np.float32)
    if abs(float(np.linalg.norm(normalized)) - 1.0) > 2e-5:
        raise FeatureContractError("CAM++ embedding normalization failed")
    return normalized


def extract_camp_embedding(
    extractor: Any,
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Extract one normalized CAM++ enrollment vector through DI or sherpa."""

    if hasattr(extractor, "create_stream"):
        try:
            from build_dacf_campp_features import extract_embedding as _extract_embedding
        except Exception as exc:  # pragma: no cover - real CLI dependency diagnostic
            raise FeatureContractError("existing CAM++ feature helper is unavailable") from exc
        try:
            value = _extract_embedding(
                extractor,
                np.asarray(waveform, dtype=np.float32),
                int(sample_rate),
                expected_dim=CAMPP_DIM,
            )
        except Exception as exc:  # pragma: no cover - sherpa diagnostic
            raise FeatureContractError("sherpa CAM++ enrollment extraction failed") from exc
        return _normalize_camp_embedding(value)
    if hasattr(extractor, "extract"):
        value = extractor.extract(waveform, sample_rate)
    elif callable(extractor):
        try:
            value = extractor(waveform, sample_rate)
        except TypeError as exc:
            if "positional" not in str(exc) and "argument" not in str(exc):
                raise
            value = extractor(waveform)
    else:
        raise FeatureContractError("CAM++ dependency injection must be callable or sherpa-like")
    return _normalize_camp_embedding(value)


@dataclass
class _PreparedRow:
    split: str
    row_id: str
    group_id: str
    query_role: str
    query_role_id: int
    target_present: bool
    query_speaker_id: str
    speaker_ids: set[str]
    manifest_path: Path
    mixture_path: Path
    enrollment_path: Path
    view2_path: Path
    clean_target_path: Optional[Path]
    activity_path: Path
    mixture_sha256: str
    mixture_audio_sha256: str
    enrollment_audio_sha256: str
    view2_audio_sha256: str
    clean_target_audio_sha256: Optional[str]
    target_activity_source_sha256: str
    resolved_audio_paths: Mapping[str, str]
    audio_sha256: Mapping[str, str]
    resolved_source_paths: Mapping[str, Any]
    source_sha256: Mapping[str, Any]
    source_path_set: set[str]
    enrollment_source_path_set: set[str]
    original_row: Mapping[str, Any]


def _prepared_contract_payload(row: _PreparedRow) -> dict[str, Any]:
    """Canonical semantic binding from one source-manifest row to one cache row."""

    return {
        "split": row.split,
        "row_id": row.row_id,
        "group_id": row.group_id,
        "query_role": row.query_role,
        "query_role_id": row.query_role_id,
        "target_present": row.target_present,
        "query_speaker_id": row.query_speaker_id,
        "speaker_ids": sorted(row.speaker_ids),
        "manifest_path": row.manifest_path.as_posix(),
        "mixture_path": row.mixture_path.as_posix(),
        "enrollment_path": row.enrollment_path.as_posix(),
        "view2_path": row.view2_path.as_posix(),
        "clean_target_path": (
            row.clean_target_path.as_posix() if row.clean_target_path is not None else None
        ),
        "activity_path": row.activity_path.as_posix(),
        "mixture_sha256": row.mixture_sha256,
        "mixture_audio_sha256": row.mixture_audio_sha256,
        "enrollment_audio_sha256": row.enrollment_audio_sha256,
        "view2_audio_sha256": row.view2_audio_sha256,
        "clean_target_audio_sha256": row.clean_target_audio_sha256,
        "target_activity_source_sha256": row.target_activity_source_sha256,
        "resolved_audio_paths": row.resolved_audio_paths,
        "audio_sha256": row.audio_sha256,
        "resolved_source_paths": row.resolved_source_paths,
        "source_sha256": row.source_sha256,
    }


def _resolve_artifact_path(
    raw: Any,
    *,
    field: str,
    artifact_roots: Sequence[Path],
) -> Path:
    # The first root is the project root; manifest parents make small local
    # fixtures and generated manifests with relative paths unambiguous.
    return _resolve_existing_file(
        raw,
        field=field,
        relative_roots=artifact_roots,
        allowed_roots=artifact_roots,
    )


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FeatureContractError(f"row {row.get('id', '<unknown>')} requires {field}")
    return str(value).strip()


def _prepare_rows(
    manifest_path: Path,
    *,
    split: str,
    artifact_roots: Sequence[Path],
    source_root: Path,
    source_sha_cache: MutableMapping[Path, str],
) -> list[_PreparedRow]:
    rows = _read_jsonl(manifest_path)
    prepared: list[_PreparedRow] = []
    seen_ids: set[str] = set()
    groups: MutableMapping[str, list[_PreparedRow]] = {}
    for index, row in enumerate(rows, 1):
        label = f"{manifest_path}:{index}"
        declared_split = str(row.get("split", ""))
        if declared_split not in DECLARED_SPLIT_ALIASES.get(split, {split}):
            raise FeatureContractError(
                f"{label} declares split={row.get('split')!r}, expected one of "
                f"{sorted(DECLARED_SPLIT_ALIASES.get(split, {split}))!r}"
            )
        protocol_split = row.get("protocol_split")
        if protocol_split is not None and str(protocol_split) != split:
            raise FeatureContractError(
                f"{label} protocol_split={protocol_split!r} disagrees with logical split {split!r}"
            )
        if str(row.get("source_corpus", "")) != SOURCE_CORPUS:
            raise FeatureContractError(f"{label} requires source_corpus=='{SOURCE_CORPUS}'")
        if _truthy(row.get("dataset_a_used", False)):
            raise FeatureContractError(f"{label} has dataset_a_used=true")
        row_id = _required_text(row, "id")
        if row_id in seen_ids:
            raise FeatureContractError(f"duplicate row id: {row_id}")
        seen_ids.add(row_id)
        group_id = _required_text(row, "base_mixture_id")
        role = _required_text(row, "query_role")
        if role not in ROLE_TO_ID:
            raise FeatureContractError(f"{label} has invalid query_role={role!r}")
        raw_role_id = row.get("query_role_id")
        if isinstance(raw_role_id, bool) or raw_role_id is None:
            raise FeatureContractError(f"{label} requires integer query_role_id as audit label")
        try:
            role_id = int(raw_role_id)
        except (TypeError, ValueError) as exc:
            raise FeatureContractError(f"{label} has invalid query_role_id") from exc
        if role_id != ROLE_TO_ID[role]:
            raise FeatureContractError(f"{label} query_role/query_role_id mismatch")
        target_present = _truthy(row.get("target_present"))
        if target_present != (role != "absent_C"):
            raise FeatureContractError(f"{label} target_present disagrees with query_role")
        query_speaker_id = _required_text(row, "query_speaker_id")
        mixture_sha = _required_text(row, "mixture_sha256").casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", mixture_sha):
            raise FeatureContractError(f"{label} mixture_sha256 is not a SHA256")

        mixture_path = _resolve_artifact_path(
            _first_path(row, MIXTURE_KEYS, field="recognition_audio"),
            field=f"{label}.recognition_audio",
            artifact_roots=artifact_roots,
        )
        enrollment_path = _resolve_artifact_path(
            _first_path(row, ENROLLMENT_KEYS, field="enrollment_audio"),
            field=f"{label}.enrollment_audio",
            artifact_roots=artifact_roots,
        )
        view2_path = _resolve_artifact_path(
            _first_path(row, VIEW2_KEYS, field="enrollment_audio_view2"),
            field=f"{label}.enrollment_audio_view2",
            artifact_roots=artifact_roots,
        )
        activity_path = _resolve_artifact_path(
            _required_text(row, "target_activity"),
            field=f"{label}.target_activity",
            artifact_roots=artifact_roots,
        )
        clean_raw = next(
            (row.get(key) for key in CLEAN_TARGET_KEYS if isinstance(row.get(key), (str, Path)) and str(row.get(key)).strip()),
            None,
        )
        clean_target_path = (
            _resolve_artifact_path(
                clean_raw,
                field=f"{label}.clean_target_audio",
                artifact_roots=artifact_roots,
            )
            if clean_raw is not None
            else None
        )

        mixture_audio_sha = _sha256_file(mixture_path)
        if mixture_audio_sha.casefold() != mixture_sha:
            raise FeatureContractError(
                f"{label} mixture SHA mismatch: declared={mixture_sha}, actual={mixture_audio_sha}"
            )
        enrollment_audio_sha = _sha256_file(enrollment_path)
        view2_audio_sha = _sha256_file(view2_path)
        if enrollment_path == view2_path or enrollment_audio_sha == view2_audio_sha:
            raise FeatureContractError(
                f"{label} enrollment views must be distinct audio artifacts"
            )
        if str(row.get("enrollment_spk", "")) != query_speaker_id:
            raise FeatureContractError(
                f"{label} enrollment_spk must equal query_speaker_id"
            )
        if int(row.get("enrollment_view_count", 0)) != 2:
            raise FeatureContractError(f"{label} requires exactly two enrollment views")
        if row.get("identity_positive") is not True:
            raise FeatureContractError(f"{label} enrollment identity-positive contract is missing")
        main_noise_sha = str(row.get("enrollment_noise_raw_sha256", ""))
        view2_noise_sha = str(row.get("enrollment_view2_noise_raw_sha256", ""))
        if not main_noise_sha or not view2_noise_sha or main_noise_sha == view2_noise_sha:
            raise FeatureContractError(
                f"{label} enrollment views do not prove independent augmentation"
            )
        for field, actual in (
            ("enrollment_sha256", enrollment_audio_sha),
            ("enrollment_view2_sha256", view2_audio_sha),
        ):
            declared = row.get(field)
            if declared is not None and str(declared).strip().casefold() != actual.casefold():
                raise FeatureContractError(
                    f"{label} {field} mismatch: declared={declared}, actual={actual}"
                )
        clean_target_audio_sha = (
            _sha256_file(clean_target_path) if clean_target_path is not None else None
        )
        activity_source_sha = _sha256_file(activity_path)
        activity = _load_activity(activity_path)
        if role == "absent_C" and np.any(activity > 0.0):
            raise FeatureContractError(f"{label} absent_C target_activity must be all zero")
        if role != "absent_C" and not np.any(activity > 0.0):
            raise FeatureContractError(f"{label} present target_activity has no active frame")
        if role == "absent_C":
            if clean_target_path is None:
                raise FeatureContractError(f"{label} absent_C requires clean_target_audio")
            clean_wave, _ = _read_audio(clean_target_path)
            if np.any(np.abs(clean_wave) > 1e-7):
                raise FeatureContractError(
                    f"{label} absent_C clean_target_audio is not physically blank"
                )

        resolved_sources: dict[str, Any] = {}
        source_hashes: dict[str, Any] = {}
        source_path_set: set[str] = set()
        for field in SOURCE_FIELDS:
            if field not in row or row[field] is None:
                continue
            resolved, hashes = _resolve_source_tree(
                row[field],
                source_root=source_root,
                field=f"{label}.{field}",
                source_sha_cache=source_sha_cache,
                path_set=source_path_set,
            )
            resolved_sources[field] = resolved
            source_hashes[field] = hashes
        if not _flatten_path_set(resolved_sources.get("enrollment_src")):
            raise FeatureContractError(f"{label} requires a resolved enrollment_src")
        enrollment_source_path_set = _flatten_path_set(resolved_sources.get("enrollment_src"))
        if protocol_split is not None:
            expected_source_split = {"train": "train", "dev": "dev", "final": "test"}[split]
            marker = f"/wav/{expected_source_split}/"
            wrong = sorted(path for path in source_path_set if marker not in path)
            if wrong:
                raise FeatureContractError(
                    f"{label} source lineage is outside AISHELL official {expected_source_split}: "
                    f"{wrong[:3]}"
                )

        resolved_audio: dict[str, str] = {
            "recognition_audio": mixture_path.as_posix(),
            "enrollment_audio": enrollment_path.as_posix(),
            "enrollment_audio_view2": view2_path.as_posix(),
        }
        audio_hashes: dict[str, str] = {
            "recognition_audio": mixture_audio_sha,
            "enrollment_audio": enrollment_audio_sha,
            "enrollment_audio_view2": view2_audio_sha,
        }
        if clean_target_path is not None and clean_target_audio_sha is not None:
            resolved_audio["clean_target_audio"] = clean_target_path.as_posix()
            audio_hashes["clean_target_audio"] = clean_target_audio_sha

        prepared_row = _PreparedRow(
            split=split,
            row_id=row_id,
            group_id=group_id,
            query_role=role,
            query_role_id=role_id,
            target_present=target_present,
            query_speaker_id=query_speaker_id,
            speaker_ids=_collect_speakers(row),
            manifest_path=manifest_path,
            mixture_path=mixture_path,
            enrollment_path=enrollment_path,
            view2_path=view2_path,
            clean_target_path=clean_target_path,
            activity_path=activity_path,
            mixture_sha256=mixture_sha,
            mixture_audio_sha256=mixture_audio_sha,
            enrollment_audio_sha256=enrollment_audio_sha,
            view2_audio_sha256=view2_audio_sha,
            clean_target_audio_sha256=clean_target_audio_sha,
            target_activity_source_sha256=activity_source_sha,
            resolved_audio_paths=resolved_audio,
            audio_sha256=audio_hashes,
            resolved_source_paths=resolved_sources,
            source_sha256=source_hashes,
            source_path_set=source_path_set,
            enrollment_source_path_set=enrollment_source_path_set,
            original_row=dict(row),
        )
        prepared.append(prepared_row)
        groups.setdefault(group_id, []).append(prepared_row)

    for group_id, group in groups.items():
        if len(group) != 3:
            raise FeatureContractError(
                f"{split} group {group_id!r} must contain exactly A/B/C rows, got {len(group)}"
            )
        if sorted(item.query_role for item in group) != sorted(ROLES):
            raise FeatureContractError(f"{split} group {group_id!r} must contain A/B/C exactly once")
        mixture_hashes = {item.mixture_sha256 for item in group}
        mixture_paths = {_canonical_path(item.mixture_path) for item in group}
        if len(mixture_hashes) != 1 or len(mixture_paths) != 1:
            raise FeatureContractError(f"{split} group {group_id!r} is not one byte-identical mixture")
        mixture_bytes = {item.mixture_path.read_bytes() for item in group}
        if len(mixture_bytes) != 1:
            raise FeatureContractError(f"{split} group {group_id!r} mixture audio bytes differ")
    return prepared


def _split_sets(rows: Sequence[_PreparedRow], split: str) -> dict[str, set[str]]:
    subset = [row for row in rows if row.split == split]
    return {
        "speaker": set().union(*(row.speaker_ids for row in subset)) if subset else set(),
        "source_path": set().union(*(row.source_path_set for row in subset)) if subset else set(),
        "source_sha256": (
            set().union(*(_flatten_path_set(row.source_sha256) for row in subset))
            if subset
            else set()
        ),
        "mixture_sha256": {row.mixture_sha256.casefold() for row in subset},
        "mixture_path": {_canonical_path(row.mixture_path) for row in subset},
        "group_id": {row.group_id.casefold() for row in subset},
        "enrollment_source": set().union(*(row.enrollment_source_path_set for row in subset)) if subset else set(),
        "enrollment_sha256": (
            {value.casefold() for row in subset for value in (row.enrollment_audio_sha256, row.view2_audio_sha256)}
        ),
    }


def _overlap_audit(rows: Sequence[_PreparedRow]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for left, right in (("train", "dev"), ("train", "final"), ("dev", "final")):
        left_sets = _split_sets(rows, left)
        right_sets = _split_sets(rows, right)
        result[f"{left}_vs_{right}"] = {
            f"{field}_overlap": sorted(left_sets[field] & right_sets[field])
            for field in left_sets
        }
    return result


def _assert_no_overlap(audit: Mapping[str, Mapping[str, Sequence[str]]]) -> None:
    for pair, fields in audit.items():
        for field, values in fields.items():
            if values:
                raise FeatureContractError(f"{pair} {field} is not empty: {list(values)[:8]}")


def _payload_sha256(paths: Sequence[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: _relative(item, root)):
        digest.update(_relative(path, root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _scalar_text(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise FeatureContractError(f"{field} must be a scalar, got {array.shape}")
    text = str(array.item()).strip()
    if not text:
        raise FeatureContractError(f"{field} is empty")
    return text


def _scalar_bool(value: Any, *, field: str) -> bool:
    return _truthy(_scalar_text(value, field=field))


def _scalar_int(value: Any, *, field: str) -> int:
    text = _scalar_text(value, field=field)
    try:
        return int(text)
    except ValueError as exc:
        raise FeatureContractError(f"{field} is not an integer: {text!r}") from exc


def _prepare_output_root(output_dir: str | Path, *, project_root: Path) -> Path:
    _guard_value(output_dir, field="output")
    raw = Path(output_dir)
    if raw.exists():
        _assert_no_symlink_components(raw, field="output")
    output = raw.resolve(strict=False)
    _assert_under_any(output, (project_root,), field="output")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("mixture", "query"):
        child = output / name
        if child.exists():
            _assert_no_symlink_components(child, field=f"output/{name}")
        child.mkdir(parents=True, exist_ok=True)
    if (output / "features_manifest.jsonl").exists() or (output / "cache_report.json").exists():
        raise FeatureContractError(
            f"output already contains a DACF-v3 cache; use a fresh output directory: {output}"
        )
    return output


def _validate_manifest_bundle(
    manifests: Mapping[str, ManifestInput],
    *,
    project_root: Path,
    source_root: Path,
) -> tuple[
    list[_PreparedRow],
    dict[str, dict[str, list[str]]],
    list[Path],
    list[Path],
    MutableMapping[Path, str],
]:
    manifest_paths: list[Path] = []
    split_manifest_paths: list[tuple[str, Path]] = []
    for split, raw in manifests.items():
        raw_paths = [raw] if isinstance(raw, (str, Path)) else list(raw)
        if not raw_paths:
            raise FeatureContractError(f"{split} requires at least one manifest")
        for raw_path in raw_paths:
            unresolved_path = Path(raw_path)
            _assert_no_symlink_components(unresolved_path, field=f"{split} manifest")
            path = unresolved_path.resolve(strict=True)
            if not path.is_file():
                raise FeatureContractError(f"{split} manifest is not a file: {path}")
            _guard_value(path, field=f"{split} manifest")
            manifest_paths.append(path)
            split_manifest_paths.append((split, path))
    artifact_roots: list[Path] = [project_root]
    for path in manifest_paths:
        parent = path.parent.resolve()
        if parent not in artifact_roots:
            artifact_roots.append(parent)
    source_sha_cache: MutableMapping[Path, str] = {}
    prepared: list[_PreparedRow] = []
    for split, path in split_manifest_paths:
        prepared.extend(
            _prepare_rows(
                path,
                split=split,
                artifact_roots=artifact_roots,
                source_root=source_root,
                source_sha_cache=source_sha_cache,
            )
        )
    row_ids: set[tuple[str, str]] = set()
    group_mixtures: dict[tuple[str, str], str] = {}
    mixture_groups: dict[tuple[str, str], str] = {}
    mixture_prefixes: dict[tuple[str, str], str] = {}
    query_safe_names: dict[tuple[str, str], str] = {}
    for row in prepared:
        row_key = (row.split, row.row_id)
        if row_key in row_ids:
            raise FeatureContractError(
                f"duplicate row_id across {row.split} manifests: {row.row_id}"
            )
        row_ids.add(row_key)
        group_key = (row.split, row.group_id)
        previous_mixture = group_mixtures.get(group_key)
        if previous_mixture is not None and previous_mixture != row.mixture_sha256:
            raise FeatureContractError(
                f"duplicate base_mixture_id maps to multiple mixtures across "
                f"{row.split} manifests: {row.group_id}"
            )
        group_mixtures[group_key] = row.mixture_sha256
        mixture_key = (row.split, row.mixture_sha256)
        previous_group = mixture_groups.get(mixture_key)
        if previous_group is not None and previous_group != row.group_id:
            raise FeatureContractError(
                f"one mixture SHA is reused by multiple {row.split} groups: "
                f"{previous_group}, {row.group_id}"
            )
        mixture_groups[mixture_key] = row.group_id
        prefix_key = (row.split, row.mixture_sha256[:32])
        previous_sha = mixture_prefixes.get(prefix_key)
        if previous_sha is not None and previous_sha != row.mixture_sha256:
            raise FeatureContractError(
                f"mixture cache filename prefix collision in {row.split}: {row.mixture_sha256[:32]}"
            )
        mixture_prefixes[prefix_key] = row.mixture_sha256
        safe_key = (row.split, _safe_name(row.row_id))
        previous_row_id = query_safe_names.get(safe_key)
        if previous_row_id is not None and previous_row_id != row.row_id:
            raise FeatureContractError(
                f"query cache filename collision in {row.split}: "
                f"{previous_row_id!r}, {row.row_id!r}"
            )
        query_safe_names[safe_key] = row.row_id
    audit = _overlap_audit(prepared)
    _assert_no_overlap(audit)
    return prepared, audit, manifest_paths, artifact_roots, source_sha_cache


def _make_real_camp_extractor(model_path: Path, num_threads: int) -> Any:
    try:
        from build_dacf_campp_features import _make_sherpa_extractor
    except Exception as exc:  # pragma: no cover - real CLI dependency diagnostic
        raise FeatureContractError("existing sherpa CAM++ helper is unavailable") from exc
    try:
        return _make_sherpa_extractor(model_path, int(num_threads))
    except Exception as exc:  # pragma: no cover - sherpa diagnostic
        raise FeatureContractError(f"cannot load local CAM++ ONNX model: {model_path}") from exc


def _feature_spec_hash(spec: Mapping[str, Any]) -> str:
    return _sha256_json({key: spec[key] for key in REQUIRED_FEATURE_SPEC} | {"sampling_rate": SAMPLE_RATE})


def build_feature_cache(
    train_manifest: ManifestInput,
    dev_manifest: ManifestInput,
    final_manifest: ManifestInput | None = None,
    *,
    qwen_config_dir: str | Path,
    campp_model: str | Path,
    output_dir: str | Path,
    qwen_extractor: Any | None = None,
    campp_extractor: Any | None = None,
    allowed_source_root: str | Path = DEFAULT_ALLOWED_SOURCE_ROOT,
    local_files_only: bool = True,
    num_threads: int = 2,
    include_clean_target_logmel: bool = True,
    expected_groups: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Build a train/dev cache, optionally adding final only after model freeze."""

    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    source_root = _resolve_dir(allowed_source_root, field="allowed source root")
    if source_root.name.casefold() != "data_aishell":
        # The path may be a fixture root in unit tests; the corpus identity is
        # still fixed by the manifest and report.  Do not infer a corpus from
        # a directory name.
        pass
    qwen_dir = _resolve_dir(qwen_config_dir, field="Qwen config dir")
    qwen_preprocessor_path, qwen_config_values = _load_preprocessor_config(qwen_dir)
    qwen_config_sha = _sha256_file(qwen_preprocessor_path)
    campp_path = _resolve_file_any(campp_model, field="CAM++ ONNX model")
    campp_model_sha = _sha256_file(campp_path)
    output = _prepare_output_root(output_dir, project_root=project_root)

    manifests: dict[str, ManifestInput] = {
        "train": train_manifest,
        "dev": dev_manifest,
    }
    if final_manifest is not None:
        manifests["final"] = final_manifest
    active_splits = tuple(manifests)
    input_manifests: dict[str, list[dict[str, str]]] = {}
    for split, raw in manifests.items():
        raw_paths = [raw] if isinstance(raw, (str, Path)) else list(raw)
        input_manifests[split] = [
            {
                "path": Path(raw_path).resolve(strict=True).as_posix(),
                "sha256": _sha256_file(Path(raw_path).resolve(strict=True)),
            }
            for raw_path in raw_paths
        ]
    prepared, audit, manifest_paths, artifact_roots, source_sha_cache = _validate_manifest_bundle(
        manifests,
        project_root=project_root,
        source_root=source_root,
    )
    if expected_groups is not None:
        for split, expected in expected_groups.items():
            actual = len({row.group_id for row in prepared if row.split == split})
            if actual != int(expected):
                raise FeatureContractError(
                    f"{split} group contract requires {expected}, got {actual}"
                )

    if qwen_extractor is None:
        qwen_extractor = make_qwen_feature_extractor(
            qwen_dir, local_files_only=local_files_only
        )
    feature_spec = validate_feature_extractor_spec(
        qwen_extractor, config_values=qwen_config_values
    )
    feature_spec_sha = _feature_spec_hash(feature_spec)
    if campp_extractor is None:
        campp_extractor = _make_real_camp_extractor(campp_path, num_threads)

    stats: MutableMapping[str, int] = {
        "rows": len(prepared),
        "groups": len({(row.split, row.group_id) for row in prepared}),
        "unique_mixture_sha256": 0,
        "qwen_feature_extractor_calls": 0,
        "qwen_mixture_feature_calls": 0,
        "qwen_clean_target_feature_calls": 0,
        "campp_enrollment_extractor_calls": 0,
        "source_hash_calls": len(source_sha_cache),
    }
    mixture_cache: dict[str, dict[str, Any]] = {}
    mixture_group_ids: dict[str, set[str]] = {}
    for row in prepared:
        mixture_group_ids.setdefault(row.mixture_sha256, set()).add(row.group_id)

    ordered_rows = sorted(
        prepared,
        key=lambda row: (row.split, row.group_id, row.query_role_id, row.row_id),
    )
    for row in ordered_rows:
        if row.mixture_sha256 not in mixture_cache:
            mixture_wave, mixture_sr = _read_audio(row.mixture_path)
            mixture_features, mixture_mask = extract_qwen_features(
                qwen_extractor, mixture_wave, mixture_sr
            )
            stats["qwen_feature_extractor_calls"] += 1
            stats["qwen_mixture_feature_calls"] += 1
            mixture_path = output / "mixture" / (
                f"mixture__{_safe_name(row.mixture_sha256[:32])}.npz"
            )
            np.savez_compressed(
                mixture_path,
                input_features=mixture_features,
                feature_attention_mask=mixture_mask,
                mixture_sha256=np.asarray(row.mixture_sha256),
                mixture_audio_path=np.asarray(row.mixture_path.as_posix()),
                mixture_audio_sha256=np.asarray(row.mixture_audio_sha256),
                source_corpus=np.asarray(SOURCE_CORPUS),
                base_mixture_ids_json=np.asarray(
                    json.dumps(sorted(mixture_group_ids[row.mixture_sha256]), ensure_ascii=False)
                ),
                qwen_config_sha256=np.asarray(qwen_config_sha),
                qwen_feature_spec_sha256=np.asarray(feature_spec_sha),
                feature_array_sha256=np.asarray(_sha256_array(mixture_features)),
                feature_size=np.asarray(FEATURE_SIZE, dtype=np.int32),
                n_fft=np.asarray(400, dtype=np.int32),
                hop_length=np.asarray(160, dtype=np.int32),
                dither=np.asarray(0.0, dtype=np.float32),
                sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int32),
            )
            mixture_cache[row.mixture_sha256] = {
                "features": mixture_features,
                "mask": mixture_mask,
                "path": mixture_path,
                "file_sha256": _sha256_file(mixture_path),
                "array_sha256": _sha256_array(mixture_features),
                "frame_count": int(mixture_features.shape[1]),
                "audio_path": row.mixture_path,
                "audio_sha256": row.mixture_audio_sha256,
            }
            stats["unique_mixture_sha256"] += 1

    query_rows: list[dict[str, Any]] = []
    alignment_records: list[dict[str, int]] = []
    for row in ordered_rows:
        mixture = mixture_cache[row.mixture_sha256]
        mixture_features = np.asarray(mixture["features"], dtype=np.float32)
        frame_count = int(mixture["frame_count"])
        enrollment_wave, enrollment_sr = _read_audio(row.enrollment_path)
        view2_wave, view2_sr = _read_audio(row.view2_path)
        enrollment_embedding = extract_camp_embedding(
            campp_extractor, enrollment_wave, enrollment_sr
        )
        view2_embedding = extract_camp_embedding(campp_extractor, view2_wave, view2_sr)
        stats["campp_enrollment_extractor_calls"] += 2

        raw_activity = _load_activity(row.activity_path)
        aligned_activity, activity_alignment = _align_tail(
            raw_activity, frame_count, fill=0.0
        )
        if not row.target_present and np.any(aligned_activity > 0.0):
            raise FeatureContractError(
                f"{row.split}:{row.row_id} absent_C aligned target_activity is not zero"
            )

        query_values: dict[str, Any] = {
            "enrollment_embedding": enrollment_embedding,
            "enrollment_embedding_view2": view2_embedding,
            "target_activity": aligned_activity.astype(np.float32),
            "row_id": np.asarray(row.row_id),
            "split": np.asarray(row.split),
            "base_mixture_id": np.asarray(row.group_id),
            "query_role": np.asarray(row.query_role),
            "query_speaker_id": np.asarray(row.query_speaker_id),
            "target_present": np.asarray(row.target_present),
            "source_corpus": np.asarray(SOURCE_CORPUS),
            "mixture_sha256": np.asarray(row.mixture_sha256),
            "mixture_feature_sha256": np.asarray(mixture["file_sha256"]),
            "mixture_input_features_sha256": np.asarray(mixture["array_sha256"]),
            "mixture_audio_path": np.asarray(row.mixture_path.as_posix()),
            "mixture_audio_sha256": np.asarray(row.mixture_audio_sha256),
            "enrollment_audio_path": np.asarray(row.enrollment_path.as_posix()),
            "enrollment_audio_sha256": np.asarray(row.enrollment_audio_sha256),
            "enrollment_audio_view2_path": np.asarray(row.view2_path.as_posix()),
            "enrollment_audio_view2_sha256": np.asarray(row.view2_audio_sha256),
            "target_activity_path": np.asarray(row.activity_path.as_posix()),
            "target_activity_sha256": np.asarray(row.target_activity_source_sha256),
            "target_activity_array_sha256": np.asarray(_sha256_array(aligned_activity)),
            "resolved_source_paths_json": np.asarray(
                json.dumps(row.resolved_source_paths, ensure_ascii=False, sort_keys=True)
            ),
            "source_sha256_json": np.asarray(
                json.dumps(row.source_sha256, ensure_ascii=False, sort_keys=True)
            ),
            "resolved_audio_paths_json": np.asarray(
                json.dumps(row.resolved_audio_paths, ensure_ascii=False, sort_keys=True)
            ),
            "audio_sha256_json": np.asarray(
                json.dumps(row.audio_sha256, ensure_ascii=False, sort_keys=True)
            ),
            "activity_alignment_json": np.asarray(
                json.dumps(activity_alignment, ensure_ascii=False, sort_keys=True)
            ),
            "qwen_config_sha256": np.asarray(qwen_config_sha),
            "qwen_feature_spec_sha256": np.asarray(feature_spec_sha),
            "campp_model_sha256": np.asarray(campp_model_sha),
            "enrollment_embedding_sha256": np.asarray(_sha256_array(enrollment_embedding)),
            "enrollment_embedding_view2_sha256": np.asarray(_sha256_array(view2_embedding)),
        }
        clean_alignment: Optional[dict[str, Any]] = None
        if include_clean_target_logmel and row.clean_target_path is not None:
            clean_wave, clean_sr = _read_audio(row.clean_target_path)
            clean_features, clean_mask = extract_qwen_features(
                qwen_extractor, clean_wave, clean_sr
            )
            stats["qwen_feature_extractor_calls"] += 1
            stats["qwen_clean_target_feature_calls"] += 1
            aligned_clean, clean_alignment = _align_tail(
                clean_features, frame_count, fill=0.0
            )
            aligned_clean_mask, _ = _align_tail(clean_mask, frame_count, fill=0)
            query_values.update(
                {
                    "clean_target_input_features": aligned_clean.astype(np.float32),
                    "clean_target_feature_attention_mask": aligned_clean_mask.astype(np.int8),
                    "clean_target_audio_path": np.asarray(row.clean_target_path.as_posix()),
                    "clean_target_audio_sha256": np.asarray(row.clean_target_audio_sha256),
                    "clean_target_input_features_sha256": np.asarray(_sha256_array(aligned_clean)),
                    "clean_target_raw_frame_count": np.asarray(clean_features.shape[1], dtype=np.int32),
                    "clean_target_alignment_json": np.asarray(
                        json.dumps(clean_alignment, ensure_ascii=False, sort_keys=True)
                    ),
                }
            )
        query_path = output / "query" / (
            f"{_safe_name(row.split)}__{_safe_name(row.row_id)}.npz"
        )
        np.savez_compressed(query_path, **query_values)
        query_sha = _sha256_file(query_path)
        manifest_row = dict(row.original_row)
        manifest_row.update(
            {
                "cache_schema": SCHEMA,
                "source_manifest_path": row.manifest_path.as_posix(),
                "source_manifest_row_sha256": _sha256_json(row.original_row),
                "source_prepared_contract_sha256": _sha256_json(
                    _prepared_contract_payload(row)
                ),
                "split": row.split,
                "row_id": row.row_id,
                "base_mixture_id": row.group_id,
                "query_role": row.query_role,
                "query_role_id": row.query_role_id,
                "query_role_id_is_audit_only": True,
                "query_role_id_used_as_model_input": False,
                "target_present": row.target_present,
                "query_speaker_id": row.query_speaker_id,
                "speaker_ids": sorted(row.speaker_ids),
                "source_corpus": SOURCE_CORPUS,
                "dataset_a_used": False,
                "resolved_source_paths": row.resolved_source_paths,
                "source_sha256": row.source_sha256,
                "resolved_audio_paths": row.resolved_audio_paths,
                "audio_sha256": row.audio_sha256,
                "resolved_feature_paths": {"target_activity": row.activity_path.as_posix()},
                "mixture_sha256": row.mixture_sha256,
                "mixture_audio_sha256": row.mixture_audio_sha256,
                "enrollment_audio_sha256": row.enrollment_audio_sha256,
                "enrollment_audio_view2_sha256": row.view2_audio_sha256,
                "target_activity_sha256": row.target_activity_source_sha256,
                "target_activity_array_sha256": _sha256_array(aligned_activity),
                "mixture_feature": _relative(mixture["path"], output),
                "mixture_feature_sha256": mixture["file_sha256"],
                "mixture_input_features_sha256": mixture["array_sha256"],
                "query_feature": _relative(query_path, output),
                "query_npz_sha256": query_sha,
                "activity_alignment": activity_alignment,
                "clean_target_alignment": clean_alignment,
                "qwen_feature_spec_sha256": feature_spec_sha,
                "qwen_config_sha256": qwen_config_sha,
                "campp_model_sha256": campp_model_sha,
                "provenance": {
                    "audio_not_copied": True,
                    "source_paths_are_lineage_only": True,
                    "model_input_fields_exclude_query_role_id": True,
                },
            }
        )
        query_rows.append(manifest_row)
        alignment_records.append(
            {
                "pad": int(activity_alignment["tail_pad_frames"]),
                "crop": int(activity_alignment["tail_crop_frames"]),
            }
        )

    manifest_path = output / "features_manifest.jsonl"
    _write_jsonl(
        manifest_path,
        sorted(query_rows, key=lambda row: (str(row["split"]), str(row["base_mixture_id"]), int(row["query_role_id"]))),
    )
    manifest_sha = _sha256_file(manifest_path)
    payload_paths = [manifest_path]
    payload_paths.extend(sorted((output / "mixture").glob("*.npz")))
    payload_paths.extend(sorted((output / "query").glob("*.npz")))
    cache_sha = _payload_sha256(payload_paths, root=output)
    split_counts = {
        split: {
            "rows": sum(row.split == split for row in prepared),
            "groups": len({row.group_id for row in prepared if row.split == split}),
            "query_speakers": len({row.query_speaker_id for row in prepared if row.split == split}),
        }
        for split in active_splits
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "cache_schema": SCHEMA,
        "builder_code": Path(__file__).resolve().as_posix(),
        "builder_code_sha256": _sha256_file(Path(__file__).resolve()),
        "dataset_a_used": False,
        "dataset_a_policy": "hard reject forbidden-corpus markers and true flags; forbidden corpus is not read",
        "source_corpus": SOURCE_CORPUS,
        "allowed_source_root": source_root.as_posix(),
        "artifact_roots": [root.as_posix() for root in sorted(set(artifact_roots), key=str)],
        "qwen_config_dir": qwen_dir.as_posix(),
        "qwen_preprocessor_config": qwen_preprocessor_path.as_posix(),
        "qwen_config_sha256": qwen_config_sha,
        "local_files_only": True,
        "campp_model": campp_path.as_posix(),
        "campp_model_sha256": campp_model_sha,
        "feature_extractor_spec": feature_spec,
        "feature_extractor_spec_sha256": feature_spec_sha,
        "split_contract": {
            "splits": list(active_splits),
            "final_gate_split": "final" if "final" in manifests else None,
            "final_deferred": "final" not in manifests,
            "dev_role": "observation_only",
            "query_role_id": "audit_only; excluded from query NPZ/model input",
            "expected_groups": dict(expected_groups) if expected_groups is not None else None,
        },
        "input_manifests": input_manifests,
        "protocol_binding": {
            "protocol_split_metadata_bound": all(
                row.original_row.get("protocol_split") == row.split for row in prepared
            ),
            "official_source_route_checked_when_bound": True,
            "source_rows_semantically_bound": True,
            "logical_to_source_split": {"train": "train", "dev": "dev", "final": "test"},
        },
        "counts": {
            **{key: int(value) for key, value in stats.items()},
            "split_counts": split_counts,
            "mixture_npz": len(mixture_cache),
            "query_npz": len(query_rows),
            "feature_manifest_rows": len(query_rows),
        },
        "deduplication": {
            "unique_mixture_sha256": len(mixture_cache),
            "qwen_mixture_feature_calls": int(stats["qwen_mixture_feature_calls"]),
            "one_call_per_unique_mixture": int(stats["qwen_mixture_feature_calls"]) == len(mixture_cache),
            "query_rows_reusing_mixture_features": len(query_rows),
            "audio_copied_to_cache": False,
        },
        "activity_alignment": {
            "policy": "tail-only zero-pad or crop to mixture input_features T",
            "rows_with_tail_pad": sum(item["pad"] > 0 for item in alignment_records),
            "rows_with_tail_crop": sum(item["crop"] > 0 for item in alignment_records),
            "tail_pad_frames_total": sum(item["pad"] for item in alignment_records),
            "tail_crop_frames_total": sum(item["crop"] for item in alignment_records),
        },
        "overlap_audit": audit,
        "manifest_sha256": manifest_sha,
        "cache_sha256": cache_sha,
        "cache_sha256_scope": "features_manifest.jsonl plus every mixture/*.npz and query/*.npz; excludes cache_report.json",
        "artifacts": {
            "manifest": "features_manifest.jsonl",
            "cache_report": "cache_report.json",
            "mixture_dir": "mixture",
            "query_dir": "query",
        },
        "limitations": [
            "This is a feature cache only; no training, CER, RR, RTF, or GO claim is made.",
            "CAM++ is extracted only for enrollment and independent view2; mixture identity is bound by audio SHA and Qwen feature SHA.",
            "clean_target_input_features is a low-weight diagnostic target and is not a submission result.",
            "When final_deferred is true, no final manifest or final feature was read by this cache build.",
        ],
        "runtime_sec": float(time.perf_counter() - started),
    }
    report_path = output / "cache_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_cache(output)
    return report


def _validate_path_mapping(
    value: Any,
    *,
    field: str,
    allowed_roots: Sequence[Path],
    expected_hashes: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise FeatureContractError(f"{field} must be a mapping")
    result: dict[str, str] = {}
    for key, raw in value.items():
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise FeatureContractError(f"{field}.{key} must be a path")
        path = _resolve_existing_file(
            raw,
            field=f"{field}.{key}",
            relative_roots=allowed_roots,
            allowed_roots=allowed_roots,
        )
        canonical = path.as_posix()
        result[str(key)] = canonical
        if expected_hashes is not None:
            expected = expected_hashes.get(str(key))
            if expected is None or _sha256_file(path).casefold() != str(expected).casefold():
                raise FeatureContractError(f"{field}.{key} SHA256 mismatch")
    return result


def _validate_source_mapping(
    value: Any,
    hashes: Any,
    *,
    field: str,
    source_root: Path,
) -> set[str]:
    if isinstance(value, Mapping):
        if not isinstance(hashes, Mapping):
            raise FeatureContractError(f"{field} source hash shape mismatch")
        result: set[str] = set()
        for key, child in value.items():
            result.update(
                _validate_source_mapping(
                    child,
                    hashes.get(key),
                    field=f"{field}.{key}",
                    source_root=source_root,
                )
            )
        return result
    if isinstance(value, list):
        if not isinstance(hashes, list) or len(value) != len(hashes):
            raise FeatureContractError(f"{field} source hash shape mismatch")
        result = set()
        for index, child in enumerate(value):
            result.update(
                _validate_source_mapping(
                    child,
                    hashes[index],
                    field=f"{field}[{index}]",
                    source_root=source_root,
                )
            )
        return result
    if value is None:
        return set()
    path = _resolve_existing_file(
        value,
        field=field,
        relative_roots=(source_root,),
        allowed_roots=(source_root,),
    )
    if not isinstance(hashes, str) or _sha256_file(path).casefold() != hashes.casefold():
        raise FeatureContractError(f"{field} source SHA256 mismatch")
    return {_canonical_path(path)}


def _validate_overlap_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for left, right in (("train", "dev"), ("train", "final"), ("dev", "final")):
        left_rows = [row for row in rows if row.get("split") == left]
        right_rows = [row for row in rows if row.get("split") == right]

        def sets(items: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
            return {
                "speaker": set().union(*(set(str(x).casefold() for x in item.get("speaker_ids", [])) for item in items)) if items else set(),
                "source_path": set().union(*(_flatten_path_set(item.get("resolved_source_paths", {})) for item in items)) if items else set(),
                "source_sha256": set().union(*(_flatten_path_set(item.get("source_sha256", {})) for item in items)) if items else set(),
                "mixture_sha256": {str(item.get("mixture_sha256", "")).casefold() for item in items},
                "mixture_path": {_path_text(item.get("resolved_audio_paths", {}).get("recognition_audio", "")).casefold() for item in items},
                "group_id": {str(item.get("base_mixture_id", "")).casefold() for item in items},
                "enrollment_source": set().union(*(_flatten_path_set(item.get("resolved_source_paths", {}).get("enrollment_src")) for item in items)) if items else set(),
                "enrollment_sha256": {str(value).casefold() for item in items for value in (item.get("enrollment_audio_sha256", ""), item.get("enrollment_audio_view2_sha256", "")) if value},
            }

        a, b = sets(left_rows), sets(right_rows)
        result[f"{left}_vs_{right}"] = {
            f"{field}_overlap": sorted(a[field] & b[field]) for field in a
        }
    return result


def validate_cache(cache_root: str | Path) -> dict[str, Any]:
    """Revalidate cache files, metadata, input hashes, and split isolation."""

    project_root = Path(__file__).resolve().parents[2]
    root = _resolve_dir(cache_root, field="cache root")
    _assert_under_any(root, (project_root,), field="cache root")
    report_path = root / "cache_report.json"
    manifest_path = root / "features_manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        raise FeatureContractError("cache requires cache_report.json and features_manifest.jsonl")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _guard_value(report, field="cache_report")
    if report.get("schema") != REPORT_SCHEMA or report.get("cache_schema") != SCHEMA:
        raise FeatureContractError("cache report schema mismatch")
    builder_path = _resolve_file_any(
        report.get("builder_code"), field="cache builder code"
    )
    if builder_path != Path(__file__).resolve():
        raise FeatureContractError("cache was emitted by a different builder path")
    if _sha256_file(builder_path) != str(report.get("builder_code_sha256", "")):
        raise FeatureContractError("cache builder code SHA256 mismatch")
    if report.get("source_corpus") != SOURCE_CORPUS or report.get("dataset_a_used") is not False:
        raise FeatureContractError("cache report source/Dataset-A contract mismatch")
    source_root = _resolve_dir(report.get("allowed_source_root"), field="cache allowed source root")
    artifact_roots = [
        _resolve_dir(value, field="cache artifact root")
        for value in report.get("artifact_roots", [])
    ]
    if not artifact_roots:
        artifact_roots = [project_root]
    qwen_config_path = _resolve_file_any(
        report.get("qwen_preprocessor_config"), field="cache Qwen preprocessor config"
    )
    if _sha256_file(qwen_config_path) != str(report.get("qwen_config_sha256", "")):
        raise FeatureContractError("cache Qwen config SHA256 mismatch")
    campp_path = _resolve_file_any(report.get("campp_model"), field="cache CAM++ model")
    if _sha256_file(campp_path) != str(report.get("campp_model_sha256", "")):
        raise FeatureContractError("cache CAM++ model SHA256 mismatch")

    rows = _read_jsonl(manifest_path)
    split_contract = report.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise FeatureContractError("cache report lacks split_contract")
    declared_splits = split_contract.get("splits")
    if not isinstance(declared_splits, list) or not declared_splits:
        raise FeatureContractError("cache report split_contract.splits is invalid")
    if len(declared_splits) != len(set(declared_splits)) or any(
        split not in {"train", "dev", "final"} for split in declared_splits
    ):
        raise FeatureContractError("cache report declares duplicate or unknown splits")
    observed_splits = {str(row.get("split", "")) for row in rows}
    if observed_splits != set(declared_splits):
        raise FeatureContractError(
            "cache report split contract does not match manifest: "
            f"declared={declared_splits}, observed={sorted(observed_splits)}"
        )
    final_deferred = split_contract.get("final_deferred")
    expected_deferred = "final" not in declared_splits
    if final_deferred is not expected_deferred:
        raise FeatureContractError("cache report final_deferred flag is inconsistent")
    expected_final_gate = None if expected_deferred else "final"
    if split_contract.get("final_gate_split") != expected_final_gate:
        raise FeatureContractError("cache report final_gate_split is inconsistent")
    input_manifests = report.get("input_manifests")
    if not isinstance(input_manifests, Mapping) or set(input_manifests) != set(declared_splits):
        raise FeatureContractError("cache report input_manifests does not match split contract")
    manifest_inputs: dict[str, list[Path]] = {}
    for split, entries in input_manifests.items():
        if not isinstance(entries, list) or not entries:
            raise FeatureContractError(f"cache report has no input manifest for {split}")
        manifest_inputs[str(split)] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise FeatureContractError(f"cache input manifest {split}:{index} is invalid")
            source_manifest = _resolve_file_any(
                entry.get("path"), field=f"cache input manifest {split}:{index}"
            )
            if _sha256_file(source_manifest) != str(entry.get("sha256", "")):
                raise FeatureContractError(
                    f"cache input manifest SHA changed for {split}:{index}"
                )
            manifest_inputs[str(split)].append(source_manifest)
    semantic_rows, semantic_audit, _, _, _ = _validate_manifest_bundle(
        manifest_inputs,
        project_root=project_root,
        source_root=source_root,
    )
    semantic_by_key = {
        (row.split, row.row_id): row for row in semantic_rows
    }
    if len(semantic_by_key) != len(semantic_rows):
        raise FeatureContractError("source manifests contain duplicate semantic row keys")
    protocol_binding = report.get("protocol_binding")
    if not isinstance(protocol_binding, Mapping):
        raise FeatureContractError("cache report lacks protocol_binding")
    observed_protocol_binding = all(
        row.get("protocol_split") == row.get("split") for row in rows
    )
    if protocol_binding.get("protocol_split_metadata_bound") is not observed_protocol_binding:
        raise FeatureContractError("cache protocol_split binding flag is inconsistent")
    if protocol_binding.get("source_rows_semantically_bound") is not True:
        raise FeatureContractError("cache report lacks source-row semantic binding")
    if any("query_role_id" in npz.files for npz in []):  # pragma: no cover - documentation guard
        raise FeatureContractError("query_role_id must not enter model input")
    for path in root.rglob("*.wav"):
        raise FeatureContractError(f"cache contains copied audio: {path}")
    mixture_seen: dict[str, Path] = {}
    seen_semantic_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        label = f"cache manifest:{index}"
        if row.get("cache_schema") != SCHEMA:
            raise FeatureContractError(f"{label} has wrong cache_schema")
        if row.get("source_corpus") != SOURCE_CORPUS or _truthy(row.get("dataset_a_used", False)):
            raise FeatureContractError(f"{label} source/Dataset-A contract mismatch")
        if row.get("query_role_id_used_as_model_input") is not False:
            raise FeatureContractError(f"{label} query_role_id model-input flag is not false")
        semantic_key = (str(row.get("split", "")), str(row.get("row_id", row.get("id", ""))))
        expected_semantic = semantic_by_key.get(semantic_key)
        if expected_semantic is None:
            raise FeatureContractError(f"{label} has no matching source-manifest row")
        if semantic_key in seen_semantic_keys:
            raise FeatureContractError(f"{label} repeats a source-manifest semantic row")
        seen_semantic_keys.add(semantic_key)
        if str(row.get("source_manifest_path", "")) != expected_semantic.manifest_path.as_posix():
            raise FeatureContractError(f"{label} source manifest path mismatch")
        if str(row.get("source_manifest_row_sha256", "")) != _sha256_json(
            expected_semantic.original_row
        ):
            raise FeatureContractError(f"{label} source manifest row SHA mismatch")
        if str(row.get("source_prepared_contract_sha256", "")) != _sha256_json(
            _prepared_contract_payload(expected_semantic)
        ):
            raise FeatureContractError(f"{label} prepared semantic contract mismatch")
        mixture_path = _resolve_existing_file(
            row.get("mixture_feature"),
            field=f"{label}.mixture_feature",
            relative_roots=(root,),
            allowed_roots=(root,),
        )
        query_path = _resolve_existing_file(
            row.get("query_feature"),
            field=f"{label}.query_feature",
            relative_roots=(root,),
            allowed_roots=(root,),
        )
        if _sha256_file(query_path) != str(row.get("query_npz_sha256", "")):
            raise FeatureContractError(f"{label} query_npz_sha256 mismatch")
        if _sha256_file(mixture_path) != str(row.get("mixture_feature_sha256", "")):
            raise FeatureContractError(f"{label} mixture_feature_sha256 mismatch")
        audio_hashes = row.get("audio_sha256")
        audio_paths = _validate_path_mapping(
            row.get("resolved_audio_paths"),
            field=f"{label}.resolved_audio_paths",
            allowed_roots=artifact_roots,
            expected_hashes=audio_hashes,
        )
        source_hashes = row.get("source_sha256")
        source_paths = row.get("resolved_source_paths")
        _validate_source_mapping(
            source_paths,
            source_hashes,
            field=f"{label}.resolved_source_paths",
            source_root=source_root,
        )
        activity_path = _resolve_existing_file(
            row.get("resolved_feature_paths", {}).get("target_activity"),
            field=f"{label}.target_activity",
            relative_roots=artifact_roots,
            allowed_roots=artifact_roots,
        )
        activity = _load_activity(activity_path)

        with np.load(mixture_path, allow_pickle=False) as mixture:
            required_mixture = {
                "input_features",
                "feature_attention_mask",
                "mixture_sha256",
                "mixture_audio_path",
                "mixture_audio_sha256",
                "qwen_config_sha256",
                "qwen_feature_spec_sha256",
                "feature_array_sha256",
            }
            missing = sorted(required_mixture - set(mixture.files))
            if missing:
                raise FeatureContractError(f"{label} mixture NPZ lacks {missing}")
            features = np.asarray(mixture["input_features"])
            if features.ndim != 2 or features.shape[0] != FEATURE_SIZE or features.shape[1] < 1:
                raise FeatureContractError(f"{label} mixture feature shape is invalid: {features.shape}")
            if not np.isfinite(features).all():
                raise FeatureContractError(f"{label} mixture features are non-finite")
            if _scalar_text(mixture["mixture_sha256"], field="mixture_sha256").casefold() != str(row["mixture_sha256"]).casefold():
                raise FeatureContractError(f"{label} mixture SHA metadata mismatch")
            if _scalar_text(mixture["mixture_audio_sha256"], field="mixture_audio_sha256").casefold() != str(row["mixture_audio_sha256"]).casefold():
                raise FeatureContractError(f"{label} mixture audio SHA metadata mismatch")
            if _scalar_text(mixture["qwen_config_sha256"], field="qwen_config_sha256") != str(report["qwen_config_sha256"]):
                raise FeatureContractError(f"{label} mixture Qwen config binding mismatch")
            if _scalar_text(mixture["feature_array_sha256"], field="feature_array_sha256") != str(row["mixture_input_features_sha256"]):
                raise FeatureContractError(f"{label} mixture feature array hash mismatch")
            mask = np.asarray(mixture["feature_attention_mask"])
            if mask.shape != (features.shape[1],):
                raise FeatureContractError(f"{label} mixture attention mask shape mismatch")
        mixture_key = str(row["mixture_sha256"]).casefold()
        previous = mixture_seen.get(mixture_key)
        if previous is not None and previous != mixture_path:
            raise FeatureContractError(f"{label} mixture SHA maps to two cache files")
        mixture_seen[mixture_key] = mixture_path

        with np.load(query_path, allow_pickle=False) as query:
            if "query_role_id" in query.files:
                raise FeatureContractError(f"{label} query NPZ contains forbidden query_role_id")
            required_query = {
                "enrollment_embedding",
                "enrollment_embedding_view2",
                "target_activity",
                "row_id",
                "split",
                "base_mixture_id",
                "query_role",
                "query_speaker_id",
                "target_present",
                "source_corpus",
                "mixture_feature_sha256",
                "target_activity_sha256",
                "target_activity_array_sha256",
            }
            missing = sorted(required_query - set(query.files))
            if missing:
                raise FeatureContractError(f"{label} query NPZ lacks {missing}")
            for key, expected in (
                ("row_id", row["row_id"]),
                ("split", row["split"]),
                ("base_mixture_id", row["base_mixture_id"]),
                ("query_role", row["query_role"]),
                ("query_speaker_id", row["query_speaker_id"]),
                ("source_corpus", SOURCE_CORPUS),
                ("mixture_feature_sha256", row["mixture_feature_sha256"]),
                ("target_activity_sha256", row["target_activity_sha256"]),
            ):
                if _scalar_text(query[key], field=key) != str(expected):
                    raise FeatureContractError(f"{label} query metadata mismatch for {key}")
            if _scalar_bool(query["target_present"], field="target_present") != bool(row["target_present"]):
                raise FeatureContractError(f"{label} target_present metadata mismatch")
            embedding = np.asarray(query["enrollment_embedding"])
            embedding_view2 = np.asarray(query["enrollment_embedding_view2"])
            if embedding.shape != (CAMPP_DIM,) or embedding_view2.shape != (CAMPP_DIM,):
                raise FeatureContractError(f"{label} CAM++ enrollment shape mismatch")
            for name, value, expected in (
                ("enrollment_embedding", embedding, row.get("enrollment_embedding_sha256", "")),
                ("enrollment_embedding_view2", embedding_view2, row.get("enrollment_embedding_view2_sha256", "")),
            ):
                if abs(float(np.linalg.norm(value)) - 1.0) > 2e-5:
                    raise FeatureContractError(f"{label} {name} is not L2-normalized")
                if expected and _sha256_array(value) != str(expected):
                    raise FeatureContractError(f"{label} {name} array hash mismatch")
            aligned_activity = np.asarray(query["target_activity"], dtype=np.float32)
            alignment = row.get("activity_alignment")
            if not isinstance(alignment, Mapping):
                raise FeatureContractError(f"{label} lacks activity_alignment")
            expected_activity, _ = _align_tail(activity, int(features.shape[1]), fill=0.0)
            if not np.array_equal(aligned_activity, expected_activity):
                raise FeatureContractError(f"{label} target_activity alignment mismatch")
            if _scalar_text(query["target_activity_array_sha256"], field="target_activity_array_sha256") != str(row["target_activity_array_sha256"]):
                raise FeatureContractError(f"{label} target activity array hash metadata mismatch")
            if _sha256_array(aligned_activity) != str(row["target_activity_array_sha256"]):
                raise FeatureContractError(f"{label} target activity array hash mismatch")
            if "clean_target_input_features" in query.files:
                clean = np.asarray(query["clean_target_input_features"])
                if clean.shape != (FEATURE_SIZE, features.shape[1]):
                    raise FeatureContractError(f"{label} clean target diagnostic shape mismatch")

    if seen_semantic_keys != set(semantic_by_key):
        missing = sorted(set(semantic_by_key) - seen_semantic_keys)
        raise FeatureContractError(f"cache omits source-manifest rows: {missing[:3]}")
    if semantic_audit != report.get("overlap_audit"):
        raise FeatureContractError("source-manifest semantic overlap audit changed")
    recomputed_audit = _validate_overlap_from_rows(rows)
    _assert_no_overlap(recomputed_audit)
    if recomputed_audit != report.get("overlap_audit"):
        raise FeatureContractError("cache report overlap_audit does not match manifest")
    expected_counts = report.get("counts", {})
    if int(expected_counts.get("rows", -1)) != len(rows):
        raise FeatureContractError("cache report row count mismatch")
    manifest_sha = _sha256_file(manifest_path)
    payload_paths = [manifest_path]
    payload_paths.extend(sorted((root / "mixture").glob("*.npz")))
    payload_paths.extend(sorted((root / "query").glob("*.npz")))
    cache_sha = _payload_sha256(payload_paths, root=root)
    if manifest_sha != str(report.get("manifest_sha256", "")):
        raise FeatureContractError("cache report manifest_sha256 mismatch")
    if cache_sha != str(report.get("cache_sha256", "")):
        raise FeatureContractError("cache report cache_sha256 mismatch")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        action="append",
        required=True,
        help="train manifest; repeat for deterministic protocol chunks",
    )
    parser.add_argument(
        "--dev-manifest",
        action="append",
        required=True,
        help="dev manifest; repeat only when the preregistered split is chunked",
    )
    parser.add_argument(
        "--final-manifest",
        action="append",
        help="optional frozen-final manifest; omit during train/dev model selection",
    )
    parser.add_argument("--qwen-config-dir", required=True)
    parser.add_argument("--campp-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allowed-source-root",
        default=str(DEFAULT_ALLOWED_SOURCE_ROOT),
        help="explicit AISHELL-1 source root; source lineage must stay below it",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--expected-train-groups", type=int, default=96)
    parser.add_argument("--expected-dev-groups", type=int, default=12)
    parser.add_argument("--expected-final-groups", type=int, default=6)
    parser.add_argument(
        "--no-clean-target-logmel",
        action="store_true",
        help="omit the optional low-weight clean-target diagnostic feature",
    )
    args = parser.parse_args(argv)
    expected_groups = {
        "train": args.expected_train_groups,
        "dev": args.expected_dev_groups,
    }
    if args.final_manifest is not None:
        expected_groups["final"] = args.expected_final_groups
    report = build_feature_cache(
        args.train_manifest,
        args.dev_manifest,
        args.final_manifest,
        qwen_config_dir=args.qwen_config_dir,
        campp_model=args.campp_model,
        output_dir=args.output,
        allowed_source_root=args.allowed_source_root,
        local_files_only=True,
        num_threads=args.num_threads,
        include_clean_target_logmel=not args.no_clean_target_logmel,
        expected_groups=expected_groups,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
