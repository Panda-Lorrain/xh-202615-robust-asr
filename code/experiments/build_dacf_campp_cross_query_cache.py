"""Build the provenance-locked DACF CAM++ cross-query cache v0.2.

This module is deliberately a converter, not a trainer.  It reads the
``features_manifest.jsonl`` emitted by
``build_dacf_campp_frame_features.py``, copies only frozen feature arrays into
small NPZ files, and keeps the original DACF row as an audit field.  Audio is
never copied or rewritten.

The pre-registered scale split is 48/16/16 groups for train/val/final.  The
``expected_groups`` argument is overridable only for small contract fixtures;
the CLI defaults to that fixed split.  ``val`` is an observation split and
``final`` is the only split intended for the final mechanism gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


SCHEMA = "dacf-campp-cross-query-cache-v0.2"
REPORT_SCHEMA = "dacf-campp-cross-query-cache-report-v0.2"
SOURCE_LINEAGE_SCHEMA = "dacf-source-lineage-v0.2"
DEFAULT_ALLOWED_SOURCE_ROOT = Path("E:/midea_datasets/data_aishell")
EXPECTED_GROUPS = {"train": 48, "val": 16, "final": 16}
SPLITS = ("train", "val", "final")
ROLE_ORDER = ("present_A", "present_B", "absent_C")
ROLE_TO_ID = {"present_A": 0, "present_B": 1, "absent_C": 2}
FEATURE_DIM = 512

DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)

# These are source lineage fields, not generated feature/audio artifacts.
SOURCE_FIELDS = (
    "mixture_sources",
    "enrollment_src",
    "target_src",
    "interferer_srcs",
    "noise_src",
    "rir_src",
)
AUDIO_FIELDS = (
    "recognition_audio",
    "mixture_audio",
    "enrollment_audio",
    "enrollment_audio_view2",
    "clean_target_audio",
    "target_audio",
)
FEATURE_FIELDS = (
    "mixture_feature_npz",
    "enrollment_prepool_npy",
    "enrollment_final_embedding_npy",
    "enrollment_view2_prepool_npy",
    "enrollment_view2_final_embedding_npy",
)


class CacheContractError(ValueError):
    """Raised when provenance, split, or artifact contracts are violated."""


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _looks_like_dataset_a(value: Any) -> bool:
    text = _path_text(value).casefold()
    if any(marker in text for marker in DATASET_A_MARKERS):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "dataseta" in compact


def _true_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return False


def _guard_dataset_a(value: Any, *, field: str) -> None:
    """Reject Dataset-A flags/markers recursively before opening artifacts."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in {"dataset_a_used", "dataset_a", "used_dataset_a"} and _true_value(child):
                raise CacheContractError(f"{field}.{key}=true is forbidden: Dataset-A")
            _guard_dataset_a(child, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _guard_dataset_a(child, field=f"{field}[{index}]")
        return
    if isinstance(value, (str, Path)) and _looks_like_dataset_a(value):
        raise CacheContractError(f"{field} contains forbidden Dataset-A marker: {value}")


def _has_parent_segment(value: Any) -> bool:
    text = _path_text(value)
    return any(part == ".." for part in text.split("/"))


def _assert_no_parent_segment(value: Any, *, field: str) -> None:
    if _has_parent_segment(value):
        raise CacheContractError(f"{field} contains a forbidden '..' path segment: {value}")


def _assert_no_symlink_components(path: Path, *, field: str) -> None:
    """Reject symlinks in every component, including the final file."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.normpath(str(candidate)))
    anchor = Path(candidate.anchor) if candidate.anchor else Path.cwd().anchor
    current = Path(anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                raise CacheContractError(f"{field} traverses a symlink: {candidate}")
        except OSError as exc:
            raise CacheContractError(f"cannot inspect path component {current}: {exc}") from exc


def _resolve_strict_root(raw: str | Path, *, field: str, require_exists: bool = True) -> Path:
    _guard_dataset_a(raw, field=field)
    _assert_no_parent_segment(raw, field=field)
    raw_path = Path(raw)
    if require_exists:
        try:
            candidate = raw_path.resolve(strict=True)
        except OSError as exc:
            raise CacheContractError(f"cannot resolve {field}: {raw}") from exc
    else:
        candidate = raw_path.resolve(strict=False)
    _assert_no_symlink_components(raw_path, field=field)
    if not candidate.is_dir():
        raise CacheContractError(f"{field} must be a directory: {candidate}")
    return candidate


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _assert_under_any(path: Path, roots: Sequence[Path], *, field: str) -> None:
    if not any(_is_under(path, root) for root in roots):
        shown = ", ".join(str(root) for root in roots)
        raise CacheContractError(f"{field} is outside allowed roots ({shown}): {path}")


def _resolve_existing_file(
    raw: Any,
    *,
    field: str,
    relative_roots: Sequence[Path],
    allowed_roots: Sequence[Path],
) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise CacheContractError(f"missing path field {field}")
    _guard_dataset_a(raw, field=field)
    _assert_no_parent_segment(raw, field=field)
    raw_path = Path(str(raw))
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [root / raw_path for root in relative_roots]

    existing: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        _assert_no_symlink_components(candidate, field=field)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CacheContractError(f"cannot resolve {field}={raw!r}") from exc
        if not resolved.is_file():
            raise CacheContractError(f"{field} is not a file: {resolved}")
        _assert_under_any(resolved, allowed_roots, field=field)
        if resolved not in existing:
            existing.append(resolved)

    if not existing:
        tried = ", ".join(str(path.resolve(strict=False)) for path in candidates)
        raise CacheContractError(f"cannot resolve {field}={raw!r}; tried {tried}")
    if len(existing) > 1:
        raise CacheContractError(f"ambiguous {field}={raw!r}: {existing}")
    return existing[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_path(path: Path) -> str:
    return path.as_posix().casefold()


def _scalar_text(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise CacheContractError(f"{field} must be a scalar NPZ field, got {array.shape}")
    text = str(array.item()).strip()
    if not text:
        raise CacheContractError(f"{field} is empty")
    return text


def _scalar_int(value: Any, *, field: str) -> int:
    text = _scalar_text(value, field=field)
    try:
        return int(text)
    except ValueError as exc:
        raise CacheContractError(f"{field} is not an integer: {text!r}") from exc


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CacheContractError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise CacheContractError(f"manifest row is not an object at {path}:{line_number}")
            _guard_dataset_a(value, field=f"{path}:{line_number}")
            rows.append(value)
    if not rows:
        raise CacheContractError(f"manifest is empty: {path}")
    return rows


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_strings(child)
    elif isinstance(value, (str, Path)) and str(value).strip():
        yield str(value)


def _replace_nested(value: Any, fn: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _replace_nested(child, fn) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_nested(child, fn) for child in value]
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return fn(value)


def _resolve_sources(
    row: Mapping[str, Any], *, allowed_source_root: Path, field_prefix: str
) -> tuple[dict[str, Any], set[str]]:
    resolved: dict[str, Any] = {}
    source_set: set[str] = set()
    for field in SOURCE_FIELDS:
        if field not in row or row[field] is None:
            continue

        def resolve_one(value: Any, *, source_field: str = field) -> str:
            path = _resolve_existing_file(
                value,
                field=f"{field_prefix}.{source_field}",
                relative_roots=(allowed_source_root,),
                allowed_roots=(allowed_source_root,),
            )
            canonical = _canonical_path(path)
            source_set.add(canonical)
            return path.as_posix()

        resolved[field] = _replace_nested(row[field], resolve_one)
    return resolved, source_set


def _resolve_artifact(
    raw: Any,
    *,
    field: str,
    frame_root: Path,
    project_root: Path,
    manifest_path: Path,
) -> Path:
    return _resolve_existing_file(
        raw,
        field=field,
        relative_roots=(frame_root, project_root, manifest_path.parent),
        allowed_roots=(frame_root, project_root),
    )


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CacheContractError(f"row {row.get('id', '<unknown>')} requires {field}")
    return str(value).strip()


def _required_feature_block(row: Mapping[str, Any]) -> Mapping[str, Any]:
    block = row.get("campp_frame_features")
    if not isinstance(block, Mapping):
        raise CacheContractError(f"row {row.get('id', '<unknown>')} lacks campp_frame_features")
    if not str(block.get("mixture_feature_npz", "")).strip():
        raise CacheContractError("campp_frame_features lacks mixture_feature_npz")
    for field in FEATURE_FIELDS[1:]:
        if not str(block.get(field, "")).strip():
            raise CacheContractError(f"campp_frame_features lacks {field}")
    return block


def _load_prepool(path: Path, *, expected_sha: str, group_id: str) -> np.ndarray:
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise CacheContractError(f"cannot load mixture feature {path}") from exc
    with archive:
        if "prepool" in archive:
            tokens = np.asarray(archive["prepool"])
        elif "tokens" in archive:
            tokens = np.asarray(archive["tokens"])
        else:
            raise CacheContractError(f"mixture feature lacks prepool/tokens: {path}")
        if "mixture_sha256" not in archive:
            raise CacheContractError(f"mixture feature lacks mixture_sha256: {path}")
        embedded_sha = _scalar_text(archive["mixture_sha256"], field=f"{path}:mixture_sha256")
        if embedded_sha.casefold() != expected_sha.casefold():
            raise CacheContractError(
                f"mixture feature SHA mismatch for {group_id}: {embedded_sha} != {expected_sha}"
            )
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != FEATURE_DIM or tokens.shape[0] < 1:
        raise CacheContractError(
            f"mixture prepool/tokens must be [T,{FEATURE_DIM}], got {tokens.shape}: {path}"
        )
    if not np.isfinite(tokens).all():
        raise CacheContractError(f"mixture prepool/tokens contain non-finite values: {path}")
    return np.ascontiguousarray(tokens)


def _load_vector(path: Path, *, field: str) -> np.ndarray:
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except Exception as exc:
        raise CacheContractError(f"cannot load {field}: {path}") from exc
    if value.shape != (FEATURE_DIM,):
        raise CacheContractError(f"{field} must have shape [{FEATURE_DIM}], got {value.shape}: {path}")
    if not np.isfinite(value).all():
        raise CacheContractError(f"{field} contains non-finite values: {path}")
    return np.ascontiguousarray(value)


def _load_activity(path: Path) -> np.ndarray:
    try:
        value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    except Exception as exc:
        raise CacheContractError(f"cannot load target_activity: {path}") from exc
    if value.ndim != 1 or value.size < 1:
        raise CacheContractError(f"target_activity must be non-empty 1-D: {path}")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise CacheContractError(f"target_activity must be finite and in [0,1]: {path}")
    return np.ascontiguousarray(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CacheContractError(f"artifact is outside cache root: {path}") from exc


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "item"


def _group_lineage(
    *,
    source_corpus: str,
    group_id: str,
    mixture_sha256: str,
    mixture_audio: Path,
    resolved_sources: Mapping[str, Any],
) -> str:
    payload = {
        "schema": SOURCE_LINEAGE_SCHEMA,
        "kind": "mixture",
        "source_corpus": source_corpus,
        "base_mixture_id": group_id,
        "mixture_sha256": mixture_sha256,
        "resolved_mixture_audio": _canonical_path(mixture_audio),
        "resolved_source_fields": resolved_sources.get("mixture_sources", {}),
    }
    return _sha256_json(payload)


def _query_lineage(
    *,
    source_corpus: str,
    group_lineage: str,
    row_id: str,
    group_id: str,
    role: str,
    speaker: str,
    resolved_sources: Mapping[str, Any],
    resolved_audio: Mapping[str, str],
    enrollment_sha256: str,
    enrollment_view2_sha256: str,
) -> str:
    payload = {
        "schema": SOURCE_LINEAGE_SCHEMA,
        "kind": "query",
        "source_corpus": source_corpus,
        "mixture_source_lineage_sha256": group_lineage,
        "row_id": row_id,
        "base_mixture_id": group_id,
        "query_role": role,
        "query_speaker_id": speaker,
        "resolved_source_fields": resolved_sources,
        "resolved_audio_paths": dict(resolved_audio),
        "enrollment_sha256": enrollment_sha256,
        "enrollment_view2_sha256": enrollment_view2_sha256,
    }
    return _sha256_json(payload)


def _flatten_resolved_paths(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            result.update(_flatten_resolved_paths(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            result.update(_flatten_resolved_paths(child))
    elif isinstance(value, (str, Path)) and str(value).strip():
        result.add(str(value).casefold())
    return result


def _overlap_audit(prepared: Sequence["PreparedRow"]) -> dict[str, Any]:
    by_split: dict[str, list[PreparedRow]] = {split: [] for split in SPLITS}
    for item in prepared:
        by_split[item.split].append(item)

    def split_sets(split: str) -> dict[str, set[str]]:
        rows = by_split[split]
        return {
            "query_speaker": {item.query_speaker_id.casefold() for item in rows},
            "source_path": set().union(*(item.source_path_set for item in rows)) if rows else set(),
            "mixture_sha256": {item.mixture_sha256.casefold() for item in rows},
            "mixture_path": {item.mixture_audio.as_posix().casefold() for item in rows},
            "mixture_feature_path": {item.mixture_feature.as_posix().casefold() for item in rows},
            "group_id": {item.group_id.casefold() for item in rows},
        }

    pairs = (("train", "val"), ("train", "final"), ("val", "final"))
    result: dict[str, Any] = {}
    for left, right in pairs:
        left_sets = split_sets(left)
        right_sets = split_sets(right)
        result[f"{left}_vs_{right}"] = {
            f"{name}_overlap": sorted(left_sets[name] & right_sets[name])
            for name in left_sets
        }
    return result


def _assert_no_overlap(audit: Mapping[str, Any]) -> None:
    for pair, fields in audit.items():
        for field, values in fields.items():
            if values:
                raise CacheContractError(f"{pair} {field} is not empty: {values[:8]}")


@dataclass
class PreparedRow:
    split: str
    row_id: str
    group_id: str
    query_role: str
    query_role_id: int
    target_present: bool
    query_speaker_id: str
    mixture_sha256: str
    mixture_audio: Path
    mixture_feature: Path
    mixture_tokens: np.ndarray
    mixture_tokens_sha256: str
    enrollment_audio: Path
    enrollment_audio_view2: Path
    enrollment_sha256: str
    enrollment_view2_sha256: str
    embedding: np.ndarray
    embedding_view2: np.ndarray
    target_activity: np.ndarray
    embedding_sha256: str
    embedding_view2_sha256: str
    target_activity_sha256: str
    source_corpus: str
    resolved_source_paths: Mapping[str, Any]
    source_path_set: set[str]
    resolved_audio_paths: Mapping[str, str]
    resolved_feature_paths: Mapping[str, str]
    input_feature_sha256: Mapping[str, str]
    mixture_source_lineage_sha256: str
    source_lineage_sha256: str
    original_row: Mapping[str, Any]


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    frame_root: Path,
    project_root: Path,
    manifest_path: Path,
    allowed_source_root: Path,
) -> list[PreparedRow]:
    prepared: list[PreparedRow] = []
    mixture_cache: dict[Path, tuple[str, np.ndarray]] = {}
    for row_index, row in enumerate(rows):
        row_label = f"{manifest_path}:{row_index + 1}"
        if str(row.get("split", "")) != split:
            raise CacheContractError(f"{row_label} declares split={row.get('split')!r}, expected {split!r}")
        if str(row.get("source_corpus", "")) != "AISHELL-1":
            raise CacheContractError(f"{row_label} requires source_corpus=='AISHELL-1'")
        if _true_value(row.get("dataset_a_used", False)):
            raise CacheContractError(f"{row_label} has dataset_a_used=true")

        row_id = _required_text(row, "id")
        group_id = _required_text(row, "base_mixture_id")
        role = _required_text(row, "query_role")
        if role not in ROLE_TO_ID:
            raise CacheContractError(f"{row_label} has invalid query_role={role!r}")
        try:
            role_id = int(row.get("query_role_id"))
        except (TypeError, ValueError) as exc:
            raise CacheContractError(f"{row_label} has invalid query_role_id") from exc
        if role_id != ROLE_TO_ID[role]:
            raise CacheContractError(f"{row_label} query_role/query_role_id mismatch")
        if isinstance(row.get("target_present"), bool):
            target_present = bool(row["target_present"])
        else:
            target_present = _true_value(row.get("target_present"))
        if target_present != (role != "absent_C"):
            raise CacheContractError(f"{row_label} target_present disagrees with query_role")
        speaker = _required_text(row, "query_speaker_id")
        mixture_sha = _required_text(row, "mixture_sha256").casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", mixture_sha):
            raise CacheContractError(f"{row_label} mixture_sha256 is not a SHA256")

        resolved_sources, source_set = _resolve_sources(
            row, allowed_source_root=allowed_source_root, field_prefix=row_label
        )

        resolved_audio: dict[str, str] = {}
        for field in AUDIO_FIELDS:
            raw = row.get(field)
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                continue
            path = _resolve_artifact(
                raw,
                field=f"{row_label}.{field}",
                frame_root=frame_root,
                project_root=project_root,
                manifest_path=manifest_path,
            )
            resolved_audio[field] = path.as_posix()

        recognition_raw = row.get("recognition_audio", row.get("mixture_audio"))
        mixture_audio = _resolve_artifact(
            recognition_raw,
            field=f"{row_label}.recognition_audio",
            frame_root=frame_root,
            project_root=project_root,
            manifest_path=manifest_path,
        )
        actual_mixture_sha = _sha256_file(mixture_audio).casefold()
        if actual_mixture_sha != mixture_sha:
            raise CacheContractError(
                f"{row_label} mixture audio SHA mismatch: {actual_mixture_sha} != {mixture_sha}"
            )
        resolved_audio["recognition_audio"] = mixture_audio.as_posix()

        enrollment_audio = _resolve_artifact(
            _required_text(row, "enrollment_audio"),
            field=f"{row_label}.enrollment_audio",
            frame_root=frame_root,
            project_root=project_root,
            manifest_path=manifest_path,
        )
        enrollment_view2 = _resolve_artifact(
            _required_text(row, "enrollment_audio_view2"),
            field=f"{row_label}.enrollment_audio_view2",
            frame_root=frame_root,
            project_root=project_root,
            manifest_path=manifest_path,
        )
        enrollment_sha = _sha256_file(enrollment_audio).casefold()
        enrollment_view2_sha = _sha256_file(enrollment_view2).casefold()
        if _required_text(row, "enrollment_sha256").casefold() != enrollment_sha:
            raise CacheContractError(f"{row_label} enrollment_sha256 does not bind enrollment_audio")
        if _required_text(row, "enrollment_view2_sha256").casefold() != enrollment_view2_sha:
            raise CacheContractError(
                f"{row_label} enrollment_view2_sha256 does not bind enrollment_audio_view2"
            )
        resolved_audio["enrollment_audio"] = enrollment_audio.as_posix()
        resolved_audio["enrollment_audio_view2"] = enrollment_view2.as_posix()

        block = _required_feature_block(row)
        resolved_features: dict[str, str] = {}
        feature_hashes: dict[str, str] = {}
        for field in FEATURE_FIELDS:
            path = _resolve_artifact(
                block[field],
                field=f"{row_label}.campp_frame_features.{field}",
                frame_root=frame_root,
                project_root=project_root,
                manifest_path=manifest_path,
            )
            resolved_features[field] = path.as_posix()
            feature_hashes[field] = _sha256_file(path)

        mixture_feature = Path(resolved_features["mixture_feature_npz"])
        if mixture_feature not in mixture_cache:
            mixture_cache[mixture_feature] = (
                mixture_sha,
                _load_prepool(mixture_feature, expected_sha=mixture_sha, group_id=group_id),
            )
        cached_sha, mixture_tokens = mixture_cache[mixture_feature]
        if cached_sha != mixture_sha:
            raise CacheContractError(f"{row_label} reuses feature path with a different mixture SHA")

        embedding = _load_vector(
            Path(resolved_features["enrollment_final_embedding_npy"]),
            field="enrollment embedding",
        )
        embedding_view2 = _load_vector(
            Path(resolved_features["enrollment_view2_final_embedding_npy"]),
            field="enrollment view2 embedding",
        )
        activity_path = _resolve_artifact(
            _required_text(row, "target_activity"),
            field=f"{row_label}.target_activity",
            frame_root=frame_root,
            project_root=project_root,
            manifest_path=manifest_path,
        )
        resolved_features["target_activity"] = activity_path.as_posix()
        feature_hashes["target_activity"] = _sha256_file(activity_path)
        target_activity = _load_activity(activity_path)
        mixture_tokens_sha256 = _sha256_array(mixture_tokens)
        embedding_sha256 = _sha256_array(embedding)
        embedding_view2_sha256 = _sha256_array(embedding_view2)
        target_activity_sha256 = _sha256_array(target_activity)

        actual_feature_mixture_sha = str(block.get("mixture_audio_sha256_actual", mixture_sha)).casefold()
        if actual_feature_mixture_sha != mixture_sha:
            raise CacheContractError(f"{row_label} frame feature metadata has wrong mixture SHA")
        for field, actual in (
            ("enrollment_audio_sha256_actual", enrollment_sha),
            ("enrollment_view2_audio_sha256_actual", enrollment_view2_sha),
        ):
            declared = block.get(field)
            if declared is not None and str(declared).casefold() != actual:
                raise CacheContractError(f"{row_label} frame feature metadata has wrong {field}")

        mixture_lineage = _group_lineage(
            source_corpus="AISHELL-1",
            group_id=group_id,
            mixture_sha256=mixture_sha,
            mixture_audio=mixture_audio,
            resolved_sources=resolved_sources,
        )
        query_lineage = _query_lineage(
            source_corpus="AISHELL-1",
            group_lineage=mixture_lineage,
            row_id=row_id,
            group_id=group_id,
            role=role,
            speaker=speaker,
            resolved_sources=resolved_sources,
            resolved_audio=resolved_audio,
            enrollment_sha256=enrollment_sha,
            enrollment_view2_sha256=enrollment_view2_sha,
        )
        declared_query_lineage = row.get("source_lineage_sha256")
        if declared_query_lineage is not None and str(declared_query_lineage).strip():
            if str(declared_query_lineage).casefold() != query_lineage:
                raise CacheContractError(f"{row_label} source_lineage_sha256 metadata is tampered")
        declared_mix_lineage = block.get("source_lineage_sha256")
        if declared_mix_lineage is not None and str(declared_mix_lineage).strip():
            if str(declared_mix_lineage).casefold() != mixture_lineage:
                raise CacheContractError(f"{row_label} mixture source_lineage_sha256 metadata is tampered")

        prepared.append(
            PreparedRow(
                split=split,
                row_id=row_id,
                group_id=group_id,
                query_role=role,
                query_role_id=role_id,
                target_present=target_present,
                query_speaker_id=speaker,
                mixture_sha256=mixture_sha,
                mixture_audio=mixture_audio,
                mixture_feature=mixture_feature,
                mixture_tokens=mixture_tokens,
                mixture_tokens_sha256=mixture_tokens_sha256,
                enrollment_audio=enrollment_audio,
                enrollment_audio_view2=enrollment_view2,
                enrollment_sha256=enrollment_sha,
                enrollment_view2_sha256=enrollment_view2_sha,
                embedding=embedding,
                embedding_view2=embedding_view2,
                target_activity=target_activity,
                embedding_sha256=embedding_sha256,
                embedding_view2_sha256=embedding_view2_sha256,
                target_activity_sha256=target_activity_sha256,
                source_corpus="AISHELL-1",
                resolved_source_paths=resolved_sources,
                source_path_set=source_set,
                resolved_audio_paths=resolved_audio,
                resolved_feature_paths=resolved_features,
                input_feature_sha256=feature_hashes,
                mixture_source_lineage_sha256=mixture_lineage,
                source_lineage_sha256=query_lineage,
                original_row=dict(row),
            )
        )
    return prepared


def _validate_group_contract(prepared: Sequence[PreparedRow]) -> None:
    by_split_group: dict[tuple[str, str], list[PreparedRow]] = {}
    row_ids: set[str] = set()
    for item in prepared:
        if item.row_id in row_ids:
            raise CacheContractError(f"duplicate row id: {item.row_id}")
        row_ids.add(item.row_id)
        by_split_group.setdefault((item.split, item.group_id), []).append(item)
    for (split, group_id), rows in by_split_group.items():
        if len(rows) != 3:
            raise CacheContractError(f"{split} group {group_id!r} must contain exactly A/B/C rows")
        if sorted(item.query_role for item in rows) != sorted(ROLE_ORDER):
            raise CacheContractError(f"{split} group {group_id!r} does not contain A/B/C exactly once")
        sha_values = {item.mixture_sha256 for item in rows}
        audio_paths = {item.mixture_audio for item in rows}
        feature_paths = {item.mixture_feature for item in rows}
        lineage_values = {item.mixture_source_lineage_sha256 for item in rows}
        if (
            len(sha_values) != 1
            or len(audio_paths) != 1
            or len(feature_paths) != 1
            or len(lineage_values) != 1
        ):
            raise CacheContractError(f"{split} group {group_id!r} is not A/B/C byte-identical")
        audio_bytes = {item.mixture_audio.read_bytes() for item in rows}
        if len(audio_bytes) != 1:
            raise CacheContractError(f"{split} group {group_id!r} mixture audio bytes differ")
        token_bytes = {item.mixture_tokens.tobytes() for item in rows}
        if len(token_bytes) != 1:
            raise CacheContractError(f"{split} group {group_id!r} feature tokens differ")


def _payload_sha256(paths: Sequence[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: _relative(value, root)):
        label = _relative(path, root)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _prepare_output_root(output_root: str | Path, *, project_root: Path) -> Path:
    _guard_dataset_a(output_root, field="output")
    _assert_no_parent_segment(output_root, field="output")
    raw = Path(output_root)
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
    return output


def build_cache(
    frame_feature_root: str | Path,
    output_root: str | Path,
    *,
    allowed_source_root: str | Path = DEFAULT_ALLOWED_SOURCE_ROOT,
    expected_groups: Optional[Mapping[str, int]] = EXPECTED_GROUPS,
) -> dict[str, Any]:
    """Convert a frame-feature root into a provenance-locked v0.2 cache."""

    project_root = Path(__file__).resolve().parents[2]
    frame_root = _resolve_strict_root(frame_feature_root, field="frame-feature-root")
    _assert_under_any(frame_root, (project_root,), field="frame-feature-root")
    source_root = _resolve_strict_root(allowed_source_root, field="allowed-source-root")
    output = _prepare_output_root(output_root, project_root=project_root)
    manifest_path = frame_root / "features_manifest.jsonl"
    if not manifest_path.is_file():
        raise CacheContractError(f"missing frame feature manifest: {manifest_path}")
    _assert_no_symlink_components(manifest_path, field="frame features manifest")
    rows = _read_jsonl(manifest_path)

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in rows:
        split = str(row.get("split", ""))
        if split not in split_rows:
            raise CacheContractError(f"unsupported split={split!r}; expected train/val/final")
        split_rows[split].append(row)
    missing_splits = [split for split in SPLITS if not split_rows[split]]
    if missing_splits:
        raise CacheContractError(f"frame manifest lacks required split rows: {missing_splits}")

    if expected_groups is not None:
        for split in SPLITS:
            if split not in expected_groups:
                raise CacheContractError(f"expected_groups lacks {split}")
            row_group_ids = {str(row.get("base_mixture_id", "")) for row in split_rows[split]}
            expected = int(expected_groups[split])
            if len(row_group_ids) != expected:
                raise CacheContractError(
                    f"{split} group contract requires {expected}, got {len(row_group_ids)}"
                )

    prepared: list[PreparedRow] = []
    for split in SPLITS:
        prepared.extend(
            _prepare_rows(
                split_rows[split],
                split=split,
                frame_root=frame_root,
                project_root=project_root,
                manifest_path=manifest_path,
                allowed_source_root=source_root,
            )
        )
    _validate_group_contract(prepared)
    audit = _overlap_audit(prepared)
    _assert_no_overlap(audit)

    row_outputs: list[dict[str, Any]] = []
    mixture_outputs: dict[tuple[str, str], tuple[Path, str]] = {}
    query_paths: list[Path] = []
    mixture_paths: list[Path] = []

    by_group: dict[tuple[str, str], list[PreparedRow]] = {}
    for item in prepared:
        by_group.setdefault((item.split, item.group_id), []).append(item)

    for (split, group_id), group_rows in sorted(by_group.items()):
        first = sorted(group_rows, key=lambda item: item.query_role_id)[0]
        mixture_path = output / "mixture" / (
            f"{_safe_name(split)}__{_safe_name(group_id)}__{first.mixture_sha256[:12]}.npz"
        )
        np.savez_compressed(
            mixture_path,
            tokens=first.mixture_tokens,
            prepool=first.mixture_tokens,
            mixture_sha256=np.asarray(first.mixture_sha256),
            base_mixture_id=np.asarray(first.group_id),
            source_lineage_sha256=np.asarray(first.mixture_source_lineage_sha256),
            tokens_sha256=np.asarray(first.mixture_tokens_sha256),
            prepool_sha256=np.asarray(first.mixture_tokens_sha256),
        )
        mixture_paths.append(mixture_path)
        mixture_outputs[(split, group_id)] = (
            mixture_path,
            _sha256_file(mixture_path),
        )

        for item in sorted(group_rows, key=lambda value: value.query_role_id):
            query_path = output / "query" / (
                f"{_safe_name(split)}__{_safe_name(item.row_id)}__{item.enrollment_sha256[:12]}.npz"
            )
            np.savez_compressed(
                query_path,
                embedding=item.embedding,
                embedding_view2=item.embedding_view2,
                target_activity=item.target_activity,
                row_id=np.asarray(item.row_id),
                base_mixture_id=np.asarray(item.group_id),
                query_role=np.asarray(item.query_role),
                query_role_id=np.asarray(item.query_role_id, dtype=np.int32),
                query_speaker_id=np.asarray(item.query_speaker_id),
                enrollment_sha256=np.asarray(item.enrollment_sha256),
                enrollment_view2_sha256=np.asarray(item.enrollment_view2_sha256),
                mixture_sha256=np.asarray(item.mixture_sha256),
                source_lineage_sha256=np.asarray(item.source_lineage_sha256),
                embedding_sha256=np.asarray(item.embedding_sha256),
                embedding_view2_sha256=np.asarray(item.embedding_view2_sha256),
                target_activity_sha256=np.asarray(item.target_activity_sha256),
            )
            query_paths.append(query_path)
            mixture_rel = _relative(mixture_path, output)
            query_rel = _relative(query_path, output)
            original = dict(item.original_row)
            cache_row = dict(original)
            # Keep the complete input row intact in an explicit immutable audit copy;
            # cache consumer paths below intentionally point at the v0.2 NPZ files.
            cache_row["original_dacf_provenance"] = original
            cache_row.update(
                {
                    "cache_schema": SCHEMA,
                    "row_id": item.row_id,
                    "base_mixture_id": item.group_id,
                    "query_role": item.query_role,
                    "query_role_id": item.query_role_id,
                    "target_present": item.target_present,
                    "query_speaker_id": item.query_speaker_id,
                    "mixture_sha256": item.mixture_sha256,
                    "enrollment_sha256": item.enrollment_sha256,
                    "enrollment_view2_sha256": item.enrollment_view2_sha256,
                    "source_corpus": item.source_corpus,
                    "dataset_a_used": False,
                    "source_lineage_sha256": item.source_lineage_sha256,
                    "mixture_source_lineage_sha256": item.mixture_source_lineage_sha256,
                    "tokens_sha256": item.mixture_tokens_sha256,
                    "embedding_sha256": item.embedding_sha256,
                    "embedding_view2_sha256": item.embedding_view2_sha256,
                    "target_activity_sha256": item.target_activity_sha256,
                    "mixture_feature": mixture_rel,
                    "query_feature": query_rel,
                    "target_activity": query_rel,
                    "mixture_npz_sha256": mixture_outputs[(split, group_id)][1],
                    "query_npz_sha256": _sha256_file(query_path),
                    "mixture_feature_source": item.mixture_feature.as_posix(),
                    "target_activity_source": item.resolved_feature_paths["target_activity"],
                    "resolved_source_paths": item.resolved_source_paths,
                    "resolved_audio_paths": item.resolved_audio_paths,
                    "resolved_feature_paths": item.resolved_feature_paths,
                    "input_feature_sha256": item.input_feature_sha256,
                    "cache_provenance": {
                        "mixture_npz": mixture_rel,
                        "query_npz": query_rel,
                        "mixture_tokens_shape": [int(value) for value in item.mixture_tokens.shape],
                        "embedding_shape": [int(value) for value in item.embedding.shape],
                        "embedding_view2_shape": [int(value) for value in item.embedding_view2.shape],
                        "target_activity_shape": [int(value) for value in item.target_activity.shape],
                        "query_role_id_is_metadata_only": True,
                    },
                }
            )
            row_outputs.append(cache_row)

    output_manifest = output / "features_manifest.jsonl"
    _write_jsonl(output_manifest, sorted(row_outputs, key=lambda row: (str(row["split"]), str(row["base_mixture_id"]), int(row["query_role_id"]))))
    manifest_sha = _sha256_file(output_manifest)
    payload_paths = [output_manifest, *mixture_paths, *query_paths]
    cache_sha = _payload_sha256(payload_paths, root=output)
    split_counts = {
        split: {
            "rows": sum(1 for item in prepared if item.split == split),
            "groups": len({item.group_id for item in prepared if item.split == split}),
            "query_speakers": len({item.query_speaker_id for item in prepared if item.split == split}),
        }
        for split in SPLITS
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "cache_schema": SCHEMA,
        "dataset_a_used": False,
        "dataset_a_policy": "hard reject forbidden-corpus markers and true flags; no forbidden corpus path is read",
        "source_corpus": "AISHELL-1",
        "frame_feature_root": frame_root.as_posix(),
        "allowed_source_root": source_root.as_posix(),
        "output_root": output.as_posix(),
        "split_contract": {
            "expected_groups": dict(expected_groups) if expected_groups is not None else None,
            "final_gate_split": "final",
            "val_role": "observation_only",
            "train_role": "training_only",
        },
        "counts": {
            "rows": len(prepared),
            "groups": len(by_group),
            "mixture_npz": len(mixture_paths),
            "query_npz": len(query_paths),
            "splits": split_counts,
        },
        "overlap_audit": audit,
        "manifest_sha256": manifest_sha,
        "cache_sha256": cache_sha,
        "cache_sha256_scope": "features_manifest.jsonl plus every generated mixture/*.npz and query/*.npz; excludes self-referential cache_report.json",
        "artifacts": {
            "manifest": "features_manifest.jsonl",
            "cache_report": "cache_report.json",
            "mixture_dir": "mixture",
            "query_dir": "query",
        },
    }
    report_path = output / "cache_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_cache(output)
    return report


def _validate_output_path(raw: Any, *, root: Path, field: str) -> Path:
    return _resolve_existing_file(
        raw,
        field=field,
        relative_roots=(root,),
        allowed_roots=(root,),
    )


def _validate_source_tree(value: Any, *, allowed_root: Path, field: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_source_tree(child, allowed_root=allowed_root, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_source_tree(child, allowed_root=allowed_root, field=f"{field}[{index}]")
        return
    if value is None:
        return
    path = Path(str(value))
    if not path.is_absolute():
        raise CacheContractError(f"{field} is not an absolute resolved source path")
    _assert_no_parent_segment(value, field=field)
    _assert_no_symlink_components(path, field=field)
    resolved = path.resolve(strict=True)
    _assert_under_any(resolved, (allowed_root,), field=field)
    if not resolved.is_file():
        raise CacheContractError(f"{field} is not a source file: {resolved}")


def validate_cache(cache_root: str | Path) -> dict[str, Any]:
    """Revalidate output NPZ metadata, audio/source bindings, and cache hashes."""

    project_root = Path(__file__).resolve().parents[2]
    root = _resolve_strict_root(cache_root, field="cache-root")
    _assert_under_any(root, (project_root,), field="cache-root")
    report_path = root / "cache_report.json"
    manifest_path = root / "features_manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        raise CacheContractError("cache requires cache_report.json and features_manifest.jsonl")
    _assert_no_symlink_components(report_path, field="cache report")
    _assert_no_symlink_components(manifest_path, field="cache manifest")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _guard_dataset_a(report, field="cache_report")
    if report.get("dataset_a_used") is not False:
        raise CacheContractError("cache_report.dataset_a_used must be false")
    allowed_source_root = _resolve_strict_root(
        report.get("allowed_source_root"), field="cache_report.allowed_source_root"
    )
    rows = _read_jsonl(manifest_path)
    prepared_like: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = f"cache manifest:{index + 1}"
        if row.get("cache_schema") != SCHEMA:
            raise CacheContractError(f"{label} has wrong cache_schema")
        if row.get("source_corpus") != "AISHELL-1":
            raise CacheContractError(f"{label} has wrong source_corpus")
        if _true_value(row.get("dataset_a_used", False)):
            raise CacheContractError(f"{label} uses Dataset-A")
        _validate_source_tree(row.get("resolved_source_paths", {}), allowed_root=allowed_source_root, field=f"{label}.sources")
        for field in ("resolved_audio_paths", "resolved_feature_paths"):
            values = row.get(field)
            if not isinstance(values, Mapping):
                raise CacheContractError(f"{label} lacks {field}")
            for key, value in values.items():
                path = _validate_output_path(value, root=project_root, field=f"{label}.{field}.{key}")
                if field == "resolved_audio_paths" and key in {"recognition_audio", "enrollment_audio", "enrollment_audio_view2"}:
                    pass
                if field == "resolved_feature_paths":
                    declared = row.get("input_feature_sha256", {}).get(key)
                    if declared and _sha256_file(path) != str(declared):
                        raise CacheContractError(f"{label} input feature SHA mismatch for {key}")

        mixture_path = _validate_output_path(row.get("mixture_feature"), root=root, field=f"{label}.mixture_feature")
        query_path = _validate_output_path(row.get("query_feature"), root=root, field=f"{label}.query_feature")
        if _sha256_file(mixture_path) != str(row.get("mixture_npz_sha256", "")):
            raise CacheContractError(f"{label} mixture_npz_sha256 mismatch")
        if _sha256_file(query_path) != str(row.get("query_npz_sha256", "")):
            raise CacheContractError(f"{label} query_npz_sha256 mismatch")

        audio_paths = row["resolved_audio_paths"]
        mixture_audio = Path(str(audio_paths["recognition_audio"]))
        enrollment_audio = Path(str(audio_paths["enrollment_audio"]))
        enrollment_view2 = Path(str(audio_paths["enrollment_audio_view2"]))
        if _sha256_file(mixture_audio).casefold() != str(row["mixture_sha256"]).casefold():
            raise CacheContractError(f"{label} mixture audio no longer matches manifest SHA")
        if _sha256_file(enrollment_audio).casefold() != str(row["enrollment_sha256"]).casefold():
            raise CacheContractError(f"{label} enrollment audio no longer matches manifest SHA")
        if _sha256_file(enrollment_view2).casefold() != str(row["enrollment_view2_sha256"]).casefold():
            raise CacheContractError(f"{label} enrollment view2 audio no longer matches manifest SHA")

        with np.load(mixture_path, allow_pickle=False) as mixture:
            for key in (
                "tokens",
                "prepool",
                "mixture_sha256",
                "base_mixture_id",
                "source_lineage_sha256",
                "tokens_sha256",
                "prepool_sha256",
            ):
                if key not in mixture:
                    raise CacheContractError(f"{label} mixture NPZ lacks {key}")
            tokens = np.asarray(mixture["tokens"])
            prepool = np.asarray(mixture["prepool"])
            if tokens.shape != prepool.shape or not np.array_equal(tokens, prepool):
                raise CacheContractError(f"{label} mixture tokens/prepool mismatch")
            if _scalar_text(mixture["mixture_sha256"], field="mixture_sha256").casefold() != str(row["mixture_sha256"]).casefold():
                raise CacheContractError(f"{label} mixture SHA metadata mismatch")
            if _scalar_text(mixture["base_mixture_id"], field="base_mixture_id") != str(row["base_mixture_id"]):
                raise CacheContractError(f"{label} mixture group metadata mismatch")
            if _scalar_text(mixture["source_lineage_sha256"], field="source_lineage_sha256") != str(row["mixture_source_lineage_sha256"]):
                raise CacheContractError(f"{label} mixture lineage metadata mismatch")
            if _scalar_text(mixture["tokens_sha256"], field="tokens_sha256") != str(row["tokens_sha256"]):
                raise CacheContractError(f"{label} mixture token SHA metadata mismatch")
            if _scalar_text(mixture["prepool_sha256"], field="prepool_sha256") != str(row["tokens_sha256"]):
                raise CacheContractError(f"{label} mixture prepool SHA metadata mismatch")
            input_tokens = _load_prepool(
                Path(row["resolved_feature_paths"]["mixture_feature_npz"]),
                expected_sha=str(row["mixture_sha256"]),
                group_id=str(row["base_mixture_id"]),
            )
            if not np.array_equal(tokens, input_tokens):
                raise CacheContractError(f"{label} mixture tokens differ from input feature")

        with np.load(query_path, allow_pickle=False) as query:
            required = (
                "embedding",
                "embedding_view2",
                "target_activity",
                "row_id",
                "base_mixture_id",
                "query_role",
                "query_speaker_id",
                "enrollment_sha256",
                "enrollment_view2_sha256",
                "source_lineage_sha256",
                "embedding_sha256",
                "embedding_view2_sha256",
                "target_activity_sha256",
            )
            for key in required:
                if key not in query:
                    raise CacheContractError(f"{label} query NPZ lacks {key}")
            for key, expected in (
                ("row_id", row["row_id"]),
                ("base_mixture_id", row["base_mixture_id"]),
                ("query_role", row["query_role"]),
                ("query_speaker_id", row["query_speaker_id"]),
                ("enrollment_sha256", row["enrollment_sha256"]),
                ("enrollment_view2_sha256", row["enrollment_view2_sha256"]),
                ("source_lineage_sha256", row["source_lineage_sha256"]),
            ):
                if _scalar_text(query[key], field=key) != str(expected):
                    raise CacheContractError(f"{label} query NPZ metadata mismatch for {key}")
            if np.asarray(query["embedding"]).shape != (FEATURE_DIM,) or np.asarray(query["embedding_view2"]).shape != (FEATURE_DIM,):
                raise CacheContractError(f"{label} query embeddings have wrong shape")
            activity = np.asarray(query["target_activity"])
            if activity.ndim != 1 or activity.size < 1:
                raise CacheContractError(f"{label} query activity has wrong shape")
            input_embedding = _load_vector(
                Path(row["resolved_feature_paths"]["enrollment_final_embedding_npy"]),
                field="enrollment embedding",
            )
            input_embedding_view2 = _load_vector(
                Path(row["resolved_feature_paths"]["enrollment_view2_final_embedding_npy"]),
                field="enrollment view2 embedding",
            )
            input_activity = _load_activity(Path(row["resolved_feature_paths"]["target_activity"]))
            if not np.array_equal(np.asarray(query["embedding"]), input_embedding):
                raise CacheContractError(f"{label} query embedding differs from input feature")
            if not np.array_equal(np.asarray(query["embedding_view2"]), input_embedding_view2):
                raise CacheContractError(f"{label} query view2 embedding differs from input feature")
            if not np.array_equal(activity, input_activity):
                raise CacheContractError(f"{label} query activity differs from input feature")
            for key, value in (
                ("embedding_sha256", input_embedding),
                ("embedding_view2_sha256", input_embedding_view2),
                ("target_activity_sha256", input_activity),
            ):
                expected_hash = _sha256_array(value)
                if str(row.get(key, "")) != expected_hash:
                    raise CacheContractError(f"{label} manifest {key} does not bind input feature")
                if _scalar_text(query[key], field=key) != expected_hash:
                    raise CacheContractError(f"{label} {key} does not bind input feature")

        prepared_like.append(
            {
                "split": str(row["split"]),
                "group_id": str(row["base_mixture_id"]),
                "query_speaker_id": str(row["query_speaker_id"]),
                "source_paths": _flatten_resolved_paths(row.get("resolved_source_paths", {})),
                "mixture_sha256": str(row["mixture_sha256"]).casefold(),
                "mixture_path": _canonical_path(mixture_audio),
                "mixture_feature_path": _canonical_path(mixture_path),
            }
        )

    # Re-run the three-way audit from output rows, so a hand-edited report is
    # not accepted as evidence of split isolation.
    audit: dict[str, Any] = {}
    for left, right in (("train", "val"), ("train", "final"), ("val", "final")):
        left_rows = [row for row in prepared_like if row["split"] == left]
        right_rows = [row for row in prepared_like if row["split"] == right]
        fields = {
            "query_speaker": {row["query_speaker_id"].casefold() for row in left_rows} & {row["query_speaker_id"].casefold() for row in right_rows},
            "source_path": set().union(*(row["source_paths"] for row in left_rows)) & set().union(*(row["source_paths"] for row in right_rows)),
            "mixture_sha256": {row["mixture_sha256"] for row in left_rows} & {row["mixture_sha256"] for row in right_rows},
            "mixture_path": {row["mixture_path"] for row in left_rows} & {row["mixture_path"] for row in right_rows},
            "mixture_feature_path": {row["mixture_feature_path"] for row in left_rows} & {row["mixture_feature_path"] for row in right_rows},
            "group_id": {row["group_id"].casefold() for row in left_rows} & {row["group_id"].casefold() for row in right_rows},
        }
        audit[f"{left}_vs_{right}"] = {f"{key}_overlap": sorted(value) for key, value in fields.items()}
    _assert_no_overlap(audit)
    if audit != report.get("overlap_audit"):
        raise CacheContractError("cache_report overlap_audit does not match manifest")

    manifest_sha = _sha256_file(manifest_path)
    payload_paths = [manifest_path]
    payload_paths.extend(sorted((root / "mixture").glob("*.npz")))
    payload_paths.extend(sorted((root / "query").glob("*.npz")))
    cache_sha = _payload_sha256(payload_paths, root=root)
    if manifest_sha != report.get("manifest_sha256"):
        raise CacheContractError("cache_report manifest_sha256 mismatch")
    if cache_sha != report.get("cache_sha256"):
        raise CacheContractError("cache_report cache_sha256 mismatch")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-source-root", default=str(DEFAULT_ALLOWED_SOURCE_ROOT))
    parser.add_argument("--expected-train-groups", type=int, default=EXPECTED_GROUPS["train"])
    parser.add_argument("--expected-val-groups", type=int, default=EXPECTED_GROUPS["val"])
    parser.add_argument("--expected-final-groups", type=int, default=EXPECTED_GROUPS["final"])
    args = parser.parse_args(argv)
    report = build_cache(
        args.frame_feature_root,
        args.output,
        allowed_source_root=args.allowed_source_root,
        expected_groups={
            "train": args.expected_train_groups,
            "val": args.expected_val_groups,
            "final": args.expected_final_groups,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
