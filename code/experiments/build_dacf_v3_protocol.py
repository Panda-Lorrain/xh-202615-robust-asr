#!/usr/bin/env python3
"""Build the preregistered DACF-v3 set on official AISHELL-1 splits.

The protocol has three fixed source routes:

    protocol train -> AISHELL-1 train -> optimization
    protocol dev   -> AISHELL-1 dev   -> one fixed observation
    protocol final -> AISHELL-1 test  -> one frozen decision

The existing counterfactual builder caps one invocation at 80 mixtures.  The
default train request is 96 groups, so this wrapper partitions the already
selected, speaker-disjoint pool into deterministic generation chunks.  Each
chunk calls the existing builder and keeps its own generated audio directory;
the final report lists every manifest path instead of copying audio.

AISHELL-1 is read speech, not a verified home-command hard-negative source.
``hard_negative_verified_count`` therefore must remain zero and is reported as
a limitation rather than being presented as real home-command rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build_dacf_counterfactual import (
    MAX_SCALE_FALSIFICATION_MIXTURES,
    _assert_not_dataset_a,
    _group_by_speaker,
    build_dacf_counterfactual,
    load_aishell_items,
    normalize_source_items,
)


PROTOCOL_SCHEMA = "dacf-v3-official-aishell-protocol-v0.2"
EXPERIMENT_ID = "dacf-v3-official-aishell-v0.2"
REAUDIT_SCHEMA = "dacf-v3-official-aishell-reaudit-v0.1"
SOURCE_CORPUS = "AISHELL-1"

# These are the official AISHELL-1 speaker counts on this machine.  They are
# an upper-bound guard, not a substitute for measuring the actually loaded
# source inventory below.
OFFICIAL_SPEAKER_COUNTS = {"train": 340, "dev": 40, "test": 20}
OFFICIAL_GROUP_CAPACITY = {
    split: count // 3 for split, count in OFFICIAL_SPEAKER_COUNTS.items()
}

PROTOCOL_TO_SOURCE_SPLIT = {"train": "train", "dev": "dev", "final": "test"}
PROTOCOL_TO_BUILDER_SPLIT = {"train": "train", "dev": "val", "final": "final"}
DEFAULT_GROUP_COUNTS = {"train": 96, "dev": 12, "final": 6}
DEFAULT_SEEDS = {"train": 2026080603, "dev": 2026080604, "final": 2026080605}
AUGMENTATION_PROFILE = "balanced"
HOP_SAMPLES = 160
MAX_GENERATION_CHUNK_GROUPS = MAX_SCALE_FALSIFICATION_MIXTURES
SOURCE_PATH_FIELDS = (
    "mixture_sources",
    "enrollment_src",
    "target_src",
    "interferer_srcs",
    "hard_negative_interferer_srcs",
    "noise_src",
    "rir_src",
)

FIXED_GATE = {
    "presence_threshold": 0.50,
    "presence_auc": 0.85,
    "activity_auc": 0.80,
    "present_recall": 0.80,
    "absent_rr": 0.95,
    "query_response_mean": 0.20,
    "query_response_group_bootstrap_ci_lower": 0.05,
    "query_permutation_auc_drop_min": 0.15,
    "same_text_different_speaker_auc_deferred": 0.75,
    "same_text_different_speaker_required_stage": (
        "designed real-home-command hard-negative gate before Qwen integration; "
        "not part of the AISHELL mechanism gate because this protocol does not "
        "construct same-text speaker swaps"
    ),
    "both_enrollment_views_required": True,
    "paired_cer_delta_max": -0.05,
    "paired_cer_bootstrap_ci_upper": 0.0,
    "clean_cer_regression_max": 0.005,
    "negative_rr_drop_max": 0.01,
    "batch1_rtf_increment_max": 0.05,
}

SELECTION_POLICY = {
    "train": "train is the only split allowed to optimize parameters",
    "dev": "dev may be observed once after the checkpoint and threshold are fixed; it cannot change them",
    "final": "final is opened once only after code, seed, update count, checkpoint, and threshold are frozen",
    "final_selects": "none",
    "forbidden": "no split may select architecture, seed, augmentation, hop, gate, update count, checkpoint, or threshold after preregistration",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _assert_no_symlink_components(path: Path, *, label: str) -> None:
    """Reject links before ``resolve`` can erase evidence of traversal."""

    candidate = path if path.is_absolute() else Path.cwd() / path
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"{label} contains a forbidden '..' component: {path}")
    current = Path(candidate.anchor)
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            if current.is_symlink():
                raise ValueError(f"{label} traverses a symlink: {path}")
        except OSError as exc:
            raise ValueError(f"cannot inspect {label} path component {current}: {exc}") from exc


def _iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_values(child)
    elif value is not None:
        yield value


def _canonical_file(
    value: str | Path,
    *,
    label: str,
    root: Path | None = None,
) -> Path:
    """Resolve an existing file, reject Dataset-A, and optionally bind it to root."""

    _assert_not_dataset_a(value)
    raw_path = Path(str(value))
    _assert_no_symlink_components(raw_path, label=label)
    path = raw_path.resolve(strict=True)
    _assert_no_symlink_components(path, label=label)
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {value}")
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"{label} escapes official AISHELL root {root}: {path}"
            ) from exc
    return path


def _canonical_dir(value: str | Path, *, label: str) -> Path:
    _assert_not_dataset_a(value)
    raw_path = Path(str(value))
    _assert_no_symlink_components(raw_path, label=label)
    path = raw_path.resolve(strict=True)
    _assert_no_symlink_components(path, label=label)
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {value}")
    return path


def _aishell_path_speaker(
    path: Path,
    *,
    aishell_root: Path,
    source_split: str,
    label: str,
) -> str:
    try:
        relative = path.relative_to(aishell_root)
    except ValueError as exc:  # defensive; _canonical_file already checks this
        raise ValueError(f"{label} escapes official AISHELL root: {path}") from exc
    parts = relative.parts
    if len(parts) < 4 or parts[0].casefold() != "wav" or parts[1] != source_split:
        raise ValueError(
            f"{label} is outside official AISHELL wav/{source_split}: {path}"
        )
    speaker = str(parts[2])
    if not speaker:
        raise ValueError(f"{label} has no speaker directory: {path}")
    return speaker


def _wav_nonzero_sample_count(path: Path) -> int:
    """Read PCM samples so an absent target cannot rely on manifest metadata."""

    with wave.open(str(path), "rb") as handle:
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    dtype_by_width = {1: np.uint8, 2: np.dtype("<i2"), 4: np.dtype("<i4")}
    if sample_width not in dtype_by_width:
        raise ValueError(f"unsupported PCM sample width in clean target: {sample_width}")
    values = np.frombuffer(frames, dtype=dtype_by_width[sample_width])
    if sample_width == 1:
        values = values.astype(np.int16) - 128
    return int(np.count_nonzero(values))


def _normalise_protocol_groups(
    group_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    counts = dict(DEFAULT_GROUP_COUNTS if group_counts is None else group_counts)
    if set(counts) != set(DEFAULT_GROUP_COUNTS):
        raise ValueError(
            "group_counts must contain exactly train/dev/final; "
            f"got {sorted(counts)}"
        )
    for split, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{split} group count must be a positive integer")
    return counts


def _validate_source_items(
    source_items: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
    aishell_root: Path,
) -> list[dict[str, Any]]:
    """Enforce the official split/corpus route before any generation."""

    items = normalize_source_items(source_items)
    for item in items:
        if _as_bool(item.get("dataset_a_used", False)):
            raise ValueError("Dataset-A is forbidden by the DACF-v3 protocol")
        if str(item.get("source_corpus", "")).strip() != SOURCE_CORPUS:
            raise ValueError(
                f"source corpus must be {SOURCE_CORPUS}, got {item.get('source_corpus')!r}"
            )
        if str(item.get("split", "")) != source_split:
            raise ValueError(
                f"official route {source_split} received non-official item split "
                f"{item.get('split')!r}"
            )
        source_path = _canonical_file(
            item["wav"], label="AISHELL source", root=aishell_root
        )
        path_speaker = _aishell_path_speaker(
            source_path,
            aishell_root=aishell_root,
            source_split=source_split,
            label="AISHELL source",
        )
        if path_speaker != str(item.get("spk", "")):
            raise ValueError(
                f"AISHELL source speaker/path mismatch: {item.get('spk')!r} != "
                f"{path_speaker!r}"
            )
    return items


def capacity_report(
    source_split: str,
    source_items: Sequence[Mapping[str, Any]],
    requested_groups: int,
) -> dict[str, int | str]:
    """Return measured capacity and fail before generation if it is exceeded."""

    if source_split not in OFFICIAL_SPEAKER_COUNTS:
        raise ValueError(f"unknown official AISHELL split: {source_split}")
    if requested_groups < 1:
        raise ValueError("requested_groups must be positive")
    items = normalize_source_items(source_items)
    groups = _group_by_speaker(items)
    all_speakers = set(groups)
    eligible_speakers = {
        speaker for speaker, rows in groups.items() if len(rows) >= 2
    }
    official_speakers = OFFICIAL_SPEAKER_COUNTS[source_split]
    official_capacity = OFFICIAL_GROUP_CAPACITY[source_split]
    available_capacity = len(eligible_speakers) // 3
    report: dict[str, int | str] = {
        "source_split": source_split,
        "requested_groups": int(requested_groups),
        "official_speaker_count": int(official_speakers),
        "official_capacity_upper_bound_groups": int(official_capacity),
        "available_speaker_count": int(len(all_speakers)),
        "eligible_speaker_count": int(len(eligible_speakers)),
        "available_capacity_groups": int(available_capacity),
        "effective_capacity_groups": int(min(official_capacity, available_capacity)),
    }
    if requested_groups > official_capacity:
        raise ValueError(
            f"{source_split} requests {requested_groups} groups, but the official "
            f"AISHELL-1 capacity is only {official_capacity} groups from "
            f"{official_speakers} speakers; capacity_report={json.dumps(report, sort_keys=True)}"
        )
    if requested_groups > available_capacity:
        raise ValueError(
            f"{source_split} requests {requested_groups} groups, but the loaded "
            f"inventory has only {available_capacity} groups from "
            f"{len(eligible_speakers)} eligible speakers; capacity_report="
            f"{json.dumps(report, sort_keys=True)}"
        )
    return report


def preregistration_payload(
    *,
    aishell_root: Path,
    group_counts: Mapping[str, int],
    capacity_reports: Mapping[str, Mapping[str, int | str]],
) -> dict[str, Any]:
    """Build the immutable protocol payload written before generation."""

    split_protocol: dict[str, Any] = {}
    for protocol_split in ("train", "dev", "final"):
        source_split = PROTOCOL_TO_SOURCE_SPLIT[protocol_split]
        split_protocol[protocol_split] = {
            "source_split": source_split,
            "groups": int(group_counts[protocol_split]),
            "seed": int(DEFAULT_SEEDS[protocol_split]),
            "augmentation_profile": AUGMENTATION_PROFILE,
            "hop_samples": HOP_SAMPLES,
            "builder_manifest_split": PROTOCOL_TO_BUILDER_SPLIT[protocol_split],
            "capacity": dict(capacity_reports[protocol_split]),
        }
    return {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_a_used": False,
        "dataset_a_policy": "hard-reject paths, flags, and non-AISHELL source corpus before generation",
        "source_corpus": SOURCE_CORPUS,
        "aishell_root": aishell_root.as_posix(),
        "official_speaker_counts": dict(OFFICIAL_SPEAKER_COUNTS),
        "official_group_capacity_upper_bound": dict(OFFICIAL_GROUP_CAPACITY),
        "split_protocol": split_protocol,
        "augmentation_profile": AUGMENTATION_PROFILE,
        "hop_samples": HOP_SAMPLES,
        "fixed_gate": dict(FIXED_GATE),
        "selection_policy": dict(SELECTION_POLICY),
        "generation_policy": {
            "builder": "build_dacf_counterfactual",
            "max_generation_chunk_groups": MAX_GENERATION_CHUNK_GROUPS,
            "chunk_seed_policy": "split seed plus zero-based generation chunk index",
            "manifest_layout": "generation/<protocol_split>/chunk_<index>/<builder_split>/manifest.jsonl",
            "audio_copy": False,
        },
        "limitations": [
            "AISHELL-1 is read speech, not verified home-command audio.",
            "hard_negative_verified_count=0 is required and is not real home-command rejection evidence.",
            "same-text/different-speaker AUC is deferred to a deliberately constructed hard-negative suite before Qwen integration; it is not computable by construction here.",
            "final is not used to select a checkpoint, threshold, seed, or implementation.",
        ],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _partition_source_items(
    source_items: Sequence[Mapping[str, Any]],
    *,
    requested_groups: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Select exactly three unique eligible speakers per group/chunk."""

    items = normalize_source_items(source_items)
    grouped = _group_by_speaker(items)
    eligible = {speaker: rows for speaker, rows in grouped.items() if len(rows) >= 2}
    speakers = sorted(eligible)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    selected = speakers[: 3 * requested_groups]
    if len(selected) != 3 * requested_groups:
        raise AssertionError("capacity was not preserved while partitioning speakers")

    chunks: list[list[dict[str, Any]]] = []
    cursor = 0
    remaining = requested_groups
    while remaining:
        chunk_groups = min(remaining, MAX_GENERATION_CHUNK_GROUPS)
        chunk_speakers = set(selected[cursor : cursor + 3 * chunk_groups])
        chunks.append(
            [item for item in items if str(item["spk"]) in chunk_speakers]
        )
        cursor += 3 * chunk_groups
        remaining -= chunk_groups
    return chunks


def _builder_counts(protocol_split: str, groups: int) -> dict[str, int]:
    counts = {"n_train_mixtures": 0, "n_val_mixtures": 0, "n_final_mixtures": 0}
    field = {
        "train": "n_train_mixtures",
        "dev": "n_val_mixtures",
        "final": "n_final_mixtures",
    }[protocol_split]
    counts[field] = int(groups)
    return counts


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _assert_not_dataset_a(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest was not emitted: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest row is not an object: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _namespace_chunk_manifest_ids(
    manifest_path: Path,
    *,
    protocol_split: str,
    chunk_index: int,
    global_group_offset: int,
    expected_groups: int,
) -> dict[str, Any]:
    """Rewrite builder-local ids into one collision-free protocol namespace."""

    rows = _read_jsonl(manifest_path)
    local_group_ids: list[str] = []
    for row in rows:
        local_id = str(row.get("base_mixture_id", ""))
        if not local_id:
            raise ValueError("cannot namespace a manifest row without base_mixture_id")
        if local_id not in local_group_ids:
            local_group_ids.append(local_id)
    if len(local_group_ids) != int(expected_groups):
        raise ValueError(
            f"chunk {chunk_index} emitted {len(local_group_ids)} local group ids, "
            f"expected {expected_groups}"
        )
    mapping = {
        local_id: f"{protocol_split}_mix_{global_group_offset + index:04d}"
        for index, local_id in enumerate(local_group_ids)
    }
    for row in rows:
        local_id = str(row["base_mixture_id"])
        global_id = mapping[local_id]
        row["protocol_original_base_mixture_id"] = local_id
        row["protocol_chunk_index"] = int(chunk_index)
        row["protocol_split"] = protocol_split
        row["base_mixture_id"] = global_id
        if row.get("id") is not None:
            local_row_id = str(row["id"])
            prefix = local_id + "__"
            if not local_row_id.startswith(prefix):
                raise ValueError(
                    f"row id {local_row_id!r} is not namespaced by {local_id!r}"
                )
            row["protocol_original_id"] = local_row_id
            row["id"] = global_id + local_row_id[len(local_id) :]
        if row.get("query_role_id") is not None:
            row["counterfactual_group_key"] = (
                f"{global_id}:{int(row['query_role_id'])}"
            )
        row["environment_id"] = f"{protocol_split}:{global_id}:env"
        row["speaker_disjoint_group"] = f"{protocol_split}:{global_id}"
    _write_jsonl(manifest_path, rows)
    return {
        "chunk_index": int(chunk_index),
        "global_group_start": int(global_group_offset),
        "global_group_stop_exclusive": int(global_group_offset + expected_groups),
        "local_to_global_group_id": mapping,
    }


def _pairwise_overlap(values_by_split: Mapping[str, set[str]]) -> dict[str, list[str]]:
    names = list(values_by_split)
    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap = sorted(values_by_split[left] & values_by_split[right])
            if overlap:
                overlaps[f"{left}/{right}"] = overlap
    return overlaps


def _row_source_paths(
    row: Mapping[str, Any],
    *,
    aishell_root: Path,
    source_split: str,
) -> tuple[set[Path], Path]:
    mixture_sources = row.get("mixture_sources")
    if not isinstance(mixture_sources, Mapping) or set(mixture_sources) != {"A", "B"}:
        raise ValueError("row must expose exactly mixture_sources A/B")
    mixture_paths = {
        _canonical_file(value, label="mixture source", root=aishell_root)
        for value in mixture_sources.values()
    }
    enrollment_source = _canonical_file(
        row.get("enrollment_src", ""),
        label="enrollment source",
        root=aishell_root,
    )
    if enrollment_source in mixture_paths:
        raise ValueError("enrollment source reuses a recognition source within a query row")
    all_source_paths: set[Path] = set()
    for field in SOURCE_PATH_FIELDS:
        for value in _iter_values(row.get(field)):
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{field} contains a non-path source value: {value!r}")
            path = _canonical_file(
                value, label=f"source lineage {field}", root=aishell_root
            )
            _aishell_path_speaker(
                path,
                aishell_root=aishell_root,
                source_split=source_split,
                label=f"source lineage {field}",
            )
            all_source_paths.add(path)

    mixture_speakers = row.get("mixture_speakers")
    if not isinstance(mixture_speakers, Mapping) or set(mixture_speakers) != {"A", "B"}:
        raise ValueError("row must expose exactly mixture_speakers A/B")
    for key in ("A", "B"):
        path = _canonical_file(
            mixture_sources[key], label=f"mixture source {key}", root=aishell_root
        )
        speaker = _aishell_path_speaker(
            path,
            aishell_root=aishell_root,
            source_split=source_split,
            label=f"mixture source {key}",
        )
        if speaker != str(mixture_speakers[key]):
            raise ValueError(f"mixture source {key} speaker/path mismatch")
    enrollment_speaker = _aishell_path_speaker(
        enrollment_source,
        aishell_root=aishell_root,
        source_split=source_split,
        label="enrollment source",
    )
    if enrollment_speaker != str(row.get("enrollment_spk", "")):
        raise ValueError("enrollment source speaker/path mismatch")

    target_src = row.get("target_src")
    if target_src is not None:
        target_path = _canonical_file(
            target_src, label="target source", root=aishell_root
        )
        target_speaker = _aishell_path_speaker(
            target_path,
            aishell_root=aishell_root,
            source_split=source_split,
            label="target source",
        )
        if target_speaker != str(row.get("target_spk", "")):
            raise ValueError("target source speaker/path mismatch")

    interferer_srcs = list(row.get("interferer_srcs") or [])
    interferer_spks = [str(value) for value in (row.get("interferer_spks") or [])]
    if len(interferer_srcs) != len(interferer_spks):
        raise ValueError("interferer source/speaker counts disagree")
    for index, (value, expected_speaker) in enumerate(
        zip(interferer_srcs, interferer_spks)
    ):
        path = _canonical_file(
            value, label=f"interferer source {index}", root=aishell_root
        )
        actual_speaker = _aishell_path_speaker(
            path,
            aishell_root=aishell_root,
            source_split=source_split,
            label=f"interferer source {index}",
        )
        if actual_speaker != expected_speaker:
            raise ValueError(f"interferer source {index} speaker/path mismatch")
    return all_source_paths, enrollment_source


def _audit_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    aishell_root: Path,
    protocol_split: str,
) -> tuple[Path, str, set[Path], Path]:
    if len(rows) != 3:
        raise ValueError(f"counterfactual group must contain exactly A/B/C, got {len(rows)}")
    query_ids = [str(row.get("query_id", "")) for row in rows]
    if set(query_ids) != {"A", "B", "C"}:
        raise ValueError(f"counterfactual group roles are not exactly A/B/C: {query_ids}")

    mixture_paths: set[Path] = set()
    mixture_hashes: set[str] = set()
    source_paths: set[Path] = set()
    enrollment_sources: list[Path] = []
    for row in rows:
        if _as_bool(row.get("dataset_a_used", False)):
            raise ValueError("Dataset-A row is forbidden in DACF-v3 manifests")
        if str(row.get("source_corpus", "")) != SOURCE_CORPUS:
            raise ValueError("manifest source_corpus is not fixed to AISHELL-1")
        if _as_bool(row.get("hard_negative_complete_instruction_verified", False)):
            raise ValueError("AISHELL protocol cannot contain verified home-command hard negatives")
        if str(row.get("protocol_split", "")) != protocol_split:
            raise ValueError(
                f"manifest protocol_split disagrees with route {protocol_split}: "
                f"{row.get('protocol_split')!r}"
            )
        row_mixture_path = _canonical_file(
            row.get("recognition_audio", ""), label="recognition mixture"
        )
        row_mixture_hash = str(row.get("mixture_sha256", ""))
        actual_hash = _sha256_file(row_mixture_path)
        if actual_hash != row_mixture_hash:
            raise ValueError(
                f"mixture SHA mismatch for {row_mixture_path}: "
                f"manifest={row_mixture_hash}, actual={actual_hash}"
            )
        mixture_paths.add(row_mixture_path)
        mixture_hashes.add(row_mixture_hash)
        row_mixture_sources, enrollment_source = _row_source_paths(
            row,
            aishell_root=aishell_root,
            source_split=PROTOCOL_TO_SOURCE_SPLIT[protocol_split],
        )
        source_paths.update(row_mixture_sources)
        source_paths.add(enrollment_source)
        enrollment_sources.append(enrollment_source)

    if len(mixture_paths) != 1 or len(mixture_hashes) != 1:
        raise ValueError("A/B/C recognition mixtures are not byte-identical")
    if len(set(enrollment_sources)) != 3:
        raise ValueError("A/B/C enrollment source paths must remain distinct")

    mixture_path = next(iter(mixture_paths))
    mixture_hash = next(iter(mixture_hashes))
    mixture_speakers = set(
        str(value) for value in rows[0].get("mixture_speakers", {}).values()
    )
    if len(mixture_speakers) != 2:
        raise ValueError("mixture_speakers must expose exactly two speakers")
    for row in rows:
        query_id = str(row["query_id"])
        expected_role, expected_role_id = {
            "A": ("present_A", 0),
            "B": ("present_B", 1),
            "C": ("absent_C", 2),
        }[query_id]
        if str(row.get("query_role", "")) != expected_role:
            raise ValueError(f"query_role disagrees with query_id={query_id}")
        if int(row.get("query_role_id", -1)) != expected_role_id:
            raise ValueError(f"query_role_id disagrees with query_id={query_id}")
        query_speaker = str(row.get("query_speaker_id", ""))
        if str(row.get("enrollment_spk", "")) != query_speaker:
            raise ValueError("query_speaker_id and enrollment_spk disagree")
        if int(row.get("enrollment_view_count", -1)) != 2:
            raise ValueError("each query must expose exactly two enrollment views")
        if not _as_bool(row.get("identity_positive", False)):
            raise ValueError("enrollment views are not marked identity-positive")
        enrollment_path = _canonical_file(
            row.get("enrollment_audio", ""), label="enrollment view1"
        )
        enrollment_view2_path = _canonical_file(
            row.get("enrollment_audio_view2", ""), label="enrollment view2"
        )
        if enrollment_path == enrollment_view2_path:
            raise ValueError("enrollment view1/view2 reuse one path")
        enrollment_sha = str(row.get("enrollment_sha256", ""))
        enrollment_view2_sha = str(row.get("enrollment_view2_sha256", ""))
        if _sha256_file(enrollment_path) != enrollment_sha:
            raise ValueError("enrollment view1 SHA mismatch")
        if _sha256_file(enrollment_view2_path) != enrollment_view2_sha:
            raise ValueError("enrollment view2 SHA mismatch")
        if not enrollment_sha or not enrollment_view2_sha or enrollment_sha == enrollment_view2_sha:
            raise ValueError("enrollment view1/view2 must be byte-distinct")
        noise_sha = str(row.get("enrollment_noise_raw_sha256", ""))
        noise_view2_sha = str(row.get("enrollment_view2_noise_raw_sha256", ""))
        if not noise_sha or not noise_view2_sha or noise_sha == noise_view2_sha:
            raise ValueError("enrollment view1/view2 must use independent noise segments")
        row_mixture_speakers = set(
            str(value) for value in row.get("mixture_speakers", {}).values()
        )
        if row_mixture_speakers != mixture_speakers:
            raise ValueError("A/B/C mixture speaker identities disagree")
        if query_id == "C":
            if _as_bool(row.get("target_present", True)):
                raise ValueError("absent-C row is marked target_present")
            if row.get("target_spk") is not None:
                raise ValueError("absent-C row must have target_spk=null")
            if query_speaker in mixture_speakers:
                raise ValueError("absent-C enrollment speaker appears in mixture")
            if not _as_bool(row.get("query_C_enrollment_only", False)):
                raise ValueError("absent-C row is missing enrollment-only contract")
            if not _as_bool(row.get("clean_target_is_empty", False)):
                raise ValueError("absent-C clean target is not marked empty")
            if int(row.get("clean_target_nonzero_samples", -1)) != 0:
                raise ValueError("absent-C clean target contains nonzero samples")
            clean_target_path = _canonical_file(
                row.get("clean_target_audio", ""), label="absent-C clean target"
            )
            if _wav_nonzero_sample_count(clean_target_path) != 0:
                raise ValueError("absent-C clean target is not physically blank")
            activity_path = _canonical_file(
                row.get("target_activity", ""), label="absent-C activity"
            )
            activity = np.load(activity_path, allow_pickle=False)
            if np.asarray(activity).size == 0 or np.any(np.asarray(activity) != 0):
                raise ValueError("absent-C target activity is not all zero")
        else:
            if not _as_bool(row.get("target_present", False)):
                raise ValueError(f"present-{query_id} row is not target_present")
            if query_id not in {"A", "B"}:
                raise ValueError(f"unknown present query id: {query_id}")
            if str(row.get("target_spk", "")) != query_speaker:
                raise ValueError(f"present-{query_id} target speaker disagrees with query")
            if query_speaker not in mixture_speakers:
                raise ValueError(f"present-{query_id} query speaker is not in mixture")
            activity_path = _canonical_file(
                row.get("target_activity", ""), label=f"present-{query_id} activity"
            )
            activity = np.asarray(np.load(activity_path, allow_pickle=False))
            if activity.size == 0 or not np.any(activity > 0):
                raise ValueError(f"present-{query_id} target activity has no active frame")
    return mixture_path, mixture_hash, source_paths, enrollment_sources[0]


def audit_manifests(
    manifest_paths_by_split: Mapping[str, Sequence[str | Path]],
    *,
    aishell_root: str | Path,
    expected_groups: Mapping[str, int],
) -> dict[str, Any]:
    """Audit emitted manifests and all cross-official-split identities."""

    root = _canonical_dir(aishell_root, label="official AISHELL root")
    split_audits: dict[str, Any] = {}
    speaker_sets: dict[str, set[str]] = {}
    source_path_sets: dict[str, set[str]] = {}
    source_sha_sets: dict[str, set[str]] = {}
    enrollment_path_sets: dict[str, set[str]] = {}
    enrollment_sha_sets: dict[str, set[str]] = {}
    mixture_path_sets: dict[str, set[str]] = {}
    mixture_sha_sets: dict[str, set[str]] = {}

    for protocol_split in ("train", "dev", "final"):
        paths = [
            _canonical_file(path, label=f"{protocol_split} manifest")
            for path in manifest_paths_by_split[protocol_split]
        ]
        rows: list[dict[str, Any]] = []
        manifest_hashes: dict[str, str] = {}
        for path in paths:
            manifest_hashes[path.as_posix()] = _sha256_file(path)
            rows.extend(_read_jsonl(path))
        if len(rows) != 3 * int(expected_groups[protocol_split]):
            raise ValueError(
                f"{protocol_split} emitted {len(rows)} records, expected "
                f"{3 * int(expected_groups[protocol_split])}"
            )

        groups: dict[tuple[str, Path], list[dict[str, Any]]] = {}
        base_id_to_path: dict[str, Path] = {}
        row_ids: set[str] = set()
        for row in rows:
            base_id = str(row.get("base_mixture_id", ""))
            recognition_path = _canonical_file(
                row.get("recognition_audio", ""), label="recognition mixture"
            )
            if not base_id:
                raise ValueError("manifest row has no base_mixture_id")
            previous_path = base_id_to_path.get(base_id)
            if previous_path is not None and previous_path != recognition_path:
                raise ValueError(
                    f"duplicate base_mixture_id maps to multiple mixtures in "
                    f"{protocol_split}: {base_id}"
                )
            base_id_to_path[base_id] = recognition_path
            row_id = str(row.get("id", ""))
            if row_id:
                if row_id in row_ids:
                    raise ValueError(
                        f"duplicate row id within {protocol_split}: {row_id}"
                    )
                row_ids.add(row_id)
            groups.setdefault((base_id, recognition_path), []).append(row)

        split_speakers: set[str] = set()
        split_source_paths: set[str] = set()
        split_source_hashes: set[str] = set()
        split_enrollment_paths: set[str] = set()
        split_enrollment_hashes: set[str] = set()
        split_mixture_paths: set[str] = set()
        split_mixture_hashes: set[str] = set()
        group_mixture_paths: set[str] = set()
        group_mixture_hashes: set[str] = set()
        used_group_speakers: set[str] = set()
        used_group_source_paths: set[str] = set()
        used_group_source_hashes: set[str] = set()

        for group_rows in groups.values():
            mixture_path, mixture_hash, source_paths, _ = _audit_group(
                group_rows, aishell_root=root, protocol_split=protocol_split
            )
            group_speakers = {
                str(value)
                for row in group_rows
                for value in row.get("mixture_speakers", {}).values()
            }
            group_speakers.update(
                str(row.get("enrollment_spk", "")) for row in group_rows
            )
            repeated_speakers = sorted(used_group_speakers & group_speakers)
            if repeated_speakers:
                raise ValueError(
                    f"speaker reused by two groups within {protocol_split}: "
                    f"{repeated_speakers}"
                )
            group_source_paths = {path.as_posix() for path in source_paths}
            repeated_source_paths = sorted(
                used_group_source_paths & group_source_paths
            )
            if repeated_source_paths:
                raise ValueError(
                    f"source path reused by two groups within {protocol_split}: "
                    f"{repeated_source_paths}"
                )
            group_source_hashes = {_sha256_file(path) for path in source_paths}
            repeated_source_hashes = sorted(
                used_group_source_hashes & group_source_hashes
            )
            if repeated_source_hashes:
                raise ValueError(
                    f"source SHA reused by two groups within {protocol_split}: "
                    f"{repeated_source_hashes}"
                )
            used_group_speakers.update(group_speakers)
            used_group_source_paths.update(group_source_paths)
            used_group_source_hashes.update(group_source_hashes)
            mixture_path_text = mixture_path.as_posix()
            if mixture_path_text in group_mixture_paths:
                raise ValueError(f"mixture path reused by two groups in {protocol_split}")
            group_mixture_paths.add(mixture_path_text)
            if mixture_hash in group_mixture_hashes:
                raise ValueError(
                    f"mixture SHA reused by two groups within {protocol_split}: "
                    f"{mixture_hash}"
                )
            group_mixture_hashes.add(mixture_hash)
            split_mixture_paths.add(mixture_path_text)
            split_mixture_hashes.add(mixture_hash)
            for row in group_rows:
                row_speakers = {
                    str(value) for value in row.get("mixture_speakers", {}).values()
                }
                row_speakers.add(str(row.get("enrollment_spk", "")))
                split_speakers.update(row_speakers)
                row_mixture_sources, enrollment_source = _row_source_paths(
                    row,
                    aishell_root=root,
                    source_split=PROTOCOL_TO_SOURCE_SPLIT[protocol_split],
                )
                for source_path in (*row_mixture_sources, enrollment_source):
                    source_text = source_path.as_posix()
                    source_hash = _sha256_file(source_path)
                    split_source_paths.add(source_text)
                    split_source_hashes.add(source_hash)
                enrollment_text = enrollment_source.as_posix()
                split_enrollment_paths.add(enrollment_text)
                split_enrollment_hashes.add(_sha256_file(enrollment_source))

        if len(groups) != int(expected_groups[protocol_split]):
            raise ValueError(
                f"{protocol_split} emitted {len(groups)} groups, expected "
                f"{expected_groups[protocol_split]}"
            )
        speaker_sets[protocol_split] = split_speakers
        source_path_sets[protocol_split] = split_source_paths
        source_sha_sets[protocol_split] = split_source_hashes
        enrollment_path_sets[protocol_split] = split_enrollment_paths
        enrollment_sha_sets[protocol_split] = split_enrollment_hashes
        mixture_path_sets[protocol_split] = split_mixture_paths
        mixture_sha_sets[protocol_split] = split_mixture_hashes
        split_audits[protocol_split] = {
            "groups": len(groups),
            "records": len(rows),
            "manifest_paths": [path.as_posix() for path in paths],
            "manifest_sha256": manifest_hashes,
            "speaker_count": len(split_speakers),
            "source_path_count": len(split_source_paths),
            "source_sha256_count": len(split_source_hashes),
            "enrollment_source_path_count": len(split_enrollment_paths),
            "enrollment_source_sha256_count": len(split_enrollment_hashes),
            "mixture_path_count": len(split_mixture_paths),
            "mixture_sha256_count": len(split_mixture_hashes),
            "speakers": sorted(split_speakers),
            "source_path_sha256": {
                path: _sha256_file(Path(path)) for path in sorted(split_source_paths)
            },
            "enrollment_source_path_sha256": {
                path: _sha256_file(Path(path))
                for path in sorted(split_enrollment_paths)
            },
            "mixture_path_sha256": {
                path: _sha256_file(Path(path)) for path in sorted(split_mixture_paths)
            },
        }

    overlaps = {
        "speaker": _pairwise_overlap(speaker_sets),
        "source_path": _pairwise_overlap(source_path_sets),
        "source_sha256": _pairwise_overlap(source_sha_sets),
        "enrollment_source_path": _pairwise_overlap(enrollment_path_sets),
        "enrollment_source_sha256": _pairwise_overlap(enrollment_sha_sets),
        "mixture_path": _pairwise_overlap(mixture_path_sets),
        "mixture_sha256": _pairwise_overlap(mixture_sha_sets),
    }
    nonzero = {name: value for name, value in overlaps.items() if value}
    if nonzero:
        raise ValueError(
            "train/dev/final overlap audit failed: "
            + json.dumps(nonzero, ensure_ascii=False, sort_keys=True)
        )
    return {
        "splits": split_audits,
        "cross_split_overlaps": overlaps,
        "all_cross_split_overlaps_zero": True,
        "hard_negative_verified_count": 0,
    }


def _build_protocol_split(
    protocol_split: str,
    source_items: Sequence[Mapping[str, Any]],
    *,
    groups: int,
    seed: int,
    generation_root: Path,
) -> dict[str, Any]:
    chunks = _partition_source_items(
        source_items, requested_groups=groups, seed=seed
    )
    chunk_reports: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    groups_left = groups
    for chunk_index, chunk_items in enumerate(chunks):
        chunk_groups = min(groups_left, MAX_GENERATION_CHUNK_GROUPS)
        chunk_dir = generation_root / protocol_split / f"chunk_{chunk_index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        builder_seed = seed + chunk_index
        builder_result = build_dacf_counterfactual(
            chunk_items,
            chunk_dir,
            **_builder_counts(protocol_split, chunk_groups),
            seed=builder_seed,
            augmentation_profile=AUGMENTATION_PROFILE,
            hop_samples=HOP_SAMPLES,
            max_mixtures=MAX_GENERATION_CHUNK_GROUPS,
        )
        if not isinstance(builder_result, Mapping):
            raise ValueError("existing DACF builder did not return a result mapping")
        if int(builder_result.get("hard_negative_verified_count", -1)) != 0:
            raise ValueError(
                "official AISHELL protocol requires hard_negative_verified_count=0"
            )
        builder_split = PROTOCOL_TO_BUILDER_SPLIT[protocol_split]
        manifest_path = chunk_dir / builder_split / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"existing builder did not emit {manifest_path}")
        global_group_offset = groups - groups_left
        namespace_report = _namespace_chunk_manifest_ids(
            manifest_path,
            protocol_split=protocol_split,
            chunk_index=chunk_index,
            global_group_offset=global_group_offset,
            expected_groups=chunk_groups,
        )
        manifest_paths.append(manifest_path.resolve())
        chunk_reports.append(
            {
                "chunk_index": chunk_index,
                "groups": chunk_groups,
                "seed": builder_seed,
                "generation_dir": chunk_dir.resolve().as_posix(),
                "manifest_path": manifest_path.resolve().as_posix(),
                "manifest_sha256": _sha256_file(manifest_path),
                "id_namespace": namespace_report,
                "builder_result": dict(builder_result),
            }
        )
        groups_left -= chunk_groups
    if groups_left != 0:
        raise AssertionError("protocol split chunk accounting failed")
    return {
        "source_split": PROTOCOL_TO_SOURCE_SPLIT[protocol_split],
        "groups": groups,
        "seed": seed,
        "generation_chunks": chunk_reports,
        "manifest_paths": [path.as_posix() for path in manifest_paths],
    }


def build_official_protocol(
    aishell_root: str | Path,
    out_dir: str | Path,
    *,
    group_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the fixed official protocol; ``group_counts`` is test-only sizing."""

    _assert_not_dataset_a(aishell_root)
    _assert_not_dataset_a(out_dir)
    root = _canonical_dir(aishell_root, label="official AISHELL root")
    raw_output = Path(out_dir)
    _assert_no_symlink_components(raw_output, label="protocol output")
    output = raw_output.resolve(strict=False)
    _assert_no_symlink_components(output, label="protocol output")
    if output.exists() and not output.is_dir():
        raise ValueError(f"protocol output is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"protocol output must be new or empty: {output}")
    counts = _normalise_protocol_groups(group_counts)

    source_items_by_split: dict[str, list[dict[str, Any]]] = {}
    capacity_reports: dict[str, dict[str, int | str]] = {}
    for protocol_split in ("train", "dev", "final"):
        source_split = PROTOCOL_TO_SOURCE_SPLIT[protocol_split]
        loaded = load_aishell_items(root, (source_split,))
        items = _validate_source_items(
            loaded, source_split=source_split, aishell_root=root
        )
        source_items_by_split[protocol_split] = items
        capacity_reports[protocol_split] = capacity_report(
            source_split, items, counts[protocol_split]
        )

    output.mkdir(parents=True, exist_ok=True)
    prereg_path = output / "PREREGISTRATION.json"
    prereg = preregistration_payload(
        aishell_root=root,
        group_counts=counts,
        capacity_reports=capacity_reports,
    )
    # This is intentionally before the first call to build_dacf_counterfactual.
    _write_json(prereg_path, prereg)
    prereg_sha256 = _sha256_file(prereg_path)

    generation_root = output / "generation"
    split_reports: dict[str, Any] = {}
    manifest_paths_by_split: dict[str, list[str]] = {}
    for protocol_split in ("train", "dev", "final"):
        split_report = _build_protocol_split(
            protocol_split,
            source_items_by_split[protocol_split],
            groups=counts[protocol_split],
            seed=DEFAULT_SEEDS[protocol_split],
            generation_root=generation_root,
        )
        split_report["capacity"] = capacity_reports[protocol_split]
        split_reports[protocol_split] = split_report
        manifest_paths_by_split[protocol_split] = list(split_report["manifest_paths"])

    audit = audit_manifests(
        manifest_paths_by_split,
        aishell_root=root,
        expected_groups=counts,
    )
    report: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "dataset_a_used": False,
        "source_corpus": SOURCE_CORPUS,
        "aishell_root": root.as_posix(),
        "preregistration": prereg_path.resolve().as_posix(),
        "preregistration_sha256": prereg_sha256,
        "group_counts": counts,
        "split_reports": split_reports,
        "audit": audit,
        "hard_negative_verified_count": 0,
        "selection_policy": dict(SELECTION_POLICY),
        "limitations": list(prereg["limitations"]),
    }
    report_path = output / "build_report.json"
    report["report_path"] = report_path.resolve().as_posix()
    _write_json(report_path, report)
    return report


def reaudit_existing_protocol(
    protocol_dir: str | Path,
    output_report: str | Path,
) -> dict[str, Any]:
    """Re-run the current audit code against an already generated protocol."""

    protocol_root = _canonical_dir(protocol_dir, label="protocol directory")
    source_report_path = _canonical_file(
        protocol_root / "build_report.json",
        label="source build report",
        root=protocol_root,
    )
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if not isinstance(source_report, Mapping):
        raise ValueError("source build report is not an object")
    if source_report.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("source build report protocol schema mismatch")
    if source_report.get("dataset_a_used") is not False:
        raise ValueError("source build report violates Dataset-A policy")
    if source_report.get("source_corpus") != SOURCE_CORPUS:
        raise ValueError("source build report source corpus mismatch")

    prereg_path = _canonical_file(
        source_report.get("preregistration", ""),
        label="protocol preregistration",
        root=protocol_root,
    )
    prereg_sha = _sha256_file(prereg_path)
    if prereg_sha != str(source_report.get("preregistration_sha256", "")):
        raise ValueError("protocol preregistration SHA differs from build report")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(prereg, Mapping) or prereg.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("protocol preregistration schema mismatch")

    counts = _normalise_protocol_groups(source_report.get("group_counts"))
    split_reports = source_report.get("split_reports")
    if not isinstance(split_reports, Mapping):
        raise ValueError("source build report lacks split_reports")
    manifests: dict[str, list[str]] = {}
    manifest_bindings: dict[str, dict[str, str]] = {}
    for split in ("train", "dev", "final"):
        split_report = split_reports.get(split)
        if not isinstance(split_report, Mapping):
            raise ValueError(f"source build report lacks {split} split report")
        paths = split_report.get("manifest_paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"source build report lacks {split} manifests")
        manifests[split] = []
        manifest_bindings[split] = {}
        for value in paths:
            path = _canonical_file(
                value,
                label=f"{split} protocol manifest",
                root=protocol_root,
            )
            manifests[split].append(path.as_posix())
            manifest_bindings[split][path.as_posix()] = _sha256_file(path)

    audit = audit_manifests(
        manifests,
        aishell_root=source_report.get("aishell_root", ""),
        expected_groups=counts,
    )
    result: dict[str, Any] = {
        "schema": REAUDIT_SCHEMA,
        "protocol_schema": PROTOCOL_SCHEMA,
        "experiment_id": source_report.get("experiment_id"),
        "dataset_a_used": False,
        "source_corpus": SOURCE_CORPUS,
        "source_build_report": source_report_path.as_posix(),
        "source_build_report_sha256": _sha256_file(source_report_path),
        "protocol_preregistration": prereg_path.as_posix(),
        "protocol_preregistration_sha256": prereg_sha,
        "audit_code": Path(__file__).resolve().as_posix(),
        "audit_code_sha256": _sha256_file(Path(__file__).resolve()),
        "group_counts": counts,
        "manifest_sha256": manifest_bindings,
        "audit": audit,
        "hard_negative_verified_count": 0,
    }
    raw_output = Path(output_report)
    _assert_no_symlink_components(raw_output, label="reaudit output")
    output = raw_output.resolve(strict=False)
    try:
        output.relative_to(protocol_root)
    except ValueError as exc:
        raise ValueError("reaudit output must stay inside the protocol directory") from exc
    if output.exists():
        raise FileExistsError(f"reaudit output already exists: {output}")
    result["report_path"] = output.as_posix()
    _write_json(output, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root")
    parser.add_argument("--out")
    parser.add_argument("--reaudit-existing")
    parser.add_argument("--reaudit-report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.reaudit_existing or args.reaudit_report:
        if not args.reaudit_existing or not args.reaudit_report:
            raise SystemExit("reaudit mode requires --reaudit-existing and --reaudit-report")
        if args.aishell_root or args.out:
            raise SystemExit("reaudit mode does not accept --aishell-root/--out")
        result = reaudit_existing_protocol(args.reaudit_existing, args.reaudit_report)
    else:
        if not args.aishell_root or not args.out:
            raise SystemExit("build mode requires --aishell-root and --out")
        result = build_official_protocol(args.aishell_root, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
