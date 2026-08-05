"""Auditable mini-G2 trainer/evaluator for the frozen CAM++ cross-query cache.

The cache contains one frozen mixture-token tensor for each A/B/C group and
two enrollment embeddings for every query.  This probe trains only the
low-rank :class:`DACFCAMPPQueryMatcher`; ``query_role_id`` is retained as an
audit field and is never passed to the model.

This is a fixed mechanism gate.  It does not tune a threshold or select a
checkpoint on validation data, and it does not measure CER or hard-negative
rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from dacf_campp_cross_query import (
    DACFCAMPPQueryMatcher,
    CrossQueryOutput,
    class_balanced_bce,
    counterfactual_margin_loss,
    resize_activity_targets,
)
from build_dacf_campp_cross_query_cache import (
    SCHEMA as PROVENANCE_CACHE_SCHEMA,
    validate_cache as validate_provenance_cache,
)


DEFAULT_CACHE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "dacf_campp_cross_query_probe_20260806"
)
DEFAULT_MANIFEST_NAME = "features_manifest.jsonl"
DEFAULT_SEED = 20260806
DEFAULT_UPDATES = 240
BOOTSTRAP_SAMPLES = 2_000
MAX_UPDATES = 600
FEATURE_DIM = 512
QUERY_DIM = 32
LOGIT_SCALE = 10.0
TOP_FRACTION = 0.25
FIXED_PRESENCE_THRESHOLD = 0.5
PARAMETER_LIMIT = 100_000
ABC_MARGIN = 0.20

# These are deliberately literals, not validation-selected hyperparameters.
LOSS_WEIGHTS = {
    "presence_bce": 1.0,
    "activity_bce": 1.0,
    "abc_margin": 1.0,
    "view_logits_consistency": 0.10,
}
OPTIMIZER_WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0

ROLE_ORDER = ("present_A", "present_B", "absent_C")
SPLIT_ORDER = ("train", "val", "final")
STRICT_COUNT_PROFILES = (
    ({"train": 72, "val": 24, "final": 0}, {"train": 24, "val": 8, "final": 0}),
    ({"train": 144, "val": 48, "final": 48}, {"train": 48, "val": 16, "final": 16}),
)
REQUIRED_FIELDS = {
    "split",
    "base_mixture_id",
    "id",
    "query_role",
    "query_role_id",
    "target_present",
    "mixture_feature",
    "query_feature",
    "target_activity",
    "mixture_sha256",
    "query_speaker_id",
    "dataset_a_used",
}
GATE_THRESHOLDS = {
    "auc": 0.80,
    "present_recall": 0.75,
    "absent_rr": 0.75,
    "query_response_mean": 0.20,
    "activity_auc": 0.70,
}
DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)


@dataclass(frozen=True)
class QueryFeature:
    """One row's frozen query features and audit-only metadata."""

    row_id: str
    role: str
    query_role_id_audit: Any
    target_present: bool
    query_speaker_id: str
    embedding: Tensor
    embedding_view2: Tensor
    target_activity: Tensor
    query_feature_path: Path
    target_activity_path: Path


@dataclass(frozen=True)
class FeatureGroup:
    """A complete A/B/C group with one shared mixture tensor object."""

    split: str
    base_mixture_id: str
    mixture_sha256: str
    mixture_feature_path: Path
    mixture_tokens: Tensor
    queries: Mapping[str, QueryFeature]


@dataclass(frozen=True)
class LoadedCache:
    root: Path
    manifest_path: Path
    train_groups: tuple[FeatureGroup, ...]
    val_groups: tuple[FeatureGroup, ...]
    final_groups: tuple[FeatureGroup, ...]
    audit: Mapping[str, Any]


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _looks_like_dataset_a(value: Any) -> bool:
    text = _path_text(value).casefold()
    if any(marker in text for marker in DATASET_A_MARKERS):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "dataseta" in compact


def _guard_dataset_a(value: Any, *, field: str) -> None:
    """Reject explicit Dataset-A flags and path markers before feature reads."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if key_text in {"dataset_a_used", "dataset_a", "used_dataset_a"}:
                if bool(child):
                    raise ValueError(f"{field}.{key}=true is forbidden: Dataset-A")
            _guard_dataset_a(child, field=f"{field}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, child in enumerate(value):
            _guard_dataset_a(child, field=f"{field}[{index}]")
        return
    if isinstance(value, (str, Path)) and _looks_like_dataset_a(value):
        raise ValueError(f"{field} contains forbidden Dataset-A marker: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.resolve(strict=True)
    _guard_dataset_a(str(path), field="manifest")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            _guard_dataset_a(row, field=f"manifest:{line_number}")
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(
                    f"manifest line {line_number} missing required fields: {missing}"
                )
            if bool(row["dataset_a_used"]):
                raise ValueError(f"manifest line {line_number} uses Dataset-A")
            if str(row["split"]) not in set(SPLIT_ORDER):
                raise ValueError(
                    f"manifest line {line_number} has invalid split={row['split']!r}"
                )
            if str(row["query_role"]) not in ROLE_ORDER:
                raise ValueError(
                    f"manifest line {line_number} has invalid query_role="
                    f"{row['query_role']!r}"
                )
            if not str(row["query_speaker_id"]).strip():
                raise ValueError(f"manifest line {line_number} has empty query speaker")
            try:
                int(row["query_role_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"manifest line {line_number} query_role_id is not audit numeric"
                ) from exc
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def _project_root_for_cache(cache_root: Path) -> Path:
    # The real cache is project/code/runs/<cache>.  For a temporary fixture the
    # cache itself is the useful root and this fallback is harmless.
    if len(cache_root.parents) >= 3:
        return cache_root.parents[2]
    return cache_root


def _resolve_artifact(raw: Any, *, cache_root: Path, manifest_path: Path, field: str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError(f"missing path field {field}")
    _guard_dataset_a(raw, field=field)
    raw_path = Path(str(raw))
    if raw_path.is_absolute():
        candidates = [raw_path]
    else:
        candidates = [
            cache_root / raw_path,
            _project_root_for_cache(cache_root) / raw_path,
            manifest_path.parent / raw_path,
            Path.cwd() / raw_path,
        ]
    existing: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.exists() and resolved.is_file():
            if resolved not in existing:
                existing.append(resolved)
    if not existing:
        shown = ", ".join(str(path.resolve(strict=False)) for path in candidates)
        raise FileNotFoundError(f"cannot resolve {field}={raw!r}; tried {shown}")
    if len(existing) > 1:
        raise ValueError(f"ambiguous {field}={raw!r}; existing paths: {existing}")
    return existing[0]


def _scalar_text(value: Any, *, field: str) -> str:
    try:
        scalar = value.item()
    except AttributeError:
        scalar = value
    text = str(scalar).strip()
    if not text:
        raise ValueError(f"empty scalar field {field}")
    return text


def _load_mixture(path: Path, declared_sha256: str) -> Tensor:
    with np.load(path, allow_pickle=False) as archive:
        if "tokens" not in archive or "mixture_sha256" not in archive:
            raise ValueError(f"mixture feature lacks tokens/SHA fields: {path}")
        embedded_sha = _scalar_text(archive["mixture_sha256"], field="mixture_sha256")
        if embedded_sha != declared_sha256:
            raise ValueError(
                f"mixture SHA mismatch in {path}: {embedded_sha} != {declared_sha256}"
            )
        tokens = np.asarray(archive["tokens"])
    if tokens.ndim != 2 or tokens.shape[1] != FEATURE_DIM or tokens.shape[0] < 1:
        raise ValueError(f"mixture tokens must be [T,{FEATURE_DIM}], got {tokens.shape}")
    tokens = np.asarray(tokens, dtype=np.float32)
    if not np.isfinite(tokens).all():
        raise ValueError(f"mixture tokens contain non-finite values: {path}")
    return torch.from_numpy(np.ascontiguousarray(tokens))


def _load_query(row: Mapping[str, Any], *, cache_root: Path, manifest_path: Path) -> QueryFeature:
    query_path = _resolve_artifact(
        row["query_feature"],
        cache_root=cache_root,
        manifest_path=manifest_path,
        field="query_feature",
    )
    activity_path = _resolve_artifact(
        row["target_activity"],
        cache_root=cache_root,
        manifest_path=manifest_path,
        field="target_activity",
    )
    with np.load(query_path, allow_pickle=False) as archive:
        for key in ("embedding", "embedding_view2"):
            if key not in archive:
                raise ValueError(f"query feature lacks {key}: {query_path}")
        embedding = np.asarray(archive["embedding"])
        embedding_view2 = np.asarray(archive["embedding_view2"])
        embedded_activity = (
            np.asarray(archive["target_activity"])
            if "target_activity" in archive
            else None
        )
        if row.get("cache_schema") == PROVENANCE_CACHE_SCHEMA:
            expected_metadata = {
                "row_id": row.get("row_id", row["id"]),
                "base_mixture_id": row["base_mixture_id"],
                "query_role": row["query_role"],
                "query_speaker_id": row["query_speaker_id"],
                "enrollment_sha256": row["enrollment_sha256"],
                "enrollment_view2_sha256": row["enrollment_view2_sha256"],
                "source_lineage_sha256": row["source_lineage_sha256"],
            }
            for key, expected in expected_metadata.items():
                if key not in archive:
                    raise ValueError(f"provenance query feature lacks {key}: {query_path}")
                actual = _scalar_text(archive[key], field=key)
                if actual != str(expected):
                    raise ValueError(
                        f"query feature metadata mismatch for {key}: "
                        f"{actual!r} != {expected!r}"
                    )
    for name, value in (("embedding", embedding), ("embedding_view2", embedding_view2)):
        if value.shape != (FEATURE_DIM,):
            raise ValueError(
                f"{name} must have shape [{FEATURE_DIM}], got {value.shape}: {query_path}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values: {query_path}")
    if embedded_activity is not None:
        if activity_path != query_path:
            raise ValueError(
                "provenance cache target_activity must be bound inside query NPZ"
            )
        activity = embedded_activity
    else:
        activity = np.asarray(np.load(activity_path, allow_pickle=False))
    if activity.ndim != 1 or activity.size < 1:
        raise ValueError(f"target_activity must be non-empty 1-D: {activity_path}")
    activity = np.asarray(activity, dtype=np.float32)
    if not np.isfinite(activity).all() or np.any(activity < 0.0) or np.any(activity > 1.0):
        raise ValueError(f"target_activity must be finite and in [0,1]: {activity_path}")
    return QueryFeature(
        row_id=str(row["id"]),
        role=str(row["query_role"]),
        # This is copied into the report only.  It is never used below to
        # choose labels, reorder tensors, or call the matcher.
        query_role_id_audit=row["query_role_id"],
        target_present=bool(row["target_present"]),
        query_speaker_id=str(row["query_speaker_id"]),
        embedding=torch.from_numpy(np.ascontiguousarray(embedding, dtype=np.float32)),
        embedding_view2=torch.from_numpy(
            np.ascontiguousarray(embedding_view2, dtype=np.float32)
        ),
        target_activity=torch.from_numpy(np.ascontiguousarray(activity)),
        query_feature_path=query_path,
        target_activity_path=activity_path,
    )


def _canonical_file_label(path: Path, *, cache_root: Path) -> str:
    try:
        return "cache/" + path.relative_to(cache_root).as_posix()
    except ValueError:
        project_root = _project_root_for_cache(cache_root)
        try:
            return "project/" + path.relative_to(project_root).as_posix()
        except ValueError:
            return "absolute/" + path.as_posix()


def _input_cache_sha256(
    *, cache_root: Path, manifest_path: Path, artifact_paths: Sequence[Path]
) -> tuple[str, int]:
    paths: dict[str, Path] = {
        "cache/features_manifest.jsonl": manifest_path,
    }
    for path in artifact_paths:
        paths[_canonical_file_label(path, cache_root=cache_root)] = path
    digest = hashlib.sha256()
    for label in sorted(paths):
        path = paths[label]
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(paths)


def _validate_and_load_cache(
    cache_root: str | Path,
    *,
    manifest: str | Path | None = None,
    strict_counts: bool = True,
) -> LoadedCache:
    root = Path(cache_root).resolve(strict=True)
    manifest_path = (
        Path(manifest).resolve(strict=True)
        if manifest is not None
        else (root / DEFAULT_MANIFEST_NAME).resolve(strict=True)
    )
    rows = _read_manifest(manifest_path)
    cache_schemas = {
        str(row.get("cache_schema"))
        for row in rows
        if row.get("cache_schema") is not None
    }
    has_final_rows = any(str(row["split"]) == "final" for row in rows)
    provenance_report: Optional[Mapping[str, Any]] = None
    if cache_schemas:
        if cache_schemas != {PROVENANCE_CACHE_SCHEMA}:
            raise ValueError(f"mixed/unknown cache schemas: {sorted(cache_schemas)}")
        expected_manifest = (root / DEFAULT_MANIFEST_NAME).resolve(strict=True)
        if manifest_path != expected_manifest:
            raise ValueError(
                "provenance cache must use its root features_manifest.jsonl"
            )
        provenance_report = validate_provenance_cache(root)
    elif has_final_rows:
        raise ValueError(
            "final holdout requires provenance-locked cross-query cache v0.2"
        )
    by_split_group: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in SPLIT_ORDER
    }
    for row in rows:
        split = str(row["split"])
        group_id = str(row["base_mixture_id"])
        by_split_group[split].setdefault(group_id, []).append(row)

    speakers_by_split = {
        split: {
            str(row["query_speaker_id"])
            for row in rows
            if str(row["split"]) == split
        }
        for split in SPLIT_ORDER
    }
    pairwise_identity_overlap: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            key = f"{left}/{right}"
            overlaps = {
                "group": sorted(
                    set(by_split_group[left]) & set(by_split_group[right])
                ),
                "speaker": sorted(
                    speakers_by_split[left] & speakers_by_split[right]
                ),
            }
            pairwise_identity_overlap[key] = overlaps
            contaminated = {name: value for name, value in overlaps.items() if value}
            if contaminated:
                raise ValueError(f"{key} identity overlap: {contaminated}")

    row_counts = {
        split: sum(len(group) for group in by_split_group[split].values())
        for split in SPLIT_ORDER
    }
    group_counts = {
        split: len(by_split_group[split]) for split in SPLIT_ORDER
    }
    if strict_counts and not any(
        row_counts == expected_rows and group_counts == expected_groups
        for expected_rows, expected_groups in STRICT_COUNT_PROFILES
    ):
        raise ValueError(
            "cache does not match either fixed 24/8 mini-G2 or preregistered "
            f"48/16/16 scale profile: rows={row_counts}, groups={group_counts}"
        )

    mixture_cache: dict[Path, Tensor] = {}
    mixture_sha_by_path: dict[Path, str] = {}
    loaded_groups: dict[str, list[FeatureGroup]] = {
        split: [] for split in SPLIT_ORDER
    }
    all_artifacts: list[Path] = []
    role_id_audit: dict[str, list[Any]] = {role: [] for role in ROLE_ORDER}
    for split in SPLIT_ORDER:
        for group_id in sorted(by_split_group[split]):
            group_rows = by_split_group[split][group_id]
            if len(group_rows) != 3:
                raise ValueError(
                    f"{split} group {group_id!r} must have exactly A/B/C rows"
                )
            roles = [str(row["query_role"]) for row in group_rows]
            if sorted(roles) != sorted(ROLE_ORDER):
                raise ValueError(
                    f"{split} group {group_id!r} roles are not A/B/C: {roles}"
                )
            mixture_features = {str(row["mixture_feature"]) for row in group_rows}
            if len(mixture_features) != 1:
                raise ValueError(
                    f"{split} group {group_id!r} does not reuse one mixture feature"
                )
            mixture_hashes = {
                str(row["mixture_sha256"]).strip() for row in group_rows
            }
            if len(mixture_hashes) != 1 or not next(iter(mixture_hashes)):
                raise ValueError(f"{split} group {group_id!r} has inconsistent mixture SHA")
            mixture_path = _resolve_artifact(
                group_rows[0]["mixture_feature"],
                cache_root=root,
                manifest_path=manifest_path,
                field="mixture_feature",
            )
            mixture_sha = next(iter(mixture_hashes))
            prior_sha = mixture_sha_by_path.get(mixture_path)
            if prior_sha is not None and prior_sha != mixture_sha:
                raise ValueError(
                    f"mixture path {mixture_path} is declared with multiple SHA256 "
                    f"values: {prior_sha} != {mixture_sha}"
                )
            mixture_sha_by_path[mixture_path] = mixture_sha
            if mixture_path not in mixture_cache:
                mixture_cache[mixture_path] = _load_mixture(mixture_path, mixture_sha)
            mixture_tokens = mixture_cache[mixture_path]
            all_artifacts.append(mixture_path)

            queries: dict[str, QueryFeature] = {}
            for row in group_rows:
                role = str(row["query_role"])
                expected_present = role != "absent_C"
                if bool(row["target_present"]) != expected_present:
                    raise ValueError(
                        f"{split} group {group_id!r} target_present disagrees with role"
                    )
                query_role_id = row["query_role_id"]
                role_id_audit[role].append(query_role_id)
                query = _load_query(row, cache_root=root, manifest_path=manifest_path)
                queries[role] = query
                all_artifacts.extend(
                    [query.query_feature_path, query.target_activity_path]
                )
            loaded_groups[split].append(
                FeatureGroup(
                    split=split,
                    base_mixture_id=group_id,
                    mixture_sha256=mixture_sha,
                    mixture_feature_path=mixture_path,
                    mixture_tokens=mixture_tokens,
                    queries={role: queries[role] for role in ROLE_ORDER},
                )
            )

    mixture_sha_by_split = {
        split: {group.mixture_sha256 for group in loaded_groups[split]}
        for split in SPLIT_ORDER
    }
    mixture_path_by_split = {
        split: {group.mixture_feature_path for group in loaded_groups[split]}
        for split in SPLIT_ORDER
    }
    query_path_by_split = {
        split: {
            query.query_feature_path
            for group in loaded_groups[split]
            for query in group.queries.values()
        }
        for split in SPLIT_ORDER
    }
    activity_path_by_split = {
        split: {
            query.target_activity_path
            for group in loaded_groups[split]
            for query in group.queries.values()
        }
        for split in SPLIT_ORDER
    }
    pairwise_artifact_overlap: dict[str, dict[str, list[str]]] = {}
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            key = f"{left}/{right}"
            overlap = {
                "mixture_sha": sorted(
                    mixture_sha_by_split[left] & mixture_sha_by_split[right]
                ),
                "mixture_path": sorted(
                    path.as_posix()
                    for path in mixture_path_by_split[left]
                    & mixture_path_by_split[right]
                ),
                "query_feature_path": sorted(
                    path.as_posix()
                    for path in query_path_by_split[left] & query_path_by_split[right]
                ),
                "activity_path": sorted(
                    path.as_posix()
                    for path in activity_path_by_split[left]
                    & activity_path_by_split[right]
                ),
            }
            pairwise_artifact_overlap[key] = overlap
            contaminated = {name: values for name, values in overlap.items() if values}
            if contaminated:
                raise ValueError(f"{key} artifact overlap: {contaminated}")

    loader_cache_sha, cache_file_count = _input_cache_sha256(
        cache_root=root,
        manifest_path=manifest_path,
        artifact_paths=all_artifacts,
    )
    audit = {
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "cache_sha256": (
            str(provenance_report["cache_sha256"])
            if provenance_report is not None
            else loader_cache_sha
        ),
        "cache_sha256_scope": (
            str(provenance_report["cache_sha256_scope"])
            if provenance_report is not None
            else "manifest plus referenced mixture/query/activity files"
        ),
        "loader_cache_sha256": loader_cache_sha,
        "cache_file_count": cache_file_count,
        **{f"{split}_rows": row_counts[split] for split in SPLIT_ORDER},
        **{f"{split}_groups": group_counts[split] for split in SPLIT_ORDER},
        **{
            f"{split}_speakers": len(speakers_by_split[split])
            for split in SPLIT_ORDER
        },
        "speaker_overlap": pairwise_identity_overlap["train/val"]["speaker"],
        "group_overlap": pairwise_identity_overlap["train/val"]["group"],
        "pairwise_identity_overlap": pairwise_identity_overlap,
        "split_artifact_overlap": pairwise_artifact_overlap,
        "dataset_a_used": False,
        "query_role_id_audit_values": role_id_audit,
        "query_role_id_used_as_model_input": False,
        "provenance_locked": provenance_report is not None,
        "provenance_cache_validation": (
            {
                "schema": provenance_report["schema"],
                "cache_schema": provenance_report["cache_schema"],
                "manifest_sha256": provenance_report["manifest_sha256"],
                "cache_sha256": provenance_report["cache_sha256"],
                "source_corpus": provenance_report["source_corpus"],
                "allowed_source_root": provenance_report["allowed_source_root"],
                "split_contract": provenance_report["split_contract"],
                "overlap_audit": provenance_report["overlap_audit"],
            }
            if provenance_report is not None
            else None
        ),
    }
    return LoadedCache(
        root=root,
        manifest_path=manifest_path,
        train_groups=tuple(loaded_groups["train"]),
        val_groups=tuple(loaded_groups["val"]),
        final_groups=tuple(loaded_groups["final"]),
        audit=audit,
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return resolved


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _group_inputs(
    group: FeatureGroup, *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build two query batches while retaining one shared mixture tensor."""

    # Ordering is controlled by the role name for the ABC loss.  The audit-only
    # query_role_id is intentionally absent from this function's inputs.
    queries = [group.queries[role] for role in ROLE_ORDER]
    mixture = group.mixture_tokens.to(device=device)
    mixture_batch = mixture.unsqueeze(0).expand(len(ROLE_ORDER), -1, -1)
    query_main = torch.stack([query.embedding for query in queries]).to(device)
    query_view2 = torch.stack([query.embedding_view2 for query in queries]).to(device)
    labels = torch.tensor(
        [float(query.target_present) for query in queries],
        dtype=torch.float32,
        device=device,
    )
    activity = torch.stack([query.target_activity for query in queries]).to(device)
    return mixture_batch, query_main, query_view2, labels, activity


def _forward_group(
    model: DACFCAMPPQueryMatcher,
    group: FeatureGroup,
    *,
    device: Optional[torch.device] = None,
) -> tuple[CrossQueryOutput, CrossQueryOutput, Tensor, Tensor]:
    """Forward main and view2 against the same ABC mixture-token batch."""

    resolved = device or next(model.parameters()).device
    mixture_batch, query_main, query_view2, labels, activity = _group_inputs(
        group, device=resolved
    )
    main_output = model(mixture_batch, query_main)
    view2_output = model(mixture_batch, query_view2)
    return main_output, view2_output, labels, activity


def compute_group_loss(
    model: DACFCAMPPQueryMatcher,
    group: FeatureGroup,
    *,
    device: Optional[torch.device] = None,
) -> tuple[Tensor, dict[str, Tensor], tuple[CrossQueryOutput, CrossQueryOutput]]:
    """Compute the fixed four-term loss for one complete ABC group."""

    resolved = device or next(model.parameters()).device
    main_output, view2_output, labels, activity = _forward_group(
        model, group, device=resolved
    )
    activity_target = resize_activity_targets(activity, main_output.frame_logits.shape[1])
    presence_losses = []
    activity_losses = []
    margin_losses = []
    for output in (main_output, view2_output):
        presence_losses.append(class_balanced_bce(output.presence_logits, labels))
        activity_losses.append(
            F.binary_cross_entropy_with_logits(output.frame_logits, activity_target)
        )
        margin_losses.append(
            counterfactual_margin_loss(
                output.presence_logits.unsqueeze(0), margin=ABC_MARGIN
            )
        )
    consistency = 0.5 * (
        F.mse_loss(main_output.frame_logits, view2_output.frame_logits)
        + F.mse_loss(main_output.presence_logits, view2_output.presence_logits)
    )
    components = {
        "presence_bce": 0.5 * (presence_losses[0] + presence_losses[1]),
        "activity_bce": 0.5 * (activity_losses[0] + activity_losses[1]),
        "abc_margin": 0.5 * (margin_losses[0] + margin_losses[1]),
        "view_logits_consistency": consistency,
    }
    total = sum(
        LOSS_WEIGHTS[name] * components[name] for name in LOSS_WEIGHTS
    )
    components["total"] = total
    return total, components, (main_output, view2_output)


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Tie-aware probability that a positive score outranks a negative score."""

    if not positive or not negative:
        raise ValueError("ROC AUC requires both positive and negative scores")
    scores = np.asarray([*negative, *positive], dtype=np.float64)
    labels = np.asarray(
        [False] * len(negative) + [True] * len(positive), dtype=np.bool_
    )
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        value = scores[order[start]]
        while end < order.size and scores[order[end]] == value:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    rank_sum = float(ranks[labels].sum())
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def _binary_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positive = [float(score) for score, label in zip(scores, labels) if label >= 0.5]
    negative = [float(score) for score, label in zip(scores, labels) if label < 0.5]
    return _roc_auc(positive, negative)


def _bootstrap_group_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Cluster bootstrap by byte-identical A/B/C mixture group."""

    if not records:
        raise ValueError("group bootstrap requires at least one group")
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auc": [],
        "present_recall": [],
        "absent_rr": [],
        "query_response_mean": [],
        "activity_auc_group_mean": [],
    }
    for _ in range(samples):
        sampled = rng.integers(0, len(records), size=len(records))
        chosen = [records[int(index)] for index in sampled]
        present = [
            float(score) for record in chosen for score in record["present"]
        ]
        absent = [float(record["absent"]) for record in chosen]
        values["auc"].append(_roc_auc(present, absent))
        values["present_recall"].append(
            float(np.mean([score >= FIXED_PRESENCE_THRESHOLD for score in present]))
        )
        values["absent_rr"].append(
            float(np.mean([score < FIXED_PRESENCE_THRESHOLD for score in absent]))
        )
        values["query_response_mean"].append(
            float(np.mean([float(record["response"]) for record in chosen]))
        )
        values["activity_auc_group_mean"].append(
            float(np.mean([float(record["activity_auc"]) for record in chosen]))
        )
    return {
        "unit": "base_mixture_id",
        "samples": samples,
        "seed": seed,
        "ci95": {
            name: [
                float(np.percentile(metric_values, 2.5)),
                float(np.percentile(metric_values, 97.5)),
            ]
            for name, metric_values in values.items()
        },
    }


def _evaluate_groups(
    model: DACFCAMPPQueryMatcher,
    groups: Sequence[FeatureGroup],
    *,
    device: Optional[torch.device] = None,
    bootstrap_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Evaluate one split at the preregistered threshold for both views."""

    resolved = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    view_scores: dict[str, dict[str, list[float]]] = {
        "main": {
            "present": [],
            "absent": [],
            "activity_scores": [],
            "activity_labels": [],
            "responses": [],
            "c_lower_both": [],
            "group_records": [],
        },
        "view2": {
            "present": [],
            "absent": [],
            "activity_scores": [],
            "activity_labels": [],
            "responses": [],
            "c_lower_both": [],
            "group_records": [],
        },
    }
    try:
        with torch.no_grad():
            for group in groups:
                main, view2, labels, activity = _forward_group(
                    model, group, device=resolved
                )
                target_activity = resize_activity_targets(
                    activity, main.frame_logits.shape[1]
                )
                for view_name, output in (("main", main), ("view2", view2)):
                    scores = torch.sigmoid(output.presence_logits).detach().cpu().tolist()
                    frame_scores = (
                        torch.sigmoid(output.frame_logits).detach().cpu().reshape(-1).tolist()
                    )
                    frame_labels = target_activity.detach().cpu().reshape(-1).tolist()
                    bucket = view_scores[view_name]
                    bucket["present"].extend([float(scores[0]), float(scores[1])])
                    bucket["absent"].append(float(scores[2]))
                    bucket["activity_scores"].extend(float(value) for value in frame_scores)
                    bucket["activity_labels"].extend(float(value) for value in frame_labels)
                    response = min(
                        float(scores[0]) - float(scores[2]),
                        float(scores[1]) - float(scores[2]),
                    )
                    bucket["responses"].append(response)
                    bucket["c_lower_both"].append(
                        float(scores[2] < scores[0] and scores[2] < scores[1])
                    )
                    bucket["group_records"].append(
                        {
                            "present": [float(scores[0]), float(scores[1])],
                            "absent": float(scores[2]),
                            "response": response,
                            "activity_auc": _binary_auc(frame_scores, frame_labels),
                        }
                    )
    finally:
        model.train(was_training)

    metrics: dict[str, Any] = {}
    for view_name, bucket in view_scores.items():
        present = bucket["present"]
        absent = bucket["absent"]
        auc = _roc_auc(present, absent)
        activity_auc = _binary_auc(
            bucket["activity_scores"], bucket["activity_labels"]
        )
        metrics[view_name] = {
            "groups": len(groups),
            "present_queries": len(present),
            "absent_queries": len(absent),
            "fixed_presence_threshold": FIXED_PRESENCE_THRESHOLD,
            "auc": auc,
            "roc_auc": auc,
            "present_recall": float(
                np.mean([score >= FIXED_PRESENCE_THRESHOLD for score in present])
            ),
            "absent_rr": float(
                np.mean([score < FIXED_PRESENCE_THRESHOLD for score in absent])
            ),
            "query_response_mean": float(np.mean(bucket["responses"])),
            "query_response_min_margin_mean": float(np.mean(bucket["responses"])),
            "c_lower_both_ratio": float(np.mean(bucket["c_lower_both"])),
            "activity_auc": activity_auc,
            "present_probability_mean": float(np.mean(present)),
            "absent_probability_mean": float(np.mean(absent)),
            "group_bootstrap": _bootstrap_group_metrics(
                bucket["group_records"],
                seed=bootstrap_seed + (0 if view_name == "main" else 1),
            ),
        }
    return metrics


def _conditional_gate(metrics: Mapping[str, Any]) -> tuple[bool, dict[str, bool]]:
    """Apply the fixed single-view gate without any threshold selection."""

    auc = float(metrics["auc"] if "auc" in metrics else metrics["roc_auc"])
    query_response = float(
        metrics[
            "query_response_mean"
            if "query_response_mean" in metrics
            else "query_response_min_margin_mean"
        ]
    )
    checks = {
        "auc": auc >= GATE_THRESHOLDS["auc"],
        "present_recall": float(metrics["present_recall"])
        >= GATE_THRESHOLDS["present_recall"],
        "absent_rr": float(metrics["absent_rr"])
        >= GATE_THRESHOLDS["absent_rr"],
        "query_response_mean": query_response >= GATE_THRESHOLDS["query_response_mean"],
        "activity_auc": float(metrics["activity_auc"])
        >= GATE_THRESHOLDS["activity_auc"],
    }
    return all(checks.values()), checks


def _two_view_gate(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    by_view: dict[str, Any] = {}
    for view_name in ("main", "view2"):
        passed, checks = _conditional_gate(metrics[view_name])
        by_view[view_name] = {"passed": passed, "checks": checks}
    return all(item["passed"] for item in by_view.values()), by_view


def _metric_means(history: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not history:
        return {}
    return {
        name: float(np.mean([float(item[name]) for item in history]))
        for name in history[0]
    }


def run_probe(
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    *,
    manifest: str | Path | None = None,
    updates: int = DEFAULT_UPDATES,
    seed: int = DEFAULT_SEED,
    device: str = "auto",
    learning_rate: float = 3e-3,
    strict_counts: bool = True,
    output_json: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train the fixed mini-G2 matcher and write the auditable result."""

    if isinstance(updates, bool) or int(updates) != updates:
        raise ValueError(f"updates must be an integer in [1, {MAX_UPDATES}]")
    updates = int(updates)
    if updates < 1 or updates > MAX_UPDATES:
        raise ValueError(f"updates must be in [1, {MAX_UPDATES}], got {updates}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if isinstance(cache_root, (str, Path)) and Path(cache_root).is_file():
        if manifest is not None:
            raise ValueError("pass either cache_root directory or manifest, not both")
        manifest = cache_root
        cache_root = Path(cache_root).parent
    started = time.perf_counter()
    cache = _validate_and_load_cache(
        cache_root,
        manifest=manifest,
        strict_counts=strict_counts,
    )
    resolved_device = _resolve_device(device)
    _seed_everything(seed)

    model = DACFCAMPPQueryMatcher(
        feature_dim=FEATURE_DIM,
        query_dim=QUERY_DIM,
        logit_scale=LOGIT_SCALE,
        top_fraction=TOP_FRACTION,
    ).to(resolved_device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters >= PARAMETER_LIMIT:
        raise RuntimeError(
            f"low-rank matcher has {trainable_parameters} trainable parameters, "
            f"limit is < {PARAMETER_LIMIT}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=OPTIMIZER_WEIGHT_DECAY
    )
    order = list(range(len(cache.train_groups)))
    rng = random.Random(seed)
    loss_history: list[float] = []
    component_history: list[dict[str, float]] = []
    for update in range(updates):
        if update % len(order) == 0:
            rng.shuffle(order)
        group = cache.train_groups[order[update % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        loss, components, _ = compute_group_loss(
            model, group, device=resolved_device
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite loss at update {update}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))
        component_history.append(
            {
                name: float(value.detach().cpu())
                for name, value in components.items()
            }
        )

    train_metrics = _evaluate_groups(
        model,
        cache.train_groups,
        device=resolved_device,
        bootstrap_seed=seed + 10_000,
    )
    val_metrics = _evaluate_groups(
        model,
        cache.val_groups,
        device=resolved_device,
        bootstrap_seed=seed + 20_000,
    )
    final_metrics = (
        _evaluate_groups(
            model,
            cache.final_groups,
            device=resolved_device,
            bootstrap_seed=seed + 30_000,
        )
        if cache.final_groups
        else None
    )
    decision_split = "final" if final_metrics is not None else "val"
    decision_metrics = final_metrics if final_metrics is not None else val_metrics
    passed, gate = _two_view_gate(decision_metrics)

    output_path = Path(output_json) if output_json is not None else cache.root / "matcher_result.json"
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint is not None
        else cache.root / "matcher_checkpoint.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "schema": "dacf-campp-cross-query-checkpoint-v0.1",
        "state_dict": model.state_dict(),
        "config": {
            "seed": seed,
            "updates": updates,
            "learning_rate": learning_rate,
            "query_dim": QUERY_DIM,
            "logit_scale": LOGIT_SCALE,
            "top_fraction": TOP_FRACTION,
            "fixed_presence_threshold": FIXED_PRESENCE_THRESHOLD,
            "abc_margin": ABC_MARGIN,
            "loss_weights": dict(LOSS_WEIGHTS),
        },
        "manifest_sha256": cache.audit["manifest_sha256"],
        "cache_sha256": cache.audit["cache_sha256"],
        "train_groups": len(cache.train_groups),
        "val_groups": len(cache.val_groups),
        "final_groups": len(cache.final_groups),
        "dataset_a_used": False,
        "trainable_parameter_count": trainable_parameters,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    result: dict[str, Any] = {
        "schema": "dacf-campp-cross-query-report-v0.2",
        "verdict": "conditional-GO" if passed else "implementation-NO-GO",
        "verdict_scope": f"fixed matcher mechanism gate on {decision_split} split only",
        "dataset_a_used": False,
        "cer_measured": False,
        "hard_negative_verified": False,
        "threshold_tuned": False,
        "val_used_for_selection": False,
        "final_used_for_selection": False,
        "decision_split": decision_split,
        "manifest_sha256": cache.audit["manifest_sha256"],
        "cache_sha256": cache.audit["cache_sha256"],
        "cache_sha256_scope": cache.audit["cache_sha256_scope"],
        "cache_root": cache.root.as_posix(),
        "manifest": cache.manifest_path.as_posix(),
        "seed": seed,
        "updates": updates,
        "learning_rate": learning_rate,
        "device": str(resolved_device),
        "config": {
            "query_dim": QUERY_DIM,
            "logit_scale": LOGIT_SCALE,
            "top_fraction": TOP_FRACTION,
            "fixed_presence_threshold": FIXED_PRESENCE_THRESHOLD,
            "abc_margin": ABC_MARGIN,
            "loss_weights": dict(LOSS_WEIGHTS),
            "optimizer": "AdamW",
            "optimizer_weight_decay": OPTIMIZER_WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
        },
        "model": {
            "class": "DACFCAMPPQueryMatcher",
            "input_features": ["mixture_tokens", "enrollment_embedding"],
            "query_role_id_used_as_model_input": False,
            "campp_parameters_unfrozen": False,
            "trainable_parameter_count": trainable_parameters,
            "parameter_limit": PARAMETER_LIMIT,
            "parameter_limit_passed": trainable_parameters < PARAMETER_LIMIT,
        },
        "audit": dict(cache.audit),
        "train": train_metrics,
        "val": val_metrics,
        **({"final": final_metrics} if final_metrics is not None else {}),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "gate": gate,
        "both_views_passed": passed,
        "train_loss_first": loss_history[0],
        "train_loss_last": loss_history[-1],
        "train_loss_tail_mean": float(np.mean(loss_history[-min(24, len(loss_history)) :])),
        "train_loss_component_means": _metric_means(component_history),
        "artifacts": {
            "result_json": output_path.resolve(strict=False).as_posix(),
            "checkpoint": checkpoint_path.resolve(strict=False).as_posix(),
        },
        "limitations": [
            "Dataset-A was false and not read; it is forbidden for this probe.",
            "CER was not measured and the submission chain was not used.",
            "A verified home-command hard-negative set was not available or validated.",
            f"The fixed {decision_split} gate is a mechanism gate, not a CER/RR/RTF integration gate.",
        ],
        "runtime_sec": float(time.perf_counter() - started),
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--allow-nonfixed-counts",
        action="store_true",
        help="allow small synthetic fixtures instead of the fixed 24/8-group cache",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_probe(
        args.cache_root,
        manifest=args.manifest,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        strict_counts=not args.allow_nonfixed_counts,
        output_json=args.output_json,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABC_MARGIN",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_SEED",
    "DEFAULT_UPDATES",
    "FIXED_PRESENCE_THRESHOLD",
    "GATE_THRESHOLDS",
    "LOSS_WEIGHTS",
    "MAX_UPDATES",
    "ROLE_ORDER",
    "FeatureGroup",
    "LoadedCache",
    "QueryFeature",
    "_conditional_gate",
    "_evaluate_groups",
    "_forward_group",
    "_input_cache_sha256",
    "_roc_auc",
    "_validate_and_load_cache",
    "compute_group_loss",
    "run_probe",
]
