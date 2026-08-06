"""Preregister and train the DACF-v4b cross-environment all-pairs mechanism.

DACF-v4b is the corrected membership test for the falsified v3/v4 line.

The earlier ``falsify frozen CAM++ DACF matcher`` and ``falsify ECST identity
front end`` results both relied on a *single-environment-per-speaker* cache:
each optimised enrollment appeared in exactly one mixture, so a query-only
speaker-to-label lookup could minimise the objective without ever reading the
mixture.  Commit ``a36d280`` exposed that identity gap.  v4b removes the
shortcut by reusing every speaker across six counterfactual environments
(2A+2B+2C role rotation).  A speaker is therefore *present* in several
mixtures and *absent* in several others, and the only honest way to score
correctly is to read the mixture acoustics.

This trainer consumes the provenance-locked ``dacf-v4b-feature-cache-v0.1``
cache (96 train / 24 dev AISHELL-1 groups, speaker-disjoint across splits).
It reuses the v4 relation model and objective unchanged -- both are
environment-agnostic by construction -- but replaces the v4 batch builder,
which hard-coded the three single-environment invariants that v4b exists to
break ("unique speakers across groups", "positive exactly once", "foreign
everywhere else").

The v4b bank is de-duplicated by speaker: one enrollment identity per speaker,
scored against every mixture in the batch.  ``presence`` is recomputed as
``query speaker in destination mixture_speakers`` for every (mixture, query)
pair, so a query may be positive for several mixtures in the same update.

Discipline is inherited from v4: preregistration locks the contract before
training; the trainable head stays under 2M parameters; the checkpoint is
frozen before the single dev observation and SHA-verified after; Dataset-A is
forbidden; the official final split is never read; no CER / RR / RTF / Qwen
decoding is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dacf_v4_objective import DACFV4LossWeights
from dacf_v4b_objective import compute_dacf_v4b_loss
from dacf_v4_relation import CAMPP_EMBEDDING_DIM, MIXTURE_MEL_BINS, DACFV4Relation


PROTOCOL_VERSION = "dacf-v4b-allpairs-mechanism-v0.1"
PREREGISTRATION_SCHEMA = "dacf-v4b-allpairs-preregistration-v0.1"
REPORT_SCHEMA = "dacf-v4b-allpairs-report-v0.1"
CACHE_REPORT_SCHEMA = "dacf-v4b-feature-cache-report-v0.1"
CACHE_SCHEMA = "dacf-v4b-feature-cache-v0.1"
THRESHOLD = 0.5

# Same mechanism gate as v4 minus the two single-role recalls (present_A/B
# recall do not survive speaker de-duplication, since a bank query has no one
# role).  query_to_mixture_top1 is redefined for multi-positive membership:
# a query's top mixture is "correct" iff that mixture actually contains it.
MECHANISM_GATE: Mapping[str, float] = {
    "presence_auc": 0.85,
    "activity_frame_auc": 0.80,
    "present_recall": 0.80,
    "foreign_rr": 0.95,
    "mixture_top2_exact": 0.75,
    "query_to_mixture_top1": 0.75,
    "query_response_mean": 0.20,
    "query_response_ci_lower": 0.05,
    "mixture_permutation_auc_drop": 0.20,
    "collapsed_query_auc_max": 0.55,
    "collapsed_mixture_auc_max": 0.55,
}

MEL_BINS = MIXTURE_MEL_BINS
CAMPP_DIM = CAMPP_EMBEDDING_DIM
EXPECTED_TRAIN_GROUPS = 96
EXPECTED_DEV_GROUPS = 24


class MechanismContractError(RuntimeError):
    """Raised when the cache, pairing, or training violates the v4b contract."""


# --------------------------------------------------------------------------- #
# Provenance data structures (mirror the v3 shapes; local to keep v4b decoupled
# from the v3 training module's import chain).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QueryRecord:
    row_id: str
    group_id: str
    role: str
    speaker_id: str
    target_present: bool
    embedding: np.ndarray  # CAM++ view1, L2-normalised [512]
    embedding_view2: np.ndarray  # CAM++ view2, L2-normalised [512]
    target_activity: np.ndarray  # per-frame activity in the BASE mixture [T]


@dataclass(frozen=True)
class CounterfactualGroup:
    split: str
    group_id: str
    mixture_feature_path: Path
    mixture_feature_sha256: str
    mixture_features: np.ndarray  # Qwen log-mel [128, T]
    mixture_speaker_ids: tuple[str, ...]  # the two present speakers (A, B)
    rows: tuple[QueryRecord, QueryRecord, QueryRecord]  # A present, B present, C absent


@dataclass(frozen=True)
class FeatureCache:
    root: Path
    report: Mapping[str, Any]
    groups: Mapping[str, tuple[CounterfactualGroup, ...]]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    # Must match build_dacf_v3_features._sha256_array byte-for-byte: the digest
    # covers dtype + shape + NUL + raw bytes, not just the payload.
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scalar_text(value: Any, field: str) -> str:
    if isinstance(value, bytes):
        raise MechanismContractError(f"{field} must be text, got bytes")
    return str(value)


def _device_for(config: "DACFV4BTrainingConfig") -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DACFV4BTrainingConfig:
    seed: int = 2026080617
    epochs: int = 30
    groups_per_batch: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    hard_foreign_margin: float = 0.20
    bootstrap_replicates: int = 2000
    device: str = "auto"

    def validate(self, train_group_count: int) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("epochs", "groups_per_batch", "bootstrap_replicates"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.groups_per_batch < 2:
            raise ValueError("groups_per_batch must be at least two")
        if train_group_count % self.groups_per_batch:
            raise ValueError(
                f"train group count ({train_group_count}) must divide evenly "
                f"into fixed batches of {self.groups_per_batch}"
            )
        for name in (
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
            "hard_foreign_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate <= 0 or self.grad_clip_norm <= 0:
            raise ValueError("learning rate and gradient clip norm must be positive")

    @property
    def optimiser_updates(self) -> int:
        # The formal v4b cache is fixed to 96 train groups.
        return self.epochs * (EXPECTED_TRAIN_GROUPS // self.groups_per_batch)


# --------------------------------------------------------------------------- #
# Cache loading
# --------------------------------------------------------------------------- #


def _validate_mixture_npz(path: Path, row: Mapping[str, Any]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "input_features",
            "feature_attention_mask",
            "mixture_audio_sha256",
            "qwen_config_sha256",
            "qwen_feature_spec_sha256",
            "input_features_sha256",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise MechanismContractError(f"mixture NPZ lacks {missing}: {path}")
        features = np.asarray(data["input_features"])
        if features.ndim != 2 or features.shape[0] != MEL_BINS or features.shape[1] < 1:
            raise MechanismContractError(
                f"invalid mixture input_features shape: {features.shape}"
            )
        if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
            raise MechanismContractError("mixture input_features must be finite floating point")
        attention = np.asarray(data["feature_attention_mask"])
        if attention.shape != (features.shape[1],) or not np.isfinite(attention).all():
            raise MechanismContractError("mixture feature_attention_mask shape/value mismatch")
        if np.any((attention != 0) & (attention != 1)):
            raise MechanismContractError("mixture feature_attention_mask must be binary")
        if not np.all(attention == 1):
            raise MechanismContractError(
                "v4b relation head consumes no attention mask; all cached frames must be valid"
            )
        if _scalar_text(data["mixture_audio_sha256"], "mixture_audio_sha256").casefold() != str(
            row["mixture_sha256"]
        ).casefold():
            raise MechanismContractError("mixture audio SHA metadata mismatch")
        if _sha256_array(features) != str(row["mixture_input_features_sha256"]):
            raise MechanismContractError("mixture feature array SHA mismatch")
    return np.asarray(features, dtype=np.float32)


def _validate_query_npz(
    path: Path,
    row: Mapping[str, Any],
    mixture_frames: int,
) -> QueryRecord:
    with np.load(path, allow_pickle=False) as data:
        forbidden = {"query_role_id"}
        present_forbidden = sorted(forbidden & set(data.files))
        if present_forbidden:
            raise MechanismContractError(
                f"query NPZ contains forbidden label-id keys {present_forbidden}: {path}"
            )
        required = {
            "enrollment_embedding",
            "enrollment_embedding_view2",
            "target_activity",
            "row_id",
            "split",
            "base_mixture_id",
            "query_role",
            "query_speaker_id",
            "mixture_speakers_json",
            "mixture_feature_sha256",
            "target_activity_array_sha256",
            "enrollment_embedding_sha256",
            "enrollment_embedding_view2_sha256",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise MechanismContractError(f"query NPZ lacks {missing}: {path}")

        if _scalar_text(data["row_id"], "row_id") != str(row["row_id"]):
            raise MechanismContractError(f"query row_id mismatch: {path}")
        if _scalar_text(data["split"], "split") != str(row["split"]):
            raise MechanismContractError(f"query split mismatch: {path}")
        if _scalar_text(data["base_mixture_id"], "base_mixture_id") != str(row["base_mixture_id"]):
            raise MechanismContractError(f"query base_mixture_id mismatch: {path}")
        if _scalar_text(data["query_role"], "query_role") != str(row["query_role"]):
            raise MechanismContractError(f"query query_role mismatch: {path}")
        if _scalar_text(data["query_speaker_id"], "query_speaker_id") != str(row["query_speaker_id"]):
            raise MechanismContractError(f"query query_speaker_id mismatch: {path}")

        embedding = np.asarray(data["enrollment_embedding"], dtype=np.float32)
        embedding_view2 = np.asarray(data["enrollment_embedding_view2"], dtype=np.float32)
        for name, value in (
            ("enrollment_embedding", embedding),
            ("enrollment_embedding_view2", embedding_view2),
        ):
            if value.shape != (CAMPP_DIM,) or not np.isfinite(value).all():
                raise MechanismContractError(f"{name} must be finite with shape (512,): {path}")
            if abs(float(np.linalg.norm(value)) - 1.0) > 2e-5:
                raise MechanismContractError(f"{name} must be L2-normalised: {path}")
            expected_hash = str(
                row.get(
                    "enrollment_embedding_sha256"
                    if name == "enrollment_embedding"
                    else "enrollment_embedding_view2_sha256",
                    "",
                )
            )
            if expected_hash and _sha256_array(value) != expected_hash:
                raise MechanismContractError(f"{name} array SHA mismatch: {path}")

        activity = np.asarray(data["target_activity"], dtype=np.float32)
        if activity.shape != (mixture_frames,) or not np.isfinite(activity).all():
            raise MechanismContractError(
                f"target_activity shape mismatch (got {activity.shape}, want ({mixture_frames},)): {path}"
            )
        if np.any((activity < 0.0) | (activity > 1.0)):
            raise MechanismContractError("target_activity must be in [0,1]")
        if _sha256_array(activity) != str(row["target_activity_array_sha256"]):
            raise MechanismContractError("query target_activity array SHA mismatch")

        # membership label recomputed from the destination mixture, never read
        # from a stored boolean (label_contract: presence = query_speaker_id in
        # destination mixture_speakers).
        speakers = json.loads(_scalar_text(data["mixture_speakers_json"], "mixture_speakers_json"))
        speaker_set = set(speakers.values()) if isinstance(speakers, dict) else set(speakers)
        query_speaker = str(row["query_speaker_id"])
        target_present = query_speaker in speaker_set

        # An absent query (role C) carries no activity; a present query must.
        if not target_present and np.any(activity != 0.0):
            raise MechanismContractError(
                f"absent query carries nonzero activity: {path}"
            )

    return QueryRecord(
        row_id=str(row["row_id"]),
        group_id=str(row["base_mixture_id"]),
        role=str(row["query_role"]),
        speaker_id=query_speaker,
        target_present=target_present,
        embedding=embedding,
        embedding_view2=embedding_view2,
        target_activity=activity,
    )


def _load_manifest(cache_root: Path) -> list[Mapping[str, Any]]:
    manifest_path = cache_root / "features_manifest.jsonl"
    rows: list[Mapping[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_v4b_feature_cache(cache_root: str | Path) -> FeatureCache:
    root = Path(cache_root).resolve()
    report_path = root / "cache_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != CACHE_REPORT_SCHEMA:
        raise MechanismContractError(
            f"cache report schema is {report.get('schema')!r}, expected {CACHE_REPORT_SCHEMA!r}"
        )
    if report.get("cache_schema") != CACHE_SCHEMA:
        raise MechanismContractError(
            f"cache schema is {report.get('cache_schema')!r}, expected {CACHE_SCHEMA!r}"
        )
    if bool(report.get("dataset_a_used", True)):
        raise MechanismContractError("Dataset-A is forbidden")
    if not bool(report.get("final_deferred", False)):
        raise MechanismContractError("v4b requires a train/dev cache with final deferred")
    loaded = list(report.get("loaded_splits", []))
    if loaded != ["train", "dev"]:
        raise MechanismContractError(f"loaded_splits must bind train/dev only, got {loaded}")
    if str(report.get("source_corpus", "")) != "AISHELL-1":
        raise MechanismContractError("v4b cache must be sourced from AISHELL-1")

    rows = _load_manifest(root)
    for index, row in enumerate(rows):
        if row.get("cache_schema") != CACHE_SCHEMA:
            raise MechanismContractError(f"manifest row {index} has wrong cache_schema")
        if bool(row.get("dataset_a_used", False)):
            raise MechanismContractError(f"manifest row {index} violates Dataset-A firewall")
        if row.get("query_role_id_used_as_model_input") not in (False, None):
            raise MechanismContractError(
                f"manifest row {index} leaks query_role_id into model input"
            )

    # group manifest rows by (split, base_mixture_id); each group has 3 rows.
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["split"]), str(row["base_mixture_id"]))
        grouped.setdefault(key, []).append(row)

    groups_by_split: dict[str, list[CounterfactualGroup]] = {"train": [], "dev": []}
    for (split, group_id), group_rows in sorted(grouped.items()):
        if split not in groups_by_split:
            raise MechanismContractError(f"unexpected split {split!r}")
        if len(group_rows) != 3:
            raise MechanismContractError(
                f"{split}/{group_id} must have exactly three query rows, got {len(group_rows)}"
            )
        roles = sorted(str(r["query_role"]) for r in group_rows)
        if roles != ["absent_C", "present_A", "present_B"]:
            raise MechanismContractError(
                f"{split}/{group_id} role rotation must be A/B/C, got {roles}"
            )

        mixture_path = root / str(group_rows[0]["mixture_feature"])
        mixture_features = _validate_mixture_npz(mixture_path, group_rows[0])
        mixture_frames = int(mixture_features.shape[1])

        query_records: list[QueryRecord] = []
        for row in group_rows:
            query_path = root / str(row["query_feature"])
            query_records.append(_validate_query_npz(query_path, row, mixture_frames))

        present_rows = [r for r in query_records if r.target_present]
        if len(present_rows) != 2:
            raise MechanismContractError(
                f"{split}/{group_id} must have exactly two present speakers"
            )
        mixture_speaker_ids = tuple(sorted(r.speaker_id for r in present_rows))
        # order rows as A present, B present, C absent for a stable contract
        ordered = tuple(
            sorted(query_records, key=lambda r: (0 if r.target_present else 1, r.role))
        )
        groups_by_split[split].append(
            CounterfactualGroup(
                split=split,
                group_id=group_id,
                mixture_feature_path=mixture_path,
                mixture_feature_sha256=str(group_rows[0]["mixture_feature_sha256"]),
                mixture_features=mixture_features,
                mixture_speaker_ids=mixture_speaker_ids,
                rows=ordered,  # type: ignore[arg-type]
            )
        )

    for split in ("train", "dev"):
        if not groups_by_split[split]:
            raise MechanismContractError(f"cache has no {split} groups")

    return FeatureCache(
        root=root,
        report=report,
        groups={
            "train": tuple(groups_by_split["train"]),
            "dev": tuple(groups_by_split["dev"]),
        },
    )


# --------------------------------------------------------------------------- #
# Cache contract
# --------------------------------------------------------------------------- #


def _cache_contract(cache: FeatureCache) -> Mapping[str, Any]:
    if set(cache.groups) != {"train", "dev"}:
        raise MechanismContractError("v4b cache must expose only train and dev")
    train_groups = cache.groups["train"]
    dev_groups = cache.groups["dev"]
    if len(train_groups) != EXPECTED_TRAIN_GROUPS or len(dev_groups) != EXPECTED_DEV_GROUPS:
        raise MechanismContractError(
            f"formal v4b cache must contain {EXPECTED_TRAIN_GROUPS} train / "
            f"{EXPECTED_DEV_GROUPS} dev groups, got "
            f"{len(train_groups)}/{len(dev_groups)}"
        )

    # speaker-disjoint train/dev is the firewall that makes dev an honest test.
    train_speakers = {s for g in train_groups for s in g.mixture_speaker_ids}
    dev_speakers = {s for g in dev_groups for s in g.mixture_speaker_ids}
    overlap = sorted(train_speakers & dev_speakers)
    if overlap:
        raise MechanismContractError(
            f"train/dev speakers must be disjoint, overlap: {overlap[:5]}"
        )

    # Full-split construction proves the multi-positive label invariant before
    # either preregistration or training.
    train_batch = build_v4b_allpairs_batch(train_groups)
    dev_batch = build_v4b_allpairs_batch(dev_groups)

    positive_per_query_train = train_batch.presence_labels.sum(axis=0)
    positive_per_query_dev = dev_batch.presence_labels.sum(axis=0)
    return {
        "train_groups": len(train_groups),
        "dev_groups": len(dev_groups),
        "train_bank_speakers": len(train_batch.speaker_ids),
        "dev_bank_speakers": len(dev_batch.speaker_ids),
        "train_query_positive_min": int(positive_per_query_train.min()),
        "train_query_positive_max": int(positive_per_query_train.max()),
        "dev_query_positive_min": int(positive_per_query_dev.min()),
        "dev_query_positive_max": int(positive_per_query_dev.max()),
        # v4b deliberately allows >1; this records that the single-env shortcut
        # is gone (max >= 2 on the full split).
        "single_environment_shortcut_removed": bool(
            int(positive_per_query_train.max()) >= 2
            or int(positive_per_query_dev.max()) >= 2
        ),
        "c_only_queries_optimised": 0,
        "final_deferred": True,
        "dataset_a_used": False,
    }


def _source_hashes() -> Mapping[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        root / "code" / "experiments" / "dacf_v4_relation.py",
        root / "code" / "experiments" / "dacf_v4_objective.py",
        root / "code" / "experiments" / "dacf_v4b_objective.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in paths
    }


# --------------------------------------------------------------------------- #
# All-pairs batching (v4b: speaker-de-duplicated bank, multi-positive labels)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class V4BAllPairsBatch:
    groups: tuple[CounterfactualGroup, ...]
    speaker_ids: tuple[str, ...]  # de-duplicated bank, one identity per speaker
    source_group_ids: tuple[str, ...]  # group_id each bank query first came from
    embeddings_view1: np.ndarray  # [Q, 512]
    embeddings_view2: np.ndarray  # [Q, 512]
    presence_labels: np.ndarray  # [G, Q] -- 1 iff bank speaker in that mixture
    activity_targets: tuple[np.ndarray, ...]  # per mixture [Q, T]


def _present_rows(group: CounterfactualGroup) -> tuple[QueryRecord, QueryRecord]:
    rows = tuple(row for row in group.rows if row.target_present)
    if len(rows) != 2:
        raise MechanismContractError(
            f"{group.split}/{group.group_id} must have exactly two present rows"
        )
    if {row.speaker_id for row in rows} != set(group.mixture_speaker_ids):
        raise MechanismContractError(
            f"{group.split}/{group.group_id} present speakers disagree with mixture"
        )
    return rows


def build_v4b_allpairs_batch(groups: Sequence[CounterfactualGroup]) -> V4BAllPairsBatch:
    """Build a speaker-de-duplicated bank with multi-positive presence labels.

    Unlike the v4 builder, the same speaker may appear in several mixtures.
    The bank keeps one enrolment identity per speaker; ``labels[mixture, query]``
    is 1 iff that speaker is one of the mixture's two present speakers, so a
    query can be positive for multiple mixtures in the same batch.
    """

    selected = tuple(groups)
    if len(selected) < 2:
        raise MechanismContractError("all-pairs batch needs at least two mixtures")
    split_names = {group.split for group in selected}
    if len(split_names) != 1:
        raise MechanismContractError("all-pairs batch cannot cross train/dev splits")
    group_ids = [group.group_id for group in selected]
    if len(group_ids) != len(set(group_ids)):
        raise MechanismContractError("all-pairs batch repeats a mixture group")

    # De-duplicate by speaker, keeping the first enrolment row seen.  Different
    # present rows of the same speaker share the same enrolment identity; the
    # per-mixture activity target is still looked up from each mixture's own
    # rows below, so de-duplication only collapses the scored embedding.
    bank_order: list[str] = []
    bank_row: dict[str, QueryRecord] = {}
    bank_source: list[str] = []
    for group in selected:
        for row in _present_rows(group):
            if row.speaker_id not in bank_row:
                bank_row[row.speaker_id] = row
                bank_order.append(row.speaker_id)
                bank_source.append(group.group_id)
    if not bank_order:
        raise MechanismContractError("all-pairs batch produced an empty query bank")

    num_groups = len(selected)
    num_queries = len(bank_order)
    labels = np.zeros((num_groups, num_queries), dtype=np.float32)
    activity_targets: list[np.ndarray] = []
    for group_index, group in enumerate(selected):
        activity_by_speaker = {
            row.speaker_id: np.asarray(row.target_activity, dtype=np.float32)
            for row in _present_rows(group)
        }
        frames = int(group.mixture_features.shape[1])
        target = np.zeros((num_queries, frames), dtype=np.float32)
        for query_index, speaker in enumerate(bank_order):
            if speaker in activity_by_speaker:
                labels[group_index, query_index] = 1.0
                source_activity = activity_by_speaker[speaker]
                if source_activity.shape != (frames,):
                    raise MechanismContractError(
                        f"{group.split}/{group.group_id} activity timeline disagrees with mixture"
                    )
                target[query_index] = source_activity
        positive_count = int(labels[group_index].sum())
        negative_count = num_queries - positive_count
        if positive_count < 1:
            raise MechanismContractError(
                f"{group.split}/{group.group_id} has no positive query in the bank"
            )
        if negative_count < 1:
            raise MechanismContractError(
                f"{group.split}/{group.group_id} has no foreign query in the bank"
            )
        activity_targets.append(target)

    embeddings_view1 = np.stack(
        [bank_row[speaker].embedding for speaker in bank_order]
    ).astype(np.float32, copy=False)
    embeddings_view2 = np.stack(
        [bank_row[speaker].embedding_view2 for speaker in bank_order]
    ).astype(np.float32, copy=False)

    return V4BAllPairsBatch(
        groups=selected,
        speaker_ids=tuple(bank_order),
        source_group_ids=tuple(bank_source),
        embeddings_view1=embeddings_view1,
        embeddings_view2=embeddings_view2,
        presence_labels=labels,
        activity_targets=tuple(activity_targets),
    )


def epoch_group_batches(
    groups: Sequence[CounterfactualGroup],
    *,
    groups_per_batch: int,
    seed: int,
    epoch: int,
) -> tuple[tuple[CounterfactualGroup, ...], ...]:
    ordered = list(groups)
    random.Random(seed + 1_000_003 * epoch).shuffle(ordered)
    if len(ordered) % groups_per_batch:
        raise MechanismContractError("fixed all-pairs schedule cannot emit a short batch")
    return tuple(
        tuple(ordered[index : index + groups_per_batch])
        for index in range(0, len(ordered), groups_per_batch)
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    truth = np.asarray(labels, dtype=np.bool_).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(truth.sum())
    negative = int((~truth).sum())
    if positive < 1 or negative < 1:
        raise MechanismContractError("AUC requires both classes")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        stop = cursor + 1
        while stop < values.size and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = 0.5 * ((cursor + 1) + stop)
        ranks[order[cursor:stop]] = rank
        cursor = stop
    rank_sum = float(ranks[truth].sum())
    return (rank_sum - positive * (positive + 1) / 2.0) / (positive * negative)


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Mapping[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.isfinite(data).all():
        raise MechanismContractError("mean bootstrap needs at least two finite values")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, data.size, size=data.size)
        draws[index] = float(data[sample].mean())
    return {
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
    }


def _evaluate_view(
    model: DACFV4Relation,
    batch: V4BAllPairsBatch,
    *,
    embeddings: np.ndarray,
    device: torch.device,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> Mapping[str, Any]:
    model.eval()
    bank = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
    # Query-collapse bank: every position becomes the mean enrolment, so a
    # query-only shortcut cannot separate present from foreign within a mixture
    # and the AUC must fall back to ~0.5.
    collapsed_bank = bank.mean(dim=0, keepdim=True).expand_as(bank).contiguous()
    mixture_count = len(batch.groups)
    query_count = len(batch.speaker_ids)
    probabilities = np.empty((mixture_count, query_count), dtype=np.float64)
    collapsed_probabilities = np.empty_like(probabilities)
    activity_scores: list[np.ndarray] = []
    activity_labels: list[np.ndarray] = []

    with torch.no_grad():
        for mixture_index, group in enumerate(batch.groups):
            mixture = torch.from_numpy(group.mixture_features).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            state = model.encode_mixture(mixture)
            out = model.score_queries(state, bank)
            collapsed_out = model.score_queries(state, collapsed_bank)
            probabilities[mixture_index] = torch.sigmoid(
                out.presence_logits[0]
            ).cpu().numpy()
            collapsed_probabilities[mixture_index] = torch.sigmoid(
                collapsed_out.presence_logits[0]
            ).cpu().numpy()
            positive = batch.presence_labels[mixture_index] > 0.5
            activity_scores.append(
                out.activity_probability[0, positive].cpu().numpy().reshape(-1)
            )
            activity_labels.append(
                batch.activity_targets[mixture_index][positive].reshape(-1)
            )

    labels = batch.presence_labels.astype(bool)
    presence_auc = _binary_auc(labels, probabilities)
    collapsed_query_auc = _binary_auc(labels, collapsed_probabilities)
    # Mixture-collapse: replace each query's across-mixture scores with that
    # query's mean.  A model that ignores the mixture survives this; the role
    # rotation requires it to fall back to ~0.5.
    collapsed_mixture_probabilities = np.broadcast_to(
        probabilities.mean(axis=0, keepdims=True), probabilities.shape
    )
    collapsed_mixture_auc = _binary_auc(labels, collapsed_mixture_probabilities)
    permutation_auc = _binary_auc(labels, np.roll(probabilities, shift=1, axis=0))
    predictions = probabilities >= THRESHOLD
    present_recall = float(predictions[labels].mean())
    foreign_rr = float((~predictions[~labels]).mean())
    responses = np.asarray(
        [
            float(probabilities[index][labels[index]].mean())
            - float(probabilities[index][~labels[index]].max())
            for index in range(mixture_count)
        ],
        dtype=np.float64,
    )
    top2_exact = float(
        np.mean(
            [
                set(np.argsort(probabilities[index])[-2:].tolist())
                == set(np.flatnonzero(labels[index]).tolist())
                for index in range(mixture_count)
            ]
        )
    )
    # v4b redefinition: a query may be present in several mixtures, so its top
    # mixture is correct iff that mixture actually contains the speaker.
    top1_present = np.asarray(
        [bool(labels[probabilities[:, q].argmax(), q]) for q in range(query_count)]
    )
    query_to_mixture_top1 = float(top1_present.mean())
    activity_frame_auc = _binary_auc(
        np.concatenate(activity_labels) > 0.5, np.concatenate(activity_scores)
    )
    return {
        "pair_count": int(labels.size),
        "positive_pairs": int(labels.sum()),
        "foreign_pairs": int((~labels).sum()),
        "presence_auc": float(presence_auc),
        "activity_frame_auc": float(activity_frame_auc),
        "present_recall": present_recall,
        "foreign_rr": foreign_rr,
        "mixture_top2_exact": top2_exact,
        "query_to_mixture_top1": query_to_mixture_top1,
        "query_response_mean": float(responses.mean()),
        "query_response_ci_95": _bootstrap_mean_ci(
            responses, replicates=bootstrap_replicates, seed=bootstrap_seed
        ),
        "mixture_permutation_auc": float(permutation_auc),
        "mixture_permutation_auc_drop": float(presence_auc - permutation_auc),
        "collapsed_query_auc": float(collapsed_query_auc),
        "collapsed_mixture_auc": float(collapsed_mixture_auc),
        "threshold": THRESHOLD,
    }


def _gate(metrics_by_view: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    checks: dict[str, Any] = {}
    passed = True
    for view_name in ("view1", "view2"):
        metrics = metrics_by_view[view_name]
        view_checks: dict[str, Any] = {}
        for name, threshold in MECHANISM_GATE.items():
            if name == "query_response_ci_lower":
                value = float(metrics["query_response_ci_95"]["lower"])
                relation = ">"
                ok = value > threshold
            elif name in {"collapsed_query_auc_max", "collapsed_mixture_auc_max"}:
                metric_name = (
                    "collapsed_query_auc"
                    if name == "collapsed_query_auc_max"
                    else "collapsed_mixture_auc"
                )
                value = float(metrics[metric_name])
                relation = "<="
                ok = value <= threshold
            else:
                value = float(metrics[name])
                relation = ">="
                ok = value >= threshold
            view_checks[name] = {
                "value": value,
                "threshold": threshold,
                "relation": relation,
                "passed": bool(ok),
            }
            passed = passed and bool(ok)
        checks[view_name] = view_checks
    return {
        "passed": passed,
        "gated_views": ["view1", "view2"],
        "fixed_gate": dict(MECHANISM_GATE),
        "checks": checks,
    }


def _evaluate_split(
    model: DACFV4Relation,
    groups: Sequence[CounterfactualGroup],
    *,
    device: torch.device,
    bootstrap_replicates: int,
    seed: int,
) -> Mapping[str, Any]:
    batch = build_v4b_allpairs_batch(groups)
    view1 = _evaluate_view(
        model,
        batch,
        embeddings=batch.embeddings_view1,
        device=device,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 11,
    )
    view2 = _evaluate_view(
        model,
        batch,
        embeddings=batch.embeddings_view2,
        device=device,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=seed + 29,
    )
    return {
        "view1": view1,
        "view2": view2,
        "gate": _gate({"view1": view1, "view2": view2}),
        "bank_speakers": len(batch.speaker_ids),
    }


# --------------------------------------------------------------------------- #
# Preregistration
# --------------------------------------------------------------------------- #


def preregistration_payload(
    cache: FeatureCache,
    config: DACFV4BTrainingConfig,
) -> Mapping[str, Any]:
    contract = _cache_contract(cache)
    config.validate(contract["train_groups"])
    model = DACFV4Relation()
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_a_used": False,
        "cache": {
            "root": cache.root.as_posix(),
            "report_sha256": _sha256_file(cache.root / "cache_report.json"),
            "cache_sha256": str(cache.report.get("cache_sha256", "")),
            "features_manifest_sha256": str(cache.report.get("features_manifest_sha256", "")),
            "contract": contract,
        },
        "model": {
            "class": "DACFV4Relation",
            "trainable_parameter_count": int(model.trainable_parameter_count()),
            "parameter_cap": 2_000_000,
            "input_allowlist": list(cache.report.get("model_input_allowlist", [])),
        },
        "gate": {
            "threshold": THRESHOLD,
            "mechanism_gate": dict(MECHANISM_GATE),
            "gated_views": ["view1", "view2"],
        },
        "config": asdict(config),
        "source_hashes": _source_hashes(),
    }


def write_preregistration(cache_root: str | Path, output: str | Path) -> Mapping[str, Any]:
    cache = load_v4b_feature_cache(cache_root)
    path = Path(output).resolve()
    if path.exists():
        raise FileExistsError(f"preregistration output already exists: {path}")
    payload = preregistration_payload(cache, DACFV4BTrainingConfig())
    _json_write(path, payload)
    return {**payload, "preregistration_sha256": _sha256_file(path)}


def validate_preregistration(
    cache: FeatureCache,
    preregistration_path: str | Path,
    config: DACFV4BTrainingConfig,
) -> Mapping[str, Any]:
    path = Path(preregistration_path).resolve(strict=True)
    prereg = json.loads(path.read_text(encoding="utf-8"))
    expected = preregistration_payload(cache, config)
    if prereg.get("schema") != PREREGISTRATION_SCHEMA:
        raise MechanismContractError("preregistration schema mismatch")
    if prereg.get("protocol_version") != PROTOCOL_VERSION:
        raise MechanismContractError("preregistration protocol mismatch")
    # The contract and gate must match byte-for-byte what the code now derives.
    for key in ("cache", "model", "gate", "config"):
        if prereg.get(key) != expected.get(key):
            raise MechanismContractError(
                f"external DACF-v4b preregistration {key} does not match code/cache contract"
            )
    actual_sha = _sha256_file(path)
    stored_sha = prereg.get("preregistration_sha256")
    if stored_sha and stored_sha != actual_sha:
        raise MechanismContractError("preregistration file SHA mismatch")
    return {**prereg, "preregistration_sha256": actual_sha}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def run_training(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    preregistration_path: str | Path,
    config: DACFV4BTrainingConfig | None = None,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    config = config or DACFV4BTrainingConfig()
    cache = load_v4b_feature_cache(cache_root)
    contract = _cache_contract(cache)
    config.validate(int(contract["train_groups"]))
    prereg = validate_preregistration(cache, preregistration_path, config)

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _set_determinism(config.seed)
    device = _device_for(config)

    model = DACFV4Relation().to(device)
    if model.trainable_parameter_count() >= 2_000_000:
        raise MechanismContractError("DACF-v4b exceeds the preregistered parameter cap")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_weights = DACFV4LossWeights()
    loss_trace: list[float] = []
    positive_pairs_seen = 0
    foreign_pairs_seen = 0

    train_groups = cache.groups["train"]
    model.train()
    for epoch in range(config.epochs):
        batches = epoch_group_batches(
            train_groups,
            groups_per_batch=config.groups_per_batch,
            seed=config.seed,
            epoch=epoch,
        )
        for batch_groups in batches:
            pair_batch = build_v4b_allpairs_batch(batch_groups)
            query1 = torch.from_numpy(pair_batch.embeddings_view1).to(
                device=device, dtype=torch.float32
            )
            query2 = torch.from_numpy(pair_batch.embeddings_view2).to(
                device=device, dtype=torch.float32
            )
            optimiser.zero_grad(set_to_none=True)
            batch_loss = 0.0
            for mixture_index, group in enumerate(pair_batch.groups):
                mixture = torch.from_numpy(group.mixture_features).unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                labels = torch.from_numpy(
                    pair_batch.presence_labels[mixture_index : mixture_index + 1]
                ).to(device=device, dtype=torch.float32)
                activity = torch.from_numpy(
                    pair_batch.activity_targets[mixture_index][None]
                ).to(device=device, dtype=torch.float32)
                state = model.encode_mixture(mixture)
                out1 = model.score_queries(state, query1)
                out2 = model.score_queries(state, query2)
                loss = compute_dacf_v4b_loss(
                    out1,
                    out2,
                    presence_labels=labels,
                    activity_targets=activity,
                    margin=config.hard_foreign_margin,
                    weights=loss_weights,
                )
                scaled = loss.total / len(pair_batch.groups)
                scaled.backward()
                batch_loss += float(loss.total.detach().cpu()) / len(pair_batch.groups)
                positive_pairs_seen += int(labels.sum().item())
                foreign_pairs_seen += int(labels.numel() - labels.sum().item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimiser.step()
            loss_trace.append(batch_loss)

    checkpoint = output / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "DACFV4Relation",
            "protocol_version": PROTOCOL_VERSION,
            "config": asdict(config),
        },
        checkpoint,
    )
    checkpoint_before = _sha256_file(checkpoint)

    # Train is diagnostic only.  Dev is a single observation after the final
    # checkpoint is frozen; neither result selects a checkpoint.
    train_metrics = _evaluate_split(
        model,
        train_groups,
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 10_000,
    )
    dev_evaluation_count = 1
    dev_metrics = _evaluate_split(
        model,
        cache.groups["dev"],
        device=device,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.seed + 20_000,
    )
    checkpoint_after = _sha256_file(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise MechanismContractError("checkpoint changed during frozen evaluation")

    report: Mapping[str, Any] = {
        "schema": REPORT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "verdict": (
            "conditional-GO" if dev_metrics["gate"]["passed"] else "implementation-NO-GO"
        ),
        "verdict_scope": {
            "scope": "v4b cross-environment all-pairs identity mechanism only",
            "single_environment_shortcut_removed": bool(
                contract["single_environment_shortcut_removed"]
            ),
            "final_opened": False,
            "qwen_integration": "not run",
            "cer": "not measured",
            "official_negative_rr": "not measured",
            "rtf": "not measured",
            "same_env_enrollment_caveat": (
                "AISHELL-1 enrolment shares the mixture noise type/RIR class "
                "(same_env_enrollment=true); real-domain transfer is unproven."
            ),
        },
        "dataset_a_used": False,
        "cache_contract": contract,
        "pairing_contract": {
            "bank_deduplication": "one enrolment identity per speaker",
            "same_query_both_labels_per_update": True,
            "c_only_queries_optimised": 0,
            "positive_pairs_seen": positive_pairs_seen,
            "foreign_pairs_seen": foreign_pairs_seen,
        },
        "training": {
            "config": asdict(config),
            "loss_weights": asdict(loss_weights),
            "optimizer": "AdamW",
            "scheduler": "none",
            "early_stop": False,
            "hyperparameter_scan": False,
            "precision": "FP32",
            "device": str(device),
            "updates": len(loss_trace),
            "loss_first": loss_trace[0],
            "loss_last": loss_trace[-1],
        },
        "checkpoint": {
            "path": checkpoint.resolve().as_posix(),
            "sha256_before_dev": checkpoint_before,
            "sha256_after_dev": checkpoint_after,
            "frozen_before_dev": True,
        },
        "preregistration": prereg,
        "train_diagnostic": train_metrics,
        "dev": dev_metrics,
        "dev_evaluation_count": dev_evaluation_count,
        "runtime_sec": float(time.perf_counter() - started),
    }
    _json_write(output / "mechanism_report.json", report)
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--write-preregistration")
    parser.add_argument("--output-dir")
    parser.add_argument("--preregistration")
    args = parser.parse_args(argv)

    if args.write_preregistration:
        if args.output_dir or args.preregistration:
            raise SystemExit("preregistration mode does not accept training outputs")
        payload = write_preregistration(args.cache_root, args.write_preregistration)
        print(json.dumps({"preregistration_sha256": payload["preregistration_sha256"]}, indent=2))
        return 0

    if not args.output_dir or not args.preregistration:
        raise SystemExit("training requires --output-dir and --preregistration")
    report = run_training(
        args.cache_root,
        args.output_dir,
        preregistration_path=args.preregistration,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "dev_gate_passed": report["dev"]["gate"]["passed"],
                "dev_presence_auc_view1": report["dev"]["view1"]["presence_auc"],
                "dev_presence_auc_view2": report["dev"]["view2"]["presence_auc"],
                "dev_activity_frame_auc_view1": report["dev"]["view1"]["activity_frame_auc"],
                "dev_activity_frame_auc_view2": report["dev"]["view2"]["activity_frame_auc"],
            },
            indent=2,
        )
    )
    return 0


__all__ = [
    "CACHE_REPORT_SCHEMA",
    "CACHE_SCHEMA",
    "CounterfactualGroup",
    "DACFV4BTrainingConfig",
    "FeatureCache",
    "MECHANISM_GATE",
    "PROTOCOL_VERSION",
    "PREREGISTRATION_SCHEMA",
    "REPORT_SCHEMA",
    "V4BAllPairsBatch",
    "build_v4b_allpairs_batch",
    "load_v4b_feature_cache",
    "main",
    "preregistration_payload",
    "run_training",
    "validate_preregistration",
    "write_preregistration",
]


if __name__ == "__main__":
    raise SystemExit(main())
