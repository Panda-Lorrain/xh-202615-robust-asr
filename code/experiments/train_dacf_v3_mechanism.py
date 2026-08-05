"""Train and audit the DACF-v3 ECST mechanism on a train/dev cache only.

This module is intentionally not an ASR trainer.  It implements one narrow
stage of the DACF-v3 contract:

* the cache must contain exactly ``train`` and ``dev``;
* every sampled item is a complete byte-identical A/B/C group;
* two independent enrollment views are optimized with the fixed ECST loss;
* there is no dev checkpoint selection: the final fixed-update state is saved,
  hashed, and only then evaluated once on dev;
* final, CER, submission RR, and RTF are outside this module's scope.

The implementation uses only Python, NumPy, and PyTorch.  It deliberately
does not import the feature builder's audio dependencies: a feature cache is
validated from its manifest, NPZ metadata, and SHA256 payload before any
model input is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from dacf_v3_ecst import DACFV3ECST, DACFV3ECSTOutput
from dacf_v3_objective import DACFV3LossWeights, compute_dacf_v3_loss


EXPECTED_CACHE_SCHEMA = "dacf-v3-feature-cache-v0.1"
EXPECTED_REPORT_SCHEMA = "dacf-v3-feature-cache-report-v0.1"
EXPECTED_SOURCE_CORPUS = "AISHELL-1"
EXPECTED_SPLITS = ("train", "dev")
EXPECTED_ROLES = ("present_A", "present_B", "absent_C")
MEL_BINS = 128
CAMPP_DIM = 512
THRESHOLD = 0.50
CHECKPOINT_POLICY = "fixed_updates_final_checkpoint_then_one_dev_evaluation"
PROTOCOL_VERSION = "dacf-v3-mechanism-protocol-v0.2"
TRAINING_PREREGISTRATION_SCHEMA = "dacf-v3-training-preregistration-v0.1"
EXPECTED_DATA_PROTOCOL_SCHEMA = "dacf-v3-official-aishell-protocol-v0.2"
EXPECTED_PROTOCOL_REAUDIT_SCHEMA = "dacf-v3-official-aishell-reaudit-v0.1"
SAME_TEXT_GATE_THRESHOLD = 0.75

# These are mechanism gates only.  They are intentionally duplicated here as
# immutable trainer constants so a command-line invocation cannot silently
# tune a gate from dev.
MECHANISM_GATE = {
    "presence_auc": 0.85,
    "activity_frame_auc": 0.80,
    "present_recall": 0.80,
    "absent_rr": 0.95,
    "query_response_mean": 0.20,
    "query_response_ci_lower": 0.05,
    "query_permutation_auc_drop": 0.15,
}

TRAINING_PREREGISTRATION = {
    "seed": 2026080606,
    "epochs": 20,
    "groups_per_epoch": 96,
    "groups_per_update": 1,
    "updates": 1920,
    "one_complete_abc_group_per_step": True,
    "optimizer": "AdamW",
    "learning_rate": 3.0e-4,
    "weight_decay": 1.0e-4,
    "gradient_clip_norm": 5.0,
    "precision": "FP32",
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "model_architecture": "DACFV3ECST",
    "loss_weights": {
        "presence": 1.0,
        "activity": 1.0,
        "counterfactual_margin": 0.5,
        "view_consistency": 0.1,
        "absent_latent_energy": 0.05,
    },
    "counterfactual_margin": 0.20,
    "scheduler": "none",
    "early_stop": False,
    "hyperparameter_scan": False,
    "checkpoint": "epoch_20_final_only",
    "dev_evaluation_count": 1,
    "threshold": THRESHOLD,
    "bootstrap_replicates": 2000,
    "bootstrap_seed_offset": 7001,
    "query_permutation_policy": "sorted dev groups cyclic +1; all source speakers absent from destination mixture",
}

_DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)
_LINEAGE_FIELDS = {
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
    "resolved_audio_paths",
    "resolved_source_paths",
    "resolved_feature_paths",
}
_SAME_TEXT_FIELDS = (
    "same_text_different_speaker_eval",
    "same_text_different_speaker",
    "hard_negative_same_text",
)


class MechanismContractError(ValueError):
    """Raised before training when the cache or model-input contract fails."""


@dataclass(frozen=True)
class MechanismTrainingConfig:
    """Fixed optimization settings; tests may use a shorter explicit config."""

    seed: int = 2026080606
    epochs: int = 20
    groups_per_epoch: int = 96
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    counterfactual_margin: float = 0.20
    bootstrap_replicates: int = 2000
    device: str = "auto"

    def validate(self) -> None:
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        for name in ("epochs", "groups_per_epoch", "bootstrap_replicates"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
            "counterfactual_margin",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")

    @property
    def fixed_updates(self) -> int:
        return self.epochs * self.groups_per_epoch


@dataclass(frozen=True)
class QueryRecord:
    row_id: str
    group_id: str
    role: str
    speaker_id: str
    target_present: bool
    query_role_id: int
    embedding: np.ndarray
    embedding_view2: np.ndarray
    target_activity: np.ndarray
    same_text_eval: bool | None


@dataclass(frozen=True)
class CounterfactualGroup:
    split: str
    group_id: str
    mixture_feature_path: Path
    mixture_feature_sha256: str
    mixture_features: np.ndarray
    mixture_speaker_ids: tuple[str, str]
    rows: tuple[QueryRecord, QueryRecord, QueryRecord]


@dataclass(frozen=True)
class FeatureCache:
    root: Path
    report: Mapping[str, Any]
    groups: Mapping[str, tuple[CounterfactualGroup, ...]]
    same_text_status: Mapping[str, str]


@dataclass(frozen=True)
class ScoreRecord:
    group_id: str
    speaker_id: str
    target_present: bool
    presence_probability: float
    permuted_presence_probability: float
    activity_probability: np.ndarray
    activity_target: np.ndarray
    same_text_eval: bool | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Match the feature-builder array hash contract exactly."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def cache_payload_sha256(cache_root: str | Path) -> str:
    """Hash the exact cache payload scope used by build_dacf_v3_features."""

    root = Path(cache_root).resolve(strict=True)
    manifest = root / "features_manifest.jsonl"
    paths = [manifest]
    paths.extend(sorted((root / "mixture").glob("*.npz")))
    paths.extend(sorted((root / "query").glob("*.npz")))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _scalar_text(value: Any, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise MechanismContractError(f"{field} must be a scalar NPZ field")
    item = array.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    text = str(item).strip()
    if not text:
        raise MechanismContractError(f"{field} must not be empty")
    return text


def _scalar_bool(value: Any, field: str) -> bool:
    array = np.asarray(value)
    if array.ndim != 0:
        raise MechanismContractError(f"{field} must be a scalar NPZ field")
    item = array.item()
    if isinstance(item, (bool, np.bool_)):
        return bool(item)
    if isinstance(item, (int, np.integer)) and int(item) in (0, 1):
        return bool(item)
    text = str(item).strip().casefold()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise MechanismContractError(f"{field} is not a boolean scalar")


def _scalar_int(value: Any, field: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0:
        raise MechanismContractError(f"{field} must be a scalar")
    item = array.item()
    if isinstance(item, (bool, np.bool_)):
        raise MechanismContractError(f"{field} must be an integer, not bool")
    try:
        result = int(item)
    except (TypeError, ValueError) as exc:
        raise MechanismContractError(f"{field} is not an integer") from exc
    if isinstance(item, (float, np.floating)) and float(item) != result:
        raise MechanismContractError(f"{field} is not an integer")
    return result


def _contains_dataset_a(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_dataset_a(child) for child in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_dataset_a(child) for child in value)
    if isinstance(value, str):
        text = value.casefold()
        return any(marker in text for marker in _DATASET_A_MARKERS)
    return False


def _assert_inside(path: Path, root: Path, field: str) -> Path:
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise MechanismContractError(f"{field} escapes cache root: {resolved}")
    return resolved


def _cache_path(root: Path, raw: Any, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise MechanismContractError(f"{field} must be a non-empty path")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return _assert_inside(candidate, root, field)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MechanismContractError(f"invalid manifest JSON at line {line_number}") from exc
        if not isinstance(value, dict):
            raise MechanismContractError(f"manifest line {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise MechanismContractError("features_manifest.jsonl is empty")
    return rows


def _same_text_flag(row: Mapping[str, Any]) -> tuple[bool | None, str]:
    present = [key for key in _SAME_TEXT_FIELDS if key in row]
    if not present:
        return None, "protocol has no explicit same-text/different-speaker evaluation field"
    values: list[bool] = []
    for key in present:
        value = row[key]
        if not isinstance(value, (bool, int)) or isinstance(value, bool) and value not in (True, False):
            return None, f"protocol field {key} is not a reliable boolean"
        if isinstance(value, int) and value not in (0, 1):
            return None, f"protocol field {key} is not a reliable boolean"
        values.append(bool(value))
    if len(set(values)) != 1:
        return None, "protocol same-text fields disagree"
    return values[0], "explicit protocol field"


def _validate_mixture_npz(
    path: Path,
    row: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "input_features",
            "feature_attention_mask",
            "mixture_sha256",
            "qwen_config_sha256",
            "qwen_feature_spec_sha256",
            "feature_array_sha256",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise MechanismContractError(f"mixture NPZ lacks {missing}: {path}")
        features = np.asarray(data["input_features"])
        if features.ndim != 2 or features.shape[0] != MEL_BINS or features.shape[1] < 1:
            raise MechanismContractError(f"invalid mixture input_features shape: {features.shape}")
        if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
            raise MechanismContractError("mixture input_features must be finite floating point")
        attention = np.asarray(data["feature_attention_mask"])
        if attention.shape != (features.shape[1],) or not np.isfinite(attention).all():
            raise MechanismContractError("mixture feature_attention_mask shape/value mismatch")
        if np.any((attention != 0) & (attention != 1)):
            raise MechanismContractError("mixture feature_attention_mask must be binary")
        if not np.all(attention == 1):
            raise MechanismContractError(
                "ECST does not consume an attention mask; cached mixture frames must all be valid"
            )
        if _scalar_text(data["mixture_sha256"], "mixture_sha256").casefold() != str(row["mixture_sha256"]).casefold():
            raise MechanismContractError("mixture audio SHA metadata mismatch")
        qwen_config_sha = str(report.get("qwen_config_sha256", ""))
        if qwen_config_sha and _scalar_text(data["qwen_config_sha256"], "qwen_config_sha256") != qwen_config_sha:
            raise MechanismContractError("mixture Qwen config binding mismatch")
        expected_spec = str(report.get("feature_extractor_spec_sha256", ""))
        actual_spec = _scalar_text(data["qwen_feature_spec_sha256"], "qwen_feature_spec_sha256")
        if expected_spec and actual_spec != expected_spec:
            raise MechanismContractError("mixture Qwen feature spec binding mismatch")
        if _scalar_text(data["feature_array_sha256"], "feature_array_sha256") != sha256_array(features):
            raise MechanismContractError("mixture feature array SHA mismatch")
    return np.asarray(features, dtype=np.float32), _sha256_file(path)


def _validate_query_npz(
    path: Path,
    row: Mapping[str, Any],
    mixture_features: np.ndarray,
) -> QueryRecord:
    with np.load(path, allow_pickle=False) as data:
        if "query_role_id" in data.files:
            raise MechanismContractError("query NPZ contains forbidden query_role_id")
        required = {
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
        missing = sorted(required - set(data.files))
        if missing:
            raise MechanismContractError(f"query NPZ lacks {missing}: {path}")
        for field in ("row_id", "split", "base_mixture_id", "query_role", "query_speaker_id", "source_corpus"):
            if _scalar_text(data[field], field) != str(row[field] if field != "row_id" else row["id"]):
                raise MechanismContractError(f"query metadata mismatch for {field}: {path}")
        expected_present = bool(row["target_present"])
        if _scalar_bool(data["target_present"], "target_present") != expected_present:
            raise MechanismContractError("query target_present metadata mismatch")
        embedding = np.asarray(data["enrollment_embedding"], dtype=np.float32)
        embedding_view2 = np.asarray(data["enrollment_embedding_view2"], dtype=np.float32)
        for name, value in (("enrollment_embedding", embedding), ("enrollment_embedding_view2", embedding_view2)):
            if value.shape != (CAMPP_DIM,) or not np.isfinite(value).all():
                raise MechanismContractError(f"{name} must be finite with shape (512,)")
            if abs(float(np.linalg.norm(value)) - 1.0) > 2e-5:
                raise MechanismContractError(f"{name} must be L2-normalized")
            expected_hash = row.get(
                "enrollment_embedding_sha256"
                if name == "enrollment_embedding"
                else "enrollment_embedding_view2_sha256",
                "",
            )
            if expected_hash and sha256_array(value) != str(expected_hash):
                raise MechanismContractError(f"{name} array SHA mismatch")
        activity = np.asarray(data["target_activity"], dtype=np.float32)
        if activity.shape != (mixture_features.shape[1],) or not np.isfinite(activity).all():
            raise MechanismContractError("target_activity shape/value mismatch")
        if np.any((activity < 0.0) | (activity > 1.0)):
            raise MechanismContractError("target_activity must be in [0,1]")
        if not expected_present and np.any(activity != 0.0):
            raise MechanismContractError("absent_C target_activity must be zero")
        if _scalar_text(data["mixture_feature_sha256"], "mixture_feature_sha256") != str(row["mixture_feature_sha256"]):
            raise MechanismContractError("query mixture feature SHA metadata mismatch")
        if _scalar_text(data["target_activity_sha256"], "target_activity_sha256") != str(row["target_activity_sha256"]):
            raise MechanismContractError("query target activity source SHA metadata mismatch")
        if _scalar_text(data["target_activity_array_sha256"], "target_activity_array_sha256") != str(row["target_activity_array_sha256"]):
            raise MechanismContractError("query target activity array SHA metadata mismatch")
        if sha256_array(activity) != str(row["target_activity_array_sha256"]):
            raise MechanismContractError("query target activity array SHA mismatch")
    same_text, _ = _same_text_flag(row)
    return QueryRecord(
        row_id=str(row["id"]),
        group_id=str(row["base_mixture_id"]),
        role=str(row["query_role"]),
        speaker_id=str(row["query_speaker_id"]),
        target_present=expected_present,
        query_role_id=int(row["query_role_id"]),
        embedding=embedding,
        embedding_view2=embedding_view2,
        target_activity=activity,
        same_text_eval=same_text,
    )


def _nonempty_overlap(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_nonempty_overlap(child) for child in value.values())
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return False


def _validate_report_contract(root: Path, report: Mapping[str, Any]) -> None:
    if report.get("schema") != EXPECTED_REPORT_SCHEMA:
        raise MechanismContractError("cache report schema mismatch")
    if report.get("cache_schema") != EXPECTED_CACHE_SCHEMA:
        raise MechanismContractError("cache schema mismatch")
    builder_path = Path(str(report.get("builder_code", ""))).resolve(strict=True)
    expected_builder = Path(__file__).resolve().with_name("build_dacf_v3_features.py")
    if builder_path != expected_builder:
        raise MechanismContractError("cache builder path mismatch")
    if _sha256_file(builder_path) != str(report.get("builder_code_sha256", "")):
        raise MechanismContractError("cache builder code SHA changed")
    if report.get("source_corpus") != EXPECTED_SOURCE_CORPUS or report.get("dataset_a_used") is not False:
        raise MechanismContractError("cache source/Dataset-A contract mismatch")
    split_contract = report.get("split_contract")
    if not isinstance(split_contract, Mapping):
        raise MechanismContractError("cache report lacks split_contract")
    if tuple(split_contract.get("splits", ())) != EXPECTED_SPLITS:
        raise MechanismContractError("trainer accepts only cache splits train and dev")
    if split_contract.get("final_deferred") is not True:
        raise MechanismContractError("cache report final_deferred must be true")
    if "final_deferred" in report and report.get("final_deferred") is not True:
        raise MechanismContractError("cache report final_deferred must be true")
    if split_contract.get("final_gate_split") is not None:
        raise MechanismContractError("cache report has a final gate split")
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise MechanismContractError("cache report lacks counts")
    split_counts = counts.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != set(EXPECTED_SPLITS):
        raise MechanismContractError("cache report split_counts must contain only train/dev")
    overlap = report.get("overlap_audit")
    if isinstance(overlap, Mapping) and _nonempty_overlap(overlap):
        raise MechanismContractError("cache report contains non-empty split overlap")
    for field in ("allowed_source_root", "qwen_preprocessor_config", "campp_model"):
        if _contains_dataset_a(report.get(field, "")):
            raise MechanismContractError(f"Dataset-A path marker in cache report {field}")


def _make_grouped_cache(root: Path, report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> FeatureCache:
    if int(report["counts"].get("rows", -1)) != len(rows):
        raise MechanismContractError("cache report row count mismatch")
    seen_ids: set[str] = set()
    expected_query_paths: set[str] = set()
    expected_mixture_paths: set[str] = set()
    mixture_cache: dict[Path, tuple[np.ndarray, str]] = {}
    by_split_group: dict[tuple[str, str], list[tuple[Mapping[str, Any], QueryRecord, Path, str]]] = {}
    same_text_reasons: list[str] = []

    for index, row in enumerate(rows, 1):
        label = f"manifest row {index}"
        required = {
            "cache_schema",
            "id",
            "split",
            "base_mixture_id",
            "query_role",
            "query_role_id",
            "query_role_id_used_as_model_input",
            "query_speaker_id",
            "target_present",
            "source_corpus",
            "dataset_a_used",
            "mixture_sha256",
            "mixture_feature",
            "mixture_feature_sha256",
            "mixture_input_features_sha256",
            "query_feature",
            "query_npz_sha256",
            "target_activity_sha256",
            "target_activity_array_sha256",
            "mixture_speakers",
        }
        missing = sorted(required - set(row))
        if missing:
            raise MechanismContractError(f"{label} lacks {missing}")
        if row.get("cache_schema") != EXPECTED_CACHE_SCHEMA:
            raise MechanismContractError(f"{label} cache schema mismatch")
        split = str(row["split"])
        if split not in EXPECTED_SPLITS:
            raise MechanismContractError(f"{label} has forbidden split {split!r}")
        if row.get("source_corpus") != EXPECTED_SOURCE_CORPUS or row.get("dataset_a_used") is not False:
            raise MechanismContractError(f"{label} source/Dataset-A contract mismatch")
        if row.get("query_role_id_used_as_model_input") is not False:
            raise MechanismContractError(f"{label} query_role_id is not audit-only")
        if _contains_dataset_a({key: row.get(key) for key in _LINEAGE_FIELDS if key in row}):
            raise MechanismContractError(f"{label} contains a Dataset-A lineage marker")
        row_id = str(row["id"])
        if not row_id or row_id in seen_ids:
            raise MechanismContractError(f"{label} has a duplicate/empty id")
        seen_ids.add(row_id)
        role = str(row["query_role"])
        if role not in EXPECTED_ROLES:
            raise MechanismContractError(f"{label} has invalid query_role {role!r}")
        if not isinstance(row["query_role_id"], int) or isinstance(row["query_role_id"], bool):
            raise MechanismContractError(f"{label} query_role_id is not an audit integer")
        expected_role_id = EXPECTED_ROLES.index(role)
        if int(row["query_role_id"]) != expected_role_id:
            raise MechanismContractError(
                f"{label} query_role_id disagrees with query_role: "
                f"{row['query_role_id']} != {expected_role_id}"
            )
        expected_present = role != "absent_C"
        if bool(row["target_present"]) != expected_present:
            raise MechanismContractError(f"{label} target_present disagrees with query_role")
        mixture_path = _cache_path(root, row["mixture_feature"], f"{label}.mixture_feature")
        query_path = _cache_path(root, row["query_feature"], f"{label}.query_feature")
        expected_query_paths.add(query_path.relative_to(root).as_posix())
        expected_mixture_paths.add(mixture_path.relative_to(root).as_posix())
        query_sha = _sha256_file(query_path)
        if query_sha != str(row["query_npz_sha256"]):
            raise MechanismContractError(f"{label} query_npz_sha256 mismatch")
        mixture_sha = _sha256_file(mixture_path)
        if mixture_sha != str(row["mixture_feature_sha256"]):
            raise MechanismContractError(f"{label} mixture_feature_sha256 mismatch")
        cached = mixture_cache.get(mixture_path)
        if cached is None:
            cached = _validate_mixture_npz(mixture_path, row, report)
            mixture_cache[mixture_path] = cached
        mixture_features, actual_mixture_file_sha = cached
        if actual_mixture_file_sha != str(row["mixture_feature_sha256"]):
            raise MechanismContractError(f"{label} mixture file SHA mismatch")
        if str(row["mixture_input_features_sha256"]) != sha256_array(mixture_features):
            raise MechanismContractError(f"{label} mixture input feature SHA mismatch")
        query = _validate_query_npz(query_path, row, mixture_features)
        same_text, reason = _same_text_flag(row)
        if same_text is None:
            same_text_reasons.append(reason)
        elif reason not in same_text_reasons:
            same_text_reasons.append(reason)
        query = replace(query, same_text_eval=same_text)
        by_split_group.setdefault((split, str(row["base_mixture_id"])), []).append(
            (row, query, mixture_path, str(row["mixture_feature_sha256"]))
        )

    actual_npz_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.npz")
    }
    expected_npz_paths = expected_query_paths | expected_mixture_paths
    if actual_npz_paths != expected_npz_paths:
        extra = sorted(actual_npz_paths - expected_npz_paths)
        missing = sorted(expected_npz_paths - actual_npz_paths)
        raise MechanismContractError(f"cache NPZ set mismatch; extra={extra[:3]}, missing={missing[:3]}")

    groups_by_split: dict[str, list[CounterfactualGroup]] = {"train": [], "dev": []}
    for (split, group_id), items in sorted(by_split_group.items()):
        if len(items) != 3:
            raise MechanismContractError(f"{split}/{group_id} is not a complete A/B/C group")
        roles = [item[1].role for item in items]
        if set(roles) != set(EXPECTED_ROLES):
            raise MechanismContractError(f"{split}/{group_id} roles are not exactly A/B/C")
        items = sorted(items, key=lambda item: EXPECTED_ROLES.index(item[1].role))
        mixture_paths = {item[2] for item in items}
        mixture_file_shas = {item[3] for item in items}
        audio_shas = {str(item[0]["mixture_sha256"]).casefold() for item in items}
        feature_array_shas = {str(item[0]["mixture_input_features_sha256"]) for item in items}
        if len(mixture_paths) != 1 or len(mixture_file_shas) != 1 or len(audio_shas) != 1 or len(feature_array_shas) != 1:
            raise MechanismContractError(f"{split}/{group_id} A/B/C mixtures are not byte-identical")
        query_rows = tuple(item[1] for item in items)
        if len({query.speaker_id for query in query_rows}) != 3:
            raise MechanismContractError(f"{split}/{group_id} query speakers are not distinct")
        mixture_speaker_sets: list[set[str]] = []
        for source_row, _, _, _ in items:
            raw_speakers = source_row.get("mixture_speakers")
            if not isinstance(raw_speakers, Mapping) or set(raw_speakers) != {"A", "B"}:
                raise MechanismContractError(
                    f"{split}/{group_id} mixture_speakers must expose A/B"
                )
            speakers = {str(value) for value in raw_speakers.values() if str(value)}
            if len(speakers) != 2:
                raise MechanismContractError(
                    f"{split}/{group_id} mixture speaker identities are invalid"
                )
            mixture_speaker_sets.append(speakers)
        if any(value != mixture_speaker_sets[0] for value in mixture_speaker_sets[1:]):
            raise MechanismContractError(
                f"{split}/{group_id} mixture speaker identities differ across A/B/C rows"
            )
        mixture_speaker_ids = tuple(sorted(mixture_speaker_sets[0]))
        for query in query_rows:
            in_mixture = query.speaker_id in mixture_speaker_sets[0]
            if query.target_present is not in_mixture:
                raise MechanismContractError(
                    f"{split}/{group_id}/{query.role} query/mixture speaker membership mismatch"
                )
        mixture_path = items[0][2]
        groups_by_split[split].append(
            CounterfactualGroup(
                split=split,
                group_id=group_id,
                mixture_feature_path=mixture_path,
                mixture_feature_sha256=items[0][3],
                mixture_features=mixture_cache[mixture_path][0],
                mixture_speaker_ids=mixture_speaker_ids,  # type: ignore[arg-type]
                rows=query_rows,  # type: ignore[arg-type]
            )
        )

    if not groups_by_split["train"] or not groups_by_split["dev"]:
        raise MechanismContractError("train and dev must both contain at least one complete group")
    train_mixtures = {group.mixture_feature_sha256 for group in groups_by_split["train"]}
    dev_mixtures = {group.mixture_feature_sha256 for group in groups_by_split["dev"]}
    if train_mixtures & dev_mixtures:
        raise MechanismContractError("train/dev mixture feature overlap")
    train_speakers = {row.speaker_id for group in groups_by_split["train"] for row in group.rows}
    dev_speakers = {row.speaker_id for group in groups_by_split["dev"] for row in group.rows}
    if train_speakers & dev_speakers:
        raise MechanismContractError("train/dev query speaker overlap")
    for split, split_groups in groups_by_split.items():
        seen_split_speakers: set[str] = set()
        for group in split_groups:
            group_speakers = {row.speaker_id for row in group.rows}
            repeated = seen_split_speakers & group_speakers
            if repeated:
                raise MechanismContractError(
                    f"{split} query speaker reused across groups: {sorted(repeated)}"
                )
            seen_split_speakers.update(group_speakers)
        _assert_cyclic_permutation_foreign(split_groups)
    all_query_rows = [
        row
        for group_list in groups_by_split.values()
        for group in group_list
        for row in group.rows
    ]
    all_have_same_text_field = all(row.same_text_eval is not None for row in all_query_rows)
    has_same_text_subset = any(row.same_text_eval is True for row in all_query_rows)
    if all_have_same_text_field and has_same_text_subset:
        same_text_status = "available"
        same_text_reason = "; ".join(sorted(set(same_text_reasons))) or "explicit protocol field"
    elif not same_text_reasons:
        same_text_status = "deferred"
        same_text_reason = "protocol has no explicit same-text/different-speaker evaluation field"
    elif not all_have_same_text_field:
        same_text_status = "deferred"
        same_text_reason = "same-text evaluation field is missing from at least one cache row"
    else:
        same_text_status = "deferred"
        same_text_reason = "explicit same-text subset is empty"
    status = {"status": same_text_status, "reason": same_text_reason}
    return FeatureCache(
        root=root,
        report=report,
        groups={key: tuple(value) for key, value in groups_by_split.items()},
        same_text_status=status,
    )


def _assert_cyclic_permutation_foreign(
    groups: Sequence[CounterfactualGroup],
) -> None:
    """Prove the fixed +1 permutation supplies only foreign enrollments."""

    if len(groups) < 2:
        raise MechanismContractError(
            "cross-group query permutation requires at least two complete groups"
        )
    ordered = tuple(sorted(groups, key=lambda item: item.group_id))
    sources = ordered[1:] + ordered[:1]
    for destination, source in zip(ordered, sources):
        source_speakers = {row.speaker_id for row in source.rows}
        overlap = source_speakers & set(destination.mixture_speaker_ids)
        if overlap:
            raise MechanismContractError(
                "cyclic query permutation is not foreign to destination mixture: "
                f"destination={destination.group_id}, source={source.group_id}, "
                f"overlap={sorted(overlap)}"
            )


def _validate_full_feature_cache_provenance(cache_root: str | Path) -> None:
    """Run the feature builder's source-manifest and artifact revalidation."""

    try:
        from build_dacf_v3_features import FeatureContractError, validate_cache

        validate_cache(cache_root)
    except (FeatureContractError, OSError, ValueError) as exc:
        raise MechanismContractError(
            f"full feature-cache provenance validation failed: {exc}"
        ) from exc


def load_feature_cache(cache_root: str | Path) -> FeatureCache:
    """Validate and load only the train/dev feature cache payload."""

    root = Path(cache_root).resolve(strict=True)
    project_root = Path(__file__).resolve().parents[2]
    if root != project_root and project_root not in root.parents:
        raise MechanismContractError("cache root must stay inside the project workspace")
    report_path = root / "cache_report.json"
    manifest_path = root / "features_manifest.jsonl"
    if not report_path.is_file() or not manifest_path.is_file():
        raise MechanismContractError("cache requires cache_report.json and features_manifest.jsonl")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MechanismContractError("cache_report.json is not valid JSON") from exc
    if not isinstance(report, dict):
        raise MechanismContractError("cache_report.json must be an object")
    _validate_report_contract(root, report)
    rows = _read_jsonl(manifest_path)
    if any(str(row.get("split", "")) == "final" for row in rows):
        raise MechanismContractError("final row found; mechanism trainer refuses final")
    if _sha256_file(manifest_path) != str(report.get("manifest_sha256", "")):
        raise MechanismContractError("cache report manifest_sha256 mismatch")
    cache = _make_grouped_cache(root, report, rows)
    if cache_payload_sha256(root) != str(report.get("cache_sha256", "")):
        raise MechanismContractError("cache report cache_sha256 mismatch")
    return cache


def _source_bindings() -> dict[str, str]:
    experiments = Path(__file__).resolve().parent
    project_root = experiments.parents[1]
    paths = (
        Path(__file__).resolve(),
        experiments / "dacf_v3_ecst.py",
        experiments / "dacf_v3_objective.py",
        experiments / "build_dacf_v3_features.py",
        experiments / "build_dacf_v3_protocol.py",
    )
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in paths
    }


def write_training_preregistration(
    cache_root: str | Path,
    protocol_preregistration: str | Path,
    protocol_audit_report: str | Path,
    output_path: str | Path,
    *,
    allow_synthetic_cache: bool = False,
) -> dict[str, Any]:
    """Bind code and train/dev cache hashes before any optimizer is created."""

    if not allow_synthetic_cache:
        _validate_full_feature_cache_provenance(cache_root)
    cache = load_feature_cache(cache_root)
    project_root = Path(__file__).resolve().parents[2]
    protocol_path = Path(protocol_preregistration).resolve(strict=True)
    if protocol_path != project_root and project_root not in protocol_path.parents:
        raise MechanismContractError("data protocol preregistration must stay in the project")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, Mapping):
        raise MechanismContractError("data protocol preregistration must be an object")
    if protocol.get("schema") != EXPECTED_DATA_PROTOCOL_SCHEMA:
        raise MechanismContractError("data protocol preregistration schema mismatch")
    if protocol.get("dataset_a_used") is not False or protocol.get("source_corpus") != EXPECTED_SOURCE_CORPUS:
        raise MechanismContractError("data protocol source/Dataset-A contract mismatch")
    protocol_gate = protocol.get("fixed_gate")
    if not isinstance(protocol_gate, Mapping):
        raise MechanismContractError("data protocol preregistration lacks fixed_gate")
    expected_protocol_gate = {
        "presence_auc": MECHANISM_GATE["presence_auc"],
        "activity_auc": MECHANISM_GATE["activity_frame_auc"],
        "present_recall": MECHANISM_GATE["present_recall"],
        "absent_rr": MECHANISM_GATE["absent_rr"],
        "query_response_mean": MECHANISM_GATE["query_response_mean"],
        "query_response_group_bootstrap_ci_lower": MECHANISM_GATE["query_response_ci_lower"],
        "query_permutation_auc_drop_min": MECHANISM_GATE["query_permutation_auc_drop"],
        "presence_threshold": THRESHOLD,
    }
    for name, expected in expected_protocol_gate.items():
        if protocol_gate.get(name) != expected:
            raise MechanismContractError(
                f"data protocol gate {name} differs from trainer contract"
            )
    audit_path = Path(protocol_audit_report).resolve(strict=True)
    if audit_path != project_root and project_root not in audit_path.parents:
        raise MechanismContractError("protocol audit report must stay in the project")
    protocol_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(protocol_audit, Mapping):
        raise MechanismContractError("protocol audit report must be an object")
    if protocol_audit.get("schema") != EXPECTED_PROTOCOL_REAUDIT_SCHEMA:
        raise MechanismContractError("protocol audit report schema mismatch")
    if protocol_audit.get("protocol_schema") != EXPECTED_DATA_PROTOCOL_SCHEMA:
        raise MechanismContractError("protocol audit report protocol schema mismatch")
    if protocol_audit.get("dataset_a_used") is not False:
        raise MechanismContractError("protocol audit report violates Dataset-A policy")
    if protocol_audit.get("hard_negative_verified_count") != 0:
        raise MechanismContractError("protocol audit hard-negative count must remain zero")
    protocol_audit_body = protocol_audit.get("audit")
    if not isinstance(protocol_audit_body, Mapping) or protocol_audit_body.get(
        "all_cross_split_overlaps_zero"
    ) is not True:
        raise MechanismContractError("protocol audit did not prove zero cross-split overlap")
    if str(protocol_audit.get("protocol_preregistration_sha256", "")) != _sha256_file(
        protocol_path
    ):
        raise MechanismContractError("protocol audit is not bound to this preregistration")
    audit_code_path = Path(str(protocol_audit.get("audit_code", ""))).resolve(strict=True)
    if _sha256_file(audit_code_path) != str(protocol_audit.get("audit_code_sha256", "")):
        raise MechanismContractError("protocol audit code SHA changed")
    source_build_report_path = Path(
        str(protocol_audit.get("source_build_report", ""))
    ).resolve(strict=True)
    if _sha256_file(source_build_report_path) != str(
        protocol_audit.get("source_build_report_sha256", "")
    ):
        raise MechanismContractError("protocol source build report SHA changed")

    cache_input_manifests = cache.report.get("input_manifests")
    audit_manifest_sha = protocol_audit.get("manifest_sha256")
    if not isinstance(cache_input_manifests, Mapping) or not isinstance(
        audit_manifest_sha, Mapping
    ):
        raise MechanismContractError("cache/audit manifest bindings are missing")
    for split in EXPECTED_SPLITS:
        cache_entries = cache_input_manifests.get(split)
        audit_entries = audit_manifest_sha.get(split)
        if not isinstance(cache_entries, list) or not isinstance(audit_entries, Mapping):
            raise MechanismContractError(f"cache/audit lacks {split} manifest bindings")
        cache_binding = {
            Path(str(entry.get("path", ""))).resolve(strict=True).as_posix(): str(
                entry.get("sha256", "")
            )
            for entry in cache_entries
            if isinstance(entry, Mapping)
        }
        normalised_audit_binding = {
            Path(str(path)).resolve(strict=True).as_posix(): str(value)
            for path, value in audit_entries.items()
        }
        if cache_binding != normalised_audit_binding:
            raise MechanismContractError(
                f"feature cache {split} manifests differ from protocol reaudit"
            )
    report_path = cache.root / "cache_report.json"
    manifest_path = cache.root / "features_manifest.jsonl"
    payload: dict[str, Any] = {
        "schema": TRAINING_PREREGISTRATION_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "dataset_a_used": False,
        "source_corpus": EXPECTED_SOURCE_CORPUS,
        "data_protocol": {
            "path": protocol_path.as_posix(),
            "sha256": _sha256_file(protocol_path),
            "schema": protocol.get("schema"),
            "experiment_id": protocol.get("experiment_id"),
        },
        "protocol_reaudit": {
            "path": audit_path.as_posix(),
            "sha256": _sha256_file(audit_path),
            "schema": protocol_audit.get("schema"),
            "audit_code": audit_code_path.as_posix(),
            "audit_code_sha256": protocol_audit.get("audit_code_sha256"),
            "source_build_report": source_build_report_path.as_posix(),
            "source_build_report_sha256": protocol_audit.get(
                "source_build_report_sha256"
            ),
            "all_cross_split_overlaps_zero": True,
            "manifest_sha256": {
                split: dict(audit_manifest_sha[split]) for split in EXPECTED_SPLITS
            },
        },
        "feature_cache": {
            "root": cache.root.as_posix(),
            "report_sha256": _sha256_file(report_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "cache_sha256": str(cache.report["cache_sha256"]),
            "splits": list(EXPECTED_SPLITS),
            "final_deferred": True,
            "qwen_config_sha256": cache.report.get("qwen_config_sha256"),
            "feature_extractor_spec_sha256": cache.report.get("feature_extractor_spec_sha256"),
            "campp_model_sha256": cache.report.get("campp_model_sha256"),
            "builder_code": cache.report.get("builder_code"),
            "builder_code_sha256": cache.report.get("builder_code_sha256"),
        },
        "source_sha256": _source_bindings(),
        "training_contract": dict(TRAINING_PREREGISTRATION),
        "mechanism_gate": dict(MECHANISM_GATE),
        "same_text_gate": {
            "current_mechanism_gate": False,
            "threshold_before_qwen": SAME_TEXT_GATE_THRESHOLD,
            "required_suite": "real home-command same-text/different-speaker hard negatives",
        },
        "selection_policy": {
            "train_only_optimization": True,
            "checkpoint": CHECKPOINT_POLICY,
            "dev_evaluation_count": 1,
            "final_opened": False,
        },
    }
    output = Path(output_path).resolve()
    if output != project_root and project_root not in output.parents:
        raise MechanismContractError("training preregistration output must stay in the project")
    if output.exists():
        raise MechanismContractError(f"training preregistration already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_training_preregistration(
    path: str | Path,
    cache: FeatureCache,
) -> dict[str, Any]:
    """Fail closed if code, data, or fixed settings changed after preregistration."""

    prereg_path = Path(path).resolve(strict=True)
    payload = json.loads(prereg_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != TRAINING_PREREGISTRATION_SCHEMA:
        raise MechanismContractError("training preregistration schema mismatch")
    if payload.get("dataset_a_used") is not False or _contains_dataset_a(payload):
        raise MechanismContractError("training preregistration violates Dataset-A policy")
    if payload.get("training_contract") != TRAINING_PREREGISTRATION:
        raise MechanismContractError("training preregistration settings changed")
    if payload.get("mechanism_gate") != MECHANISM_GATE:
        raise MechanismContractError("training preregistration mechanism gate changed")
    cache_binding = payload.get("feature_cache")
    if not isinstance(cache_binding, Mapping):
        raise MechanismContractError("training preregistration lacks feature_cache binding")
    expected_cache = {
        "root": cache.root.as_posix(),
        "report_sha256": _sha256_file(cache.root / "cache_report.json"),
        "manifest_sha256": _sha256_file(cache.root / "features_manifest.jsonl"),
        "cache_sha256": str(cache.report["cache_sha256"]),
        "splits": list(EXPECTED_SPLITS),
        "final_deferred": True,
        "qwen_config_sha256": cache.report.get("qwen_config_sha256"),
        "feature_extractor_spec_sha256": cache.report.get("feature_extractor_spec_sha256"),
        "campp_model_sha256": cache.report.get("campp_model_sha256"),
        "builder_code": cache.report.get("builder_code"),
        "builder_code_sha256": cache.report.get("builder_code_sha256"),
    }
    if dict(cache_binding) != expected_cache:
        raise MechanismContractError("training preregistration cache binding changed")
    if payload.get("source_sha256") != _source_bindings():
        raise MechanismContractError("training preregistration source SHA changed")
    data_protocol = payload.get("data_protocol")
    if not isinstance(data_protocol, Mapping):
        raise MechanismContractError("training preregistration lacks data protocol binding")
    protocol_path = Path(str(data_protocol.get("path", ""))).resolve(strict=True)
    if _sha256_file(protocol_path) != str(data_protocol.get("sha256", "")):
        raise MechanismContractError("data protocol preregistration SHA changed")
    audit_binding = payload.get("protocol_reaudit")
    if not isinstance(audit_binding, Mapping):
        raise MechanismContractError("training preregistration lacks protocol reaudit binding")
    audit_path = Path(str(audit_binding.get("path", ""))).resolve(strict=True)
    if _sha256_file(audit_path) != str(audit_binding.get("sha256", "")):
        raise MechanismContractError("protocol reaudit report SHA changed")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit_payload, Mapping):
        raise MechanismContractError("protocol reaudit report is not an object")
    if audit_payload.get("schema") != EXPECTED_PROTOCOL_REAUDIT_SCHEMA:
        raise MechanismContractError("protocol reaudit report schema changed")
    if audit_payload.get("audit", {}).get("all_cross_split_overlaps_zero") is not True:
        raise MechanismContractError("protocol reaudit no longer proves zero overlap")
    if audit_payload.get("protocol_preregistration_sha256") != data_protocol.get("sha256"):
        raise MechanismContractError("protocol reaudit/preregistration binding changed")
    audit_code_path = Path(str(audit_binding.get("audit_code", ""))).resolve(strict=True)
    if _sha256_file(audit_code_path) != str(audit_binding.get("audit_code_sha256", "")):
        raise MechanismContractError("protocol audit code SHA changed")
    build_report_path = Path(
        str(audit_binding.get("source_build_report", ""))
    ).resolve(strict=True)
    if _sha256_file(build_report_path) != str(
        audit_binding.get("source_build_report_sha256", "")
    ):
        raise MechanismContractError("protocol source build report SHA changed")
    if audit_binding.get("manifest_sha256") != {
        split: dict(audit_payload.get("manifest_sha256", {}).get(split, {}))
        for split in EXPECTED_SPLITS
    }:
        raise MechanismContractError("protocol reaudit manifest binding changed")
    return payload


def _set_fixed_seed(seed: int) -> None:
    # CUDA >= 10.2 requires this before the first CuBLAS operation when
    # deterministic algorithms are enabled.  It is part of the fixed runtime
    # contract, not a caller-controlled tuning knob.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except TypeError:  # pragma: no cover - older torch compatibility
        torch.use_deterministic_algorithms(True)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    return device


def _group_tensors(group: CounterfactualGroup, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    mixture = torch.from_numpy(group.mixture_features).to(device=device, dtype=torch.float32)
    mixture = mixture.unsqueeze(0).expand(3, -1, -1)
    embedding = torch.from_numpy(np.stack([row.embedding for row in group.rows])).to(device=device, dtype=torch.float32)
    embedding_view2 = torch.from_numpy(np.stack([row.embedding_view2 for row in group.rows])).to(device=device, dtype=torch.float32)
    activity = torch.from_numpy(np.stack([row.target_activity for row in group.rows])).to(device=device, dtype=torch.float32)
    labels = torch.tensor([float(row.target_present) for row in group.rows], device=device, dtype=torch.float32)
    return mixture, embedding, embedding_view2, activity, labels


def _validate_output(output: DACFV3ECSTOutput, batch: int, frames: int) -> None:
    for name in (
        "activity_logits",
        "presence_logits",
        "activity_probability",
        "query_conditioned_frames",
    ):
        if not hasattr(output, name):
            raise MechanismContractError(f"model output lacks {name}")
    if tuple(output.activity_logits.shape) != (batch, frames):
        raise MechanismContractError("model activity_logits shape mismatch")
    if tuple(output.presence_logits.shape) != (batch,):
        raise MechanismContractError("model presence_logits shape mismatch")
    if tuple(output.activity_probability.shape) != (batch, frames):
        raise MechanismContractError("model activity_probability shape mismatch")
    if tuple(output.query_conditioned_frames.shape) != (batch, frames, 128):
        raise MechanismContractError("model query_conditioned_frames shape mismatch")
    for name in (
        "activity_logits",
        "presence_logits",
        "activity_probability",
        "query_conditioned_frames",
    ):
        value = getattr(output, name)
        if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all().item()):
            raise MechanismContractError(f"model output {name} is non-finite/non-floating")


def _train_group(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group: CounterfactualGroup,
    device: torch.device,
    config: MechanismTrainingConfig,
    loss_weights: DACFV3LossWeights,
) -> float:
    model.train()
    mixture, embedding, embedding_view2, activity, labels = _group_tensors(group, device)
    group_index = torch.zeros(3, device=device, dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    main = model(mixture, embedding)
    view2 = model(mixture, embedding_view2)
    _validate_output(main, 3, mixture.shape[-1])
    _validate_output(view2, 3, mixture.shape[-1])
    loss = compute_dacf_v3_loss(
        main,
        view2,
        presence_labels=labels,
        activity_targets=activity,
        group_index=group_index,
        margin=config.counterfactual_margin,
        weights=loss_weights,
    )
    if not bool(torch.isfinite(loss.total).item()):
        raise MechanismContractError("training loss is non-finite")
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    return float(loss.total.detach().cpu().item())


def _binary_auc(labels: Sequence[bool | int | float], scores: Sequence[float]) -> float | None:
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("AUC labels and scores must be one-dimensional and aligned")
    if not np.isfinite(s).all():
        return None
    positive = y > 0
    negative = ~positive
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_scores = s[order]
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _group_query_response(records: Sequence[ScoreRecord]) -> list[float]:
    by_group: dict[str, list[ScoreRecord]] = {}
    for record in records:
        by_group.setdefault(record.group_id, []).append(record)
    values: list[float] = []
    for group_records in by_group.values():
        present = [item.presence_probability for item in group_records if item.target_present]
        absent = [item.presence_probability for item in group_records if not item.target_present]
        if len(present) == 2 and len(absent) == 1:
            values.append(float(np.mean(present) - absent[0]))
    return values


def _same_text_metric(records: Sequence[ScoreRecord], reason: str) -> dict[str, Any]:
    if not records or any(record.same_text_eval is None for record in records):
        return {
            "value": None,
            "status": "deferred",
            "state": "not_constructed",
            "reason": reason,
        }
    subset = [record for record in records if record.same_text_eval]
    value = _binary_auc(
        [record.target_present for record in subset],
        [record.presence_probability for record in subset],
    ) if subset else None
    if value is None:
        return {
            "value": None,
            "status": "deferred",
            "state": "not_constructed",
            "reason": "explicit same-text subset lacks both present and absent labels",
        }
    return {
        "value": float(value),
        "status": "available",
        "state": "constructed_but_not_current_gate",
        "reason": "explicit protocol subset",
    }


def _compute_metrics(records: Sequence[ScoreRecord], same_text_reason: str) -> dict[str, Any]:
    labels = [record.target_present for record in records]
    scores = [record.presence_probability for record in records]
    permutation_scores = [record.permuted_presence_probability for record in records]
    present_records = [record for record in records if record.target_present]
    activity_labels = (
        np.concatenate([record.activity_target for record in present_records])
        if present_records
        else np.asarray([])
    )
    activity_scores = (
        np.concatenate([record.activity_probability for record in present_records])
        if present_records
        else np.asarray([])
    )
    presence_auc = _binary_auc(labels, scores)
    permutation_auc = _binary_auc(labels, permutation_scores)
    query_values = _group_query_response(records)
    present_scores = np.asarray([score for score, label in zip(scores, labels) if label], dtype=np.float64)
    absent_scores = np.asarray([score for score, label in zip(scores, labels) if not label], dtype=np.float64)
    metrics: dict[str, Any] = {
        "presence_auc": presence_auc,
        "activity_frame_auc": _binary_auc(activity_labels > 0.5, activity_scores.tolist()),
        "activity_frame_auc_scope": "present rows only; absent all-zero frames excluded",
        "threshold": THRESHOLD,
        "present_recall": float(np.mean(present_scores >= THRESHOLD)) if present_scores.size else None,
        "absent_rr": float(np.mean(absent_scores < THRESHOLD)) if absent_scores.size else None,
        "query_response": {
            "mean": float(np.mean(query_values)) if query_values else None,
            "group_count": len(query_values),
        },
        "query_permutation_auc_drop": (
            float(presence_auc - permutation_auc)
            if presence_auc is not None and permutation_auc is not None
            else None
        ),
        "same_text_different_speaker_auc": _same_text_metric(records, same_text_reason),
    }
    return metrics


def _metric_value(metrics: Mapping[str, Any], name: str) -> float | None:
    if name == "query_response_mean":
        value = metrics.get("query_response", {}).get("mean")
    elif name == "same_text_different_speaker_auc":
        value = metrics.get(name, {}).get("value")
    else:
        value = metrics.get(name)
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _clone_records(records: Iterable[ScoreRecord], group_suffix: str) -> list[ScoreRecord]:
    return [replace(record, group_id=f"{record.group_id}__{group_suffix}") for record in records]


def _bootstrap_ci(
    records: Sequence[ScoreRecord],
    cluster_field: str,
    *,
    replicates: int,
    seed: int,
    same_text_reason: str,
) -> dict[str, Any]:
    if cluster_field not in {"group_id", "speaker_id"}:
        raise ValueError("cluster_field must be group_id or speaker_id")
    clusters = sorted({getattr(record, cluster_field) for record in records})
    if len(clusters) < 2:
        return {
            "status": "unavailable",
            "reason": f"fewer than two {cluster_field} clusters",
            "clusters": len(clusters),
        }
    by_cluster = {
        cluster: [record for record in records if getattr(record, cluster_field) == cluster]
        for cluster in clusters
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "presence_auc": [],
        "activity_frame_auc": [],
        "present_recall": [],
        "absent_rr": [],
        "query_response_mean": [],
        "query_permutation_auc_drop": [],
        "same_text_different_speaker_auc": [],
    }
    for replicate in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_records: list[ScoreRecord] = []
        for occurrence, cluster in enumerate(sampled):
            sampled_records.extend(_clone_records(by_cluster[str(cluster)], f"b{replicate}_{occurrence}"))
        metrics = _compute_metrics(sampled_records, same_text_reason)
        for name in values:
            metric = _metric_value(metrics, name)
            if metric is not None:
                values[name].append(metric)
    result: dict[str, Any] = {
        "status": "ok",
        "clusters": len(clusters),
        "replicates": replicates,
    }
    for name, observations in values.items():
        if len(observations) < max(20, replicates // 10):
            result[name] = {
                "status": "unavailable",
                "reason": "too few valid bootstrap replicates",
                "valid_replicates": len(observations),
            }
        else:
            result[name] = {
                "status": "ok",
                "lower": float(np.percentile(observations, 2.5)),
                "upper": float(np.percentile(observations, 97.5)),
                "valid_replicates": len(observations),
            }
    return result


def _average_records(first: Sequence[ScoreRecord], second: Sequence[ScoreRecord]) -> list[ScoreRecord]:
    if len(first) != len(second):
        raise MechanismContractError("enrollment view evaluation lengths differ")
    result: list[ScoreRecord] = []
    for left, right in zip(first, second):
        if (left.group_id, left.speaker_id, left.target_present) != (right.group_id, right.speaker_id, right.target_present):
            raise MechanismContractError("enrollment view evaluation ordering differs")
        result.append(
            replace(
                left,
                presence_probability=0.5 * (left.presence_probability + right.presence_probability),
                permuted_presence_probability=0.5 * (
                    left.permuted_presence_probability + right.permuted_presence_probability
                ),
                activity_probability=0.5 * (left.activity_probability + right.activity_probability),
            )
        )
    return result


def _evaluate_split(
    model: nn.Module,
    groups: Sequence[CounterfactualGroup],
    device: torch.device,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    same_text_reason: str,
) -> dict[str, Any]:
    model.eval()
    view_records: dict[str, list[ScoreRecord]] = {"view1": [], "view2": []}
    _assert_cyclic_permutation_foreign(groups)
    ordered_groups = tuple(sorted(groups, key=lambda item: item.group_id))
    permutation_sources = ordered_groups[1:] + ordered_groups[:1]
    with torch.inference_mode():
        for group, permutation_source in zip(ordered_groups, permutation_sources):
            mixture, embedding, embedding_view2, activity, _ = _group_tensors(group, device)
            out1 = model(mixture, embedding)
            out2 = model(mixture, embedding_view2)
            permutation_embedding = torch.from_numpy(
                np.stack([row.embedding for row in permutation_source.rows])
            ).to(device=device, dtype=torch.float32)
            permutation_embedding_view2 = torch.from_numpy(
                np.stack([row.embedding_view2 for row in permutation_source.rows])
            ).to(device=device, dtype=torch.float32)
            permuted1 = model(mixture, permutation_embedding)
            permuted2 = model(mixture, permutation_embedding_view2)
            for output in (out1, out2, permuted1, permuted2):
                _validate_output(output, 3, mixture.shape[-1])
            p1 = torch.sigmoid(out1.presence_logits).detach().cpu().numpy()
            p2 = torch.sigmoid(out2.presence_logits).detach().cpu().numpy()
            pp1 = torch.sigmoid(permuted1.presence_logits).detach().cpu().numpy()
            pp2 = torch.sigmoid(permuted2.presence_logits).detach().cpu().numpy()
            a1 = out1.activity_probability.detach().cpu().numpy()
            a2 = out2.activity_probability.detach().cpu().numpy()
            targets = activity.detach().cpu().numpy()
            for index, row in enumerate(group.rows):
                view_records["view1"].append(
                    ScoreRecord(
                        group_id=group.group_id,
                        speaker_id=row.speaker_id,
                        target_present=row.target_present,
                        presence_probability=float(p1[index]),
                        permuted_presence_probability=float(pp1[index]),
                        activity_probability=np.asarray(a1[index], dtype=np.float64),
                        activity_target=np.asarray(targets[index], dtype=np.float64),
                        same_text_eval=row.same_text_eval,
                    )
                )
                view_records["view2"].append(
                    ScoreRecord(
                        group_id=group.group_id,
                        speaker_id=row.speaker_id,
                        target_present=row.target_present,
                        presence_probability=float(p2[index]),
                        permuted_presence_probability=float(pp2[index]),
                        activity_probability=np.asarray(a2[index], dtype=np.float64),
                        activity_target=np.asarray(targets[index], dtype=np.float64),
                        same_text_eval=row.same_text_eval,
                    )
                )
    view_records["view_averaged"] = _average_records(view_records["view1"], view_records["view2"])
    evaluated: dict[str, Any] = {}
    for offset, view_name in enumerate(("view1", "view2", "view_averaged")):
        records = view_records[view_name]
        metrics = _compute_metrics(records, same_text_reason)
        group_ci = _bootstrap_ci(
            records,
            "group_id",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + offset * 101,
            same_text_reason=same_text_reason,
        )
        speaker_ci = _bootstrap_ci(
            records,
            "speaker_id",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1000 + offset * 101,
            same_text_reason=same_text_reason,
        )
        query_ci = group_ci.get("query_response_mean")
        metrics["query_response"]["bootstrap_ci_95"] = query_ci
        evaluated[view_name] = {
            "metrics": metrics,
            "cluster_bootstrap_ci_95": {
                "group": group_ci,
                "speaker": speaker_ci,
            },
            "row_count": len(records),
            "group_count": len(groups),
            "query_permutation_policy": (
                "sorted dev groups cyclic +1; within-split speaker uniqueness "
                "makes every permuted enrollment absent from the destination mixture"
            ),
        }
    return evaluated


def _gate_checks(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for metric_name, threshold in MECHANISM_GATE.items():
        key = metric_name
        if metric_name == "query_response_mean":
            value = _metric_value(metrics, metric_name)
            check_name = "query_response_mean"
            relation = ">="
        elif metric_name == "query_response_ci_lower":
            query_ci = metrics.get("query_response", {}).get("bootstrap_ci_95")
            value = None if not isinstance(query_ci, Mapping) else _metric_value(query_ci, "lower")
            check_name = "query_response_ci_lower"
            relation = ">"
        else:
            value = _metric_value(metrics, key)
            check_name = metric_name
            relation = ">="
        passed = value is not None and (value > threshold if relation == ">" else value >= threshold)
        checks[check_name] = {
            "value": value,
            "threshold": threshold,
            "relation": relation,
            "passed": bool(passed),
            "status": "ok" if value is not None else "unavailable",
        }
    return checks


def _mechanism_gate(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    checks_by_view = {
        view: _gate_checks(evaluation[view]["metrics"])
        for view in ("view1", "view2")
    }
    passed = all(
        check["passed"]
        for checks in checks_by_view.values()
        for check in checks.values()
    )
    return {
        "threshold": THRESHOLD,
        "fixed_gate": dict(MECHANISM_GATE),
        "gated_views": ["view1", "view2"],
        "checkpoint_selection": "none; dev is observed once after final checkpoint freeze",
        "checks": checks_by_view,
        "passed": bool(passed),
    }


def run_training(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    config: MechanismTrainingConfig | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    allow_prereg_override: bool = False,
    training_preregistration: str | Path | None = None,
) -> dict[str, Any]:
    """Run fixed train-only optimization and dev-only mechanism evaluation."""

    supplied_config = config
    config = config or MechanismTrainingConfig()
    config.validate()
    if not allow_prereg_override and (
        (supplied_config is not None and config != MechanismTrainingConfig())
        or model_factory is not None
    ):
        raise MechanismContractError(
            "formal training accepts only the default preregistered config and DACFV3ECST; "
            "use allow_prereg_override=True only for synthetic tests"
        )
    if not allow_prereg_override and training_preregistration is None:
        raise MechanismContractError(
            "formal training requires an independently written training preregistration"
        )
    if not allow_prereg_override:
        _validate_full_feature_cache_provenance(cache_root)
    started = time.perf_counter()
    cache = load_feature_cache(cache_root)
    external_preregistration: dict[str, Any] | None = None
    external_preregistration_path: Path | None = None
    if not allow_prereg_override:
        if training_preregistration is None:
            raise MechanismContractError(
                "formal training requires an independently written training preregistration"
            )
        external_preregistration_path = Path(training_preregistration).resolve(strict=True)
        external_preregistration = validate_training_preregistration(
            external_preregistration_path, cache
        )
    elif training_preregistration is not None:
        external_preregistration_path = Path(training_preregistration).resolve(strict=True)
        external_preregistration = validate_training_preregistration(
            external_preregistration_path, cache
        )
    if not allow_prereg_override:
        actual_groups = {split: len(cache.groups[split]) for split in EXPECTED_SPLITS}
        if actual_groups != {"train": 96, "dev": 12}:
            raise MechanismContractError(
                f"formal mechanism cache must contain 96 train/12 dev groups, got {actual_groups}"
            )
    device = _resolve_device(config.device)
    _set_fixed_seed(config.seed)
    model = (model_factory or DACFV3ECST)()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return torch.nn.Module")
    model = model.to(device=device, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise MechanismContractError("formal mechanism training requires FP32 model parameters")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_weights = DACFV3LossWeights()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise MechanismContractError(
            f"formal training output must be new or empty: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "final_checkpoint.pt"
    loss_trace: list[float] = []
    train_groups = cache.groups["train"]
    if len(train_groups) != config.groups_per_epoch:
        raise MechanismContractError(
            "train group count does not match the fixed groups_per_epoch contract: "
            f"{len(train_groups)} != {config.groups_per_epoch}"
        )
    shuffle_rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        order = list(range(len(train_groups)))
        shuffle_rng.shuffle(order)
        for group_index in order:
            group = train_groups[group_index]
            loss_trace.append(_train_group(model, optimizer, group, device, config, loss_weights))
    if len(loss_trace) != config.fixed_updates:
        raise MechanismContractError("completed update count differs from fixed epoch contract")
    deterministic_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    training_preregistration_expected: dict[str, Any] = dict(TRAINING_PREREGISTRATION)
    training_preregistration_actual: dict[str, Any] = {
        "seed": config.seed,
        "epochs": config.epochs,
        "groups_per_epoch": len(train_groups),
        "groups_per_update": 1,
        "updates": config.fixed_updates,
        "one_complete_abc_group_per_step": True,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip_norm": config.grad_clip_norm,
        "precision": "FP32",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": deterministic_algorithms,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "model_architecture": model.__class__.__name__,
        "loss_weights": asdict(loss_weights),
        "counterfactual_margin": config.counterfactual_margin,
        "scheduler": "none",
        "early_stop": False,
        "hyperparameter_scan": False,
        "checkpoint": "epoch_20_final_only" if config.epochs == 20 else "fixed_final_only_override",
        "dev_evaluation_count": 1,
        "threshold": THRESHOLD,
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_seed_offset": 7001,
        "query_permutation_policy": "sorted dev groups cyclic +1; all source speakers absent from destination mixture",
    }
    training_preregistration_matches = (
        training_preregistration_actual == training_preregistration_expected
    )
    if not allow_prereg_override and not training_preregistration_matches:
        raise MechanismContractError(
            "runtime training contract differs from the fixed preregistration"
        )
    # The final training state is the only checkpoint.  No dev forward pass
    # happens above this boundary, so the checkpoint cannot be selected by dev.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": config.fixed_updates,
            "epoch": config.epochs,
            "seed": config.seed,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "model_class": model.__class__.__name__,
        },
        checkpoint_path,
    )
    if not checkpoint_path.is_file():
        raise MechanismContractError("final checkpoint was not produced")
    checkpoint_sha_before_dev = _sha256_file(checkpoint_path)
    checkpoint_frozen_before_dev = True
    dev_evaluation_count = 0
    evaluation = _evaluate_split(
        model,
        cache.groups["dev"],
        device,
        bootstrap_replicates=config.bootstrap_replicates,
        bootstrap_seed=config.seed + 7001,
        same_text_reason=str(cache.same_text_status["reason"]),
    )
    dev_evaluation_count += 1
    checkpoint_sha_after_dev = _sha256_file(checkpoint_path)
    if checkpoint_sha_after_dev != checkpoint_sha_before_dev:
        raise MechanismContractError("checkpoint changed during the one dev evaluation")
    gate = _mechanism_gate(evaluation)
    verdict = "conditional-GO" if gate["passed"] else "implementation-NO-GO"
    report: dict[str, Any] = {
        "schema": "dacf-v3-mechanism-training-report-v0.2",
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "threshold": THRESHOLD,
        "same_text_metric": {
            "status": "deferred",
            "state": "not_constructed",
            "value": None,
            "current_mechanism_gate": False,
            "threshold_before_qwen": SAME_TEXT_GATE_THRESHOLD,
            "enforcement_stage": "before Qwen integration",
            "required_suite": "real home-command hard-negative suite with same-text/different-speaker swaps",
            "reason": "current AISHELL protocol does not construct same-text/different-speaker speaker swaps",
        },
        "dev_evaluation_count": dev_evaluation_count,
        "checkpoint_frozen_before_dev": checkpoint_frozen_before_dev,
        "verdict_scope": {
            "scope": "mechanism-only",
            "cer": "not measured",
            "official_negative_rr": "not measured",
            "rtf": "not measured",
            "qwen_integration": "not run",
            "final_opened": False,
        },
        "cache": {
            "root": cache.root.as_posix(),
            "cache_sha256": str(cache.report["cache_sha256"]),
            "manifest_sha256": str(cache.report["manifest_sha256"]),
            "splits": list(EXPECTED_SPLITS),
            "final_deferred": True,
            "dataset_a_used": False,
            "groups": {split: len(cache.groups[split]) for split in EXPECTED_SPLITS},
        },
        "training_preregistration": {
            "protocol_version": PROTOCOL_VERSION,
            "expected": training_preregistration_expected,
            "actual": training_preregistration_actual,
            "matches_default_contract": training_preregistration_matches,
            "independent_preregistration_must_match": True,
            "external_path": (
                external_preregistration_path.as_posix()
                if external_preregistration_path is not None
                else None
            ),
            "external_sha256": (
                _sha256_file(external_preregistration_path)
                if external_preregistration_path is not None
                else None
            ),
            "external_validated": external_preregistration is not None,
        },
        "training": {
            "config": asdict(config),
            "device": str(device),
            "seed": config.seed,
            "optimizer": "AdamW",
            "precision": "FP32",
            "gradient_clip_norm": config.grad_clip_norm,
            "loss_weights": asdict(loss_weights),
            "updates_completed": config.fixed_updates,
            "fixed_updates": config.fixed_updates,
            "epochs": config.epochs,
            "groups_per_epoch": config.groups_per_epoch,
            "groups_per_update": 1,
            "final_epoch": config.epochs,
            "checkpoint_policy": CHECKPOINT_POLICY,
            "scheduler": "none",
            "early_stop": False,
            "hyperparameter_scan": False,
            "dev_evaluation_count": dev_evaluation_count,
            "checkpoint_frozen_before_dev": checkpoint_frozen_before_dev,
            "threshold": THRESHOLD,
            "loss_first": loss_trace[0],
            "loss_last": loss_trace[-1],
            "full_group_sampling": True,
            "query_role_id_used_as_model_input": False,
        },
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "exists": checkpoint_path.is_file(),
            "model_class": model.__class__.__name__,
            "sha256_before_dev": checkpoint_sha_before_dev,
            "sha256_after_dev": checkpoint_sha_after_dev,
            "frozen_before_dev": checkpoint_frozen_before_dev,
        },
        "dev": evaluation["view_averaged"],
        "dev_views": {
            "view1": evaluation["view1"],
            "view2": evaluation["view2"],
        },
        "mechanism_gate": gate,
        "same_text_different_speaker": dict(cache.same_text_status),
        "runtime_sec": float(time.perf_counter() - started),
    }
    report_path = output / "mechanism_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--training-preregistration")
    parser.add_argument("--protocol-preregistration")
    parser.add_argument("--protocol-audit-report")
    parser.add_argument("--write-preregistration")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.write_preregistration:
        if not args.protocol_preregistration or not args.protocol_audit_report:
            raise SystemExit(
                "--protocol-preregistration and --protocol-audit-report are required "
                "when writing preregistration"
            )
        if args.output_dir or args.training_preregistration:
            raise SystemExit(
                "preregistration mode does not accept --output-dir/--training-preregistration"
            )
        report = write_training_preregistration(
            args.cache_root,
            args.protocol_preregistration,
            args.protocol_audit_report,
            args.write_preregistration,
        )
    else:
        if not args.output_dir or not args.training_preregistration:
            raise SystemExit(
                "formal training requires --output-dir and --training-preregistration"
            )
        if args.protocol_preregistration or args.protocol_audit_report:
            raise SystemExit(
                "--protocol-preregistration/--protocol-audit-report are only used in "
                "preregistration mode"
            )
        report = run_training(
            args.cache_root,
            args.output_dir,
            training_preregistration=args.training_preregistration,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CAMPP_DIM",
    "CounterfactualGroup",
    "EXPECTED_ROLES",
    "EXPECTED_SPLITS",
    "FeatureCache",
    "MechanismContractError",
    "MechanismTrainingConfig",
    "MECHANISM_GATE",
    "TRAINING_PREREGISTRATION",
    "TRAINING_PREREGISTRATION_SCHEMA",
    "CHECKPOINT_POLICY",
    "PROTOCOL_VERSION",
    "SAME_TEXT_GATE_THRESHOLD",
    "ScoreRecord",
    "_binary_auc",
    "_compute_metrics",
    "cache_payload_sha256",
    "load_feature_cache",
    "run_training",
    "sha256_array",
    "validate_training_preregistration",
    "write_training_preregistration",
]


if __name__ == "__main__":
    raise SystemExit(main())
