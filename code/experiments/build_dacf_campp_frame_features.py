"""Build an auditable, frozen CAM++ frame-feature capacity probe.

The source model is the local CAM++ ONNX file used by the existing D9
features.  CAM++'s statistics-pooling input is exposed by a small dynamic
protobuf descriptor: the source ONNX is never modified.  The modified graph
has two outputs, the original 512-dimensional embedding and the pre-pooling
``[B, 512, T]`` tensor.  The latter is stored as ``[T, 512]``.

The sherpa-onnx stream remains the single source of truth for CAM++ fbank
features.  ``get_frames(0, max(1, num_samples // 160))`` is reshaped from its
flat ``n * 80`` return value to ``[n, 80]`` and fed directly to ORT.  This
deliberately drops at most one partial 160-sample tail frame because sherpa's
C++ binding aborts on an out-of-range ``get_frames`` request.  No training,
threshold selection, validation-window search, Dataset-A data, or query-role
feature input is present in this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16_000
FBANK_DIM = 80
PREPOOL_DIM = 512
FINAL_DIM = 512
FRAME_SHIFT_SAMPLES = 160
TOP25_FRACTION = 0.25
EXPECTED_TRAIN_GROUPS = 24
EXPECTED_VAL_GROUPS = 8
EXPECTED_SCALE_GROUPS = {"train": 48, "val": 16, "final": 16}
ROLE_IDS = (0, 1, 2)
PRESENT_ROLE_IDS = (0, 1)
ABSENT_ROLE_ID = 2
BACKEND = "sherpa-campp-onnx-prepool"
PREPOOL_OUTPUT_NAME = "onnx::ReduceMean_4602"
DEFAULT_MODEL = Path("E:/hf_cache/campplus/campplus.onnx")


_DATASET_A_MARKERS = (
    "dataset-a",
    "dataset_a",
    "dataseta",
    "test_wav/dataset",
    "test_wav\\dataset",
)
_MIXTURE_KEYS = ("recognition_audio", "mixture_audio", "mixture_path")
_ENROLLMENT_KEYS = (
    "enrollment_audio",
    "clean_enrollment_audio",
    "enrollment_clean_audio",
    "enrollment_path",
)
_ENROLLMENT_VIEW2_KEYS = (
    "enrollment_audio_view2",
    "noisy_enrollment_audio",
    "enrollment_noisy_audio",
    "enrollment_view2_audio",
    "view2_audio",
    "enrollment_audio2",
)


class ManifestContractError(ValueError):
    """Raised before audio access when a DACF manifest is not auditable."""


class CamppInterfaceError(RuntimeError):
    """Raised when the local protobuf/ORT/sherpa interface cannot be used."""


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _looks_like_dataset_a(value: Any) -> bool:
    text = _path_text(value).casefold()
    if any(marker in text for marker in _DATASET_A_MARKERS):
        return True
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return "dataseta" in compact


def _assert_not_dataset_a(value: Any, *, field: str = "path") -> None:
    if _looks_like_dataset_a(value):
        raise ManifestContractError(
            f"{field} contains forbidden Dataset-A path/marker: {value}"
        )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return bool(value)


def _guard_dataset_a(value: Any, field: str) -> None:
    """Recursively reject Dataset-A paths and explicit use flags."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_field = f"{field}.{key}"
            if str(key).casefold() in {
                "dataset_a_used",
                "dataset_a",
                "used_dataset_a",
            } and _as_bool(child):
                raise ManifestContractError(
                    f"{child_field}=true is forbidden: Dataset-A may not enter features"
                )
            _guard_dataset_a(child, child_field)
        return
    if isinstance(value, (list, tuple, set)):
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
    candidates = (
        (raw_path,) if raw_path.is_absolute() else (Path.cwd() / raw_path, manifest_path.parent / raw_path)
    )
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
    declared = {"A": "present_A", "B": "present_B", "C": "absent_C"}.get(
        str(row.get("query_role", "")).strip(), str(row.get("query_role", "")).strip()
    )
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
        "enrollment_utt",
        "target_src",
        "target_audio",
        "interferer_srcs",
        "mixture_sources",
        "noise_src",
    ):
        for value in _iter_values(row.get(key)):
            if value is None or not str(value).strip():
                continue
            text = str(value).strip()
            if re.search(r"[\\/]", text) or Path(text).suffix:
                values.add(str(Path(text).resolve(strict=False)).casefold())
            else:
                values.add(text.casefold())
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
        _first_path(row, _ENROLLMENT_VIEW2_KEYS, field="enrollment_audio_view2")
        group = _group_id(row)
        groups[group].append(dict(row))
        speakers.update(_speaker_ids(row))
        sources.update(_source_ids(row))

    for group, group_rows in groups.items():
        if len(group_rows) != 3:
            raise ManifestContractError(
                f"{split} group {group!r} must contain exactly A/B/C rows, got {len(group_rows)}"
            )
        roles = [_role_id(row) for row in group_rows]
        if sorted(roles) != [0, 1, 2]:
            raise ManifestContractError(
                f"{split} group {group!r} must contain role ids 0,1,2, got {roles}"
            )
        hashes = {str(row.get("mixture_sha256", "")).strip().casefold() for row in group_rows}
        if "" in hashes or len(hashes) != 1:
            raise ManifestContractError(
                f"{split} group {group!r} has inconsistent/missing mixture_sha256"
            )
    return groups, speakers, sources


def validate_manifests(
    train_manifest: str | Path,
    val_manifest: str | Path,
    *,
    final_manifest: str | Path | None = None,
    expected_train_groups: Optional[int] = EXPECTED_TRAIN_GROUPS,
    expected_val_groups: Optional[int] = EXPECTED_VAL_GROUPS,
    expected_final_groups: Optional[int] = None,
) -> dict[str, Any]:
    """Validate A/B/C manifests and every split pair before opening audio."""

    manifest_inputs: dict[str, str | Path] = {
        "train": train_manifest,
        "val": val_manifest,
    }
    if final_manifest is not None:
        manifest_inputs["final"] = final_manifest
    expected = {
        "train": expected_train_groups,
        "val": expected_val_groups,
        "final": expected_final_groups,
    }
    bundle: dict[str, Any] = {}
    for split, manifest in manifest_inputs.items():
        path = Path(manifest).resolve(strict=True)
        _assert_not_dataset_a(path, field=f"{split}-manifest")
        rows = _read_jsonl(path)
        groups, speakers, sources = _validate_split_rows(rows, split=split)
        required = expected[split]
        if required is not None and len(groups) != required:
            raise ManifestContractError(
                f"fixed {split} group contract requires {required}, got {len(groups)}"
            )
        bundle[split] = {
            "path": path,
            "rows": rows,
            "groups": groups,
            "speakers": speakers,
            "sources": sources,
        }

    pairwise_overlap: dict[str, dict[str, list[str]]] = {}
    splits = tuple(bundle)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            key = f"{left}/{right}"
            overlap = {
                "group": sorted(
                    set(bundle[left]["groups"]) & set(bundle[right]["groups"])
                ),
                "speaker": sorted(
                    bundle[left]["speakers"] & bundle[right]["speakers"]
                ),
                "source": sorted(
                    bundle[left]["sources"] & bundle[right]["sources"]
                ),
            }
            pairwise_overlap[key] = overlap
            for kind, values in overlap.items():
                if values:
                    raise ManifestContractError(
                        f"{key} {kind} overlap is forbidden: {values[:8]}"
                    )

    train_val = pairwise_overlap.get(
        "train/val", {"group": [], "speaker": [], "source": []}
    )
    bundle["audit"] = {
        **{
            f"{split}_{name}": len(bundle[split][name])
            for split in splits
            for name in ("rows", "groups", "speakers", "sources")
        },
        "speaker_overlap": train_val["speaker"],
        "source_overlap": train_val["source"],
        "group_overlap": train_val["group"],
        "pairwise_overlap": pairwise_overlap,
        "dataset_a_used": False,
    }
    return bundle


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_audio_bytes(data: bytes, *, source: Path) -> tuple[np.ndarray, int]:
    try:
        import io

        waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:  # pragma: no cover - decoder-specific diagnostic
        raise RuntimeError(f"cannot decode audio bytes from {source}") from exc
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise RuntimeError(f"audio is empty or not mono-decodable: {source}")
    if int(sample_rate) != SAMPLE_RATE:
        raise RuntimeError(
            f"CAM++ exact-fbank probe requires 16 kHz audio, got {sample_rate} for {source}"
        )
    return np.ascontiguousarray(waveform), int(sample_rate)


def _verify_mixture_group(
    group: Sequence[Mapping[str, Any]], manifest_path: Path
) -> tuple[str, Path, bytes]:
    actual_hashes: list[str] = []
    paths: list[Path] = []
    byte_values: list[bytes] = []
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
        byte_values.append(raw)
    if len(set(actual_hashes)) != 1 or len(set(byte_values)) != 1:
        raise ManifestContractError(
            f"three DACF rows do not share byte-identical mixture: {actual_hashes}"
        )
    return actual_hashes[0], paths[0], byte_values[0]


def _verify_optional_audio_sha(row: Mapping[str, Any], fields: Sequence[str], actual: str) -> None:
    for field in fields:
        declared = row.get(field)
        if declared is not None and str(declared).strip():
            if str(declared).strip().casefold() != actual:
                raise ManifestContractError(
                    f"{field} mismatch for row {row.get('id')}: declared={declared}, actual={actual}"
                )


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("._") or "item"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _normalize_vector(value: Any, expected_dim: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != expected_dim:
        raise CamppInterfaceError(
            f"CAM++ vector dimension {vector.size}, expected {expected_dim}"
        )
    if not np.isfinite(vector).all():
        raise CamppInterfaceError("CAM++ vector contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise CamppInterfaceError("CAM++ vector is near zero")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def _normalize_frame_rows(value: Any, expected_dim: int = PREPOOL_DIM) -> np.ndarray:
    frames = np.asarray(value, dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != expected_dim or frames.shape[0] <= 0:
        raise CamppInterfaceError(
            f"pre-pool feature must be [T,{expected_dim}], got {frames.shape}"
        )
    if not np.isfinite(frames).all():
        raise CamppInterfaceError("pre-pool features contain non-finite values")
    return np.ascontiguousarray(frames, dtype=np.float32)


def _add_field(message: Any, name: str, number: int, field_type: int, *, repeated: bool = False, type_name: str = "") -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = 3 if repeated else 1  # LABEL_REPEATED / LABEL_OPTIONAL
    field.type = field_type
    if type_name:
        field.type_name = type_name


def _onnx_model_class() -> Any:
    """Return a minimal dynamic ONNX ModelProto class without importing onnx."""

    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "dacf_minimal_onnx.proto"
    file_proto.package = "onnx"
    file_proto.syntax = "proto3"

    operator_set = file_proto.message_type.add()
    operator_set.name = "OperatorSetIdProto"
    _add_field(operator_set, "domain", 1, 9)
    _add_field(operator_set, "version", 2, 3)

    dimension = file_proto.message_type.add()
    dimension.name = "TensorShapeProto_Dimension"
    _add_field(dimension, "dim_value", 1, 3)
    _add_field(dimension, "dim_param", 2, 9)
    _add_field(dimension, "denotation", 3, 9)

    shape = file_proto.message_type.add()
    shape.name = "TensorShapeProto"
    _add_field(shape, "dim", 1, 11, repeated=True, type_name=".onnx.TensorShapeProto_Dimension")
    _add_field(shape, "denotation", 2, 9, repeated=True)

    type_proto = file_proto.message_type.add()
    type_proto.name = "TypeProto"
    tensor_type = type_proto.nested_type.add()
    tensor_type.name = "Tensor"
    _add_field(tensor_type, "elem_type", 1, 5)
    _add_field(tensor_type, "shape", 2, 11, type_name=".onnx.TensorShapeProto")
    _add_field(type_proto, "tensor_type", 1, 11, type_name=".onnx.TypeProto.Tensor")
    _add_field(type_proto, "denotation", 6, 9)

    value_info = file_proto.message_type.add()
    value_info.name = "ValueInfoProto"
    _add_field(value_info, "name", 1, 9)
    _add_field(value_info, "type", 2, 11, type_name=".onnx.TypeProto")

    node = file_proto.message_type.add()
    node.name = "NodeProto"
    _add_field(node, "input", 1, 9, repeated=True)
    _add_field(node, "output", 2, 9, repeated=True)
    _add_field(node, "name", 3, 9)
    _add_field(node, "op_type", 4, 9)

    graph = file_proto.message_type.add()
    graph.name = "GraphProto"
    _add_field(graph, "node", 1, 11, repeated=True, type_name=".onnx.NodeProto")
    _add_field(graph, "name", 2, 9)
    _add_field(graph, "doc_string", 10, 9)
    _add_field(graph, "input", 11, 11, repeated=True, type_name=".onnx.ValueInfoProto")
    _add_field(graph, "output", 12, 11, repeated=True, type_name=".onnx.ValueInfoProto")
    _add_field(graph, "value_info", 13, 11, repeated=True, type_name=".onnx.ValueInfoProto")

    model = file_proto.message_type.add()
    model.name = "ModelProto"
    _add_field(model, "ir_version", 1, 3)
    _add_field(model, "producer_name", 2, 9)
    _add_field(model, "producer_version", 3, 9)
    _add_field(model, "domain", 4, 9)
    _add_field(model, "model_version", 5, 3)
    _add_field(model, "doc_string", 6, 9)
    _add_field(model, "graph", 7, 11, type_name=".onnx.GraphProto")
    _add_field(model, "opset_import", 8, 11, repeated=True, type_name=".onnx.OperatorSetIdProto")

    pool = descriptor_pool.DescriptorPool()
    try:
        pool.Add(file_proto)
    except Exception as exc:  # pragma: no cover - protobuf version diagnostic
        raise CamppInterfaceError(f"cannot build minimal ONNX protobuf descriptor: {exc}") from exc
    descriptor = pool.FindMessageTypeByName("onnx.ModelProto")
    getter = getattr(message_factory, "GetMessageClass", None)
    if getter is not None:
        return getter(descriptor)
    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


def add_prepool_output(original_path: str | Path, modified_path: str | Path) -> dict[str, Any]:
    """Copy an ONNX graph and append the fixed CAM++ pre-pool tensor output."""

    source = Path(original_path).resolve(strict=True)
    target = Path(modified_path).resolve()
    _assert_not_dataset_a(source, field="CAM++ source model")
    _assert_not_dataset_a(target, field="CAM++ modified model")
    if source == target:
        raise CamppInterfaceError("modified ONNX path must differ from original path")
    model_class = _onnx_model_class()
    model = model_class()
    try:
        model.ParseFromString(source.read_bytes())
    except Exception as exc:
        raise CamppInterfaceError(f"cannot parse CAM++ ONNX with protobuf: {source}") from exc
    if not model.HasField("graph"):
        raise CamppInterfaceError("CAM++ ONNX has no graph")

    candidate = ""
    for node in model.graph.node:
        if PREPOOL_OUTPUT_NAME in node.output:
            candidate = PREPOOL_OUTPUT_NAME
            break
        if node.name == PREPOOL_OUTPUT_NAME and node.output:
            candidate = str(node.output[0])
            break
    if not candidate:
        raise CamppInterfaceError(
            f"CAM++ graph does not contain the expected pre-pool node/output {PREPOOL_OUTPUT_NAME!r}"
        )
    existing_outputs = {str(output.name) for output in model.graph.output}
    if candidate not in existing_outputs:
        output = model.graph.output.add()
        output.name = candidate
        output.type.tensor_type.elem_type = 1  # TensorProto.FLOAT
        dimensions = output.type.tensor_type.shape.dim
        dimensions.add().dim_param = "B"
        dimensions.add().dim_value = PREPOOL_DIM
        dimensions.add().dim_param = "T"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(model.SerializeToString())
    return {
        "original_path": str(source),
        "original_sha256": sha256_file(source),
        "modified_path": str(target),
        "modified_sha256": sha256_file(target),
        "prepool_output_name": candidate,
        "prepool_layout": "[B,512,T] -> [T,512]",
        "source_unchanged": True,
    }


def _frame_output_to_time_major(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 1:
        raise CamppInterfaceError(f"pre-pool ORT output must be [1,C,T] or [1,T,C], got {array.shape}")
    array = array[0]
    if array.shape[0] == PREPOOL_DIM:
        return _normalize_frame_rows(array.T)
    if array.shape[1] == PREPOOL_DIM:
        return _normalize_frame_rows(array)
    raise CamppInterfaceError(f"cannot identify 512 feature axis in pre-pool output {array.shape}")


def safe_fbank_frame_count(num_samples: int) -> int:
    """Return the sherpa-safe floor frame count for a non-empty waveform."""

    samples = int(num_samples)
    if samples <= 0:
        raise CamppInterfaceError("cannot derive fbank frames from empty audio")
    # ceil() can ask sherpa for a partial tail frame and abort in C++; this
    # floor rule discards at most one incomplete 160-sample tail.
    return max(1, samples // FRAME_SHIFT_SAMPLES)


def _call_get_frames(stream: Any, frame_count: int) -> np.ndarray:
    if not hasattr(stream, "get_frames"):
        raise CamppInterfaceError("sherpa stream has no get_frames(frame_index, n) API")
    flat = np.asarray(stream.get_frames(0, int(frame_count)), dtype=np.float32).reshape(-1)
    expected = int(frame_count) * FBANK_DIM
    if flat.size != expected:
        raise CamppInterfaceError(
            f"sherpa get_frames returned {flat.size} values, expected {expected} for {frame_count} frames"
        )
    return np.ascontiguousarray(flat.reshape(int(frame_count), FBANK_DIM), dtype=np.float32)


class CamppFrameBackend:
    """Frozen ORT CAM++ graph plus sherpa exact-fbank stream."""

    def __init__(self, original_model: str | Path, modified_model: str | Path, *, num_threads: int = 2):
        try:
            import onnxruntime as ort
            import sherpa_onnx
        except Exception as exc:  # pragma: no cover - environment-specific
            raise CamppInterfaceError(
                "CAM++ frame backend requires existing onnxruntime and sherpa_onnx; no installation is attempted"
            ) from exc
        self._sherpa = sherpa_onnx
        self._original_model = Path(original_model).resolve(strict=True)
        self._modified_model = Path(modified_model).resolve(strict=True)
        try:
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=str(self._original_model), num_threads=int(num_threads), debug=False
                )
            )
            self._session = ort.InferenceSession(
                str(self._modified_model), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # pragma: no cover - environment-specific
            raise CamppInterfaceError(f"cannot load local CAM++ backend: {exc}") from exc
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise CamppInterfaceError(f"expected one CAM++ ORT input, got {len(inputs)}")
        self._input_name = inputs[0].name
        outputs = self._session.get_outputs()
        names = [str(item.name) for item in outputs]
        if len(outputs) < 2:
            raise CamppInterfaceError("modified CAM++ graph must expose final and pre-pool outputs")
        self._prepool_index = next(
            (index for index, name in enumerate(names) if name == PREPOOL_OUTPUT_NAME), None
        )
        if self._prepool_index is None:
            self._prepool_index = 1
        self._embedding_index = next(
            (index for index, item in enumerate(outputs) if len(item.shape) == 2), 0
        )
        self.calls = 0

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if signal.size <= 0:
            raise CamppInterfaceError("cannot extract CAM++ features from empty audio")
        if int(sample_rate) != SAMPLE_RATE:
            raise CamppInterfaceError(f"CAM++ exact-fbank input must be {SAMPLE_RATE} Hz")
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(signal))
        stream.input_finished()
        frame_count = safe_fbank_frame_count(signal.size)
        fbank = _call_get_frames(stream, frame_count)
        try:
            outputs = self._session.run(None, {self._input_name: fbank[None, :, :]})
        except Exception as exc:  # pragma: no cover - ORT-specific diagnostic
            raise CamppInterfaceError(
                f"CAM++ modified graph failed on exact sherpa fbank {fbank.shape}: {exc}"
            ) from exc
        final_embedding = _normalize_vector(outputs[self._embedding_index], FINAL_DIM)
        prepool = _frame_output_to_time_major(outputs[self._prepool_index])
        self.calls += 1
        return {
            "prepool": prepool,
            "embedding": final_embedding,
            "fbank": fbank,
            "fbank_frame_count": np.asarray(frame_count, dtype=np.int64),
        }


def _l2_row_normalize(frames: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    if np.any(norms <= 1e-8) or not np.isfinite(norms).all():
        raise CamppInterfaceError("pre-pool frame contains a near-zero/non-finite row")
    return np.asarray(frames / norms, dtype=np.float32)


def capacity_scores(query_prepool: np.ndarray, mixture_prepool: np.ndarray) -> dict[str, float]:
    """Compute fixed, untrained whole/frame/top25 pre-pool cosine scores."""

    query = _normalize_frame_rows(query_prepool).mean(axis=0)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 1e-8:
        raise CamppInterfaceError("pooled query pre-pool feature is near zero")
    query = query / query_norm
    mixture_rows = _l2_row_normalize(_normalize_frame_rows(mixture_prepool))
    mixture_pool = mixture_rows.mean(axis=0)
    mixture_pool = mixture_pool / max(float(np.linalg.norm(mixture_pool)), 1e-8)
    frame_scores = mixture_rows @ query
    top_k = max(1, int(frame_scores.size * TOP25_FRACTION))
    top25 = float(np.sort(frame_scores)[-top_k:].mean())
    return {
        "whole_pre": float(np.dot(query, mixture_pool)),
        "frame_max": float(frame_scores.max()),
        "top25_pre": top25,
        "top_k": float(top_k),
        "mixture_frame_count": float(frame_scores.size),
    }


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        raise ValueError("present-vs-absent ROC-AUC requires both classes")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += float(pos > neg) + 0.5 * float(pos == neg)
    return float(wins / (len(positive) * len(negative)))


def _capacity_metrics(records: Sequence[Mapping[str, Any]], *, query_key: str) -> dict[str, Any]:
    positive: dict[str, list[float]] = defaultdict(list)
    negative: dict[str, list[float]] = defaultdict(list)
    for record in records:
        scores = capacity_scores(record[query_key], record["mixture_prepool"])
        target = positive if int(record["query_role_id"]) in PRESENT_ROLE_IDS else negative
        for key in ("whole_pre", "frame_max", "top25_pre"):
            target[key].append(float(scores[key]))
    return {
        "query_view": query_key,
        "positive_count": len(positive["whole_pre"]),
        "absent_count": len(negative["whole_pre"]),
        "whole_pre_auc": _roc_auc(positive["whole_pre"], negative["whole_pre"]),
        "frame_max_auc": _roc_auc(positive["frame_max"], negative["frame_max"]),
        "top25_pre_auc": _roc_auc(positive["top25_pre"], negative["top25_pre"]),
        "whole_pooled_cosine_auc": _roc_auc(
            positive["whole_pre"], negative["whole_pre"]
        ),
        "top25_fraction_fixed": TOP25_FRACTION,
        "role_ordering_audit": _role_ordering_audit(records, query_key=query_key),
        "interpretation": (
            "frozen untrained pre-pool cosine capacity only; no threshold, window, "
            "model-selection, CER, RR, or submission claim"
        ),
    }


def _role_ordering_audit(
    records: Sequence[Mapping[str, Any]], *, query_key: str
) -> dict[str, Any]:
    """Report fixed A/B/C ordering without using role as a feature input."""

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for record in records:
        group_id = str(record.get("group_id", "")).strip()
        if not group_id:
            raise ManifestContractError("role ordering audit requires group_id")
        role_id = int(record["query_role_id"])
        score = capacity_scores(record[query_key], record["mixture_prepool"])
        grouped[group_id][role_id] = float(score["top25_pre"])
    if not grouped or any(set(roles) != set(ROLE_IDS) for roles in grouped.values()):
        raise ManifestContractError("role ordering audit requires exactly A/B/C per group")
    per_metric: dict[str, dict[str, Any]] = {}
    for metric in ("whole_pre", "frame_max", "top25_pre"):
        ordered: dict[str, dict[int, float]] = defaultdict(dict)
        for record in records:
            group_id = str(record["group_id"])
            role_id = int(record["query_role_id"])
            score = capacity_scores(record[query_key], record["mixture_prepool"])
            ordered[group_id][role_id] = float(score[metric])
        lower_a = sum(values[ABSENT_ROLE_ID] < values[0] for values in ordered.values())
        lower_b = sum(values[ABSENT_ROLE_ID] < values[1] for values in ordered.values())
        lower_both = sum(
            values[ABSENT_ROLE_ID] < values[0] and values[ABSENT_ROLE_ID] < values[1]
            for values in ordered.values()
        )
        count = len(ordered)
        per_metric[metric] = {
            "c_lower_than_a_count": int(lower_a),
            "c_lower_than_a_fraction": float(lower_a / count),
            "c_lower_than_b_count": int(lower_b),
            "c_lower_than_b_fraction": float(lower_b / count),
            "c_lower_than_both_count": int(lower_both),
            "c_lower_than_both_fraction": float(lower_both / count),
        }
    return {
        "query_view": query_key,
        "group_count": len(grouped),
        "per_metric": per_metric,
        "interpretation": "descriptive A/B/C ordering only; no role-conditioned feature or threshold",
    }


def _view_agreement(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("enrollment dual-view agreement requires records")
    pre_scores = []
    final_scores = []
    for record in records:
        left = _normalize_frame_rows(record["enrollment_prepool"]).mean(axis=0)
        right = _normalize_frame_rows(record["view2_prepool"]).mean(axis=0)
        left = left / max(float(np.linalg.norm(left)), 1e-8)
        right = right / max(float(np.linalg.norm(right)), 1e-8)
        pre_scores.append(float(np.dot(left, right)))
        final_scores.append(
            float(
                np.dot(
                    _normalize_vector(record["enrollment_embedding"], FINAL_DIM),
                    _normalize_vector(record["view2_embedding"], FINAL_DIM),
                )
            )
        )
    pre = np.asarray(pre_scores, dtype=np.float64)
    final = np.asarray(final_scores, dtype=np.float64)
    return {
        "count": int(pre.size),
        "prepool_cosine_mean": float(pre.mean()),
        "prepool_cosine_median": float(np.median(pre)),
        "prepool_cosine_p10": float(np.percentile(pre, 10)),
        "prepool_cosine_min": float(pre.min()),
        "final_embedding_cosine_mean": float(final.mean()),
        "final_embedding_cosine_median": float(np.median(final)),
        "final_embedding_cosine_min": float(final.min()),
        "interpretation": "same-speaker enrollment dual-view agreement; not a threshold",
    }


def _final_embedding_same_different_auc(
    records: Sequence[Mapping[str, Any]],
    *,
    left_key: str = "enrollment_embedding",
    right_key: str = "view2_embedding",
    query_view: str = "clean enrollment final embedding vs enrollment view2 final embedding",
) -> dict[str, Any]:
    """Audit one final-embedding view against a second view.

    Each row contributes one same-speaker pair (its clean enrollment versus
    its view2 enrollment).  Every cross-row pair is a different-speaker pair;
    the speaker id is used only for this audit label, never by the backend.
    """

    positive: list[float] = []
    negative: list[float] = []
    for left in records:
        left_embedding = _normalize_vector(left[left_key], FINAL_DIM)
        left_speaker = str(left.get("query_speaker_id", "")).strip()
        if not left_speaker:
            raise ManifestContractError(
                "final embedding same/different audit requires query_speaker_id"
            )
        for right in records:
            right_embedding = _normalize_vector(right[right_key], FINAL_DIM)
            score = float(np.dot(left_embedding, right_embedding))
            if left_speaker == str(right.get("query_speaker_id", "")).strip():
                positive.append(score)
            else:
                negative.append(score)
    return {
        "same_speaker_count": len(positive),
        "different_speaker_count": len(negative),
        "auc": _roc_auc(positive, negative),
        "query_view": query_view,
        "interpretation": (
            "speaker-disjoint same/different embedding audit; speaker ids are audit labels only"
        ),
    }


def _load_d9_index(d9_root: Optional[Path]) -> dict[str, Any]:
    if d9_root is None or not d9_root.exists():
        return {"rows": {}, "mixtures": {}}
    manifest = d9_root / "features_manifest.jsonl"
    index: dict[str, Any] = {"rows": {}, "mixtures": {}}
    if not manifest.exists():
        return index
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("campp_features", {})
        row_id = str(row.get("id", ""))
        if not row_id:
            continue
        index["rows"][row_id] = {
            "enrollment": d9_root / str(meta.get("enrollment_embedding_npy", "")),
            "view2": d9_root / str(meta.get("enrollment_view2_embedding_npy", "")),
            "mixture": d9_root / str(meta.get("mixture_feature_npz", "")),
        }
        mixture_sha = str(meta.get("mixture_audio_sha256_actual", row.get("mixture_sha256", "")))
        if mixture_sha:
            index["mixtures"][mixture_sha.casefold()] = d9_root / str(
                meta.get("mixture_feature_npz", "")
            )
    return index


def _load_d9_vector(path: Any) -> Optional[np.ndarray]:
    candidate = Path(path) if path is not None else None
    if candidate is None or not candidate.exists() or not candidate.is_file():
        return None
    return _normalize_vector(np.load(candidate, allow_pickle=False), FINAL_DIM)


def _compare_d9_vector(value: np.ndarray, path: Path) -> Optional[dict[str, float]]:
    if not path.exists() or not path.is_file():
        return None
    reference = np.load(path, allow_pickle=False)
    left = _normalize_vector(value, FINAL_DIM)
    right = _normalize_vector(reference, FINAL_DIM)
    return {
        "cosine": float(np.dot(left, right)),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def _compare_d9_mixture(value: np.ndarray, path: Path) -> Optional[dict[str, float]]:
    if not path.exists() or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "mixture_embedding" not in data:
            return None
        reference = np.asarray(data["mixture_embedding"], dtype=np.float32)
    if not reference.size:
        return None
    left = _normalize_vector(value, FINAL_DIM)
    right = _normalize_vector(reference, FINAL_DIM)
    return {
        "cosine": float(np.dot(left, right)),
        "max_abs": float(np.max(np.abs(left - right))),
    }


def _summarize_d9(values: Sequence[Mapping[str, float]], *, root: Optional[Path]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "root": str(root) if root else None}
    cosine = np.asarray([float(item["cosine"]) for item in values], dtype=np.float64)
    max_abs = np.asarray([float(item["max_abs"]) for item in values], dtype=np.float64)
    return {
        "count": int(cosine.size),
        "root": str(root) if root else None,
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_p10": float(np.percentile(cosine, 10)),
        "max_abs_max": float(max_abs.max()),
        "interpretation": "new final 512d embedding versus existing D9 NPY/NPZ, no retraining",
    }


def _default_d9_root() -> Optional[Path]:
    root = Path(__file__).resolve().parents[2]
    candidate = root / "code" / "runs" / "dacf_counterfactual_probe32_campp_20260806"
    return candidate if candidate.exists() else None


def build_feature_dataset(
    train_manifest: str | Path,
    val_manifest: str | Path,
    *,
    final_manifest: str | Path | None = None,
    model_path: str | Path = DEFAULT_MODEL,
    output_dir: str | Path,
    d9_root: str | Path | None = None,
    backend: Any | None = None,
    backend_factory: Optional[Callable[[Path, Path, int], Any]] = None,
    num_threads: int = 2,
) -> dict[str, Any]:
    """Extract the fixed mini-G2 or preregistered 48/16/16 feature layer."""

    started = time.perf_counter()
    model = Path(model_path).resolve(strict=True)
    _assert_not_dataset_a(model, field="CAM++ model")
    output = Path(output_dir).resolve()
    _assert_not_dataset_a(output, field="feature output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mixture").mkdir(parents=True, exist_ok=True)
    (output / "query").mkdir(parents=True, exist_ok=True)
    (output / "model").mkdir(parents=True, exist_ok=True)

    if final_manifest is None:
        bundle = validate_manifests(train_manifest, val_manifest)
    else:
        bundle = validate_manifests(
            train_manifest,
            val_manifest,
            final_manifest=final_manifest,
            expected_train_groups=EXPECTED_SCALE_GROUPS["train"],
            expected_val_groups=EXPECTED_SCALE_GROUPS["val"],
            expected_final_groups=EXPECTED_SCALE_GROUPS["final"],
        )
    split_names = tuple(split for split in ("train", "val", "final") if split in bundle)
    graph_info: dict[str, Any]
    modified_model = output / "model" / "campplus_frame.onnx"
    if backend is None:
        graph_info = add_prepool_output(model, modified_model)
        if backend_factory is not None:
            backend = backend_factory(model, modified_model, int(num_threads))
        else:
            backend = CamppFrameBackend(model, modified_model, num_threads=num_threads)
    else:
        graph_info = {
            "backend_injected": True,
            "original_path": str(model),
            "original_sha256": sha256_file(model),
            "modified_path": None,
            "source_unchanged": True,
        }

    d9_path = Path(d9_root).resolve() if d9_root is not None else _default_d9_root()
    if d9_path is not None:
        _assert_not_dataset_a(d9_path, field="D9 feature root")
    d9_index = _load_d9_index(d9_path)
    d9_values: dict[str, list[dict[str, float]]] = {
        "enrollment_audio": [],
        "enrollment_audio_view2": [],
        "mixture": [],
    }
    d9_metric_records: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_names
    }

    stats: MutableMapping[str, int] = {
        "backend_calls": 0,
        "mixture_feature_count": 0,
        "enrollment_feature_count": 0,
        "enrollment_view2_feature_count": 0,
    }
    mixture_cache: dict[str, dict[str, Any]] = {}
    metric_records: dict[str, list[dict[str, Any]]] = {
        split: [] for split in split_names
    }
    enhanced_rows: list[dict[str, Any]] = []
    split_runtime: dict[str, float] = {}

    def extract(waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        result = backend(waveform, sample_rate)
        if not isinstance(result, Mapping):
            raise CamppInterfaceError("CAM++ backend must return a mapping")
        prepool = _normalize_frame_rows(result["prepool"], PREPOOL_DIM)
        embedding = _normalize_vector(result["embedding"], FINAL_DIM)
        raw_fbank_count = result.get("fbank_frame_count", prepool.shape[0])
        fbank_frame_count = int(np.asarray(raw_fbank_count).reshape(-1)[0])
        if fbank_frame_count <= 0:
            raise CamppInterfaceError("CAM++ backend returned a non-positive fbank frame count")
        stats["backend_calls"] += 1
        return {
            "prepool": prepool,
            "embedding": embedding,
            "fbank_frame_count": np.asarray(fbank_frame_count, dtype=np.int64),
        }

    for split in split_names:
        split_started = time.perf_counter()
        split_info = bundle[split]
        manifest_path: Path = split_info["path"]
        for group_id in sorted(split_info["groups"]):
            group = split_info["groups"][group_id]
            mixture_sha, mixture_path, mixture_bytes = _verify_mixture_group(group, manifest_path)
            if mixture_sha not in mixture_cache:
                mixture_wave, mixture_sr = _read_audio_bytes(mixture_bytes, source=mixture_path)
                mixture_features = extract(mixture_wave, mixture_sr)
                mixture_file = output / "mixture" / f"{_safe_name(split + '_' + group_id)}_{mixture_sha[:12]}.npz"
                np.savez_compressed(
                    mixture_file,
                    prepool=mixture_features["prepool"],
                    mixture_embedding=mixture_features["embedding"],
                    mixture_sha256=np.asarray(mixture_sha),
                    frame_count=np.asarray(mixture_features["prepool"].shape[0], dtype=np.int64),
                    fbank_frame_count=mixture_features["fbank_frame_count"],
                    feature_dim=np.asarray(PREPOOL_DIM, dtype=np.int32),
                    sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int32),
                    frame_shift_samples=np.asarray(FRAME_SHIFT_SAMPLES, dtype=np.int32),
                )
                mixture_cache[mixture_sha] = {
                    **mixture_features,
                    "path": mixture_file,
                    "owner": f"{split}:{group_id}",
                    "mixture_audio_sha256": mixture_sha,
                }
                stats["mixture_feature_count"] += 1
                d9_mixture = d9_index["mixtures"].get(mixture_sha.casefold())
                if d9_mixture is not None:
                    comparison = _compare_d9_mixture(mixture_features["embedding"], d9_mixture)
                    if comparison is not None:
                        d9_values["mixture"].append(comparison)
            mixture = mixture_cache[mixture_sha]

            for row in sorted(group, key=_role_id):
                row_id = _safe_name(row.get("id", f"{split}_{group_id}_{_role_id(row)}"))
                view1_path = _resolve_path(
                    _first_path(row, _ENROLLMENT_KEYS, field="enrollment_audio"),
                    manifest_path,
                    field="enrollment_audio",
                )
                view2_path = _resolve_path(
                    _first_path(row, _ENROLLMENT_VIEW2_KEYS, field="enrollment_audio_view2"),
                    manifest_path,
                    field="enrollment_audio_view2",
                )
                view1_bytes = view1_path.read_bytes()
                view2_bytes = view2_path.read_bytes()
                view1_sha = _sha256_bytes(view1_bytes)
                view2_sha = _sha256_bytes(view2_bytes)
                _verify_optional_audio_sha(row, ("enrollment_sha256", "clean_enrollment_sha256"), view1_sha)
                _verify_optional_audio_sha(
                    row,
                    ("enrollment_view2_sha256", "noisy_enrollment_sha256"),
                    view2_sha,
                )
                view1_wave, view1_sr = _read_audio_bytes(view1_bytes, source=view1_path)
                view2_wave, view2_sr = _read_audio_bytes(view2_bytes, source=view2_path)
                view1_features = extract(view1_wave, view1_sr)
                view2_features = extract(view2_wave, view2_sr)
                stats["enrollment_feature_count"] += 1
                stats["enrollment_view2_feature_count"] += 1

                base = output / "query" / f"{split}_{row_id}"
                view1_pre = base.with_name(base.name + "__enrollment_prepool.npy")
                view1_final = base.with_name(base.name + "__enrollment_embedding.npy")
                view2_pre = base.with_name(base.name + "__view2_prepool.npy")
                view2_final = base.with_name(base.name + "__view2_embedding.npy")
                np.save(view1_pre, view1_features["prepool"])
                np.save(view1_final, view1_features["embedding"])
                np.save(view2_pre, view2_features["prepool"])
                np.save(view2_final, view2_features["embedding"])

                d9_row = d9_index["rows"].get(str(row.get("id", "")), {})
                for key, value, fallback in (
                    ("enrollment_audio", view1_features["embedding"], base.with_name(base.name + "__enrollment.npy")),
                    ("enrollment_audio_view2", view2_features["embedding"], base.with_name(base.name + "__view2.npy")),
                ):
                    reference = d9_row.get("enrollment" if key == "enrollment_audio" else "view2", fallback)
                    comparison = _compare_d9_vector(value, reference)
                    if comparison is not None:
                        d9_values[key].append(comparison)

                role_id = _role_id(row)
                metric_records[split].append(
                    {
                        "group_id": group_id,
                        "query_speaker_id": str(row.get("query_speaker_id", "")),
                        "query_role_id": role_id,
                        "enrollment_prepool": view1_features["prepool"],
                        "view2_prepool": view2_features["prepool"],
                        "enrollment_embedding": view1_features["embedding"],
                        "view2_embedding": view2_features["embedding"],
                        "mixture_prepool": mixture["prepool"],
                    }
                )
                d9_enrollment = _load_d9_vector(d9_row.get("enrollment"))
                d9_view2 = _load_d9_vector(d9_row.get("view2"))
                if d9_enrollment is not None and d9_view2 is not None:
                    d9_metric_records[split].append(
                        {
                            "query_speaker_id": str(row.get("query_speaker_id", "")),
                            "enrollment_embedding": d9_enrollment,
                            "view2_embedding": d9_view2,
                        }
                    )
                enhanced = dict(row)
                enhanced["campp_frame_features"] = {
                    "schema": "dacf-campp-frame-features-v0.1",
                    "backend": BACKEND,
                    "feature_dim": PREPOOL_DIM,
                    "final_embedding_dim": FINAL_DIM,
                    "mixture_audio_sha256_actual": mixture_sha,
                    "mixture_feature_npz": _relative(mixture["path"], output),
                    "mixture_feature_owner": mixture["owner"],
                    "mixture_feature_reused_across_counterfactual_rows": True,
                    "mixture_prepool_shape": [int(value) for value in mixture["prepool"].shape],
                    "mixture_fbank_frame_count": int(mixture["fbank_frame_count"]),
                    "mixture_final_embedding_npy": _relative(mixture["path"], output) + "::mixture_embedding",
                    "enrollment_prepool_npy": _relative(view1_pre, output),
                    "enrollment_final_embedding_npy": _relative(view1_final, output),
                    "enrollment_view2_prepool_npy": _relative(view2_pre, output),
                    "enrollment_view2_final_embedding_npy": _relative(view2_final, output),
                    "enrollment_audio_sha256_actual": view1_sha,
                    "enrollment_view2_audio_sha256_actual": view2_sha,
                    "enrollment_prepool_shape": [int(value) for value in view1_features["prepool"].shape],
                    "enrollment_view2_prepool_shape": [int(value) for value in view2_features["prepool"].shape],
                    "enrollment_fbank_frame_count": int(view1_features["fbank_frame_count"]),
                    "enrollment_view2_fbank_frame_count": int(view2_features["fbank_frame_count"]),
                    "fbank_frame_shift_samples": FRAME_SHIFT_SAMPLES,
                    "query_role_id_used_as_feature_input": False,
                    "query_role_id_is_audit_label_only": True,
                    "capacity_scores_enrollment": capacity_scores(
                        view1_features["prepool"], mixture["prepool"]
                    ),
                    "capacity_scores_enrollment_view2": capacity_scores(
                        view2_features["prepool"], mixture["prepool"]
                    ),
                    "d9_final_embedding_reference": str(d9_row.get("enrollment")) if d9_row.get("enrollment") else None,
                }
                enhanced_rows.append(enhanced)
        split_runtime[split] = time.perf_counter() - split_started

    enhanced_manifest = output / "features_manifest.jsonl"
    with enhanced_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in enhanced_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metrics: dict[str, Any] = {}
    for split in split_names:
        metrics[split] = {
            "enrollment_audio": _capacity_metrics(
                metric_records[split], query_key="enrollment_prepool"
            ),
            "enrollment_audio_view2": _capacity_metrics(
                metric_records[split], query_key="view2_prepool"
            ),
            "enrollment_view_agreement": _view_agreement(metric_records[split]),
            "clean_enrollment_final_embedding_same_vs_different": (
                _final_embedding_same_different_auc(metric_records[split])
            ),
            "d9_clean_enrollment_final_embedding_same_vs_different": (
                _final_embedding_same_different_auc(
                    d9_metric_records[split],
                    query_view="D9 clean enrollment final embedding vs D9 enrollment view2 final embedding",
                )
                if d9_metric_records[split]
                else {
                    "same_speaker_count": 0,
                    "different_speaker_count": 0,
                    "auc": None,
                    "query_view": "D9 clean enrollment final embedding vs D9 enrollment view2 final embedding",
                    "interpretation": "D9 reference files unavailable",
                }
            ),
        }

    report: dict[str, Any] = {
        "schema": "dacf-campp-frame-feature-report-v0.1",
        "backend": BACKEND,
        "verdict": "implementation-NO-GO",
        "verdict_scope": (
            "raw CAM++ pre-pool capacity implementation only; not a direction-NO-GO "
            "and not a CER/RR/RTF/integration claim"
        ),
        "dataset_a_policy": "hard reject path/flag; not read",
        "dataset_a_used": False,
        "training_allowed": False,
        "threshold_selection_allowed": False,
        "validation_window_tuning_allowed": False,
        "model": graph_info,
        "inputs": {
            "train_manifest": str(bundle["train"]["path"]),
            "val_manifest": str(bundle["val"]["path"]),
            **(
                {"final_manifest": str(bundle["final"]["path"])}
                if "final" in bundle
                else {}
            ),
        },
        "audit": bundle["audit"],
        "counts": {
            **{key: int(value) for key, value in stats.items()},
            "input_rows": len(enhanced_rows),
            "train_groups": len(bundle["train"]["groups"]),
            "val_groups": len(bundle["val"]["groups"]),
            **(
                {"final_groups": len(bundle["final"]["groups"])}
                if "final" in bundle
                else {}
            ),
            "groups": sum(len(bundle[split]["groups"]) for split in split_names),
            "unique_mixtures": len(mixture_cache),
            "feature_manifest_rows": len(enhanced_rows),
        },
        "fbank_contract": {
            "source": "sherpa_onnx.SpeakerEmbeddingExtractor stream.get_frames",
            "sample_rate": SAMPLE_RATE,
            "frame_shift_samples": FRAME_SHIFT_SAMPLES,
            "frame_count_rule": "max(1, num_samples//160)",
            "partial_tail_policy": (
                "discard at most one partial 160-sample tail frame; never request it from sherpa"
            ),
            "shape_after_reshape": "[T,80]",
            "direct_ort_input": True,
            "query_role_id_used_as_feature_input": False,
        },
        "prepool_contract": {
            "source_node": PREPOOL_OUTPUT_NAME,
            "ort_layout": "[B,512,T]",
            "saved_layout": "[T,512]",
            "top25_rule": "mean of the largest max(1, floor(0.25*T)) frame cosines",
            "query_role_id_used_as_feature_input": False,
        },
        "capacity_metrics": metrics,
        "d9_final_embedding_comparison": {
            "root": str(d9_path) if d9_path else None,
            "enrollment_audio": _summarize_d9(d9_values["enrollment_audio"], root=d9_path),
            "enrollment_audio_view2": _summarize_d9(
                d9_values["enrollment_audio_view2"], root=d9_path
            ),
            "mixture": _summarize_d9(d9_values["mixture"], root=d9_path),
        },
        "runtime_sec": float(time.perf_counter() - started),
        "split_runtime_sec": {key: float(value) for key, value in split_runtime.items()},
        "limitations": [
            "whole_pre/frame_max/top25_pre are frozen untrained cosine capacity metrics only.",
            "top25 fraction is fixed at 0.25; no validation tuning or operating threshold is used.",
            "No Dataset-A, optimizer, CER, rejection-rate, or submission result is measured here.",
            "A/B/C rows reuse one byte-verified mixture feature artifact per SHA256.",
            "D9 comparison is an embedding parity check, not a new training signal.",
        ],
        "artifacts": {
            "enhanced_manifest": _relative(enhanced_manifest, output),
            "report": "report.json",
            "modified_onnx": _relative(Path(graph_info["modified_path"]), output)
            if graph_info.get("modified_path")
            else None,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_failure_report(output_dir: str | Path, error: BaseException, *, model_path: str | Path) -> Path:
    """Write a small diagnostic report when the frozen local interface is unavailable."""

    output = Path(output_dir).resolve()
    _assert_not_dataset_a(output, field="failure output")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report = {
        "schema": "dacf-campp-frame-feature-report-v0.1",
        "backend": BACKEND,
        "status": "blocked",
        "verdict": "implementation-NO-GO",
        "verdict_scope": "local frozen CAM++ frame-feature interface only",
        "dataset_a_used": False,
        "training_allowed": False,
        "threshold_selection_allowed": False,
        "model_path": str(Path(model_path).resolve(strict=False)),
        "error": {"type": type(error).__name__, "message": str(error)},
        "next_action": "inspect the local ORT/sherpa/protobuf interface; do not install or download dependencies",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument(
        "--final-manifest",
        default=None,
        help="preregistered final holdout; when present enforces fixed 48/16/16 counts",
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--output",
        default="code/runs/dacf_campplus_frame_features_20260806",
    )
    parser.add_argument(
        "--d9-root",
        default=None,
        help="Existing D9 feature root; omitted uses the local 24/8 CAM++ D9 output when present",
    )
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = build_feature_dataset(
            args.train_manifest,
            args.val_manifest,
            final_manifest=args.final_manifest,
            model_path=args.model,
            output_dir=args.output,
            d9_root=args.d9_root,
            num_threads=args.num_threads,
        )
    except Exception as exc:  # CLI keeps a diagnostic report for interface failures.
        path = write_failure_report(args.output, exc, model_path=args.model)
        print(f"CAM++ frame probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"diagnostic report: {path}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
