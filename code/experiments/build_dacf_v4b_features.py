"""Build the provenance-locked DACF-v4b train/dev feature cache.

DACF-v4b deliberately reuses every speaker across six counterfactual
environments.  This builder therefore validates a *global* source-WAV ledger
and the 2A+2B+2C role-rotation contract instead of inheriting the v3
one-environment-per-speaker assumption.  It extracts one exact Qwen log-mel
tensor per byte-identical mixture and two CAM++ enrollment views per query.

The official final/test split is not an input to this program.  Query roles
and role ids are retained only as provenance metadata; downstream membership
labels must be recomputed as ``query speaker in destination mixture``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from build_dacf_v3_features import (
    CAMPP_DIM,
    FEATURE_SIZE,
    REQUIRED_FEATURE_SPEC,
    SAMPLE_RATE,
    FeatureContractError,
    _align_tail,
    _feature_spec_hash,
    _load_activity,
    _load_preprocessor_config,
    _make_real_camp_extractor,
    _payload_sha256,
    _read_audio,
    _sha256_array,
    _sha256_file,
    _sha256_json,
    extract_camp_embedding,
    extract_qwen_features,
    make_qwen_feature_extractor,
    validate_feature_extractor_spec,
)


SCHEMA = "dacf-v4b-feature-cache-v0.1"
REPORT_SCHEMA = "dacf-v4b-feature-cache-report-v0.1"
MANIFEST_SCHEMA = "dacf-v4b-query-manifest-v0.1"
AUDIO_SCHEMA = "dacf-v4b-role-rotation-audio-v0.1"
SOURCE_CORPUS = "AISHELL-1"
DEFAULT_SOURCE_ROOT = Path("E:/midea_datasets/data_aishell")
ROLES = ("present_A", "present_B", "absent_C")
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}
ROLE_SHORT = {"present_A": "A", "present_B": "B", "absent_C": "C"}
EXPECTED_GROUPS = {"train": 96, "dev": 24}
EXPECTED_SPEAKERS = {"train": 48, "dev": 12}
DATASET_A_MARKERS = ("dataset-a", "dataset_a", "dataseta")


@dataclass(frozen=True)
class PreparedRow:
    split: str
    row_id: str
    group_id: str
    round_index: int
    role: str
    role_id: int
    query_speaker_id: str
    query_speaker_label: int
    mixture_speakers: tuple[str, str]
    mixture_path: Path
    mixture_sha256: str
    enrollment_path: Path
    enrollment_sha256: str
    enrollment_view2_path: Path
    enrollment_view2_sha256: str
    activity_path: Path
    activity_sha256: str
    clean_target_path: Path
    clean_target_sha256: str
    enrollment_source_path: Path
    enrollment_source_sha256: str
    enrollment_view2_source_path: Path
    enrollment_view2_source_sha256: str
    mixture_source_paths: tuple[Path, Path]
    mixture_source_sha256: tuple[str, str]
    manifest_path: Path
    original_row: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FeatureContractError(f"JSON root must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FeatureContractError(f"{path}:{index} is not a JSON object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _contains_dataset_a(value: Any) -> bool:
    text = str(value).replace("\\", "/").casefold()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return any(marker in text for marker in DATASET_A_MARKERS) or "dataseta" in compact


def _assert_no_symlink(path: Path, *, field: str) -> None:
    candidate = path.absolute()
    for item in (candidate, *candidate.parents):
        if item.exists() and item.is_symlink():
            raise FeatureContractError(f"{field} contains a symlink component: {item}")


def _resolve_file(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FeatureContractError(f"{field} requires a file path")
    if _contains_dataset_a(value):
        raise FeatureContractError(f"{field} contains a forbidden Dataset-A marker")
    if ".." in Path(str(value)).parts:
        raise FeatureContractError(f"{field} contains forbidden '..'")
    raw = Path(value)
    _assert_no_symlink(raw, field=field)
    path = raw.resolve(strict=True)
    if not path.is_file():
        raise FeatureContractError(f"{field} is not a file: {path}")
    return path


def _resolve_dir(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FeatureContractError(f"{field} requires a directory path")
    if _contains_dataset_a(value):
        raise FeatureContractError(f"{field} contains a forbidden Dataset-A marker")
    raw = Path(value)
    _assert_no_symlink(raw, field=field)
    path = raw.resolve(strict=True)
    if not path.is_dir():
        raise FeatureContractError(f"{field} is not a directory: {path}")
    return path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise FeatureContractError(f"{label} requires {field}")
    return str(value).strip()


def _declared_sha(row: Mapping[str, Any], field: str, actual: str, *, label: str) -> str:
    declared = _required_text(row, field, label=label).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise FeatureContractError(f"{label}.{field} is not a SHA256")
    if declared != actual.casefold():
        raise FeatureContractError(
            f"{label}.{field} mismatch: declared={declared}, actual={actual}"
        )
    return declared


def _source_file(
    value: Any,
    declared_sha: Any,
    *,
    field: str,
    split: str,
    speaker: str,
    source_root: Path,
    sha_cache: MutableMapping[Path, str],
) -> tuple[Path, str]:
    path = _resolve_file(value, field=field)
    if not _is_under(path, source_root):
        raise FeatureContractError(f"{field} is outside the AISHELL source root: {path}")
    marker = f"/wav/{split}/"
    if marker not in path.as_posix():
        raise FeatureContractError(f"{field} is outside AISHELL official {split}: {path}")
    if path.parent.name != speaker:
        raise FeatureContractError(
            f"{field} parent speaker {path.parent.name!r} != {speaker!r}"
        )
    actual = sha_cache.setdefault(path, _sha256_file(path))
    declared = str(declared_sha).strip().casefold()
    if declared != actual:
        raise FeatureContractError(
            f"{field} source SHA mismatch: declared={declared}, actual={actual}"
        )
    return path, actual


def _prepare_row(
    row: Mapping[str, Any],
    *,
    split: str,
    manifest_path: Path,
    source_root: Path,
    source_sha_cache: MutableMapping[Path, str],
) -> PreparedRow:
    row_id = _required_text(row, "id", label=f"{manifest_path}")
    label = f"{manifest_path}:{row_id}"
    if row.get("schema_version") != MANIFEST_SCHEMA:
        raise FeatureContractError(f"{label} schema_version changed")
    if row.get("protocol_version") != AUDIO_SCHEMA:
        raise FeatureContractError(f"{label} protocol_version changed")
    if row.get("split") != split or row.get("protocol_split") != split:
        raise FeatureContractError(f"{label} split/protocol_split mismatch")
    if row.get("source_split") != split:
        raise FeatureContractError(f"{label} source_split mismatch")
    if row.get("source_corpus") != SOURCE_CORPUS:
        raise FeatureContractError(f"{label} source_corpus changed")
    # ``dataset_a_policy`` intentionally contains the corpus name in a
    # prohibition sentence.  Guard actual booleans and every path at the path
    # resolver instead of stringifying the whole metadata object.
    if bool(row.get("dataset_a_used")):
        raise FeatureContractError(f"{label} violates Dataset-A firewall")
    if row.get("query_role_id_model_input") is not False:
        raise FeatureContractError(f"{label} query_role_id_model_input must be false")

    role = _required_text(row, "query_role", label=label)
    if role not in ROLE_TO_ID:
        raise FeatureContractError(f"{label} invalid query_role={role!r}")
    role_id = int(row.get("query_role_id", -1))
    if role_id != ROLE_TO_ID[role]:
        raise FeatureContractError(f"{label} query_role/query_role_id mismatch")
    if bool(row.get("target_present")) != (role != "absent_C"):
        raise FeatureContractError(f"{label} target_present disagrees with role")
    if int(row.get("target_present_label", -1)) != int(role != "absent_C"):
        raise FeatureContractError(f"{label} target_present_label disagrees with role")

    group_id = _required_text(row, "base_mixture_id", label=label)
    query_speaker = _required_text(row, "query_speaker_id", label=label)
    if row.get("enrollment_spk") != query_speaker:
        raise FeatureContractError(f"{label} enrollment_spk disagrees with query speaker")
    if row.get("enrollment_view2_spk") != query_speaker:
        raise FeatureContractError(f"{label} view2 speaker disagrees with query speaker")
    if int(row.get("enrollment_view_count", 0)) != 2:
        raise FeatureContractError(f"{label} must contain exactly two enrollment views")
    if row.get("identity_positive") is not True:
        raise FeatureContractError(f"{label} identity_positive contract is missing")
    if row.get("enrollment_views_distinct_source_wavs") is not True:
        raise FeatureContractError(f"{label} distinct-source view contract is missing")
    if int(row.get("global_source_path_use_count", 0)) != 1:
        raise FeatureContractError(f"{label} global source path count changed")
    if int(row.get("global_source_sha_use_count", 0)) != 1:
        raise FeatureContractError(f"{label} global source SHA count changed")

    mixture_speakers_raw = row.get("mixture_speakers")
    if not isinstance(mixture_speakers_raw, dict):
        raise FeatureContractError(f"{label} mixture_speakers must be a mapping")
    speaker_a = str(mixture_speakers_raw.get("A", "")).strip()
    speaker_b = str(mixture_speakers_raw.get("B", "")).strip()
    if not speaker_a or not speaker_b or speaker_a == speaker_b:
        raise FeatureContractError(f"{label} invalid mixture speaker pair")
    if role == "present_A" and query_speaker != speaker_a:
        raise FeatureContractError(f"{label} present_A query identity mismatch")
    if role == "present_B" and query_speaker != speaker_b:
        raise FeatureContractError(f"{label} present_B query identity mismatch")
    if role == "absent_C" and query_speaker in {speaker_a, speaker_b}:
        raise FeatureContractError(f"{label} absent_C speaker is present in mixture")

    mixture_path = _resolve_file(row.get("recognition_audio"), field=f"{label}.recognition_audio")
    mixture_actual_sha = _sha256_file(mixture_path)
    mixture_sha = _declared_sha(row, "mixture_sha256", mixture_actual_sha, label=label)
    enrollment_path = _resolve_file(row.get("enrollment_audio"), field=f"{label}.enrollment_audio")
    enrollment_sha = _declared_sha(
        row, "enrollment_sha256", _sha256_file(enrollment_path), label=label
    )
    view2_path = _resolve_file(
        row.get("enrollment_audio_view2"), field=f"{label}.enrollment_audio_view2"
    )
    view2_sha = _declared_sha(
        row, "enrollment_view2_sha256", _sha256_file(view2_path), label=label
    )
    if enrollment_path == view2_path or enrollment_sha == view2_sha:
        raise FeatureContractError(f"{label} enrollment views are not byte-distinct")

    activity_path = _resolve_file(row.get("target_activity"), field=f"{label}.target_activity")
    activity_sha = _sha256_file(activity_path)
    activity = _load_activity(activity_path)
    clean_path = _resolve_file(row.get("target_audio"), field=f"{label}.target_audio")
    clean_sha = _sha256_file(clean_path)
    if role == "absent_C":
        if np.any(activity != 0):
            raise FeatureContractError(f"{label} absent_C activity is not zero")
        clean_wave, _ = _read_audio(clean_path)
        if np.any(np.abs(clean_wave) > 1e-7):
            raise FeatureContractError(f"{label} absent_C clean target is not blank")
        if row.get("clean_target_is_empty") is not True:
            raise FeatureContractError(f"{label} absent_C empty-target audit changed")
    elif not np.any(activity > 0):
        raise FeatureContractError(f"{label} present activity is empty")

    enrollment_source, enrollment_source_sha = _source_file(
        row.get("enrollment_src"),
        row.get("enrollment_src_sha256"),
        field=f"{label}.enrollment_src",
        split=split,
        speaker=query_speaker,
        source_root=source_root,
        sha_cache=source_sha_cache,
    )
    view2_source, view2_source_sha = _source_file(
        row.get("enrollment_src_view2"),
        row.get("enrollment_src_view2_sha256"),
        field=f"{label}.enrollment_src_view2",
        split=split,
        speaker=query_speaker,
        source_root=source_root,
        sha_cache=source_sha_cache,
    )
    if enrollment_source == view2_source or enrollment_source_sha == view2_source_sha:
        raise FeatureContractError(f"{label} enrollment source WAVs are not distinct")

    mixture_sources_raw = row.get("mixture_sources")
    mixture_hashes_raw = row.get("mixture_source_sha256")
    if not isinstance(mixture_sources_raw, dict) or not isinstance(mixture_hashes_raw, dict):
        raise FeatureContractError(f"{label} mixture source lineage is incomplete")
    source_a, source_a_sha = _source_file(
        mixture_sources_raw.get("A"),
        mixture_hashes_raw.get("A"),
        field=f"{label}.mixture_sources.A",
        split=split,
        speaker=speaker_a,
        source_root=source_root,
        sha_cache=source_sha_cache,
    )
    source_b, source_b_sha = _source_file(
        mixture_sources_raw.get("B"),
        mixture_hashes_raw.get("B"),
        field=f"{label}.mixture_sources.B",
        split=split,
        speaker=speaker_b,
        source_root=source_root,
        sha_cache=source_sha_cache,
    )
    if source_a == source_b or source_a_sha == source_b_sha:
        raise FeatureContractError(f"{label} mixture source WAVs are not distinct")

    expected_target = {"present_A": source_a, "present_B": source_b, "absent_C": None}[role]
    raw_target = row.get("target_src")
    if expected_target is None:
        if raw_target not in (None, ""):
            raise FeatureContractError(f"{label} absent_C unexpectedly has target_src")
    elif _resolve_file(raw_target, field=f"{label}.target_src") != expected_target:
        raise FeatureContractError(f"{label} target_src does not bind to mixture membership")

    return PreparedRow(
        split=split,
        row_id=row_id,
        group_id=group_id,
        round_index=int(row.get("round_index", -1)),
        role=role,
        role_id=role_id,
        query_speaker_id=query_speaker,
        query_speaker_label=int(row.get("query_speaker_label", -1)),
        mixture_speakers=(speaker_a, speaker_b),
        mixture_path=mixture_path,
        mixture_sha256=mixture_sha,
        enrollment_path=enrollment_path,
        enrollment_sha256=enrollment_sha,
        enrollment_view2_path=view2_path,
        enrollment_view2_sha256=view2_sha,
        activity_path=activity_path,
        activity_sha256=activity_sha,
        clean_target_path=clean_path,
        clean_target_sha256=clean_sha,
        enrollment_source_path=enrollment_source,
        enrollment_source_sha256=enrollment_source_sha,
        enrollment_view2_source_path=view2_source,
        enrollment_view2_source_sha256=view2_source_sha,
        mixture_source_paths=(source_a, source_b),
        mixture_source_sha256=(source_a_sha, source_b_sha),
        manifest_path=manifest_path,
        original_row=dict(row),
    )


def validate_input_bundle(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    audio_build_report: str | Path,
    *,
    allowed_source_root: str | Path = DEFAULT_SOURCE_ROOT,
    expected_groups: Optional[Mapping[str, int]] = None,
) -> tuple[list[PreparedRow], dict[str, Any]]:
    """Validate v4b audio, role rotation, and the global source ledger."""

    source_root = _resolve_dir(allowed_source_root, field="allowed_source_root")
    report_path = _resolve_file(audio_build_report, field="audio_build_report")
    report = _read_json(report_path)
    if report.get("schema") != AUDIO_SCHEMA:
        raise FeatureContractError("audio build report schema changed")
    if bool(report.get("dataset_a_used")) or bool(report.get("official_test_loaded")):
        raise FeatureContractError("audio report violates Dataset-A/final firewall")
    if report.get("loaded_splits") != ["train", "dev"] or report.get("final_deferred") is not True:
        raise FeatureContractError("audio report must bind train/dev with final deferred")

    manifests = {
        "train": _resolve_file(train_manifest, field="train_manifest"),
        "dev": _resolve_file(dev_manifest, field="dev_manifest"),
    }
    expected = dict(EXPECTED_GROUPS if expected_groups is None else expected_groups)
    all_rows: list[PreparedRow] = []
    source_sha_cache: MutableMapping[Path, str] = {}
    original_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, manifest in manifests.items():
        report_manifest = report.get("manifests", {}).get(split, {})
        if Path(str(report_manifest.get("path", ""))).resolve(strict=True) != manifest:
            raise FeatureContractError(f"{split} manifest path disagrees with audio report")
        if str(report_manifest.get("sha256", "")) != _sha256_file(manifest):
            raise FeatureContractError(f"{split} manifest SHA disagrees with audio report")
        original_rows = _read_jsonl(manifest)
        original_rows_by_split[split] = original_rows
        all_rows.extend(
            _prepare_row(
                row,
                split=split,
                manifest_path=manifest,
                source_root=source_root,
                source_sha_cache=source_sha_cache,
            )
            for row in original_rows
        )

    audit: dict[str, Any] = {
        "audio_build_report": {
            "path": report_path.as_posix(),
            "sha256": _sha256_file(report_path),
        },
        "manifests": {},
        "cross_split_overlap": {},
    }
    split_sets: dict[str, dict[str, set[str]]] = {}
    for split in ("train", "dev"):
        rows = [row for row in all_rows if row.split == split]
        groups: dict[str, list[PreparedRow]] = defaultdict(list)
        role_counts: dict[str, Counter[str]] = defaultdict(Counter)
        round_counts: dict[int, Counter[str]] = defaultdict(Counter)
        labels: dict[str, int] = {}
        for row in rows:
            groups[row.group_id].append(row)
            role_counts[row.query_speaker_id][ROLE_SHORT[row.role]] += 1
            round_counts[row.round_index][row.query_speaker_id] += 1
            previous_label = labels.setdefault(row.query_speaker_id, row.query_speaker_label)
            if previous_label != row.query_speaker_label:
                raise FeatureContractError(f"{split} speaker label is not stable")
        if len(groups) != int(expected[split]) or len(rows) != int(expected[split]) * 3:
            raise FeatureContractError(
                f"{split} expected {expected[split]} groups/{int(expected[split]) * 3} rows, "
                f"got {len(groups)}/{len(rows)}"
            )
        if len(labels) != int(report["generated_audit"]["split_speakers"][split]):
            raise FeatureContractError(f"{split} speaker count disagrees with audio report")
        expected_labels = {speaker: index for index, speaker in enumerate(sorted(labels))}
        if labels != expected_labels:
            raise FeatureContractError(f"{split} speaker labels are not the stable sorted mapping")
        expected_roles = Counter({"A": 2, "B": 2, "C": 2})
        if any(counts != expected_roles for counts in role_counts.values()):
            raise FeatureContractError(f"{split} role rotation is not 2A+2B+2C")
        if set(round_counts) != set(range(6)):
            raise FeatureContractError(f"{split} must contain six rounds")
        if any(set(counts) != set(labels) or any(value != 1 for value in counts.values()) for counts in round_counts.values()):
            raise FeatureContractError(f"{split} each speaker must occur exactly once per round")

        semantic_sources: list[tuple[str, str]] = []
        generated_paths: set[str] = set()
        generated_shas: set[str] = set()
        mixture_paths: set[str] = set()
        mixture_shas: set[str] = set()
        for group_id, group in groups.items():
            if len(group) != 3 or {row.role for row in group} != set(ROLES):
                raise FeatureContractError(f"{split}/{group_id} is not exactly A/B/C")
            if len({row.mixture_path for row in group}) != 1 or len({row.mixture_sha256 for row in group}) != 1:
                raise FeatureContractError(f"{split}/{group_id} does not share byte-identical mixture")
            if len({row.mixture_speakers for row in group}) != 1:
                raise FeatureContractError(f"{split}/{group_id} mixture membership changed by query")
            first = group[0]
            mixture_paths.add(first.mixture_path.as_posix())
            mixture_shas.add(first.mixture_sha256)
            generated_paths.add(first.mixture_path.as_posix())
            generated_shas.add(first.mixture_sha256)
            semantic_sources.extend(
                (path.as_posix(), sha)
                for path, sha in zip(first.mixture_source_paths, first.mixture_source_sha256)
            )
            for row in group:
                semantic_sources.extend(
                    (
                        (row.enrollment_source_path.as_posix(), row.enrollment_source_sha256),
                        (row.enrollment_view2_source_path.as_posix(), row.enrollment_view2_source_sha256),
                    )
                )
                generated_paths.update(
                    (row.enrollment_path.as_posix(), row.enrollment_view2_path.as_posix())
                )
                generated_shas.update((row.enrollment_sha256, row.enrollment_view2_sha256))

        source_paths = [path for path, _ in semantic_sources]
        source_shas = [sha for _, sha in semantic_sources]
        if len(source_paths) != len(set(source_paths)) or len(source_shas) != len(set(source_shas)):
            raise FeatureContractError(f"{split} global source WAV ledger contains reuse")
        expected_source_uses = len(labels) * 16
        if len(source_paths) != expected_source_uses:
            raise FeatureContractError(
                f"{split} source ledger requires 16 WAVs/speaker, got {len(source_paths)}"
            )
        split_sets[split] = {
            "speaker": set(labels),
            "source_path": set(source_paths),
            "source_sha256": set(source_shas),
            "generated_path": generated_paths,
            "generated_sha256": generated_shas,
            "mixture_path": mixture_paths,
            "mixture_sha256": mixture_shas,
        }
        audit["manifests"][split] = {
            "path": manifests[split].as_posix(),
            "sha256": _sha256_file(manifests[split]),
            "rows": len(rows),
            "groups": len(groups),
            "speakers": len(labels),
            "source_uses": len(source_paths),
            "generated_artifacts": len(generated_paths),
            "role_counts_per_speaker": {"A": 2, "B": 2, "C": 2},
            "rounds": 6,
        }

    for field in split_sets["train"]:
        overlap = sorted(split_sets["train"][field] & split_sets["dev"][field])
        audit["cross_split_overlap"][field] = overlap
        if overlap:
            raise FeatureContractError(f"train/dev {field} overlap: {overlap[:8]}")

    if len(source_sha_cache) != sum(
        int(audit["manifests"][split]["source_uses"]) for split in ("train", "dev")
    ):
        raise FeatureContractError("source SHA cache count disagrees with global ledger")
    if int(report.get("source_audit", {}).get("verified_source_files", -1)) != len(source_sha_cache):
        raise FeatureContractError("audio report source count changed")
    audit["source_files_reverified"] = len(source_sha_cache)
    audit["final_deferred"] = True
    audit["official_test_loaded"] = False
    audit["dataset_a_used"] = False
    return sorted(all_rows, key=lambda row: (row.split, row.round_index, row.group_id, row.role_id)), audit


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "item"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _prepare_output(output_dir: str | Path, *, project_root: Path) -> Path:
    if _contains_dataset_a(output_dir):
        raise FeatureContractError("output contains a forbidden Dataset-A marker")
    output = Path(output_dir).resolve(strict=False)
    if not _is_under(output, project_root):
        raise FeatureContractError(f"output must stay inside project root: {output}")
    if output.exists() and any(output.iterdir()):
        raise FeatureContractError(f"feature output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "mixture").mkdir(exist_ok=True)
    (output / "query").mkdir(exist_ok=True)
    return output


def build_feature_cache(
    train_manifest: str | Path,
    dev_manifest: str | Path,
    audio_build_report: str | Path,
    *,
    qwen_config_dir: str | Path,
    campp_model: str | Path,
    output_dir: str | Path,
    allowed_source_root: str | Path = DEFAULT_SOURCE_ROOT,
    qwen_extractor: Any | None = None,
    campp_extractor: Any | None = None,
    num_threads: int = 2,
    expected_groups: Optional[Mapping[str, int]] = None,
) -> dict[str, Any]:
    """Build a train/dev-only Qwen/CAM++ cache after full provenance audit."""

    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    rows, input_audit = validate_input_bundle(
        train_manifest,
        dev_manifest,
        audio_build_report,
        allowed_source_root=allowed_source_root,
        expected_groups=expected_groups,
    )
    qwen_dir = _resolve_dir(qwen_config_dir, field="qwen_config_dir")
    preprocessor_path, qwen_values = _load_preprocessor_config(qwen_dir)
    qwen_config_sha = _sha256_file(preprocessor_path)
    campp_path = _resolve_file(campp_model, field="campp_model")
    campp_model_sha = _sha256_file(campp_path)
    if qwen_extractor is None:
        qwen_extractor = make_qwen_feature_extractor(qwen_dir, local_files_only=True)
    feature_spec = validate_feature_extractor_spec(qwen_extractor, config_values=qwen_values)
    feature_spec_sha = _feature_spec_hash(feature_spec)
    if campp_extractor is None:
        campp_extractor = _make_real_camp_extractor(campp_path, num_threads)
    output = _prepare_output(output_dir, project_root=project_root)

    mixture_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.split, row.mixture_sha256)
        if key in mixture_cache:
            continue
        waveform, sample_rate = _read_audio(row.mixture_path)
        features, attention_mask = extract_qwen_features(qwen_extractor, waveform, sample_rate)
        cache_path = output / "mixture" / f"{row.split}__{_safe_name(row.group_id)}.npz"
        np.savez_compressed(
            cache_path,
            input_features=features,
            feature_attention_mask=attention_mask,
            split=np.asarray(row.split),
            base_mixture_id=np.asarray(row.group_id),
            mixture_audio_path=np.asarray(row.mixture_path.as_posix()),
            mixture_audio_sha256=np.asarray(row.mixture_sha256),
            input_features_sha256=np.asarray(_sha256_array(features)),
            qwen_config_sha256=np.asarray(qwen_config_sha),
            qwen_feature_spec_sha256=np.asarray(feature_spec_sha),
            feature_size=np.asarray(FEATURE_SIZE, dtype=np.int32),
            sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int32),
        )
        mixture_cache[key] = {
            "path": cache_path,
            "file_sha256": _sha256_file(cache_path),
            "array_sha256": _sha256_array(features),
            "features": features,
            "frame_count": int(features.shape[1]),
        }

    feature_rows: list[dict[str, Any]] = []
    alignment_records: list[dict[str, int]] = []
    for row in rows:
        mixture = mixture_cache[(row.split, row.mixture_sha256)]
        view1_wave, view1_sr = _read_audio(row.enrollment_path)
        view2_wave, view2_sr = _read_audio(row.enrollment_view2_path)
        embedding = extract_camp_embedding(campp_extractor, view1_wave, view1_sr)
        embedding_view2 = extract_camp_embedding(campp_extractor, view2_wave, view2_sr)
        activity, alignment = _align_tail(
            _load_activity(row.activity_path), int(mixture["frame_count"]), fill=0.0
        )
        if row.role == "absent_C" and np.any(activity != 0):
            raise FeatureContractError(f"{row.row_id} absent activity changed during alignment")
        query_path = output / "query" / f"{row.split}__{_safe_name(row.row_id)}.npz"
        np.savez_compressed(
            query_path,
            enrollment_embedding=embedding,
            enrollment_embedding_view2=embedding_view2,
            target_activity=activity.astype(np.float32),
            split=np.asarray(row.split),
            row_id=np.asarray(row.row_id),
            base_mixture_id=np.asarray(row.group_id),
            query_speaker_id=np.asarray(row.query_speaker_id),
            query_speaker_label=np.asarray(row.query_speaker_label, dtype=np.int32),
            query_role=np.asarray(row.role),
            mixture_speakers_json=np.asarray(json.dumps(row.mixture_speakers)),
            mixture_audio_sha256=np.asarray(row.mixture_sha256),
            mixture_feature_sha256=np.asarray(mixture["file_sha256"]),
            enrollment_audio_sha256=np.asarray(row.enrollment_sha256),
            enrollment_audio_view2_sha256=np.asarray(row.enrollment_view2_sha256),
            target_activity_source_sha256=np.asarray(row.activity_sha256),
            target_activity_array_sha256=np.asarray(_sha256_array(activity)),
            enrollment_embedding_sha256=np.asarray(_sha256_array(embedding)),
            enrollment_embedding_view2_sha256=np.asarray(_sha256_array(embedding_view2)),
            campp_model_sha256=np.asarray(campp_model_sha),
            qwen_feature_spec_sha256=np.asarray(feature_spec_sha),
            activity_alignment_json=np.asarray(json.dumps(alignment, sort_keys=True)),
        )
        manifest_row = dict(row.original_row)
        manifest_row.update(
            {
                "cache_schema": SCHEMA,
                "source_manifest_path": row.manifest_path.as_posix(),
                "source_manifest_row_sha256": _sha256_json(row.original_row),
                "split": row.split,
                "row_id": row.row_id,
                "base_mixture_id": row.group_id,
                "query_role_id_is_audit_only": True,
                "query_role_id_used_as_model_input": False,
                "membership_label_policy": "recompute query_speaker_id in destination mixture_speakers",
                "mixture_feature": _relative(mixture["path"], output),
                "mixture_feature_sha256": mixture["file_sha256"],
                "mixture_input_features_sha256": mixture["array_sha256"],
                "query_feature": _relative(query_path, output),
                "query_npz_sha256": _sha256_file(query_path),
                "target_activity_array_sha256": _sha256_array(activity),
                "enrollment_embedding_sha256": _sha256_array(embedding),
                "enrollment_embedding_view2_sha256": _sha256_array(embedding_view2),
                "qwen_config_sha256": qwen_config_sha,
                "qwen_feature_spec_sha256": feature_spec_sha,
                "campp_model_sha256": campp_model_sha,
                "activity_alignment": alignment,
                "provenance": {
                    "audio_not_copied": True,
                    "final_not_read": True,
                    "query_role_excluded_from_model_allowlist": True,
                    "labels_recomputed_against_destination_mixture": True,
                },
            }
        )
        feature_rows.append(manifest_row)
        alignment_records.append(
            {
                "pad": int(alignment["tail_pad_frames"]),
                "crop": int(alignment["tail_crop_frames"]),
            }
        )

    feature_manifest = output / "features_manifest.jsonl"
    _write_jsonl(feature_manifest, feature_rows)
    payload_paths = [feature_manifest]
    payload_paths.extend(sorted((output / "mixture").glob("*.npz")))
    payload_paths.extend(sorted((output / "query").glob("*.npz")))
    helper_path = Path(__file__).with_name("build_dacf_v3_features.py").resolve(strict=True)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "cache_schema": SCHEMA,
        "builder_code": Path(__file__).resolve().as_posix(),
        "builder_code_sha256": _sha256_file(Path(__file__).resolve()),
        "feature_helper_code": helper_path.as_posix(),
        "feature_helper_code_sha256": _sha256_file(helper_path),
        "dataset_a_used": False,
        "official_test_loaded": False,
        "loaded_splits": ["train", "dev"],
        "final_deferred": True,
        "source_corpus": SOURCE_CORPUS,
        "input_audit": input_audit,
        "qwen_config_dir": qwen_dir.as_posix(),
        "qwen_preprocessor_config": preprocessor_path.as_posix(),
        "qwen_config_sha256": qwen_config_sha,
        "qwen_feature_spec": feature_spec,
        "qwen_feature_spec_sha256": feature_spec_sha,
        "campp_model": campp_path.as_posix(),
        "campp_model_sha256": campp_model_sha,
        "counts": {
            "rows": len(rows),
            "groups": len(mixture_cache),
            "unique_mixture_sha256": len(mixture_cache),
            "qwen_mixture_feature_calls": len(mixture_cache),
            "campp_enrollment_extractor_calls": len(rows) * 2,
            "mixture_npz": len(mixture_cache),
            "query_npz": len(feature_rows),
            "split_counts": {
                split: {
                    "rows": sum(row.split == split for row in rows),
                    "groups": len({row.group_id for row in rows if row.split == split}),
                    "speakers": len({row.query_speaker_id for row in rows if row.split == split}),
                }
                for split in ("train", "dev")
            },
        },
        "deduplication": {
            "one_qwen_call_per_unique_mixture": True,
            "two_campp_calls_per_query": True,
            "clean_target_qwen_calls": 0,
        },
        "model_input_allowlist": [
            "mixture.input_features",
            "mixture.feature_attention_mask",
            "query.enrollment_embedding",
            "query.enrollment_embedding_view2",
        ],
        "label_contract": {
            "presence": "query_speaker_id in destination mixture_speakers",
            "activity": "destination-group target activity for a present speaker; zero otherwise",
            "origin_query_role_is_label": False,
            "origin_query_role_id_is_model_input": False,
        },
        "activity_alignment": {
            "policy": "tail-only zero-pad or crop to destination mixture frame count",
            "rows_with_tail_pad": sum(item["pad"] > 0 for item in alignment_records),
            "rows_with_tail_crop": sum(item["crop"] > 0 for item in alignment_records),
        },
        "features_manifest": "features_manifest.jsonl",
        "features_manifest_sha256": _sha256_file(feature_manifest),
        "cache_sha256": _payload_sha256(payload_paths, root=output),
        "cache_sha256_scope": "features_manifest.jsonl plus all mixture/query NPZ files",
        "limitations": [
            "This is a feature cache, not a CER/RR/RTF or integration result.",
            "AISHELL read speech does not verify same-command home hard negatives.",
            "The official final/test split was neither accepted nor read.",
        ],
        "runtime_sec": float(time.perf_counter() - started),
    }
    _write_json(output / "cache_report.json", report)
    validate_cache(output)
    return report


def _scalar_text(value: Any, *, field: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise FeatureContractError(f"{field} must be scalar")
    return str(array.item())


def validate_cache(cache_root: str | Path) -> dict[str, Any]:
    """Re-read the cache and every bound hash without touching final/test."""

    root = _resolve_dir(cache_root, field="cache_root")
    report_path = _resolve_file(root / "cache_report.json", field="cache_report")
    report = _read_json(report_path)
    if report.get("schema") != REPORT_SCHEMA or report.get("cache_schema") != SCHEMA:
        raise FeatureContractError("cache/report schema changed")
    if report.get("loaded_splits") != ["train", "dev"]:
        raise FeatureContractError("cache loaded split contract changed")
    if report.get("final_deferred") is not True or report.get("official_test_loaded") is not False:
        raise FeatureContractError("cache final/test firewall changed")
    if bool(report.get("dataset_a_used")):
        raise FeatureContractError("cache declares Dataset-A use")
    builder = _resolve_file(report.get("builder_code"), field="builder_code")
    helper = _resolve_file(report.get("feature_helper_code"), field="feature_helper_code")
    if _sha256_file(builder) != report.get("builder_code_sha256"):
        raise FeatureContractError("builder code SHA mismatch")
    if _sha256_file(helper) != report.get("feature_helper_code_sha256"):
        raise FeatureContractError("feature helper code SHA mismatch")

    audio_binding = report.get("input_audit", {}).get("audio_build_report", {})
    audio_report = _resolve_file(audio_binding.get("path"), field="bound audio report")
    if _sha256_file(audio_report) != audio_binding.get("sha256"):
        raise FeatureContractError("bound audio report SHA mismatch")
    source_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for split in ("train", "dev"):
        binding = report.get("input_audit", {}).get("manifests", {}).get(split, {})
        manifest = _resolve_file(binding.get("path"), field=f"bound {split} manifest")
        if _sha256_file(manifest) != binding.get("sha256"):
            raise FeatureContractError(f"bound {split} manifest SHA mismatch")
        for row in _read_jsonl(manifest):
            key = (manifest.as_posix(), str(row.get("id", "")))
            if key in source_rows:
                raise FeatureContractError("duplicate source manifest semantic key")
            source_rows[key] = row

    manifest_path = _resolve_file(root / str(report.get("features_manifest")), field="features_manifest")
    if _sha256_file(manifest_path) != report.get("features_manifest_sha256"):
        raise FeatureContractError("features manifest SHA mismatch")
    rows = _read_jsonl(manifest_path)
    if len(rows) != int(report.get("counts", {}).get("rows", -1)):
        raise FeatureContractError("feature manifest row count changed")
    mixture_files: set[Path] = set()
    query_files: set[Path] = set()
    for row in rows:
        source_manifest = _resolve_file(row.get("source_manifest_path"), field="source_manifest_path")
        source_key = (source_manifest.as_posix(), str(row.get("row_id", "")))
        source = source_rows.get(source_key)
        if source is None or _sha256_json(source) != row.get("source_manifest_row_sha256"):
            raise FeatureContractError("source manifest row SHA mismatch")

        mixture_path = _resolve_file(root / str(row.get("mixture_feature")), field="mixture_feature")
        query_path = _resolve_file(root / str(row.get("query_feature")), field="query_feature")
        if not _is_under(mixture_path, root) or not _is_under(query_path, root):
            raise FeatureContractError("cache artifact escaped cache root")
        mixture_files.add(mixture_path)
        query_files.add(query_path)
        if _sha256_file(mixture_path) != row.get("mixture_feature_sha256"):
            raise FeatureContractError("mixture feature file SHA mismatch")
        if _sha256_file(query_path) != row.get("query_npz_sha256"):
            raise FeatureContractError("query NPZ SHA mismatch")
        with np.load(mixture_path, allow_pickle=False) as mixture:
            features = np.asarray(mixture["input_features"], dtype=np.float32)
            mask = np.asarray(mixture["feature_attention_mask"])
            if features.ndim != 2 or features.shape[0] != FEATURE_SIZE:
                raise FeatureContractError("mixture Qwen feature shape changed")
            if mask.shape != (features.shape[1],):
                raise FeatureContractError("mixture attention-mask shape changed")
            if _sha256_array(features) != row.get("mixture_input_features_sha256"):
                raise FeatureContractError("mixture feature array SHA mismatch")
        with np.load(query_path, allow_pickle=False) as query:
            if "query_role_id" in query.files:
                raise FeatureContractError("query_role_id leaked into query NPZ")
            embedding = np.asarray(query["enrollment_embedding"], dtype=np.float32)
            view2 = np.asarray(query["enrollment_embedding_view2"], dtype=np.float32)
            activity = np.asarray(query["target_activity"], dtype=np.float32)
            if embedding.shape != (CAMPP_DIM,) or view2.shape != (CAMPP_DIM,):
                raise FeatureContractError("CAM++ embedding shape changed")
            if activity.shape != (features.shape[1],):
                raise FeatureContractError("target activity frame count changed")
            if _scalar_text(query["row_id"], field="row_id") != row.get("row_id"):
                raise FeatureContractError("query row_id metadata mismatch")
            if _scalar_text(query["query_speaker_id"], field="query_speaker_id") != row.get("query_speaker_id"):
                raise FeatureContractError("query speaker metadata mismatch")
            if _sha256_array(embedding) != row.get("enrollment_embedding_sha256"):
                raise FeatureContractError("view1 embedding SHA mismatch")
            if _sha256_array(view2) != row.get("enrollment_embedding_view2_sha256"):
                raise FeatureContractError("view2 embedding SHA mismatch")
            if _sha256_array(activity) != row.get("target_activity_array_sha256"):
                raise FeatureContractError("activity array SHA mismatch")

    actual_mixtures = set((root / "mixture").glob("*.npz"))
    actual_queries = set((root / "query").glob("*.npz"))
    if mixture_files != actual_mixtures or query_files != actual_queries:
        raise FeatureContractError("cache contains missing or unreferenced NPZ files")
    if len(mixture_files) != int(report["counts"]["mixture_npz"]):
        raise FeatureContractError("mixture NPZ count changed")
    if len(query_files) != int(report["counts"]["query_npz"]):
        raise FeatureContractError("query NPZ count changed")
    payload_paths = [manifest_path, *sorted(mixture_files), *sorted(query_files)]
    if _payload_sha256(payload_paths, root=root) != report.get("cache_sha256"):
        raise FeatureContractError("cache payload SHA mismatch")
    if list(root.rglob("*.wav")):
        raise FeatureContractError("audio was copied into feature cache")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--audio-build-report", required=True)
    parser.add_argument("--qwen-config-dir", required=True)
    parser.add_argument("--campp-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--num-threads", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    report = build_feature_cache(
        args.train_manifest,
        args.dev_manifest,
        args.audio_build_report,
        qwen_config_dir=args.qwen_config_dir,
        campp_model=args.campp_model,
        output_dir=args.output,
        allowed_source_root=args.allowed_source_root,
        num_threads=args.num_threads,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PreparedRow",
    "SCHEMA",
    "build_feature_cache",
    "validate_cache",
    "validate_input_bundle",
]


if __name__ == "__main__":
    raise SystemExit(main())
