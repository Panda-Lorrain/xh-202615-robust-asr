"""Generate DACF-v4b train/dev audio from an audited role-rotation schedule.

The schedule, not this builder, chooses every speaker and source WAV.  This
builder verifies all source SHAs again, emits byte-identical A/B/C query rows,
uses distinct enrollment source WAVs for view1/view2, and never loads an
official final/test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build_dacf_counterfactual import (
    QUERY_ROLE_IDS,
    SR,
    _activity_from_clean,
    _assert_not_dataset_a,
    _build_base,
    _load_audio,
    _make_enrollment,
    _make_independent_enrollment_noise,
    _row_for_query,
    _sample_params,
    _sha256_file,
    _write_wav,
    resolve_noise_type,
)
from build_dacf_v4b_schedule import (
    AUDIT_SCHEMA as SCHEDULE_AUDIT_SCHEMA,
    DEFAULT_SEEDS,
    ROLES,
    SCHEDULE_SCHEMA,
)


AUDIO_SCHEMA = "dacf-v4b-role-rotation-audio-v0.1"
MANIFEST_SCHEMA_VERSION = "dacf-v4b-query-manifest-v0.1"


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _source_item(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "wav": str(source["path"]),
        "spk": str(source["speaker_id"]),
        "utt": str(source["utterance_id"]),
        "ref": str(source.get("transcript", "")),
        "source_corpus": str(source.get("source_corpus", "AISHELL-1")),
    }


def validate_schedule_binding(
    schedule_path: str | Path,
    audit_path: str | Path,
) -> tuple[Path, Path, Mapping[str, Any], Mapping[str, Any]]:
    schedule_file = Path(schedule_path).resolve(strict=True)
    audit_file = Path(audit_path).resolve(strict=True)
    _assert_not_dataset_a(schedule_file)
    _assert_not_dataset_a(audit_file)
    schedule = _read_json(schedule_file)
    audit = _read_json(audit_file)
    if schedule.get("schema") != SCHEDULE_SCHEMA:
        raise ValueError("schedule schema mismatch")
    if audit.get("schema") != SCHEDULE_AUDIT_SCHEMA:
        raise ValueError("schedule audit schema mismatch")
    if str(audit.get("schedule_sha256", "")) != _sha256_file(schedule_file):
        raise ValueError("schedule bytes do not match the audited SHA")
    if bool(schedule.get("dataset_a_used", True)) or bool(audit.get("dataset_a_used", True)):
        raise ValueError("Dataset-A is forbidden")
    if schedule.get("loaded_source_splits") != ["train", "dev"]:
        raise ValueError("schedule must bind only train/dev source splits")
    if bool(schedule.get("official_test_loaded", True)) or not bool(schedule.get("final_deferred", False)):
        raise ValueError("official final/test must remain deferred")
    if set(schedule.get("splits", {})) != {"train", "dev"}:
        raise ValueError("schedule must expose train/dev only")
    return schedule_file, audit_file, schedule, audit


def _verify_source_bytes(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    path_counts: Counter[str] = Counter()
    sha_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()
    verified = 0
    for split in ("train", "dev"):
        for group in schedule["splits"][split]:
            for role in ROLES:
                role_row = group["roles"][role]
                speaker = str(role_row["speaker_id"])
                for source in role_row["sources"].values():
                    path = Path(source["path"]).resolve(strict=True)
                    _assert_not_dataset_a(path)
                    if path.parent.name != speaker or str(source["speaker_id"]) != speaker:
                        raise ValueError("source speaker/path metadata mismatch")
                    digest = _sha256_file(path)
                    if digest != str(source["sha256"]):
                        raise ValueError(f"source SHA changed since scheduling: {path}")
                    path_counts[path.as_posix()] += 1
                    sha_counts[digest] += 1
                    speaker_counts[speaker] += 1
                    verified += 1
    if any(value != 1 for value in path_counts.values()):
        raise ValueError("source path reuse detected before audio generation")
    if any(value != 1 for value in sha_counts.values()):
        raise ValueError("source SHA reuse detected before audio generation")
    if any(value != 16 for value in speaker_counts.values()):
        raise ValueError("source uses per speaker are not exactly 16")
    return {
        "verified_source_files": verified,
        "unique_source_paths": len(path_counts),
        "unique_source_sha256": len(sha_counts),
        "source_uses_per_speaker_min": min(speaker_counts.values()),
        "source_uses_per_speaker_max": max(speaker_counts.values()),
    }


def _generate_group(
    group: Mapping[str, Any],
    *,
    split_output: Path,
    seed: int,
    speaker_labels: Mapping[str, int],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    rng = random.Random(seed)
    audio_cache: dict[str, np.ndarray] = {}
    roles = group["roles"]
    source_a = _source_item(roles["A"]["sources"]["recognition"])
    source_b = _source_item(roles["B"]["sources"]["recognition"])
    enroll_sources = {
        role: {
            "view1": _source_item(roles[role]["sources"]["enrollment_view1"]),
            "view2": _source_item(roles[role]["sources"]["enrollment_view2"]),
        }
        for role in ROLES
    }
    enrollment_lengths = [
        _load_audio(source["wav"], audio_cache).size
        for role in ROLES
        for source in enroll_sources[role].values()
    ]
    max_enroll_sec = min(enrollment_lengths) / SR
    params = _sample_params(rng, "balanced", max_enroll_sec)
    # V4b removes the old A-only speed transform.  A and B still define the
    # signed SIR orientation, but every speaker occupies A and B exactly twice
    # across the schedule and no speaker identity is tied to one side.
    params["target_speed_rate"] = 1.0
    noise_resolution = resolve_noise_type(
        str(params["noise_type"]), has_external_noise=False
    )
    params.update(noise_resolution)
    params["noise_type"] = params["effective_noise_type"]
    enroll_samples = int(round(float(params["enroll_dur_sec"]) * SR))

    base = _build_base(
        _load_audio(source_a["wav"], audio_cache),
        _load_audio(source_b["wav"], audio_cache),
        params,
        rng,
        None,
        None,
        None,
    )
    view1_noise = _make_independent_enrollment_noise(
        base,
        enroll_samples,
        rng,
        avoid_offsets=(base["recognition_noise_offset_samples"],),
        avoid_seeds=(base["recognition_noise_seed"],),
        avoid_hashes=(base["recognition_noise_raw_sha256"],),
    )
    view2_noise: dict[str, tuple[np.ndarray, int, int, str]] = {}
    for role in ROLES:
        view2_noise[role] = _make_independent_enrollment_noise(
            base,
            enroll_samples,
            rng,
            avoid_offsets=(base["recognition_noise_offset_samples"], view1_noise[2]),
            avoid_seeds=(base["recognition_noise_seed"], view1_noise[1]),
            avoid_hashes=(base["recognition_noise_raw_sha256"], view1_noise[3]),
        )

    enroll_audio: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    enroll_start: dict[str, dict[str, int]] = defaultdict(dict)
    for role in ROLES:
        audio, start = _make_enrollment(
            _load_audio(enroll_sources[role]["view1"]["wav"], audio_cache),
            enroll_samples,
            rng,
            view1_noise[0],
            base["rir"],
        )
        enroll_audio[role]["view1"] = audio
        enroll_start[role]["view1"] = start
        audio, start = _make_enrollment(
            _load_audio(enroll_sources[role]["view2"]["wav"], audio_cache),
            enroll_samples,
            rng,
            view2_noise[role][0],
            base["rir"],
        )
        enroll_audio[role]["view2"] = audio
        enroll_start[role]["view2"] = start

    recognition = base["recognition"]
    clean = {
        "A": base["clean_a"],
        "B": base["clean_b"],
        "C": np.zeros_like(recognition, dtype=np.float32),
    }
    peaks = [
        float(np.max(np.abs(value))) if value.size else 0.0
        for value in [recognition, clean["A"], clean["B"]]
        + [enroll_audio[role][view] for role in ROLES for view in ("view1", "view2")]
    ]
    peak_norm_scale = 1.0
    if max(peaks) > 0.99:
        peak_norm_scale = 0.99 / max(peaks)
        recognition = recognition * peak_norm_scale
        clean["A"] = clean["A"] * peak_norm_scale
        clean["B"] = clean["B"] * peak_norm_scale
        for role in ROLES:
            for view in ("view1", "view2"):
                enroll_audio[role][view] = enroll_audio[role][view] * peak_norm_scale

    base_id = str(group["base_mixture_id"])
    recognition_path = split_output / "recognition" / f"{base_id}.wav"
    _write_wav(recognition_path, recognition)
    clean_paths = {
        role: split_output / "clean_target" / f"{base_id}__{'present' if role != 'C' else 'absent'}_{role}.wav"
        for role in ROLES
    }
    enrollment_paths = {
        role: split_output / "enrollment" / f"{base_id}__{'present' if role != 'C' else 'absent'}_{role}.wav"
        for role in ROLES
    }
    enrollment_view2_paths = {
        role: split_output / "enrollment_view2" / f"{base_id}__{'present' if role != 'C' else 'absent'}_{role}.wav"
        for role in ROLES
    }
    activity_paths = {
        role: split_output / "activity" / f"{base_id}__{'present' if role != 'C' else 'absent'}_{role}.npy"
        for role in ROLES
    }
    for role in ROLES:
        _write_wav(clean_paths[role], clean[role])
        _write_wav(enrollment_paths[role], enroll_audio[role]["view1"])
        _write_wav(enrollment_view2_paths[role], enroll_audio[role]["view2"])
        activity = (
            _activity_from_clean(clean[role], 160)
            if role != "C"
            else np.zeros(max(1, int(math.ceil(recognition.size / 160))), dtype=np.uint8)
        )
        activity_paths[role].parent.mkdir(parents=True, exist_ok=True)
        np.save(activity_paths[role], activity)

    enrollment_sha = {
        role: _sha256_file(enrollment_paths[role]) for role in ROLES
    }
    enrollment_view2_sha = {
        role: _sha256_file(enrollment_view2_paths[role]) for role in ROLES
    }
    common = {
        "mixture_sha256": _sha256_file(recognition_path),
        "noise_id": base["noise_id"],
        "noise_type": base["noise_type"],
        "requested_noise_type": base["requested_noise_type"],
        "effective_noise_type": base["effective_noise_type"],
        "noise_type_reason": base["noise_type_reason"],
        "noise_src": base["noise_src"],
        "noise_seed": base["noise_seed"],
        "recognition_noise_seed": base["recognition_noise_seed"],
        "recognition_noise_offset_samples": base["recognition_noise_offset_samples"],
        "recognition_noise_raw_sha256": base["recognition_noise_raw_sha256"],
        "enrollment_noise_seed": int(view1_noise[1]),
        "enrollment_noise_offset_samples": int(view1_noise[2]),
        "enrollment_noise_raw_sha256": view1_noise[3],
        "enrollment_view2_noise_seed": {role: int(view2_noise[role][1]) for role in ROLES},
        "enrollment_view2_noise_offset_samples": {role: int(view2_noise[role][2]) for role in ROLES},
        "enrollment_view2_noise_raw_sha256": {role: view2_noise[role][3] for role in ROLES},
        "enrollment_sha256": enrollment_sha,
        "enrollment_view2_sha256": enrollment_view2_sha,
        "rir_id": "none",
        "rir_src": None,
        "environment_id": f"{group['protocol_split']}:{base_id}:env",
        "augmentation": {
            **{key: value for key, value in params.items() if key != "enroll_pollute_p"},
            "ab_symmetry_policy": "A-only speed disabled; every speaker exactly 2A and 2B",
        },
        "timing": base["timing"],
        "measured_sir_db": base["measured_sir_db"],
        "measured_snr_db": base["measured_snr_db"],
        "peak_norm_scale": peak_norm_scale,
    }

    source_items = {"A": source_a, "B": source_b}
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        present = role != "C"
        row = _row_for_query(
            split=str(group["protocol_split"]),
            base_id=base_id,
            query_id=role,
            query_speaker=str(group["roles"][role]["speaker_id"]),
            query_speaker_label=speaker_labels[str(group["roles"][role]["speaker_id"])],
            target_present=present,
            ref=source_items[role]["ref"] if present else "",
            enrollment_src=enroll_sources[role]["view1"],
            recognition_path=recognition_path,
            enrollment_path=enrollment_paths[role],
            enrollment_view2_path=enrollment_view2_paths[role],
            clean_path=clean_paths[role],
            activity_path=activity_paths[role],
            clean=clean[role],
            common=common,
            source_a=source_a,
            source_b=source_b,
            enrollment_start=enroll_start[role]["view1"],
            enrollment_view2_start=enroll_start[role]["view2"],
            enrollment_duration_sec=enroll_samples / SR,
            hop_samples=160,
        )
        view1_source = group["roles"][role]["sources"]["enrollment_view1"]
        view2_source = group["roles"][role]["sources"]["enrollment_view2"]
        row.update(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "protocol_version": AUDIO_SCHEMA,
                "protocol_split": str(group["protocol_split"]),
                "source_split": str(group["source_split"]),
                "round_index": int(group["round_index"]),
                "round_group_index": int(group["round_group_index"]),
                "global_group_index": int(group["global_group_index"]),
                "enrollment_src_sha256": str(view1_source["sha256"]),
                "enrollment_src_view2": str(view2_source["path"]),
                "enrollment_src_view2_sha256": str(view2_source["sha256"]),
                "enrollment_view2_spk": str(view2_source["speaker_id"]),
                "enrollment_view2_utt": str(view2_source["utterance_id"]),
                "enrollment_views_distinct_source_wavs": True,
                "identity_positive_contract": (
                    "view1/view2 use different source WAVs from the same "
                    "query speaker under independent noise draws"
                ),
                "global_source_path_use_count": 1,
                "global_source_sha_use_count": 1,
                "mixture_source_sha256": {
                    "A": str(group["roles"]["A"]["sources"]["recognition"]["sha256"]),
                    "B": str(group["roles"]["B"]["sources"]["recognition"]["sha256"]),
                },
                "role_rotation_contract": "each selected speaker exactly 2A+2B+2C",
                "query_role_id_model_input": False,
                "hard_negative_complete_instruction_verified": False,
                "hard_negative_status": (
                    "requires_real_home_command_source" if role == "C" else "not_applicable"
                ),
            }
        )
        rows.append(row)
    return rows, {
        "base_mixture_id": base_id,
        "mixture_sha256": common["mixture_sha256"],
        "duration_samples": int(recognition.size),
        "enrollment_duration_samples": enroll_samples,
        "measured_sir_db": float(base["measured_sir_db"]),
        "measured_snr_db": float(base["measured_snr_db"]),
        "overlap_ratio": float(params["overlap_ratio"]),
        "target_speed_rate": float(params["target_speed_rate"]),
    }


def _audit_rows(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    group_reports: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    split_speakers: dict[str, set[str]] = {}
    generated_paths: dict[str, set[str]] = {}
    generated_shas: dict[str, set[str]] = {}
    round_roles: dict[str, dict[str, Counter[str]]] = {}
    for split in ("train", "dev"):
        rows = list(rows_by_split[split])
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["base_mixture_id"])].append(row)
        expected_groups = 96 if split == "train" else 24
        if len(groups) != expected_groups or len(rows) != expected_groups * 3:
            raise ValueError(f"{split} generated group/row count mismatch")
        speakers: set[str] = set()
        role_counts: dict[str, Counter[str]] = defaultdict(Counter)
        paths: set[str] = set()
        shas: set[str] = set()
        for group_id, group_rows in groups.items():
            if {str(row["query_id"]) for row in group_rows} != set(ROLES):
                raise ValueError(f"{split}/{group_id} does not contain A/B/C")
            if len({str(row["mixture_sha256"]) for row in group_rows}) != 1:
                raise ValueError("counterfactual rows do not share mixture bytes")
            if any(not bool(row["enrollment_views_distinct_source_wavs"]) for row in group_rows):
                raise ValueError("enrollment view source distinction is false")
            for row in group_rows:
                speaker = str(row["query_speaker_id"])
                role = str(row["query_id"])
                speakers.add(speaker)
                role_counts[speaker][role] += 1
                if str(row["enrollment_src"]) == str(row["enrollment_src_view2"]):
                    raise ValueError("view1/view2 source WAVs are identical")
                for field in (
                    "recognition_audio",
                    "enrollment_audio",
                    "enrollment_audio_view2",
                ):
                    path = Path(row[field]).resolve(strict=True)
                    paths.add(path.as_posix())
                    shas.add(_sha256_file(path))
                activity = np.load(str(row["target_activity"]), allow_pickle=False)
                if _sha256_array(activity) == "":
                    raise AssertionError("unreachable activity hash guard")
                if role == "C":
                    if np.any(activity != 0) or int(row["clean_target_nonzero_samples"]) != 0:
                        raise ValueError("absent-C target is not physically blank")
                elif not np.any(activity > 0):
                    raise ValueError("present target activity is empty")
        expected_roles = Counter({"A": 2, "B": 2, "C": 2})
        if any(counts != expected_roles for counts in role_counts.values()):
            raise ValueError(f"{split} generated role rotation changed")
        if any(float(value["target_speed_rate"]) != 1.0 for value in group_reports[split]):
            raise ValueError("A-only speed transform was not fully disabled")
        split_speakers[split] = speakers
        generated_paths[split] = paths
        generated_shas[split] = shas
        round_roles[split] = role_counts

    overlaps = {
        "speaker": sorted(split_speakers["train"] & split_speakers["dev"]),
        "generated_path": sorted(generated_paths["train"] & generated_paths["dev"]),
        "generated_sha256": sorted(generated_shas["train"] & generated_shas["dev"]),
    }
    if any(overlaps.values()):
        raise ValueError("train/dev generated artifact overlap detected")
    return {
        "cross_split_overlap": overlaps,
        "split_speakers": {key: len(value) for key, value in split_speakers.items()},
        "generated_artifacts": {
            split: {
                "unique_paths": len(generated_paths[split]),
                "unique_sha256": len(generated_shas[split]),
            }
            for split in ("train", "dev")
        },
        "role_counts_per_speaker": {"A": 2, "B": 2, "C": 2},
        "a_only_speed_disabled": True,
    }


def build_audio(
    schedule_path: str | Path,
    schedule_audit_path: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    schedule_file, audit_file, schedule, _ = validate_schedule_binding(
        schedule_path, schedule_audit_path
    )
    _assert_not_dataset_a(output_dir)
    output = Path(output_dir).resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"audio output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_audit = _verify_source_bytes(schedule)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    group_reports: dict[str, list[Mapping[str, Any]]] = {}
    manifest_paths: dict[str, str] = {}
    for split in ("train", "dev"):
        rows: list[dict[str, Any]] = []
        reports: list[Mapping[str, Any]] = []
        split_speakers = sorted(
            {
                str(group["roles"][role]["speaker_id"])
                for group in schedule["splits"][split]
                for role in ROLES
            }
        )
        speaker_labels = {
            speaker: index for index, speaker in enumerate(split_speakers)
        }
        for group in schedule["splits"][split]:
            group_rows, group_report = _generate_group(
                group,
                split_output=output / split,
                seed=DEFAULT_SEEDS[split] + int(group["global_group_index"]) * 1009,
                speaker_labels=speaker_labels,
            )
            rows.extend(group_rows)
            reports.append(group_report)
        manifest = output / split / "manifest.jsonl"
        _write_jsonl(manifest, rows)
        rows_by_split[split] = rows
        group_reports[split] = reports
        manifest_paths[split] = manifest.as_posix()

    generated_audit = _audit_rows(rows_by_split, group_reports)
    report: Mapping[str, Any] = {
        "schema": AUDIO_SCHEMA,
        "dataset_a_used": False,
        "source_corpus": "AISHELL-1",
        "schedule": {
            "path": schedule_file.as_posix(),
            "sha256": _sha256_file(schedule_file),
            "audit_path": audit_file.as_posix(),
            "audit_sha256": _sha256_file(audit_file),
        },
        "loaded_splits": ["train", "dev"],
        "official_test_loaded": False,
        "final_deferred": True,
        "source_audit": source_audit,
        "manifests": {
            split: {
                "path": manifest_paths[split],
                "sha256": _sha256_file(Path(manifest_paths[split])),
                "groups": len(group_reports[split]),
                "rows": len(rows_by_split[split]),
            }
            for split in ("train", "dev")
        },
        "generated_audit": generated_audit,
        "augmentation_audit": {
            split: {
                "target_speed_rate_values": sorted(
                    {float(row["target_speed_rate"]) for row in group_reports[split]}
                ),
                "sir_db_min": min(float(row["measured_sir_db"]) for row in group_reports[split]),
                "sir_db_max": max(float(row["measured_sir_db"]) for row in group_reports[split]),
                "snr_db_min": min(float(row["measured_snr_db"]) for row in group_reports[split]),
                "snr_db_max": max(float(row["measured_snr_db"]) for row in group_reports[split]),
                "overlap_values": sorted({float(row["overlap_ratio"]) for row in group_reports[split]}),
            }
            for split in ("train", "dev")
        },
        "hard_negative_verified_count": 0,
        "same_text_different_speaker": "deferred to real home-command suite",
        "runtime_sec": float(time.perf_counter() - started),
        "limitations": [
            "This is generated mechanism audio, not a CER/RR/RTF result.",
            "AISHELL read speech does not verify same-command home hard negatives.",
            "The official final/test split remains unopened.",
        ],
    }
    _write_json(output / "build_report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_audio(args.schedule, args.schedule_audit, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AUDIO_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "build_audio",
    "validate_schedule_binding",
]


if __name__ == "__main__":
    raise SystemExit(main())
