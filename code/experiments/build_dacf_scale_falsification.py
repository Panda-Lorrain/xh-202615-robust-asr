#!/usr/bin/env python3
"""Build the one preregistered DACF scale-only falsification set.

This wrapper deliberately exposes no knobs for dataset size, seed, source
splits, or augmentation profile.  It excludes every speaker used by the first
24/8 mini-G2 probe, then creates one new 48-train/16-validation/16-final split
from the remaining AISHELL-1 train speakers only.  Dataset-A is forbidden.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_dacf_counterfactual import (
    MAX_SCALE_FALSIFICATION_MIXTURES,
    _assert_not_dataset_a,
    _group_by_speaker,
    build_dacf_counterfactual,
    load_aishell_items,
)


EXPERIMENT_ID = "dacf-campp-cross-query-scale-only-v0.1"
BUILD_SEED = 2026080602
MATCHER_SEED = 20260806
N_TRAIN_MIXTURES = 48
N_VAL_MIXTURES = 16
N_FINAL_MIXTURES = 16
SOURCE_SPLITS = ("train",)
AUGMENTATION_PROFILE = "balanced"
MATCHER_UPDATES = 480
MATCHER_LEARNING_RATE = 3e-3
FIXED_PRESENCE_THRESHOLD = 0.5
ACTIVITY_AUC_GATE = 0.70


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_speaker_values(row: Mapping[str, Any]) -> Iterable[str]:
    for field in ("query_speaker_id", "enrollment_spk", "target_spk"):
        value = row.get(field)
        if value is not None and str(value).strip():
            yield str(value)
    for field in ("mixture_speakers",):
        value = row.get(field)
        if isinstance(value, Mapping):
            for speaker in value.values():
                if speaker is not None and str(speaker).strip():
                    yield str(speaker)


def _iter_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            yield from _iter_values(child)
    elif value is not None:
        yield value


def _guard_manifest_paths(row: Mapping[str, Any]) -> None:
    for field in (
        "recognition_audio",
        "enrollment_audio",
        "enrollment_audio_view2",
        "target_audio",
        "target_activity",
        "target_src",
        "enrollment_src",
        "mixture_sources",
        "interferer_srcs",
        "noise_src",
        "rir_src",
    ):
        for value in _iter_values(row.get(field)):
            _assert_not_dataset_a(value)
    for field in ("interferer_spks", "hard_negative_interferer_spks"):
        value = row.get(field)
        if isinstance(value, (list, tuple)):
            for speaker in value:
                if speaker is not None and str(speaker).strip():
                    yield str(speaker)


def read_excluded_speakers(
    manifests: Sequence[str | Path],
) -> tuple[set[str], dict[str, str]]:
    """Return every source speaker named by prior non-A manifests."""

    if not manifests:
        raise ValueError("at least one prior manifest is required for exclusion")
    speakers: set[str] = set()
    hashes: dict[str, str] = {}
    for raw_path in manifests:
        _assert_not_dataset_a(raw_path)
        path = Path(raw_path).resolve(strict=True)
        _assert_not_dataset_a(path)
        hashes[path.as_posix()] = _sha256_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"manifest row is not an object: {path}:{line_number}")
                if bool(row.get("dataset_a_used", False)):
                    raise ValueError(f"Dataset-A row is forbidden: {path}:{line_number}")
                _guard_manifest_paths(row)
                speakers.update(_iter_speaker_values(row))
    if not speakers:
        raise ValueError("prior manifests did not expose any speakers to exclude")
    return speakers, hashes


def preregistration_payload(
    *, aishell_root: Path, excluded_manifest_hashes: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": "dacf-scale-falsification-preregistration-v0.1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_a_used": False,
        "source_corpus": "AISHELL-1",
        "aishell_root": aishell_root.as_posix(),
        "source_splits": list(SOURCE_SPLITS),
        "excluded_manifest_sha256": dict(sorted(excluded_manifest_hashes.items())),
        "build": {
            "seed": BUILD_SEED,
            "train_mixtures": N_TRAIN_MIXTURES,
            "val_mixtures": N_VAL_MIXTURES,
            "final_mixtures": N_FINAL_MIXTURES,
            "augmentation_profile": AUGMENTATION_PROFILE,
        },
        "matcher": {
            "architecture": "DACFCAMPPQueryMatcher-v0.1-unchanged",
            "seed": MATCHER_SEED,
            "updates": MATCHER_UPDATES,
            "learning_rate": MATCHER_LEARNING_RATE,
            "fixed_presence_threshold": FIXED_PRESENCE_THRESHOLD,
        },
        "fixed_gate": {
            "auc": 0.80,
            "present_recall": 0.75,
            "absent_rr": 0.75,
            "query_response_mean": 0.20,
            "activity_auc": ACTIVITY_AUC_GATE,
            "both_enrollment_views_required": True,
        },
        "selection_policy": (
            "one scale-only run; train updates parameters, validation is observed "
            "once after fixed training, and final holdout alone decides the gate; "
            "no split selects seed, threshold, learning rate, update count, "
            "architecture, or checkpoint"
        ),
        "limitations": [
            "AISHELL-1 read speech is not a verified home-command hard negative.",
            "Passing this gate is only conditional-GO for mechanism scaling.",
            "CER, negative-set RR, and RTF are not measured at this stage.",
        ],
    }


def build_scale_falsification(
    aishell_root: str | Path,
    out_dir: str | Path,
    *,
    exclude_manifests: Sequence[str | Path],
) -> dict[str, Any]:
    _assert_not_dataset_a(aishell_root)
    _assert_not_dataset_a(out_dir)
    root = Path(aishell_root).resolve(strict=True)
    output = Path(out_dir).resolve(strict=False)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"scale-falsification output must be new or empty, got {output}"
        )

    excluded, manifest_hashes = read_excluded_speakers(exclude_manifests)
    source_items = load_aishell_items(root, SOURCE_SPLITS)
    source_speakers_before = set(_group_by_speaker(source_items))
    filtered = [item for item in source_items if str(item["spk"]) not in excluded]
    source_speakers_after = set(_group_by_speaker(filtered))
    if source_speakers_after & excluded:
        raise AssertionError("excluded speaker survived source filtering")
    required = 3 * (
        N_TRAIN_MIXTURES + N_VAL_MIXTURES + N_FINAL_MIXTURES
    )
    eligible_after = {
        speaker
        for speaker, rows in _group_by_speaker(filtered).items()
        if len(rows) >= 2
    }
    if len(eligible_after) < required:
        raise ValueError(
            f"scale falsification needs {required} unused eligible speakers, "
            f"found {len(eligible_after)}"
        )

    output.mkdir(parents=True, exist_ok=True)
    prereg = preregistration_payload(
        aishell_root=root, excluded_manifest_hashes=manifest_hashes
    )
    prereg_path = output / "PREREGISTRATION.json"
    prereg_text = json.dumps(prereg, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    prereg_path.write_text(prereg_text, encoding="utf-8")

    result = build_dacf_counterfactual(
        filtered,
        output,
        n_train_mixtures=N_TRAIN_MIXTURES,
        n_val_mixtures=N_VAL_MIXTURES,
        n_final_mixtures=N_FINAL_MIXTURES,
        seed=BUILD_SEED,
        augmentation_profile=AUGMENTATION_PROFILE,
        max_mixtures=MAX_SCALE_FALSIFICATION_MIXTURES,
    )
    result.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "preregistered": True,
            "preregistration": prereg_path.as_posix(),
            "preregistration_sha256": _sha256_file(prereg_path),
            "excluded_manifest_sha256": manifest_hashes,
            "excluded_speakers": len(excluded),
            "source_speakers_before_exclusion": len(source_speakers_before),
            "source_speakers_after_exclusion": len(source_speakers_after),
            "eligible_speakers_after_exclusion": len(eligible_after),
            "source_splits": list(SOURCE_SPLITS),
        }
    )
    report_path = output / "build_report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aishell-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        required=True,
        help="prior DACF manifest whose speakers must not enter this run; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_scale_falsification(
        args.aishell_root,
        args.out,
        exclude_manifests=args.exclude_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
