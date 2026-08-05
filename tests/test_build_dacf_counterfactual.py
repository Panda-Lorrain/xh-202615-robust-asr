from __future__ import annotations

import json
import hashlib
import sys
import difflib
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "runs"
    / "dacf_counterfactual_builder_test_fixture"
)
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_counterfactual import (  # noqa: E402
    SR,
    build_dacf_counterfactual,
    normalize_source_items,
    resolve_noise_type,
)


def _make_source_fixture(root: Path, *, speaker_count: int = 6):
    items = []
    t = np.arange(int(2.4 * SR), dtype=np.float32) / SR
    for speaker_index in range(speaker_count):
        speaker = f"spk_{speaker_index:02d}"
        for utterance_index in range(2):
            frequency = 180.0 + speaker_index * 35.0 + utterance_index * 7.0
            audio = (
                0.18
                * np.sin(2.0 * np.pi * frequency * t)
                * (0.75 + 0.1 * utterance_index)
            ).astype(np.float32)
            path = root / "sources" / speaker / f"utt_{utterance_index:02d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, SR, subtype="PCM_16")
            items.append(
                {
                    "wav": str(path),
                    "spk": speaker,
                    "utt": f"utt_{utterance_index:02d}",
                    "ref": f"完整指令 {speaker} {utterance_index}",
                    "complete_instruction": True,
                }
            )
    return items


def _read_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _exercise_contract(tmp_path: Path) -> None:
    source_items = _make_source_fixture(tmp_path)
    noise_t = np.arange(8 * SR, dtype=np.float32) / SR
    noise_audio = (
        0.03 * np.sin(2.0 * np.pi * 31.0 * noise_t)
        + 0.01 * np.sin(2.0 * np.pi * 73.0 * noise_t)
    ).astype(np.float32)
    noise_path = tmp_path / "noise" / "room_noise.wav"
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(noise_path, noise_audio, SR, subtype="PCM_16")
    out_dir = tmp_path / "dacf"
    result = build_dacf_counterfactual(
        source_items,
        out_dir,
        n_train_mixtures=1,
        n_val_mixtures=1,
        seed=20260806,
        noise_items=[{"wav": str(noise_path), "id": "room_noise_01"}],
    )

    assert result["total_records"] == 6
    train_rows = _read_rows(out_dir / "train" / "manifest.jsonl")
    val_rows = _read_rows(out_dir / "val" / "manifest.jsonl")
    assert len(train_rows) == len(val_rows) == 3

    train_roles = {row["query_role"] for row in train_rows}
    assert train_roles == {"present_A", "present_B", "absent_C"}
    assert {row["target_present"] for row in train_rows} == {True, False}

    base_ids = {row["base_mixture_id"] for row in train_rows}
    assert len(base_ids) == 1
    assert len({row["recognition_audio"] for row in train_rows}) == 1
    assert len({row["noise_id"] for row in train_rows}) == 1
    assert len({row["mixture_sha256"] for row in train_rows}) == 1
    assert len({row["rir_id"] for row in train_rows}) == 1
    assert len({row["environment_id"] for row in train_rows}) == 1
    mixture_path = Path(train_rows[0]["recognition_audio"])
    expected_mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
    assert train_rows[0]["mixture_sha256"] == expected_mixture_sha

    assert {row["query_role_id"] for row in train_rows} == {0, 1, 2}
    assert {
        row["query_role"]: row["query_role_id"] for row in train_rows
    } == {"present_A": 0, "present_B": 1, "absent_C": 2}
    assert len({row["query_speaker_id"] for row in train_rows}) == 3
    assert len({row["query_speaker_label"] for row in train_rows}) == 3
    for row in train_rows:
        assert row["query_speaker_id"] not in {"A", "B", "C"}
        assert row["contrastive_identity_key"] == "query_speaker_id"
        assert row["counterfactual_group_key"] == (
            f"{row['base_mixture_id']}:{row['query_role_id']}"
        )

    train_speakers = set(train_rows[0]["mixture_speakers"].values())
    train_speakers.update(row["enrollment_spk"] for row in train_rows)
    val_speakers = set(val_rows[0]["mixture_speakers"].values())
    val_speakers.update(row["enrollment_spk"] for row in val_rows)
    assert train_speakers.isdisjoint(val_speakers)

    by_role = {row["query_role"]: row for row in train_rows}
    for row in train_rows:
        enrollment, enrollment_sr = sf.read(row["enrollment_audio"], dtype="float32")
        enrollment_view2, enrollment_view2_sr = sf.read(
            row["enrollment_audio_view2"], dtype="float32"
        )
        clean, clean_sr = sf.read(row["clean_target_audio"], dtype="float32")
        recognition, recognition_sr = sf.read(row["recognition_audio"], dtype="float32")
        activity = np.load(row["target_activity"])
        assert enrollment_sr == enrollment_view2_sr == clean_sr == recognition_sr == SR
        assert 1.0 <= len(enrollment) / SR <= 2.0
        assert 1.0 <= len(enrollment_view2) / SR <= 2.0
        assert len(clean) == len(recognition)
        assert activity.dtype == np.uint8
        assert len(activity) == int(np.ceil(len(recognition) / row["target_activity_hop_samples"]))
        assert row["dataset_a_used"] is False
        assert row["target_latent"] is None
        assert row["enrollment_src"] != row["target_src"] if row["target_src"] else True
        assert row["identity_positive"] is True
        assert row["identity_positive_group"] == f"speaker:{row['query_speaker_label']}"
        assert row["enrollment_sha256"] != row["enrollment_view2_sha256"]
        assert row["same_noise_identity"] is True
        assert row["same_noise_anchor"] is False
        assert row["same_noise_anchor_contract"].startswith("identity/type shared")
        assert row["exact_noise_segment_shared"] is False
        assert row["same_rir_anchor"] is True
        assert row["noise_type"] == "env"
        assert row["requested_noise_type"] in {"white", "pink", "babble"}
        assert row["effective_noise_type"] == "env"
        assert row["noise_type"] == row["effective_noise_type"]
        assert row["noise_type_reason"] == "explicit_external_noise_manifest"
        assert row["noise_id"] == "room_noise_01"
        assert row["recognition_noise_offset_samples"] != row["enrollment_noise_offset_samples"]
        assert row["enrollment_noise_offset_samples"] != row["enrollment_view2_noise_offset_samples"]
        assert row["recognition_noise_raw_sha256"] != row["enrollment_noise_raw_sha256"]
        assert row["enrollment_noise_raw_sha256"] != row["enrollment_view2_noise_raw_sha256"]
        assert row["recognition_noise_seed"] != row["enrollment_noise_seed"]
        assert row["enrollment_noise_seed"] != row["enrollment_view2_noise_seed"]

    assert np.count_nonzero(np.abs(sf.read(by_role["present_A"]["clean_target_audio"], dtype="float32")[0])) > 0
    assert np.count_nonzero(np.abs(sf.read(by_role["present_B"]["clean_target_audio"], dtype="float32")[0])) > 0
    absent_clean = sf.read(by_role["absent_C"]["clean_target_audio"], dtype="float32")[0]
    assert np.count_nonzero(absent_clean) == 0
    assert np.count_nonzero(np.load(by_role["absent_C"]["target_activity"])) == 0
    assert by_role["absent_C"]["hard_negative"] is True
    assert by_role["absent_C"]["hard_negative_complete_instruction_verified"] is True

    manifest_before = (out_dir / "train" / "manifest.jsonl").read_bytes()
    build_dacf_counterfactual(
        source_items,
        out_dir,
        n_train_mixtures=1,
        n_val_mixtures=1,
        seed=20260806,
        noise_items=[{"wav": str(noise_path), "id": "room_noise_01"}],
    )
    manifest_after = (out_dir / "train" / "manifest.jsonl").read_bytes()
    if manifest_before != manifest_after:
        diff = "\n".join(
            difflib.unified_diff(
                manifest_before.decode().splitlines(),
                manifest_after.decode().splitlines(),
                lineterm="",
            )
        )
        raise AssertionError(f"same-seed manifest changed:\n{diff}")

    generated_resolution = resolve_noise_type("babble", has_external_noise=False)
    assert generated_resolution == {
        "requested_noise_type": "babble",
        "effective_noise_type": "pink",
        "noise_type_reason": (
            "babble_disabled_without_external_noise_to_avoid_"
            "query_speaker_content_leakage"
        ),
    }
    environment_resolution = resolve_noise_type("babble", has_external_noise=True)
    assert environment_resolution["requested_noise_type"] == "babble"
    assert environment_resolution["effective_noise_type"] == "env"
    assert environment_resolution["noise_type_reason"] == "explicit_external_noise_manifest"

    generated_dir = tmp_path / "dacf_generated_noise"
    forced_babble_params = {
        "overlap_ratio": 0.5,
        "sir_db": 0.0,
        "snr_db": 0.0,
        "noise_type": "babble",
        "target_gain_db": 0.0,
        "target_speed_rate": 1.0,
        "enroll_pollute_p": 0.0,
    }
    with patch(
        "build_dacf_counterfactual.sample_tse_aug_params",
        return_value=forced_babble_params,
    ):
        build_dacf_counterfactual(
            source_items,
            generated_dir,
            n_train_mixtures=1,
            n_val_mixtures=0,
            seed=20260807,
        )
    generated_rows = _read_rows(generated_dir / "train" / "manifest.jsonl")
    assert len(generated_rows) == 3
    for row in generated_rows:
        assert row["noise_src"] is None
        assert row["requested_noise_type"] == "babble"
        assert row["effective_noise_type"] == "pink"
        assert row["noise_type"] == row["effective_noise_type"]
        assert "query_speaker_content_leakage" in row["noise_type_reason"]
        assert row["noise_id"] == "generated:pink"


def test_dacf_counterfactual_contract(tmp_path):
    _exercise_contract(tmp_path)


def test_dataset_a_path_is_rejected_before_audio_read():
    try:
        normalize_source_items(
            [{"wav": "E:/Dataset-A/pos/cmd.wav", "spk": "A", "utt": "u", "ref": "x"}]
        )
    except ValueError as exc:
        assert "Dataset-A" in str(exc)
    else:
        raise AssertionError("Dataset-A source path must be rejected")


def test_final_split_is_pairwise_speaker_and_source_disjoint(tmp_path: Path):
    source_items = _make_source_fixture(tmp_path, speaker_count=9)
    output = tmp_path / "dacf_final"
    result = build_dacf_counterfactual(
        source_items,
        output,
        n_train_mixtures=1,
        n_val_mixtures=1,
        n_final_mixtures=1,
        seed=20260808,
    )
    assert result["final_mixtures"] == 1
    rows_by_split = {
        split: _read_rows(output / split / "manifest.jsonl")
        for split in ("train", "val", "final")
    }
    speaker_sets = {
        split: {row["query_speaker_id"] for row in rows}
        for split, rows in rows_by_split.items()
    }
    source_sets = {
        split: {
            source
            for row in rows
            for source in [*row["mixture_sources"].values(), row["enrollment_src"]]
        }
        for split, rows in rows_by_split.items()
    }
    for left, right in (("train", "val"), ("train", "final"), ("val", "final")):
        assert speaker_sets[left].isdisjoint(speaker_sets[right])
        assert source_sets[left].isdisjoint(source_sets[right])


if __name__ == "__main__":
    root = FIXTURE_ROOT.resolve(strict=False)
    runs_root = (Path(__file__).resolve().parents[1] / "code" / "runs").resolve()
    if runs_root not in root.parents:
        raise RuntimeError(f"unsafe fixture root: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    _exercise_contract(root / "base")
    test_final_split_is_pairwise_speaker_and_source_disjoint(root / "final")
    print("ALL PASS: DACF counterfactual contract")
