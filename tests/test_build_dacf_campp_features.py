from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_counterfactual_probe32_campp_20260806" / "_test_fixture"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_campp_features import (  # noqa: E402
    ManifestContractError,
    _pad_signal,
    build_feature_dataset,
    validate_manifests,
    window_schedule,
)


class _FakeStream:
    def __init__(self) -> None:
        self.waveform = None

    def accept_waveform(self, sample_rate: int, waveform: np.ndarray) -> None:
        del sample_rate
        self.waveform = np.asarray(waveform, dtype=np.float32).copy()

    def input_finished(self) -> None:
        return None


class _FakeExtractor:
    """A deterministic 512d extractor with the sherpa stream contract."""

    dim = 512

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def create_stream(self) -> _FakeStream:
        return _FakeStream()

    def compute(self, stream: _FakeStream) -> np.ndarray:
        assert stream.waveform is not None
        wave = np.asarray(stream.waveform, dtype=np.float32)
        self.inputs.append(wave.copy())
        signature = float(np.mean(wave) * 13.0 + np.std(wave) * 7.0)
        signature += float(np.mean(np.abs(wave)) * 5.0)
        index = np.arange(self.dim, dtype=np.float32)
        return np.sin(index * 0.017 + signature) + 0.2 * np.cos(
            index * 0.031 - signature
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audio(path: Path, frequency: float, *, seconds: float = 2.5) -> None:
    sample_count = int(round(seconds * 16_000))
    t = np.arange(sample_count, dtype=np.float32) / 16_000.0
    wave = (0.2 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, 16_000, subtype="FLOAT")


def _make_rows(root: Path, split: str, *, group_id: str, speaker_prefix: str) -> list[dict]:
    split_dir = root / split
    mixture = split_dir / "recognition" / f"{group_id}.wav"
    _write_audio(mixture, 300.0 if split == "train" else 500.0)
    mixture_sha = _sha256(mixture)
    rows = []
    roles = ((0, "present_A", "A"), (1, "present_B", "B"), (2, "absent_C", "C"))
    for role_id, role, role_letter in roles:
        enrollment = split_dir / "enrollment" / f"{group_id}_{role_letter}.wav"
        view2 = split_dir / "enrollment_view2" / f"{group_id}_{role_letter}.wav"
        _write_audio(enrollment, 600.0 + role_id * 100.0, seconds=1.2)
        _write_audio(view2, 900.0 + role_id * 100.0, seconds=1.1)
        speaker = f"{speaker_prefix}_{role_letter}"
        rows.append(
            {
                "id": f"{group_id}__{role}",
                "base_mixture_id": group_id,
                "split": split,
                "query_role": role,
                "query_role_id": role_id,
                "query_speaker_id": speaker,
                "target_spk": speaker,
                "enrollment_spk": speaker,
                "enrollment_audio": str(enrollment),
                "enrollment_audio_view2": str(view2),
                "recognition_audio": str(mixture),
                "mixture_sha256": mixture_sha,
                "dataset_a_used": False,
                "dataset_a_policy": "forbidden",
                "target_src": f"{split}_source_{role_letter}.wav",
                "enrollment_src": f"{split}_enroll_source_{role_letter}.wav",
                "interferer_srcs": [f"{split}_interferer_{role_letter}.wav"],
                "mixture_sources": {
                    "A": f"{split}_mixture_A_{group_id}.wav",
                    "B": f"{split}_mixture_B_{group_id}.wav",
                },
                "mixture_speakers": {"A": f"{speaker_prefix}_A", "B": f"{speaker_prefix}_B"},
            }
        )
    return rows


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class BuildDacfCamppFeaturesTest(unittest.TestCase):
    def test_build_deduplicates_mixture_and_keeps_two_query_views(self) -> None:
        root = FIXTURE_ROOT / "build_dedup"
        root.mkdir(parents=True, exist_ok=True)
        train_rows = _make_rows(root, "train", group_id="train_mix", speaker_prefix="tr")
        val_rows = _make_rows(root, "val", group_id="val_mix", speaker_prefix="va")
        train_manifest = root / "train_manifest.jsonl"
        val_manifest = root / "val_manifest.jsonl"
        _write_manifest(train_manifest, train_rows)
        _write_manifest(val_manifest, val_rows)
        model = root / "campplus.onnx"
        model.write_bytes(b"fake-campp-model")
        output = root / "features"
        fake = _FakeExtractor()

        report = build_feature_dataset(
            train_manifest,
            val_manifest,
            model_path=model,
            output_dir=output,
            extractor=fake,
        )

        self.assertEqual(report["counts"]["groups"], 2)
        self.assertEqual(report["counts"]["unique_mixtures"], 2)
        self.assertEqual(report["counts"]["mixture_feature_count"], 2)
        self.assertEqual(report["counts"]["enrollment_embedding_count"], 6)
        self.assertEqual(report["counts"]["enrollment_view2_embedding_count"], 6)
        self.assertFalse(report["dataset_a_used"])
        self.assertEqual(report["audit"]["speaker_overlap"], [])
        self.assertEqual(report["audit"]["source_overlap"], [])

        output_rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        self.assertEqual(len(output_rows), 6)
        for group in ("train_mix", "val_mix"):
            group_rows = [row for row in output_rows if row["base_mixture_id"] == group]
            mixture_paths = {
                row["campp_features"]["mixture_feature_npz"] for row in group_rows
            }
            self.assertEqual(len(mixture_paths), 1)
            self.assertTrue(
                all(
                    row["campp_features"]["query_role_id_used_as_feature_input"] is False
                    for row in group_rows
                )
            )
            enrollment_paths = {
                row["campp_features"]["enrollment_embedding_npy"]
                for row in group_rows
            }
            view2_paths = {
                row["campp_features"]["enrollment_view2_embedding_npy"]
                for row in group_rows
            }
            self.assertEqual(len(enrollment_paths), 3)
            self.assertEqual(len(view2_paths), 3)

        for path in (output / "query").glob("*.npy"):
            embedding = np.load(path)
            self.assertEqual(embedding.shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
        mixture_files = {
            output / row["campp_features"]["mixture_feature_npz"]
            for row in output_rows
        }
        self.assertEqual(len(mixture_files), 2)
        with np.load(sorted(mixture_files)[0]) as data:
            self.assertEqual(data["mixture_embedding"].shape, (512,))
            self.assertEqual(data["window_embeddings"].shape[1], 512)
            self.assertEqual(
                data["window_start_samples"][-1] + 24_000,
                data["window_end_samples"][-1] + data["window_padded_samples"][-1],
            )
        for split in ("train", "val"):
            for key in (
                "whole_cosine_auc",
                "window_max_auc",
                "top25_mean_auc",
            ):
                self.assertIn(key, report["capacity_metrics"][split]["enrollment_audio"])
            agreement = report["capacity_metrics"][split]["enrollment_view_agreement"]
            self.assertGreater(agreement["count"], 0)
            self.assertTrue(-1.0 <= agreement["cosine_min"] <= 1.0)

        # 2 mixtures x (whole + 3 windows) + 12 enrollment views.
        self.assertEqual(len(fake.inputs), 20)

    def test_window_tail_and_short_audio_padding(self) -> None:
        windows = window_schedule(41_000)
        self.assertEqual(windows[0], (0, 24_000))
        self.assertEqual(windows[-1], (17_000, 41_000))
        self.assertEqual(_pad_signal(np.ones(100, dtype=np.float32), 24_000)[0].shape, (24_000,))
        self.assertEqual(window_schedule(1_000), [(0, 1_000)])

    def test_dataset_a_path_and_flag_are_hard_rejected(self) -> None:
        root = FIXTURE_ROOT / "dataset_a_guard"
        root.mkdir(parents=True, exist_ok=True)
        bad_path = root / "bad_dataset_a.jsonl"
        bad_path.write_text(
            json.dumps(
                {
                    "dataset_a_used": True,
                    "recognition_audio": "safe.wav",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ManifestContractError):
            validate_manifests(bad_path, bad_path)
        bad_marker = root / "bad_marker.jsonl"
        bad_marker.write_text(
            json.dumps({"recognition_audio": "E:/test_wav/dataset/raw/a.wav"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ManifestContractError):
            validate_manifests(bad_marker, bad_marker)

    def test_speaker_and_source_split_audit_rejects_overlap(self) -> None:
        root = FIXTURE_ROOT / "split_audit"
        root.mkdir(parents=True, exist_ok=True)
        train_rows = _make_rows(root, "train", group_id="train_mix", speaker_prefix="same")
        val_rows = _make_rows(root, "val", group_id="val_mix", speaker_prefix="other")
        train_manifest = root / "train_manifest.jsonl"
        val_manifest = root / "val_manifest.jsonl"
        _write_manifest(train_manifest, train_rows)
        _write_manifest(val_manifest, val_rows)

        val_rows[0]["query_speaker_id"] = train_rows[0]["query_speaker_id"]
        val_rows[0]["target_spk"] = train_rows[0]["target_spk"]
        val_rows[0]["enrollment_spk"] = train_rows[0]["enrollment_spk"]
        _write_manifest(val_manifest, val_rows)
        with self.assertRaises(ManifestContractError):
            validate_manifests(train_manifest, val_manifest)

        val_rows[0]["query_speaker_id"] = "other_A"
        val_rows[0]["target_spk"] = "other_A"
        val_rows[0]["enrollment_spk"] = "other_A"
        val_rows[0]["target_src"] = train_rows[0]["target_src"]
        _write_manifest(val_manifest, val_rows)
        with self.assertRaises(ManifestContractError):
            validate_manifests(train_manifest, val_manifest)


if __name__ == "__main__":
    unittest.main()
