"""Contract tests for the frozen CAM++ pre-pool frame probe.

The fixture is a stable repository-local directory rather than
``TemporaryDirectory`` because Windows ACL handling for temporary paths is
known to make this project's subprocess/ORT tests flaky.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_campplus_frame_features_20260806" / "_test_fixture"
ORIGINAL_MODEL = Path("E:/hf_cache/campplus/campplus.onnx")
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_campp_frame_features import (  # noqa: E402
    PREPOOL_OUTPUT_NAME,
    ManifestContractError,
    add_prepool_output,
    build_feature_dataset,
    capacity_scores,
    safe_fbank_frame_count,
    validate_manifests,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_audio(path: Path, frequency: float, *, seconds: float = 0.04) -> None:
    count = max(640, int(round(seconds * 16_000)))
    t = np.arange(count, dtype=np.float32) / 16_000.0
    wave = (0.2 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave, 16_000, subtype="FLOAT")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_fixed_manifests(root: Path) -> tuple[Path, Path, Path]:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    d9_root = root / "d9"
    d9_query = d9_root / "query"
    d9_mixture = d9_root / "mixture"
    d9_query.mkdir(parents=True, exist_ok=True)
    d9_mixture.mkdir(parents=True, exist_ok=True)
    d9_manifest_rows: list[dict] = []

    for split, group_count, destination in (
        ("train", 24, train_rows),
        ("val", 8, val_rows),
    ):
        for group_index in range(group_count):
            group_id = f"{split}_mix_{group_index:04d}"
            mixture = root / split / "recognition" / f"{group_id}.wav"
            _write_audio(mixture, 220.0 + group_index + (0 if split == "train" else 100.0))
            mixture_sha = _sha256(mixture)
            mixture_sources = {
                "A": f"{split}_mix_source_A_{group_index:04d}.wav",
                "B": f"{split}_mix_source_B_{group_index:04d}.wav",
            }
            for role_id, role in ((0, "present_A"), (1, "present_B"), (2, "absent_C")):
                speaker = f"{split}_speaker_{group_index:04d}_{role_id}"
                enrollment = root / split / "enrollment" / f"{group_id}_{role_id}.wav"
                view2 = root / split / "enrollment_view2" / f"{group_id}_{role_id}.wav"
                _write_audio(enrollment, 300.0 + group_index * 2 + role_id)
                _write_audio(view2, 500.0 + group_index * 2 + role_id)
                row = {
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
                    "target_src": f"{split}_target_{group_index:04d}_{role_id}.wav",
                    "enrollment_src": f"{split}_enrollment_{group_index:04d}_{role_id}.wav",
                    "interferer_srcs": [f"{split}_interferer_{group_index:04d}_{role_id}.wav"],
                    "mixture_sources": mixture_sources,
                    "mixture_speakers": {
                        "A": f"{split}_speaker_{group_index:04d}_0",
                        "B": f"{split}_speaker_{group_index:04d}_1",
                    },
                }
                destination.append(row)
                # D9-style references deliberately match the fake backend output.
                d9_query_path = d9_query / f"{split}_{row['id']}__enrollment.npy"
                d9_view2_path = d9_query / f"{split}_{row['id']}__view2.npy"
                np.save(d9_query_path, np.ones(512, dtype=np.float32))
                np.save(d9_view2_path, np.ones(512, dtype=np.float32))
                d9_manifest_rows.append(
                    {
                        "id": row["id"],
                        "mixture_sha256": mixture_sha,
                        "campp_features": {
                            "enrollment_embedding_npy": str(d9_query_path.relative_to(d9_root)),
                            "enrollment_view2_embedding_npy": str(d9_view2_path.relative_to(d9_root)),
                            "mixture_audio_sha256_actual": mixture_sha,
                            "mixture_feature_npz": str(
                                (d9_mixture / f"{split}_{group_id}.npz").relative_to(d9_root)
                            ),
                        },
                    }
                )
            np.savez(
                d9_mixture / f"{split}_{group_id}.npz",
                mixture_embedding=np.ones(512, dtype=np.float32),
            )

    train_manifest = root / "train_manifest.jsonl"
    val_manifest = root / "val_manifest.jsonl"
    _write_manifest(train_manifest, train_rows)
    _write_manifest(val_manifest, val_rows)
    _write_manifest(d9_root / "features_manifest.jsonl", d9_manifest_rows)
    marker = root / "fake_campplus.onnx"
    marker.write_bytes(b"fake marker for injected backend")
    return train_manifest, val_manifest, d9_root


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
        assert sample_rate == 16_000
        wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
        self.calls.append(wave.copy())
        signature = float(np.std(wave) + np.mean(np.abs(wave)))
        index = np.arange(512, dtype=np.float32)
        vector = np.sin(index * 0.013 + signature) + 0.1 * np.cos(index * 0.031)
        prepool = np.stack(
            [vector + 0.01 * np.sin(index * 0.007 + frame) for frame in range(4)],
            axis=0,
        ).astype(np.float32)
        return {"prepool": prepool, "embedding": np.ones(512, dtype=np.float32)}


class BuildDacfCamppFrameFeaturesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = FIXTURE_ROOT
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.train_manifest, cls.val_manifest, cls.d9_root = _make_fixed_manifests(cls.root)

    def test_dynamic_graph_appends_output_without_touching_source(self) -> None:
        self.assertTrue(ORIGINAL_MODEL.exists())
        original_bytes = ORIGINAL_MODEL.read_bytes()
        modified = self.root / "graph" / "campplus_frame.onnx"
        info = add_prepool_output(ORIGINAL_MODEL, modified)
        self.assertEqual(ORIGINAL_MODEL.read_bytes(), original_bytes)
        self.assertEqual(info["prepool_output_name"], PREPOOL_OUTPUT_NAME)
        self.assertTrue(info["source_unchanged"])
        self.assertNotEqual(modified.read_bytes(), original_bytes)

    def test_capacity_score_is_fixed_whole_frame_top25(self) -> None:
        query = np.ones((4, 512), dtype=np.float32)
        mixture = np.ones((4, 512), dtype=np.float32)
        scores = capacity_scores(query, mixture)
        self.assertAlmostEqual(scores["whole_pre"], 1.0, places=5)
        self.assertAlmostEqual(scores["frame_max"], 1.0, places=5)
        self.assertAlmostEqual(scores["top25_pre"], 1.0, places=5)
        self.assertEqual(scores["top_k"], 1.0)

    def test_sherpa_frame_count_uses_floor_and_keeps_one_frame_minimum(self) -> None:
        self.assertEqual(safe_fbank_frame_count(160), 1)
        self.assertEqual(safe_fbank_frame_count(161), 1)
        self.assertEqual(safe_fbank_frame_count(31999), 199)
        with self.assertRaises(Exception):
            safe_fbank_frame_count(0)

    def test_fixed_24_8_build_reuses_mixture_and_keeps_d9_comparison(self) -> None:
        fake = _FakeBackend()
        output = self.root / "features"
        if output.exists():
            shutil.rmtree(output)
        report = build_feature_dataset(
            self.train_manifest,
            self.val_manifest,
            model_path=self.root / "fake_campplus.onnx",
            output_dir=output,
            d9_root=self.d9_root,
            backend=fake,
        )
        self.assertEqual(report["backend"], "sherpa-campp-onnx-prepool")
        self.assertEqual(report["counts"]["train_groups"], 24)
        self.assertEqual(report["counts"]["val_groups"], 8)
        self.assertEqual(report["counts"]["input_rows"], 96)
        self.assertEqual(report["counts"]["unique_mixtures"], 32)
        self.assertEqual(len(fake.calls), 32 + 2 * 96)
        self.assertFalse(report["dataset_a_used"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["threshold_selection_allowed"])
        self.assertEqual(report["audit"]["speaker_overlap"], [])
        self.assertEqual(report["audit"]["source_overlap"], [])
        for split in ("train", "val"):
            metrics = report["capacity_metrics"][split]["enrollment_audio"]
            for key in ("whole_pre_auc", "frame_max_auc", "top25_pre_auc"):
                self.assertIn(key, metrics)
            self.assertEqual(
                report["capacity_metrics"][split]["enrollment_view_agreement"]["count"],
                3 * (24 if split == "train" else 8),
            )
            self.assertIn(
                "d9_clean_enrollment_final_embedding_same_vs_different",
                report["capacity_metrics"][split],
            )
        self.assertEqual(
            report["d9_final_embedding_comparison"]["enrollment_audio"]["count"], 96
        )
        self.assertEqual(
            report["d9_final_embedding_comparison"]["enrollment_audio_view2"]["count"], 96
        )
        rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(rows), 96)
        group_rows = [row for row in rows if row["base_mixture_id"] == "train_mix_0000"]
        self.assertEqual(
            len({row["campp_frame_features"]["mixture_feature_npz"] for row in group_rows}),
            1,
        )
        self.assertTrue(
            all(
                row["campp_frame_features"]["query_role_id_used_as_feature_input"] is False
                for row in group_rows
            )
        )
        mixture_files = list((output / "mixture").glob("*.npz"))
        self.assertEqual(len(mixture_files), 32)
        with np.load(mixture_files[0], allow_pickle=False) as data:
            self.assertEqual(data["prepool"].shape, (4, 512))
            self.assertEqual(data["mixture_embedding"].shape, (512,))

    def test_dataset_a_and_split_overlap_are_rejected_before_feature_build(self) -> None:
        bad = self.root / "bad_dataset_a.jsonl"
        bad.write_text(json.dumps({"dataset_a_used": True}) + "\n", encoding="utf-8")
        with self.assertRaises(ManifestContractError):
            validate_manifests(bad, bad, expected_train_groups=None, expected_val_groups=None)

        rows = [
            json.loads(line)
            for line in self.val_manifest.read_text(encoding="utf-8").splitlines()
        ]
        train_rows = [
            json.loads(line)
            for line in self.train_manifest.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["query_speaker_id"] = train_rows[0]["query_speaker_id"]
        overlap_manifest = self.root / "val_manifest_speaker_overlap.jsonl"
        _write_manifest(overlap_manifest, rows)
        with self.assertRaisesRegex(ManifestContractError, "speaker overlap"):
            validate_manifests(self.train_manifest, overlap_manifest)

    def test_final_holdout_is_pairwise_disjoint(self) -> None:
        rows = [
            json.loads(line)
            for line in self.val_manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]

        def rewrite(value):
            if isinstance(value, dict):
                return {key: rewrite(child) for key, child in value.items()}
            if isinstance(value, list):
                return [rewrite(child) for child in value]
            if isinstance(value, str):
                return value.replace("val", "final_unique")
            return value

        final_rows = [rewrite(row) for row in rows]
        for row in final_rows:
            row["split"] = "final"
        final_manifest = self.root / "final_manifest.jsonl"
        _write_manifest(final_manifest, final_rows)
        bundle = validate_manifests(
            self.train_manifest,
            self.val_manifest,
            final_manifest=final_manifest,
            expected_final_groups=8,
        )
        self.assertEqual(len(bundle["final"]["groups"]), 8)
        self.assertFalse(
            any(
                values
                for values in bundle["audit"]["pairwise_overlap"]["train/final"].values()
            )
        )

        train_first = json.loads(
            self.train_manifest.read_text(encoding="utf-8").splitlines()[0]
        )
        final_rows[0]["query_speaker_id"] = train_first["query_speaker_id"]
        contaminated = self.root / "final_manifest_contaminated.jsonl"
        _write_manifest(contaminated, final_rows)
        with self.assertRaisesRegex(ManifestContractError, "train/final speaker overlap"):
            validate_manifests(
                self.train_manifest,
                self.val_manifest,
                final_manifest=contaminated,
                expected_final_groups=8,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
