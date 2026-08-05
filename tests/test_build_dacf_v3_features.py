"""Fake-only contract tests for the DACF-v3 feature cache.

The fixture is deliberately repo-local because TemporaryDirectory and
Windows symlink creation are not reliable on this machine.  No real Qwen or
CAM++ model is loaded and no full dataset is touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import wave
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_v3_feature_cache_test_fixture"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_v3_features import (  # noqa: E402
    FeatureContractError,
    build_feature_cache,
    validate_cache,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_wav(path: Path, samples: int, *, amplitude: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timeline = np.arange(samples, dtype=np.float32)
    waveform = amplitude * np.sin(2.0 * np.pi * timeline / 37.0)
    pcm = np.clip(waveform * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class FakeQwenFeatureExtractor:
    feature_size = 128
    n_fft = 400
    hop_length = 160
    dither = 0.0
    sampling_rate = 16_000

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, waveform, *, sampling_rate, return_tensors="np", padding=False):
        assert sampling_rate == 16_000
        assert return_tensors == "np"
        assert padding is False
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        frames = max(1, int(np.ceil(signal.size / 160.0)))
        call_index = len(self.calls) + 1
        self.calls.append(int(signal.size))
        values = np.arange(128 * frames, dtype=np.float32).reshape(1, 128, frames)
        values = values + np.float32(call_index) / 100.0
        return {
            "input_features": values,
            "feature_attention_mask": np.ones((1, frames), dtype=np.int8),
        }


class FakeCamppExtractor:
    dim = 512

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, waveform, sample_rate):
        assert sample_rate == 16_000
        self.calls += 1
        return np.arange(1, 513, dtype=np.float32) + np.float32(self.calls)


class BuildDacfV3FeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        if FIXTURE_ROOT.exists():
            shutil.rmtree(FIXTURE_ROOT)
        self.case_root = FIXTURE_ROOT / "case"
        self.artifact_root = self.case_root / "artifacts"
        self.source_root = self.case_root / "aishell_source"
        self.config_root = self.case_root / "qwen_config"
        self.manifest_root = self.case_root / "manifests"
        self.model_path = self.case_root / "campplus.onnx"
        self.config_root.mkdir(parents=True, exist_ok=True)
        (self.config_root / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "feature_extractor_type": "WhisperFeatureExtractor",
                    "feature_size": 128,
                    "n_fft": 400,
                    "hop_length": 160,
                    "dither": 0.0,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.model_path.write_bytes(b"fake-local-campp-onnx")
        self.manifests = self._make_manifests()
        self.qwen = FakeQwenFeatureExtractor()
        self.campp = FakeCamppExtractor()

    def tearDown(self) -> None:
        if FIXTURE_ROOT.exists():
            shutil.rmtree(FIXTURE_ROOT)

    def _make_manifests(self) -> dict[str, Path]:
        manifests: dict[str, Path] = {}
        for split, split_index in (("train", 0), ("dev", 1), ("final", 2)):
            group_id = f"{split}_mix_0000"
            mixture = self.artifact_root / split / "recognition" / f"{group_id}.wav"
            _write_wav(mixture, 640, amplitude=0.08 + split_index * 0.01)
            mixture_sha = _sha256(mixture)
            speakers = [f"{split}_spk_A", f"{split}_spk_B", f"{split}_spk_C"]
            source_a = self.source_root / split / f"{group_id}_source_A.wav"
            source_b = self.source_root / split / f"{group_id}_source_B.wav"
            source_c = self.source_root / split / f"{group_id}_source_C.wav"
            for source, offset in ((source_a, 0), (source_b, 1), (source_c, 2)):
                # Source identity is audited by raw SHA, so synthetic speakers
                # from different logical splits must not accidentally have
                # byte-identical waveforms.
                _write_wav(
                    source,
                    320 + offset * 16,
                    amplitude=0.02 + split_index * 0.007 + offset * 0.001,
                )

            rows: list[dict] = []
            for role_id, role in enumerate(("present_A", "present_B", "absent_C")):
                row_id = f"{group_id}__{role}"
                speaker = speakers[role_id]
                enrollment = self.artifact_root / split / "enrollment" / f"{row_id}.wav"
                view2 = self.artifact_root / split / "enrollment_view2" / f"{row_id}.wav"
                clean_target = self.artifact_root / split / "clean_target" / f"{row_id}.wav"
                _write_wav(
                    enrollment,
                    320 + role_id * 16,
                    amplitude=0.04 + (split_index * 3 + role_id) * 0.01,
                )
                _write_wav(
                    view2,
                    336 + role_id * 16,
                    amplitude=0.05 + (split_index * 3 + role_id) * 0.01,
                )
                _write_wav(
                    clean_target,
                    480,
                    amplitude=0.0 if role == "absent_C" else 0.06 + split_index * 0.01,
                )
                activity_path = self.artifact_root / split / "activity" / f"{row_id}.npy"
                activity_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(
                    activity_path,
                    np.zeros(5, dtype=np.float32)
                    if role == "absent_C"
                    else np.asarray([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32),
                )
                rows.append(
                    {
                        "id": row_id,
                        "split": split,
                        "base_mixture_id": group_id,
                        "query_role": role,
                        "query_role_id": role_id,
                        "query_speaker_id": speaker,
                        "enrollment_spk": speaker,
                        "enrollment_view_count": 2,
                        "identity_positive": True,
                        "enrollment_noise_raw_sha256": f"main-noise-{split}-{role_id}",
                        "enrollment_view2_noise_raw_sha256": f"view2-noise-{split}-{role_id}",
                        "target_present": role != "absent_C",
                        "source_corpus": "AISHELL-1",
                        "dataset_a_used": False,
                        "dataset_a_policy": "forbidden",
                        "recognition_audio": str(mixture),
                        "mixture_sha256": mixture_sha,
                        "enrollment_audio": str(enrollment),
                        "enrollment_audio_view2": str(view2),
                        "enrollment_sha256": _sha256(enrollment),
                        "enrollment_view2_sha256": _sha256(view2),
                        "clean_target_audio": str(clean_target),
                        "target_activity": str(activity_path),
                        "enrollment_src": str(source_a if role == "present_A" else source_b if role == "present_B" else source_c),
                        "target_src": str(source_a if role == "present_A" else source_b if role == "present_B" else source_c),
                        "interferer_srcs": [str(source_b if role == "present_A" else source_a)],
                        "mixture_sources": {"A": str(source_a), "B": str(source_b)},
                        "mixture_speakers": {"A": speakers[0], "B": speakers[1]},
                        "target_spk": speaker,
                        "interferer_spks": [speakers[1] if role == "present_A" else speakers[0]],
                    }
                )
            manifest = self.manifest_root / f"{split}.jsonl"
            _write_jsonl(manifest, rows)
            manifests[split] = manifest
        return manifests

    def _build(self, *, output_name: str = "cache") -> Path:
        output = self.case_root / output_name
        build_feature_cache(
            self.manifests["train"],
            self.manifests["dev"],
            self.manifests["final"],
            qwen_config_dir=self.config_root,
            campp_model=self.model_path,
            output_dir=output,
            qwen_extractor=self.qwen,
            campp_extractor=self.campp,
            allowed_source_root=self.source_root,
        )
        return output

    def _rows(self, split: str) -> list[dict]:
        path = self.manifests[split]
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_dedup_shape_normalization_alignment_and_provenance(self) -> None:
        output = self._build()
        report = validate_cache(output)
        self.assertEqual(report["counts"]["unique_mixture_sha256"], 3)
        self.assertEqual(report["counts"]["qwen_mixture_feature_calls"], 3)
        self.assertEqual(report["counts"]["qwen_clean_target_feature_calls"], 9)
        self.assertEqual(report["counts"]["qwen_feature_extractor_calls"], 12)
        self.assertEqual(report["counts"]["campp_enrollment_extractor_calls"], 18)
        self.assertTrue(report["deduplication"]["one_call_per_unique_mixture"])
        for pair in report["overlap_audit"].values():
            for values in pair.values():
                self.assertEqual(values, [])
        self.assertFalse(list(output.rglob("*.wav")))

        mixture_path = next((output / "mixture").glob("*.npz"))
        with np.load(mixture_path, allow_pickle=False) as mixture:
            self.assertEqual(mixture["input_features"].shape, (128, 4))
            self.assertEqual(mixture["feature_attention_mask"].shape, (4,))
            self.assertEqual(int(mixture["feature_size"]), 128)
            self.assertEqual(int(mixture["n_fft"]), 400)
            self.assertEqual(int(mixture["hop_length"]), 160)
            self.assertEqual(float(mixture["dither"]), 0.0)

        rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        present = next(row for row in rows if row["split"] == "train" and row["query_role"] == "present_A")
        absent = next(row for row in rows if row["split"] == "train" and row["query_role"] == "absent_C")
        with np.load(output / present["query_feature"], allow_pickle=False) as query:
            self.assertNotIn("query_role_id", query.files)
            self.assertEqual(query["enrollment_embedding"].shape, (512,))
            self.assertEqual(query["enrollment_embedding_view2"].shape, (512,))
            self.assertAlmostEqual(float(np.linalg.norm(query["enrollment_embedding"])), 1.0, places=5)
            self.assertAlmostEqual(float(np.linalg.norm(query["enrollment_embedding_view2"])), 1.0, places=5)
            np.testing.assert_array_equal(query["target_activity"], np.asarray([1, 0, 1, 0], dtype=np.float32))
            self.assertEqual(query["clean_target_input_features"].shape, (128, 4))
            self.assertEqual(json.loads(str(query["activity_alignment_json"].item()))["policy"], "tail_crop")
        with np.load(output / absent["query_feature"], allow_pickle=False) as query:
            self.assertFalse(bool(query["target_present"]))
            self.assertTrue(np.all(query["target_activity"] == 0.0))

    def test_chunked_train_and_deferred_final_do_not_read_final(self) -> None:
        second_rows = self._rows("train")
        second_mixture = self.artifact_root / "train" / "recognition" / "train_mix_0001.wav"
        _write_wav(second_mixture, 672, amplitude=0.13)
        second_mixture_sha = _sha256(second_mixture)
        for row in second_rows:
            row["id"] = str(row["id"]).replace("train_mix_0000", "train_mix_0001")
            row["base_mixture_id"] = "train_mix_0001"
            row["recognition_audio"] = str(second_mixture)
            row["mixture_sha256"] = second_mixture_sha
        second_manifest = self.manifest_root / "train_chunk_0001.jsonl"
        _write_jsonl(second_manifest, second_rows)

        output = self.case_root / "cache_deferred_final"
        build_feature_cache(
            [self.manifests["train"], second_manifest],
            [self.manifests["dev"]],
            None,
            qwen_config_dir=self.config_root,
            campp_model=self.model_path,
            output_dir=output,
            qwen_extractor=self.qwen,
            campp_extractor=self.campp,
            allowed_source_root=self.source_root,
            include_clean_target_logmel=False,
        )
        report = validate_cache(output)
        self.assertEqual(report["split_contract"]["splits"], ["train", "dev"])
        self.assertTrue(report["split_contract"]["final_deferred"])
        self.assertIsNone(report["split_contract"]["final_gate_split"])
        self.assertEqual(report["counts"]["split_counts"]["train"]["groups"], 2)
        self.assertEqual(report["counts"]["split_counts"]["dev"]["groups"], 1)
        self.assertNotIn("final", report["counts"]["split_counts"])
        rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual({row["split"] for row in rows}, {"train", "dev"})
        self.assertEqual(report["counts"]["qwen_clean_target_feature_calls"], 0)

    def test_duplicate_chunk_ids_are_rejected_before_cache_writes(self) -> None:
        with self.assertRaisesRegex(FeatureContractError, "duplicate row_id"):
            build_feature_cache(
                [self.manifests["train"], self.manifests["train"]],
                self.manifests["dev"],
                None,
                qwen_config_dir=self.config_root,
                campp_model=self.model_path,
                output_dir=self.case_root / "cache_duplicate_chunk_ids",
                qwen_extractor=self.qwen,
                campp_extractor=self.campp,
                allowed_source_root=self.source_root,
                include_clean_target_logmel=False,
            )

    def test_protocol_val_alias_is_normalized_to_logical_dev(self) -> None:
        dev_rows = self._rows("dev")
        for row in dev_rows:
            row["split"] = "val"
        _write_jsonl(self.manifests["dev"], dev_rows)
        output = self.case_root / "cache_val_alias"
        build_feature_cache(
            self.manifests["train"],
            self.manifests["dev"],
            None,
            qwen_config_dir=self.config_root,
            campp_model=self.model_path,
            output_dir=output,
            qwen_extractor=self.qwen,
            campp_extractor=self.campp,
            allowed_source_root=self.source_root,
            include_clean_target_logmel=False,
        )
        report = validate_cache(output)
        self.assertEqual(report["split_contract"]["splits"], ["train", "dev"])
        rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual({row["split"] for row in rows}, {"train", "dev"})

    def test_exact_qwen_frontend_spec_guard(self) -> None:
        config_path = self.config_root / "preprocessor_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["n_fft"] = 512
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(FeatureContractError, "preprocessor_config.n_fft"):
            self._build()

    def test_audio_declared_hash_tamper_is_rejected(self) -> None:
        rows = self._rows("train")
        rows[0]["enrollment_sha256"] = "0" * 64
        _write_jsonl(self.manifests["train"], rows)
        with self.assertRaisesRegex(FeatureContractError, "enrollment_sha256"):
            self._build()

    def test_source_row_semantics_remain_bound_after_report_hash_rewrite(self) -> None:
        output = self._build(output_name="source_row_binding")
        rows = self._rows("train")
        rows[0]["ref"] = "synthetic but semantically changed source row"
        _write_jsonl(self.manifests["train"], rows)
        report_path = output / "cache_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["input_manifests"]["train"][0]["sha256"] = _sha256(
            self.manifests["train"]
        )
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(FeatureContractError, "source manifest row SHA mismatch"):
            validate_cache(output)

    def test_query_npz_metadata_tamper_is_rejected(self) -> None:
        output = self._build()
        manifest_rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        query_path = output / next(row["query_feature"] for row in manifest_rows)
        with np.load(query_path, allow_pickle=False) as query:
            values = {key: query[key] for key in query.files}
        values["row_id"] = np.asarray("tampered-row")
        np.savez_compressed(query_path, **values)
        with self.assertRaisesRegex(FeatureContractError, "query_npz_sha256 mismatch"):
            validate_cache(output)

    def test_deferred_final_report_tamper_is_rejected(self) -> None:
        output = self.case_root / "cache_deferred_tamper"
        build_feature_cache(
            self.manifests["train"],
            self.manifests["dev"],
            None,
            qwen_config_dir=self.config_root,
            campp_model=self.model_path,
            output_dir=output,
            qwen_extractor=self.qwen,
            campp_extractor=self.campp,
            allowed_source_root=self.source_root,
            include_clean_target_logmel=False,
        )
        report_path = output / "cache_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["split_contract"]["final_deferred"] = False
        report_path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(FeatureContractError, "final_deferred"):
            validate_cache(output)

    def test_cross_split_speaker_source_and_mixture_overlap_are_rejected(self) -> None:
        train_rows = self._rows("train")
        dev_rows = self._rows("dev")
        dev_rows[0]["query_speaker_id"] = train_rows[0]["query_speaker_id"]
        dev_rows[0]["enrollment_spk"] = train_rows[0]["query_speaker_id"]
        _write_jsonl(self.manifests["dev"], dev_rows)
        with self.assertRaisesRegex(FeatureContractError, "train_vs_dev speaker_overlap"):
            self._build(output_name="speaker_overlap")

        self.setUp()
        train_rows = self._rows("train")
        dev_rows = self._rows("dev")
        dev_rows[0]["enrollment_src"] = train_rows[0]["enrollment_src"]
        _write_jsonl(self.manifests["dev"], dev_rows)
        with self.assertRaisesRegex(FeatureContractError, "train_vs_dev source_path_overlap"):
            self._build(output_name="source_overlap")

        self.setUp()
        train_rows = self._rows("train")
        final_rows = self._rows("final")
        for final_row in final_rows:
            final_row["recognition_audio"] = train_rows[0]["recognition_audio"]
            final_row["mixture_sha256"] = train_rows[0]["mixture_sha256"]
        _write_jsonl(self.manifests["final"], final_rows)
        with self.assertRaisesRegex(FeatureContractError, "train_vs_final mixture_sha256_overlap"):
            self._build(output_name="mixture_overlap")

    def test_dataset_a_and_parent_escape_are_rejected(self) -> None:
        rows = self._rows("train")
        rows[0]["dataset_a_used"] = True
        _write_jsonl(self.manifests["train"], rows)
        with self.assertRaises(FeatureContractError):
            self._build(output_name="dataset_a_flag")

        self.setUp()
        rows = self._rows("train")
        rows[0]["enrollment_src"] = "../outside.wav"
        _write_jsonl(self.manifests["train"], rows)
        with self.assertRaisesRegex(FeatureContractError, "forbidden '..'"):
            self._build(output_name="parent_escape")

    def test_symlink_escape_is_rejected_when_windows_allows_symlinks(self) -> None:
        real_source = Path(self._rows("train")[0]["enrollment_src"])
        link = self.source_root / "train" / "symlink_source.wav"
        try:
            os.symlink(real_source, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable on this Windows runner")
        rows = self._rows("train")
        rows[0]["enrollment_src"] = str(link)
        _write_jsonl(self.manifests["train"], rows)
        with self.assertRaisesRegex(FeatureContractError, "symlink"):
            self._build(output_name="symlink_escape")


if __name__ == "__main__":
    unittest.main(verbosity=2)
