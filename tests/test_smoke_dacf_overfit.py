from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import soundfile as sf

EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from smoke_dacf_overfit import (  # noqa: E402
    LOSS_NAMES,
    load_dacf_group,
    main,
    run_overfit,
)


SAMPLE_RATE = 16_000


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def _make_manifest(fixture_root: Path) -> Path:
    root = fixture_root / "dacf_fixture"
    sample_count = 1_600
    t = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    target_a = 0.20 * np.sin(2.0 * np.pi * 300.0 * t)
    target_b = 0.15 * np.sin(2.0 * np.pi * 700.0 * t)
    mixture = target_a + target_b
    mixture_path = root / "recognition" / "mix.wav"
    _write_wav(mixture_path, mixture)
    mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()

    rows = []
    specs = (
        ("present_A", 0, "speaker-global-A", 10, True, target_a, "开灯"),
        ("present_B", 1, "speaker-global-B", 11, True, target_b, "关灯"),
        ("absent_C", 2, "speaker-global-C", 12, False, np.zeros_like(mixture), ""),
    )
    for role, role_id, speaker, speaker_label, present, clean, transcript in specs:
        enrollment = target_a if role == "present_A" else target_b if role == "present_B" else target_a * 0.5
        enrollment_path = root / "enrollment" / f"{role}.wav"
        enrollment_view2_path = root / "enrollment_view2" / f"{role}.wav"
        clean_path = root / "clean" / f"{role}.wav"
        activity_path = root / "activity" / f"{role}.npy"
        _write_wav(enrollment_path, enrollment)
        _write_wav(
            enrollment_view2_path,
            enrollment + 0.01 * np.cos(2.0 * np.pi * 90.0 * t),
        )
        _write_wav(clean_path, clean)
        activity = np.zeros(sample_count // 160, dtype=np.uint8)
        if present:
            activity[3:8] = 1
        activity_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(activity_path, activity)
        rows.append(
            {
                "base_mixture_id": "synthetic_mix_0",
                "query_role": role,
                "query_role_id": role_id,
                "query_speaker_id": speaker,
                "query_speaker_label": speaker_label,
                "environment_id": "synthetic_env_0",
                "enrollment_view_count": 2,
                "target_present": present,
                "target_transcript": transcript,
                "recognition_audio": str(mixture_path),
                "mixture_sha256": mixture_sha,
                "enrollment_audio": str(enrollment_path),
                "enrollment_audio_view2": str(enrollment_view2_path),
                "clean_target_audio": str(clean_path),
                "target_activity": str(activity_path),
            }
        )

    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


class SmokeDACFOverfitTest(unittest.TestCase):
    def temporary_directory(self, prefix: str):
        # Keep test artifacts in the attached workspace.  TemporaryDirectory
        # still owns cleanup; this avoids the locked system TEMP on this host.
        return tempfile.TemporaryDirectory(
            prefix=prefix,
            dir=Path(__file__).resolve().parents[1],
        )

    def test_dataset_a_guard_happens_before_audio_read(self):
        with self.temporary_directory("_dacf_guard_") as tmp:
            fixture_root = Path(tmp)
            manifest = _make_manifest(fixture_root)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            forbidden = fixture_root / "Dataset-A" / "forbidden.wav"
            rows[0]["recognition_audio"] = str(forbidden)
            forbidden_manifest = fixture_root / "guard_manifest.jsonl"
            forbidden_manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                load_dacf_group(forbidden_manifest)

    def test_mixture_sha_mismatch_fails_fast(self):
        with self.temporary_directory("_dacf_sha_") as tmp:
            fixture_root = Path(tmp)
            manifest = _make_manifest(fixture_root)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            rows[2]["mixture_sha256"] = "0" * 64
            bad_manifest = fixture_root / "sha_manifest.jsonl"
            bad_manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "mixture_sha256 mismatch"):
                load_dacf_group(bad_manifest)

    def test_repo_root_relative_audio_paths_resolve_from_cwd(self):
        with self.temporary_directory("_dacf_repo_relative_") as tmp:
            fixture_root = Path(tmp)
            manifest = _make_manifest(fixture_root)
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            repo_root = Path(__file__).resolve().parents[1]
            path_fields = (
                "recognition_audio",
                "enrollment_audio",
                "enrollment_audio_view2",
                "clean_target_audio",
                "target_activity",
            )
            for row in rows:
                for field in path_fields:
                    row[field] = os.path.relpath(row[field], repo_root)
            relative_manifest = fixture_root / "repo_relative_manifest.jsonl"
            relative_manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            loaded = load_dacf_group(relative_manifest)
            self.assertEqual(loaded.effective_batch_size, 6)
            self.assertEqual(loaded.mixture_path, Path(rows[0]["recognition_audio"]).resolve())

    def test_short_overfit_backprop_and_metric_contract(self):
        with self.temporary_directory("_dacf_overfit_") as tmp:
            manifest = _make_manifest(Path(tmp))
            result = run_overfit(
                manifest,
                steps=2,
                seed=17,
                device="cpu",
            )

            self.assertEqual(result["status"], "smoke_only")
            self.assertFalse(result["dataset_a_read"])
            self.assertFalse(result["submission_chain_used"])
            self.assertEqual(result["steps"], 2)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["seed"], 17)
            self.assertTrue(result["backward_ok"])
            self.assertEqual(result["effective_batch_size"], 6)
            self.assertEqual(result["identity_positive_pairs"], 3)
            self.assertEqual(result["identity_source"], "query_speaker_label")
            self.assertEqual(result["counterfactual_source"], "query_role_id")
            self.assertEqual(result["model"]["n_fft"], 400)
            self.assertEqual(result["model"]["hop_length"], 160)
            self.assertEqual(result["model"]["win_length"], 400)
            self.assertEqual(result["activity_hop_samples"], 160)
            self.assertEqual(result["vocab"]["blank_index"], 0)
            self.assertGreater(result["vocab"]["char_to_id"]["开"], 0)
            self.assertGreater(result["vocab"]["char_to_id"]["关"], 0)

            for phase in ("before", "after"):
                self.assertEqual(set(result[phase]["losses"]), set(LOSS_NAMES))
                self.assertTrue(np.isfinite(result[phase]["losses"]["total"]))
                self.assertEqual(
                    set(result[phase]["presence_prob"]),
                    {"present_A", "present_B", "absent_C"},
                )
                self.assertEqual(
                    set(result[phase]["target_l1"]),
                    {"present_A", "present_B", "absent_C"},
                )
                self.assertTrue(np.isfinite(result[phase]["absent_output_rms"]))
                self.assertTrue(
                    np.isfinite(
                        result[phase]["query_swap_audio_delta"]["mean_pairwise"]
                    )
                )

            negative = result["permutation_negative_control"]
            self.assertTrue(negative["labels_unchanged"])
            self.assertEqual(negative["permutation"], [2, 3, 0, 1, 4, 5])
            self.assertEqual(set(negative["loss_change"]), set(LOSS_NAMES))
            self.assertEqual(
                set(negative["presence_prob_change"]),
                {"present_A", "present_B", "absent_C"},
            )
            self.assertIn("not a statistical", negative["note"])

            absent_control = result["permutation_absent_control"]
            self.assertTrue(absent_control["labels_unchanged"])
            self.assertEqual(absent_control["permutation"], [4, 5, 2, 3, 0, 1])
            self.assertEqual(
                set(absent_control["presence_prob_change"]),
                {"present_A", "present_B", "absent_C"},
            )

            loaded = load_dacf_group(manifest)
            targets = loaded.targets(device="cpu")
            self.assertIn("query_role_id", targets)
            self.assertIn("query_speaker_label", targets)
            self.assertNotIn("query_id", targets)
            self.assertEqual(
                targets["query_speaker_label"].tolist(), [0, 0, 1, 1, 2, 2]
            )
            self.assertEqual(targets["query_role_id"].tolist(), [0, 0, 1, 1, 2, 2])

    def test_disable_ctc_is_explicitly_reported(self):
        with self.temporary_directory("_dacf_no_ctc_") as tmp:
            manifest = _make_manifest(Path(tmp))
            result = run_overfit(
                manifest,
                steps=1,
                seed=19,
                device="cpu",
                disable_ctc=True,
            )
            self.assertFalse(result["ctc_enabled"])
            self.assertIsNone(result["before"]["losses"]["ctc"])
            self.assertIsNone(result["after"]["losses"]["ctc"])

    def test_cli_output_json_and_cpu_peak_memory_field(self):
        with self.temporary_directory("_dacf_output_json_") as tmp:
            fixture_root = Path(tmp)
            manifest = _make_manifest(fixture_root)
            output_path = fixture_root / "nested" / "smoke-result.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main(
                    [
                        "--manifest",
                        str(manifest),
                        "--steps",
                        "0",
                        "--device",
                        "cpu",
                        "--output-json",
                        str(output_path),
                    ]
                )

            saved_text = output_path.read_text(encoding="utf-8")
            saved = json.loads(saved_text)
            self.assertTrue(saved_text.endswith("\n"))
            self.assertIsNone(saved["cuda_peak_memory_mib"])
            self.assertEqual(saved["steps"], 0)
            self.assertIn('"开"', saved_text)
            self.assertIn('"灯"', saved_text)
            self.assertEqual(json.loads(stdout.getvalue())["base_mixture_id"], "synthetic_mix_0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
