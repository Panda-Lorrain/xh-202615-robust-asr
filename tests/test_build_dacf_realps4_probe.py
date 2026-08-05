from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_realps4_probe import (  # noqa: E402
    MAX_BASE_MIXTURES,
    SAMPLE_RATE,
    _canonical_path_key,
    build_dacf_realps4_probe,
    discover_segments,
    parse_segment_filename,
)


def _make_fixture(root: Path) -> Path:
    source_root = root / "REAL-PS4" / "AISHELL-4" / "enrolment_speakers"
    duration_samples = int(1.25 * SAMPLE_RATE)
    timeline = np.arange(duration_samples, dtype=np.float32) / SAMPLE_RATE
    for session_index in range(1, 3):
        for speaker_index in range(1, 5):
            speaker = f"{speaker_index:03d}-{'M' if speaker_index < 3 else 'F'}"
            for fragment_index in range(2):
                frequency = 180.0 + 23.0 * speaker_index + 5.0 * fragment_index
                audio = (
                    0.15
                    * np.sin(2.0 * np.pi * frequency * timeline)
                    * (1.0 - 0.1 * session_index)
                ).astype(np.float32)
                name = (
                    f"AISHELL-4_20200616_M_R001S{session_index:02d}C01_"
                    f"{speaker}_{100.0 + fragment_index * 2:.2f}_"
                    f"{101.25 + fragment_index * 2:.2f}.wav"
                )
                path = source_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return source_root


def _read_rows(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]


class TestDacfRealPs4Probe(unittest.TestCase):
    def test_parse_and_validate_realps4_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = _make_fixture(Path(temp))
            segments = discover_segments(source_root)
            self.assertEqual(len(segments), 16)
            parsed = parse_segment_filename(segments[0]["path"])
            self.assertIn(parsed["session"], {"S01", "S02"})
            self.assertIn(
                parsed["speaker_id"], {"001-M", "002-M", "003-F", "004-F"}
            )
            self.assertLess(parsed["start_sec"], parsed["end_sec"])
            self.assertEqual(
                {item["audio_format"]["samplerate"] for item in segments},
                {SAMPLE_RATE},
            )
            self.assertEqual(
                {item["audio_format"]["channels"] for item in segments}, {1}
            )

    def test_probe_is_deterministic_and_emits_complete_2x2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            source_root = _make_fixture(tmp_path)
            source_hashes = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_root.glob("*.wav")
            }
            out_dir = tmp_path / "probe"
            first_summary = build_dacf_realps4_probe(
                source_root,
                out_dir,
                base_mixtures=1,
                seed=20260806,
                sir_db=3.0,
            )
            manifest_before = (out_dir / "manifest.jsonl").read_bytes()
            summary_before = (out_dir / "summary.json").read_bytes()
            second_summary = build_dacf_realps4_probe(
                source_root,
                out_dir,
                base_mixtures=1,
                seed=20260806,
                sir_db=3.0,
            )
            self.assertEqual(
                (out_dir / "manifest.jsonl").read_bytes(), manifest_before
            )
            self.assertEqual((out_dir / "summary.json").read_bytes(), summary_before)
            self.assertEqual(first_summary, second_summary)
            self.assertEqual(
                {
                    path: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in source_root.glob("*.wav")
                },
                source_hashes,
            )

            rows = _read_rows(out_dir)
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {row["condition"] for row in rows}, {"same_env", "cross_env"}
            )
            self.assertEqual(
                {row["query_role"] for row in rows},
                {"present_A", "present_B", "absent_C"},
            )
            self.assertEqual(
                {(row["condition"], row["target_present"]) for row in rows},
                {
                    ("same_env", True),
                    ("same_env", False),
                    ("cross_env", True),
                    ("cross_env", False),
                },
            )
            self.assertEqual({row["query_role_id"] for row in rows}, {0, 1, 2})
            self.assertEqual(len({row["recognition_audio"] for row in rows}), 1)
            self.assertEqual(len({row["mixture_sha256"] for row in rows}), 1)
            recognition_path = Path(rows[0]["recognition_audio"])
            self.assertEqual(
                hashlib.sha256(recognition_path.read_bytes()).hexdigest(),
                rows[0]["mixture_sha256"],
            )
            self.assertEqual(
                first_summary["probe"]["comparison_grid_counts"],
                {
                    "same_env__correct": 2,
                    "same_env__wrong": 1,
                    "cross_env__correct": 2,
                    "cross_env__wrong": 1,
                },
            )
            self.assertEqual(first_summary["probe"]["record_count"], 6)
            self.assertIs(first_summary["source"]["dataset_a_used"], False)
            self.assertEqual(
                first_summary["source"]["speaker_label_map"],
                {"001-M": 0, "002-M": 1, "003-F": 2, "004-F": 3},
            )
            self.assertIs(
                first_summary["probe"]["synthetic_sum_of_real_fragments"], True
            )
            self.assertIs(
                first_summary["probe"]["mixture_is_native_real_overlap"], False
            )
            self.assertIs(first_summary["probe"]["background_is_superposed"], True)
            self.assertIs(first_summary["policy"]["training_allowed"], False)
            self.assertIs(
                first_summary["policy"]["threshold_selection_allowed"], False
            )
            self.assertIn("synthetic sum", first_summary["readme"]["design"].lower())
            self.assertIn("not a native", first_summary["readme"]["design"].lower())
            self.assertTrue(first_summary["readme"]["not_for"])

            for row in rows:
                self.assertIs(row["dataset_a_used"], False)
                self.assertIs(row["transcript_unavailable"], True)
                self.assertIs(row["enrollment_is_recognition_source"], False)
                self.assertNotIn(
                    row["enrollment_source_audio"], row["recognition_source_audio"]
                )
                self.assertEqual(
                    row["enrollment_source_key"],
                    _canonical_path_key(row["enrollment_source_audio"]),
                )
                self.assertNotIn(
                    row["enrollment_source_key"], row["recognition_source_keys"]
                )
                self.assertEqual(
                    row["recognition_source_keys"],
                    [
                        _canonical_path_key(source)
                        for source in row["recognition_source_audio"]
                    ],
                )
                self.assertEqual(
                    row["query_speaker_label"],
                    first_summary["source"]["speaker_label_map"][
                        row["query_speaker_id"]
                    ],
                )
                self.assertTrue(
                    row["mixture_environment_id"].startswith("synthetic_sum|")
                )
                self.assertIn("session=", row["enrollment_environment_id"])
                self.assertIs(row["mixture_is_native_real_overlap"], False)
                self.assertIs(row["background_is_superposed"], True)
                if row["condition"] == "same_env":
                    self.assertEqual(row["enrollment_session"], row["mixture_session"])
                else:
                    self.assertNotEqual(
                        row["enrollment_session"], row["mixture_session"]
                    )

            by_role = {
                row["query_role"]: row
                for row in rows
                if row["condition"] == "same_env"
            }
            absent = by_role["absent_C"]
            absent_clean, absent_sr = sf.read(
                absent["clean_target_audio"], dtype="float32"
            )
            absent_activity = np.load(absent["target_activity"])
            self.assertEqual(absent_sr, SAMPLE_RATE)
            self.assertEqual(np.count_nonzero(absent_clean), 0)
            self.assertEqual(np.count_nonzero(absent_activity), 0)

            present_a = by_role["present_A"]
            present_clean, _ = sf.read(
                present_a["clean_target_audio"], dtype="float32"
            )
            self.assertGreater(np.count_nonzero(present_clean), 0)

    def test_canonical_source_key_normalizes_windows_slashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            self.assertEqual(
                _canonical_path_key(str(source)),
                _canonical_path_key(source.as_posix()),
            )

    def test_multiple_bases_round_robin_mixture_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            source_root = _make_fixture(tmp_path)
            out_dir = tmp_path / "probe_two_bases"
            summary = build_dacf_realps4_probe(
                source_root,
                out_dir,
                base_mixtures=2,
                seed=20260806,
            )
            manifest_before = (out_dir / "manifest.jsonl").read_bytes()
            summary_before = (out_dir / "summary.json").read_bytes()
            summary_again = build_dacf_realps4_probe(
                source_root,
                out_dir,
                base_mixtures=2,
                seed=20260806,
            )
            self.assertEqual(summary_again, summary)
            self.assertEqual((out_dir / "manifest.jsonl").read_bytes(), manifest_before)
            self.assertEqual((out_dir / "summary.json").read_bytes(), summary_before)
            self.assertEqual(summary["probe"]["base_mixtures"], 2)
            rows = _read_rows(out_dir)
            mixture_sessions = [
                next(
                    row["mixture_session"]
                    for row in rows
                    if row["base_mixture_id"] == base_id
                )
                for base_id in sorted(
                    {row["base_mixture_id"] for row in rows}
                )
            ]
            self.assertEqual(len(mixture_sessions), 2)
            self.assertEqual(len(set(mixture_sessions)), 2)

    def test_dataset_a_path_guard_and_wav_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            tmp_path = Path(temp)
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                discover_segments(tmp_path / "Dataset-A" / "enrolment_speakers")

            bad_root = tmp_path / "bad" / "enrolment_speakers"
            bad_root.mkdir(parents=True)
            bad_name = (
                bad_root / "AISHELL-4_20200616_M_R001S01C01_001-M_1.00_2.00.wav"
            )
            sf.write(
                bad_name,
                np.zeros(8000, dtype=np.float32),
                8000,
                subtype="PCM_16",
            )
            with self.assertRaisesRegex(ValueError, "16 kHz mono"):
                discover_segments(bad_root)

    def test_base_mixture_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_root = _make_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, str(MAX_BASE_MIXTURES)):
                build_dacf_realps4_probe(
                    source_root, Path(temp) / "too_many", base_mixtures=17
                )


if __name__ == "__main__":
    unittest.main()
