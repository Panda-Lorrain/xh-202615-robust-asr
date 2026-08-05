from __future__ import annotations

import hashlib
import json
import shutil
import sys
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_v3_protocol import (  # noqa: E402
    DEFAULT_GROUP_COUNTS,
    FIXED_GATE,
    OFFICIAL_GROUP_CAPACITY,
    OFFICIAL_SPEAKER_COUNTS,
    PROTOCOL_TO_SOURCE_SPLIT,
    _namespace_chunk_manifest_ids,
    audit_manifests,
    build_official_protocol,
    capacity_report,
    reaudit_existing_protocol,
)


TEST_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "code"
    / "runs"
    / "dacf_v3_protocol_test_fixture"
)


@contextmanager
def _fixture_directory():
    """Use a repo-local fixture; this machine's system temp is ACL-locked."""

    if TEST_FIXTURE_ROOT.exists():
        shutil.rmtree(TEST_FIXTURE_ROOT)
    TEST_FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        yield TEST_FIXTURE_ROOT
    finally:
        shutil.rmtree(TEST_FIXTURE_ROOT)


def _source_fixture(root: Path, split: str, speaker_count: int = 6) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for speaker_index in range(speaker_count):
        speaker = f"{split}_spk_{speaker_index:02d}"
        for utterance_index in range(2):
            path = (
                root
                / "wav"
                / split
                / speaker
                / f"{speaker}_{utterance_index:02d}.wav"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                f"source:{split}:{speaker}:{utterance_index}".encode("utf-8")
            )
            items.append(
                {
                    "wav": str(path),
                    "spk": speaker,
                    "utt": f"{speaker}_{utterance_index:02d}",
                    "ref": f"read speech {speaker} {utterance_index}",
                    "split": split,
                    "source_corpus": "AISHELL-1",
                }
            )
    return items


def _write_pcm_wav(path: Path, *, amplitude: int, samples: int = 640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.full(samples, amplitude, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm.tobytes())


def _write_fake_builder_output(
    items: list[dict[str, str]],
    out_dir: Path,
    **kwargs: int,
) -> dict[str, int]:
    local_split, group_count = next(
        (name, kwargs[key])
        for name, key in (
            ("train", "n_train_mixtures"),
            ("val", "n_val_mixtures"),
            ("final", "n_final_mixtures"),
        )
        if kwargs[key]
    )
    protocol_split = {"train": "train", "val": "dev", "final": "final"}[local_split]
    by_speaker: dict[str, list[dict[str, str]]] = {}
    for item in items:
        by_speaker.setdefault(item["spk"], []).append(item)
    speakers = sorted(by_speaker)
    if len(speakers) != 3 * group_count:
        raise AssertionError("fake builder received a non-disjoint chunk")

    manifest_path = out_dir / local_split / "manifest.jsonl"
    rows: list[dict[str, object]] = []
    for group_index in range(group_count):
        speaker_a, speaker_b, speaker_c = speakers[group_index * 3 : group_index * 3 + 3]
        base_id = f"{local_split}_mix_{group_index:04d}"
        mixture_path = out_dir / local_split / "recognition" / f"{base_id}.wav"
        mixture_path.parent.mkdir(parents=True, exist_ok=True)
        mixture_path.write_bytes(f"mixture:{out_dir}:{base_id}".encode("utf-8"))
        mixture_sha = hashlib.sha256(mixture_path.read_bytes()).hexdigest()
        mixture_sources = {
            "A": by_speaker[speaker_a][0]["wav"],
            "B": by_speaker[speaker_b][0]["wav"],
        }
        mixture_speakers = {"A": speaker_a, "B": speaker_b}
        for query_id, query_speaker, present in (
            ("A", speaker_a, True),
            ("B", speaker_b, True),
            ("C", speaker_c, False),
        ):
            role = f"present_{query_id}" if present else "absent_C"
            role_id = {"A": 0, "B": 1, "C": 2}[query_id]
            activity_path = (
                out_dir / local_split / "activity" / f"{base_id}__{role}.npy"
            )
            activity_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                activity_path,
                np.asarray([1, 0, 1, 0], dtype=np.uint8)
                if present
                else np.zeros(4, dtype=np.uint8),
            )
            enrollment_source = by_speaker[query_speaker][1]["wav"]
            enrollment_path = (
                out_dir / local_split / "enrollment" / f"{base_id}__{role}.wav"
            )
            enrollment_view2_path = (
                out_dir / local_split / "enrollment_view2" / f"{base_id}__{role}.wav"
            )
            clean_target_path = (
                out_dir / local_split / "clean_target" / f"{base_id}__{role}.wav"
            )
            _write_pcm_wav(enrollment_path, amplitude=100 + role_id)
            _write_pcm_wav(enrollment_view2_path, amplitude=200 + role_id)
            _write_pcm_wav(clean_target_path, amplitude=300 + role_id if present else 0)
            if query_id == "A":
                target_src = mixture_sources["A"]
                interferer_srcs = [mixture_sources["B"]]
                interferer_spks = [speaker_b]
            elif query_id == "B":
                target_src = mixture_sources["B"]
                interferer_srcs = [mixture_sources["A"]]
                interferer_spks = [speaker_a]
            else:
                target_src = None
                interferer_srcs = [mixture_sources["A"], mixture_sources["B"]]
                interferer_spks = [speaker_a, speaker_b]
            rows.append(
                {
                    "id": f"{base_id}__{role}",
                    "base_mixture_id": base_id,
                    "query_id": query_id,
                    "query_role": role,
                    "query_role_id": role_id,
                    "counterfactual_group_key": f"{base_id}:{role_id}",
                    "environment_id": f"{local_split}:{base_id}:env",
                    "speaker_disjoint_group": f"{local_split}:{base_id}",
                    "protocol_split": protocol_split,
                    "query_speaker_id": query_speaker,
                    "enrollment_spk": query_speaker,
                    "enrollment_src": enrollment_source,
                    "target_src": target_src,
                    "interferer_srcs": interferer_srcs,
                    "interferer_spks": interferer_spks,
                    "enrollment_audio": str(enrollment_path.resolve()),
                    "enrollment_audio_view2": str(enrollment_view2_path.resolve()),
                    "enrollment_sha256": hashlib.sha256(enrollment_path.read_bytes()).hexdigest(),
                    "enrollment_view2_sha256": hashlib.sha256(enrollment_view2_path.read_bytes()).hexdigest(),
                    "enrollment_noise_raw_sha256": f"noise-main-{out_dir}-{base_id}-{role_id}",
                    "enrollment_view2_noise_raw_sha256": f"noise-view2-{out_dir}-{base_id}-{role_id}",
                    "enrollment_view_count": 2,
                    "identity_positive": True,
                    "mixture_speakers": mixture_speakers,
                    "mixture_sources": mixture_sources,
                    "recognition_audio": str(mixture_path.resolve()),
                    "mixture_sha256": mixture_sha,
                    "source_corpus": "AISHELL-1",
                    "dataset_a_used": False,
                    "hard_negative_complete_instruction_verified": False,
                    "target_present": present,
                    "target_spk": query_speaker if present else None,
                    "query_C_enrollment_only": not present,
                    "clean_target_is_empty": not present,
                    "clean_target_nonzero_samples": 1 if present else 0,
                    "clean_target_audio": str(clean_target_path.resolve()),
                    "target_activity": str(activity_path.resolve()),
                }
            )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "total_mixtures": group_count,
        "total_records": len(rows),
        "hard_negative_verified_count": 0,
    }


class DacfV3ProtocolTests(unittest.TestCase):
    def test_chunk_ids_are_rewritten_into_one_global_protocol_namespace(self) -> None:
        with _fixture_directory() as tmp_path:
            source = _source_fixture(tmp_path / "data_aishell", "train", 3)
            out = tmp_path / "chunk_001"
            _write_fake_builder_output(
                source,
                out,
                n_train_mixtures=1,
                n_val_mixtures=0,
                n_final_mixtures=0,
            )
            manifest = out / "train" / "manifest.jsonl"
            report = _namespace_chunk_manifest_ids(
                manifest,
                protocol_split="train",
                chunk_index=1,
                global_group_offset=80,
                expected_groups=1,
            )
            rows = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual({row["base_mixture_id"] for row in rows}, {"train_mix_0080"})
            self.assertEqual(
                {row["id"] for row in rows},
                {
                    "train_mix_0080__present_A",
                    "train_mix_0080__present_B",
                    "train_mix_0080__absent_C",
                },
            )
            self.assertEqual(
                {row["protocol_original_base_mixture_id"] for row in rows},
                {"train_mix_0000"},
            )
            self.assertEqual(
                {row["environment_id"] for row in rows},
                {"train:train_mix_0080:env"},
            )
            self.assertEqual(report["global_group_start"], 80)
            self.assertEqual(report["global_group_stop_exclusive"], 81)

    def test_official_capacities_are_340_40_20_not_sixteen(self) -> None:
        self.assertEqual(OFFICIAL_SPEAKER_COUNTS, {"train": 340, "dev": 40, "test": 20})
        self.assertEqual(OFFICIAL_GROUP_CAPACITY, {"train": 113, "dev": 13, "test": 6})

        items = [
            {
                "wav": f"/fake/{speaker}/{index}.wav",
                "spk": f"spk_{speaker:02d}",
                "utt": f"utt_{speaker:02d}_{index}",
                "split": "dev",
                "source_corpus": "AISHELL-1",
            }
            for speaker in range(40)
            for index in range(2)
        ]
        report = capacity_report("dev", items, 13)
        self.assertEqual(report["available_capacity_groups"], 13)
        with self.assertRaisesRegex(ValueError, "capacity is only 13"):
            capacity_report("dev", items, 14)

        test_items = [
            {
                "wav": f"/fake/test/{speaker}/{index}.wav",
                "spk": f"test_spk_{speaker:02d}",
                "utt": f"utt_{speaker:02d}_{index}",
                "split": "test",
                "source_corpus": "AISHELL-1",
            }
            for speaker in range(20)
            for index in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "capacity is only 6"):
            capacity_report("test", test_items, 7)

    def test_official_routing_prereg_first_and_hash_bound_report(self) -> None:
        with _fixture_directory() as tmp_path:
            aishell_root = tmp_path / "data_aishell"
            source_by_split = {
                split: _source_fixture(aishell_root, split)
                for split in ("train", "dev", "test")
            }
            output = tmp_path / "protocol"
            loader_calls: list[tuple[Path, tuple[str, ...]]] = []
            builder_calls: list[Path] = []

            def fake_loader(root: Path, source_splits: tuple[str, ...]):
                loader_calls.append((root, tuple(source_splits)))
                return source_by_split[source_splits[0]]

            def fake_builder(items, out_dir, **kwargs):
                prereg_path = output / "PREREGISTRATION.json"
                self.assertTrue(prereg_path.is_file(), "generation began before preregistration")
                builder_calls.append(Path(out_dir))
                return _write_fake_builder_output(list(items), Path(out_dir), **kwargs)

            with patch("build_dacf_v3_protocol.load_aishell_items", side_effect=fake_loader), patch(
                "build_dacf_v3_protocol.build_dacf_counterfactual",
                side_effect=fake_builder,
            ):
                result = build_official_protocol(
                    aishell_root,
                    output,
                    group_counts={"train": 1, "dev": 1, "final": 1},
                )

            self.assertEqual(
                [call[1] for call in loader_calls],
                [("train",), ("dev",), ("test",)],
            )
            self.assertEqual(len(builder_calls), 3)
            self.assertEqual(result["group_counts"], {"train": 1, "dev": 1, "final": 1})
            prereg_path = output / "PREREGISTRATION.json"
            prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {key: value["source_split"] for key, value in prereg["split_protocol"].items()},
                PROTOCOL_TO_SOURCE_SPLIT,
            )
            self.assertEqual(prereg["fixed_gate"], FIXED_GATE)
            self.assertEqual(prereg["selection_policy"]["final_selects"], "none")
            report_path = output / "build_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            expected_prereg_sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
            self.assertEqual(report["preregistration_sha256"], expected_prereg_sha)
            self.assertEqual(report["hard_negative_verified_count"], 0)
            self.assertTrue(report["audit"]["all_cross_split_overlaps_zero"])
            self.assertIn("hard_negative_verified_count=0", " ".join(report["limitations"]))
            for protocol_split in ("train", "dev", "final"):
                self.assertEqual(
                    report["split_reports"][protocol_split]["source_split"],
                    PROTOCOL_TO_SOURCE_SPLIT[protocol_split],
                )
                self.assertTrue(report["split_reports"][protocol_split]["manifest_paths"])

            reaudit_path = output / "build_report_reaudit_v02.json"
            reaudit = reaudit_existing_protocol(output, reaudit_path)
            self.assertTrue(reaudit["audit"]["all_cross_split_overlaps_zero"])
            self.assertEqual(
                reaudit["source_build_report_sha256"],
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(reaudit["hard_negative_verified_count"], 0)

    def test_audit_rejects_hash_mismatch_absent_c_and_cross_split_overlap(self) -> None:
        with _fixture_directory() as tmp_path:
            root = tmp_path / "data_aishell"
            source_by_split = {
                split: _source_fixture(root, split)
                for split in ("train", "dev", "test")
            }
            valid_paths: dict[str, list[Path]] = {}
            for protocol_split, source_split in PROTOCOL_TO_SOURCE_SPLIT.items():
                out = tmp_path / "valid" / protocol_split
                source_items = source_by_split[source_split]
                selected_speakers = sorted({item["spk"] for item in source_items})[:3]
                chunk_items = [
                    item for item in source_items if item["spk"] in selected_speakers
                ]
                _write_fake_builder_output(chunk_items, out, n_train_mixtures=1 if protocol_split == "train" else 0, n_val_mixtures=1 if protocol_split == "dev" else 0, n_final_mixtures=1 if protocol_split == "final" else 0)
                local_split = {"train": "train", "dev": "val", "final": "final"}[protocol_split]
                valid_paths[protocol_split] = [out / local_split / "manifest.jsonl"]

            audit_manifests(valid_paths, aishell_root=root, expected_groups={"train": 1, "dev": 1, "final": 1})

            def rewrite(protocol_split: str, mutate):
                original = valid_paths[protocol_split][0]
                rows = [json.loads(line) for line in original.read_text(encoding="utf-8").splitlines()]
                mutate(rows)
                replacement = tmp_path / f"{protocol_split}_mutated.jsonl"
                replacement.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                return replacement

            bad_hash = rewrite("train", lambda rows: rows[1].update({"mixture_sha256": "0" * 64}))
            bad_paths = dict(valid_paths)
            bad_paths["train"] = [bad_hash]
            with self.assertRaisesRegex(ValueError, "mixture SHA mismatch"):
                audit_manifests(bad_paths, aishell_root=root, expected_groups={"train": 1, "dev": 1, "final": 1})

            bad_absent = rewrite(
                "train",
                lambda rows: rows[2].update({"query_speaker_id": rows[0]["mixture_speakers"]["A"], "enrollment_spk": rows[0]["mixture_speakers"]["A"]}),
            )
            bad_paths = dict(valid_paths)
            bad_paths["train"] = [bad_absent]
            with self.assertRaisesRegex(ValueError, "enrollment source speaker/path mismatch"):
                audit_manifests(bad_paths, aishell_root=root, expected_groups={"train": 1, "dev": 1, "final": 1})

            train_rows = [json.loads(line) for line in valid_paths["train"][0].read_text(encoding="utf-8").splitlines()]
            overlap_source = train_rows[0]["enrollment_src"]
            dev_rows = [
                json.loads(line)
                for line in valid_paths["dev"][0].read_text(encoding="utf-8").splitlines()
            ]
            dev_source = Path(dev_rows[0]["enrollment_src"])
            original_dev_bytes = dev_source.read_bytes()
            try:
                # Keep the official dev path/speaker metadata valid while
                # manufacturing a cross-split raw-SHA collision.
                dev_source.write_bytes(Path(overlap_source).read_bytes())
                with self.assertRaisesRegex(ValueError, "overlap audit failed"):
                    audit_manifests(
                        valid_paths,
                        aishell_root=root,
                        expected_groups={"train": 1, "dev": 1, "final": 1},
                    )
            finally:
                dev_source.write_bytes(original_dev_bytes)

            bad_speaker_path = rewrite(
                "train",
                lambda rows: rows[0].update(
                    {"enrollment_src": rows[2]["enrollment_src"]}
                ),
            )
            bad_paths = dict(valid_paths)
            bad_paths["train"] = [bad_speaker_path]
            with self.assertRaisesRegex(ValueError, "enrollment source speaker/path mismatch"):
                audit_manifests(
                    bad_paths,
                    aishell_root=root,
                    expected_groups={"train": 1, "dev": 1, "final": 1},
                )

            bad_official_split = rewrite(
                "train",
                lambda rows: rows[0].update(
                    {"target_src": source_by_split["dev"][0]["wav"]}
                ),
            )
            bad_paths = dict(valid_paths)
            bad_paths["train"] = [bad_official_split]
            with self.assertRaisesRegex(ValueError, "outside official AISHELL wav/train"):
                audit_manifests(
                    bad_paths,
                    aishell_root=root,
                    expected_groups={"train": 1, "dev": 1, "final": 1},
                )

    def test_dataset_a_is_hard_rejected(self) -> None:
        with _fixture_directory() as tmp_path:
            output = tmp_path / "protocol"
            with self.assertRaisesRegex(ValueError, "Dataset-A"):
                build_official_protocol(tmp_path / "Dataset-A", output)

    def test_default_protocol_counts_are_fixed(self) -> None:
        self.assertEqual(DEFAULT_GROUP_COUNTS, {"train": 96, "dev": 12, "final": 6})
        self.assertEqual(FIXED_GATE["same_text_different_speaker_auc_deferred"], 0.75)
        self.assertIn(
            "before Qwen integration",
            FIXED_GATE["same_text_different_speaker_required_stage"],
        )


if __name__ == "__main__":
    unittest.main()
