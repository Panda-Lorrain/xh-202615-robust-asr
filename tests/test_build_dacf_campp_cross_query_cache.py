"""Lightweight contract tests for the DACF CAM++ cache converter v0.2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
FIXTURE_ROOT = ROOT / "code" / "runs" / "dacf_campp_cross_query_cache_20260806" / "_test_fixture"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_campp_cross_query_cache import (  # noqa: E402
    CacheContractError,
    SCHEMA,
    build_cache,
    validate_cache,
)
from probe_dacf_campp_cross_query import _validate_and_load_cache  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_fixture(root: Path) -> tuple[Path, Path, Path]:
    frame_root = root / "frame_features"
    allowed_root = root / "allowed_source_root"
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    final_rows: list[dict] = []

    for split, destination in (
        ("train", train_rows),
        ("val", val_rows),
        ("final", final_rows),
    ):
        group_id = f"{split}_mix_0000"
        mixture_audio = frame_root / "audio" / split / f"{group_id}.wav"
        _write_bytes(mixture_audio, f"mixture:{split}".encode("ascii"))
        mixture_sha = _sha256(mixture_audio)
        source_a = allowed_root / split / f"{group_id}_source_A.wav"
        source_b = allowed_root / split / f"{group_id}_source_B.wav"
        _write_bytes(source_a, f"source-A:{split}".encode("ascii"))
        _write_bytes(source_b, f"source-B:{split}".encode("ascii"))

        mixture_feature = frame_root / "mixture" / f"{group_id}.npz"
        tokens = np.arange(4 * 512, dtype=np.float32).reshape(4, 512) + len(destination)
        mixture_feature.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mixture_feature,
            prepool=tokens,
            mixture_sha256=np.asarray(mixture_sha),
        )

        for role_id, role in enumerate(("present_A", "present_B", "absent_C")):
            row_id = f"{group_id}__{role}"
            speaker = f"{split}_speaker_{role_id}"
            enrollment_src = allowed_root / split / f"{group_id}_enroll_{role_id}.wav"
            target_src = allowed_root / split / f"{group_id}_target_{role_id}.wav"
            _write_bytes(enrollment_src, f"enrollment-source:{split}:{role_id}".encode("ascii"))
            _write_bytes(target_src, f"target-source:{split}:{role_id}".encode("ascii"))
            enrollment_audio = frame_root / "audio" / split / f"{row_id}.wav"
            enrollment_view2 = frame_root / "audio" / split / f"{row_id}__view2.wav"
            _write_bytes(enrollment_audio, f"enrollment:{split}:{role_id}".encode("ascii"))
            _write_bytes(enrollment_view2, f"view2:{split}:{role_id}".encode("ascii"))
            enrollment_sha = _sha256(enrollment_audio)
            enrollment_view2_sha = _sha256(enrollment_view2)

            query_dir = frame_root / "query"
            query_dir.mkdir(parents=True, exist_ok=True)
            embedding = np.linspace(-1.0, 1.0, 512, dtype=np.float32) + role_id
            embedding_view2 = embedding[::-1].copy()
            enrollment_embedding = query_dir / f"{row_id}__enrollment.npy"
            view2_embedding = query_dir / f"{row_id}__view2.npy"
            enrollment_prepool = query_dir / f"{row_id}__enrollment_prepool.npy"
            view2_prepool = query_dir / f"{row_id}__view2_prepool.npy"
            activity = query_dir / f"{row_id}__activity.npy"
            np.save(enrollment_embedding, embedding)
            np.save(view2_embedding, embedding_view2)
            np.save(enrollment_prepool, np.ones((2, 512), dtype=np.float32) * role_id)
            np.save(view2_prepool, np.ones((2, 512), dtype=np.float32) * (role_id + 1))
            np.save(activity, np.asarray([float(role_id != 2), 0.0, 0.5], dtype=np.float32))

            destination.append(
                {
                    "id": row_id,
                    "split": split,
                    "base_mixture_id": group_id,
                    "query_role": role,
                    "query_role_id": role_id,
                    "query_speaker_id": speaker,
                    "target_present": role_id != 2,
                    "target_activity": str(activity),
                    "source_corpus": "AISHELL-1",
                    "dataset_a_used": False,
                    "dataset_a_policy": "forbidden",
                    "recognition_audio": str(mixture_audio),
                    "enrollment_audio": str(enrollment_audio),
                    "enrollment_audio_view2": str(enrollment_view2),
                    "mixture_sha256": mixture_sha,
                    "enrollment_sha256": enrollment_sha,
                    "enrollment_view2_sha256": enrollment_view2_sha,
                    "enrollment_src": str(enrollment_src),
                    "mixture_sources": {"A": str(source_a), "B": str(source_b)},
                    "target_src": str(target_src),
                    "interferer_srcs": [str(source_b if role == "present_A" else source_a)],
                    "campp_frame_features": {
                        "schema": "dacf-campp-frame-features-v0.1",
                        "mixture_feature_npz": str(mixture_feature.relative_to(frame_root)),
                        "enrollment_prepool_npy": str(enrollment_prepool.relative_to(frame_root)),
                        "enrollment_final_embedding_npy": str(enrollment_embedding.relative_to(frame_root)),
                        "enrollment_view2_prepool_npy": str(view2_prepool.relative_to(frame_root)),
                        "enrollment_view2_final_embedding_npy": str(view2_embedding.relative_to(frame_root)),
                        "mixture_audio_sha256_actual": mixture_sha,
                        "enrollment_audio_sha256_actual": enrollment_sha,
                        "enrollment_view2_audio_sha256_actual": enrollment_view2_sha,
                    },
                    "original_marker": {"record_kind": "dacf_query"},
                }
            )

    manifest = frame_root / "features_manifest.jsonl"
    _write_jsonl(manifest, train_rows + val_rows + final_rows)
    return frame_root, allowed_root, manifest


class BuildDacfCamppCrossQueryCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        # Keep the physical fixture path neutral: the production marker guard
        # must inspect real corpus markers, not unittest method names.
        self.case_root = FIXTURE_ROOT / "case"
        if self.case_root.exists():
            shutil.rmtree(self.case_root)
        self.frame_root, self.allowed_root, self.input_manifest = _make_fixture(self.case_root)
        self.expected_groups = {"train": 1, "val": 1, "final": 1}

    def tearDown(self) -> None:
        if self.case_root.exists():
            shutil.rmtree(self.case_root)

    def _build(self) -> Path:
        output = self.case_root / "cache"
        build_cache(
            self.frame_root,
            output,
            allowed_source_root=self.allowed_root,
            expected_groups=self.expected_groups,
        )
        return output

    def _rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.input_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _rewrite_rows(self, rows: list[dict]) -> None:
        _write_jsonl(self.input_manifest, rows)

    def test_normal_fixture_writes_final_and_keeps_audio_out_of_cache(self) -> None:
        output = self._build()
        report = validate_cache(output)
        self.assertEqual(report["cache_schema"], SCHEMA)
        self.assertFalse(report["dataset_a_used"])
        self.assertEqual(report["counts"]["splits"]["train"]["groups"], 1)
        self.assertEqual(report["counts"]["splits"]["val"]["groups"], 1)
        self.assertEqual(report["counts"]["splits"]["final"]["groups"], 1)
        self.assertEqual(report["split_contract"]["final_gate_split"], "final")
        for pair_audit in report["overlap_audit"].values():
            for values in pair_audit.values():
                self.assertEqual(values, [])
        self.assertFalse(list(output.rglob("*.wav")))
        loaded = _validate_and_load_cache(output, strict_counts=False)
        self.assertTrue(loaded.audit["provenance_locked"])
        self.assertEqual(len(loaded.train_groups), 1)
        self.assertEqual(len(loaded.val_groups), 1)
        self.assertEqual(len(loaded.final_groups), 1)
        self.assertEqual(
            loaded.final_groups[0].queries["present_A"].target_activity.ndim, 1
        )

        rows = [
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 9)
        final_row = next(row for row in rows if row["split"] == "final" and row["query_role"] == "present_A")
        self.assertIn("original_dacf_provenance", final_row)
        self.assertEqual(final_row["source_corpus"], "AISHELL-1")
        self.assertEqual(Path(final_row["resolved_source_paths"]["enrollment_src"]).anchor, self.allowed_root.anchor)
        self.assertTrue(Path(output / final_row["mixture_feature"]).is_file())
        self.assertTrue(Path(output / final_row["query_feature"]).is_file())
        with np.load(output / final_row["mixture_feature"], allow_pickle=False) as mixture:
            self.assertEqual(mixture["tokens"].shape, (4, 512))
            self.assertEqual(mixture["tokens"].shape, mixture["prepool"].shape)
            for key in ("mixture_sha256", "base_mixture_id", "source_lineage_sha256"):
                self.assertIn(key, mixture)
        with np.load(output / final_row["query_feature"], allow_pickle=False) as query:
            for key in (
                "embedding",
                "embedding_view2",
                "target_activity",
                "row_id",
                "base_mixture_id",
                "query_role",
                "query_speaker_id",
                "enrollment_sha256",
                "enrollment_view2_sha256",
                "source_lineage_sha256",
            ):
                self.assertIn(key, query)

    def test_default_scale_contract_requires_48_16_16_groups(self) -> None:
        with self.assertRaisesRegex(CacheContractError, "train group contract requires 48"):
            build_cache(self.frame_root, self.case_root / "cache", allowed_source_root=self.allowed_root)

    def test_tampered_input_metadata_and_output_metadata_are_rejected(self) -> None:
        rows = self._rows()
        rows[0]["enrollment_sha256"] = "0" * 64
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "enrollment_sha256"):
            self._build()

        self.setUp()
        output = self._build()
        row = next(
            json.loads(line)
            for line in (output / "features_manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        query_path = output / row["query_feature"]
        with np.load(query_path, allow_pickle=False) as query:
            values = {key: query[key] for key in query.files}
        values["row_id"] = np.asarray("tampered-row")
        np.savez_compressed(query_path, **values)
        with self.assertRaisesRegex(CacheContractError, "query_npz_sha256 mismatch"):
            validate_cache(output)

    def test_train_val_final_mixture_sha_and_path_overlap_is_rejected(self) -> None:
        rows = self._rows()
        train_row = next(row for row in rows if row["split"] == "train")
        for row in rows:
            if row["split"] == "final":
                row["recognition_audio"] = train_row["recognition_audio"]
                row["mixture_sha256"] = train_row["mixture_sha256"]
                row["campp_frame_features"]["mixture_feature_npz"] = train_row[
                    "campp_frame_features"
                ]["mixture_feature_npz"]
                row["campp_frame_features"]["mixture_audio_sha256_actual"] = train_row["mixture_sha256"]
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "train_vs_final"):
            self._build()

    def test_train_val_final_source_overlap_is_rejected(self) -> None:
        rows = self._rows()
        train_row = next(row for row in rows if row["split"] == "train")
        final_row = next(row for row in rows if row["split"] == "final")
        final_row["enrollment_src"] = train_row["enrollment_src"]
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "train_vs_final"):
            self._build()

    def test_dataset_a_marker_out_of_root_and_parent_escape_are_rejected(self) -> None:
        rows = self._rows()
        rows[0]["source_corpus"] = "Dataset-A"
        self._rewrite_rows(rows)
        with self.assertRaises(CacheContractError):
            self._build()

        self.setUp()
        rows = self._rows()
        outside = self.case_root / "outside_source.wav"
        _write_bytes(outside, b"outside")
        rows[0]["enrollment_src"] = str(outside)
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "outside allowed roots"):
            self._build()

        self.setUp()
        rows = self._rows()
        source = Path(rows[0]["enrollment_src"])
        rows[0]["enrollment_src"] = str(source.parent / "subdir" / ".." / source.name)
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "forbidden '..'"):
            self._build()

    def test_symlink_escape_is_rejected_when_platform_allows_symlinks(self) -> None:
        real_source = Path(self._rows()[0]["enrollment_src"])
        link = self.case_root / "allowed_source_root" / "link.wav"
        try:
            os.symlink(real_source, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation is unavailable on this Windows runner")
        rows = self._rows()
        rows[0]["enrollment_src"] = str(link)
        self._rewrite_rows(rows)
        with self.assertRaisesRegex(CacheContractError, "symlink"):
            self._build()


if __name__ == "__main__":
    unittest.main(verbosity=2)
