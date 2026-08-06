from __future__ import annotations

import sys
import hashlib
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_dacf_v4b_schedule import (  # noqa: E402
    MAX_PAIR_USE,
    SOURCES_PER_SPEAKER,
    allocate_source_wavs,
    audit_role_rotation,
    build_role_rotation,
)


def _fake_items(root: Path, speakers: list[str]):
    rows = {}
    for speaker in speakers:
        speaker_dir = root / speaker
        values = []
        for index in range(SOURCES_PER_SPEAKER + 2):
            path = speaker_dir / f"{speaker}_{index:03d}.wav"
            values.append(
                {
                    "wav": str(path),
                    "spk": speaker,
                    "utt": path.stem,
                    "ref": f"text {speaker} {index}",
                    "source_corpus": "AISHELL-1",
                }
            )
        rows[speaker] = values
    return rows


class RoleRotationTests(unittest.TestCase):
    def _assert_schedule(self, speaker_count: int) -> None:
        speakers = [f"S{index:03d}" for index in range(speaker_count)]
        groups = build_role_rotation(speakers, seed=12345)
        audit = audit_role_rotation(groups, speakers=speakers)
        self.assertEqual(audit["groups"], speaker_count * 2)
        self.assertEqual(audit["role_count_per_speaker"], {"A": 2, "B": 2, "C": 2})
        self.assertEqual(audit["present_count_per_speaker"], 4)
        self.assertEqual(audit["absent_count_per_speaker"], 2)
        self.assertEqual(audit["unique_ab_pairs"], speaker_count * 2)
        self.assertLessEqual(audit["pair_use_max"], MAX_PAIR_USE)
        for round_index in range(6):
            seen = [
                speaker
                for group in groups
                if group["round_index"] == round_index
                for speaker in group["speakers"].values()
            ]
            self.assertEqual(Counter(seen), Counter(speakers))

    def test_train_48_speaker_schedule(self) -> None:
        self._assert_schedule(48)

    def test_dev_12_speaker_schedule(self) -> None:
        self._assert_schedule(12)

    def test_schedule_is_deterministic(self) -> None:
        speakers = [f"S{index:03d}" for index in range(12)]
        self.assertEqual(
            build_role_rotation(speakers, seed=77),
            build_role_rotation(speakers, seed=77),
        )

    def test_global_source_ledger_uses_path_and_sha_once(self) -> None:
        speakers = [f"S{index:03d}" for index in range(12)]
        groups = build_role_rotation(speakers, seed=31)
        items = _fake_items(ROOT / "_logical_v4b_fixture", speakers)
        logical_resolver = lambda value: Path(value).resolve(strict=False)
        logical_sha = lambda path: hashlib.sha256(path.as_posix().encode()).hexdigest()
        attached, ledger = allocate_source_wavs(
            groups,
            speaker_items=items,
            seed=31,
            path_resolver=logical_resolver,
            sha_resolver=logical_sha,
        )
        self.assertEqual(len(attached), 24)
        self.assertEqual(ledger["unique_source_paths"], 12 * SOURCES_PER_SPEAKER)
        self.assertEqual(ledger["unique_source_sha256"], 12 * SOURCES_PER_SPEAKER)
        self.assertEqual(ledger["source_path_use_count_min"], 1)
        self.assertEqual(ledger["source_path_use_count_max"], 1)
        self.assertEqual(ledger["source_sha_use_count_min"], 1)
        self.assertEqual(ledger["source_sha_use_count_max"], 1)
        for group in attached:
            for role in ("A", "B", "C"):
                sources = group["roles"][role]["sources"]
                self.assertNotEqual(
                    sources["enrollment_view1"]["path"],
                    sources["enrollment_view2"]["path"],
                )
                if role in {"A", "B"}:
                    self.assertNotIn(
                        sources["recognition"]["path"],
                        {
                            sources["enrollment_view1"]["path"],
                            sources["enrollment_view2"]["path"],
                        },
                    )

    def test_reused_source_bytes_are_rejected(self) -> None:
        speakers = [f"S{index:03d}" for index in range(12)]
        groups = build_role_rotation(speakers, seed=9)
        items = _fake_items(ROOT / "_logical_v4b_fixture_dup", speakers)
        with self.assertRaisesRegex(ValueError, "byte SHA was reused"):
            allocate_source_wavs(
                groups,
                speaker_items=items,
                seed=9,
                sha_resolver=lambda _: "0" * 64,
                path_resolver=lambda value: Path(value).resolve(strict=False),
            )

    def test_invalid_cardinality_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of three"):
            build_role_rotation(["a", "b", "c", "d"], seed=1)


if __name__ == "__main__":
    unittest.main()
