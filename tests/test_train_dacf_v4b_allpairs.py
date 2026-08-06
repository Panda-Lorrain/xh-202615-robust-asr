from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from train_dacf_v4b_allpairs import (  # noqa: E402
    CounterfactualGroup,
    MechanismContractError,
    QueryRecord,
    _binary_auc,
    build_v4b_allpairs_batch,
    epoch_group_batches,
)


def _group(
    index: int,
    split: str = "train",
    speakers: tuple[str, str, str] = ("sA", "sB", "sC"),
    frames: int = 12,
) -> CounterfactualGroup:
    """Synthetic group: speakers = (present_A, present_B, absent_C).

    present_A/B each carry a balanced (active+inactive) timeline so the
    objective term is well defined; absent_C carries zeros.
    """

    rows = []
    for role_index, role in enumerate(("present_A", "present_B", "absent_C")):
        present = role_index < 2
        activity = np.zeros(frames, dtype=np.float32)
        if present:
            activity[2 + role_index : 7 + role_index] = 1.0
        rows.append(
            QueryRecord(
                row_id=f"g{index}_{role}",
                group_id=f"g{index}",
                role=role,
                speaker_id=speakers[role_index],
                target_present=present,
                embedding=np.full(512, index * 10 + role_index, dtype=np.float32),
                embedding_view2=np.full(
                    512, index * 10 + role_index + 0.5, dtype=np.float32
                ),
                target_activity=activity,
            )
        )
    return CounterfactualGroup(
        split=split,
        group_id=f"g{index}",
        mixture_feature_path=Path(f"g{index}.npz"),
        mixture_feature_sha256=f"sha{index}",
        mixture_features=np.zeros((128, frames), dtype=np.float32),
        mixture_speaker_ids=(speakers[0], speakers[1]),
        rows=tuple(rows),  # type: ignore[arg-type]
    )


class V4BAllPairsContractTests(unittest.TestCase):
    def test_bank_dedups_repeated_speaker_and_allows_multi_positive(self) -> None:
        # s0 is present in BOTH g0 and g1.  The bank keeps one enrolment for s0
        # but its column is positive for two mixtures -- the v4b multi-positive
        # invariant that the v4 builder rejected ("unique A/B speakers").
        g0 = _group(0, speakers=("s0", "s1", "s9"))
        g1 = _group(1, speakers=("s0", "s2", "s8"))
        g2 = _group(2, speakers=("s3", "s4", "s7"))
        batch = build_v4b_allpairs_batch([g0, g1, g2])

        self.assertEqual(batch.speaker_ids, ("s0", "s1", "s2", "s3", "s4"))
        self.assertEqual(batch.presence_labels.shape, (3, 5))
        # absent C speakers never enter the bank
        self.assertFalse(any(s in {"s9", "s8", "s7"} for s in batch.speaker_ids))
        # every mixture has exactly two positive queries
        np.testing.assert_array_equal(
            batch.presence_labels.sum(axis=1), np.full(3, 2)
        )
        # s0 is positive in two mixtures (multi-positive); every other once
        column_sums = batch.presence_labels.sum(axis=0)
        self.assertEqual(column_sums[0], 2)
        self.assertTrue(np.all(column_sums[1:] == 1))

    def test_each_mixture_has_positive_and_foreign_queries(self) -> None:
        batch = build_v4b_allpairs_batch(
            [
                _group(0, speakers=("s0", "s1", "s9")),
                _group(1, speakers=("s2", "s3", "s8")),
                _group(2, speakers=("s4", "s5", "s7")),
            ]
        )
        bank_size = len(batch.speaker_ids)
        for mixture_index in range(3):
            positives = int(batch.presence_labels[mixture_index].sum())
            self.assertGreaterEqual(positives, 1)
            self.assertGreaterEqual(bank_size - positives, 1)

    def test_activity_targets_are_per_mixture_and_foreign_is_zero(self) -> None:
        g0 = _group(0, speakers=("s0", "s1", "s9"))
        g1 = _group(1, speakers=("s0", "s2", "s8"))
        batch = build_v4b_allpairs_batch([g0, g1])
        s0_index = batch.speaker_ids.index("s0")
        frames = g0.mixture_features.shape[1]

        for mixture_index, group in enumerate(batch.groups):
            target = batch.activity_targets[mixture_index]
            self.assertEqual(target.shape, (len(batch.speaker_ids), frames))
            negative = batch.presence_labels[mixture_index] == 0
            # foreign queries carry exactly zero activity
            self.assertTrue(np.all(target[negative] == 0.0))
            # a present query's activity matches that speaker's timeline in THIS
            # mixture (looked up per-mixture, not from the deduplicated bank row)
            if "s0" in set(group.mixture_speaker_ids):
                np.testing.assert_array_equal(
                    target[s0_index],
                    next(
                        row.target_activity
                        for row in group.rows
                        if row.speaker_id == "s0"
                    ),
                )

    def test_rejects_split_mixing(self) -> None:
        with self.assertRaisesRegex(MechanismContractError, "cannot cross"):
            build_v4b_allpairs_batch([_group(0, "train"), _group(1, "dev")])

    def test_rejects_single_group(self) -> None:
        with self.assertRaisesRegex(MechanismContractError, "at least two"):
            build_v4b_allpairs_batch([_group(0)])

    def test_epoch_batches_are_deterministic_complete_and_not_short(self) -> None:
        groups = [_group(index) for index in range(8)]
        left = epoch_group_batches(groups, groups_per_batch=4, seed=9, epoch=2)
        right = epoch_group_batches(groups, groups_per_batch=4, seed=9, epoch=2)
        self.assertEqual(
            [[g.group_id for g in batch] for batch in left],
            [[g.group_id for g in batch] for batch in right],
        )
        self.assertEqual(
            {g.group_id for batch in left for g in batch},
            {f"g{i}" for i in range(8)},
        )
        with self.assertRaisesRegex(MechanismContractError, "short batch"):
            epoch_group_batches(groups[:7], groups_per_batch=4, seed=9, epoch=2)

    def test_binary_auc_handles_ties(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        self.assertEqual(_binary_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(_binary_auc(labels, np.ones(4)), 0.5)


if __name__ == "__main__":
    unittest.main()
