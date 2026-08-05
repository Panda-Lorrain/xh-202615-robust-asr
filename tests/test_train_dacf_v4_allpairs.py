from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from train_dacf_v3_mechanism import (  # noqa: E402
    CounterfactualGroup,
    MechanismContractError,
    QueryRecord,
)
from train_dacf_v4_allpairs import (  # noqa: E402
    _binary_auc,
    build_allpairs_batch,
    epoch_group_batches,
)


def _group(index: int, split: str = "train") -> CounterfactualGroup:
    frames = 12
    rows = []
    for role_index, role in enumerate(("present_A", "present_B", "absent_C")):
        present = role_index < 2
        activity = np.zeros(frames, dtype=np.float32)
        if present:
            activity[2 + role_index : 7 + role_index] = 1.0
        rows.append(
            QueryRecord(
                row_id=f"g{index}_{role_index}",
                group_id=f"g{index}",
                role=role,
                speaker_id=f"s{index}_{role_index}",
                target_present=present,
                query_role_id=role_index,
                embedding=np.full(512, index * 10 + role_index, dtype=np.float32),
                embedding_view2=np.full(512, index * 10 + role_index + 0.5, dtype=np.float32),
                target_activity=activity,
                same_text_eval=None,
            )
        )
    return CounterfactualGroup(
        split=split,
        group_id=f"g{index}",
        mixture_feature_path=Path(f"g{index}.npz"),
        mixture_feature_sha256=f"sha{index}",
        mixture_features=np.zeros((128, frames), dtype=np.float32),
        mixture_speaker_ids=(f"s{index}_0", f"s{index}_1"),
        rows=tuple(rows),
    )


class AllPairsContractTests(unittest.TestCase):
    def test_every_a_b_query_gets_both_labels_and_c_is_excluded(self) -> None:
        batch = build_allpairs_batch([_group(0), _group(1), _group(2)])
        self.assertEqual(batch.presence_labels.shape, (3, 6))
        np.testing.assert_array_equal(batch.presence_labels.sum(axis=0), np.ones(6))
        np.testing.assert_array_equal(batch.presence_labels.sum(axis=1), np.full(3, 2))
        self.assertFalse(any(speaker.endswith("_2") for speaker in batch.speaker_ids))
        self.assertEqual(batch.query_roles.count("present_A"), 3)
        self.assertEqual(batch.query_roles.count("present_B"), 3)
        for mixture_index, target in enumerate(batch.activity_targets):
            self.assertEqual(target.shape, (6, 12))
            negative = batch.presence_labels[mixture_index] == 0
            self.assertTrue(np.all(target[negative] == 0))

    def test_duplicate_speaker_across_groups_is_rejected(self) -> None:
        left = _group(0)
        right = _group(1)
        repeated = list(right.rows)
        repeated[0] = QueryRecord(
            **{**repeated[0].__dict__, "speaker_id": left.rows[0].speaker_id}
        )
        right = CounterfactualGroup(
            **{
                **right.__dict__,
                "mixture_speaker_ids": (left.rows[0].speaker_id, right.mixture_speaker_ids[1]),
                "rows": tuple(repeated),
            }
        )
        with self.assertRaisesRegex(MechanismContractError, "unique A/B"):
            build_allpairs_batch([left, right])

    def test_split_mixing_is_rejected(self) -> None:
        with self.assertRaisesRegex(MechanismContractError, "cannot cross"):
            build_allpairs_batch([_group(0, "train"), _group(1, "dev")])

    def test_epoch_batches_are_deterministic_complete_and_not_short(self) -> None:
        groups = [_group(index) for index in range(8)]
        left = epoch_group_batches(groups, groups_per_batch=4, seed=9, epoch=2)
        right = epoch_group_batches(groups, groups_per_batch=4, seed=9, epoch=2)
        self.assertEqual(
            [[g.group_id for g in batch] for batch in left],
            [[g.group_id for g in batch] for batch in right],
        )
        self.assertEqual({g.group_id for batch in left for g in batch}, {f"g{i}" for i in range(8)})
        with self.assertRaisesRegex(MechanismContractError, "short batch"):
            epoch_group_batches(groups[:7], groups_per_batch=4, seed=9, epoch=2)

    def test_binary_auc_handles_ties(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        self.assertEqual(_binary_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(_binary_auc(labels, np.ones(4)), 0.5)


if __name__ == "__main__":
    unittest.main()
