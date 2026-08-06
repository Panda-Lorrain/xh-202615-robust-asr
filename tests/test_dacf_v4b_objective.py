from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v4b_objective import compute_dacf_v4b_loss  # noqa: E402
from dacf_v4_relation import DACFV4Relation  # noqa: E402


class DACFV4BObjectiveTests(unittest.TestCase):
    """The v4b objective is the v4 objective with one robustness fix:
    ``_balanced_positive_activity`` must not raise when a present pair is
    active (or inactive) on every frame.  The v4b cache contains read speech
    where a present speaker can be voiced for all frames of a mixture.
    """

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = DACFV4Relation(
            stem_channels=8,
            stem_out_channels=8,
            hidden_dim=16,
            relation_dim=8,
            relation_hidden_dim=4,
        )

    def _outputs(self):
        state = self.model.encode_mixture(torch.randn(1, 128, 16))
        return (
            self.model.score_queries(state, torch.randn(6, 512)),
            self.model.score_queries(state, torch.randn(6, 512)),
        )

    def test_finite_loss_and_gradients(self) -> None:
        view1, view2 = self._outputs()
        labels = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        activity = torch.zeros(1, 6, 16)
        activity[:, :2, 3:10] = 1.0
        result = compute_dacf_v4b_loss(
            view1, view2, presence_labels=labels, activity_targets=activity
        )
        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(
            set(result.components),
            {
                "presence",
                "positive_activity",
                "foreign_activity",
                "hard_foreign_margin",
                "view_consistency",
                "absent_latent_energy",
            },
        )
        result.total.backward()
        self.assertGreater(float(self.model.key_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(self.model.query_projection[-1].weight.grad.abs().sum()), 0.0
        )

    def test_rejects_foreign_nonzero_activity(self) -> None:
        view1, view2 = self._outputs()
        labels = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        activity = torch.zeros(1, 6, 16)
        activity[:, :2, 2:8] = 1.0
        activity[0, 3, 4] = 1.0  # a foreign query carrying activity
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            compute_dacf_v4b_loss(
                view1, view2, presence_labels=labels, activity_targets=activity
            )

    def test_rejects_batches_without_both_labels(self) -> None:
        view1, view2 = self._outputs()
        activity = torch.zeros(1, 6, 16)
        with self.assertRaisesRegex(ValueError, "both positive"):
            compute_dacf_v4b_loss(
                view1,
                view2,
                presence_labels=torch.ones(1, 6),
                activity_targets=activity,
            )

    def test_all_active_present_pair_does_not_raise(self) -> None:
        # The v4b robustness fix: a present speaker voiced on every frame must
        # not abort training.  v4 raises "active and inactive" on this input.
        view1, view2 = self._outputs()
        labels = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        activity = torch.zeros(1, 6, 16)
        activity[:, :2] = 1.0  # both present pairs active on every frame
        result = compute_dacf_v4b_loss(
            view1, view2, presence_labels=labels, activity_targets=activity
        )
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertGreater(float(self.model.key_projection.weight.grad.abs().sum()), 0.0)

    def test_all_inactive_present_pair_does_not_raise(self) -> None:
        view1, view2 = self._outputs()
        labels = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        activity = torch.zeros(1, 6, 16)  # present pairs fully inactive
        result = compute_dacf_v4b_loss(
            view1, view2, presence_labels=labels, activity_targets=activity
        )
        self.assertTrue(torch.isfinite(result.total))


if __name__ == "__main__":
    unittest.main()
