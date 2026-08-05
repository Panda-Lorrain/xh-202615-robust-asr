import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v3_ecst import DACFV3ECSTOutput  # noqa: E402
from dacf_v3_objective import (  # noqa: E402
    balanced_binary_bce,
    compute_dacf_v3_loss,
    resize_activity_targets,
)


def _output(activity_logits: torch.Tensor, latent_scale: float = 1.0):
    probability = torch.sigmoid(activity_logits)
    presence = activity_logits.topk(
        max(1, (activity_logits.shape[1] + 3) // 4), dim=1
    ).values.mean(dim=1)
    latent = probability.unsqueeze(-1).expand(-1, -1, 128) * latent_scale
    return DACFV3ECSTOutput(
        activity_logits=activity_logits,
        presence_logits=presence,
        activity_probability=probability,
        query_conditioned_frames=latent,
    )


class DACFV3ObjectiveTests(unittest.TestCase):
    def test_resize_activity_nearest_and_guards(self):
        target = torch.tensor([[0.0, 1.0]])
        resized = resize_activity_targets(target, 4)
        torch.testing.assert_close(resized, torch.tensor([[0.0, 0.0, 1.0, 1.0]]))
        with self.assertRaises(ValueError):
            resize_activity_targets(torch.tensor([[2.0]]), 1)

    def test_balanced_bce_handles_both_and_single_class_rows(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        loss = balanced_binary_bce(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_counterfactual_loss_is_finite_and_backpropagates(self):
        main_logits = torch.randn(3, 8, requires_grad=True)
        view_logits = torch.randn(3, 8, requires_grad=True)
        labels = torch.tensor([1.0, 1.0, 0.0])
        activity = torch.tensor(
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.float32
        )
        result = compute_dacf_v3_loss(
            _output(main_logits),
            _output(view_logits),
            presence_labels=labels,
            activity_targets=activity,
            group_index=torch.tensor([0, 0, 0]),
        )
        self.assertEqual(
            set(result.components),
            {
                "presence",
                "activity",
                "counterfactual_margin",
                "view_consistency",
                "absent_latent_energy",
            },
        )
        self.assertTrue(torch.isfinite(result.total))
        result.total.backward()
        self.assertGreater(main_logits.grad.abs().sum().item(), 0.0)
        self.assertGreater(view_logits.grad.abs().sum().item(), 0.0)

    def test_margin_rewards_two_present_above_absent_without_role_ids(self):
        good = torch.tensor(
            [[6.0, 6.0, 6.0, 6.0], [5.0, 5.0, 5.0, 5.0], [-6.0, -6.0, -6.0, -6.0]]
        )
        bad = -good
        labels = torch.tensor([1.0, 1.0, 0.0])
        activity = labels[:, None].expand(-1, 4)
        good_loss = compute_dacf_v3_loss(
            _output(good),
            _output(good.clone()),
            presence_labels=labels,
            activity_targets=activity,
            group_index=torch.zeros(3, dtype=torch.long),
        ).components["counterfactual_margin"]
        bad_loss = compute_dacf_v3_loss(
            _output(bad),
            _output(bad.clone()),
            presence_labels=labels,
            activity_targets=activity,
            group_index=torch.zeros(3, dtype=torch.long),
        ).components["counterfactual_margin"]
        self.assertEqual(good_loss.item(), 0.0)
        self.assertGreater(bad_loss.item(), 0.2)

    def test_group_contract_and_shape_fail_fast(self):
        logits = torch.zeros(3, 4)
        output = _output(logits)
        with self.assertRaisesRegex(ValueError, "two present and one absent"):
            compute_dacf_v3_loss(
                output,
                output,
                presence_labels=torch.tensor([1.0, 0.0, 0.0]),
                activity_targets=torch.zeros(3, 4),
                group_index=torch.zeros(3, dtype=torch.long),
            )
        with self.assertRaises(TypeError):
            compute_dacf_v3_loss(
                output,
                output,
                presence_labels=torch.tensor([1.0, 1.0, 0.0]),
                activity_targets=torch.zeros(3, 4),
                group_index=torch.zeros(3),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
