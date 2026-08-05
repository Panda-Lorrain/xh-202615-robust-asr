import inspect
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_campp_cross_query import (  # noqa: E402
    DACFCAMPPQueryMatcher,
    class_balanced_bce,
    counterfactual_margin_loss,
    resize_activity_targets,
)


class DACFCAMPPQueryMatcherTest(unittest.TestCase):
    def test_shapes_identity_initialization_and_backward(self):
        torch.manual_seed(7)
        model = DACFCAMPPQueryMatcher(feature_dim=8, query_dim=4)
        mixture = torch.randn(3, 20, 8)
        enrollment = torch.randn(3, 8)
        output = model(mixture, enrollment)

        self.assertEqual(tuple(output.frame_logits.shape), (3, 20))
        self.assertEqual(tuple(output.presence_logits.shape), (3,))
        self.assertEqual(tuple(output.query_aware_tokens.shape), (3, 20, 8))
        self.assertTrue(torch.equal(output.query_aware_tokens, mixture.float()))

        labels = torch.tensor([1.0, 1.0, 0.0])
        loss = class_balanced_bce(output.presence_logits, labels)
        loss.backward()
        self.assertGreater(model.mixture_projection.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.enrollment_projection.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNone(model.query_residual.weight.grad)

    def test_enrollment_is_the_only_query_input(self):
        signature = inspect.signature(DACFCAMPPQueryMatcher.forward)
        self.assertEqual(
            list(signature.parameters),
            ["self", "mixture_tokens", "enrollment_embedding"],
        )

        model = DACFCAMPPQueryMatcher(feature_dim=4, query_dim=2, logit_scale=1.0)
        with torch.no_grad():
            model.mixture_projection.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
            model.enrollment_projection.weight.copy_(
                torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
            )
        mixture = torch.tensor(
            [[[3.0, 1.0, 0.0, -1.0], [1.0, 3.0, 0.0, -1.0]]]
        )
        query_a = torch.tensor([[3.0, 1.0, 0.0, -1.0]])
        query_b = torch.tensor([[1.0, 3.0, 0.0, -1.0]])
        logits_a = model(mixture, query_a).frame_logits
        logits_b = model(mixture, query_b).frame_logits
        self.assertFalse(torch.allclose(logits_a, logits_b))
        self.assertGreater(logits_a[0, 0].item(), logits_a[0, 1].item())
        self.assertGreater(logits_b[0, 1].item(), logits_b[0, 0].item())

    def test_top_fraction_has_fixed_ceil_semantics(self):
        model = DACFCAMPPQueryMatcher(
            feature_dim=4, query_dim=2, logit_scale=1.0, top_fraction=0.25
        )
        with torch.no_grad():
            model.mixture_projection.weight.zero_()
            model.enrollment_projection.weight.zero_()
            model.frame_bias.fill_(2.5)
        output = model(torch.randn(1, 5, 4), torch.randn(1, 4))
        self.assertAlmostEqual(output.presence_logits.item(), 2.5, places=6)

    def test_resize_activity_is_bounded_and_differentiable_target(self):
        source = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        resized = resize_activity_targets(source, 3)
        self.assertEqual(tuple(resized.shape), (1, 3))
        self.assertTrue(torch.all((0.0 <= resized) & (resized <= 1.0)))
        self.assertGreater(resized[0, -1].item(), resized[0, 0].item())

    def test_balanced_bce_is_finite_for_single_class(self):
        logits = torch.tensor([0.2, -0.3], requires_grad=True)
        loss = class_balanced_bce(logits, torch.zeros(2))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_counterfactual_margin_uses_predictions_in_abc_order(self):
        passing = torch.tensor([[1.0, 0.8, 0.0]])
        failing = torch.tensor([[0.1, 0.8, 0.0]])
        self.assertEqual(counterfactual_margin_loss(passing).item(), 0.0)
        self.assertGreater(counterfactual_margin_loss(failing).item(), 0.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            counterfactual_margin_loss(torch.zeros(3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
