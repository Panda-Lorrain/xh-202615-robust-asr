import inspect
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v3_ecst import (  # noqa: E402
    CAMPP_EMBEDDING_DIM,
    DACFV3ECST,
    MIXTURE_MEL_BINS,
    QUERY_CONDITIONED_DIM,
    fixed_top25_presence,
)


class DACFV3ECSTTests(unittest.TestCase):
    def test_shapes_and_variable_timeline(self):
        torch.manual_seed(11)
        model = DACFV3ECST()
        output = model(
            torch.randn(2, MIXTURE_MEL_BINS, 13),
            torch.randn(2, CAMPP_EMBEDDING_DIM),
        )

        self.assertEqual(tuple(output.activity_logits.shape), (2, 13))
        self.assertEqual(tuple(output.presence_logits.shape), (2,))
        self.assertEqual(tuple(output.activity_probability.shape), (2, 13))
        self.assertEqual(tuple(output.query_conditioned_frames.shape), (2, 13, 128))
        self.assertEqual(tuple(output.diagnostic_spectral_mask.shape), (2, 128, 13))
        self.assertTrue(torch.isfinite(output.query_conditioned_frames).all())

    def test_parameter_budget_and_public_count(self):
        model = DACFV3ECST()
        count = model.trainable_parameter_count()
        self.assertEqual(count, sum(p.numel() for p in model.parameters() if p.requires_grad))
        self.assertLess(count, 2_000_000)

        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            DACFV3ECST(stem_channels=47)

    def test_same_mixture_different_query_changes_outputs(self):
        torch.manual_seed(12)
        model = DACFV3ECST()
        mixture = torch.randn(1, MIXTURE_MEL_BINS, 17)
        query_a = torch.randn(1, CAMPP_EMBEDDING_DIM)
        query_b = torch.randn(1, CAMPP_EMBEDDING_DIM)
        output_a = model(mixture, query_a)
        output_b = model(mixture, query_b)

        self.assertFalse(torch.allclose(output_a.activity_logits, output_b.activity_logits))
        self.assertFalse(
            torch.allclose(
                output_a.query_conditioned_frames,
                output_b.query_conditioned_frames,
            )
        )

    def test_backward_reaches_mixture_stem_and_query_projection(self):
        torch.manual_seed(13)
        model = DACFV3ECST()
        mixture = torch.randn(2, MIXTURE_MEL_BINS, 9)
        enrollment = torch.randn(2, CAMPP_EMBEDDING_DIM)
        output = model(mixture, enrollment)
        loss = output.presence_logits.mean() + output.query_conditioned_frames.square().mean()
        loss.backward()

        self.assertIsNotNone(model.stem_conv.weight.grad)
        self.assertGreater(model.stem_conv.weight.grad.abs().sum().item(), 0.0)
        query_linear = model.query_projection[1]
        self.assertIsInstance(query_linear, nn.Linear)
        self.assertIsNotNone(query_linear.weight.grad)
        self.assertGreater(query_linear.weight.grad.abs().sum().item(), 0.0)

    def test_presence_is_fixed_top25_ceil_mean(self):
        activity = torch.tensor([[0.1, 4.0, -1.0, 3.0, 2.0, 0.2, 7.0]])
        expected = torch.tensor([(7.0 + 4.0) / 2.0])
        torch.testing.assert_close(fixed_top25_presence(activity), expected)

        model = DACFV3ECST()
        output = model(torch.randn(1, 128, 7), torch.randn(1, 512))
        expected_model = output.activity_logits.topk(2, dim=1).values.mean(dim=1)
        torch.testing.assert_close(output.presence_logits, expected_model)

    def test_forward_has_only_mel_and_enrollment_inputs(self):
        signature = inspect.signature(DACFV3ECST.forward)
        self.assertEqual(
            list(signature.parameters),
            ["self", "mixture_logmel", "enrollment_embedding"],
        )
        self.assertNotIn("role", signature.parameters)
        self.assertNotIn("environment", signature.parameters)

    def test_bad_inputs_fail_fast(self):
        model = DACFV3ECST()
        query = torch.randn(1, CAMPP_EMBEDDING_DIM)

        with self.assertRaisesRegex(ValueError, "128 mel bins"):
            model(torch.randn(1, 80, 4), query)
        with self.assertRaisesRegex(ValueError, "rank-3"):
            model(torch.randn(128, 4), query)
        with self.assertRaisesRegex(ValueError, "512 features"):
            model(torch.randn(1, 128, 4), torch.randn(1, 256))
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            model(torch.randn(2, 128, 4), query)
        bad = torch.zeros(1, 128, 4)
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            model(bad, query)
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 128, 4, dtype=torch.int64), query)

    def test_shortest_one_frame_input(self):
        model = DACFV3ECST()
        output = model(torch.randn(1, 128, 1), torch.randn(1, 512))
        self.assertEqual(tuple(output.activity_logits.shape), (1, 1))
        self.assertEqual(tuple(output.presence_logits.shape), (1,))
        self.assertEqual(tuple(output.query_conditioned_frames.shape), (1, 1, 128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
