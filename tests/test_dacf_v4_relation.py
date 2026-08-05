from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v4_relation import (  # noqa: E402
    DACFV4Relation,
    fixed_top25_presence,
)


class DACFV4RelationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = DACFV4Relation(
            stem_channels=8,
            stem_out_channels=8,
            hidden_dim=16,
            relation_dim=8,
            relation_hidden_dim=4,
        )

    def test_shapes_for_shared_query_bank(self) -> None:
        mixture = torch.randn(2, 128, 31)
        queries = torch.randn(5, 512)
        output = self.model(mixture, queries)
        self.assertEqual(tuple(output.activity_logits.shape), (2, 5, 31))
        self.assertEqual(tuple(output.presence_logits.shape), (2, 5))
        self.assertEqual(tuple(output.query_conditioned_frames.shape), (2, 5, 31, 128))
        self.assertEqual(tuple(output.mixture_keys.shape), (2, 31, 8))
        self.assertEqual(tuple(output.query_embeddings.shape), (2, 5, 8))

    def test_mixture_state_is_query_independent(self) -> None:
        mixture = torch.randn(1, 128, 19)
        state = self.model.encode_mixture(mixture)
        left = self.model.score_queries(state, torch.randn(3, 512))
        right = self.model.score_queries(state, torch.randn(4, 512))
        self.assertTrue(torch.equal(left.mixture_keys, right.mixture_keys))
        self.assertEqual(tuple(left.activity_logits.shape), (1, 3, 19))
        self.assertEqual(tuple(right.activity_logits.shape), (1, 4, 19))

    def test_query_order_is_exactly_equivariant(self) -> None:
        mixture = torch.randn(1, 128, 19)
        queries = torch.randn(5, 512)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        direct = self.model(mixture, queries)
        permuted = self.model(mixture, queries[permutation])
        self.assertTrue(
            torch.allclose(
                permuted.activity_logits,
                direct.activity_logits[:, permutation],
                atol=1e-6,
            )
        )

    def test_content_has_only_multiplicative_query_gate(self) -> None:
        mixture = torch.randn(1, 128, 13)
        queries = torch.randn(2, 512)
        state = self.model.encode_mixture(mixture)
        output = self.model.score_queries(state, queries)
        expected = state.content_frames.unsqueeze(1) * output.activity_probability.unsqueeze(-1)
        self.assertTrue(torch.allclose(output.query_conditioned_frames, expected))

    def test_keys_and_queries_are_l2_normalised(self) -> None:
        output = self.model(torch.randn(1, 128, 17), torch.randn(3, 512))
        key_norm = output.mixture_keys.norm(dim=-1)
        query_norm = output.query_embeddings.norm(dim=-1)
        self.assertTrue(torch.allclose(key_norm, torch.ones_like(key_norm), atol=1e-5))
        self.assertTrue(torch.allclose(query_norm, torch.ones_like(query_norm), atol=1e-5))

    def test_pair_loss_reaches_both_encoders(self) -> None:
        output = self.model(torch.randn(1, 128, 11), torch.randn(3, 512))
        output.presence_logits.square().mean().backward()
        key_grad = self.model.key_projection.weight.grad
        query_grad = self.model.query_projection[-1].weight.grad
        self.assertIsNotNone(key_grad)
        self.assertIsNotNone(query_grad)
        self.assertGreater(float(key_grad.abs().sum()), 0.0)
        self.assertGreater(float(query_grad.abs().sum()), 0.0)

    def test_relation_mlp_has_no_bias_parameters(self) -> None:
        names = [name for name, _ in self.model.relation_mlp.named_parameters()]
        self.assertTrue(names)
        self.assertTrue(all("bias" not in name for name in names))

    def test_parameter_budget(self) -> None:
        model = DACFV4Relation()
        self.assertLess(model.trainable_parameter_count(), 2_000_000)

    def test_fixed_top25_short_and_exact(self) -> None:
        one = torch.tensor([[[2.0]]])
        self.assertEqual(float(fixed_top25_presence(one)), 2.0)
        four = torch.tensor([[[1.0, 4.0, 2.0, 3.0]]])
        self.assertEqual(float(fixed_top25_presence(four)), 4.0)

    def test_rejects_wrong_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "128 mel"):
            self.model(torch.randn(1, 80, 10), torch.randn(2, 512))
        with self.assertRaisesRegex(ValueError, "512"):
            self.model(torch.randn(1, 128, 10), torch.randn(2, 192))
        with self.assertRaisesRegex(ValueError, "batch"):
            self.model(torch.randn(2, 128, 10), torch.randn(1, 3, 512))


if __name__ == "__main__":
    unittest.main()
