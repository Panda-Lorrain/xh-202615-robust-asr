from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_v4_anchor_relation import DACFV4AnchorRelation  # noqa: E402


class DACFV4AnchorRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        self.model = DACFV4AnchorRelation(
            stem_channels=8,
            stem_out_channels=8,
            hidden_dim=16,
        )

    def test_raw_campp_query_is_preserved_exactly_after_normalisation(self) -> None:
        raw = torch.randn(5, 512)
        encoded = self.model.encode_queries(raw, batch_size=2)
        expected = F.normalize(raw, dim=-1).unsqueeze(0).expand(2, -1, -1)
        self.assertTrue(torch.equal(encoded, expected))

    def test_model_has_no_query_projection_or_relation_mlp(self) -> None:
        names = [name for name, _ in self.model.named_parameters()]
        self.assertFalse(any("query_projection" in name for name in names))
        self.assertFalse(any("relation_mlp" in name for name in names))

    def test_shapes_and_query_order_equivariance(self) -> None:
        mixture = torch.randn(1, 128, 21)
        queries = torch.randn(4, 512)
        permutation = torch.tensor([2, 0, 3, 1])
        direct = self.model(mixture, queries)
        permuted = self.model(mixture, queries[permutation])
        self.assertEqual(tuple(direct.activity_logits.shape), (1, 4, 21))
        self.assertEqual(tuple(direct.mixture_keys.shape), (1, 21, 512))
        self.assertTrue(
            torch.allclose(
                permuted.activity_logits,
                direct.activity_logits[:, permutation],
                atol=1e-6,
            )
        )

    def test_pair_loss_reaches_mixture_key_but_not_query_parameters(self) -> None:
        output = self.model(torch.randn(1, 128, 15), torch.randn(3, 512))
        output.presence_logits.square().mean().backward()
        self.assertGreater(float(self.model.key_projection.weight.grad.abs().sum()), 0.0)
        self.assertFalse(any("query" in name for name, _ in self.model.named_parameters()))

    def test_parameter_budget(self) -> None:
        self.assertLess(DACFV4AnchorRelation().trainable_parameter_count(), 2_000_000)


if __name__ == "__main__":
    unittest.main()
