"""Short shape/gradient/absent-query self-tests for the DACF POC.

These tests use synthetic tensors only.  They do not touch Dataset-A, load a
checkpoint, run a long training loop, or alter the submission chain.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


EXPERIMENTS = Path(__file__).resolve().parent / "code" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from dacf_frontend import (  # noqa: E402
    DACFFrontend,
    compute_dacf_loss,
    identity_contrastive_loss,
    make_absent_targets,
)


class DACFFrontendTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260806)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_model(self) -> DACFFrontend:
        return DACFFrontend(
            n_fft=64,
            hop_length=32,
            win_length=64,
            d_model=32,
            n_heads=4,
            vocab_size=13,
        ).to(self.device)

    def test_forward_shapes_and_separate_cross_attention(self) -> None:
        model = self.make_model()
        mixture = torch.randn(2, 512, device=self.device)
        enrollment = torch.randn(2, 256, device=self.device)

        outputs = model(mixture, enrollment)
        batch, freq, frames = outputs["mixture_stft"].shape
        self.assertEqual((batch, freq), (2, 33))
        self.assertEqual(outputs["target_stft"].shape, (batch, freq, frames))
        self.assertEqual(outputs["target_mask"].shape, (batch, freq, frames))
        self.assertEqual(outputs["target_audio"].shape, mixture.shape)
        self.assertTrue(torch.equal(outputs["mixture_audio"], mixture))
        self.assertEqual(outputs["target_present_logits"].shape, (batch,))
        self.assertEqual(outputs["target_activity_logits"].shape, (batch, frames))
        self.assertEqual(outputs["ctc_logits"].shape, (batch, frames, 13))
        self.assertEqual(outputs["speaker_anchor"].shape, (batch, 32))
        self.assertEqual(outputs["environment_anchor"].shape, (batch, 32))
        self.assertEqual(outputs["speaker_frame_weights"].shape, (batch, enrollment.shape[-1] // 32 + 1))
        self.assertTrue(
            torch.allclose(
                outputs["speaker_frame_weights"] + outputs["environment_frame_weights"],
                torch.ones_like(outputs["speaker_frame_weights"]),
            )
        )
        self.assertEqual(outputs["speaker_query_frames"].shape, (batch, frames, 32))
        self.assertEqual(outputs["query_aware_frames"].shape, (batch, frames, 32))
        self.assertTrue(torch.is_complex(outputs["target_stft"]))
        self.assertIsInstance(model.speaker_cross, nn.MultiheadAttention)
        self.assertIsInstance(model.environment_cross, nn.MultiheadAttention)
        self.assertIsNot(model.speaker_cross, model.environment_cross)

    def test_one_batch_backward_covers_heads_and_conditioning(self) -> None:
        model = self.make_model()
        base_mixture = torch.randn(2, 512, device=self.device)
        mixture = base_mixture.repeat_interleave(2, dim=0)
        enrollment = torch.randn(4, 256, device=self.device)
        outputs = model(mixture, enrollment)
        frames = outputs["target_activity_logits"].shape[1]

        target_scale = torch.tensor([0.3, 0.1, 0.2, 0.0], device=self.device).unsqueeze(1)

        targets = {
            "target_present": torch.tensor([1, 0, 1, 0], device=self.device),
            "target_activity": torch.rand(4, frames, device=self.device),
            "target_audio": target_scale * mixture,
            "transcript": torch.tensor(
                [[1, 2, 0], [4, 0, 0], [0, 0, 0], [5, 6, 7]],
                dtype=torch.long,
                device=self.device,
            ),
            "transcript_lengths": torch.tensor([2, 1, 0, 3], device=self.device),
            # Query role is group-local; speaker label is global across groups.
            "query_role_id": torch.tensor([0, 1, 0, 2], device=self.device),
            "query_speaker_label": torch.tensor([10, 11, 10, 12], device=self.device),
            "mixture_id": torch.tensor([0, 0, 1, 1], device=self.device),
            "environment_id": torch.tensor([0, 0, 1, 1], device=self.device),
        }
        losses = compute_dacf_loss(outputs, targets)
        self.assertEqual(
            set(losses),
            {
                "presence",
                "activity",
                "reconstruction",
                "counterfactual",
                "ctc",
                "identity",
                "environment",
                "disentangle",
                "total",
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]).item())
        losses["total"].backward()

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        for parameter in trainable:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        cross_grads = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if ("speaker_cross" in name or "environment_cross" in name)
        ]
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in cross_grads))
        self.assertGreater(float(model.mask_head.weight.grad.abs().sum().detach().cpu()), 0.0)

    def test_presence_head_has_no_direct_environment_branch_gradient(self) -> None:
        model = self.make_model()
        mixture = torch.randn(2, 512, device=self.device)
        enrollment = torch.randn(2, 256, device=self.device)
        outputs = model(mixture, enrollment)

        environment_parameters = list(model.environment_branch.parameters()) + list(
            model.environment_cross.parameters()
        )
        gradients = torch.autograd.grad(
            outputs["target_present_logits"].sum(),
            environment_parameters,
            allow_unused=True,
        )
        self.assertTrue(all(gradient is None for gradient in gradients))

    def test_single_class_contrastive_term_is_inactive(self) -> None:
        anchors = torch.randn(4, 8, device=self.device, requires_grad=True)
        labels = torch.zeros(4, dtype=torch.long, device=self.device)
        loss = identity_contrastive_loss(anchors, labels)
        self.assertEqual(float(loss.detach().cpu()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(anchors.grad, torch.zeros_like(anchors.grad)))

    def test_mask_starts_at_identity_but_can_represent_silence(self) -> None:
        model = self.make_model()
        mixture = torch.randn(1, 512, device=self.device)
        enrollment = torch.randn(1, 256, device=self.device)

        initial = model(mixture, enrollment)
        self.assertTrue(
            torch.allclose(initial["target_mask"], torch.ones_like(initial["target_mask"]), atol=1e-6)
        )
        self.assertTrue(torch.allclose(initial["target_audio"], mixture, atol=1e-5, rtol=1e-5))

        with torch.no_grad():
            model.mask_head.weight.zero_()
            model.mask_head.bias.fill_(-40.0)
        suppressed = model(mixture, enrollment)
        self.assertLess(float(suppressed["target_mask"].max().detach().cpu()), 1e-12)
        self.assertLess(float(suppressed["target_audio"].abs().max().detach().cpu()), 1e-10)

    def test_absent_targets_are_zero_activity_silent_and_blank_ctc(self) -> None:
        model = self.make_model()
        mixture = torch.randn(2, 512, device=self.device)
        absent_enrollment = torch.randn(2, 256, device=self.device)
        outputs = model(mixture, absent_enrollment)
        absent = make_absent_targets(
            batch_size=2,
            num_frames=outputs["target_activity_logits"].shape[1],
            num_samples=mixture.shape[-1],
            device=self.device,
        )

        self.assertTrue(torch.equal(absent["target_present"], torch.zeros_like(absent["target_present"])))
        self.assertTrue(torch.equal(absent["target_activity"], torch.zeros_like(absent["target_activity"])))
        self.assertTrue(torch.equal(absent["target_audio"], torch.zeros_like(absent["target_audio"])))
        self.assertEqual(tuple(absent["transcript"].shape), (2, 0))
        self.assertTrue(torch.equal(absent["transcript_lengths"], torch.zeros(2, dtype=torch.long, device=self.device)))

        losses = compute_dacf_loss(outputs, absent)
        self.assertIn("ctc", losses)
        self.assertTrue(torch.isfinite(losses["ctc"]).item())
        self.assertTrue(torch.isfinite(losses["total"]).item())
        losses["total"].backward()

    def test_same_mixture_changes_when_enrollment_query_is_swapped(self) -> None:
        model = self.make_model()
        sample_index = torch.linspace(-1.0, 1.0, 256, device=self.device)
        enrollment_a = torch.sin(7.0 * sample_index)
        enrollment_b = torch.cos(11.0 * sample_index)
        enrollment = torch.stack((enrollment_a, enrollment_b), dim=0)
        mixture = torch.randn(1, 512, device=self.device).expand(2, -1).clone()

        outputs = model(mixture, enrollment)
        difference = (
            outputs["query_aware_frames"][0] - outputs["query_aware_frames"][1]
        ).abs().mean()
        self.assertGreater(float(difference.detach().cpu()), 1e-6)
        head_difference = (
            outputs["target_present_logits"][0]
            - outputs["target_present_logits"][1]
        ).abs()
        self.assertGreater(float(head_difference.detach().cpu()), 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
