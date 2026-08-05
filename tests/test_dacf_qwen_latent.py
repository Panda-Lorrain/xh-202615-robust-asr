"""Fake-only unittest contract tests for the D7 DACF/Qwen latent bridge.

No Dataset-A, processor, Qwen checkpoint, or long training is used here.  The
tests verify only the interface needed before a real Qwen batch=1 smoke.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor, nn
from torch.nn import functional as F


EXPERIMENTS = Path(__file__).resolve().parents[1] / "code" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from dacf_qwen_latent import DACFBridgeConfig, FrozenQwenDACFLatent  # noqa: E402


class FakeDACF(nn.Module):
    def __init__(self, dim: int = 4, *, detach_query: bool = False) -> None:
        super().__init__()
        self.d_model = dim
        self.query = nn.Linear(1, dim)
        self.presence = nn.Linear(dim, 1)
        self.activity = nn.Linear(dim, 1)
        self.detach_query = detach_query

    def forward(self, mixture_waveform: Tensor, enrollment_waveform: Tensor):
        del enrollment_waveform
        query = self.query(mixture_waveform.unsqueeze(-1))
        if self.detach_query:
            query = query.detach()
        present = self.presence(query.mean(dim=1)).squeeze(-1)
        activity = self.activity(query).squeeze(-1)
        return {
            # Speaker-only query/head outputs are the only fields consumed by
            # the wrapper for presence/activity supervision.
            "speaker_query_frames": query,
            "target_present_logits": present,
            "target_activity_logits": activity,
        }


class FakeAudioLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        del args, kwargs
        return self.linear(hidden_states)


class FakeAudioTower(nn.Module):
    def __init__(
        self,
        hidden: int = 6,
        layers: int = 2,
        *,
        layout: str = "utterance",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=hidden)
        self.layers = nn.ModuleList([FakeAudioLayer(hidden) for _ in range(layers)])
        self.layout = layout

    def forward(self, input_features: Tensor, **kwargs):
        del kwargs
        hidden = input_features
        if self.layout == "bad_ambiguous":
            hidden = hidden.unsqueeze(0).expand(2, -1, -1)
        elif self.layout == "batch_first":
            hidden = hidden.unsqueeze(0)
        elif self.layout == "time_first":
            hidden = hidden.unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class FakeThinker(nn.Module):
    def __init__(
        self,
        *,
        layers: int = 2,
        tower_layout: str = "utterance",
        return_loss: bool = True,
    ) -> None:
        super().__init__()
        hidden = 6
        self.audio_tower = FakeAudioTower(hidden, layers, layout=tower_layout)
        self.lm_head = nn.Linear(hidden, 3)
        self.return_loss = return_loss

    def forward(self, input_features: Tensor, labels: Tensor | None = None, **kwargs):
        del kwargs
        hidden = self.audio_tower(input_features)
        if hidden.ndim == 3:
            if hidden.shape[0] == 1:
                hidden = hidden[0]
            elif hidden.shape[1] == 1:
                hidden = hidden[:, 0]
            else:
                hidden = hidden[0]
        logits = self.lm_head(hidden)
        loss = None
        if self.return_loss and labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss, logits=logits)


def _inputs(time: int = 7) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    features = torch.randn(time, 6)
    mixture = torch.randn(1, time + 3)
    enrollment = torch.randn(1, time)
    labels = torch.randint(0, 3, (time,))
    return features, mixture, enrollment, labels


def _wrapper(
    *,
    tower_layout: str = "utterance",
    return_loss: bool = True,
    dacf=None,
    presence_loss_weight: float = 0.1,
    activity_loss_weight: float = 0.1,
):
    torch.manual_seed(7)
    thinker = FakeThinker(tower_layout=tower_layout, return_loss=return_loss)
    dacf = FakeDACF() if dacf is None else dacf
    wrapper = FrozenQwenDACFLatent(
        thinker,
        dacf,
        DACFBridgeConfig(
            layers=2,
            rank=3,
            dacf_dim=4,
            presence_loss_weight=presence_loss_weight,
            activity_loss_weight=activity_loss_weight,
        ),
    )
    return thinker, dacf, wrapper


class FrozenQwenDACFLatentTests(unittest.TestCase):
    def test_zero_residual_is_exact_frozen_thinker_equivalence(self):
        # Touch only the standard-library temp location; no data file or
        # cleanup-sensitive temporary directory is needed for this in-memory
        # fake test.
        self.assertTrue(Path(tempfile.gettempdir()).exists())
        features, mixture, enrollment, labels = _inputs()
        torch.manual_seed(7)
        thinker = FakeThinker()
        dacf = FakeDACF()
        baseline = thinker(input_features=features, labels=labels)
        wrapper = FrozenQwenDACFLatent(
            thinker,
            dacf,
            DACFBridgeConfig(layers=2, rank=3, dacf_dim=4),
        )

        output = wrapper(
            mixture,
            enrollment,
            input_features=features,
            labels=labels,
        )
        self.assertTrue(all(bridge.gate.item() == 0.0 for bridge in wrapper.bridges))
        torch.testing.assert_close(output.logits, baseline.logits, rtol=0.0, atol=0.0)
        torch.testing.assert_close(output.loss, baseline.loss, rtol=0.0, atol=0.0)
        torch.testing.assert_close(output.asr_loss, baseline.loss, rtol=0.0, atol=0.0)

    def test_asr_loss_reaches_dacf_and_bridge_but_not_frozen_thinker(self):
        features, mixture, enrollment, labels = _inputs()
        thinker, dacf, wrapper = _wrapper(
            presence_loss_weight=0.0,
            activity_loss_weight=0.0,
        )
        with torch.no_grad():
            for bridge in wrapper.bridges:
                bridge.gate.fill_(0.5)

        output = wrapper(
            mixture,
            enrollment,
            input_features=features,
            labels=labels,
        )
        self.assertIsNotNone(output.loss)
        self.assertTrue(output.loss.requires_grad)
        # Backpropagate the pure thinker CE, not total loss.  This prevents an
        # auxiliary presence/activity loss from masquerading as ASR gradient.
        output.asr_loss.backward()
        wrapper.assert_gradient_contract(output.asr_loss)
        self.assertTrue(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in dacf.parameters()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in thinker.parameters()))

    def test_zero_gate_first_asr_step_only_opens_gate(self):
        features, mixture, enrollment, labels = _inputs()
        thinker, dacf, wrapper = _wrapper(
            presence_loss_weight=0.0,
            activity_loss_weight=0.0,
        )
        output = wrapper(
            mixture,
            enrollment,
            input_features=features,
            labels=labels,
        )
        output.asr_loss.backward()

        self.assertTrue(
            any(
                bridge.gate.grad is not None
                and torch.isfinite(bridge.gate.grad).all()
                and bridge.gate.grad.abs().sum() > 0
                for bridge in wrapper.bridges
            )
        )
        # residual = gate * projection, so an exact-zero gate deliberately
        # blocks ASR gradients to DACF/projection on the very first step.
        self.assertFalse(
            any(
                parameter.grad is not None and parameter.grad.abs().sum() > 0
                for parameter in dacf.parameters()
            )
        )
        self.assertTrue(all(parameter.grad is None for parameter in thinker.parameters()))

    def test_restore_returns_exact_original_thinker_layers_and_forward(self):
        thinker = FakeThinker()
        original_forward = thinker.audio_tower.forward
        original_layers = tuple(thinker.audio_tower.layers)
        wrapper = FrozenQwenDACFLatent(
            thinker,
            FakeDACF(),
            DACFBridgeConfig(layers=2, rank=3, dacf_dim=4),
        )
        self.assertFalse(
            all(
                thinker.audio_tower.layers[index] is layer
                for index, layer in enumerate(original_layers)
            )
        )

        wrapper.restore()
        wrapper.restore()  # idempotent
        self.assertTrue(
            all(
                thinker.audio_tower.layers[index] is layer
                for index, layer in enumerate(original_layers)
            )
        )
        self.assertEqual(thinker.audio_tower.forward, original_forward)
        features, mixture, enrollment, labels = _inputs()
        with self.assertRaisesRegex(RuntimeError, "restored"):
            wrapper(mixture, enrollment, input_features=features, labels=labels)

    def test_presence_activity_auxiliary_losses_and_probs_are_exposed(self):
        features, mixture, enrollment, labels = _inputs()
        _, dacf, wrapper = _wrapper()
        output = wrapper(
            mixture,
            enrollment,
            target_present=torch.ones(1),
            target_activity=torch.tensor([[0.0, 1.0, 0.0]]),
            input_features=features,
            labels=labels,
        )
        self.assertEqual(output.presence_loss.ndim, 0)
        self.assertEqual(output.activity_loss.ndim, 0)
        self.assertEqual(output.aux_loss.ndim, 0)
        self.assertEqual(tuple(output.presence_probs.shape), (1,))
        self.assertEqual(output.activity_probs.shape[0], 1)
        self.assertIsNotNone(output.loss)
        self.assertTrue(
            torch.allclose(
                output.loss,
                output.asr_loss + output.aux_loss.to(output.asr_loss.device),
            )
        )
        output.loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in dacf.parameters()
            )
        )

    def test_missing_asr_loss_with_training_target_fails_fast(self):
        features, mixture, enrollment, _ = _inputs()
        _, _, wrapper = _wrapper(return_loss=False)
        with self.assertRaisesRegex(RuntimeError, "output.loss is None"):
            wrapper(
                mixture,
                enrollment,
                target_present=torch.ones(1),
                input_features=features,
            )

    def test_shape_and_layout_mismatch_fail_fast(self):
        features, mixture, enrollment, labels = _inputs()
        _, _, wrapper = _wrapper(tower_layout="bad_ambiguous")
        with self.assertRaisesRegex(ValueError, "layout"):
            wrapper(mixture, enrollment, input_features=features, labels=labels)

        features, _, enrollment, labels = _inputs()
        _, _, wrapper = _wrapper()
        with self.assertRaisesRegex(ValueError, "batch"):
            wrapper(
                torch.randn(2, mixture.shape[-1]),
                enrollment,
                input_features=features,
                labels=labels,
            )

        _, mixture, enrollment, labels = _inputs()
        _, _, wrapper = _wrapper()
        with self.assertRaisesRegex(ValueError, "target_present"):
            wrapper(
                mixture,
                enrollment,
                target_present=torch.ones(1, 2),
                input_features=features,
                labels=labels,
            )

        features, _, _, labels = _inputs()
        _, _, wrapper = _wrapper()
        with self.assertRaisesRegex(ValueError, "call count"):
            wrapper(
                torch.randn(2, features.shape[0] + 3),
                torch.randn(2, features.shape[0]),
                input_features=features,
                labels=labels,
            )

    def test_single_utterance_batch_first_and_time_first_layouts(self):
        features, mixture, enrollment, labels = _inputs()
        for layout in ("batch_first", "time_first"):
            _, _, wrapper = _wrapper(tower_layout=layout)
            output = wrapper(
                mixture,
                enrollment,
                input_features=features,
                labels=labels,
            )
            self.assertEqual(tuple(output.logits.shape), (features.shape[0], 3))
            self.assertIsNotNone(output.loss)

    def test_detached_dacf_and_no_grad_are_detected(self):
        features, mixture, enrollment, labels = _inputs()
        detached = FakeDACF(detach_query=True)
        _, _, wrapper = _wrapper(dacf=detached)
        with self.assertRaisesRegex(RuntimeError, "detached"):
            wrapper(mixture, enrollment, input_features=features, labels=labels)

        _, _, wrapper = _wrapper()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "no_grad"):
            wrapper(mixture, enrollment, input_features=features, labels=labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
