import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tse_qwen_sidecar import (
    FrozenQwenSidecar,
    SidecarConfig,
    _TargetActivityHead,
)


class DummyLayer(nn.Module):
    def forward(self, hidden, *_args, **_kwargs):
        return (hidden + 1,)


class DummyTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"d_model": 8})()
        self.layers = nn.ModuleList([DummyLayer(), DummyLayer()])

    def forward(self, features, **_kwargs):
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return type("Output", (), {"last_hidden_state": hidden})()


class DummyThinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_tower = DummyTower()

    def forward(self, features, labels=None):
        result = self.audio_tower(features)
        loss = result.last_hidden_state.mean() if labels is not None else None
        return type("Output", (), {"loss": loss})()


def test_sidecar_is_identity_at_initialization_and_only_sidecar_is_trainable():
    thinker = DummyThinker()
    wrapped = FrozenQwenSidecar(thinker, SidecarConfig(layers=2, rank=4, enrollment_dim=3))
    features = torch.randn(7, 8)
    baseline = features + 2
    output = wrapped(
        features,
        labels=torch.tensor([1]),
        enrollment_embeddings=torch.randn(1, 3),
        target_activity=torch.ones(1, 5),
    )
    assert torch.allclose(thinker.audio_tower(features).last_hidden_state, baseline)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert wrapped.trainable_parameter_count > 0
    assert all(
        not parameter.requires_grad
        for layer in wrapped.thinker.audio_tower.layers
        for parameter in layer.base.parameters()
    )
    assert all(parameter.requires_grad for parameter in wrapped.adapters.parameters())


def test_sidecar_tracks_frozen_backbone_dtype_and_can_reload_checkpoint():
    thinker = DummyThinker().to(torch.bfloat16)
    wrapped = FrozenQwenSidecar(
        thinker, SidecarConfig(layers=1, rank=4, enrollment_dim=3)
    )
    assert next(wrapped.adapters.parameters()).dtype == torch.bfloat16
    checkpoint = wrapped.trainable_state_dict()
    wrapped.load_trainable_state_dict(checkpoint)
    output = wrapped(
        torch.randn(5, 8, dtype=torch.bfloat16),
        labels=torch.tensor([1]),
        enrollment_embeddings=torch.randn(1, 3),
        target_activity=torch.ones(1, 4),
    )
    assert torch.isfinite(output.loss)


def test_target_activity_head_depends_on_enrollment():
    torch.manual_seed(7)
    head = _TargetActivityHead(hidden_size=8, enrollment_dim=3, rank=4)
    hidden = torch.randn(5, 8)
    first = head(hidden, torch.zeros(1, 3))
    second = head(hidden, torch.ones(1, 3))
    assert not torch.allclose(first, second)


def test_inference_context_is_cleared_after_generation_scope():
    wrapped = FrozenQwenSidecar(
        DummyThinker(),
        SidecarConfig(layers=1, rank=4, enrollment_dim=3),
    )
    with wrapped.inference_context(torch.randn(1, 3)):
        assert wrapped._enrollment is not None
    assert wrapped._enrollment is None
    assert wrapped._activity_logits == []
