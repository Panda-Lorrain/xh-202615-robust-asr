import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tse_qwen_sidecar import FrozenQwenSidecar, SidecarConfig


class DummyLayer(nn.Module):
    def forward(self, hidden, *_args, **_kwargs):
        return (hidden + 1,)


class DummyTower(nn.Module):
    def __init__(self):
        super().__init__()
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
