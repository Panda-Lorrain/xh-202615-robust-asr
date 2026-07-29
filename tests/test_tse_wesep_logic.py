import os
import sys

import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from tse_wesep_train import (
    QwenLogMelLoss,
    collate_batch,
    normalized_waveform_l1,
    require_batch_size_one,
    validate_speaker_disjoint,
)
from tse_data_aug import sample_tse_aug_params


def test_wesep_collate_preserves_lengths():
    batch = [
        (torch.ones(3), torch.ones(3), torch.ones(4), "a"),
        (torch.ones(2), torch.ones(2), torch.ones(4), "b"),
    ]
    mix, clean, embedding, lengths, ids = collate_batch(batch)
    assert mix.shape == clean.shape == (2, 3)
    assert embedding.shape == (2, 4)
    assert lengths.tolist() == [3, 2]
    assert ids == ["a", "b"]
    print("test_wesep_collate_preserves_lengths OK")


def test_wesep_validation_rejects_speaker_overlap():
    try:
        validate_speaker_disjoint(
            [{"target_spk": "s1"}], [{"target_spk": "s1"}]
        )
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("speaker-overlapping validation must fail")
    validate_speaker_disjoint(
        [{"target_spk": "s1"}], [{"target_spk": "s2"}]
    )
    try:
        validate_speaker_disjoint(
            [{"interferer_spk": "s3"}], [{"target_spk": "s3"}]
        )
    except ValueError as exc:
        assert "speakers overlap" in str(exc)
    else:
        raise AssertionError("cross-role speaker leakage must fail")
    try:
        validate_speaker_disjoint(
            [{"target_src": "shared.wav"}],
            [{"enrollment_src": "shared.wav"}],
        )
    except ValueError as exc:
        assert "source audio overlap" in str(exc)
    else:
        raise AssertionError("cross-role source leakage must fail")
    print("test_wesep_validation_rejects_speaker_overlap OK")


def test_wesep_requires_batch_size_one():
    require_batch_size_one(1)
    try:
        require_batch_size_one(2)
    except ValueError as exc:
        assert "right padding" in str(exc)
    else:
        raise AssertionError("variable-length batch >1 must fail")
    print("test_wesep_requires_batch_size_one OK")


def test_qwen_logmel_loss_is_scale_sensitive():
    target = torch.sin(torch.linspace(0, 200, 6400)).unsqueeze(0)
    lengths = torch.tensor([target.size(1)])
    loss_fn = QwenLogMelLoss()
    identical = loss_fn(target, target, lengths)
    attenuated = loss_fn(target * 0.5, target, lengths)
    assert identical.item() < 1e-7
    assert attenuated.item() > 0.05
    print("test_qwen_logmel_loss_is_scale_sensitive OK")


def test_normalized_waveform_l1_penalizes_wrong_gain():
    target = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    lengths = torch.tensor([4])
    assert normalized_waveform_l1(target, target, lengths).item() == 0.0
    wrong_gain = normalized_waveform_l1(target * 5.0, target, lengths)
    assert wrong_gain.item() > 3.9
    print("test_normalized_waveform_l1_penalizes_wrong_gain OK")


def test_balanced_tse_profile_does_not_compound_target_gain():
    rng = random.Random(9)
    rows = [sample_tse_aug_params(rng, "balanced") for _ in range(200)]
    assert all(row["target_gain_db"] == 0.0 for row in rows)
    assert all(-5.0 <= row["sir_db"] <= 5.0 for row in rows)
    assert all(-5.0 <= row["snr_db"] <= 5.0 for row in rows)
    print("test_balanced_tse_profile_does_not_compound_target_gain OK")


if __name__ == "__main__":
    test_wesep_collate_preserves_lengths()
    test_wesep_validation_rejects_speaker_overlap()
    test_wesep_requires_batch_size_one()
    test_qwen_logmel_loss_is_scale_sensitive()
    test_normalized_waveform_l1_penalizes_wrong_gain()
    test_balanced_tse_profile_does_not_compound_target_gain()
    print("ALL PASS")
