import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from tse_wesep_train import (
    collate_batch,
    require_batch_size_one,
    validate_speaker_disjoint,
)


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


if __name__ == "__main__":
    test_wesep_collate_preserves_lengths()
    test_wesep_validation_rejects_speaker_overlap()
    test_wesep_requires_batch_size_one()
    print("ALL PASS")
