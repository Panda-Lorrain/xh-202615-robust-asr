import os
import sys
import random

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))

from tse_data_aug import (
    _active_rms,
    add_noise_relative_to_target,
    mix_with_random_timing,
    synthesize_tse_triple,
)
from tse_train import apply_complex_mask, collate_fn, si_snr
from build_aishell_manifest import split_train_val_speakers


def test_random_timing_keeps_full_target_and_sir():
    target = np.sin(np.linspace(0, 200, 32000, dtype=np.float32))
    interferer = np.cos(np.linspace(0, 260, 40000, dtype=np.float32))
    mixed, clean, meta = mix_with_random_timing(
        target, interferer, overlap_ratio=0.5, sir_db=3.0,
        rng=random.Random(7),
    )
    start = meta["target_start_sample"]
    assert len(mixed) == len(clean)
    assert np.allclose(clean[start:start + len(target)], target)
    assert np.count_nonzero(clean[:start]) == 0
    interference = mixed - clean
    measured_sir = 20.0 * np.log10(
        _active_rms(clean) / _active_rms(interference)
    )
    assert abs(measured_sir - 3.0) < 0.05, measured_sir
    assert meta["overlap_samples"] == 16000
    assert 0.45 <= meta["active_overlap_ratio"] <= 0.55
    print("test_random_timing_keeps_full_target_and_sir OK")


def test_noise_snr_is_relative_to_clean_target():
    clean = np.zeros(48000, dtype=np.float32)
    clean[8000:40000] = 0.2
    mixed = clean.copy()
    noise = np.linspace(-1.0, 1.0, len(clean), dtype=np.float32)
    noisy = add_noise_relative_to_target(mixed, clean, noise, snr_db=-2.0)
    added = noisy - mixed
    measured = 20.0 * np.log10(_active_rms(clean) / _active_rms(added))
    assert abs(measured - (-2.0)) < 0.05, measured
    print("test_noise_snr_is_relative_to_clean_target OK")


def test_enrollment_is_required():
    wav = np.ones(16000, dtype=np.float32)
    try:
        synthesize_tse_triple(
            wav, wav, None, overlap_ratio=1.0, snr_db=0.0,
            enroll_pollute_p=0.0, rng=random.Random(1),
        )
    except ValueError as exc:
        assert "enrollment_wav" in str(exc)
    else:
        raise AssertionError("same-utterance enrollment fallback must not be allowed")
    print("test_enrollment_is_required OK")


def test_environment_noise_is_not_silently_replaced():
    wav = np.ones(16000, dtype=np.float32)
    try:
        synthesize_tse_triple(
            wav,
            wav,
            None,
            enrollment_wav=wav,
            overlap_ratio=1.0,
            snr_db=0.0,
            noise_type="env",
            enroll_pollute_p=0.0,
            rng=random.Random(2),
        )
    except ValueError as exc:
        assert "noise_wav" in str(exc)
    else:
        raise AssertionError("missing environment noise must fail")
    print("test_environment_noise_is_not_silently_replaced OK")


def test_complex_mask_uses_complex_multiplication():
    mix = torch.tensor([[[[2.0]], [[3.0]]]])
    mask = torch.tensor([[[[5.0]], [[7.0]]]])
    out = apply_complex_mask(mix, mask)
    assert out[0, 0, 0, 0].item() == -11.0
    assert out[0, 1, 0, 0].item() == 29.0
    print("test_complex_mask_uses_complex_multiplication OK")


def test_collate_preserves_lengths_and_sisnr_ignores_padding():
    batch = [
        (
            torch.tensor([1.0, -1.0, 0.5]),
            torch.tensor([1.0, -1.0, 0.5]),
            torch.zeros(4),
            torch.tensor([1, 2]),
            "a",
        ),
        (
            torch.tensor([0.2, -0.2]),
            torch.tensor([0.2, -0.2]),
            torch.zeros(4),
            torch.tensor([3]),
            "b",
        ),
    ]
    mixes, cleans, enrolls, wav_lens, tokens, token_lens, ids = collate_fn(batch)
    assert mixes.shape == cleans.shape == (2, 3)
    assert wav_lens.tolist() == [3, 2]
    assert token_lens.tolist() == [2, 1]
    scores = si_snr(mixes, cleans, wav_lens)
    assert torch.all(scores > 60), scores
    print("test_collate_preserves_lengths_and_sisnr_ignores_padding OK")


def test_train_val_speakers_are_disjoint():
    items = []
    for speaker_idx in range(8):
        for utt_idx in range(2):
            items.append({
                "wav": f"s{speaker_idx}_u{utt_idx}.wav",
                "ref": "测试",
                "spk": f"s{speaker_idx}",
                "utt": f"u{utt_idx}",
            })
    pools = split_train_val_speakers(
        items,
        n_target_speakers=2,
        n_interferer_speakers=2,
        n_val_target_speakers=2,
        n_val_interferer_speakers=2,
        max_utts_per_target=2,
        max_utts_per_interferer=2,
        seed=42,
    )
    train_target, train_interferer, val_target, val_interferer = pools
    train_target_spks = {row["spk"] for row in train_target}
    train_interferer_spks = {row["spk"] for row in train_interferer}
    val_target_spks = {row["spk"] for row in val_target}
    val_interferer_spks = {row["spk"] for row in val_interferer}
    assert train_target_spks == train_interferer_spks
    assert val_target_spks == val_interferer_spks
    assert train_target_spks.isdisjoint(val_target_spks)
    print("test_train_val_speakers_are_disjoint OK")


def test_quiet_target_changes_effective_sir_and_snr():
    target = np.ones(16000, dtype=np.float32)
    interferer = np.ones(16000, dtype=np.float32)
    enrollment = np.ones(20000, dtype=np.float32)
    _, recognition, clean, meta = synthesize_tse_triple(
        target,
        interferer,
        None,
        enrollment_wav=enrollment,
        overlap_ratio=1.0,
        snr_db=0.0,
        noise_type="white",
        target_gain_db=-6.0,
        sir_db=0.0,
        enroll_pollute_p=0.0,
        rng=random.Random(4),
    )
    assert len(recognition) == len(clean)
    assert abs(meta["measured_sir_db"] - (-6.0)) < 0.1
    assert abs(meta["measured_snr_db"] - (-6.0)) < 0.1
    print("test_quiet_target_changes_effective_sir_and_snr OK")


if __name__ == "__main__":
    test_random_timing_keeps_full_target_and_sir()
    test_noise_snr_is_relative_to_clean_target()
    test_enrollment_is_required()
    test_environment_noise_is_not_silently_replaced()
    test_complex_mask_uses_complex_multiplication()
    test_collate_preserves_lengths_and_sisnr_ignores_padding()
    test_train_val_speakers_are_disjoint()
    test_quiet_target_changes_effective_sir_and_snr()
    print("ALL PASS")
