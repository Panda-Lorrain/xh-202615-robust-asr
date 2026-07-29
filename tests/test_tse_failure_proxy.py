#!/usr/bin/env python3
"""Pure logic tests for TSE mel-distortion failure proxies."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tse_failure_proxy import (  # noqa: E402
    distortion_features,
    fit_threshold,
    speaker_loso,
)


def test_identical_waveforms_have_zero_distortion():
    waveform = np.random.default_rng(7).normal(size=3200).astype(np.float32)
    features = distortion_features(waveform, waveform.copy())
    assert features["logmel_l1"] == 0.0
    assert features["logmel_l2"] == 0.0
    assert features["logmel_delta_l1"] == 0.0
    assert features["spectral_convergence"] == 0.0
    assert features["residual_rms_ratio"] == 0.0
    assert np.isclose(features["waveform_correlation"], 1.0)


def _sample(uid, speaker, value, raw_errors, enhanced_errors):
    return {
        "id": uid,
        "target_spk": speaker,
        "reference_units": 10,
        "raw_errors": raw_errors,
        "enhanced_errors": enhanced_errors,
        "features": {"logmel_l1": value},
    }


def test_threshold_fit_prefers_safe_low_distortion_candidates():
    samples = [
        _sample("a", "s1", 0.1, 5, 1),
        _sample("b", "s1", 0.2, 5, 2),
        _sample("c", "s2", 0.8, 2, 7),
    ]
    fitted = fit_threshold(samples, "logmel_l1", "<=")
    assert fitted["accepted"] == 2
    assert fitted["training_errors"] == 5
    assert 0.2 <= fitted["threshold"] < 0.8


def test_loso_never_trains_on_held_out_speaker():
    samples = [
        _sample("a", "s1", 0.1, 4, 1),
        _sample("b", "s1", 0.9, 1, 5),
        _sample("c", "s2", 0.2, 4, 1),
        _sample("d", "s2", 0.8, 1, 5),
    ]
    result = speaker_loso(samples, "logmel_l1", "<=")
    assert result["decisions"] == {
        "a": True,
        "b": False,
        "c": False,
        "d": False,
    }
    assert result["errors"] == 7
    assert result["folds"][0]["threshold"] == 0.2
    assert result["folds"][1]["threshold"] == 0.1


if __name__ == "__main__":
    test_identical_waveforms_have_zero_distortion()
    test_threshold_fit_prefers_safe_low_distortion_candidates()
    test_loso_never_trains_on_held_out_speaker()
    print("test_tse_failure_proxy OK")
