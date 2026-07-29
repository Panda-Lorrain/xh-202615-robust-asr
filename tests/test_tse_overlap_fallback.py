#!/usr/bin/env python3
"""Logic tests for overlap-only TSE and online-safe fallback routing."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from tse_overlap_fallback import (  # noqa: E402
    derive_overlap_intervals,
    normalize_intervals,
    route_segment,
    splice_segment,
)


def test_interval_normalization_and_oracle_derivation():
    assert normalize_intervals(
        [[8, 12], [-2, 3], [3, 5], [20, 20]], 10
    ) == [(0, 5), (8, 10)]
    row = {
        "target_start_sample": 100,
        "interferer_start_sample": 300,
        "overlap_samples": 250,
    }
    assert derive_overlap_intervals(row, 1000) == [(300, 550)]
    explicit = {
        "overlap_intervals": [[0.1, 0.2]],
        "overlap_interval_unit": "seconds",
    }
    assert derive_overlap_intervals(explicit, 10000) == [(1600, 3200)]


def test_splice_keeps_non_overlap_exact():
    base = np.arange(20, dtype=np.float32)
    original = base.copy()
    splice_segment(
        base,
        np.full(8, 100.0, dtype=np.float32),
        start=6,
        end=14,
        fade_samples=2,
    )
    assert np.array_equal(base[:6], original[:6])
    assert np.array_equal(base[14:], original[14:])
    assert base[6] == original[6]
    assert base[8] == 100.0


def test_no_overlap_is_exact_passthrough():
    row = {"overlap_samples": 0}
    mixture = np.random.default_rng(1).normal(size=100).astype(np.float32)
    output = mixture.copy()
    for start, end in derive_overlap_intervals(row, mixture.size):
        splice_segment(
            output, np.zeros(end - start, np.float32), start, end, 2
        )
    assert np.array_equal(output, mixture)


def test_failure_router_reasons():
    thresholds = {
        "cosine_min": 0.2,
        "rms_ratio_min": 0.1,
        "rms_ratio_max": 4.0,
        "peak_ratio_max": 4.0,
        "clipping_fraction_max": 0.01,
        "min_duration_sec": 0.05,
    }
    good = {
        "finite": True,
        "output_enrollment_cosine": 0.5,
        "rms_ratio": 0.8,
        "peak_ratio": 1.2,
        "clipping_fraction": 0.0,
        "duration_sec": 0.5,
    }
    assert route_segment(good, thresholds) == (True, [])
    bad = dict(good)
    bad.update(
        {
            "output_enrollment_cosine": 0.1,
            "rms_ratio": 8.0,
            "clipping_fraction": 0.1,
        }
    )
    accepted, reasons = route_segment(bad, thresholds)
    assert not accepted
    assert reasons == ["cosine_low", "rms_too_high", "clipping"]


if __name__ == "__main__":
    test_interval_normalization_and_oracle_derivation()
    test_splice_keeps_non_overlap_exact()
    test_no_overlap_is_exact_passthrough()
    test_failure_router_reasons()
    print("test_tse_overlap_fallback OK")
