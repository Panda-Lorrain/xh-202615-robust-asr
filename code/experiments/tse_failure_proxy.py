#!/usr/bin/env python3
"""Evaluate online-safe TSE failure proxies with target-speaker LOSO.

The online features compare the raw overlap with the enhanced candidate only.
References and Qwen errors are used exclusively for offline threshold fitting
and evaluation; they are never emitted as runtime route features.
"""

import argparse
import json
import os
import string
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

SR = 16000
EPS = 1e-8
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128

# A lower distortion is safer. Correlation has the opposite direction.
FEATURE_DIRECTIONS = {
    "logmel_l1": "<=",
    "logmel_l2": "<=",
    "logmel_delta_l1": "<=",
    "spectral_convergence": "<=",
    "residual_rms_ratio": "<=",
    "waveform_correlation": ">=",
}


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SR:
        raise ValueError(f"expected {SR} Hz, got {sample_rate}: {path}")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def _hz_to_mel(frequency: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (np.power(10.0, mel / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    frequencies = np.linspace(0.0, SR / 2.0, N_FFT // 2 + 1)
    mel_edges = np.linspace(
        _hz_to_mel(np.asarray([0.0]))[0],
        _hz_to_mel(np.asarray([SR / 2.0]))[0],
        N_MELS + 2,
    )
    hz_edges = _mel_to_hz(mel_edges)
    filters = np.zeros((N_MELS, frequencies.size), dtype=np.float32)
    for index in range(N_MELS):
        left, center, right = hz_edges[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, EPS),
                (right - frequencies) / max(right - center, EPS),
            ),
        )
    return filters


MEL_FILTERBANK = mel_filterbank()


def log_mel_spectrogram(
    waveform: np.ndarray, feature_extractor=None
) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    if feature_extractor is not None:
        features = feature_extractor(
            waveform,
            sampling_rate=SR,
            return_tensors="np",
            padding=False,
            return_attention_mask=True,
        )["input_features"][0]
        return np.asarray(features.T, dtype=np.float32)
    if waveform.size < N_FFT:
        waveform = np.pad(waveform, (0, N_FFT - waveform.size))
    frame_count = 1 + (waveform.size - N_FFT) // HOP_LENGTH
    frames = np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, N_FFT),
        strides=(waveform.strides[0] * HOP_LENGTH, waveform.strides[0]),
        writeable=False,
    )
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = spectrum @ MEL_FILTERBANK.T
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0).astype(np.float32)


def distortion_features(
    raw: np.ndarray, enhanced: np.ndarray, feature_extractor=None
) -> dict:
    length = min(raw.size, enhanced.size)
    raw = np.asarray(raw[:length], dtype=np.float32)
    enhanced = np.asarray(enhanced[:length], dtype=np.float32)
    raw_mel = log_mel_spectrogram(raw, feature_extractor)
    enhanced_mel = log_mel_spectrogram(enhanced, feature_extractor)
    frame_count = min(raw_mel.shape[0], enhanced_mel.shape[0])
    raw_mel = raw_mel[:frame_count]
    enhanced_mel = enhanced_mel[:frame_count]
    difference = enhanced_mel - raw_mel
    raw_delta = np.diff(raw_mel, axis=0)
    enhanced_delta = np.diff(enhanced_mel, axis=0)
    raw_centered = raw - raw.mean()
    enhanced_centered = enhanced - enhanced.mean()
    correlation = float(
        np.dot(raw_centered, enhanced_centered)
        / max(
            float(np.linalg.norm(raw_centered) * np.linalg.norm(enhanced_centered)),
            EPS,
        )
    )
    return {
        "logmel_l1": float(np.mean(np.abs(difference))),
        "logmel_l2": float(np.sqrt(np.mean(np.square(difference)))),
        "logmel_delta_l1": float(
            np.mean(np.abs(enhanced_delta - raw_delta))
            if raw_delta.size
            else 0.0
        ),
        "spectral_convergence": float(
            np.linalg.norm(difference) / max(float(np.linalg.norm(raw_mel)), EPS)
        ),
        "residual_rms_ratio": float(
            np.sqrt(np.mean(np.square(enhanced - raw)))
            / max(float(np.sqrt(np.mean(np.square(raw)))), EPS)
        ),
        "waveform_correlation": correlation,
    }


def _accepted(value: float, threshold: float, direction: str) -> bool:
    return value <= threshold if direction == "<=" else value >= threshold


def fit_threshold(
    samples: Sequence[dict], feature: str, direction: str
) -> dict:
    usable = [row for row in samples if row["features"].get(feature) is not None]
    if not usable:
        return {
            "threshold": 0.0,
            "accepted": 0,
            "training_errors": int(
                sum(row["raw_errors"] for row in samples)
            ),
        }
    values = sorted({float(row["features"][feature]) for row in usable})
    margin = max(abs(values[0]), abs(values[-1]), 1.0) * 1e-9
    thresholds = [values[0] - margin, *values, values[-1] + margin]
    candidates = []
    for threshold in thresholds:
        accepted_count = 0
        errors = 0
        for row in samples:
            value = row["features"].get(feature)
            use_enhanced = value is not None and _accepted(
                float(value), threshold, direction
            )
            errors += (
                row["enhanced_errors"] if use_enhanced else row["raw_errors"]
            )
            accepted_count += int(use_enhanced)
        # Prefer the conservative route on exact error ties.
        candidates.append((errors, accepted_count, threshold))
    errors, accepted_count, threshold = min(candidates)
    return {
        "threshold": float(threshold),
        "accepted": int(accepted_count),
        "training_errors": int(errors),
    }


def speaker_loso(samples: Sequence[dict], feature: str, direction: str) -> dict:
    speakers = sorted({row["target_spk"] for row in samples})
    decisions = {}
    folds = []
    for speaker in speakers:
        training = [row for row in samples if row["target_spk"] != speaker]
        held_out = [row for row in samples if row["target_spk"] == speaker]
        fitted = fit_threshold(training, feature, direction)
        accepted_count = 0
        for row in held_out:
            value = row["features"].get(feature)
            use_enhanced = value is not None and _accepted(
                float(value), fitted["threshold"], direction
            )
            decisions[row["id"]] = use_enhanced
            accepted_count += int(use_enhanced)
        folds.append(
            {
                "held_out_speaker": speaker,
                **fitted,
                "held_out_n": len(held_out),
                "held_out_accepted": accepted_count,
            }
        )
    errors = sum(
        row["enhanced_errors"] if decisions[row["id"]] else row["raw_errors"]
        for row in samples
    )
    return {"errors": int(errors), "decisions": decisions, "folds": folds}


def bootstrap_delta_ci(
    samples: Sequence[dict],
    decisions: Dict[str, bool],
    bootstrap_samples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    raw_errors = np.asarray([row["raw_errors"] for row in samples])
    routed_errors = np.asarray(
        [
            row["enhanced_errors"] if decisions[row["id"]] else row["raw_errors"]
            for row in samples
        ]
    )
    units = np.asarray([row["reference_units"] for row in samples])
    deltas = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        chosen = rng.integers(0, len(samples), len(samples))
        denominator = max(int(units[chosen].sum()), 1)
        deltas[index] = (
            routed_errors[chosen].sum() - raw_errors[chosen].sum()
        ) / denominator
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def _normalize_reference(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return "".join(
        character
        for character in normalized
        if character not in string.whitespace
        and not unicodedata.category(character).startswith("P")
    )


def build_samples(
    manifest: Sequence[dict], comparison: dict, feature_extractor=None
) -> List[dict]:
    comparison_by_id = {
        str(row["id"]): row for row in comparison["per_sample"]
    }
    samples = []
    for source in manifest:
        uid = str(source["id"])
        compared = comparison_by_id[uid]
        mixture = _load_audio(source["recognition_audio"])
        weighted_features = {name: [] for name in FEATURE_DIRECTIONS}
        weights = []
        for segment in source["segments"]:
            start = int(segment["start_sample"])
            end = int(segment["end_sample"])
            enhanced = np.load(segment["candidate_npy"]).astype(np.float32)
            features = distortion_features(
                mixture[start:end], enhanced, feature_extractor
            )
            weight = max(end - start, 1)
            weights.append(weight)
            for name, value in features.items():
                weighted_features[name].append(value)
        aggregate = {}
        for name, values in weighted_features.items():
            aggregate[name] = (
                float(np.average(values, weights=weights)) if values else None
            )
        target_spk = str(
            source.get("target_spk")
            or uid.split("_", 1)[0].replace("BAC009", "")
        )
        samples.append(
            {
                "id": uid,
                "target_spk": target_spk,
                "reference_units": max(
                    len(_normalize_reference(compared["ref"])), 1
                ),
                "raw_errors": int(compared["raw_errors"]),
                "enhanced_errors": int(compared["enhanced_errors"]),
                "features": aggregate,
            }
        )
    return samples


def evaluate(args) -> None:
    feature_extractor = None
    if args.qwen_model:
        from transformers import WhisperFeatureExtractor

        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            args.qwen_model, local_files_only=True
        )
    manifest = _read_jsonl(args.manifest)
    comparison = json.loads(Path(args.qwen_compare).read_text(encoding="utf-8"))
    samples = build_samples(manifest, comparison, feature_extractor)
    denominator = sum(row["reference_units"] for row in samples)
    raw_errors = sum(row["raw_errors"] for row in samples)
    enhanced_errors = sum(row["enhanced_errors"] for row in samples)
    proxies = {}
    for feature, direction in FEATURE_DIRECTIONS.items():
        loso = speaker_loso(samples, feature, direction)
        ci = bootstrap_delta_ci(
            samples,
            loso["decisions"],
            args.bootstrap_samples,
            args.seed,
        )
        global_fit = fit_threshold(samples, feature, direction)
        proxies[feature] = {
            "direction": direction,
            "loso_cer": loso["errors"] / denominator,
            "loso_delta_cer": (loso["errors"] - raw_errors) / denominator,
            "loso_delta_bootstrap_ci95": list(ci),
            "loso_accepted": sum(loso["decisions"].values()),
            "folds": loso["folds"],
            "deployment_fit_all_validation": global_fit,
        }
    best_name = min(
        proxies,
        key=lambda name: (
            proxies[name]["loso_cer"],
            proxies[name]["loso_accepted"],
        ),
    )
    best_decisions = speaker_loso(
        samples, best_name, FEATURE_DIRECTIONS[best_name]
    )["decisions"]
    result = {
        "n": len(samples),
        "speaker_count": len({row["target_spk"] for row in samples}),
        "raw_cer": raw_errors / denominator,
        "enhanced_all_cer": enhanced_errors / denominator,
        "best_loso_proxy": best_name,
        "best_loso_cer": proxies[best_name]["loso_cer"],
        "best_loso_delta_cer": proxies[best_name]["loso_delta_cer"],
        "feature_frontend": (
            f"Qwen WhisperFeatureExtractor: {args.qwen_model}"
            if feature_extractor is not None
            else "built-in 128-bin log-mel approximation"
        ),
        "proxies": proxies,
        "per_sample": [
            {
                **row,
                "best_loso_route": (
                    "enhanced" if best_decisions[row["id"]] else "raw"
                ),
            }
            for row in samples
        ],
        "safety_note": (
            "reference/Qwen errors are offline labels only; runtime routing "
            "uses raw-versus-enhanced distortion features"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_sample"},
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--qwen-compare", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--qwen-model",
        default=os.environ.get(
            "MODEL_QWEN3_ASR",
            "E:/hf_cache/Qwen3-ASR-1.7B" if os.name == "nt" else "",
        ),
        help=(
            "local Qwen3-ASR path used for its exact 128-bin Whisper frontend; "
            "pass an empty string to use the dependency-free approximation"
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
