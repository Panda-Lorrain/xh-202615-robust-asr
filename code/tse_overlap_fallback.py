#!/usr/bin/env python3
"""Overlap-only pBSRNN inference with feature-based failure fallback.

The two commands intentionally run in the project's existing isolated envs:

1. ``infer`` (``.venv_tse``) runs pBSRNN only on overlap intervals and saves
   float32 candidate segments plus waveform-only failure features.
2. ``finalize`` (``.venv_campp``) adds CAM++ output/enrollment cosine, applies
   online-safe thresholds, and writes both overlap-only and fallback audio.

No reference transcript, clean target, SI-SNR, or ASR result is used for
routing.  The final manifest keeps all per-segment features and route reasons.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SR = 16000
EPS = 1e-8


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_intervals(
    intervals: Sequence[Sequence[int]], num_samples: int
) -> List[Tuple[int, int]]:
    """Clip, discard empty intervals, and merge touching intervals."""
    clipped = sorted(
        (
            max(0, int(interval[0])),
            min(num_samples, int(interval[1])),
        )
        for interval in intervals
        if len(interval) == 2
    )
    merged: List[Tuple[int, int]] = []
    for start, end in clipped:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def derive_overlap_intervals(
    row: dict, num_samples: int
) -> List[Tuple[int, int]]:
    """Read explicit intervals or derive the synthetic oracle intersection."""
    explicit = row.get("overlap_intervals")
    if explicit is not None:
        unit = row.get("overlap_interval_unit", "samples")
        scale = SR if unit in {"second", "seconds", "sec"} else 1
        return normalize_intervals(
            [
                (round(float(interval[0]) * scale),
                 round(float(interval[1]) * scale))
                for interval in explicit
            ],
            num_samples,
        )

    if row.get("overlap_start_sample") is not None:
        return normalize_intervals(
            [[row["overlap_start_sample"], row["overlap_end_sample"]]],
            num_samples,
        )

    overlap_samples = int(
        row.get("active_overlap_samples", row.get("overlap_samples", 0))
    )
    # active_overlap_samples describes activity amount, not necessarily the
    # geometric end point.  The generator's overlap_samples is authoritative
    # for the continuous source intersection when available.
    if row.get("overlap_samples") is not None:
        overlap_samples = int(row["overlap_samples"])
    if overlap_samples <= 0:
        return []
    start = max(
        int(row.get("target_start_sample", 0)),
        int(row.get("interferer_start_sample", 0)),
    )
    return normalize_intervals([[start, start + overlap_samples]], num_samples)


def expanded_interval(
    start: int, end: int, num_samples: int, context_samples: int
) -> Tuple[int, int]:
    return (
        max(0, start - context_samples),
        min(num_samples, end + context_samples),
    )


def waveform_features(
    mixture: np.ndarray, estimate: np.ndarray
) -> Dict[str, float]:
    finite = bool(np.isfinite(estimate).all())
    safe = np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
    mix_rms = float(np.sqrt(np.mean(np.square(mixture)) + EPS))
    out_rms = float(np.sqrt(np.mean(np.square(safe)) + EPS))
    mix_peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    out_peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    return {
        "finite": finite,
        "nan_count": int(np.isnan(estimate).sum()),
        "inf_count": int(np.isinf(estimate).sum()),
        "mixture_rms": mix_rms,
        "output_rms": out_rms,
        "rms_ratio": out_rms / max(mix_rms, EPS),
        "mixture_peak": mix_peak,
        "output_peak": out_peak,
        "peak_ratio": out_peak / max(mix_peak, EPS),
        "clipping_fraction": float(np.mean(np.abs(safe) >= 0.999)),
    }


def splice_segment(
    base: np.ndarray,
    estimate: np.ndarray,
    start: int,
    end: int,
    fade_samples: int,
) -> None:
    """Replace [start,end), keeping every non-overlap sample untouched."""
    length = end - start
    if length <= 0 or estimate.shape[0] != length:
        raise ValueError("estimate length must equal splice interval length")
    weight = np.ones(length, dtype=np.float32)
    fade = min(max(0, fade_samples), length // 2)
    if fade:
        # Endpoints remain raw.  Cross-fade is entirely inside the declared
        # overlap interval, so non-overlap passthrough remains exact.
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        weight[:fade] = ramp
        weight[-fade:] = ramp[::-1]
    original = base[start:end].copy()
    base[start:end] = original * (1.0 - weight) + estimate * weight


def route_segment(features: dict, thresholds: dict) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not features.get("finite", False):
        reasons.append("non_finite")
    cosine = features.get("output_enrollment_cosine")
    if cosine is None or not np.isfinite(cosine):
        reasons.append("missing_cosine")
    elif cosine < thresholds["cosine_min"]:
        reasons.append("cosine_low")
    if features["rms_ratio"] < thresholds["rms_ratio_min"]:
        reasons.append("rms_too_low")
    if features["rms_ratio"] > thresholds["rms_ratio_max"]:
        reasons.append("rms_too_high")
    if features["peak_ratio"] > thresholds["peak_ratio_max"]:
        reasons.append("peak_ratio_high")
    if features["clipping_fraction"] > thresholds["clipping_fraction_max"]:
        reasons.append("clipping")
    if features["duration_sec"] < thresholds["min_duration_sec"]:
        reasons.append("segment_too_short")
    return not reasons, reasons


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != SR:
        import librosa

        audio = librosa.resample(
            audio.mean(axis=1), orig_sr=sample_rate, target_sr=SR
        )
    else:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _save_audio(path: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR, subtype="FLOAT")


def _load_model(args, device):
    import torch

    from tse_wesep_train import load_bsrnn_class

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    bsrnn = load_bsrnn_class(args.wesep_root)
    model = bsrnn(
        spk_emb_dim=int(checkpoint["embedding_dim"]),
        sr=SR,
        win=512,
        stride=128,
        feature_dim=int(model_args["feature_dim"]),
        num_repeat=int(model_args["num_repeat"]),
        use_spk_transform=False,
        spk_fuse_type="multiply",
        multi_fuse=True,
        joint_training=False,
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def run_infer(args) -> None:
    import torch

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    model = _load_model(args, device)
    output_dir = Path(args.output_dir)
    candidate_dir = output_dir / "candidate_segments"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    context_samples = round(args.context_sec * SR)
    rows_out: List[dict] = []

    with torch.no_grad():
        for row_index, source in enumerate(_read_jsonl(args.manifest)):
            uid = str(source.get("id", row_index))
            mixture = _load_audio(source["recognition_audio"])
            intervals = derive_overlap_intervals(source, mixture.size)
            embedding = np.load(source["enrollment_embedding"]).astype(
                np.float32
            )
            embedding /= max(float(np.linalg.norm(embedding)), EPS)
            segment_rows = []
            for segment_index, (start, end) in enumerate(intervals):
                ctx_start, ctx_end = expanded_interval(
                    start, end, mixture.size, context_samples
                )
                model_input = torch.from_numpy(
                    mixture[ctx_start:ctx_end]
                ).unsqueeze(0).to(device)
                spk_embedding = torch.from_numpy(
                    embedding
                ).unsqueeze(0).to(device)
                estimate, _ = model(model_input, spk_embedding)
                estimate = estimate[0, : model_input.shape[-1]].float().cpu()
                core_start = start - ctx_start
                core_end = core_start + (end - start)
                raw_candidate = estimate[
                    core_start:core_end
                ].numpy().copy()
                raw_features = waveform_features(
                    mixture[start:end], raw_candidate
                )
                # SI-SNR is scale-invariant, so the separator is free to emit
                # an arbitrary global gain.  Match local mixture RMS before
                # splicing instead of relying on WAV integer clipping.
                gain_match = (
                    raw_features["mixture_rms"]
                    / max(raw_features["output_rms"], EPS)
                )
                candidate = raw_candidate * gain_match
                candidate_path = (
                    candidate_dir / f"{uid}_{segment_index:02d}.npy"
                )
                np.save(candidate_path, candidate)
                features = waveform_features(
                    mixture[start:end], candidate
                )
                features.update(
                    {
                        "model_output_rms": raw_features["output_rms"],
                        "model_output_peak": raw_features["output_peak"],
                        "model_rms_ratio": raw_features["rms_ratio"],
                        "model_peak_ratio": raw_features["peak_ratio"],
                        "gain_match_scale": gain_match,
                    }
                )
                features["duration_sec"] = (end - start) / SR
                segment_rows.append(
                    {
                        "index": segment_index,
                        "start_sample": start,
                        "end_sample": end,
                        "context_start_sample": ctx_start,
                        "context_end_sample": ctx_end,
                        "candidate_npy": str(candidate_path),
                        "features": features,
                    }
                )
            rows_out.append(
                {
                    **source,
                    "overlap_intervals_samples": [
                        list(interval) for interval in intervals
                    ],
                    "segments": segment_rows,
                }
            )
            print(f"[infer] {uid}: overlap_segments={len(segment_rows)}")

    output_manifest = output_dir / "candidates.jsonl"
    _write_jsonl(output_manifest, rows_out)
    print(f"[infer] rows={len(rows_out)} -> {output_manifest}")


def _campp_extractor(model_path: str, num_threads: int):
    import sherpa_onnx

    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
            debug=False,
        )
    )


def _campp_embedding(extractor, waveform: np.ndarray) -> np.ndarray:
    if waveform.size == 0:
        raise ValueError("cannot embed an empty waveform")
    if waveform.size < SR:
        waveform = np.tile(
            waveform, int(np.ceil(SR / waveform.size))
        )[:SR]
    stream = extractor.create_stream()
    stream.accept_waveform(
        SR, np.ascontiguousarray(waveform, dtype=np.float32)
    )
    stream.input_finished()
    embedding = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("CAM++ returned an invalid embedding")
    return embedding / norm


def run_finalize(args) -> None:
    extractor = _campp_extractor(args.campp_model, args.num_threads)
    output_dir = Path(args.output_dir)
    overlap_dir = output_dir / "overlap_only_audio"
    fallback_dir = output_dir / "fallback_audio"
    thresholds = {
        "cosine_min": args.cosine_min,
        "rms_ratio_min": args.rms_ratio_min,
        "rms_ratio_max": args.rms_ratio_max,
        "peak_ratio_max": args.peak_ratio_max,
        "clipping_fraction_max": args.clipping_fraction_max,
        "min_duration_sec": args.min_duration_sec,
    }
    output_rows = []

    for row_index, source in enumerate(_read_jsonl(args.candidates)):
        uid = str(source.get("id", row_index))
        mixture = _load_audio(source["recognition_audio"])
        overlap_only = mixture.copy()
        fallback = mixture.copy()
        enrollment = np.load(source["enrollment_embedding"]).astype(
            np.float32
        )
        enrollment /= max(float(np.linalg.norm(enrollment)), EPS)
        segment_results = []
        for segment in source["segments"]:
            candidate = np.load(segment["candidate_npy"]).astype(np.float32)
            features = dict(segment["features"])
            try:
                output_embedding = _campp_embedding(extractor, candidate)
                features["output_enrollment_cosine"] = float(
                    np.dot(output_embedding, enrollment)
                )
            except (ValueError, RuntimeError) as error:
                features["output_enrollment_cosine"] = None
                features["embedding_error"] = str(error)
            accepted, reasons = route_segment(features, thresholds)
            start = int(segment["start_sample"])
            end = int(segment["end_sample"])
            splice_segment(
                overlap_only,
                candidate,
                start,
                end,
                round(args.fade_ms * SR / 1000),
            )
            if accepted:
                splice_segment(
                    fallback,
                    candidate,
                    start,
                    end,
                    round(args.fade_ms * SR / 1000),
                )
            segment_results.append(
                {
                    **segment,
                    "features": features,
                    "route": "enhanced" if accepted else "fallback_raw",
                    "reasons": reasons,
                }
            )

        overlap_path = overlap_dir / f"{uid}.wav"
        fallback_path = fallback_dir / f"{uid}.wav"
        if source["segments"]:
            _save_audio(overlap_path, overlap_only)
        else:
            overlap_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source["recognition_audio"], overlap_path)
        if any(
            segment["route"] == "enhanced"
            for segment in segment_results
        ):
            _save_audio(fallback_path, fallback)
        else:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source["recognition_audio"], fallback_path)
        output_rows.append(
            {
                **source,
                "overlap_only_audio": str(overlap_path),
                "fallback_audio": str(fallback_path),
                "route_thresholds": thresholds,
                "segments": segment_results,
                "enhanced_segment_count": sum(
                    segment["route"] == "enhanced"
                    for segment in segment_results
                ),
                "fallback_segment_count": sum(
                    segment["route"] == "fallback_raw"
                    for segment in segment_results
                ),
            }
        )
        print(
            f"[finalize] {uid}: enhanced="
            f"{output_rows[-1]['enhanced_segment_count']} fallback="
            f"{output_rows[-1]['fallback_segment_count']}"
        )

    output_manifest = output_dir / "manifest.jsonl"
    _write_jsonl(output_manifest, output_rows)
    print(f"[finalize] rows={len(output_rows)} -> {output_manifest}")


def si_snr_np(estimate: np.ndarray, target: np.ndarray) -> float:
    length = min(estimate.size, target.size)
    estimate = estimate[:length].astype(np.float64)
    target = target[:length].astype(np.float64)
    estimate -= estimate.mean()
    target -= target.mean()
    projection = (
        np.dot(estimate, target)
        * target
        / max(float(np.dot(target, target)), EPS)
    )
    noise = estimate - projection
    return float(
        10.0
        * np.log10(
            (float(np.dot(projection, projection)) + EPS)
            / (float(np.dot(noise, noise)) + EPS)
        )
    )


def _summarize_improvements(
    scores: Sequence[float], raw_scores: Sequence[float]
) -> dict:
    scores_array = np.asarray(scores)
    improvements = scores_array - np.asarray(raw_scores)
    return {
        "si_snr": float(scores_array.mean()),
        "si_snri": float(improvements.mean()),
        "si_snri_median": float(np.median(improvements)),
        "si_snri_p10": float(np.percentile(improvements, 10)),
        "nondegraded_rate": float(np.mean(improvements > 0.0)),
    }


def run_evaluate(args) -> None:
    rows = _read_jsonl(args.manifest)
    unconditional_by_id = {}
    if args.unconditional_manifest:
        unconditional_by_id = {
            str(row["id"]): row["recognition_audio"]
            for row in _read_jsonl(args.unconditional_manifest)
        }
    systems = {
        "raw": lambda row: row["recognition_audio"],
        "overlap_only": lambda row: row["overlap_only_audio"],
        "overlap_fallback": lambda row: row["fallback_audio"],
    }
    if unconditional_by_id:
        systems["unconditional_bsrnn"] = lambda row: unconditional_by_id[
            str(row["id"])
        ]
    scores = {name: [] for name in systems}
    passthrough_checks = []
    for row in rows:
        clean = _load_audio(row["clean_target_audio"])
        raw = _load_audio(row["recognition_audio"])
        for name, path_getter in systems.items():
            estimate = _load_audio(path_getter(row))
            scores[name].append(si_snr_np(estimate, clean))
        if not row["segments"]:
            fallback = _load_audio(row["fallback_audio"])
            overlap_only = _load_audio(row["overlap_only_audio"])
            passthrough_checks.append(
                bool(
                    np.array_equal(raw, fallback)
                    and np.array_equal(raw, overlap_only)
                )
            )
    raw_scores = scores["raw"]
    result = {
        "n": len(rows),
        "non_overlap_n": len(passthrough_checks),
        "non_overlap_exact_passthrough": all(passthrough_checks),
        "systems": {
            name: _summarize_improvements(values, raw_scores)
            for name, values in scores.items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_calibrate(args) -> None:
    """Sweep a cosine gate on validation clean audio; never an online input."""
    rows = _read_jsonl(args.manifest)
    cosines = sorted(
        {
            float(segment["features"]["output_enrollment_cosine"])
            for row in rows
            for segment in row["segments"]
            if segment["features"].get("output_enrollment_cosine") is not None
        }
    )
    thresholds = [-1.0] + cosines + [1.000001]
    raw_scores = []
    cached = []
    for row in rows:
        mixture = _load_audio(row["recognition_audio"])
        clean = _load_audio(row["clean_target_audio"])
        raw_scores.append(si_snr_np(mixture, clean))
        cached.append((row, mixture, clean))

    sweep = []
    for cosine_min in thresholds:
        routed_scores = []
        accepted_count = 0
        for row, mixture, clean in cached:
            routed = mixture.copy()
            for segment in row["segments"]:
                features = segment["features"]
                cosine = features.get("output_enrollment_cosine")
                if cosine is None or cosine < cosine_min:
                    continue
                candidate = np.load(segment["candidate_npy"]).astype(
                    np.float32
                )
                splice_segment(
                    routed,
                    candidate,
                    int(segment["start_sample"]),
                    int(segment["end_sample"]),
                    round(args.fade_ms * SR / 1000),
                )
                accepted_count += 1
            routed_scores.append(si_snr_np(routed, clean))
        metrics = _summarize_improvements(routed_scores, raw_scores)
        sweep.append(
            {
                "cosine_min": cosine_min,
                "accepted_segments": accepted_count,
                **metrics,
            }
        )
    # Mean SI-SNRi is the primary phase-2 acoustic criterion; prefer the more
    # conservative threshold on exact ties.
    best = max(sweep, key=lambda row: (row["si_snri"], row["cosine_min"]))
    result = {
        "n": len(rows),
        "selection_metric": "mean_full_utterance_si_snri",
        "online_features": ["output_enrollment_cosine"],
        "offline_selection_only": ["clean_target_audio", "si_snri"],
        "best": best,
        "sweep": sweep,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--wesep-root", default="code/WeSep")
    infer.add_argument("--device", default="cuda")
    infer.add_argument("--context-sec", type=float, default=0.25)
    infer.set_defaults(func=run_infer)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--candidates", required=True)
    finalize.add_argument("--output-dir", required=True)
    finalize.add_argument(
        "--campp-model", default="E:/hf_cache/campplus/campplus.onnx"
    )
    finalize.add_argument("--num-threads", type=int, default=2)
    finalize.add_argument("--fade-ms", type=float, default=10.0)
    finalize.add_argument("--cosine-min", type=float, default=-1.0)
    finalize.add_argument("--rms-ratio-min", type=float, default=0.1)
    finalize.add_argument("--rms-ratio-max", type=float, default=4.0)
    finalize.add_argument("--peak-ratio-max", type=float, default=4.0)
    finalize.add_argument(
        "--clipping-fraction-max", type=float, default=0.01
    )
    finalize.add_argument("--min-duration-sec", type=float, default=0.05)
    finalize.set_defaults(func=run_finalize)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--unconditional-manifest")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=run_evaluate)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--manifest", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--fade-ms", type=float, default=10.0)
    calibrate.set_defaults(func=run_calibrate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
