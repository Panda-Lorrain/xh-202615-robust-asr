#!/usr/bin/env python3
"""Prepare Phase-3 Sidecar manifests from the audited Phase-2 triples.

Adds a target-activity numpy vector (10 ms hop) to each row while preserving
the cached CAM++ enrollment embedding and all source/augmentation provenance.
The script never touches Dataset A; input manifests are synthetic AISHELL/MUSAN
triples only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def activity_from_clean(path: Path, hop_samples: int, threshold_db: float) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    frames = max(1, int(np.ceil(wav.size / hop_samples)))
    padded = np.pad(wav, (0, frames * hop_samples - wav.size))
    rms = np.sqrt(np.mean(padded.reshape(frames, hop_samples) ** 2, axis=1) + 1e-10)
    floor = max(float(rms.max()) * 10.0 ** (threshold_db / 20.0), 1e-5)
    return (rms >= floor).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Phase-2 manifest_campp.jsonl")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--activity-dir", required=True)
    parser.add_argument("--hop-samples", type=int, default=160)
    parser.add_argument("--threshold-db", type=float, default=-40.0)
    args = parser.parse_args()
    if args.hop_samples <= 0:
        parser.error("--hop-samples must be positive")

    source = Path(args.manifest)
    output = Path(args.output_manifest)
    activity_dir = Path(args.activity_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    activity_dir.mkdir(parents=True, exist_ok=True)
    rows = 0
    active_rates = []
    with source.open(encoding="utf-8") as inp, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("id", "clean_target_audio", "enrollment_embedding", "recognition_audio", "ref"):
                if not row.get(field):
                    raise ValueError(f"row {rows} missing required field {field}")
            activity_path = activity_dir / f"{row['id']}.npy"
            activity = activity_from_clean(Path(row["clean_target_audio"]), args.hop_samples, args.threshold_db)
            np.save(activity_path, activity)
            row["target_activity"] = str(activity_path)
            row["target_activity_hop_samples"] = args.hop_samples
            row["target_activity_source"] = "clean_target_audio"
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            active_rates.append(float(activity.mean()))
    if not rows:
        raise ValueError(f"empty manifest: {source}")
    print(f"[phase3] rows={rows} active_ratio={np.mean(active_rates):.3f} -> {output}")


if __name__ == "__main__":
    main()
