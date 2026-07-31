#!/usr/bin/env python3
"""Attach cached CAM++ enrollment embeddings to a TSE manifest.

Run this script with ``code/.venv_campp``.  The output manifest remains usable
by the regular TSE tools and adds one field: ``enrollment_embedding``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import sherpa_onnx


SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--cache-dir", default="code/tse_embedding_cache/campp"
    )
    parser.add_argument(
        "--model", default="E:/hf_cache/campplus/campplus.onnx"
    )
    parser.add_argument("--num-threads", type=int, default=2)
    return parser.parse_args()


def cache_key(audio_path: Path) -> str:
    stat = audio_path.stat()
    identity = (
        f"{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    ).encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def extract_embedding(extractor, audio_path: Path) -> np.ndarray:
    wav, _ = librosa.load(audio_path, sr=SR, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty enrollment audio: {audio_path}")
    if wav.size < SR:
        wav = np.tile(wav, int(np.ceil(SR / wav.size)))[:SR]
    stream = extractor.create_stream()
    stream.accept_waveform(SR, np.ascontiguousarray(wav, dtype=np.float32))
    stream.input_finished()
    embedding = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"invalid CAM++ embedding: {audio_path}")
    return embedding / norm


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_path = Path(args.output_manifest)
    cache_dir = Path(args.cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.model,
            num_threads=args.num_threads,
            debug=False,
        )
    )

    total = 0
    created = 0
    with manifest_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            enrollment = Path(row["enrollment_audio"])
            key = cache_key(enrollment)
            embedding_path = cache_dir / f"{key}.npy"
            if not embedding_path.exists():
                embedding = extract_embedding(extractor, enrollment)
                if embedding.shape != (extractor.dim,):
                    raise ValueError(
                        f"unexpected embedding shape {embedding.shape}"
                    )
                np.save(embedding_path, embedding)
                created += 1
            row["enrollment_embedding"] = str(embedding_path)
            row["enrollment_embedding_model"] = str(args.model)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += 1

    if total == 0:
        raise ValueError(f"empty manifest: {manifest_path}")
    print(
        f"[CAM++] rows={total} new_embeddings={created} "
        f"dim={extractor.dim} -> {output_path}"
    )


if __name__ == "__main__":
    main()
