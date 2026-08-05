#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a small DACF counterfactual query set from local source manifests.

DACF keeps one A+B mixture fixed and changes only the enrollment/query:

    mix(A+B) + enroll(A) -> target A
    mix(A+B) + enroll(B) -> target B
    mix(A+B) + enroll(C) -> blank (C is absent)

The builder is deliberately a small, manifest-auditable POC.  It uses only
local AISHELL-style source manifests (or ``--aishell-root``), never Dataset-A,
and does not run a model or create embeddings.  All three query records share
the same noise identity/type, RIR identity, environment identity, and exact
mixture bytes.  Enrollment noise is generated from the same noise identity but
with an independent seed/offset; it is never copied from the recognition
segment.  Generated babble is intentionally disabled: without an explicit
external environment-noise manifest, a babble request deterministically falls
back to pink noise so A/B speech cannot become a noise source.  For absent-C
the clean target is a same-length zero waveform and the activity vector is all
zero, so downstream code can keep a single tensor shape.

Manifest identity contract:
  * ``query_role``/``query_id`` and ``query_role_id`` (A=0, B=1, C=2) are
    counterfactual roles only, never speaker identities.
  * ``query_speaker_id`` is the source speaker identity and
    ``query_speaker_label`` is its stable sorted integer label.  Speaker
    contrastive training uses this identity; grouping the three records uses
    ``base_mixture_id + query_role_id``.
  * ``enrollment_audio_view2`` is an independent augmentation of the same
    source speaker and is the explicit identity-positive pair.

AISHELL-1 is read speech rather than verified home-command speech.  Therefore
absent-C rows carry a hard-negative *candidate* plus an explicit verification
flag; the builder never labels AISHELL text as a complete home instruction.
Annotated source rows can set ``complete_instruction`` to true when a real
command recording is supplied later.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf


_HERE = Path(__file__).resolve().parent
_CODE_DIR = _HERE.parent
for _import_path in (str(_CODE_DIR), str(_HERE)):
    if _import_path not in sys.path:
        sys.path.insert(0, _import_path)

# Reuse the project's existing AISHELL manifest parser and same-environment
# augmentation primitives.  No new package is needed for this POC.
from build_aishell_manifest import collect_wav_items, parse_transcript  # noqa: E402
from build_dataset import gen_pink, gen_white  # noqa: E402
from data_aug_recipe import make_fast  # noqa: E402
from rir_augment import apply_rir  # noqa: E402
from tse_data_aug import (  # noqa: E402
    SR,
    _active_rms,
    add_noise_relative_to_target,
    mix_with_random_timing,
    sample_tse_aug_params,
)


SCHEMA_VERSION = "dacf-counterfactual-v0.2"
MAX_POC_MIXTURES = 32
MAX_SCALE_FALSIFICATION_MIXTURES = 80
MIN_ENROLL_SEC = 1.0
MAX_ENROLL_SEC = 2.0
QUERY_ROLE_IDS = {"A": 0, "B": 1, "C": 2}
_DATASET_A_MARKERS = ("dataseta", "dataset-a", "dataset_a", "test_wav/dataset")


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _assert_not_dataset_a(value: Any) -> None:
    """Reject Dataset-A paths before any audio is opened."""
    text = _path_text(value).casefold()
    if any(marker in text for marker in _DATASET_A_MARKERS):
        raise ValueError(f"Dataset-A path is forbidden for DACF construction: {value}")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "是"}
    return bool(value)


def normalize_source_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize an AISHELL-style row without changing its source path."""
    wav = item.get("wav") or item.get("path") or item.get("audio")
    speaker = item.get("spk") or item.get("speaker_id") or item.get("speaker")
    if not wav:
        raise ValueError(f"source row has no wav/path: {item}")
    if not speaker:
        raise ValueError(f"source row has no speaker id: {item}")
    _assert_not_dataset_a(wav)
    wav_text = _path_text(wav)
    utt = item.get("utt") or item.get("utterance_id") or Path(wav_text).stem
    ref = item.get("ref")
    if ref is None:
        ref = item.get("text", "")
    return {
        "wav": wav_text,
        "spk": str(speaker),
        "utt": str(utt),
        "ref": str(ref),
        "split": str(item.get("split", "")),
        "source_corpus": str(item.get("source_corpus", "AISHELL-1")),
        "complete_instruction": _as_bool(
            item.get("complete_instruction", item.get("is_complete_instruction", False))
        ),
    }


def normalize_source_items(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and deterministically order source rows."""
    out = [normalize_source_item(item) for item in items]
    if not out:
        raise ValueError("source manifest is empty")
    seen = set()
    for item in out:
        key = (item["spk"], item["wav"])
        if key in seen:
            raise ValueError(f"duplicate source row: {key}")
        seen.add(key)
    return sorted(out, key=lambda row: (row["spk"], row["utt"], row["wav"]))


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a small JSONL source/noise manifest; no file is downloaded."""
    _assert_not_dataset_a(path)
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def load_aishell_items(aishell_root: str | Path, source_splits: Sequence[str] = ("train",)) -> List[Dict[str, Any]]:
    """Load only the requested official AISHELL source splits."""
    root = Path(aishell_root)
    _assert_not_dataset_a(root)
    transcript_path = root / "transcript" / "aishell_transcript_v0.8.txt"
    wav_root = root / "wav"
    transcript = parse_transcript(str(transcript_path))
    items = collect_wav_items(str(wav_root), transcript)
    allowed = set(source_splits)
    if any(item.get("split") != "all" for item in items):
        items = [item for item in items if item.get("split") in allowed]
    if not items:
        raise ValueError(
            f"no AISHELL audio in {root} for source splits {sorted(allowed)}"
        )
    return normalize_source_items(items)


def _group_by_speaker(items: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item["spk"]), []).append(dict(item))
    for rows in groups.values():
        rows.sort(key=lambda row: (row["utt"], row["wav"]))
    return groups


def _source_key(item: Mapping[str, Any]) -> str:
    return f"{item['spk']}::{item['utt']}"


def _duration_samples(path: str, cache: MutableMapping[str, int]) -> int:
    """Read container metadata only when possible; fallback to decoding."""
    if path in cache:
        return cache[path]
    try:
        info = sf.info(path)
        samples = int(round(info.frames * SR / info.samplerate))
    except Exception:  # noqa: BLE001 - source validation reports the path later
        wav, _ = librosa.load(path, sr=SR, mono=True)
        samples = int(wav.size)
    cache[path] = samples
    return samples


def _load_audio(path: str, cache: MutableMapping[str, np.ndarray]) -> np.ndarray:
    if path not in cache:
        _assert_not_dataset_a(path)
        wav, _ = librosa.load(path, sr=SR, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or wav.size == 0:
            raise ValueError(f"empty or non-mono source audio: {path}")
        if not np.all(np.isfinite(wav)):
            raise ValueError(f"non-finite source audio: {path}")
        cache[path] = wav
    return cache[path]


def _capacity_seconds(group: Sequence[Mapping[str, Any]], need: int, duration_cache: MutableMapping[str, int]) -> float:
    durations = sorted(
        (_duration_samples(str(item["wav"]), duration_cache) / SR for item in group),
        reverse=True,
    )
    if len(durations) < need:
        return 0.0
    return float(durations[need - 1])


def _pick_rec_and_enrollment(
    group: Sequence[Mapping[str, Any]],
    min_enroll_samples: int,
    rng: random.Random,
    audio_cache: MutableMapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = list(group)
    rng.shuffle(candidates)
    for recognition in candidates:
        rec_audio = _load_audio(str(recognition["wav"]), audio_cache)
        if rec_audio.size == 0:
            continue
        enroll_candidates = [
            item
            for item in candidates
            if item["wav"] != recognition["wav"]
            and _load_audio(str(item["wav"]), audio_cache).size >= min_enroll_samples
        ]
        if enroll_candidates:
            return dict(recognition), dict(rng.choice(enroll_candidates))
    raise ValueError(
        f"speaker {group[0]['spk']} has no distinct enrollment utterance "
        f"of at least {min_enroll_samples / SR:.2f}s"
    )


def _pick_enrollment(
    group: Sequence[Mapping[str, Any]],
    min_enroll_samples: int,
    rng: random.Random,
    audio_cache: MutableMapping[str, np.ndarray],
) -> Dict[str, Any]:
    candidates = [
        item
        for item in group
        if _load_audio(str(item["wav"]), audio_cache).size >= min_enroll_samples
    ]
    if not candidates:
        raise ValueError(
            f"speaker {group[0]['spk']} has no enrollment utterance "
            f"of at least {min_enroll_samples / SR:.2f}s"
        )
    return dict(rng.choice(candidates))


def _sample_params(rng: random.Random, profile: str, max_enroll_sec: float) -> Dict[str, Any]:
    params = dict(sample_tse_aug_params(rng, profile=profile))
    upper = min(MAX_ENROLL_SEC, float(max_enroll_sec))
    if upper < MIN_ENROLL_SEC:
        raise ValueError("selected speakers do not have 1.0s enrollment audio")
    params["enroll_dur_sec"] = float(
        MIN_ENROLL_SEC if upper == MIN_ENROLL_SEC else rng.uniform(MIN_ENROLL_SEC, upper)
    )
    return params


def _load_rir_records(rir_root: str | Path) -> List[Dict[str, Any]]:
    """Load local RIRs with stable, auditable IDs."""
    root = Path(rir_root)
    _assert_not_dataset_a(root)
    records: List[Dict[str, Any]] = []
    for path_text in sorted(glob.glob(str(root / "**" / "*.wav"), recursive=True)):
        path = Path(path_text)
        try:
            rir, _ = librosa.load(str(path), sr=SR, mono=True)
        except Exception:  # noqa: BLE001 - skip invalid local RIRs like rir_augment.py
            continue
        rir = np.asarray(rir, dtype=np.float32)
        if rir.size == 0 or float(np.max(np.abs(rir))) < 1e-6:
            continue
        rir = (rir / (float(np.max(np.abs(rir))) + 1e-8)).astype(np.float32)
        rel = _path_text(path.relative_to(root))
        records.append({"id": f"rir:{rel}", "path": _path_text(path), "audio": rir})
    if not records:
        raise ValueError(f"no valid local RIR wav found under {root}")
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_audio(audio: np.ndarray) -> str:
    data = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    return hashlib.sha256(data.tobytes()).hexdigest()


def resolve_noise_type(requested_noise_type: str, has_external_noise: bool) -> Dict[str, str]:
    """Resolve a requested noise type without using query-speaker content.

    ``sample_tse_aug_params`` still samples the legacy ``babble`` request so
    the POC preserves the existing augmentation distribution at the manifest
    boundary.  The actual generated branch has no babble implementation:
    ``babble`` becomes deterministic pink noise unless an explicit external
    environment-noise manifest is present, in which case the effective type
    is ``env``.  This makes it impossible for A/B recognition audio to become
    an implicit noise corpus or cross-split speaker source.
    """
    requested = str(requested_noise_type).strip().casefold() or "white"
    if has_external_noise:
        return {
            "requested_noise_type": requested,
            "effective_noise_type": "env",
            "noise_type_reason": "explicit_external_noise_manifest",
        }
    if requested == "babble":
        return {
            "requested_noise_type": requested,
            "effective_noise_type": "pink",
            "noise_type_reason": (
                "babble_disabled_without_external_noise_to_avoid_"
                "query_speaker_content_leakage"
            ),
        }
    if requested in {"white", "pink"}:
        return {
            "requested_noise_type": requested,
            "effective_noise_type": requested,
            "noise_type_reason": "generated_noise_has_no_speaker_content_source",
        }
    return {
        "requested_noise_type": requested,
        "effective_noise_type": "white",
        "noise_type_reason": "unsupported_noise_type_fallback_without_speaker_content",
    }


def _cyclic_segment(source: np.ndarray, length: int, offset: int) -> np.ndarray:
    """Take a deterministic source segment, wrapping only at source bounds."""
    source = np.asarray(source, dtype=np.float32)
    if source.size == 0:
        raise ValueError("noise source is empty")
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    indices = (np.arange(length, dtype=np.int64) + int(offset)) % source.size
    return source[indices].astype(np.float32)


def _generate_noise(
    noise_type: str,
    length: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if noise_type == "pink":
        return np.asarray(gen_pink(length, rng), dtype=np.float32)
    if noise_type == "white":
        return np.asarray(gen_white(length, rng), dtype=np.float32)
    raise ValueError(
        "generated noise supports only white/pink; resolve requested noise "
        "type before construction (generated babble is disabled)"
    )


def _scale_noise_to_target(
    noise: np.ndarray,
    target: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    desired_rms = _active_rms(target) / (10.0 ** (float(snr_db) / 20.0))
    return (
        np.asarray(noise, dtype=np.float32)
        * (desired_rms / max(_active_rms(noise), 1e-8))
    ).astype(np.float32)


def _make_enrollment(
    source: np.ndarray,
    duration_samples: int,
    rng: random.Random,
    noise_anchor: np.ndarray,
    rir: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    if source.size < duration_samples:
        raise ValueError(
            f"enrollment source is {source.size / SR:.2f}s, "
            f"shorter than requested {duration_samples / SR:.2f}s"
        )
    start = rng.randrange(0, source.size - duration_samples + 1)
    enrollment = source[start : start + duration_samples].astype(np.float32)
    if rir is not None:
        enrollment = apply_rir(enrollment, rir)
    return (enrollment + noise_anchor).astype(np.float32), start


def _make_independent_enrollment_noise(
    base: Mapping[str, Any],
    length: int,
    rng: random.Random,
    *,
    avoid_offsets: Sequence[int],
    avoid_seeds: Sequence[int],
    avoid_hashes: Sequence[str],
) -> Tuple[np.ndarray, int, int, str]:
    """Generate an enrollment noise view from the same identity, not same samples."""
    source = base.get("noise_source_audio")
    source_size = 0 if source is None else int(np.asarray(source).size)
    for _ in range(64):
        seed = int(rng.randrange(2**31))
        if seed in avoid_seeds:
            continue
        if base["noise_type"] == "env":
            if source_size <= 1:
                raise ValueError(
                    "environment noise source must contain at least two samples "
                    "to make independent enrollment offsets"
                )
            offset = int(rng.randrange(source_size))
            if offset in avoid_offsets:
                continue
            raw = _cyclic_segment(np.asarray(source), length, offset)
        else:
            offset = 0
            raw = _generate_noise(str(base["noise_type"]), length, seed)
        raw_hash = _sha256_audio(raw)
        if raw_hash in avoid_hashes:
            continue
        scaled = _scale_noise_to_target(
            raw, base["nominal_target"], float(base["snr_db"])
        )
        return scaled, seed, offset, raw_hash
    raise RuntimeError("could not create an independent enrollment noise view")


def _activity_from_clean(clean: np.ndarray, hop_samples: int, threshold_db: float = -40.0) -> np.ndarray:
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    frames = max(1, int(math.ceil(clean.size / hop_samples)))
    padded = np.pad(clean.astype(np.float32), (0, frames * hop_samples - clean.size))
    rms = np.sqrt(
        np.mean(padded.reshape(frames, hop_samples) ** 2, axis=1) + 1e-10
    )
    peak = float(rms.max()) if rms.size else 0.0
    threshold = max(peak * 10.0 ** (threshold_db / 20.0), 1e-5)
    return (rms >= threshold).astype(np.uint8)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), SR, subtype="PCM_16")


def _hard_negative_verified(items: Sequence[Mapping[str, Any]]) -> bool:
    return any(_as_bool(item.get("complete_instruction", False)) for item in items)


def _build_base(
    a_audio: np.ndarray,
    b_audio: np.ndarray,
    params: Mapping[str, Any],
    rng: random.Random,
    noise_item: Optional[Mapping[str, Any]],
    noise_audio: Optional[np.ndarray],
    rir_record: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Create one A+B mixture and aligned A/B clean sources."""
    target = a_audio.astype(np.float32)
    if float(params["target_speed_rate"]) != 1.0:
        target = make_fast(target, float(params["target_speed_rate"]))
    interferer = b_audio.astype(np.float32)
    rir = None if rir_record is None else np.asarray(rir_record["audio"], dtype=np.float32)
    if rir is not None:
        target = apply_rir(target, rir)
        interferer = apply_rir(interferer, rir)

    nominal_mixed, nominal_a, timing = mix_with_random_timing(
        target,
        interferer,
        float(params["overlap_ratio"]),
        float(params["sir_db"]),
        rng,
    )
    nominal_b = nominal_mixed - nominal_a
    target_gain = 10.0 ** (float(params["target_gain_db"]) / 20.0)
    clean_a = nominal_a * target_gain
    mixed = clean_a + nominal_b

    noise_seed = int(rng.randrange(2**31))
    noise_type = str(params["effective_noise_type"])
    if noise_type == "env":
        if noise_audio is None:
            raise ValueError("noise_type='env' requires a noise manifest row")
        noise_source = np.asarray(noise_audio, dtype=np.float32)
        recognition_noise_offset = int(rng.randrange(noise_source.size))
        noise = _cyclic_segment(noise_source, mixed.size, recognition_noise_offset)
    else:
        noise_source = None
        recognition_noise_offset = 0
        noise = _generate_noise(noise_type, mixed.size, noise_seed)
    noise = np.asarray(noise, dtype=np.float32)
    recognition = add_noise_relative_to_target(
        mixed, nominal_a, noise, float(params["snr_db"])
    )
    scaled_noise = (recognition - mixed).astype(np.float32)
    noise_id = (
        str(noise_item.get("noise_id") or noise_item.get("id") or noise_item.get("wav"))
        if noise_item is not None
        else f"generated:{noise_type}"
    )
    return {
        "recognition": recognition.astype(np.float32),
        "clean_a": clean_a.astype(np.float32),
        "clean_b": nominal_b.astype(np.float32),
        "scaled_noise": scaled_noise,
        "nominal_target": nominal_a.astype(np.float32),
        "rir": rir,
        "noise_id": noise_id,
        "noise_type": noise_type,
        "requested_noise_type": str(params["requested_noise_type"]),
        "effective_noise_type": noise_type,
        "noise_type_reason": str(params["noise_type_reason"]),
        "noise_src": None if noise_item is None else _path_text(noise_item.get("wav", "")),
        "noise_seed": int(noise_seed),
        "recognition_noise_seed": int(noise_seed),
        "recognition_noise_offset_samples": int(recognition_noise_offset),
        "recognition_noise_raw_sha256": _sha256_audio(noise),
        "noise_source_audio": noise_source,
        "snr_db": float(params["snr_db"]),
        "timing": timing,
        "measured_sir_db": float(
            20.0 * np.log10(_active_rms(clean_a) / max(_active_rms(nominal_b), 1e-8))
        ),
        "measured_snr_db": float(
            20.0 * np.log10(_active_rms(clean_a) / max(_active_rms(scaled_noise), 1e-8))
        ),
    }


def _row_for_query(
    *,
    split: str,
    base_id: str,
    query_id: str,
    query_speaker: str,
    query_speaker_label: int,
    target_present: bool,
    ref: str,
    enrollment_src: Mapping[str, Any],
    recognition_path: Path,
    enrollment_path: Path,
    enrollment_view2_path: Path,
    clean_path: Path,
    activity_path: Path,
    clean: np.ndarray,
    common: Mapping[str, Any],
    source_a: Mapping[str, Any],
    source_b: Mapping[str, Any],
    enrollment_start: int,
    enrollment_view2_start: int,
    enrollment_duration_sec: float,
    hop_samples: int,
) -> Dict[str, Any]:
    target_src = source_a["wav"] if query_id == "A" else source_b["wav"] if query_id == "B" else None
    target_spk = source_a["spk"] if query_id == "A" else source_b["spk"] if query_id == "B" else None
    interferer_sources = [source_b["wav"]] if query_id == "A" else [source_a["wav"]] if query_id == "B" else [source_a["wav"], source_b["wav"]]
    interferer_speakers = [source_b["spk"]] if query_id == "A" else [source_a["spk"]] if query_id == "B" else [source_a["spk"], source_b["spk"]]
    complete_verified = _hard_negative_verified((source_a, source_b))
    hard_negative = query_id == "C"
    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "dacf_query",
        "split": split,
        "id": f"{base_id}__present_{query_id}" if target_present else f"{base_id}__absent_{query_id}",
        "base_mixture_id": base_id,
        "query_id": query_id,
        "query_role": "present_A" if query_id == "A" else "present_B" if query_id == "B" else "absent_C",
        "query_role_id": int(QUERY_ROLE_IDS[query_id]),
        "query_role_id_contract": {"A": 0, "B": 1, "C": 2},
        "query_speaker_id": query_speaker,
        "query_speaker_label": int(query_speaker_label),
        "query_speaker_label_contract": "sorted source speaker id -> stable integer; identity, never A/B/C role",
        "contrastive_identity_key": "query_speaker_id",
        "counterfactual_group_key": f"{base_id}:{QUERY_ROLE_IDS[query_id]}",
        "target_present": bool(target_present),
        "target_present_label": int(target_present),
        "target_spk": target_spk,
        "target_src": target_src,
        "target_transcript": ref if target_present else "",
        "ref": ref if target_present else "",
        "target_latent": None,
        "target_latent_status": "not_generated_poc",
        "recognition_audio": _path_text(recognition_path),
        "enrollment_audio": _path_text(enrollment_path),
        "enrollment_audio_view2": _path_text(enrollment_view2_path),
        "clean_target_audio": _path_text(clean_path),
        "target_audio": _path_text(clean_path),
        "target_activity": _path_text(activity_path),
        "target_activity_hop_samples": int(hop_samples),
        "target_activity_source": "clean_target_audio",
        "clean_target_is_empty": not target_present,
        "clean_target_nonzero_samples": int(np.count_nonzero(np.abs(clean) > 1e-7)),
        "enrollment_src": enrollment_src["wav"],
        "enrollment_spk": enrollment_src["spk"],
        "enrollment_utt": enrollment_src["utt"],
        "enrollment_start_sample": int(enrollment_start),
        "enrollment_view2_start_sample": int(enrollment_view2_start),
        "enrollment_duration_sec": float(enrollment_duration_sec),
        "enrollment_view_count": 2,
        "identity_positive": True,
        "identity_positive_group": f"speaker:{query_speaker_label}",
        "identity_positive_contract": "enrollment_audio and enrollment_audio_view2 share query_speaker_id/label but use independent augmentation",
        "mixture_speakers": {"A": source_a["spk"], "B": source_b["spk"]},
        "mixture_sources": {"A": source_a["wav"], "B": source_b["wav"]},
        "interferer_spks": interferer_speakers,
        "interferer_srcs": interferer_sources,
        "query_C_enrollment_only": bool(query_id == "C"),
        "noise_id": common["noise_id"],
        "noise_type": common["noise_type"],
        "requested_noise_type": common["requested_noise_type"],
        "effective_noise_type": common["effective_noise_type"],
        "noise_type_reason": common["noise_type_reason"],
        "noise_src": common["noise_src"],
        "noise_seed": common["noise_seed"],
        "recognition_noise_seed": common["recognition_noise_seed"],
        "recognition_noise_offset_samples": common["recognition_noise_offset_samples"],
        "recognition_noise_raw_sha256": common["recognition_noise_raw_sha256"],
        "enrollment_noise_seed": common["enrollment_noise_seed"],
        "enrollment_noise_offset_samples": common["enrollment_noise_offset_samples"],
        "enrollment_noise_raw_sha256": common["enrollment_noise_raw_sha256"],
        "enrollment_view2_noise_seed": common["enrollment_view2_noise_seed"][query_id],
        "enrollment_view2_noise_offset_samples": common["enrollment_view2_noise_offset_samples"][query_id],
        "enrollment_view2_noise_raw_sha256": common["enrollment_view2_noise_raw_sha256"][query_id],
        "same_noise_identity": True,
        "exact_noise_segment_shared": False,
        "same_noise_anchor_contract": "identity/type shared; exact recognition noise segment not shared",
        "enrollment_environment_anchor_shared": True,
        "rir_id": common["rir_id"],
        "rir_src": common["rir_src"],
        "environment_id": common["environment_id"],
        "same_env_enrollment": True,
        "same_noise_anchor": False,
        "same_rir_anchor": True,
        "mixture_sha256": common["mixture_sha256"],
        "enrollment_sha256": common["enrollment_sha256"][query_id],
        "enrollment_view2_sha256": common["enrollment_view2_sha256"][query_id],
        "hard_negative": hard_negative,
        "hard_negative_type": (
            "target_absent_with_complete_interferer_instruction"
            if hard_negative and complete_verified
            else "target_absent_interferer_speech_unverified"
            if hard_negative
            else None
        ),
        "hard_negative_complete_instruction_required": hard_negative,
        "hard_negative_complete_instruction_verified": bool(hard_negative and complete_verified),
        "hard_negative_status": (
            "verified_candidate" if hard_negative and complete_verified
            else "requires_real_home_command_source" if hard_negative
            else "not_applicable"
        ),
        "hard_negative_interferer_spks": [source_a["spk"], source_b["spk"]],
        "hard_negative_interferer_refs": [source_a["ref"], source_b["ref"]],
        "speaker_disjoint_group": f"{split}:{base_id}",
        "dataset_a_used": False,
        "dataset_a_policy": "forbidden",
        "source_corpus": source_a.get("source_corpus", "AISHELL-1"),
        "augmentation": dict(common["augmentation"]),
        "timing": dict(common["timing"]),
        "measured_sir_db": common["measured_sir_db"],
        "measured_snr_db": common["measured_snr_db"],
        "peak_norm_scale": common["peak_norm_scale"],
    }
    return row


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_speaker_disjoint(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate every emitted split pair for speaker and source separation."""
    by_split: Dict[str, set[str]] = {}
    source_by_split: Dict[str, set[str]] = {}
    for row in rows:
        split = str(row["split"])
        speakers = set(str(value) for value in row["mixture_speakers"].values())
        speakers.add(str(row["enrollment_spk"]))
        by_split.setdefault(split, set()).update(speakers)
        sources = set(row["mixture_sources"].values())
        sources.add(str(row["enrollment_src"]))
        source_by_split.setdefault(split, set()).update(sources)
    split_names = sorted(by_split)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = by_split[left] & by_split[right]
            if overlap:
                raise ValueError(
                    f"{left}/{right} speaker overlap: {sorted(overlap)}"
                )
            source_overlap = source_by_split[left] & source_by_split[right]
            if source_overlap:
                raise ValueError(
                    f"{left}/{right} source overlap: {sorted(source_overlap)}"
                )


def build_dacf_counterfactual(
    source_items: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    n_train_mixtures: int = 1,
    n_val_mixtures: int = 1,
    n_final_mixtures: int = 0,
    seed: int = 42,
    noise_items: Optional[Sequence[Mapping[str, Any]]] = None,
    rir_records: Optional[Sequence[Mapping[str, Any]]] = None,
    augmentation_profile: str = "balanced",
    hop_samples: int = 160,
    max_mixtures: int = MAX_POC_MIXTURES,
) -> Dict[str, Any]:
    """Build a deterministic, small DACF train/val set.

    ``source_items`` must contain AISHELL-style ``wav``, ``spk``, ``utt`` and
    ``ref`` fields.  Each selected speaker needs two distinct utterances so a
    recognition utterance and an enrollment utterance cannot be the same file.
    ``noise_items`` and ``rir_records`` are optional local assets; generated
    white/pink noise and ``rir_id='none'`` are explicit fallbacks.  A sampled
    ``babble`` request is deterministically downgraded to pink unless
    ``noise_items`` supplies an explicit environment source; A/B audio is
    never used as a generated babble source.
    """
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    if n_train_mixtures < 0 or n_val_mixtures < 0 or n_final_mixtures < 0:
        raise ValueError("mixture counts must be non-negative")
    total_mixtures = n_train_mixtures + n_val_mixtures + n_final_mixtures
    if total_mixtures <= 0:
        raise ValueError("at least one train or val mixture is required")
    if max_mixtures < 1 or max_mixtures > MAX_SCALE_FALSIFICATION_MIXTURES:
        raise ValueError(
            "max_mixtures must stay within the audited research cap "
            f"[1, {MAX_SCALE_FALSIFICATION_MIXTURES}], got {max_mixtures}"
        )
    if total_mixtures > max_mixtures:
        raise ValueError(
            f"DACF build is capped at {max_mixtures} base mixtures for this stage; "
            "use the preregistered scale-falsification wrapper rather than "
            "raising the cap ad hoc"
        )
    if augmentation_profile not in {"balanced", "hard", "legacy"}:
        raise ValueError(f"unknown augmentation profile: {augmentation_profile}")

    items = normalize_source_items(source_items)
    for item in items:
        _assert_not_dataset_a(item["wav"])
    noise_rows = [dict(row) for row in (noise_items or [])]
    for row in noise_rows:
        if row.get("wav"):
            _assert_not_dataset_a(row["wav"])
    rir_rows = [dict(row) for row in (rir_records or [])]
    groups = _group_by_speaker(items)
    eligible = {
        speaker: rows for speaker, rows in groups.items() if len(rows) >= 2
    }
    speaker_labels = {
        speaker: label for label, speaker in enumerate(sorted(groups))
    }
    required_speakers = 3 * total_mixtures
    if len(eligible) < required_speakers:
        raise ValueError(
            f"need {required_speakers} speakers with >=2 utterances, "
            f"found {len(eligible)}"
        )

    rng = random.Random(seed)
    speakers = sorted(eligible)
    rng.shuffle(speakers)
    duration_cache: Dict[str, int] = {}
    audio_cache: Dict[str, np.ndarray] = {}
    output_root = Path(out_dir)
    all_rows: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "out_dir": _path_text(output_root),
        "seed": int(seed),
        "n_train_mixtures": int(n_train_mixtures),
        "n_val_mixtures": int(n_val_mixtures),
        "n_final_mixtures": int(n_final_mixtures),
        "max_mixtures": int(max_mixtures),
        "records_per_mixture": 3,
        "dataset_a_used": False,
        "train_manifest": _path_text(output_root / "train" / "manifest.jsonl"),
        "val_manifest": _path_text(output_root / "val" / "manifest.jsonl"),
        "final_manifest": _path_text(output_root / "final" / "manifest.jsonl"),
        "hard_negative_verified_count": 0,
    }

    split_specs = (
        ("train", n_train_mixtures),
        ("val", n_val_mixtures),
        ("final", n_final_mixtures),
    )
    speaker_cursor = 0
    for split, mixture_count in split_specs:
        split_rows: List[Dict[str, Any]] = []
        for mixture_index in range(mixture_count):
            selected = speakers[speaker_cursor : speaker_cursor + 3]
            speaker_cursor += 3
            speaker_a, speaker_b, speaker_c = selected
            capacity = min(
                _capacity_seconds(eligible[speaker_a], 2, duration_cache),
                _capacity_seconds(eligible[speaker_b], 2, duration_cache),
                _capacity_seconds(eligible[speaker_c], 1, duration_cache),
            )
            params = _sample_params(rng, augmentation_profile, capacity)
            enroll_samples = int(round(float(params["enroll_dur_sec"]) * SR))
            source_a, enroll_a = _pick_rec_and_enrollment(
                eligible[speaker_a], enroll_samples, rng, audio_cache
            )
            source_b, enroll_b = _pick_rec_and_enrollment(
                eligible[speaker_b], enroll_samples, rng, audio_cache
            )
            enroll_c = _pick_enrollment(
                eligible[speaker_c], enroll_samples, rng, audio_cache
            )
            a_audio = _load_audio(source_a["wav"], audio_cache)
            b_audio = _load_audio(source_b["wav"], audio_cache)
            noise_resolution = resolve_noise_type(
                str(params["noise_type"]), has_external_noise=bool(noise_rows)
            )
            params.update(noise_resolution)
            noise_item: Optional[Dict[str, Any]] = None
            noise_audio: Optional[np.ndarray] = None
            if noise_rows:
                noise_item = dict(rng.choice(noise_rows))
                if not noise_item.get("wav"):
                    raise ValueError("noise row has no wav")
                noise_audio = _load_audio(_path_text(noise_item["wav"]), audio_cache)
            params["noise_type"] = params["effective_noise_type"]
            rir_record = dict(rng.choice(rir_rows)) if rir_rows else None
            base = _build_base(
                a_audio,
                b_audio,
                params,
                rng,
                noise_item,
                noise_audio,
                rir_record,
            )
            duration_samples = enroll_samples
            enrollment_noise, enrollment_noise_seed, enrollment_noise_offset, enrollment_noise_hash = _make_independent_enrollment_noise(
                base,
                duration_samples,
                rng,
                avoid_offsets=(base["recognition_noise_offset_samples"],),
                avoid_seeds=(base["recognition_noise_seed"],),
                avoid_hashes=(base["recognition_noise_raw_sha256"],),
            )
            view2_noise_by_query: Dict[str, Tuple[np.ndarray, int, int, str]] = {}
            for query_id in ("A", "B", "C"):
                view2_noise_by_query[query_id] = _make_independent_enrollment_noise(
                    base,
                    duration_samples,
                    rng,
                    avoid_offsets=(
                        base["recognition_noise_offset_samples"],
                        enrollment_noise_offset,
                    ),
                    avoid_seeds=(
                        base["recognition_noise_seed"],
                        enrollment_noise_seed,
                    ),
                    avoid_hashes=(
                        base["recognition_noise_raw_sha256"],
                        enrollment_noise_hash,
                    ),
                )
            enroll_audio_a, enroll_start_a = _make_enrollment(
                _load_audio(enroll_a["wav"], audio_cache),
                duration_samples,
                rng,
                enrollment_noise,
                base["rir"],
            )
            enroll_audio_b, enroll_start_b = _make_enrollment(
                _load_audio(enroll_b["wav"], audio_cache),
                duration_samples,
                rng,
                enrollment_noise,
                base["rir"],
            )
            enroll_audio_c, enroll_start_c = _make_enrollment(
                _load_audio(enroll_c["wav"], audio_cache),
                duration_samples,
                rng,
                enrollment_noise,
                base["rir"],
            )
            enroll_audio_view2_a, enroll_view2_start_a = _make_enrollment(
                _load_audio(enroll_a["wav"], audio_cache),
                duration_samples,
                rng,
                view2_noise_by_query["A"][0],
                base["rir"],
            )
            enroll_audio_view2_b, enroll_view2_start_b = _make_enrollment(
                _load_audio(enroll_b["wav"], audio_cache),
                duration_samples,
                rng,
                view2_noise_by_query["B"][0],
                base["rir"],
            )
            enroll_audio_view2_c, enroll_view2_start_c = _make_enrollment(
                _load_audio(enroll_c["wav"], audio_cache),
                duration_samples,
                rng,
                view2_noise_by_query["C"][0],
                base["rir"],
            )

            recognition = base["recognition"]
            clean_a = base["clean_a"]
            clean_b = base["clean_b"]
            clean_c = np.zeros_like(recognition, dtype=np.float32)
            peak = max(
                float(np.max(np.abs(recognition))) if recognition.size else 0.0,
                float(np.max(np.abs(enroll_audio_a))) if enroll_audio_a.size else 0.0,
                float(np.max(np.abs(enroll_audio_b))) if enroll_audio_b.size else 0.0,
                float(np.max(np.abs(enroll_audio_c))) if enroll_audio_c.size else 0.0,
                float(np.max(np.abs(enroll_audio_view2_a))) if enroll_audio_view2_a.size else 0.0,
                float(np.max(np.abs(enroll_audio_view2_b))) if enroll_audio_view2_b.size else 0.0,
                float(np.max(np.abs(enroll_audio_view2_c))) if enroll_audio_view2_c.size else 0.0,
            )
            peak_norm_scale = 1.0
            if peak > 0.99:
                peak_norm_scale = 0.99 / peak
                recognition = recognition * peak_norm_scale
                clean_a = clean_a * peak_norm_scale
                clean_b = clean_b * peak_norm_scale
                enroll_audio_a = enroll_audio_a * peak_norm_scale
                enroll_audio_b = enroll_audio_b * peak_norm_scale
                enroll_audio_c = enroll_audio_c * peak_norm_scale
                enroll_audio_view2_a = enroll_audio_view2_a * peak_norm_scale
                enroll_audio_view2_b = enroll_audio_view2_b * peak_norm_scale
                enroll_audio_view2_c = enroll_audio_view2_c * peak_norm_scale

            base_id = f"{split}_mix_{mixture_index:04d}"
            split_dir = output_root / split
            recognition_path = split_dir / "recognition" / f"{base_id}.wav"
            _write_wav(recognition_path, recognition)
            mixture_sha256 = _sha256_file(recognition_path)
            clean_paths = {
                "A": split_dir / "clean_target" / f"{base_id}__present_A.wav",
                "B": split_dir / "clean_target" / f"{base_id}__present_B.wav",
                "C": split_dir / "clean_target" / f"{base_id}__absent_C.wav",
            }
            enrollment_paths = {
                "A": split_dir / "enrollment" / f"{base_id}__present_A.wav",
                "B": split_dir / "enrollment" / f"{base_id}__present_B.wav",
                "C": split_dir / "enrollment" / f"{base_id}__absent_C.wav",
            }
            enrollment_view2_paths = {
                "A": split_dir / "enrollment_view2" / f"{base_id}__present_A.wav",
                "B": split_dir / "enrollment_view2" / f"{base_id}__present_B.wav",
                "C": split_dir / "enrollment_view2" / f"{base_id}__absent_C.wav",
            }
            activity_paths = {
                key: split_dir / "activity" / f"{base_id}__{key}.npy"
                for key in ("present_A", "present_B", "absent_C")
            }
            _write_wav(clean_paths["A"], clean_a)
            _write_wav(clean_paths["B"], clean_b)
            _write_wav(clean_paths["C"], clean_c)
            _write_wav(enrollment_paths["A"], enroll_audio_a)
            _write_wav(enrollment_paths["B"], enroll_audio_b)
            _write_wav(enrollment_paths["C"], enroll_audio_c)
            _write_wav(enrollment_view2_paths["A"], enroll_audio_view2_a)
            _write_wav(enrollment_view2_paths["B"], enroll_audio_view2_b)
            _write_wav(enrollment_view2_paths["C"], enroll_audio_view2_c)
            activities = {
                "present_A": _activity_from_clean(clean_a, hop_samples),
                "present_B": _activity_from_clean(clean_b, hop_samples),
                "absent_C": np.zeros(max(1, int(math.ceil(recognition.size / hop_samples))), dtype=np.uint8),
            }
            for key, activity in activities.items():
                activity_paths[key].parent.mkdir(parents=True, exist_ok=True)
                np.save(activity_paths[key], activity)

            rir_id = "none" if rir_record is None else str(rir_record["id"])
            common = {
                "mixture_sha256": mixture_sha256,
                "noise_id": base["noise_id"],
                "noise_type": base["noise_type"],
                "requested_noise_type": base["requested_noise_type"],
                "effective_noise_type": base["effective_noise_type"],
                "noise_type_reason": base["noise_type_reason"],
                "noise_src": base["noise_src"],
                "noise_seed": base["noise_seed"],
                "recognition_noise_seed": base["recognition_noise_seed"],
                "recognition_noise_offset_samples": base["recognition_noise_offset_samples"],
                "recognition_noise_raw_sha256": base["recognition_noise_raw_sha256"],
                "enrollment_noise_seed": int(enrollment_noise_seed),
                "enrollment_noise_offset_samples": int(enrollment_noise_offset),
                "enrollment_noise_raw_sha256": enrollment_noise_hash,
                "enrollment_view2_noise_seed": {
                    key: int(value[1]) for key, value in view2_noise_by_query.items()
                },
                "enrollment_view2_noise_offset_samples": {
                    key: int(value[2]) for key, value in view2_noise_by_query.items()
                },
                "enrollment_view2_noise_raw_sha256": {
                    key: value[3] for key, value in view2_noise_by_query.items()
                },
                "enrollment_sha256": {
                    "A": _sha256_file(enrollment_paths["A"]),
                    "B": _sha256_file(enrollment_paths["B"]),
                    "C": _sha256_file(enrollment_paths["C"]),
                },
                "enrollment_view2_sha256": {
                    "A": _sha256_file(enrollment_view2_paths["A"]),
                    "B": _sha256_file(enrollment_view2_paths["B"]),
                    "C": _sha256_file(enrollment_view2_paths["C"]),
                },
                "rir_id": rir_id,
                "rir_src": None if rir_record is None else _path_text(rir_record.get("path", "")),
                "environment_id": f"{split}:{base_id}:env",
                "augmentation": {
                    key: value for key, value in params.items() if key != "enroll_pollute_p"
                },
                "timing": base["timing"],
                "measured_sir_db": base["measured_sir_db"],
                "measured_snr_db": base["measured_snr_db"],
                "peak_norm_scale": float(peak_norm_scale),
            }
            query_specs = (
                ("A", speaker_a, True, source_a["ref"], enroll_a, enroll_start_a, enroll_view2_start_a, "present_A"),
                ("B", speaker_b, True, source_b["ref"], enroll_b, enroll_start_b, enroll_view2_start_b, "present_B"),
                ("C", speaker_c, False, "", enroll_c, enroll_start_c, enroll_view2_start_c, "absent_C"),
            )
            for query_id, query_speaker, present, ref, enrollment_src, enrollment_start, enrollment_view2_start, activity_key in query_specs:
                clean = clean_a if query_id == "A" else clean_b if query_id == "B" else clean_c
                row = _row_for_query(
                    split=split,
                    base_id=base_id,
                    query_id=query_id,
                    query_speaker=query_speaker,
                    query_speaker_label=speaker_labels[query_speaker],
                    target_present=present,
                    ref=ref,
                    enrollment_src=enrollment_src,
                    recognition_path=recognition_path,
                    enrollment_path=enrollment_paths[query_id],
                    enrollment_view2_path=enrollment_view2_paths[query_id],
                    clean_path=clean_paths[query_id],
                    activity_path=activity_paths[activity_key],
                    clean=clean,
                    common=common,
                    source_a=source_a,
                    source_b=source_b,
                    enrollment_start=enrollment_start,
                    enrollment_view2_start=enrollment_view2_start,
                    enrollment_duration_sec=float(duration_samples / SR),
                    hop_samples=hop_samples,
                )
                split_rows.append(row)
                all_rows.append(row)
                if row["hard_negative_complete_instruction_verified"]:
                    result["hard_negative_verified_count"] += 1
        _write_manifest(output_root / split / "manifest.jsonl", split_rows)
        result[f"{split}_records"] = len(split_rows)
        result[f"{split}_mixtures"] = mixture_count

    validate_speaker_disjoint(all_rows)
    result["total_records"] = len(all_rows)
    result["total_mixtures"] = total_mixtures
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--aishell-root", help="local AISHELL-1 root containing wav/ and transcript/")
    source.add_argument("--source-manifest", help="local AISHELL-style JSONL with wav/spk/utt/ref")
    parser.add_argument("--noise-manifest", default="", help="optional local noise JSONL; one env noise is shared per base mixture")
    parser.add_argument("--rir-root", default="", help="optional local RIR directory")
    parser.add_argument("--out", required=True, help="small POC output directory")
    parser.add_argument("--n-train-mixtures", type=int, default=1)
    parser.add_argument("--n-val-mixtures", type=int, default=1)
    parser.add_argument("--n-final-mixtures", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augmentation-profile", choices=("balanced", "hard", "legacy"), default="balanced")
    parser.add_argument("--hop-samples", type=int, default=160)
    parser.add_argument("--source-splits", nargs="+", default=["train"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.aishell_root:
        source_items = load_aishell_items(args.aishell_root, args.source_splits)
    else:
        source_items = normalize_source_items(read_jsonl(args.source_manifest))
    noise_items = read_jsonl(args.noise_manifest) if args.noise_manifest else []
    rir_records = _load_rir_records(args.rir_root) if args.rir_root else []
    result = build_dacf_counterfactual(
        source_items,
        args.out,
        n_train_mixtures=args.n_train_mixtures,
        n_val_mixtures=args.n_val_mixtures,
        n_final_mixtures=args.n_final_mixtures,
        seed=args.seed,
        noise_items=noise_items,
        rir_records=rir_records,
        augmentation_profile=args.augmentation_profile,
        hop_samples=args.hop_samples,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["hard_negative_verified_count"] == 0:
        print(
            "[dacf] absent-C hard-negative candidates were built, but no source row "
            "was annotated complete_instruction=true; AISHELL read speech is not a "
            "verified home-command negative."
        )


if __name__ == "__main__":
    main()
