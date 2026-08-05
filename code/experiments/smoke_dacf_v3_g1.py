"""One-group real-feature controllability smoke for DACF-v3 ECST.

This script reuses one provenance-locked v0.2 CAM++ query cache group and
extracts the mixture's exact local-Qwen 128-bin log-mel.  It is intentionally
only G1: memorising one byte-identical A/B/C group proves query controllability
and gradient plumbing, not unseen-speaker generalisation or CER improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from dacf_v3_ecst import DACFV3ECST
from dacf_v3_objective import compute_dacf_v3_loss


FORBIDDEN_MARKERS = ("dataset-a", "dataset_a", "dataseta")


def _path_text(value: Any) -> str:
    return str(value).replace("\\", "/")


def _guard_not_dataset_a(value: Any, field: str) -> None:
    text = _path_text(value).lower()
    if any(marker in text for marker in FORBIDDEN_MARKERS):
        raise ValueError(f"Dataset-A is forbidden in {field}: {value}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(manifest: Path, group_id: str) -> list[dict[str, Any]]:
    _guard_not_dataset_a(manifest, "manifest")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if str(row.get("base_mixture_id")) == group_id]
    if len(selected) != 3:
        raise ValueError(f"group {group_id!r} must contain exactly 3 rows, got {len(selected)}")
    selected.sort(key=lambda row: str(row.get("query_role")))
    required_roles = {"present_A", "present_B", "absent_C"}
    roles = {str(row.get("query_role")) for row in selected}
    if roles != required_roles:
        raise ValueError(f"group roles must be {sorted(required_roles)}, got {sorted(roles)}")
    mixture_hashes = {str(row.get("mixture_sha256")) for row in selected}
    recognition_paths = {_path_text(row.get("recognition_audio")) for row in selected}
    if len(mixture_hashes) != 1 or len(recognition_paths) != 1:
        raise ValueError("A/B/C must share one byte-identical mixture path and SHA")
    for row in selected:
        if bool(row.get("dataset_a_used")):
            raise ValueError("Dataset-A row is forbidden")
        if str(row.get("source_corpus")) != "AISHELL-1":
            raise ValueError("G1 source_corpus must be AISHELL-1")
        for key in ("recognition_audio", "enrollment_audio", "enrollment_audio_view2"):
            _guard_not_dataset_a(row.get(key), key)
    return selected


def _resolve_cache_artifact(manifest: Path, raw: Any) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = manifest.parent / path
    return path.resolve()


def _load_group_features(
    manifest: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import soundfile as sf
    from transformers import WhisperFeatureExtractor

    recognition = Path(str(rows[0]["recognition_audio"])).resolve()
    if not recognition.is_file():
        raise FileNotFoundError(recognition)
    actual_mixture_sha = _sha256_file(recognition)
    if actual_mixture_sha != str(rows[0]["mixture_sha256"]):
        raise ValueError("recognition audio SHA does not match the cache manifest")

    config_dir = Path(os.environ.get("DACF_QWEN_CONFIG", r"E:\hf_cache\Qwen3-ASR-1.7B"))
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(config_dir), local_files_only=True
    )
    contract = {
        "feature_size": int(extractor.feature_size),
        "n_fft": int(extractor.n_fft),
        "hop_length": int(extractor.hop_length),
        "dither": float(extractor.dither),
    }
    if contract != {"feature_size": 128, "n_fft": 400, "hop_length": 160, "dither": 0.0}:
        raise ValueError(f"local Qwen feature contract changed: {contract}")
    waveform, sample_rate = sf.read(
        str(recognition), dtype="float32", always_2d=False
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    batch = extractor(
        waveform,
        sampling_rate=int(sample_rate),
        return_tensors="np",
        padding=False,
    )
    mixture = np.asarray(batch["input_features"][0], dtype=np.float32)
    if mixture.ndim != 2 or mixture.shape[0] != 128:
        raise ValueError(f"unexpected Qwen log-mel shape {mixture.shape}")

    main_embeddings: list[np.ndarray] = []
    view2_embeddings: list[np.ndarray] = []
    activities: list[np.ndarray] = []
    labels: list[float] = []
    query_hashes: list[str] = []
    for row in rows:
        query_path = _resolve_cache_artifact(manifest, row["query_feature"])
        actual_query_sha = _sha256_file(query_path)
        if actual_query_sha != str(row["query_npz_sha256"]):
            raise ValueError(f"query NPZ SHA mismatch: {query_path}")
        with np.load(query_path, allow_pickle=False) as archive:
            main = np.asarray(archive["embedding"], dtype=np.float32)
            view2 = np.asarray(archive["embedding_view2"], dtype=np.float32)
            activity = np.asarray(archive["target_activity"], dtype=np.float32).reshape(-1)
        if main.shape != (512,) or view2.shape != (512,):
            raise ValueError("CAM++ query embeddings must both be 512d")
        if activity.size < mixture.shape[1]:
            activity = np.pad(activity, (0, mixture.shape[1] - activity.size))
        activity = activity[: mixture.shape[1]]
        main_embeddings.append(main)
        view2_embeddings.append(view2)
        activities.append(activity)
        labels.append(float(bool(row["target_present"])))
        query_hashes.append(actual_query_sha)
    return (
        mixture,
        np.stack(main_embeddings),
        np.stack(view2_embeddings),
        np.stack(activities),
        {
            "labels": labels,
            "mixture_sha256": actual_mixture_sha,
            "qwen_logmel_sha256": hashlib.sha256(mixture.tobytes()).hexdigest(),
            "qwen_feature_contract": contract,
            "query_npz_sha256": query_hashes,
            "recognition_audio": _path_text(recognition),
        },
    )


def _probabilities(model: DACFV3ECST, mixture: torch.Tensor, query: torch.Tensor) -> list[float]:
    model.eval()
    with torch.no_grad():
        values = torch.sigmoid(model(mixture, query).presence_logits)
    return [float(value) for value in values.cpu().tolist()]


def run_g1(
    feature_manifest: str | Path,
    *,
    group_id: str,
    updates: int = 60,
    learning_rate: float = 0.003,
    seed: int = 20260806,
    device: str = "auto",
) -> dict[str, Any]:
    if updates < 1 or learning_rate <= 0:
        raise ValueError("updates and learning_rate must be positive")
    manifest = Path(feature_manifest).resolve()
    rows = _read_rows(manifest, group_id)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(
        "cuda:0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    mixture_np, main_np, view2_np, activity_np, audit = _load_group_features(manifest, rows)
    mixture = torch.from_numpy(mixture_np).unsqueeze(0).repeat(3, 1, 1).to(resolved_device)
    query_main = torch.from_numpy(main_np).to(resolved_device)
    query_view2 = torch.from_numpy(view2_np).to(resolved_device)
    activity = torch.from_numpy(activity_np).to(resolved_device)
    labels = torch.tensor(audit.pop("labels"), dtype=torch.float32, device=resolved_device)
    group_index = torch.zeros(3, dtype=torch.long, device=resolved_device)

    model = DACFV3ECST().to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    initial_main = _probabilities(model, mixture, query_main)
    initial_view2 = _probabilities(model, mixture, query_view2)
    history: list[float] = []
    started = time.perf_counter()
    model.train()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        main = model(mixture, query_main)
        view2 = model(mixture, query_view2)
        objective = compute_dacf_v3_loss(
            main,
            view2,
            presence_labels=labels,
            activity_targets=activity,
            group_index=group_index,
        )
        objective.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        history.append(float(objective.total.detach().cpu()))
    runtime_sec = time.perf_counter() - started

    final_main = _probabilities(model, mixture, query_main)
    final_view2 = _probabilities(model, mixture, query_view2)
    present_indices = [index for index, label in enumerate(labels.tolist()) if label > 0.5]
    absent_index = next(index for index, label in enumerate(labels.tolist()) if label <= 0.5)
    swap_index = present_indices[0]
    swapped = query_main.clone()
    swapped[[swap_index, absent_index]] = swapped[[absent_index, swap_index]]
    swapped_main = _probabilities(model, mixture, swapped)
    swap_error = max(
        abs(swapped_main[swap_index] - final_main[absent_index]),
        abs(swapped_main[absent_index] - final_main[swap_index]),
    )
    threshold = 0.5
    both_views_pass = all(
        probabilities[index] >= 0.80
        for probabilities in (final_main, final_view2)
        for index in present_indices
    ) and all(
        probabilities[absent_index] <= 0.20
        for probabilities in (final_main, final_view2)
    )
    return {
        "schema": "dacf-v3-ecst-g1-v0.1",
        "verdict": "conditional-GO" if both_views_pass and swap_error < 1e-6 else "implementation-NO-GO",
        "verdict_scope": "one-group controllability only; no unseen-speaker, CER, RR, hard-negative, or RTF claim",
        "dataset_a_used": False,
        "group_id": group_id,
        "device": str(resolved_device),
        "seed": seed,
        "updates": updates,
        "learning_rate": learning_rate,
        "trainable_parameters": model.trainable_parameter_count(),
        "loss_first": history[0],
        "loss_last": history[-1],
        "runtime_sec": runtime_sec,
        "presence": {
            "initial_main": initial_main,
            "initial_view2": initial_view2,
            "final_main": final_main,
            "final_view2": final_view2,
            "swapped_main": swapped_main,
            "swap_indices": [swap_index, absent_index],
            "swap_equivariance_error": swap_error,
        },
        "audit": audit,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--qwen-config", default=r"E:\hf_cache\Qwen3-ASR-1.7B")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DACF_QWEN_CONFIG"] = str(Path(args.qwen_config).resolve())
    result = run_g1(
        args.feature_manifest,
        group_id=args.group_id,
        updates=args.updates,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["verdict"] == "conditional-GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["_read_rows", "run_g1"]
