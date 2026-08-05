#!/usr/bin/env python3
"""D8: strict real-local Qwen batch=1 DACF latent bridge smoke.

This script is deliberately a contract smoke, not a training job.  It loads
only the already-present local Qwen checkpoint and the first present row of a
non-Dataset-A AISHELL counterfactual manifest, then performs:

1. a frozen-thinker baseline ``forward(labels)``;
2. a zero-gate DACF bridge forward with an equivalence check;
3. one pure ``asr_loss.backward()`` at zero gate;
4. one pure ``asr_loss.backward()`` after setting the bridge gate to 0.01.

No optimizer, checkpoint, Dataset-A read, network access, or submission-chain
integration is allowed here.  Every failure is written to the one permitted
result JSON as ``implementation-NO-GO``.  A passing result is only a
``conditional-GO`` for this local interface smoke; the research direction
remains ``direction-unresolved`` until held-out training/evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


# Prevent imports below from creating any extra repository files.
sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(r"E:\hf_cache\Qwen3-ASR-1.7B")
MANIFEST = REPO_ROOT / "code" / "runs" / "dacf_counterfactual_poc_20260806" / "train" / "manifest.jsonl"
RESULT_PATH = REPO_ROOT / "code" / "runs" / "dacf_qwen_real_smoke_20260806" / "result.json"
EXPERIMENTS_DIR = REPO_ROOT / "code" / "experiments"

BANNED_PATH_MARKERS = ("dataseta", "dataset_a", "dataset-a")
SAMPLE_REQUIRED_FIELDS = {
    "id",
    "recognition_audio",
    "enrollment_audio",
    "target_present",
    "clean_target_is_empty",
    "target_src",
    "source_corpus",
    "split",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _empty_result() -> dict[str, Any]:
    return {
        "schema_version": "dacf-qwen-real-smoke-v0.1",
        "role": "D8",
        "created_at_utc": _now(),
        "verdict": "implementation-NO-GO",
        "direction": "direction-unresolved",
        "status": "not-run",
        "contract": {
            "offline_only": True,
            "local_files_only": True,
            "dataset_a_read": False,
            "optimizer_used": False,
            "optimizer_step_used": False,
            "training_run": False,
            "batch_size": 1,
            "pure_asr_backward": True,
            "auxiliary_loss_weights": {
                "presence": 0.0,
                "activity": 0.0,
            },
        },
        "paths": {
            "model_dir": str(MODEL_DIR),
            "manifest": str(MANIFEST),
            "result": str(RESULT_PATH),
        },
        "sample": {},
        "runtime": {},
        "audio_layer": {},
        "checks": {},
        "gradients": {},
        "error": None,
    }


def _assert_no_dataset_a(value: Any, label: str) -> None:
    """Reject any path/serialized record that mentions Dataset-A."""

    text = str(value).replace("\\", "/").lower()
    if any(marker in text for marker in BANNED_PATH_MARKERS):
        raise RuntimeError(f"Dataset-A path/content rejected at {label}: {value}")


def _assert_no_dataset_a_values(value: Any, label: str) -> None:
    """Inspect record values recursively without treating metadata keys as paths."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_dataset_a_values(item, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_dataset_a_values(item, f"{label}[{index}]")
    elif isinstance(value, str):
        _assert_no_dataset_a(value, label)


def _resolve_repo_path(raw: Any, label: str) -> Path:
    if not isinstance(raw, (str, Path)) or not str(raw).strip():
        raise ValueError(f"{label} must be a non-empty path string")
    _assert_no_dataset_a(raw, label)
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve(strict=False)
    _assert_no_dataset_a(path, label)
    return path


def _load_first_present_row() -> tuple[dict[str, Any], Path]:
    manifest = _resolve_repo_path(MANIFEST, "manifest")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            _assert_no_dataset_a_values(row, f"manifest line {line_number}")
            if bool(row.get("target_present")) and not bool(row.get("clean_target_is_empty")):
                rows.append(row)
                break
    if not rows:
        raise ValueError("no present non-empty row found in the manifest")
    row = rows[0]
    missing = sorted(SAMPLE_REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"present row is missing fields: {missing}")
    if row.get("dataset_a_used") is not False:
        raise RuntimeError(f"manifest row dataset_a_used is not false: {row.get('id')}")
    if row.get("dataset_a_policy") != "forbidden":
        raise RuntimeError(f"manifest row dataset_a_policy is not forbidden: {row.get('id')}")
    if row.get("source_corpus") != "AISHELL-1":
        raise RuntimeError(f"unexpected source corpus: {row.get('source_corpus')}")
    if row.get("split") != "train":
        raise RuntimeError(f"unexpected split for smoke row: {row.get('split')}")
    target_src = str(row.get("target_src", "")).replace("\\", "/").lower()
    if "/wav/train/" not in target_src:
        raise RuntimeError(f"smoke row target source is not AISHELL train: {row.get('target_src')}")
    return row, manifest


def _read_audio(path: Path, label: str):
    import numpy as np
    import soundfile as sf

    _assert_no_dataset_a(path, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} audio not found: {path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    if audio.ndim != 1 or audio.size < 2:
        raise ValueError(f"{label} must be non-empty mono audio, got {audio.shape}")
    if int(sample_rate) != 16000:
        raise ValueError(f"{label} sample rate must be 16000, got {sample_rate}")
    if not np.isfinite(audio).all():
        raise ValueError(f"{label} contains non-finite samples")
    return audio, int(sample_rate)


def _build_qwen_inputs(processor: Any, tokenizer: Any, audio: Any, reference: str):
    import torch

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    eos = tokenizer.eos_token or "<|endoftext|>"
    inputs = processor(
        text=prompt + "language Chinese<asr_text>" + reference + eos,
        audio=audio,
        return_tensors="pt",
        padding=False,
    )
    tensors = {
        name: value
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    if "input_ids" not in tensors or "input_features" not in tensors:
        raise RuntimeError(f"processor did not return input_ids/input_features: {sorted(tensors)}")
    asr_text_token = tokenizer.convert_tokens_to_ids("<asr_text>")
    if asr_text_token is None or asr_text_token < 0:
        raise RuntimeError("Qwen tokenizer does not expose <asr_text>")
    labels = tensors["input_ids"].clone()
    positions = (labels == asr_text_token).nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise RuntimeError("processor output does not contain <asr_text>")
    first_position = int(positions[0, -1].item())
    labels[:, : first_position + 1] = -100
    tensors["labels"] = labels
    return tensors


def _output_field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _hidden_from_layer_output(output: Any):
    import torch

    if isinstance(output, torch.Tensor):
        return output, "tensor"
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0], "sequence"
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value, name
    raise TypeError(f"cannot inspect audio layer output type {type(output).__name__}")


def _gradient_stats(module: Any, *, exclude_names: tuple[str, ...] = ()) -> dict[str, Any]:
    import torch

    total_norm = 0.0
    nonzero = 0
    finite = True
    parameter_count = 0
    by_name: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or any(name.endswith(item) for item in exclude_names):
            continue
        parameter_count += 1
        grad = parameter.grad
        if grad is None:
            continue
        grad_float = grad.detach().float()
        is_finite = bool(torch.isfinite(grad_float).all().item())
        finite = finite and is_finite
        norm = float(grad_float.norm().item()) if is_finite else float("nan")
        by_name[name] = norm
        total_norm += norm if is_finite else 0.0
        if is_finite and bool(grad_float.abs().sum().item() > 0.0):
            nonzero += 1
    return {
        "trainable_parameter_tensors": parameter_count,
        "gradient_tensors": len(by_name),
        "nonzero_gradient_tensors": nonzero,
        "finite": finite,
        "sum_of_l2_norms": total_norm,
        "by_name": by_name,
    }


def _clear_grads(module: Any) -> None:
    for parameter in module.parameters():
        parameter.grad = None


def _frozen_gradient_leaks(thinker: Any) -> list[str]:
    return [
        name
        for name, parameter in thinker.named_parameters()
        if parameter.grad is not None
    ]


def _describe_audio_layout(shape: tuple[int, ...], hidden_size: int) -> str:
    if len(shape) == 2 and shape[-1] == hidden_size:
        return "utterance_[T,H]"
    if len(shape) == 3 and shape[-1] == hidden_size:
        if shape[0] == 1:
            return "utterance_batch_first_[1,T,H]"
        if shape[1] == 1:
            return "utterance_time_first_[T,1,H]"
        return "batched_or_ambiguous_[B,T,H-or-T,B,H]"
    return "unsupported"


def _record_exception(result: dict[str, Any], exc: BaseException) -> None:
    lines = traceback.format_exc().strip().splitlines()
    result["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": lines[-12:],
    }
    result["verdict"] = "implementation-NO-GO"
    result["status"] = "failed"


def _run(result: dict[str, Any]) -> None:
    # Offline flags are set before importing Hugging Face/Qwen modules.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    import numpy as np
    import torch
    from transformers import AutoProcessor
    from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this D8 smoke requires the local CUDA Qwen environment")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"local Qwen model directory not found: {MODEL_DIR}")
    _assert_no_dataset_a(MODEL_DIR, "model_dir")
    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"local Qwen config missing: {MODEL_DIR / 'config.json'}")
    if not any(MODEL_DIR.glob("*.safetensors")):
        raise FileNotFoundError(f"local Qwen safetensors missing: {MODEL_DIR}")

    row, manifest = _load_first_present_row()
    recognition_path = _resolve_repo_path(row["recognition_audio"], "recognition_audio")
    enrollment_path = _resolve_repo_path(row["enrollment_audio"], "enrollment_audio")
    target_source_path = _resolve_repo_path(row["target_src"], "target_src")
    result["paths"].update(
        {
            "model_dir": str(MODEL_DIR),
            "manifest": str(manifest),
            "recognition_audio": str(recognition_path),
            "enrollment_audio": str(enrollment_path),
            "target_source": str(target_source_path),
            "result": str(RESULT_PATH),
        }
    )
    result["sample"] = {
        "id": row["id"],
        "query_role": row.get("query_role"),
        "query_role_id": row.get("query_role_id"),
        "target_present": bool(row["target_present"]),
        "clean_target_is_empty": bool(row["clean_target_is_empty"]),
        "query_speaker_id": row.get("query_speaker_id"),
        "target_speaker_id": row.get("target_spk"),
        "source_corpus": row.get("source_corpus"),
        "split": row.get("split"),
        "reference_chars": len(str(row.get("target_transcript", row.get("ref", "")))),
        "dataset_a_used": row.get("dataset_a_used"),
    }

    recognition_audio, recognition_sr = _read_audio(recognition_path, "recognition")
    enrollment_audio, enrollment_sr = _read_audio(enrollment_path, "enrollment")
    result["sample"].update(
        {
            "recognition_sample_rate": recognition_sr,
            "enrollment_sample_rate": enrollment_sr,
            "recognition_samples": int(recognition_audio.size),
            "enrollment_samples": int(enrollment_audio.size),
        }
    )

    device = torch.device("cuda:0")
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        fix_mistral_regex=True,
        local_files_only=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    thinker = model.thinker
    thinker.eval()
    load_seconds = time.perf_counter() - load_start
    thinker_devices = sorted({str(parameter.device) for parameter in thinker.parameters()})
    thinker_dtypes = sorted({str(parameter.dtype) for parameter in thinker.parameters()})
    result["runtime"].update(
        {
            "model_load_seconds": load_seconds,
            "cuda_available": True,
            "device": str(device),
            "qwen_autocast": "cuda:bfloat16",
            "thinker_parameter_devices": thinker_devices,
            "thinker_parameter_dtypes": thinker_dtypes,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    )
    if thinker_devices != [str(device)]:
        raise RuntimeError(f"thinker is not fully on cuda:0: {thinker_devices}")

    processor_inputs = _build_qwen_inputs(
        processor,
        processor.tokenizer,
        recognition_audio,
        str(row.get("ref", row.get("target_transcript", ""))),
    )
    processor_inputs = {
        name: value.to(device=device, non_blocking=True)
        for name, value in processor_inputs.items()
    }
    mixture = torch.from_numpy(np.ascontiguousarray(recognition_audio)).to(device=device)
    enrollment = torch.from_numpy(np.ascontiguousarray(enrollment_audio)).to(device=device)
    mixture = mixture.unsqueeze(0)
    enrollment = enrollment.unsqueeze(0)
    result["runtime"].update(
        {
            "processor_tensor_shapes": {
                name: list(value.shape) for name, value in processor_inputs.items()
            },
            "processor_tensor_dtypes": {
                name: str(value.dtype) for name, value in processor_inputs.items()
            },
            "waveform_dtype": str(mixture.dtype),
            "waveform_device": str(mixture.device),
        }
    )

    # A clean baseline is established before the bridge mutates the live
    # thinker.  It is used only for same-process equivalence, never for tuning.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            baseline_output = thinker(**processor_inputs)
    baseline_logits = _output_field(baseline_output, "logits")
    baseline_loss = _output_field(baseline_output, "loss")
    if not isinstance(baseline_logits, torch.Tensor) or not isinstance(baseline_loss, torch.Tensor):
        raise RuntimeError("baseline thinker.forward(labels) did not expose tensor logits and loss")
    baseline_logits_cpu = baseline_logits.detach().float().cpu()
    baseline_loss_value = float(baseline_loss.detach().float().cpu().item())

    if str(EXPERIMENTS_DIR) not in sys.path:
        sys.path.insert(0, str(EXPERIMENTS_DIR))
    from dacf_frontend import DACFFrontend
    from dacf_qwen_latent import DACFBridgeConfig, FrozenQwenDACFLatent

    dacf = DACFFrontend(
        n_fft=400,
        hop_length=160,
        d_model=32,
        n_heads=4,
        vocab_size=64,
    ).to(device=device, dtype=torch.float32)

    class _Float32DACF(torch.nn.Module):
        """Keep STFT/ISTFT and compact front-end arithmetic in float32."""

        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base
            self.d_model = int(getattr(base, "d_model"))

        def forward(self, mixture_waveform: torch.Tensor, enrollment_waveform: torch.Tensor):
            with torch.autocast(device_type="cuda", enabled=False):
                return self.base(mixture_waveform.float(), enrollment_waveform.float())

    dacf_for_bridge = _Float32DACF(dacf)
    bridge_config = DACFBridgeConfig(
        layers=1,
        rank=4,
        dacf_dim=32,
        presence_loss_weight=0.0,
        activity_loss_weight=0.0,
        audio_layout="auto",
    )
    wrapper = FrozenQwenDACFLatent(thinker, dacf_for_bridge, bridge_config)
    wrapper.eval()

    audio_layer_records: list[dict[str, Any]] = []
    first_audio_layer = thinker.audio_tower.layers[0]

    def _audio_layer_hook(_module: Any, _args: Any, output: Any) -> None:
        hidden, output_kind = _hidden_from_layer_output(output)
        audio_layer_records.append(
            {
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": str(hidden.device),
                "output_kind": output_kind,
            }
        )

    hook_handle = first_audio_layer.register_forward_hook(_audio_layer_hook)
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            zero_output = wrapper(
                mixture,
                enrollment,
                **processor_inputs,
            )
        zero_logits = _output_field(zero_output, "logits")
        zero_asr_loss = _output_field(zero_output, "asr_loss")
        if not isinstance(zero_logits, torch.Tensor) or not isinstance(zero_asr_loss, torch.Tensor):
            raise RuntimeError("zero-gate bridge forward did not expose logits and asr_loss")
        zero_diff = (zero_logits.detach().float().cpu() - baseline_logits_cpu).abs()
        zero_loss_diff = abs(float(zero_asr_loss.detach().float().cpu().item()) - baseline_loss_value)
        result["checks"].update(
            {
                "baseline_forward_labels_ok": True,
                "zero_gate_forward_labels_ok": True,
                "zero_gate_logits_exact_equal": bool(torch.equal(zero_logits.detach().float().cpu(), baseline_logits_cpu)),
                "zero_gate_logits_max_abs_diff": float(zero_diff.max().item()),
                "zero_gate_loss_exact_equal": float(zero_loss_diff) == 0.0,
                "zero_gate_loss_abs_diff": float(zero_loss_diff),
                "zero_gate_equivalent": bool(float(zero_diff.max().item()) == 0.0 and zero_loss_diff == 0.0),
            }
        )

        # First pure-ASR backward: at exact zero the scalar gate is the only
        # parameter that can receive a nonzero derivative by construction.
        _clear_grads(wrapper.bridges)
        _clear_grads(dacf)
        _clear_grads(thinker)
        zero_asr_loss.backward()
        zero_gate_stats = _gradient_stats(wrapper.bridges, exclude_names=("down.weight", "up.weight", "norm.weight", "norm.bias"))
        zero_bridge_stats = _gradient_stats(wrapper.bridges)
        zero_dacf_stats = _gradient_stats(dacf)
        zero_frozen_leaks = _frozen_gradient_leaks(thinker)
        result["gradients"]["zero_gate_first_pure_asr"] = {
            "gate": zero_gate_stats,
            "bridge_all": zero_bridge_stats,
            "dacf": zero_dacf_stats,
            "frozen_thinker_gradient_leaks": zero_frozen_leaks,
        }

        # Second pure-ASR backward: manually open the gate a little.  This is
        # not an optimizer step and exists only to test that ASR supervision
        # reaches both DACF and the latent bridge without auxiliary losses.
        _clear_grads(wrapper.bridges)
        _clear_grads(dacf)
        _clear_grads(thinker)
        with torch.no_grad():
            for bridge in wrapper.bridges:
                bridge.gate.fill_(0.01)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            small_output = wrapper(
                mixture,
                enrollment,
                **processor_inputs,
            )
        small_asr_loss = _output_field(small_output, "asr_loss")
        if not isinstance(small_asr_loss, torch.Tensor):
            raise RuntimeError("small-gate bridge forward did not expose asr_loss")
        small_asr_loss.backward()
        small_gate_stats = _gradient_stats(wrapper.bridges, exclude_names=("down.weight", "up.weight", "norm.weight", "norm.bias"))
        small_bridge_stats = _gradient_stats(wrapper.bridges)
        small_dacf_stats = _gradient_stats(dacf)
        small_frozen_leaks = _frozen_gradient_leaks(thinker)
        result["gradients"]["small_gate_0.01_pure_asr"] = {
            "gate": small_gate_stats,
            "bridge_all": small_bridge_stats,
            "dacf": small_dacf_stats,
            "frozen_thinker_gradient_leaks": small_frozen_leaks,
        }
        try:
            wrapper.assert_gradient_contract(small_asr_loss)
            result["checks"]["strict_gradient_contract"] = True
        except Exception as contract_error:  # noqa: BLE001 - serialize as a gate failure
            result["checks"]["strict_gradient_contract"] = False
            result["checks"]["strict_gradient_contract_error"] = str(contract_error)

        result["checks"].update(
            {
                "zero_gate_first_gate_gradient_nonzero": zero_gate_stats["nonzero_gradient_tensors"] > 0,
                "zero_gate_first_dacf_gradient_zero": zero_dacf_stats["nonzero_gradient_tensors"] == 0,
                "small_gate_asr_reaches_gate": small_gate_stats["nonzero_gradient_tensors"] > 0,
                "small_gate_asr_reaches_bridge_non_gate": any(
                    name not in {"0.gate"} and value > 0.0
                    for name, value in small_bridge_stats["by_name"].items()
                ),
                "small_gate_asr_reaches_dacf": small_dacf_stats["nonzero_gradient_tensors"] > 0,
                "frozen_thinker_requires_grad_false": all(
                    not parameter.requires_grad for parameter in thinker.parameters()
                ),
                "zero_backward_no_frozen_thinker_grad": not zero_frozen_leaks,
                "small_backward_no_frozen_thinker_grad": not small_frozen_leaks,
                "auxiliary_loss_weights_zero": (
                    bridge_config.presence_loss_weight == 0.0
                    and bridge_config.activity_loss_weight == 0.0
                ),
            }
        )
    finally:
        hook_handle.remove()

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    result["runtime"].update(
        {
            "smoke_wall_seconds_baseline_zero_small": wall_seconds,
            "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_cuda_memory_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
            "peak_cuda_memory_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024**2)),
        }
    )
    layer_base = getattr(first_audio_layer, "base", first_audio_layer)
    result["audio_layer"] = {
        "wrapper_class": type(first_audio_layer).__name__,
        "base_class": type(layer_base).__name__,
        "hidden_size": int(getattr(wrapper, "hidden_size", 0)),
        "layer_index": 0,
        "hook_call_count_zero_plus_small": len(audio_layer_records),
        "hook_records": audio_layer_records,
        "observed_layouts": sorted(
            {
                _describe_audio_layout(tuple(item["shape"]), int(getattr(wrapper, "hidden_size", 0)))
                for item in audio_layer_records
            }
        ),
    }

    checks = result["checks"]
    required_checks = (
        "baseline_forward_labels_ok",
        "zero_gate_forward_labels_ok",
        "zero_gate_equivalent",
        "zero_gate_first_gate_gradient_nonzero",
        "zero_gate_first_dacf_gradient_zero",
        "small_gate_asr_reaches_gate",
        "small_gate_asr_reaches_bridge_non_gate",
        "small_gate_asr_reaches_dacf",
        "frozen_thinker_requires_grad_false",
        "zero_backward_no_frozen_thinker_grad",
        "small_backward_no_frozen_thinker_grad",
        "auxiliary_loss_weights_zero",
        "strict_gradient_contract",
    )
    all_checks_pass = all(bool(checks.get(name)) for name in required_checks)
    result["checks"]["required_contract_checks"] = list(required_checks)
    result["checks"]["all_required_contract_checks_pass"] = all_checks_pass
    if len(audio_layer_records) < 2:
        # The hook is expected once in each of the zero/small forwards.
        all_checks_pass = False
        result["checks"]["audio_layer_hook_two_forwards"] = False
    else:
        result["checks"]["audio_layer_hook_two_forwards"] = True
    if len(audio_layer_records) != 2:
        all_checks_pass = False
        result["checks"]["audio_layer_call_count_per_forward_is_one"] = False
    else:
        result["checks"]["audio_layer_call_count_per_forward_is_one"] = True
    result["checks"]["all_required_contract_checks_pass"] = all_checks_pass
    result["verdict"] = "conditional-GO" if all_checks_pass else "implementation-NO-GO"
    result["status"] = "passed-contract-smoke" if all_checks_pass else "failed-contract-check"
    result["runtime"]["baseline_loss"] = baseline_loss_value
    result["runtime"]["zero_gate_loss"] = float(zero_asr_loss.detach().float().cpu().item())
    result["runtime"]["small_gate_loss"] = float(small_asr_loss.detach().float().cpu().item())
    result["runtime"]["small_gate_value"] = 0.01


def _write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    result = _empty_result()
    started = time.perf_counter()
    try:
        _run(result)
    except BaseException as exc:  # noqa: BLE001 - result JSON is the failure contract
        _record_exception(result, exc)
    result["runtime"]["process_wall_seconds_including_failure"] = time.perf_counter() - started
    _write_result(result)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
