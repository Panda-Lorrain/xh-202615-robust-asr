#!/usr/bin/env python3
"""Train a WeSep pBSRNN with frozen, precomputed speaker embeddings.

This is the phase-2 separation POC.  It deliberately trains only the target
speech extractor.  ASR is evaluated downstream with the unchanged Qwen3-ASR
pipeline so that separation gains cannot be hidden by a temporary CTC head.
"""

import argparse
import importlib.util
import json
import random
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from tse_train import si_snr


SR = 16000


def load_bsrnn_class(wesep_root: str):
    """Load only WeSep's BSRNN core without importing optional CLI packages."""
    root = Path(wesep_root).resolve()
    source = root / "wesep" / "models" / "bsrnn.py"
    if not source.exists():
        raise FileNotFoundError(
            f"WeSep BSRNN not found at {source}; clone wenet-e2e/wesep"
        )

    for name, package_path in (
        ("wesep", root / "wesep"),
        ("wesep.models", root / "wesep" / "models"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(package_path)]
        sys.modules[name] = module

    # bsrnn.py imports this symbol at module load time, although it is unused
    # when joint_training=False.
    wespeaker = types.ModuleType("wespeaker")
    wespeaker.__path__ = []
    wespeaker_models = types.ModuleType("wespeaker.models")
    wespeaker_models.__path__ = []
    speaker_model = types.ModuleType("wespeaker.models.speaker_model")

    def unavailable_speaker_model(_):
        raise RuntimeError(
            "This POC uses joint_training=False and external embeddings"
        )

    speaker_model.get_speaker_model = unavailable_speaker_model
    sys.modules["wespeaker"] = wespeaker
    sys.modules["wespeaker.models"] = wespeaker_models
    sys.modules["wespeaker.models.speaker_model"] = speaker_model

    spec = importlib.util.spec_from_file_location(
        "wesep.models.bsrnn", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BSRNN


class WeSepDataset(Dataset):
    def __init__(self, manifest_path: str):
        with open(manifest_path, encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _load_wav(path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.float()

    def __getitem__(self, index: int):
        row = self.rows[index]
        if "enrollment_embedding" not in row:
            raise KeyError(
                f"{row.get('id', index)} lacks enrollment_embedding; "
                "run tse_campp_embeddings.py first"
            )
        mix = self._load_wav(row["recognition_audio"])
        clean = self._load_wav(row["clean_target_audio"])
        if mix.numel() != clean.numel():
            raise ValueError(
                f"{row.get('id', index)} mix/clean length mismatch: "
                f"{mix.numel()} != {clean.numel()}"
            )
        embedding = torch.from_numpy(
            np.load(row["enrollment_embedding"]).astype(np.float32)
        )
        if embedding.ndim != 1:
            raise ValueError(
                f"{row.get('id', index)} embedding must be one-dimensional"
            )
        embedding = torch.nn.functional.normalize(embedding, dim=0)
        return mix, clean, embedding, row.get("id", str(index))


def validate_speaker_disjoint(
    train_rows: List[dict], val_rows: List[dict]
) -> None:
    """Reject speaker or source-audio leakage across train/validation."""
    speaker_keys = ("target_spk", "interferer_spk", "enrollment_spk")
    source_keys = ("target_src", "interferer_src", "enrollment_src")

    def values(rows, keys):
        return {
            str(row[key])
            for row in rows
            for key in keys
            if row.get(key)
        }

    train_speakers = values(train_rows, speaker_keys)
    val_speakers = values(val_rows, speaker_keys)
    if train_speakers and val_speakers:
        overlap = train_speakers & val_speakers
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"train/validation speakers overlap: {preview}"
            )
    source_overlap = values(train_rows, source_keys) & values(
        val_rows, source_keys
    )
    if source_overlap:
        preview = ", ".join(sorted(source_overlap)[:3])
        raise ValueError(
            f"train/validation source audio overlap: {preview}"
        )


def require_batch_size_one(batch_size: int) -> None:
    """Keep bidirectional BSRNN independent of right-padding context."""
    if batch_size != 1:
        raise ValueError(
            "pBSRNN variable-length training/inference requires "
            "--batch-size=1; right padding changes bidirectional LSTM output"
        )


def collate_batch(batch):
    mixes, cleans, embeddings, ids = zip(*batch)
    lengths = torch.tensor([wav.numel() for wav in mixes], dtype=torch.long)
    max_length = int(lengths.max())
    mix_batch = torch.zeros(len(batch), max_length)
    clean_batch = torch.zeros_like(mix_batch)
    for index, (mix, clean) in enumerate(zip(mixes, cleans)):
        mix_batch[index, : mix.numel()] = mix
        clean_batch[index, : clean.numel()] = clean
    return (
        mix_batch,
        clean_batch,
        torch.stack(embeddings),
        lengths,
        list(ids),
    )


def evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    totals = {"count": 0, "mix_si_snr": 0.0, "est_si_snr": 0.0}
    improvements: List[float] = []
    with torch.no_grad():
        for mix, clean, embedding, lengths, _ in loader:
            mix = mix.to(device)
            clean = clean.to(device)
            embedding = embedding.to(device)
            lengths = lengths.to(device)
            estimate, _ = model(mix, embedding)
            batch_size = mix.size(0)
            totals["count"] += batch_size
            mix_scores = si_snr(mix, clean, lengths)
            est_scores = si_snr(estimate, clean, lengths)
            totals["mix_si_snr"] += float(mix_scores.sum().cpu())
            totals["est_si_snr"] += float(est_scores.sum().cpu())
            improvements.extend(
                (est_scores - mix_scores).detach().cpu().tolist()
            )
    count = totals.pop("count")
    if count == 0:
        raise ValueError("validation loader is empty")
    mix_score = totals["mix_si_snr"] / count
    est_score = totals["est_si_snr"] / count
    return {
        "mix_si_snr": mix_score,
        "est_si_snr": est_score,
        "si_snri": est_score - mix_score,
        "si_snri_median": float(np.median(improvements)),
        "si_snri_p10": float(np.percentile(improvements, 10)),
        "nondegraded_rate": float(
            np.mean(np.asarray(improvements) > 0.0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wesep-root", default="code/WeSep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--num-repeat", type=int, default=6)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_batch_size_one(args.batch_size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    train_set = WeSepDataset(args.train_manifest)
    val_set = WeSepDataset(args.val_manifest)
    validate_speaker_disjoint(train_set.rows, val_set.rows)
    embedding_dim = int(train_set[0][2].numel())
    if int(val_set[0][2].numel()) != embedding_dim:
        raise ValueError("train/validation embedding dimensions differ")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    BSRNN = load_bsrnn_class(args.wesep_root)
    model = BSRNN(
        spk_emb_dim=embedding_dim,
        sr=SR,
        win=512,
        stride=128,
        feature_dim=args.feature_dim,
        num_repeat=args.num_repeat,
        use_spk_transform=False,
        spk_fuse_type="multiply",
        multi_fuse=True,
        joint_training=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_si_snri = float("-inf")
    step = 0
    stop = False
    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for mix, clean, embedding, lengths, _ in train_loader:
            mix = mix.to(device)
            clean = clean.to(device)
            embedding = embedding.to(device)
            lengths = lengths.to(device)
            estimate, _ = model(mix, embedding)
            loss = -si_snr(estimate, clean, lengths).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_interval == 0:
                print(
                    f"[train] epoch={epoch} step={step} "
                    f"loss={float(loss.detach().cpu()):.4f}"
                )
            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break

        metrics = evaluate(model, val_loader, device)
        metrics.update({"epoch": epoch, "step": step})
        history.append(metrics)
        print(
            f"[val] epoch={epoch} SI-SNR={metrics['est_si_snr']:.3f} "
            f"SI-SNRi={metrics['si_snri']:+.3f} "
            f"median={metrics['si_snri_median']:+.3f} "
            f"p10={metrics['si_snri_p10']:+.3f} "
            f"nondegraded={metrics['nondegraded_rate']:.1%}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "embedding_dim": embedding_dim,
            "metrics": metrics,
            "wesep_source": str(
                Path(args.wesep_root, "wesep", "models", "bsrnn.py").resolve()
            ),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["si_snri"] > best_si_snri:
            best_si_snri = metrics["si_snri"]
            torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if stop:
            break

    print(
        f"[done] steps={step} best_val_si_snri={best_si_snri:+.3f} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
