#!/usr/bin/env python3
"""Export pBSRNN estimates and a manifest for downstream Qwen3-ASR CER."""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader

from tse_wesep_train import (
    SR,
    WeSepDataset,
    collate_batch,
    load_bsrnn_class,
    require_batch_size_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wesep-root", default="code/WeSep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--cap-rms-to-mixture",
        action="store_true",
        help="attenuate estimates whose RMS exceeds the online mixture RMS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_batch_size_one(args.batch_size)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint["args"]
    embedding_dim = int(checkpoint["embedding_dim"])
    BSRNN = load_bsrnn_class(args.wesep_root)
    model = BSRNN(
        spk_emb_dim=embedding_dim,
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
    model.to(device).eval()

    dataset = WeSepDataset(args.manifest, args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {
        str(row.get("id", index)): row
        for index, row in enumerate(dataset.rows)
    }
    output_manifest = output_dir / "manifest.jsonl"

    count = 0
    with output_manifest.open("w", encoding="utf-8") as handle:
        with torch.no_grad():
            for mix, _, embedding, lengths, ids in loader:
                estimate, _ = model(
                    mix.to(device), embedding.to(device)
                )
                estimate = estimate.cpu()
                for index, uid in enumerate(ids):
                    length = int(lengths[index])
                    output = estimate[index, :length]
                    gain_scale = 1.0
                    if args.cap_rms_to_mixture:
                        mixture_rms = mix[index, :length].square().mean().sqrt()
                        estimate_rms = output.square().mean().sqrt()
                        gain_scale = min(
                            1.0,
                            float(
                                mixture_rms
                                / estimate_rms.clamp_min(1e-8)
                            ),
                        )
                        output = output * gain_scale
                    output_audio = audio_dir / f"{uid}.wav"
                    torchaudio.save(
                        str(output_audio),
                        output.unsqueeze(0),
                        SR,
                    )
                    source = rows_by_id[uid]
                    row = {
                        "id": uid,
                        "recognition_audio": str(output_audio),
                        "ref": source.get("ref", ""),
                        "source_recognition_audio": source[
                            "recognition_audio"
                        ],
                        "clean_target_audio": source["clean_target_audio"],
                        "target_spk": source.get("target_spk"),
                        "output_gain_scale": gain_scale,
                    }
                    handle.write(
                        json.dumps(row, ensure_ascii=False) + "\n"
                    )
                    count += 1
    print(f"[export] rows={count} -> {output_manifest}")


if __name__ == "__main__":
    main()
