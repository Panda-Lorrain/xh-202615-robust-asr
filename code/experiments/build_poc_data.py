#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""POC 数据装配: TTS 干净音频 + MUSAN + MIT RIR → build_tse_pairs 三件套。

读 CosyVoice 合成的 target/interferer manifest (干净音频) + MUSAN noise,
调 build_tse_pairs (含 RIR 卷积) 合成带难度(房间混响+双人重叠+噪声+小声/快语速)
的 TSE 三件套, speaker-disjoint split (train/val)。

⚠️ 在 Windows .venv_tse 跑 (不是 WSL):
  E:/midea_target_asr/code/.venv_tse/Scripts/python.exe E:/midea_target_asr/code/build_poc_data.py

红线: 不碰 Dataset-A; speaker-disjoint train/val (说话人不交叉)。
"""
import argparse
import json
import os
import random
import sys


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_win_path(p):
    """TTS 写的 /mnt/e/... → E:/..."""
    if p.startswith("/mnt/e/"):
        return "E:/" + p[len("/mnt/e/"):]
    return p


def main():
    sys.path.insert(0, "E:/midea_target_asr/code")
    from tse_data_aug import build_tse_pairs

    p = argparse.ArgumentParser(description="装配 POC 三件套 (TTS+RIR+重叠+噪声)")
    p.add_argument("--tts-pool", default="E:/midea_target_asr/code/_tts_pool")
    p.add_argument("--noise-manifest",
                   default="E:/midea_target_asr/code/_aug_manifests_poc1k/noise.jsonl")
    p.add_argument("--rir-root", default="E:/midea_datasets/mit_ir_survey/Audio")
    p.add_argument("--out-root", default="E:/midea_target_asr/code/_home_tse_poc")
    p.add_argument("--n-val-spk", type=int, default=20)
    p.add_argument("--n-per-target", type=int, default=1, help="每 target 生成几 triple")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # 1) 读 TTS manifest + 路径转换 (/mnt/e/ → E:/)
    targets = read_jsonl(f"{args.tts_pool}/target_manifest.jsonl")
    interferers = read_jsonl(f"{args.tts_pool}/interferer_manifest.jsonl")
    for r in targets:
        r["wav"] = to_win_path(r["wav"])
    for r in interferers:
        r["wav"] = to_win_path(r["wav"])
    print(f"target={len(targets)} interferer={len(interferers)}")

    # 2) MUSAN noise
    noises = read_jsonl(args.noise_manifest)
    print(f"noise={len(noises)}")

    # 3) speaker-disjoint split (按 spk)
    spks = sorted({r["spk"] for r in targets})
    rng = random.Random(args.seed)
    rng.shuffle(spks)
    val_spks = set(spks[: args.n_val_spk])
    train_spks = set(spks[args.n_val_spk:])
    train_targets = [r for r in targets if r["spk"] in train_spks]
    val_targets = [r for r in targets if r["spk"] in val_spks]
    print(f"split: train_spk={len(train_spks)} val_spk={len(val_spks)} "
          f"train_target={len(train_targets)} val_target={len(val_targets)}")

    # 4) build_tse_pairs (train + val, 含 RIR + 重叠 + 噪声)
    for split, titems in [("train", train_targets), ("val", val_targets)]:
        out_dir = f"{args.out_root}/{split}"
        cnt = build_tse_pairs(
            target_items=titems,
            interferer_items=interferers,
            noise_items=noises,
            out_dir=out_dir,
            n_per_target=args.n_per_target,
            seed=args.seed,
            augmentation_profile="balanced",
            rir_root=args.rir_root,
        )
        print(f"[{split}] {cnt} triples -> {out_dir}")


if __name__ == "__main__":
    main()
