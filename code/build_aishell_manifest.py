#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aishell-1 → data_aug_recipe.build_pairs 输入清单装配器

Aishell-1 目录结构 (解压 data_aishell.tgz 后):
    data_aishell/wav/<speaker_id>/<utt_id>.wav     # 16k mono, 中文朗读
    data_aishell/transcript/aishell_transcript_v0.8.txt
        (行格式: "<utt_id> 而 对 楼市 销售 ..." 带空格分词)

本脚本:
  - 解析 transcript → {utt_id: 去空格文本}
  - 遍历 wav 目录 → [{wav, ref, spk}]
  - 按说话人切分: 一部分当 target, 一部分当 interferer
    (跨说话人切保证 target≠interferer, 同说话人 utterance 自重叠不出现)
  - 输出 3 个 jsonl: target / interferer / (noise 走 MUSAN 另建)

注意: Aishell 是新闻朗读 (非家居指令), 文本不是空调/灯/温度类。
      recipe 文档已说明: 当前骨架沿用原始 transcript 验证增广链工作;
      家居指令化替换/TTS 留给后续工作 (见 data_aug_recipe.HOME_CMD_TEMPLATES)。
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import random
from typing import List, Dict


def parse_transcript(tsv_path: str) -> Dict[str, str]:
    """解析 aishell_transcript_v0.8.txt → {utt_id: 去空格文本}。"""
    out = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            utt_id, text = parts
            # 去空格 (Aishell 用空格分词, 题目要纯中文)
            text = text.replace(" ", "").strip()
            if text:
                out[utt_id] = text
    return out


def collect_wav_items(wav_root: str, transcript: Dict[str, str]) -> List[Dict]:
    """遍历 wav_root/<split>/<spk>/<utt>.wav (train/dev/test 三 split) → [{wav, ref, spk, utt}]。

    Aishell-1 实际结构: wav/{train,dev,test}/SXXXX/BAC009SXXXXWYYYY.wav
    兼容 (无 split): wav/SXXXX/BAC009SXXXXWYYYY.wav
    """
    items = []
    # 优先三层 split 结构
    for wav_path in glob.glob(os.path.join(wav_root, "*", "*", "*.wav")):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        if utt_id not in transcript:
            continue
        spk = os.path.basename(os.path.dirname(wav_path))
        items.append({
            "wav": wav_path.replace("\\", "/"),
            "ref": transcript[utt_id],
            "spk": spk,
            "utt": utt_id,
            "split": os.path.basename(os.path.dirname(os.path.dirname(wav_path))),
        })
    if items:
        return items
    # 退化: 两层结构 (无 split)
    for wav_path in glob.glob(os.path.join(wav_root, "*", "*.wav")):
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        if utt_id not in transcript:
            continue
        spk = os.path.basename(os.path.dirname(wav_path))
        items.append({
            "wav": wav_path.replace("\\", "/"),
            "ref": transcript[utt_id],
            "spk": spk,
            "utt": utt_id,
            "split": "all",
        })
    return items


def split_target_interferer(items: List[Dict],
                            n_target_speakers: int = 10,
                            n_interferer_speakers: int = 10,
                            max_utts_per_target: int = 5,
                            max_utts_per_interferer: int = 5,
                            seed: int = 42) -> tuple:
    """按说话人切分: 选 n_target_speakers 当 target, n_interferer_speakers 当干扰。

    跨说话人切 → 保证 target 和 interferer 不是同一人 (避免自重叠泄漏)。
    """
    # 按说话人分组
    by_spk: Dict[str, List[Dict]] = {}
    for it in items:
        by_spk.setdefault(it["spk"], []).append(it)
    speakers = sorted(by_spk.keys())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    if len(speakers) < n_target_speakers + n_interferer_speakers:
        raise SystemExit(
            f"[split] 说话人不够: 共 {len(speakers)} 个, 需要 "
            f"{n_target_speakers + n_interferer_speakers} 个"
        )
    target_spks = speakers[:n_target_speakers]
    interf_spks = speakers[n_target_speakers:n_target_speakers + n_interferer_speakers]

    target_items = []
    for spk in target_spks:
        utts = by_spk[spk]
        rng.shuffle(utts)
        for it in utts[:max_utts_per_target]:
            target_items.append({"wav": it["wav"], "ref": it["ref"],
                                 "spk": it["spk"], "utt": it["utt"]})
    interferer_items = []
    for spk in interf_spks:
        utts = by_spk[spk]
        rng.shuffle(utts)
        for it in utts[:max_utts_per_interferer]:
            interferer_items.append({"wav": it["wav"]})  # ref 不需要
    return target_items, interferer_items


def collect_musan_noise(musan_root: str,
                        max_items: int = 20,
                        only_types=("noise",)) -> List[Dict]:
    """收集 MUSAN 噪声 wav → [{wav}]。

    MUSAN 结构: musan/{noise,music,speech}/<subdir>/*.wav (子目录 free-sound/sound-bible 等)。
    只取 noise (题目 babble 已由程序合成, 这里补 env 噪声多样性)。
    """
    out = []
    for t in only_types:
        # 递归两层: noise/<subdir>/*.wav (MUSAN 实际把 wav 放在子目录里)
        for wav_path in glob.glob(os.path.join(musan_root, t, "*", "*.wav")):
            out.append({"wav": wav_path.replace("\\", "/")})
            if len(out) >= max_items:
                return out
        # 兼容: 也有可能直接 noise/*.wav
        for wav_path in glob.glob(os.path.join(musan_root, t, "*.wav")):
            out.append({"wav": wav_path.replace("\\", "/")})
            if len(out) >= max_items:
                return out
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aishell-root", required=True,
                    help="data_aishell/ 根 (含 wav/ 和 transcript/)")
    ap.add_argument("--musan-root", default="",
                    help="musan/ 根 (含 noise/ music/ speech/); 留空则不输出 noise 清单")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-target-speakers", type=int, default=10)
    ap.add_argument("--n-interferer-speakers", type=int, default=10)
    ap.add_argument("--max-utts-per-target", type=int, default=5)
    ap.add_argument("--max-utts-per-interferer", type=int, default=5)
    ap.add_argument("--max-noise", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    transcript_path = os.path.join(args.aishell_root, "transcript",
                                    "aishell_transcript_v0.8.txt")
    wav_root = os.path.join(args.aishell_root, "wav")
    print(f"[build] 解析 transcript: {transcript_path}")
    transcript = parse_transcript(transcript_path)
    print(f"[build] transcript 条数: {len(transcript)}")
    print(f"[build] 收集 wav: {wav_root}")
    items = collect_wav_items(wav_root, transcript)
    print(f"[build] 总 wav 条数 (有 transcript): {len(items)}")

    target_items, interferer_items = split_target_interferer(
        items,
        n_target_speakers=args.n_target_speakers,
        n_interferer_speakers=args.n_interferer_speakers,
        max_utts_per_target=args.max_utts_per_target,
        max_utts_per_interferer=args.max_utts_per_interferer,
        seed=args.seed,
    )
    print(f"[build] target utts: {len(target_items)} "
          f"({args.n_target_speakers} 说话人 × ≤{args.max_utts_per_target})")
    print(f"[build] interferer utts: {len(interferer_items)} "
          f"({args.n_interferer_speakers} 说话人 × ≤{args.max_utts_per_interferer})")

    target_path = os.path.join(args.out_dir, "target.jsonl")
    interf_path = os.path.join(args.out_dir, "interferer.jsonl")
    with open(target_path, "w", encoding="utf-8") as f:
        for it in target_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(interf_path, "w", encoding="utf-8") as f:
        for it in interferer_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[build] 写: {target_path}")
    print(f"[build] 写: {interf_path}")

    if args.musan_root:
        noise_items = collect_musan_noise(args.musan_root, max_items=args.max_noise)
        noise_path = os.path.join(args.out_dir, "noise.jsonl")
        with open(noise_path, "w", encoding="utf-8") as f:
            for it in noise_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[build] MUSAN noise 条数: {len(noise_items)}")
        print(f"[build] 写: {noise_path}")
    print("[build] OK")


if __name__ == "__main__":
    main()
