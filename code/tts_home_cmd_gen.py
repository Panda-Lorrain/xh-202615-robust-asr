#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CosyVoice2 批量合成家居指令(target) + 干扰闲话(interferer) 干净音频。

用 AISHELL-1 说话人做 zero-shot 克隆 prompt (声纹多样性, 覆盖题目 20+ 唤醒人声),
合成家居指令 (来自 generate_home_corpus 的 _home_cmd_corpus.jsonl) 作 target,
干扰闲话 (_chitchat_corpus.jsonl) 作 interferer。

⚠️ 在 WSL 内跑 (CosyVoice 装在 ~/cosyvoice/.venv_cosyvoice):
  wsl bash -lc "cd ~/cosyvoice/CosyVoice && source ../.venv_cosyvoice/bin/activate && \
    export PYTHONPATH=\$PWD/third_party/Matcha-TTS:\$PYTHONPATH && \
    python /mnt/e/midea_target_asr/code/tts_home_cmd_gen.py <args>"

红线: 不碰 Dataset-A; 仅 AISHELL-1 (Apache-2.0) + 人工语料。
产物固化: 合成 wav 作训练数据源; CosyVoice 不进提交推理链 (零外部运行时依赖)。
"""
import argparse
import json
import os
import random


def parse_aishell_transcript(transcript_file):
    """解析 aishell_transcript_v0.8.txt -> {utt_id: text(去空格)}."""
    utt2text = {}
    with open(transcript_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1].replace(" ", "")
    return utt2text


def collect_aishell_speakers(aishell_root, utt2text):
    """train 集 -> {spk_id: [(utt_id, wav_path, text), ...]}."""
    wav_root = os.path.join(aishell_root, "wav", "train")
    if not os.path.isdir(wav_root):
        raise FileNotFoundError(f"AISHELL train wav root 不存在: {wav_root}")
    spk2utts = {}
    for spk in sorted(os.listdir(wav_root)):
        spk_dir = os.path.join(wav_root, spk)
        if not os.path.isdir(spk_dir):
            continue
        for wav in sorted(os.listdir(spk_dir)):
            if not wav.endswith(".wav"):
                continue
            utt_id = os.path.splitext(wav)[0]
            if utt_id in utt2text:
                spk2utts.setdefault(spk, []).append(
                    (utt_id, os.path.join(spk_dir, wav), utt2text[utt_id])
                )
    return spk2utts


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(cv, out_wav, chunks, sr_out):
    import torch
    import torchaudio
    # 拼接所有 chunk: CosyVoice 非流式一般 yield 单段, 但长文本/流式可能多段;
    # 只取 [-1] 会丢前段音频 → cat 拼接防丢 (对抗性审查修正)。
    speech = torch.cat([c["tts_speech"].cpu() for c in chunks], dim=-1)
    torchaudio.save(out_wav, speech, sr_out)


def main():
    p = argparse.ArgumentParser(description="CosyVoice2 批量合成家居指令 + 闲话")
    p.add_argument("--model-dir",
                   default="/home/lorrain/cosyvoice/CosyVoice/pretrained_models/CosyVoice2-0.5B")
    p.add_argument("--aishell-root", default="/mnt/e/midea_datasets/data_aishell")
    p.add_argument("--home-corpus",
                   default="/mnt/e/midea_target_asr/code/_home_cmd_corpus.jsonl")
    p.add_argument("--chat-corpus",
                   default="/mnt/e/midea_target_asr/code/_chitchat_corpus.jsonl")
    p.add_argument("--out-dir", default="/mnt/e/midea_target_asr/code/_tts_pool")
    p.add_argument("--n-spk", type=int, default=20, help="克隆声纹数 (POC 20)")
    p.add_argument("--n-cmd-per-spk", type=int, default=50, help="每声纹 target 指令数")
    p.add_argument("--n-interferer", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir + "/target", exist_ok=True)
    os.makedirs(args.out_dir + "/interferer", exist_ok=True)

    # 1) AISHELL-1 说话人池
    print("=== 加载 AISHELL-1 说话人池 ===")
    utt2text = parse_aishell_transcript(
        os.path.join(args.aishell_root, "transcript", "aishell_transcript_v0.8.txt")
    )
    spk2utts = collect_aishell_speakers(args.aishell_root, utt2text)
    speakers = sorted(spk2utts.keys())
    if len(speakers) < args.n_spk:
        raise ValueError(f"AISHELL train 仅 {len(speakers)} 说话人, 要 {args.n_spk}")
    sel_spks = speakers[: args.n_spk]
    print(f"  选 {len(sel_spks)} 说话人: {sel_spks[:5]}...")

    # 2) 加载 CosyVoice2
    print("=== 加载 CosyVoice2 ===")
    from cosyvoice.cli.cosyvoice import AutoModel
    cv = AutoModel(model_dir=args.model_dir)
    sr_out = cv.sample_rate
    print(f"  output sample_rate={sr_out}")

    # 3) 注册 zero-shot 声纹 (每说话人 1 条 utt 做 prompt)
    print("=== 注册 zero-shot 声纹 ===")
    spk_prompt = {}
    for spk in sel_spks:
        prompt = rng.choice(spk2utts[spk])  # (utt_id, wav_path, text)
        spk_prompt[spk] = prompt
        try:
            cv.add_zero_shot_spk(prompt[2], prompt[1], spk)
        except Exception as e:
            print(f"  [warn] add_zero_shot_spk {spk} 失败: {e} (将 fallback 直传 prompt)")
    print(f"  注册 {len(spk_prompt)} 声纹完成")

    # 4) 合成 target 家居指令
    print("=== 合成 target 家居指令 ===")
    homes = read_jsonl(args.home_corpus)
    print(f"  home 语料 {len(homes)} 条")
    target_manifest, ti = [], 0
    for si, spk in enumerate(sel_spks):
        cmds = rng.choices(homes, k=args.n_cmd_per_spk)
        for cmd in cmds:
            out_wav = f"{args.out_dir}/target/{spk}_{ti:05d}.wav"
            try:
                chunks = list(cv.inference_zero_shot(
                    cmd["text"], "", "", zero_shot_spk_id=spk
                ))
                _save(cv, out_wav, chunks, sr_out)
            except Exception as e:
                # fallback: 直接传 prompt audio + text
                pp = spk_prompt[spk]
                try:
                    chunks = list(cv.inference_zero_shot(cmd["text"], pp[2], pp[1]))
                    _save(cv, out_wav, chunks, sr_out)
                except Exception as e2:
                    print(f"  [fail] target {spk}_{ti} '{cmd['text']}': {e2}")
                    continue
            target_manifest.append({
                "id": f"target_{ti:05d}", "wav": out_wav, "ref": cmd["text"],
                "spk": spk, "category": cmd["category"],
            })
            ti += 1
        print(f"  [{si+1}/{len(sel_spks)}] {spk}: 累计 {ti} target")

    # 5) 合成 interferer 闲话
    print("=== 合成 interferer 闲话 ===")
    chats = read_jsonl(args.chat_corpus)
    interf_manifest = []
    for ci, chat in enumerate(chats[: args.n_interferer]):
        spk = rng.choice(sel_spks)
        out_wav = f"{args.out_dir}/interferer/{spk}_{ci:05d}.wav"
        try:
            chunks = list(cv.inference_zero_shot(
                chat["text"], "", "", zero_shot_spk_id=spk
            ))
            _save(cv, out_wav, chunks, sr_out)
        except Exception as e:
            print(f"  [fail] interferer {ci}: {e}")
            continue
        interf_manifest.append({
            "id": f"interferer_{ci:05d}", "wav": out_wav, "ref": chat["text"],
            "spk": spk, "category": "chitchat",
        })

    # 6) 写 manifest
    with open(f"{args.out_dir}/target_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in target_manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out_dir}/interferer_manifest.jsonl", "w", encoding="utf-8") as f:
        for r in interf_manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"=== DONE: target={len(target_manifest)} interferer={len(interf_manifest)} "
          f"-> {args.out_dir} ===")


if __name__ == "__main__":
    main()
