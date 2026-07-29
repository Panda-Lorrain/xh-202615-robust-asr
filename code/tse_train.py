#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自训中文 TSE + ASR 联合训练 (V1) - 治本路线

================================================================
为什么这样设计 (CLAUDE.md 2026-07-27 三红线)
================================================================
1. 中文数据: Aishell-1 合成 (data_aug_recipe / tse_data_aug)
2. 避免 SI-SDR 纯陷阱 (EoW 感知-识别鸿沟): **联合 loss**
   L = α · (-SI-SNR) + β · CTC
   SI-SNR 保分离可懂, CTC 把"分离为识别服务"压力传到 mask net,
   避免"分离听着干净但 mel 失真 ASR 转错"。
3. 数据对齐题目: -5~5dB SNR / 0-100% 重叠 / ~1.8s enrollment (tse_data_aug 已实现)

================================================================
架构 (主方案 V1, 自写可控, 复用 SpeechBrain 思路但无外部权重依赖)
================================================================
  mix_wav ──STFT──► mix_ri [B,2,F,T]
                              │
  enroll_wav ──mel─► spk_enc ─► d_vec [B,D]
                              │   ├── (mix_ri, d_vec) → MaskNet → mask [B,2,F,T]
                              │   │
                              ▼   ▼
                       est_ri = complex_mul(mix_ri, mask) ──ISTFT──► est_wav
                                          │
                                          ▼ mel
                                       ASR encoder + CTC head → ctc_logits ─► CTC(logits, ref_tokens)

参数量 (n_fft=512, d_vec=192, mask_hidden=256, mask_blocks=4, asr_hidden=256, asr_layers=4):
  - SpeakerEncoder ~ 0.2M
  - MaskNet       ~ 1.5M
  - ASREncoder    ~ 1.5M
  - CTC head      ~ 0.2M (vocab=2000)
  合计 ~ 3.4M, 单 GPU 微 batch 可训

================================================================
备选 V2 (文档讲不实现): Whisper-Sidecar 思路自训
  - frozen Whisper encoder + Sidecar mask 分支 (embedding 空间分离)
  - 纯 ASR loss (无 SI-SDR)
  - 优点: 完全绕 SI-SDR; 缺点: 训练复杂需 4×GPU × 数天
================================================================

================================================================
用法
================================================================
冒烟 (本机 4060, CPU 也可):
  uv run --python code/.venv_tse python code/tse_train.py \
      --manifest code/_tse_aug_smoke/manifest.jsonl \
      --out-dir code/_tse_train_smoke --steps 10 --batch-size 2 --device cuda

全量 (租算力 L20/A100):
  uv run --python code/.venv_tse python code/tse_train.py \
      --manifest /path/full_manifest.jsonl \
      --out-dir /path/full_out --steps 100000 --batch-size 8 \
      --device cuda --lr 2e-4
"""
from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader


SR = 16000


# ============================================================================
# 1. SI-SNR loss (zero-mean, scale-invariant)
# ============================================================================
def si_snr(est: torch.Tensor, target: torch.Tensor,
           lengths: Optional[torch.Tensor] = None,
           eps: float = 1e-8) -> torch.Tensor:
    """SI-SNR (scale-invariant SNR) for batches.

    Args:
        est:    [B, T]
        target: [B, T]
    Returns:
        [B] per-sample SI-SNR in dB
    """
    if lengths is None:
        lengths = torch.full(
            (est.size(0),), est.size(1), dtype=torch.long, device=est.device
        )
    time = torch.arange(est.size(1), device=est.device).unsqueeze(0)
    mask = (time < lengths.unsqueeze(1)).to(est.dtype)
    denom = lengths.clamp_min(1).to(est.dtype).unsqueeze(1)
    est = (est - (est * mask).sum(dim=-1, keepdim=True) / denom) * mask
    target = (
        target - (target * mask).sum(dim=-1, keepdim=True) / denom
    ) * mask
    dot = (est * target).sum(dim=-1, keepdim=True)
    energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / energy) * target
    e_noise = est - s_target
    ratio = (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


# ============================================================================
# 2. 子模块
# ============================================================================
class SpeakerEncoder(nn.Module):
    """mel spectrogram -> d_vec (轻量 CNN, 不依赖 ECAPA-TDNN 复杂结构)。"""

    def __init__(self, n_mels: int = 80, d_out: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, d_out),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T] -> d_vec: [B, d_out]"""
        x = mel.unsqueeze(1)  # [B, 1, n_mels, T]
        return F.normalize(self.net(x), dim=-1)


class TCNBlock(nn.Module):
    """TCN block (depthwise conv on time axis + FF + residual)."""

    def __init__(self, ch: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=(1, kernel_size),
                               padding=(0, pad), dilation=(1, dilation))
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        return x + h


class MaskNet(nn.Module):
    """(mix_stft_realimag, d_vec) -> complex mask in [-1, 1].

    输入:
      mix_ri: [B, 2, F, T]
      dvec:   [B, D]
    输出:
      mask:   [B, 2, F, T]
    """

    def __init__(self, n_fft: int = 512, d_vec: int = 192,
                 hidden: int = 256, n_blocks: int = 4):
        super().__init__()
        self.proj_in = nn.Conv2d(2, hidden, 1)
        self.dvec_proj = nn.Linear(d_vec, hidden)
        self.tcn_blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size=5, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        self.proj_out = nn.Conv2d(hidden, 2, 1)

    def forward(self, mix_ri: torch.Tensor,
                dvec: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(mix_ri)  # [B, hidden, F, T]
        d = self.dvec_proj(dvec).unsqueeze(-1).unsqueeze(-1)  # [B, hidden, 1, 1]
        x = x + d  # broadcast across F, T
        for blk in self.tcn_blocks:
            x = blk(x)
        return torch.tanh(self.proj_out(x))  # [B, 2, F, T]


def apply_complex_mask(mix_ri: torch.Tensor,
                       mask_ri: torch.Tensor) -> torch.Tensor:
    """Apply a complex ratio mask to [real, imag] tensors."""
    if mix_ri.shape != mask_ri.shape or mix_ri.size(1) != 2:
        raise ValueError(
            f"expected matching [B,2,F,T] tensors, got "
            f"{tuple(mix_ri.shape)} and {tuple(mask_ri.shape)}"
        )
    xr, xi = mix_ri[:, 0], mix_ri[:, 1]
    mr, mi = mask_ri[:, 0], mask_ri[:, 1]
    return torch.stack((xr * mr - xi * mi, xr * mi + xi * mr), dim=1)


class ASRLayer(nn.Module):
    """simple Conformer-ish: depthwise conv + FF + residual + LayerNorm."""

    def __init__(self, d: int, kernel_size: int = 15):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(d, d, kernel_size, padding=pad, groups=d)
        self.norm1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4), nn.ReLU(), nn.Linear(d * 4, d)
        )
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d]
        h = x.transpose(1, 2)  # [B, d, T]
        h = self.conv(h).transpose(1, 2)  # [B, T, d]
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class ASREncoder(nn.Module):
    """simple trainable Conformer-like encoder for CTC (Whisper-frozen 是 V2 升级)。"""

    def __init__(self, n_mels: int = 80, d_model: int = 256, n_layers: int = 4):
        super().__init__()
        self.proj_in = nn.Linear(n_mels, d_model)
        self.layers = nn.ModuleList([ASRLayer(d_model) for _ in range(n_layers)])

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel: [B, T, n_mels] -> hidden: [B, T, d_model]"""
        x = self.proj_in(mel)
        for L in self.layers:
            x = L(x)
        return x


class TSE_ASR_Joint(nn.Module):
    """完整 TSE + ASR 联合模型。

    forward(mix, enroll):
      est_wav (SI-SDR 监督), ctc_logits (CTC 监督)
    """

    def __init__(self, vocab_size: int,
                 n_fft: int = 512, hop: int = 128, win: int = 512,
                 n_mels: int = 80, d_vec: int = 192,
                 mask_hidden: int = 256, mask_blocks: int = 4,
                 asr_hidden: int = 256, asr_layers: int = 4):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.n_mels = n_mels
        # 子模块
        self.spk_encoder = SpeakerEncoder(n_mels, d_vec)
        self.mask_net = MaskNet(n_fft, d_vec, mask_hidden, mask_blocks)
        self.asr_encoder = ASREncoder(n_mels, asr_hidden, asr_layers)
        self.ctc_head = nn.Linear(asr_hidden, vocab_size)
        # mel transform (Kaldi-like: n_fft=512, hop=128, mel=80) + STFT window
        self.register_buffer("window", torch.hann_window(win))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=hop, win_length=win,
            power=2.0, n_mels=n_mels, window_fn=torch.hann_window,
        )

    def _mel(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> log-mel [B, n_mels, T_m]"""
        mel = self.mel_transform(wav)  # [B, n_mels, T_m]
        return torch.log(mel + 1e-6)

    def forward(self, mix: torch.Tensor,
                enroll: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mix, enroll: [B, T]
        # 1) speaker embedding from enrollment
        mel_e = self._mel(enroll)  # [B, n_mels, T_e]
        d_vec = self.spk_encoder(mel_e)  # [B, d_vec]
        # 2) STFT of mix
        stft = torch.stft(mix, self.n_fft, self.hop, window=self.window,
                          return_complex=True, center=True)  # [B, F, T]
        mix_ri = torch.stack([stft.real, stft.imag], dim=1)  # [B, 2, F, T]
        # 3) mask
        m = self.mask_net(mix_ri, d_vec)  # [B, 2, F, T]
        est_ri = apply_complex_mask(mix_ri, m)
        est_stft = torch.complex(est_ri[:, 0], est_ri[:, 1])
        T_in = mix.size(-1)
        est_wav = torch.istft(est_stft, self.n_fft, self.hop,
                              window=self.window, center=True, length=T_in)
        # 4) ASR
        mel_est = self._mel(est_wav)  # [B, n_mels, T_est]
        asr_hidden = self.asr_encoder(mel_est.transpose(1, 2))  # [B, T_est, d_asr]
        ctc_logits = self.ctc_head(asr_hidden)  # [B, T_est, vocab]
        return est_wav, ctc_logits


# ============================================================================
# 3. 词表 (字符级)
# ============================================================================
PAD_TOK, BLANK_TOK = "<pad>", "<blank>"  # CTC blank id = 0 by convention


def build_vocab(refs: List[str]) -> Dict[str, int]:
    """字符级词表: blank=0, 其余按出现顺序。"""
    vocab = {BLANK_TOK: 0}
    for r in refs:
        for ch in r:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    vocab[PAD_TOK] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab[ch] for ch in text if ch in vocab]


# ============================================================================
# 4. Dataset / collate
# ============================================================================
class TSEDataset(Dataset):
    def __init__(self, manifest_path: str, vocab: Dict[str, int],
                 seg_samples: int = 0, enroll_samples: int = 28800):
        """Load complete labeled utterances.

        seg_samples is a deprecated safety limit. Labeled utterances are never
        cropped because a segment-level transcript/forced alignment is absent.
        """
        self.rows = [json.loads(l) for l in open(manifest_path, encoding="utf-8")
                     if l.strip()]
        self.vocab = vocab
        self.seg = seg_samples
        self.enroll_n = enroll_samples

    def __len__(self):
        return len(self.rows)

    def _load_wav(self, path: str) -> np.ndarray:
        wav, sr = torchaudio.load(path)  # [1, T]
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(dim=0).numpy().astype(np.float32)  # [T]
        return wav

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        mix = self._load_wav(r["recognition_audio"])
        clean = self._load_wav(r["clean_target_audio"])
        if len(mix) != len(clean):
            raise ValueError(
                f"{r['id']}: recognition/clean length mismatch "
                f"{len(mix)} != {len(clean)}"
            )
        if self.seg > 0 and len(mix) > self.seg:
            raise ValueError(
                f"{r['id']}: {len(mix)} samples exceed --seg-samples={self.seg}; "
                "cropping is forbidden without segment-level transcripts"
            )
        enroll = self._load_wav(r["enrollment_audio"])
        if len(enroll) >= self.enroll_n:
            max_start = len(enroll) - self.enroll_n
            start = random.randint(0, max_start) if max_start > 0 else 0
            enroll = enroll[start:start + self.enroll_n]
        else:
            enroll = np.pad(enroll, (0, self.enroll_n - len(enroll)))
        tokens = encode_text(r.get("ref", ""), self.vocab)
        return (torch.from_numpy(mix), torch.from_numpy(clean),
                torch.from_numpy(enroll), torch.tensor(tokens, dtype=torch.long),
                r["id"])


def collate_fn(batch):
    wav_lens = torch.tensor([len(b[0]) for b in batch], dtype=torch.long)
    mixes = torch.nn.utils.rnn.pad_sequence(
        [b[0] for b in batch], batch_first=True, padding_value=0.0
    )
    cleans = torch.nn.utils.rnn.pad_sequence(
        [b[1] for b in batch], batch_first=True, padding_value=0.0
    )
    enrolls = torch.stack([b[2] for b in batch], dim=0)
    tokens = [b[3] for b in batch]
    ids = [b[4] for b in batch]
    token_lens = torch.tensor([len(t) for t in tokens], dtype=torch.long)
    if any(len(t) for t in tokens):
        tokens_padded = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=0
        )
    else:
        tokens_padded = torch.empty((len(batch), 0), dtype=torch.long)
    return mixes, cleans, enrolls, wav_lens, tokens_padded, token_lens, ids


# ============================================================================
# 5. 训练循环
# ============================================================================
def train(args):
    device = torch.device(args.device)
    print(f"[train] device={device}")

    # 1) 词表 (从 manifest 构建)
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")
            if l.strip()]
    refs = [r.get("ref", "") for r in rows]
    vocab = build_vocab(refs)
    print(f"[train] manifest: {len(rows)} 条, vocab size: {len(vocab)}")

    # 2) 模型
    model = TSE_ASR_Joint(
        vocab_size=len(vocab),
        n_fft=args.n_fft, hop=args.hop, win=args.win,
        n_mels=args.n_mels, d_vec=args.d_vec,
        mask_hidden=args.mask_hidden, mask_blocks=args.mask_blocks,
        asr_hidden=args.asr_hidden, asr_layers=args.asr_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params/1e6:.2f}M")

    # 3) 数据
    ds = TSEDataset(args.manifest, vocab,
                    seg_samples=args.seg_samples,
                    enroll_samples=args.enroll_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_fn, drop_last=False, num_workers=0)
    if len(ds) == 0:
        raise ValueError("empty training manifest")

    # 4) 优化器
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5) 训练循环
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log_f = open(log_path, "w", encoding="utf-8")
    step = 0
    t0 = time.time()
    model.train()
    print(f"[train] 开始训练, 目标 {args.steps} steps, alpha={args.alpha} (SI-SNR) beta={args.beta} (CTC)")
    while step < args.steps:
        for mix, clean, enroll, wav_lens, tokens, token_lens, ids in dl:
            mix = mix.to(device); clean = clean.to(device)
            enroll = enroll.to(device)
            wav_lens = wav_lens.to(device)
            tokens = tokens.to(device); token_lens = token_lens.to(device)

            est_wav, ctc_logits = model(mix, enroll)
            # SI-SNR loss (越大越好 → 取负)
            L_sisnr = -si_snr(est_wav, clean, wav_lens).mean()
            # CTC loss (log-probs, target, input_len, target_len, blank=0)
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V]
            input_lens = (wav_lens // args.hop + 1).clamp_max(log_probs.size(0))
            if torch.any(token_lens > input_lens):
                bad = [
                    ids[i] for i in range(len(ids))
                    if token_lens[i] > input_lens[i]
                ]
                raise ValueError(f"CTC target longer than input frames: {bad}")
            L_ctc = F.ctc_loss(log_probs, tokens, input_lens, token_lens,
                               blank=0, zero_infinity=False)
            loss = args.alpha * L_sisnr + args.beta * L_ctc

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            step += 1
            elapsed = time.time() - t0
            msg = (f"step {step}/{args.steps} | loss={loss.item():.3f} "
                   f"sisnr={-L_sisnr.item():.2f}dB ctc={L_ctc.item():.3f} "
                   f"| {elapsed:.1f}s")
            print(msg)
            log_f.write(msg + "\n"); log_f.flush()
            if step % args.log_every == 0 or step <= 3:
                # 早期打印梯度范数 (冒烟健康检查)
                grad_norm = sum(p.grad.norm().item() ** 2
                                for p in model.parameters()
                                if p.grad is not None) ** 0.5
                msg2 = f"  └ grad_norm={grad_norm:.3f}"
                print(msg2); log_f.write(msg2 + "\n")
            if step >= args.steps:
                break

    # 保存词表 + 模型
    ckpt_path = os.path.join(args.out_dir, "tse_asr_joint.pt")
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab,
        "config": vars(args),
        "step": step,
    }, ckpt_path)
    vocab_path = os.path.join(args.out_dir, "vocab.json")
    json.dump(vocab, open(vocab_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[train] DONE, {step} steps, 模型 → {ckpt_path}, vocab → {vocab_path}")
    print(f"[train] 总耗时 {time.time()-t0:.1f}s")
    log_f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="tse_data_aug.py 输出的 manifest.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seg-samples", type=int, default=0,
                    help="弃用的安全上限；0=整句。无分段标注时禁止裁剪")
    ap.add_argument("--enroll-samples", type=int, default=28800,
                    help="enrollment 段长 (28.8k=1.8s 题目规格)")
    # 模型超参
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--n-mels", type=int, default=80)
    ap.add_argument("--d-vec", type=int, default=192)
    ap.add_argument("--mask-hidden", type=int, default=256)
    ap.add_argument("--mask-blocks", type=int, default=4)
    ap.add_argument("--asr-hidden", type=int, default=256)
    ap.add_argument("--asr-layers", type=int, default=4)
    # loss 权重
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="SI-SNR loss 权重")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="CTC loss 权重 (建议>=alpha, 让 ASR 主导)")
    # 训练
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
