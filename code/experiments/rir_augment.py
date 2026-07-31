#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIT IR Survey RIR 卷积增广 (CC-BY-4.0, Traer & McDermott 2016 PNAS).

为零新依赖设计: 纯 numpy FFT 卷积 (项目未装 torchaudio / scipy.signal.fftconvolve,
符合 CLAUDE.md "可选依赖必须显式声明" 红线)。仅在 tse_data_aug.synthesize_tse_triple
内被调用, 给 target/interferer 各自卷积不同 RIR, 补"真实双人重叠相位 + 房间混响"
(破 LoRA POC mix_overlap 纯线性叠加的域不匹配根因)。

数据源: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html (271 真实环境 RIR)
本机: E:/midea_datasets/mit_ir_survey/Audio/*.wav

设计要点 (见 docs/家居指令数据集制作计划_2026-07-31.md 阶段3 + RIR 核查报告):
- 卷积在前, SIR/SNR 标定在后 → 下游 _scale_to_sir / add_noise_relative_to_target
  自动以卷积后 RMS 标定, SIR/SNR 自动正确 (无需手动能量归一)。
- 卷积后裁回原长 (取前 N 样本), 下游 mix_with_random_timing 长度假设兼容。
- RIR 本身 peak 归一到 [-1,1] 防 IR 峰值差异致数值爆; 卷积后不做 RMS 归一
  (保留房间共振色彩, 这正是真实房间的物理特征)。
- enrollment 不卷积 (另一 utterance 的近场参考, 保持声纹干净)。
"""
from __future__ import annotations

import glob
import os
import random
from typing import List, Tuple

import librosa
import numpy as np

SR = 16000  # 与 data_aug_recipe.SR 一致


def load_rir_pool(rir_root: str, sr: int = SR) -> List[np.ndarray]:
    """加载 rir_root 下全部 wav (递归), 重采样到 sr, peak 归一到 [-1,1]。

    返回 List[np.ndarray], 每个是 1D float32 变长。跳过空 / 全零 / 异常文件。
    """
    paths = sorted(glob.glob(os.path.join(rir_root, "**", "*.wav"), recursive=True))
    if len(paths) < 2:
        raise ValueError(
            f"RIR pool 太小: {rir_root} 仅找到 {len(paths)} 个 wav "
            f"(需 ≥2, 期望 MIT IR Survey ~270)"
        )
    pool: List[np.ndarray] = []
    skipped = 0
    for p in paths:
        try:
            rir, _ = librosa.load(p, sr=sr, mono=True)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if rir.size == 0 or np.abs(rir).max() < 1e-6:
            skipped += 1
            continue
        rir = (rir / (np.abs(rir).max() + 1e-8)).astype(np.float32)
        pool.append(rir)
    if len(pool) < 2:
        raise ValueError(
            f"有效 RIR 不足: {rir_root} (跳过 {skipped} 个, 仅 {len(pool)} 个有效)"
        )
    print(f"  [rir] 加载 {len(pool)} 个 RIR (跳过 {skipped}) from {rir_root}")
    return pool


def _fft_convolve(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """numpy 快速卷积, 等价 scipy.signal.fftconvolve(x, rir, 'full')。

    输出长度 = len(x) + len(rir) - 1 (线性卷积全长)。
    """
    x = np.asarray(x, dtype=np.float32)
    rir = np.asarray(rir, dtype=np.float32)
    n = x.size + rir.size - 1
    nfft = 1 << int(np.ceil(np.log2(n)))  # 下一个 2 的幂, FFT 最快
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(rir, nfft), nfft)[:n]
    return y.astype(np.float32)


def apply_rir(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """用 RIR 卷积 wav, 裁回原长度 (取前 N 样本, 保留 direct sound + 尾部混响)。

    输入: wav [N] float32, rir [M] float32 (已 peak 归一)。
    输出: [N] float32 (长度不变, 下游 mix_with_random_timing 兼容)。

    不做 RMS 归一: RIR 卷积改变能量是真实房间物理特征, 下游 SIR/SNR 标定会吸收。
    """
    n = wav.size
    if n == 0 or rir.size == 0:
        return np.asarray(wav, dtype=np.float32)
    out = _fft_convolve(wav, rir)[:n]
    return out.astype(np.float32)


def sample_two_rirs(
    pool: List[np.ndarray], rng: random.Random
) -> Tuple[np.ndarray, np.ndarray]:
    """无放回抽 2 个不同 RIR (target, interferer), 模拟两人在不同位置。"""
    if len(pool) < 2:
        raise ValueError("RIR pool 至少需要 2 个")
    rir_t, rir_i = rng.sample(pool, 2)
    return rir_t, rir_i


if __name__ == "__main__":
    # 冒烟: 加载 MIT IR, 对 1s 正弦波卷积, 验证长度不变 + 有限值 + 不爆。
    import argparse as _ap

    _p = _ap.ArgumentParser(description="rir_augment 冒烟自测")
    _p.add_argument("--rir-root", default="E:/midea_datasets/mit_ir_survey/Audio")
    args = _p.parse_args()

    pool = load_rir_pool(args.rir_root)
    _rng = random.Random(42)
    _t = (np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR).astype(np.float32) * 0.3)
    rt, ri = sample_two_rirs(pool, _rng)
    yo = apply_rir(_t, rt)
    assert yo.shape == _t.shape, f"长度变了: {yo.shape} vs {_t.shape}"
    assert np.all(np.isfinite(yo)), "卷积结果含 NaN/inf"
    # 卷积后能量应变化 (房间色彩), 但 peak 归一 RIR 后不会爆 100x
    rms_in = np.sqrt(np.mean(_t ** 2))
    rms_out = np.sqrt(np.mean(yo ** 2))
    print(f"  in rms={rms_in:.4f} out rms={rms_out:.4f} "
          f"out peak={np.abs(yo).max():.4f} len={yo.size}")
    print(f"  rir_target len={rt.size} rir_interferer len={ri.size} (不同: {rt is not ri})")
    assert rt is not ri, "target/interferer 抽到同一 RIR"
    print("ALL PASS: rir_augment 冒烟通过")
