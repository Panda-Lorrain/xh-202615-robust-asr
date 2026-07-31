#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TSE 训练数据增广 (3 件套) - 自训中文 TSE 专用

================================================================
与 data_aug_recipe.py 的关系
================================================================
data_aug_recipe.synthesize_one 只输出 (enrollment, recognition),
适合 qwen ASR 微调 (不需 clean target 监督)。

但 TSE 训练需要 **3 件套**:
  - enrollment.wav   (~1.8s 目标说话人参考, 给 speaker encoder)
  - recognition.wav  (混合音频 = target + interferer + noise, 给 separator)
  - clean_target.wav (干净 target 波形 = SI-SDR 监督; ⚠️ 包含小声/快语速预处理后版本, 与 recognition 中 target 完全对齐)

本脚本复用 data_aug_recipe 的 target/enrollment 增广工具，并单独实现
TSE 所需的严格对齐混合。enrollment 必须来自同说话人的另一条 utterance，
避免模型利用同句内容和录音条件走捷径。

================================================================
设计红线对齐 (CLAUDE.md 2026-07-27)
================================================================
1. 中文数据: Aishell-1 + MUSAN, 不碰 A 集 (lessons-pitfalls §14)
2. 题目规格对齐:
   - SNR −5~5dB (DEFAULT_SNR_BUCKETS)
   - 重叠 0~100% (DEFAULT_OVERLAP_BUCKETS, 偏中高)
   - enrollment ~1.5–2.5s (题目 ~1.8s)
   - enrollment 可选污染 (pollute_p=0.3)
   - target 小声化 / 快语速 (死区失败模式)
3. 失败分布加权 (memory overlap-is-cer-failure-rootcause):
   - 失败组重叠中位 45% / 75% 重叠 / 97% 双人 → overlap 权重偏中高
   - 死区 sim<0.4 (78.8%) → SNR 偏低 + babble 多

================================================================
产物 (每条训练对)
================================================================
  out_dir/enrollment/<uid>.wav   (1.5–2.5s, mono 16kHz)
  out_dir/recognition/<uid>.wav  (target/interferer 随机时序混合, mono 16kHz)
  out_dir/clean_target/<uid>.wav (= 与 recognition 等长的 target 波形, 含小声/快预处理, 用于 SI-SDR 监督)
  out_dir/manifest.jsonl         (路径 + 文本 + 增广参数)

recognition 与 clean_target 严格等长，clean_target 在 target 非活动区补零。
target 全句不会被数据生成器截断，因此 ref 与监督波形保持一致。
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import random
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

# 复用项目内增广工具
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from data_aug_recipe import (  # noqa: E402
    sample_aug_params as sample_legacy_aug_params, make_fast, cut_enrollment,
    pollute_enrollment, SR,
)
from simulate_pipeline import _fit_noise  # noqa: E402
from build_dataset import gen_white, gen_pink, gen_babble  # noqa: E402


TSE_AUGMENTATION_PROFILES = {
    "balanced": {
        "overlap_weights": [0.10, 0.20, 0.30, 0.25, 0.15],
        "sir_weights": [0.10, 0.20, 0.40, 0.20, 0.10],
        "snr_weights": [0.10, 0.20, 0.40, 0.20, 0.10],
        "noise_weights": [0.30, 0.30, 0.40],
        "fast_probability": 0.15,
        "enroll_pollute_probability": 0.15,
    },
    "hard": {
        "overlap_weights": [0.05, 0.10, 0.25, 0.35, 0.25],
        "sir_weights": [0.20, 0.25, 0.30, 0.15, 0.10],
        "snr_weights": [0.20, 0.25, 0.30, 0.15, 0.10],
        "noise_weights": [0.20, 0.25, 0.55],
        "fast_probability": 0.30,
        "enroll_pollute_probability": 0.30,
    },
}


def sample_tse_aug_params(
    rng: random.Random, profile: str = "balanced"
) -> Dict:
    """Sample TSE difficulty without compounding quiet gain and SIR/SNR.

    Relative target loudness is already represented by SIR and SNR. Applying
    an additional target-only gain after calibration pushed the historical
    synthetic set down to about -12 dB, outside the advertised [-5, 5] range.
    """
    if profile == "legacy":
        return sample_legacy_aug_params(rng)
    if profile not in TSE_AUGMENTATION_PROFILES:
        raise ValueError(
            f"unknown TSE augmentation profile {profile!r}; expected "
            f"{sorted(TSE_AUGMENTATION_PROFILES)} or 'legacy'"
        )
    config = TSE_AUGMENTATION_PROFILES[profile]
    overlap = rng.choices(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        config["overlap_weights"],
        k=1,
    )[0]
    sir = rng.choices(
        [-5.0, -3.0, 0.0, 3.0, 5.0], config["sir_weights"], k=1
    )[0]
    snr = rng.choices(
        [-5.0, -3.0, 0.0, 3.0, 5.0], config["snr_weights"], k=1
    )[0]
    noise_type = rng.choices(
        ["white", "pink", "babble"], config["noise_weights"], k=1
    )[0]
    speed = (
        rng.uniform(1.1, 1.4)
        if rng.random() < config["fast_probability"]
        else 1.0
    )
    return {
        "overlap_ratio": float(overlap),
        "sir_db": float(sir),
        "snr_db": float(snr),
        "noise_type": noise_type,
        # Do not shift the calibrated final SIR/SNR a second time.
        "target_gain_db": 0.0,
        "target_speed_rate": float(speed),
        "enroll_dur_sec": float(rng.uniform(1.5, 2.5)),
        "enroll_pollute_p": float(
            config["enroll_pollute_probability"]
        ),
        "enroll_pollute_snr_db": float(rng.uniform(8.0, 15.0)),
    }


def _active_rms(wav: np.ndarray, eps: float = 1e-8) -> float:
    """RMS over non-negligible samples, used for explicit SIR/SNR control."""
    x = np.asarray(wav, dtype=np.float32)
    active = x[np.abs(x) > 1e-5]
    if active.size == 0:
        active = x
    return float(np.sqrt(np.mean(active ** 2) + eps))


def _speech_activity_mask(
    wav: np.ndarray, frame_samples: int = 320, top_db: float = 40.0
) -> np.ndarray:
    """Expand a frame-RMS speech decision to a sample-level mask."""
    x = np.asarray(wav, dtype=np.float32)
    if x.size == 0:
        return np.zeros(0, dtype=bool)
    n_frames = int(np.ceil(x.size / frame_samples))
    padded = np.pad(x, (0, n_frames * frame_samples - x.size))
    rms = np.sqrt(
        np.mean(padded.reshape(n_frames, frame_samples) ** 2, axis=1)
        + 1e-10
    )
    threshold = max(float(rms.max()) * 10.0 ** (-top_db / 20.0), 1e-5)
    return np.repeat(rms >= threshold, frame_samples)[:x.size]


def _scale_to_sir(target: np.ndarray, interferer: np.ndarray,
                  sir_db: float) -> np.ndarray:
    """Scale interferer so target/interferer active-speech RMS matches SIR."""
    target_rms = _active_rms(target)
    interferer_rms = _active_rms(interferer)
    desired_interferer_rms = target_rms / (10.0 ** (sir_db / 20.0))
    return (interferer * (desired_interferer_rms / max(interferer_rms, 1e-8))).astype(
        np.float32
    )


def mix_with_random_timing(
    target: np.ndarray,
    interferer: np.ndarray,
    overlap_ratio: float,
    sir_db: float,
    rng: random.Random,
    max_context_sec: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Mix full utterances with controlled overlap and random context."""
    target = np.asarray(target, dtype=np.float32)
    interferer = _scale_to_sir(
        target, np.asarray(interferer, dtype=np.float32), sir_db
    )
    lt, li = len(target), len(interferer)
    if lt == 0 or li == 0:
        raise ValueError("target/interferer must be non-empty")

    overlap_samples = int(round(np.clip(overlap_ratio, 0.0, 1.0) * min(lt, li)))
    context = rng.randint(0, max(0, int(max_context_sec * SR)))
    target_start = context
    placement = rng.choice(("leading", "trailing"))

    if overlap_samples == 0:
        gap = rng.randint(0, max(1, int(0.25 * SR)))
        if placement == "leading":
            interferer_start = 0
            target_start = li + gap + context
        else:
            interferer_start = target_start + lt + gap
    elif placement == "leading":
        interferer_start = target_start + overlap_samples - li
        if interferer_start < 0:
            shift = -interferer_start
            interferer_start = 0
            target_start += shift
    else:
        interferer_start = target_start + lt - overlap_samples

    out_len = max(target_start + lt, interferer_start + li)
    clean_target = np.zeros(out_len, dtype=np.float32)
    interference = np.zeros(out_len, dtype=np.float32)
    clean_target[target_start:target_start + lt] = target
    interference[interferer_start:interferer_start + li] = interferer
    mixed = clean_target + interference
    target_activity = _speech_activity_mask(clean_target)
    interferer_activity = _speech_activity_mask(interference)
    active_denominator = min(
        int(target_activity.sum()), int(interferer_activity.sum())
    )
    active_overlap = int((target_activity & interferer_activity).sum())
    meta = {
        "sir_db": float(sir_db),
        "target_start_sample": int(target_start),
        "interferer_start_sample": int(interferer_start),
        "overlap_samples": int(overlap_samples),
        "active_overlap_samples": active_overlap,
        "active_overlap_ratio": (
            float(active_overlap / active_denominator)
            if active_denominator > 0 else 0.0
        ),
        "placement": placement,
    }
    return mixed.astype(np.float32), clean_target, meta


def add_noise_relative_to_target(
    mixed: np.ndarray,
    clean_target: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Add noise with SNR defined against target speech, not the mixture."""
    target_rms = _active_rms(clean_target)
    noise_rms = _active_rms(noise)
    desired_noise_rms = target_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (desired_noise_rms / max(noise_rms, 1e-8))
    return (mixed + scaled_noise).astype(np.float32)


def synthesize_tse_triple(
    target_wav: np.ndarray,
    interferer_wav: np.ndarray,
    noise_wav: Optional[np.ndarray],
    enrollment_wav: Optional[np.ndarray] = None,
    nontarget_pool: Optional[List[np.ndarray]] = None,
    *,
    overlap_ratio: float,
    snr_db: float,
    noise_type: str = "white",
    target_gain_db: float = 0.0,
    target_speed_rate: float = 1.0,
    enroll_dur_sec: float = 1.8,
    enroll_pollute_p: float = 0.3,
    enroll_pollute_snr_db: float = 10.0,
    sir_db: Optional[float] = None,
    rng: Optional[random.Random] = None,
    rir_pool: Optional[List[np.ndarray]] = None,
    rir_prob: float = 0.5,
):
    """合成一条 TSE 训练三元组: (enrollment, recognition, clean_target, params)。

    与 data_aug_recipe.synthesize_one 的区别: 额外返回 clean_target
    (= 小声/快预处理后的 target 波形, 未混合干扰和噪声, 作 SI-SDR 监督)。

    Returns:
        enrollment_audio: np.float32 [T_enroll]
        recognition_audio: np.float32 [T_mix]
        clean_target_audio: np.float32 [T_mix]  # 与 recognition 等长
        metadata: dict
    """
    rng = rng or random.Random()
    nprng = np.random.default_rng(rng.randrange(2**31))

    # 1) target 预处理。Quiet gain is applied after SIR/SNR calibration so
    # it actually creates a relatively quiet target instead of scaling every
    # source together.
    t = target_wav.astype(np.float32)
    if target_speed_rate != 1.0:
        t = make_fast(t, target_speed_rate)

    # 1b) 可选 RIR 卷积 (MIT IR Survey, 见 rir_augment.py)。卷积在前/标定在后:
    # 下游 _scale_to_sir 与 add_noise_relative_to_target 自动以卷积后 RMS 标定,
    # SIR/SNR 自动正确。target 与 clean_target 共同一 RIR → SI-SDR 监督严格对齐;
    # enrollment 走独立路径 (cut_enrollment) 不卷积。两人抽不同 RIR 模拟不同位置。
    rir_applied = False
    rir_interferer = None
    if rir_pool is not None and len(rir_pool) >= 2 and rng.random() < rir_prob:
        from rir_augment import apply_rir, sample_two_rirs
        _rir_t, rir_interferer = sample_two_rirs(rir_pool, rng)
        t = apply_rir(t, _rir_t)
        rir_applied = True

    # 2) enrollment 必须来自同 speaker 的另一 utterance。
    if enrollment_wav is None:
        raise ValueError("enrollment_wav is required and must be a distinct utterance")
    enroll = cut_enrollment(
        np.asarray(enrollment_wav, dtype=np.float32), enroll_dur_sec, SR, rng
    )

    # 3) enrollment 污染 (可选)
    enroll_polluted = rng.random() < enroll_pollute_p and noise_wav is not None
    if enroll_polluted:
        enroll = pollute_enrollment(enroll, noise_wav,
                                     enroll_pollute_snr_db, rng, pollute_p=1.0)

    # 4) 完整 target + 随机前后重叠。SIR 独立于背景噪声 SNR。
    sir_db = float(rng.uniform(-5.0, 5.0) if sir_db is None else sir_db)
    # interferer RIR 注入 (与 step1 target 同批抽取的 rir_interferer)。
    # 在 mix_with_random_timing 前卷积, 保证 interference = nominal_mixed -
    # nominal_clean_target 数学上 = 卷积后的 interferer buffer, SI-SDR 对齐成立。
    interferer_proc = interferer_wav
    if rir_applied and rir_interferer is not None:
        from rir_augment import apply_rir
        interferer_proc = apply_rir(
            np.asarray(interferer_wav, dtype=np.float32), rir_interferer
        )
    nominal_mixed, nominal_clean_target, timing_meta = mix_with_random_timing(
        t, interferer_proc, overlap_ratio, sir_db, rng
    )
    interference = nominal_mixed - nominal_clean_target
    gain = 10.0 ** (float(target_gain_db) / 20.0)
    clean_target = nominal_clean_target * gain
    mixed = clean_target + interference
    n = len(mixed)

    # 5) 加噪。SNR 以 clean target 为基准，避免干扰越强噪声也被错误放大。
    if noise_type == "env":
        if noise_wav is None:
            raise ValueError("noise_type='env' requires a valid noise_wav")
        noise = _fit_noise(noise_wav, n)
    elif noise_type == "pink":
        noise = gen_pink(n, nprng)
    elif noise_type == "babble":
        noise = gen_babble(nontarget_pool or [interferer_wav], n, nprng)
    else:  # white / fallback
        noise = gen_white(n, nprng)
    noise = noise.astype(np.float32)
    if len(noise) < n:
        noise = np.pad(noise, (0, n - len(noise)))
    else:
        noise = noise[:n]
    recognition = add_noise_relative_to_target(
        mixed, nominal_clean_target, noise, snr_db
    )
    scaled_noise = recognition - mixed
    measured_sir_db = 20.0 * np.log10(
        _active_rms(clean_target) / max(_active_rms(interference), 1e-8)
    )
    measured_snr_db = 20.0 * np.log10(
        _active_rms(clean_target) / max(_active_rms(scaled_noise), 1e-8)
    )

    peak = float(np.max(np.abs(recognition))) if recognition.size else 0.0
    if peak > 0.99:
        scale = 0.99 / peak
        recognition *= scale
        clean_target *= scale

    metadata = {
        "overlap_ratio": float(overlap_ratio),
        "snr_db": float(snr_db),
        "noise_type": noise_type,
        "target_gain_db": float(target_gain_db),
        "measured_sir_db": float(measured_sir_db),
        "measured_snr_db": float(measured_snr_db),
        "target_speed_rate": float(target_speed_rate),
        "enroll_dur_sec": float(enroll_dur_sec),
        "enroll_pollute": bool(enroll_polluted),
        "enroll_pollute_snr_db": float(enroll_pollute_snr_db),
        "rir_applied": bool(rir_applied),
        **timing_meta,
    }
    return (enroll.astype(np.float32),
            recognition.astype(np.float32),
            clean_target,
            metadata)


def build_tse_pairs(
    target_items: List[Dict],
    interferer_items: List[Dict],
    noise_items: List[Dict],
    out_dir: str,
    n_per_target: int = 10,
    seed: int = 42,
    progress_every: int = 5,
    augmentation_profile: str = "balanced",
    rir_root: str = "",
):
    """批量生成 TSE 训练三元组 → out_dir/{enrollment,recognition,clean_target}/*.wav
    + manifest.jsonl。

    manifest 每行字段:
        id, enrollment_audio, recognition_audio, clean_target_audio, ref,
        target_src, **asdict(AugParams)
    """
    os.makedirs(os.path.join(out_dir, "enrollment"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "recognition"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "clean_target"), exist_ok=True)
    rng = random.Random(seed)
    rir_pool = None
    if rir_root:
        from rir_augment import load_rir_pool
        rir_pool = load_rir_pool(rir_root, sr=SR)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    cnt = 0
    t0 = time.time()

    # Preload target utterances. Enrollment lookup requires a different
    # utterance from the same speaker.
    target_audio = {}
    target_by_spk: Dict[str, List[Dict]] = {}
    for ti in target_items:
        spk = ti.get("spk")
        if not spk:
            raise ValueError("target manifest must contain 'spk' for distinct enrollment")
        target_by_spk.setdefault(spk, []).append(ti)
        try:
            target_audio[ti["wav"]] = librosa.load(ti["wav"], sr=SR)[0]
        except Exception as e:
            print(f"  [warn] 跳过 target {ti['wav']}: {e}")

    # 预加载干扰/噪声池 (小, 一次性)
    nontarget_records = []
    for ii in interferer_items:
        try:
            w, _ = librosa.load(ii["wav"], sr=SR)
            nontarget_records.append({
                "wav": w,
                "src": ii["wav"],
                "spk": ii.get("spk"),
            })
        except Exception as e:
            print(f"  [warn] 跳过干扰 {ii['wav']}: {e}")
    if not nontarget_records:
        raise ValueError("no valid interferer audio could be loaded")
    nontarget_pool = [record["wav"] for record in nontarget_records]
    noise_pool = []
    for ni in noise_items:
        try:
            w, _ = librosa.load(ni["wav"], sr=SR)
            noise_pool.append(w)
        except Exception:
            pass

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for ti_idx, ti in enumerate(target_items):
            t_wav = target_audio.get(ti["wav"])
            if t_wav is None:
                continue
            enrollment_candidates = [
                x for x in target_by_spk[ti["spk"]]
                if x["wav"] != ti["wav"] and x["wav"] in target_audio
            ]
            if not enrollment_candidates:
                print(f"  [warn] 跳过 {ti['wav']}: speaker {ti['spk']} 无独立 enrollment utterance")
                continue
            t_ref = ti.get("ref", "")
            valid_interferers = [
                record for record in nontarget_records
                if not record["spk"] or record["spk"] != ti["spk"]
            ]
            if not valid_interferers:
                raise ValueError(
                    f"no different-speaker interferer for target {ti['spk']}"
                )
            for k in range(n_per_target):
                params = sample_tse_aug_params(rng, augmentation_profile)
                interferer = rng.choice(valid_interferers)
                interferer_wav = interferer["wav"]
                noise_wav = (noise_pool[rng.randrange(len(noise_pool))]
                             if noise_pool else None)
                enrollment_item = rng.choice(enrollment_candidates)
                enroll, recog, clean_tgt, aug = synthesize_tse_triple(
                    t_wav, interferer_wav, noise_wav,
                    enrollment_wav=target_audio[enrollment_item["wav"]],
                    nontarget_pool=nontarget_pool, rng=rng,
                    rir_pool=rir_pool, **params,
                )
                uid = f"{os.path.splitext(os.path.basename(ti['wav']))[0]}_k{k:03d}"
                enr_path = os.path.join(out_dir, "enrollment", uid + ".wav")
                rec_path = os.path.join(out_dir, "recognition", uid + ".wav")
                ct_path = os.path.join(out_dir, "clean_target", uid + ".wav")
                sf.write(enr_path, enroll, SR)
                sf.write(rec_path, recog, SR)
                sf.write(ct_path, clean_tgt, SR)
                rec_line = {
                    "id": uid,
                    "enrollment_audio": enr_path,
                    "recognition_audio": rec_path,
                    "clean_target_audio": ct_path,
                    "ref": t_ref,
                    "target_src": ti["wav"],
                    "target_spk": ti["spk"],
                    "interferer_src": interferer["src"],
                    "interferer_spk": interferer["spk"],
                    "enrollment_src": enrollment_item["wav"],
                    "augmentation_profile": augmentation_profile,
                    **aug,
                }
                fout.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
                cnt += 1
            if (ti_idx + 1) % progress_every == 0 or (ti_idx + 1) == len(target_items):
                elapsed = time.time() - t0
                rate = cnt / max(elapsed, 1e-6)
                print(f"  [build_tse] {ti_idx+1}/{len(target_items)} targets "
                      f"({cnt} triples, {rate:.1f} triple/s)")
    print(f"[build_tse] 共 {cnt} 训练三元组 → {out_dir} "
          f"(耗时 {time.time()-t0:.0f}s)")
    return cnt


# ============================================================================
# Smoke 自测
# ============================================================================
def _smoke_self_test(out_dir: str, seed: int = 42, n_target: int = 2,
                     n_per_target: int = 2):
    """冒烟: 用 test_wav/zh_target_*.wav + zh_nontarget_*.wav 跑 4 条三元组。

    验证:
      - 链路不崩, 三件套全部写出
      - clean_target 与 recognition 等长 (对齐 SI-SDR)
      - enrollment ~1.5–2.5s
      - manifest 字段完整
    """
    _root = os.path.dirname(_HERE)
    t_wavs = sorted(glob.glob(os.path.join(_root, "test_wav", "zh_target_*.wav")))
    n_wavs = sorted(glob.glob(os.path.join(_root, "test_wav", "zh_nontarget_*.wav")))
    if not t_wavs or not n_wavs:
        raise SystemExit(f"[smoke] 缺测试音频: target={len(t_wavs)} "
                         f"nontarget={len(n_wavs)}")
    rng = random.Random(seed)
    smoke_refs = {
        "zh_target_01": "请把客厅的空调温度调到二十六度",
        "zh_target_02": "小美小美打开卧室的灯",
        "zh_target_03": "把电视的声音关小一点",
        "zh_target_04": "帮我定一个明天早上七点的闹钟",
    }
    # Smoke fixtures share one synthetic target voice; use different files as
    # enrollment and recognition utterances.
    target_items = [{
        "wav": w,
        "ref": smoke_refs[os.path.splitext(os.path.basename(w))[0]],
        "spk": "smoke_target",
    }
                    for w in t_wavs[:max(2, n_target)]]
    interferer_items = [{"wav": w} for w in n_wavs]
    noise_items = []
    n = build_tse_pairs(target_items, interferer_items, noise_items,
                        out_dir=out_dir, n_per_target=n_per_target, seed=seed)
    # 自检
    mpath = os.path.join(out_dir, "manifest.jsonl")
    rows = [json.loads(l) for l in open(mpath, encoding="utf-8") if l.strip()]
    print(f"\n[smoke] OK: {n} 三元组 → {out_dir}")
    for r in rows[:2]:
        e, _ = sf.read(r["enrollment_audio"])
        x, _ = sf.read(r["recognition_audio"])
        c, _ = sf.read(r["clean_target_audio"])
        print(f"  - {r['id']}: enroll={len(e)/SR:.2f}s "
              f"recog={len(x)/SR:.2f}s clean_tgt={len(c)/SR:.2f}s "
              f"(等长={len(x)==len(c)}) "
              f"overlap={r['overlap_ratio']} snr={r['snr_db']} "
              f"noise={r['noise_type']} gain={r['target_gain_db']:.1f} "
              f"speed={r['target_speed_rate']:.2f}")
    print(f"[smoke] 人工听: {out_dir}/recognition/ vs clean_target/ 验证混合/对齐")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="冒烟自测 (test_wav)")
    p_smoke.add_argument("--out", default=os.path.join(_HERE, "_tse_aug_smoke"))
    p_smoke.add_argument("--n-target", type=int, default=2)
    p_smoke.add_argument("--n-per-target", type=int, default=2)
    p_smoke.add_argument("--seed", type=int, default=42)

    p_build = sub.add_parser("build", help="批量生成 (基于 build_aishell_manifest 输出)")
    p_build.add_argument("--target-manifest", required=True,
                         help="target.jsonl (来自 build_aishell_manifest.py)")
    p_build.add_argument("--interferer-manifest", required=True,
                         help="interferer.jsonl")
    p_build.add_argument("--noise-manifest", default="",
                         help="noise.jsonl (可选)")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--n-per-target", type=int, default=10)
    p_build.add_argument(
        "--augmentation-profile",
        choices=["balanced", "hard", "legacy"],
        default="balanced",
        help=(
            "balanced keeps final measured SIR/SNR in [-5,5] dB; hard "
            "biases overlap/babble; legacy reproduces compounded quiet gain"
        ),
    )
    p_build.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "smoke":
        _smoke_self_test(args.out, seed=args.seed,
                         n_target=args.n_target, n_per_target=args.n_per_target)
    elif args.cmd == "build":
        def _load(p):
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        target_items = _load(args.target_manifest)
        interferer_items = _load(args.interferer_manifest)
        noise_items = (_load(args.noise_manifest) if args.noise_manifest else [])
        build_tse_pairs(target_items, interferer_items, noise_items,
                        out_dir=args.out, n_per_target=args.n_per_target,
                        seed=args.seed,
                        augmentation_profile=args.augmentation_profile)


if __name__ == "__main__":
    main()
