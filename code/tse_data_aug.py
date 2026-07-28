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

本脚本复用 data_aug_recipe.synthesize_one 的所有预处理/混合逻辑,
额外把处理后的 target 波形 (mix_overlap 之前) 保存为 clean_target.wav。

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
  out_dir/recognition/<uid>.wav  (= enroll_dur 到 target 全长, mono 16kHz)
  out_dir/clean_target/<uid>.wav (= 与 recognition 等长的 target 波形, 含小声/快预处理, 用于 SI-SDR 监督)
  out_dir/manifest.jsonl         (路径 + 文本 + 增广参数)

⚠️ recognition 与 clean_target 等长, 用 mix_overlap 的输出长度对齐
   (mix_overlap 取 min(target, interferer) → 可能短于 target 全长,
    但 SI-SDR 只需对应段等长, 不影响训练)。
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import random
import argparse
from dataclasses import asdict
from typing import List, Dict, Optional

import numpy as np
import librosa
import soundfile as sf

# 复用项目内增广工具
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from data_aug_recipe import (  # noqa: E402
    sample_aug_params, make_quiet, make_fast, cut_enrollment,
    pollute_enrollment, AugParams, SR,
)
from simulate_pipeline import mix_overlap, add_noise, _fit_noise  # noqa: E402
from build_dataset import gen_white, gen_pink, gen_babble  # noqa: E402


def synthesize_tse_triple(
    target_wav: np.ndarray,
    interferer_wav: np.ndarray,
    noise_wav: Optional[np.ndarray],
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
    rng: Optional[random.Random] = None,
):
    """合成一条 TSE 训练三元组: (enrollment, recognition, clean_target, params)。

    与 data_aug_recipe.synthesize_one 的区别: 额外返回 clean_target
    (= 小声/快预处理后的 target 波形, 未混合干扰和噪声, 作 SI-SDR 监督)。

    Returns:
        enrollment_audio: np.float32 [T_enroll]
        recognition_audio: np.float32 [T_mix]
        clean_target_audio: np.float32 [T_mix]  # 与 recognition 等长
        params: AugParams
    """
    rng = rng or random.Random()
    nprng = np.random.default_rng(rng.randrange(2**31))

    # 1) target 预处理 (小声 + 快语速)
    t = target_wav.astype(np.float32)
    if target_gain_db != 0.0:
        t = make_quiet(t, target_gain_db, rng)
    if target_speed_rate != 1.0:
        t = make_fast(t, target_speed_rate)

    # 2) 切 enrollment (从预处理后的 t 切, 与 recognition 中 target 同风格)
    enroll = cut_enrollment(t, enroll_dur_sec, SR, rng)

    # 3) enrollment 污染 (可选)
    if rng.random() < enroll_pollute_p and noise_wav is not None:
        enroll = pollute_enrollment(enroll, noise_wav,
                                     enroll_pollute_snr_db, rng, pollute_p=1.0)

    # 4) 重叠混合 → recognition 的目标段长 = min(len(t), len(interferer))
    mixed = mix_overlap(t, interferer_wav, overlap_ratio)  # [T_mix]
    n = len(mixed)

    # 5) clean_target 对齐: 取 t 前 n 个样本 (= recognition 中 target 的贡献)
    clean_target = np.zeros(n, dtype=np.float32)
    copy_len = min(len(t), n)
    clean_target[:copy_len] = t[:copy_len]

    # 6) 加噪 → recognition
    if noise_type == "env" and noise_wav is not None:
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
    recognition = add_noise(mixed, noise, snr_db).astype(np.float32)

    params = AugParams(
        overlap_ratio=overlap_ratio, snr_db=snr_db, noise_type=noise_type,
        target_gain_db=target_gain_db, target_speed_rate=target_speed_rate,
        enroll_dur_sec=enroll_dur_sec,
        enroll_pollute=(enroll_pollute_p > 0 and rng.random() < enroll_pollute_p),
        enroll_pollute_snr_db=enroll_pollute_snr_db,
    )
    return (enroll.astype(np.float32),
            recognition.astype(np.float32),
            clean_target,
            params)


def build_tse_pairs(
    target_items: List[Dict],
    interferer_items: List[Dict],
    noise_items: List[Dict],
    out_dir: str,
    n_per_target: int = 10,
    seed: int = 42,
    progress_every: int = 5,
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
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    cnt = 0
    t0 = time.time()

    # 预加载干扰/噪声池 (小, 一次性)
    nontarget_pool = []
    for ii in interferer_items:
        try:
            w, _ = librosa.load(ii["wav"], sr=SR)
            nontarget_pool.append(w)
        except Exception as e:
            print(f"  [warn] 跳过干扰 {ii['wav']}: {e}")
    if not nontarget_pool:
        nontarget_pool = [np.zeros(SR * 2, dtype=np.float32)]
    noise_pool = []
    for ni in noise_items:
        try:
            w, _ = librosa.load(ni["wav"], sr=SR)
            noise_pool.append(w)
        except Exception:
            pass

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for ti_idx, ti in enumerate(target_items):
            try:
                t_wav, _ = librosa.load(ti["wav"], sr=SR)
            except Exception as e:
                print(f"  [warn] 跳过 target {ti['wav']}: {e}")
                continue
            t_ref = ti.get("ref", "")
            for k in range(n_per_target):
                params = sample_aug_params(rng)
                interferer_wav = nontarget_pool[rng.randrange(len(nontarget_pool))]
                noise_wav = (noise_pool[rng.randrange(len(noise_pool))]
                             if noise_pool else None)
                enroll, recog, clean_tgt, aug = synthesize_tse_triple(
                    t_wav, interferer_wav, noise_wav,
                    nontarget_pool=nontarget_pool, rng=rng, **params,
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
                    **asdict(aug),
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
    from data_aug_recipe import sample_home_cmd
    target_items = [{"wav": w, "ref": sample_home_cmd(rng)}
                    for w in t_wavs[:n_target]]
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
                        seed=args.seed)


if __name__ == "__main__":
    main()
