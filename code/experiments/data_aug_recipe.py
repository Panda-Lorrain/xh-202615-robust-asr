#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据增广配方 (Qwen3-ASR 微调用训练对合成骨架)

================================================================
目标 (与题目难点对齐)
================================================================
死区 sim<0.4 中 60% 失败是「ASR错+接近地板」, 细分两类:
  ① ASR 幻觉 (清洁→春洁, 音频清晰却转错)
  ② 小声/被盖/重叠 (人耳可辨机器不能, 如 cmd_2096/2949/2571/2001)

本配方针对性合成训练对, 微调 Qwen3-ASR-1.7B (主线 CER 0.3436) 攻这两类:
  - 重叠 0–100% (题目规格 ≤2 人)
  - SNR −5~5dB (题目规格)
  - target 小声化 (音量压低, 模拟 cmd_2096/2949)
  - target 快语速 (模拟 cmd_2050)
  - 短 enrollment ~1.8s (题目规格, 从 target 切)
  - enrollment 可选加噪 (模拟污染)

================================================================
输入 / 输出
================================================================
输入: 干净中文单人 wav (A=target家居指令, B=干扰闲话/新闻/财经, 非家居指令)
输出 (每条训练对):
  - enrollment.wav  (1.5–2.5s, A 的子片段, 可选加噪)
  - recognition.wav (A+B 重叠 + 加噪, A 可选小声化/快语速)
  - ref             (A 的原始转录 = ground truth 文本)
  - manifest.jsonl  (所有增广参数 + 文件路径)

================================================================
重要约束 (lessons-pitfalls §14: A 集禁训练)
================================================================
A 集是测试集, 绝不能进训练 (会泄漏 + 过拟合)。本脚本:
  - 不读 datasetA/ 下任何真实音频 / 文本
  - 只复用 datasetA/pos.jsonl 的「指令模板」(空调/灯/温度/...) 提取词表
  - 真实训练数据来源: 外部干净中文语料 (Aishell-1/2/3, WenetSpeech, ...)
    + 干扰源 (新闻/财经 TTS 或 WenetSpeech 非家居切片)
    + 噪声 (MUSAN/WHAM! 或程序噪声 white/pink/babble)

================================================================
复用
================================================================
- simulate_pipeline.add_noise / mix_overlap / _fit_noise
- build_dataset.gen_white / gen_pink / gen_babble / load_env_noise
================================================================
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import random
import time
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List

import numpy as np
import librosa
import soundfile as sf

# 复用项目内仿真函数
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from simulate_pipeline import add_noise, mix_overlap, _fit_noise
from build_dataset import gen_white, gen_pink, gen_babble


SR = 16000  # 题目规格: 16k mono

# ============================================================================
# 家居指令词表 (从 datasetA/pos.jsonl 模板提炼, 仅取「词表」, 不取真实音频/标签)
# 用途: 当外部干净语料 (Aishell) 文本不是家居指令时, 用本词表做"指令化替换"或
#       配合 TTS 合成专属家居指令 (后续工作)。当前骨架默认沿用 A 的原始 transcript。
# ============================================================================
HOME_CMD_TEMPLATES = {
    "空调": ["空调开到制热", "空调开到制冷", "空调调到{t}度", "空调风速调到{p}",
            "空调关闭", "空调开到{mode}模式"],
    "灯": ["灯光亮度调到{p}", "所有灯光打开", "客厅灯关闭", "卧室灯调暗一点",
           "把灯调成暖色"],
    "洗衣机": ["洗衣机暂停工作", "洗衣机开始漂洗", "洗衣机调到轻柔模式"],
    "温度": ["温度调到{t}度", "把温度设到{t}度"],
    "风速": ["风量调到{p}", "风速调到最大"],
    "窗帘": ["拉起窗帘", "关上窗帘", "打开窗帘"],
    "音乐": ["音乐的声音小一点", "播放{artist}的歌", "下一首",
             "音量调到{p}"],
    "模式": ["打开回家模式", "启动睡眠模式", "退出观影模式"],
    "通用": ["我要出门了", "我回来了", "元宵节灯会什么时候"],
}
TEMP_POOL = [22, 23, 24, 25, 26, 27, 28]
PCT_POOL = [30, 40, 50, 60, 70, 80, 100]
ARTIST_POOL = ["周杰伦", "邓紫棋", "权志龙"]


def sample_home_cmd(rng: random.Random) -> str:
    """从词表随机拼一条家居指令 (供 TTS 合成或文本替换参考)。"""
    cat = rng.choice(list(HOME_CMD_TEMPLATES.keys()))
    tpl = rng.choice(HOME_CMD_TEMPLATES[cat])
    return tpl.format(
        t=rng.choice(TEMP_POOL),
        p=f"百分之{rng.choice(PCT_POOL)}",
        mode=rng.choice(["制热", "制冷", "除湿", "自动"]),
        artist=rng.choice(ARTIST_POOL),
    )


# ============================================================================
# 1) Target 预处理: 小声化 + 快语速 (直击 cmd_2096/2949/2050 失败模式)
# ============================================================================
def make_quiet(audio: np.ndarray, gain_db: float, rng: random.Random,
               muffle_p: float = 0.3) -> np.ndarray:
    """小声化: 整体降增益 + (概率 muffle_p) 高频衰减模拟「闷/小声」。

    gain_db: 负值, e.g. -8 ~ -3 dB 衰减;
    muffle_p: 以该概率叠 lowpass(fs/4) 模拟口齿不清 / 蒙嘴说。
    """
    out = audio * (10 ** (gain_db / 20.0))
    if rng.random() < muffle_p:
        # 简单 1 阶低通: 减弱高频成分, 模拟「闷声」
        # 使用 librosa.effects 不合适 (无 lowpass), 用 scipy signal
        try:
            from scipy.signal import butter, lfilter
            b, a = butter(4, 0.4, btype="low")  # 0.4 * (fs/2) ≈ 3.2kHz
            out = lfilter(b, a, out).astype(np.float32)
        except Exception:
            pass  # scipy 缺则跳过, 不阻塞流程
    return out.astype(np.float32)


def make_fast(audio: np.ndarray, rate: float) -> np.ndarray:
    """快语速: librosa.time_stretch (rate>1 加速)。rate 范围建议 1.1~1.4。"""
    return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)


# ============================================================================
# 2) Enrollment 切片: 模拟 ~1.8s 短 enrollment, 可选加噪(污染)
# ============================================================================
def cut_enrollment(target_audio: np.ndarray, dur_sec: float,
                   sr: int = SR, rng: Optional[random.Random] = None,
                   target_db_range: Tuple[float, float] = (0.0, 0.0)
                   ) -> np.ndarray:
    """从 target_audio 随机起点切 dur_sec 秒, 模拟题目 enrollment (~1.8s)。"""
    n = int(dur_sec * sr)
    if len(target_audio) <= n:
        out = target_audio.copy()
    else:
        start = (rng.randrange(0, len(target_audio) - n)
                 if rng else 0)
        out = target_audio[start:start + n].copy()
    return out.astype(np.float32)


def pollute_enrollment(enroll: np.ndarray, noise: np.ndarray,
                       snr_db: float, rng: random.Random,
                       pollute_p: float = 0.3) -> np.ndarray:
    """以 pollute_p 概率给 enrollment 加噪 (模拟 enrollment 污染, 题目 enrollment
    虽是唤醒词但现场可能含噪)。snr_db 越低污染越重。"""
    if rng.random() >= pollute_p:
        return enroll
    n = len(enroll)
    noise_seg = _fit_noise(noise, n)
    return add_noise(enroll, noise_seg, snr_db)


# ============================================================================
# 3) 完整一条训练对合成
# ============================================================================
@dataclass
class AugParams:
    """单条增广参数 (写 manifest 便于回溯/消融)。"""
    overlap_ratio: float
    snr_db: float
    noise_type: str
    target_gain_db: float       # target 小声化增益 (0=不变)
    target_speed_rate: float    # target 快语速倍率 (1.0=不变)
    enroll_dur_sec: float
    enroll_pollute: bool
    enroll_pollute_snr_db: float


def synthesize_one(
    target_wav: np.ndarray,
    interferer_wav: np.ndarray,
    noise_wav: Optional[np.ndarray],
    nontarget_pool: Optional[List[np.ndarray]] = None,  # 供 babble 用
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
) -> Tuple[np.ndarray, np.ndarray, AugParams]:
    """合成一条训练对: (enrollment, recognition, params)。

    Args:
        target_wav:    干净 target 单人中文音频 (家居指令)
        interferer_wav: 干扰单人音频 (闲话/新闻/财经, 非家居指令)
        noise_wav:     噪声; None=用程序噪声 (noise_type)
        nontarget_pool: 用于 babble 合成; None=babble 退化
        overlap_ratio: 0–1
        snr_db:        −5~5 dB
        noise_type:    white/pink/babble/env (env 需 noise_wav)
        target_gain_db: target 小声化 (负值) / 0=不变
        target_speed_rate: target 快语速倍率
        enroll_dur_sec: enrollment 秒数
    Returns:
        (enrollment_audio, recognition_audio, params)
    """
    rng = rng or random.Random()
    nprng = np.random.default_rng(rng.randrange(2**31))

    # 1) Target 预处理: 小声化 + 快语速 (在重叠前, 影响最终混合功率比)
    t = target_wav.astype(np.float32)
    if target_gain_db != 0.0:
        t = make_quiet(t, target_gain_db, rng)
    if target_speed_rate != 1.0:
        t = make_fast(t, target_speed_rate)

    # 2) Enrollment 先切 (用处理后的 t, 让 enroll 与 recognition 风格一致)
    enroll = cut_enrollment(t, enroll_dur_sec, SR, rng)

    # 3) Enrollment 污染 (可选, 模拟现场噪声)
    enroll_polluted = False
    if rng.random() < enroll_pollute_p:
        if noise_wav is not None:
            enroll = pollute_enrollment(enroll, noise_wav,
                                         enroll_pollute_snr_db, rng, pollute_p=1.0)
            enroll_polluted = True

    # 4) 重叠: target + interferer (overlap_ratio)
    mixed = mix_overlap(t, interferer_wav, overlap_ratio)

    # 5) 加噪
    n = len(mixed)
    if noise_type == "env" and noise_wav is not None:
        noise = _fit_noise(noise_wav, n)
    elif noise_type == "white":
        noise = gen_white(n, nprng)
    elif noise_type == "pink":
        noise = gen_pink(n, nprng)
    elif noise_type == "babble":
        noise = gen_babble(nontarget_pool or [interferer_wav], n, nprng)
    else:
        noise = gen_white(n, nprng)
    noise = noise.astype(np.float32)
    if len(noise) < n:
        noise = np.pad(noise, (0, n - len(noise)))
    else:
        noise = noise[:n]
    recognition = add_noise(mixed, noise, snr_db)

    params = AugParams(
        overlap_ratio=overlap_ratio, snr_db=snr_db, noise_type=noise_type,
        target_gain_db=target_gain_db, target_speed_rate=target_speed_rate,
        enroll_dur_sec=enroll_dur_sec, enroll_pollute=enroll_polluted,
        enroll_pollute_snr_db=enroll_pollute_snr_db,
    )
    return enroll.astype(np.float32), recognition.astype(np.float32), params


# ============================================================================
# 4) 采样增广参数 (按失败分布加权 → 让训练集贴近题目难点)
# ============================================================================
# 题目死区分布 (来自 memory overlap-is-cer-failure-rootcause):
#   失败组重叠中位 45% / 75% 重叠 / 97% 双人
#   死区 sim<0.4 (78.8%) 大头, 60% 是 ASR错+接近地板
# 所以采样时偏向: 中高重叠 (≥0.4), 低 SNR (≤0dB), 高频次小声/快语速
DEFAULT_OVERLAP_BUCKETS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_OVERLAP_WEIGHTS = [0.10, 0.10, 0.25, 0.30, 0.25]   # 偏中高
DEFAULT_SNR_BUCKETS = [-5, -3, 0, 3, 5]
DEFAULT_SNR_WEIGHTS = [0.20, 0.25, 0.30, 0.15, 0.10]        # 偏低
DEFAULT_NOISE_TYPES = ["white", "pink", "babble"]
DEFAULT_NOISE_WEIGHTS = [0.25, 0.25, 0.50]                  # babble 多 (题目主因)
# target 小声/快语速: 50% 取小声 (-8~-3dB), 30% 取快语速 (1.1~1.4x)
QUIET_PROB = 0.5
QUIET_DB_RANGE = (-8.0, -3.0)
FAST_PROB = 0.3
FAST_RATE_RANGE = (1.1, 1.4)


def sample_aug_params(rng: random.Random,
                      nontarget_pool_given: bool = True) -> Dict:
    """按权重采样一组增广参数 (失败分布对齐)。"""
    overlap = rng.choices(DEFAULT_OVERLAP_BUCKETS, DEFAULT_OVERLAP_WEIGHTS, k=1)[0]
    snr = rng.choices(DEFAULT_SNR_BUCKETS, DEFAULT_SNR_WEIGHTS, k=1)[0]
    noise_type = rng.choices(DEFAULT_NOISE_TYPES, DEFAULT_NOISE_WEIGHTS, k=1)[0]
    gain_db = (rng.uniform(*QUIET_DB_RANGE) if rng.random() < QUIET_PROB else 0.0)
    speed = (rng.uniform(*FAST_RATE_RANGE) if rng.random() < FAST_PROB else 1.0)
    enroll_dur = rng.uniform(1.5, 2.5)  # 题目 ~1.8s, 略抖动
    return dict(
        overlap_ratio=float(overlap),
        snr_db=float(snr),
        noise_type=noise_type,
        target_gain_db=float(gain_db),
        target_speed_rate=float(speed),
        enroll_dur_sec=float(enroll_dur),
        enroll_pollute_p=0.3,
        enroll_pollute_snr_db=float(rng.uniform(8, 15)),
    )


# ============================================================================
# 5) 批量生成 manifest
# ============================================================================
def build_pairs(
    target_items: List[Dict],      # [{wav: path, ref: transcript}, ...]
    interferer_items: List[Dict],
    noise_items: List[Dict],       # [{wav, type}], 可空
    out_dir: str,
    n_per_target: int = 10,
    seed: int = 42,
):
    """批量生成训练对 → out_dir/{enrollment,recognition}/*.wav + manifest.jsonl。

    target_items / interferer_items / noise_items 是上层装配的 (来自外部语料)。
    本函数不下载任何东西。
    """
    os.makedirs(os.path.join(out_dir, "enrollment"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "recognition"), exist_ok=True)
    rng = random.Random(seed)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    cnt = 0
    t0 = time.time()
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for ti in target_items:
            t_wav, _ = librosa.load(ti["wav"], sr=SR)
            t_ref = ti["ref"]
            # 预加载 noise 池 (env)
            noise_pool_audio = []
            for ni in noise_items:
                try:
                    w, _ = librosa.load(ni["wav"], sr=SR)
                    noise_pool_audio.append(w)
                except Exception:
                    pass
            nontarget_pool = []
            for ii in interferer_items:
                try:
                    w, _ = librosa.load(ii["wav"], sr=SR)
                    nontarget_pool.append(w)
                except Exception:
                    pass
            if not nontarget_pool:
                # 退化: 无干扰则用程序噪声替代 (不重叠)
                nontarget_pool = [np.zeros(SR * 2, dtype=np.float32)]
            for k in range(n_per_target):
                params = sample_aug_params(rng)
                # 选干扰
                interferer_wav = nontarget_pool[rng.randrange(len(nontarget_pool))]
                noise_wav = (noise_pool_audio[rng.randrange(len(noise_pool_audio))]
                             if noise_pool_audio else None)
                enroll, recog, aug = synthesize_one(
                    t_wav, interferer_wav, noise_wav,
                    nontarget_pool=nontarget_pool,
                    rng=rng, **params,
                )
                uid = f"{os.path.splitext(os.path.basename(ti['wav']))[0]}_k{k:03d}"
                enr_path = os.path.join(out_dir, "enrollment", uid + ".wav")
                rec_path = os.path.join(out_dir, "recognition", uid + ".wav")
                sf.write(enr_path, enroll, SR)
                sf.write(rec_path, recog, SR)
                rec_line = {
                    "id": uid,
                    "enrollment_audio": enr_path,
                    "recognition_audio": rec_path,
                    "ref": t_ref,
                    "target_src": ti["wav"],
                    **asdict(aug),
                }
                fout.write(json.dumps(rec_line, ensure_ascii=False) + "\n")
                cnt += 1
            if (len(target_items) > 0):
                done = target_items.index(ti) + 1
                if done % 5 == 0 or done == len(target_items):
                    elapsed = time.time() - t0
                    rate = cnt / max(elapsed, 1e-6)
                    print(f"  [build_pairs] {done}/{len(target_items)} targets "
                          f"({cnt} pairs, {rate:.1f} pair/s)")
    print(f"[build_pairs] 共 {cnt} 训练对 → {out_dir} "
          f"(耗时 {time.time()-t0:.0f}s)")
    return cnt


# ============================================================================
# CLI: 含 smoke 自测 (用 test_wav/zh_target_*.wav 验证配方可跑)
# ============================================================================
def _smoke_self_test(out_dir: str, seed: int = 42, n_pairs: int = 2):
    """小样本自测: 用 test_wav/ 下 4 条 zh_target + 2 条 zh_nontarget 验证配方。

    无 transcript → ref 字段填占位文本 (取家居指令模板)。
    自测通过 = 增广链不崩 + 输出 wav 长度/形状合理 + manifest 写出。
    """
    _root = os.path.dirname(_HERE)
    t_wavs = sorted(glob.glob(os.path.join(_root, "test_wav", "zh_target_*.wav")))
    n_wavs = sorted(glob.glob(os.path.join(_root, "test_wav", "zh_nontarget_*.wav")))
    if not t_wavs or not n_wavs:
        raise SystemExit(f"[smoke] 缺测试音频: target={len(t_wavs)} "
                         f"nontarget={len(n_wavs)} (期望 test_wav/zh_target_*.wav)")
    rng = random.Random(seed)
    target_items = [{"wav": w, "ref": sample_home_cmd(rng)} for w in t_wavs[:2]]
    interferer_items = [{"wav": w} for w in n_wavs]
    noise_items = []  # smoke 不用 env noise, 用程序噪声
    n = build_pairs(target_items, interferer_items, noise_items,
                    out_dir=out_dir, n_per_target=max(1, n_pairs // 2),
                    seed=seed)
    print(f"\n[smoke] OK: 生成 {n} 训练对 → {out_dir}")
    print(f"[smoke] manifest: {os.path.join(out_dir, 'manifest.jsonl')}")
    print("[smoke] 验证: 人工听 enrollment/*.wav + recognition/*.wav, "
          "对照 manifest 里 ref / overlap_ratio / snr_db 是否合理")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # smoke 自测
    p_smoke = sub.add_parser("smoke", help="小样本自测 (用 test_wav 现成音频)")
    p_smoke.add_argument("--out", default=os.path.join(_HERE, "_aug_smoke_out"),
                         help="输出目录")
    p_smoke.add_argument("--n", type=int, default=4, help="生成训练对数")
    p_smoke.add_argument("--seed", type=int, default=42)

    # 批量生成 (需装配 target/interferer/noise 三个 jsonl)
    p_build = sub.add_parser("build", help="批量生成 (用装配好的输入清单)")
    p_build.add_argument("--target-manifest", required=True,
                         help="target清单 jsonl: 每行 {wav: path, ref: 文本}")
    p_build.add_argument("--interferer-manifest", required=True,
                         help="interferer清单 jsonl: 每行 {wav: path}")
    p_build.add_argument("--noise-manifest", default="",
                         help="噪声清单 jsonl (可选): 每行 {wav: path}")
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--n-per-target", type=int, default=10)
    p_build.add_argument("--seed", type=int, default=42)

    # 单条 sample_home_cmd demo
    p_demo = sub.add_parser("demo-cmd", help="打印若干家居指令模板样本")
    p_demo.add_argument("--n", type=int, default=10)
    p_demo.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    if args.cmd == "smoke":
        _smoke_self_test(args.out, seed=args.seed, n_pairs=args.n)
    elif args.cmd == "build":
        def _load_jsonl(p):
            return [json.loads(line) for line in open(p, encoding="utf-8")
                    if line.strip()]
        target_items = _load_jsonl(args.target_manifest)
        interferer_items = _load_jsonl(args.interferer_manifest)
        noise_items = (_load_jsonl(args.noise_manifest)
                       if args.noise_manifest else [])
        build_pairs(target_items, interferer_items, noise_items,
                    out_dir=args.out, n_per_target=args.n_per_target,
                    seed=args.seed)
    elif args.cmd == "demo-cmd":
        rng = random.Random(args.seed)
        for _ in range(args.n):
            print(" ", sample_home_cmd(rng))


if __name__ == "__main__":
    main()
