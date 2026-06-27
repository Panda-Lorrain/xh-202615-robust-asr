"""W2 数据仿真 pipeline：造中文多人重叠 + SNR −5~5dB 仿真集（题目分布）。
- add_noise(audio, noise, snr_db): 按目标 SNR(dB) 加噪
- mix_overlap(target, interferer, overlap_ratio): 重叠混合（overlap_ratio 0~1）
- simulate(target_wav, interferer_wav, noise_wav, snr_db, overlap_ratio): 完整仿真一条
- build_set(src_dir, noise_wav, ...): 批量造仿真集（真实单人中文音频来时用）

真实单人音频来源：AISHELL-1/2/3、WenetSpeech、MagicData-RAMC、或关代理后 edge-tts 合成。
"""
import os, glob, numpy as np, librosa, soundfile as sf


def add_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按目标 SNR(dB) 把 noise 混入 audio。snr_db 越低噪声越大（−5dB 很吵）。"""
    p_audio = np.mean(audio ** 2) + 1e-10
    p_noise_target = p_audio / (10 ** (snr_db / 10))
    p_noise = np.mean(noise ** 2) + 1e-10
    scale = np.sqrt(p_noise_target / p_noise)
    return audio + scale * noise


def mix_overlap(target: np.ndarray, interferer: np.ndarray, overlap_ratio: float = 1.0) -> np.ndarray:
    """把 interferer 与 target 重叠。overlap_ratio=1.0 完全重叠(题面100%)，0 无重叠。"""
    n = min(len(target), len(interferer))
    overlap_len = int(n * overlap_ratio)
    out = target[:n].copy().astype(np.float32)
    out[:overlap_len] += interferer[:overlap_len].astype(np.float32)
    return out


def _fit_noise(noise: np.ndarray, n: int) -> np.ndarray:
    if len(noise) < n:
        reps = n // len(noise) + 1
        noise = np.tile(noise, reps)[:n]
    else:
        noise = noise[:n]
    return noise


def simulate(target_wav, interferer_wav, noise_wav, snr_db=0.0, overlap_ratio=1.0, sr=16000):
    target, _ = librosa.load(target_wav, sr=sr)
    interferer, _ = librosa.load(interferer_wav, sr=sr)
    noise, _ = librosa.load(noise_wav, sr=sr)
    mixed = mix_overlap(target, interferer, overlap_ratio)
    noise = _fit_noise(noise, len(mixed))
    final = add_noise(mixed, noise, snr_db)
    return final, sr


def build_set(src_dir, noise_wav, out_dir, snr_list=(-5, 0, 5), overlap_list=(0.0, 0.5, 1.0), sr=16000):
    """批量：src_dir 单人 wav 两两配对造重叠 + 多 SNR/重叠率。"""
    os.makedirs(out_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    noise, _ = librosa.load(noise_wav, sr=sr)
    cnt = 0
    for i in range(len(wavs)):
        for j in range(len(wavs)):
            if i == j:
                continue
            for snr in snr_list:
                for ov in overlap_list:
                    final, _ = simulate(wavs[i], wavs[j], noise_wav, snr, ov, sr)
                    name = f"t{i:03d}_int{j:03d}_snr{snr}_ov{int(ov*100):03d}.wav"
                    sf.write(os.path.join(out_dir, name), final, sr)
                    cnt += 1
    print(f"[build_set] 生成 {cnt} 条仿真音频 → {out_dir}")
    return cnt


if __name__ == "__main__":
    print("=== W2 仿真 pipeline 自测（生成信号验证混合/加噪逻辑）===")
    sr = 16000
    t = np.arange(sr * 3) / sr
    target = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)      # 目标 220Hz
    interferer = 0.5 * np.sin(2 * np.pi * 330 * t).astype(np.float32)  # 干扰 330Hz
    noise = (np.random.randn(sr * 3) * 0.1).astype(np.float32)
    mixed = mix_overlap(target, interferer, overlap_ratio=1.0)
    noisy_0db = add_noise(mixed, noise, snr_db=0)
    noisy_neg5 = add_noise(mixed, noise, snr_db=-5)
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _test_wav = os.path.join(_proj_root, "test_wav")
    os.makedirs(_test_wav, exist_ok=True)
    sf.write(os.path.join(_test_wav, "sim_test_0db.wav"), noisy_0db, sr)
    sf.write(os.path.join(_test_wav, "sim_test_neg5db.wav"), noisy_neg5, sr)
    p_tar = np.mean(target ** 2)
    p_mix = np.mean(mixed ** 2)
    print(f"  target 功率={p_tar:.4f}, 重叠后 mixed 功率={p_mix:.4f}（≈2x，重叠叠加）")
    print(f"  写出 sim_test_0db.wav / sim_test_neg5db.wav（3s, 220+330Hz 重叠 + 白噪）")
    print("  [说明] 真实单人中文音频(AISHELL/TTS)→ simulate()/build_set() 造重叠带噪集")
