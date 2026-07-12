"""音频转码 + 混音工具(ffmpeg 转码 + simulate_pipeline 复用混音)。

lazy import simulate_pipeline: 转码/时长功能不依赖它, 只有 mix_* 才 import。
"""
import os, sys, subprocess, glob, random, contextlib, wave
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))      # demo_web/
_CODE = os.path.dirname(_HERE)                           # code/
if _CODE not in sys.path:
    sys.path.insert(0, _CODE)


def to_wav_16k_mono(src_path, dst_path):
    """ffmpeg 任意格式(webm/mp4/wav/m4a) → 16k mono s16 wav。失败抛 CalledProcessError。"""
    cmd = ["ffmpeg", "-y", "-i", src_path, "-vn", "-ar", "16000",
           "-ac", "1", "-sample_fmt", "s16", dst_path]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration_s(wav_path):
    """wav 时长(秒), 纯 stdlib wave。读失败返回 0.0。"""
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def _load_mono(path, sr=16000):
    w, _ = librosa.load(path, sr=sr)
    return w.astype(np.float32)


def mix_babble(test_wav, out_wav, snr_db, babble_pool):
    """test + babble 噪声(从 babble_pool/*.wav 随机采一段, 不足 tile) @ snr_db, 写到 out_wav。

    babble_pool 为空目录 → 退化白噪 fallback。snr_db 越低越吵(-5 很吵, 10 轻微)。
    """
    from simulate_pipeline import add_noise  # lazy: 隔离重 import
    audio = _load_mono(test_wav)
    wavs = sorted(glob.glob(os.path.join(babble_pool, "*.wav")))
    if not wavs:
        noise = np.random.standard_normal(len(audio)).astype(np.float32)
    else:
        nw, _ = librosa.load(random.choice(wavs), sr=16000)
        if len(nw) < len(audio):
            nw = np.tile(nw, len(audio) // len(nw) + 1)
        noise = nw[:len(audio)].astype(np.float32)
    mixed = add_noise(audio, noise, snr_db)
    sf.write(out_wav, mixed.astype(np.float32), 16000)


def mix_voice(test_wav, interferer_wav, out_wav, overlap_ratio):
    """test + 第二人声重叠 @ overlap_ratio(0~1, 1.0=完全重叠), 写到 out_wav。"""
    from simulate_pipeline import mix_overlap  # lazy
    target = _load_mono(test_wav)
    interf = _load_mono(interferer_wav)
    mixed = mix_overlap(target, interf, overlap_ratio=overlap_ratio)
    sf.write(out_wav, mixed.astype(np.float32), 16000)
