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
