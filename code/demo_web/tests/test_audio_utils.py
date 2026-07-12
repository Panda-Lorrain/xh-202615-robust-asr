"""audio_utils 裸 assert 单测。用法: cd code && .venv/Scripts/python.exe demo_web/tests/test_audio_utils.py"""
import os, sys, tempfile, contextlib, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # demo_web/
import numpy as np
import soundfile as sf
import audio_utils


def _mkwav(path, sr=16000, dur=1.0, freq=220):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    w = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, w, sr)


def test_to_wav_16k_mono():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.wav"); dst = os.path.join(d, "out.wav")
        _mkwav(src, sr=44100, dur=1.0)  # 故意 44.1k → 转码有意义
        audio_utils.to_wav_16k_mono(src, dst)
        assert os.path.exists(dst), "dst 未生成"
        with contextlib.closing(wave.open(dst, "rb")) as w:
            assert w.getframerate() == 16000, f"sr={w.getframerate()}"
            assert w.getnchannels() == 1, f"ch={w.getnchannels()}"
            assert w.getsampwidth() == 2, f"sw={w.getsampwidth()}"  # s16=2字节
    print("test_to_wav_16k_mono OK")


if __name__ == "__main__":
    test_to_wav_16k_mono()
    print("ALL OK")
