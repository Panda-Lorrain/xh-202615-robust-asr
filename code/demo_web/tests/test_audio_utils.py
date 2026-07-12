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


def test_mix_babble_duration():
    with tempfile.TemporaryDirectory() as d:
        test = os.path.join(d, "t.wav"); out = os.path.join(d, "m.wav")
        _mkwav(test, sr=16000, dur=1.0)
        audio_utils.mix_babble(test, out, snr_db=0.0, babble_pool=d)  # 空 pool → 白噪 fallback
        assert os.path.exists(out)
        d_test = audio_utils.duration_s(test); d_out = audio_utils.duration_s(out)
        assert abs(d_test - d_out) < 0.05, (d_test, d_out)  # 时长不变
    print("test_mix_babble_duration OK")


def test_mix_voice_overlap_zero():
    # overlap_ratio=0 应等于 target(simulate_pipeline.mix_overlap 语义)
    with tempfile.TemporaryDirectory() as d:
        a = os.path.join(d, "a.wav"); b = os.path.join(d, "b.wav"); out = os.path.join(d, "o.wav")
        _mkwav(a, freq=220); _mkwav(b, freq=440)
        audio_utils.mix_voice(a, b, out, overlap_ratio=0.0)
        ta, _ = sf.read(a); to_, _ = sf.read(out)
        assert len(ta) == len(to_)
        assert np.allclose(ta, to_, atol=1e-5), "overlap=0 应 == target"
    print("test_mix_voice_overlap_zero OK")


if __name__ == "__main__":
    test_to_wav_16k_mono()
    test_mix_babble_duration()
    test_mix_voice_overlap_zero()
    print("ALL OK")
