import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import numpy as np
from text_utils import to_simplified, cut_target_timeline

def test_to_simplified():
    assert to_simplified("空調開到二十六度") == "空调开到二十六度"
    assert to_simplified("") == ""
    assert to_simplified(None) is None
    assert to_simplified("已经简体") == "已经简体"
    print("test_to_simplified OK")

def test_cut_basic():
    sr = 16000
    audio = np.ones(sr * 10, dtype=np.float32)
    out = cut_target_timeline(audio, [(1.0, 3.0), (5.0, 7.0)], sr=sr)
    assert len(out) == sr * 4
    print("test_cut_basic OK")

def test_cut_keeps_overlap_region():
    sr = 16000
    audio = np.arange(sr * 5, dtype=np.float32)
    out = cut_target_timeline(audio, [(0.0, 2.0), (3.0, 5.0)], sr=sr)
    assert len(out) == sr * 4
    assert out[0] == audio[0]
    assert out[sr * 2] == audio[sr * 3]  # 段2 拼接到 2s 位置
    print("test_cut_keeps_overlap_region OK")

def test_cut_too_short_fallback():
    sr = 16000
    audio = np.ones(sr * 5, dtype=np.float32)
    out = cut_target_timeline(audio, [(0.0, 0.1)], sr=sr)  # 0.1s < 0.3s
    assert len(out) == sr * 5  # 退化整条
    print("test_cut_too_short_fallback OK")

def test_cut_empty_fallback():
    sr = 16000
    audio = np.ones(sr * 3, dtype=np.float32)
    assert len(cut_target_timeline(audio, [], sr=sr)) == sr * 3
    print("test_cut_empty_fallback OK")

if __name__ == "__main__":
    test_to_simplified()
    test_cut_basic()
    test_cut_keeps_overlap_region()
    test_cut_too_short_fallback()
    test_cut_empty_fallback()
    print("ALL PASS")
