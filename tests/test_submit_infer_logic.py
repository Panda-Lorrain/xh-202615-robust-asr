import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from submit_infer import utt_id_from_path, audio_duration_s

def test_utt_id():
    assert utt_id_from_path("E:/x/rec_001.wav") == "rec_001"
    assert utt_id_from_path("rec_002.WAV") == "rec_002"
    assert utt_id_from_path("/a/b/c.wav") == "c"
    print("test_utt_id OK")

def test_audio_duration():
    p = "E:/midea_target_asr/test_wav/zh_target_01.wav"
    if os.path.exists(p):
        d = audio_duration_s(p)
        assert d > 0, f"duration should be >0, got {d}"
        print(f"test_audio_duration OK ({d:.2f}s)")
    else:
        print("test_audio_duration SKIP (no fixture)")

if __name__ == "__main__":
    test_utt_id()
    test_audio_duration()
    print("ALL PASS")
