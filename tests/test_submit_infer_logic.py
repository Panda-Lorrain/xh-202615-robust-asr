import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from submit_infer import utt_id_from_path, audio_duration_s
from submit_infer import expand_inputs, load_pairs
from submit_infer import decide_reject
from submit_infer import bucket_by_atten

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

def test_load_pairs():
    import tempfile, json as _json
    d = tempfile.mkdtemp()
    pj = os.path.join(d, "pairs.json")
    _json.dump([{"enrollment": "a.wav", "recognition": "b.wav"},
                {"enrollment": "a.wav", "recognition": "c.wav"}],
               open(pj, "w"))
    pairs = load_pairs(pj)
    assert pairs == [("a.wav", "b.wav"), ("a.wav", "c.wav")], pairs
    print("test_load_pairs OK")

def test_expand_inputs_folder():
    class A: pass
    import tempfile
    d = tempfile.mkdtemp()
    for n in ("r1.wav", "r2.wav"):
        open(os.path.join(d, n), "w").close()
    a = A(); a.pairs=None; a.enrollment="e.wav"; a.recognition_folder=d
    out = expand_inputs(a)
    assert ("e.wav", os.path.join(d, "r1.wav")) in out
    assert ("e.wav", os.path.join(d, "r2.wav")) in out
    assert len(out) == 2
    print("test_expand_inputs_folder OK")

def test_decide_reject():
    assert decide_reject(0.10, "accept", "sim_only", 0.2, True) == True
    assert decide_reject(0.30, "reject", "sim_only", 0.2, True) == False
    assert decide_reject(0.05, "accept", "llm_only", 0.2, True) == False
    assert decide_reject(0.90, "reject", "llm_only", 0.2, True) == True
    assert decide_reject(0.10, "accept", "llm_or_sim", 0.2, True) == False
    assert decide_reject(0.30, "reject", "llm_or_sim", 0.2, True) == False
    assert decide_reject(0.10, "reject", "llm_or_sim", 0.2, True) == True
    assert decide_reject(0.10, "accept", "llm_or_sim", 0.2, False) == True
    assert decide_reject(0.30, "accept", "llm_or_sim", 0.2, False) == False
    print("test_decide_reject OK")

def test_bucket_by_atten():
    rows = [
        {"file": "a.wav", "atten_lim_db": 0},
        {"file": "b.wav", "atten_lim_db": 6},
        {"file": "c.wav", "atten_lim_db": 0},
        {"file": "d.wav", "atten_lim_db": 6},
    ]
    b = bucket_by_atten(rows)
    assert set(b.keys()) == {0, 6}
    assert sorted(b[0]) == ["a.wav", "c.wav"]
    assert sorted(b[6]) == ["b.wav", "d.wav"]
    assert bucket_by_atten([]) == {}
    print("test_bucket_by_atten OK")

if __name__ == "__main__":
    test_utt_id()
    test_audio_duration()
    test_load_pairs()
    test_expand_inputs_folder()
    test_decide_reject()
    test_bucket_by_atten()
    print("ALL PASS")
