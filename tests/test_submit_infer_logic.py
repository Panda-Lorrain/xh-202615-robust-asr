import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from submit_infer import utt_id_from_path, audio_duration_s
from submit_infer import expand_inputs, load_pairs
from submit_infer import decide_reject
from submit_infer import bucket_by_atten
from submit_infer import build_result, build_timing

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

def test_build_result():
    items = [{
        "utt_id": "r1", "enrollment": "e.wav", "recognition": "r1.wav",
        "text": "你好", "rejected": False, "score": 0.30,
        "max_sim": 0.30, "llm_verdict": "accept",
        "noise_type": "white", "atten_lim_db": 0, "diar_fail": False,
    }, {
        "utt_id": "r2", "enrollment": "e.wav", "recognition": "r2.wav",
        "text": "", "rejected": True, "score": 0.05,
        "max_sim": 0.05, "llm_verdict": "reject",
        "noise_type": None, "atten_lim_db": None, "diar_fail": False,
    }]
    cfg = {"se": True, "llm": True, "strategy": "llm_or_sim", "sim_thr": 0.2, "device": "cuda:0"}
    out = build_result(items, cfg)
    assert out["task_id"] == "XH-202615"
    assert out["n_utt"] == 2
    assert out["config"] == cfg
    assert len(out["results"]) == 2
    assert out["results"][0]["text"] == "你好"
    assert out["results"][1]["rejected"] == True
    assert "generated_at" in out
    print("test_build_result OK")

def test_build_timing():
    t = build_timing(
        device="cuda:0", n_utt=2,
        total_audio_sec=10.0, total_wall_sec=3.0,
        phases={"noise_classify": {"wall_sec": 0.5}, "se": {"wall_sec": 1.0, "n": 2},
                "enroll_diar_dicow": {"wall_sec": 1.0, "mean_rtf": 0.1},
                "llm": {"wall_sec": 0.5}},
        per_utt=[{"utt_id": "r1", "audio_sec": 5.0, "wall_sec": 1.5, "rtf": 0.3}])
    assert t["device"] == "cuda:0"
    assert t["n_utt"] == 2
    assert abs(t["overall_rtf"] - 0.3) < 1e-6
    assert t["phases"]["se"]["n"] == 2
    assert len(t["per_utt"]) == 1
    print("test_build_timing OK")

if __name__ == "__main__":
    test_utt_id()
    test_audio_duration()
    test_load_pairs()
    test_expand_inputs_folder()
    test_decide_reject()
    test_bucket_by_atten()
    test_build_result()
    test_build_timing()
    print("ALL PASS")
