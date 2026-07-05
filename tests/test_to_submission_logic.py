import os, sys, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from to_submission import convert, _utt_id_stripped

def test_utt_id_stripped():
    assert _utt_id_stripped("utt0012_cmd_3.wav") == "cmd_3"
    assert _utt_id_stripped("cmd_5.wav") == "cmd_5"
    print("test_utt_id_stripped OK")

def _write(d, name, obj):
    p = os.path.join(d, name); json.dump(obj, open(p, "w")); return p

def test_convert_pos_accept():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "utt0001_cmd_10.wav", "text": "空調開到二十六度", "rejected": False}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_10.wav",
                                   "ref": "空调开到二十六度"}])
    sub = convert(rj, pj, duration_infer_sec=12.3)
    row = sub["result"]["results"][0]
    assert row["id"] == "cmd_10"
    assert row["content"] == "空调开到二十六度"  # 繁简归一
    assert row["label"] == "accept"
    assert row["cer"] == 0.0  # 完美匹配(_norm_zh 含繁简)
    assert sub["result"]["final_cer"] == 0.0
    assert sub["result"]["duration"] == 12.3
    print("test_convert_pos_accept OK")

def test_convert_pos_rejected():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "cmd_1.wav", "text": "", "rejected": True}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_1.wav",
                                   "ref": "打开空调"}])
    sub = convert(rj, pj)
    row = sub["result"]["results"][0]
    assert row["content"] == ""
    assert row["label"] == "reject"
    assert row["cer"] == 1.0
    assert sub["result"]["final_cer"] == 1.0
    print("test_convert_pos_rejected OK")

def test_convert_neg():
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "cmd_2.wav", "text": "", "rejected": True}]})
    pj = _write(d, "pairs.json", [{"enrollment": "e.wav", "recognition": "cmd_2.wav",
                                   "ref": ""}])  # neg ref 空
    sub = convert(rj, pj)
    row = sub["result"]["results"][0]
    assert row["label"] == "reject"
    assert row["cer"] == ""  # neg 不评 CER 评 RR
    print("test_convert_neg OK")

def test_convert_duration_from_infer_sec():
    # duration 不显式传时, 从 per-utt infer_sec 累加
    d = tempfile.mkdtemp()
    rj = _write(d, "result.json", {"results": [
        {"recognition": "cmd_1.wav", "text": "开灯", "rejected": False, "infer_sec": 1.5},
        {"recognition": "cmd_2.wav", "text": "", "rejected": True, "infer_sec": 0.5}]})
    pj = _write(d, "pairs.json", [
        {"enrollment": "e.wav", "recognition": "cmd_1.wav", "ref": "开灯"},
        {"enrollment": "e.wav", "recognition": "cmd_2.wav", "ref": ""}])
    sub = convert(rj, pj)
    assert sub["result"]["duration"] == 2.0  # 1.5 + 0.5
    print("test_convert_duration_from_infer_sec OK")

if __name__ == "__main__":
    test_utt_id_stripped()
    test_convert_pos_accept()
    test_convert_pos_rejected()
    test_convert_neg()
    test_convert_duration_from_infer_sec()
    print("ALL PASS")
