"""对照转写 spk0/spk1 的 exclusive vs full 段, 证明 diar 已正确分开(exclusive 纯净) vs downstream 用 full 致混(full 混两人)。
用 vanilla whisper-large-v3-turbo(本地缓存)。"""
import inspect as _inspect
_orig = _inspect.getmodule
def _safe(*a, **k):
    try: return _orig(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe

import os, sys, json, glob
sys.path.insert(0, "E:/midea_target_asr/code")
import torch, numpy as np, librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
from repro import set_global_seed, resolve_model

set_global_seed(42)
device = torch.device("cuda:0")
dtype = torch.float16
model_id = resolve_model("VANILLA")
print(f"[load] {model_id}")
m = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
tok = AutoTokenizer.from_pretrained(model_id)
fe = AutoFeatureExtractor.from_pretrained(model_id)

def transcribe(wav_path):
    a, sr = librosa.load(wav_path, sr=16000)
    if len(a) < 1600:  # <0.1s skip
        return "<too short>"
    inp = fe(a, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    with torch.inference_mode():
        ids = m.generate(inp, language="zh", task="transcribe", max_new_tokens=100)
    return tok.batch_decode(ids, skip_special_tokens=True)[0].strip()

CFG_DIR = "E:/midea_target_asr/code/runs/_diag_2637_diar/default"
out = {}
for f in sorted(glob.glob(os.path.join(CFG_DIR, "spk*.wav"))):
    name = os.path.basename(f).replace(".wav", "")
    txt = transcribe(f)
    print(f"  {name}: {txt!r}")
    out[name] = txt

# enrollment + 原始 recognition 也跑一遍对照
ENR = "E:/midea_target_asr/datasetA/pos/kws_2637.wav"
REC = "E:/midea_target_asr/datasetA/pos/cmd_2637.wav"
out["enrollment (kws_2637)"] = transcribe(ENR)
out["recognition_original (cmd_2637)"] = transcribe(REC)
print(f"  enrollment: {out['enrollment (kws_2637)']!r}")
print(f"  recognition_original: {out['recognition_original (cmd_2637)']!r}")

with open("E:/midea_target_asr/code/runs/_diag_2637_diar/transcription_check.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("saved transcription_check.json")
