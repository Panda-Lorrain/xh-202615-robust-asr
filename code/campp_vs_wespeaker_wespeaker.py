"""wespeaker 对照侧(主 .venv, 复用 DiariZen._embedding 抽 256d emb)。
配合 campp_vs_wespeaker.py(CAM++ 侧, .venv_campp sherpa-onnx 512d) 做公平 A/B。
两者样本/enrollment/加噪种子完全一致(seed=42), 输出 JSON 可直接对照。

运行(主 .venv, 含 DiariZen/torch/transformers):
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/campp_vs_wespeaker_wespeaker.py
"""
import os, sys, json
import numpy as np
import librosa
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[load] DiariZen (wespeaker via _embedding) on {device}")
from diarizen.pipelines.inference import DiariZenPipeline
diar = DiariZenPipeline.from_pretrained("E:/hf_cache/diarizen-wavlm-large-s80-md").to(device)


def emb_wespeaker(wav_np):
    w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
    if w.dim() == 1:
        w = w[None, None]
    with torch.no_grad():
        e = diar._embedding(w)            # (1,256)
    e = torch.as_tensor(e, dtype=torch.float32).squeeze(0)
    return F.normalize(e, dim=-1)


from simulate_pipeline import add_noise

ENROLL = os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav")
T01 = os.path.join(_ROOT, "test_wav/dataset/raw/target/t_01.wav")
N01 = os.path.join(_ROOT, "test_wav/dataset/raw/nontarget/n_01.wav")
enroll, _ = librosa.load(ENROLL, sr=16000)
t01, _ = librosa.load(T01, sr=16000)
n01, _ = librosa.load(N01, sr=16000)
rng = np.random.default_rng(42)   # 与 CAM++ 侧同种子
t01_snr5 = add_noise(t01, rng.standard_normal(len(t01)).astype(np.float32), 5)
t01_snrn5 = add_noise(t01, rng.standard_normal(len(t01)).astype(np.float32), -5)
samples = [("冰糖t_01干净(同人)", t01), ("冰糖t_01+白噪+5dB", t01_snr5),
           ("冰糖t_01+白噪-5dB", t01_snrn5), ("苏打n_01干净(异人)", n01)]

results = {"wespeaker": {}}
e_en = emb_wespeaker(enroll)
for nm, w in samples:
    e = emb_wespeaker(w)
    results["wespeaker"][nm] = float(torch.dot(e_en, e))

print(f"\n{'样本':<24} {'wespeaker(256d)':>16}")
print("-" * 42)
for nm, _ in samples:
    print(f"{nm:<24} {results['wespeaker'][nm]:>16.3f}")

ws_same = results['wespeaker']['冰糖t_01干净(同人)']
ws_drop = ws_same - results['wespeaker']['冰糖t_01+白噪+5dB']
print(f"\nwespeaker 同人干净 sim: {ws_same:.3f}")
print(f"wespeaker +5dB 带噪 sim 衰减: {ws_drop:.3f}")
results["ws_noise_drop"] = ws_drop
json.dump(results, open(os.path.join(_HERE, 'wespeaker_quick_result.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print("\n[done] → code/wespeaker_quick_result.json (与 campp_quick_result.json 对照)")
