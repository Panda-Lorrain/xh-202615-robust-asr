"""快速验证 CAM++ vs wespeaker 声纹鲁棒性假设（中文+噪声）。T17 实验。

不做 diarization/转写, 聚焦声纹质量: 对 enrollment + 各样本整段抽 emb, 比余弦 sim。
若 CAM++ 带噪 sim 衰减更小 + 同人/不同人区分更好 → 假设成立, 值得换用 + 完整对比。

样本(enrollment=冰糖长干净):
  - 冰糖 t_01 干净          → 同人干净, 两模型都应高 sim
  - 冰糖 t_01 + 白噪 +5dB   → 同人带噪, 看 sim 衰减(CAM++ 是否降更少)
  - 冰糖 t_01 + 白噪 -5dB   → 同人重噪
  - 苏打 n_01 干净          → 异人, 两模型都应低 sim

环境: source code/setenv.sh && export HF_HUB_OFFLINE=1
      code/.venv/Scripts/python.exe code/campp_vs_wespeaker.py
"""
import os, sys, tempfile, json
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
sys.path.insert(0, _HERE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# wespeaker: 复用 DiariZen._embedding(已验证接口)
print("[load] DiariZen (wespeaker via _embedding)")
from diarizen.pipelines.inference import DiariZenPipeline
diar = DiariZenPipeline.from_pretrained("E:/hf_cache/diarizen-wavlm-large-s80-md").to(device)

# campp: modelscope
print("[load] CAM++ (modelscope, 首次会下载 ~29MB)")
from modelscope.pipelines import pipeline as ms_pipeline
from modelscope.utils.constant import Tasks
sv = ms_pipeline(task=Tasks.speaker_verification, model='iic/speech_campplus_sv_zh-cn_16k-common')


def emb_wespeaker(wav_np):
    w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
    if w.dim() == 1:
        w = w[None, None]
    with torch.no_grad():
        e = diar._embedding(w)            # (1,256) np
    return torch.as_tensor(e, dtype=torch.float32).squeeze(0)


def emb_campp(wav_np):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    sf.write(tmp, wav_np.astype(np.float32), 16000)
    res = sv(tmp, output_embs=True)
    os.unlink(tmp)
    # 接口容错: res 可能 dict/list, emb 在 'embs' 键
    if isinstance(res, dict) and 'embs' in res:
        e = res['embs']
    elif isinstance(res, list) and isinstance(res[0], dict) and 'embs' in res[0]:
        e = res[0]['embs']
    else:
        e = res
    e = np.array(e).reshape(-1)
    return torch.as_tensor(e, dtype=torch.float32)


from simulate_pipeline import add_noise

ENROLL = os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav")
T01 = os.path.join(_ROOT, "test_wav/dataset/raw/target/t_01.wav")
N01 = os.path.join(_ROOT, "test_wav/dataset/raw/nontarget/n_01.wav")
enroll, _ = librosa.load(ENROLL, sr=16000)
t01, _ = librosa.load(T01, sr=16000)
n01, _ = librosa.load(N01, sr=16000)
rng = np.random.default_rng(42)
t01_snr5 = add_noise(t01, rng.standard_normal(len(t01)).astype(np.float32), 5)
t01_snrn5 = add_noise(t01, rng.standard_normal(len(t01)).astype(np.float32), -5)
samples = [("冰糖t_01干净(同人)", t01), ("冰糖t_01+白噪+5dB", t01_snr5),
           ("冰糖t_01+白噪-5dB", t01_snrn5), ("苏打n_01干净(异人)", n01)]

results = {"wespeaker": {}, "CAM++": {}}
for model, fn in [("wespeaker", emb_wespeaker), ("CAM++", emb_campp)]:
    e_en = F.normalize(fn(enroll), dim=-1)
    for nm, w in samples:
        e = F.normalize(fn(w), dim=-1)
        results[model][nm] = float(torch.dot(e_en, e))

print(f"\n{'样本':<24} {'wespeaker':>11} {'CAM++':>8}")
print("-" * 46)
for nm, _ in samples:
    print(f"{nm:<24} {results['wespeaker'][nm]:>11.3f} {results['CAM++'][nm]:>8.3f}")

ws_same = results['wespeaker']['冰糖t_01干净(同人)']
cp_same = results['CAM++']['冰糖t_01干净(同人)']
ws_drop = ws_same - results['wespeaker']['冰糖t_01+白噪+5dB']
cp_drop = cp_same - results['CAM++']['冰糖t_01+白噪+5dB']
print(f"\n同人干净 sim: wespeaker={ws_same:.3f}  CAM++={cp_same:.3f}")
print(f"异人干净 sim: wespeaker={results['wespeaker']['苏打n_01干净(异人)']:.3f}  CAM++={results['CAM++']['苏打n_01干净(异人)']:.3f}")
print(f"+5dB 带噪 sim 衰减: wespeaker={ws_drop:.3f}  CAM++={cp_drop:.3f}  (越小越鲁棒)")
verdict = "✅ CAM++ 带噪更鲁棒(假设成立, 值得换用+完整对比)" if cp_drop < ws_drop - 0.02 else \
          ("wespeaker 更鲁棒或持平(假设不成立)" if ws_drop < cp_drop - 0.02 else "两者接近")
print(f"结论: {verdict}")

results["verdict"] = verdict
results["ws_noise_drop"] = ws_drop
results["cp_noise_drop"] = cp_drop
json.dump(results, open(os.path.join(_HERE, 'campp_vs_wespeaker_result.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"\n[done] → code/campp_vs_wespeaker_result.json")
