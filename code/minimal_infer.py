"""最小推理:加载 DiCoW_v3_2,构造全-target STNO,generate 测 RTF + 输出。
验证:模型加载 + trust_remote_code forward + GPU 推理 + RTF(解除零实测)。
真实 target-speaker 需 DiariZen diarization STNO(后续完整 pipeline)。
用法: python minimal_infer.py [audio.wav] [lang]
"""
import time, sys, torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

MODEL = "E:/hf_cache/DiCoW_v3_2"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "E:/midea_papers/code/DiCoW-inference/DiariZen/example/EN2002a_30s.wav"
LANG = sys.argv[2] if len(sys.argv) > 2 else "en"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
print(f"[setup] device={device} dtype={dtype} model={MODEL}")

t0 = time.time()
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
fe = AutoFeatureExtractor.from_pretrained(MODEL)
print(f"[load] {time.time()-t0:.1f}s | params={sum(p.numel() for p in model.parameters())/1e9:.2f}G")

audio, sr = librosa.load(AUDIO, sr=16000)
dur = len(audio) / sr
print(f"[audio] {AUDIO} | {dur:.1f}s")

inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device, dtype)   # [1, 80, 3000]
frames = input_features.shape[-1] // 2                       # 1500 (50Hz 帧)
# STNO: [sil=0, target=1, nontarget=0, overlap=0] —— 整段当目标说话人(最小验证)
stno = torch.zeros(1, 4, frames, device=device, dtype=dtype)
stno[0, 1] = 1.0
attention_mask = torch.ones(1, input_features.shape[-1], dtype=torch.bool, device=device)

if device.startswith("cuda"):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t1 = time.time()
with torch.no_grad():
    out = model.generate(input_features=input_features, attention_mask=attention_mask,
                         stno_mask=stno, language=LANG, task="transcribe", max_new_tokens=200)
if device.startswith("cuda"):
    torch.cuda.synchronize()
    print(f"[mem] peak GPU mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
dt = time.time() - t1
seqs = out["sequences"] if isinstance(out, dict) else out
text = tok.batch_decode(seqs, skip_special_tokens=True)[0]
print(f"[infer] {dt:.2f}s for {dur:.2f}s audio | RTF={dt/dur:.3f}")
print(f"[text] {text}")
