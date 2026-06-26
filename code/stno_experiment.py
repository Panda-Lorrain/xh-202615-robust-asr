"""STNO 实验:验证 FDDT/STNO 如何控制 target-speaker 转写(答辩素材)。
对 EN2002a(30s 英文会议,多人)构造 4 种 STNO,看 DiCoW 输出差异:
  A 全-target(基线,转所有人) B 前半target+后半silence C 前半silence+后半target D 全-non-target(应拒识/空)
预期:B 只转前半 C 只转后半 D 空/拒识 → 证明 STNO 的 target/silence/nontarget 类控制转写。"""
import time, torch, librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

MODEL = "E:/hf_cache/DiCoW_v3_2"
AUDIO = "E:/midea_papers/code/DiCoW-inference/DiariZen/example/EN2002a_30s.wav"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"[load] {MODEL} on {device}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
fe = AutoFeatureExtractor.from_pretrained(MODEL)
audio, sr = librosa.load(AUDIO, sr=16000)
ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
frames = ifp.shape[-1] // 2
am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
half = frames // 2

def run(stno, label):
    t = time.time()
    with torch.no_grad():
        out = model.generate(input_features=ifp, attention_mask=am, stno_mask=stno,
                             language="en", task="transcribe", max_new_tokens=200)
    dt = time.time() - t
    seqs = out["sequences"] if isinstance(out, dict) else out
    text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
    print(f"\n=== {label} | infer {dt:.2f}s | {len(text)} chars ===\n{text[:350]}")

# A 全 target
sA = torch.zeros(1, 4, frames, device=device, dtype=dtype); sA[0, 1] = 1.0
run(sA, "A 全-target(基线,转所有人)")
# B 前半 target + 后半 silence
sB = torch.zeros(1, 4, frames, device=device, dtype=dtype); sB[0, 1, :half] = 1.0; sB[0, 0, half:] = 1.0
run(sB, "B 前半target+后半silence(应只转前半)")
# C 前半 silence + 后半 target
sC = torch.zeros(1, 4, frames, device=device, dtype=dtype); sC[0, 0, :half] = 1.0; sC[0, 1, half:] = 1.0
run(sC, "C 前半silence+后半target(应只转后半)")
# D 全 non-target(应拒识/空)
sD = torch.zeros(1, 4, frames, device=device, dtype=dtype); sD[0, 2] = 1.0
run(sD, "D 全-non-target(应拒识/空)")
print("\n[done] STNO 控制验证完成")
