"""中文 STNO 拒识验证（XH-202615 答辩素材，拒识占评分 40%）。

STNO non-target mask 应让 DiCoW 输出空(拒识非目标)。此前 STNO 拒识实验
(stno_experiment.py D 组)只有英文 EN2002a 证据,本脚本补中文。

对 zh_target_01(冰糖「请把客厅的空调温度调到二十六度」)跑两种 STNO:
  target STNO([0,1]=1)      → 应转出指令
  non-target STNO([0,2]=1)  → 应 0 字(拒识)
STNO 4 类: [silence=0, target=1, nontarget=2, overlap=3]

用法: source code/setenv.sh && python code/zh_stno_reject.py
"""
import os, json, torch, librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

MODEL = "E:/hf_cache/DiCoW_v3_2"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
AUDIO = os.path.join(_ROOT, "test_wav", "zh_target_01.wav")
REF = "请把客厅的空调温度调到二十六度"
OUT_JSON = os.path.join(_HERE, "zh_stno_reject_result.json")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print(f"[load] {MODEL} on {device} | audio={AUDIO}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
fe = AutoFeatureExtractor.from_pretrained(MODEL)
audio, sr = librosa.load(AUDIO, sr=16000)
ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
frames = ifp.shape[-1] // 2
am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)


def run(stno):
    with torch.no_grad():
        out = model.generate(input_features=ifp, attention_mask=am, stno_mask=stno,
                             language="zh", task="transcribe", max_new_tokens=200)
    seqs = out["sequences"] if isinstance(out, dict) else out
    return tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()


# target STNO([0,1]=1) → 转写指令
s_tgt = torch.zeros(1, 4, frames, device=device, dtype=dtype)
s_tgt[0, 1] = 1.0
t_tgt = run(s_tgt)

# non-target STNO([0,2]=1) → 拒识(应空)
s_non = torch.zeros(1, 4, frames, device=device, dtype=dtype)
s_non[0, 2] = 1.0
t_non = run(s_non)

print(f"\n[target STNO]      {len(t_tgt)}字: {t_tgt}")
print(f"[non-target STNO]  {len(t_non)}字: '{t_non}'  ← 应为空=拒识成功")

result = {
    "audio": AUDIO, "ref": REF,
    "target_stno_transcript": t_tgt, "target_stno_chars": len(t_tgt),
    "nontarget_stno_transcript": t_non, "nontarget_stno_chars": len(t_non),
    "reject_success": len(t_non) == 0,  # non-target→0字=拒识成功
    "conclusion": "non-target STNO → 0字输出 = FDDT 内建拒识(中文证实)" if len(t_non) == 0
                  else f"non-target STNO 仍有 {len(t_non)}字输出(拒识不完全)",
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n[done] reject_success={result['reject_success']} → {OUT_JSON}")
