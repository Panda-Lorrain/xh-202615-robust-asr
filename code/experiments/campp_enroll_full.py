"""450 矩阵 CAM++ enrollment sim 全量计算(.venv_campp, CPU, sherpa-onnx)。
输出与主 wespeaker enroll_wespeaker_full.json 同结构(每条 max_sim/rejected/transcript占位),
供 eval_enrollment.py 分档对照(注: 本脚本只算声纹 sim, 不转写——转写需 DiCoW GPU)。

逻辑(复用 enroll_infer.py 的声纹部分, 但 CAM++ 替代 wespeaker):
  - final_manifest.json 每条: target_ref(ground truth) + file(450条带噪重叠 wav)
  - enrollment = raw/enrollment/target_long_01.wav (干净, 同 enroll_infer)
  - 识别音频里无 diarization speaker 分段(纯声纹整段对比) → 直接抽整段 CAM++ emb
  - sim = cos(enroll_emb, 整段识别音频 emb)
  注: 这是"整段声纹 sim"(无 diar 分离), 是 CAM++ 带噪鲁棒性下限估计。
       真正的 per-speaker sim 需 DiariZen diarization(主 .venv+GPU), 见 verdict 说明。

运行:
  E:/midea_target_asr/code/.venv_campp/Scripts/python.exe code/campp_enroll_full.py
对照评估(主 venv):
  code/.venv/Scripts/python.exe code/eval_enrollment.py --enroll-json code/campp_enroll_full.json --label CAMpp-integral
"""
import os, sys, json, time, glob
import numpy as np
import librosa
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"
FINAL = os.path.join(_ROOT, "test_wav/dataset/final/final_manifest.json")

_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=CAMPP_ONNX, num_threads=4))


def emb(wav):
    w = np.ascontiguousarray(wav.astype(np.float32))
    st = _ext.create_stream(); st.accept_waveform(16000, w); st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


mani = json.load(open(FINAL, encoding="utf-8"))
ENROLL = os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav")
e_en = emb(librosa.load(ENROLL, sr=16000)[0])
print(f"[enroll] {ENROLL} → emb {e_en.shape}; 共 {len(mani['items'])} 条识别音频")

results = []
t0 = time.time()
for i, it in enumerate(mani["items"]):
    f = it["file"]
    if not os.path.isabs(f):
        f = os.path.join(os.path.dirname(FINAL), os.path.basename(f))
    w, _ = librosa.load(f, sr=16000)
    e = emb(w)
    sim = float(np.dot(e_en, e))
    results.append({
        "recognition": it["file"], "enrollment": ENROLL,
        "max_sim": sim, "reject_threshold": 0.5,
        "rejected": sim < 0.5, "transcript": "", "chars": 0,  # 声纹侧不转写
        "_note": "CAM++ 整段声纹 sim (无 diar 分离); 转写需主 venv DiCoW",
    })
    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(mani['items'])}] {os.path.basename(it['file'])} sim={sim:.3f} ({time.time()-t0:.0f}s)")

out = os.path.join(_HERE, "campp_enroll_full.json")
json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
sims = [r["max_sim"] for r in results]
print(f"\n[done] {len(results)} 条 → {out}")
print(f"CAM++ 整段均 sim={np.mean(sims):.3f} (主线 wespeaker 450 矩阵均 sim≈0.218)")
print(f"  max={np.max(sims):.3f} min={np.min(sims):.3f} >0.3比例={np.mean(np.array(sims)>0.3):.2f}")
print(f"对照: python code/eval_enrollment.py --enroll-json code/campp_enroll_full.json --label CAMpp-integral")
