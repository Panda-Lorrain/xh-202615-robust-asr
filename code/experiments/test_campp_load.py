"""CPU验证: sherpa-onnx campplus 加载 + 抽emb (解modelscope挂起的核心de-risk)。
不依赖transformers/DiariZen, 纯ONNX runtime。
用法: E:/midea_target_asr/code/.venv_campp/Scripts/python.exe code/test_campp_load.py
"""
import os, sys, time
import numpy as np
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
MODEL = "E:/hf_cache/campplus/campplus.onnx"

print(f"[1] import sherpa_onnx OK (no hang)")
print(f"[2] build SpeakerEmbeddingExtractor (model={MODEL})")
t0 = time.time()
cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=1, debug=False)
ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
print(f"    loaded in {time.time()-t0:.2f}s, dim={ext.dim}")

import soundfile as sf
def load16(p):
    # test_wav raw 都是16k mono, soundfile 直接读; 非16k才需要重采样
    w, sr = sf.read(p, dtype="float32")
    if sr != 16000:
        import librosa
        w = librosa.load(p, sr=16000)[0].astype(np.float32)
    if w.ndim > 1:
        w = w.mean(axis=1)
    return np.ascontiguousarray(w, dtype=np.float32)

samples = {
    "enroll": os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav"),
    "target_t01": os.path.join(_ROOT, "test_wav/dataset/raw/target/t_01.wav"),
    "nontarget_n01": os.path.join(_ROOT, "test_wav/dataset/raw/nontarget/n_01.wav"),
}

embs = {}
print(f"\n[3] extract embeddings on CPU:")
for nm, p in samples.items():
    if not os.path.isfile(p):
        print(f"    [skip] {nm}: missing {p}")
        continue
    w = load16(p)
    s = sherpa_onnx.OfflineSpeakerEmbeddingExtractor if False else ext  # ext is online-compatible API
    stream = ext.create_stream()
    stream.accept_waveform(16000, w)
    stream.input_finished()
    emb = np.asarray(ext.compute(stream), dtype=np.float32)  # flat list[dim] → np
    embs[nm] = emb
    print(f"    {nm:14s} dur={len(w)/16000:.1f}s dim={emb.shape[0]} norm={np.linalg.norm(emb):.4f} mean={emb.mean():+.4f}")

print(f"\n[4] cosine similarity (interface aligned to wespeaker: wav -> normalized emb):")
def cos(a, b):
    a, b = a / (np.linalg.norm(a)+1e-9), b / (np.linalg.norm(b)+1e-9)
    return float(a @ b)
if "enroll" in embs and "target_t01" in embs:
    print(f"    enroll vs target_t01  (same speaker): {cos(embs['enroll'], embs['target_t01']):.3f}  (期望高)")
if "enroll" in embs and "nontarget_n01" in embs:
    print(f"    enroll vs nontarget_n01(diff speaker): {cos(embs['enroll'], embs['nontarget_n01']):.3f}  (期望低)")
print("\n[done] sherpa-onnx CAM++ 加载+抽emb 成功, import 不挂起。")
