"""诊断: CAM++ 在 TTS 数据集上的同人/异人 margin(多对样本)。
若 margin 仍接近0/负 → 模型在TTS数据上区分弱(信号问题, 非加载问题);
若 margin 变正 → 单条样本噪声(扩样本即可)。
"""
import os, glob
import numpy as np
import librosa
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model="E:/hf_cache/campplus/campplus.onnx", num_threads=2))

def emb(wav):
    w = np.ascontiguousarray(wav.astype(np.float32))
    st = _ext.create_stream(); st.accept_waveform(16000, w); st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

ENROLL = os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav")
e_en = emb(librosa.load(ENROLL, sr=16000)[0])

targets = sorted(glob.glob(os.path.join(_ROOT, "test_wav/dataset/raw/target/t_*.wav")))
nons = sorted(glob.glob(os.path.join(_ROOT, "test_wav/dataset/raw/nontarget/n_*.wav")))

sims_t, sims_n = [], []
print(f"{'target file':<14} {'sim':>7}     {'nontarget file':<14} {'sim':>7}")
for tf, nf in zip(targets, nons):
    st = emb(librosa.load(tf, sr=16000)[0]); sn = emb(librosa.load(nf, sr=16000)[0])
    cst = float(np.dot(e_en, st)); csn = float(np.dot(e_en, sn))
    sims_t.append(cst); sims_n.append(csn)
    print(f"{os.path.basename(tf):<14} {cst:>7.3f}     {os.path.basename(nf):<14} {csn:>7.3f}")

mt, mn = np.mean(sims_t), np.mean(sims_n)
print(f"\nCAM++ 均值: target(同人)={mt:.3f}  nontarget(异人)={mn:.3f}  margin={mt-mn:+.3f}")
print("解读: margin>0.05 → 模型可区分(可用); margin<0 → 模型在该数据上失效(TTS声码器线索弱化说话人)。")
