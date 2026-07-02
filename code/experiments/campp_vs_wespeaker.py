"""快速验证 CAM++ vs wespeaker 声纹鲁棒性假设(中文+噪声)。T17 实验。

CAM++ 加载方式 = sherpa-onnx(方案B, 隔离 .venv_campp, 不碰 transformers, ONNX runtime 自带)。
  解了 modelscope 1.37+Windows+torch 挂起问题。
  模型 = Wespeaker/wespeaker-voxceleb-campplus (voxceleb_CAM++.onnx, 512d, 已加 sherpa meta)。

接口对齐 wespeaker: 输入 wav(np.float32@16k) → L2 归一化 emb 向量。
  wespeaker(主线 DiariZen._embedding) 抽 256d; CAM++ 抽 512d —— 维度不同但接口同形(归一化向量, 余弦内积即可比)。

样本(enrollment=冰糖长干净):
  - 冰糖 t_01 干净          → 同人干净, 两模型都应高 sim
  - 冰糖 t_01 + 白噪 +5dB   → 同人带噪, 看 sim 衰减(CAM++ 是否降更少)
  - 冰糖 t_01 + 白噪 -5dB   → 同人重噪
  - 苏打 n_01 干净          → 异人, 两模型都应低 sim

运行(独立 venv, 不碰主 .venv):
  E:/midea_target_asr/code/.venv_campp/Scripts/python.exe code/campp_vs_wespeaker.py
"""
import os, sys, json
import numpy as np
import librosa
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"

print(f"[load] CAM++ (sherpa-onnx, 模型已含 meta) {CAMPP_ONNX}")
_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=CAMPP_ONNX, num_threads=2, debug=False)
_ext = sherpa_onnx.SpeakerEmbeddingExtractor(_cfg)
print(f"    loaded, dim={_ext.dim}")


def emb_campp(wav_np):
    """wav(np.float32@16k) → L2归一化 emb (np.float32, dim)。接口对齐 wespeaker。"""
    w = np.ascontiguousarray(wav_np.astype(np.float32))
    st = _ext.create_stream()
    st.accept_waveform(16000, w)
    st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)   # (dim,)
    n = np.linalg.norm(e) + 1e-9
    return e / n   # 已归一化, 内积即余弦


def cos(a, b):
    return float(np.dot(a, b))   # a,b 已归一化


# wespeaker emb: 本 venv 无 DiariZen/torch GPU pipeline, 不能直接抽。
# → CAM++ 单独算, 与主线 enroll JSON 的 wespeaker 0.218 均值 / 450 矩阵结果对照(见 campp_450 命令)。
# 此脚本聚焦"快速鲁棒性信号"(干净/带噪/异人 4 样本 sim), wespeaker 对照值由主 venv 跑 campp_vs_wespeaker_wespeaker.py。
WESPEAKER_PLACEHOLDER = None  # 见下方对照说明


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

results = {"CAM++": {}}
e_en = emb_campp(enroll)
print(f"[enrollment] {os.path.basename(ENROLL)} ({len(enroll)/16000:.1f}s) → emb {e_en.shape}")
for nm, w in samples:
    e = emb_campp(w)
    results["CAM++"][nm] = cos(e_en, e)

print(f"\n{'样本':<24} {'CAM++(512d)':>12}")
print("-" * 38)
for nm, _ in samples:
    print(f"{nm:<24} {results['CAM++'][nm]:>12.3f}")

cp_same = results['CAM++']['冰糖t_01干净(同人)']
cp_diff = results['CAM++']['苏打n_01干净(异人)']
cp_drop = cp_same - results['CAM++']['冰糖t_01+白噪+5dB']
print(f"\nCAM++ 同人干净 sim: {cp_same:.3f}")
print(f"CAM++ 异人干净 sim: {cp_diff:.3f}")
print(f"CAM++ +5dB 带噪 sim 衰减: {cp_drop:.3f}  (越小越鲁棒)")
print(f"CAM++ 同人-异人 区分度(margin): {cp_same - cp_diff:+.3f}  (越大越能分同人/异人)")

results["verdict_note"] = ("CAM++ 单独结果。wespeaker 对照需主 venv 跑 campp_vs_wespeaker_wespeaker.py "
                            "(主 .venv 有 DiariZen._embedding)。主线 wespeaker 450 矩阵均 sim=0.218。")
results["cp_noise_drop"] = cp_drop
results["cp_margin"] = cp_same - cp_diff
out = os.path.join(_HERE, 'campp_quick_result.json')
json.dump(results, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f"\n[done] → {out}")
print("[提示] 主 .venv 跑 wespeaker 对照: 见 code/campp_vs_wespeaker_wespeaker.py")
