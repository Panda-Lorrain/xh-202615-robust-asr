#!/usr/bin/env python3
"""声纹强化重开 POC —— CAM++ 在 datasetA 死区 B 类(音频可辨但 wespeaker 声纹失败)的识别力。

⚠️ 缘起(2026-07-11 A2 连带): 死区 sim<0.2 经听音坐实是混合桶, B类(音频可辨 qwen 突破 H1)
+ A类(真摧毁 H2)。spk-oracle-poc 用 wespeaker oracle 证伪声纹强化, 但有 sim 代理缺陷
(wespeaker 在 B类失败 ≠ 更强声纹器也失败)。本 POC 直接测 CAM++ 在死区能否区分 B/A。

T17 仿真集 campp_vs_wespeaker.py 显示 CAM++ margin -0.022(白噪, 无区分力) → 本 POC
在真实 babble 死区复测, 判 CAM++ 是否值得集成。

机制: 对死区 target 切片(E:/target_slices_full/) + enrollment(kws_N) 用 CAM++ 算 sim,
对比 wespeaker sim(poc json), 看 B类(qwen cer<0.1) vs A类(qwen cer>0.5) 的 CAM++ margin。

运行(独立 .venv_campp, sherpa-onnx, 不碰主 venv):
  code/.venv_campp/Scripts/python.exe code/exp_spk_campp_deadzone.py
"""
import os, json, statistics
import numpy as np
import librosa
import sherpa_onnx

CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"
SLICE_DIR = "E:/target_slices_full"
_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=CAMPP_ONNX, num_threads=2, debug=False))
print(f"[load] CAM++ {CAMPP_ONNX} dim={_ext.dim}")


def emb(wav):
    st = _ext.create_stream()
    st.accept_waveform(16000, np.ascontiguousarray(wav.astype(np.float32)))
    st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


# uid -> enrollment 映射
pairs = json.load(open("code/pos_pairs_datasetA.json", encoding="utf-8"))
uid2enr = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p["enrollment"] for p in pairs}

poc = json.load(open("code/poc_qwen_asr_full_result.json", encoding="utf-8"))
dead = [r for r in poc["rows"] if r["sim"] < 0.2]
print(f"死区 sim<0.2: {len(dead)} 条, CAM++ 算 sim...")

out, miss = [], 0
for i, r in enumerate(dead):
    uid = r["uid"]
    enr = uid2enr.get(uid)
    slc = os.path.join(SLICE_DIR, uid + ".wav")
    if not enr or not os.path.exists(enr) or not os.path.exists(slc):
        miss += 1
        continue
    e_en = emb(librosa.load(enr, sr=16000)[0])
    e_sl = emb(librosa.load(slc, sr=16000)[0])
    out.append({**r, "sim_campp": float(np.dot(e_en, e_sl))})
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(dead)} (miss {miss})")

print(f"\n有效 {len(out)} 条 (miss {miss})")
ws = [r["sim"] for r in out]
cp = [r["sim_campp"] for r in out]
print(f"\nsim_wespeaker: mean={statistics.mean(ws):.3f} median={statistics.median(ws):.3f} std={statistics.pstdev(ws):.3f}")
print(f"sim_campp:     mean={statistics.mean(cp):.3f} median={statistics.median(cp):.3f} std={statistics.pstdev(cp):.3f}")

# B 类(qwen 转对, 音频可辨) vs A 类(qwen 翻车, 音频摧毁/切错) —— CAM++ 能否区分
B = [r for r in out if r["qwen_cer"] < 0.1]
A = [r for r in out if r["qwen_cer"] > 0.5]
print(f"\n=== B 类(qwen cer<0.1 转对, n={len(B)}) vs A 类(qwen cer>0.5 翻车, n={len(A)}) ===")
print(f"  B 类 wespeaker sim mean={statistics.mean(r['sim'] for r in B):.3f} | campp sim mean={statistics.mean(r['sim_campp'] for r in B):.3f}")
print(f"  A 类 wespeaker sim mean={statistics.mean(r['sim'] for r in A):.3f} | campp sim mean={statistics.mean(r['sim_campp'] for r in A):.3f}")
cp_margin = statistics.mean(r["sim_campp"] for r in B) - statistics.mean(r["sim_campp"] for r in A)
ws_margin = statistics.mean(r["sim"] for r in B) - statistics.mean(r["sim"] for r in A)
print(f"\n  CAM++     B-A margin: {cp_margin:+.3f}  (正=CAM++ 能识别 B 类 target 信号 → 有识别力)")
print(f"  wespeaker B-A margin: {ws_margin:+.3f}  (对照)")

# CAM++ 救回 B 类(>=0.27 wespeaker 阈值, 仅参考绝对值)
B_rescued = sum(1 for r in B if r["sim_campp"] >= 0.27)
print(f"\n  B 类 CAM++ sim>=0.27(救回不拒, 参考): {B_rescued}/{len(B)} = {B_rescued/len(B):.1%}")

# 听音坐实样本
print("\n=== 听音坐实样本(cmd_2091/2137, 用户确认 H1 真实听对) ===")
for r in out:
    if r["uid"] in ("cmd_2091", "cmd_2137"):
        print(f"  {r['uid']}: wespeaker={r['sim']:.3f} campp={r['sim_campp']:.3f} qwen_cer={r['qwen_cer']:.2f} | ref={r['ref']}")

print("\n=== 判别 ===")
if cp_margin > 0.05:
    print(f"  CAM++ B-A margin {cp_margin:+.3f}>0.05 → CAM++ 在死区有识别力(B类 sim 显著高于A类), 声纹强化值得完整集成(加 neg 校准阈值)")
elif cp_margin > 0:
    print(f"  CAM++ B-A margin {cp_margin:+.3f} 弱正 → CAM++ 有微弱识别力但不强, 需 neg 对比确认是否值得")
else:
    print(f"  CAM++ B-A margin {cp_margin:+.3f}≤0 → CAM++ 在死区无识别力(与 T17 仿真集 margin-0.022 一致), 声纹强化方向关闭(真 POC 证伪)")

json.dump(out, open("code/exp_spk_campp_deadzone.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\n[done] code/exp_spk_campp_deadzone.json")
