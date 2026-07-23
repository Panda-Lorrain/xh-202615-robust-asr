"""exp_campp_select_cer.py — CAM++ 替换 wespeaker 选 target 的 CER 收益 POC。

【背景】ASE 诊断(exp_ase_keyframe_diag)发现: CAM++ enrollment argmax 选 target 比 wespeaker 更准
(主战场选对率 78.3% vs 68.3%, wespeaker miss 救回 73.7%)。这推翻 spk-oracle "声纹强化关闭"的笼统结论
(它证伪的是"用sim区分音频可辨度", 非"选target speaker"的who任务)。

【问】CAM++ argmax 选 target → CER 是否显著低于 wespeaker argmax(原系统)?
【方法】复用 exp_spk_oracle_qwen 已算的 per_spk_cer(各 speaker 的 qwen CER), CAM++ argmax 选 idx → per_spk_cer[idx],
  不重跑 qwen(最便宜)。对比 wespeaker argmax_cer + oracle_cer(下限)。

用法: code/.venv_campp/Scripts/python.exe code/exp_campp_select_cer.py
"""
import os, json, numpy as np, librosa, sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"
_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=CAMPP_ONNX, num_threads=2, debug=False))
print(f"[load] CAM++ dim={_ext.dim}")


def emb(wav):
    st = _ext.create_stream()
    st.accept_waveform(16000, np.ascontiguousarray(wav.astype(np.float32)))
    st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

BUCKETS = [("dead", "E:/target_slices_oracle_qwen_dead", "死区[0,0.2)"),
           ("main", "E:/target_slices_oracle_qwen_main", "主战场[0.2,0.4)")]
overall = {}
for bucket, slc_dir, label in BUCKETS:
    jp = os.path.join(_HERE, "runs", f"exp_spk_oracle_qwen_{bucket}.json")
    data = json.load(open(jp, encoding="utf-8"))
    rows = [r for r in data["results"]
            if "error" not in r and "oracle_idx" in r and "per_spk_cer" in r]
    print(f"\n{'='*60}\n=== {label} (n={len(rows)}) ===")
    ws_cers, cp_cers, orc_cers = [], [], []
    cp_better = cp_worse = cp_tie = 0
    for r in rows:
        uid = r["uid"]
        pair = uid2pair.get(uid)
        if not pair:
            continue
        per_spk_cer = r["per_spk_cer"]
        n_spk = r["n_spk"]
        if len(per_spk_cer) != n_spk:
            continue
        spk_embs, ok = [], True
        for k in range(n_spk):
            sf_path = os.path.join(slc_dir, f"{uid}__spk{k}.wav")
            if not os.path.exists(sf_path):
                ok = False
                break
            spk_embs.append(emb(librosa.load(sf_path, sr=16000)[0]))
        if not ok:
            continue
        enr_emb = emb(librosa.load(pair["enrollment"], sr=16000)[0])
        sims = np.array([enr_emb @ e for e in spk_embs])
        cp_argmax = int(np.argmax(sims))
        ws_argmax = r["argmax_idx"]
        oracle_idx = r["oracle_idx"]
        ws_cer = per_spk_cer[ws_argmax]
        cp_cer = per_spk_cer[cp_argmax]
        orc_cer = per_spk_cer[oracle_idx]
        ws_cers.append(ws_cer)
        cp_cers.append(cp_cer)
        orc_cers.append(orc_cer)
        if cp_cer < ws_cer - 0.01:
            cp_better += 1
        elif cp_cer > ws_cer + 0.01:
            cp_worse += 1
        else:
            cp_tie += 1
    ws_m, cp_m, orc_m = float(np.mean(ws_cers)), float(np.mean(cp_cers)), float(np.mean(orc_cers))
    print(f"  wespeaker argmax CER(原系统): {ws_m:.3f}")
    print(f"  CAM++ argmax CER:             {cp_m:.3f}  (Δ{cp_m-ws_m:+.3f})")
    print(f"  oracle CER(完美选 target 下限): {orc_m:.3f}  (oracle vs wespeaker Δ{orc_m-ws_m:+.3f})")
    print(f"  逐条: CAM++ 更优 {cp_better} / 更差 {cp_worse} / 持平 {cp_tie} (n={len(ws_cers)})")
    overall[bucket] = {"ws_cer": round(ws_m, 4), "cp_cer": round(cp_m, 4),
                       "oracle_cer": round(orc_m, 4), "delta": round(cp_m - ws_m, 4),
                       "oracle_delta": round(orc_m - ws_m, 4),
                       "n_better": cp_better, "n_worse": cp_worse, "n_tie": cp_tie, "n": len(ws_cers)}

print(f"\n{'='*60}\n[CAM++ 替换 wespeaker 选 target CER 总结]")
for b, d in overall.items():
    share = d['n_better'] / (d['n_better'] + d['n_worse']) if (d['n_better'] + d['n_worse']) else 0
    print(f"  {b}: wespeaker {d['ws_cer']:.3f} → CAM++ {d['cp_cer']:.3f} (Δ{d['delta']:+.3f}) | "
          f"更优{d['n_better']}/更差{d['n_worse']} | oracle下限{d['oracle_cer']:.3f}")
    if d['delta'] < -0.03:
        print(f"    ✅ CAM++ CER 降 {d['delta']:+.3f} 显著 → CAM++ 替换 wespeaker 值得全量集成")
    elif d['delta'] < 0:
        print(f"    🟡 CAM++ 略降 {d['delta']:+.3f} → 边际, 慎投")
    else:
        print(f"    ❌ CAM++ 更差 {d['delta']:+.3f} → 不替换")

json.dump(overall, open(os.path.join(_HERE, "runs", "exp_campp_select_cer.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[done] runs/exp_campp_select_cer.json")
