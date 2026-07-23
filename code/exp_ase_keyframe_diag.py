"""exp_ase_keyframe_diag.py — ASE 关键帧选 target 诊断 POC(判域失配生死)。

【假设】近场 enrollment(干净唤醒词) / 远场 recognition(重噪) 域失配 → wespeaker argmax 用近场声纹
选远场 target 选错(主战场 22% 可救面)。ASE(美的论文#2 ASE-PVAD arXiv2601.12769 延伸): 从混合音
选与 enrollment 最似的关键帧(远场 target 真实帧), 用关键帧 emb 选 target(同域, 绕过近/远场失配)。

【问】CAM++ 关键帧(ASE)选 target 准确率 vs CAM++ enrollment argmax(近场) vs wespeaker argmax(原系统) vs oracle。
  ASE(关键帧)选对率 >> enrollment argmax, 且 miss 中救回>30% → 域失配确认, ASE 有戏(进 CER POC)
  ≈ 或更低 → 域失配非主因或 CAM++ 不可靠, ASE 证伪(spk-oracle CAM++ B/A margin≈0 印证)

【复用】exp_spk_oracle_qwen 切的 speaker 切片(E:/target_slices_oracle_qwen_*) + CAM++(.venv_campp sherpa-onnx)。
无需 diar/转写, 纯 CAM++ emb 选 target 诊断, 最便宜。

用法: code/.venv_campp/Scripts/python.exe code/exp_ase_keyframe_diag.py
"""
import os, json, numpy as np, librosa, sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"
_ext = sherpa_onnx.SpeakerEmbeddingExtractor(
    sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=CAMPP_ONNX, num_threads=2, debug=False))
print(f"[load] CAM++ {CAMPP_ONNX} dim={_ext.dim}")


def emb(wav):
    st = _ext.create_stream()
    st.accept_waveform(16000, np.ascontiguousarray(wav.astype(np.float32)))
    st.input_finished()
    e = np.asarray(_ext.compute(st), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def keyframe_emb(mix_wav, enroll_emb, sr=16000, win=1.0, hop=0.5, topk=5):
    """滑窗选与 enrollment 最似 top-K 帧, 平均得关键帧 emb(远场 target 代表)。"""
    win_n, hop_n = int(win * sr), int(hop * sr)
    frames = []
    s = 0
    while s < max(win_n, len(mix_wav)):
        seg = mix_wav[s:s + win_n]
        if len(seg) < sr:
            seg = np.tile(seg, sr // len(seg) + 1)[:sr]
        else:
            seg = seg[:sr]
        frames.append(emb(seg))
        s += hop_n
        if s >= len(mix_wav):
            break
    if not frames:
        seg = mix_wav[:sr] if len(mix_wav) >= sr else np.tile(mix_wav, sr // len(mix_wav) + 1)[:sr]
        frames = [emb(seg)]
    frames = np.stack(frames)
    sims = frames @ enroll_emb
    top = np.argsort(sims)[-topk:]
    return frames[top].mean(axis=0), float(np.mean(sims[top]))


pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

BUCKETS = [("dead", "E:/target_slices_oracle_qwen_dead", "sim[0,0.2) 死区"),
           ("main", "E:/target_slices_oracle_qwen_main", "sim[0.2,0.4) 主战场")]
overall = {}
for bucket, slc_dir, label in BUCKETS:
    jp = os.path.join(_HERE, "runs", f"exp_spk_oracle_qwen_{bucket}.json")
    if not os.path.exists(jp):
        print(f"[skip] {jp} 不存在")
        continue
    data = json.load(open(jp, encoding="utf-8"))
    rows = [r for r in data["results"] if "error" not in r and "oracle_idx" in r]
    print(f"\n{'='*60}\n=== {label} (n={len(rows)}) ===")
    enr_ok = key_ok = ws_ok = n = 0
    miss_enr = miss_key = n_miss = 0
    for i, r in enumerate(rows):
        uid = r["uid"]
        oracle_idx = r["oracle_idx"]
        ws_argmax = r["argmax_idx"]
        pair = uid2pair.get(uid)
        if not pair:
            continue
        spk_embs, ok = [], True
        for k in range(r["n_spk"]):
            sf_path = os.path.join(slc_dir, f"{uid}__spk{k}.wav")
            if not os.path.exists(sf_path):
                ok = False
                break
            spk_embs.append(emb(librosa.load(sf_path, sr=16000)[0]))
        if not ok or len(spk_embs) != r["n_spk"]:
            continue
        enr_emb = emb(librosa.load(pair["enrollment"], sr=16000)[0])
        mix_wav = librosa.load(pair["recognition"], sr=16000)[0]
        key_emb, _ = keyframe_emb(mix_wav, enr_emb)
        enr_sims = np.array([enr_emb @ e for e in spk_embs])
        key_sims = np.array([key_emb @ e for e in spk_embs])
        enr_argmax = int(np.argmax(enr_sims))
        key_argmax = int(np.argmax(key_sims))
        enr_ok += (enr_argmax == oracle_idx)
        key_ok += (key_argmax == oracle_idx)
        ws_ok += (ws_argmax == oracle_idx)
        n += 1
        if ws_argmax != oracle_idx:
            n_miss += 1
            miss_enr += (enr_argmax == oracle_idx)
            miss_key += (key_argmax == oracle_idx)
        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(rows)}]")
    if n == 0:
        print("  无有效样本")
        continue
    print(f"\n  选对率(vs oracle 完美选 target):")
    print(f"    wespeaker argmax(原系统): {ws_ok}/{n} = {ws_ok/n:.1%}")
    print(f"    CAM++ enrollment argmax:  {enr_ok}/{n} = {enr_ok/n:.1%}")
    print(f"    CAM++ 关键帧 ASE argmax:  {key_ok}/{n} = {key_ok/n:.1%}")
    if n_miss:
        print(f"  wespeaker miss 子集(n={n_miss}, ASE 真正价值在这):")
        print(f"    CAM++ enrollment 救回: {miss_enr}/{n_miss} = {miss_enr/n_miss:.1%}")
        print(f"    CAM++ 关键帧 ASE 救回: {miss_key}/{n_miss} = {miss_key/n_miss:.1%}")
    overall[bucket] = {"n": n, "ws": ws_ok/n, "enr": enr_ok/n, "key": key_ok/n,
                       "n_miss": n_miss, "miss_key_rate": miss_key/n_miss if n_miss else None}

print(f"\n{'='*60}\n[ASE 关键帧诊断总结]")
for b, d in overall.items():
    mk = d["miss_key_rate"]
    mk_str = f"{mk:.0%}" if mk is not None else "NA"
    verdict = (f"ASE关键帧 miss 救回 {mk_str}>30% → 域失配确认, ASE 有戏"
               if mk and mk > 0.3 else
               f"ASE关键帧 miss 救回 {mk_str}≤30% → 域失配非主因(关键帧被babble污染), ASE关键帧证伪; 但看 CAM++ enrollment 是否>wespeaker(选target活路)")
    print(f"  {b}: ASE 选对 {d['key']:.1%}(vs ws {d['ws']:.1%}/enr {d['enr']:.1%}) | {verdict}")

json.dump(overall, open(os.path.join(_HERE, "runs", "exp_ase_keyframe_diag.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\n[done] runs/exp_ase_keyframe_diag.json")
