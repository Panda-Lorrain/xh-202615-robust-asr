"""verify_sample_success.py — 选「双人重叠+成功(CER=0)」样本生成 8 工位齐全验收包。

对照 cmd_2637(双人重叠失败), 让用户听"分离成功"长啥样。
8 工位 + SepFormer 分离, 每工位产出存盘 + 写索引 md。

环境: code/.venv(主) + code/.venv_qwen(qwen 转写 subprocess)
种子: 42  不改主线代码  不 git commit
"""
# speechbrain 1.1 lazy proxy + Win inspect guard (复刻 enroll_infer.py:24-29 + exp_sepformer_qwen.py)
import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import os, sys, json, shutil, time, subprocess, glob
import torch
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

DICOW_INF = os.path.join(_HERE, "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p): sys.path.insert(0, _p)

# speechbrain Win patch (sepformer)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import speechbrain.utils.importutils as _sb_iu
import importlib as _importlib
def _patched_ensure_module(self, stacklevel):
    fr = None
    try:
        fr = _inspect.getframeinfo(sys._getframe(stacklevel + 1))
    except AttributeError:
        pass
    if fr is not None and fr.filename.replace("\\", "/").endswith("/inspect.py"):
        raise AttributeError()
    if self.lazy_module is None:
        try:
            self.lazy_module = (_importlib.import_module(self.target) if self.package is None
                                else _importlib.import_module("." + self.target, self.package))
        except Exception as e:
            raise ImportError("Lazy import failed (patched)") from e
    return self.lazy_module
_sb_iu.LazyModule.ensure_module = _patched_ensure_module

import pyarrow  # 预热: 避免 pyannote 扫 sys.path 触发 WinError 6714

from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import to_simplified, digit_postproc, brand_homophone_fix, cut_target_timeline, is_valid_command
from repro import set_global_seed, resolve_model


def main():
    set_global_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ============ load diar + sepformer (一次) ============
    print("[load] DiariZen...")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)

    print("[load] SepFormer (sepformer-whamr16k)...")
    from speechbrain.inference.separation import SepformerSeparation as separator
    from speechbrain.utils.fetching import LocalStrategy
    sep_model = separator.from_hparams(source="speechbrain/sepformer-whamr16k",
                                       savedir="E:/hf_cache/sepformer-whamr16k",
                                       local_strategy=LocalStrategy.COPY,
                                       run_opts={"device": str(device)})

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1: w = w[None, None]
        elif w.dim() == 2: w = w[None]
        with torch.inference_mode():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    def separate(mix_np):
        mix = torch.from_numpy(np.ascontiguousarray(mix_np.astype(np.float32))).to(sep_model.device)[None, :]
        est = sep_model.separate_batch(mix).squeeze(0).detach().cpu().numpy()  # [T, n_src]
        return est.T  # [n_src, T]

    # ============ step 1: 扫候选, 挑双人重叠率最高 ============
    poc = json.load(open(os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    pairs = {r["id"]: r for r in json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))}
    rows = poc["rows"]
    cands = [r for r in rows if r.get("sim", 0) >= 0.4 and r.get("qwen_cer", 1) < 0.001]
    cands.sort(key=lambda r: -r["sim"])
    print(f"\n{len(cands)} success cands (sim>=0.4 & qwen_cer<0.001); scanning ALL for overlap + 2-spk")

    def diar_info(rec_path):
        audio, sr = librosa.load(rec_path, sr=16000)
        try:
            diar_out = diar(rec_path)
        except Exception as e:
            return None
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        total_spk_sec = sum((e - s) for tl in per_spk for (s, e) in tl)
        audio_sec = len(audio) / sr
        overlap_rate = max(0.0, (total_spk_sec - audio_sec) / audio_sec)
        return {"n_spk": len(speakers), "overlap": overlap_rate, "speakers": speakers, "per_spk": per_spk, "audio": audio, "sr": sr}

    scan = []
    for r in cands:
        uid_num = int(r["uid"].split("_")[1])
        pair = pairs.get(uid_num)
        if not pair: continue
        info = diar_info(pair["recognition"])
        if info is None: continue
        scan.append({"uid": r["uid"], "uid_num": uid_num, "sim": r["sim"], "qwen_cer": r["qwen_cer"],
                     "ref": r["ref"], "kws_txt": pair.get("kws_txt", ""),
                     "enrollment": pair["enrollment"], "recognition": pair["recognition"], **info})
        print(f"  {r['uid']} n_spk={info['n_spk']} overlap={info['overlap']:.3f} sim={r['sim']:.3f}")

    two_spk = [s for s in scan if s["n_spk"] == 2]
    if two_spk:
        best = max(two_spk, key=lambda s: s["overlap"])
    else:
        print("[warn] 无 2-spk 候选, 取 overlap 最高的样本(可能单人)")
        best = max(scan, key=lambda s: s["overlap"])
    uid_label = best["uid"]
    uid_num = best["uid_num"]
    pair = pairs[uid_num]
    print(f"\n[selected] {uid_label} sim={best['sim']:.3f} cer={best['qwen_cer']:.3f} "
          f"overlap={best['overlap']:.3f} n_spk={best['n_spk']}")
    print(f"  ref: {best['ref']}")
    print(f"  kws_txt: {best['kws_txt']}")

    # ============ 工位产出 ============
    outdir = os.path.join(_HERE, "runs", f"_verify_{uid_label}")
    if os.path.isdir(outdir): shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    qwen_dir = os.path.join(outdir, "_qwen_slices")
    os.makedirs(qwen_dir, exist_ok=True)

    enr_path = best["enrollment"]; rec_path = best["recognition"]

    # WS0 输入
    shutil.copy(enr_path, os.path.join(outdir, "enrollment.wav"))
    shutil.copy(rec_path, os.path.join(outdir, "recognition.wav"))

    # WS1 enroll diar + emb
    print("\n[WS1] enroll diar + enroll_emb(整段, 主线用法)")
    enr_wav, _ = librosa.load(enr_path, sr=16000)
    enr_info_path = os.path.join(outdir, "enroll_diar.json")
    try:
        enr_diar = diar(enr_path)
        enr_spk = list(enr_diar.labels())
        enr_per_spk = [enr_diar.label_timeline(s) for s in enr_spk]
        for i, spk in enumerate(enr_spk):
            segs = [enr_wav[int(s * 16000):int(e * 16000)] for s, e in enr_per_spk[i]]
            seg = np.concatenate(segs) if segs else np.zeros(16000, dtype=np.float32)
            sf.write(os.path.join(outdir, f"enr_spk{i}.wav"), seg, 16000)
        enr_n_spk = len(enr_spk)
        enr_dur_sec = len(enr_wav) / 16000
        json.dump({"enr_n_spk": enr_n_spk, "enr_dur_sec": enr_dur_sec,
                   "enr_spk_segs_sec": {f"spk{i}": [(float(s), float(e)) for s, e in enr_per_spk[i]] for i in range(enr_n_spk)}},
                  open(enr_info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  enroll diar: n_spk={enr_n_spk} dur={enr_dur_sec:.2f}s")
    except Exception as e:
        print(f"  enroll diar fail: {e}")
        enr_n_spk = 0
    enroll_emb = get_emb(enr_wav)  # mainline: 整段抽

    # WS2 rec diar
    print("\n[WS2] rec diar → 每 speaker 完整段 + 重叠率")
    rec_wav = best["audio"]; sr = best["sr"]
    speakers = best["speakers"]; per_spk = best["per_spk"]
    audio_len_frames = int(len(rec_wav) / sr * 50)
    diar_mask = get_diarization_mask(per_spk, audio_len_frames)
    for i, spk in enumerate(speakers):
        segs = [rec_wav[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
        full = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
        sf.write(os.path.join(outdir, f"rec_spk{i}_full.wav"), full, sr)
    total_spk_sec = sum((e - s) for tl in per_spk for s, e in tl)
    audio_sec = len(rec_wav) / sr
    overlap_rate = max(0.0, (total_spk_sec - audio_sec) / audio_sec)
    print(f"  speakers={speakers} audio={audio_sec:.2f}s total_spk={total_spk_sec:.2f}s overlap={overlap_rate:.3f}")

    # WS3 每 spk emb 细节(collect_clean_audio)★ 补 2637 缺
    print("\n[WS3] per-spk emb 细节 (collect_clean_audio 独占帧)")
    spk_emb_info = []
    spk_embs = []
    for i in range(len(speakers)):
        seg_excl = collect_clean_audio(rec_wav, diar_mask, i, sr=sr)
        info = {"speaker": speakers[i]}
        if seg_excl is None or len(seg_excl) < sr * 0.3:
            info["excl_is_none"] = seg_excl is None
            info["excl_sec"] = 0.0 if seg_excl is None else len(seg_excl) / sr
            info["fallback_full_timeline"] = True
            segs = [rec_wav[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
            seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
        else:
            info["excl_is_none"] = False
            info["excl_sec"] = len(seg_excl) / sr
            info["fallback_full_timeline"] = False
            seg = seg_excl
        # 存独占帧原始(无论是否 fallback, 有则存)
        if seg_excl is not None and len(seg_excl) > 0:
            sf.write(os.path.join(outdir, f"rec_spk{i}_excl_raw.wav"), seg_excl, sr)
        # tile 到 1s(主线逻辑)
        if len(seg) < sr:
            info["tiled_to_1s"] = True
            info["pre_tile_sec"] = len(seg) / sr
            seg = np.tile(seg, sr // len(seg) + 1)[:sr]
        else:
            info["tiled_to_1s"] = False
            info["pre_tile_sec"] = len(seg) / sr
        info["emb_input_sec"] = len(seg) / sr
        sf.write(os.path.join(outdir, f"rec_spk{i}_emb_input.wav"), seg, sr)  # 实际喂 wespeaker 的
        emb = get_emb(seg)
        spk_embs.append(emb)
        spk_emb_info.append(info)
        print(f"  spk{i}({speakers[i]}): excl={info['excl_sec']:.2f}s fallback={info['fallback_full_timeline']} "
              f"tiled={info['tiled_to_1s']} emb_input={info['emb_input_sec']:.2f}s")

    # WS4 选 target
    print("\n[WS4] 选 target (余弦 argmax)")
    sims = {speakers[i]: float(torch.dot(enroll_emb, spk_embs[i])) for i in range(len(speakers))}
    target_idx = int(np.argmax([sims[s] for s in speakers]))
    target_speaker = speakers[target_idx]
    max_sim = sims[target_speaker]
    json.dump({"sims": sims, "target_idx": target_idx, "target_speaker": target_speaker,
               "max_sim": max_sim, "note": "argmax 选 target, 对比 WS-选对/选错"},
              open(os.path.join(outdir, "sims.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  sims={sims} → target=spk{target_idx}({target_speaker}) sim={max_sim:.3f}")

    # WS5 切 target timeline
    print("\n[WS5] cut_target_timeline (含重叠区)")
    target_audio = cut_target_timeline(rec_wav, per_spk[target_idx], sr=sr)
    sf.write(os.path.join(outdir, "target_slice.wav"), target_audio.astype(np.float32), sr)
    # 也存"假如选另一 speaker" 切片(对照用, 类似 2637)
    for i in range(len(speakers)):
        if i == target_idx: continue
        alt = cut_target_timeline(rec_wav, per_spk[i], sr=sr)
        sf.write(os.path.join(outdir, f"假如选spk{i}_当target.wav"), alt.astype(np.float32), sr)
    print(f"  target_slice {len(target_audio)/sr:.2f}s; 备选切片已存")

    # SepFormer 分离(放在 qwen 转写前, 一起进 batch)
    print("\n[SepFormer] separate recognition → 2 路")
    sep_out = separate(rec_wav)  # [n_src, T]
    n_src = sep_out.shape[0]
    print(f"  SepFormer 输出 {n_src} 路")
    sep_labels = [chr(ord('A') + i) for i in range(n_src)]
    for i, lbl in enumerate(sep_labels):
        sf.write(os.path.join(outdir, f"sep_source{lbl}.wav"), sep_out[i].astype(np.float32), sr)

    # WS6 qwen 转写(target_slice + sep_A + sep_B 一次 batch)
    print("\n[WS6] qwen transcribe (target_slice + sep sources 一起)")
    # 清空 qwen_dir 防止 stale
    for f in glob.glob(os.path.join(qwen_dir, "*.wav")): os.remove(f)
    target_uid = f"{uid_label}__target"
    sf.write(os.path.join(qwen_dir, f"{target_uid}.wav"), target_audio.astype(np.float32), sr)
    sep_uids = []
    for i, lbl in enumerate(sep_labels):
        u = f"{uid_label}__sep{lbl}"
        sep_uids.append(u)
        sf.write(os.path.join(qwen_dir, f"{u}.wav"), sep_out[i].astype(np.float32), sr)

    uid2text_path = os.path.join(qwen_dir, "_uid2text.json")
    py_qwen = os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe")
    subprocess.check_call([py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
                           "--slice-dir", qwen_dir, "--out", uid2text_path, "--seed", "42"])
    uid2text = json.load(open(uid2text_path, encoding="utf-8"))
    raw_text = uid2text.get(target_uid, "")
    print(f"  raw target text: {raw_text}")

    # WS7 后处理(逐步)★ 补 2637 缺
    print("\n[WS7] 后处理 4 步(to_simplified → digit_postproc → brand_homophone_fix)")
    t1 = to_simplified(raw_text)
    t2 = digit_postproc(t1)
    t3 = brand_homophone_fix(t2)
    steps = {"raw": raw_text, "to_simplified": t1, "+digit_postproc": t2, "+brand_homophone_fix": t3}
    json.dump(steps, open(os.path.join(outdir, "postprocess_steps.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for k, v in steps.items(): print(f"  {k:25s}: {v}")
    final_text = t3

    # WS8 拒识决策★
    print("\n[WS8] 拒识决策 (sim_thr=0.27 + content_gate)")
    sim_thr = 0.27
    sim_rejected = max_sim < sim_thr
    cg_valid = is_valid_command(final_text)
    cg_reject = (not sim_rejected) and (not cg_valid)
    reject_info = {
        "max_sim": max_sim, "sim_thr": sim_thr,
        "sim_rejected": sim_rejected,
        "content_gate_on": True,
        "is_valid_command": cg_valid,
        "content_gate_reject": cg_reject,
        "final_rejected": sim_rejected or cg_reject,
        "note": "pos 样本期望 final_rejected=False; max_sim 应过 thr 且 is_valid_command=True",
    }
    json.dump(reject_info, open(os.path.join(outdir, "reject_decision.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"  sim_rejected={sim_rejected} cg_valid={cg_valid} final_rejected={reject_info['final_rejected']}")

    # SepFormer 各路 sim + transcript
    print("\n[SepFormer] 各路 sim vs enroll_emb + 转写")
    sep_result = {}
    for i, lbl in enumerate(sep_labels):
        emb = get_emb(sep_out[i])
        sim = float(torch.dot(enroll_emb, emb))
        u = sep_uids[i]
        sep_result[lbl] = {
            "sim": sim,
            "transcript": uid2text.get(u, ""),
            "audio": f"sep_source{lbl}.wav",
            "picked_as_target": sim == max(v["sim"] for v in sep_result.values()) if sep_result else False,
        }
    # 标 picked
    if sep_result:
        best_src = max(sep_result.keys(), key=lambda k: sep_result[k]["sim"])
        for k in sep_result: sep_result[k]["picked_as_target"] = (k == best_src)
    json.dump(sep_result, open(os.path.join(outdir, "sepformer_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for lbl, r in sep_result.items():
        print(f"  source{lbl}: sim={r['sim']:.3f} picked={r['picked_as_target']} text={r['transcript']}")

    # summary
    summary = {
        "uid": uid_label, "uid_num": uid_num,
        "ref": best["ref"], "kws_txt": best["kws_txt"],
        "poc_sim": best["sim"], "poc_qwen_cer": best["qwen_cer"],
        "n_spk": best["n_spk"], "overlap_rate": overlap_rate,
        "audio_sec": audio_sec,
        "enr_n_spk": enr_n_spk,
        "speakers": speakers,
        "target_speaker": target_speaker, "target_idx": target_idx,
        "max_sim": max_sim,
        "final_text": final_text,
        "final_rejected": reject_info["final_rejected"],
        "spk_emb_info": spk_emb_info,
        "sep_result": sep_result,
        "poc_scan_top": [{"uid": s["uid"], "n_spk": s["n_spk"], "overlap": s["overlap"], "sim": s["sim"]} for s in scan[:15]],
    }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[done] {outdir}")
    print(f"\nSummary: uid={uid_label} overlap={overlap_rate:.3f} sim={max_sim:.3f} "
          f"final_text='{final_text}' ref='{best['ref']}'")
    return summary


if __name__ == "__main__":
    main()
