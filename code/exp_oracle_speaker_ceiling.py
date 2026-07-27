#!/usr/bin/env python3
"""Oracle speaker 天花板 POC(2026-07-27).

问题: 双人重叠场景 diar 次分是 CER 失败主因(已坐实: 失败组 75% 有重叠/97% 双人)。
本脚本: 在"双人重叠失败"子集上, 测"如果选对 speaker(而非 argmax), 能救回多少 CER"。
判别: 救得动→问题在选 target 策略; 救不动→必须改善分离本身。

链路(复刻 enroll_infer.py, 不动主线代码):
  diar(recognition) → speakers + per_spk timelines
  → 每 speaker i cut_target_timeline(含重叠区)切片存 wav
  → 复用 diar._embedding(wespeaker) + collect_clean_audio 算各 speaker sim(同 enroll_infer argmax 逻辑)
  → argmax target_idx(交叉验证应≈poc qwen_cer)
  → 批量 qwen 转写所有切片(独立 venv_qwen)
  → CERMetric 官方口径(to_simplified+digit_postproc+brand_homophone_fix)
  → oracle_CER = min(各 speaker CER); argmax_CER = argmax speaker CER

数据源: code/runs/poc_qwen_asr_full_result.json
  失败组: sim>=0.4 & qwen_cer>0.8 (取前 40)
  对照组: sim>=0.4 & qwen_cer<0.001 (取前 20)

用法: code/.venv/Scripts/python.exe code/exp_oracle_speaker_ceiling.py
产物: code/runs/_oracle_speaker/ (切片+uid2text+metadata+summary)
      docs/oracle_speaker_ceiling_A.md (报告)
"""
import inspect as _inspect  # speechbrain lazy proxy patch(同 enroll_infer.py:24-29)
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import os, sys, json, time, subprocess, glob
import numpy as np
import torch
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
import pyarrow  # 预热(避免 pyannote 扫 sys.path 崩)

from repro import resolve_model
from text_utils import to_simplified, cut_target_timeline, digit_postproc, brand_homophone_fix
from eval_metrics import CERMetric

# 复刻 enroll_infer.py:52-90 的 diar_mask 构造 + collect_clean_audio(声纹提独占帧避开重叠污染)
def get_diarization_mask(per_speaker_samples, audio_length):
    mask = torch.zeros(len(per_speaker_samples), audio_length)
    for i, spk_samples in enumerate(per_speaker_samples):
        for start, end in spk_samples:
            mask[i, round(start * 50):round(end * 50)] = 1
    return mask

def collect_clean_audio(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    others = diar_mask.sum(axis=0) - diar_mask[i]
    clean = (diar_mask[i] > 0) & (others == 0)
    T = clean.shape[0]
    pieces, idx = [], 0
    min_frames = int(min_seg_sec / frame_sec)
    while idx < T:
        if clean[idx]:
            start = idx
            while idx < T and clean[idx]:
                idx += 1
            if idx - start >= min_frames:
                pieces.append(audio[int(start * sr * frame_sec):int(idx * sr * frame_sec)])
        else:
            idx += 1
    return np.concatenate(pieces) if pieces else None


def submit_norm(text):
    """复刻 enroll_infer 提交链路归一(to_simplified → digit_postproc → brand_homophone_fix)。"""
    return brand_homophone_fix(digit_postproc(to_simplified(text or "")))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sr = 16000

    poc_path = os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json")
    out_dir = os.path.join(_HERE, "runs", "_oracle_speaker")
    slices_dir = os.path.join(out_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    d = json.load(open(poc_path, encoding="utf-8"))
    rows = d["rows"]
    fail_all = [r for r in rows if r["sim"] >= 0.4 and r["qwen_cer"] > 0.8]
    succ_all = [r for r in rows if r["sim"] >= 0.4 and r["qwen_cer"] < 0.001]
    fail = fail_all[:40]
    succ = succ_all[:20]
    print(f"[select] fail {len(fail)}/{len(fail_all)}, succ {len(succ)}/{len(succ_all)}")

    # ---- 加载 diar(一次) ----
    print(f"[load] DiariZen {resolve_model('DIAR')} on {device}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.inference_mode():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    # ---- 主循环: diar + 切片 + argmax ----
    meta = []  # 每条样本的元信息
    t0 = time.time()
    for ri, r in enumerate(fail + succ):
        uid = r["uid"]
        group = "fail" if ri < len(fail) else "succ"
        cmd_wav = os.path.join(_ROOT, "datasetA", "pos", f"{uid}.wav")
        kws_wav = os.path.join(_ROOT, "datasetA", "pos", f"kws_{uid[4:]}.wav")
        if not (os.path.isfile(cmd_wav) and os.path.isfile(kws_wav)):
            print(f"  [{ri+1}/{len(fail)+len(succ)}] {uid} SKIP missing audio")
            continue
        try:
            audio, _ = librosa.load(cmd_wav, sr=sr)
            enr_w, _ = librosa.load(kws_wav, sr=sr)
            enr_emb = get_emb(enr_w)
            diar_out = diar(cmd_wav)
            speakers = list(diar_out.labels())
            per_spk = [diar_out.label_timeline(s) for s in speakers]

            # argmax target(复刻 enroll_infer collect_clean_audio 逻辑, 交叉验证 POC)
            audio_len = int(librosa.get_duration(y=audio, sr=sr) * 50)
            # 用 mel 帧数对齐 enroll_infer; 退化估算: len(audio)/sr * 50
            diar_mask = get_diarization_mask(per_spk, audio_len)
            spk_embs = []
            for i in range(len(speakers)):
                seg = collect_clean_audio(audio, diar_mask, i, sr)
                if seg is None or len(seg) < sr * 0.3:
                    segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                    seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
                min_len = sr * 1
                if len(seg) < min_len:
                    seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
                spk_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enr_emb, e) for e in spk_embs]) if spk_embs else torch.tensor([])
            target_idx = int(torch.argmax(sims)) if len(sims) else -1

            # 切每 speaker timeline(含重叠)存盘
            spk_slice_files = []
            for i in range(len(speakers)):
                seg = cut_target_timeline(audio, per_spk[i], sr=sr)
                fn = os.path.join(slices_dir, f"{uid}_spk{i}.wav")
                import soundfile as sf
                sf.write(fn, seg.astype(np.float32), sr)
                spk_slice_files.append(fn)
            meta.append({
                "uid": uid, "group": group, "ref": r["ref"], "poc_qwen_cer": r["qwen_cer"],
                "poc_sim": r["sim"], "speakers": speakers,
                "sims": {speakers[i]: float(sims[i]) for i in range(len(speakers))},
                "target_idx": target_idx, "argmax_speaker": speakers[target_idx] if target_idx >= 0 else None,
                "n_spk": len(speakers), "slice_files": [os.path.basename(f) for f in spk_slice_files],
            })
            dt = time.time() - t0
            print(f"  [{ri+1}/{len(fail)+len(succ)}] {uid}({group}) n_spk={len(speakers)} argmax={target_idx} "
                  f"sim={[round(float(x),2) for x in sims]} ({dt:.0f}s)")
        except Exception as e:
            print(f"  [{ri+1}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "group": group, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    meta_path = os.path.join(out_dir, "meta.json")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[slice done] {sum(1 for m in meta if 'error' not in m)}/{len(fail)+len(succ)} ok → {meta_path}")

    # ---- qwen 批量转写所有切片(独立 venv) ----
    py_qwen = os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe")
    if not os.path.isfile(py_qwen):
        py_qwen = os.environ.get("PY_QWEN", py_qwen)
    uid2text_path = os.path.join(slices_dir, "_uid2text.json")
    print(f"\n[qwen] 批量转写 {slices_dir} (py={py_qwen}) ...")
    try:
        subprocess.check_call(
            [py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
             "--slice-dir", slices_dir, "--out", uid2text_path,
             "--seed", "42", "--batch-size", "16"])
    except subprocess.CalledProcessError as e:
        print(f"[qwen] FAIL rc={e.returncode}; 退到逐条模式重试一次")
        subprocess.check_call(
            [py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
             "--slice-dir", slices_dir, "--out", uid2text_path,
             "--seed", "42", "--batch-size", "0"])

    uid2text = json.load(open(uid2text_path, encoding="utf-8"))
    print(f"[qwen] {len(uid2text)} slices transcribed")

    # ---- 算 CER 每 speaker 切片 + 聚合 oracle/argmax ----
    summary = []  # 每条样本的 oracle vs argmax
    for m in meta:
        if "error" in m:
            continue
        uid = m["uid"]
        ref = m["ref"]
        ref_n = submit_norm(ref)
        per_spk_cer = []
        spk_texts = {}
        for i, fn in enumerate(m["slice_files"]):
            key = os.path.splitext(fn)[0]  # cmd_N_spkI
            txt = uid2text.get(key, "")
            txt_n = submit_norm(txt)
            cm = CERMetric(); cm.update([txt_n], [ref_n])
            cer = float(cm.compute()["cer"])
            per_spk_cer.append(cer)
            spk_texts[str(i)] = {"text": txt, "cer": cer}
        if not per_spk_cer:
            continue
        oracle_cer = min(per_spk_cer)
        oracle_idx = int(np.argmin(per_spk_cer))
        argmax_idx = m["target_idx"]
        argmax_cer = per_spk_cer[argmax_idx] if argmax_idx >= 0 and argmax_idx < len(per_spk_cer) else None
        summary.append({
            "uid": uid, "group": m["group"], "n_spk": m["n_spk"],
            "argmax_idx": argmax_idx, "argmax_speaker": m.get("argmax_speaker"),
            "argmax_cer": argmax_cer, "oracle_cer": oracle_cer, "oracle_idx": oracle_idx,
            "delta_argmax_oracle": (argmax_cer - oracle_cer) if argmax_cer is not None else None,
            "poc_qwen_cer": m["poc_qwen_cer"], "poc_sim": m["poc_sim"],
            "per_spk_cer": per_spk_cer, "speakers": m["speakers"],
            "sims": m["sims"], "spk_texts": spk_texts, "ref": ref,
        })

    summary_path = os.path.join(out_dir, "summary.json")
    json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 聚合打印 ----
    def agg(items):
        if not items:
            return None
        am = np.array([x["argmax_cer"] for x in items if x["argmax_cer"] is not None])
        om = np.array([x["oracle_cer"] for x in items])
        rescued = [x for x in items if x["delta_argmax_oracle"] is not None and x["delta_argmax_oracle"] > 0.1]
        stuck = [x for x in items if x["delta_argmax_oracle"] is not None and x["delta_argmax_oracle"] <= 0.1]
        rescue_amounts = [x["delta_argmax_oracle"] for x in rescued]
        return {
            "n": len(items),
            "argmax_mean_cer": float(am.mean()) if len(am) else None,
            "oracle_mean_cer": float(om.mean()),
            "delta": float(am.mean() - om.mean()) if len(am) else None,
            "rescued_n": len(rescued),
            "stuck_n": len(stuck),
            "rescue_mean_drop": float(np.mean(rescue_amounts)) if rescue_amounts else 0.0,
            "rescued_uids": [x["uid"] for x in rescued],
            "stuck_uids": [x["uid"] for x in stuck],
        }
    fail_items = [s for s in summary if s["group"] == "fail"]
    succ_items = [s for s in summary if s["group"] == "succ"]
    fail_agg = agg(fail_items)
    succ_agg = agg(succ_items)
    out = {"fail": fail_agg, "succ": succ_agg}
    json.dump(out, open(os.path.join(out_dir, "agg.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 聚合 =====")
    print(f"[FAIL] n={fail_agg['n']} argmax_mean={fail_agg['argmax_mean_cer']:.3f} "
          f"oracle_mean={fail_agg['oracle_mean_cer']:.3f} Δ={fail_agg['delta']:.3f}")
    print(f"       rescued={fail_agg['rescued_n']} stuck={fail_agg['stuck_n']} "
          f"rescue_mean_drop={fail_agg['rescue_mean_drop']:.3f}")
    print(f"[SUCC] n={succ_agg['n']} argmax_mean={succ_agg['argmax_mean_cer']:.3f} "
          f"oracle_mean={succ_agg['oracle_mean_cer']:.3f} Δ={succ_agg['delta']:.3f}")
    print(f"\n[done] → {summary_path}, {out_dir}/agg.json")


if __name__ == "__main__":
    main()
