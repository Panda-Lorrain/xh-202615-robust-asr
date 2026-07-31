"""exp_pipeline_steps.py — 把 SepFormer 分离管线的每一步中间音频 dump 成 wav, 供用户逐步听.

任务核心约束(**不报 CER, 不下好坏结论**): 每步只客观描述"这步做了什么处理", 好坏由
用户亲耳/亲眼评. 这是验收 multi-voice 路线证伪的"摆样本"工具, 不是判别实验.

样本(3 个, 各代表一类):
  - cmd_146:  B 类主战场(argmax 完美转写"打开观影模式"), 已有 _verify_mv_fail 产物可复用
  - cmd_2637: 双人重叠(diar 内容分对但 cut_timeline 切了重叠区), 典型重叠 bug
  - cmd_18:   死区(sim 极低), 用来对照死区是不是分离问题(若 diar 只 1 人, 如实展示)

每个样本输出到 code/runs/_pipeline_steps/<uid>/:
  00_enrollment.wav        — 原始 enrollment(目标说话人参考)
  01_recognition.wav       — 原始 recognition 混音
  02_spk{i}_exclusive.wav  — diar 分出 spk{i} 独占段(不含重叠帧)拼接(每有一个 spk 一个文件)
  02_overlap_region.wav    — 重叠区(>=2 speaker 同时活跃帧)音频(无重叠则跳过)
  03_slice_full.wav        — cut_target_timeline 切的 target 全 timeline(含重叠区)= 主线喂 ASR 切片
  03_slice_exclusive.wav   — 只切 target 独占段(不含重叠区)的切片(对照; 无独占段则跳过)
  04_sepformer_input.wav   — 实际喂 SepFormer 的音频(代码已确认 = recognition 原混音)
  05_sep_srcA.wav          — SepFormer 输出源 A
  05_sep_srcB.wav          — SepFormer 输出源 B
  README.md                — 客观逐步描述(不评价好坏, 不报 CER)
  meta.json                — diar speakers / target_idx / sims / overlap_ratio 等结构化元信息

复用:
  - cmd_146 的 sep_sourceA/B + argmax_target_slice + enrollment + recognition_original
    直接拷自 code/runs/_verify_mv_fail/B_cmd_146/
  - cmd_18/2637 的 argmax_target_slice 拷自 code/stability_matrix/_slices/cmd_<N>.wav
  - 其余(diar 独占段/重叠区/SepFormer cmd_18,2637)现场算

环境:
  code/.venv(uv run) + speechbrain/sepformer-whamr16k(E:/hf_cache) + diar(复用 enroll_infer sys.path)
  不调 qwen(本任务只 dump 音频, 不强制转写).

用法:
  code/.venv/Scripts/python.exe code/exp_pipeline_steps.py
"""
import os, sys, json, shutil, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容(复刻 exp_sepformer_qwen.py, SB 1.1.0 LazyModule inspect-guard) ----
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import speechbrain.utils.importutils as _sb_iu
import inspect as _inspect, importlib as _importlib


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

# 触发 enroll_infer 内部 DiariZen/pyannote sys.path 注册 + speechbrain patch
from enroll_infer import get_diarization_mask, collect_clean_audio  # noqa: F401
from text_utils import cut_target_timeline  # noqa: E402
from repro import set_global_seed, resolve_model  # noqa: E402

import numpy as np
import torch
import librosa
import soundfile as sf


# ---------- diar Mask 派生音频(本任务专用, 参照 collect_clean_audio 复刻) ----------
def collect_overlap_audio(audio, diar_mask, sr=16000, frame_sec=0.02, min_seg_sec=0.05):
    """从 diar_mask 提取重叠区音频(>=2 speaker 同时活跃的帧), 按 >=min_seg_sec 连续段拼接.

    返回 (ndarray|None, list[(start_frame, end_frame)]). None 表示该样本无重叠帧.
    """
    overlap = (diar_mask.sum(axis=0) >= 2)
    T = overlap.shape[0]
    pieces, spans, idx = [], [], 0
    min_frames = int(min_seg_sec / frame_sec)
    while idx < T:
        if overlap[idx]:
            start = idx
            while idx < T and overlap[idx]:
                idx += 1
            if idx - start >= min_frames:
                spans.append((int(start), int(idx)))
                pieces.append(audio[int(start * sr * frame_sec):int(idx * sr * frame_sec)])
        else:
            idx += 1
    if not pieces:
        return None, []
    return np.concatenate(pieces), spans


def collect_speaker_exclusive(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    """= enroll_infer.collect_clean_audio, 多返回段信息(用于 README 描述)."""
    others = diar_mask.sum(axis=0) - diar_mask[i]
    clean = (diar_mask[i] > 0) & (others == 0)
    T = clean.shape[0]
    pieces, spans, idx = [], [], 0
    min_frames = int(min_seg_sec / frame_sec)
    while idx < T:
        if clean[idx]:
            start = idx
            while idx < T and clean[idx]:
                idx += 1
            if idx - start >= min_frames:
                spans.append((int(start), int(idx)))
                pieces.append(audio[int(start * sr * frame_sec):int(idx * sr * frame_sec)])
        else:
            idx += 1
    if not pieces:
        return None, []
    return np.concatenate(pieces), spans


def save_wav(path, wav_np, sr=16000):
    if wav_np is None:
        return False
    sf.write(path, np.ascontiguousarray(wav_np.astype(np.float32)), sr)
    return True


def _fmt_spans_sec(spans, frame_sec=0.02):
    """帧区间 → 秒区间字符串列表."""
    return [(round(s * frame_sec, 3), round(e * frame_sec, 3)) for s, e in spans]


# ---------- 单样本处理 ----------
def process_one(uid, pair, args, diar, sep_model, get_emb, device, out_root):
    uid_num = int(uid.split("_")[1])
    enr_src = pair["enrollment"]
    rec_src = pair["recognition"]
    ref_txt = pair.get("ref", "")
    out_dir = os.path.join(out_root, uid)
    os.makedirs(out_dir, exist_ok=True)

    # 唤醒词(可选, pos.jsonl)
    kws_txt = ""
    pos_jsonl = os.path.join(_ROOT, "datasetA", "pos.jsonl")
    if os.path.exists(pos_jsonl):
        for line in open(pos_jsonl, encoding="utf-8"):
            d = json.loads(line)
            if d.get("id") == uid_num:
                kws_txt = d.get("唤醒文本", "")
                break

    # 优先复用: cmd_146 从 _verify_mv_fail/B_cmd_146 拷
    verify_dir = os.path.join(_HERE, "runs", "_verify_mv_fail", f"B_{uid}")
    verify_available = os.path.isdir(verify_dir)
    # argmax slice 全部样本都从 stability_matrix/_slices 拷
    stability_slice = os.path.join(_HERE, "stability_matrix", "_slices", f"{uid}.wav")

    meta = {"uid": uid, "uid_num": uid_num, "ref": ref_txt, "kws_txt": kws_txt,
            "enrollment_src": enr_src, "recognition_src": rec_src, "steps": {}, "skipped": []}

    # ---- 00 / 01: 原始音频(优先复用 verify_mv_fail, 否则拷数据集) ----
    f00 = os.path.join(out_dir, "00_enrollment.wav")
    f01 = os.path.join(out_dir, "01_recognition.wav")
    enr_src_use = os.path.join(verify_dir, "enrollment.wav") if verify_available else enr_src
    rec_src_use = os.path.join(verify_dir, "recognition_original.wav") if verify_available else rec_src
    shutil.copy2(enr_src_use, f00)
    shutil.copy2(rec_src_use, f01)
    meta["steps"]["00_enrollment"] = {"src": enr_src_use}
    meta["steps"]["01_recognition"] = {"src": rec_src_use}

    # ---- 读 recognition wav 算 diar ----
    audio, sr = librosa.load(rec_src_use, sr=16000)
    audio_dur = len(audio) / sr
    meta["recognition_sec"] = round(audio_dur, 3)

    # diar(DiariZen pyannote Pipeline) → per_spk timeline
    try:
        diar_out = diar(rec_src_use)
    except Exception as e:
        meta["diar_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        meta["skipped"].append("diar_failed")
        # 仍拷 argmax slice 和 sep 输入(无 diar 不能算 02_* / 03_slice_exclusive)
        if os.path.exists(stability_slice):
            shutil.copy2(stability_slice, os.path.join(out_dir, "03_slice_full.wav"))
            meta["steps"]["03_slice_full"] = {"src": stability_slice,
                                               "note": "diar 失败, 此文件复用主线已存的切片, 未现场重算"}
        # SepFormer 输入 = recognition
        shutil.copy2(f01, os.path.join(out_dir, "04_sepformer_input.wav"))
        meta["steps"]["04_sepformer_input"] = {"src": f01,
                                               "note": "代码确认 SepFormer 输入 = recognition 原混音(非切片)"}
        _maybe_reuse_or_run_sep(uid, out_dir, verify_dir, audio, sep_model, args, meta, skip_sep=False)
        _write_readme(out_dir, uid, meta)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    speakers = list(diar_out.labels())
    per_spk = [diar_out.label_timeline(s) for s in speakers]
    # 转 [(start, end), ...] 列表(供 cut_target_timeline 用)
    per_spk_segs = [[(float(s), float(e)) for s, e in tl] for tl in per_spk]

    # diar_mask @ 50Hz
    audio_len_frames = int(len(audio) / sr * 50) + 1
    diar_mask = get_diarization_mask(per_spk_segs, audio_len_frames)

    # enrollment + 各 speaker emb, argmax target
    w_enr, _ = librosa.load(enr_src_use, sr=16000)
    enroll_emb = get_emb(w_enr)
    spk_embs = []
    for i in range(len(speakers)):
        seg = collect_clean_audio(audio, diar_mask, i, sr)
        if seg is None or len(seg) < sr * 0.3:
            segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk_segs[i]]
            seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
        if len(seg) < sr:
            seg = np.tile(seg, sr // len(seg) + 1)[:sr]
        spk_embs.append(get_emb(seg))
    sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
    target_idx = int(torch.argmax(sims))

    # 各 speaker 独占时长统计
    spk_excl_sec = []
    for i in range(len(speakers)):
        clean_frames = int(((diar_mask[i] > 0) & ((diar_mask.sum(axis=0) - diar_mask[i]) == 0)).sum())
        spk_excl_sec.append(round(clean_frames * 0.02, 3))
    overlap_frames = int((diar_mask.sum(axis=0) >= 2).sum())
    overlap_sec = round(overlap_frames * 0.02, 3)
    overlap_ratio = round(overlap_frames / diar_mask.shape[1], 4) if diar_mask.shape[1] else 0.0

    meta.update({
        "n_speakers": len(speakers),
        "speakers": speakers,
        "per_spk_segs_sec": [[(round(s, 3), round(e, 3)) for s, e in tl] for tl in per_spk_segs],
        "spk_exclusive_sec": spk_excl_sec,
        "overlap_sec": overlap_sec,
        "overlap_ratio": overlap_ratio,
        "sims_enroll_vs_spk": [round(float(x), 4) for x in sims],
        "target_idx": target_idx,
        "target_sim": round(float(sims[target_idx]), 4),
        "diar_mask_frames": int(diar_mask.shape[1]),
    })

    # ---- 02_spk{i}_exclusive: 每 speaker 独占段 ----
    for i in range(len(speakers)):
        wav_excl, spans = collect_speaker_exclusive(audio, diar_mask, i, sr)
        fname = os.path.join(out_dir, f"02_spk{i}_exclusive.wav")
        if wav_excl is None:
            meta["skipped"].append(f"02_spk{i}_exclusive(no exclusive frames)")
            meta["steps"][f"02_spk{i}_exclusive"] = {"wav_sec": 0.0, "n_segs": 0, "spans_sec": [],
                                                     "note": f"speaker {speakers[i]} 无 >=0.3s 独占连续段"}
        else:
            save_wav(fname, wav_excl, sr)
            meta["steps"][f"02_spk{i}_exclusive"] = {
                "wav_sec": round(len(wav_excl) / sr, 3), "n_segs": len(spans),
                "spans_sec": _fmt_spans_sec(spans),
            }

    # ---- 02_overlap_region: 重叠区 ----
    wav_ov, ov_spans = collect_overlap_audio(audio, diar_mask, sr)
    fname = os.path.join(out_dir, "02_overlap_region.wav")
    if wav_ov is None:
        meta["skipped"].append("02_overlap_region(no overlap frames)")
        meta["steps"]["02_overlap_region"] = {"wav_sec": 0.0, "n_segs": 0, "spans_sec": [],
                                              "note": "无 >=2 speaker 同时活跃帧(单人 diar 或无重叠)"}
    else:
        save_wav(fname, wav_ov, sr)
        meta["steps"]["02_overlap_region"] = {
            "wav_sec": round(len(wav_ov) / sr, 3), "n_segs": len(ov_spans),
            "spans_sec": _fmt_spans_sec(ov_spans),
        }

    # ---- 03_slice_full: cut_target_timeline(含重叠区) ----
    # 优先复用主线 stability_matrix/_slices(等价于现场算 cut_target_timeline), 节省不必要再算
    f03f = os.path.join(out_dir, "03_slice_full.wav")
    if os.path.exists(stability_slice):
        shutil.copy2(stability_slice, f03f)
        meta["steps"]["03_slice_full"] = {"src": stability_slice,
                                          "note": "复用主线 enroll_infer qwen 切片(== cut_target_timeline(per_spk[target_idx]), 含重叠区)"}
    else:
        tgt_audio = cut_target_timeline(audio, per_spk_segs[target_idx], sr=sr)
        save_wav(f03f, tgt_audio, sr)
        meta["steps"]["03_slice_full"] = {"wav_sec": round(len(tgt_audio) / sr, 3),
                                          "note": "现场 cut_target_timeline(per_spk[target_idx]) 含重叠区"}
    # 同时现场重算一份用于校验(不写文件, 只记时长差异)
    tgt_audio_check = cut_target_timeline(audio, per_spk_segs[target_idx], sr=sr)
    meta["steps"]["03_slice_full"]["recomputed_sec"] = round(len(tgt_audio_check) / sr, 3)

    # ---- 03_slice_exclusive: 只切 target 独占段(= collect_clean_audio(target_idx)) ----
    wav_excl_t, spans_t = collect_speaker_exclusive(audio, diar_mask, target_idx, sr)
    f03e = os.path.join(out_dir, "03_slice_exclusive.wav")
    if wav_excl_t is None:
        meta["skipped"].append("03_slice_exclusive(no exclusive frames for target)")
        meta["steps"]["03_slice_exclusive"] = {"wav_sec": 0.0, "n_segs": 0, "spans_sec": [],
                                               "note": f"target(sp{speakers[target_idx]}) 无 >=0.3s 独占段(全在重叠或太碎)"}
    else:
        save_wav(f03e, wav_excl_t, sr)
        meta["steps"]["03_slice_exclusive"] = {
            "wav_sec": round(len(wav_excl_t) / sr, 3), "n_segs": len(spans_t),
            "spans_sec": _fmt_spans_sec(spans_t),
        }

    # ---- 04_sepformer_input: = recognition 原混音(代码已确认) ----
    f04 = os.path.join(out_dir, "04_sepformer_input.wav")
    shutil.copy2(f01, f04)
    meta["steps"]["04_sepformer_input"] = {"src": f01,
                                           "note": "代码确认(exp_sepformer_b2.py:147 librosa.load(rec)) SepFormer 输入 = recognition 原混音, 非 target slice"}

    # ---- 05_sep_srcA/B ----
    _maybe_reuse_or_run_sep(uid, out_dir, verify_dir, audio, sep_model, args, meta)

    # ---- README + meta ----
    _write_readme(out_dir, uid, meta)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _maybe_reuse_or_run_sep(uid, out_dir, verify_dir, audio, sep_model, args, meta, skip_sep=False):
    """05_sep_srcA/B: 优先复用 _verify_mv_fail(B_cmd_146), 否则现场 SepFormer 分离."""
    fA = os.path.join(out_dir, "05_sep_srcA.wav")
    fB = os.path.join(out_dir, "05_sep_srcB.wav")
    reuse_A = os.path.join(verify_dir, "sep_sourceA.wav")
    reuse_B = os.path.join(verify_dir, "sep_sourceB.wav")
    if os.path.isdir(verify_dir) and os.path.exists(reuse_A) and os.path.exists(reuse_B) and not skip_sep:
        shutil.copy2(reuse_A, fA)
        shutil.copy2(reuse_B, fB)
        meta["steps"]["05_sep_srcA"] = {"src": reuse_A,
                                        "note": "复用 _verify_mv_fail 已跑的 SepFormer 输出(同模型 sepformer-whamr16k)"}
        meta["steps"]["05_sep_srcB"] = {"src": reuse_B,
                                        "note": "复用 _verify_mv_fail 已跑的 SepFormer 输出(同模型 sepformer-whamr16k)"}
        return
    if skip_sep or sep_model is None:
        meta["skipped"].append("05_sep_srcA/B (sep_model unavailable)")
        return
    # 现场分离
    from exp_sepformer_qwen import separate  # 局部 import 避免顶层 torch 掺杂
    sources = separate(audio, sep_model)  # [n_src, T]
    n_src = sources.shape[0]
    for i in range(min(n_src, 2)):
        fname = fA if i == 0 else fB
        save_wav(fname, sources[i], 16000)
        meta["steps"][f"05_sep_src{'A' if i == 0 else 'B'}"] = {
            "wav_sec": round(len(sources[i]) / 16000, 3),
            "src_index_in_sepformer": i,
            "note": f"现场 SepFormer 分离第 {i} 路输出(SepFormer 不指定哪路是 target, 两路平权)",
        }
    if n_src > 2:
        meta["steps"]["05_extra_srcs"] = {"n_src": n_src, "note": f"SepFormer 输出 {n_src} 路, 仅存前 2"}
    elif n_src < 2:
        meta["steps"]["05_extra_srcs"] = {"n_src": n_src, "note": f"SepFormer 仅输出 {n_src} 路"}


# ---------- README ----------
def _write_readme(out_dir, uid, meta):
    ref = meta.get("ref", "")
    kws = meta.get("kws_txt", "")
    rec_sec = meta.get("recognition_sec", "?")
    n_spk = meta.get("n_speakers", "?")
    speakers = meta.get("speakers", [])
    sims = meta.get("sims_enroll_vs_spk", [])
    target_idx = meta.get("target_idx", None)
    target_sim = meta.get("target_sim", None)
    overlap_sec = meta.get("overlap_sec", 0)
    overlap_ratio = meta.get("overlap_ratio", 0)
    spk_excl = meta.get("spk_exclusive_sec", [])

    lines = []
    lines.append(f"# {uid} — pipeline 逐步音频\n")
    lines.append("本目录把 SepFormer 分离管线的每一步中间音频 dump 成 wav, 供逐步听/看.")
    lines.append("**这是客观摆样本, 不报 CER, 不下好坏结论. 每步只描述'这步做了什么', 好坏由你听自己评.**\n")
    lines.append("## 元信息")
    lines.append(f"- 目标说话人真值(ref): `{ref}`")
    lines.append(f"- enrollment 唤醒文本(kws): `{kws}`")
    lines.append(f"- recognition 时长: {rec_sec}s")
    if n_spk == "?":
        lines.append(f"- diar 状态: **失败**({meta.get('diar_error', '未知')}) → 02_* / 03_slice_exclusive 未产出")
    else:
        lines.append(f"- diar(DiariZen) 分出说话人数: {n_spk} ({', '.join(str(s) for s in speakers)})")
        sim_str = ", ".join(f"{speakers[i]}:{sims[i]:.3f}" for i in range(len(speakers))) if sims else ""
        tgt_label = speakers[target_idx] if target_idx is not None else "?"
        sim_display = f"{target_sim:.3f}" if target_sim is not None else "?"
        lines.append(f"- 各 speaker vs enrollment 余弦 sim: {sim_str}")
        lines.append(f"- argmax 选 target: {tgt_label} (sim={sim_display})")
        lines.append(f"- 各 speaker 独占段时长(秒): {spk_excl}")
        lines.append(f"- 重叠区时长: {overlap_sec}s (占 recognition {overlap_ratio*100:.1f}%)")
    lines.append("")

    lines.append("## 步骤音频\n")
    step_descriptions = [
        ("00_enrollment.wav", "原始 enrollment 音频",
         "无处理, 直接拷贝数据集 enrollment(目标说话人 ~1.8s 短参考音频, 用于算 wespeaker 声纹).",
         "数据集 kws wav", "同上(16k mono)"),
        ("01_recognition.wav", "原始 recognition 混音",
         "无处理, 直接拷贝数据集 recognition(带噪 + 多人重叠的识别音频, 是整个管线的输入).",
         "数据集 cmd wav", "同上(16k mono)"),
    ]
    for fname, title, desc, inp, outp in step_descriptions:
        lines.append(f"### {fname} — {title}")
        lines.append(f"- 这步做了什么: {desc}")
        lines.append(f"- 输入: {inp}")
        lines.append(f"- 输出: {outp}")
        if fname in meta.get("steps", {}):
            lines.append(f"- 来源: `{meta['steps'][fname].get('src', '')}`")
        lines.append("")

    # 02_spk{i}_exclusive
    spk_keys = sorted([k for k in meta.get("steps", {}) if k.startswith("02_spk") and k.endswith("_exclusive")])
    for k in spk_keys:
        i = k.replace("02_spk", "").replace("_exclusive", "")
        info = meta["steps"][k]
        lines.append(f"### {k}.wav — diar 分出 spk{i} 的独占段拼接")
        lines.append(f"- 这步做了什么: diar(DiariZen/VBx) 分出 spk{i} 在 recognition 里的所有时间段; "
                     f"只保留**独占帧**(其他 speaker 不活跃, 避开重叠区污染), 按 >=0.3s 连续段拼接.")
        lines.append(f"- 输入: recognition wav + diar 输出 spk{i} timeline")
        if info.get("wav_sec", 0) == 0:
            lines.append(f"- 输出: **未产出** — {info.get('note', '')}")
        else:
            lines.append(f"- 输出: {info['wav_sec']}s 拼接音频, {info['n_segs']} 段(秒区间): {info['spans_sec']}")
        lines.append("")

    # 02_overlap_region
    if "02_overlap_region" in meta.get("steps", {}):
        info = meta["steps"]["02_overlap_region"]
        lines.append(f"### 02_overlap_region.wav — 重叠区音频(>=2 speaker 同时活跃)")
        lines.append(f"- 这步做了什么: 从 diar_mask 找出 >=2 speaker 同时活跃的帧, 按 >=0.05s 连续段拼接.")
        lines.append(f"- 输入: recognition wav + diar_mask")
        if info.get("wav_sec", 0) == 0:
            lines.append(f"- 输出: **未产出** — {info.get('note', '')}")
        else:
            lines.append(f"- 输出: {info['wav_sec']}s 拼接音频, {info['n_segs']} 段(秒区间): {info['spans_sec']}")
        lines.append("")

    # 03_slice_full
    if "03_slice_full" in meta.get("steps", {}):
        info = meta["steps"]["03_slice_full"]
        lines.append(f"### 03_slice_full.wav — target 全 timeline 切片(含重叠区) = 主线喂 ASR 的切片")
        lines.append(f"- 这步做了什么: cut_target_timeline 切 target speaker 的整条 timeline(= per_spk[target_idx])"
                     f"并拼接; **包含重叠区**(target 与其他 speaker 同时说话的部分也收进切片).")
        lines.append(f"- 输入: recognition wav + target timeline")
        if info.get("recomputed_sec") is not None:
            lines.append(f"- 输出: {info['recomputed_sec']}s 切片(现场重算); 主线已存切片复用自 `{info.get('src', '')}`")
        else:
            lines.append(f"- 输出: {info.get('wav_sec', '?')}s 切片")
        lines.append(f"- 注: 这是当前主线 enroll_infer --asr-backend {{qwen,vanilla}} 真正喂给 ASR 的音频.")
        lines.append("")

    # 03_slice_exclusive
    if "03_slice_exclusive" in meta.get("steps", {}):
        info = meta["steps"]["03_slice_exclusive"]
        lines.append(f"### 03_slice_exclusive.wav — 只切 target 独占段(不含重叠区), 对照用")
        lines.append(f"- 这步做了什么: = collect_clean_audio(audio, diar_mask, target_idx), "
                     f"只取 target 独占帧(>=0.3s 连续段), **不含重叠区**. 与 03_slice_full 对照听.")
        lines.append(f"- 输入: recognition wav + diar_mask(target 独占帧掩码)")
        if info.get("wav_sec", 0) == 0:
            lines.append(f"- 输出: **未产出** — {info.get('note', '')}")
        else:
            lines.append(f"- 输出: {info['wav_sec']}s 拼接音频, {info['n_segs']} 段(秒区间): {info['spans_sec']}")
        lines.append("")

    # 04_sepformer_input
    if "04_sepformer_input" in meta.get("steps", {}):
        info = meta["steps"]["04_sepformer_input"]
        lines.append(f"### 04_sepformer_input.wav — 实际喂 SepFormer 的音频")
        lines.append(f"- 这步做了什么: **无处理**(就是 recognition 原混音的拷贝). "
                     f"从代码确认(exp_sepformer_b2.py:147 `audio, sr = librosa.load(rec, sr=16000)` → "
                     f"`separate(audio, sep_model)`), SepFormer 的输入是 **recognition 原混音**, "
                     f"不是 target slice, 也不是 diar 切出来的任何片段.")
        lines.append(f"- 输入: = 01_recognition.wav")
        lines.append(f"- 输出: 同上(16k mono)")
        lines.append("")

    # 05_sep_srcA/B
    for k in ["05_sep_srcA", "05_sep_srcB"]:
        if k in meta.get("steps", {}):
            info = meta["steps"][k]
            idx = "A" if "A" in k else "B"
            lines.append(f"### {k}.wav — SepFormer 输出源 {idx}")
            lines.append(f"- 这步做了什么: SepFormer(speechbrain/sepformer-whamr16k, 英文 WHAM+reverb 权重) "
                         f"对 04_sepformer_input 做 2 路盲分离, 输出 src{idx} 这一路. "
                         f"SepFormer 不指定哪路是 target, 两路平权输出(A/B 顺序由模型内部决定).")
            lines.append(f"- 输入: 04_sepformer_input.wav")
            if "wav_sec" in info:
                lines.append(f"- 输出: {info['wav_sec']}s(16k mono); {info.get('note', '')}")
            else:
                lines.append(f"- 来源: `{info.get('src', '')}`; {info.get('note', '')}")
            lines.append("")

    # skipped 总结
    if meta.get("skipped"):
        lines.append("## 因样本特性跳过的步骤")
        for s in meta["skipped"]:
            lines.append(f"- {s}")
        lines.append("")

    lines.append("## 听音建议(顺序)")
    lines.append("1. `00_enrollment.wav` 记住 target 说话人声纹.")
    lines.append("2. `01_recognition.wav` 整体听混音, 找 target 在哪段说话、有几人重叠.")
    lines.append("3. `02_spk0_exclusive.wav` / `02_spk1_exclusive.wav` — diar 分出的两路独占段, 判断 diar 分对没.")
    lines.append("4. `02_overlap_region.wav` — 重叠区(若有), 听这段是不是物理上两人同时说话.")
    lines.append("5. `03_slice_full.wav` — 主线喂 ASR 的切片, 听 target 在这里面是否清晰(尤其是含重叠区那段).")
    lines.append("6. `03_slice_exclusive.wav` — 同样 target 但只独占段, 与 03_slice_full 对照听重叠区对切片的影响.")
    lines.append("7. `04_sepformer_input.wav` — 应和 01 一模一样, 确认 SepFormer 吃的是原混音.")
    lines.append("8. `05_sep_srcA.wav` / `05_sep_srcB.wav` — SepFormer 分离后两路, 听每路听感、有无 artifact.")
    lines.append("")
    lines.append("> 你自己听完后再判断每步处理对/错/有无 artifact. 本目录不下任何结论.")

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="dump SepFormer 分离管线每步中间音频")
    ap.add_argument("--uids", default="cmd_146,cmd_2637,cmd_18",
                    help="逗号分 uid 列表(默认 3 个代表样本)")
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-root", default=os.path.join(_HERE, "runs", "_pipeline_steps"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sepformer", action="store_true",
                    help="只跑 diar + 切片, 不加载 SepFormer(cmd_146 已有可复用就走这条快路径)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_root, exist_ok=True)

    pairs_all = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {}
    for p in pairs_all:
        uid = p.get("uid") or os.path.splitext(os.path.basename(p["recognition"]))[0]
        uid2pair[uid] = p

    uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    print(f"[plan] uids = {uids}")
    print(f"[plan] out_root = {args.out_root}")
    print(f"[plan] device = {device}")

    # ---- 加载模型(按需) ----
    # diar 总是要算(02/03_slice_exclusive 依赖)
    print(f"[load] DiariZen diar {args.diarization_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.no_grad():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    # SepFormer: 是否所有 uid 都有 _verify_mv_fail 可复用?
    need_sep = False
    for uid in uids:
        verify_dir = os.path.join(_HERE, "runs", "_verify_mv_fail", f"B_{uid}")
        if not (os.path.isdir(verify_dir)
                and os.path.exists(os.path.join(verify_dir, "sep_sourceA.wav"))
                and os.path.exists(os.path.join(verify_dir, "sep_sourceB.wav"))):
            need_sep = True
            break
    sep_model = None
    if need_sep and not args.skip_sepformer:
        print(f"[load] SepFormer sepformer-whamr16k → {args.sepformer_dir}")
        from exp_sepformer_qwen import load_sepformer
        sep_model = load_sepformer(device, args.sepformer_dir)
        print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")
    elif not need_sep:
        print(f"[load] 所有 uid 都有 _verify_mv_fail 可复用, 跳过加载 SepFormer")
    else:
        print(f"[load] --skip-sepformer: 05_sep_srcA/B 仅对可复用 uid 产出")

    # ---- 主循环 ----
    t0 = time.time()
    summary = []
    for n, uid in enumerate(uids):
        if uid not in uid2pair:
            print(f"[{n+1}/{len(uids)}] {uid} 不在 pairs, 跳过")
            continue
        print(f"\n[{n+1}/{len(uids)}] === {uid} ===")
        try:
            m = process_one(uid, uid2pair[uid], args, diar, sep_model, get_emb, device, args.out_root)
            summary.append(m)
            print(f"  done: n_spk={m.get('n_speakers')} target_idx={m.get('target_idx')} "
                  f"overlap={m.get('overlap_sec')}s skipped={m.get('skipped', [])}")
        except Exception as e:
            import traceback
            print(f"  FAIL {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()
            summary.append({"uid": uid, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    # 顶层 README + 索引
    top = ["# _pipeline_steps — SepFormer 分离管线逐步音频索引", "",
           "本目录把每个样本的 SepFormer 分离管线每一步中间音频 dump 成 wav.",
           "**不报 CER, 不下好坏结论.** 用户亲自听每步音频, 自己判断每步处理的好坏.\n",
           "## 步骤说明(每个样本目录里都有)", "",
           "| 文件 | 这步做了什么 |",
           "|---|---|",
           "| 00_enrollment.wav | 数据集 enrollment(目标说话人 ~1.8s 参考), 无处理 |",
           "| 01_recognition.wav | 数据集 recognition 混音, 无处理 |",
           "| 02_spk{i}_exclusive.wav | diar 分出 spk{i} 独占段(避开重叠区)拼接, 每 speaker 一个 |",
           "| 02_overlap_region.wav | 重叠区(>=2 speaker 同时活跃)音频; 无重叠则不产出 |",
           "| 03_slice_full.wav | cut_target_timeline 切 target 全 timeline(**含重叠区**) = 主线喂 ASR 的切片 |",
           "| 03_slice_exclusive.wav | 只切 target 独占段(不含重叠区), 对照用 |",
           "| 04_sepformer_input.wav | 实际喂 SepFormer 的音频(**代码确认 = recognition 原混音, 非 slice**) |",
           "| 05_sep_srcA/B.wav | SepFormer 输出两路(模型不指定哪路是 target, 两路平权) |",
           "| README.md | 该样本逐步客观描述(不评价好坏) |",
           "| meta.json | diar speakers / target_idx / sims / overlap_ratio 等结构化元信息 |",
           "",
           "## 样本清单", ""]
    for m in summary:
        if "error" in m:
            top.append(f"- **{m['uid']}** — ERROR: {m['error']}")
            continue
        skipped = m.get("skipped", [])
        skip_str = f" (跳过: {', '.join(skipped)})" if skipped else ""
        top.append(f"- **{m['uid']}** — ref=`{m.get('ref', '')}` "
                   f"n_spk={m.get('n_speakers', '?')} "
                   f"overlap={m.get('overlap_sec', 0)}s "
                   f"target_sim={m.get('target_sim', '?')}{skip_str}")
    top += ["", "## 关键代码结论", "",
            "- **SepFormer 输入 = recognition 原混音**, 非 target slice. 证据: `exp_sepformer_b2.py:147` "
            "`audio, sr = librosa.load(rec, sr=16000)` → `sources = separate(audio, sep_model)`.",
            "- **cut_target_timeline 含重叠区**: `text_utils.cut_target_timeline` 拼接 target 全 timeline "
            "(`per_spk[target_idx]`), 不区分独占帧 vs 重叠帧. 这就是当前主线喂 ASR 的切片.",
            "- **collect_clean_audio(独占段)**: 在 enroll_infer 里用于算 speaker 声纹(避开重叠污染); "
              "本目录借用同一函数产出 02_spk{i}_exclusive 和 03_slice_exclusive, 作为对照."]
    with open(os.path.join(args.out_root, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(top))
    print(f"\n[done] {args.out_root} (总耗时 {(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
