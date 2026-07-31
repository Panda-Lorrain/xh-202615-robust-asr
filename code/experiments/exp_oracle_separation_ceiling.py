#!/usr/bin/env python3
"""Oracle separation 天花板 POC B1(2026-07-27).

【和 A 对称】: A 测"选 speaker 天花板"(在 diar 现有输出里选对的 spk), B1 测"切时间段天花板"
              (用 ref forced-align 出 target 真实说话时间段 ≈ oracle 完美分离)。

判别:
- B1 CER 大降 → 即便近似 oracle 也能救, "分离/切时间段"有真实空间, 升级 diar 或 source separation 有杠杆
- B1 CER 没降 → 不能断言"分离无空间"(forced alignment 在重叠区可能对齐到 louder 干扰人, 是近似下界),
                需 B2 SepFormer 真做源分离实测验证

链路(不动主线):
  MMS_FA(ref→IPA token forced align) → 每个 ref 字符的 [start,end] 帧 → union 成 oracle timeline
  → cut_target_timeline(audio, oracle_timeline) 拼接(同主线 diar 切法)
  → qwen_asr_backend 批转 → CERMetric 官方口径
  → 与 A 实验 argmax=1.216 / oracle 选 spk=0.850 对照

【局限(必标)】forced alignment 在重叠区会把 ref 对齐到 louder 说话人(可能干扰人), B1 oracle 不完美.
- 但 token 时间位置大体正确(model 在 target 说该音素时概率最高)
- 测的是"近似下界", 留 align_score 供报告判断对齐质量

数据源: code/runs/_oracle_speaker/meta.json (复用 A 的 40 条 fail 样本, 完全可比)
用法: code/.venv/Scripts/python.exe code/exp_oracle_separation_ceiling.py
产物: code/runs/_oracle_separation/{meta.json, summary.json, agg.json, slices/}
      docs/separation_ceiling_B1.md
"""
import os, sys, json, time, subprocess
import numpy as np
import torch, torchaudio
import soundfile as sf
import librosa
from pypinyin import lazy_pinyin
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from torchaudio.functional import forced_align
from text_utils import to_simplified, digit_postproc, brand_homophone_fix, cut_target_timeline
from eval_metrics import CERMetric

# ---- MMS_FA forced alignment 配置 ----
BUNDLE = torchaudio.pipelines.MMS_FA
FA_SR = BUNDLE.sample_rate  # 16000
LABELS = BUNDLE.get_labels()  # 29 chars ('-','a',...,'q','x','*')
LAB2IDX = {c: i for i, c in enumerate(LABELS)}
BLANK = len(LABELS) - 1  # '*' index 28 (MMS_FA 实际 blank; 验证 avg_prob 0.85 vs blank=0 的 0.36)


def pyize(text):
    """中文 → 每字拼音字母 token(只保留在 MMS labels 里的字符; ü/特殊自动丢弃).
    Returns list[list[char]], 每个字一组."""
    chunks = []
    for p in lazy_pinyin(text):
        chunks.append([c for c in p.lower() if c in LAB2IDX])
    return chunks


def align_ref(model, audio_np, sr, ref):
    """对齐 ref 到 audio, 返回 (oracle_timeline: [(start_s, end_s), ...], avg_prob, n_frames).
    oracle_timeline: 每个 ref 字符 -> [first_token_frame, last_token_frame] -> (start_s, end_s).
    avg_prob: 非 blank 帧的平均 exp(score), 用于判断对齐质量."""
    if sr != FA_SR:
        audio_np = librosa.resample(np.ascontiguousarray(audio_np, dtype=np.float32),
                                    orig_sr=sr, target_sr=FA_SR)
    wav = torch.from_numpy(np.ascontiguousarray(audio_np, dtype=np.float32)).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        emis, _ = model(wav)
    n_frames = emis.shape[1]

    chunks = pyize(ref)
    flat_tokens = []
    char_lens = []
    for ch_chars in chunks:
        flat_tokens.extend(ch_chars)
        char_lens.append(len(ch_chars))
    if not flat_tokens:
        return [], 0.0, n_frames
    target_ids = [LAB2IDX[c] for c in flat_tokens]
    # forced_align 在 GPU 上部分 input 触发 IndexError(torchaudio 2.5 已知问题), 落 CPU 稳定可靠
    emis_cpu = emis.detach().cpu()
    targets = torch.tensor([target_ids], dtype=torch.int)
    aligned, scores = forced_align(
        emis_cpu, targets,
        input_lengths=torch.tensor([n_frames]),
        target_lengths=torch.tensor([len(target_ids)]),
        blank=BLANK,
    )
    aligned = aligned[0].tolist()
    scores = scores[0].tolist()

    hop = (len(audio_np) / FA_SR) / n_frames  # 秒/帧

    # === timeline 提取: dilation 策略 ===
    # MMS FA 把每个 target token 对齐到 1-2 帧(周围都是 blank), 直接当 timeline 段过短会被
    # cut_target_timeline min_sec 退化整条. dilation: 每个非 blank 帧 ±D 帧(D=3, 覆盖 ~100ms 上下文
    # 即该音素真实发声区间), union 重叠/相邻段 → 稠密 timeline 覆盖 target 整个说话区间.
    D = 3
    hit = sorted(set(fi for fi in range(n_frames) if aligned[fi] != BLANK))
    if not hit:
        return [], 0.0, n_frames
    # 把每帧扩展 [fi-D, fi+D+1), union 相邻
    segs = []
    for fi in hit:
        s, e = max(0, fi - D), min(n_frames, fi + D + 1)
        if segs and s <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], e)
        else:
            segs.append([s, e])
    timeline = [(s * hop, e * hop) for s, e in segs]

    # 对齐置信度: 非 blank 帧的平均 exp(score), 反映 model 对该 token 序列的置信度
    # 用于报告判断对齐质量(>0.5 高质量, <0.3 重叠严重可能对齐到 louder 干扰人, 是已知局限)
    probs = [float(np.exp(scores[fi])) for fi in range(n_frames) if aligned[fi] != BLANK]
    avg_prob = float(np.mean(probs)) if probs else 0.0
    return timeline, avg_prob, n_frames


def submit_norm(text):
    return brand_homophone_fix(digit_postproc(to_simplified(text or "")))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(_HERE, "runs", "_oracle_separation")
    slices_dir = os.path.join(out_dir, "slices")
    os.makedirs(slices_dir, exist_ok=True)

    # 复用 A 的 fail 样本(完全可比)
    a_meta_path = os.path.join(_HERE, "runs", "_oracle_speaker", "meta.json")
    a_meta = json.load(open(a_meta_path, encoding="utf-8"))
    fail = [m for m in a_meta if m.get("group") == "fail"]
    # 同时取 A summary 拿 argmax_cer/oracle_cer
    a_sum_path = os.path.join(_HERE, "runs", "_oracle_speaker", "summary.json")
    a_sum = {s["uid"]: s for s in json.load(open(a_sum_path, encoding="utf-8"))}
    print(f"[load] A fail n={len(fail)}")

    # 加载 MMS_FA
    print(f"[load] MMS_FA model on {device}")
    model = BUNDLE.get_model().to(device).eval()

    # ---- 主循环: forced align + 切片 ----
    meta = []
    t0 = time.time()
    for ri, r in enumerate(fail):
        uid = r["uid"]
        ref = r["ref"]
        cmd_wav = os.path.join(_ROOT, "datasetA", "pos", f"{uid}.wav")
        if not os.path.isfile(cmd_wav):
            print(f"  {uid} SKIP missing audio")
            continue
        try:
            audio, asr = sf.read(cmd_wav)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            timeline, score, n_frames = align_ref(model, audio, asr, ref)
            if not timeline:
                # 极端情况: 对齐完全失败(拼音 token 全空), 退化整条音频但 score 保留真实(0)
                timeline = [(0.0, len(audio) / asr)]
            # 切片(同 enroll_infer 用的 cut_target_timeline, 含 union & min_sec 退化)
            seg = cut_target_timeline(audio, timeline, sr=asr)
            fn = os.path.join(slices_dir, f"{uid}_oracle.wav")
            sf.write(fn, seg.astype(np.float32), asr)
            total_dur = float(sum(e - s for s, e in timeline))
            as_rec = a_sum.get(uid, {})
            meta.append({
                "uid": uid, "ref": ref,
                "poc_qwen_cer": r["poc_qwen_cer"], "poc_sim": r["poc_sim"],
                "argmax_cer_A": as_rec.get("argmax_cer"),
                "oracle_speaker_cer_A": as_rec.get("oracle_cer"),
                "n_spk_A": r.get("n_spk"),
                "align_score": score,
                "timeline_n_segs": len(timeline),
                "timeline_total_dur": total_dur,
                "audio_dur": len(audio) / asr,
                "slice_file": os.path.basename(fn),
            })
            dt = time.time() - t0
            print(f"  [{ri+1}/{len(fail)}] {uid} score={score:.3f} segs={len(timeline)} "
                  f"tl_dur={total_dur:.2f}s/audio={len(audio)/asr:.2f}s ({dt:.0f}s)")
        except Exception as e:
            print(f"  [{ri+1}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    meta_path = os.path.join(out_dir, "meta.json")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok_n = sum(1 for m in meta if "error" not in m)
    print(f"\n[slice done] {ok_n}/{len(fail)} ok → {meta_path}")

    # ---- qwen 批量转写 ----
    py_qwen = os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe")
    if not os.path.isfile(py_qwen):
        py_qwen = os.environ.get("PY_QWEN", py_qwen)
    uid2text_path = os.path.join(slices_dir, "_uid2text.json")
    print(f"\n[qwen] 批量转写 {slices_dir} ...")
    try:
        subprocess.check_call(
            [py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
             "--slice-dir", slices_dir, "--out", uid2text_path,
             "--seed", "42", "--batch-size", "16"])
    except subprocess.CalledProcessError as e:
        print(f"[qwen] FAIL rc={e.returncode}; 退到逐条")
        subprocess.check_call(
            [py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
             "--slice-dir", slices_dir, "--out", uid2text_path,
             "--seed", "42", "--batch-size", "0"])

    uid2text = json.load(open(uid2text_path, encoding="utf-8"))
    print(f"[qwen] {len(uid2text)} slices transcribed")

    # ---- 算 CER ----
    summary = []
    for m in meta:
        if "error" in m:
            continue
        uid = m["uid"]
        ref = m["ref"]
        ref_n = submit_norm(ref)
        key = os.path.splitext(m["slice_file"])[0]  # cmd_N_oracle
        txt = uid2text.get(key, "")
        txt_n = submit_norm(txt)
        cm = CERMetric()
        cm.update([txt_n], [ref_n])
        cer = float(cm.compute()["cer"])
        summary.append({
            "uid": uid, "ref": ref,
            "oracle_separation_cer": cer,
            "oracle_separation_text": txt,
            "argmax_cer_A": m["argmax_cer_A"],
            "oracle_speaker_cer_A": m["oracle_speaker_cer_A"],
            "poc_qwen_cer": m["poc_qwen_cer"],
            "poc_sim": m["poc_sim"],
            "n_spk_A": m["n_spk_A"],
            "align_score": m["align_score"],
            "timeline_total_dur": m["timeline_total_dur"],
            "audio_dur": m["audio_dur"],
            "timeline_n_segs": m["timeline_n_segs"],
        })
    summary_path = os.path.join(out_dir, "summary.json")
    json.dump(summary, open(summary_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 聚合 ----
    n = len(summary)
    argmax_arr = np.array([s["argmax_cer_A"] for s in summary if s["argmax_cer_A"] is not None])
    orspk_arr = np.array([s["oracle_speaker_cer_A"] for s in summary if s["oracle_speaker_cer_A"] is not None])
    orsep_arr = np.array([s["oracle_separation_cer"] for s in summary])
    score_arr = np.array([s["align_score"] for s in summary])

    rescued_orsep = int(sum(1 for c in orsep_arr if c < 0.5))
    near_orsep = int(sum(1 for c in orsep_arr if c < 0.1))
    # 比 oracle_speaker 进一步救回的条数
    further = int(sum(1 for s in summary
                      if s["oracle_speaker_cer_A"] is not None
                      and s["oracle_separation_cer"] < s["oracle_speaker_cer_A"] - 0.1))

    agg = {
        "n": n,
        "argmax_mean_CER": float(argmax_arr.mean()) if len(argmax_arr) else None,
        "oracle_speaker_mean_CER": float(orspk_arr.mean()) if len(orspk_arr) else None,
        "oracle_separation_mean_CER": float(orsep_arr.mean()),
        "delta_argmax_orsep": float(argmax_arr.mean() - orsep_arr.mean()) if len(argmax_arr) else None,
        "delta_orspk_orsep": float(orspk_arr.mean() - orsep_arr.mean()) if len(orspk_arr) else None,
        "rescued_oracle_separation_n": rescued_orsep,
        "rescued_oracle_separation_pct": rescued_orsep / n if n else 0.0,
        "near_perfect_oracle_separation_n": near_orsep,
        "further_than_oracle_speaker_n": further,
        "align_score_mean": float(score_arr.mean()),
        "align_score_median": float(np.median(score_arr)),
        "align_score_min": float(score_arr.min()),
        "align_score_max": float(score_arr.max()),
    }
    json.dump(agg, open(os.path.join(out_dir, "agg.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("\n===== 聚合 (FAIL n={}) =====".format(n))
    print(f"argmax mean CER               = {agg['argmax_mean_CER']:.4f}  (基线: argmax 选 spk)")
    print(f"oracle speaker mean CER       = {agg['oracle_speaker_mean_CER']:.4f}  (A 实验: 完美选 spk)")
    print(f"oracle separation mean CER    = {agg['oracle_separation_mean_CER']:.4f}  (B1: forced alignment)")
    print(f"Δ(argmax - oracle_sep)        = {agg['delta_argmax_orsep']:.4f}")
    print(f"Δ(oracle_spk - oracle_sep)    = {agg['delta_orspk_orsep']:.4f}")
    print(f"rescued (CER<0.5)             = {rescued_orsep}/{n} ({rescued_orsep/n*100:.1f}%)")
    print(f"near perfect (CER<0.1)        = {near_orsep}/{n}")
    print(f"比 oracle_speaker 进一步救回    = {further}/{n}")
    print(f"align score mean={agg['align_score_mean']:.3f} "
          f"median={agg['align_score_median']:.3f} min={agg['align_score_min']:.3f} "
          f"max={agg['align_score_max']:.3f}")
    print(f"\n[done] → {summary_path}, {out_dir}/agg.json")


if __name__ == "__main__":
    main()
