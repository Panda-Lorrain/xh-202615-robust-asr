"""exp_fix_cut_boundary.py — cut_target_timeline 边界丢字"度"诊断 + POC 修复 + 验证。

【背景】用户听 cmd_2098 验收包发现: target 说"调到二十八度", 但切片
  `假如选spk1_当target.wav`(spk1 假设是 target 的切片) 只含"调到二十八", 丢了"度"字。

【已排除】md5 验证: `rec_spk{i}_full.wav` == `假如选spk{i}_当target.wav`(同 hash),
  cut_target_timeline 与 raw diar timeline 拼接产物完全一致 —— 不是 cut 内部截断 bug,
  是 diar 边界 / 缺 padding 致末段尾部短音被切。本脚本定位 + POC padding 修复。

【cut_target_timeline_v2 修复】每段尾部向后扩 pad_sec(默认 80ms), 夹到下一段 start 或音频末。
  - 最后一段无下邻 → 夹到 audio_length, 直接捕获末尾被 diar 划给"无主"或下一段首的尾音。
  - 中间段 → 夹到下一段 start, 不重叠不漏(单纯衔接更平滑)。
  - 首段额外向前 pad_sec 夹到 0(捕首字被截)。
  pad_sec=80ms 选型依据: 普通话字长 200-400ms, diar 标注分辨率~20ms, 80ms 覆盖末音节 1/3-1/2
  足以捕获"度"类短尾字, 不至于吞整字污染。

【3 phase】
  diagnose cmd_2098: diar timeline 打印 + 切片 v1/v2 存盘 + qwen 转写对比
  sample30:        随机 30 条抽样, argmax target 切片 v1 vs v2 qwen CER 对比(不退化验证)
  all:             顺序跑 diagnose + sample30

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_fix_cut_boundary.py --phase diagnose
  code/.venv/Scripts/python.exe code/exp_fix_cut_boundary.py --phase sample30
  code/.venv/Scripts/python.exe code/exp_fix_cut_boundary.py --phase all

产物:
  code/runs/_fix_cut_boundary/
    cmd_2098/diar_timeline.json + spk{0,1}_v1.wav + spk{0,1}_v2.wav + rec_last_500ms.wav
    sample30_v1/ sample30_v2/  (切片 wav)
    cmd_2098_qwen_compare.json + sample30_cer_compare.json
"""
import os, sys, json, time, argparse, subprocess, random

# ---- speechbrain lazy inspect patch(enroll_infer.py:24-29 复刻) ----
import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import numpy as np
import torch
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
_OUT_ROOT = os.path.join(_HERE, "runs", "_fix_cut_boundary")

from text_utils import to_simplified, digit_postproc, brand_homophone_fix, cut_target_timeline
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

# DiariZen / pyannote 路径
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"),
           os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)


# ============ cut_target_timeline_v2(POC 修复版) ============
def cut_target_timeline_v2(audio, per_spk_timeline, sr=16000, min_sec=0.3, pad_sec=0.08):
    """v2: 每段尾向后扩 pad_sec, 夹到下段 start 或 audio 末; 首段额外前扩 pad_sec。

    核心动机: diar 时间帧分辨率 ~20ms, 段尾短字(如"度"~200ms)常被划给"无主"或下一段首,
    原版直接按 (s*sr : e*sr) 切, 末音节被截。padding 80ms 可捕获被截尾字。
    中间段夹到下段 start 不重复(覆盖 gap), 末段夹到 audio 末(覆盖尾音)。

    audio: np.ndarray 全条; per_spk_timeline: list[(start,end)]; pad_sec: 单位秒。
    """
    segs = sorted((float(s), float(e)) for s, e in per_spk_timeline)
    n = len(audio)
    if not segs:
        out = np.asarray(audio)
    else:
        clips = []
        n_seg = len(segs)
        for i, (s, e) in enumerate(segs):
            # 末段: 夹到 audio 末; 中间段: 夹到下段 start(覆盖 gap 不重叠)
            if i + 1 < n_seg:
                e_pad = min(e + pad_sec, segs[i + 1][0])
            else:
                e_pad = min(e + pad_sec, n / sr)
            # 首段额外前扩 pad_sec(捕首字); 其他段不前扩(避免与上段尾重叠)
            if i == 0:
                s_pad = max(0.0, s - pad_sec)
            else:
                s_pad = s
            i_start = int(round(s_pad * sr))
            i_end = int(round(e_pad * sr))
            i_start = max(0, min(i_start, n))
            i_end = max(i_start, min(i_end, n))
            clips.append(audio[i_start:i_end])
        out = np.concatenate(clips) if clips else np.asarray(audio)
    if len(out) < sr * min_sec:
        out = np.asarray(audio)
    return out


# ============ diar + get_emb 复刻(enroll_infer.py:170-187) ============
def load_diar(device):
    from diarizen.pipelines.inference import DiariZenPipeline
    # pyannote Pipeline.to 要 torch.device 不收 str(enroll_infer 老用法可能因 transformers 版本而异, 这里强转)
    dev = torch.device(device) if isinstance(device, str) else device
    print(f"[load] DiariZen {resolve_model('DIAR')} on {dev}")
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(dev)
    return diar


def make_get_emb(diar, device):
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
    return get_emb


def collect_clean_audio(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    """enroll_infer.py:73 复刻, 抽 speaker i 的 non-overlap 独占段(避重叠污染声纹)。"""
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


def get_diarization_mask(per_speaker_samples, audio_length):
    """enroll_infer.py:52 复刻, list[(s,e)] → [N, T@50Hz] mask。"""
    mask = torch.zeros(len(per_speaker_samples), audio_length)
    for i, spk_samples in enumerate(per_speaker_samples):
        for start, end in spk_samples:
            mask[i, round(start * 50):round(end * 50)] = 1
    return mask


# ============ 通用辅助 ============
def _cer(text, ref):
    t = brand_homophone_fix(digit_postproc(to_simplified(text or "")))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref or "")))
    return float(cer_official(t, r))


# ============ Phase 1: cmd_2098 诊断 ============
def phase_diagnose(args):
    set_global_seed(args.seed)
    os.makedirs(_OUT_ROOT, exist_ok=True)
    out_dir = os.path.join(_OUT_ROOT, "cmd_2098")
    os.makedirs(out_dir, exist_ok=True)
    device = args.device

    pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    pair = next(p for p in pairs if "cmd_2098" in p["recognition"])
    enr_path, rec_path = pair["enrollment"], pair["recognition"]
    ref = pair["ref"]
    print(f"\n[diagnose] uid=cmd_2098  ref='{ref}'  kws='{pair['kws_txt']}'")

    # 加载模型
    diar = load_diar(device)
    get_emb = make_get_emb(diar, device)

    # 音频
    w_enr, _ = librosa.load(enr_path, sr=16000)
    w_rec, sr = librosa.load(rec_path, sr=16000)
    audio_sec = len(w_rec) / sr
    sf.write(os.path.join(out_dir, "recognition.wav"), w_rec.astype(np.float32), sr)
    sf.write(os.path.join(out_dir, "enrollment.wav"), w_enr.astype(np.float32), sr)
    print(f"  audio: rec={len(w_rec)/sr:.3f}s ({len(w_rec)} samples)  enr={len(w_enr)/sr:.3f}s")

    # diar
    diar_out = diar(rec_path)
    speakers = list(diar_out.labels())
    per_spk = [diar_out.label_timeline(s) for s in speakers]
    print(f"  diar speakers: {speakers}  n_seg/spk: {[len(tl) for tl in per_spk]}")

    # 各 spk argmax sim(复刻 enroll_infer 流程)
    enr_emb = get_emb(w_enr)
    audio_len = len(w_rec) // 320
    diar_mask = get_diarization_mask(per_spk, audio_len)
    spk_embs, spk_sims = [], []
    for i in range(len(speakers)):
        excl = collect_clean_audio(w_rec, diar_mask, i, sr)
        if excl is None or len(excl) < sr * 0.3:
            segs = [w_rec[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
            seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
            fallback = True
        else:
            seg = excl
            fallback = False
        min_len = sr
        if len(seg) < min_len:
            seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
        emb = get_emb(seg)
        spk_embs.append(emb)
        spk_sims.append(float(torch.dot(enr_emb, emb)))
    target_idx = int(np.argmax(spk_sims))
    print(f"  sims: {[round(s,4) for s in spk_sims]}  argmax target=spk{target_idx}")

    # ---- timeline 详情: 每段 start/end/dur, 末段距 audio 末的距离 ----
    tl_info = {}
    for i, spk_l in enumerate(speakers):
        segs = sorted((float(s), float(e)) for s, e in per_spk[i])
        info = []
        for j, (s, e) in enumerate(segs):
            gap_to_next = (segs[j + 1][0] - e) if j + 1 < len(segs) else None
            info.append({"idx": j, "start": round(s, 4), "end": round(e, 4),
                         "dur": round(e - s, 4),
                         "gap_to_next": round(gap_to_next, 4) if gap_to_next is not None else None})
        last_end = segs[-1][1] if segs else 0.0
        tl_info[f"spk{i}"] = {
            "label": spk_l, "n_seg": len(segs),
            "segs": info,
            "last_end": round(last_end, 4),
            "audio_end": round(audio_sec, 4),
            "tail_gap_to_audio_end": round(audio_sec - last_end, 4),
            "excluded_at_tail_sec": round(audio_sec - last_end, 4),
        }
        print(f"  [spk{i} '{spk_l}'] n_seg={len(segs)} last_end={last_end:.3f}s "
              f"audio_end={audio_sec:.3f}s tail_gap={audio_sec-last_end:.3f}s")
        for j, (s, e) in enumerate(segs):
            tag = ""
            if j == len(segs) - 1:
                tag = " <- 末段(本 spk 最后一段)"
            print(f"      seg{j}: [{s:.3f}, {e:.3f}] dur={e-s:.3f}s{tag}")

    # ---- 诊断小切片: 末尾音频 + 各 spk 末段 + padding 扩展(听"度"位置) ----
    # rec 末尾多档(听"度"在 audio 末哪一段)
    for tail_ms in (300, 500, 800):
        tail_start = max(0, len(w_rec) - int(tail_ms / 1000 * sr))
        sf.write(os.path.join(out_dir, f"rec_last_{tail_ms}ms.wav"),
                 w_rec[tail_start:].astype(np.float32), sr)
    # 关键: spk1(用户实际 target) 末段后不同长度 pad(80/200/400/560ms), 看 "度" 在哪段补回
    # 560ms = spk1 末段到 audio 末的距离(0.558s) —— 全补回
    pad_sweep = [args.pad_sec, 0.20, 0.40, 0.56]
    # 各 spk: v1 + v2(各 pad_sec) + 末段独立存
    for i in range(len(speakers)):
        v1 = cut_target_timeline(w_rec, per_spk[i], sr=sr)
        sf.write(os.path.join(out_dir, f"spk{i}_v1.wav"), v1.astype(np.float32), sr)
        for psec in pad_sweep:
            v2 = cut_target_timeline_v2(w_rec, per_spk[i], sr=sr, pad_sec=psec)
            tag = f"{int(psec*1000)}ms"
            sf.write(os.path.join(out_dir, f"spk{i}_v2_pad{tag}.wav"), v2.astype(np.float32), sr)
            if psec == args.pad_sec:
                delta_ms = (len(v2) - len(v1)) / sr * 1000
                print(f"  [spk{i}] v1={len(v1)/sr:.3f}s  v2(pad={psec}s)={len(v2)/sr:.3f}s  Δ=+{delta_ms:.0f}ms")
        # 末段 + 末段+pad(听末段是否含"度", pad 是否补回)
        if per_spk[i]:
            last_s, last_e = sorted((float(s), float(e)) for s, e in per_spk[i])[-1]
            last_raw = w_rec[int(last_s*sr):int(last_e*sr)]
            last_pad_end = min(last_e + args.pad_sec, audio_sec)
            last_pad = w_rec[int(last_s*sr):int(last_pad_end*sr)]
            sf.write(os.path.join(out_dir, f"spk{i}_last_seg_raw.wav"),
                     last_raw.astype(np.float32), sr)
            sf.write(os.path.join(out_dir, f"spk{i}_last_seg_+pad.wav"),
                     last_pad.astype(np.float32), sr)

    # timeline 存盘
    diag_json = {
        "uid": "cmd_2098", "ref": ref, "audio_sec": round(audio_sec, 4),
        "speakers": speakers, "argmax_target_idx": target_idx,
        "spk_sims": [round(s, 4) for s in spk_sims],
        "timelines": tl_info,
        "pad_sec": args.pad_sec,
        "hypothesis": (
            "若 spk_i 末段 last_end < audio_end 且 tail_gap ~ '度' 长度(~0.2s), "
            "说明 diar 把末段尾部'度'划给了'无主'或下一段首; cut_v2 padding 可捕获。"
        ),
    }
    json.dump(diag_json, open(os.path.join(out_dir, "diar_timeline.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- qwen 转写对比(各 spk v1 vs v2 + pad_sweep + tail): 用 venv_qwen 跑 ----
    if not args.skip_qwen:
        print("\n  [qwen] 准备 v1/v2/pad_sweep/tail 切片目录转写...")
        slice_cmp_dir = os.path.join(_OUT_ROOT, "cmd_2098_qwen_sweep")
        os.makedirs(slice_cmp_dir, exist_ok=True)
        # 重点: spk1(用户 hearing 实际 target)+ spk0(argmax) 在不同 pad 下转写
        sweep_inputs = {}
        for i in range(len(speakers)):
            sweep_inputs[f"spk{i}_v1"] = cut_target_timeline(w_rec, per_spk[i], sr=sr)
            for psec in [args.pad_sec, 0.20, 0.40, 0.56]:
                tag = f"{int(psec*1000)}ms"
                sweep_inputs[f"spk{i}_v2_pad{tag}"] = \
                    cut_target_timeline_v2(w_rec, per_spk[i], sr=sr, pad_sec=psec)
        # rec tail 各档(定位"度"在 audio 哪段)
        for tail_ms in (300, 500, 800):
            tail_start = max(0, len(w_rec) - int(tail_ms / 1000 * sr))
            sweep_inputs[f"rec_tail_{tail_ms}ms"] = w_rec[tail_start:]
        # 存盘 + qwen 跑
        for k, arr in sweep_inputs.items():
            sf.write(os.path.join(slice_cmp_dir, f"cmd_2098_{k}.wav"),
                     arr.astype(np.float32), sr)
        sweep_uid2 = _run_qwen_backend(slice_cmp_dir, args)
        cmp = {"ref": ref, "argmax_target_idx": target_idx, "pad_sec_default": args.pad_sec,
               "results": [], "tail_locating": {}}
        # 主对比: 每个 spk 各 pad 文本 + CER
        for i in range(len(speakers)):
            row = {"spk": i, "label": speakers[i], "pads": []}
            for psec, key in [(0.0, f"spk{i}_v1")] + \
                    [(p, f"spk{i}_v2_pad{int(p*1000)}ms") for p in [args.pad_sec, 0.20, 0.40, 0.56]]:
                t = sweep_uid2.get(f"cmd_2098_{key}", "")
                row["pads"].append({
                    "pad_sec": psec, "key": key, "text": t,
                    "cer_vs_ref": round(_cer(t, ref), 4),
                    "has_du": "度" in t,
                })
            cmp["results"].append(row)
            # 打印 spk 各 pad 转写
            print(f"    [spk{i} '{speakers[i]}'] ref='{ref}'")
            for p in row["pads"]:
                mark = "  ★含'度'" if p["has_du"] else ""
                print(f"      pad={p['pad_sec']:.2f}s  CER={p['cer_vs_ref']:.3f}  "
                      f"text='{p['text']}'{mark}")
        # tail 定位
        for tail_ms in (300, 500, 800):
            t = sweep_uid2.get(f"cmd_2098_rec_tail_{tail_ms}ms", "")
            cmp["tail_locating"][f"rec_last_{tail_ms}ms"] = {"text": t, "has_du": "度" in t}
            print(f"    [rec tail {tail_ms}ms] '{t}'  含'度'={'度' in t}")
        json.dump(cmp, open(os.path.join(out_dir, "cmd_2098_qwen_compare.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)

    print(f"\n[diagnose] 产物: {out_dir}")


# ============ Phase 2: 30 条抽样 v1 vs v2 CER 对比(不退化验证) ============
def phase_sample30(args):
    set_global_seed(args.seed)
    os.makedirs(_OUT_ROOT, exist_ok=True)
    device = args.device

    pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    qfull = json.load(open(os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    qfull_map = {r["uid"]: r for r in qfull["rows"]}

    # 随机 30(覆盖三桶: 死区/主战场/接近解决 各 10 条, 让分布更鲁棒)
    rng = random.Random(args.seed)
    by_bucket = {"<0.2 死区": [], "[0.2, 0.4) 主战场": [], ">=0.4 接近解决": []}
    for r in qfull["rows"]:
        b = r.get("bucket", "")
        for k in by_bucket:
            if k.startswith(b.rstrip()) or b.startswith(k.split()[0]):
                by_bucket[k].append(r["uid"])
                break
    sampled = []
    for k in by_bucket:
        pool = []
        for u in by_bucket[k]:
            try:
                cid = int(u.split("_")[1])
            except Exception:
                continue
            if any(p["id"] == cid for p in pairs):
                pool.append(u)
        rng.shuffle(pool)
        sampled.extend(pool[:10])
    sampled = sampled[:30]
    print(f"[sample30] 抽样 {len(sampled)} 条 (3 桶×10): {sampled[:5]}...")

    # 加载模型
    diar = load_diar(device)
    get_emb = make_get_emb(diar, device)

    pair_by_id = {p["id"]: p for p in pairs}
    # 多档 pad_sweep(每档一个切片目录): 0(v1) + 0.03/0.05/0.08/0.12/0.20s
    pad_list = [0.0, 0.03, 0.05, 0.08, 0.12, 0.20]
    slice_dirs = {f"pad{int(p*1000):03d}ms": os.path.join(_OUT_ROOT, f"sample30_pad{int(p*1000):03d}ms")
                  for p in pad_list}
    for d in slice_dirs.values():
        os.makedirs(d, exist_ok=True)
    meta = []
    # 缓存 diar 结果, 不每档重跑
    cache = {}

    for n, uid in enumerate(sampled):
        cid = int(uid.split("_")[1])
        p = pair_by_id[cid]
        enr_path, rec_path, ref = p["enrollment"], p["recognition"], p["ref"]
        try:
            w_rec, sr = librosa.load(rec_path, sr=16000)
            w_enr, _ = librosa.load(enr_path, sr=16000)
            diar_out = diar(rec_path)
            speakers = list(diar_out.labels())
            per_spk = [diar_out.label_timeline(s) for s in speakers]
            enr_emb = get_emb(w_enr)
            audio_len = len(w_rec) // 320
            diar_mask = get_diarization_mask(per_spk, audio_len)
            spk_sims = []
            for i in range(len(speakers)):
                excl = collect_clean_audio(w_rec, diar_mask, i, sr)
                if excl is None or len(excl) < sr * 0.3:
                    segs = [w_rec[int(s*sr):int(e*sr)] for s, e in per_spk[i]]
                    seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
                else:
                    seg = excl
                min_len = sr
                if len(seg) < min_len:
                    seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
                emb = get_emb(seg)
                spk_sims.append(float(torch.dot(enr_emb, emb)))
            target_idx = int(np.argmax(spk_sims))
            cache[uid] = {"w_rec": w_rec, "sr": sr, "per_spk": per_spk, "target_idx": target_idx}
            # 各 pad 切片存盘
            for psec in pad_list:
                key = f"pad{int(psec*1000):03d}ms"
                if psec == 0.0:
                    sl = cut_target_timeline(w_rec, per_spk[target_idx], sr=sr)
                else:
                    sl = cut_target_timeline_v2(w_rec, per_spk[target_idx], sr=sr, pad_sec=psec)
                sf.write(os.path.join(slice_dirs[key], f"{uid}.wav"), sl.astype(np.float32), sr)
            meta.append({
                "uid": uid, "ref": ref, "argmax_target_idx": target_idx,
                "n_spk": len(speakers), "max_sim": round(max(spk_sims), 4),
                "bucket": qfull_map.get(uid, {}).get("bucket", "?"),
                "orig_qwen_cer": qfull_map.get(uid, {}).get("qwen_cer"),
            })
            if (n + 1) % 5 == 0 or n == 0:
                print(f"  [{n+1}/{len(sampled)}] {uid} spk{target_idx}")
        except Exception as e:
            print(f"  [skip] {uid}: {type(e).__name__}: {str(e)[:80]}")
            meta.append({"uid": uid, "ref": p["ref"], "error": str(e)[:120]})

    json.dump(meta, open(os.path.join(_OUT_ROOT, "sample30_meta.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- 各 pad 档 qwen 转写 + CER ----
    if not args.skip_qwen:
        all_summary = {"n_meta": len(meta), "pad_sweep": []}
        ref_map = {m["uid"]: m["ref"] for m in meta if "error" not in m}
        valid_uids = list(ref_map.keys())
        per_pad_mean = {}
        for key, d in slice_dirs.items():
            print(f"\n  [qwen] {key} 转写...")
            uid2 = _run_qwen_backend(d, args)
            cers = []
            rows = []
            for uid in valid_uids:
                t = uid2.get(uid, "")
                c = _cer(t, ref_map[uid])
                cers.append(c)
                rows.append({"uid": uid, "text": t, "cer": round(c, 4)})
            mean_c = sum(cers) / max(len(cers), 1)
            per_pad_mean[key] = mean_c
            all_summary["pad_sweep"].append({
                "pad_key": key, "mean_cer": round(mean_c, 4),
                "n": len(cers), "rows": rows,
            })
            print(f"    {key}: mean CER = {mean_c:.4f}  (n={len(cers)})")
        # v1 (pad000ms) vs 各 pad 档对比
        v1_mean = per_pad_mean.get("pad000ms", 0.0)
        all_summary["v1_mean_cer"] = round(v1_mean, 4)
        all_summary["delta_vs_v1"] = {k: round(v - v1_mean, 4) for k, v in per_pad_mean.items()}
        all_summary["verdict"] = {
            k: ("PASS" if v - v1_mean <= 0.005 else "FAIL") for k, v in per_pad_mean.items()
        }
        json.dump(all_summary, open(os.path.join(_OUT_ROOT, "sample30_cer_compare.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        print("\n[sample30 pad_sweep 总结]")
        print(f"  v1(pad=0): mean CER = {v1_mean:.4f}")
        for k, v in per_pad_mean.items():
            if k == "pad000ms": continue
            d = v - v1_mean
            tag = "PASS" if d <= 0.005 else "FAIL"
            print(f"  {k}: mean CER = {v:.4f}  Δ={d:+.4f}  [{tag}]")


# ============ qwen backend 子进程封装 ============
def _run_qwen_backend(slice_dir, args):
    """子进程调 venv_qwen python qwen_asr_backend.py, 返回 uid→text。"""
    out_json = os.path.join(slice_dir, "_uid2text.json")
    py = args.py_qwen
    if not os.path.exists(py):
        print(f"  [warn] venv_qwen python 不存在 {py}, 跳过 qwen 转写")
        return {}
    cmd = [py, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json,
           "--batch-size", str(args.qwen_batch_size),
           "--seed", str(args.seed)]
    print(f"  [subprocess] {' '.join(cmd[:4])} ...")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=_HERE)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"  [error] qwen backend rc={r.returncode}  elapsed={elapsed:.0f}s")
        print("  stderr tail:", r.stderr[-500:] if r.stderr else "(empty)")
        return {}
    if not os.path.exists(out_json):
        print(f"  [warn] {out_json} 未生成")
        return {}
    print(f"  [ok] qwen backend done {elapsed:.0f}s")
    return json.load(open(out_json, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="cut_target_timeline 边界 padding 修复 POC")
    ap.add_argument("--phase", choices=["diagnose", "sample30", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad-sec", type=float, default=0.08,
                    help="段尾向后 padding 秒数(默认 80ms)")
    ap.add_argument("--py-qwen", default=os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))
    ap.add_argument("--qwen-batch-size", type=int, default=16)
    ap.add_argument("--skip-qwen", action="store_true", help="跳过 qwen 转写(只诊断 timeline + 切片存盘)")
    args = ap.parse_args()

    print(f"[exp_fix_cut_boundary] phase={args.phase} pad_sec={args.pad_sec}s seed={args.seed}")
    if args.phase in ("diagnose", "all"):
        phase_diagnose(args)
    if args.phase in ("sample30", "all"):
        phase_sample30(args)
    print("\n[done] 产物根目录:", _OUT_ROOT)


if __name__ == "__main__":
    main()
