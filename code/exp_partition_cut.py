"""exp_partition_cut.py — 分区切 timeline POC (2026-07-27)。

【背景】链路 bug 已定位: enroll_infer 的 cut_target_timeline 切的是 target 的 full timeline
(含重叠帧)。重叠区物理上是两人混合, 切进去 → ASR 收混合 mel → 转错(如 cmd_2637 spk0 切
片含两人)。但 diar 是对的(spk0/spk1 内容可分)。

【POC 设计】对 target 的 timeline 分两类区域区别处理:
  - exclusive 区(target 说话 + 其他都不说话)→ 纯 target, 切原始 audio
  - overlap 区(target + 其他同时说话)→ 混合, 切 SepFormer 分离的 target 路
  按时间顺序拼接 → qwen 转 → 算 CER

【核心创新: 用 exclusive 段纯 target 声纹选 SepFormer 路】
B2 证明 enroll emb 选路只有 25% 对(enroll 污染 + SI-SDR 破坏声纹)。本 POC 改用 exclusive
段(从原始 audio 抽的纯 target 未经 SepFormer)作锚匹配 SepFormer 两路。理论上绕开两类污染。

【3 切法 × 3 选路 = 9 组合对照】(full/excl 不依赖 SepFormer 选路, no-op; 重点 partition × 3)
  - full: cut_target_timeline 全 timeline(当前主线 baseline)
  - exclusive-only: 只切 exclusive 段(对照, ASR 退化预期)
  - partition: exclusive 原始 + overlap SepFormer target 路 × 3 选路(enroll/exclusive/oracle)

【样本】8 条 B2 fail 组(覆盖 argmax 对/错 × sep 对/错):
  cmd_2637(argmax 错, 双重错对照) / 2188/2302/2347/2503/2630/2766(argmax 对 sep 错, 主战场)
  / 2890(argmax 对 sep 对, 不毁已对基线)

【复用】B2 已分离 sepformer 两路 wav(runs/_sepformer_b2/slices/), 不重新分离, 省 15min。
用法:
  code/.venv/Scripts/python.exe code/exp_partition_cut.py
产物: code/runs/_partition_poc/{slices/, _uid2text.json, summary.json, diag.json}
"""
import os, sys, json, time, argparse, subprocess, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain 1.1 LazyModule inspect-guard patch (Win 兼容, 复刻 enroll_infer) ----
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

# enroll_infer 的副作用(DiariZen/pyannote sys.path 注入) + cut_target_timeline 复用
from enroll_infer import get_diarization_mask, collect_clean_audio  # noqa: F401
from text_utils import to_simplified, cut_target_timeline, digit_postproc, brand_homophone_fix
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

import numpy as np
import torch
import librosa
import soundfile as sf

# DiCoW-inference / DiariZen sys.path (enroll_infer 已注入, 这里冗余兜底)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))

SAMPLE_UIDS = ['cmd_2637', 'cmd_2188', 'cmd_2302', 'cmd_2347',
               'cmd_2503', 'cmd_2630', 'cmd_2766', 'cmd_2890']


def load_diar(diar_model, device):
    from diarizen.pipelines.inference import DiariZenPipeline
    return DiariZenPipeline.from_pretrained(diar_model).to(device)


def get_emb_factory(diar, device):
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


def cer_normalized(text, ref):
    t = brand_homophone_fix(digit_postproc(to_simplified(text or "")))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref or "")))
    return float(cer_official(t, r))


def align_sep_to_audio(sep_wav, audio_len):
    """SepFormer 输出对齐到原 audio 长度(截断/pad zero)。"""
    sep_wav = np.ascontiguousarray(sep_wav.astype(np.float32))
    if len(sep_wav) >= audio_len:
        return sep_wav[:audio_len]
    return np.concatenate([sep_wav, np.zeros(audio_len - len(sep_wav), dtype=np.float32)])


def split_target_intervals(per_spk, target_idx):
    """把 target timeline 切成 exclusive/overlap 子区间(按时间顺序)。
    返回 [(type, start, end), ...] type ∈ {'excl','ov'}; 单位秒。
    """
    if target_idx >= len(per_spk):
        return []
    target_intervals = sorted((float(s), float(e)) for s, e in per_spk[target_idx])
    others = []
    for i, spk_samples in enumerate(per_spk):
        if i == target_idx:
            continue
        for s, e in spk_samples:
            others.append((float(s), float(e)))
    others.sort()

    out = []
    for ts, te in target_intervals:
        # 收集其他 speaker 与 [ts,te] 的交集
        overlaps = []
        for os_, oe in others:
            if oe <= ts or os_ >= te:
                continue
            overlaps.append((max(os_, ts), min(oe, te)))
        overlaps.sort()
        cur = ts
        for os_, oe in overlaps:
            if os_ > cur:
                out.append(("excl", cur, os_))
            out.append(("ov", os_, oe))
            cur = oe
        if cur < te:
            out.append(("excl", cur, te))
    return out


def crossfade_concat(pieces, sr=16000, fade_ms=10):
    """pieces: list[(audio_np, start_sec)] 按时间顺序; 边界线性 crossfade 拼接。
    无 fade 也能用(cut_target_timeline 就是直接 concat), 这里加 fade 避免边界咔哒声。
    """
    if not pieces:
        return np.zeros(sr, dtype=np.float32)
    fade_n = int(sr * fade_ms / 1000)
    out = np.array([], dtype=np.float32)
    for i, (a, _) in enumerate(pieces):
        a = np.ascontiguousarray(a.astype(np.float32))
        if i == 0:
            out = a
            continue
        # 在边界做 overlap-add: 前 tail 与 后 head 各 fade_n 样本
        if fade_n > 0 and len(out) >= fade_n and len(a) >= fade_n:
            tail = out[-fade_n:].copy()
            head = a[:fade_n].copy()
            ramp = np.linspace(1, 0, fade_n, dtype=np.float32)
            out[-fade_n:] = tail * ramp + head * (1 - ramp)
            out = np.concatenate([out, a[fade_n:]])
        else:
            out = np.concatenate([out, a])
    return out


def build_slices(audio, sep_a, sep_b, per_spk, target_idx, sr=16000, min_seg_sec=0.1):
    """切 5 类切片音频: full / exclusive-only / partition_enroll / partition_excl / partition_oracle。
    sep_a, sep_b: SepFormer 两路对齐到原 audio 长度的 np 数组。
    返回 dict[cut_method] = audio_np。
    """
    audio_len = len(audio)
    pieces = split_target_intervals(per_spk, target_idx)
    min_n = int(min_seg_sec * sr)

    # full: 全 timeline 含 overlap (cut_target_timeline 一致, 原 audio)
    full_pieces = [(audio[int(s*sr):int(e*sr)], s) for _, s, e in pieces if (e-s)*sr >= min_n]
    full_audio = crossfade_concat(full_pieces, sr) if full_pieces else np.array(audio)

    # exclusive-only: 只 target 独占段(原 audio)
    excl_pieces = [(audio[int(s*sr):int(e*sr)], s) for t, s, e in pieces if t == "excl" and (e-s)*sr >= min_n]
    excl_audio = crossfade_concat(excl_pieces, sr) if excl_pieces else np.zeros(int(0.3*sr), dtype=np.float32)

    # partition: excl 用原 audio, ov 用 sep_target_audio; 三种 sep 选路不同 sep_target
    def _partition(sep_target):
        pp = []
        for t, s, e in pieces:
            seg_audio = audio if t == "excl" else sep_target
            seg = seg_audio[int(s*sr):int(e*sr)]
            if len(seg) >= min_n or t == "ov":  # overlap 段即使短也保留(否则切掉等于丢目标内容)
                pp.append((seg, s))
            # 太短的 excl 子段跳过(噪声)
        return crossfade_concat(pp, sr) if pp else np.array(audio)

    return {
        "full": full_audio,
        "excl": excl_audio,
        "part_enroll": _partition(sep_a),  # 路由 sep_a/enroll 选
        "part_excl":   _partition(sep_a),  # 占位, 路由由 main 决定后传入; 这里临时同 part_enroll, 实际下面覆盖
        "part_oracle": _partition(sep_a),  # 同上
    }


def run_qwen_batch(slice_dir, out_json, batch_size=16):
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--seed", "42",
           "--batch-size", str(batch_size)]
    print(f"[qwen] subprocess 转写 {slice_dir}")
    subprocess.run(cmd, check=True)
    return json.load(open(out_json, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="分区切 timeline POC")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--b2-summary", default=os.path.join(_HERE, "runs/_sepformer_b2/summary.json"))
    ap.add_argument("--b2-slices", default=os.path.join(_HERE, "runs/_sepformer_b2/slices"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs/_partition_poc"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    slice_dir = os.path.join(args.out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(slice_dir, "*.wav")):
        os.remove(f)

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    b2 = json.load(open(args.b2_summary, encoding="utf-8"))
    b2_by_uid = {r["uid"]: r for r in b2["results"]}

    samples = [u for u in SAMPLE_UIDS if u in uid2pair and u in b2_by_uid]
    missing = [u for u in SAMPLE_UIDS if u not in samples]
    if missing:
        print(f"[WARN] missing pairs/b2: {missing}")
    print(f"[data] {len(samples)} samples: {samples}")

    print(f"[load] diar {args.diarization_model}")
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] ok, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    slice_meta = {}    # uid -> {slice_uid_by_key}
    diag = []          # 每条诊断
    sep_streams = {}   # uid -> {"src_idx_embs": [emb_per_src], "src_idx_to_b2_uid": ...}
    t0 = time.time()

    for n, uid in enumerate(samples):
        pair = uid2pair[uid]
        enr, rec, ref = pair["enrollment"], pair["recognition"], pair["ref"]
        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)
            audio_len = len(audio)

            diar_out = diar(rec)
            speakers = list(diar_out.labels())
            per_spk = [diar_out.label_timeline(s) for s in speakers]

            # argmax target_idx (复刻 enroll_infer)
            audio_length = int(len(audio) / sr * 50) + 1
            diar_mask = get_diarization_mask(per_spk, audio_length)
            spk_embs = []
            for i in range(len(speakers)):
                seg = collect_clean_audio(audio, diar_mask, i, sr)
                if seg is None or len(seg) < sr * 0.3:
                    segs = [audio[int(s*sr):int(e*sr)] for s, e in per_spk[i]]
                    seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                spk_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
            target_idx = int(torch.argmax(sims))

            # ---- 切 target 子区间(exclusive/overlap)----
            pieces = split_target_intervals(per_spk, target_idx)
            excl_dur = sum(e - s for t, s, e in pieces if t == "excl")
            ov_dur = sum(e - s for t, s, e in pieces if t == "ov")
            target_total = sum(e - s for _, s, e in pieces)

            # ---- 抽 exclusive_target_emb (创新核心) ----
            excl_target_audio = collect_clean_audio(audio, diar_mask, target_idx, sr)
            excl_short = excl_target_audio is None or len(excl_target_audio) < sr * 0.3
            if excl_short:
                excl_target_audio = excl_target_audio if excl_target_audio is not None else np.zeros(sr, dtype=np.float32)
                excl_target_emb = enroll_emb  # fallback enroll
                excl_emb_src = "fallback_enroll"
            else:
                # 至少 0.3s, 可抽 emb; 若 < 1s tile 到 1s(wespeaker 期望 ≥1s 稳)
                ea = excl_target_audio
                if len(ea) < sr:
                    ea = np.tile(ea, sr // len(ea) + 1)[:sr]
                excl_target_emb = get_emb(ea)
                excl_emb_src = f"exclusive({len(excl_target_audio)/sr:.2f}s)"

            # ---- load B2 已分离两路 sep wav + 抽 stream_embs ----
            b2_r = b2_by_uid[uid]
            per_src = b2_r["per_src"]  # [{src_idx, slice_uid, ...}]
            assert len(per_src) == 2, f"B2 expected 2 src, got {len(per_src)}"
            src_embs = []
            src_wavs_aligned = []
            for src in per_src:
                wav_path = os.path.join(args.b2_slices, src["slice_uid"] + ".wav")
                wav, _ = librosa.load(wav_path, sr=16000)
                wav_al = align_sep_to_audio(wav, audio_len)
                src_wavs_aligned.append(wav_al)
                # 抽 emb (≥1s tile)
                ew = wav_al
                if len(ew) < sr:
                    ew = np.tile(ew, sr // len(ew) + 1)[:sr]
                src_embs.append(get_emb(ew))

            # ---- 3 种选路: enroll / exclusive / oracle ----
            sims_enroll = [float(torch.dot(enroll_emb, e)) for e in src_embs]
            sims_excl   = [float(torch.dot(excl_target_emb, e)) for e in src_embs]
            enroll_pick = int(np.argmax(sims_enroll))
            excl_pick   = int(np.argmax(sims_excl))
            oracle_pick = int(np.argmin([cer_normalized(src["text"], ref) for src in per_src]))
            # 注意: per_src 顺序对应 src_idx 0,1; B2 slice_uids 列表已按 src_idx 排序

            sep_a_wav = src_wavs_aligned[0]
            sep_b_wav = src_wavs_aligned[1]

            # ---- 切 5 类切片 ----
            # full / excl 不依赖 sep 选路(no-op); partition 三种选路
            def _cut_partition(sep_target_wav):
                min_n = int(0.1 * sr)
                pp = []
                for t, s, e in pieces:
                    seg_audio = audio if t == "excl" else sep_target_wav
                    seg = seg_audio[int(s*sr):int(e*sr)]
                    if t == "ov" or len(seg) >= min_n:
                        pp.append((seg, s))
                return crossfade_concat(pp, sr) if pp else np.array(audio)

            full_audio = cut_target_timeline(audio, per_spk[target_idx], sr=sr)
            excl_only_audio = excl_target_audio if not excl_short else np.array(audio)

            partition_enroll_audio = _cut_partition(src_wavs_aligned[enroll_pick])
            partition_excl_audio   = _cut_partition(src_wavs_aligned[excl_pick])
            partition_oracle_audio = _cut_partition(src_wavs_aligned[oracle_pick])

            # ---- 存盘 ----
            slice_uids = {}
            for name, wav in [
                ("full", full_audio),
                ("excl", excl_only_audio),
                ("part_enroll", partition_enroll_audio),
                ("part_excl", partition_excl_audio),
                ("part_oracle", partition_oracle_audio),
            ]:
                suid = f"{uid}__{name}"
                sf.write(os.path.join(slice_dir, suid + ".wav"),
                         np.ascontiguousarray(wav.astype(np.float32)), 16000)
                slice_uids[name] = suid

            slice_meta[uid] = slice_uids
            diag.append({
                "uid": uid, "ref": ref,
                "speakers": speakers, "target_idx": target_idx,
                "n_spk": len(speakers),
                "target_total_sec": round(target_total, 2),
                "excl_dur_sec": round(excl_dur, 2),
                "overlap_dur_sec": round(ov_dur, 2),
                "excl_ratio": round(excl_dur / max(target_total, 0.01), 3),
                "overlap_ratio": round(ov_dur / max(target_total, 0.01), 3),
                "excl_emb_src": excl_emb_src,
                "sims_enroll_vs_sep": [round(s, 4) for s in sims_enroll],
                "sims_excl_vs_sep":   [round(s, 4) for s in sims_excl],
                "enroll_pick": enroll_pick, "excl_pick": excl_pick, "oracle_pick": oracle_pick,
                "enroll_picks_oracle": enroll_pick == oracle_pick,
                "excl_picks_oracle":   excl_pick == oracle_pick,
                "argmax_cer_A": b2_r.get("argmax_cer_A"),
                "oracle_speaker_cer_A": b2_r.get("oracle_speaker_cer_A"),
                "sep_sim_cer": b2_r.get("sep_cer"),
                "sep_oracle_cer": b2_r.get("oracle_cer"),
                "B2_target_idx_from_enroll_on_diar": b2_r.get("target_idx"),
                "B2_argmax_target_idx_correct": b2_r.get("target_idx") == b2_r.get("oracle_src_idx"),
            })
            elapsed = time.time() - t0
            print(f"  [{n+1}/{len(samples)}] {uid} spk={speakers} tgt={target_idx} "
                  f"excl={excl_dur:.1f}s ov={ov_dur:.1f}s ({elapsed:.0f}s) "
                  f"picks enroll/excl/oracle = {enroll_pick}/{excl_pick}/{oracle_pick} "
                  f"{'[EXCL救]' if (enroll_pick != oracle_pick and excl_pick == oracle_pick) else ''}")
        except Exception as e:
            import traceback
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            diag.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:200]}"})

    # ---- qwen 批转写 ----
    print(f"\n[qwen] 转写 {sum(len(v) for v in slice_meta.values())} 切片 (5/样本)...")
    qwen_out = os.path.join(args.out_dir, "_uid2text.json")
    uid2text = run_qwen_batch(slice_dir, qwen_out, args.batch_size)

    # ---- 算 CER ----
    results = []
    for d in diag:
        if "error" in d:
            results.append(d)
            continue
        uid = d["uid"]
        ref = d["ref"]
        suids = slice_meta[uid]
        out = {**d}
        for name, suid in suids.items():
            txt = uid2text.get(suid, "")
            out[f"text_{name}"] = txt
            out[f"cer_{name}"] = round(cer_normalized(txt, ref), 4)
        results.append(out)

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = time.time() - t0

    # ---- 汇总 ----
    def _mean(rs, key):
        vs = [r[key] for r in rs if key in r]
        return round(float(np.mean(vs)), 4) if vs else None

    # 切法主效应(各切法独立, part_* 三选路分开报)
    means = {
        "full":          _mean(valid, "cer_full"),
        "excl_only":     _mean(valid, "cer_excl"),
        "part_enroll":   _mean(valid, "cer_part_enroll"),
        "part_excl":     _mean(valid, "cer_part_excl"),
        "part_oracle":   _mean(valid, "cer_part_oracle"),
    }
    # 选路选对率 (enroll vs excl vs oracle, oracle 永远 100%)
    n_enroll_pick = sum(1 for r in valid if r.get("enroll_picks_oracle"))
    n_excl_pick   = sum(1 for r in valid if r.get("excl_picks_oracle"))
    pick_rate = {
        "enroll_pick_rate": round(n_enroll_pick / n_valid, 4) if n_valid else 0,
        "excl_pick_rate":   round(n_excl_pick / n_valid, 4) if n_valid else 0,
        "n_enroll_pick_oracle": n_enroll_pick,
        "n_excl_pick_oracle":   n_excl_pick,
        "n_valid": n_valid,
    }
    # argmax 对/错子集分析
    argmax_correct = [r for r in valid if r.get("B2_argmax_target_idx_correct")]
    argmax_wrong   = [r for r in valid if not r.get("B2_argmax_target_idx_correct")]
    subset_stats = {
        "argmax_correct_n": len(argmax_correct),
        "argmax_correct_enroll_pick": (sum(1 for r in argmax_correct if r.get("enroll_picks_oracle")) if argmax_correct else None),
        "argmax_correct_excl_pick":   (sum(1 for r in argmax_correct if r.get("excl_picks_oracle")) if argmax_correct else None),
        "argmax_wrong_n": len(argmax_wrong),
        "argmax_wrong_enroll_pick": (sum(1 for r in argmax_wrong if r.get("enroll_picks_oracle")) if argmax_wrong else None),
        "argmax_wrong_excl_pick":   (sum(1 for r in argmax_wrong if r.get("excl_picks_oracle")) if argmax_wrong else None),
    }

    print(f"\n{'='*72}\n[分区切 timeline POC] 有效 {n_valid}/{len(samples)}, 总耗时 {total_dt/60:.1f}min")
    print(f"\n[9 组合 CER 均值]")
    print(f"  baseline full (当前主线):     {means['full']}")
    print(f"  exclusive-only (退化对照):    {means['excl_only']}")
    print(f"  partition × enroll 选路 (B2): {means['part_enroll']}")
    print(f"  partition × EXCL 选路 (创新): {means['part_excl']}")
    print(f"  partition × oracle (天花板):  {means['part_oracle']}")
    print(f"\n[选路选对率(对比 B2 baseline 25%)]")
    print(f"  enroll 选路: {pick_rate['enroll_pick_rate']*100:.0f}% ({n_enroll_pick}/{n_valid})")
    print(f"  EXCL 选路:   {pick_rate['excl_pick_rate']*100:.0f}% ({n_excl_pick}/{n_valid})")
    print(f"\n[子集分析: argmax target_idx 对/错(即 enroll 是否一开始就选对 target)]")
    print(f"  argmax 对: n={subset_stats['argmax_correct_n']}, "
          f"enroll 选对 {subset_stats['argmax_correct_enroll_pick']}, "
          f"excl 选对 {subset_stats['argmax_correct_excl_pick']}")
    print(f"  argmax 错: n={subset_stats['argmax_wrong_n']}, "
          f"enroll 选对 {subset_stats['argmax_wrong_enroll_pick']}, "
          f"excl 选对 {subset_stats['argmax_wrong_excl_pick']}")
    print(f"\n[逐条 CER]")
    print(f"  {'uid':<10} {'argmaxA':>8} {'full':>6} {'excl':>6} {'p.enr':>6} {'p.excl':>6} {'p.orc':>6}  ref")
    for r in valid:
        print(f"  {r['uid']:<10} {r.get('argmax_cer_A',0):>8.2f} "
              f"{r.get('cer_full',0):>6.2f} {r.get('cer_excl',0):>6.2f} "
              f"{r.get('cer_part_enroll',0):>6.2f} {r.get('cer_part_excl',0):>6.2f} "
              f"{r.get('cer_part_oracle',0):>6.2f}  {r['ref'][:18]}")

    summary_out = {
        "verdict_short": {
            "n_samples": n_valid,
            "baseline_full_mean": means["full"],
            "excl_only_mean": means["excl_only"],
            "partition_enroll_mean": means["part_enroll"],
            "partition_excl_mean": means["part_excl"],
            "partition_oracle_mean": means["part_oracle"],
            "enroll_pick_rate": pick_rate["enroll_pick_rate"],
            "excl_pick_rate":   pick_rate["excl_pick_rate"],
        },
        "means": means,
        "pick_rate": pick_rate,
        "subset_stats": subset_stats,
        "n_samples": len(samples), "n_valid": n_valid, "total_min": round(total_dt/60, 2),
        "results": results,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    diag_json = os.path.join(args.out_dir, "diag.json")
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_json} + {diag_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
