"""诊断 cmd_2637 DiariZen diar 为什么 spk0/spk1 都混两人。

样本: cmd_2637.wav (2.48s, 男女双人并行, 音色差极大音量相当)。
假设: 男女音色差大, diar 理应轻松分开却没分开; 查清是参数问题还是能力问题。

三步:
  1) 默认 diar 跑 2637, 检查: 独占段/emb 可分性/重叠占比/聚类数
  2) 扫参数: 强制 num_speakers=2, ahc_threshold, Fa/Fb
  3) 给结论: 找到分开的参数 → 参数问题(可修); 否则能力问题

产物:
  - runs/_diag_2637_diar/<config>/spk*.wav (各 speaker 段, 离线听)
  - runs/_diag_2637_diar/summary.json (各 config 的诊断指标)
"""
import inspect as _inspect  # speechbrain lazy proxy workaround (enroll_infer:24-29)
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import os, sys, json, time, copy, traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

DICOW_INF = os.path.join(_HERE, "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

import pyarrow  # 预热(避开 pyannote sys.path 扫描 WinError 6714)

import numpy as np
import torch
import librosa
import soundfile as sf
from diarizen.pipelines.inference import DiariZenPipeline
from repro import resolve_model, set_global_seed


REC_WAV = "E:/midea_target_asr/datasetA/pos/cmd_2637.wav"
ENR_WAV = "E:/midea_target_asr/datasetA/pos/kws_2637.wav"
OUT_DIR = "E:/midea_target_asr/code/runs/_diag_2637_diar"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42


def get_diarization_mask(per_speaker_samples, audio_length):
    """复刻 enroll_infer.py:52 — per_spk timeline → [N,T@50Hz] mask。"""
    mask = torch.zeros(len(per_speaker_samples), audio_length)
    for i, spk_samples in enumerate(per_speaker_samples):
        for start, end in spk_samples:
            mask[i, round(start * 50):round(end * 50)] = 1
    return mask


def collect_clean_audio(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    """复刻 enroll_infer.py:73 — speaker i 的 non-overlap 独占连续段。"""
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


def build_pipeline(config_override):
    """传 config_parse override 聚类参数, 否则用 model 默认 config.toml。

    config_override 形如:
      {"min_speakers":2, "max_speakers":2, "ahc_threshold":0.3, "Fa":0.07, "Fb":0.8}
    """
    diar_hub = Path(resolve_model("DIAR")).expanduser().absolute()
    config_path = diar_hub / "config.toml"
    import toml
    base = toml.load(config_path.as_posix())
    base_inf = dict(base["inference"]["args"])
    base_clu = dict(base["clustering"]["args"])
    base_clu.update(config_override)
    config_parse = {"inference": {"args": base_inf}, "clustering": {"args": base_clu}}
    diar = DiariZenPipeline(
        diarizen_hub=diar_hub,
        embedding_model=_resolve_wespeaker(),
        config_parse=config_parse,
    ).to(DEVICE)
    return diar, base_clu


_WESP_CACHE = None
def _resolve_wespeaker():
    """wespeaker 路径(本地 cache, 避开 gated)。"""
    global _WESP_CACHE
    if _WESP_CACHE:
        return _WESP_CACHE
    from huggingface_hub import hf_hub_download
    _WESP_CACHE = hf_hub_download(
        repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
        filename="pytorch_model.bin",
        local_files_only=True,
    )
    return _WESP_CACHE


def run_one_config(audio_np, sr, label, clu_override, get_emb_fn, save_dir):
    """跑一组 diar 配置 + 全套诊断。

    返回 dict(label, num_spk, timelines, exclusive_sec, overlap_pct, spk_embs_cos, cos_min, cos_max).
    """
    print(f"\n========== {label} ==========")
    print(f"  override: {clu_override}")
    set_global_seed(SEED)
    diar, used_clu = build_pipeline(clu_override)
    print(f"  used clustering args: {used_clu}")

    t0 = time.time()
    try:
        # 写临时 wav 让 diar 走 torchaudio.load(它只收 str)
        tmp_wav = os.path.join(save_dir, "_tmp_input.wav")
        sf.write(tmp_wav, audio_np, sr)
        diar_out = diar(tmp_wav, sess_name="cmd_2637")
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [FAIL] {type(e).__name__}: {e}\n{tb}")
        return {"label": label, "override": clu_override, "error": f"{type(e).__name__}: {str(e)[:200]}", "traceback": tb}
    finally:
        try: diar.cpu(); del diar; torch.cuda.empty_cache()
        except Exception: pass
    elapsed = time.time() - t0

    speakers = list(diar_out.labels())
    per_spk = [diar_out.label_timeline(s) for s in speakers]
    n_spk = len(speakers)
    dur = len(audio_np) / sr
    audio_len_50Hz = int(np.ceil(dur * 50))
    diar_mask = get_diarization_mask(per_spk, audio_len_50Hz)

    # ----- 独占段分析 -----
    exclusive_segs = []  # 各 speaker 独占总秒数
    exclusive_audios = []  # 各 speaker 独占音频(np.ndarray)
    for i in range(n_spk):
        clean = collect_clean_audio(audio_np, diar_mask, i, sr=sr)
        if clean is None:
            exclusive_segs.append(0.0)
            exclusive_audios.append(np.zeros(0, dtype=np.float32))
        else:
            exclusive_segs.append(len(clean) / sr)
            exclusive_audios.append(clean)

    # 重叠帧占比: 两 speaker 同时 active 的 50Hz 帧 / 总帧
    if n_spk >= 2:
        # diar_mask 是 torch.Tensor, 用 numpy 桥接(避免 .astype on Tensor 错误)
        active_count = diar_mask.sum(dim=0)  # (T,) 每帧多少 speaker 活跃
        any_active = (active_count > 0).sum().item()
        both_active = (active_count >= 2).sum().item()
        # 仅看有人 active 的帧里多少是重叠
        overlap_pct_in_active = both_active / any_active if any_active > 0 else 0.0
        overlap_pct_in_total = both_active / diar_mask.shape[1]
    else:
        overlap_pct_in_active = 0.0
        overlap_pct_in_total = 0.0

    # 两人并行(全程无独占段)判定
    total_excl = sum(exclusive_segs)

    # ----- emb 可分性 -----
    spk_embs = []
    for i in range(n_spk):
        clean = exclusive_audios[i] if len(exclusive_audios[i]) > sr * 0.3 else None
        if clean is None or len(clean) < sr * 0.3:
            # fallback 全 timeline
            segs = [audio_np[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
            clean = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
        # 至少 1s, 不足 tile
        if len(clean) < sr * 1:
            clean = np.tile(clean, sr // max(1, len(clean)) + 1)[:sr]
        spk_embs.append(get_emb_fn(clean))
    cos_matrix = None
    cos_off_diag = []
    if n_spk >= 2:
        E = torch.stack(spk_embs)  # (n, d) normalized
        cos_matrix = (E @ E.T).cpu().numpy()
        iu = np.triu_indices(n_spk, k=1)
        cos_off_diag = cos_matrix[iu].tolist()

    # timeline 详情
    timelines_str = []
    for i, s in enumerate(speakers):
        segs = [(round(st, 2), round(en, 2)) for st, en in per_spk[i]]
        timelines_str.append({"spk": str(s), "segments": segs, "total_sec": round(sum(e - s for s, e in per_spk[i]), 2),
                              "exclusive_sec": round(exclusive_segs[i], 2)})

    # 存各 spk wav(独占 + 全 timeline) 便于人工听
    cfg_dir = os.path.join(save_dir, label)
    os.makedirs(cfg_dir, exist_ok=True)
    for i in range(n_spk):
        if len(exclusive_audios[i]) > 0:
            sf.write(os.path.join(cfg_dir, f"spk{i}_exclusive.wav"), exclusive_audios[i], sr)
        full = np.concatenate([audio_np[int(s * sr):int(e * sr)] for s, e in per_spk[i]]) if per_spk[i] else np.zeros(sr, dtype=np.float32)
        sf.write(os.path.join(cfg_dir, f"spk{i}_full.wav"), full, sr)

    # enrollment 余弦(参考)
    enr_wav, _ = librosa.load(ENR_WAV, sr=16000)
    enr_emb = get_emb_fn(enr_wav)
    enr_sims = [float(torch.dot(enr_emb, e)) for e in spk_embs]

    result = {
        "label": label,
        "override": clu_override,
        "used_clustering_args": {k: v for k, v in used_clu.items()},
        "num_speakers_detected": n_spk,
        "elapsed_sec": round(elapsed, 2),
        "audio_dur_sec": round(dur, 2),
        "timelines": timelines_str,
        "total_exclusive_sec": round(total_excl, 2),
        "overlap_pct_in_active_frames": round(overlap_pct_in_active, 3),
        "overlap_pct_in_total_frames": round(overlap_pct_in_total, 3),
        "exclusive_sec_per_spk": [round(x, 2) for x in exclusive_segs],
        "cos_matrix_spk": [[round(float(x), 3) for x in row] for row in cos_matrix] if cos_matrix is not None else None,
        "cos_off_diag": [round(x, 3) for x in cos_off_diag],
        "cos_min": round(min(cos_off_diag), 3) if cos_off_diag else None,
        "cos_max": round(max(cos_off_diag), 3) if cos_off_diag else None,
        "enr_sims": [round(x, 3) for x in enr_sims],
    }
    print(f"  n_spk={n_spk}, total_excl={total_excl:.2f}s, "
          f"overlap%_active={overlap_pct_in_active:.3f}, overlap%_total={overlap_pct_in_total:.3f}")
    if cos_off_diag:
        print(f"  spk pairwise cos: {cos_off_diag} (min={min(cos_off_diag):.3f}, max={max(cos_off_diag):.3f})")
    print(f"  enr_sims: {enr_sims}")
    for tl in timelines_str:
        print(f"    spk{tl['spk']}: total={tl['total_sec']}s, excl={tl['exclusive_sec']}s, segs={tl['segments']}")
    return result


def main():
    set_global_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[load] audio {REC_WAV}")
    audio, sr = librosa.load(REC_WAV, sr=16000)
    print(f"  dur={len(audio)/sr:.2f}s, sr={sr}")

    # 先用默认 diar 仅取 _embedding(后续各 config 复用同一 get_emb, diar 实例会被释放)
    print(f"[load] default diar for embedding fn")
    diar_hub = Path(resolve_model("DIAR")).expanduser().absolute()
    diar_default = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(DEVICE)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(DEVICE)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.inference_mode():
            emb = diar_default._embedding(w)
        emb = torch.as_tensor(emb, device=DEVICE, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    # 验证 get_emb 可用
    _ = get_emb(audio[:16000])

    # ----- 各 config 跑一遍 -----
    configs = [
        ("default", {}),
    ] if len(sys.argv) < 2 or sys.argv[1] != "all" else [
        ("default", {}),
        ("force_nspk2", {"min_speakers": 2, "max_speakers": 2}),
        ("ahc_0.3", {"ahc_threshold": 0.3}),
        ("ahc_0.4", {"ahc_threshold": 0.4}),
        ("ahc_0.5", {"ahc_threshold": 0.5}),
        ("ahc_0.3_nspk2", {"min_speakers": 2, "max_speakers": 2, "ahc_threshold": 0.3}),
        ("Fa_0.15", {"Fa": 0.15}),
        ("Fb_1.5", {"Fb": 1.5}),
        ("Fa_0.15_Fb_1.5_nspk2", {"Fa": 0.15, "Fb": 1.5, "min_speakers": 2, "max_speakers": 2}),
        ("Fb_0.3", {"Fb": 0.3}),
        ("Fb_0.3_nspk2", {"min_speakers": 2, "max_speakers": 2, "Fb": 0.3}),
        ("agc_ahc0.3", {"method": "AgglomerativeClustering", "ahc_threshold": 0.3, "min_cluster_size": 5}),
        ("agc_ahc0.5", {"method": "AgglomerativeClustering", "ahc_threshold": 0.5, "min_cluster_size": 5}),
        ("agc_ahc0.7", {"method": "AgglomerativeClustering", "ahc_threshold": 0.7, "min_cluster_size": 5}),
    ]
    all_results = []
    for label, override in configs:
        try:
            r = run_one_config(audio, sr, label, override, get_emb, OUT_DIR)
        except Exception as e:
            r = {"label": label, "override": override, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            print(f"[FAIL {label}] {type(e).__name__}: {str(e)[:200]}")
            torch.cuda.empty_cache()
        all_results.append(r)
        # 写中间结果(增量, 防中断丢)
        with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"recognition": REC_WAV, "enrollment": ENR_WAV, "results": all_results},
                      f, ensure_ascii=False, indent=2)

    print("\n========== SUMMARY ==========")
    print(f"{'label':<28} {'n_spk':<6} {'excl_s':<8} {'ovl%act':<9} {'cos_min':<8} {'cos_max':<8}")
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<28} ERROR: {r['error'][:60]}")
            continue
        print(f"{r['label']:<28} {r['num_speakers_detected']:<6} "
              f"{r['total_exclusive_sec']:<8} {r['overlap_pct_in_active_frames']:<9} "
              f"{str(r['cos_min']):<8} {str(r['cos_max']):<8}")


if __name__ == "__main__":
    main()
