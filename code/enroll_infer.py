"""Part1: enrollment→wespeaker 锁定唯一 target(方案 B 独立脚本, 不动 pipeline.py)。

链路:
  DiariZen diarization(找 N 个 speaker 时间段)
  → 复用 diar._embedding(wespeaker) 抽 enrollment + 各 speaker 声纹
  → 余弦匹配选 target_idx(max sim)
  → 构造 target 的 STNO mask(复制 pipeline.py 的 get_stno_mask 逻辑)
  → DiCoW generate 只转 target 一个(不再全 speaker 转)
  → 兜底拒识: max_sim < reject_threshold → 输出空(target 不在场, 对接拒识 40%)

向后兼容: 原 inference.py 不变, 本脚本独立。

环境(同完整 pipeline, 需先 source setenv):
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/enroll_infer.py \
    --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
    --recognition E:/midea_target_asr/test_wav/dataset/final/<xxx>.wav

验证场景:
  sanity(自匹配): enrollment == recognition → sim≈1, 锁定正确
  target 选择:    enrollment=target 音频, recognition=重叠(target+nontarget) → 锁定 target
  兜底拒识:       enrollment 与 recognition 不同人 → sim 低 → 拒识空输出
"""
import os, sys, json, argparse, glob, time
import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DICOW_MODEL = "E:/hf_cache/DiCoW_v3_2"
DIAR_MODEL = "E:/hf_cache/diarizen-wavlm-large-s80-md"

# 自包含: 把 DiCoW-inference / DiariZen / pyannote-audio 加入 sys.path(等价运行时 export PYTHONPATH)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)


# ---- STNO 构造(复制自 pipeline.py, 保证与原 pipeline 帧率/语义一致) ----
def get_diarization_mask(per_speaker_samples, audio_length):
    """per_speaker_samples: list of [(start,end),...]; audio_length: 50Hz 帧数。→ [N, T] float."""
    mask = torch.zeros(len(per_speaker_samples), audio_length)
    for i, spk_samples in enumerate(per_speaker_samples):
        for start, end in spk_samples:
            mask[i, round(start * 50):round(end * 50)] = 1
    return mask


def get_stno_mask(diar_mask, s_index):
    """diar_mask: [N, T]; s_index: target。→ [4, T] (sil/target/nontarget/overlap 每帧 one-hot)。"""
    non_target = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
    non_target[s_index] = False
    sil = (1 - diar_mask).prod(axis=0)
    anyone_else = (1 - diar_mask[non_target]).prod(axis=0)
    target_spk = diar_mask[s_index] * anyone_else
    non_target_spk = (1 - diar_mask[s_index]) * (1 - anyone_else)
    overlap = diar_mask[s_index] - target_spk
    return torch.stack([sil, target_spk, non_target_spk, overlap], axis=0)


def collect_clean_audio(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    """从 diar_mask 提取 speaker i 的 non-overlap 独占连续段音频(避开重叠区污染声纹)。
    diar_mask: [N, T@50Hz]; 返回拼接 np.ndarray, 无足够独占段则 None。"""
    others = diar_mask.sum(axis=0) - diar_mask[i]      # 其他 speaker 占用帧数
    clean = (diar_mask[i] > 0) & (others == 0)         # speaker i 独占帧
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


def main():
    ap = argparse.ArgumentParser(description="enrollment→wespeaker 锁定唯一 target")
    ap.add_argument("--enrollment", required=True, help="目标说话人参考音频 wav")
    ap.add_argument("--recognition", help="识别音频 wav(单条)")
    ap.add_argument("--recognition-folder", help="识别音频文件夹(批量)")
    ap.add_argument("--reject-threshold", type=float, default=0.5, help="兜底拒识余弦阈值")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dicow-model", default=DICOW_MODEL)
    ap.add_argument("--diarization-model", default=DIAR_MODEL)
    ap.add_argument("--language", default="zh")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "enroll_infer_result.json"))
    ap.add_argument("--always-generate", action="store_true",
                    help="总generate(不因sim拒识跳过), 供 eval 扫阈值; 拒识条仍记 transcript+rejected=True")
    ap.add_argument("--enroll-augment", action="store_true",
                    help="enrollment 加噪增强: 干净+多档加噪 emb 均值, 提声纹鲁棒")
    ap.add_argument("--aug-snrs", default="10,5,0", help="enrollment 加噪增强的 SNR 档(逗号分)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"[load] DiCoW {args.dicow_model} on {device}")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.dicow_model)
    fe = AutoFeatureExtractor.from_pretrained(args.dicow_model)

    print(f"[load] DiariZen {args.diarization_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)

    # wespeaker embedding 抽取(复用 diar._embedding, 零额外加载)
    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]          # (1, 1, n)
        elif w.dim() == 2:
            w = w[None]                # (1, 1, n)
        with torch.no_grad():
            emb = diar._embedding(w)   # (batch, dim) np.ndarray
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)   # (dim,) 已归一化

    # enrollment embedding(可选加噪增强: 干净 + 多档加噪 emb 均值, 提带噪鲁棒)
    enroll_wav, _ = librosa.load(args.enrollment, sr=16000)
    if args.enroll_augment:
        from simulate_pipeline import add_noise
        aug_snrs = [int(s) for s in args.aug_snrs.split(",") if s.strip()]
        rng = np.random.default_rng(0)
        embs = [get_emb(enroll_wav)]
        for snr in aug_snrs:
            noisy = add_noise(enroll_wav, rng.standard_normal(len(enroll_wav)).astype(np.float32), snr)
            embs.append(get_emb(noisy))
        enroll_emb = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=-1)
        print(f"[enrollment] 加噪增强: 干净 + {len(aug_snrs)}档加噪{aug_snrs}dB emb 均值 → {tuple(enroll_emb.shape)}")
    else:
        enroll_emb = get_emb(enroll_wav)
        print(f"[enrollment] {args.enrollment} ({len(enroll_wav)/16000:.1f}s) → emb {tuple(enroll_emb.shape)}")

    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav"))) \
        if args.recognition_folder else [args.recognition]

    results = []
    for rec in recs:
        t0 = time.time()
        audio, sr = librosa.load(rec, sr=16000)
        dur = len(audio) / sr

        # 1) diarization → 各 speaker 时间段(恶劣音频可能触发 pyannote reconstruct 边界 bug, 容错跳过)
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [diar-fail] {os.path.basename(rec)} {type(e).__name__}: {str(e)[:80]} → 跳过该条")
            results.append({"recognition": rec, "enrollment": args.enrollment,
                            "error": f"{type(e).__name__}: {str(e)[:120]}",
                            "rejected": True, "transcript": "", "chars": 0})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]

        # 1.5) 提前算 mel 特征 + diar_mask(抽声纹用 non-overlap 帧, 转 target 也要用)
        ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        audio_len = ifp.shape[-1] // 2          # 50Hz 帧数
        diar_mask = get_diarization_mask(per_spk, audio_len)

        # 2) 各 speaker 声纹: 优先 non-overlap 独占帧(避开重叠污染), 不足则 fallback 全 timeline
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

        # 3) 余弦匹配(已归一化 → 内积即余弦)
        sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
        target_idx = int(torch.argmax(sims))
        max_sim = float(sims[target_idx])
        sim_str = ", ".join(f"{speakers[i]}:{float(sims[i]):.3f}" for i in range(len(speakers)))
        print(f"\n[rec] {os.path.basename(rec)} ({dur:.1f}s) speakers={speakers}")
        print(f"  [match] {{{sim_str}}} → target={speakers[target_idx]} sim={max_sim:.3f}")

        # 4) 兜底拒识 / 转写(ifp/diar_mask 已在 1.5 算好)
        rejected = max_sim < args.reject_threshold
        if rejected and not args.always_generate:
            text, verdict = "", f"REJECT(target 不在场, max_sim={max_sim:.3f}<{args.reject_threshold})"
        else:
            stno = get_stno_mask(diar_mask, target_idx)    # [4, T]
            am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                out = dicow.generate(input_features=ifp, attention_mask=am,
                                     stno_mask=stno[None].to(device, dtype),
                                     language=args.language, task="transcribe", max_new_tokens=200)
            seqs = out["sequences"] if isinstance(out, dict) else out
            text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
            verdict = (f"REJECT_GEN(max_sim={max_sim:.3f}<{args.reject_threshold}, always-generate 仍转)" if rejected
                       else f"TRANSCRIBE(target={speakers[target_idx]})")

        dt = time.time() - t0
        print(f"  [{verdict}] {len(text)}字 ({dt:.1f}s, RTF={dt/dur:.3f}): {text}")
        results.append({
            "recognition": rec, "enrollment": args.enrollment,
            "speakers": speakers,
            "sims": {speakers[i]: float(sims[i]) for i in range(len(speakers))},
            "target_idx": target_idx, "target_speaker": speakers[target_idx],
            "max_sim": max_sim, "reject_threshold": args.reject_threshold,
            "rejected": max_sim < args.reject_threshold,
            "transcript": text, "chars": len(text), "rtf": dt / dur,
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {len(results)} 条 → {args.out_json}")


if __name__ == "__main__":
    main()
