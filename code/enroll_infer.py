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
from text_utils import to_simplified, cut_target_timeline, digit_postproc
from repro import set_global_seed, resolve_model, reset_peak_gpu, peak_gpu_mib
import pyarrow  # 预热: 避免 import pyannote 时扫描 sys.path 的 DiariZen 目录触发 WinError 6714

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# 模型路径走 repro.resolve_model(env override → HF repo id), 不再裸硬编码(spec 可复现性 C5)

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


def _sample_babble(noise_dir, n, rng):
    """从 noise_dir/*.wav 随机采一段长 n 的 babble 噪声(真实语音, 比白噪更对症 babble 场景)。"""
    wavs = sorted(glob.glob(os.path.join(noise_dir, "*.wav")))
    if not wavs:
        return rng.standard_normal(n).astype(np.float32)
    pick = wavs[int(rng.integers(len(wavs)))]
    nw, _ = librosa.load(pick, sr=16000)
    if len(nw) < n:
        nw = np.tile(nw, n // len(nw) + 1)
    return nw[:n].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="enrollment→wespeaker 锁定唯一 target")
    ap.add_argument("--enrollment", help="目标说话人参考音频 wav(配 --recognition / --recognition-folder)")
    ap.add_argument("--recognition", help="识别音频 wav(单条)")
    ap.add_argument("--recognition-folder", help="识别音频文件夹(批量)")
    ap.add_argument("--pairs", help="[{enrollment,recognition}, ...] json(单进程批量化,模型加载1次;优先于 --enrollment)")
    ap.add_argument("--reject-threshold", type=float, default=0.5, help="兜底拒识余弦阈值")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dicow-model", default=resolve_model("DICOW"))
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--language", default="zh")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "enroll_infer_result.json"))
    ap.add_argument("--always-generate", action="store_true",
                    help="总generate(不因sim拒识跳过), 供 eval 扫阈值; 拒识条仍记 transcript+rejected=True")
    ap.add_argument("--enroll-augment", action="store_true",
                    help="enrollment 加噪增强: 干净+多档加噪 emb 均值, 提声纹鲁棒")
    ap.add_argument("--aug-snrs", default="10,5,0", help="enrollment 加噪增强的 SNR 档(逗号分)")
    ap.add_argument("--aug-noise-dir", help="babble 噪声池目录(增强比白噪更对症真实 babble; 不填则用白噪)")
    ap.add_argument("--asr-backend", default="dicow", choices=["dicow", "vanilla"],
                    help="ASR 后端: dicow(FDDT/STNO 条件化, fallback) / vanilla(切target timeline+whisper, 主线 CER 减半)")
    ap.add_argument("--vanilla-model", default=resolve_model("VANILLA"),
                    help="vanilla 后端 Whisper 模型(默认 large-v3-turbo)")
    ap.add_argument("--seed", type=int, default=42, help="全局种子(可复现性, repro.set_global_seed)")
    args = ap.parse_args()
    set_global_seed(args.seed)  # 可复现性: 固定 torch/numpy/cudnn(FAQ 核查硬要求 2)
    if not args.pairs and not args.enrollment:
        ap.error("--pairs 与 --enrollment 至少填一个")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    if args.asr_backend == "vanilla":
        print(f"[load] vanilla Whisper {args.vanilla_model} on {device}")
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.vanilla_model, torch_dtype=dtype).to(device).eval()
        _model_path = args.vanilla_model
    else:
        print(f"[load] DiCoW {args.dicow_model} on {device}")
        asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
        _model_path = args.dicow_model
    tok = AutoTokenizer.from_pretrained(_model_path)
    fe = AutoFeatureExtractor.from_pretrained(_model_path)

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

    # ---- 构造 (enrollment, recognition) 对列表(3 种输入归一) ----
    if args.pairs:
        pair_rows = json.load(open(args.pairs, encoding="utf-8"))
        pairs = [(r["enrollment"], r["recognition"]) for r in pair_rows]
        print(f"[pairs] {args.pairs}: {len(pairs)} 对 (单进程批量化, 模型加载 1 次)")
    elif args.recognition_folder:
        recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav")))
        pairs = [(args.enrollment, rec) for rec in recs]
    else:
        pairs = [(args.enrollment, args.recognition)]

    # enrollment embedding 缓存(按路径;datasetA 每条不同 enr 也能命中同说话人复用,
    # 且 --enrollment+folder 老模式同一 enr 只提一次)
    _enroll_cache = {}
    def get_enroll_emb(enr_path):
        if enr_path in _enroll_cache:
            return _enroll_cache[enr_path]
        w, _ = librosa.load(enr_path, sr=16000)
        if args.enroll_augment:
            from simulate_pipeline import add_noise
            aug_snrs = [int(s) for s in args.aug_snrs.split(",") if s.strip()]
            rng = np.random.default_rng(args.seed)  # 可复现性: 消费全局 seed(避免双套种子)
            use_babble = bool(args.aug_noise_dir)
            embs = [get_emb(w)]
            for snr in aug_snrs:
                noise = _sample_babble(args.aug_noise_dir, len(w), rng) if use_babble \
                        else rng.standard_normal(len(w)).astype(np.float32)
                noisy = add_noise(w, noise, snr)
                embs.append(get_emb(noisy))
            emb = torch.nn.functional.normalize(torch.stack(embs).mean(0), dim=-1)
            src = f"babble({os.path.basename(os.path.dirname(args.aug_noise_dir.rstrip('/')))}{os.path.basename(args.aug_noise_dir.rstrip('/'))})" if use_babble else "白噪"
            print(f"[enrollment] {os.path.basename(enr_path)} ({len(w)/16000:.1f}s) +{len(aug_snrs)}档{src}增强 → cached")
        else:
            emb = get_emb(w)
            print(f"[enrollment] {os.path.basename(enr_path)} ({len(w)/16000:.1f}s) → cached")
        _enroll_cache[enr_path] = emb
        return emb

    results = []
    for enr, rec in pairs:
        enroll_emb = get_enroll_emb(enr)
        reset_peak_gpu()  # 可复现性: 每条循环开始重置, 记每条整条峰值(不被 langfix retry 二次 generate 重置)
        t0 = time.time()
        audio, sr = librosa.load(rec, sr=16000)
        dur = len(audio) / sr

        # 1) diarization → 各 speaker 时间段(恶劣音频可能触发 pyannote reconstruct 边界 bug, 容错跳过)
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [diar-fail] {os.path.basename(rec)} {type(e).__name__}: {str(e)[:80]} → 跳过该条")
            results.append({"recognition": rec, "enrollment": enr,
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
        # STNO target 帧占比(target 独占无重叠帧 / 总帧) — 三路融合第三信号
        non_tg = torch.ones((diar_mask.shape[0],), dtype=torch.bool)
        non_tg[target_idx] = False
        anyone_else = (1 - diar_mask[non_tg]).prod(axis=0)
        stno_target_ratio = float((diar_mask[target_idx] * anyone_else).mean())
        target_active_ratio = float(diar_mask[target_idx].mean())
        sim_str = ", ".join(f"{speakers[i]}:{float(sims[i]):.3f}" for i in range(len(speakers)))
        print(f"\n[rec] {os.path.basename(rec)} ({dur:.1f}s) speakers={speakers}")
        print(f"  [match] {{{sim_str}}} → target={speakers[target_idx]} sim={max_sim:.3f}")

        # 4) 兜底拒识 / 转写(ifp/diar_mask 已在 1.5 算好)
        rejected = max_sim < args.reject_threshold
        if rejected and not args.always_generate:
            text, verdict = "", f"REJECT(target 不在场, max_sim={max_sim:.3f}<{args.reject_threshold})"
        else:
            if args.asr_backend == "vanilla":
                # vanilla: 切 target timeline(含重叠区)拼接 → vanilla.generate 无条件化
                target_audio = cut_target_timeline(audio, per_spk[target_idx], sr=sr)
                ifp_v = fe(target_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
                am_v = torch.ones(1, ifp_v.shape[-1], dtype=torch.bool, device=device)
                with torch.no_grad():
                    out = asr_model.generate(input_features=ifp_v, attention_mask=am_v,
                                             language=args.language, task="transcribe", max_new_tokens=200)
                seqs = out["sequences"] if isinstance(out, dict) else out
                text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
                # vanilla 不跑 langfix(英文幻觉 0.59%, langfix 是 dicow 治标)
            else:
                # dicow: stno_mask 条件化 + SE-DiCoW 自登记 + langfix retry(原逻辑保留)
                stno = get_stno_mask(diar_mask, target_idx)    # [4, T]
                am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
                gen_kwargs = dict(input_features=ifp, attention_mask=am,
                                  stno_mask=stno[None].to(device, dtype),
                                  language=args.language, task="transcribe", max_new_tokens=200)
                # ---- SE-DiCoW(config.uses_enrollments=True): 自登记 enrollment, cross-attn 解重叠 ----
                # 自登记 ≠ 外部 enrollment wav —— 从 recognition mel+target STNO 提 target 最密 30s 窗,
                # 复刻 DiCoW-inference/pipeline.py::_process_enrollment_sample + max_ones_window。
                if bool(getattr(asr_model.config, "uses_enrollments", False) or getattr(asr_model.config, "use_enrollments", False)):
                    if not getattr(asr_model, "_se_setup_done", False):
                        vocab = tok.get_vocab()
                        tok.upper_cased_tokens = {}
                        for _t, _i in vocab.items():
                            if len(_t) < 1:
                                continue
                            _lo = (_t[0] + _t[1].lower() + (_t[2:] if len(_t) > 2 else '')) \
                                  if (_t[0] == 'Ġ' and len(_t) > 1) else (_t[0].lower() + _t[1:])
                            if _lo != _t and vocab.get(_lo) is not None:
                                tok.upper_cased_tokens[vocab[_lo]] = _i
                        if hasattr(asr_model, "set_tokenizer"):
                            asr_model.set_tokenizer(tok)
                        asr_model.config.model_type = "whisper"
                        asr_model._se_setup_done = True
                        print("[load] SE-DiCoW uses_enrollments=True → cross-attn 内部启用(self-enrolled)")
                    # SE-DiCoW 的 self-enrollment 在模型内部从 stno_mask target 行自动提取,
                    # config.uses_enrollments 控制 cross-attn 路径; generate 接口与 DiCoW 相同(实测不接受 enrollments kwarg)。
                with torch.no_grad():
                    out = asr_model.generate(**gen_kwargs)
                seqs = out["sequences"] if isinstance(out, dict) else out
                text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
                # [langfix 加强] 英文漂移 → prompt_ids 中文偏置 + greedy 重生成(retry 复用 gen_kwargs 含 enrollments)
                _L = [c for c in text if c.isalpha()]
                _er = sum(c.isascii() for c in _L) / len(_L) if len(_L) >= 4 else 0.0
                if _er > 0.4:
                    if not hasattr(asr_model, "_zh_prompt_ids"):
                        asr_model._zh_prompt_ids = torch.tensor(
                            tok("以下是普通话的句子。", add_special_tokens=False).input_ids, device=device)
                    retry_kwargs = dict(gen_kwargs)
                    retry_kwargs.update(num_beams=1, prompt_ids=asr_model._zh_prompt_ids)
                    with torch.no_grad():
                        out2 = asr_model.generate(**retry_kwargs)
                    seqs2 = out2["sequences"] if isinstance(out2, dict) else out2
                    text2 = tok.batch_decode(seqs2, skip_special_tokens=True)[0].strip()
                    _PFX = "以下是普通话的句子。"
                    if text2.startswith(_PFX):
                        text2 = text2[len(_PFX):].strip()
                    _L2 = [c for c in text2 if c.isalpha()]
                    _er2 = sum(c.isascii() for c in _L2) / len(_L2) if len(_L2) >= 4 else 0.0
                    if _er2 < _er:
                        text = text2
                    print(f"  [langfix-retry] 英文率 {_er:.2f}→{_er2:.2f} {'采纳重生成' if _er2 < _er else '保留首次'}")
            # 统一繁简归一(dicow + vanilla 都过)
            text = to_simplified(text)
            # 数字后处理: 阿拉伯→中文数字对齐 ref 口径(从 MiMo 对比学到的 quick win, 两口径都不亏)
            text = digit_postproc(text)
            verdict = (f"REJECT_GEN(max_sim={max_sim:.3f}<{args.reject_threshold}, always-generate 仍转)" if rejected
                       else f"TRANSCRIBE(target={speakers[target_idx]}, backend={args.asr_backend})")

        dt = time.time() - t0
        peak = peak_gpu_mib()  # 可复现性: 每条峰值显存(FAQ 核查硬要求 5)
        print(f"  [{verdict}] {len(text)}字 ({dt:.1f}s, RTF={dt/dur:.3f}, batch=1, peak_mem={peak}MiB): {text}")
        results.append({
            "recognition": rec, "enrollment": enr,
            "speakers": speakers,
            "sims": {speakers[i]: float(sims[i]) for i in range(len(speakers))},
            "target_idx": target_idx, "target_speaker": speakers[target_idx],
            "max_sim": max_sim, "reject_threshold": args.reject_threshold,
            "stno_target_ratio": stno_target_ratio, "target_active_ratio": target_active_ratio,
            "rejected": max_sim < args.reject_threshold,
            "asr_backend": args.asr_backend,
            "infer_sec": round(dt, 3),  # 单条纯推理(不含模型加载), 对齐官方 batch=1 duration
            "batch_size": 1, "peak_mem_mib": peak,  # 可复现性: batch/显存(FAQ 核查硬要求 4/5)
            "transcript": text, "chars": len(text), "rtf": dt / dur,
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {len(results)} 条 → {args.out_json}")


if __name__ == "__main__":
    main()
