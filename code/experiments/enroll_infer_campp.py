"""CAM++ per-speaker 公平对照 enrollment(主 code/.venv, 复用 enroll_infer.py diar 逻辑)。

目的: 定论 CAM++ 是否值得替代 wespeaker 作主线声纹。
  线B已发现 CAM++ 整段 sim=0.121 < wespeaker 0.218 —— 但那是**不公平对比**:
  CAM++ 抽整段混音, wespeaker 抽 diarization 分离后的 per-speaker clean audio。
  本脚本让 CAM++ 也走 per-speaker 分离 → 公平对照。

链路(对齐 enroll_infer.py, 仅声纹模型换 CAM++):
  DiariZen diarization(主 .venv GPU) → collect_clean_audio 各 speaker non-overlap 段
  → 存临时 wav → subprocess 调 .venv_campp/emb_campp.py 抽 CAM++ emb(512d, L2归一)
  → 余弦 sim 选 target_idx

跨 venv 通信: 主 venv 把 wav 写临时文件, subprocess 跑 .venv_campp/Scripts/python.exe
  emb_campp.py --wav tmp.wav --out tmp.npy; 主 venv 读 .npy 还原 emb 向量。

输出 json 结构对齐 enroll_infer.py, 供 code/eval_enrollment.py 直接对比:
  recognition/enrollment/speakers/sims/target_idx/target_speaker/max_sim/
  reject_threshold/rejected(注: 本脚本不转写, transcript="" 字段保留对齐)

运行(需先 source setenv.sh, HF_HUB_OFFLINE=1):
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/enroll_infer_campp.py \
    --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
    --recognition-folder E:/midea_target_asr/test_wav/dataset/final \
    --out-json code/enroll_infer_campp_result.json
"""
import os, sys, json, argparse, glob, time, subprocess, tempfile
import numpy as np
import soundfile as sf
import torch
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# ---- 跨 venv 路径(硬约束: CAM++ 在 .venv_campp) ----
CAMPP_PY = os.path.join(_HERE, ".venv_campp", "Scripts", "python.exe")
EMB_SCRIPT = os.path.join(_HERE, "emb_campp.py")

DIAR_MODEL = "E:/hf_cache/diarizen-wavlm-large-s80-md"

# 自包含: 把 DiariZen / pyannote 加入 sys.path(等价 setenv.sh 的 PYTHONPATH)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        if _p not in sys.path:
            sys.path.insert(0, _p)


# ============ 复用 enroll_infer.py 的 diar 辅助(复制, 不 import 以保自包含) ============
def get_diarization_mask(per_speaker_samples, audio_length):
    """per_speaker_samples: list of [(start,end),...]; audio_length: 50Hz 帧数。→ [N, T] float."""
    mask = torch.zeros(len(per_speaker_samples), audio_length)
    for i, spk_samples in enumerate(per_speaker_samples):
        for start, end in spk_samples:
            mask[i, round(start * 50):round(end * 50)] = 1
    return mask


def collect_clean_audio(audio, diar_mask, i, sr=16000, frame_sec=0.02, min_seg_sec=0.3):
    """从 diar_mask 提取 speaker i 的 non-overlap 独占连续段音频(避开重叠污染声纹)。
    与 enroll_infer.py 完全一致。"""
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


# ============ 跨 venv CAM++ emb 抽取 ============
def campp_emb_via_subprocess(wav_np, sr=16000, tmpdir=None, num_threads=2):
    """把 wav 写临时文件 → subprocess 调 .venv_campp/emb_campp.py → 读回 L2归一化 emb。
    wav_np: np.float32 @sr。返回 (emb np.float32, None) 或 (None, err_str)。"""
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="campp_emb_")
    wav_path = os.path.join(tmpdir, "seg.wav")
    np_path = os.path.join(tmpdir, "seg.npy")
    # 极短段补齐(对齐 emb_campp.emb_of 的 min_len=1s, 但显式在此也补一次更稳)
    if len(wav_np) < sr:
        wav_np = np.tile(wav_np, sr // max(1, len(wav_np)) + 1)[:sr]
    try:
        sf.write(wav_path, np.ascontiguousarray(wav_np.astype(np.float32)), sr, subtype="FLOAT")
    except Exception:
        sf.write(wav_path, np.ascontiguousarray(wav_np.astype(np.float32)), sr)
    cmd = [CAMPP_PY, EMB_SCRIPT, "--wav", wav_path, "--out", np_path,
           "--num-threads", str(num_threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                           env=os.environ.copy())
    except subprocess.TimeoutExpired:
        return None, "emb_campp subprocess timeout(120s)"
    if r.returncode != 0:
        return None, f"emb_campp rc={r.returncode} stderr={r.stderr[:200]}"
    if not os.path.exists(np_path):
        return None, f"emb_campp no output npy. stdout={r.stdout[:200]}"
    e = np.load(np_path)
    return e.astype(np.float32), None


def campp_emb_batch_via_subprocess(wav_paths, num_threads=4):
    """批量: 一次 subprocess 跑 emb_campp.py --batch → 返回 {path: emb}。
    减少 subprocess 启动开销(450 条时显著)。wav 已落盘。"""
    tmpdir = tempfile.mkdtemp(prefix="campp_batch_")
    list_path = os.path.join(tmpdir, "list.json")
    out_path = os.path.join(tmpdir, "out.json")
    json.dump(list(wav_paths), open(list_path, "w", encoding="utf-8"))
    cmd = [CAMPP_PY, EMB_SCRIPT, "--batch", list_path, "--out-batch", out_path,
           "--num-threads", str(num_threads)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           env=os.environ.copy())
    except subprocess.TimeoutExpired:
        return None, "batch timeout(600s)"
    if r.returncode != 0 or not os.path.exists(out_path):
        return None, f"batch rc={r.returncode} stderr={r.stderr[:200]} stdout={r.stdout[:200]}"
    d = json.load(open(out_path, encoding="utf-8"))
    emb_map = {p: np.asarray(v, dtype=np.float32) for p, v in d.get("embeddings", {}).items()}
    return emb_map, None


# ============ 主流程 ============
def main():
    ap = argparse.ArgumentParser(description="CAM++ per-speaker 公平对照 enrollment")
    ap.add_argument("--enrollment", required=True, help="目标说话人参考音频 wav")
    ap.add_argument("--recognition", help="识别音频 wav(单条)")
    ap.add_argument("--recognition-folder", help="识别音频文件夹(批量)")
    ap.add_argument("--reject-threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=DIAR_MODEL)
    ap.add_argument("--out-json", default=os.path.join(_HERE, "enroll_infer_campp_result.json"))
    ap.add_argument("--batch-emb", action="store_true",
                    help="批量模式: 先把所有 enrollment+各 speaker clean 段落盘, "
                         "一次 subprocess 抽完再回算 sim(450 条时省 subprocess 开销)")
    ap.add_argument("--campp-threads", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CAMPP_PY):
        sys.exit(f"[fatal] CAM++ venv python 不存在: {CAMPP_PY}\n"
                 f"        请先建 .venv_campp 并装 sherpa-onnx(线B 已完成)")
    if not os.path.exists(EMB_SCRIPT):
        sys.exit(f"[fatal] emb_campp.py 不存在: {EMB_SCRIPT}")

    # ---- 加载 DiariZen(仅 diar, 不加载 DiCoW——本脚本不转写) ----
    print(f"[load] DiariZen {args.diarization_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)

    # ---- enrollment CAM++ emb(跨 venv) ----
    print(f"[enrollment] {args.enrollment} (写临时 wav → .venv_campp 抽 CAM++)")
    enroll_wav, _ = librosa.load(args.enrollment, sr=16000)
    enroll_emb, err = campp_emb_via_subprocess(enroll_wav, 16000, num_threads=args.campp_threads)
    if enroll_emb is None:
        sys.exit(f"[fatal] enrollment CAM++ emb 抽取失败: {err}")
    print(f"  → enroll emb {enroll_emb.shape} (L2 norm={np.linalg.norm(enroll_emb):.4f})")

    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav"))) \
        if args.recognition_folder else [args.recognition]

    results = []
    for rec in recs:
        t0 = time.time()
        audio, sr = librosa.load(rec, sr=16000)
        dur = len(audio) / sr

        # 1) diarization
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [diar-fail] {os.path.basename(rec)} {type(e).__name__}: {str(e)[:80]} → 跳过")
            results.append({"recognition": rec, "enrollment": args.enrollment,
                            "error": f"{type(e).__name__}: {str(e)[:120]}",
                            "rejected": True, "transcript": "", "chars": 0,
                            "_voicemodel": "CAMpp-per-speaker"})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        audio_len = int(len(audio) / sr * 50)   # 50Hz 帧数(对齐 enroll_infer: ifp.shape[-1]//2)
        diar_mask = get_diarization_mask(per_spk, audio_len)

        # 2) 各 speaker clean 段(同 enroll_infer: 独占段优先, 不足 fallback 全 timeline)
        segs_by_spk = []
        for i in range(len(speakers)):
            seg = collect_clean_audio(audio, diar_mask, i, sr)
            if seg is None or len(seg) < sr * 0.3:
                segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
            min_len = sr * 1
            if len(seg) < min_len:
                seg = np.tile(seg, min_len // max(1, len(seg)) + 1)[:min_len]
            segs_by_spk.append(seg)

        # 3) 抽 CAM++ emb: 逐条 subprocess(单条/小批量) 或 批量(默认逐条, 稳)
        spk_embs = []
        if args.batch_emb:
            tmpdir = tempfile.mkdtemp(prefix="campp_recs_")
            paths = []
            for i, seg in enumerate(segs_by_spk):
                p = os.path.join(tmpdir, f"spk{i}.wav")
                sf.write(p, np.ascontiguousarray(seg.astype(np.float32)), sr, subtype="FLOAT")
                paths.append(p)
            emap, eerr = campp_emb_batch_via_subprocess(paths, num_threads=args.campp_threads)
            if emap is None:
                print(f"  [batch-emb-fail] {eerr} → 退逐条")
                for seg in segs_by_spk:
                    e, eerr2 = campp_emb_via_subprocess(seg, sr, num_threads=args.campp_threads)
                    spk_embs.append(e if e is not None else np.zeros_like(enroll_emb))
            else:
                for p in paths:
                    spk_embs.append(emap.get(p, np.zeros_like(enroll_emb)))
        else:
            for seg in segs_by_spk:
                e, eerr = campp_emb_via_subprocess(seg, sr, num_threads=args.campp_threads)
                if e is None:
                    print(f"    [emb-fail] {eerr} → 用零向量占位")
                    spk_embs.append(np.zeros_like(enroll_emb))
                else:
                    spk_embs.append(e)

        # 4) 余弦匹配(已 L2 归一 → 内积即余弦)
        sims = np.array([float(np.dot(enroll_emb, e)) for e in spk_embs])
        target_idx = int(np.argmax(sims))
        max_sim = float(sims[target_idx])
        sim_str = ", ".join(f"{speakers[i]}:{float(sims[i]):.3f}" for i in range(len(speakers)))
        dt = time.time() - t0
        print(f"\n[rec] {os.path.basename(rec)} ({dur:.1f}s) speakers={speakers}")
        print(f"  [match-CAM++] {{{sim_str}}} → target={speakers[target_idx]} sim={max_sim:.3f} ({dt:.1f}s)")

        results.append({
            "recognition": rec, "enrollment": args.enrollment,
            "speakers": speakers,
            "sims": {speakers[i]: float(sims[i]) for i in range(len(speakers))},
            "target_idx": target_idx, "target_speaker": speakers[target_idx],
            "max_sim": max_sim, "reject_threshold": args.reject_threshold,
            "rejected": max_sim < args.reject_threshold,
            "transcript": "", "chars": 0, "rtf": dt / dur,
            "_voicemodel": "CAMpp-per-speaker", "_dim": int(enroll_emb.shape[0]),
        })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    sims_all = [r["max_sim"] for r in results if "max_sim" in r]
    print(f"\n[done] {len(results)} 条 → {args.out_json}")
    if sims_all:
        print(f"CAM++ per-speaker 均 sim={np.mean(sims_all):.3f} "
              f"(对照 wespeaker 450 矩阵 ≈0.218, CAM++ 整段 ≈0.121)")
        print(f"  max={np.max(sims_all):.3f} min={np.min(sims_all):.3f} "
              f">0.3比例={np.mean(np.array(sims_all)>0.3):.2f}")
    print(f"评估: code/.venv/Scripts/python.exe code/eval_enrollment.py "
          f"--enroll-json {os.path.basename(args.out_json)} --label CAMpp-per-speaker")


if __name__ == "__main__":
    main()
