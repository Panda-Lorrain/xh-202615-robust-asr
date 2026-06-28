"""CAM++ per-speaker 声纹抽取器(.venv_campp, sherpa-onnx, CPU)。

被 enroll_infer_campp.py 跨 venv 用 subprocess 调用, 也可独立 CLI:
  单条:  .venv_campp/Scripts/python.exe code/emb_campp.py --wav a.wav --out a.npy
  批量:  .venv_campp/Scripts/python.exe code/emb_campp.py --batch list.json --out-batch emb_dict.json
         (list.json = ["a.wav","b.wav"] 或 [{"path":"a.wav"},...]; 输出 {path: [512 floats]})

接口对齐 enroll_infer.py 的 wespeaker get_emb: 输入 wav(np.float32@16k) → L2 归一化 emb。
模型 = E:/hf_cache/campplus/campplus.onnx (512d, 含 sherpa meta, 同 campp_vs_wespeaker.py)。

注意: 只在 .venv_campp 跑, 严禁被主 code/.venv(transformers4.42.4) import。
"""
import os, sys, json, argparse
import numpy as np
import soundfile as sf  # sherpa-onnx venv 自带; 用于读临时 wav(避免依赖 librosa)
import sherpa_onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
CAMPP_ONNX = "E:/hf_cache/campplus/campplus.onnx"


def make_extractor(num_threads=2):
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=CAMPP_ONNX, num_threads=num_threads, debug=False)
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def load_wav_16k(path):
    """读 wav → np.float32 mono @16k。失败返回 None。"""
    try:
        wav, sr = sf.read(path, dtype="float32")
    except Exception:
        # 兜底: scipy(LINEAR16) 或 librosa(若装了)
        try:
            import librosa
            wav, sr = librosa.load(path, sr=16000)
            return np.ascontiguousarray(wav.astype(np.float32)), 16000
        except Exception:
            return None, None
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        # sherpa-onnx 自带 resample
        try:
            wav = sherpa_onnx.resample(wav, sr, 16000)
        except Exception:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return np.ascontiguousarray(wav.astype(np.float32)), 16000


def emb_of(ext, wav_np, sr=16000, min_len=1600):
    """wav(np.float32) → L2归一化 emb (np.float32, dim)。极短段补齐到 ≥1s。"""
    w = np.ascontiguousarray(wav_np.astype(np.float32))
    if len(w) < min_len:
        w = np.tile(w, min_len // max(1, len(w)) + 1)[:min_len]
    st = ext.create_stream()
    st.accept_waveform(sr, w)
    st.input_finished()
    e = np.asarray(ext.compute(st), dtype=np.float32)
    n = np.linalg.norm(e) + 1e-9
    return e / n


def main():
    ap = argparse.ArgumentParser(description="CAM++ 声纹抽取(.venv_campp)")
    ap.add_argument("--wav", help="单条 wav 路径")
    ap.add_argument("--out", help="单条输出 .npy(默认 <wav>.npy)")
    ap.add_argument("--batch", help="批量输入 json: [path,...] 或 [{path},...]")
    ap.add_argument("--out-batch", help="批量输出 json/npy(默认 emb_dict.json)")
    ap.add_argument("--model", default=CAMPP_ONNX)
    ap.add_argument("--num-threads", type=int, default=2)
    args = ap.parse_args()

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.model, num_threads=args.num_threads, debug=False)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
    dim = ext.dim
    print(f"[emb_campp] loaded dim={dim} model={args.model}", flush=True)

    if args.batch:
        items = json.load(open(args.batch, encoding="utf-8"))
        paths = []
        for it in items:
            if isinstance(it, str):
                paths.append(it)
            elif isinstance(it, dict):
                paths.append(it.get("path") or it.get("wav") or it.get("file"))
        out = {}
        ok, fail = 0, 0
        for p in paths:
            w, sr = load_wav_16k(p)
            if w is None:
                print(f"  [fail-load] {p}", flush=True)
                fail += 1
                continue
            try:
                e = emb_of(ext, w, sr)
                out[p] = e.tolist()
                ok += 1
            except Exception as ex:
                print(f"  [fail-emb] {p}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
                fail += 1
        out_path = args.out_batch or os.path.join(_HERE, "emb_dict.json")
        if out_path.endswith(".json"):
            d = {"model": args.model, "dim": dim, "count": len(out), "embeddings": out}
            json.dump(d, open(out_path, "w", encoding="utf-8"))
        else:  # .npy: 存 dict[path→ ndarray]
            np.save(out_path, {p: np.asarray(v, dtype=np.float32) for p, v in out.items()},
                    allow_pickle=True)
        print(f"[done] batch ok={ok} fail={fail} → {out_path}", flush=True)
        return

    if not args.wav:
        ap.error("must give --wav or --batch")
    w, sr = load_wav_16k(args.wav)
    if w is None:
        sys.exit(f"[emb_campp] load failed: {args.wav}")
    e = emb_of(ext, w, sr)
    out_path = args.out or (os.path.splitext(args.wav)[0] + ".npy")
    np.save(out_path, e)
    print(f"[done] {args.wav} → {out_path} shape={e.shape} norm={np.linalg.norm(e):.4f}", flush=True)


if __name__ == "__main__":
    main()
