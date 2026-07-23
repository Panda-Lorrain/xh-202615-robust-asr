"""exp_sepformer_qwen.py — SepFormer 源分离 + Qwen3 转写攻死区 CER POC。

【背景】exp_sepformer_poc.py(vanilla 转写)证伪(CER 0.859 vs vanilla 0.918, Δ-0.059 噪声内,
oracle 0.752 反劣)。但 Qwen3-ASR 更鲁棒(ExtremeNoise 4×), Qwen3 下重跑 SepFormer 可能有效。
本 POC 验证用户假设: Qwen3-ASR + SepFormer 协同可能翻盘。

【问】SepFormer 分离 target 流 + Qwen3 转写, 死区 CER 是否显著降 vs qwen argmax 基线?
  GO=是: Δ<-0.10 + correct 升 → 源分离+Qwen3 协同有效, 投集成
  GO=否: 持平/更差 → 死区物理极限, Qwen3 也救不动(vanilla 证伪在 Qwen3 复现)

【方法】复用 exp_sepformer_poc.py 框架, 转写换 Qwen3(subprocess 批量):
  Phase1 主venv: load SepFormer+diar(声纹选流), 遍历死区样本分离→选 target/other 流→存 wav
  Phase2 subprocess: qwen_asr_backend.py 批量转写
  Phase3: 算 CER + go/no-go

【数据源】poc_qwen_asr_full_result.json 筛死区(sim<0.2), 基线 = qwen argmax CER(同 uid)。

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_sepformer_qwen.py [--n-sample 40]
产物: code/exp_sepformer_qwen.json + stdout go/no-go。
"""
import os, sys, json, time, argparse, subprocess, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容(复刻 exp_sepformer_poc.py, SB 1.1.0 LazyModule inspect-guard) ----
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

import numpy as np
import torch
import librosa
import soundfile as sf

from enroll_infer import get_diarization_mask, collect_clean_audio  # noqa: F401 (副作用: 触发 enroll_infer 的 DiariZen/pyannote sys.path 设置; sepformer 选流用 diar._embedding)
from text_utils import to_simplified, digit_postproc  # noqa: F401 (qwen backend 已归一)
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))


def load_sepformer(device, savedir):
    from speechbrain.inference.separation import SepformerSeparation as separator
    from speechbrain.utils.fetching import LocalStrategy
    return separator.from_hparams(source="speechbrain/sepformer-whamr16k", savedir=savedir,
                                  local_strategy=LocalStrategy.COPY, run_opts={"device": str(device)})


def separate(mix_np, model):
    mix = torch.from_numpy(np.ascontiguousarray(mix_np.astype(np.float32))).to(model.device)[None, :]
    est = model.separate_batch(mix).squeeze(0).detach().cpu().numpy()  # [T, n_src]
    return est.T  # [n_src, T]


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
        with torch.no_grad():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)
    return get_emb


def run_qwen_batch(slice_dir, out_json, batch_size=16):
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--batch-size", str(batch_size)]
    print(f"[qwen] subprocess 转写 {slice_dir}")
    subprocess.run(cmd, check=True)
    return json.load(open(out_json, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="SepFormer+Qwen3 攻死区 CER POC")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=40)
    ap.add_argument("--sim-max", type=float, default=0.2)
    ap.add_argument("--qwen-full", default=os.path.join(_HERE, "poc_qwen_asr_full_result.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--slice-dir", default="E:/target_slices_sepformer_qwen")
    ap.add_argument("--qwen-out", default=os.path.join(_HERE, "_sepformer_qwen_uid2text.json"))
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_sepformer_qwen.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.slice_dir, "*.wav")):
        os.remove(f)

    # ---- 1. 数据 ----
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows = qfull["rows"] if isinstance(qfull, dict) and "rows" in qfull else qfull
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    deadzone = [d for d in rows if d.get("sim") is not None and d["sim"] < args.sim_max
                and d.get("ref") and d["uid"] in uid2pair]
    print(f"[data] qwen_full={len(rows)} 死区(sim<{args.sim_max})={len(deadzone)}")
    rng = np.random.default_rng(args.seed)
    samples = [deadzone[i] for i in sorted(rng.permutation(len(deadzone))[:args.n_sample])]
    print(f"[data] 抽样 {len(samples)} 条 (seed={args.seed})")

    # ---- 2. load ----
    print(f"[load] SepFormer whamr16k → {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. Phase1: 分离 → 选 target/other 流 → 存 wav ----
    meta, slice_uids_all = [], []
    t0 = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        enr, rec, uid, ref = pair["enrollment"], pair["recognition"], d["uid"], d["ref"]
        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]

            stream_embs = []
            for i in range(n_src):
                seg = sources[i]
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                stream_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in stream_embs])
            target_idx = int(torch.argmax(sims))
            target_audio = sources[target_idx]
            other_idx = (1 - target_idx) if n_src == 2 else -1

            t_uid = f"{uid}__target"
            sf.write(os.path.join(args.slice_dir, t_uid + ".wav"),
                     np.ascontiguousarray(target_audio.astype(np.float32)), 16000)
            slice_uids_all.append(t_uid)
            o_uid = None
            if other_idx >= 0:
                o_uid = f"{uid}__other"
                sf.write(os.path.join(args.slice_dir, o_uid + ".wav"),
                         np.ascontiguousarray(sources[other_idx].astype(np.float32)), 16000)
                slice_uids_all.append(o_uid)

            meta.append({"uid": uid, "ref": ref, "n_src": n_src,
                         "stream_sims": [round(float(s), 4) for s in sims],
                         "target_idx": target_idx, "target_sim": round(float(sims[target_idx]), 4),
                         "other_idx": other_idx, "target_uid": t_uid, "other_uid": o_uid,
                         "qwen_argmax_cer": d.get("qwen_cer"), "qwen_sim": d.get("sim")})
            print(f"  [{n+1}/{len(samples)}] {uid} n_src={n_src} sep_sim={float(sims[target_idx]):.3f} "
                  f"qwen_argmax_CER={d.get('qwen_cer')} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- 4. Phase2: 批量 qwen ----
    print(f"\n[qwen] 转写 {len(slice_uids_all)} 流...")
    uid2text = run_qwen_batch(args.slice_dir, args.qwen_out, args.batch_size)

    # ---- 5. Phase3: 算 CER + go/no-go ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        sep_text = uid2text.get(m["target_uid"], "")
        sep_cer = float(cer_official(sep_text, m["ref"]))
        other_cer = None
        if m["other_idx"] >= 0 and m["other_uid"]:
            other_text = uid2text.get(m["other_uid"], "")
            other_cer = float(cer_official(other_text, m["ref"]))
        results.append({**m, "sep_text": sep_text, "sep_cer": round(sep_cer, 4),
                        "other_cer": round(other_cer, 4) if other_cer is not None else None,
                        "qwen_argmax_cer": m["qwen_argmax_cer"]})

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = time.time() - t0
    print(f"\n{'='*70}\n[SepFormer+Qwen3 POC] 有效 {n_valid}/{len(samples)}, 总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本, 无法判定。")
        return

    sep_cers = np.array([r["sep_cer"] for r in valid])
    paired = [r for r in valid if r.get("qwen_argmax_cer") is not None]
    v_cers = np.array([r["qwen_argmax_cer"] for r in paired])
    sep_paired = np.array([r["sep_cer"] for r in paired])
    sep_mean = float(np.mean(sep_cers))
    sep_correct = float(np.mean(sep_cers < 0.5))
    v_mean = float(np.mean(v_cers)) if paired else float("nan")
    v_correct = float(np.mean(v_cers < 0.5)) if paired else float("nan")
    delta = float(np.mean(sep_paired - v_cers)) if paired else float("nan")
    both = [r for r in valid if r.get("other_cer") is not None]
    oracle_sep = np.array([min(r["sep_cer"], r["other_cer"]) for r in both])
    oracle_mean = float(np.mean(oracle_sep)) if both else float("nan")
    better = sum(1 for r in paired if r["sep_cer"] < r["qwen_argmax_cer"] - 0.01)
    worse = sum(1 for r in paired if r["sep_cer"] > r["qwen_argmax_cer"] + 0.01)

    print(f"\n[核心指标] 死区 sim<{args.sim_max}")
    print(f"  SepFormer+Qwen3 CER 均值:    {sep_mean:.3f} (correct<0.5: {sep_correct*100:.0f}%)")
    print(f"  Qwen3 argmax 基线 CER 均值:  {v_mean:.3f} (correct: {v_correct*100:.0f}%)")
    print(f"  Δ(SepFormer - argmax):       {delta:+.3f}")
    print(f"  逐条: 更优 {better} / 更差 {worse} (n={len(paired)})")
    print(f"  SepFormer oracle(两流取优):  {oracle_mean:.3f}")

    print(f"\n{'='*70}\n[go/no-go 判定]")
    significant = (delta < -0.10) and (sep_correct > v_correct + 0.05)
    oracle_recoverable = (oracle_mean < v_mean - 0.10) and not significant and not (oracle_mean != oracle_mean)
    if significant:
        verdict = "GO=是"
        reason = (f"SepFormer+Qwen3 死区 CER {sep_mean:.3f} 显著低于 qwen argmax {v_mean:.3f} "
                  f"(Δ{delta:+.3f}, correct {sep_correct*100:.0f}% vs {v_correct*100:.0f}%)。"
                  f"源分离+Qwen3 协同有效, 投 SepFormer 集成方向。")
    elif oracle_recoverable:
        verdict = "GO=偏是(改选流)"
        reason = (f"argmax {sep_mean:.3f} 未胜基线, 但 oracle(两流取优){oracle_mean:.3f} < argmax {v_mean:.3f} "
                  f"→ 分离拎出干净 target 只是选流选错, 改进 target 流选择能救。")
    elif oracle_mean >= v_mean - 0.05:
        verdict = "GO=否"
        reason = (f"SepFormer 即便 oracle(两流取优){oracle_mean:.3f} 也不显著低于 qwen argmax {v_mean:.3f} "
                  f"→ 分离没拎出更干净 target mel, 死区物理极限, Qwen3 也救不动(vanilla 证伪在 Qwen3 复现)。"
                  f"接受 qwen 0.3436 天花板。")
    else:
        verdict = "GO=否(偏)"
        reason = (f"oracle {oracle_mean:.3f} 略低于 {v_mean:.3f} 但收益边际(Δ<{0.10}), 投大工程不划算。")

    print(f"  判定: {verdict}")
    print(f"  理由: {reason}")

    summary = {"verdict": verdict, "reason": reason, "n_sample": len(samples), "n_valid": n_valid,
               "sep_cer_mean": round(sep_mean, 4), "sep_correct_rate": round(sep_correct, 4),
               "qwen_argmax_cer_mean": round(v_mean, 4) if paired else None,
               "qwen_argmax_correct_rate": round(v_correct, 4) if paired else None,
               "delta_mean": round(delta, 4) if paired else None,
               "n_better": better, "n_worse": worse,
               "sepformer_oracle_cer_mean": round(oracle_mean, 4) if both else None,
               "model": "speechbrain/sepformer-whamr16k + Qwen3-ASR-1.7B",
               "total_min": round(total_dt / 60, 2)}
    out = {"summary": summary, "results": results}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
