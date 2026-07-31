"""exp_spk_oracle_qwen.py — Qwen3 oracle POC: 判 CAM++/声纹强化在 Qwen3 后端下的生死。

【背景】exp_spk_oracle.py(vanilla) 证伪声纹强化(死区 oracle_CER 0.607, 完美选 target 也
不及格 → 音频摧毁)。但那是 vanilla 评估。Qwen3-ASR 更强(死区 qwen 0.459 vs vanilla 0.828),
"Qwen3 下选 target 是否还限制 CER"是开放问题。用户核心假设: CAM++ 更强声纹选 target 更准
→ CER 降。本 POC 是判这个的前提。

【问】Qwen3 oracle(作弊完美选 target → qwen 转写)死区/主战场 CER vs qwen argmax:
  - oracle << argmax(cer_gain>0.10) + miss 中 oracle_sim≥0.2>30% → 选 target 还是大问题,
    CAM++ 有戏 → 投端到端集成
  - oracle ≈ argmax → mel 是主问题, CAM++ 救不了, 声纹强化死(GO=否)

【方法】复用 exp_spk_oracle.py 框架, 转写换 Qwen3(subprocess 批量, 因 qwen 在 .venv_qwen):
  Phase1 主venv: load diar, 遍历样本 diar+切每个 speaker timeline → 存临时 wav(uid__spkI)
  Phase2 subprocess: qwen_asr_backend.py 批量转写 → uid2text.json
  Phase3: 读回算各 speaker CER, oracle=argmin, go/no-go

【数据源】poc_qwen_asr_full_result.json(qwen 全量, 字段 sim/qwen_cer/ref/uid)筛死区/主战场。
  基线 = qwen argmax CER(同 uid), 公平对比。

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_spk_oracle_qwen.py [--sim-max 0.2] [--n-sample 60]
  # 主战场: --sim-min 0.2 --sim-max 0.4
产物: code/exp_spk_oracle_qwen.json + stdout go/no-go。
"""
import os, sys, json, time, argparse, subprocess, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import AutoFeatureExtractor

from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import cut_target_timeline, to_simplified, digit_postproc  # noqa: F401 (to_simplified/digit_postproc 复用 qwen 已做, 保 import 一致)
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))


def get_emb_factory(diar, device):
    """复用 diar._embedding(wespeaker) 抽声纹, 与 enroll_infer.get_emb 一致。"""
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
    """subprocess 调 qwen_asr_backend.py 批量转写 slice_dir 下所有 wav → uid2text.json。"""
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--batch-size", str(batch_size)]
    print(f"[qwen] subprocess 转写 {slice_dir} (batch={batch_size})")
    env = dict(os.environ)
    subprocess.run(cmd, check=True, env=env)
    return json.load(open(out_json, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Qwen3 oracle POC: 判 CAM++/声纹强化生死")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--fe-model", default=resolve_model("VANILLA"), help="Whisper fe, 仅算 audio_len")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=60)
    ap.add_argument("--sim-min", type=float, default=0.0, help="死区0/主战场0.2")
    ap.add_argument("--sim-max", type=float, default=0.2, help="死区0.2/主战场0.4")
    ap.add_argument("--qwen-full", default=os.path.join(_HERE, "poc_qwen_asr_full_result.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--slice-dir", default="E:/target_slices_oracle_qwen")
    ap.add_argument("--qwen-out", default=os.path.join(_HERE, "_oracle_qwen_uid2text.json"))
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_spk_oracle_qwen.json"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.slice_dir, "*.wav")):  # 清旧切片
        os.remove(f)

    # ---- 1. 数据: qwen_full 筛桶 + 抽样 ----
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows = qfull["rows"] if isinstance(qfull, dict) and "rows" in qfull else qfull
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    bucket = [r for r in rows
              if r.get("sim") is not None and args.sim_min <= r["sim"] < args.sim_max
              and r.get("ref") and r["uid"] in uid2pair]
    print(f"[data] qwen_full={len(rows)} 桶 sim[{args.sim_min},{args.sim_max})={len(bucket)}")
    rng = np.random.default_rng(args.seed)
    samples = [bucket[i] for i in sorted(rng.permutation(len(bucket))[:args.n_sample])]
    print(f"[data] 抽样 {len(samples)} 条 (seed={args.seed})")

    # ---- 2. load diar + fe ----
    print(f"[load] DiariZen {args.diarization_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)
    get_emb = get_emb_factory(diar, device)
    fe = AutoFeatureExtractor.from_pretrained(args.fe_model)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. Phase1: diar + 切每个 speaker timeline → 存 wav ----
    meta, slice_uids_all = [], []
    t0 = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        enr, rec, uid, ref = pair["enrollment"], pair["recognition"], d["uid"], d["ref"]
        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)
            diar_out = diar(rec)
            speakers = list(diar_out.labels())
            per_spk = [diar_out.label_timeline(s) for s in speakers]
            n_spk = len(speakers)

            ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features
            audio_len = ifp.shape[-1] // 2
            diar_mask = get_diarization_mask(per_spk, audio_len)

            spk_embs = []
            for i in range(n_spk):
                seg = collect_clean_audio(audio, diar_mask, i, sr)
                if seg is None or len(seg) < sr * 0.3:
                    segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                    seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                spk_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
            argmax_idx = int(torch.argmax(sims))

            slice_uids = []
            for i in range(n_spk):
                tgt = cut_target_timeline(audio, per_spk[i], sr=sr)
                suid = f"{uid}__spk{i}"
                sf.write(os.path.join(args.slice_dir, suid + ".wav"),
                         np.ascontiguousarray(tgt.astype(np.float32)), 16000)
                slice_uids.append(suid)
                slice_uids_all.append(suid)

            meta.append({"uid": uid, "ref": ref, "n_spk": n_spk, "speakers": speakers,
                         "sims": [round(float(s), 4) for s in sims], "argmax_idx": argmax_idx,
                         "max_sim": round(float(sims[argmax_idx]), 4),
                         "qwen_argmax_cer": d.get("qwen_cer"),
                         "qwen_argmax_text": d.get("qwen_text") or d.get("text"),
                         "slice_uids": slice_uids})
            print(f"  [{n+1}/{len(samples)}] {uid} spk={n_spk} max_sim={float(sims[argmax_idx]):.3f} "
                  f"qwen_argmax_CER={d.get('qwen_cer')} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- 4. Phase2: 批量 qwen 转写 ----
    print(f"\n[qwen] 转写 {len(slice_uids_all)} 个 speaker slice...")
    uid2text = run_qwen_batch(args.slice_dir, args.qwen_out, args.batch_size)

    # ---- 5. Phase3: 算 oracle + go/no-go ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        per_spk_text = [uid2text.get(su, "") for su in m["slice_uids"]]
        per_spk_cer = [float(cer_official(t, m["ref"])) for t in per_spk_text]
        oracle_idx = int(np.argmin(per_spk_cer))
        argmax_cer = per_spk_cer[m["argmax_idx"]]
        oracle_cer = per_spk_cer[oracle_idx]
        oracle_sim = m["sims"][oracle_idx]
        results.append({**m, "per_spk_text": per_spk_text,
                        "per_spk_cer": [round(c, 4) for c in per_spk_cer],
                        "oracle_idx": oracle_idx, "oracle_cer": round(oracle_cer, 4),
                        "argmax_cer": round(argmax_cer, 4), "oracle_sim": oracle_sim,
                        "argmax_correct": m["argmax_idx"] == oracle_idx})

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = time.time() - t0
    print(f"\n{'='*70}\n[Qwen3 oracle POC] 有效 {n_valid}/{len(samples)} 条, 总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本, 无法判定。")
        return

    argmax_correct_rate = float(np.mean([r["argmax_correct"] for r in valid]))
    argmax_cer_mean = float(np.mean([r["argmax_cer"] for r in valid]))
    oracle_cer_mean = float(np.mean([r["oracle_cer"] for r in valid]))
    cer_gain = argmax_cer_mean - oracle_cer_mean
    oracle_sims = np.array([r["oracle_sim"] for r in valid])
    oracle_sim_mean = float(np.mean(oracle_sims))

    miss = [r for r in valid if not r["argmax_correct"]]
    n_miss = len(miss)
    miss_oracle_sim = float(np.mean([r["oracle_sim"] for r in miss])) if miss else float("nan")
    miss_argmax_cer = float(np.mean([r["argmax_cer"] for r in miss])) if miss else float("nan")
    miss_oracle_cer = float(np.mean([r["oracle_cer"] for r in miss])) if miss else float("nan")
    miss_recognizable = (float(np.mean([r["oracle_sim"] >= 0.2 for r in miss])) if miss else float("nan"))

    multi = [r for r in valid if r["n_spk"] >= 2]
    multi_correct = float(np.mean([r["argmax_correct"] for r in multi])) if multi else float("nan")

    print(f"\n[核心指标] 桶 sim[{args.sim_min},{args.sim_max})")
    print(f"  argmax 选对率:              {argmax_correct_rate*100:.1f}% ({sum(r['argmax_correct'] for r in valid)}/{n_valid})")
    print(f"    ↳ 多speaker(≥2)子集:      {multi_correct*100:.1f}%" if multi else "    ↳ 无多speaker样本")
    print(f"  Qwen3 argmax_CER 均值(系统): {argmax_cer_mean:.3f}")
    print(f"  Qwen3 oracle_CER 均值(完美): {oracle_cer_mean:.3f}")
    print(f"  CER 收益(argmax→oracle):    {cer_gain:+.3f} ({cer_gain/argmax_cer_mean*100 if argmax_cer_mean>0 else 0:+.1f}%)")
    print(f"  oracle_sim 均值:            {oracle_sim_mean:.3f}")
    print(f"\n[miss 子集(argmax 选错, n={n_miss})]")
    print(f"  miss argmax_CER: {miss_argmax_cer:.3f} | oracle_CER: {miss_oracle_cer:.3f} | oracle_sim: {miss_oracle_sim:.3f}")
    print(f"  miss 中 oracle_sim≥0.2(声纹强化可救): {miss_recognizable*100:.1f}%")

    # ---- go/no-go 判 CAM++ 生死 ----
    print(f"\n{'='*70}\n[go/no-go 判 CAM++ 生死]")
    gainable = (cer_gain > 0.10) and (n_miss >= 3) and (miss_recognizable > 0.3)
    if gainable:
        verdict = "GO=是(CAM++ 有戏)"
        reason = (f"Qwen3 oracle_CER {oracle_cer_mean:.3f} 显著低于 argmax {argmax_cer_mean:.3f} "
                  f"(Δ{cer_gain:+.3f}), miss {n_miss} 条中 {miss_recognizable*100:.0f}% 正确 target "
                  f"声纹 sim≥0.2 本可识别 → 选 target 还是大问题, CAM++ 更强声纹能救。投端到端集成。")
    elif argmax_correct_rate >= 0.5 and cer_gain <= 0.10:
        verdict = "GO=否(CAM++ 死)"
        reason = (f"argmax 选对率 {argmax_correct_rate*100:.0f}%≥50%, Qwen3 oracle_CER {oracle_cer_mean:.3f} "
                  f"≈ argmax {argmax_cer_mean:.3f}(Δ{cer_gain:+.3f}≤0.10) → mel 是主问题非选 target, "
                  f"CAM++ 救不了(声纹编码 who 不编码 clarity, 与 07-11 CAM++ B/A 证伪一致)。声纹强化关闭。")
    else:
        verdict = "GO=否(偏)"
        reason = (f"argmax 选对率 {argmax_correct_rate*100:.0f}%, oracle_CER {oracle_cer_mean:.3f} 仍高, "
                  f"cer_gain {cer_gain:+.3f} 不足 → 选 target 非主瓶颈, CAM++ 收益有限。")

    print(f"  判定: {verdict}")
    print(f"  理由: {reason}")

    summary = {
        "verdict": verdict, "reason": reason,
        "bucket": f"sim[{args.sim_min},{args.sim_max})", "n_sample": len(samples), "n_valid": n_valid,
        "argmax_correct_rate": round(argmax_correct_rate, 4),
        "multi_spk_correct_rate": round(multi_correct, 4) if multi else None,
        "argmax_cer_mean": round(argmax_cer_mean, 4), "oracle_cer_mean": round(oracle_cer_mean, 4),
        "cer_gain": round(cer_gain, 4), "oracle_sim_mean": round(oracle_sim_mean, 4),
        "n_miss": n_miss, "miss_oracle_sim_mean": round(miss_oracle_sim, 4),
        "miss_argmax_cer_mean": round(miss_argmax_cer, 4), "miss_oracle_cer_mean": round(miss_oracle_cer, 4),
        "miss_oracle_recognizable_rate": round(miss_recognizable, 4),
        "total_min": round(total_dt / 60, 2),
    }
    out = {"summary": summary, "results": results}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
