"""exp_ase_pvad_poc.py — ASE-PVAD self-augmentation POC(出题方 ICASSP 2026 论文 #02 arXiv:2601.12769 复现)

背景: 主战场桶 sim[0.2,0.4) 全量 668 oracle POC GO=否(`exp_spk_oracle_mainbattle_full.json`):
  argmax_CER 0.65 = 选错target 0.14(22%)+ 音频摧毁 0.51(78%); miss 172 中 oracle_sim≥0.2(ASE 可救面)仅 21%。
ASE-PVAD 估算收益 CER 腿 +0.2(乐观)。本 POC 实跑验证 ASE 机制真实救回率(非估算)。

ASE-PVAD 配方(论文):
  - 1s 窗 / 0.2s shift 扫 recognition 混合音频(帧级 emb)
  - 选单一 keyframe: cos(enroll, frame) 最大(不用 top-k, 论证单说话人占多数时间)
  - Eq.6 加性融合: aug = λ*enroll + (1-λ)*avg(aug, keyframe), λ=0.1, L2 normalize, 迭代 5 次
  - 帧来源 recognition(非 enrollment), 修 near-field enroll vs far-field test domain mismatch
  - 推理期做(模型参数不动), zero-training 友好

本 POC 增强(比论文更严): 用 collect_clean_audio 取 speaker 独占帧(避开 overlap 污染)扫 keyframe。
护栏: best_sim < min_best_sim → enroll 严重不可靠(sim 反向风险), fallback baseline argmax。

三档(每条转写所有 speaker, 复用 per_spk_cer):
  baseline = argmax(声纹 sims)   ASE = argmax(aug_enroll sims)   oracle = argmin(per_spk_cer)
判据: ASE 救回率(baseline miss 里 ASE 选对的占比) + ASE CER 实际改善(vs oracle 上限)。

用法:
  code/.venv/Scripts/python.exe code/exp_ase_pvad_poc.py --sim-min 0.2 --sim-max 0.4 --n-sample 700
产物: code/exp_ase_pvad_poc.json + stdout 三档对比 + miss 救回细分。
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import cut_target_timeline  # to_simplified/digit_postproc 在 transcribe 内
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model
from exp_spk_oracle import load_models, get_emb_factory, transcribe


def ase_augment_keyframe(audio, per_spk, diar_mask, enroll_emb, get_emb,
                         sr=16000, frame_sec=1.0, shift_sec=0.2,
                         lam=0.1, n_iter=5, min_best_sim=0.15):
    """ASE-PVAD 单 keyframe 加性增强(论文 Eq.6)。

    对每 speaker i 的独占段(collect_clean_audio 避开 overlap)1s/0.2s 滑窗扫,
    算 cos(enroll_emb, frame_emb), 全局选最高帧 Eselected。
    Eq.5-6: aug = λ*enroll + (1-λ)*avg(aug, keyframe), L2 normalize, 迭代 n_iter。

    护栏: best_sim < min_best_sim → enroll 严重不可靠(sim 反向风险), 返回 (enroll, -1, sim) fallback。
    返回 (aug_emb, best_spk, best_sim)。best_spk=-1 表示 fallback(未增强)。
    """
    candidates = []  # (spk_idx, frame_emb_tensor, cos_float)
    win = int(sr * frame_sec)
    hop = max(1, int(sr * shift_sec))
    for i in range(len(per_spk)):
        clean = collect_clean_audio(audio, diar_mask, i, sr=sr)
        if clean is None or len(clean) < win:
            # fallback: 扫 timeline 整段(含 overlap, 论文原样)
            segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]] if per_spk[i] else []
            clean = np.concatenate(segs) if segs else None
            if clean is None or len(clean) < win:
                continue
        n_frames = max(1, (len(clean) - win) // hop + 1)
        for k in range(n_frames):
            start = k * hop
            seg = clean[start:start + win]
            if len(seg) < win:
                break
            fe = get_emb(seg)
            cos = float(torch.dot(enroll_emb, fe))
            candidates.append((i, fe, cos))
    if not candidates:
        return enroll_emb, -1, 0.0
    best_spk, best_frame, best_sim = max(candidates, key=lambda x: x[2])
    if best_sim < min_best_sim:
        return enroll_emb, -1, best_sim  # fallback(enroll 不可靠)
    aug = enroll_emb.clone()
    for _ in range(n_iter):
        avg = 0.5 * (aug + best_frame)
        aug = lam * enroll_emb + (1 - lam) * avg
        aug = torch.nn.functional.normalize(aug, dim=-1)
    return aug, best_spk, best_sim


def pool_cer(rs, key):
    """累计池 CER 近似: sum(cer_i * char_i)/sum(char_i), char_i 用 len(ref) 近似(诊断趋势够, 答辩数字用 CERMetric 精算)。"""
    total_err = sum(r[key] * len(r["ref"]) for r in rs)
    total_char = sum(len(r["ref"]) for r in rs)
    return total_err / total_char if total_char else 0.0


def main():
    ap = argparse.ArgumentParser(description="ASE-PVAD self-augmentation POC: ASE 机制真实救回率")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vanilla-model", default=resolve_model("VANILLA"))
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=700, help="主战场全量 668(>668 取全, 避抽样偏差)")
    ap.add_argument("--sim-min", type=float, default=0.2)
    ap.add_argument("--sim-max", type=float, default=0.4)
    ap.add_argument("--vanilla-full", default=os.path.join(_HERE, "exp_vanilla_full.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_ase_pvad_poc.json"))
    ap.add_argument("--lam", type=float, default=0.1, help="Eq.6 λ(论文 §4.3 最优 [0.05,0.1])")
    ap.add_argument("--n-iter", type=int, default=5, help="迭代次数(论文 5 次达 parity)")
    ap.add_argument("--frame-sec", type=float, default=1.0, help="keyframe 窗长(论文 1s)")
    ap.add_argument("--shift-sec", type=float, default=0.2, help="滑窗 shift(论文 0.2s)")
    ap.add_argument("--min-best-sim", type=float, default=0.15, help="护栏: best_sim 低于此 fallback(防 sim 反向放大错误)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    full = json.load(open(args.vanilla_full, encoding="utf-8"))
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    bucket = [d for d in full
              if d.get("max_sim") is not None and args.sim_min <= d["max_sim"] < args.sim_max
              and d.get("ref") and d["uid"] in uid2pair]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(bucket))[:args.n_sample]
    samples = [bucket[i] for i in sorted(idx)]
    print(f"[ASE-PVAD POC] 桶 {len(bucket)} 抽样 {len(samples)} (seed={args.seed})")
    print(f"[ASE 超参] λ={args.lam} n_iter={args.n_iter} frame={args.frame_sec}s shift={args.shift_sec}s guard={args.min_best_sim}")

    asr_model, tok, fe, diar = load_models(device, args.vanilla_model, args.diarization_model, dtype)
    get_emb = get_emb_factory(diar, device)

    results = []
    t_start = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        uid = d["uid"]; ref = d["ref"]
        enr, rec = pair["enrollment"], pair["recognition"]
        rec_t0 = time.time()

        w_enr, _ = librosa.load(enr, sr=16000)
        enroll_emb = get_emb(w_enr)

        audio, sr = librosa.load(rec, sr=16000)
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} diar-fail {type(e).__name__} → 跳过")
            results.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        n_spk = len(speakers)

        ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        audio_len = ifp.shape[-1] // 2
        diar_mask = get_diarization_mask(per_spk, audio_len)

        spk_embs = []
        for i in range(n_spk):
            seg = collect_clean_audio(audio, diar_mask, i, sr)
            if seg is None or len(seg) < sr * 0.3:
                segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
            min_len = sr * 1
            if len(seg) < min_len:
                seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
            spk_embs.append(get_emb(seg))

        sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
        argmax_idx = int(torch.argmax(sims))
        max_sim = float(sims[argmax_idx])

        # ---- ASE-PVAD 增强 ----
        aug_emb, ase_spk, best_sim = ase_augment_keyframe(
            audio, per_spk, diar_mask, enroll_emb, get_emb, sr=sr,
            frame_sec=args.frame_sec, shift_sec=args.shift_sec,
            lam=args.lam, n_iter=args.n_iter, min_best_sim=args.min_best_sim)
        sims_aug = torch.stack([torch.dot(aug_emb, e) for e in spk_embs])
        aug_argmax_idx = int(torch.argmax(sims_aug))
        ase_fallback = (ase_spk == -1)

        # ---- 转写每个 speaker → CER(复用 per_spk_cer, baseline/ASE/oracle 都从这里取) ----
        per_spk_cer = []
        per_spk_text = []
        for i in range(n_spk):
            tgt_audio = cut_target_timeline(audio, per_spk[i], sr=sr)
            txt = transcribe(asr_model, tok, fe, tgt_audio, device, dtype)
            c = cer_official(txt, ref)
            per_spk_text.append(txt)
            per_spk_cer.append(float(c))

        oracle_idx = int(np.argmin(per_spk_cer))
        oracle_sim = float(sims[oracle_idx])
        baseline_cer = per_spk_cer[argmax_idx]
        ase_cer = per_spk_cer[aug_argmax_idx]
        baseline_correct = (argmax_idx == oracle_idx)
        ase_correct = (aug_argmax_idx == oracle_idx)
        ase_changed = (aug_argmax_idx != argmax_idx)

        rec_dt = time.time() - rec_t0
        mark = ("ASE_RESCUE" if (not baseline_correct and ase_correct) else
                ("ASE_BREAK" if (baseline_correct and not ase_correct) else
                 ("ASE_FLIP" if ase_changed else "ASE_SAME")))
        print(f"  [{n+1}/{len(samples)}] {uid} spk={n_spk} base={baseline_cer:.2f} "
              f"ase={ase_cer:.2f} oracle={per_spk_cer[oracle_idx]:.2f} "
              f"best_sim={best_sim:.2f} {mark} ({rec_dt:.1f}s)")

        results.append({
            "uid": uid, "ref": ref, "n_spk": n_spk, "speakers": speakers,
            "sims": [round(float(s), 4) for s in sims],
            "argmax_idx": argmax_idx, "max_sim": round(max_sim, 4),
            "aug_argmax_idx": aug_argmax_idx, "ase_spk": ase_spk,
            "best_sim": round(best_sim, 4), "ase_fallback": ase_fallback,
            "oracle_idx": oracle_idx, "oracle_sim": round(oracle_sim, 4),
            "baseline_cer": round(baseline_cer, 4),
            "ase_cer": round(ase_cer, 4),
            "oracle_cer": round(per_spk_cer[oracle_idx], 4),
            "baseline_correct": baseline_correct, "ase_correct": ase_correct,
            "ase_changed": ase_changed,
            "baseline_text": per_spk_text[argmax_idx],
            "ase_text": per_spk_text[aug_argmax_idx],
            "oracle_text": per_spk_text[oracle_idx],
            "rec_sec": round(rec_dt, 2),
        })

    total_dt = time.time() - t_start
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    print(f"\n{'='*70}\n[ASE-PVAD POC 结论] 有效 {n_valid}/{len(samples)}, 总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本。"); return

    baseline_correct_rate = float(np.mean([r["baseline_correct"] for r in valid]))
    ase_correct_rate = float(np.mean([r["ase_correct"] for r in valid]))
    multi = [r for r in valid if r["n_spk"] >= 2]
    multi_baseline = float(np.mean([r["baseline_correct"] for r in multi])) if multi else None
    multi_ase = float(np.mean([r["ase_correct"] for r in multi])) if multi else None

    baseline_cer_pool = pool_cer(valid, "baseline_cer")
    ase_cer_pool = pool_cer(valid, "ase_cer")
    oracle_cer_pool = pool_cer(valid, "oracle_cer")

    baseline_miss = [r for r in valid if not r["baseline_correct"]]
    n_miss = len(baseline_miss)
    n_rescue = sum(1 for r in baseline_miss if r["ase_correct"])          # baseline 错 → ASE 对
    n_break = sum(1 for r in valid if r["baseline_correct"] and not r["ase_correct"])  # baseline 对 → ASE 错
    n_flip = sum(1 for r in valid if r["ase_changed"])
    n_fallback = sum(1 for r in valid if r["ase_fallback"])
    miss_recognizable = float(np.mean([r["oracle_sim"] >= 0.2 for r in baseline_miss])) if baseline_miss else None

    print(f"\n[三档对比(主战场 sim[0.2,0.4), n={n_valid})]")
    print(f"  {'档位':<20} {'选对率':<10} {'多spk选对':<12} {'累计池CER':<10}")
    print(f"  {'baseline argmax':<20} {baseline_correct_rate*100:<10.1f} {(multi_baseline*100 if multi_baseline is not None else 0):<12.1f} {baseline_cer_pool:<10.4f}")
    print(f"  {'ASE-PVAD aug':<20} {ase_correct_rate*100:<10.1f} {(multi_ase*100 if multi_ase is not None else 0):<12.1f} {ase_cer_pool:<10.4f}")
    print(f"  {'oracle(上限)':<20} {'100.0':<10} {'100.0':<12} {oracle_cer_pool:<10.4f}")
    print(f"\n[ASE 救回细分]")
    print(f"  baseline miss:           {n_miss}/{n_valid} ({n_miss/n_valid*100:.1f}%)")
    print(f"  ASE 救回(错→对):         {n_rescue}/{n_miss} ({(n_rescue/n_miss*100) if n_miss else 0:.1f}%)")
    print(f"  ASE 改错(对→错):         {n_break}/{n_valid}")
    print(f"  ASE 改了选择(flip):      {n_flip}/{n_valid}")
    print(f"  ASE fallback(护栏触发):  {n_fallback}/{n_valid}")
    print(f"  baseline miss oracle_sim≥0.2(理论可救面): {(miss_recognizable*100) if miss_recognizable is not None else 0:.1f}%")
    print(f"\n[CER 改善(累计池近似)]")
    print(f"  baseline→ASE:  {baseline_cer_pool:.4f} → {ase_cer_pool:.4f}  (Δ{(ase_cer_pool-baseline_cer_pool)*100:+.2f}%)")
    print(f"  ASE→oracle:    {ase_cer_pool:.4f} → {oracle_cer_pool:.4f}  (上限 Δ{(oracle_cer_pool-ase_cer_pool)*100:+.2f}%)")

    cer_gain_ase = baseline_cer_pool - ase_cer_pool  # 正=ASE 改善
    print(f"\n{'='*70}\n[决策]")
    if cer_gain_ase > 0.02 and n_rescue >= 5 and n_rescue > n_break * 2:
        print(f"  ✅ ASE 有效: CER 改善 {cer_gain_ase:+.4f}(累计池), 救回 {n_rescue} 条 miss(改错 {n_break})。")
        print(f"  → 值得迁移 qwen 后端估真实收益 + 集成 ASE-PVAD(改 enroll_infer get_enroll_emb)。")
    elif cer_gain_ase > 0:
        print(f"  🟡 ASE 微正: CER Δ{cer_gain_ase:+.4f}, 救回 {n_rescue}/改错 {n_break}。收益边际, 集成价值有限。")
    else:
        print(f"  ❌ ASE 无效/负: CER Δ{cer_gain_ase:+.4f}, 救回 {n_rescue}/改错 {n_break}。")
        print(f"  → 归档, 答辩诚实归因(复现出题方 ASE-PVAD 但 wespeaker+diar argmax 架构下收益不显著)。")

    summary = {
        "n_sample": len(samples), "n_valid": n_valid,
        "baseline_correct_rate": round(baseline_correct_rate, 4),
        "ase_correct_rate": round(ase_correct_rate, 4),
        "multi_baseline_correct": round(multi_baseline, 4) if multi_baseline is not None else None,
        "multi_ase_correct": round(multi_ase, 4) if multi_ase is not None else None,
        "baseline_cer_pool": round(baseline_cer_pool, 4),
        "ase_cer_pool": round(ase_cer_pool, 4),
        "oracle_cer_pool": round(oracle_cer_pool, 4),
        "cer_gain_ase": round(cer_gain_ase, 4),
        "n_miss": n_miss, "n_rescue": n_rescue, "n_break": n_break,
        "n_flip": n_flip, "n_fallback": n_fallback,
        "miss_recognizable_rate": round(miss_recognizable, 4) if miss_recognizable is not None else None,
        "ase_hyperparams": {"lam": args.lam, "n_iter": args.n_iter, "frame_sec": args.frame_sec,
                            "shift_sec": args.shift_sec, "min_best_sim": args.min_best_sim},
        "total_min": round(total_dt / 60, 2),
    }
    out = {"summary": summary, "results": results}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
