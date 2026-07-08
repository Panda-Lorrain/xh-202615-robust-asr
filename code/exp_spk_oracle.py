"""exp_spk_oracle.py — 攻 CER 切片死区(sim<0.2)的 oracle POC(go/no-go 门槛)。

问: 死区 pos(max_sim<0.2)是
  (A) diar+声纹 argmax 选错 target(切了非目标 timeline) → 声纹强化(CAM++/帧选择)能救 → GO=是
  (B) target 选对了但音频被 babble 摧毁(切对了但 mel 退化转写崩) → 声纹救不了 → GO=否, 转答辩诚实归因

方法: 对死区抽样, 转写【每个 speaker】的 timeline, oracle target = argmin CER(转写最接近 ref=真 target)。
  - argmax 选对率(argmax_idx==oracle_idx) 高 → 选对了, 音频摧毁 → GO=否
  - oracle_sim 普遍 >0.2(正确 target 声纹可识别)但 argmax 选了别的 → 选错 → GO=是
  - oracle_CER << argmax_CER → 切错是主因 → GO=是

复用 enroll_infer 的模型加载 + diar._embedding(wespeaker) + collect_clean_audio/cut_target_timeline,
不加载 DiCoW(死区 vanilla 路线, dicow 不需要)。

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_spk_oracle.py
产物: code/exp_spk_oracle.json + stdout 明确 go/no-go 判定。
"""
import os, sys, json, time, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

# 复用 enroll_infer 的 sys.path 设置(DiariZen/pyannote) + 纯函数 + repro
# enroll_infer 模块级只做 import/sys.path/函数定义(main 在 __main__ 守卫内, import 安全)
from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import cut_target_timeline, to_simplified, digit_postproc
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model, reset_peak_gpu


def load_models(device, vanilla_model, diar_model, dtype):
    """加载 vanilla Whisper + DiariZen diar(复用 enroll_infer 的加载逻辑, 不加载 DiCoW)。"""
    print(f"[load] vanilla Whisper {vanilla_model} on {device}")
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        vanilla_model, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(vanilla_model)
    fe = AutoFeatureExtractor.from_pretrained(vanilla_model)

    print(f"[load] DiariZen {diar_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(diar_model).to(device)
    return asr_model, tok, fe, diar


def get_emb_factory(diar, device):
    """复用 diar._embedding(wespeaker) 抽声纹, 与 enroll_infer.get_emb 完全一致。"""
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


def transcribe(asr_model, tok, fe, audio_np, device, dtype, language="zh"):
    """vanilla Whisper 转写一段 audio(已切好 target timeline), 返回 submit 归一后的 text。"""
    ifp = fe(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
    with torch.no_grad():
        out = asr_model.generate(input_features=ifp, attention_mask=am,
                                 language=language, task="transcribe", max_new_tokens=200)
    seqs = out["sequences"] if isinstance(out, dict) else out
    text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
    text = to_simplified(text)
    text = digit_postproc(text)
    return text


def main():
    ap = argparse.ArgumentParser(description="死区 oracle POC: argmax 选错 vs 音频摧毁")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vanilla-model", default=resolve_model("VANILLA"))
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=60, help="死区抽样条数(控制总时间)")
    ap.add_argument("--sim-max", type=float, default=0.2, help="死区上界 max_sim<sim_max")
    ap.add_argument("--vanilla-full", default=os.path.join(_HERE, "exp_vanilla_full.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_spk_oracle.json"))
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # ---- 1. 载入 vanilla_full + pairs, 筛死区, 抽样 ----
    full = json.load(open(args.vanilla_full, encoding="utf-8"))
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    deadzone = [d for d in full
                if d.get("max_sim") is not None and d["max_sim"] < args.sim_max
                and d.get("ref") and d["uid"] in uid2pair]
    print(f"[data] vanilla_full={len(full)} 死区(max_sim<{args.sim_max}, 有ref)={len(deadzone)}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(deadzone))[:args.n_sample]
    samples = [deadzone[i] for i in sorted(idx)]
    print(f"[data] 抽样 {len(samples)} 条(seed={args.seed})")

    # ---- 2. 加载模型 ----
    asr_model, tok, fe, diar = load_models(device, args.vanilla_model,
                                           args.diarization_model, dtype)
    get_emb = get_emb_factory(diar, device)

    # ---- 3. 主循环 ----
    results = []
    t_start = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        enr, rec = pair["enrollment"], pair["recognition"]
        ref = d["ref"]
        uid = d["uid"]
        rec_t0 = time.time()

        # enrollment emb
        w_enr, _ = librosa.load(enr, sr=16000)
        enroll_emb = get_emb(w_enr)

        # diar
        audio, sr = librosa.load(rec, sr=16000)
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} diar-fail {type(e).__name__} → 跳过")
            results.append({"uid": uid, "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        n_spk = len(speakers)

        # diar_mask + 各 speaker 声纹(完全复刻 enroll_infer:222-231)
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

        # 转写每个 speaker → CER
        per_spk_cer = []
        per_spk_text = []
        for i in range(n_spk):
            tgt_audio = cut_target_timeline(audio, per_spk[i], sr=sr)
            txt = transcribe(asr_model, tok, fe, tgt_audio, device, dtype)
            c = cer_official(txt, ref)
            per_spk_text.append(txt)
            per_spk_cer.append(float(c))

        oracle_idx = int(np.argmin(per_spk_cer))
        oracle_cer = per_spk_cer[oracle_idx]
        oracle_sim = float(sims[oracle_idx])
        argmax_cer = per_spk_cer[argmax_idx]
        # argmax 选对 = argmax 的转写就是最优(与 oracle 同一 speaker)
        argmax_correct = (argmax_idx == oracle_idx)

        rec_dt = time.time() - rec_t0
        mark = "ARGMAX_OK" if argmax_correct else "ARGMAX_MISS"
        print(f"  [{n+1}/{len(samples)}] {uid} spk={n_spk} max_sim={max_sim:.3f} "
              f"argmax_CER={argmax_cer:.3f} oracle_CER={oracle_cer:.3f} "
              f"oracle_sim={oracle_sim:.3f} {mark} ({rec_dt:.1f}s)")
        if not argmax_correct:
            print(f"      argmax#{argmax_idx}(sim{max_sim:.2f}):\"{per_spk_text[argmax_idx][:30]}\" "
                  f"vs oracle#{oracle_idx}(sim{oracle_sim:.2f}):\"{per_spk_text[oracle_idx][:30]}\"")

        results.append({
            "uid": uid, "ref": ref,
            "n_spk": n_spk,
            "speakers": speakers,
            "sims": [round(float(s), 4) for s in sims],
            "argmax_idx": argmax_idx, "max_sim": round(max_sim, 4),
            "oracle_idx": oracle_idx, "oracle_sim": round(oracle_sim, 4),
            "argmax_cer": round(argmax_cer, 4), "oracle_cer": round(oracle_cer, 4),
            "argmax_correct": argmax_correct,
            "per_spk_cer": [round(c, 4) for c in per_spk_cer],
            "per_spk_text": per_spk_text,
            "vanilla_full_max_sim": d["max_sim"],
            "vanilla_full_cer": d.get("vanilla_cer"),
            "vanilla_full_text": d.get("vanilla_text"),
            "rec_sec": round(rec_dt, 2),
        })

    total_dt = time.time() - t_start

    # ---- 4. 统计 + go/no-go 判定 ----
    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    print(f"\n{'='*70}\n[oracle POC 结论] 有效 {n_valid}/{len(samples)} 条, 总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本, 无法判定。")
        return

    argmax_correct_rate = np.mean([r["argmax_correct"] for r in valid])
    argmax_cer_mean = np.mean([r["argmax_cer"] for r in valid])
    oracle_cer_mean = np.mean([r["oracle_cer"] for r in valid])
    cer_gain = argmax_cer_mean - oracle_cer_mean

    # oracle_sim 分桶(正确 target 声纹可识别性)
    oracle_sims = np.array([r["oracle_sim"] for r in valid])
    sim_buckets = {
        "oracle_sim<0.1": float(np.mean(oracle_sims < 0.1)),
        "0.1<=sim<0.2": float(np.mean((oracle_sims >= 0.1) & (oracle_sims < 0.2))),
        "0.2<=sim<0.3": float(np.mean((oracle_sims >= 0.2) & (oracle_sims < 0.3))),
        "oracle_sim>=0.3": float(np.mean(oracle_sims >= 0.3)),
    }
    oracle_sim_mean = float(np.mean(oracle_sims))

    # miss 子集(argmax 选错)细看: 这些条里 oracle 是否声纹可识别 + CER 差距
    miss = [r for r in valid if not r["argmax_correct"]]
    n_miss = len(miss)
    miss_oracle_sim = np.mean([r["oracle_sim"] for r in miss]) if miss else float("nan")
    miss_argmax_cer = np.mean([r["argmax_cer"] for r in miss]) if miss else float("nan")
    miss_oracle_cer = np.mean([r["oracle_cer"] for r in miss]) if miss else float("nan")
    # miss 中 oracle_sim>=0.2(正确 target 本可被声纹识别)占比 → 声纹强化直接收益面
    miss_oracle_recognizable = (np.mean([r["oracle_sim"] >= 0.2 for r in miss]) if miss else float("nan"))

    # 单 speaker 样本(argmax 必然==oracle, 无信息)单独看
    multi = [r for r in valid if r["n_spk"] >= 2]
    n_multi = len(multi)
    multi_correct = np.mean([r["argmax_correct"] for r in multi]) if multi else float("nan")

    print(f"\n[核心指标]")
    print(f"  argmax 选对率(死区):           {argmax_correct_rate*100:.1f}% ({sum(r['argmax_correct'] for r in valid)}/{n_valid})")
    print(f"    ↳ 多 speaker(≥2)子集选对率:  {multi_correct*100:.1f}% ({sum(r['argmax_correct'] for r in multi)}/{n_multi})"
          if multi else "    ↳ 无多 speaker 样本")
    print(f"  argmax_CER 均值(当前系统):     {argmax_cer_mean:.3f}")
    print(f"  oracle_CER 均值(完美选 target):{oracle_cer_mean:.3f}")
    print(f"  CER 收益(argmax→oracle):       {cer_gain:+.3f}  ({cer_gain/argmax_cer_mean*100 if argmax_cer_mean>0 else 0:+.1f}%)")
    print(f"\n[oracle_sim 分布(正确 target 的声纹相似度)]")
    print(f"  oracle_sim 均值: {oracle_sim_mean:.3f}")
    for k, v in sim_buckets.items():
        print(f"    {k}: {v*100:.1f}%")
    print(f"\n[miss 子集(argmax 选错, n={n_miss})]")
    print(f"  miss_argmax_CER 均值:  {miss_argmax_cer:.3f}")
    print(f"  miss_oracle_CER 均值:  {miss_oracle_cer:.3f}")
    print(f"  miss_oracle_sim 均值:  {miss_oracle_sim:.3f}")
    print(f"  miss 中 oracle_sim≥0.2(声纹强化可救): {miss_oracle_recognizable*100:.1f}%")

    # ---- go/no-go 判定 ----
    print(f"\n{'='*70}\n[go/no-go 判定]")
    # 判据1: argmax 选对率高 → 音频摧毁, 声纹救不了
    # 判据2: miss 中 oracle 声纹可识别(≥0.2) 且 oracle_CER 显著低于 argmax_CER → 声纹强化有价值
    gainable = (cer_gain > 0.10) and (n_miss >= 3) and (miss_oracle_recognizable > 0.3)
    if argmax_correct_rate >= 0.5 and not gainable:
        verdict = "GO=否"
        reason = (f"argmax 选对率 {argmax_correct_rate*100:.0f}% ≥50% → 死区主因是【音频被 babble 摧毁】"
                  f"(target 切对了但 mel 退化转写崩), 非选错 target。声纹强化(CAM++/帧选择)救不了。"
                  f"oracle_CER 仍 {oracle_cer_mean:.2f}(完美选 target 也这么差)坐实音频摧毁。"
                  f"→ 转答辩诚实归因 + babble 专用源分离(SepFormer 提 mel)方向。")
    elif gainable:
        verdict = "GO=是"
        reason = (f"argmax 选错 {n_miss}/{n_valid} 条, 其中 {miss_oracle_recognizable*100:.0f}% 的正确 target "
                  f"声纹 sim≥0.2(本可识别) 但 argmax 选了别的; 完美选 target CER {argmax_cer_mean:.2f}→"
                  f"{oracle_cer_mean:.2f}(Δ{cer_gain:+.2f})。→ 投声纹强化让 argmax 选对能救。")
    else:
        # 混合: 选对率不高但 oracle_CER 也差 → 音频摧毁为主
        verdict = "GO=否(偏)"
        reason = (f"argmax 选对率 {argmax_correct_rate*100:.0f}%, 但 oracle_CER {oracle_cer_mean:.2f} 仍高 "
                  f"(完美选 target 也救不回), 死区以音频摧毁为主。声纹强化收益有限。"
                  f"miss 中 oracle 可识别比例 {miss_oracle_recognizable*100:.0f}%。")

    print(f"  判定: {verdict}")
    print(f"  理由: {reason}")

    # 推荐
    if "GO=是" in verdict:
        print(f"\n[推荐优先投](基于 miss 子集特征)")
        if miss_oracle_recognizable >= 0.5:
            print(f"  1️⃣ 帧选择 overlap-aware 修复(修 collect_clean_audio 回退污染): "
                  f"miss oracle_sim≥0.2 占 {miss_oracle_recognizable*100:.0f}%, 说明正确 target 声纹"
                  f"本可识别, argmax 选错多因独占帧不足回退整条 timeline 污染 → 帧选择修复最直接。")
            print(f"  2️⃣ CAM++ per-speaker 替换 wespeaker(更强声纹, 提升可识别边界)。")
            print(f"  3️⃣ US-PVAD(短参考优化, 但 enrollment~1.8s 已不算超短, 收益面窄)。")
        else:
            print(f"  1️⃣ CAM++ per-speaker(3D-Speaker, 192d, 更强声纹): miss oracle_sim 偏低"
                  f"(均值 {miss_oracle_sim:.2f}), 需更强声纹拉开 target/非 target 边界。")
            print(f"  2️⃣ 帧选择 overlap-aware 修复(减少声纹提取污染)。")
    else:
        print(f"\n[推荐方向](GO=否 → 不投声纹强化)")
        print(f"  死区是音频摧毁(target 对了但 mel 退化), 声纹强化破不了。")
        print(f"  → babble 专用源分离(SepFormer 提 target mel, 高成本) 或 答辩诚实归因(架构极限)。")

    summary = {
        "verdict": verdict, "reason": reason,
        "n_sample": len(samples), "n_valid": n_valid,
        "argmax_correct_rate": round(argmax_correct_rate, 4),
        "multi_spk_correct_rate": round(multi_correct, 4) if multi else None,
        "argmax_cer_mean": round(argmax_cer_mean, 4),
        "oracle_cer_mean": round(oracle_cer_mean, 4),
        "cer_gain": round(cer_gain, 4),
        "oracle_sim_mean": round(oracle_sim_mean, 4),
        "oracle_sim_buckets": {k: round(v, 4) for k, v in sim_buckets.items()},
        "n_miss": n_miss, "miss_oracle_sim_mean": round(miss_oracle_sim, 4),
        "miss_argmax_cer_mean": round(miss_argmax_cer, 4),
        "miss_oracle_cer_mean": round(miss_oracle_cer, 4),
        "miss_oracle_recognizable_rate": round(miss_oracle_recognizable, 4),
        "total_min": round(total_dt / 60, 2),
    }

    out = {"summary": summary, "results": results}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {args.out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
