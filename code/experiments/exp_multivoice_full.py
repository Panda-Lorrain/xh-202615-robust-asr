"""exp_multivoice_full.py — multi-voice 内容判别选路 全量泛化验证 (主战场260条).

【背景】multi-voice 内容判别 POC 在**失败组40条**已突破:
  heuristic 选路 (content_gate + 设备词/动作词/品牌锚点 + news 黑名单)
  mean CER **0.654** / 选对率 87.5%, 逼近 oracle 0.603, 碾压 sim1.249 / argmax1.216.
  产物: code/runs/_multivoice_route/summary.json

【本任务】验证全量泛化 — 在主战场 (sim≥0.4 & qwen_cer<0.8, 非失败组) 跑 multi-voice:
  1. SepFormer 分离 recognition → sourceA/sourceB
  2. 两路 qwen 批量转写
  3. heuristic 选路 (复用 exp_multivoice_route.route_heuristic)
  4. CER + 选对率 + RTF
  5. 合并失败组40条结果加权外推整体 CER, 对比主线 (mean 0.3848 / 池 0.3436)

【判别】
  主战场 heuristic mean CER 显著低于 argmax (Δ<-0.05) → 内容判别泛化有效
  选对率 ≥80% (近失败组 87.5%) → 启发式在主战场仍鲁棒
  整体外推 CER 降 ≥ 0.02 → 集成值得 (考虑 RTF 代价)

【RTF】
  SepFormer 分离总耗时 + 两路 qwen 总耗时 / 总音频时长 = multi-voice 全链路 RTF
  对比主线 RTF ~0.24 @4060 (qwen 单路+diar)
  L20 外推 (4060×1.5~2)

【边界】
  - SepFormer 失败条跳过记录, 不阻塞
  - n_src != 2 (理论上 SepFormer-whamr16k 恒输出 2 路) 视为失败
  - heuristic 逻辑完全复刻 exp_multivoice_route.route_heuristic (已验证)

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_multivoice_full.py
产物: code/runs/_multivoice_full/{slices/, _uid2text.json, summary.json}
"""
import os, sys, json, time, argparse, subprocess, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容 (复刻 exp_sepformer_qwen.py 的 SB 1.1.0 LazyModule inspect-guard) ----
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

# 复用 exp_sepformer_qwen 的 load_sepformer / separate / load_diar / get_emb_factory (经过验证)
from exp_sepformer_qwen import load_sepformer, separate, load_diar, get_emb_factory
# 复用 POC 验证过的 heuristic 选路 (策略3) + cmd_score
from exp_multivoice_route import route_heuristic, route_content_gate, cmd_score
from text_utils import to_simplified, digit_postproc, brand_homophone_fix
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))

# 失败组 (sim≥0.4 & qwen_cer>0.8) 已有产物 (复用, 不重跑)
FAIL_SUMMARY = os.path.join(_HERE, "runs", "_multivoice_route", "summary.json")
# 失败组 B2 SepFormer 两路转写原始结果 (取 per_src / oracle)
FAIL_B2 = os.path.join(_HERE, "runs", "_sepformer_b2", "summary.json")


def cer_normalized(text, ref):
    """官方口径 CER + 同 enroll_infer 归一链 (繁简→数字→品牌同音)。"""
    t = brand_homophone_fix(digit_postproc(to_simplified(text)))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref)))
    return float(cer_official(t, r))


def run_qwen_batch_timed(slice_dir, out_json, batch_size=16):
    """subprocess 调 qwen backend 批量转写, 返回 (uid2text, 耗时秒)。"""
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--seed", "42",
           "--batch-size", str(batch_size)]
    print(f"[qwen] subprocess 转写 {slice_dir} ({len(cmd)} args)")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    uid2text = json.load(open(out_json, encoding="utf-8"))
    return uid2text, dt


def main():
    ap = argparse.ArgumentParser(description="multi-voice 内容判别选路 全量泛化验证 (主战场)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=260)
    ap.add_argument("--qwen-full", default=os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs", "_multivoice_full"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    slice_dir = os.path.join(args.out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(slice_dir, "*.wav")):
        os.remove(f)

    # ---- 1. 数据: 主战场 (sim≥0.4 & qwen_cer<0.8) ----
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows = qfull["rows"] if isinstance(qfull, dict) and "rows" in qfull else qfull
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    mainfield_pool = [r for r in rows
                      if r.get("sim") is not None and r["sim"] >= 0.4
                      and r.get("qwen_cer") is not None and r["qwen_cer"] < 0.8
                      and r["uid"] in uid2pair and r.get("ref")]
    fail_pool = [r for r in rows
                 if r.get("sim") is not None and r["sim"] >= 0.4
                 and r.get("qwen_cer") is not None and r["qwen_cer"] > 0.8]
    deadzone_pool = [r for r in rows if r.get("sim") is not None and r["sim"] < 0.4]
    total = len(rows)
    print(f"[data] qwen_full rows={total}")
    print(f"[data] 主战场池 (sim≥0.4 & cer<0.8): {len(mainfield_pool)} ({len(mainfield_pool)/total*100:.1f}%)")
    print(f"[data] 失败组池 (sim≥0.4 & cer>0.8): {len(fail_pool)} ({len(fail_pool)/total*100:.1f}%)")
    print(f"[data] 死区池   (sim<0.4):          {len(deadzone_pool)} ({len(deadzone_pool)/total*100:.1f}%)")

    rng = np.random.default_rng(args.seed)
    perm = sorted(rng.permutation(len(mainfield_pool))[:args.n_sample])
    samples = [mainfield_pool[i] for i in perm]
    print(f"[data] 抽样 {len(samples)} 条 (seed={args.seed})")

    # ---- 2. load SepFormer + DiariZen ----
    print(f"[load] SepFormer whamr16k → {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. Phase1: SepFormer 分离 → 两路都存 (不预设 target, 让 heuristic 选) ----
    meta = []
    slice_uids_all = []
    total_audio_sec = 0.0
    n_sep_fail = 0
    sep_t0 = time.time()
    for n, d in enumerate(samples):
        pair = uid2pair[d["uid"]]
        enr, rec, uid, ref = pair["enrollment"], pair["recognition"], d["uid"], d["ref"]
        try:
            audio, sr = librosa.load(rec, sr=16000)
            total_audio_sec += len(audio) / sr
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]
            if n_src != 2:
                # heuristic 假设 2 路; 非 2 路记 fail 跳过
                raise RuntimeError(f"SepFormer n_src={n_src} 非 2 路, heuristic 选路仅支持 2 路")

            # 诊断: sim 选 target (仅供对照, 不作主选路)
            embs = []
            for i in range(n_src):
                seg = sources[i]
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in embs])
            sim_pick_idx = int(torch.argmax(sims))

            # 两路都存 srcA / srcB (不按 sim 排序, 让 heuristic 看 text 决定)
            slice_uids = []
            for i in range(n_src):
                suid = f"{uid}__src{chr(65 + i)}"  # cmd_0__srcA, cmd_0__srcB
                sf.write(os.path.join(slice_dir, suid + ".wav"),
                         np.ascontiguousarray(sources[i].astype(np.float32)), 16000)
                slice_uids_all.append(suid)
                slice_uids.append(suid)

            meta.append({
                "uid": uid, "ref": ref, "n_src": n_src,
                "slice_uids": slice_uids,
                "sep_sims": [round(float(s), 4) for s in sims],
                "sim_pick_idx": sim_pick_idx,
                "argmax_qwen_cer": d.get("qwen_cer"), "sim": d.get("sim"),
                "audio_sec": round(len(audio) / sr, 2),
            })
            if (n + 1) % 20 == 0 or n == len(samples) - 1:
                print(f"  [{n+1}/{len(samples)}] sep done ({time.time()-sep_t0:.0f}s, "
                      f"audio={total_audio_sec:.0f}s)")
        except Exception as e:
            n_sep_fail += 1
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    sep_total_sec = time.time() - sep_t0
    print(f"\n[sep] 分离完成: {len(slice_uids_all)} 路, 耗时 {sep_total_sec:.0f}s "
          f"({sep_total_sec/60:.1f}min), 失败 {n_sep_fail} 条")
    print(f"[sep] 总音频时长 {total_audio_sec:.0f}s ({total_audio_sec/60:.1f}min), "
          f"分离 RTF={sep_total_sec/total_audio_sec:.3f}")

    # ---- 4. Phase2: 批量 qwen 转写两路 ----
    print(f"\n[qwen] 转写 {len(slice_uids_all)} 路 (两路全转)...")
    qwen_out_path = os.path.join(args.out_dir, "_uid2text.json")
    uid2text, qwen_total_sec = run_qwen_batch_timed(slice_dir, qwen_out_path, args.batch_size)
    print(f"[qwen] 转写完成: {len(uid2text)} 条, 耗时 {qwen_total_sec:.0f}s ({qwen_total_sec/60:.1f}min)")
    print(f"[qwen] 转写 RTF={qwen_total_sec/total_audio_sec:.3f} (两路合计)")

    # ---- 5. Phase3: 选路 + CER ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        per_src = []
        for i, suid in enumerate(m["slice_uids"]):
            t = uid2text.get(suid, "")
            c = cer_normalized(t, m["ref"])
            per_src.append({"src_idx": i, "slice_uid": suid, "text": t, "cer": round(c, 4)})
        oracle = min(per_src, key=lambda x: x["cer"])

        # heuristic 选路 (复刻 POC 验证逻辑)
        per_src_for_route = [{"text": s["text"]} for s in per_src]
        h_idx, h_reason = route_heuristic(per_src_for_route)
        h_cer = per_src[h_idx]["cer"]
        h_correct = (h_idx == oracle["src_idx"])

        # 诊断: content_gate 二值选路
        cg_idx, cg_reason = route_content_gate(per_src_for_route)
        cg_cer = per_src[cg_idx]["cer"]
        cg_correct = (cg_idx == oracle["src_idx"])

        # 诊断: sim 选路 (SepFormer 分离 + embedding 相似度挑 target)
        sim_idx = m["sim_pick_idx"]
        sim_cer = per_src[sim_idx]["cer"]
        sim_correct = (sim_idx == oracle["src_idx"])

        results.append({
            **m,
            "per_src": per_src,
            "oracle_idx": oracle["src_idx"], "oracle_cer": oracle["cer"],
            "heuristic_idx": h_idx, "heuristic_cer": h_cer,
            "heuristic_correct": h_correct, "heuristic_reason": h_reason,
            "heuristic_scores": [round(cmd_score(s["text"]), 2) for s in per_src],
            "content_gate_idx": cg_idx, "content_gate_cer": cg_cer,
            "content_gate_correct": cg_correct,
            "sim_route_idx": sim_idx, "sim_route_cer": sim_cer,
            "sim_route_correct": sim_correct,
            "argmax_cer": m["argmax_qwen_cer"],
        })

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = sep_total_sec + qwen_total_sec
    print(f"\n{'='*70}\n[multi-voice 主战场验证] 有效 {n_valid}/{len(samples)}, "
          f"总耗时 {total_dt/60:.1f}min")
    if n_valid == 0:
        print("无有效样本, 无法判定。")
        return

    # ---- 6. 汇总指标 ----
    h_cers = np.array([r["heuristic_cer"] for r in valid])
    orc_cers = np.array([r["oracle_cer"] for r in valid])
    cg_cers = np.array([r["content_gate_cer"] for r in valid])
    sim_cers = np.array([r["sim_route_cer"] for r in valid])
    argmax_paired = [r for r in valid if r.get("argmax_cer") is not None]
    argmax_cers = np.array([r["argmax_cer"] for r in argmax_paired])
    h_paired = np.array([r["heuristic_cer"] for r in argmax_paired])

    def _stats(cers):
        return {
            "mean": round(float(np.mean(cers)), 4),
            "median": round(float(np.median(cers)), 4),
            "correct_rate": round(float(np.mean(cers < 0.5)), 4),
            "n": len(cers),
        }

    h_accuracy = float(np.mean([r["heuristic_correct"] for r in valid]))
    cg_accuracy = float(np.mean([r["content_gate_correct"] for r in valid]))
    sim_accuracy = float(np.mean([r["sim_route_correct"] for r in valid]))

    mainfield_stats = {
        "n": n_valid,
        "heuristic": _stats(h_cers),
        "oracle": _stats(orc_cers),
        "content_gate": _stats(cg_cers),
        "sim_route": _stats(sim_cers),
        "argmax_qwen_paired": _stats(argmax_cers) if len(argmax_cers) else None,
        "heuristic_accuracy_vs_oracle": round(h_accuracy, 4),
        "content_gate_accuracy_vs_oracle": round(cg_accuracy, 4),
        "sim_route_accuracy_vs_oracle": round(sim_accuracy, 4),
        "delta_heuristic_vs_argmax": (round(float(np.mean(h_paired - argmax_cers)), 4)
                                      if len(argmax_cers) == len(h_paired) else None),
    }

    # 选错案例样本 (heuristic 选错即 != oracle)
    h_wrong = [r for r in valid if not r["heuristic_correct"]]

    # ---- 7. 外推整体 ----
    # 三分桶: 死区 (sim<0.4) / 主战场 (sim≥0.4 & cer<0.8) / 失败组 (sim≥0.4 & cer>0.8)
    n_dead = len(deadzone_pool)
    n_main = len(mainfield_pool)
    n_fail = len(fail_pool)
    # 死区不上 multi-voice (重 sim 区, 分离反而恶化); 用主线 argmax
    deadzone_cers = [r["qwen_cer"] for r in deadzone_pool if r.get("qwen_cer") is not None]
    deadzone_mean = float(np.mean(deadzone_cers)) if deadzone_cers else 0.0

    # 失败组: 复用 _multivoice_route/summary.json 的 heuristic mean CER
    fail_heur_mean = None
    fail_oracle_mean = None
    fail_argmax_mean = None
    fail_n = None
    if os.path.exists(FAIL_SUMMARY):
        fs = json.load(open(FAIL_SUMMARY, encoding="utf-8"))
        if "strategies" in fs and "heuristic" in fs["strategies"]:
            fail_heur_mean = fs["strategies"]["heuristic"]["mean_cer"]
            fail_n = fs["strategies"]["heuristic"]["n"]
        if "strategies" in fs and "content_gate" in fs["strategies"]:
            fail_cg_mean = fs["strategies"]["content_gate"]["mean_cer"]
        if "baselines" in fs:
            fail_argmax_mean = fs["baselines"].get("argmax_main")
            fail_oracle_mean = fs["baselines"].get("oracle_B2")

    # 外推整体 (按真实分桶比例加权)
    overall_mainfield_heur = mainfield_stats["heuristic"]["mean"]
    overall = (
        n_dead / total * deadzone_mean +
        n_main / total * overall_mainfield_heur +
        n_fail / total * (fail_heur_mean if fail_heur_mean is not None else 0.0)
    )

    # 主线整体对照
    main_line_mean = float(np.mean([r["qwen_cer"] for r in rows if r.get("qwen_cer") is not None]))
    main_line_pool_cer = 0.3436  # 池口径 (qwen_official, SSOT)

    # multi-voice 全链路 RTF
    rtf_sep = sep_total_sec / total_audio_sec if total_audio_sec else 0
    rtf_qwen = qwen_total_sec / total_audio_sec if total_audio_sec else 0
    rtf_total = (sep_total_sec + qwen_total_sec) / total_audio_sec if total_audio_sec else 0
    main_line_rtf_4060 = 0.24  # 主线 qwen 单路+diar @4060 (引用)
    # L20 外推: 4060 AD107, L20 AD102 同代 sm_89, 速度×1.5~2 (efficiency-portability-audit)
    l20_speedup_low, l20_speedup_high = 1.5, 2.0
    rtf_l20_low = rtf_total / l20_speedup_high
    rtf_l20_high = rtf_total / l20_speedup_low

    # ---- 8. 打印核心 ----
    print(f"\n{'='*70}\n[主战场260条 multi-voice 结果]")
    print(f"  {'策略':<14} {'mean CER':>10} {'median':>10} {'correct<0.5':>14}")
    for name, st in [("heuristic", mainfield_stats["heuristic"]),
                     ("oracle", mainfield_stats["oracle"]),
                     ("content_gate", mainfield_stats["content_gate"]),
                     ("sim_route", mainfield_stats["sim_route"])]:
        print(f"  {name:<14} {st['mean']:>10.4f} {st['median']:>10.4f} "
              f"{st['correct_rate']*100:>13.1f}%")
    if mainfield_stats["argmax_qwen_paired"]:
        a = mainfield_stats["argmax_qwen_paired"]
        print(f"  {'argmax主线':<14} {a['mean']:>10.4f} {a['median']:>10.4f} "
              f"{a['correct_rate']*100:>13.1f}%  (paired n={a['n']})")

    print(f"\n[选对率 (vs oracle)]")
    print(f"  heuristic    : {h_accuracy*100:.1f}% ({int(h_accuracy*n_valid)}/{n_valid})")
    print(f"  content_gate : {cg_accuracy*100:.1f}% ({int(cg_accuracy*n_valid)}/{n_valid})")
    print(f"  sim_route    : {sim_accuracy*100:.1f}% ({int(sim_accuracy*n_valid)}/{n_valid})")
    print(f"  Δ(heuristic - argmax) mean CER: {mainfield_stats['delta_heuristic_vs_argmax']}")

    print(f"\n[选错案例] heuristic 选错 {len(h_wrong)} 条 / {n_valid}")
    for r in h_wrong[:8]:
        orc = r["per_src"][r["oracle_idx"]]
        h_p = r["per_src"][r["heuristic_idx"]]
        print(f"  {r['uid']} ref={r['ref'][:18]}.. "
              f"oracle[src{r['oracle_idx']}]cer={orc['cer']:.2f} '{orc['text'][:20]}' "
              f"<- heu[src{r['heuristic_idx']}]cer={h_p['cer']:.2f} '{h_p['text'][:20]}' "
              f"scores={r['heuristic_scores']} reason={r['heuristic_reason']}")

    print(f"\n{'='*70}\n[整体外推]")
    print(f"  分桶占比 (n={total}): 死区 {n_dead}({n_dead/total*100:.1f}%) / "
          f"主战场 {n_main}({n_main/total*100:.1f}%) / 失败组 {n_fail}({n_fail/total*100:.1f}%)")
    print(f"  死区 mean CER (主线 argmax): {deadzone_mean:.4f}")
    print(f"  主战场 heuristic mean CER:  {overall_mainfield_heur:.4f}")
    print(f"  失败组 heuristic mean CER:  {fail_heur_mean} (复用 _multivoice_route)")
    print(f"  → 外推整体 multi-voice mean CER: {overall:.4f}")
    print(f"  主线 mean CER (逐条 argmax):    {main_line_mean:.4f}")
    print(f"  主线 mean CER (官方池):         {main_line_pool_cer}")
    print(f"  Δ(外推 - 主线逐条): {overall - main_line_mean:+.4f}")

    print(f"\n[RTF 成本]")
    print(f"  SepFormer 分离: {sep_total_sec:.0f}s  RTF={rtf_sep:.3f}")
    print(f"  两路 qwen 转写: {qwen_total_sec:.0f}s  RTF={rtf_qwen:.3f} (两路合计)")
    print(f"  multi-voice 全链路 RTF: {rtf_total:.3f}  vs 主线 ~{main_line_rtf_4060}")
    print(f"  L20 外推 (×{l20_speedup_low}-{l20_speedup_high}): RTF={rtf_l20_low:.3f}-{rtf_l20_high:.3f}")

    # 判定
    delta_argmax = mainfield_stats["delta_heuristic_vs_argmax"]
    overall_delta = overall - main_line_mean
    argmax_mean_paired = (mainfield_stats["argmax_qwen_paired"]["mean"]
                          if mainfield_stats["argmax_qwen_paired"] else None)
    if delta_argmax is not None and delta_argmax < -0.05 and h_accuracy >= 0.80:
        verdict_mainfield = "GO=主战场泛化有效"
        reason_mainfield = (f"主战场 heuristic mean CER {overall_mainfield_heur:.4f} "
                            f"显著低于 argmax {argmax_mean_paired:.4f} "
                            f"(Δ{delta_argmax:+.3f}), 选对率 {h_accuracy*100:.1f}% ≥80%, "
                            f"启发式在主战场仍鲁棒。")
    elif delta_argmax is not None and delta_argmax < 0:
        verdict_mainfield = "GO=偏是 (收益边际)"
        reason_mainfield = (f"Δ{delta_argmax:+.3f} 为负但弱 (>-0.05), 主战场泛化可行但收益边际; "
                            f"看整体外推 CER Δ{overall_delta:+.4f} 是否值得 RTF 代价。")
    else:
        verdict_mainfield = "GO=否 (主战场退化)"
        reason_mainfield = (f"主战场 heuristic Δ={delta_argmax} 不显著或正, "
                            f"multi-voice 在主战场退化; 不集成。")
    print(f"\n[判定] {verdict_mainfield}")
    print(f"  {reason_mainfield}")

    # ---- 9. 写出 ----
    summary_out = {
        "verdict_mainfield": verdict_mainfield,
        "reason_mainfield": reason_mainfield,
        "model": "speechbrain/sepformer-whamr16k + Qwen3-ASR-1.7B + heuristic route (exp_multivoice_route)",
        "seed": args.seed,
        "n_sample": len(samples),
        "n_valid": n_valid,
        "n_sep_fail": n_sep_fail,
        "total_audio_sec": round(total_audio_sec, 1),
        "total_compute_sec": round(total_dt, 1),
        "mainfield_stats": mainfield_stats,
        "h_wrong_examples": [{
            "uid": r["uid"], "ref": r["ref"],
            "oracle_text": r["per_src"][r["oracle_idx"]]["text"],
            "oracle_cer": r["per_src"][r["oracle_idx"]]["cer"],
            "heuristic_text": r["per_src"][r["heuristic_idx"]]["text"],
            "heuristic_cer": r["per_src"][r["heuristic_idx"]]["cer"],
            "scores": r["heuristic_scores"], "reason": r["heuristic_reason"],
            "texts": [s["text"] for s in r["per_src"]],
            "cers": [s["cer"] for s in r["per_src"]],
        } for r in h_wrong[:30]],
        "n_h_wrong": len(h_wrong),
        "extrapolation": {
            "buckets": {
                "deadzone": {"n": n_dead, "share": round(n_dead/total, 4),
                             "mean_cer_main": round(deadzone_mean, 4),
                             "use": "主线 argmax (sim<0.4 不上 multi-voice)"},
                "mainfield": {"n": n_main, "share": round(n_main/total, 4),
                              "mean_cer_heuristic": overall_mainfield_heur,
                              "use": "本次主战场验证"},
                "fail": {"n": n_fail, "share": round(n_fail/total, 4),
                         "mean_cer_heuristic": fail_heur_mean,
                         "mean_cer_oracle": fail_oracle_mean,
                         "mean_cer_argmax": fail_argmax_mean,
                         "use": "复用 _multivoice_route (已验证)"},
            },
            "overall_mean_cer_multivoice": round(overall, 4),
            "main_line_mean_cer_per_utt": round(main_line_mean, 4),
            "main_line_pool_cer": main_line_pool_cer,
            "delta_overall_vs_mainline": round(overall - main_line_mean, 4),
        },
        "rtf": {
            "sep_sec": round(sep_total_sec, 1),
            "qwen_sec": round(qwen_total_sec, 1),
            "total_sec": round(total_dt, 1),
            "audio_sec": round(total_audio_sec, 1),
            "rtf_sep": round(rtf_sep, 3),
            "rtf_qwen_two_paths": round(rtf_qwen, 3),
            "rtf_total": round(rtf_total, 3),
            "main_line_rtf_4060": main_line_rtf_4060,
            "rtf_l20_extrapolated": [round(rtf_l20_low, 3), round(rtf_l20_high, 3)],
            "note": "L20 外推基于 4060 AD107 / L20 AD102 同代 sm_89 (efficiency-portability-audit); "
                    "绝对 RTF 不可直接外推, 仅相对排序可参考",
        },
        "fail_group_summary_ref": FAIL_SUMMARY,
        "results": results,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
