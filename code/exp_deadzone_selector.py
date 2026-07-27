"""exp_deadzone_selector.py — 死区(sim<0.4) SepFormer 两路 + heuristic 选路 POC.

【任务】验证"主战场 argmax + 死区 multi-voice heuristic"分区 selector 能否降整体 CER。
  死区占 78.8% 贡献 87% CER, 其中"听错人 30%"是 argmax 选错 target. 多 voice 整体替换
  已证伪 (主战场 argmax 0.059 已优), 但死区没专门测 heuristic. 本脚本补这个 POC.

【流程】
  1. 从 poc_qwen_asr_full_result.json 取死区 (sim<0.4) 1064 条, 随机抽 200 (seed=42)
  2. SepFormer (whamr16k) 分离 recognition → 2 路 wav
  3. 两路 wav 都跑 qwen3-asr 转写
  4. heuristic 选路 (cmd_score: content_gate + 设备词/动作词/功能/品牌/news黑名单) 挑 target
  5. 算每条 CER (heuristic_pick vs oracle 取近 ref), 对比 argmax (poc 的 qwen_cer)
  6. 整体外推: 主战场 (sim>=0.4) 复用 poc qwen_cer + 死区 heuristic → 加权整体, 对比主线 0.3436/0.3848

【复用】exp_sepformer_qwen.load_sepformer/separate/load_diar/get_emb_factory,
       qwen_asr_backend subprocess, exp_multivoice_route.cmd_score/route_heuristic,
       text_utils + eval_metrics.cer_official.

【产物】 code/runs/_deadzone_selector/{slices/, uid2text_target.json, uid2text_other.json,
                                          summary.json, per_sample.json}
"""
import os, sys, json, time, argparse, subprocess, glob, random, statistics
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容 (复刻 exp_sepformer_b2.py) ----
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

from exp_sepformer_qwen import load_sepformer, separate, load_diar, get_emb_factory
from exp_multivoice_route import cmd_score, route_heuristic, _strip_punct
from text_utils import to_simplified, digit_postproc, brand_homophone_fix, is_valid_command
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))

POC_RESULT = os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json")
PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")


def cer_normalized(text, ref):
    """应用官方归一链 (繁简→数字→品牌同音) 再算 CER."""
    t = brand_homophone_fix(digit_postproc(to_simplified(text)))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref)))
    return float(cer_official(t, r))


def run_qwen_batch(slice_dir, out_json, batch_size=16):
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--seed", "42",
           "--batch-size", str(batch_size)]
    print(f"[qwen] subprocess 转写 {slice_dir} ({' '.join(cmd)})")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"[qwen] 转写 {time.time()-t0:.1f}s 完成")
    return json.load(open(out_json, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="死区 SepFormer + heuristic selector POC")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-sample", type=int, default=200, help="死区抽样条数")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs/_deadzone_selector"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sim-deadzone-max", type=float, default=0.4, help="死区上界 (exclusive)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    slice_dir = os.path.join(args.out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(slice_dir, "*.wav")):
        os.remove(f)

    # ---- 1. 取死区子集 ----
    poc = json.load(open(POC_RESULT, encoding="utf-8"))
    pairs = json.load(open(PAIRS, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    rows = poc["rows"]
    deadzone_all = [r for r in rows if r["sim"] < args.sim_deadzone_max]
    print(f"[data] 总 {len(rows)}, 死区(sim<{args.sim_deadzone_max}) {len(deadzone_all)} "
          f"({len(deadzone_all)/len(rows)*100:.1f}%)")

    # 必须有 pair 文件存在
    deadzone = [r for r in deadzone_all if r["uid"] in uid2pair]
    print(f"[data] 有 pair 的死区: {len(deadzone)}")

    rng = random.Random(args.seed)
    rng.shuffle(deadzone)
    sample = deadzone[:args.n_sample]
    print(f"[data] 抽样 {len(sample)} 条 (seed={args.seed})")

    # 分桶 (供汇总分析)
    bucket_counts = {
        "<0.2": sum(1 for r in sample if r["sim"] < 0.2),
        "[0.2,0.4)": sum(1 for r in sample if 0.2 <= r["sim"] < 0.4),
    }
    print(f"[data] 分桶: {bucket_counts}")

    # ---- 2. load models ----
    print(f"[load] SepFormer whamr16k → {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. 分离 → 两路 wav ----
    meta, slice_uids_all = [], []
    t_sep_start = time.time()
    sep_total_audio_sec = 0.0
    sep_count = 0
    for n, s in enumerate(sample):
        uid, ref = s["uid"], s["ref"]
        pair = uid2pair[uid]
        enr, rec = pair["enrollment"], pair["recognition"]
        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)
            sep_total_audio_sec += len(audio) / sr
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]
            sep_count += 1

            stream_embs = []
            for i in range(n_src):
                seg = sources[i]
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                stream_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in stream_embs])
            target_idx = int(torch.argmax(sims))

            slice_uids = []
            for i in range(n_src):
                tag = "src%d" % i
                suid = f"{uid}__{tag}"
                sf.write(os.path.join(slice_dir, suid + ".wav"),
                         np.ascontiguousarray(sources[i].astype(np.float32)), 16000)
                slice_uids_all.append(suid)
                slice_uids.append(suid)

            meta.append({
                "uid": uid, "ref": ref, "sim": s["sim"], "n_src": n_src,
                "stream_sims": [round(float(x), 4) for x in sims],
                "sep_target_idx": target_idx, "sep_target_sim": round(float(sims[target_idx]), 4),
                "slice_uids": slice_uids,
                "argmax_qwen_cer_poc": s.get("qwen_cer"),
                "argmax_qwen_text_poc": s.get("qwen"),
                "argmax_vanilla_cer_poc": s.get("vanilla_cer"),
            })
            if (n + 1) % 20 == 0 or n == len(sample) - 1:
                print(f"  [{n+1}/{len(sample)}] sep 完成 ({time.time()-t_sep_start:.0f}s, "
                      f"audio={sep_total_audio_sec:.0f}s)")
        except Exception as e:
            print(f"  [{n+1}/{len(sample)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "sim": s["sim"],
                         "argmax_qwen_cer_poc": s.get("qwen_cer"),
                         "argmax_qwen_text_poc": s.get("qwen"),
                         "error": f"{type(e).__name__}: {str(e)[:120]}"})

    t_sep_total = time.time() - t_sep_start
    print(f"\n[sep] 分离 {sep_count}/{len(sample)} 完成, 总耗时 {t_sep_total/60:.1f}min, "
          f"音频总时长 {sep_total_audio_sec:.0f}s")

    # ---- 4. 两路 qwen 批转写 ----
    print(f"\n[qwen] 转写 {len(slice_uids_all)} 路 (src0+src1 全转)...")
    qwen_out_path = os.path.join(args.out_dir, "_uid2text.json")
    uid2text = run_qwen_batch(slice_dir, qwen_out_path, args.batch_size)
    t_qwen_total = time.time() - t_sep_start - t_sep_total

    # ---- 5. heuristic 选路 + 算 CER ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        per_src = []
        for i, suid in enumerate(m["slice_uids"]):
            t = uid2text.get(suid, "")
            c = cer_normalized(t, m["ref"])
            per_src.append({"src_idx": i, "slice_uid": suid, "text": t, "cer": round(c, 4),
                            "score": round(cmd_score(t), 3)})
        oracle = min(per_src, key=lambda x: x["cer"])

        # heuristic 选路
        heur_idx, heur_reason = route_heuristic(per_src)
        heur_text = per_src[heur_idx]["text"]
        heur_cer = per_src[heur_idx]["cer"]

        # sim 选路 (B2 复刻, 对照)
        sim_idx = m["sep_target_idx"]
        sim_cer = per_src[sim_idx]["cer"] if sim_idx < len(per_src) else None

        results.append({
            **m,
            "per_src": per_src,
            "oracle_src_idx": oracle["src_idx"],
            "oracle_cer": round(oracle["cer"], 4),
            "heuristic_idx": heur_idx,
            "heuristic_reason": heur_reason,
            "heuristic_text": heur_text,
            "heuristic_cer": round(heur_cer, 4),
            "heuristic_picks_oracle": heur_idx == oracle["src_idx"],
            "sim_pick_idx": sim_idx,
            "sim_pick_cer": round(sim_cer, 4) if sim_cer is not None else None,
        })

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = time.time() - t_sep_start
    # ---- 6. 汇总 ----
    def _stats(rs, label):
        if not rs:
            return {"label": label, "n": 0}
        heur = np.array([r["heuristic_cer"] for r in rs])
        orc = np.array([r["oracle_cer"] for r in rs])
        sim_arr = np.array([r["sim_pick_cer"] for r in rs if r.get("sim_pick_cer") is not None])
        # argmax 从 poc 拿
        ag = np.array([r["argmax_qwen_cer_poc"] for r in rs if r.get("argmax_qwen_cer_poc") is not None])

        out = {
            "label": label, "n": len(rs),
            "argmax_mean_cer": round(float(np.mean(ag)), 4) if len(ag) else None,
            "heuristic_mean_cer": round(float(np.mean(heur)), 4),
            "oracle_mean_cer": round(float(np.mean(orc)), 4),
            "sim_pick_mean_cer": round(float(np.mean(sim_arr)), 4) if len(sim_arr) else None,
            "delta_heur_vs_argmax": (round(float(np.mean(heur) - np.mean(ag)), 4)
                                     if len(ag) == len(rs) else None),
            "delta_oracle_vs_argmax": (round(float(np.mean(orc) - np.mean(ag)), 4)
                                       if len(ag) == len(rs) else None),
            "heuristic_correct_rate": round(float(np.mean(heur < 0.5)), 4),
            "oracle_correct_rate": round(float(np.mean(orc < 0.5)), 4),
            "argmax_correct_rate": (round(float(np.mean(ag < 0.5)), 4) if len(ag) else None),
            "n_recovered_by_heur": int(sum(1 for r in rs if r["heuristic_cer"] < 0.5)),
            "n_recovered_by_oracle": int(sum(1 for r in rs if r["oracle_cer"] < 0.5)),
            "n_recovered_by_argmax": (int(sum(1 for r in rs if r.get("argmax_qwen_cer_poc", 1) < 0.5))
                                      if len(ag) else None),
            "heuristic_pick_accuracy": round(float(np.mean([r["heuristic_picks_oracle"] for r in rs])), 4),
            "heuristic_pick_accuracy_count": int(sum(1 for r in rs if r["heuristic_picks_oracle"])),
        }
        return out

    overall_stats = _stats(valid, "死区整体 (n=%d)" % n_valid)
    sim_lo = [r for r in valid if r["sim"] < 0.2]
    sim_mid = [r for r in valid if 0.2 <= r["sim"] < 0.4]
    sim_lo_stats = _stats(sim_lo, "死区 sim<0.2 (重 babble)")
    sim_mid_stats = _stats(sim_mid, "死区 [0.2,0.4) (中)")

    # 失败子集 (argmax_cer>0.8) 上 heuristic 救回率
    fail_sub = [r for r in valid if (r.get("argmax_qwen_cer_poc") or 0) > 0.8]
    fail_stats = _stats(fail_sub, "死区失败子集 (argmax>0.8)")

    # 控制台
    print(f"\n{'='*70}")
    print(f"[死区 selector POC] 有效 {n_valid}/{len(sample)}, 总耗时 {total_dt/60:.1f}min")
    print(f"{'='*70}")
    def _print_block(s):
        if s.get("n", 0) == 0:
            print(f"\n[{s['label']}] n=0")
            return
        print(f"\n[{s['label']}]")
        print(f"  argmax (主线):     mean CER {s.get('argmax_mean_cer')}  "
              f"(correct<0.5: {(s.get('argmax_correct_rate') or 0)*100:.0f}%)")
        print(f"  heuristic 选路:    mean CER {s['heuristic_mean_cer']:.4f}  "
              f"(correct<0.5: {s['heuristic_correct_rate']*100:.0f}%)")
        print(f"  oracle 选路:       mean CER {s['oracle_mean_cer']:.4f}  "
              f"(correct<0.5: {s['oracle_correct_rate']*100:.0f}%)")
        if s.get("sim_pick_mean_cer") is not None:
            print(f"  sim 选路 (B2):     mean CER {s['sim_pick_mean_cer']:.4f}")
        print(f"  Δ(heur - argmax):  {s['delta_heur_vs_argmax']}")
        print(f"  Δ(oracle - argmax):{s['delta_oracle_vs_argmax']}")
        print(f"  救回(CER<0.5): heur {s['n_recovered_by_heur']} / "
              f"oracle {s['n_recovered_by_oracle']} / "
              f"argmax {s.get('n_recovered_by_argmax')}")
        print(f"  heuristic 选对 oracle 路: {s['heuristic_pick_accuracy_count']}/{s['n']} "
              f"({s['heuristic_pick_accuracy']*100:.0f}%)")

    _print_block(overall_stats)
    _print_block(sim_lo_stats)
    _print_block(sim_mid_stats)
    _print_block(fail_stats)

    # ---- 7. 整体外推 ----
    # 主战场 (sim>=0.4) 复用 poc qwen_cer (主线), 死区用本次 heuristic / argmax / oracle
    rows = poc["rows"]
    mainfield = [r for r in rows if r["sim"] >= args.sim_deadzone_max]
    deadzone_full = [r for r in rows if r["sim"] < args.sim_deadzone_max]
    n_main = len(mainfield)
    n_dead = len(deadzone_full)
    n_total = n_main + n_dead
    # 主线整体 (逐条 mean)
    overall_argmax = float(np.mean([r["qwen_cer"] for r in rows]))
    main_argmax_mean = float(np.mean([r["qwen_cer"] for r in mainfield]))
    dead_argmax_mean = float(np.mean([r["qwen_cer"] for r in deadzone_full]))
    # 死区 selector 假设: 把抽样上的 Δ(heur - argmax) 加到死区均值上
    delta_heur = overall_stats["delta_heur_vs_argmax"]
    delta_oracle = overall_stats["delta_oracle_vs_argmax"]
    dead_heur_extrapolated = dead_argmax_mean + (delta_heur if delta_heur is not None else 0)
    dead_oracle_extrapolated = dead_argmax_mean + (delta_oracle if delta_oracle is not None else 0)
    # 加权整体 (按真实占比)
    overall_heur_extrapolated = (main_argmax_mean * n_main + dead_heur_extrapolated * n_dead) / n_total
    overall_oracle_extrapolated = (main_argmax_mean * n_main + dead_oracle_extrapolated * n_dead) / n_total

    print(f"\n{'='*70}")
    print(f"[整体外推 — 主战场 argmax + 死区 selector]")
    print(f"{'='*70}")
    print(f"  主战场 (sim>={args.sim_deadzone_max}) n={n_main} ({n_main/n_total*100:.1f}%) "
          f"argmax mean CER {main_argmax_mean:.4f}")
    print(f"  死区   (sim<{args.sim_deadzone_max}) n={n_dead} ({n_dead/n_total*100:.1f}%) "
          f"argmax mean CER {dead_argmax_mean:.4f}")
    print(f"  死区 200 抽样 Δ(heur - argmax) = {delta_heur}")
    print(f"  死区 200 抽样 Δ(oracle - argmax) = {delta_oracle}")
    print(f"  → 死区 heuristic 外推 CER = {dead_heur_extrapolated:.4f}")
    print(f"  → 死区 oracle    外推 CER = {dead_oracle_extrapolated:.4f}")
    print(f"  整体主线 argmax (逐条 mean): {overall_argmax:.4f}")
    print(f"  整体 heuristic 外推:        {overall_heur_extrapolated:.4f}  "
          f"(Δ {overall_heur_extrapolated-overall_argmax:+.4f})")
    print(f"  整体 oracle    外推 (上限): {overall_oracle_extrapolated:.4f}  "
          f"(Δ {overall_oracle_extrapolated-overall_argmax:+.4f})")
    # 官方池口径参考 (CLAUDE.md: qwen 主线 0.3436)
    print(f"  对照主线官方池 CER 0.3436 (从含拒评分公式)")

    # ---- 8. RTF 增量 ----
    sep_audio_dur = sep_total_audio_sec
    sep_rtf = t_sep_total / sep_audio_dur if sep_audio_dur > 0 else None
    # qwen 转写 2 路: 200条 × 2 = 400 路, 音频 ~ 2 × sep_audio_dur
    qwen_rtf = t_qwen_total / (2 * sep_audio_dur) if sep_audio_dur > 0 else None
    print(f"\n[RTF 增量 (200 条死区)]")
    print(f"  分离 SepFormer: {t_sep_total:.0f}s / 音频 {sep_audio_dur:.0f}s = RTF {sep_rtf:.3f}")
    print(f"  两路 qwen 转写: {t_qwen_total:.0f}s / 音频 {2*sep_audio_dur:.0f}s = RTF {qwen_rtf:.3f}")
    print(f"  死区增量 RTF = {sep_rtf + qwen_rtf:.3f} (per 死区音频秒)")
    print(f"  整体增量 (死区占比 {n_dead/n_total*100:.0f}%): "
          f"{(sep_rtf + qwen_rtf) * n_dead/n_total:.3f}")

    # ---- 9. heuristic 选错案例分析 ----
    # 失败 = heuristic_cer > argmax_cer (选路比 argmax 还差)
    worse = [r for r in valid if r["heuristic_cer"] > (r.get("argmax_qwen_cer_poc") or 0) + 0.05]
    better = [r for r in valid if r["heuristic_cer"] < (r.get("argmax_qwen_cer_poc") or 1) - 0.05]
    tie = [r for r in valid if r not in worse and r not in better]

    # 两路都过 content_gate (TRAP)
    n_both_valid = sum(1 for r in valid
                       if all(is_valid_command(s["text"]) for s in r["per_src"]))
    n_one_valid = sum(1 for r in valid
                      if sum(is_valid_command(s["text"]) for s in r["per_src"]) == 1)
    n_both_invalid = sum(1 for r in valid
                         if not any(is_valid_command(s["text"]) for s in r["per_src"]))

    print(f"\n[heuristic vs argmax 逐条胜负]")
    print(f"  better (heur 比 argmax 强 ≥0.05): {len(better)}")
    print(f"  tie (|Δ| < 0.05):                 {len(tie)}")
    print(f"  worse (heur 比 argmax 差 ≥0.05):  {len(worse)}")
    print(f"\n[内容判别 TRAP 分布]")
    print(f"  两路都过 content_gate (TRAP): {n_both_valid}/{n_valid}")
    print(f"  一路过 / 一路拒:              {n_one_valid}/{n_valid}")
    print(f"  两路都拒:                     {n_both_invalid}/{n_valid}")

    # 选错案例特征
    err_features = []
    for r in worse[:10]:
        err_features.append({
            "uid": r["uid"], "sim": r["sim"],
            "argmax_cer": r.get("argmax_qwen_cer_poc"),
            "heur_cer": r["heuristic_cer"], "heur_reason": r["heuristic_reason"],
            "texts": [s["text"] for s in r["per_src"]],
            "ref": r["ref"],
        })

    summary_out = {
        "verdict": "死区 SepFormer + heuristic selector POC",
        "seed": args.seed, "n_sample": args.n_sample, "n_valid": n_valid,
        "bucket_counts": bucket_counts,
        "sep_time_min": round(t_sep_total / 60, 2),
        "qwen_time_min": round(t_qwen_total / 60, 2),
        "total_time_min": round(total_dt / 60, 2),
        "sep_audio_total_sec": round(sep_total_audio_sec, 1),
        "sep_rtf": round(sep_rtf, 3) if sep_rtf else None,
        "qwen_rtf_2path": round(qwen_rtf, 3) if qwen_rtf else None,
        "stats": {
            "overall_deadzone": overall_stats,
            "sim_lt_0.2": sim_lo_stats,
            "sim_0.2_0.4": sim_mid_stats,
            "fail_subset_argmax_gt_0.8": fail_stats,
        },
        "extrapolation": {
            "n_main": n_main, "n_dead": n_dead, "n_total": n_total,
            "main_argmax_mean": round(main_argmax_mean, 4),
            "dead_argmax_mean": round(dead_argmax_mean, 4),
            "dead_heur_extrapolated": round(dead_heur_extrapolated, 4),
            "dead_oracle_extrapolated": round(dead_oracle_extrapolated, 4),
            "overall_argmax_baseline": round(overall_argmax, 4),
            "overall_heur_extrapolated": round(overall_heur_extrapolated, 4),
            "overall_oracle_extrapolated": round(overall_oracle_extrapolated, 4),
            "delta_heur_overall": round(overall_heur_extrapolated - overall_argmax, 4),
            "delta_oracle_overall": round(overall_oracle_extrapolated - overall_argmax, 4),
            "baseline_official_pool": 0.3436,
        },
        "heur_vs_argmax": {
            "n_better": len(better), "n_tie": len(tie), "n_worse": len(worse),
            "TRAP_both_valid": n_both_valid,
            "one_valid_clean": n_one_valid,
            "both_invalid_fallback": n_both_invalid,
        },
        "worse_examples": err_features,
        "results": results,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_json}")


if __name__ == "__main__":
    main()
