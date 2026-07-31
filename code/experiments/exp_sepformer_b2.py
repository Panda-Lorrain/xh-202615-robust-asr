"""exp_sepformer_b2.py — SepFormer 真源分离攻双人重叠失败组第③层(B2)。

【背景】前置 A+B1 把 fail 组(sim≥0.4 & qwen_cer>0.8, 40 条)拆三层:
  ①~20% 选 speaker 能救(A 实验, sim 反向)
  ②~20% 切时间段能救(B1 forced alignment 整体 0.940, 对齐可靠子集 0.55)
  ③~60-80% 现有方法够不着(B1 对齐失效子集 n=32, align_score<0.4) ← B2 攻这层
基线: argmax mean CER 1.216 / A oracle 选 speaker 0.850 / B1 forced alignment 整体 0.940

【B2 假设】SepFormer 把 recognition 分离成 2 路单说话人 → 选 target 路 → qwen 转 → CER,
直接测源分离能否攻第③层。

【方法】复用 exp_sepformer_qwen.py 的 SepFormer whamr16k + diar._embedding 选路 + qwen
subprocess 批转写框架, 改:
  ①数据源: _oracle_speaker/summary.json 中 group==fail 的 40 条(主战场双人重叠)
  ②关联 _oracle_separation/meta.json 拿 align_score, 分桶对齐失效(align_score<0.4)
  ③两路(target/other)都转写, 同时报 sim 选路 + oracle 选路 CER

【判别】
  SepFormer+oracle 选路 ≪ 0.85 → 源分离有效, 攻得动第③层
  ≈ 1.216 或更高 → SepFormer 在中文重叠无效(英文 OOD / SI-SDR 陷阱)
  中间 → 部分有效, 看救回几条

【局限】
  1. SepFormer 训练在英文 WHAM/LibriMix, 中文 OOD 风险
  2. SI-SDR 陷阱: 波形分离优化 SI-DR ≠ ASR 友好, mel 可能更糟
  3. B2 失败不能断言"源分离无空间", 只能说 SepFormer 中文 OOD 或 SI-SDR 陷阱

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_sepformer_b2.py
产物: code/runs/_sepformer_b2/{slices/, _uid2text.json, summary.json}
"""
import os, sys, json, time, argparse, subprocess, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# ---- speechbrain Windows 兼容(复刻 exp_sepformer_qwen.py, SB 1.1.0 LazyModule inspect-guard) ----
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

# 复用 exp_sepformer_qwen 的 load_sepformer / separate / load_diar / get_emb_factory
from exp_sepformer_qwen import load_sepformer, separate, load_diar, get_emb_factory
from text_utils import to_simplified, digit_postproc, brand_homophone_fix
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model

PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))


def run_qwen_batch(slice_dir, out_json, batch_size=16, context=""):
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", out_json, "--seed", "42",
           "--batch-size", str(batch_size)]
    if context:
        cmd += ["--context", context]
    print(f"[qwen] subprocess 转写 {slice_dir} ({' '.join(cmd)})")
    subprocess.run(cmd, check=True)
    return json.load(open(out_json, encoding="utf-8"))


def cer_normalized(text, ref):
    """应用官方归一链(繁简→数字→品牌同音)再算 CER, 与 enroll_infer 一致。"""
    t = brand_homophone_fix(digit_postproc(to_simplified(text)))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref)))
    return float(cer_official(t, r))


def main():
    ap = argparse.ArgumentParser(description="SepFormer 真源分离攻双人重叠失败组第③层(B2)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diarization-model", default=resolve_model("DIAR"))
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle-summary", default=os.path.join(_HERE, "runs/_oracle_speaker/summary.json"))
    ap.add_argument("--sep-meta", default=os.path.join(_HERE, "runs/_oracle_separation/meta.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs/_sepformer_b2"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--align-fail-thr", type=float, default=0.4,
                    help="B1 对齐失效阈值(align_score<thr 视为第③层, 默认 0.4)")
    args = ap.parse_args()

    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    slice_dir = os.path.join(args.out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(slice_dir, "*.wav")):
        os.remove(f)

    # ---- 1. 数据 ----
    summary = json.load(open(args.oracle_summary, encoding="utf-8"))
    sep_meta = json.load(open(args.sep_meta, encoding="utf-8"))
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    uid2align = {m["uid"]: m.get("align_score") for m in sep_meta}
    fail = [s for s in summary if s.get("group") == "fail" and s["uid"] in uid2pair]
    print(f"[data] summary={len(summary)} fail={len(fail)} (主战场双人重叠组)")
    print(f"[data] 对齐失效子集(align_score<{args.align_fail_thr}): "
          f"{sum(1 for s in fail if (uid2align.get(s['uid']) or 0) < args.align_fail_thr)} 条")

    # ---- 2. load ----
    print(f"[load] SepFormer whamr16k → {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(args.diarization_model, device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. Phase1: 分离 → 两路都存 wav(为 oracle 选路准备) ----
    meta, slice_uids_all = [], []
    t0 = time.time()
    for n, s in enumerate(fail):
        uid, ref = s["uid"], s["ref"]
        pair = uid2pair[uid]
        enr, rec = pair["enrollment"], pair["recognition"]
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

            slice_uids = []
            for i in range(n_src):
                tag = "target" if i == target_idx else f"src{i}"
                suid = f"{uid}__{tag}"
                sf.write(os.path.join(slice_dir, suid + ".wav"),
                         np.ascontiguousarray(sources[i].astype(np.float32)), 16000)
                slice_uids_all.append(suid)
                slice_uids.append(suid)

            meta.append({
                "uid": uid, "ref": ref, "n_src": n_src,
                "stream_sims": [round(float(x), 4) for x in sims],
                "target_idx": target_idx, "target_sim": round(float(sims[target_idx]), 4),
                "slice_uids": slice_uids,
                "argmax_cer_A": s.get("argmax_cer"),
                "oracle_speaker_cer_A": s.get("oracle_cer"),
                "poc_qwen_cer": s.get("poc_qwen_cer"),
                "poc_sim": s.get("poc_sim"),
                "align_score": uid2align.get(uid),
            })
            print(f"  [{n+1}/{len(fail)}] {uid} n_src={n_src} sep_sim={float(sims[target_idx]):.3f} "
                  f"argmax_CER={s.get('argmax_cer')} align={uid2align.get(uid)} ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{n+1}/{len(fail)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- 4. Phase2: 批量 qwen(两路都转) ----
    print(f"\n[qwen] 转写 {len(slice_uids_all)} 路 (target+other 全转, 为 oracle 选路)...")
    qwen_out_path = os.path.join(args.out_dir, "_uid2text.json")
    uid2text = run_qwen_batch(slice_dir, qwen_out_path, args.batch_size)

    # ---- 5. Phase3: 算 CER(sim 选路 + oracle 选路) ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        # sim 选 target 路
        target_uid = m["slice_uids"][m["target_idx"]]
        target_text = uid2text.get(target_uid, "")
        sep_cer = cer_normalized(target_text, m["ref"])
        # oracle 选路(两路取 CER 最小)
        per_src = []
        for i, suid in enumerate(m["slice_uids"]):
            t = uid2text.get(suid, "")
            c = cer_normalized(t, m["ref"])
            per_src.append({"src_idx": i, "slice_uid": suid, "text": t, "cer": round(c, 4)})
        oracle = min(per_src, key=lambda x: x["cer"])
        results.append({
            **m,
            "target_uid": target_uid, "sep_text": target_text, "sep_cer": round(sep_cer, 4),
            "per_src": per_src,
            "oracle_src_idx": oracle["src_idx"],
            "oracle_text": oracle["text"], "oracle_cer": round(oracle["cer"], 4),
            "sep_picks_oracle": (m["target_idx"] == oracle["src_idx"]),
        })

    valid = [r for r in results if "error" not in r]
    n_valid = len(valid)
    total_dt = time.time() - t0

    # ---- 6. 汇总(分桶) ----
    def _stats(rs):
        if not rs:
            return {"n": 0}
        sep = np.array([r["sep_cer"] for r in rs])
        orc = np.array([r["oracle_cer"] for r in rs])
        ag = np.array([r["argmax_cer_A"] for r in rs if r.get("argmax_cer_A") is not None])
        return {
            "n": len(rs),
            "sep_sim_mean": round(float(np.mean(sep)), 4),
            "sep_oracle_mean": round(float(np.mean(orc)), 4),
            "argmax_mean": round(float(np.mean(ag)), 4) if len(ag) else None,
            "sep_correct_rate": round(float(np.mean(sep < 0.5)), 4),
            "oracle_correct_rate": round(float(np.mean(orc < 0.5)), 4),
            "delta_sep_vs_argmax": (round(float(np.mean(sep) - np.mean(ag)), 4)
                                    if len(ag) == len(rs) else None),
            "delta_oracle_vs_argmax": (round(float(np.mean(orc) - np.mean(ag)), 4)
                                       if len(ag) == len(rs) else None),
            "n_recovered_by_sep": int(sum(1 for r in rs if r["sep_cer"] < 0.5)),
            "n_recovered_by_oracle": int(sum(1 for r in rs if r["oracle_cer"] < 0.5)),
            "n_sim_picks_oracle": int(sum(1 for r in rs if r["sep_picks_oracle"])),
        }

    fail_stats = _stats(valid)
    align_fail = [r for r in valid if (r.get("align_score") or 0) < args.align_fail_thr]
    align_fail_stats = _stats(align_fail)
    align_ok = [r for r in valid if (r.get("align_score") or 0) >= args.align_fail_thr]
    align_ok_stats = _stats(align_ok)

    print(f"\n{'='*70}\n[B2 SepFormer 真源分离 — 双人重叠失败组第③层]")
    print(f"有效 {n_valid}/{len(fail)}, 总耗时 {total_dt/60:.1f}min\n")
    print(f"[失败组 40 条]")
    print(f"  SepFormer+sim选路  mean CER: {fail_stats['sep_sim_mean']:.3f}  "
          f"(correct<0.5: {fail_stats['sep_correct_rate']*100:.0f}%)")
    print(f"  SepFormer+oracle选路 mean CER: {fail_stats['sep_oracle_mean']:.3f}  "
          f"(correct<0.5: {fail_stats['oracle_correct_rate']*100:.0f}%)")
    print(f"  对照 argmax mean CER:        {fail_stats['argmax_mean']}")
    print(f"  Δ(sep - argmax):   {fail_stats['delta_sep_vs_argmax']}")
    print(f"  Δ(oracle - argmax):{fail_stats['delta_oracle_vs_argmax']}")
    print(f"  sep 救回(CER<0.5): {fail_stats['n_recovered_by_sep']} / "
          f"oracle 救回: {fail_stats['n_recovered_by_oracle']}")
    print(f"  sim 选对 oracle 路: {fail_stats['n_sim_picks_oracle']}/{fail_stats['n']}")
    print(f"\n[B1 对齐失效子集(align_score<{args.align_fail_thr}) — 第③层重点] n={align_fail_stats['n']}")
    if align_fail_stats["n"]:
        print(f"  SepFormer+sim选路  mean CER: {align_fail_stats['sep_sim_mean']:.3f}  "
              f"(correct<0.5: {align_fail_stats['sep_correct_rate']*100:.0f}%)")
        print(f"  SepFormer+oracle选路 mean CER: {align_fail_stats['sep_oracle_mean']:.3f}  "
              f"(correct<0.5: {align_fail_stats['oracle_correct_rate']*100:.0f}%)")
        print(f"  对照 argmax mean CER:        {align_fail_stats['argmax_mean']}")
        print(f"  sep 救回(CER<0.5): {align_fail_stats['n_recovered_by_sep']} / "
              f"oracle 救回: {align_fail_stats['n_recovered_by_oracle']}")
    print(f"\n[B1 对齐可靠子集(align_score≥{args.align_fail_thr})] n={align_ok_stats['n']}")
    if align_ok_stats["n"]:
        print(f"  SepFormer+sim选路  mean CER: {align_ok_stats['sep_sim_mean']:.3f}")
        print(f"  SepFormer+oracle选路 mean CER: {align_ok_stats['sep_oracle_mean']:.3f}")

    print(f"\n{'='*70}\n[判定]")
    if fail_stats["sep_oracle_mean"] < 0.85:
        verdict = "源分离有效"
        reason = (f"SepFormer+oracle选路 mean CER {fail_stats['sep_oracle_mean']:.3f} ≪ 0.85, "
                  f"显著低于 argmax {fail_stats['argmax_mean']} 和 A oracle_speaker 0.850, "
                  f"攻得动第③层。")
    elif fail_stats["sep_oracle_mean"] >= (fail_stats["argmax_mean"] or 1.0) - 0.05:
        verdict = "SepFormer 在中文重叠无效(可能英文OOD/SI-SDR陷阱, 不能断言源分离无空间)"
        reason = (f"SepFormer+oracle选路 mean CER {fail_stats['sep_oracle_mean']:.3f} "
                  f"≥ argmax {fail_stats['argmax_mean']} (Δ≥-0.05), 第③层 SepFormer 救不动, "
                  f"但中文 OOD/SI-SDR 陷阱是可能原因, 需 TSE 或中文分离模型验证。")
    else:
        verdict = "部分有效"
        reason = (f"oracle {fail_stats['sep_oracle_mean']:.3f} 介于 0.85 和 argmax 之间, "
                  f"看救回 {fail_stats['n_recovered_by_oracle']}/{fail_stats['n']} 条。")
    print(f"  判定: {verdict}")
    print(f"  理由: {reason}")

    summary_out = {
        "verdict": verdict, "reason": reason,
        "model": "speechbrain/sepformer-whamr16k + Qwen3-ASR-1.7B + DiariZen wespeaker-emb",
        "n_fail": len(fail), "n_valid": n_valid, "total_min": round(total_dt / 60, 2),
        "fail_group_stats": fail_stats,
        "align_fail_subset_stats": align_fail_stats,
        "align_ok_subset_stats": align_ok_stats,
        "baseline_ref": {"argmax_mean_40": 1.216, "oracle_speaker_A": 0.850,
                         "B1_forced_align_overall": 0.940},
        "limits": [
            "SepFormer 训练在英文 WHAM/LibriMix, 中文 OOD 风险",
            "SI-SDR 陷阱: 波形分离优化 SI-DR ≠ ASR 友好, mel 可能更糟",
            "B2 失败只能证伪 SepFormer, 不能断言'源分离无空间'(需 TSE/中文分离模型验证)"
        ],
        "results": results,
    }
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_json} (总耗时 {total_dt/60:.1f}min)")


if __name__ == "__main__":
    main()
