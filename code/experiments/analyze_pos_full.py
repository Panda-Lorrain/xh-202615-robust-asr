#!/usr/bin/env python3
"""pos 全量结果深度分析：整体 CER + max_sim 分布 + sim_thr 工作点扫描 + 分组诊断 + 极差样本。

pos 全量跑完后立即用，为 sim_thr 调参 / babble 攻坚提供数据依据。
用法:
  code/.venv/Scripts/python.exe code/analyze_pos_full.py <result.json> <pos_pairs_datasetA.json>
"""
import json
import sys
import os
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_datasetA import _norm_zh
from eval_metrics import cer


def main():
    result_path = sys.argv[1]
    manifest_path = sys.argv[2]
    results = {os.path.splitext(os.path.basename(it["recognition"]))[0]: it
               for it in json.load(open(result_path, encoding="utf-8"))["results"]}
    man = json.load(open(manifest_path, encoding="utf-8"))

    rows = []
    miss = 0
    for r in man:
        k = os.path.splitext(os.path.basename(r["recognition"]))[0]
        if k not in results:
            miss += 1
            continue
        res = results[k]
        ref = r["ref"] or ""
        text = res.get("text", "") or ""
        rejected = bool(res.get("rejected"))
        c = 1.0 if (rejected or not text) else cer(_norm_zh(text), _norm_zh(ref))
        rows.append({
            "id": r["id"], "kws_txt": r.get("kws_txt"), "max_sim": float(res.get("max_sim", 0) or 0),
            "cer": c, "rejected": rejected, "noise_type": res.get("noise_type"),
            "ref_len": len(ref), "diar_fail": bool(res.get("diar_fail")),
            "hyp": text, "ref": ref,
        })

    n = len(rows)
    print(f"===== pos 全量深度分析 ({n} 条, manifest 未匹配 {miss}) =====")

    # 1. 整体 CER
    overall = sum(r["cer"] for r in rows) / n
    correct = sum(1 for r in rows if r["cer"] < 0.5) / n
    near = sum(1 for r in rows if r["cer"] < 0.1) / n
    acc = [r for r in rows if not r["rejected"]]
    cer_acc = sum(r["cer"] for r in acc) / len(acc) if acc else 0.0
    n_diar_fail = sum(1 for r in rows if r["diar_fail"])
    n_empty = sum(1 for r in rows if not r["hyp"])
    print(f"overall CER            : {overall:.4f}")
    print(f"correct_rate (CER<0.5) : {correct:.2%}")
    print(f"near_perfect (CER<0.1) : {near:.2%}")
    print(f"cer_accepted_only      : {cer_acc:.4f}  ({len(acc)} 条未被拒)")
    print(f"diar_fail              : {n_diar_fail}  | 空转写: {n_empty}")

    # 2. max_sim 分布
    sims = sorted(r["max_sim"] for r in rows)
    print(f"\n-- max_sim 分布 --")
    print(f"  min={sims[0]:.3f} p25={sims[n//4]:.3f} median={sims[n//2]:.3f} "
          f"p75={sims[3*n//4]:.3f} max={sims[-1]:.3f} mean={statistics.mean(sims):.3f}")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    for i in range(len(bins) - 1):
        cnt = sum(1 for s in sims if bins[i] <= s < bins[i + 1])
        bar = "█" * int(cnt / n * 50)
        print(f"  [{bins[i]:.1f},{bins[i+1]:.1f}): {cnt:5d} ({cnt/n:5.1%}) {bar}")

    # 3. sim_thr 工作点扫描(pos 不该拒, 误拒伤 CER; max_sim<Thr 视为拒识)
    print(f"\n-- sim_thr 工作点扫描 (pos: 误拒=伤CER; 选 neg 拒识阈值的关键依据) --")
    print(f"  {'thr':<6} {'误拒率':<9} {'CER(误拒=1.0)':<15} {'CER(排除误拒)':<14}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        kept = [r for r in rows if r["max_sim"] >= thr]
        fr = 1 - len(kept) / n
        cer_full = sum((1.0 if r["max_sim"] < thr else r["cer"]) for r in rows) / n
        cer_excl = sum(r["cer"] for r in kept) / len(kept) if kept else 1.0
        print(f"  {thr:<6.2f} {fr:<9.2%} {cer_full:<15.4f} {cer_excl:<14.4f}")

    # 4. CER 分桶
    print(f"\n-- CER 分桶 --")
    for lo, hi, name in [(0, 0.05, "完美"), (0.05, 0.1, "优"), (0.1, 0.3, "良"),
                         (0.3, 0.5, "可"), (0.5, 1.01, "差/误拒")]:
        cnt = sum(1 for r in rows if lo <= r["cer"] < hi)
        print(f"  [{lo:.2f},{hi:.2f}) {name:<7}: {cnt:5d} ({cnt/n:5.1%})")

    # 5. 按 noise_type 分组
    print(f"\n-- 按 noise_type 分组 --")
    by_nt = defaultdict(list)
    for r in rows:
        by_nt[r["noise_type"]].append(r["cer"])
    for nt, cs in sorted(by_nt.items(), key=lambda x: -len(x[1])):
        print(f"  {str(nt):<10}: n={len(cs):5d} mean_cer={sum(cs)/len(cs):.4f} "
              f"correct={sum(1 for c in cs if c<0.5)/len(cs):.2%}")

    # 6. 按唤醒词分组
    print(f"\n-- 按唤醒词分组 --")
    by_kw = defaultdict(list)
    for r in rows:
        by_kw[r["kws_txt"]].append(r["cer"])
    for kw, cs in sorted(by_kw.items(), key=lambda x: -len(x[1])):
        print(f"  {kw:<12}: n={len(cs):4d} mean_cer={sum(cs)/len(cs):.4f}")

    # 7. 极差样本(CER>=0.5, 诊断 babble/短指令/特定词)
    print(f"\n-- 极差样本 (CER>=0.5, 前 12, 诊断用) --")
    bad = sorted([r for r in rows if r["cer"] >= 0.5], key=lambda r: -r["cer"])[:12]
    for r in bad:
        print(f"  id{r['id']:4d} CER={r['cer']:.2f} sim={r['max_sim']:.3f} nt={r['noise_type']} "
              f"kw={r['kws_txt']} | hyp={r['hyp'][:25]} | ref={r['ref'][:25]}")

    print(f"\n[评分映射] CER 40% <- overall_cer={overall:.4f} (含误拒惩罚) / cer_accepted_only={cer_acc:.4f} (纯转写质量)")


if __name__ == "__main__":
    main()
