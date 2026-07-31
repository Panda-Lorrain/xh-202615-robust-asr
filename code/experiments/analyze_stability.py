#!/usr/bin/env python
"""稳定性/鲁棒性测试分析器: 汇总所有 run JSON → 波动判定 → 根因归因 → report + per_utt。

spec §6/§8。适配 B1 改动(batch=1×1, batch=16 复用 A_s42_r0):
- R1 GPU非确定: A dim 内多遍(A_s42_r0..r9) transcript 波动
- R2 batch口径: B1_b1_r0(batch=1) vs A_s42_r0(batch=16) 差异(开发16→提交1桥, 主办方默认batch=1)
- R5 数值边界: B2 变种子 dim 内(5种子×2)波动
- R3 输入泛化: B3 微扰 vs A_s42_r0 差异
- R4 声纹锁定: B4_auguon vs A_s42_r0 差异

用法: code/.venv/Scripts/python.exe code/analyze_stability.py [--limit N]
产物: code/stability_matrix/stability_report.json + per_utt_volatility.json
"""
import os, sys, json, glob, argparse, statistics
from collections import defaultdict, Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(_HERE, "stability_matrix")
sys.path.insert(0, _HERE)
from eval_metrics import cer
from eval_datasetA import _norm_zh


def _uid(rec):
    return os.path.splitext(os.path.basename(rec))[0]


def load_runs():
    """返回 {run_id: {uid: result_dict}}。加载所有 A_*/B[1234]_* result JSON(容错缺失)。"""
    runs = {}
    pats = [os.path.join(MATRIX, "A_*.json")] + [os.path.join(MATRIX, f"B{i}_*.json") for i in range(1, 5)]
    fs = set()
    for p in pats:
        fs.update(glob.glob(p))
    for f in sorted(fs):
        rid = os.path.splitext(os.path.basename(f))[0]
        try:
            rows = json.load(open(f, encoding="utf-8"))
            runs[rid] = {_uid(r["recognition"]): r for r in rows}
        except Exception as e:
            print(f"[warn] 跳过 {rid}: {e}")
    return runs


def load_refs():
    rows = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    return {_uid(r["recognition"]): r.get("ref", "") for r in rows}


def group_runs(runs):
    g = defaultdict(list)
    for rid in runs:
        g[rid.split("_")[0]].append(rid)  # A/B1/B2/B3/B4
    return g


def _transcripts_in(uid, runs, run_ids):
    return [runs[r].get(uid, {}).get("transcript", "") for r in run_ids if r in runs and uid in runs[r]]


def classify_root(uid, runs, groups, baseline_rid="A_s42_r0"):
    """根因归因(适配 B1 batch=1×1)。返回 (causes, fix)。"""
    causes, fix = [], []
    base_t = runs.get(baseline_rid, {}).get(uid, {}).get("transcript", "")

    # R1: A dim 内多遍波动(GPU 残余非确定)
    if len(set(_transcripts_in(uid, runs, groups.get("A", [])))) > 1:
        causes.append("R1_gpu_nondeterminism"); fix.append("use_deterministic_algorithms")

    # R2: batch=1 vs batch=16 差异(开发→提交口径桥)
    b1_t = runs.get("B1_b1_r0", {}).get(uid, {}).get("transcript", "")
    if base_t and b1_t and b1_t != base_t:
        causes.append("R2_batch_mismatch"); fix.append("submit_batch1_dev_caveat")

    # R5: B2 变种子 dim 内波动(数值边界)
    if len(set(_transcripts_in(uid, runs, groups.get("B2", [])))) > 1:
        causes.append("R5_numeric_boundary"); fix.append("archive_holdout_reject")

    # R3: B3 微扰 vs 基线 差异(输入泛化短板)
    b3_ts = _transcripts_in(uid, runs, groups.get("B3", []))
    if base_t and any(t and t != base_t for t in b3_ts):
        causes.append("R3_input_generalization"); fix.append("archive_external_training")

    # R4: B4_auguon vs 基线 差异(声纹锁定不稳)
    b4on_t = runs.get("B4_auguon", {}).get(uid, {}).get("transcript", "")
    if base_t and b4on_t and b4on_t != base_t:
        causes.append("R4_voice_locking"); fix.append("archive_enroll_augment_holdout")

    return causes, fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只分析前 N 条(0=全部)")
    args = ap.parse_args()
    runs = load_runs()
    refs = load_refs()
    groups = group_runs(runs)
    print(f"[load] {len(runs)} runs, dims: { {k: len(v) for k, v in groups.items()} }")
    if "A_s42_r0" not in runs:
        print("[warn] 基线 A_s42_r0 缺失, R2/R3/R4 无基线对比(实跑未跑到?)")

    all_uids = sorted({uid for r in runs.values() for uid in r})
    if args.limit > 0:
        all_uids = all_uids[:args.limit]

    # 各 dim 整体 transcript 波动率(dim 内多遍不一致)
    dim_stats = {}
    for dim, ids in groups.items():
        vol_n, total = 0, 0
        for uid in all_uids:
            ts = _transcripts_in(uid, runs, ids)
            if not ts:
                continue
            total += 1
            if len(set(ts)) > 1:
                vol_n += 1
        dim_stats[dim] = {"n_runs": len(ids), "volatile_rate": round(vol_n / total, 4) if total else 0,
                          "volatile_n": vol_n, "total_n": total}

    # R2 整体(batch=1 vs batch=16 不一致率) — 核心: 开发口径数字能否外推提交口径
    if "B1_b1_r0" in runs and "A_s42_r0" in runs:
        diff_n, total = 0, 0
        for uid in all_uids:
            bt = runs["B1_b1_r0"].get(uid, {}).get("transcript", "")
            at = runs["A_s42_r0"].get(uid, {}).get("transcript", "")
            if bt or at:
                total += 1
                if bt != at:
                    diff_n += 1
        dim_stats["B1_vs_A(batch1vs16)"] = {"diff_rate": round(diff_n / total, 4) if total else 0,
                                            "diff_n": diff_n, "total_n": total,
                                            "note": "开发口径(16)→提交口径(1)桥, 0=可外推"}

    per_utt = {}
    for uid in all_uids:
        causes, fix = classify_root(uid, runs, groups)
        if not causes:
            continue
        all_ts, all_cers, all_rej = [], [], []
        for rid in runs:
            r = runs[rid].get(uid)
            if r is None:
                continue
            t = r.get("transcript", "") or ""
            all_ts.append(t)
            all_rej.append(bool(r.get("rejected", False)))
            ref = refs.get(uid, "")
            c = 1.0 if (r.get("rejected") or not t) else cer(_norm_zh(t), _norm_zh(ref))
            all_cers.append(c)
        max_sim = max((runs[r].get(uid, {}).get("max_sim", 0) for r in runs), default=0)
        per_utt[uid] = {
            "ref": refs.get(uid, ""),
            "max_sim": round(max_sim, 4),
            "n_runs_seen": len(all_ts),
            "n_distinct_transcripts": len(set(all_ts)),
            "top_transcripts": dict(Counter(all_ts).most_common(3)),
            "cer_mean": round(statistics.mean(all_cers), 4) if all_cers else None,
            "cer_std": round(statistics.pstdev(all_cers), 4) if len(all_cers) > 1 else 0.0,
            "cer_max": round(max(all_cers), 4) if all_cers else None,
            "decision_flips": len(set(all_rej)) > 1,
            "root_causes": causes,
            "fix_action": fix,
        }

    report = {
        "n_runs_total": len(runs),
        "dim_stats": dim_stats,
        "n_volatile_utts": len(per_utt),
        "root_cause_distribution": dict(Counter(c for u in per_utt for c in per_utt[u]["root_causes"])),
    }
    json.dump(report, open(os.path.join(MATRIX, "stability_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(per_utt, open(os.path.join(MATRIX, "per_utt_volatility.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[done] report + per_utt → stability_matrix/ ({len(per_utt)} 条波动)")
    print("\n=== 根因分布 ===")
    for c, n in report["root_cause_distribution"].items():
        print(f"  {c}: {n}")
    print("\n=== 各维度统计 ===")
    for k, s in dim_stats.items():
        print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
