"""run-twice 可复现性验证: 同 seed 跑两遍 enroll_infer, 比对 transcript 一致性 + CER delta。

量化 fp16 + CUDA 残余非确定(spec docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md §7)。
FAQ 核查硬要求 6(seed 固定后跑两遍比对)。CER delta > 0.01 警告(决策是否该段升 fp32, 不阻塞)。

用法:
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/verify_reproducibility.py \
    --pairs code/pos_pairs_datasetA.json --limit 20 --seed 42
"""
import os, sys, json, argparse, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.join(_HERE, ".venv", "Scripts", "python.exe")
sys.path.insert(0, _HERE)
from eval_metrics import cer as _cer
from eval_datasetA import _norm_zh


def run_enroll_once(pairs_file, out_json, seed, device, backend):
    """subprocess 跑 enroll_infer --pairs(同 seed), 返回 (wall_sec)。失败抛 RuntimeError。"""
    import time
    cmd = [_PY, os.path.join(_HERE, "enroll_infer.py"),
           "--pairs", pairs_file, "--out-json", out_json,
           "--always-generate", "--reject-threshold", "0",
           "--asr-backend", backend, "--device", device, "--seed", str(seed)]
    print(f"[run] enroll_infer --seed {seed} --backend {backend} → {os.path.basename(out_json)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"enroll_infer 失败({r.returncode}): {r.stderr[-500:]}")
    print(f"      done ({dt:.1f}s)")
    return dt


def _uid(rec_path):
    return os.path.splitext(os.path.basename(rec_path))[0]


def main():
    ap = argparse.ArgumentParser(description="run-twice 可复现性验证(fp16 残余非确定量化)")
    ap.add_argument("--pairs", required=True, help="pos_pairs_datasetA.json")
    ap.add_argument("--limit", type=int, default=20, help="前 N 条(0=全部)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--backend", default="vanilla", choices=["vanilla", "dicow", "qwen"])
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "verify_repro_out"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 临时 pairs 子集(enroll_infer 无 --limit, 用前 N 条临时 manifest)
    rows = json.load(open(args.pairs, encoding="utf-8"))
    if args.limit > 0:
        rows = rows[: args.limit]
    tmp_pairs = os.path.join(args.out_dir, "pairs_subset.json")
    json.dump(rows, open(tmp_pairs, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[verify] {len(rows)} 条, seed={args.seed}, backend={args.backend}")

    # 同 seed 跑两遍
    run1 = os.path.join(args.out_dir, "run1.json")
    run2 = os.path.join(args.out_dir, "run2.json")
    run_enroll_once(tmp_pairs, run1, args.seed, args.device, args.backend)
    run_enroll_once(tmp_pairs, run2, args.seed, args.device, args.backend)

    # 比对(两次 transcript 互比, 衡量输出差异 —— 不需 ref)
    r1 = {_uid(x["recognition"]): x for x in json.load(open(run1, encoding="utf-8"))}
    r2 = {_uid(x["recognition"]): x for x in json.load(open(run2, encoding="utf-8"))}

    n, n_match, deltas = len(r1), 0, []
    per_utt = []
    for uid in r1:
        t1 = r1[uid].get("transcript", "") or ""
        t2 = (r2.get(uid) or {}).get("transcript", "") or ""
        match = (t1 == t2)
        if match:
            n_match += 1
            d = 0.0
        elif t1 and t2:
            d = _cer(_norm_zh(t1), _norm_zh(t2))
        else:
            d = 1.0  # 一空一非空
        deltas.append(d)
        per_utt.append({"uid": uid, "match": match, "cer_delta": round(d, 4),
                        "t1": t1[:40], "t2": t2[:40]})

    match_rate = n_match / n if n else 0
    max_d = max(deltas) if deltas else 0
    mean_d = sum(deltas) / len(deltas) if deltas else 0
    n_over = sum(1 for d in deltas if d > 0.01)

    print(f"\n===== run-twice 验证({n} 条, seed={args.seed}, backend={args.backend}) =====")
    print(f"text 完全一致率      : {match_rate:.2%} ({n_match}/{n})")
    print(f"CER delta 平均       : {mean_d:.4f}")
    print(f"CER delta 最大       : {max_d:.4f}")
    print(f"CER delta >0.01 条数 : {n_over}/{n}")
    if max_d > 0.01:
        print(f"[WARN] fp16 残余非确定超阈(max={max_d:.4f}>0.01), 建议该段升 fp32(spec §6)")
    else:
        print(f"[OK] fp16 残余非确定在阈内(max={max_d:.4f}≤0.01), 可复现性达标")

    summary = {"n": n, "seed": args.seed, "backend": args.backend,
               "text_match_rate": round(match_rate, 4),
               "cer_delta_mean": round(mean_d, 4),
               "cer_delta_max": round(max_d, 4),
               "n_over_0.01": n_over, "per_utt": per_utt}
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[done] → {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
