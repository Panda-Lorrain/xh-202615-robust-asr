#!/usr/bin/env python
"""稳定性/鲁棒性测试编排器: 按扰动矩阵逐遍跑 enroll_infer, 每遍产 run-id JSON。

spec: docs/superpowers/specs/2026-07-19-stability-robustness-test-design.md
断点续跑: 已存在的 run-id JSON 自动跳过。

用法:
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 10            # 全量
  code/.venv/Scripts/python.exe code/stability_test.py --phase A --runs 2 --limit 5   # dry-run
  code/.venv/Scripts/python.exe code/stability_test.py --phase B1|B2|B3|B4|all
"""
import os, sys, json, time, subprocess, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY_MAIN = os.environ.get("PY_MAIN") or os.path.join(_HERE, ".venv", "Scripts", "python.exe")
PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")
OUT_DIR = os.path.join(_HERE, "stability_matrix")
SIM_THR = 0.27  # 主线提交 thr(memory unified-thr-decision), 决策翻盘维度需要


def _subset_pairs(run_id, pairs, limit):
    """limit>0 取前 N 条写临时 manifest(enroll_infer 无 --limit), 返回新 pairs 路径。"""
    if limit <= 0:
        return pairs
    rows = json.load(open(pairs, encoding="utf-8"))[:limit]
    tmp = os.path.join(OUT_DIR, f"_pairs_{run_id}.json")
    json.dump(rows, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    return tmp


def run_once(run_id, pairs, seed, batch_size, enroll_augment=False, limit=0):
    """跑一遍 enroll_infer(qwen), 输出 stability_matrix/<run_id>.json。已存在则跳过。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, f"{run_id}.json")
    if os.path.exists(out_json):
        print(f"[skip] {run_id} 已存在")
        return out_json
    pairs = _subset_pairs(run_id, pairs, limit)
    cmd = [_PY_MAIN, os.path.join(_HERE, "enroll_infer.py"),
           "--pairs", pairs, "--out-json", out_json,
           "--always-generate", "--reject-threshold", str(SIM_THR),
           "--asr-backend", "qwen", "--seed", str(seed),
           "--asr-batch-size", str(batch_size)]
    if enroll_augment:
        cmd += ["--enroll-augment"]
    print(f"[run] {run_id} (seed={seed}, batch={batch_size}, aug={enroll_augment})")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"{run_id} 失败({r.returncode}): {r.stderr[-800:]}")
    print(f"      done {run_id} ({dt:.0f}s)")
    return out_json


def phase_A(runs, seed, batch, limit):
    for r in range(runs):
        run_once(f"A_s{seed}_r{r}", PAIRS, seed, batch, False, limit)


def main():
    ap = argparse.ArgumentParser(description="稳定性/鲁棒性测试编排器")
    ap.add_argument("--phase", required=True, choices=["A", "B1", "B2", "B3", "B4", "all"])
    ap.add_argument("--runs", type=int, default=10, help="A 阶段遍数(默认10)")
    ap.add_argument("--seed", type=int, default=42, help="A 阶段种子(默认42)")
    ap.add_argument("--batch", type=int, default=16, help="A 阶段 batch(默认16)")
    ap.add_argument("--limit", type=int, default=0, help="每遍只跑前 N 条(0=全量, dry-run 用 5)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.phase == "A":
        phase_A(args.runs, args.seed, args.batch, args.limit)
    elif args.phase in ("B1", "B2", "B3", "B4", "all"):
        print(f"[TODO] phase {args.phase} 在 Task 5/6 实现")
    print("[done] stability_test phase", args.phase)


if __name__ == "__main__":
    main()
