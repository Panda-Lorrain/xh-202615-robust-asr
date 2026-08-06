"""一键提交装配: run_baodi(pos+neg) → to_submission(pos+neg) → 合并单文件 submission.json。

产出官方格式(2026-08-06 FAQ):
  {"result":{"results":[{"id","content","label","cer"}],"final_cer","avg_rr","duration"}}

- final_cer = pos 累计池 CER(neg 不评 CER), 取 pos submission 的 final_cer
- avg_rr    = neg 正确拒识率(FAQ 第1条), 取 neg submission 的 avg_rr
- duration  = pos total_wall + neg total_wall(端到端含模型加载, batch=1)
- id 加 pos_/neg_ 前缀避免 cmd_N 在 1000-1363 撞键(⚠️ 占位, 待主办方确认 id 方案)

用法:
  python code/build_submission.py                       # 全量一键(scene_route 开, thr0.27)
  python code/build_submission.py --skip-infer          # 跳过推理, 只装配(已有 out_{pos,neg}_baodi)
  python code/build_submission.py --no-scene-route      # 关 scene_route 走老主线(对比用)

⚠️ duration 是本机实测(device 见 timing.json), 非官方 3090 真值 —— 提交前须租 3090 重跑。
"""
import os, sys, json, subprocess, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = (os.path.join(HERE, ".venv", "Scripts", "python.exe") if os.name == "nt"
      else os.path.join(HERE, ".venv", "bin", "python"))


def run(cmd, cwd=ROOT, env=None):
    print(f"[run] {' '.join(cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, env=env)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"命令失败({r.returncode}): {' '.join(cmd)}")
    print(f"     done {dt:.1f}s")
    return dt


def main():
    ap = argparse.ArgumentParser(description="一键装配官方 submission.json(scene_route 提交主线)")
    ap.add_argument("--thr", type=float, default=0.27, help="sim_thr(默认 0.27, T27 推荐)")
    ap.add_argument("--skip-infer", action="store_true", help="跳过 run_baodi, 用已有 out_{pos,neg}_baodi")
    ap.add_argument("--no-scene-route", action="store_true", help="关 scene_route(BAODI_SCENE_ROUTE=0, 老主线对比)")
    ap.add_argument("--pos-out", default=os.path.join(HERE, "out_pos_baodi"))
    ap.add_argument("--neg-out", default=os.path.join(HERE, "out_neg_baodi"))
    ap.add_argument("--out", default=os.path.join(HERE, "submission_final.json"))
    ap.add_argument("--no-id-prefix", action="store_true", help="不加 pos_/neg_ 前缀(⚠️ cmd_N 会撞键)")
    args = ap.parse_args()

    scene_env = "0" if args.no_scene_route else "1"

    # 阶段1: run_baodi pos + neg(thr=0.27 提交口径; scene_route 默认开; run_baodi.sh 内部 source setenv 设 HF_HOME/MODEL_*)
    if not args.skip_infer:
        env = dict(os.environ, BAODI_SCENE_ROUTE=scene_env)
        run(["bash", os.path.join(HERE, "run_baodi.sh"), "pos", str(args.thr)], env=env)
        run(["bash", os.path.join(HERE, "run_baodi.sh"), "neg", str(args.thr)], env=env)

    # 阶段2: to_submission pos + neg(各自产出 submission_{pos,neg}.json, duration 取 total_wall_sec)
    pos_result = os.path.join(args.pos_out, "result.json")
    neg_result = os.path.join(args.neg_out, "result.json")
    pos_pairs = os.path.join(HERE, "pos_pairs_datasetA.json")
    neg_pairs = os.path.join(HERE, "neg_pairs_datasetA.json")
    for result, pairs, out_dir, tag in [
        (pos_result, pos_pairs, args.pos_out, "pos"),
        (neg_result, neg_pairs, args.neg_out, "neg"),
    ]:
        if not os.path.exists(result):
            sys.exit(f"[error] {result} 不存在; 先去掉 --skip-infer 跑 run_baodi {tag} {args.thr}, 或检查 --pos-out/--neg-out")
        run([PY, os.path.join(HERE, "to_submission.py"),
             "--result-json", result, "--pairs", pairs,
             "--out", os.path.join(out_dir, f"submission_{tag}.json")])

    # 阶段3: 合并 pos+neg 单文件(final_cer 取 pos 累计池; avg_rr 取 neg; duration = pos wall + neg wall)
    pos_sub = json.load(open(os.path.join(args.pos_out, "submission_pos.json"), encoding="utf-8"))["result"]
    neg_sub = json.load(open(os.path.join(args.neg_out, "submission_neg.json"), encoding="utf-8"))["result"]

    merged = []
    for tag, sub in [("pos", pos_sub), ("neg", neg_sub)]:
        for row in sub["results"]:
            r = dict(row)
            if not args.no_id_prefix:
                r["id"] = f"{tag}_{row['id']}"   # 占位前缀避免 cmd_N 撞键(待主办方确认 id 方案)
            merged.append(r)

    final = {"result": {
        "results": merged,
        "final_cer": pos_sub["final_cer"],                                              # pos 累计池(neg 不评 CER)
        "avg_rr": neg_sub["avg_rr"],                                                    # neg 正确拒识率(FAQ 第1条)
        "duration": round((pos_sub.get("duration") or 0) + (neg_sub.get("duration") or 0), 3),  # pos+neg 端到端 wall
    }}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # 诚实标注 duration 的 device(本机 4060 / 3090 实测)
    pos_timing = os.path.join(args.pos_out, "timing.json")
    dev = json.load(open(pos_timing, encoding="utf-8")).get("device", "?") if os.path.exists(pos_timing) else "?"
    n_pos, n_neg = len(pos_sub["results"]), len(neg_sub["results"])
    print(f"\n[done] {args.out}")
    print(f"  条数: pos {n_pos} + neg {n_neg} = {n_pos + n_neg}")
    print(f"  final_cer(累计池) = {final['result']['final_cer']}   (scene_route 开对标 ~0.59 / 关 ~0.62)")
    print(f"  avg_rr            = {final['result']['avg_rr']}    (对标 ~0.94)")
    print(f"  duration(端到端)  = {final['result']['duration']}s (device={dev}, ⚠️ 非 3090 官方真值, 提交须租 3090 实测)")


if __name__ == "__main__":
    main()
