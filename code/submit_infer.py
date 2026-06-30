"""标准化推理脚本: enrollment + recognition -> result.json + timing.json

纯 stdlib subprocess 顶层编排器,零侵入复用 noise_classify / se_denoise /
enroll_infer / llm_reject。详见 spec:
docs/superpowers/specs/2026-07-01-submit-infer-and-deliverables-design.md
"""
import os
import json
import glob
import time
import shutil
import wave
import argparse
import subprocess
import contextlib
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))      # code/
ROOT = os.path.dirname(HERE)                            # 项目根
PY_MAIN = os.path.join(HERE, ".venv", "Scripts", "python.exe")
PY_SE   = os.path.join(HERE, ".venv_se", "Scripts", "python.exe")
PY_LLM  = os.path.join(ROOT, ".venv_llm", "Scripts", "python.exe")


def utt_id_from_path(p):
    """文件名去扩展名作 utt_id。'E:/x/rec_001.wav' -> 'rec_001'。"""
    return os.path.splitext(os.path.basename(p))[0]


def audio_duration_s(p):
    """wav 时长(秒),纯 stdlib wave。读失败返回 0.0。"""
    try:
        with contextlib.closing(wave.open(p, "rb")) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def load_pairs(pairs_json):
    """读 --pairs manifest: [{enrollment, recognition}, ...] -> [(enr, rec), ...]。"""
    rows = json.load(open(pairs_json, encoding="utf-8"))
    return [(r["enrollment"], r["recognition"]) for r in rows]


def expand_inputs(args):
    """把 CLI 输入展开为 [(enrollment, recognition), ...] 统一列表。"""
    if args.pairs:
        return load_pairs(args.pairs)
    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav")))
    return [(args.enrollment, r) for r in recs]


def decide_reject(max_sim, llm_verdict, strategy, sim_thr, use_llm):
    """融合拒识决策。返回 True=拒识。
    - 无 LLM 或 sim_only: 拒 iff max_sim < sim_thr
    - llm_only:           拒 iff llm != accept
    - llm_or_sim(默认):   拒 iff (llm != accept) AND (max_sim < sim_thr)
    """
    if not use_llm or strategy == "sim_only":
        return max_sim < sim_thr
    if strategy == "llm_only":
        return llm_verdict != "accept"
    return llm_verdict != "accept" and max_sim < sim_thr


def bucket_by_atten(noise_rows):
    """按 atten_lim_db 把识别音频分桶 -> {atten: [basename, ...]}。"""
    buckets = {}
    for r in noise_rows:
        a = int(r.get("atten_lim_db", 0))
        buckets.setdefault(a, []).append(r["file"])
    return buckets


def build_result(items, config):
    """组装 result.json schema(见 spec §8)。"""
    return {
        "task_id": "XH-202615",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "n_utt": len(items),
        "results": items,
    }


def build_timing(device, n_utt, total_audio_sec, total_wall_sec, phases, per_utt):
    """组装 timing.json schema(见 spec §8)。overall_rtf = wall/audio。"""
    return {
        "device": device,
        "n_utt": n_utt,
        "total_audio_sec": round(total_audio_sec, 3),
        "total_wall_sec": round(total_wall_sec, 3),
        "overall_rtf": round(total_wall_sec / total_audio_sec, 4) if total_audio_sec else None,
        "phases": phases,
        "per_utt": per_utt,
    }


def _run(cmd, py):
    """subprocess 跑 [py] + cmd,返回 (wall_sec)。失败抛 RuntimeError。"""
    t0 = time.perf_counter()
    full = [py] + cmd
    print(f"  [run] {os.path.basename(py)} {' '.join(cmd[:2])} ...")
    r = subprocess.run(full, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"subprocess failed ({r.returncode}): {r.stderr[-500:]}")
    return dt


def run_noise_classify(rec_dir, out_json, py=PY_MAIN):
    """阶段0: 估每条噪声类型 -> out_json(list[{file,atten_lim_db,...}])。"""
    return _run([os.path.join(HERE, "noise_classify.py"),
                 "--in-dir", rec_dir, "--out", out_json], py)


def run_se_bucket(in_dir, out_dir, atten_db, py=PY_SE):
    """阶段1(单桶): se_denoise 对 in_dir 全体用统一 atten_db 降噪 -> out_dir。"""
    return _run([os.path.join(HERE, "se_denoise.py"),
                 "--in-dir", in_dir, "--out-dir", out_dir,
                 "--atten-lim-db", str(atten_db)], py)


def run_enroll_infer(enrollment, rec_dir, out_json, device, sim_thr, py=PY_MAIN):
    """阶段2: enroll声纹锁定+diar+DiCoW批量转写 -> out_json(list[{...}])。"""
    return _run([os.path.join(HERE, "enroll_infer.py"),
                 "--enrollment", enrollment, "--recognition-folder", rec_dir,
                 "--out-json", out_json, "--always-generate",
                 "--reject-threshold", str(sim_thr),
                 "--device", device], py)


def run_llm(infer_json, out_json, device, py=PY_LLM):
    """阶段3: llm_reject 对每条 transcript 判 accept/reject。"""
    return _run([os.path.join(HERE, "llm_reject.py"),
                 "--infer-json", infer_json, "--out-json", out_json,
                 "--device", device], py)


def main():
    ap = argparse.ArgumentParser(description="标准化推理: enrollment+recognition -> result.json+timing.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", help="[{enrollment,recognition}] manifest(一对一)")
    g.add_argument("--enrollment", help="目标说话人参考 wav(配 --recognition-folder)")
    ap.add_argument("--recognition-folder", help="识别 wav 目录(配 --enrollment,一对多)")
    ap.add_argument("--no-se", action="store_true", help="跳过 SE 条件化降噪")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 拒识")
    ap.add_argument("--sim-thr", type=float, default=0.2, help="声纹拒识阈值(T20 最优)")
    ap.add_argument("--strategy", default="llm_or_sim",
                    choices=["llm_or_sim", "sim_only", "llm_only"], help="融合策略")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "submit_out"))
    ap.add_argument("--work-dir", default=None, help="中间产物(默认 <out-dir>/_work)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部)")
    args = ap.parse_args()

    if not args.pairs and not args.recognition_folder:
        ap.error("--enrollment 需配 --recognition-folder")
    use_llm = not args.no_llm
    use_se = not args.no_se
    work_dir = args.work_dir or os.path.join(args.out_dir, "_work")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    pairs = expand_inputs(args)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    # 识别音频目录视图(复制到 work/rec_in,统一喂下游)
    rec_in = os.path.join(work_dir, "rec_in")
    os.makedirs(rec_in, exist_ok=True)
    rec_paths = []
    for i, (enr, rec) in enumerate(pairs):
        dst = os.path.join(rec_in, f"utt{i:04d}_{utt_id_from_path(rec)}.wav")
        if not os.path.exists(dst):
            shutil.copy(rec, dst)
        rec_paths.append((enr, rec, dst))
    t_total0 = time.perf_counter()
    phases = {}
    total_audio = sum(audio_duration_s(dst) for _, _, dst in rec_paths)

    # --- 阶段0+1: SE 条件化 ---
    rec_for_enroll = rec_in
    if use_se:
        t0 = time.perf_counter()
        noise_est = os.path.join(work_dir, "noise_est.json")
        phases.setdefault("noise_classify", {})["wall_sec"] = round(run_noise_classify(rec_in, noise_est), 3)
        rows = json.load(open(noise_est, encoding="utf-8"))
        buckets = bucket_by_atten(rows)
        se_out = os.path.join(work_dir, "se_out")
        os.makedirs(se_out, exist_ok=True)
        se_wall = 0.0
        for atten, files in buckets.items():
            bin_ = os.path.join(work_dir, f"se_in_{atten}")
            os.makedirs(bin_, exist_ok=True)
            for f in files:
                src = os.path.join(rec_in, f)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(bin_, f))
            bout = os.path.join(work_dir, f"se_out_{atten}")
            se_wall += run_se_bucket(bin_, bout, atten)
            for f in files:
                s = os.path.join(bout, f)
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(se_out, f))
        phases["se"] = {"wall_sec": round(se_wall, 3), "n": sum(len(v) for v in buckets.values())}
        rec_for_enroll = se_out
        t1 = time.perf_counter()
        print(f"[se] 阶段0+1 用时 {t1-t0:.1f}s ({len(rows)} 条, 桶={list(buckets.keys())})")

    # --- 阶段2: enroll_infer 转写(按 enrollment 分组,每组一次批量) ---
    from collections import defaultdict
    groups = defaultdict(list)
    for enr, rec, dst in rec_paths:
        groups[enr].append(dst)
    enroll_jsons = []
    e_wall = 0.0
    sum_rtf = 0.0
    n_rtf = 0
    for gi, (enr, dsts) in enumerate(groups.items()):
        gdir = os.path.join(work_dir, f"enroll_g{gi}")
        os.makedirs(gdir, exist_ok=True)
        for d in dsts:
            shutil.copy(d, os.path.join(gdir, os.path.basename(d)))
        out_json = os.path.join(work_dir, f"enroll_g{gi}.json")
        e_wall += run_enroll_infer(enr, gdir, out_json, args.device, args.sim_thr)
        rows = json.load(open(out_json, encoding="utf-8"))
        for r in rows:
            sum_rtf += float(r.get("rtf", 0.0) or 0.0)
            n_rtf += 1
        enroll_jsons.append((enr, out_json))
    phases["enroll_diar_dicow"] = {"wall_sec": round(e_wall, 3),
                                   "mean_rtf": round(sum_rtf / n_rtf, 4) if n_rtf else None}

    # 汇总 enroll 输出(utt_id -> row)
    enr_map = {}
    for enr, out_json in enroll_jsons:
        for r in json.load(open(out_json, encoding="utf-8")):
            uid = utt_id_from_path(r.get("recognition", ""))
            enr_map[uid] = r

    # --- 阶段3: LLM 拒识 ---
    llm_map = {}
    if use_llm:
        llm_in = os.path.join(work_dir, "llm_in.json")
        llm_rows = [{"file": enr_map[uid].get("recognition", ""),
                     "text": enr_map[uid].get("transcript", "") or ""}
                    for uid in enr_map]
        json.dump(llm_rows, open(llm_in, "w", encoding="utf-8"), ensure_ascii=False)
        llm_out = os.path.join(work_dir, "llm_out.json")
        l_wall = run_llm(llm_in, llm_out, args.device)
        phases["llm"] = {"wall_sec": round(l_wall, 3)}
        for row in json.load(open(llm_out, encoding="utf-8")):
            llm_map[utt_id_from_path(row.get("file", ""))] = row.get("pred", "reject")

    # --- 阶段4: 融合 + 组装 result ---
    total_wall = time.perf_counter() - t_total0
    items, per_utt = [], []
    for enr, rec, dst in rec_paths:
        uid = utt_id_from_path(dst)
        r = enr_map.get(uid, {})
        max_sim = float(r.get("max_sim", 0.0) or 0.0)
        llm_v = llm_map.get(uid, "reject") if use_llm else "accept"
        rejected = decide_reject(max_sim, llm_v, args.strategy, args.sim_thr, use_llm)
        text = "" if rejected else (r.get("transcript", "") or "")
        items.append({
            "utt_id": uid, "enrollment": enr, "recognition": rec,
            "text": text, "rejected": rejected, "score": round(max_sim, 4),
            "max_sim": round(max_sim, 4), "llm_verdict": llm_v if use_llm else None,
            "noise_type": r.get("noise_type"), "atten_lim_db": r.get("atten_lim_db"),
            "diar_fail": bool(r.get("error")),
        })
        dur = audio_duration_s(dst)
        per_utt.append({"utt_id": uid, "audio_sec": round(dur, 3),
                        "wall_sec": None, "rtf": round(float(r.get("rtf", 0.0) or 0.0), 4)})

    cfg = {"se": use_se, "llm": use_llm, "strategy": args.strategy if use_llm else "sim_only",
           "sim_thr": args.sim_thr, "device": args.device}
    result = build_result(items, cfg)
    timing = build_timing(args.device, len(items), total_audio, total_wall, phases, per_utt)

    rj = os.path.join(args.out_dir, "result.json")
    tj = os.path.join(args.out_dir, "timing.json")
    json.dump(result, open(rj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(timing, open(tj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_rej = sum(1 for it in items if it["rejected"])
    print(f"\n[done] {len(items)} 条 ({n_rej} 拒识) -> {rj}")
    print(f"       overall_rtf={timing['overall_rtf']} (audio={total_audio:.1f}s wall={total_wall:.1f}s)")
    print(f"       phases={ {k: v.get('wall_sec') for k, v in phases.items()} }")


if __name__ == "__main__":
    main()
