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
from repro import set_global_seed  # 可复现性: 主进程种子(子进程各自 set_global_seed)
from text_utils import is_valid_command  # content_gate(2026-07-08): 转写内容有效性二次拒

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
    with open(pairs_json, encoding="utf-8") as f:
        rows = json.load(f)
    return [(r["enrollment"], r["recognition"]) for r in rows]


def expand_inputs(args):
    """把 CLI 输入展开为 [(enrollment, recognition), ...] 统一列表。"""
    if args.pairs:
        return load_pairs(args.pairs)
    recs = sorted(glob.glob(os.path.join(args.recognition_folder, "*.wav")))
    return [(args.enrollment, r) for r in recs]


def decide_reject(max_sim, llm_verdict, strategy, sim_thr, use_llm,
                  text="", use_content_gate=False):
    """融合拒识决策。返回 True=拒识。
    - 无 LLM 或 sim_only: 拒 iff max_sim < sim_thr           （保底走这条）
    - llm_only:           拒 iff llm != accept
    - llm_or_sim(默认):   拒 iff (llm != accept) AND (max_sim < sim_thr)
                          ⚠️ 命名历史遗留，实为 AND：LLM=accept 一票放过，故 LLM 只能
                          【减拒】不能【加拒】（GAP4 证伪"三路融合"强项定位，答辩勿列）。
    - content_gate(2026-07-08): sim≥thr 的 accept 若转写内容非有效家居指令(is_valid_command=False)
                          → 加拒。独立加拒通道(对 neg 提 RR, pos 侧顺带拒幻觉灾难降 CER), 不改原 sim/llm 逻辑。
                          hold-out 验证 val +1.6 分(L 不敏感 18-30 全正, bootstrap CI p5>0), 默认关(--content-gate 开)。
    """
    # content_gate 独立加拒通道: sim 过线但转写内容非指令 → 拒(只在 sim≥thr 时加拒, 不影响 sim<thr 的原决策)
    if use_content_gate and max_sim >= sim_thr and not is_valid_command(text):
        return True
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


def run_noise_classify(rec_dir, out_json, seed=42, py=PY_MAIN):
    """阶段0: 估每条噪声类型 -> out_json(list[{file,atten_lim_db,...}])。"""
    return _run([os.path.join(HERE, "noise_classify.py"),
                 "--in-dir", rec_dir, "--out", out_json,
                 "--seed", str(seed)], py)


def run_se_bucket(in_dir, out_dir, atten_db, seed=42, py=PY_SE):
    """阶段1(单桶): se_denoise 对 in_dir 全体用统一 atten_db 降噪 -> out_dir。"""
    return _run([os.path.join(HERE, "se_denoise.py"),
                 "--in-dir", in_dir, "--out-dir", out_dir,
                 "--atten-lim-db", str(atten_db), "--seed", str(seed)], py)


def run_enroll_infer(enrollment, rec_dir, out_json, device, sim_thr, seed=42, py=PY_MAIN):
    """阶段2: enroll声纹锁定+diar+DiCoW批量转写 -> out_json(list[{...}])。"""
    return _run([os.path.join(HERE, "enroll_infer.py"),
                 "--enrollment", enrollment, "--recognition-folder", rec_dir,
                 "--out-json", out_json, "--always-generate",
                 "--reject-threshold", str(sim_thr),
                 "--device", device, "--seed", str(seed)], py)


def run_enroll_infer_pairs(pairs_json, out_json, device, sim_thr,
                           enroll_augment=False, aug_snrs="10,5,0", aug_noise_dir=None,
                           asr_backend="dicow", seed=42, py=PY_MAIN):
    """阶段2(批量化, datasetA 用): enroll_infer --pairs 单进程跑所有对,模型加载1次。
    绕过"按 enrollment 分组多次 subprocess"的瓶颈(datasetA 每条 enr 不同 → 1838 次重载)。
    enroll_augment=True 透传 --enroll-augment(干净+多档加噪 emb 均值,提 babble 鲁棒)。
    aug_noise_dir 指定 babble 噪声池目录(比默认白噪更对症真实 babble)。"""
    cmd = [os.path.join(HERE, "enroll_infer.py"),
           "--pairs", pairs_json,
           "--out-json", out_json, "--always-generate",
           "--reject-threshold", str(sim_thr),
           "--device", device, "--seed", str(seed)]
    if asr_backend != "dicow":
        cmd += ["--asr-backend", asr_backend]
    if enroll_augment:
        cmd += ["--enroll-augment", "--aug-snrs", aug_snrs]
        if aug_noise_dir:
            cmd += ["--aug-noise-dir", aug_noise_dir]
    return _run(cmd, py)


def run_llm(infer_json, out_json, device, seed=42, py=PY_LLM):
    """阶段3: llm_reject 对每条 transcript 判 accept/reject。"""
    return _run([os.path.join(HERE, "llm_reject.py"),
                 "--infer-json", infer_json, "--out-json", out_json,
                 "--device", device, "--seed", str(seed)], py)


def main():
    ap = argparse.ArgumentParser(description="标准化推理: enrollment+recognition -> result.json+timing.json")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs", help="[{enrollment,recognition}] manifest(一对一)")
    g.add_argument("--enrollment", help="目标说话人参考 wav(配 --recognition-folder)")
    ap.add_argument("--recognition-folder", help="识别 wav 目录(配 --enrollment,一对多)")
    ap.add_argument("--no-se", action="store_true", help="跳过 SE 条件化降噪")
    ap.add_argument("--no-llm", action="store_true", help="跳过 LLM 拒识")
    ap.add_argument("--sim-thr", type=float, default=0.2, help="声纹拒识阈值(T20 最优)")
    ap.add_argument("--content-gate", action="store_true",
                    help="开 content_gate: sim≥thr 的 accept 若转写非有效家居指令则加拒"
                         "(hold-out val +1.6分证泛化, 默认关; run_baodi 用 BAODI_GATE=1 开)")
    ap.add_argument("--enroll-augment", action="store_true",
                    help="enrollment 加噪增强(干净+多档加噪 emb 均值,提 babble 下声纹鲁棒 → 提 sim)")
    ap.add_argument("--aug-snrs", default="10,5,0", help="enrollment 加噪增强 SNR 档(逗号分)")
    ap.add_argument("--aug-noise-dir", default=None,
                    help="babble 噪声池目录(默认 None; 比白噪更对症真实 babble)")
    ap.add_argument("--strategy", default="llm_or_sim",
                    choices=["llm_or_sim", "sim_only", "llm_only"],
                    help="融合策略（⚠️ llm_or_sim 实为 AND，LLM 只减拒不加拒；保底用 sim_only）")
    ap.add_argument("--asr-backend", default="dicow", choices=["dicow", "vanilla", "qwen", "firered"],
                    help="ASR 后端(透传 enroll_infer): dicow(fallback) / vanilla(主线) / qwen(Qwen3-ASR 中文原生, 含拒CER腿+4.29) / firered(FireRedASR-AED-L 中文原生, RTF 更优备选)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "submit_out"))
    ap.add_argument("--work-dir", default=None, help="中间产物(默认 <out-dir>/_work)")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部)")
    ap.add_argument("--seed", type=int, default=42, help="全局种子(透传 4 子进程 + 主进程)")
    args = ap.parse_args()
    set_global_seed(args.seed)  # 可复现性: 主进程 random/numpy(子进程各自 set_global_seed)

    if not args.pairs and not args.recognition_folder:
        ap.error("--enrollment 需配 --recognition-folder")
    use_llm = not args.no_llm
    use_se = not args.no_se

    # 保底守卫（GAP3，memory baodi-config-no-llm）：裸调默认 flag
    # （LLM ON / sim_thr=0.2 / strategy=llm_or_sim）→ RTF~1.0 + neg RR~0.77 双崩。
    # ⚠️ 统一 thr=0.27(B 集, T27 对抗验证推荐)< 0.35 守卫阈值 → 必须经 run_baodi.sh B
    # （export BAODI_OK=1 opt-in）或显式 BAODI_OK=1, 已由对抗验证背书非裸调灾难。
    if not os.environ.get("BAODI_OK") and (use_llm or args.sim_thr < 0.35):
        ap.error(
            "检测到非保底配置（LLM ON 或 sim_thr<0.35），裸调默认即灾难"
            "（RTF~1.0 + neg RR~0.77，见 memory baodi-config-no-llm）。\n"
            "  A 集分thr保底: bash code/run_baodi.sh pos|neg [thr]\n"
            "  B 集统一thr(推荐0.27, T27对抗验证): bash code/run_baodi.sh B [thr]\n"
            "  实验裸调: BAODI_OK=1 python code/submit_infer.py ... （显式 opt-in）"
        )
    if use_llm and not os.path.exists(PY_LLM):
        ap.error(f"开 LLM 但 {PY_LLM} 不存在（.venv_llm 未部署）；保底请加 --no-llm。")
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
    # noise_map: basename -> (est_noise, atten_lim_db)。SE 跳过时保持空 → result 字段为 null。
    noise_map = {}
    rec_for_enroll = rec_in
    if use_se:
        t0 = time.perf_counter()
        noise_est = os.path.join(work_dir, "noise_est.json")
        phases.setdefault("noise_classify", {})["wall_sec"] = round(run_noise_classify(rec_in, noise_est, args.seed), 3)
        with open(noise_est, encoding="utf-8") as f:
            rows = json.load(f)
        # 把 noise_classify 输出 join 回每条(按 basename),供阶段4 result 组装
        noise_map = {r["file"]: (r.get("est_noise"), r.get("atten_lim_db")) for r in rows}
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
            se_wall += run_se_bucket(bin_, bout, atten, args.seed)
            for f in files:
                s = os.path.join(bout, f)
                if os.path.exists(s):
                    shutil.copy(s, os.path.join(se_out, f))
        phases["se"] = {"wall_sec": round(se_wall, 3), "n": sum(len(v) for v in buckets.values())}
        rec_for_enroll = se_out
        t1 = time.perf_counter()
        print(f"[se] 阶段0+1 用时 {t1-t0:.1f}s ({len(rows)} 条, 桶={list(buckets.keys())})")

    # --- 阶段2: enroll_infer 转写(单进程批量化: 所有对一次喂, 模型加载1次) ---
    # 写 pairs manifest(enrollment=原enr 路径, recognition=work/rec_in/uttN.wav)。
    # enroll_infer --pairs 内部按 enr 路径缓存 enroll_emb, 同说话人/同文件复用。
    enroll_pairs = os.path.join(work_dir, "enroll_pairs.json")
    with open(enroll_pairs, "w", encoding="utf-8") as f:
        json.dump([{"enrollment": enr, "recognition": dst} for enr, rec, dst in rec_paths],
                  f, ensure_ascii=False)
    enroll_all = os.path.join(work_dir, "enroll_all.json")
    e_wall = run_enroll_infer_pairs(enroll_pairs, enroll_all, args.device, args.sim_thr,
                                    args.enroll_augment, args.aug_snrs, args.aug_noise_dir,
                                    args.asr_backend, args.seed)
    with open(enroll_all, encoding="utf-8") as f:
        all_rows = json.load(f)
    sum_rtf = sum(float(r.get("rtf", 0.0) or 0.0) for r in all_rows)
    n_rtf = len(all_rows)
    phases["enroll_diar_dicow"] = {"wall_sec": round(e_wall, 3),
                                   "mean_rtf": round(sum_rtf / n_rtf, 4) if n_rtf else None}

    # 汇总 enroll 输出(utt_id -> row)
    enr_map = {}
    for r in all_rows:
        uid = utt_id_from_path(r.get("recognition", ""))
        enr_map[uid] = r

    # --- 阶段3: LLM 拒识 ---
    llm_map = {}
    if use_llm:
        llm_in = os.path.join(work_dir, "llm_in.json")
        llm_rows = [{"file": enr_map[uid].get("recognition", ""),
                     "text": enr_map[uid].get("transcript", "") or ""}
                    for uid in enr_map]
        with open(llm_in, "w", encoding="utf-8") as f:
            json.dump(llm_rows, f, ensure_ascii=False)
        llm_out = os.path.join(work_dir, "llm_out.json")
        l_wall = run_llm(llm_in, llm_out, args.device, args.seed)
        phases["llm"] = {"wall_sec": round(l_wall, 3)}
        with open(llm_out, encoding="utf-8") as f:
            lj = json.load(f)
        verdicts = lj["rows"] if isinstance(lj, dict) else lj   # llm_reject 输出 {rows:[...]}, 兼容裸 list
        for row in verdicts:
            llm_map[utt_id_from_path(row.get("file", ""))] = row.get("pred", "reject")

    # --- 阶段4: 融合 + 组装 result ---
    total_wall = time.perf_counter() - t_total0
    duration_infer_sec = sum(float(r.get("infer_sec", 0.0) or 0.0) for r in all_rows)
    items, per_utt = [], []
    for enr, rec, dst in rec_paths:
        uid = utt_id_from_path(dst)
        r = enr_map.get(uid, {})
        max_sim = float(r.get("max_sim", 0.0) or 0.0)
        llm_v = llm_map.get(uid, "reject") if use_llm else "accept"
        rejected = decide_reject(max_sim, llm_v, args.strategy, args.sim_thr, use_llm,
                                 text=r.get("transcript", "") or "",
                                 use_content_gate=args.content_gate)
        text = "" if rejected else (r.get("transcript", "") or "")
        # noise_type/atten_lim_db 来自阶段0的 noise_classify 输出(join by basename);
        # --no-se 时 noise_map 为空 → (None, None) → null(合理: SE 跳过, 未估噪声)。
        nt, atten = noise_map.get(os.path.basename(dst), (None, None))
        items.append({
            "utt_id": uid, "enrollment": enr, "recognition": rec,
            "text": text, "rejected": rejected, "score": round(max_sim, 4),
            "max_sim": round(max_sim, 4), "llm_verdict": llm_v if use_llm else None,
            "noise_type": nt, "atten_lim_db": atten,
            "infer_sec": round(float(r.get("infer_sec", 0.0) or 0.0), 3),
            "batch_size": r.get("batch_size", 1),  # 可复现性: per-utt batch(FAQ 核查可见)
            "peak_mem_mib": r.get("peak_mem_mib"),  # 可复现性: per-utt 峰值显存(FAQ 核查硬要求 5)
            "diar_fail": bool(r.get("error")),
        })
        dur = audio_duration_s(dst)
        per_utt.append({"utt_id": uid, "audio_sec": round(dur, 3),
                        "wall_sec": None, "rtf": round(float(r.get("rtf", 0.0) or 0.0), 4)})

    cfg = {"se": use_se, "llm": use_llm, "strategy": args.strategy if use_llm else "sim_only",
           "sim_thr": args.sim_thr, "device": args.device, "asr_backend": args.asr_backend}
    result = build_result(items, cfg)
    timing = build_timing(args.device, len(items), total_audio, total_wall, phases, per_utt)
    timing["duration_infer_sec"] = round(duration_infer_sec, 3)

    rj = os.path.join(args.out_dir, "result.json")
    tj = os.path.join(args.out_dir, "timing.json")
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(tj, "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)
    n_rej = sum(1 for it in items if it["rejected"])
    print(f"\n[done] {len(items)} 条 ({n_rej} 拒识) -> {rj}")
    print(f"       overall_rtf={timing['overall_rtf']} (audio={total_audio:.1f}s wall={total_wall:.1f}s)")
    print(f"       phases={ {k: v.get('wall_sec') for k, v in phases.items()} }")
    # 后置 sanity check（GAP3）：防崩盘结果静默交付
    if timing.get("overall_rtf") and timing["overall_rtf"] > 0.6:
        print(f"[WARN] overall_rtf={timing['overall_rtf']} > 0.6，疑误开 LLM/SE（保底应 ~0.24）")
    pairs_name = os.path.basename(args.pairs or "").lower()
    if "neg" in pairs_name and len(items) and n_rej / len(items) < 0.9:
        print(f"[WARN] neg 拒识率 {n_rej/len(items):.2%} < 90%，低于保底预期（thr 太低？）")


if __name__ == "__main__":
    main()
