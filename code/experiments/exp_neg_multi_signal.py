"""exp_neg_multi_signal.py — neg 474 条多信号拒识真实 RR 损失测量.

【任务】task4 多信号拒识在 pos 上斩获 ΔCER-22% (含拒 0.5937→0.4468), 但 neg 漏拒率
当时只有代理区间 [0% (vanilla), 8.3% (pos 另一路)]. 本脚本跑 SepFormer+两路 ASR 真测
neg 上"被 gate+kw 补救接受"的比例 = 多信号 RR 损失, 用于判 net win.

【多信号规则 (复用 task4 exp_multi_signal_reject.py)】
  - 主线: wespeaker max_sim ≥ 0.27 → 接受 (主线漏拒 baseline, 与多信号无关)
  - max_sim < 0.27 (本应拒) → 看 SepFormer 分离后 heuristic 选的 target 路:
      rescue_gate_kw(text) 通过 (is_valid_command AND has_home_kw) → 补救接受 (neg 漏拒来源)
      否则 → 拒
  - 多信号 RR = 1 − (主线接受 + rescue) / 474
  - RR 损失 (abs) = rescue 数 / 474

【net win 判据】
  - RR 损失 ≤3%   → 净 +4.7 分 (强 win, pos ΔCER-22% 大赚)
  - 3% < 损失 <8% → 边际, 收紧 rescue (加 len≥5 / 更严指令判别)
  - 损失 ≥8%      → 放弃或大改

【Phase】
  Phase1 (.venv GPU SepFormer): 474 recognition → 2 路 wav (srcA/srcB), resumable
  Phase2 (.venv_qwen subprocess): 批转 948 路
  Phase3 (CPU): heuristic 选 target 路 (cmd_score) + gate+kw rescue 判别 + 漏拒率 + 拆解

【复用】
  - exp_sepformer_qwen.{load_sepformer, separate}                (分离原语)
  - exp_multivoice_route.{route_heuristic, cmd_score}            (heuristic 选路)
  - exp_multi_signal_reject.{HOME_KW, rescue_gate_kw, has_home_kw, cn_len}  (rescue 规则)
  - qwen_asr_backend.py (--slice-dir --out --batch-size --seed)  (qwen 子进程)
  - text_utils.is_valid_command                                   (content_gate)

用法:
  code/.venv/Scripts/python.exe code/exp_neg_multi_signal.py --phase all
  code/.venv/Scripts/python.exe code/exp_neg_multi_signal.py --phase sep     # 仅 SepFormer
  code/.venv/Scripts/python.exe code/exp_neg_multi_signal.py --phase asr     # 仅 qwen
  code/.venv/Scripts/python.exe code/exp_neg_multi_signal.py --phase analyze # 仅分析

产物:
  code/runs/_neg_multi_signal/{slices/, _sep_meta.json, _uid2text.json, result.json}
"""
import os, sys, json, time, argparse, subprocess, glob, re, statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

OUT_DIR = os.path.join(_HERE, "runs", "_neg_multi_signal")
SLICE_DIR = os.path.join(OUT_DIR, "slices")
SEP_META_PATH = os.path.join(OUT_DIR, "_sep_meta.json")
UID2TEXT_PATH = os.path.join(OUT_DIR, "_uid2text.json")
RESULT_PATH = os.path.join(OUT_DIR, "result.json")

THR = 0.27
PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))
NEG_PAIRS = os.path.join(_HERE, "neg_pairs_datasetA.json")
NEG_VANILLA = os.path.join(_HERE, "runs", "out_neg_vanilla_full", "result.json")


def neg_uid_of(recognition_path: str) -> str:
    """recognition 路径 → uid (cmd_XXXX 文件名 stem)."""
    base = os.path.splitext(os.path.basename(recognition_path))[0]
    return base  # e.g. cmd_1000


def load_neg_data():
    """加载 neg 474 + 主线 max_sim. 返回 list[{uid, enrollment, recognition, kws_txt, max_sim}]."""
    pairs = json.load(open(NEG_PAIRS, encoding="utf-8"))
    vanilla = json.load(open(NEG_VANILLA, encoding="utf-8"))["results"]
    sim_map = {}
    for r in vanilla:
        m = re.search(r"cmd_\d+", r["recognition"])
        if m:
            sim_map[m.group(0)] = r.get("max_sim")
    out = []
    for p in pairs:
        uid = neg_uid_of(p["recognition"])
        out.append({
            "uid": uid,
            "enrollment": p["enrollment"],
            "recognition": p["recognition"],
            "kws_txt": p.get("kws_txt"),
            "max_sim": sim_map.get(uid),
        })
    return out


# ============ Phase 1: SepFormer 分离 (GPU) ============
def phase_sep(device, batch_size):
    """SepFormer 分离 474 neg recognition → 2 路 wav. Resumable."""
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

    import numpy as np, torch, librosa, soundfile as sf
    from exp_sepformer_qwen import load_sepformer, separate
    from repro import set_global_seed
    set_global_seed(42)

    os.makedirs(SLICE_DIR, exist_ok=True)
    samples = load_neg_data()

    # resumable meta
    if os.path.exists(SEP_META_PATH):
        meta = {m["uid"]: m for m in json.load(open(SEP_META_PATH, encoding="utf-8")).get("results", [])}
    else:
        meta = {}

    todo = []
    for s in samples:
        uid = s["uid"]
        sa = os.path.join(SLICE_DIR, f"{uid}__srcA.wav")
        sb = os.path.join(SLICE_DIR, f"{uid}__srcB.wav")
        if uid in meta and "error" not in meta[uid] and os.path.exists(sa) and os.path.exists(sb):
            continue
        todo.append(s)

    print(f"\n{'='*70}\n[Phase sep] SepFormer neg 待跑 {len(todo)} 条 (skip {len(samples)-len(todo)} 已完成)")
    if not todo:
        print("[sep] 无需补跑")
        return

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[load] SepFormer whamr16k → E:/hf_cache/sepformer-whamr16k")
    sep_model = load_sepformer(dev, "E:/hf_cache/sepformer-whamr16k")
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    t0 = time.time()
    total_audio_sec = 0.0
    n_fail = 0
    for n, s in enumerate(todo):
        uid, rec = s["uid"], s["recognition"]
        try:
            audio, sr = librosa.load(rec, sr=16000)
            total_audio_sec += len(audio) / sr
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]
            if n_src != 2:
                raise RuntimeError(f"SepFormer n_src={n_src} 非 2 路")
            slice_uids = []
            for i in range(n_src):
                suid = f"{uid}__src{chr(65+i)}"
                sf.write(os.path.join(SLICE_DIR, suid + ".wav"),
                         np.ascontiguousarray(sources[i].astype(np.float32)), 16000)
                slice_uids.append(suid)
            meta[uid] = {
                "uid": uid, "n_src": n_src, "slice_uids": slice_uids,
                "audio_sec": round(len(audio)/sr, 2),
            }
            if (n+1) % 25 == 0 or n == len(todo)-1:
                json.dump({"results": list(meta.values())},
                          open(SEP_META_PATH, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                print(f"  [{n+1}/{len(todo)}] {uid} ({time.time()-t0:.0f}s, audio={total_audio_sec:.0f}s)")
        except Exception as e:
            n_fail += 1
            print(f"  [{n+1}/{len(todo)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta[uid] = {"uid": uid, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    json.dump({"results": list(meta.values())},
              open(SEP_META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    dt = time.time() - t0
    print(f"\n[sep] 完成: {len(todo)-n_fail}/{len(todo)} 成功, 失败 {n_fail}, "
          f"耗时 {dt:.0f}s ({dt/60:.1f}min), RTF={dt/max(total_audio_sec,1):.3f}")


# ============ Phase 2: Qwen 批转 (subprocess) ============
def phase_asr(batch_size):
    print(f"\n{'='*70}\n[Phase asr] qwen 转写 {SLICE_DIR}")
    if not os.path.isdir(SLICE_DIR) or not glob.glob(os.path.join(SLICE_DIR, "*.wav")):
        print("[asr] 无切片, 跳过")
        return
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", SLICE_DIR, "--out", UID2TEXT_PATH,
           "--seed", "42", "--batch-size", str(batch_size)]
    print(f"[asr] subprocess: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[asr] 完成, 耗时 {dt:.0f}s ({dt/60:.1f}min), 输出 {UID2TEXT_PATH}")


# ============ Phase 3: 分析 ============
def phase_analyze():
    from exp_multivoice_route import route_heuristic, cmd_score
    from exp_multi_signal_reject import rescue_gate_kw, has_home_kw, cn_len, HOME_KW
    import text_utils as tu

    samples = load_neg_data()
    sep_meta = {}
    if os.path.exists(SEP_META_PATH):
        for m in json.load(open(SEP_META_PATH, encoding="utf-8")).get("results", []):
            sep_meta[m["uid"]] = m
    uid2text = {}
    if os.path.exists(UID2TEXT_PATH):
        uid2text = json.load(open(UID2TEXT_PATH, encoding="utf-8"))

    print(f"\n{'='*70}\n[Phase analyze] heuristic 选路 + gate+kw rescue")
    print(f"[data] neg {len(samples)} | sep_meta {len(sep_meta)} | uid2text {len(uid2text)}")

    per_sample = []
    n_total = len(samples)
    n_mainline_accept = 0   # max_sim ≥ 0.27 (主线漏拒)
    n_mainline_reject = 0   # max_sim < 0.27 (本应拒, 多信号作用池)
    n_sep_missing = 0       # SepFormer 失败/缺失
    n_rescued = 0           # max_sim<0.27 AND rescue_gate_kw(heuristic target) True → 多信号漏拒
    n_rescued_gate_only = 0 # 仅 gate 通过 (kw 不要求, 对照)
    # RR 工作点扫 (各 rescue_pred 强度)
    rescue_counts = {"gate_only": 0, "gate+kw": 0, "gate+kw+len>=3": 0,
                     "gate+kw+len>=4": 0, "gate+kw+len>=5": 0}

    def _rescue_gate_kw_len(text, min_cn_len):
        return bool(text and tu.is_valid_command(text) and has_home_kw(text) and cn_len(text) >= min_cn_len)

    rescued_samples = []
    low_sim_no_rescue_examples = []

    for s in samples:
        uid = s["uid"]
        max_sim = s["max_sim"]
        rec = s["recognition"]
        m = sep_meta.get(uid, {})
        slice_uids = m.get("slice_uids") or [f"{uid}__srcA", f"{uid}__srcB"]
        texts = [uid2text.get(su, "") for su in slice_uids]
        # 若 slice_uids 缺失/出错 → 文本空, 不影响 (gate False → 不 rescue)

        # heuristic 选 target 路 (cmd_score 文本判别, 复用 pos 同款)
        if texts and any(t.strip() for t in texts):
            per_src = [{"text": t} for t in texts]
            try:
                h_idx, h_reason = route_heuristic(per_src)
            except Exception:
                h_idx, h_reason = 0, "route_exception"
        else:
            h_idx, h_reason = 0, "both_empty"
        target_text = texts[h_idx] if h_idx < len(texts) else ""
        other_text = texts[1 - h_idx] if len(texts) == 2 else ""

        # baseline
        if max_sim is None:
            mainline_accept = False
        else:
            mainline_accept = max_sim >= THR
        if mainline_accept:
            n_mainline_accept += 1
        else:
            n_mainline_reject += 1

        # 多信号 rescue (主工作点 gate+kw)
        if not mainline_accept:
            # 工作点扫
            rescue_counts["gate_only"] += int(bool(target_text and tu.is_valid_command(target_text)))
            rescue_counts["gate+kw"] += int(rescue_gate_kw(target_text))
            rescue_counts["gate+kw+len>=3"] += int(_rescue_gate_kw_len(target_text, 3))
            rescue_counts["gate+kw+len>=4"] += int(_rescue_gate_kw_len(target_text, 4))
            rescue_counts["gate+kw+len>=5"] += int(_rescue_gate_kw_len(target_text, 5))

            if rescue_gate_kw(target_text):
                n_rescued += 1
                if tu.is_valid_command(target_text):
                    n_rescued_gate_only += 1
                # 命中的家居关键词
                hit_kw = [k for k in HOME_KW if k in target_text]
                rescued_samples.append({
                    "uid": uid,
                    "max_sim": round(max_sim, 4) if max_sim is not None else None,
                    "kws_txt": s.get("kws_txt"),
                    "recognition": rec,
                    "srcA_text": texts[0] if len(texts) > 0 else "",
                    "srcB_text": texts[1] if len(texts) > 1 else "",
                    "heuristic_idx": h_idx,
                    "heuristic_reason": h_reason,
                    "target_text": target_text,
                    "other_text": other_text,
                    "cmd_score_target": round(cmd_score(target_text), 3),
                    "cmd_score_other": round(cmd_score(other_text), 3) if other_text else None,
                    "target_cn_len": cn_len(target_text),
                    "hit_home_kw": hit_kw,
                    "gate_pass_target": bool(tu.is_valid_command(target_text)),
                })
            else:
                # 收低 sim 未 rescue 的样本 (前 10 例, 诊断用)
                if len(low_sim_no_rescue_examples) < 10 and target_text.strip():
                    low_sim_no_rescue_examples.append({
                        "uid": uid, "max_sim": round(max_sim, 4) if max_sim is not None else None,
                        "target_text": target_text[:80],
                        "gate_pass": bool(tu.is_valid_command(target_text)),
                        "has_kw": has_home_kw(target_text),
                    })

        if "error" in m:
            n_sep_missing += 1

        per_sample.append({
            "uid": uid,
            "max_sim": round(max_sim, 4) if max_sim is not None else None,
            "mainline_accept": mainline_accept,
            "heuristic_idx": h_idx,
            "heuristic_reason": h_reason,
            "target_text": target_text,
            "other_text": other_text,
            "rescued_gate_kw": (not mainline_accept) and rescue_gate_kw(target_text),
        })

    # =================== 指标计算 ===================
    baseline_RR = n_mainline_reject / n_total   # 主线 thr0.27 RR
    # 多信号 RR (gate+kw 工作点): 主线接受 + rescue 都算"接受" → 漏拒
    n_multi_signal_accept = n_mainline_accept + n_rescued
    multi_signal_RR = (n_total - n_multi_signal_accept) / n_total
    RR_loss_abs = n_rescued / n_total            # 多信号绝对 RR 损失 (Δ 形式: -loss)
    RR_loss_pp = (n_mainline_accept + n_rescued - n_mainline_accept) / n_total * 100  # 百分点

    # net win 判定
    if RR_loss_abs <= 0.03:
        verdict = "强 win"
        verdict_reason = (f"neg RR 损失 {RR_loss_abs*100:.2f}% ≤ 3%, pos ΔCER-22% 净赚. "
                          f"预计净 +4.7 分 (CER腿+5.88 - RR腿损失)")
    elif RR_loss_abs < 0.08:
        verdict = "边际, 需收紧 rescue"
        verdict_reason = (f"neg RR 损失 {RR_loss_abs*100:.2f}% 在 3-8%, 收紧 rescue (加 len≥5 / 更严判别) "
                          f"或量化 pos 收益后再决策")
    else:
        verdict = "高风险, 放弃或大改"
        verdict_reason = (f"neg RR 损失 {RR_loss_abs*100:.2f}% ≥ 8%, neg 漏拒过多, 多信号在 neg 反噬")

    # rescue_pred 强度对比 (各工作点的 neg 漏拒数)
    rescue_pred_loss = {
        name: {"rescued_n": cnt, "RR_loss_abs": round(cnt / n_total, 4),
               "RR_loss_pp": round(cnt / n_total * 100, 2),
               "multi_signal_RR": round((n_total - (n_mainline_accept + cnt)) / n_total, 4)}
        for name, cnt in rescue_counts.items()
    }

    # 拆解: 被 rescue 的 neg 样本 SepFormer target 路文本分类
    rescued_categories = {"empty_other_after_strip": 0, "home_command_like": 0,
                          "noise_hit_kw": 0, "news_or_other": 0, "digit_hallucination": 0}
    for r in rescued_samples:
        t = r["target_text"]
        if not t.strip():
            rescued_categories["empty_other_after_strip"] += 1
        elif any(k in t for k in ["新闻", "财经", "奥运", "产业", "资本", "市场"]):
            rescued_categories["news_or_other"] += 1
        elif re.search(r"\d{4,}", t) or re.search(r"[零一二三四五六七八九]{5,}", t):
            rescued_categories["digit_hallucination"] += 1
        elif r["cmd_score_target"] >= 3.0:
            rescued_categories["home_command_like"] += 1
        else:
            rescued_categories["noise_hit_kw"] += 1

    result = {
        "task": "neg 474 多信号拒识真实 RR 损失测量",
        "rule": "主线 max_sim≥0.27 接受; max_sim<0.27 时若 SepFormer heuristic target 路 gate+kw 通过 → 补救接受",
        "n_total": n_total,
        "baseline_mainline": {
            "thr": THR,
            "n_mainline_accept": n_mainline_accept,
            "n_mainline_reject": n_mainline_reject,
            "RR_thr0.27": round(baseline_RR, 4),
            "interpretation": f"主线 thr0.27 neg RR = {baseline_RR:.4f} (published ≈0.9051 校验)",
        },
        "multi_signal_gate_kw": {
            "n_rescued": n_rescued,
            "RR_loss_abs": round(RR_loss_abs, 4),
            "RR_loss_pp": round(RR_loss_pp, 2),
            "multi_signal_RR": round(multi_signal_RR, 4),
            "multi_signal_accept_n": n_multi_signal_accept,
        },
        "rescue_pred_strength_comparison": rescue_pred_loss,
        "rescued_breakdown_categories": rescued_categories,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "task4_proxy_interval": {"lower_vanilla": 0.0, "upper_pos_other_path": 0.083},
        "rescued_samples": rescued_samples,
        "low_sim_no_rescue_examples": low_sim_no_rescue_examples,
        "n_sep_missing_or_failed": n_sep_missing,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(result, open(RESULT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # =================== 控制台 ===================
    print(f"\n{'='*70}")
    print(f"[结果 — neg 474 多信号 RR 损失]")
    print(f"{'='*70}")
    print(f"\n[校验] 主线 thr0.27: accept={n_mainline_accept}, reject={n_mainline_reject}, "
          f"RR={baseline_RR:.4f} (应≈0.9051)")
    print(f"\n[核心] gate+kw 工作点 rescue:")
    print(f"  rescue 数 (max_sim<0.27 AND gate+kw): {n_rescued}/{n_mainline_reject} (低 sim 池) "
          f"= {n_rescued/n_mainline_reject*100:.2f}% of low-sim")
    print(f"  neg 绝对 RR 损失: {n_rescued}/{n_total} = {RR_loss_abs*100:.2f}% (ΔRR={-RR_loss_abs:+.4f})")
    print(f"  多信号 neg RR: {multi_signal_RR:.4f} (vs 主线 {baseline_RR:.4f})")
    print(f"\n[task4 区间对照] 真实值 {RR_loss_abs*100:.2f}% 落在 [0%(vanilla代理), 8.3%(pos代理)]"
          f" → {'靠近下界' if RR_loss_abs < 0.03 else '靠近上界' if RR_loss_abs > 0.05 else '中部'}")

    print(f"\n[rescue_pred 强度扫 (neg RR 损失绝对值)]:")
    print(f"{'pred':<22} {'rescued':>8} {'RR_loss %':>10} {'multi_sig_RR':>14}")
    for name, cnt in rescue_counts.items():
        loss = cnt / n_total
        msr = (n_total - (n_mainline_accept + cnt)) / n_total
        print(f"{name:<22} {cnt:>8} {loss*100:>9.2f}% {msr:>14.4f}")

    print(f"\n[rescue 样本拆解 (n={n_rescued})]:")
    for k, v in rescued_categories.items():
        print(f"  {k}: {v}")

    print(f"\n[net win 判定]: {verdict}")
    print(f"  {verdict_reason}")

    print(f"\n[产物] {RESULT_PATH}")
    print(f"[done]")


def main():
    ap = argparse.ArgumentParser(description="neg 474 多信号拒识真实 RR 损失测量")
    ap.add_argument("--phase", default="all", choices=["all", "sep", "asr", "analyze"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.phase in ("all", "sep"):
        phase_sep(args.device, args.batch_size)
    if args.phase in ("all", "asr"):
        phase_asr(args.batch_size)
    if args.phase in ("all", "analyze"):
        phase_analyze()


if __name__ == "__main__":
    main()
