"""exp_scene_route_full.py — 全量坐实"分场景路由"的 CER 收益(不再外推).

【背景】
之前 483 样本外推结论: 分场景路由 transcribe CER 0.3436→0.2729 (-20.7%)。
机制: n_spk=1 走主线(SepFormer 反而破坏), n_spk=2 走 SepFormer 两路 + heuristic 二选一。
但死区 n_spk=2 只有 131 抽样, 外推到 805 (universe=1350) 未全量坐实。

【任务】全量实测 1350 universe(n_spk=1: 543, n_spk=2: 805)
  1. n_spk=1 → 主线 qwen (复用 poc_qwen_asr_full_result)
  2. n_spk=2 → SepFormer 分离 + 两路 qwen + heuristic 二选一(cmd_score 挑家居指令路)
     - 231 n_spk=2 已有 SepFormer 数据(复用 3 个 summary.json)
     - 574 n_spk=2 补跑(本脚本 SepFormer phase + qwen phase)
  3. 分场景路由: n_spk=1 用主线, n_spk=2 用 heuristic 选的 SepFormer 路
  4. CER 双口径:
     - transcribe (不拒): 全量分场景 vs 纯主线 0.3436(池口径)
     - 含拒 thr0.27 (pos 允许拒, 拒算 CER=1): 用主线 wespeaker max_sim, 分场景只改"target 说啥"不改"是不是 target"

【sim 一致性方案】
含拒口径 thr 拒识仍用主线 wespeaker max_sim(out_pos_slices_full.json 的 max_sim 字段):
分场景路由只改进"target 说了什么"的转写, 不改"是不是 target"的拒识信号。
理由: 拒识基于 enrollment 声纹 vs recognition 原混音 max_sim, 是"是否存在 target"的信号,
与转写器选路无关; 若改用 SepFormer target 路的 sim, 引入分离后声纹失真 + 与提交链路(主线 max_sim)不一致。

【复用数据】
- out_pos_slices_full.json: 1364 pos diar 输出, speakers 字段 = n_spk, max_sim 字段 = 拒识信号
- runs/poc_qwen_asr_full_result.json: 1350 主线 qwen + CER (rows[uid,sim,ref,qwen,qwen_cer])
- runs/_multivoice_full/summary.json: 243 主战场(61 n_spk=2) SepFormer 两路
- runs/_deadzone_selector/summary.json: 200 死区(131 n_spk=2) SepFormer 两路
- runs/_sepformer_b2/summary.json: 40 失败组(39 n_spk=2) SepFormer 两路
- exp_sepformer_qwen.{load_sepformer,separate,load_diar,get_emb_factory}
- exp_multivoice_route.{route_heuristic,cmd_score}

【用法】
  code/.venv/Scripts/python.exe code/exp_scene_route_full.py --phase all      # 全流程
  code/.venv/Scripts/python.exe code/exp_scene_route_full.py --phase prep     # 仅 CPU 准备
  code/.venv/Scripts/python.exe code/exp_scene_route_full.py --phase sep      # GPU SepFormer (574 缺)
  code/.venv/Scripts/python.exe code/exp_scene_route_full.py --phase asr      # qwen subprocess (新切片)
  code/.venv/Scripts/python.exe code/exp_scene_route_full.py --phase combine  # 聚合 + 写产物

产物: code/runs/_scene_route_full/{slices/,_sep_meta.json,_uid2text_new.json,
                                    _prep.json,per_sample.json,summary.json}
"""
import os, sys, json, time, argparse, subprocess, glob, re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

OUT_DIR = os.path.join(_HERE, "runs", "_scene_route_full")
SLICE_DIR_NEW = os.path.join(OUT_DIR, "slices")  # 新跑的 SepFormer 切片
SEP_META_PATH = os.path.join(OUT_DIR, "_sep_meta.json")  # 新跑的 SepFormer meta
UID2TEXT_NEW_PATH = os.path.join(OUT_DIR, "_uid2text_new.json")  # 新跑的 qwen 输出
PREP_PATH = os.path.join(OUT_DIR, "_prep.json")  # prep 阶段产物(uid 表)
PER_SAMPLE_PATH = os.path.join(OUT_DIR, "per_sample.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")

THR = 0.27  # 含拒阈值(提交口径)
PY_QWEN = os.environ.get("PY_QWEN", os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))


# ============ Phase 1: 数据准备 (CPU, 复用) ============
def phase_prep():
    print(f"\n{'='*70}\n[Phase prep] 数据准备")
    os.makedirs(OUT_DIR, exist_ok=True)

    slices = json.load(open(os.path.join(_HERE, "out_pos_slices_full.json"), encoding="utf-8"))
    slice_map = {os.path.splitext(os.path.basename(it["recognition"]))[0]: it for it in slices}

    qfull = json.load(open(os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    q_rows = {r["uid"]: r for r in qfull["rows"]}

    pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    pair_map = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    # universe = 1350 (有主线 qwen 转写)
    universe = sorted(set(q_rows) & set(slice_map))
    n_spk_dist = {}
    for u in universe:
        n = len(slice_map[u].get("speakers", []))
        n_spk_dist[n] = n_spk_dist.get(n, 0) + 1
    print(f"[prep] universe (有主线 qwen): {len(universe)}, n_spk 分布: {dict(sorted(n_spk_dist.items()))}")
    print(f"[prep] 主线整体 cer_pool (校验): ", end="")
    # 用累计池校验 0.3436
    from eval_metrics import cer_pool
    refs = [q_rows[u]["ref"] for u in universe]
    main_texts = [q_rows[u]["qwen"] for u in universe]
    pool = cer_pool(main_texts, refs)
    print(f"{pool:.4f} (vs 发布 0.3436)")

    # 14 排除样本(无主线 qwen)
    excluded = sorted(set(slice_map) - set(q_rows))
    print(f"[prep] 排除(无主线 qwen 转写): {len(excluded)}: {excluded[:5]}...")

    # 已有 SepFormer 数据覆盖的 n_spk=2 uid
    existing_uids = set()
    for p in [os.path.join(_HERE, "runs/_multivoice_full/_uid2text.json"),
              os.path.join(_HERE, "runs/_deadzone_selector/_uid2text.json"),
              os.path.join(_HERE, "runs/_sepformer_b2/_uid2text.json")]:
        d = json.load(open(p, encoding="utf-8"))
        for k in d:
            m = re.match(r"^(cmd_\d+)__", k)
            if m:
                existing_uids.add(m.group(1))
    print(f"[prep] 已有 SepFormer 数据覆盖: {len(existing_uids)} uid")

    nspk2_universe = [u for u in universe if len(slice_map[u].get("speakers", [])) == 2]
    nspk2_existing = [u for u in nspk2_universe if u in existing_uids]
    nspk2_missing = [u for u in nspk2_universe if u not in existing_uids]
    print(f"[prep] n_spk=2 universe: {len(nspk2_universe)}")
    print(f"[prep]   已有 SepFormer: {len(nspk2_existing)}")
    print(f"[prep]   需补跑      : {len(nspk2_missing)}")

    prep = {
        "universe": universe,
        "excluded": excluded,
        "n_spk_dist": {str(k): v for k, v in n_spk_dist.items()},
        "nspk2_existing": nspk2_existing,
        "nspk2_missing": nspk2_missing,
        "mainline_pool_cer": round(pool, 4),
    }
    json.dump(prep, open(PREP_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[prep] 写 {PREP_PATH}")
    return prep


def load_prep():
    if not os.path.exists(PREP_PATH):
        return phase_prep()
    return json.load(open(PREP_PATH, encoding="utf-8"))


# ============ Phase 2: SepFormer 分离 (GPU) - 仅补跑 missing ============
def phase_sep(device, batch_size):
    """SepFormer 分离 574 缺失样本, 两路(srcA/srcB)都存 wav。
    Resumable: 已存在的 wav 不重跑。"""
    # speechbrain Windows 兼容(LazyModule inspect-guard, 复刻 exp_sepformer_qwen.py)
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
    from exp_sepformer_qwen import load_sepformer, separate, load_diar, get_emb_factory
    from repro import set_global_seed, resolve_model

    set_global_seed(42)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(SLICE_DIR_NEW, exist_ok=True)

    prep = load_prep()
    missing = prep["nspk2_missing"]
    pairs = json.load(open(os.path.join(_HERE, "pos_pairs_datasetA.json"), encoding="utf-8"))
    pair_map = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    # 已有 meta + slices (resumable)
    if os.path.exists(SEP_META_PATH):
        meta = json.load(open(SEP_META_PATH, encoding="utf-8"))
    else:
        meta = {}
    meta_new = {m["uid"]: m for m in meta.get("results", [])} if "results" in meta else dict(meta)

    # 过滤: 跳过已写完整两路 wav 的
    todo = []
    for uid in missing:
        sa = os.path.join(SLICE_DIR_NEW, f"{uid}__srcA.wav")
        sb = os.path.join(SLICE_DIR_NEW, f"{uid}__srcB.wav")
        if os.path.exists(sa) and os.path.exists(sb) and uid in meta_new:
            continue
        todo.append(uid)
    print(f"\n{'='*70}\n[Phase sep] SepFormer 补跑 {len(todo)} 条 (skipped {len(missing)-len(todo)} 已完成)")
    if not todo:
        print("[sep] 无需补跑")
        return

    print(f"[load] SepFormer whamr16k → E:/hf_cache/sepformer-whamr16k")
    sep_model = load_sepformer(dev, "E:/hf_cache/sepformer-whamr16k")
    diar = load_diar(resolve_model("DIAR"), dev)
    get_emb = get_emb_factory(diar, dev)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    t0 = time.time()
    total_audio_sec = 0.0
    n_sep_fail = 0
    for n, uid in enumerate(todo):
        pair = pair_map[uid]
        enr, rec = pair["enrollment"], pair["recognition"]
        try:
            w_enr, _ = librosa.load(enr, sr=16000)
            enroll_emb = get_emb(w_enr)
            audio, sr = librosa.load(rec, sr=16000)
            total_audio_sec += len(audio) / sr
            sources = separate(audio, sep_model)  # [n_src, T]
            n_src = sources.shape[0]
            if n_src != 2:
                raise RuntimeError(f"SepFormer n_src={n_src} 非 2 路")

            # 诊断: SepFormer 分离后两路 vs enrollment sim(仅供诊断, 不作选路)
            embs = []
            for i in range(n_src):
                seg = sources[i]
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enroll_emb, e) for e in embs])
            sim_pick_idx = int(torch.argmax(sims))

            # 两路都存, 保持 SepFormer 原生顺序 (srcA, srcB) — 与 _multivoice_full 一致
            slice_uids = []
            for i in range(n_src):
                suid = f"{uid}__src{chr(65+i)}"
                sf.write(os.path.join(SLICE_DIR_NEW, suid + ".wav"),
                         np.ascontiguousarray(sources[i].astype(np.float32)), 16000)
                slice_uids.append(suid)

            meta_new[uid] = {
                "uid": uid, "n_src": n_src,
                "slice_uids": slice_uids,
                "sep_sims": [round(float(s), 4) for s in sims],
                "sim_pick_idx": sim_pick_idx,
                "audio_sec": round(len(audio)/sr, 2),
            }
            if (n+1) % 20 == 0 or n == len(todo)-1:
                # 周期落盘(resumable)
                json.dump({"results": list(meta_new.values())},
                          open(SEP_META_PATH, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                print(f"  [{n+1}/{len(todo)}] {uid} sep_sim={float(sims[sim_pick_idx]):.3f} "
                      f"({time.time()-t0:.0f}s, audio={total_audio_sec:.0f}s)")
        except Exception as e:
            n_sep_fail += 1
            print(f"  [{n+1}/{len(todo)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta_new[uid] = {"uid": uid, "error": f"{type(e).__name__}: {str(e)[:120]}"}

    json.dump({"results": list(meta_new.values())},
              open(SEP_META_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    dt = time.time() - t0
    print(f"\n[sep] 完成: {len(todo)-n_sep_fail}/{len(todo)} 成功, 失败 {n_sep_fail}, "
          f"耗时 {dt:.0f}s ({dt/60:.1f}min), 总音频 {total_audio_sec:.0f}s "
          f"({total_audio_sec/60:.1f}min), RTF={dt/total_audio_sec:.3f}")


# ============ Phase 3: Qwen 批量转写新切片 (subprocess) ============
def phase_asr(batch_size):
    print(f"\n{'='*70}\n[Phase asr] qwen 转写 {SLICE_DIR_NEW}")
    if not os.path.isdir(SLICE_DIR_NEW) or not glob.glob(os.path.join(SLICE_DIR_NEW, "*.wav")):
        print("[asr] 无新切片, 跳过")
        return
    cmd = [PY_QWEN, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", SLICE_DIR_NEW, "--out", UID2TEXT_NEW_PATH,
           "--seed", "42", "--batch-size", str(batch_size)]
    print(f"[asr] subprocess: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0
    print(f"[asr] 完成, 耗时 {dt:.0f}s ({dt/60:.1f}min), 输出 {UID2TEXT_NEW_PATH}")


# ============ Phase 4: 聚合 + CER + 写产物 ============
def phase_combine():
    print(f"\n{'='*70}\n[Phase combine] 聚合 + CER 计算")
    from text_utils import to_simplified, digit_postproc, brand_homophone_fix
    from eval_metrics import cer_pool, cer_official, normalize_text, CERMetric
    from exp_multivoice_route import route_heuristic, cmd_score

    def cer_norm(text, ref):
        t = brand_homophone_fix(digit_postproc(to_simplified(text)))
        r = brand_homophone_fix(digit_postproc(to_simplified(ref)))
        return float(cer_official(t, r))

    prep = load_prep()
    universe = prep["universe"]
    slices = json.load(open(os.path.join(_HERE, "out_pos_slices_full.json"), encoding="utf-8"))
    slice_map = {os.path.splitext(os.path.basename(it["recognition"]))[0]: it for it in slices}
    qfull = json.load(open(os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    q_rows = {r["uid"]: r for r in qfull["rows"]}

    # 收集已有 SepFormer 两路数据(uid → list of {"src_idx","text","cer"})
    existing_sep = {}
    for path in [os.path.join(_HERE, "runs/_multivoice_full/summary.json"),
                 os.path.join(_HERE, "runs/_deadzone_selector/summary.json"),
                 os.path.join(_HERE, "runs/_sepformer_b2/summary.json")]:
        d = json.load(open(path, encoding="utf-8"))
        for r in d.get("results", []):
            if "error" in r or "per_src" not in r:
                continue
            uid = r["uid"]
            if uid in existing_sep:
                continue  # 不覆盖(优先 _multivoice_full 主战场, 但实际应无重叠)
            existing_sep[uid] = {
                "slice_uids": r.get("slice_uids", []),
                "per_src_texts": [s["text"] for s in r["per_src"]],
            }

    # 新跑的 SepFormer meta + uid2text
    new_meta = {}
    if os.path.exists(SEP_META_PATH):
        d = json.load(open(SEP_META_PATH, encoding="utf-8"))
        for m in d.get("results", []):
            new_meta[m["uid"]] = m
    new_uid2text = {}
    if os.path.exists(UID2TEXT_NEW_PATH):
        new_uid2text = json.load(open(UID2TEXT_NEW_PATH, encoding="utf-8"))

    print(f"[combine] universe: {len(universe)}")
    print(f"[combine] existing SepFormer data: {len(existing_sep)}")
    print(f"[combine] new SepFormer data: {len(new_meta)} (with text: "
          f"{sum(1 for m in new_meta.values() if 'error' not in m)})")

    # 逐条构建
    per_sample = []
    mainline_pool_metric = CERMetric()      # 主线 transcribe (校验 = 0.3436)
    scene_route_pool_metric = CERMetric()   # 分场景 transcribe
    mainline_thr_metric = CERMetric()       # 主线 含拒 thr0.27
    scene_route_thr_metric = CERMetric()    # 分场景 含拒 thr0.27

    # 分桶 metric
    bucket_metrics = {
        "nspk1": {"mainline_transcribe": CERMetric(), "scene_route_transcribe": CERMetric(),
                  "mainline_thr": CERMetric(), "scene_route_thr": CERMetric()},
        "nspk2": {"mainline_transcribe": CERMetric(), "scene_route_transcribe": CERMetric(),
                  "mainline_thr": CERMetric(), "scene_route_thr": CERMetric()},
        "nspk0_3": {"mainline_transcribe": CERMetric(), "scene_route_transcribe": CERMetric(),
                    "mainline_thr": CERMetric(), "scene_route_thr": CERMetric()},
    }

    def norm_pair(text, ref):
        t = brand_homophone_fix(digit_postproc(to_simplified(text)))
        r = brand_homophone_fix(digit_postproc(to_simplified(ref)))
        return t, r

    n_no_mainline = 0
    n_sep_recovered = 0
    n_sep_failed_fallback = 0

    for uid in universe:
        qrow = q_rows[uid]
        ref = qrow["ref"]
        main_text = qrow["qwen"]
        max_sim = slice_map[uid].get("max_sim")
        speakers = slice_map[uid].get("speakers", [])
        n_spk = len(speakers)
        bucket_key = "nspk1" if n_spk == 1 else ("nspk2" if n_spk == 2 else "nspk0_3")
        rejected = (max_sim is not None and max_sim < THR)

        # 主线 transcribe
        mt, mr = norm_pair(main_text, ref)
        mainline_pool_metric.update([mt], [mr])
        bucket_metrics[bucket_key]["mainline_transcribe"].update([mt], [mr])

        # 主线 含拒 thr0.27
        if rejected:
            mainline_thr_metric.update([""], [mr])  # 拒 → 空 hyp → CER=1
            bucket_metrics[bucket_key]["mainline_thr"].update([""], [mr])
        else:
            mainline_thr_metric.update([mt], [mr])
            bucket_metrics[bucket_key]["mainline_thr"].update([mt], [mr])

        # 分场景路由 transcribe
        scene_text = main_text  # 默认主线
        route_decision = "nspk1_mainline"
        sep_info = None
        if n_spk == 2:
            # 取 SepFormer 两路
            per_src_texts = None
            slice_uids = None
            if uid in existing_sep:
                per_src_texts = existing_sep[uid]["per_src_texts"]
                slice_uids = existing_sep[uid]["slice_uids"]
                sep_source = "existing"
            elif uid in new_meta and "error" not in new_meta[uid]:
                suids = new_meta[uid]["slice_uids"]
                per_src_texts = [new_uid2text.get(s, "") for s in suids]
                slice_uids = suids
                sep_source = "new"
            else:
                sep_source = "missing"

            if per_src_texts is not None and len(per_src_texts) >= 2:
                # heuristic 二选一
                per_src_for_route = [{"text": t} for t in per_src_texts]
                h_idx, h_reason = route_heuristic(per_src_for_route)
                scene_text = per_src_texts[h_idx]
                route_decision = f"nspk2_sep_heuristic_idx{h_idx}({sep_source},{h_reason})"
                sep_info = {
                    "sep_source": sep_source,
                    "slice_uids": slice_uids,
                    "per_src_texts": per_src_texts,
                    "heuristic_idx": h_idx,
                    "heuristic_reason": h_reason,
                    "per_src_cers": [round(cer_norm(t, ref), 4) for t in per_src_texts],
                    "sep_sims": new_meta.get(uid, {}).get("sep_sims"),
                }
                # 诊断: 选对了 oracle 路吗
                per_cers = sep_info["per_src_cers"]
                oracle_idx = per_cers.index(min(per_cers))
                sep_info["oracle_idx"] = oracle_idx
                sep_info["heuristic_picks_oracle"] = (h_idx == oracle_idx)
                if cer_norm(scene_text, ref) < cer_norm(main_text, ref) - 0.05:
                    n_sep_recovered += 1
            else:
                # SepFormer 失败 → fallback 主线
                route_decision = f"nspk2_sep_missing_or_failed_fallback_mainline({sep_source})"
                n_sep_failed_fallback += 1

        st, sr = norm_pair(scene_text, ref)
        scene_route_pool_metric.update([st], [sr])
        bucket_metrics[bucket_key]["scene_route_transcribe"].update([st], [sr])
        if rejected:
            scene_route_thr_metric.update([""], [sr])
            bucket_metrics[bucket_key]["scene_route_thr"].update([""], [sr])
        else:
            scene_route_thr_metric.update([st], [sr])
            bucket_metrics[bucket_key]["scene_route_thr"].update([st], [sr])

        per_sample.append({
            "uid": uid,
            "n_spk": n_spk,
            "max_sim": round(max_sim, 4) if max_sim is not None else None,
            "rejected_thr0.27": rejected,
            "ref": ref,
            "mainline_text": main_text,
            "mainline_cer_transcribe": round(cer_norm(main_text, ref), 4),
            "scene_route_text": scene_text,
            "scene_route_cer_transcribe": round(cer_norm(scene_text, ref), 4),
            "route_decision": route_decision,
            "sep_info": sep_info,
        })

    # 汇总
    mainline_transcribe_cer = mainline_pool_metric.compute()["cer"]
    scene_route_transcribe_cer = scene_route_pool_metric.compute()["cer"]
    mainline_thr_cer = mainline_thr_metric.compute()["cer"]
    scene_route_thr_cer = scene_route_thr_metric.compute()["cer"]

    delta_transcribe = scene_route_transcribe_cer - mainline_transcribe_cer
    delta_thr = scene_route_thr_cer - mainline_thr_cer
    pct_transcribe = (delta_transcribe / mainline_transcribe_cer * 100) if mainline_transcribe_cer else 0
    pct_thr = (delta_thr / mainline_thr_cer * 100) if mainline_thr_cer else 0

    print(f"\n{'='*70}\n[结果 — 全量 {len(universe)} 条]")
    print(f"{'口径':<28} {'主线':>10} {'分场景路由':>12} {'Δ':>10} {'%':>8}")
    print(f"{'-'*70}")
    print(f"{'transcribe (不拒)':<28} {mainline_transcribe_cer:>10.4f} "
          f"{scene_route_transcribe_cer:>12.4f} {delta_transcribe:>+10.4f} {pct_transcribe:>+7.1f}%")
    print(f"{'含拒 thr0.27 (提交)':<28} {mainline_thr_cer:>10.4f} "
          f"{scene_route_thr_cer:>12.4f} {delta_thr:>+10.4f} {pct_thr:>+7.1f}%")
    print(f"\n[对照发布 baseline] qwen transcribe pool=0.3436 / 含拒 thr0.27=0.5934")
    print(f"[校验] 本次主线 transcribe cer_pool={mainline_transcribe_cer:.4f} "
          f"(应≈0.3436, 小差异来自归一链)")

    # 分桶打印
    print(f"\n[分桶 CER (transcribe / 含拒 thr0.27)]")
    print(f"{'桶':<10} {'n':>6} {'主线 trans':>12} {'分场景 trans':>14} "
          f"{'主线 thr':>10} {'分场景 thr':>12}")
    bucket_out = {}
    for k in ["nspk1", "nspk2", "nspk0_3"]:
        bk = bucket_metrics[k]
        n = bk["mainline_transcribe"].total_chars
        if n == 0:
            continue
        m_tr = bk["mainline_transcribe"].compute()["cer"]
        s_tr = bk["scene_route_transcribe"].compute()["cer"]
        m_thr = bk["mainline_thr"].compute()["cer"]
        s_thr = bk["scene_route_thr"].compute()["cer"]
        # n_samples from per_sample
        n_samp = sum(1 for p in per_sample if (
            p["n_spk"] == 1 if k == "nspk1" else
            p["n_spk"] == 2 if k == "nspk2" else
            p["n_spk"] not in (1, 2)
        ))
        bucket_out[k] = {
            "n_samples": n_samp,
            "mainline_transcribe_cer_pool": round(m_tr, 4),
            "scene_route_transcribe_cer_pool": round(s_tr, 4),
            "mainline_thr0.27_cer_pool": round(m_thr, 4),
            "scene_route_thr0.27_cer_pool": round(s_thr, 4),
            "delta_transcribe": round(s_tr - m_tr, 4),
            "delta_thr": round(s_thr - m_thr, 4),
        }
        print(f"{k:<10} {n_samp:>6} {m_tr:>12.4f} {s_tr:>14.4f} {m_thr:>10.4f} {s_thr:>12.4f}")

    # 选路统计
    n_nspk2 = sum(1 for p in per_sample if p["n_spk"] == 2)
    n_nspk2_used_sep = sum(1 for p in per_sample if p["n_spk"] == 2 and p["sep_info"] is not None)
    n_nspk2_heur_picks_oracle = sum(1 for p in per_sample
                                    if p["sep_info"] and p["sep_info"].get("heuristic_picks_oracle"))
    print(f"\n[选路统计]")
    print(f"  n_spk=2 总数:        {n_nspk2}")
    print(f"  使用 SepFormer 路径: {n_nspk2_used_sep} ({n_nspk2_used_sep/max(n_nspk2,1)*100:.1f}%)")
    if n_nspk2_used_sep:
        print(f"  heuristic 选对 oracle: {n_nspk2_heur_picks_oracle}/{n_nspk2_used_sep} "
              f"({n_nspk2_heur_picks_oracle/n_nspk2_used_sep*100:.1f}%)")
    print(f"  SepFormer 失败/缺失 fallback 主线: {n_sep_failed_fallback}")
    print(f"  SepFormer 救回(transcribe CER 改善>0.05): {n_sep_recovered}")

    # 写 per_sample
    json.dump(per_sample, open(PER_SAMPLE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[写] {PER_SAMPLE_PATH} ({len(per_sample)} 条)")

    # 写 summary
    summary = {
        "verdict": "分场景路由全量坐实",
        "task": "n_spk=1 走主线 / n_spk=2 走 SepFormer 两路 + heuristic 二选一",
        "universe_n": len(universe),
        "excluded_n": len(prep["excluded"]),
        "excluded_reason": "无主线 qwen 转写(原 enroll_infer 该 uid vanilla 输出空)",
        "n_spk_distribution": prep["n_spk_dist"],
        "thr": THR,
        "sim_consistency_note": (
            "含拒口径 thr 拒识用主线 wespeaker max_sim(out_pos_slices_full.json); "
            "分场景路由只改进 'target 说了什么' 的转写, 不改 '是不是 target' 的拒识信号; "
            "与提交链路一致。"
        ),
        "mainline_baseline_published": {
            "transcribe_pool_cer": 0.3436,
            "containing_reject_thr0.27_cer": 0.5934,
        },
        "overall": {
            "mainline_transcribe_cer_pool": round(mainline_transcribe_cer, 4),
            "scene_route_transcribe_cer_pool": round(scene_route_transcribe_cer, 4),
            "mainline_thr0.27_cer_pool": round(mainline_thr_cer, 4),
            "scene_route_thr0.27_cer_pool": round(scene_route_thr_cer, 4),
            "delta_transcribe": round(delta_transcribe, 4),
            "delta_thr": round(delta_thr, 4),
            "pct_change_transcribe": round(pct_transcribe, 2),
            "pct_change_thr": round(pct_thr, 2),
        },
        "buckets": bucket_out,
        "routing_stats": {
            "n_nspk2_total": n_nspk2,
            "n_nspk2_used_sep": n_nspk2_used_sep,
            "n_nspk2_heuristic_picks_oracle": n_nspk2_heur_picks_oracle,
            "heuristic_pick_accuracy": round(n_nspk2_heur_picks_oracle/max(n_nspk2_used_sep,1), 4),
            "n_sep_failed_fallback": n_sep_failed_fallback,
            "n_sep_recovered_vs_mainline": n_sep_recovered,
            "sep_coverage": {
                "existing_reused": len([u for u in prep["nspk2_existing"]]),
                "new_run": len([u for u in prep["nspk2_missing"]]),
            },
        },
        "external_extrapolation_compare": {
            "prior_483_extrapolation": {
                "mainline_transcribe": 0.3436,
                "scene_route_transcribe_estimate": 0.2729,
                "expected_delta": -0.0707,
                "expected_pct": -20.7,
            },
            "full_measurement": {
                "mainline_transcribe": round(mainline_transcribe_cer, 4),
                "scene_route_transcribe": round(scene_route_transcribe_cer, 4),
                "actual_delta": round(delta_transcribe, 4),
                "actual_pct": round(pct_transcribe, 2),
            },
        },
        "per_sample_path": PER_SAMPLE_PATH,
    }
    json.dump(summary, open(SUMMARY_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[写] {SUMMARY_PATH}")
    print(f"\n[done] 分场景路由全量坐实完成")


def main():
    ap = argparse.ArgumentParser(description="分场景路由全量坐实")
    ap.add_argument("--phase", default="all",
                    choices=["all", "prep", "sep", "asr", "combine"],
                    help="执行阶段(all=prep→sep→asr→combine)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.phase in ("all", "prep"):
        phase_prep()
    if args.phase in ("all", "sep"):
        phase_sep(args.device, args.batch_size)
    if args.phase in ("all", "asr"):
        phase_asr(args.batch_size)
    if args.phase in ("all", "combine"):
        phase_combine()


if __name__ == "__main__":
    main()
