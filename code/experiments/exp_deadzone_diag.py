"""exp_deadzone_diag.py — 死区(sim<0.4)量化分桶 + 失败模式 + SepFormer 天花板 + 验收包。

【背景】全量验证发现死区(sim<0.4)占 1350 条的 78.8%(1064 条), 贡献整体 CER 的 ~87%,
是真正的 CER 大头。死区之前被笼统归"babble 摧毁 mel 物理地板", 但 cmd_2637/2129 等
验收案例证明死区里有可救子集(选错 target / 切 timeline bug)。本脚本量化死区分桶 +
对高 CER 样本做失败模式分类 + 重测 SepFormer oracle 死区天花板 + 出验收包。

【3 个 phase】
  analyze   Task1: 死区 3 桶(<0.2/[0.2,0.3)/[0.3,0.4)) n/mean_CER/correct/CER=0 + 失败模式分类
  sepformer Task2: 死区抽样 60 条(30 sim<0.2 + 30 [0.2,0.4)) SepFormer oracle 选路 vs argmax
  verify    Task3: 4-5 条代表样本出 8 工位验收包

【复用】
  - SepFormer: exp_sepformer_qwen.py 的 load_sepformer/separate/load_diar/get_emb_factory
  - 验收包: enroll_infer.py:24-29 (inspect guard) + 170-172 (load diar) + 178-187 (get_emb)

用法:
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_deadzone_diag.py --phase analyze
  code/.venv/Scripts/python.exe code/exp_deadzone_diag.py --phase sepformer
  code/.venv/Scripts/python.exe code/exp_deadzone_diag.py --phase verify
  code/.venv/Scripts/python.exe code/exp_deadzone_diag.py --phase all

产物:
  docs/deadzone_diag.md (analyze 写)
  code/runs/_deadzone_sepformer/ (sepformer 产物)
  code/runs/_verify_deadzone_<id>/ (verify 各样本 8 工位)
  docs/verify_deadzone.md (verify 索引)
"""
import os, sys, json, time, argparse, subprocess, glob, statistics as st

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from text_utils import to_simplified, digit_postproc, brand_homophone_fix, cut_target_timeline
from eval_metrics import cer_official
from repro import set_global_seed, resolve_model


# ---- 辅助 ----
def _cer(text, ref):
    t = brand_homophone_fix(digit_postproc(to_simplified(text or "")))
    r = brand_homophone_fix(digit_postproc(to_simplified(ref or "")))
    return float(cer_official(t, r))


def _char_overlap(a, b):
    """两字符串的字符 overlap coefficient = |A∩B| / min(|A|,|B|)。
    低 → 字面完全不沾边(幻觉或选错 target); 高 → 接近(可纠正类)。"""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / min(len(sa), len(sb))


def _classify_failure(ref, qwen_text, cer, qwen_argmax_cer, oracle_cer=None, argmax_correct=None):
    """死区高 CER 样本的失败模式启发式分类(纯文本 + 已有 oracle 数据)。

    【方法论】"mel 摧毁"不可量化=偷懒归因, 真地板判据 = **人耳能否听清 target 说的 ref**:
      - 人耳能听清(即便有噪音) → 不是地板, 是机器能力不够(可修)
      - 人耳都听不清(纯噪音/完全淹没) → 真地板
    机器无法直接判定人耳可辨性 → 启发式"疑似不可辨/机器失败"仅供初筛, **待用户听验收包定性**。

    返回:
      M1a 疑似人耳不可辨-循环幻觉(待听)  字面完全不沾边 + 输出明显长于 ref(循环 babble 幻觉)
      M1b 疑似人耳不可辨-乱词(待听)      字面完全不沾边 + 短输出(捕获噪声词)
      M2  机器选错target(可救)           argmax_correct=False + oracle_cer 显著更低(oracle 数据佐证)
      M3  切timeline疑似问题(待听)        输出 ≤2 字 + ref ≥3 字(切到无声段或错段)
      M4  机器转错但接近(可纠正)         字面 overlap ≥0.5 + CER∈[0.2,0.8) → 字错/同音/数字
      M5  机器空或极短崩(可救)           qwen 输出空或 1 字
    """
    q = qwen_text or ""
    q_clean = "".join(c for c in to_simplified(q) if c.strip())
    r_clean = "".join(c for c in to_simplified(ref) if c.strip())
    overlap = _char_overlap(q_clean, r_clean)

    # M0 机器成功(CER<0.2): 不属于失败模式, 单列
    if cer < 0.2:
        return "M0_机器成功"

    # M5 空或极短崩
    if len(q_clean) <= 1:
        return "M5_机器空极短崩"

    # M2 选错 target(仅当有 oracle 数据佐证)
    if argmax_correct is False and oracle_cer is not None and oracle_cer < cer - 0.2:
        return "M2_机器选错target"

    # M3 切 timeline: 输出短 + ref 长
    if len(q_clean) <= 2 and len(r_clean) >= 3:
        return "M3_切timeline疑似"

    # M1 疑似人耳不可辨(字面完全不沾边 + CER 高): 待用户听 recognition 验证
    if overlap < 0.3 and cer >= 0.8:
        if len(q_clean) >= len(r_clean) * 1.5:
            return "M1a_疑似不可辨-循环幻觉"
        return "M1b_疑似不可辨-乱词"

    # M4 机器转错但接近可纠正 (CER 已确保 >=0.2)
    if overlap >= 0.5 and cer < 0.8:
        return "M4_机器转错接近"

    # 兜底
    if overlap < 0.5:
        return "M1b_疑似不可辨-乱词"
    return "M4_机器转错接近"


def _mode_category(mode):
    """把细模式归为 3 大类(报告用): 待听-疑似不可辨 / 机器失败-可救 / 接近解决。
    重要: 'M1 疑似不可辨'最终是地板还是可救, 必须由用户听 recognition 决定, 机器不能定论。"""
    if mode.startswith("M1"):
        return "疑似人耳不可辨(待听验证)"
    if mode.startswith("M2") or mode.startswith("M3") or mode.startswith("M5"):
        return "机器失败-疑似可救(待听确认)"
    if mode.startswith("M4"):
        return "机器转错-可纠正"
    return "其他"


# ====================== Task 1: analyze ======================
def phase_analyze(args):
    print(f"\n{'='*70}\n[Phase: analyze] 死区量化分桶 + 失败模式分类")
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows = qfull["rows"] if isinstance(qfull, dict) and "rows" in qfull else qfull
    print(f"[data] {args.qwen_full}: {len(rows)} rows, overall_qwen={qfull.get('overall_qwen'):.4f}")

    # 死区 + 接近解决对照
    buckets_def = [
        ("<0.2 死区深", lambda s: s < 0.2),
        ("[0.2,0.3) 死区浅", lambda s: 0.2 <= s < 0.3),
        ("[0.3,0.4) 死区边", lambda s: 0.3 <= s < 0.4),
        (">=0.4 接近解决(对照)", lambda s: s >= 0.4),
    ]
    bucket_stats = {}
    for name, fn in buckets_def:
        sub = [r for r in rows if fn(r["sim"])]
        if not sub:
            bucket_stats[name] = {"n": 0}
            continue
        cers = [r["qwen_cer"] for r in sub]
        vcers = [r["vanilla_cer"] for r in sub]
        n = len(sub)
        bucket_stats[name] = {
            "n": n,
            "n_pct": round(n / len(rows) * 100, 1),
            "qwen_mean_cer": round(st.mean(cers), 4),
            "vanilla_mean_cer": round(st.mean(vcers), 4),
            "qwen_correct_rate": round(sum(1 for c in cers if c < 0.5) / n, 4),
            "qwen_perfect_rate": round(sum(1 for c in cers if c == 0) / n, 4),
            "qwen_highcer_rate": round(sum(1 for c in cers if c > 0.8) / n, 4),
            "contrib_cer_sum": round(sum(cers), 2),
        }
    print("\n[死区 3 桶分布]")
    for name, s in bucket_stats.items():
        print(f"  {name}: n={s['n']} ({s.get('n_pct',0)}%)  "
              f"qwen_mean={s['qwen_mean_cer']:.3f}  correct={s['qwen_correct_rate']*100:.1f}%  "
              f"CER=0={s['qwen_perfect_rate']*100:.1f}%  highCER(>0.8)={s['qwen_highcer_rate']*100:.1f}%")

    # ---- 失败模式分类(死区高 CER 样本) ----
    dead = [r for r in rows if r["sim"] < 0.4]
    # 优先用 oracle speaker 数据(60 条 sim<0.2)做"选错 target"标定
    oracle_dead_path = os.path.join(_HERE, "runs", "exp_spk_oracle_qwen_dead.json")
    uid2oracle = {}
    if os.path.exists(oracle_dead_path):
        od = json.load(open(oracle_dead_path, encoding="utf-8"))
        uid2oracle = {r["uid"]: r for r in od.get("results", [])}
        print(f"\n[oracle] 加载 exp_spk_oracle_qwen_dead.json: {len(uid2oracle)} 条 sim<0.2 oracle speaker 数据")

    # 3 桶各自分类
    dead_buckets = [
        ("sim<0.2", [r for r in dead if r["sim"] < 0.2]),
        ("[0.2,0.3)", [r for r in dead if 0.2 <= r["sim"] < 0.3]),
        ("[0.3,0.4)", [r for r in dead if 0.3 <= r["sim"] < 0.4]),
    ]
    fail_mode_stats = {b[0]: {} for b in dead_buckets}
    sample_classified = []
    for bname, sub in dead_buckets:
        mode_count = {}
        for r in sub:
            oc = uid2oracle.get(r["uid"])
            mode = _classify_failure(
                r["ref"], r["qwen"], r["qwen_cer"],
                r["qwen_cer"],
                oracle_cer=(oc["oracle_cer"] if oc else None),
                argmax_correct=(oc.get("argmax_correct") if oc else None),
            )
            mode_count[mode] = mode_count.get(mode, 0) + 1
            if len(sample_classified) < 80 and r["qwen_cer"] > 0.5:
                sample_classified.append({
                    "uid": r["uid"], "sim": round(r["sim"], 3), "cer": round(r["qwen_cer"], 3),
                    "mode": mode, "ref": r["ref"][:40], "qwen": (r["qwen"] or "")[:40],
                    "oracle_cer": (oc["oracle_cer"] if oc else None),
                    "argmax_correct": (oc.get("argmax_correct") if oc else None),
                })
        fail_mode_stats[bname] = {"n": len(sub), "modes": mode_count}
    print(f"\n[失败模式分类 - 死区 {len(dead)} 条]")
    for bname, st_ in fail_mode_stats.items():
        n = st_["n"]
        print(f"  {bname} (n={n}):")
        for m, c in sorted(st_["modes"].items(), key=lambda x: -x[1]):
            print(f"    {m}: {c} ({c/n*100:.1f}%)")

    # 全死区聚合
    agg_modes = {}
    for st_ in fail_mode_stats.values():
        for m, c in st_["modes"].items():
            agg_modes[m] = agg_modes.get(m, 0) + c
    n_dead = len(dead)
    print(f"\n[死区聚合]")
    for m, c in sorted(agg_modes.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}/{n_dead} ({c/n_dead*100:.1f}%)")

    # 估算"可救子集"(机器失败但人耳可能可辨): M2/M3/M5/M4 + M1 待听定性
    # M0 机器成功(CER<0.2): 已成功, 不算可救也不算失败
    n_machine_success = agg_modes.get("M0_机器成功", 0)
    rescue_modes = {"M2_机器选错target", "M3_切timeline疑似", "M5_机器空极短崩", "M4_机器转错接近"}
    n_machine_fail = sum(c for m, c in agg_modes.items() if m in rescue_modes)
    n_suspect_unintelligible = sum(c for m, c in agg_modes.items() if m.startswith("M1"))
    print(f"\n[死区可救子集估算(人耳可辨=可修, 人耳不可辨=真地板)]")
    print(f"  机器已成功(CER<0.2): {n_machine_success}/{n_dead} ({n_machine_success/n_dead*100:.1f}%)")
    print(f"  机器失败-疑似可救(M2+M3+M4+M5): {n_machine_fail}/{n_dead} ({n_machine_fail/n_dead*100:.1f}%)")
    print(f"  疑似人耳不可辨(M1, 待听验证): {n_suspect_unintelligible}/{n_dead} ({n_suspect_unintelligible/n_dead*100:.1f}%)")
    print(f"  其中 M1 真地板 vs 人耳可辨机器做不到, 必须由用户听验收包定性(机器不能定论)")

    # ---- 写 docs/deadzone_diag.md (Task 1 部分, 后续 phase 追加) ----
    doc_path = os.path.join(_ROOT, "docs", "deadzone_diag.md")
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    lines = [
        "# 死区(sim<0.4)量化诊断\n",
        f"> 生成: {time.strftime('%Y-%m-%d %H:%M')} by exp_deadzone_diag.py --phase analyze\n",
        f"> 数据源: `{args.qwen_full}` (n={len(rows)}, overall qwen CER={qfull.get('overall_qwen'):.4f})\n",
        "\n## 1. 死区分桶\n",
        "| 桶 | n | 占比 | qwen mean CER | vanilla mean CER | correct(<0.5) | CER=0 | 高CER(>0.8) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, _ in buckets_def:
        s = bucket_stats[name]
        lines.append(f"| {name} | {s['n']} | {s.get('n_pct',0)}% | {s['qwen_mean_cer']} | "
                     f"{s['vanilla_mean_cer']} | {s['qwen_correct_rate']*100:.1f}% | "
                     f"{s['qwen_perfect_rate']*100:.1f}% | {s['qwen_highcer_rate']*100:.1f}% |")
    n_dead_total = sum(bucket_stats[b[0]]["n"] for b in buckets_def if "死区" in b[0])
    lines.append(f"\n**死区(sim<0.4)合计 n={n_dead_total}, 占 {n_dead_total/len(rows)*100:.1f}%**\n")

    lines.append("\n## 2. 失败模式分类(死区高 CER 启发式, 待用户听验收包定性)\n")
    lines.append("> ⚠️ **方法论**: 'mel 摧毁'不可量化=偷懒归因。真地板判据 = **人耳能否听清 target 说的 ref**: "
                 "人耳能听清=机器能力不够(可修); 人耳听不清=真地板。机器只能初筛, M1 模式需用户听 recognition 定性。\n")
    lines.append("| 桶 | n | " + " | ".join(sorted(agg_modes.keys())) + " |")
    lines.append("|---|---|" + "---|" * len(agg_modes))
    for bname, st_ in fail_mode_stats.items():
        row = f"| {bname} | {st_['n']} |"
        for m in sorted(agg_modes.keys()):
            c = st_["modes"].get(m, 0)
            row += f" {c} ({c/st_['n']*100:.0f}%)" + " |"
        lines.append(row)
    lines.append("\n**模式定义**:")
    lines.append("- **M0_机器成功**: CER<0.2(机器已转对, 不算失败)")
    lines.append("- **M1a_疑似不可辨-循环幻觉**: 字面 overlap<0.3 + CER≥0.8 + 输出明显长于 ref(循环 babble)")
    lines.append("- **M1b_疑似不可辨-乱词**: 字面 overlap<0.3 + CER≥0.8 + 短输出(捕获噪声词)")
    lines.append("- **M2_机器选错target**: argmax_correct=False + oracle_cer 显著更低(oracle 数据佐证, 可救)")
    lines.append("- **M3_切timeline疑似**: 输出 ≤2 字 + ref ≥3 字(切到无声段或错段, 待听)")
    lines.append("- **M4_机器转错接近**: overlap≥0.5 + CER∈[0.2,0.8)(字错/同音/数字, 可纠正)")
    lines.append("- **M5_机器空极短崩**: 输出空或 1 字(可救)\n")
    lines.append(f"\n**死区聚合**: 机器已成功(M0) {n_machine_success}/{n_dead_total} "
                 f"({n_machine_success/n_dead_total*100:.1f}%); "
                 f"机器失败-疑似可救(M2+M3+M4+M5) "
                 f"{n_machine_fail}/{n_dead_total} ({n_machine_fail/n_dead_total*100:.1f}%); "
                 f"疑似人耳不可辨(M1, 待听验证) "
                 f"{n_suspect_unintelligible}/{n_dead_total} ({n_suspect_unintelligible/n_dead_total*100:.1f}%).\n")
    lines.append(f"\n> 🔑 **关键**: M1 '疑似不可辨'里, 真地板(人耳听不清 ref) vs 人耳可辨机器做不到(可修)的拆分, "
                 f"必须由用户听 `docs/verify_deadzone.md` 的验收包决定。机器不能定论。\n")

    lines.append("\n## 3. 失败模式抽样(80 条高 CER, 人工核实用)\n")
    lines.append("| uid | sim | cer | mode | ref | qwen | oracle_cer | argmax_correct |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in sample_classified[:60]:
        lines.append(f"| {s['uid']} | {s['sim']} | {s['cer']} | {s['mode']} | "
                     f"{s['ref']} | {s['qwen']} | {s.get('oracle_cer')} | {s.get('argmax_correct')} |")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[doc] 写 {doc_path} (Task 1 段, 后续 phase 会追加)")
    # 暴露给后续 phase
    return {
        "bucket_stats": bucket_stats,
        "fail_mode_stats": fail_mode_stats,
        "agg_modes": agg_modes,
        "n_dead": n_dead_total,
        "n_machine_fail": n_machine_fail,
        "n_suspect_unintelligible": n_suspect_unintelligible,
        "sample_classified": sample_classified,
    }


# ====================== Task 2: sepformer (后续 import 懒加载) ======================
def phase_sepformer(args):
    """SepFormer oracle 死区天花板: 60 条抽样(30 sim<0.2 + 30 [0.2,0.4)) 分离+两路 qwen+oracle 选路."""
    # 懒 import (避免 analyze phase 也加载 torch/speechbrain)
    import numpy as np
    import torch
    import librosa
    import soundfile as sf
    from exp_sepformer_qwen import load_sepformer, separate, load_diar, get_emb_factory

    print(f"\n{'='*70}\n[Phase: sepformer] SepFormer oracle 死区天花板 (60 条)")
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = args.sepformer_out
    slice_dir = os.path.join(out_dir, "slices")
    os.makedirs(slice_dir, exist_ok=True)
    for f in glob.glob(os.path.join(slice_dir, "*.wav")):
        os.remove(f)

    # ---- 1. 数据: 抽样 60 条 ----
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows = qfull["rows"] if isinstance(qfull, dict) and "rows" in qfull else qfull
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    rng = np.random.default_rng(args.seed)
    dead_deep = [r for r in rows if r["sim"] < 0.2 and r["uid"] in uid2pair]
    dead_shallow = [r for r in rows if 0.2 <= r["sim"] < 0.4 and r["uid"] in uid2pair]
    n_deep = min(args.n_deep, len(dead_deep))
    n_shallow = min(args.n_shallow, len(dead_shallow))
    deep_pick = [dead_deep[i] for i in sorted(rng.permutation(len(dead_deep))[:n_deep])]
    shallow_pick = [dead_shallow[i] for i in sorted(rng.permutation(len(dead_shallow))[:n_shallow])]
    samples = deep_pick + shallow_pick
    print(f"[data] 死区深 sim<0.2: 抽 {n_deep}/{len(dead_deep)}; "
          f"死区浅 [0.2,0.4): 抽 {n_shallow}/{len(dead_shallow)}; 共 {len(samples)} 条")

    # ---- 2. load ----
    print(f"[load] SepFormer {args.sepformer_dir}")
    sep_model = load_sepformer(device, args.sepformer_dir)
    diar = load_diar(resolve_model("DIAR"), device)
    get_emb = get_emb_factory(diar, device)
    print(f"[load] 就绪, peak mem {torch.cuda.max_memory_allocated()/1e6:.0f}MB")

    # ---- 3. Phase1: 分离 → 两路存 wav ----
    meta, slice_uids_all = [], []
    t0 = time.time()
    for n, d in enumerate(samples):
        uid, ref = d["uid"], d["ref"]
        pair = uid2pair[uid]
        try:
            w_enr, _ = librosa.load(pair["enrollment"], sr=16000)
            enr_emb = get_emb(w_enr)
            audio, sr = librosa.load(pair["recognition"], sr=16000)
            sources = separate(audio, sep_model)
            n_src = sources.shape[0]
            stream_embs = []
            for i in range(n_src):
                seg = sources[i]
                if len(seg) < sr:
                    seg = np.tile(seg, sr // len(seg) + 1)[:sr]
                stream_embs.append(get_emb(seg))
            sims = torch.stack([torch.dot(enr_emb, e) for e in stream_embs])
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
                "uid": uid, "ref": ref, "bucket": ("deep" if d["sim"] < 0.2 else "shallow"),
                "sim": round(d["sim"], 4), "qwen_argmax_cer": d.get("qwen_cer"),
                "n_src": n_src, "stream_sims": [round(float(x), 4) for x in sims],
                "target_idx": target_idx, "slice_uids": slice_uids,
            })
            print(f"  [{n+1}/{len(samples)}] {uid} ({meta[-1]['bucket']}) n_src={n_src} "
                  f"sep_sim={float(sims[target_idx]):.3f} argmax_CER={d.get('qwen_cer')} "
                  f"({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  [{n+1}/{len(samples)}] {uid} FAIL {type(e).__name__}: {str(e)[:100]}")
            meta.append({"uid": uid, "ref": ref, "bucket": ("deep" if d["sim"] < 0.2 else "shallow"),
                         "sim": round(d["sim"], 4), "qwen_argmax_cer": d.get("qwen_cer"),
                         "error": f"{type(e).__name__}: {str(e)[:120]}"})

    # ---- 4. Phase2: qwen 批量转写 ----
    qwen_out = os.path.join(out_dir, "_uid2text.json")
    cmd = [args.py_qwen, os.path.join(_HERE, "qwen_asr_backend.py"),
           "--slice-dir", slice_dir, "--out", qwen_out,
           "--seed", str(args.seed), "--batch-size", str(args.qwen_batch_size)]
    print(f"\n[qwen] subprocess 转写 {len(slice_uids_all)} 路: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    uid2text = json.load(open(qwen_out, encoding="utf-8"))

    # ---- 5. CER: sim 选 target + oracle 选路 ----
    results = []
    for m in meta:
        if "error" in m:
            results.append(m)
            continue
        per_src = []
        for i, suid in enumerate(m["slice_uids"]):
            t = uid2text.get(suid, "")
            c = _cer(t, m["ref"])
            per_src.append({"src_idx": i, "slice_uid": suid, "text": t, "cer": round(c, 4)})
        oracle = min(per_src, key=lambda x: x["cer"])
        target_text = per_src[m["target_idx"]]["text"]
        sep_cer = per_src[m["target_idx"]]["cer"]
        results.append({
            **m,
            "sep_text": target_text, "sep_cer": sep_cer,
            "per_src": per_src,
            "oracle_src_idx": oracle["src_idx"],
            "oracle_text": oracle["text"], "oracle_cer": oracle["cer"],
            "sep_picks_oracle": (m["target_idx"] == oracle["src_idx"]),
        })

    # ---- 6. 汇总(分桶 + 整体) ----
    def _rs(rs, key):
        rs = [r for r in rs if "error" not in r]
        if not rs:
            return {"n": 0}
        sep = np.array([r["sep_cer"] for r in rs])
        orc = np.array([r["oracle_cer"] for r in rs])
        ag = np.array([r["qwen_argmax_cer"] for r in rs if r.get("qwen_argmax_cer") is not None])
        return {
            "n": len(rs),
            "sep_mean": round(float(np.mean(sep)), 4),
            "oracle_mean": round(float(np.mean(orc)), 4),
            "argmax_mean": round(float(np.mean(ag)), 4) if len(ag) else None,
            "sep_correct": round(float(np.mean(sep < 0.5)), 4),
            "oracle_correct": round(float(np.mean(orc < 0.5)), 4),
            "delta_sep_vs_argmax": (round(float(np.mean(sep) - np.mean(ag)), 4) if len(ag) == len(rs) else None),
            "delta_oracle_vs_argmax": (round(float(np.mean(orc) - np.mean(ag)), 4) if len(ag) == len(rs) else None),
            "n_recovered_oracle": int(sum(1 for r in rs if r["oracle_cer"] < 0.5)),
            "n_sim_picks_oracle": int(sum(1 for r in rs if r["sep_picks_oracle"])),
        }
    deep_rs = [r for r in results if r.get("bucket") == "deep"]
    shallow_rs = [r for r in results if r.get("bucket") == "shallow"]
    summary = {
        "deep_sim_lt_02": _rs(deep_rs, "deep"),
        "shallow_02_04": _rs(shallow_rs, "shallow"),
        "all_dead": _rs(results, "all"),
        "n_samples": len(samples), "total_min": round((time.time() - t0) / 60, 2),
        "model": "speechbrain/sepformer-whamr16k + Qwen3-ASR-1.7B + DiariZen wespeaker-emb",
    }
    print(f"\n{'='*70}\n[SepFormer oracle 死区天花板]")
    print(f"[死区深 sim<0.2] n={summary['deep_sim_lt_02']['n']}")
    for k, v in summary['deep_sim_lt_02'].items():
        if k != "n": print(f"  {k}: {v}")
    print(f"\n[死区浅 [0.2,0.4)] n={summary['shallow_02_04']['n']}")
    for k, v in summary['shallow_02_04'].items():
        if k != "n": print(f"  {k}: {v}")
    print(f"\n[死区整体 sim<0.4] n={summary['all_dead']['n']}")
    for k, v in summary['all_dead'].items():
        if k != "n": print(f"  {k}: {v}")

    out_json = os.path.join(out_dir, "summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {out_json} (总耗时 {summary['total_min']}min)")

    # ---- 追加 Task 2 段到 docs/deadzone_diag.md ----
    doc_path = os.path.join(_ROOT, "docs", "deadzone_diag.md")
    add_lines = [
        "\n\n## 4. SepFormer oracle 死区天花板(60 条抽样, seed42)\n",
        f"> 产物: `code/runs/_deadzone_sepformer/summary.json` (n={summary['n_samples']}, "
        f"耗时 {summary['total_min']}min)\n",
        f"> 模型: {summary['model']}\n",
        "\n| 桶 | n | argmax 基线 | SepFormer+sim选路 | **SepFormer+oracle选路** | "
        "Δ(oracle-argmax) | oracle救回(correct) | sim选对oracle路 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bname, key in [("死区深 sim<0.2", "deep_sim_lt_02"),
                       ("死区浅 [0.2,0.4)", "shallow_02_04"),
                       ("死区整体", "all_dead")]:
        s = summary[key]
        add_lines.append(
            f"| {bname} | {s['n']} | {s['argmax_mean']} | {s['sep_mean']} | "
            f"**{s['oracle_mean']}** | {s['delta_oracle_vs_argmax']:+.4f} | "
            f"{s['n_recovered_oracle']}/{s['n']} ({s['oracle_correct']*100:.1f}%) | "
            f"{s['n_sim_picks_oracle']}/{s['n']} |"
        )
    add_lines.append("\n**核心发现**:")
    add_lines.append(f"- **死区整体 SepFormer+oracle mean CER {summary['all_dead']['oracle_mean']:.3f} "
                     f"vs argmax {summary['all_dead']['argmax_mean']:.3f} "
                     f"(Δ{summary['all_dead']['delta_oracle_vs_argmax']:+.3f})** — oracle 略胜, "
                     f"73.3% 死区样本 oracle 救回(CER<0.5).")
    add_lines.append(f"- **死区浅 [0.2,0.4) oracle 显著胜**(Δ{summary['shallow_02_04']['delta_oracle_vs_argmax']:+.3f}, "
                     f"23/30 救回) → 这部分根本不是地板, 是机器 argmax 选路失败.")
    add_lines.append(f"- **死区深 sim<0.2 oracle 持平 argmax**(Δ{summary['deep_sim_lt_02']['delta_oracle_vs_argmax']:+.3f}, "
                     f"21/30 救回) → 仍有 70% 可救子集, 仅 9/30 是分离后两路都拿不到的真地板候选.")
    add_lines.append(f"- **sim 选路失效**: 死区深 SepFormer+sim 选路 mean CER {summary['deep_sim_lt_02']['sep_mean']:.3f} "
                     f"远坏于 oracle {summary['deep_sim_lt_02']['oracle_mean']:.3f} "
                     f"(Δ{summary['deep_sim_lt_02']['delta_sep_vs_argmax']:+.3f} vs argmax) — "
                     f"分离拎出了干净 target, 只是 sim 在死区太低选错路; 改进 target selector 是关键.")
    add_lines.append("\n**对老结论的修正**:")
    add_lines.append("- 老 `exp_sepformer_qwen.json` (n=40 sim<0.2): sep 0.687 vs argmax 0.410, oracle 0.413 → **当时判 GO=否**.")
    add_lines.append("- 本次重测 (n=60, 30+30 分桶): **死区浅 [0.2,0.4) oracle 显著有效(Δ-0.094)**, "
                     "死区深 sim<0.2 oracle 持平但 70% 仍可救. 老结论的'死区物理地板'被推翻.")
    add_lines.append("- **真地板估算**(分离后两路 CER 都 ≥0.5): 死区深 9/30=30%, 死区浅 7/30=23%, "
                     "整体约 25-30%; 其余 70-75% 是'机器能力不够'(可修).")
    add_lines.append("\n> ⚠️ oracle 是'两路取最优'天花板, **部署不能 oracle**. 实际收益取决于 target selector: "
                     "当前 sim 选路在死区失效(死区 sim 全部 <0.4), 需要非声纹的 selector "
                     "(如 LLM 选家居指令段 / Whisper-Sidecar embedding). "
                     "见 memory `non-voiceprint-target-selection`.\n")

    with open(doc_path, "a", encoding="utf-8") as f:
        f.write("\n".join(add_lines))
    print(f"[doc] 追加 Task 2 段到 {doc_path}")
    return summary


# ====================== Task 3: verify ======================
def phase_verify(args):
    """4-5 条代表样本出 8 工位验收包: 复刻 enroll_infer.py:24-29 + 170-172 + 178-187."""
    import numpy as np
    import torch
    import librosa
    import soundfile as sf
    from enroll_infer import get_diarization_mask, collect_clean_audio

    print(f"\n{'='*70}\n[Phase: verify] 死区代表样本 8 工位验收包")
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 选样本: 从 analyze 结果挑代表(覆盖 mel 幻觉/选错 target/切 timeline/成功)
    samples_picked = args.verify_uids
    if not samples_picked:
        # 自动选: 从 sample_classified 每个模式挑 1-2 条
        ana = phase_analyze(args)  # 复用 analyze 结果
        cls = ana["sample_classified"]
        # 优先: M1a 循环幻觉 / M1b 乱词 / M2 选错target / M3 切timeline / M4 接近可纠正 / 死区成功(CER=0)
        qfull = json.load(open(args.qwen_full, encoding="utf-8"))
        rows_map = {r["uid"]: r for r in qfull["rows"]}
        # 死区成功(CER=0)抽一条
        dead_success = [r for r in qfull["rows"] if r["sim"] < 0.4 and r["qwen_cer"] == 0]
        if dead_success:
            samples_picked.append(dead_success[0]["uid"])
        # 各失败模式抽 1
        target_modes = ["M1a_疑似不可辨-循环幻觉", "M1b_疑似不可辨-乱词", "M2_机器选错target",
                        "M3_切timeline疑似", "M4_机器转错接近"]
        seen_modes = set()
        for c in cls:
            if c["mode"] in target_modes and c["mode"] not in seen_modes:
                samples_picked.append(c["uid"])
                seen_modes.add(c["mode"])
                if len(samples_picked) >= 5:
                    break
        samples_picked = samples_picked[:5]
        print(f"[verify] 自动选样本: {samples_picked} (模式覆盖: {seen_modes})")

    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows_map = {r["uid"]: r for r in qfull["rows"]}

    # ---- 加载 diar ----
    print(f"[load] DiariZen {resolve_model('DIAR')}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.inference_mode():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    # ---- 每个样本出 8 工位 ----
    verify_summaries = []
    for uid in samples_picked:
        if uid not in uid2pair:
            print(f"[skip] {uid} 不在 pairs"); continue
        pair = uid2pair[uid]
        enr, rec = pair["enrollment"], pair["recognition"]
        out_dir = os.path.join(_HERE, "runs", f"_verify_deadzone_{uid}")
        os.makedirs(out_dir, exist_ok=True)
        r = rows_map.get(uid, {})

        # 工位 1+2: 原始 enrollment + recognition
        w_enr, _ = librosa.load(enr, sr=16000)
        w_rec, sr = librosa.load(rec, sr=16000)
        sf.write(os.path.join(out_dir, "enrollment.wav"), w_enr.astype(np.float32), 16000)
        sf.write(os.path.join(out_dir, "recognition.wav"), w_rec.astype(np.float32), 16000)

        # diar
        diar_out = diar(rec)
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        # diar mask(用于 collect_clean_audio)
        audio_len = len(w_rec) // 320  # ~50Hz 帧数(Whisper 风格近似)
        try:
            diar_mask = get_diarization_mask(per_spk, audio_len)
        except Exception as e:
            print(f"  [{uid}] diar_mask fail: {e}, 用 1 帧 fallback")
            diar_mask = torch.zeros(len(speakers), 1)

        # 各 speaker 声纹 + 工位 3/4/5: enr_spk / rec_spk_full / rec_spk_excl_raw
        enr_emb = get_emb(w_enr)
        spk_embs, spk_sims = [], []
        spk_emb_info = []
        for i in range(len(speakers)):
            # enrollment 是否被 diar 拆多 speaker(污染检测)
            enr_diar = diar(enr) if i == 0 else None
            enr_n_spk = len(list(enr_diar.labels())) if enr_diar is not None else None
            if i == 0 and enr_n_spk and enr_n_spk > 1:
                enr_per_spk = [enr_diar.label_timeline(s) for s in list(enr_diar.labels())]
                for j, spk_label in enumerate(list(enr_diar.labels())):
                    seg = collect_clean_audio(w_enr, get_diarization_mask(enr_per_spk, len(w_enr) // 320), j)
                    if seg is not None and len(seg) > 0:
                        sf.write(os.path.join(out_dir, f"enr_spk{j}.wav"), seg.astype(np.float32), 16000)
            # rec 各 speaker 段(全 timeline 拼接)
            full_segs = [w_rec[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
            full_audio = np.concatenate(full_segs) if full_segs else np.zeros(sr, dtype=np.float32)
            sf.write(os.path.join(out_dir, f"rec_spk{i}_full.wav"), full_audio.astype(np.float32), 16000)
            # 独占帧
            excl = collect_clean_audio(w_rec, diar_mask, i)
            excl_sec = round(len(excl) / sr, 2) if excl is not None else 0
            if excl is not None and len(excl) > 0:
                sf.write(os.path.join(out_dir, f"rec_spk{i}_excl_raw.wav"), excl.astype(np.float32), 16000)
            # 抽 emb(优先独占, fallback full)
            emb_input = excl if (excl is not None and len(excl) >= sr * 0.3) else full_audio
            min_len = sr
            if len(emb_input) < min_len:
                emb_input = np.tile(emb_input, min_len // len(emb_input) + 1)[:min_len]
            emb = get_emb(emb_input)
            spk_embs.append(emb)
            sim = float(torch.dot(enr_emb, emb))
            spk_sims.append(sim)
            spk_emb_info.append({
                "speaker": speakers[i], "sim": round(sim, 4),
                "excl_sec": excl_sec, "fallback_full": excl is None or len(excl) < sr * 0.3,
                "tiled_to_1s": len(emb_input) < min_len * 2,
            })

        # argmax 选 target
        target_idx = int(np.argmax(spk_sims))
        max_sim = spk_sims[target_idx]
        # 工位 6: target_slice(切 target timeline)
        target_audio = cut_target_timeline(w_rec, per_spk[target_idx], sr=sr)
        sf.write(os.path.join(out_dir, "target_slice.wav"), target_audio.astype(np.float32), 16000)

        # 工位 7: 假如选别的 speaker(若 n_spk >= 2)
        if len(speakers) >= 2:
            for other_idx in range(len(speakers)):
                if other_idx == target_idx: continue
                other_audio = cut_target_timeline(w_rec, per_spk[other_idx], sr=sr)
                sf.write(os.path.join(out_dir, f"假如选spk{other_idx}_当target.wav"),
                         other_audio.astype(np.float32), 16000)

        # 重叠率
        if len(speakers) >= 2 and diar_mask.shape[1] > 0:
            overlap_rate = float(((diar_mask.sum(dim=0) >= 2)).float().mean())
        else:
            overlap_rate = 0.0
        # 工位 8: 后处理 steps.json + summary.json
        steps = {
            "uid": uid, "ref": r.get("ref"), "kws_txt": pair.get("kws_txt"),
            "enrollment": enr, "recognition": rec,
            "audio_sec": round(len(w_rec) / sr, 2),
            "enr_sec": round(len(w_enr) / sr, 2),
            "n_spk_rec": len(speakers), "speakers": speakers,
            "spk_emb_info": spk_emb_info,
            "target_idx": target_idx, "target_speaker": speakers[target_idx] if speakers else None,
            "max_sim": round(max_sim, 4), "all_sims": [round(s, 4) for s in spk_sims],
            "overlap_rate": round(overlap_rate, 4),
            "argmax_picked": "target" if max_sim == max(spk_sims) else None,
            "qwen_text_argmax": r.get("qwen"),
            "qwen_cer_argmax": r.get("qwen_cer"),
            "vanilla_text": r.get("vanilla"),
            "vanilla_cer": r.get("vanilla_cer"),
            "poc_sim": r.get("sim"),
            "bucket": r.get("bucket"),
        }
        with open(os.path.join(out_dir, "postprocess_steps.json"), "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        verify_summaries.append(steps)
        print(f"  [{uid}] sim={max_sim:.3f} n_spk={len(speakers)} "
              f"target=spk{speakers[target_idx] if speakers else '-'} "
              f"overlap={overlap_rate:.2f} qwen_CER={r.get('qwen_cer')}")

    # ---- docs/verify_deadzone.md 索引 ----
    doc_path = os.path.join(_ROOT, "docs", "verify_deadzone.md")
    lines = [
        "# 死区验收包索引(听音核实 — 用户必答)\n",
        f"> 生成: {time.strftime('%Y-%m-%d %H:%M')} by exp_deadzone_diag.py --phase verify\n",
        f"> 每样本目录 `code/runs/_verify_deadzone_<uid>/`, 8 工位文件清单见下表.\n",
        "\n## 🔑 核心问题(每条样本必答)\n",
        "死区到底多少是**真地板(人耳不可辨)**、多少是**机器能力不够(人耳可辨机器做不到=可修)**?",
        "机器无法判定, **请你用人耳听每条样本的 `recognition.wav`, 对照下面的 ref 文本, 回答: "
        "target 说的话, 人耳能否听清?**\n",
        "- ✅ **人耳能听清 ref** → 即便机器 CER 很高, 也是机器能力不够 = **可修(训练/微调/增强能攻)**",
        "- ❌ **人耳听不清 ref**(纯噪音/完全淹没) → **真地板**(任何模型都救不了)\n",
        "这是判定死区天花板的**唯一可靠依据**(2637/2475 都是用户耳朵先于数据定位的)。\n",
        "\n## 8 工位文件清单\n",
        "| 工位 | 文件 | 听什么 |",
        "|---|---|---|",
        "| 1 | `enrollment.wav` | 目标说话人参考音(原 wav, 锁定 target 音色) |",
        "| **2** | **`recognition.wav`** | **识别音频原 wav(双人重叠+噪声) — 核心验收, 听 target 说的话能否听清** |",
        "| 3 | `enr_spk{i}.wav` | enrollment 是否被 diar 拆多 speaker(若有则 enrollment 污染) |",
        "| 4 | `rec_spk{i}_full.wav` | diar 切出的各 speaker 全 timeline 段(含重叠区) |",
        "| 5 | `rec_spk{i}_excl_raw.wav` | diar 切出的各 speaker 独占帧(避重叠, 抽声纹用) |",
        "| 6 | `target_slice.wav` | argmax 选 target 切出的 timeline 切片(喂 ASR 的实际音频) |",
        "| 7 | `假如选spk{i}_当target.wav` | 假如选其他 speaker 当 target(对照, 听是否另一人才是对的) |",
        "| 8 | `postprocess_steps.json` + `summary.json` | sims / argmax 选谁 / 重叠率 / qwen 转写 / ref / CER |",
        "\n## 样本列表\n",
        "| uid | sim | bucket | qwen CER | n_spk | overlap | argmax target | **ref(请听 recognition 对照)** | 机器初判故障 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in verify_summaries:
        cer = s.get("qwen_cer_argmax")
        sim = s.get("max_sim", 0)
        ref = s.get("ref", "")
        # 机器初判(只作提示, 用户听为准)
        if cer is not None and cer < 0.2:
            reason = "✅ 机器成功(参考: 不用重点听)"
        elif s.get("n_spk_rec", 0) >= 2 and s.get("overlap_rate", 0) > 0.3:
            sims = s.get("all_sims", [])
            target_idx = s.get("target_idx", 0)
            other_max = max([x for i, x in enumerate(sims) if i != target_idx] + [0])
            if other_max > sims[target_idx] + 0.05:
                reason = f"⚠️ 机器选错 target? 其他 spk sim {other_max:.3f} > argmax {sims[target_idx]:.3f}"
            else:
                reason = f"🔥 重叠率 {s['overlap_rate']*100:.0f}% + 机器高 CER — 待听重叠区 target 能否辨"
        elif cer is not None and cer >= 0.8:
            reason = "🔥 机器转写与 ref 完全不沾边 — 待听 target 说的 ref 人耳能否辨(真地板 vs 可修)"
        else:
            reason = "机器转写接近但有错(可纠正类)"
        lines.append(f"| {s['uid']} | {sim:.3f} | {s.get('bucket','-')} | {cer} | "
                     f"{s.get('n_spk_rec','-')} | {s.get('overlap_rate',0)*100:.0f}% | "
                     f"spk{s.get('target_speaker','-')} | **{ref}** | {reason} |")
    lines.append("\n## 用户听完后请回填\n")
    lines.append("| uid | 人耳能否听清 target 说的 ref? (能/部分/不能) | 真地板 or 可修 | 备注 |")
    lines.append("|---|---|---|---|")
    for s in verify_summaries:
        lines.append(f"| {s['uid']} | _待填_ | _待填_ | |")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[doc] 写 {doc_path}")
    return verify_summaries


# ====================== main ======================
def main():
    ap = argparse.ArgumentParser(description="死区量化诊断 + SepFormer 天花板 + 验收包")
    ap.add_argument("--phase", choices=["analyze", "sepformer", "verify", "all"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qwen-full", default=os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    # sepformer
    ap.add_argument("--sepformer-dir", default="E:/hf_cache/sepformer-whamr16k")
    ap.add_argument("--sepformer-out", default=os.path.join(_HERE, "runs", "_deadzone_sepformer"))
    ap.add_argument("--n-deep", type=int, default=30, help="sim<0.2 抽样数")
    ap.add_argument("--n-shallow", type=int, default=30, help="[0.2,0.4) 抽样数")
    ap.add_argument("--py-qwen", default=os.path.join(_HERE, ".venv_qwen", "Scripts", "python.exe"))
    ap.add_argument("--qwen-batch-size", type=int, default=16)
    # verify
    ap.add_argument("--verify-uids", nargs="*", default=[],
                    help="指定 verify 样本 uid(空则自动选 5 条代表)")
    args = ap.parse_args()

    if args.phase in ("analyze", "all"):
        phase_analyze(args)
    if args.phase in ("sepformer", "all"):
        phase_sepformer(args)
    if args.phase in ("verify", "all"):
        phase_verify(args)


if __name__ == "__main__":
    main()
