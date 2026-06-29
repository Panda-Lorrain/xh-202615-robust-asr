"""端到端融合评测: enroll_infer 转写 + LLM 拒识 → 多策略融合 → 真实组合指标。

把三线(SE 已前置→enroll_infer 转写+max_sim / LLM 语义 verdict)在结果层融合,
在 450 条 target-在场数据集上算**真实组合指标**(三线首次合到一个分数):

  - cer_final     : 竞赛 CER(拒识计 1.0=漏 target; 接受计 CER(transcript,ref))。越低越好。
  - cer_accepted  : 仅"接受"条的转写质量(衡量放进来的条有多干净)。
  - correct_rate  : 接受 且 CER<0.5 的比例 ≈ 锁定对+转写对。越高越好。
  - reject_rate   : 拒识率(target 全在场→即**误拒率**, 越低越好; 真实拒识率见 target-absent 集)。
  - rtf           : 推理实时因子均值。

融合策略(扫 sim_threshold + 策略组合), 找最优工作点:
  sim_only(thr)    : 拒 iff max_sim<thr                 (baseline, 当前 ~87% 误拒)
  llm_only         : 拒 iff LLM=reject                  (纯语义, 救回 sim 误拒)
  llm_or_sim(thr)  : 接受 iff (LLM=accept OR max_sim>=thr)   (LLM 救回 sim 误拒, 主推)
  llm_and_sim(thr) : 接受 iff (LLM=accept AND max_sim>=thr)  (保守双确认)
  weighted(w,thr)  : score=w·[llm_acc]+(1-w)·[sim>=thr]; 接受 iff score>=0.5

用法(三步, 见文末命令; fuse_eval 自带 --prep-llm-input 抽转写):
  # 1) 抽 enroll 转写 → LLM 输入
  code/.venv/Scripts/python.exe code/fuse_eval.py --prep-llm-input \
      --enroll-json code/se_denoised.json --out code/llm_input_se0.json
  # 2) 跑 LLM 拒识(推理模式, 独立 .venv_llm)
  .venv_llm/Scripts/python.exe code/llm_reject.py --infer-json code/llm_input_se0.json \
      --out-json code/llm_verdicts_se0.json --device cuda:0
  # 3) 融合评测
  code/.venv/Scripts/python.exe code/fuse_eval.py \
      --enroll-json code/se_denoised.json --llm-json code/llm_verdicts_se0.json \
      --out code/fuse_result_se0.json
"""
import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_metrics import cer  # noqa  字符级 CER


def _basename(p):
    return os.path.basename(p or "")


def load_joined(enroll_json, llm_json, manifest):
    """三源按 recognition 文件名 join → list[dict(max_sim/transcript/target_ref/llm_pred/rtf/...)]。"""
    enroll = json.load(open(enroll_json, encoding="utf-8"))
    mani = json.load(open(manifest, encoding="utf-8"))
    mani_map = {_basename(it["file"]): it for it in mani["items"]}
    llm_map = {}
    if llm_json and os.path.exists(llm_json):
        lj = json.load(open(llm_json, encoding="utf-8"))
        for row in lj.get("rows", lj if isinstance(lj, list) else []):
            llm_map[_basename(row.get("file", ""))] = row.get("pred", "reject")

    items = []
    for r in enroll:
        f = _basename(r.get("recognition", ""))
        m = mani_map.get(f, {})
        items.append({
            "file": f,
            "max_sim": float(r.get("max_sim", 0.0) or 0.0),
            "transcript": r.get("transcript", "") or "",
            "target_ref": m.get("target_ref", ""),
            "target_present": m.get("target_present", True),   # 默认在场(450 集)
            "overlap": m.get("overlap_ratio"),
            "snr": m.get("snr_db"),
            "noise_type": m.get("noise_type"),
            "llm_pred": llm_map.get(f, "reject"),              # 缺 LLM verdict → 保守拒识
            "stno_target_ratio": r.get("stno_target_ratio"),   # 三路第三信号(可能 None)
            "rtf": float(r.get("rtf", 0.0) or 0.0),
            "diar_fail": bool(r.get("error")),
        })
    return items


def evaluate(items, name, accept_fn, lock_cer=0.5):
    """accept_fn(it)->bool(True=接受/转写)。返回指标 dict。"""
    n = len(items)
    n_acc = n_rej = n_correct = 0
    cer_final_sum = 0.0          # 拒识计 1.0(漏 target)
    cer_acc_sum = 0.0            # 仅接受条
    cer_acc_n = 0
    rtf_sum = 0.0
    for it in items:
        rtf_sum += it["rtf"]
        if accept_fn(it):
            n_acc += 1
            c = cer(it["transcript"], it["target_ref"])
            cer_final_sum += c
            cer_acc_sum += c
            cer_acc_n += 1
            if c < lock_cer:
                n_correct += 1
        else:
            n_rej += 1
            # 拒识: target 在场→漏转(CER 1.0); target 缺席→正确拒(CER 0)
            cer_final_sum += 0.0 if not it["target_present"] else 1.0
    return {
        "config": name, "n": n,
        "accept_rate": round(n_acc / n, 3),
        "reject_rate": round(n_rej / n, 3),
        "cer_final": round(cer_final_sum / n, 4),
        "cer_accepted": round(cer_acc_sum / cer_acc_n, 4) if cer_acc_n else None,
        "correct_rate": round(n_correct / n, 3),
        "rtf": round(rtf_sum / n, 3) if n else 0,
    }


def build_configs(sim_thrs, weights, has_stno=False):
    """返回 [(name, accept_fn)]。accept_fn(it)->True=接受。has_stno=True 时追加三路配置。"""
    cfgs = []
    for t in sim_thrs:
        cfgs.append((f"sim_only(t={t})", lambda it, t=t: it["max_sim"] >= t))
    cfgs.append(("llm_only", lambda it: it["llm_pred"] == "accept"))
    for t in sim_thrs:
        cfgs.append((f"llm_or_sim(t={t})",
                     lambda it, t=t: it["llm_pred"] == "accept" or it["max_sim"] >= t))
    for t in sim_thrs:
        cfgs.append((f"llm_and_sim(t={t})",
                     lambda it, t=t: it["llm_pred"] == "accept" and it["max_sim"] >= t))
    for w in weights:
        for t in (0.2, 0.3):
            cfgs.append((f"weighted(w_llm={w},t={t})",
                         lambda it, w=w, t=t:
                         w * (1.0 if it["llm_pred"] == "accept" else 0.0)
                         + (1 - w) * (1.0 if it["max_sim"] >= t else 0.0) >= 0.5))
    if has_stno:
        for t in (0.01, 0.05, 0.1, 0.2):
            cfgs.append((f"stno_only(t={t})",
                         lambda it, t=t: (it["stno_target_ratio"] or 0.0) >= t))
        cfgs.append(("llm_or_stno(t=0.05)",
                     lambda it: it["llm_pred"] == "accept"
                     or (it["stno_target_ratio"] or 0.0) >= 0.05))
        # 三路加权(对齐 llm_reject.fuse_three_ways 默认 0.4/0.4/0.2)
        for (wl, ws, wst) in [(0.4, 0.4, 0.2), (0.5, 0.3, 0.2), (0.6, 0.2, 0.2)]:
            cfgs.append((f"three_way(w_llm={wl},w_sim={ws},w_stno={wst})",
                         lambda it, wl=wl, ws=ws, wst=wst:
                         wl * (1.0 if it["llm_pred"] == "accept" else 0.0)
                         + ws * (1.0 if it["max_sim"] >= 0.2 else 0.0)
                         + wst * (1.0 if (it["stno_target_ratio"] or 0.0) >= 0.05 else 0.0) >= 0.5))
    return cfgs


def prep_llm_input(enroll_json, out):
    """从 enroll JSON 抽 [{file,text}] 供 llm_reject --infer-json。"""
    enroll = json.load(open(enroll_json, encoding="utf-8"))
    rows = [{"file": _basename(r.get("recognition", "")),
             "text": r.get("transcript", "") or ""} for r in enroll]
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_empty = sum(1 for r in rows if not r["text"].strip())
    print(f"[prep] {len(rows)} 条转写 → {out} (空转写 {n_empty} 条→LLM 视为拒识)")


def main():
    ap = argparse.ArgumentParser(description="端到端融合评测(SE+enroll+LLM)")
    ap.add_argument("--enroll-json", required=True, help="enroll_infer 输出 JSON")
    ap.add_argument("--llm-json", help="llm_reject --infer-json 输出(verdicts)")
    ap.add_argument("--manifest", default="E:/midea_target_asr/test_wav/dataset/final/final_manifest.json")
    ap.add_argument("--lock-cer", type=float, default=0.5, help="判定锁定+转写正确的 CER 阈值")
    ap.add_argument("--out", help="输出融合结果 JSON")
    ap.add_argument("--prep-llm-input", action="store_true", help="只抽转写到 --out(供 llm_reject)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.prep_llm_input:
        prep_llm_input(args.enroll_json, args.out)
        return

    if not args.llm_json:
        ap.error("融合评测需 --llm-json(或用 --prep-llm-input 先抽转写)")

    items = load_joined(args.enroll_json, args.llm_json, args.manifest)
    n_present = sum(1 for it in items if it["target_present"])
    n_diar_fail = sum(1 for it in items if it["diar_fail"])
    print(f"[fuse] {len(items)} 条 (target 在场 {n_present} / diar-fail {n_diar_fail}) "
          f"{'('+args.label+')' if args.label else ''}")
    print(f"       max_sim 范围 [{min(it['max_sim'] for it in items):.3f}, "
          f"{max(it['max_sim'] for it in items):.3f}]  均值 "
          f"{sum(it['max_sim'] for it in items)/len(items):.3f}")
    n_llm_acc = sum(1 for it in items if it["llm_pred"] == "accept")
    print(f"       LLM accept {n_llm_acc}/{len(items)} ({n_llm_acc/len(items):.2%})")
    # raw 转写 CER(质量口径, 可>1: hyp 长于 ref 的幻觉/重复)——与下方 cer_final(拒识计1.0)双口径
    raw_cers = [cer(it["transcript"], it["target_ref"]) for it in items if it["target_ref"]]
    raw_cer = sum(raw_cers) / len(raw_cers) if raw_cers else float("nan")
    n_good = sum(1 for c in raw_cers if c < 0.5)
    print(f"       [双口径] raw转写CER={raw_cer:.3f}(质量,可>1; good<0.5 占 {n_good}/{len(raw_cers)}={n_good/max(len(raw_cers),1):.1%}) | "
          f"cer_final=拒识计1.0(竞赛口径,见下表各配置)")
    has_stno = all(it["stno_target_ratio"] is not None for it in items) and len(items) > 0
    if has_stno:
        print(f"       [三路] stno_target_ratio 可用(均值 "
              f"{sum(it['stno_target_ratio'] for it in items)/len(items):.3f}), 追加 stno/三路配置")

    cfgs = build_configs([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], [0.3, 0.5, 0.7], has_stno)
    results = [evaluate(items, name, fn, args.lock_cer) for name, fn in cfgs]

    # 排序: correct_rate 降序 → cer_final 升序(正确转写优先, 兼顾 CER)
    results_sorted = sorted(results, key=lambda r: (-r["correct_rate"], r["cer_final"]))

    print(f"\n{'config':<26} {'accept':>7} {'reject':>7} {'cer_fin':>8} {'cer_acc':>8} "
          f"{'correct':>8} {'rtf':>6}")
    print("-" * 76)
    for r in results_sorted:
        ca = f"{r['cer_accepted']:.3f}" if r['cer_accepted'] is not None else "  -  "
        print(f"{r['config']:<26} {r['accept_rate']:>7.2f} {r['reject_rate']:>7.2f} "
              f"{r['cer_final']:>8.3f} {ca:>8} {r['correct_rate']:>8.2f} {r['rtf']:>6.3f}")

    best = results_sorted[0]
    # sim_only(t=0.2) 作 baseline 对比
    base = next((r for r in results if r["config"] == "sim_only(t=0.2)"), None)
    print(f"\n[最优] {best['config']}: correct={best['correct_rate']:.2f} "
          f"cer_final={best['cer_final']:.3f} reject={best['reject_rate']:.2f}")
    print(f"  注: correct_rate=接受且CER<{args.lock_cer}的比例; 在弱转写数据上各配置普遍偏低(瓶颈在转写质量, 非融合)")
    if base:
        print(f"[对比 sim_only(t=0.2) baseline] correct {base['correct_rate']:.2f}→{best['correct_rate']:.2f} "
              f"(Δ{best['correct_rate']-base['correct_rate']:+.2f}), "
              f"cer_final {base['cer_final']:.3f}→{best['cer_final']:.3f} "
              f"(Δ{best['cer_final']-base['cer_final']:+.3f}), "
              f"reject {base['reject_rate']:.2f}→{best['reject_rate']:.2f}")

    if args.out:
        json.dump({"label": args.label, "n": len(items),
                   "n_target_present": n_present, "n_diar_fail": n_diar_fail,
                   "raw_cer_quality": round(raw_cer, 4),
                   "raw_cer_good_rate": round(n_good / max(len(raw_cers), 1), 4),
                   "best": best, "baseline_sim_t02": base,
                   "all_configs": results_sorted},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n[done] → {args.out}")


if __name__ == "__main__":
    main()
