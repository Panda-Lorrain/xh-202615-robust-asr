"""多信号拒识探索 (task: n_spk=2 rejected 445 条瓶颈).

核心洞察:
  当前拒识只靠 mainline wespeaker max_sim<0.27→拒. n_spk=2 重叠样本里 445/805 (55%) 被拒,
  在含拒口径下算 CER=1.0. 但低 sim 往往是 babble/enrollment 污染所致, 不代表 target 真不在说.
  如果用分离 target 路转写内容作为补救信号, 接受"分离+内容证明 target 在说话"的低 sim 样本,
  含拒 CER 能大幅下降.

数据 (纯计算模拟, 不跑 SepFormer/ASR/LLM):
  - code/runs/_scene_route_full/per_sample.json  (pos 1350, 含 mainline/scene_route 转写+CER+sep_info)
  - code/runs/out_neg_vanilla_full/result.json   (neg 474, max_sim + vanilla 混音 text)

多信号规则:
  - n_spk=1: 主线 sim<0.27 拒, 否则接受+transcribe
  - n_spk=2: 主线 sim<0.27 时, 看 SepFormer heuristic 选的 target 路 (scene_route_text):
      若 target 路通过内容判别 (gate / gate+home_kw / 严格) → 接受 (补救), 用 scene_route_text
      否则 → 拒
    主线 sim>=0.27 → 与 baseline 一致 (接受+transcribe)

工作点扫:
  - rescue_pred 强度: gate_only / gate+home_kw / gate+home_kw+min_len
  - 主线 sim thr: 0.20 / 0.27 / 0.30 (含拒基线对齐)

风险双界 (neg 没跑 SepFormer, 用代理):
  - 下界 (乐观): neg 用 vanilla 混音 text 代理 (vanilla 全空 → 损失 0)
  - 上界 (悲观): pos n_spk=2 的"另一路"(非 heuristic 选的)通过内容判别的比例作风险代理

产物:
  - code/runs/_multi_signal_reject/analysis.json   445 被拒样本的分离 target 路内容+CER 分析
  - code/runs/_multi_signal_reject/result.json      多信号 vs 单 sim 的 CER+RR + Pareto + 风险
"""
from __future__ import annotations
import json
import re
import statistics
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

import eval_metrics as em
import text_utils as tu

# ----------------------------------------------------------------------------
# Home-command keyword set (derived from pos refs bigrams + general home domain).
# 用途: 增强 content_gate, 把"看上去像家居指令"判严. 这是领域先验 (非 A 集拟合),
# 复用 text_utils.py 同源风格.
# ----------------------------------------------------------------------------
HOME_KW = [
    # 高频 ref bigrams (top from pos refs)
    "空调", "调到", "模式", "打开", "播放", "风速", "温度", "开启",
    "食物", "适合", "灯光", "百分", "关掉", "窗帘", "风量", "音量",
    "二十", "三十", "制热", "制冷", "除湿", "送风", "自动", "低速",
    "中速", "高速", "静音", "睡眠", "节能", "速热", "净化", "杀菌",
    # 家居设备 / 功能
    "电视", "音乐", "电影", "剧", "小说", "相声", "节目", "歌曲", "歌手",
    "电台", "频道", "广播", "新闻", "讲故事", "冰箱", "洗衣机", "扫地机",
    "净化器", "加湿器", "音箱", "热水器", "风扇", "窗户", "门锁", "监控",
    "定时", "设置", "提醒", "闹钟", "屏幕", "清洁", "童锁", "自清洁",
    # 动作词
    "关闭", "开启", "关掉", "调", "设定", "降低", "升高", "增加", "减少",
]


def has_home_kw(text: str) -> bool:
    if not text:
        return False
    return any(k in text for k in HOME_KW)


def cn_len(text: str) -> int:
    """中文范围字符数 (去标点空白后)."""
    if not text:
        return 0
    return sum(1 for c in re.sub(r"[^\w一-鿿]", "", text) if "一" <= c <= "鿿")


# ----------------------------------------------------------------------------
# rescue predicates (内容判别信号, 决定低 sim n_spk=2 样本是否补救接受)
# ----------------------------------------------------------------------------
def rescue_gate_only(text: str) -> bool:
    return bool(text and tu.is_valid_command(text))


def rescue_gate_kw(text: str) -> bool:
    return bool(text and tu.is_valid_command(text) and has_home_kw(text))


def rescue_gate_kw_len(text: str, min_cn_len: int = 3) -> bool:
    return bool(
        text
        and tu.is_valid_command(text)
        and has_home_kw(text)
        and cn_len(text) >= min_cn_len
    )


def rescue_gate_kw_strict(text: str, min_cn_len: int = 4) -> bool:
    """更严: gate + kw + 至少 4 中文字 (过滤 '低速风' '空调开' 等极短)."""
    return rescue_gate_kw_len(text, min_cn_len)


RESCUE_PREDS = {
    "gate_only": rescue_gate_only,
    "gate+kw": rescue_gate_kw,
    "gate+kw+len>=3": lambda t: rescue_gate_kw_len(t, 3),
    "gate+kw+len>=4": lambda t: rescue_gate_kw_len(t, 4),
    "gate+kw+len>=5": lambda t: rescue_gate_kw_len(t, 5),
}


# ----------------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------------
def cer_off(hyp: str, ref: str) -> float:
    return em.cer_official(hyp or "", ref)


def evaluate_pos(samples, rescue_pred, sim_thr=0.27):
    """pos 含拒 CER (累计池): 主线 sim<thr 拒, 否则 transcribe scene_route_text.
    n_spk=2 + sim<thr 时, 若 rescue_pred(scene_route_text) True → 补救接受."""
    m = em.CERMetric()
    cnt = {"rescued": 0, "rejected": 0, "accepted_mainline": 0, "rescue_real": 0,
           "rescue_bad": 0, "nspk2_rej": 0}
    for s in samples:
        rejected = s["max_sim"] < sim_thr
        if not rejected:
            pred = s.get("scene_route_text") or ""
            cnt["accepted_mainline"] += 1
        else:
            if s["n_spk"] == 2 and rescue_pred(s.get("scene_route_text") or ""):
                pred = s.get("scene_route_text") or ""
                cnt["rescued"] += 1
                # 真救回 vs 误接受 (基于 oracle CER vs ref)
                real_cer = cer_off(pred, s["ref"])
                if real_cer < 0.5:
                    cnt["rescue_real"] += 1
                else:
                    cnt["rescue_bad"] += 1
            else:
                pred = ""
                cnt["rejected"] += 1
                if s["n_spk"] == 2:
                    cnt["nspk2_rej"] += 1
        m.update([pred], [s["ref"]])
    return m.compute()["cer"], cnt


def evaluate_neg(neg_samples, rescue_pred, sim_thr=0.27, sep_text_proxy="vanilla"):
    """neg RR. 主线 sim<thr 拒, 否则漏拒 (误接受).
    多信号: sim<thr 时若 rescue_pred(代理 sep text) True → 补救接受 (RR 损失).
    sep_text_proxy:
      'vanilla'    用 neg vanilla 混音 text 作下界代理 (vanilla 全空 → 0 损失)
      'pos_other_path_rate' 用 pos n_spk=2 另一路通过率作上界代理 (按比例分摊)
    """
    rejected = 0
    rescued_proxy = 0
    n = len(neg_samples)
    for x in neg_samples:
        if x["max_sim"] < sim_thr:
            text = x.get("text") or ""
            if sep_text_proxy == "vanilla":
                if rescue_pred(text):
                    rescued_proxy += 1
                else:
                    rejected += 1
            else:
                # 上界代理: 假设 sep 路通过率 = pos_other_rate (传参算), 此处保守拒
                rejected += 1
        # else: 误接受, 不进 RR
    if sep_text_proxy == "vanilla":
        RR = rejected / n
        return RR, rescued_proxy
    return None, None


def neg_risk_upper_bound(pos_samples, rescue_pred):
    """neg RR 损失上界代理: pos n_spk=2 的'另一路'(非 heuristic 选的) 通过 rescue_pred 的比例.
    假设: neg 上 SepFormer 选的'target 路'(实际是干扰人) 与 pos 上'非 target 路'同分布."""
    nspk2 = [s for s in pos_samples if s["n_spk"] == 2 and s.get("sep_info")]
    other_pass = 0
    total = 0
    for s in nspk2:
        si = s["sep_info"]
        texts = si.get("per_src_texts") or []
        if len(texts) != 2:
            continue
        total += 1
        idx = si["heuristic_idx"]
        other_text = texts[1 - idx] or ""
        if rescue_pred(other_text):
            other_pass += 1
    rate = other_pass / total if total else 0.0
    return rate, other_pass, total


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    out_dir = CODE / "runs" / "_multi_signal_reject"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_samples = json.loads(
        (CODE / "runs" / "_scene_route_full" / "per_sample.json").read_text(encoding="utf-8")
    )
    neg_result = json.loads(
        (CODE / "runs" / "out_neg_vanilla_full" / "result.json").read_text(encoding="utf-8")
    )
    neg_samples = neg_result["results"]

    # ============================================================
    # A. 445 被拒 n_spk=2 样本的分离 target 路内容 + CER 分析
    # ============================================================
    nspk2_rej = [s for s in pos_samples if s["n_spk"] == 2 and s["rejected_thr0.27"]]
    n_rej = len(nspk2_rej)

    analysis = {"n_rejected_nspk2": n_rej, "samples": [
        {
            "uid": s["uid"],
            "max_sim": round(s["max_sim"], 4),
            "ref": s["ref"][:120],
            "scene_route_text": (s.get("scene_route_text") or "")[:120],
            "scene_route_cer": round(cer_off(s.get("scene_route_text") or "", s["ref"]), 4),
            "gate_pass": bool(tu.is_valid_command(s.get("scene_route_text") or "")),
            "gate_kw_pass": bool(rescue_gate_kw(s.get("scene_route_text") or "")),
            "heuristic_idx": (s.get("sep_info") or {}).get("heuristic_idx"),
            "oracle_idx": (s.get("sep_info") or {}).get("oracle_idx"),
            "per_src_cers": (s.get("sep_info") or {}).get("per_src_cers"),
        }
        for s in nspk2_rej
    ]}
    gate_pass_n = sum(1 for s in nspk2_rej if tu.is_valid_command(s.get("scene_route_text") or ""))
    gate_kw_pass_n = sum(1 for s in nspk2_rej if rescue_gate_kw(s.get("scene_route_text") or ""))

    cers_gate_pass = [cer_off(s["scene_route_text"], s["ref"])
                      for s in nspk2_rej if tu.is_valid_command(s.get("scene_route_text") or "")]
    cers_gate_kw_pass = [cer_off(s["scene_route_text"], s["ref"])
                         for s in nspk2_rej if rescue_gate_kw(s.get("scene_route_text") or "")]

    def cer_bin_stats(cers):
        return {
            "n": len(cers),
            "mean": round(statistics.mean(cers), 4) if cers else None,
            "median": round(statistics.median(cers), 4) if cers else None,
            "cer_eq_0": sum(1 for c in cers if c == 0),
            "cer_lt_0.3": sum(1 for c in cers if c < 0.3),
            "cer_lt_0.5": sum(1 for c in cers if c < 0.5),
            "cer_lt_1.0": sum(1 for c in cers if c < 1.0),
            "cer_ge_1.0": sum(1 for c in cers if c >= 1.0),
        }

    analysis["gate_pass"] = {"n": gate_pass_n, "cers": cer_bin_stats(cers_gate_pass)}
    analysis["gate_kw_pass"] = {"n": gate_kw_pass_n, "cers": cer_bin_stats(cers_gate_kw_pass)}

    # max_sim bin 分布 + 救回池 (gate+kw)
    bins = [(0, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.27)]
    analysis["sim_bin_realtarget_rate"] = []
    for lo, hi in bins:
        sub = [s for s in nspk2_rej if lo <= s["max_sim"] < hi]
        if not sub:
            continue
        cers = [cer_off(s["scene_route_text"], s["ref"]) for s in sub]
        analysis["sim_bin_realtarget_rate"].append({
            "bin": f"[{lo:.2f},{hi:.2f})",
            "n": len(sub),
            "cer_lt_0.5": sum(1 for c in cers if c < 0.5),
            "real_rate": round(sum(1 for c in cers if c < 0.5) / len(sub), 4),
            "mean_cer": round(statistics.mean(cers), 4),
        })

    # 个例: 8 个"被 sim 拒但分离 target 路是清晰指令 + CER 低"的样本
    examples_real = []
    for s in nspk2_rej:
        t = s.get("scene_route_text") or ""
        if rescue_gate_kw(t):
            c = cer_off(t, s["ref"])
            if c < 0.3:
                examples_real.append({
                    "uid": s["uid"], "max_sim": round(s["max_sim"], 4),
                    "cer": round(c, 4),
                    "ref": s["ref"][:80], "target_path_text": t[:80],
                })
                if len(examples_real) >= 10:
                    break
    analysis["rescue_real_examples"] = examples_real

    # 个例: 8 个"被 sim 拒且分离 target 路通过 gate+kw 但 CER 高"的样本 (误接受风险)
    examples_bad = []
    for s in nspk2_rej:
        t = s.get("scene_route_text") or ""
        if rescue_gate_kw(t):
            c = cer_off(t, s["ref"])
            if c >= 0.8:
                examples_bad.append({
                    "uid": s["uid"], "max_sim": round(s["max_sim"], 4),
                    "cer": round(c, 4),
                    "ref": s["ref"][:80], "target_path_text": t[:80],
                })
                if len(examples_bad) >= 8:
                    break
    analysis["rescue_misaccept_examples"] = examples_bad

    (out_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ============================================================
    # B + C. 多信号 vs 单 sim, pos CER + neg RR + Pareto
    # ============================================================
    # baseline (单 sim, scene_route_text) - 用作对照
    base_cer, base_cnt = evaluate_pos(pos_samples, lambda t: False)  # no rescue
    base_neg_rr_vanilla = sum(1 for x in neg_samples if x["max_sim"] < 0.27) / len(neg_samples)

    result = {
        "baseline_sim_only": {
            "pos_cer_with_reject": round(base_cer, 4),
            "pos_breakdown": base_cnt,
            "neg_RR_vanilla": round(base_neg_rr_vanilla, 4),
            "neg_n": len(neg_samples),
            "transcribe_field": "scene_route_text",
        },
        "multi_signal_workpoints": [],
        "pareto": [],
        "risk_bounds": {},
    }

    # neg risk upper bound (per rescue_pred): pos 另一路通过率
    for name, pred in RESCUE_PREDS.items():
        rate, other_pass, total = neg_risk_upper_bound(pos_samples, pred)
        result["risk_bounds"][name] = {
            "pos_other_path_pass_rate": round(rate, 4),
            "pos_other_pass_n": other_pass,
            "pos_nspk2_total": total,
            "neg_RR_loss_upper_bound": round(rate * base_neg_rr_vanilla, 4),  # 假设 neg 同分布
            "neg_RR_lower_bound_with_risk": round(base_neg_rr_vanilla * (1 - rate), 4),
            "interpretation": "若 neg 分离 target 路通过率与 pos 另一路相同, 则 RR 损失 ≈ rate*0.9051",
        }

    # 工作点扫
    for name, pred in RESCUE_PREDS.items():
        cer, cnt = evaluate_pos(pos_samples, pred)
        # neg RR (vanilla 代理下界)
        rr_lb, rescued_proxy = evaluate_neg(neg_samples, pred, sep_text_proxy="vanilla")
        # neg RR 上界代理 (pos 另一路 rate)
        rate = result["risk_bounds"][name]["pos_other_path_pass_rate"]
        rr_ub = base_neg_rr_vanilla * (1 - rate)
        wp = {
            "rescue_pred": name,
            "pos_cer_with_reject": round(cer, 4),
            "delta_cer_vs_baseline": round(cer - base_cer, 4),
            "delta_cer_pct": round((cer - base_cer) / base_cer * 100, 2),
            "pos_breakdown": cnt,
            "neg_RR_lower_bound_vanilla": round(rr_lb, 4),
            "neg_RR_upper_bound_posproxy": round(rr_ub, 4),
            "neg_RR_delta_lower": round(rr_lb - base_neg_rr_vanilla, 4),
            "neg_RR_delta_upper": round(rr_ub - base_neg_rr_vanilla, 4),
            "neg_rescued_proxy_vanilla": rescued_proxy,
            "rescue_precision": round(cnt["rescue_real"] / cnt["rescued"], 4) if cnt["rescued"] else None,
        }
        result["multi_signal_workpoints"].append(wp)
        result["pareto"].append({
            "rescue_pred": name,
            "pos_cer": round(cer, 4),
            "neg_RR_lower": round(rr_lb, 4),
            "neg_RR_upper": round(rr_ub, 4),
            "rescued_n": cnt["rescued"],
            "rescue_real_n": cnt["rescue_real"],
            "rescue_bad_n": cnt["rescue_bad"],
        })

    # 额外: 主线 sim_thr 扫 (固定 rescue = gate+kw) — sim_thr 影响基线 + 多信号接受边界
    result["sim_thr_scan_gate_kw"] = []
    for thr in [0.20, 0.23, 0.25, 0.27, 0.30, 0.35]:
        # 重新生成 rejected (per_sample 用 thr0.27 的 flag, 这里基于 max_sim 重算)
        cer, cnt = evaluate_pos(pos_samples, rescue_gate_kw, sim_thr=thr)
        neg_rej = sum(1 for x in neg_samples if x["max_sim"] < thr) / len(neg_samples)
        rate = result["risk_bounds"]["gate+kw"]["pos_other_path_pass_rate"]
        result["sim_thr_scan_gate_kw"].append({
            "sim_thr": thr,
            "pos_cer_with_reject": round(cer, 4),
            "pos_breakdown": cnt,
            "neg_RR_baseline_at_thr": round(neg_rej, 4),
            "neg_RR_upper_bound_posproxy": round(neg_rej * (1 - rate), 4),
        })

    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ============================================================
    # Console summary
    # ============================================================
    print("=" * 72)
    print("多信号拒识探索 — 结果汇总")
    print("=" * 72)
    print(f"\n[A] 445 被拒 n_spk=2 样本的分离 target 路分析:")
    print(f"  gate 通过: {gate_pass_n}/445 ({gate_pass_n/445*100:.1f}%)")
    print(f"    真救回 CER<0.5: {analysis['gate_pass']['cers']['cer_lt_0.5']}")
    print(f"    CER==0: {analysis['gate_pass']['cers']['cer_eq_0']}")
    print(f"    mean CER: {analysis['gate_pass']['cers']['mean']}")
    print(f"  gate+kw 通过: {gate_kw_pass_n}/445 ({gate_kw_pass_n/445*100:.1f}%)")
    print(f"    真救回 CER<0.5: {analysis['gate_kw_pass']['cers']['cer_lt_0.5']}")
    print(f"    CER==0: {analysis['gate_kw_pass']['cers']['cer_eq_0']}")
    print(f"    mean CER: {analysis['gate_kw_pass']['cers']['mean']}")

    print(f"\n[B] pos 含拒 CER + neg RR (baseline vs 多信号):")
    print(f"  baseline 单 sim thr0.27 (scene_route_text):")
    print(f"    pos CER = {base_cer:.4f}, neg RR = {base_neg_rr_vanilla:.4f}")
    for wp in result["multi_signal_workpoints"]:
        print(f"  multi-signal [{wp['rescue_pred']}]:")
        print(f"    pos CER = {wp['pos_cer_with_reject']:.4f} (Δ {wp['delta_cer_vs_baseline']:+.4f}, {wp['delta_cer_pct']:+.1f}%)")
        print(f"    neg RR 下界(vanilla代理) = {wp['neg_RR_lower_bound_vanilla']:.4f} (Δ {wp['neg_RR_delta_lower']:+.4f})")
        print(f"    neg RR 上界(pos代理) = {wp['neg_RR_upper_bound_posproxy']:.4f} (Δ {wp['neg_RR_delta_upper']:+.4f})")
        print(f"    rescued={wp['pos_breakdown']['rescued']}, 真救回={wp['pos_breakdown']['rescue_real']}, 误接受={wp['pos_breakdown']['rescue_bad']}")
        if wp['rescue_precision'] is not None:
            print(f"    救回 precision(CER<0.5): {wp['rescue_precision']*100:.1f}%")

    print(f"\n[C] sim_thr 扫 (固定 gate+kw):")
    for r in result["sim_thr_scan_gate_kw"]:
        print(f"  thr={r['sim_thr']}: pos CER={r['pos_cer_with_reject']:.4f}, neg RR_base={r['neg_RR_baseline_at_thr']:.4f}, neg RR_UB={r['neg_RR_upper_bound_posproxy']:.4f}")

    print(f"\n产物:")
    print(f"  {out_dir / 'analysis.json'}")
    print(f"  {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
