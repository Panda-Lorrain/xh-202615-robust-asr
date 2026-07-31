#!/usr/bin/env python3
"""POC: CER 腿多后端融合 oracle 上限测量（gating 决策，纯计算无 GPU）。

目的: 测 qwen+firered+vanilla 三后端逐条 min 的 oracle CER 上限，对比 qwen 单独 baseline，
      判断多后端融合有没有空间（vs 效率腿代价）。

数据源:
  - qwen + vanilla 逐条: code/poc_qwen_asr_full_result.json 的 rows[]
  - firered 逐条文本:   code/_firered_uid2text_full.json (uid -> text, n=1350 全匹配)
  - sanity: qwen transcribe CER 应=0.3436, thr0.27 含拒应=0.5934（与 recompute_qwen_official 对齐）

口径（严格复用 recompute_qwen_official.py）:
  - submit_norm(text) = digit_postproc(to_simplified(text))
  - CERMetric（eval_metrics.py 累计池 total_errors/total_chars）
  - 含拒 thr0.27: sim<thr → pred置空 → CER=1.0

oracle 计算（累计池正确算法）:
  对每条样本，CERMetric 给出 (errors, target_chars)。oracle 在该条选 min(err) 的后端；
  cumulative_oracle_CER = sum_i(min(err_q_i, err_f_i [, err_v_i])) / sum_i(target_char_i)。
  含拒版：sim<thr 的条，三后端 pred 都置空，err=ref_char（无选择空间，天然 CER=1.0）。

判断（gating）:
  - transcribe oracle gap < 0.02  → 枪毙（融合无空间）
  - gap > 0.04                    → 有空间（但要权衡效率腿 RTF 翻倍代价）
  - 0.02-0.04                    → 边界，报数字决策
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc


def submit_norm(text):
    return digit_postproc(to_simplified(text or ""))


def per_sample(pred, ref):
    """单条官方累计池 per-sample (errors, target_chars)。"""
    m = CERMetric()
    m.update([pred], [ref])
    ps = m.per_sample_results[0]
    return ps["errors"], ps["target_chars"]


def cumulative(errs_chars):
    """list of (err, char) -> cumulative CER."""
    te = sum(e for e, _ in errs_chars)
    tc = sum(c for _, c in errs_chars)
    return te / tc if tc else 0.0


def pool_oracle_transcribe(per_backend):
    """转写口径 oracle（不拒）: per_backend = list[backend] of list[(err,char)] aligned by sample idx.
    返回 cumulative_oracle_CER + 每条选中的 backend 索引。"""
    n = len(per_backend[0])
    picks = []
    ec = []
    for i in range(n):
        # 选 err 最小的（target_chars 同 per-sample，等价于选 CER 最小）
        errs = [per_backend[b][i][0] for b in range(len(per_backend))]
        best = min(range(len(errs)), key=lambda b: errs[b])
        picks.append(best)
        ec.append(per_backend[best][i])
    return cumulative(ec), picks


def pool_oracle_with_reject(per_backend, sims, thr):
    """含拒 thr 口径 oracle: sim<thr 的条三后端 pred 都置空（err=ref_char 天然 CER=1.0），
    sim>=thr 的条按 oracle 选 min err。返回 cumulative_oracle_CER。"""
    n = len(per_backend[0])
    ec = []
    for i in range(n):
        if sims[i] < thr:
            # 拒识条：pred 置空，errors = target_chars（任一 backend 的 char 即 ref char）
            ec.append((per_backend[0][i][1], per_backend[0][i][1]))
        else:
            errs = [per_backend[b][i][0] for b in range(len(per_backend))]
            best = min(range(len(errs)), key=lambda b: errs[b])
            ec.append(per_backend[best][i])
    return cumulative(ec)


def main():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    qwen_path = os.path.join(code_dir, "poc_qwen_asr_full_result.json")
    fire_path = os.path.join(code_dir, "_firered_uid2text_full.json")

    qd = json.load(open(qwen_path, encoding="utf-8"))
    rows = qd["rows"]
    n = len(rows)
    fire = json.load(open(fire_path, encoding="utf-8"))

    # 缺失检查
    missing = [r["uid"] for r in rows if r["uid"] not in fire]
    if missing:
        print(f"[警告] {len(missing)} 条 uid 在 firered 缺失，将跳过（样本数减少）")
        rows = [r for r in rows if r["uid"] in fire]
        n = len(rows)

    refs = [submit_norm(r["ref"]) for r in rows]
    sims = [r["sim"] for r in rows]
    qw_sub = [submit_norm(r["qwen"]) for r in rows]
    va_sub = [submit_norm(r["vanilla"]) for r in rows]
    fr_sub = [submit_norm(fire[r["uid"]]) for r in rows]

    print(f"===== Oracle 融合上限测量（n={n}, 三后端）=====")

    # ---- 0. 三后端 per-sample (err, char) ----
    ps_q = [per_sample(p, r) for p, r in zip(qw_sub, refs)]
    ps_f = [per_sample(p, r) for p, r in zip(fr_sub, refs)]
    ps_v = [per_sample(p, r) for p, r in zip(va_sub, refs)]

    # ---- 1. baseline sanity（官方累计池）----
    print("\n-- baseline 转写 CER（官方累计池, sanity）--")
    bq = cumulative(ps_q); bf = cumulative(ps_f); bv = cumulative(ps_v)
    print(f"  qwen    : {bq:.4f}  (期望 0.3436)")
    print(f"  firered : {bf:.4f}  (期望 ~0.3501)")
    print(f"  vanilla : {bv:.4f}  (期望 ~0.5954)")
    ok_q = abs(bq - 0.3436) < 0.001
    print(f"  qwen sanity(0.3436): {'PASS' if ok_q else 'FAIL'}")

    THR = 0.27
    print(f"\n-- baseline 含拒 thr={THR}（官方累计池, sanity）--")
    # 含拒 baseline: sim<thr 的条 pred 置空 → 单后端 CER
    def baseline_with_reject(ps):
        ec = []
        for i, s in enumerate(sims):
            if s < THR:
                ec.append((ps[i][1], ps[i][1]))  # err=char=ref_char
            else:
                ec.append(ps[i])
        return cumulative(ec)
    bq27 = baseline_with_reject(ps_q); bf27 = baseline_with_reject(ps_f); bv27 = baseline_with_reject(ps_v)
    print(f"  qwen    : {bq27:.4f}  (期望 0.5934)")
    print(f"  firered : {bf27:.4f}")
    print(f"  vanilla : {bv27:.4f}")
    ok_q27 = abs(bq27 - 0.5934) < 0.001
    print(f"  qwen thr0.27 sanity(0.5934): {'PASS' if ok_q27 else 'FAIL'}")

    if not (ok_q and ok_q27):
        print("\n[警告] sanity 未通过，口径可能不对，后续数字仅供参考。")

    # ---- 2. oracle_qf (qwen+firered) ----
    print("\n-- oracle_qf（qwen+firered 逐条 min）--")
    oqf_t, picks_qf = pool_oracle_transcribe([ps_q, ps_f])
    oqf_r = pool_oracle_with_reject([ps_q, ps_f], sims, THR)
    print(f"  transcribe CER : {oqf_t:.4f}  (qwen baseline {bq:.4f}, gap = {bq - oqf_t:+.4f})")
    print(f"  含拒 thr={THR}: {oqf_r:.4f}  (qwen baseline {bq27:.4f}, gap = {bq27 - oqf_r:+.4f})")

    # ---- 3. oracle_qfv (qwen+firered+vanilla) ----
    print("\n-- oracle_qfv（qwen+firered+vanilla 逐条 min）--")
    oqfv_t, picks_qfv = pool_oracle_transcribe([ps_q, ps_f, ps_v])
    oqfv_r = pool_oracle_with_reject([ps_q, ps_f, ps_v], sims, THR)
    print(f"  transcribe CER : {oqfv_t:.4f}  (qwen baseline {bq:.4f}, gap = {bq - oqfv_t:+.4f})")
    print(f"  含拒 thr={THR}: {oqfv_r:.4f}  (qwen baseline {bq27:.4f}, gap = {bq27 - oqfv_r:+.4f})")

    # ---- 4. verdict ----
    print("\n===== gating verdict =====")
    def verdict(gap_t, gap_r):
        # 用 transcribe 口径作主判（gating 标准），含拒辅证
        if gap_t < 0.02:
            return f"枪毙（transcribe gap {gap_t:+.4f} < 0.02）"
        if gap_t > 0.04:
            return f"有空间（transcribe gap {gap_t:+.4f} > 0.04）"
        return f"边界（transcribe gap {gap_t:+.4f} ∈ [0.02,0.04]，让决策）"
    print(f"  qf  : {verdict(bq - oqf_t, bq27 - oqf_r)}")
    print(f"  qfv : {verdict(bq - oqfv_t, bq27 - oqfv_r)}")

    # ---- 5. pairwise 互补性（per-sample cer）----
    print("\n-- pairwise 互补性（per-sample CER，n=%d）--" % n)
    cer_q = [e / max(c, 1) for e, c in ps_q]
    cer_f = [e / max(c, 1) for e, c in ps_f]
    cer_v = [e / max(c, 1) for e, c in ps_v]

    # 三后端谁是 unique winner（min CER，允许 tie）
    def argmin_tie(vals, tol=1e-9):
        m = min(vals)
        return [i for i, v in enumerate(vals) if v <= m + tol]
    qwin = fwin = vwin = tie_multi = 0
    for i in range(n):
        winners = argmin_tie([cer_q[i], cer_f[i], cer_v[i]])
        if len(winners) > 1:
            tie_multi += 1
        elif winners == [0]:
            qwin += 1
        elif winners == [1]:
            fwin += 1
        elif winners == [2]:
            vwin += 1
    print(f"  三后端 min 独赢: qwen={qwin}({qwin/n:.1%})  firered={fwin}({fwin/n:.1%})  vanilla={vwin}({vwin/n:.1%})  多后端并列最好={tie_multi}({tie_multi/n:.1%})")

    # qf 二后端互补（对齐 exp_firered_eval.json 的 pairwise 字段，校验口径）
    qf_q = qf_f = qf_tie = 0
    for i in range(n):
        if cer_q[i] < cer_f[i] - 1e-9:
            qf_q += 1
        elif cer_f[i] < cer_q[i] - 1e-9:
            qf_f += 1
        else:
            qf_tie += 1
    print(f"  qf 二后端: qwen 更优 {qf_q}({qf_q/n:.1%}) / firered 更优 {qf_f}({qf_f/n:.1%}) / tie {qf_tie}({qf_tie/n:.1%})")
    print(f"    （对照 exp_firered_eval pairwise: firered_better=167, qwen_better=206, tie=977）")

    # ---- 6. net 提示（效率腿代价）----
    print("\n===== net 提示（效率腿 20 分）=====")
    print(f"  qf transcribe gap {bq - oqf_t:+.4f} → CER 腿潜在 +{(bq - oqf_t)*40:.2f} 分")
    print(f"  qfv transcribe gap {bq - oqfv_t:+.4f} → CER 腿潜在 +{(bq - oqfv_t)*40:.2f} 分")
    print(f"  代价: 跑 2 后端 RTF ≈ 翻倍（qwen RTF0.289@4060 → ~0.58），效率腿 20 分线性 →")
    print(f"        若 L20 RTF 仍 <1（不超时），效率腿 -0~-2 分；若 >1 触发超时惩罚，-5~全部")
    print(f"  注意: 上限是 oracle（作弊选 min），现实选择器（LLM/sim/长度规则）只能兑现一部分，")
    print(f"        兑现率经验 30-60%。oracle gap 乘 0.4 ≈ 现实预期 CER 腿增益。")

    out = {
        "n": n,
        "baseline": {
            "transcribe": {"qwen": round(bq, 4), "firered": round(bf, 4), "vanilla": round(bv, 4)},
            "containing_reject_thr027": {"qwen": round(bq27, 4), "firered": round(bf27, 4), "vanilla": round(bv27, 4)},
        },
        "oracle_qf": {
            "transcribe": round(oqf_t, 4), "gap_vs_qwen": round(bq - oqf_t, 4),
            "containing_reject_thr027": round(oqf_r, 4), "gap_vs_qwen_cr": round(bq27 - oqf_r, 4),
        },
        "oracle_qfv": {
            "transcribe": round(oqfv_t, 4), "gap_vs_qwen": round(bq - oqfv_t, 4),
            "containing_reject_thr027": round(oqfv_r, 4), "gap_vs_qwen_cr": round(bq27 - oqfv_r, 4),
        },
        "pairwise_complement": {
            "three_backend_unique_win": {"qwen": qwin, "firered": fwin, "vanilla": vwin, "tie_multi": tie_multi},
            "qf_pair": {"qwen_better": qf_q, "firered_better": qf_f, "tie": qf_tie},
        },
        "sanity": {"qwen_transcribe_expect_0p3436": round(bq, 4), "qwen_thr027_expect_0p5934": round(bq27, 4),
                   "qwen_transcribe_pass": ok_q, "qwen_thr027_pass": ok_q27},
    }
    out_path = os.path.join(code_dir, "poc_oracle_fusion.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")


if __name__ == "__main__":
    main()
