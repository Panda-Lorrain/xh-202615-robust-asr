#!/usr/bin/env python
"""2026-07-18 SE bugfix A/B 权威复核 (对抗审查 4-agent 印证后的一手重算)。

背景: docs/SE_bugfix_AB结果_2026-07-18.md + commit c8c739d 有 3 处归因/事实错:
  1. compare_se_bugfix.py:41 字段名 bug (读 'transcript' 实为 'text') → "0 文本不一致" 是假象
  2. "翻转386条" 数字错
  3. "accepted-only SE 略好(0.4606<0.4777)→SE 对转写无害" 是选择性偏差(两臂 accepted 集合不同)
本脚本一次性重算所有修正数字, 作为 AB 文档重写 + memory 的权威依据。复用 eval_metrics.cer
+ eval_datasetA._norm_zh (繁简归一) 与官方评测同口径。

用法: code/.venv/Scripts/python.exe code/audit_se_bugfix.py
"""
import json, os, statistics
from eval_metrics import cer
from eval_datasetA import _norm_zh, key_from_rec

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_pairs(path):
    d = json.load(open(path, encoding="utf-8"))
    return {key_from_rec(it["recognition"]): it for it in d["results"]}


se = load_pairs(os.path.join(ROOT, "out_pos_SE_bugfixed_1364/result.json"))
no = load_pairs(os.path.join(ROOT, "out_pos_noSE_1364/result.json"))
man = json.load(open(os.path.join(ROOT, "pos_pairs_datasetA.json"), encoding="utf-8"))
ref = {key_from_rec(r["recognition"]): (r.get("ref") or "") for r in man}

common = sorted(set(se) & set(no) & set(ref))
n = len(common)


def utt_cer(it, k):
    """复用 eval_datasetA:66 逻辑 — rejected/空 → 1.0, 否则 cer(繁简归一)。"""
    if it.get("rejected") or not (it.get("text") or "").strip():
        return 1.0
    return cer(_norm_zh(it["text"]), _norm_zh(ref[k]))


# ---------- 1. overall 对账 ----------
se_overall = sum(utt_cer(se[k], k) for k in common) / n
no_overall = sum(utt_cer(no[k], k) for k in common) / n
delta = se_overall - no_overall
print("=" * 72)
print(f"[1. overall 对账] n={n}")
print(f"  SE overall   = {se_overall:.4f}  (doc 0.8292)")
print(f"  noSE overall = {no_overall:.4f}  (doc 0.7243)")
print(f"  delta        = {delta:+.4f}  (doc +0.1049)")

# ---------- 2. 文本不一致(正确字段 text) ----------
both_acc = [k for k in common if not se[k].get("rejected") and not no[k].get("rejected")]
text_diff = [k for k in both_acc
             if (se[k].get("text") or "") != (no[k].get("text") or "")]
improve, worsen, neutral = [], [], []
for k in text_diff:
    d = utt_cer(se[k], k) - utt_cer(no[k], k)
    if d < -0.01:
        improve.append((k, no[k].get("text"), se[k].get("text"), d))
    elif d > 0.01:
        worsen.append((k, no[k].get("text"), se[k].get("text"), d))
    else:
        neutral.append(k)
print("\n" + "=" * 72)
print(f"[2. 文本不一致] (修字段名 bug 后: text 非 transcript)")
print(f"  both-accepted = {len(both_acc)}")
print(f"  文本不同      = {len(text_diff)} ({len(text_diff)/max(len(both_acc),1)*100:.1f}%)  ← doc 错写 '0 条'")
print(f"  其中 SE 改善(cer降>0.01) = {len(improve)}  SE 恶化(cer升>0.01) = {len(worsen)}  中性 = {len(neutral)}")
print("  改善样例(SE 比 noSE 更准):")
for k, nt, st, d in sorted(improve, key=lambda x: x[3])[:5]:
    print(f"    {k} Δ{d:+.3f}  noSE=[{nt}] → SE=[{st}]")
print("  恶化样例(SE 比 noSE 更差):")
for k, nt, st, d in sorted(worsen, key=lambda x: -x[3])[:5]:
    print(f"    {k} Δ{d:+.3f}  noSE=[{nt}] → SE=[{st}]")

# ---------- 3. 拒识交叉表 ----------
b = {"rr": 0, "ra": 0, "ar": 0, "aa": 0}
for k in common:
    sr = "r" if se[k].get("rejected") else "a"
    nr = "r" if no[k].get("rejected") else "a"
    b[sr + nr] += 1
print("\n" + "=" * 72)
print("[3. 拒识交叉表] (se/no)")
print(f"  both_rej       = {b['rr']}")
print(f"  se_rej & no_acc= {b['ra']}   (SE 多拒的)")
print(f"  se_acc & no_rej= {b['ar']}   (SE 少拒的 lucky-accept)")
print(f"  both_acc       = {b['aa']}")
print(f"  => 翻转 = {b['ra']+b['ar']}  (doc 错写 '386')")
print(f"  => 净增拒识 = {b['ra']-b['ar']}  (= SE拒{b['rr']+b['ra']} - noSE拒{b['rr']+b['ar']})")

# ---------- 4. diar_fail ----------
se_df = [k for k in common if se[k].get("diar_fail")]
no_df = [k for k in common if no[k].get("diar_fail")]
se_df_noacc = [k for k in se_df if not no[k].get("rejected")]
print("\n" + "=" * 72)
print("[4. diar_fail] (DeepFilterNet3 过衰减→DiariZen ValueError)")
print(f"  SE   diar_fail = {len(se_df)}  (审查说 ~207)")
print(f"  noSE diar_fail = {len(no_df)}")
print(f"  SE diar_fail 且 noSE accepted = {len(se_df_noacc)}  (被 SE silence 搞崩的可转写条)")

# ---------- 5. 四桶 CER delta 分解 ----------
contrib = {"A_diar_crash": 0.0, "B_sim_drop": 0.0, "C_transcribe": 0.0,
           "D_lucky_accept": 0.0, "E_both_rej": 0.0}
cnt = dict.fromkeys(contrib, 0)
for k in common:
    d = utt_cer(se[k], k) - utt_cer(no[k], k)
    sr, nr = se[k].get("rejected"), no[k].get("rejected")
    if sr and nr:
        bucket = "E_both_rej"
    elif sr and not nr:
        bucket = "A_diar_crash" if se[k].get("diar_fail") else "B_sim_drop"
    elif not sr and nr:
        bucket = "D_lucky_accept"
    else:
        bucket = "C_transcribe"
    contrib[bucket] += d
    cnt[bucket] += 1
print("\n" + "=" * 72)
print(f"[5. 四桶 CER delta 分解] (总 delta = {delta:+.4f})")
tot = 0.0
for k in ["A_diar_crash", "B_sim_drop", "C_transcribe", "D_lucky_accept", "E_both_rej"]:
    print(f"  {k:16s} n={cnt[k]:5d}  contrib={contrib[k]:+.4f}")
    tot += contrib[k]
print(f"  {'合计':16s} n={n:5d}  contrib={tot:+.4f}  (应 = delta)")
pos_sum = abs(contrib["A_diar_crash"]) + abs(contrib["B_sim_drop"]) + abs(contrib["C_transcribe"])
print(f"  正贡献占比: diar_crash={abs(contrib['A_diar_crash'])/pos_sum*100:.0f}%  "
      f"sim_drop={abs(contrib['B_sim_drop'])/pos_sum*100:.0f}%  "
      f"transcribe={abs(contrib['C_transcribe'])/pos_sum*100:.0f}%")
print(f"  lucky_accept 抵消 = {contrib['D_lucky_accept']:+.4f}")

# ---------- 6. 交集 apples-to-apples ----------
se_int = [utt_cer(se[k], k) for k in both_acc]
no_int = [utt_cer(no[k], k) for k in both_acc]
print("\n" + "=" * 72)
print(f"[6. 交集 apples-to-apples] both-accepted {len(both_acc)} 条 (公平对比)")
print(f"  cer_SE   = {statistics.mean(se_int):.4f}")
print(f"  cer_noSE = {statistics.mean(no_int):.4f}")
print(f"  delta    = {statistics.mean(se_int)-statistics.mean(no_int):+.4f}  "
      f"(正=SE 在同集合更差)")
print(f"  ← doc 'accepted-only SE 0.4606<0SE 0.4777' 是两臂不同集合(SE 432 vs noSE 720),")
print(f"    选择性偏差不可比; 交集才是公平口径, SE 在交集反而恶化(+{statistics.mean(se_int)-statistics.mean(no_int):.4f})")

# ---------- 存 json ----------
out = {
    "n": n,
    "overall": {"SE": round(se_overall, 4), "noSE": round(no_overall, 4), "delta": round(delta, 4)},
    "text_diff": {"both_accepted": len(both_acc), "differ": len(text_diff),
                  "pct": round(len(text_diff)/max(len(both_acc), 1)*100, 1),
                  "SE_improve": len(improve), "SE_worsen": len(worsen), "neutral": len(neutral)},
    "reject_crosstab": {"both_rej": b["rr"], "se_rej_no_acc": b["ra"],
                        "se_acc_no_rej": b["ar"], "both_acc": b["aa"],
                        "flipped": b["ra"]+b["ar"], "net_increase": b["ra"]-b["ar"]},
    "diar_fail": {"SE": len(se_df), "noSE": len(no_df),
                  "SE_fail_noSE_acc": len(se_df_noacc)},
    "delta_decomp": {k: {"n": cnt[k], "contrib": round(contrib[k], 4)} for k in contrib},
    "intersection_cer": {"n": len(both_acc), "SE": round(statistics.mean(se_int), 4),
                         "noSE": round(statistics.mean(no_int), 4),
                         "delta": round(statistics.mean(se_int)-statistics.mean(no_int), 4)},
}
with open(os.path.join(ROOT, "audit_se_bugfix.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n[done] 写出 code/audit_se_bugfix.json")
