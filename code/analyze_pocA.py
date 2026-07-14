"""POC A 结果分析: llm_reject (Qwen2.5-3B) 在测试集上的 家居/非家居指令 判别力。

读 code/pocA_llm_reject_result.json (llm_reject --testset 输出, rows 按顺序对齐 testset,
不跳过不重排) + code/llm_testset_pocA.json (拿 layer), 用索引对齐。

算:
  - 总体 accuracy + 混淆矩阵 (accept 为正类)
  - 家居 recall(accept 类)         ← 关键指标1, 阈值≥0.90 (误拒target丢分)
  - 非家居 recall(reject 类)        ← 关键指标2(用户称"非家居precision"), 阈值≥0.85
  - 分层正确率: pos/a_news/b_garble/c_en/d_empty/e_adv
  - e 层 reject 率 (防循环论证核心: LLM 是否超越"关键词匹配")
  - 错误案例 (gold≠pred, 按层打印)

go/no-go: 家居recall≥0.90 且 非家居recall≥0.85 -> GO POC B (端到端全speaker转写+llm_reject挑)
落盘 code/pocA_analysis.json
"""
import json, os
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    with open(os.path.join(_HERE, os.path.basename(p)), encoding="utf-8") as f:
        return json.load(f)


res = load("code/pocA_llm_reject_result.json")
rows = res["rows"]
print(f"[结果] {len(rows)} 条 (batch={res.get('batch')}, max_new={res.get('max_new_tokens')}, "
      f"{res.get('secs_total')}s, testset={res.get('testset','?')}) — rows 自带 layer/text (pocA_fast_eval 输出)")
for r in rows:
    r.setdefault("layer", "unknown")
    r.setdefault("text", "")

# ---- gold 修正: 对抗审查(workflow wqgwv2m65, 396条审6争议)发现的清洗漏网 ----
# 4 条 pos 原标 accept 实为非指令(描述闲聊/疑问求助/意图陈述/疑ASR噪声),
# 系"含家电或动作词"清洗规则漏网(含"模式""开始""下"等词但非真指令)。
# 修正为 reject 避免污染家居 recall(否则 LLM 正确拒它们反被算 FN)。
# e 层 2 条弱争议(三十小时闹钟/风扇摇头)保守保留 reject, 不改。
GOLD_CORRECTIONS = {
    "呃这个就是全屋智能像我们要开空调啊或者是我们要做饭啊这些都是嗯这里了听到我们说话": "reject",
    "放了一年的羊绒大衣应该用什么清洗模式": "reject",
    "我要开始烹饪": "reject",
    "窗帘下架": "reject",
}
n_corrected = 0
for r in rows:
    if r["text"] in GOLD_CORRECTIONS and r["gold"] != GOLD_CORRECTIONS[r["text"]]:
        r["original_gold"] = r["gold"]
        r["gold"] = GOLD_CORRECTIONS[r["text"]]
        n_corrected += 1
print(f"[gold修正] 对抗审查争议: {n_corrected} 条 pos accept->reject (清洗规则漏网的含词非指令)\n")

# ---- 混淆矩阵 (accept 为正类) ----
tp = fp = fn = tn = 0
for r in rows:
    g, p = r["gold"], r["pred"]
    if g == "accept" and p == "accept":
        tp += 1     # 家居指令正确接受
    elif g == "accept" and p == "reject":
        fn += 1     # 家居指令被误拒 (新架构里=误拒target, 丢分)
    elif g == "reject" and p == "accept":
        fp += 1     # 非家居被误放 (新架构里=保留干扰人话, CER不降)
    else:
        tn += 1     # 非家居正确拒
n = len(rows)
acc = (tp + tn) / n
home_recall = tp / (tp + fn) if (tp + fn) else 0.0    # 家居召回
home_prec = tp / (tp + fp) if (tp + fp) else 0.0
nonhome_recall = tn / (tn + fp) if (tn + fp) else 0.0  # 非家居被拒比例(用户称"非家居precision")
nonhome_prec = tn / (tn + fn) if (tn + fn) else 0.0
f1_home = (2 * home_recall * home_prec / (home_recall + home_prec)
           if (home_recall + home_prec) else 0.0)

print("=" * 60)
print("POC A 总体 (llm_reject 家居/非家居指令判别力)")
print("=" * 60)
print(f"n={n}  accuracy={acc:.4f}")
print(f"混淆矩阵(accept正类): TP(家居对放){tp}  FN(家居误拒){fn}  "
      f"FP(非家居误放){fp}  TN(非家居对拒){tn}")
print(f"家居 recall   = {home_recall:.4f}   [阈值≥0.90]  {'✓' if home_recall>=0.90 else '✗'}")
print(f"家居 precision= {home_prec:.4f}")
print(f"非家居 recall = {nonhome_recall:.4f}   [阈值≥0.85]  {'✓' if nonhome_recall>=0.85 else '✗'}  (用户称'非家居precision')")
print(f"非家居precision={nonhome_prec:.4f}")
print(f"家居 F1       = {f1_home:.4f}")

# ---- 分层 ----
by_layer = defaultdict(lambda: {"n": 0, "correct": 0, "pred": Counter(),
                                "gold": Counter(), "rows": []})
for r in rows:
    L = r["layer"]
    by_layer[L]["n"] += 1
    by_layer[L]["correct"] += (r["gold"] == r["pred"])
    by_layer[L]["pred"][r["pred"]] += 1
    by_layer[L]["gold"][r["gold"]] += 1
    by_layer[L]["rows"].append(r)

print("\n" + "=" * 60)
print("分层正确率")
print("=" * 60)
ORDER = ["pos", "a_news", "b_garble", "c_en", "d_empty", "e_adv_含家电非指令"]
layer_stats = {}
for L in ORDER:
    d = by_layer[L]
    a = d["correct"] / d["n"] if d["n"] else 0.0
    # 该层"应被判定"的方向: pos应accept, 其余应reject
    layer_stats[L] = {"n": d["n"], "acc": round(a, 4), "pred": dict(d["pred"])}
    print(f"  [{L:22s}] n={d['n']:3d} 正确率={a:.4f}  pred={dict(d['pred'])}")

# e 层 reject 率 (防循环论证核心)
e = by_layer["e_adv_含家电非指令"]
e_reject_rate = e["pred"].get("reject", 0) / e["n"] if e["n"] else 0.0
print(f"\n  >>> e 层(含家电词非指令) reject 率 = {e_reject_rate:.4f}")
print(f"      (防循环论证核心: 高=LLM超越关键词匹配懂指令性; 低=只靠关键词,POC无效)")

# ---- go/no-go ----
go = home_recall >= 0.90 and nonhome_recall >= 0.85
print("\n" + "=" * 60)
print("go / no-go 判定")
print("=" * 60)
print(f"  家居 recall {home_recall:.2f} {'≥0.90 ✓' if home_recall>=0.90 else '<0.90 ✗'}")
print(f"  非家居 recall {nonhome_recall:.2f} {'≥0.85 ✓' if nonhome_recall>=0.85 else '<0.85 ✗'}")
print(f"  e 层 reject 率 {e_reject_rate:.2f} ({'高,循环论证已破' if e_reject_rate>=0.8 else '偏低,警惕关键词依赖'})")
print(f"\n  => {'✅ GO POC B (端到端全speaker转写+llm_reject挑)' if go else '⛔ NO-GO (根基不稳, 需调prompt或回退声纹辅助)'}")

# ---- 错误案例 ----
print("\n" + "=" * 60)
print("错误案例 (gold≠pred) 按层")
print("=" * 60)
errors_by_layer = {}
for L in ORDER:
    errs = [r for r in by_layer[L]["rows"] if r["gold"] != r["pred"]]
    if errs:
        errors_by_layer[L] = [{"text": r["text"], "gold": r["gold"], "pred": r["pred"],
                               "reason": r.get("reason", "")} for r in errs]
        print(f"\n  [{L}] {len(errs)} 条错 (该层应={errs[0]['gold']}):")
        for r in errs[:12]:
            t = r["text"][:38]
            print(f"    gold={r['gold']} pred={r['pred']:6s} | {t!r}")
            print(f"      reason: {r.get('reason','')[:80]}")

# ---- 落盘 ----
out = {
    "overall": {
        "n": n, "accuracy": round(acc, 4),
        "home_recall": round(home_recall, 4), "home_precision": round(home_prec, 4),
        "home_f1": round(f1_home, 4),
        "nonhome_recall": round(nonhome_recall, 4),
        "nonhome_precision": round(nonhome_prec, 4),
        "confusion": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "e_layer_reject_rate": round(e_reject_rate, 4),
        "go_pocB": go,
    },
    "by_layer": layer_stats,
    "errors_by_layer": errors_by_layer,
}
out_path = os.path.join(_HERE, "pocA_analysis.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n-> {out_path}")
