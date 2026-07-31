#!/usr/bin/env python
"""对比 SE bugfix 前后: qwen+SE(真生效) vs qwen+noSE 的 CER/sim/拒识。
2026-07-18: SE 输出 se_out 原是孤儿目录(从未喂 enroll_infer), 修复后 A/B 看 SE 真生效是否降 CER。
"""
import json, os, statistics

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return d.get("results", d) if isinstance(d, dict) else d

def uid(r):
    return os.path.splitext(os.path.basename(r.get("recognition", "")))[0]

se = {uid(r): r for r in load("code/out_pos_SE_bugfixed_1364/result.json")}
no = {uid(r): r for r in load("code/out_pos_noSE_1364/result.json")}
common = set(se) & set(no)
print(f"SE(bugfixed): {len(se)}条, noSE: {len(no)}条, 共同: {len(common)}")

# 1. sim score 对比(SE 改了音频, 可能改 diar/embedding → 改 sim)
sim_diff = [se[u]["max_sim"] - no[u]["max_sim"] for u in common
            if "max_sim" in se[u] and "max_sim" in no[u]]
print(f"\n=== max_sim 对比(SE真生效 - noSE) ===")
print(f"  mean={statistics.mean(sim_diff):+.4f} median={statistics.median(sim_diff):+.4f} std={statistics.stdev(sim_diff):.4f}")
print(f"  min={min(sim_diff):+.4f} max={max(sim_diff):+.4f}")
sim_changed = sum(1 for d in sim_diff if abs(d) > 1e-6)
print(f"  sim 有变化: {sim_changed}/{len(sim_diff)} ({sim_changed/len(sim_diff)*100:.1f}%)")

# 2. 拒识对比
se_rej = sum(1 for u in common if se[u].get("rejected"))
no_rej = sum(1 for u in common if no[u].get("rejected"))
print(f"\n=== 拒识对比 ===")
print(f"  SE: {se_rej} ({se_rej/len(common)*100:.1f}%) vs noSE: {no_rej} ({no_rej/len(common)*100:.1f}%)")
cross = sum(1 for u in common if se[u].get("rejected") != no[u].get("rejected"))
print(f"  拒识决策翻转: {cross}/{len(common)}")

# 3. 文本对比(仅看 accepted 条)
text_diff = []
for u in common:
    if se[u].get("rejected") or no[u].get("rejected"):
        continue
    # 2026-07-18 修字段名 bug: result.json 实际字段是 'text'(submit_infer.py:365),
    # 非 'transcript' → 原 get('transcript') 永远返回 '' → "0 条不一致" 假象。实测 109/383。
    if se[u].get("text", "") != no[u].get("text", ""):
        text_diff.append((u, no[u].get("text", ""), se[u].get("text", "")))
print(f"\n=== 文本对比(两边都 accepted 的 {sum(1 for u in common if not se[u].get('rejected') and not no[u].get('rejected'))} 条) ===")
print(f"  文本不一致: {len(text_diff)}")
for u, no_t, se_t in text_diff[:10]:
    print(f"    {u}: noSE=[{no_t}] SE=[{se_t}]")
