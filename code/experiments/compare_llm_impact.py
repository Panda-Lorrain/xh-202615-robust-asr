"""对比开关 LLM 的影响：列出 LLM 救回和误拒的具体样本"""
import json

# 开 LLM（llm_or_sim 策略）
with open("out_pos_final/result.json", encoding="utf-8") as f:
    pos_with_llm = json.load(f)
with open("out_neg_final/result.json", encoding="utf-8") as f:
    neg_with_llm = json.load(f)

# 关 LLM（sim_only 策略，thr=0.4）
with open("out_pos_noLLM/result.json", encoding="utf-8") as f:
    pos_no_llm = json.load(f)
with open("out_neg_noLLM/result.json", encoding="utf-8") as f:
    neg_no_llm = json.load(f)

print("=" * 80)
print("开关 LLM 对比分析")
print("=" * 80)
print(f"\n开 LLM 配置: {pos_with_llm['config']}")
print(f"关 LLM 配置: {pos_no_llm['config']}")

# 建立 utt_id -> result 的映射
pos_with_map = {r["utt_id"]: r for r in pos_with_llm["results"]}
pos_no_map = {r["utt_id"]: r for r in pos_no_llm["results"]}
neg_with_map = {r["utt_id"]: r for r in neg_with_llm["results"]}
neg_no_map = {r["utt_id"]: r for r in neg_no_llm["results"]}

# === POS 集分析 ===
print("\n" + "=" * 80)
print("POS 集（正样本，应该被接受）")
print("=" * 80)

pos_saved_by_llm = []  # 关 LLM 拒，开 LLM 不拒（LLM 救回）
pos_killed_by_llm = []  # 关 LLM 不拒，开 LLM 拒（LLM 误拒）

for utt_id in pos_with_map:
    with_r = pos_with_map[utt_id]
    no_r = pos_no_map.get(utt_id)
    if no_r is None:
        continue

    with_rejected = with_r["rejected"]
    no_rejected = no_r["rejected"]

    if no_rejected and not with_rejected:
        # LLM 救回来了
        pos_saved_by_llm.append({
            "utt_id": utt_id,
            "text": with_r["text"],
            "max_sim": with_r["max_sim"],
            "llm_verdict": with_r.get("llm_verdict"),
        })
    elif not no_rejected and with_rejected:
        # LLM 误拒了
        pos_killed_by_llm.append({
            "utt_id": utt_id,
            "text": with_r["text"],
            "max_sim": with_r["max_sim"],
            "llm_verdict": with_r.get("llm_verdict"),
        })

print(f"\n【LLM 救回】关 LLM 时被拒，开 LLM 后救回: {len(pos_saved_by_llm)} 条")
print("-" * 60)
for item in pos_saved_by_llm[:50]:
    print(f"  {item['utt_id']}: sim={item['max_sim']:.4f} | text='{item['text']}' | llm={item['llm_verdict']}")
if len(pos_saved_by_llm) > 50:
    print(f"  ... 还有 {len(pos_saved_by_llm) - 50} 条")

print(f"\n【LLM 误拒】关 LLM 时通过，开 LLM 后被拒: {len(pos_killed_by_llm)} 条")
print("-" * 60)
for item in pos_killed_by_llm[:50]:
    print(f"  {item['utt_id']}: sim={item['max_sim']:.4f} | text='{item['text']}' | llm={item['llm_verdict']}")
if len(pos_killed_by_llm) > 50:
    print(f"  ... 还有 {len(pos_killed_by_llm) - 50} 条")

# === NEG 集分析 ===
print("\n" + "=" * 80)
print("NEG 集（负样本，应该被拒识）")
print("=" * 80)

neg_escaped_by_llm = []  # 关 LLM 拒，开 LLM 不拒（LLM 放过了非目标）
neg_caught_by_llm = []  # 关 LLM 不拒，开 LLM 拒（LLM 抓住了漏网的）

for utt_id in neg_with_map:
    with_r = neg_with_map[utt_id]
    no_r = neg_no_map.get(utt_id)
    if no_r is None:
        continue

    with_rejected = with_r["rejected"]
    no_rejected = no_r["rejected"]

    if no_rejected and not with_rejected:
        # LLM 放过了非目标（错误）
        neg_escaped_by_llm.append({
            "utt_id": utt_id,
            "text": with_r["text"],
            "max_sim": with_r["max_sim"],
            "llm_verdict": with_r.get("llm_verdict"),
        })
    elif not no_rejected and with_rejected:
        # LLM 抓住了漏网的（正确）
        neg_caught_by_llm.append({
            "utt_id": utt_id,
            "text": with_r["text"],
            "max_sim": with_r["max_sim"],
            "llm_verdict": with_r.get("llm_verdict"),
        })

print(f"\n【LLM 放过非目标】关 LLM 时拒，开 LLM 后放行（RR 受损）: {len(neg_escaped_by_llm)} 条")
print("-" * 60)
for item in neg_escaped_by_llm[:50]:
    print(f"  {item['utt_id']}: sim={item['max_sim']:.4f} | text='{item['text']}' | llm={item['llm_verdict']}")
if len(neg_escaped_by_llm) > 50:
    print(f"  ... 还有 {len(neg_escaped_by_llm) - 50} 条")

print(f"\n【LLM 抓住漏网】关 LLM 时放行，开 LLM 后拒（RR 改善）: {len(neg_caught_by_llm)} 条")
print("-" * 60)
for item in neg_caught_by_llm[:50]:
    print(f"  {item['utt_id']}: sim={item['max_sim']:.4f} | text='{item['text']}' | llm={item['llm_verdict']}")
if len(neg_caught_by_llm) > 50:
    print(f"  ... 还有 {len(neg_caught_by_llm) - 50} 条")

# === 总结 ===
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"POS 集: LLM 救回 {len(pos_saved_by_llm)} 条 vs 误拒 {len(pos_killed_by_llm)} 条")
print(f"NEG 集: LLM 放过非目标 {len(neg_escaped_by_llm)} 条 vs 抓住漏网 {len(neg_caught_by_llm)} 条")
print(f"\n净效果:")
print(f"  POS 侧: {'利好' if len(pos_saved_by_llm) > len(pos_killed_by_llm) else '利空'} (救回{len(pos_saved_by_llm)} - 误拒{len(pos_killed_by_llm)} = {len(pos_saved_by_llm)-len(pos_killed_by_llm)})")
print(f"  NEG 侧: {'利好' if len(neg_caught_by_llm) > len(neg_escaped_by_llm) else '利空'} (抓住{len(neg_caught_by_llm)} - 放过{len(neg_escaped_by_llm)} = {len(neg_caught_by_llm)-len(neg_escaped_by_llm)})")
