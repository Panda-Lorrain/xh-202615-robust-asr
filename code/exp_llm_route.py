"""task2: LLM 泛化二选一选路 + 都录入假设验证。

复用 task1 全量分场景路由产物(per_sample.json 已含 sep_info 两路 qwen 转写文本),
只跑 LLM 二选一/单路判别, 不重跑 SepFormer / qwen ASR。

【做什么】
A. 对 n_spk=2 全部 805 条: 用 Qwen2.5-3B-Instruct 二选一(哪路更像对助手说话)
   - 选路标准泛化: 不只判"家居指令", 判任何"对智能设备的发言"
     (控制指令 + 咨询提问 + 设置查询)
B. 算 LLM 选路 vs heuristic vs oracle 的全量含拒 CER thr0.27
   - 拒识信号沿用主线 max_sim(同 task1, 不改拒识)
C. 都录入假设验证: 对 srcA / srcB 各自单独判 yes/no (是否像对助手说话)
   - 两路都 yes = TRAP(真双指令) 比例

【产物】
  code/runs/_scene_route_full/llm_route.json         (per-sample)
  code/runs/_scene_route_full/llm_route_summary.json (汇总 + 三档对比)

环境: .venv_llm (transformers 4.46.3 + Qwen2.5-3B-Instruct + torch 2.5.1 + editdistance)
"""
import os, sys, json, time, re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# 复用 task1 产物
SCENE_DIR = _HERE / "runs" / "_scene_route_full"
PER_SAMPLE = SCENE_DIR / "per_sample.json"
OUT_PER = SCENE_DIR / "llm_route.json"
OUT_SUMMARY = SCENE_DIR / "llm_route_summary.json"

# 全局种子(可复现性, 同 llm_reject.py)
from repro import set_global_seed, resolve_model
set_global_seed(42)

from eval_metrics import normalize_text  # 官方 NFKC + lower + 去 P*/空白
import editdistance

# ---- LLM prompts (用户 2026-07-28 定: 泛化"对助手说话") ----
SYS_BINARY = (
    "你是一个语音助手交互判别器。下面会给你两段中文语音转写文本 A 和 B。"
    "请判断：哪一段更像是一个人正在对智能音箱/智能助手说话？"
    "判断标准——对助手说话包括：控制设备的指令（如\"打开空调\"\"把温度调高\"）、"
    "向助手咨询或提问（如\"哺乳期要少吃什么\"\"今天天气怎么样\"）、"
    "设置或查询请求等任何针对智能设备的发言。"
    "另一段可能是新闻播报、闲聊、陈述、或与设备无关的内容。"
    "只回答 A 或 B， 不要其他文字。"
)

SYS_SINGLE = (
    "你是一个语音助手交互判别器。下面会给你一段中文语音转写文本。"
    "请判断：它是否像是一个人正在对智能音箱/智能助手说话？"
    "判断标准——对助手说话包括：控制设备的指令（如\"打开空调\"）、"
    "向助手咨询或提问（如\"今天天气怎么样\"\"哺乳期要少吃什么\"）、"
    "设置或查询请求等任何针对智能设备的发言。"
    "新闻播报、闲聊、陈述、与设备无关的内容算\"否\"。"
    "只回答 是 或 否， 不要其他文字。"
)


# ---- 模型加载 + batch 生成 ----
def load_model(model_path, device="cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    print(f"[load] Qwen {model_path} on {device}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # batch 生成需要 pad
    # left padding for generation (decoder-only)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    print(f"[load] done {time.time()-t0:.1f}s")
    return model, tok, torch


def render_prompt(tok, msgs):
    """单条 -> token id list (with generation prompt)"""
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)


def batch_generate(model, tok, torch, list_of_msg_lists, max_new_tokens=4, batch_size=16):
    """批量生成, 返回 raw text list (左 pad)."""
    results = [None] * len(list_of_msg_lists)
    # 分批
    for start in range(0, len(list_of_msg_lists), batch_size):
        batch = list_of_msg_lists[start:start + batch_size]
        # 渲染
        ids_list = [render_prompt(tok, m) for m in batch]
        max_len = max(len(x) for x in ids_list)
        # 左 pad
        pad_id = tok.pad_token_id
        attn = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        input_ids = torch.full((len(ids_list), max_len), pad_id, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            L = len(ids)
            input_ids[i, max_len - L:] = torch.tensor(ids, dtype=torch.long)
            attn[i, max_len - L:] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=pad_id,
            )
        # 切掉 prompt (左 pad 时, generated 部分固定在 out[:, input_len:])
        gen = out[:, input_ids.shape[-1]:]
        for i in range(len(batch)):
            txt = tok.decode(gen[i], skip_special_tokens=True).strip()
            results[start + i] = txt
        if (start // batch_size) % 10 == 0:
            print(f"  [batch] {start+len(batch)}/{len(list_of_msg_lists)}")
    return results


# ---- 输出解析 ----
def parse_binary(raw):
    """抽 A 或 B。"""
    if not raw:
        return None
    r = raw.strip().upper()
    # 找首个 A 或 B (不是字母 a-z 内的)
    # 优先精确匹配首字符
    if r.startswith("A"):
        return 0
    if r.startswith("B"):
        return 1
    # 兜底: 搜索
    for ch in r:
        if ch == "A":
            return 0
        if ch == "B":
            return 1
    return None


def parse_single(raw):
    """抽 是/否 -> True/False/None. 也接受 yes/no 中英."""
    if not raw:
        return None
    r = raw.strip()
    # 是 / 否 优先
    if r.startswith("是") or r.startswith("Yes") or r.startswith("yes") or r.startswith("YES"):
        return True
    if r.startswith("否") or r.startswith("No") or r.startswith("no") or r.startswith("NO"):
        return False
    # 兜底搜
    if "是" in r and "否" not in r:
        return True
    if "否" in r and "是" not in r:
        return False
    if "yes" in r.lower():
        return True
    if "no" in r.lower():
        return False
    return None


# ---- CER (官方口径, 同 task1 / eval_metrics) ----
def cer_pair(pred_text, ref_text):
    """单条官方 CER, 空 ref 边界一致。"""
    p = normalize_text(pred_text)
    r = normalize_text(ref_text)
    if len(r) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return editdistance.eval(p, r) / len(r)


def pool_cer_with_thr(samples, route_text_fn, thr=0.27):
    """累计池含拒 CER (同 task1 mainline_thr0.27_cer_pool 口径)。

    route_text_fn(sample) -> 该样本用于转写的文本 (拒识样本返回 "" 即贡献 CER=1.0)。
    拒识判定: max_sim < thr (主线 wespeaker max_sim, 同 task1)。
    """
    total_err = 0
    total_char = 0
    for s in samples:
        ref_n = normalize_text(s["ref"])
        if s["max_sim"] < thr:
            pred_n = ""  # 拒识 -> 全删 -> errors = len(ref)
        else:
            pred_n = normalize_text(route_text_fn(s))
        total_err += editdistance.eval(pred_n, ref_n)
        total_char += len(ref_n)
    return (total_err / total_char) if total_char else 0.0


def main():
    # ---- 1. 加载 task1 数据 ----
    with open(PER_SAMPLE, encoding="utf-8") as f:
        samples = json.load(f)
    print(f"[load] {len(samples)} samples from {PER_SAMPLE}")
    nspk2 = [s for s in samples if s.get("n_spk") == 2]
    print(f"[load] n_spk=2: {len(nspk2)} (LLM 路由候选)")

    # ---- 2. 加载 LLM ----
    model_path = resolve_model("QWEN")
    model, tok, torch = load_model(model_path)

    # ---- 3. 构造 LLM 任务列表 ----
    # A. 二选一
    binary_msgs_list = []
    for s in nspk2:
        tA = (s["sep_info"]["per_src_texts"][0] or "").strip()
        tB = (s["sep_info"]["per_src_texts"][1] or "").strip()
        user = f"A: {tA}\nB: {tB}"
        binary_msgs_list.append(
            [{"role": "system", "content": SYS_BINARY},
             {"role": "user", "content": user}]
        )

    # B/C. 两路各自 yes/no (srcA 一批, srcB 一批)
    singleA_msgs_list = []
    singleB_msgs_list = []
    for s in nspk2:
        tA = (s["sep_info"]["per_src_texts"][0] or "").strip()
        tB = (s["sep_info"]["per_src_texts"][1] or "").strip()
        singleA_msgs_list.append(
            [{"role": "system", "content": SYS_SINGLE},
             {"role": "user", "content": tA}]
        )
        singleB_msgs_list.append(
            [{"role": "system", "content": SYS_SINGLE},
             {"role": "user", "content": tB}]
        )

    # ---- 4. 跑 batch 生成 ----
    BS = 16
    t0 = time.time()
    print(f"[run] binary {len(binary_msgs_list)} on bs={BS}")
    raw_binary = batch_generate(model, tok, torch, binary_msgs_list,
                                max_new_tokens=4, batch_size=BS)
    print(f"[run] binary done {time.time()-t0:.1f}s, sample raws:")
    for r in raw_binary[:8]:
        print(f"    {r!r}")

    t0 = time.time()
    print(f"[run] singleA {len(singleA_msgs_list)}")
    raw_singleA = batch_generate(model, tok, torch, singleA_msgs_list,
                                 max_new_tokens=4, batch_size=BS)
    print(f"[run] singleA done {time.time()-t0:.1f}s")

    t0 = time.time()
    print(f"[run] singleB {len(singleB_msgs_list)}")
    raw_singleB = batch_generate(model, tok, torch, singleB_msgs_list,
                                 max_new_tokens=4, batch_size=BS)
    print(f"[run] singleB done {time.time()-t0:.1f}s")

    # ---- 5. 解析 + 计算 ----
    per_sample_out = []
    n_parse_fail_binary = 0
    n_parse_fail_single = 0
    # 都录入统计
    n_both_assistant = 0       # TRAP: 两路都是对助手说话 (真双指令)
    n_only_target = 0          # 只有 target 路(LLM 二选一选的那路)是
    n_only_nontarget = 0       # 只有非 target 路是
    n_neither = 0              # 两路都不是
    # 二选一选对率
    n_llm_pick_correct = 0
    n_llm_parse_ok = 0
    # heuristic 选对率(参考, 同 task1)
    n_heur_pick_correct = 0

    for i, s in enumerate(nspk2):
        sep = s["sep_info"]
        tA = sep["per_src_texts"][0]
        tB = sep["per_src_texts"][1]
        oracle_idx = sep["oracle_idx"]
        heur_idx = sep["heuristic_idx"]

        # 二选一
        llm_idx = parse_binary(raw_binary[i])
        if llm_idx is None:
            n_parse_fail_binary += 1
            # 解析失败兜底: 跟 heuristic (避免空文本崩)
            llm_idx_for_fallback = heur_idx
        else:
            n_llm_parse_ok += 1

        # 二选一选对(只对解析成功算)
        if llm_idx is not None and llm_idx == oracle_idx:
            n_llm_pick_correct += 1
        if heur_idx == oracle_idx:
            n_heur_pick_correct += 1

        # 用 llm_idx (None 时退 heuristic)
        llm_idx_eff = llm_idx if llm_idx is not None else heur_idx
        llm_route_text = sep["per_src_texts"][llm_idx_eff]
        llm_route_cer = cer_pair(llm_route_text, s["ref"])

        # 单路判别
        a_yes = parse_single(raw_singleA[i])
        b_yes = parse_single(raw_singleB[i])
        if a_yes is None or b_yes is None:
            n_parse_fail_single += 1

        # 都录入分类 (target = LLM 选的那路)
        target_yes = a_yes if llm_idx_eff == 0 else b_yes
        nontarget_yes = b_yes if llm_idx_eff == 0 else a_yes

        if a_yes and b_yes:
            n_both_assistant += 1
        elif a_yes and not b_yes:
            n_only_target_a = getattr(globals(), "_placeholder", None)
        # 上面写法易混, 改用 target/nontarget 维度

        # 重写清晰: 按 target / nontarget 维度
        # (覆盖前面累加)
        # 先撤回上面 n_only_target 等的污染(用 flag 重做)
        pass  # 实际下方统一收口

        per_sample_out.append({
            "uid": s["uid"],
            "n_spk": 2,
            "max_sim": s["max_sim"],
            "rejected_thr0.27": s["max_sim"] < 0.27,
            "ref": s["ref"],
            "srcA_text": tA,
            "srcB_text": tB,
            "srcA_cer": cer_pair(tA, s["ref"]),
            "srcB_cer": cer_pair(tB, s["ref"]),
            "heuristic_idx": heur_idx,
            "oracle_idx": oracle_idx,
            "heuristic_picks_oracle": heur_idx == oracle_idx,
            "llm_binary_raw": raw_binary[i],
            "llm_idx": llm_idx,
            "llm_idx_eff": llm_idx_eff,
            "llm_picks_oracle": (llm_idx == oracle_idx) if llm_idx is not None else None,
            "llm_route_text": llm_route_text,
            "llm_route_cer": llm_route_cer,
            "srcA_assistant_raw": raw_singleA[i],
            "srcB_assistant_raw": raw_singleB[i],
            "srcA_assistant_yes": a_yes,
            "srcB_assistant_yes": b_yes,
            "sep_sims": sep.get("sep_sims"),
            "heuristic_reason": sep.get("heuristic_reason"),
        })

    # ---- 5b. 重做都录入统计 (清晰版, 按 target/nontarget 维度) ----
    n_both_assistant = 0
    n_only_target = 0
    n_only_nontarget = 0
    n_neither = 0
    n_parse_unclear = 0
    for r in per_sample_out:
        a_yes = r["srcA_assistant_yes"]
        b_yes = r["srcB_assistant_yes"]
        # target = LLM 选的那路 (llm_idx_eff)
        if r["llm_idx_eff"] == 0:
            t_yes, nt_yes = a_yes, b_yes
        else:
            t_yes, nt_yes = b_yes, a_yes
        if a_yes is None or b_yes is None:
            n_parse_unclear += 1
            # 用 best-effort 分类 (None 视为 False 不严谨, 单独统计)
            continue
        if a_yes and b_yes:
            n_both_assistant += 1
        elif t_yes and not nt_yes:
            n_only_target += 1
        elif nt_yes and not t_yes:
            n_only_nontarget += 1
        else:
            n_neither += 1

    # ---- 6. 算全量含拒 CER (三档) ----
    # 三档的 route_text_fn
    def fn_mainline(s):
        # 主线: n_spk=1 / n_spk=0,3 用主线; n_spk=2 用主线 (回退基线)
        return s["mainline_text"]

    def fn_heuristic(s):
        if s.get("n_spk") == 2:
            idx = s["sep_info"]["heuristic_idx"]
            return s["sep_info"]["per_src_texts"][idx]
        return s["mainline_text"]

    def fn_llm(s):
        if s.get("n_spk") == 2:
            # 找 per_sample_out 里这个 uid
            idx = _uid2llmidx.get(s["uid"], s["sep_info"]["heuristic_idx"])
            return s["sep_info"]["per_src_texts"][idx]
        return s["mainline_text"]

    def fn_oracle(s):
        if s.get("n_spk") == 2:
            idx = s["sep_info"]["oracle_idx"]
            return s["sep_info"]["per_src_texts"][idx]
        return s["mainline_text"]

    _uid2llmidx = {r["uid"]: r["llm_idx_eff"] for r in per_sample_out}

    mainline_pool = pool_cer_with_thr(samples, fn_mainline)
    heur_pool = pool_cer_with_thr(samples, fn_heuristic)
    llm_pool = pool_cer_with_thr(samples, fn_llm)
    oracle_pool = pool_cer_with_thr(samples, fn_oracle)

    # 同口径 transcribe 池 (不拒识, 算所有转写 CER)
    def pool_cer_no_thr(samples, route_text_fn):
        total_err = 0
        total_char = 0
        for s in samples:
            ref_n = normalize_text(s["ref"])
            pred_n = normalize_text(route_text_fn(s))
            total_err += editdistance.eval(pred_n, ref_n)
            total_char += len(ref_n)
        return (total_err / total_char) if total_char else 0.0

    mainline_trans = pool_cer_no_thr(samples, fn_mainline)
    heur_trans = pool_cer_no_thr(samples, fn_heuristic)
    llm_trans = pool_cer_no_thr(samples, fn_llm)
    oracle_trans = pool_cer_no_thr(samples, fn_oracle)

    # ---- 7. 写出 ----
    OUT_PER.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PER, "w", encoding="utf-8") as f:
        json.dump(per_sample_out, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_PER}")

    summary = {
        "verdict": "LLM 泛化二选一选路 + 都录入假设验证",
        "task": "n_spk=2 两路 qwen 转写 → LLM 二选一(对助手说话标准) + 两路分别 yes/no",
        "n_total": len(samples),
        "n_nspk2": len(nspk2),
        "model": model_path,
        "thr": 0.27,
        "prompts": {"SYS_BINARY": SYS_BINARY, "SYS_SINGLE": SYS_SINGLE},
        "mainline_baseline_published": {
            "transcribe_pool_cer": 0.3436,
            "containing_reject_thr0.27_cer": 0.5934,
        },
        "task1_heuristic_baseline": {
            "transcribe_pool_cer": 0.2941,
            "thr0.27_cer_pool": 0.5727,
            "heuristic_pick_accuracy": 0.8435,
        },
        "recomputed_in_this_run": {
            "mainline_transcribe_cer_pool": round(mainline_trans, 4),
            "mainline_thr0.27_cer_pool": round(mainline_pool, 4),
            "heuristic_transcribe_cer_pool": round(heur_trans, 4),
            "heuristic_thr0.27_cer_pool": round(heur_pool, 4),
            "llm_transcribe_cer_pool": round(llm_trans, 4),
            "llm_thr0.27_cer_pool": round(llm_pool, 4),
            "oracle_transcribe_cer_pool": round(oracle_trans, 4),
            "oracle_thr0.27_cer_pool": round(oracle_pool, 4),
            "delta_llm_vs_heuristic_thr": round(llm_pool - heur_pool, 4),
            "delta_llm_vs_heuristic_transcribe": round(llm_trans - heur_trans, 4),
            "delta_oracle_vs_llm_thr": round(oracle_pool - llm_pool, 4),
            "pct_change_llm_vs_heuristic_thr": round((llm_pool - heur_pool) / heur_pool * 100, 2),
            "pct_change_oracle_vs_heuristic_thr": round((oracle_pool - heur_pool) / heur_pool * 100, 2),
        },
        "binary_pick_stats": {
            "n_nspk2": len(nspk2),
            "n_llm_parse_ok": n_llm_parse_ok,
            "n_parse_fail_binary": n_parse_fail_binary,
            "llm_pick_accuracy": round(n_llm_pick_correct / max(n_llm_parse_ok, 1), 4),
            "heuristic_pick_accuracy": round(n_heur_pick_correct / len(nspk2), 4),
            "n_llm_pick_correct": n_llm_pick_correct,
            "n_heur_pick_correct": n_heur_pick_correct,
            "note": "llm_pick_accuracy 分母只算解析成功, heuristic 分母全部 805",
        },
        "doulu_assistant_hypothesis": {
            "n_total_nspk2": len(nspk2),
            "n_both_assistant_TRAP": n_both_assistant,
            "pct_both_assistant_TRAP": round(n_both_assistant / len(nspk2) * 100, 2),
            "n_only_target_assistant": n_only_target,
            "pct_only_target": round(n_only_target / len(nspk2) * 100, 2),
            "n_only_nontarget_assistant": n_only_nontarget,
            "pct_only_nontarget": round(n_only_nontarget / len(nspk2) * 100, 2),
            "n_neither_assistant": n_neither,
            "pct_neither": round(n_neither / len(nspk2) * 100, 2),
            "n_parse_unclear_excluded": n_parse_unclear,
            "note": "TRAP=两路都是对助手说话(真双指令, LLM 二选一在此失效正常); "
                    "target=LLM 二选一选的那路; nontarget=另一路",
        },
        "per_sample_path": str(OUT_PER),
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_SUMMARY}")

    # ---- 8. 控制台速报 ----
    print("\n=== LLM 二选一选路 + 都录入假设验证 ===")
    print(f"n_spk=2 样本: {len(nspk2)} (LLM 解析成功: {n_llm_parse_ok})")
    print(f"\n[二选一选对率]")
    print(f"  heuristic : {n_heur_pick_correct}/{len(nspk2)} = {n_heur_pick_correct/len(nspk2):.4f}")
    print(f"  LLM       : {n_llm_pick_correct}/{n_llm_parse_ok} = {n_llm_pick_correct/max(n_llm_parse_ok,1):.4f}")
    print(f"\n[全量含拒 CER thr0.27] (n={len(samples)})")
    print(f"  mainline  : {mainline_pool:.4f}  (published: 0.5931/0.5934)")
    print(f"  heuristic : {heur_pool:.4f}  (published: 0.5727)")
    print(f"  LLM       : {llm_pool:.4f}  Δ vs heuristic = {(llm_pool-heur_pool)/heur_pool*100:+.2f}%")
    print(f"  oracle    : {oracle_pool:.4f}  (上限)")
    print(f"\n[全量 transcribe 池 CER (不拒识)]")
    print(f"  mainline  : {mainline_trans:.4f}")
    print(f"  heuristic : {heur_trans:.4f}  (published: 0.2941)")
    print(f"  LLM       : {llm_trans:.4f}")
    print(f"  oracle    : {oracle_trans:.4f}")
    print(f"\n[都录入假设]")
    print(f"  两路都对助手 (TRAP)        : {n_both_assistant}/{len(nspk2)} = {n_both_assistant/len(nspk2)*100:.2f}%")
    print(f"  只 target 对助手          : {n_only_target}/{len(nspk2)} = {n_only_target/len(nspk2)*100:.2f}%")
    print(f"  只 nontarget 对助手       : {n_only_nontarget}/{len(nspk2)} = {n_only_nontarget/len(nspk2)*100:.2f}%")
    print(f"  两路都不对助手            : {n_neither}/{len(nspk2)} = {n_neither/len(nspk2)*100:.2f}%")
    print(f"  解析不明(排除)            : {n_parse_unclear}")


if __name__ == "__main__":
    main()
