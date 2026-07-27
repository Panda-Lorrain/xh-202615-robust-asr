"""LLM 路由 (策略2): Qwen2.5-3B-Instruct 判哪路像家居指令。

读 B2 summary.json 40 条的两路 transcript, 每路调 llm_reject.LLMRejecter,
挑 verdict=accept 的那路 (都accept/都reject fallback)。
输出: code/runs/_multivoice_route/llm_routing.json
"""
import json, os, sys, time, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from llm_reject import LLMRejecter

B2 = os.path.join(HERE, "runs", "_sepformer_b2", "summary.json")
OUT = os.path.join(HERE, "runs", "_multivoice_route", "llm_routing.json")


def main():
    b2 = json.load(open(B2, encoding="utf-8"))
    results = b2["results"]
    print(f"[load] Qwen2.5-3B-Instruct (FP16 cuda:0)")
    rej = LLMRejecter(device="cuda:0")
    out_rows = []
    t0 = time.time()
    for i, r in enumerate(results):
        per_src = r["per_src"]
        verdicts = []
        for s in per_src:
            v = rej.reject(s["text"])
            verdicts.append({"src_idx": s["src_idx"], "text": s["text"],
                             "cer": s["cer"],
                             "verdict": v.get("verdict", "reject"),
                             "entity": v.get("entity", "none"),
                             "action": v.get("action", "none"),
                             "reason": v.get("reason", "")[:80]})
        # 选路逻辑
        acc = [j for j, v in enumerate(verdicts) if v["verdict"] == "accept"]
        if len(acc) == 1:
            pick = acc[0]; reason = "one_accept"
        elif len(acc) == 0:
            pick = 0; reason = "both_reject_fallback_src0"
        else:
            # 都 accept: 挑含 entity+action 更明确的 (entity!=none 加分)
            scores = []
            for v in verdicts:
                sc = 0
                if v["entity"] != "none": sc += 1
                if v["action"] != "none": sc += 1
                scores.append(sc)
            if scores[0] != scores[1]:
                pick = 0 if scores[0] > scores[1] else 1
                reason = f"both_accept_entity_action_{scores}"
            else:
                pick = 0; reason = "both_accept_tie_default_src0"
        oracle = r["oracle_src_idx"]
        out_rows.append({
            "uid": r["uid"], "ref": r["ref"],
            "oracle_idx": oracle, "oracle_cer": r["oracle_cer"],
            "pick": pick, "cer": per_src[pick]["cer"],
            "correct": pick == oracle,
            "reason": reason,
            "verdicts": verdicts,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/40] elapsed {time.time()-t0:.1f}s")
    cers = [r["cer"] for r in out_rows]
    correct = [r["correct"] for r in out_rows]
    summary = {
        "strategy": "llm",
        "mean_cer": round(statistics.mean(cers), 4),
        "median_cer": round(statistics.median(cers), 4),
        "accuracy": round(sum(correct) / len(correct), 4),
        "n_correct": sum(correct),
        "n": len(cers),
        "fallback_reasons": {},
        "per_sample": out_rows,
    }
    for r in out_rows:
        summary["fallback_reasons"][r["reason"]] = \
            summary["fallback_reasons"].get(r["reason"], 0) + 1
    json.dump(summary, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n[LLM 策略] mean CER: {summary['mean_cer']:.4f}  "
          f"准确率: {summary['accuracy']*100:.1f}%  "
          f"({summary['n_correct']}/{summary['n']})  "
          f"wall {time.time()-t0:.1f}s")
    print(f"产物: {OUT}")


if __name__ == "__main__":
    main()
