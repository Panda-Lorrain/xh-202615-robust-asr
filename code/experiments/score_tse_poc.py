#!/usr/bin/env python
"""算 TSE POC 的 CER：从 qwen 转写结果 + ref → CER + 主线对比。

用法:
  code/.venv/Scripts/python.exe code/score_tse_poc.py \
    --tse-dir E:/midea_target_asr/code/runs/_tse_poc \
    --qwen-json E:/midea_target_asr/code/runs/_tse_poc/qwen_text.json

输出: code/runs/_tse_poc/tse_poc_score.json + markdown 摘要
"""
import os, sys, json, argparse

# 用主 venv (有 eval_metrics 模块)
sys.path.insert(0, r"E:/midea_target_asr/code")
from eval_metrics import cer_official  # noqa

# 主线 baseline CER (已知翻车, 都是 ~1.0, 来自 poc_qwen_asr_full_result.json)
BASELINE_QWEN_CER = {
    "cmd_2637": 1.125,   # sim 0.585 重叠区
    "cmd_18": 1.000,     # sim 0.058 死区
    "cmd_2098": 1.000,   # sim 0.020 死区
    "cmd_2251": 1.000,   # sim 0.604 重叠组
    "cmd_2687": 1.000,   # sim 0.579 重叠组
    "cmd_2630": 2.250,   # sim 0.567 极端幻觉组
}

# ref (来自 poc_qwen_asr_full_result.json)
REFS = {
    "cmd_2637": "哺乳期要少吃什么",
    "cmd_18": "关闭灯光",
    "cmd_2098": "调到二十八度",
    "cmd_2251": "把温度调到三十度",
    "cmd_2687": "把温度调到三十度",
    "cmd_2630": "开左右风",
}

# 主线 qwen 转写 (参考用, 知道主线翻车成什么样)
BASELINE_QWEN_TEXT = {
    "cmd_2637": "(主线转写: 就已经进行过一轮教育 - 切了含重叠区)",
    "cmd_18": "(主线转写: 我我十天 - ASR 幻觉)",
    "cmd_2098": "(主线转写: 请你把温 - argmax 选错 target)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tse-dir", default=r"E:/midea_target_asr/code/runs/_tse_poc")
    ap.add_argument("--qwen-json", default=r"E:/midea_target_asr/code/runs/_tse_poc/qwen_text.json")
    args = ap.parse_args()

    qwen_text = json.load(open(args.qwen_json, encoding="utf-8"))
    summary = json.load(open(os.path.join(args.tse_dir, "tse_extract_summary.json"),
                             encoding="utf-8"))

    results = []
    for uid, ref in REFS.items():
        # TSE 输出文件命名 cmd_N_tse.wav, qwen uid 为 cmd_N_tse
        quid = f"{uid}_tse"
        if quid not in qwen_text:
            print(f"[skip] {uid}: 不在 qwen 输出 (找 {quid})")
            continue
        hyp = qwen_text[quid]
        cer = cer_official(hyp, ref)
        base_cer = BASELINE_QWEN_CER.get(uid, None)
        delta = cer - base_cer if base_cer is not None else None
        saved = delta < 0 if delta is not None else False
        results.append({
            "uid": uid,
            "ref": ref,
            "tselm_hyp": hyp,
            "tselm_cer": round(cer, 4),
            "mainline_qwen_cer": base_cer,
            "delta": round(delta, 4) if delta is not None else None,
            "saved": saved,
            "extract": summary.get(uid, {}),
        })

    # 汇总
    saved_n = sum(1 for r in results if r["saved"])
    total_n = len(results)
    avg_tse = sum(r["tselm_cer"] for r in results) / total_n
    avg_main = sum(r["mainline_qwen_cer"] for r in results) / total_n
    overall = {
        "n": total_n,
        "saved_n": saved_n,
        "avg_tselm_cer": round(avg_tse, 4),
        "avg_mainline_cer": round(avg_main, 4),
        "avg_delta": round(avg_tse - avg_main, 4),
        "verdict_short": "",
    }
    if saved_n >= total_n * 0.5:
        overall["verdict_short"] = f"TSE 救回 {saved_n}/{total_n} 条 → 治本路线有效"
    elif saved_n > 0:
        overall["verdict_short"] = f"TSE 仅救 {saved_n}/{total_n} 条 → 部分有效, 需中文训练"
    else:
        overall["verdict_short"] = f"TSE 0/{total_n} 救不回 → 英文训练 OOD/物理地板, 治本路线需中文训练"

    print("\n=== TSE POC score ===")
    for r in results:
        flag = "✓救" if r["saved"] else "✗"
        print(f"  [{flag}] {r['uid']}: TSE CER {r['tselm_cer']:.3f} "
              f"vs 主线 {r['mainline_qwen_cer']:.3f} (Δ{r['delta']:+.3f}) "
              f"| ref='{r['ref']}' hyp='{r['tselm_hyp']}'")
    print(f"\n{overall['verdict_short']}")
    print(f"  avg: TSE {overall['avg_tselm_cer']:.3f} vs 主线 {overall['avg_mainline_cer']:.3f}")

    out_json = os.path.join(args.tse_dir, "tse_poc_score.json")
    json.dump({"overall": overall, "rows": results},
              open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
