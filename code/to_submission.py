"""把 submit_infer result.json 转成官方提交格式。

官方格式(2026-08-06 FAQ 更新, 加 avg_rr):
  {"result":{"results":[{"id","content","label","cer"}],"final_cer","avg_rr","duration"}}

- final_cer: 官方累计池(total_errors/total_chars), 非逐句算术平均(2026-08-06 修)
- avg_rr:    neg 正确拒识率(FAQ 第1条, 2026-08-06 新增)
- duration:  batch=1 端到端 wall(含模型加载), 取 timing.json total_wall_sec(非 duration_infer_sec, 那个漏 Qwen 转写)
待主办方确认口径做成 SUBMISSION_DEFAULTS 常量, 主办方回复只改常量不改逻辑。
"""
import os, sys, json, re, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import to_simplified, digit_postproc
from eval_metrics import cer_pool, cer_official

# 待主办方确认口径(默认推测值, 见 spec §7)
SUBMISSION_DEFAULTS = {
    "label_accept": "accept",
    "label_reject": "reject",
    "pos_rejected_cer": 1.0,  # pos 被错拒的 cer(删除错误)
}


def _utt_id_stripped(p):
    """去 uttN_ 前缀(HANDOFF §8 坑5): utt0012_cmd_3.wav → cmd_3。无前缀则原样去 ext。"""
    base = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r"^utt\d+_", "", base)


def convert(result_json, pairs_json, duration_sec=None):
    """result.json + pairs manifest → 官方 submission dict。

    pairs_json 提供 ref(pos 有 / neg 空)。duration_sec 为端到端 wall(含模型加载),
    None 时兜底 per-utt infer_sec 累加(⚠️ qwen 后端漏 ASR 转写, 应由 main 从 timing.json total_wall_sec 传入)。
    """
    with open(result_json, encoding="utf-8") as f:
        result = json.load(f)
    with open(pairs_json, encoding="utf-8") as f:
        pair_rows = json.load(f)

    ref_map, is_neg_map = {}, {}
    for row in pair_rows:
        uid = _utt_id_stripped(row["recognition"])
        ref_map[uid] = row.get("ref", "") or ""
        is_neg_map[uid] = (not row.get("ref")) or row.get("label") == "neg"

    rows_out = []
    pos_preds, pos_refs = [], []        # 官方累计池(total_errors/total_chars), 非 pos_cers 算术平均
    neg_total, neg_correct_reject = 0, 0   # avg_rr(FAQ 第1条): neg 正确拒识率
    for r in result.get("results", []):
        uid = _utt_id_stripped(r.get("recognition", ""))
        # 提交归一 SSOT(2026-07-08 workflow④): digit+繁简, 与 enroll_infer:317-319 / recompute submit_norm 对齐
        text = digit_postproc(to_simplified(r.get("text", "") or ""))
        rejected = bool(r.get("rejected"))
        label = SUBMISSION_DEFAULTS["label_reject"] if rejected else SUBMISSION_DEFAULTS["label_accept"]
        is_neg = is_neg_map.get(uid, False)
        if is_neg:
            cer_val = ""  # neg 不评 CER(评 RR)
            neg_total += 1
            if rejected:
                neg_correct_reject += 1
        else:
            ref = ref_map.get(uid, "")
            if not ref:
                cer_val = ""  # 无 ref 边界, 不进累计池
            else:
                # rejected 时 text="" (submit_infer.py:435), CERMetric 天然 errors=len(ref)→CER1.0(删除错误)
                pos_preds.append(text)
                pos_refs.append(ref)
                cer_val = (round(SUBMISSION_DEFAULTS["pos_rejected_cer"], 3) if rejected
                           else round(cer_official(text, ref), 3))
        rows_out.append({"id": uid, "content": text, "label": label, "cer": cer_val})

    # final_cer = 官方累计池(total_errors/total_chars), 与主办方 CERMetric 逐行等价(2026-07-08 4-agent 坐实)
    final_cer = round(cer_pool(pos_preds, pos_refs), 3) if pos_refs else 0.0
    # avg_rr: neg 正确拒识数/neg 总数(单次跑 pos 时 neg_total=0 → None, 整体合并由 build_submission 取 neg 跑的值)
    avg_rr = round(neg_correct_reject / neg_total, 4) if neg_total else None
    if duration_sec is None:
        # 最后兜底: per-utt infer_sec 累加(⚠️ qwen 后端漏 ASR 转写, 严重低估, 应由 main 传 total_wall_sec)
        duration_sec = sum(float(r.get("infer_sec", 0) or 0) for r in result.get("results", []))
    return {"result": {"results": rows_out, "final_cer": final_cer,
                       "avg_rr": avg_rr, "duration": round(duration_sec, 3)}}


def main():
    ap = argparse.ArgumentParser(description="result.json → 官方提交格式")
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--duration", type=float, default=None,
                    help="batch=1 端到端总秒数(优先; 覆盖 timing.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # duration 优先级: --duration > 同目录 timing.json 的 total_wall_sec(端到端含加载) > duration_infer_sec > per-utt infer_sec
    # ⚠️ 必须用 total_wall_sec: duration_infer_sec 漏 Qwen 转写(enroll_infer 子进程), 严重低估(memory l20-eval-hardware)
    duration = args.duration
    if duration is None:
        timing_json = args.result_json.replace("result.json", "timing.json")
        if os.path.exists(timing_json):
            tj = json.load(open(timing_json, encoding="utf-8"))
            duration = tj.get("total_wall_sec") or tj.get("duration_infer_sec")
    sub = convert(args.result_json, args.pairs, duration)
    out = args.out or args.result_json.replace("result.json", "submission.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(sub['result']['results'])} 条 → {out} "
          f"(final_cer={sub['result']['final_cer']}, avg_rr={sub['result']['avg_rr']}, duration={sub['result']['duration']}s)")


if __name__ == "__main__":
    main()
