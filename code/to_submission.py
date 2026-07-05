"""把 submit_infer result.json 转成官方提交格式。

官方格式(2026-07-06):
  {"result":{"results":[{"id","content","label","cer"}],"final_cer","duration"}}

待主办方确认口径做成 SUBMISSION_DEFAULTS 常量, 主办方回复只改常量不改逻辑。
"""
import os, sys, json, re, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import to_simplified
from eval_datasetA import _norm_zh
from eval_metrics import cer as _cer

# 待主办方确认口径(默认推测值, 见 spec §7)
SUBMISSION_DEFAULTS = {
    "label_accept": "accept",
    "label_reject": "reject",
    "pos_rejected_cer": 1.0,  # pos 被错拒的 cer
}


def _utt_id_stripped(p):
    """去 uttN_ 前缀(HANDOFF §8 坑5): utt0012_cmd_3.wav → cmd_3。无前缀则原样去 ext。"""
    base = os.path.splitext(os.path.basename(p))[0]
    return re.sub(r"^utt\d+_", "", base)


def convert(result_json, pairs_json, duration_infer_sec=None):
    """result.json + pairs manifest → 官方 submission dict。

    pairs_json 提供 ref(pos 有 / neg 空)。duration_infer_sec 为 None 时从 per-utt infer_sec 累加。
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

    rows_out, pos_cers = [], []
    for r in result.get("results", []):
        uid = _utt_id_stripped(r.get("recognition", ""))
        text = to_simplified(r.get("text", "") or "")
        rejected = bool(r.get("rejected"))
        label = SUBMISSION_DEFAULTS["label_reject"] if rejected else SUBMISSION_DEFAULTS["label_accept"]
        is_neg = is_neg_map.get(uid, False)
        if is_neg:
            cer_val = ""  # neg 不评 CER(评 RR)
        else:
            ref = ref_map.get(uid, "")
            if not ref:
                cer_val = ""
            elif rejected:
                cer_val = round(SUBMISSION_DEFAULTS["pos_rejected_cer"], 3)
                pos_cers.append(SUBMISSION_DEFAULTS["pos_rejected_cer"])
            else:
                t = _norm_zh(text); rr = _norm_zh(ref)
                c = _cer(t, rr) if t else 1.0
                cer_val = round(c, 3)
                pos_cers.append(c)
        rows_out.append({"id": uid, "content": text, "label": label, "cer": cer_val})

    final_cer = round(sum(pos_cers) / len(pos_cers), 3) if pos_cers else 0.0
    if duration_infer_sec is None:
        duration_infer_sec = sum(float(r.get("infer_sec", 0) or 0) for r in result.get("results", []))
    return {"result": {"results": rows_out, "final_cer": final_cer,
                       "duration": round(duration_infer_sec, 3)}}


def main():
    ap = argparse.ArgumentParser(description="result.json → 官方提交格式")
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--duration", type=float, default=None,
                    help="batch=1 逐条推理总秒数(覆盖 infer_sec 累加)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # duration 优先级: --duration > 同目录 timing.json 的 duration_infer_sec > per-utt infer_sec 累加
    duration = args.duration
    if duration is None:
        timing_json = args.result_json.replace("result.json", "timing.json")
        if os.path.exists(timing_json):
            duration = json.load(open(timing_json, encoding="utf-8")).get("duration_infer_sec")
    sub = convert(args.result_json, args.pairs, duration)
    out = args.out or args.result_json.replace("result.json", "submission.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(sub['result']['results'])} 条 → {out} "
          f"(final_cer={sub['result']['final_cer']}, duration={sub['result']['duration']}s)")


if __name__ == "__main__":
    main()
