#!/usr/bin/env python3
"""datasetA 真实评测: pos 算 CER(误拒条 CER=1.0) / neg 算 RR(句准拒识率)。

替代 eval_full_test.py(后者硬编码仿真集文件名 ov_snr_noise + overlap/snr/noise
分组,真实数据无这些标签)。复用 eval_metrics.cer()。

用法:
  code/.venv/Scripts/python.exe code/eval_datasetA.py <result.json> <pairs.json>

pairs.json 由 make_pairs_from_datasetA.py 生成,含 id/enrollment/recognition/ref。
mode(pos/neg)由 ref 是否非空自动判定。
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import cer

try:
    import zhconv
    def _norm_zh(t):
        """繁→简归一:消除 Whisper 输出繁体 vs ref 简体的字符级 CER 虚高(冒烟 cmd_0
        繁体'空調開到自熱'对简体 ref 每个繁体字都被算错)。只作用于 CER 计算,
        不改存储的原始 text/ref。"""
        return zhconv.convert(t or "", "zh-cn")
    _HAS_ZHCONV = True
except ImportError:
    def _norm_zh(t):
        return t or ""
    _HAS_ZHCONV = False


def key_from_rec(p):
    """recognition 路径 -> join key。'E:/.../pos/cmd_0.wav' -> 'cmd_0'。
    与 submit_infer result 的 recognition 字段对齐(原路径保留)。"""
    return os.path.splitext(os.path.basename(p))[0]


def main():
    result_path = sys.argv[1]
    manifest_path = sys.argv[2]
    out_path = result_path.replace(".json", "_eval.json")

    results_list = json.load(open(result_path, encoding="utf-8"))["results"]
    results = {key_from_rec(it["recognition"]): it for it in results_list}
    man = json.load(open(manifest_path, encoding="utf-8"))

    has_ref = any(r.get("ref") for r in man)
    mode = "pos" if has_ref else "neg"

    miss = []
    if mode == "pos":
        rows = []
        for r in man:
            k = key_from_rec(r["recognition"])
            if k not in results:
                miss.append(r["id"])
                continue
            res = results[k]
            ref = r["ref"] or ""
            rejected = bool(res.get("rejected"))
            text = res.get("text", "") or ""
            # pos 全是目标指令,不该拒。误拒/空转写 -> CER=1.0(错全)
            # CER 计算前繁→简归一(消除 Whisper 繁体输出 vs 简体 ref 的虚高)
            c = 1.0 if (rejected or not text) else cer(_norm_zh(text), _norm_zh(ref))
            rows.append({"id": r["id"], "kws_txt": r.get("kws_txt"),
                         "cer": c, "rejected": rejected, "text": text, "ref": ref,
                         "max_sim": res.get("max_sim")})

        n = len(rows)
        overall = sum(x["cer"] for x in rows) / n if n else 0.0
        correct = sum(1 for x in rows if x["cer"] < 0.5) / n if n else 0.0
        near = sum(1 for x in rows if x["cer"] < 0.1) / n if n else 0.0
        acc = [x for x in rows if not x["rejected"]]
        cer_acc = sum(x["cer"] for x in acc) / len(acc) if acc else 0.0
        false_rej = sum(1 for x in rows if x["rejected"]) / n if n else 0.0

        summary = {
            "mode": "pos", "n": n, "manifest_miss": len(miss),
            "overall_cer": round(overall, 4),
            "correct_rate(CER<0.5)": round(correct, 4),
            "near_perfect(CER<0.1)": round(near, 4),
            "cer_accepted_only": round(cer_acc, 4),
            "false_reject_rate(伤CER)": round(false_rej, 4),
            "n_rejected": sum(1 for x in rows if x["rejected"]),
        }
        print(f"\n===== pos CER 实测 ({n} 条, manifest 未匹配 {len(miss)}) =====")
        print(f"overall CER             : {overall:.4f}")
        print(f"correct_rate (CER<0.5)  : {correct:.2%}")
        print(f"near_perfect(CER<0.1)   : {near:.2%}")
        print(f"cer (仅 accepted)       : {cer_acc:.4f}")
        print(f"误拒率 (pos 被拒,伤CER) : {false_rej:.2%} ({summary['n_rejected']}/{n})")

    else:  # neg
        rows = []
        for r in man:
            k = key_from_rec(r["recognition"])
            if k not in results:
                miss.append(r["id"])
                continue
            res = results[k]
            rejected = bool(res.get("rejected"))
            text = res.get("text", "") or ""
            rows.append({"id": r["id"], "kws_txt": r.get("kws_txt"),
                         "rejected": rejected, "text": text,
                         "max_sim": res.get("max_sim")})

        n = len(rows)
        rr = sum(1 for x in rows if x["rejected"]) / n if n else 0.0
        leak = [x for x in rows if not x["rejected"]]
        leak_avg_len = sum(len(x["text"]) for x in leak) / len(leak) if leak else 0.0

        summary = {
            "mode": "neg", "n": n, "manifest_miss": len(miss),
            "RR_句准拒识率": round(rr, 4),
            "漏拒率(被转写)": round(1 - rr, 4),
            "n_correct_reject": sum(1 for x in rows if x["rejected"]),
            "n_leak": len(leak),
            "漏拒条平均转写字数": round(leak_avg_len, 2),
        }
        print(f"\n===== neg 拒识实测 ({n} 条, manifest 未匹配 {len(miss)}) =====")
        print(f"RR (句准拒识率)        : {rr:.2%} ({summary['n_correct_reject']}/{n})")
        print(f"漏拒率 (neg 被转写)    : {1-rr:.2%} ({len(leak)}/{n})")
        if leak:
            print(f"漏拒条平均转写长度     : {leak_avg_len:.1f} 字")

    summary["miss_ids"] = miss[:20]
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")
    print(f"[评分映射] {'CER 40%' if mode=='pos' else '拒识率 40%'} <- 上面对应指标")


if __name__ == "__main__":
    main()
