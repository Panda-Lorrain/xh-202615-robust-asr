#!/usr/bin/env python3
"""datasetA 标注 -> submit_infer --pairs manifest 转换器。

datasetA 的 pos.jsonl/neg.jsonl 用中文 key + 相对路径,submit_infer --pairs
期望 [{"enrollment","recognition"}, ...](英文 key + 可读路径)。本脚本做适配,
保留 id / ref / kws_txt 供评测 join(submit_infer 会忽略多余字段)。

输出:
  code/pos_pairs_datasetA.json   (1364 条, ref=识别文本)
  code/neg_pairs_datasetA.json   (474 条,  ref=None)

⚠️ pos id(0-1363) 与 neg id(1000-1473)重叠,utt_id(cmd_N)会冲突 →
   pos/neg 必须分两次跑 submit_infer(见 eval_datasetA.py 用法)。
"""
import json
from pathlib import Path

ROOT = Path(r"E:\midea_target_asr\datasetA")
OUTDIR = Path(__file__).resolve().parent  # code/


def make(jsonl_name, out_name):
    rows = [json.loads(l) for l in open(ROOT / jsonl_name, encoding="utf-8") if l.strip()]
    pairs = []
    for r in rows:
        pairs.append({
            "id": r["id"],
            "enrollment": str((ROOT / r["唤醒音频"]).resolve()),
            "recognition": str((ROOT / r["识别音频"]).resolve()),
            "ref": r.get("识别文本"),   # pos 有值, neg 为 None
            "kws_txt": r.get("唤醒文本"),
        })
    out = OUTDIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    n_ref = sum(1 for p in pairs if p["ref"])
    print(f"{out_name}: {len(pairs)} 条 (ref 非空 {n_ref}) -> {out}")
    return pairs


if __name__ == "__main__":
    make("pos.jsonl", "pos_pairs_datasetA.json")
    make("neg.jsonl", "neg_pairs_datasetA.json")
    print("\n[下一步] 分两次跑 submit_infer:")
    print("  code/.venv/Scripts/python.exe code/submit_infer.py \\")
    print("    --pairs code/pos_pairs_datasetA.json --out-dir code/out_pos --limit 3   # 冒烟")
