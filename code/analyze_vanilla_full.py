#!/usr/bin/env python3
"""vanilla 全量结果分析：转写CER + thr工作点(含拒识) + sim分桶 + 数字格式 + 英文幻觉。
对比 DiCoW（同一样本的 dicow_text）。回答"vanilla 路线 CER 腿能拿多少分"。
用法: code/.venv/Scripts/python.exe code/analyze_vanilla_full.py [exp_vanilla_full.json]
"""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_datasetA import _norm_zh
from eval_metrics import cer


def cer_of(text, ref):
    t = _norm_zh(text or "")
    r = _norm_zh(ref or "")
    if not t:
        return 1.0
    return cer(t, r)


def is_eng(t):
    L = [c for c in (t or "") if c.isalpha()]
    return sum(c.isascii() for c in L) / len(L) > 0.5 if len(L) >= 4 else False


def has_digit(t):
    return bool(re.search(r"\d", t or ""))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "exp_vanilla_full.json")
    r = json.load(open(path, encoding="utf-8"))
    valid = [x for x in r if "max_sim" in x and "vanilla_text" in x]
    n = len(valid)
    print(f"===== vanilla 全量分析（{n} 条，diar 失败 {len(r)-n}）=====")

    # 1. 转写 CER（不拒）
    print("\n-- 转写 CER（always_generate，不拒）--")
    for label, key in [("vanilla", "vanilla_text"), ("dicow", "dicow_text")]:
        cs = [cer_of(x[key], x["ref"]) for x in valid]
        print(f"  {label:8}: overall={sum(cs)/n:.4f} correct(CER<0.5)={sum(1 for c in cs if c<0.5)/n:.2%} "
              f"near(<0.1)={sum(1 for c in cs if c<0.1)/n:.2%} 英文幻觉={sum(1 for x in valid if is_eng(x[key]))/n:.2%}")

    # 2. thr 工作点（含拒识，拒=1.0）—— 提交时 overall CER，最关键
    print("\n-- thr 工作点（含拒识拒=1.0 = 提交 overall CER）【最关键】--")
    print(f"  {'thr':<6} {'vanilla':<10} {'dicow':<10} {'Δ(v-d)':<10} {'van correct':<14} {'dic correct'}")
    for thr in [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]:
        v_cs = [1.0 if x["max_sim"] < thr else cer_of(x["vanilla_text"], x["ref"]) for x in valid]
        d_cs = [1.0 if x["max_sim"] < thr else cer_of(x["dicow_text"], x["ref"]) for x in valid]
        v_o = sum(v_cs) / n; d_o = sum(d_cs) / n
        v_c = sum(1 for c in v_cs if c < 0.5) / n; d_c = sum(1 for c in d_cs if c < 0.5) / n
        print(f"  {thr:<6.2f} {v_o:<10.4f} {d_o:<10.4f} {v_o-d_o:<+10.4f} {v_c:<14.2%} {d_c:.2%}")

    # 3. sim 分桶（转写 CER）
    print("\n-- sim 分桶（转写 CER，看 H3 在哪桶成立）--")
    for lo, hi in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 1.0)]:
        b = [x for x in valid if lo <= x["max_sim"] < hi]
        if not b:
            continue
        v = sum(cer_of(x["vanilla_text"], x["ref"]) for x in b) / len(b)
        d = sum(cer_of(x["dicow_text"], x["ref"]) for x in b) / len(b)
        print(f"  sim[{lo:.1f},{hi:.1f}): n={len(b)} vanilla={v:.3f} dicow={d:.3f} Δ={v-d:+.3f}")

    # 4. 数字格式影响（vanilla 唯一劣势来源）
    print("\n-- 数字格式影响（vanilla 含数字的劣势 = 后处理可修空间）--")
    for label, b in [("vanilla含数字", [x for x in valid if has_digit(x["vanilla_text"])]),
                     ("vanilla无数字", [x for x in valid if not has_digit(x["vanilla_text"])])]:
        if not b:
            continue
        v = sum(cer_of(x["vanilla_text"], x["ref"]) for x in b) / len(b)
        d = sum(cer_of(x["dicow_text"], x["ref"]) for x in b) / len(b)
        print(f"  {label}: n={len(b)} vanilla={v:.3f} dicow={d:.3f} Δ={v-d:+.3f}")


if __name__ == "__main__":
    main()
