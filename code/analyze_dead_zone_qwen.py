#!/usr/bin/env python3
"""死区(sim<0.2) Qwen3-ASR 0.459 突破的对抗验证 —— H1(真实转写鲁棒性) vs H2(LM 先验幻觉)。

⚠️ 缘起(2026-07-11 A2): Qwen3-ASR 死区官方池 0.459 < spk-oracle-poc 的 oracle 0.607
(完美选 target + vanilla 仍 0.607), 挑战"死区=babble 摧毁 mel 物理地板不可破"叙事。
本脚本纯分析(不推理, 切片已在 poc json)给 H1/H2 信号; 听音核验作可选补充。

数据源: poc_qwen_asr_full_result.json(1350 条, 死区 sim<0.2 n=396)。
口径: per-row mean(poc qwen_cer, NFKC+去标点) vs 官方累计池(CERMetric, 提交归一后)。

用法: code/.venv/Scripts/python.exe code/analyze_dead_zone_qwen.py
"""
import json, statistics, random, sys, os
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc


def sn(t):
    return digit_postproc(to_simplified(t or ""))


def pool(hyps, refs):
    m = CERMetric(); m.update(hyps, refs); return m.compute()["cer"]


def main():
    d = json.load(open(os.path.join(os.path.dirname(__file__), "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    dead = [r for r in d["rows"] if r["sim"] < 0.2]
    n = len(dead)
    print(f"===== 死区 sim<0.2 对抗验证 (n={n}) =====")

    # 口径确认
    print(f"\n[口径] per-row mean: qwen={statistics.mean(r['qwen_cer'] for r in dead):.4f} vanilla={statistics.mean(r['vanilla_cer'] for r in dead):.4f}")
    qh = [sn(r["qwen"]) for r in dead]; vh = [sn(r["vanilla"]) for r in dead]; refs = [r["ref"] for r in dead]
    print(f"[口径] 官方累计池: qwen={pool(qh, refs):.4f} vanilla={pool(vh, refs):.4f}  (handoff 0.459 坐实=官方池)")

    # H2 信号: 完美句占比(死区近 4 成完美=异常高, LM 猜模板嫌疑)
    perfect = [r for r in dead if r["qwen_cer"] < 0.01]
    vperfect = [r for r in dead if r["vanilla_cer"] < 0.01]
    print(f"\n[H2信号] qwen cer<0.01 完美句: {len(perfect)}/{n} = {len(perfect)/n:.1%}")
    print(f"[对照]   vanilla cer<0.01 完美句: {len(vperfect)}/{n} = {len(vperfect)/n:.1%}")

    win = sum(1 for r in dead if r["qwen_cer"] < r["vanilla_cer"] - 0.01)
    loss = sum(1 for r in dead if r["qwen_cer"] > r["vanilla_cer"] + 0.01)
    print(f"[qwen vs vanilla] win={win} loss={loss} tie={n-win-loss}")

    # H2 核心: ref 模板化度(死区 ref 高度重复=LM 易猜)
    ref_c = Counter(r["ref"] for r in dead)
    dup = sum(c for r, c in ref_c.items() if c >= 2)
    print(f"\n[ref 模板化] 死区 {n} 条, 独特 ref {len(ref_c)} 个, 重复>=2次条数 {dup}/{n}={dup/n:.1%}")

    # H1/H2 判别: 完美句中 ref 是高频模板 vs 唯一
    if perfect:
        pf_hf = sum(1 for r in perfect if ref_c[r["ref"]] >= 2)
        print(f"\n[H1/H2 判别] 完美句 {len(perfect)} 条, ref 均长 {statistics.mean(len(r['ref']) for r in perfect):.1f} 字")
        print(f"  完美句 ref 是高频模板(>=2次): {pf_hf}/{len(perfect)} = {pf_hf/len(perfect):.1%}  → H2 嫌疑")
        print(f"  完美句 ref 是唯一(非模板):    {len(perfect)-pf_hf}/{len(perfect)} = {(len(perfect)-pf_hf)/len(perfect):.1%}  → H1 嫌疑(含健康问答等非家居句)")

    # 完美句样本 + 最差样本(看 LM 幻觉编造样态)
    print(f"\n=== 完美句样本(ref vs qwen, 随机15) ===")
    random.seed(1)
    for r in random.sample(perfect, min(15, len(perfect))):
        tag = f"模板{ref_c[r['ref']]}x" if ref_c[r["ref"]] >= 2 else "唯一"
        print(f"  {r['uid']} sim={r['sim']:.3f} [{tag}]\n    ref:  {r['ref']}\n    qwen: {r['qwen']}")
    print(f"\n=== qwen 最差死区10条(LM 幻觉编造样态) ===")
    for r in sorted(dead, key=lambda x: -x["qwen_cer"])[:10]:
        print(f"  {r['uid']} sim={r['sim']:.3f} cer={r['qwen_cer']:.2f}\n    ref:  {r['ref']}\n    qwen: {r['qwen']}")

    print(f"\n[H1/H2 初步结论] 死区 0.459 是 H1+H2 混合, 非纯物理地板:")
    print(f"  H1 证据: {len(perfect)-pf_hf if perfect else 0} 条完美句是非高频模板(含'儿童要少吃什么'等非家居句, LM 纯猜难中)")
    print(f"  H2 证据: {len(perfect)}/{n}={len(perfect)/n:.0%} 完美句占比死区异常高 + 最差条是 LM 编造('邮政银行''刘德华冰雨')")
    print(f"  → spk-oracle-poc '死区物理地板'叙事需修正为'混合桶'; 听音核验(抽 cmd_2091/2992 等)坐实 H1/H2 比例")


if __name__ == "__main__":
    main()
