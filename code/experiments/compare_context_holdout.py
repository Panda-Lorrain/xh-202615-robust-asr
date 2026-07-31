"""qwen context 家居引导 hold-out 对比(2026-07-18, P1-③ 验证).

问: Qwen3-ASR transcribe 加 context(家居场景引导, system message) 能否降 CER?
基线: poc_qwen_asr_full_result.json(context="", 同 uid 的 qwen_cer 已归一算)
对比: out_context_on.json(enroll_infer --context "家居..." 跑主战场 30 条)

判定: mean(CER_on) < mean(CER_off) - 0.005 → context 有效(启用); 否则不启用(context="" 默认).
注意 hold-out 30 条主战场样本(sim[0.2,0.4)), 样本小看趋势 + 逐条 win/loss/tie.

用法: code/.venv/Scripts/python.exe code/compare_context_holdout.py
"""
import json, os, sys, unicodedata, statistics
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from eval_metrics import cer_official


def main():
    poc = {r["uid"]: r for r in json.load(open(os.path.join(_HERE, "poc_qwen_asr_full_result.json"), encoding="utf-8"))["rows"]}
    on_path = os.path.join(_HERE, "out_context_on.json")
    if not os.path.exists(on_path):
        print(f"⚠️ {on_path} 不存在, 先跑 enroll_infer --context")
        return
    on_rows = json.load(open(on_path, encoding="utf-8"))

    paired = []
    for r in on_rows:
        uid = os.path.splitext(os.path.basename(r.get("recognition", "")))[0]
        if uid in poc and r.get("transcript") is not None:
            ref = poc[uid]["ref"]
            on_text = r.get("transcript", "") or ""
            off_text = poc[uid].get("qwen", "") or ""
            cer_on = cer_official(on_text, ref)
            cer_off = cer_official(off_text, ref)  # 重算(等价 poc qwen_cer, 归一口径)
            paired.append({"uid": uid, "ref": ref, "off": off_text, "on": on_text,
                           "cer_off": cer_off, "cer_on": cer_on, "sim": poc[uid]["sim"]})

    n = len(paired)
    if n == 0:
        print("无配对样本")
        return
    mean_off = statistics.mean(p["cer_off"] for p in paired)
    mean_on = statistics.mean(p["cer_on"] for p in paired)
    win = sum(1 for p in paired if p["cer_on"] < p["cer_off"] - 0.01)
    loss = sum(1 for p in paired if p["cer_on"] > p["cer_off"] + 0.01)
    tie = n - win - loss

    print(f"=== qwen context 家居引导 hold-out(主战场 sim[0.2,0.4), n={n}) ===")
    print(f"  context='' 基线 CER mean = {mean_off:.4f}")
    print(f"  context=家居  CER mean = {mean_on:.4f}")
    print(f"  Δ = {mean_on-mean_off:+.4f}  (负=context降CER有效)")
    print(f"  逐条: context 更优 {win} / 更差 {loss} / 持平 {tie}")
    print(f"\n  逐条明细(更差的前 8, 看 context 是否引入偏见):")
    for p in sorted(paired, key=lambda x: x["cer_on"]-x["cer_off"], reverse=True)[:8]:
        print(f"    {p['uid']} sim={p['sim']:.3f} CER {p['cer_off']:.2f}→{p['cer_on']:.2f}")
        print(f"      ref:  {p['ref'][:40]}")
        print(f"      off:  {p['off'][:40]}")
        print(f"      on:   {p['on'][:40]}")
    print(f"\n  context 更优的前 5:")
    for p in sorted(paired, key=lambda x: x["cer_on"]-x["cer_off"])[:5]:
        print(f"    {p['uid']} CER {p['cer_off']:.2f}→{p['cer_on']:.2f}  off:{p['off'][:30]!r} → on:{p['on'][:30]!r}")

    verdict = ("✅ 有效 → 启用 context(改 enroll_infer/submit_infer 默认或 BAODI_GATE 式 env)" if mean_on < mean_off - 0.005
               else ("⚠️ 边际(噪声内) → 不启用" if abs(mean_on-mean_off) <= 0.005 else "❌ 恶化 → 不启用(context 引入偏见)"))
    print(f"\n  判定: {verdict}")


if __name__ == "__main__":
    main()
