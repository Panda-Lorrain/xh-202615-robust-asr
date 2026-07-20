"""content_gate joint 收益离线验证(qwen 后端 thr0.27, noSE)。

✅ 状态(2026-07-20): 已据此脚本实测 joint+0.826 修正 memory content-gate-decision
   (反转 BAODI_GATE=1 默认开, run_baodi.sh 同步). 本脚本为一手验证依据保留复现.
   下方"应修正"为脚本写作时措辞, 现已完成.

回应 2026-07-18 三腿盘点发现的 memory 遗漏:
  memory content-gate-decision 否决 qwen 后端用 gate, 但其依据 poc_content_gate_v2_qwen_eval.py
  只评了 pos 侧 CER 代价(Δ+0.024), **neg 侧 RR 收益从未量过**(本机无 qwen neg 明细)。
  vanilla 时代 hold-out(exp_content_gate_holdout.py) pos+neg 都评了 val+1.6 分泛化成立,
  但 qwen 后端 transcribe 更鲁棒, neg 漏拒文本特征可能不同, gate 收益需独立验证。

本脚本: 离线算 gate=off vs gate=on 的 joint(不占 GPU, gate 是纯函数 is_valid_command):
  pos 侧 CER 代价: poc_qwen_asr_full_result.json(1350 条 uid/ref/qwen/sim, 累计池 overall CER)
  neg 侧 RR 收益: out_neg_baodi/result.json(qwen noSE thr0.27 全量, max_sim+transcript)
  joint = ΔRR_leg(+gate 提升 RR) - ΔCER_leg(+gate 恶化 CER), w1=w2=0.4

判定: joint > 0 → memory content-gate-decision 应修正(qwen 主线开 gate); < 0 → 否决成立。

用法:
  code/.venv/Scripts/python.exe code/verify_content_gate_joint.py [--neg-json code/out_neg_baodi/result.json]
"""
import json, sys, os, unicodedata, argparse, hashlib
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import is_valid_command, digit_postproc, to_simplified

THR = 0.27
W = 0.4  # w1=w2=0.4 假设(排名公式)


def normalize(t):
    """官方 CER 预处理口径(NFKC+lower+去P*标点空白)."""
    t = unicodedata.normalize("NFKC", t or "").lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith("P") and not c.isspace())


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[n]


def cer(pred, ref):
    pred, ref = normalize(pred), normalize(ref)
    if not ref: return 0.0 if not pred else 1.0
    return lev(pred, ref) / len(ref)


def gate_reject(sim, text, gate_on):
    """复刻 submit_infer.decide_reject 的 content_gate 通道(sim_only, 关 LLM).
    gate_on=True 时: sim>=thr 且 not is_valid_command(text_norm) → 加拒."""
    if sim < THR:
        return True  # sim 拒(与 gate 无关)
    if gate_on:
        text_norm = digit_postproc(to_simplified(text or ""))
        if not is_valid_command(text_norm):
            return True
    return False


# ============ pos 侧 overall CER(累计池, 官方口径) ============
def pos_overall_cer(pos_path, gate_on):
    d = json.load(open(pos_path, encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    err = ch = 0
    n_gate_add = 0  # gate 在 sim>=thr 基础上额外拒的条数
    for r in rows:
        sim = float(r.get("sim", 0) or 0)
        ref = normalize(r.get("ref", ""))
        if not ref:
            continue
        qtext = r.get("qwen", "") or r.get("qwen_text", "") or r.get("text", "") or ""
        if gate_reject(sim, qtext, gate_on):
            err += len(ref)
            ch += len(ref)
            if gate_on and sim >= THR:
                n_gate_add += 1
        else:
            err += lev(normalize(qtext), ref)
            ch += len(ref)
    return (err / ch if ch else 0), n_gate_add


# ============ neg 侧 RR(句准) ============
def neg_rr(neg_path, gate_on):
    rows = json.load(open(neg_path, encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("results", rows.get("rows", []))
    n_rej = 0
    n_gate_catch = 0  # gate 在 sim>=thr 基础上拒掉的漏拒(sim 过线但非指令)
    leaks_gate_caught = []
    for r in rows:
        sim = float(r.get("max_sim", 0) or 0)
        text = r.get("transcript", "") or r.get("text", "") or ""
        if gate_reject(sim, text, gate_on):
            n_rej += 1
            if gate_on and sim >= THR:
                n_gate_catch += 1
                leaks_gate_caught.append({"uid": r.get("uid", os.path.splitext(os.path.basename(r.get("recognition", "")))[0]),
                                          "sim": round(sim, 3), "text": text[:40]})
    rr = n_rej / max(1, len(rows))
    return rr, n_gate_catch, len(rows), leaks_gate_caught


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-json", default=os.path.join(_HERE, "poc_qwen_asr_full_result.json"))
    ap.add_argument("--neg-json", default=os.path.join(_HERE, "out_neg_baodi", "result.json"),
                    help="qwen noSE thr0.27 neg 全量 result.json(需先 BAODI_BACKEND=qwen run_baodi.sh neg 0.27 跑出)")
    args = ap.parse_args()

    print(f"=== content_gate joint 验证(qwen thr={THR}, w1=w2={W}) ===\n")

    # pos 侧
    cer_off, _ = pos_overall_cer(args.pos_json, gate_on=False)
    cer_on, n_pos_gate_add = pos_overall_cer(args.pos_json, gate_on=True)
    print(f"[pos CER 代价] poc_qwen_asr_full({args.pos_json})")
    print(f"  gate=off overall CER = {cer_off:.4f}  (CER 腿 {(1-cer_off)*W*100/0.4:.2f}/40)")
    print(f"  gate=on  overall CER = {cer_on:.4f}  Δ{cer_on-cer_off:+.4f}  (pos 误拒 +{n_pos_gate_add} 条)")
    print(f"  pos CER 腿变化 = {(cer_off-cer_on)*40:+.3f} 分\n")

    # neg 侧
    if not os.path.exists(args.neg_json):
        print(f"[neg RR 收益] ⚠️ {args.neg_json} 不存在!")
        print(f"  先跑: BAODI_BACKEND=qwen bash code/run_baodi.sh neg {THR}")
        print(f"  (产 out_neg_baodi/result.json, qwen noSE thr={THR} 全量 neg 基线)")
        return
    rr_off, _, n_neg, _ = neg_rr(args.neg_json, gate_on=False)
    rr_on, n_catch, _, caught = neg_rr(args.neg_json, gate_on=True)
    print(f"[neg RR 收益] {args.neg_json} ({n_neg} 条)")
    print(f"  gate=off RR = {rr_off:.4f}  (RR 腿 {rr_off*40:.2f}/40)")
    print(f"  gate=on  RR = {rr_on:.4f}  Δ{rr_on-rr_off:+.4f}  (gate 拒掉 {n_catch} 条漏拒)")
    if caught:
        print(f"  gate 拒掉的漏拒样例(前 10):")
        for c in caught[:10]:
            print(f"    sim={c['sim']:.3f} text={c['text']!r}")
    print(f"  neg RR 腿变化 = {(rr_on-rr_off)*40:+.3f} 分\n")

    # joint
    d_cer_leg = (cer_off - cer_on) * 40      # +gate 让 CER 升 → 腿分降(负)
    d_rr_leg = (rr_on - rr_off) * 40         # +gate 让 RR 升 → 腿分升(正)
    joint = d_rr_leg + d_cer_leg
    print(f"=== JOINT (w1=w2=0.4) ===")
    print(f"  ΔCER 腿 = {d_cer_leg:+.3f}  (gate 误拒 pos 致 CER 恶化)")
    print(f"  ΔRR  腿 = {d_rr_leg:+.3f}  (gate 拒漏拒 neg 提 RR)")
    print(f"  JOINT   = {joint:+.3f} 分")
    verdict = ("✅ 净正 → memory content-gate-decision 应修正, qwen 主线开 gate(BAODI_GATE=1)" if joint > 0
               else "❌ 净负 → memory 否决成立, qwen 主线维持关 gate")
    print(f"  判定: {verdict}")


if __name__ == "__main__":
    main()
