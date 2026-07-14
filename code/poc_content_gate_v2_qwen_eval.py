"""qwen + content_gate v2 全量确认(qwen 同受益验证)。

qwen text 含阿拉伯数字 0/1350(中文原生) → digit_postproc 时间归一对 qwen no-op。
增量只在 content_gate 拒 qwen 幻觉(qwen 死区 0.459 有幻觉)。

对比(全量 char-weighted, pos 1350):
  S2 qwen thr0.27:        sim<0.27 拒(CER1.0) else cer(qwen_text, ref)   [≈0.5934 官方]
  S4 +content_gate幻觉检测: sim<0.27 拒 OR not is_valid_command(提交归一text) 拒 else S2
hold-out(uid md5 val)验证泛化。
"""
import json, sys, hashlib, unicodedata
sys.path.insert(0, "code")
from text_utils import digit_postproc, to_simplified, is_valid_command

d = json.load(open("code/poc_qwen_asr_full_result.json", encoding="utf-8"))
rows = d["rows"]
THR = 0.27


def normalize(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith("P") and not c.isspace())


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j]+1, dp[j-1]+1, prev+(a[i-1]!=b[j-1])); prev = cur
    return dp[n]


def cer(pred, ref):
    pred, ref = normalize(pred), normalize(ref)
    if not ref: return 0.0 if not pred else 1.0
    return lev(pred, ref) / len(ref)


def run(split=None):
    s = {"S2_qwen_thr027": 0, "S4_+content_gate": 0}
    tot = 0; n_gate_pos = n_gate_neg = 0
    for r in rows:
        uid = r["uid"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2
        if split == "val" and h != 1: continue
        if split == "train" and h != 0: continue
        ref = r["ref"]; qtext = r.get("qwen", "") or ""; sim = r.get("sim", 0)
        rlen = max(len(normalize(ref)), 1)
        text = digit_postproc(to_simplified(qtext))   # 提交归一(qwen 标点保留, normalize 在 cer 去)
        sim_rej = sim < THR
        gate_rej = (not is_valid_command(text)) if text.strip() else True
        cer_q = cer(qtext, ref)
        s["S2_qwen_thr027"] += (1.0 if sim_rej else cer_q) * rlen
        s["S4_+content_gate"] += (1.0 if (sim_rej or gate_rej) else cer_q) * rlen
        if gate_rej and not sim_rej:
            if cer_q > 1: n_gate_pos += 1     # 拒死区幻觉(改善)
            else: n_gate_neg += 1             # 误拒正常(恶化)
        tot += rlen
    return {k: round(v / tot, 4) for k, v in s.items()}, n_gate_pos, n_gate_neg


for split in [None, "val"]:
    s, np_, nn = run(split)
    print(f"=== {'全集' if not split else 'val(hold-out)'} ===")
    print(f"  S2 qwen thr0.27:      {s['S2_qwen_thr027']:.4f}")
    print(f"  S4 +content_gate:     {s['S4_+content_gate']:.4f}  (Δ{s['S4_+content_gate']-s['S2_qwen_thr027']:+.4f})")
    print(f"    gate 拒死区幻觉 {np_} 条 | 误拒正常 {nn} 条")
    print()
