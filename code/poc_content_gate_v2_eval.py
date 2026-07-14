"""集成后全量 CER 净效果验证(content_gate v2 + digit_postproc 时间归一)。

对比(全量 char-weighted 累计池, pos 1362):
  S2 当前提交:    sim<0.27 拒(CER1.0) else cer(vanilla_text, ref)        [vanilla thr0.27 ≈0.703]
  S3 +时间归一:   sim<0.27 拒 else cer(digit_postproc(vanilla_text), ref) [digit_postproc 时间增量]
  S4 +content_gate: sim<0.27 拒 OR not is_valid_command(dp_v2) 拒 else S3 [gate 幻觉拒增量]
hold-out(uid md5 train/val)验证泛化。
"""
import json, sys, hashlib, unicodedata
sys.path.insert(0, "code")
from text_utils import digit_postproc, is_valid_command

van = json.load(open("code/exp_vanilla_full.json", encoding="utf-8"))
pos = {f"cmd_{r['id']}": r for r in json.load(open("code/pos_pairs_datasetA.json", encoding="utf-8"))}
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
    """split: None=全集 / 'val'。返回各场景 char-weighted CER。"""
    s = {"S2_thr027": 0, "S3_+dp时间归一": 0, "S4_+content_gate": 0}
    tot = 0
    n_dp_improve = n_gate_reject = 0
    for r in van:
        uid = r["uid"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2
        if split == "val" and h != 1: continue
        if split == "train" and h != 0: continue
        ref = pos.get(uid, {}).get("ref", "")
        text = r.get("vanilla_text", "") or ""
        max_sim = r.get("max_sim", 0)
        rlen = max(len(normalize(ref)), 1)
        sim_reject = max_sim < THR
        text_dp = digit_postproc(text)
        gate_reject = (not is_valid_command(text_dp)) if text_dp.strip() else True
        cer_raw = cer(text, ref)
        cer_dp = cer(text_dp, ref)
        # S2: sim 拒
        s["S2_thr027"] += (1.0 if sim_reject else cer_raw) * rlen
        # S3: +dp 时间归一
        s["S3_+dp时间归一"] += (1.0 if sim_reject else cer_dp) * rlen
        # S4: +content_gate
        s["S4_+content_gate"] += (1.0 if (sim_reject or gate_reject) else cer_dp) * rlen
        if cer_dp < cer_raw - 0.01: n_dp_improve += 1
        if gate_reject and not sim_reject and cer_raw > 1: n_gate_reject += 1
        tot += rlen
    return {k: round(v / tot, 4) for k, v in s.items()}, n_dp_improve, n_gate_reject


for split in [None, "val"]:
    s, n_dp, n_gate = run(split)
    print(f"=== {'全集' if not split else 'val(hold-out)'} ===")
    print(f"  S2 当前提交 thr0.27:        {s['S2_thr027']:.4f}")
    print(f"  S3 +digit_postproc时间归一: {s['S3_+dp时间归一']:.4f}  (Δ{s['S3_+dp时间归一']-s['S2_thr027']:+.4f})")
    print(f"  S4 +content_gate幻觉检测:   {s['S4_+content_gate']:.4f}  (Δ{s['S4_+content_gate']-s['S2_thr027']:+.4f})")
    print(f"  dp 时间归一改善 {n_dp}条 | gate 拒死区幻觉 {n_gate}条")
    print()

json.dump({"note": "集成后 vs 当前提交 thr0.27, char-weighted"}, open("code/content_gate_v2_eval.json", "w", encoding="utf-8"), ensure_ascii=False)
