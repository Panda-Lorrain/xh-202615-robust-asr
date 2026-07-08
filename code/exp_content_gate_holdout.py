"""content_gate hold-out 泛化验证（2026-07-08）。

回应过拟合担忧（用户 2026-07-08）：PoC 的 +2.3 是 in-sample 上界（黑名单词看全 A 集漏拒 neg 定的）。
本脚本：规则用强先验版（拒纯非中文/英文为主/通用非家居类目词/超长），唯一在 train 半调的参数是
len 阈值 L；val 半用 train 选的 L 报泛化分数 + bootstrap CI。

判定：val ΔTotalScore > +0.005（即 +0.5 分，80 满分）→ 证泛化进阶段 2；崩则放弃。
"""
import json, sys, os, hashlib, editdistance, statistics, random
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import to_simplified, digit_postproc
from eval_metrics import normalize_text


def submit_norm(t):
    return digit_postproc(to_simplified(t or ""))


def nfk(t):
    return normalize_text(t or "")


# 通用非家居类目词（先验，非 A 集拟合）：新闻/财经/体育/娱乐/公司 —— 家居指令不会出现
NEWS_BLACK = [
    # 新闻/财经（简）
    "产业", "资本", "投资", "制度", "政府", "债务", "市场", "调研", "报告", "期货", "股票",
    "基金", "贷款", "住房", "房地产", "报道", "新闻", "记者", "日前", "发布", "价格", "广告", "拍摄",
    # 新闻/财经（繁）
    "期貨", "報告", "市場", "調研", "調查", "顯示", "股份", "有限公司", "落戶", "服務",
    # 体育/娱乐/其他
    "四強", "席位", "聚杯", "婚姻", "导演", "考试", "生意", "无法阻挡",
]


def nch_zh(text):
    return sum(1 for c in text if "一" <= c <= "鿿")


def is_valid_command(text, len_thr=20):
    """先验版：默认 True 保留，强非指令信号才 False。len_thr 是唯一待调参数。"""
    if not text or not text.strip():
        return False
    nch = nch_zh(text)
    if nch == 0:                                    # 纯非中文(ok/tooling)
        return False
    if len(text) >= 3 and nch < len(text) * 0.5:   # 英文为主(productive/i can't go)
        return False
    if any(w in text for w in NEWS_BLACK):          # 通用非家居类目词
        return False
    if len(text) > len_thr:                         # 超长叙述(len_thr 在 train 调)
        return False
    return True


THR = 0.27


def uid_of(r):
    return str(r.get("uid") or r.get("id") or r.get("utt_id")
               or os.path.splitext(os.path.basename(r.get("recognition", "")))[0])


def split_train_val(rows, seed=42):
    """按 uid md5 hash 固定分两半（确定性，与样本顺序无关）。"""
    train, val = [], []
    for r in rows:
        h = int(hashlib.md5(uid_of(r).encode()).hexdigest(), 16)
        (train if h % 2 == 0 else val).append(r)
    return train, val


def pos_pool_cer(pos_rows, len_thr, with_gate):
    """pos 官方累计池 CER：thr 拒/gate 拒 → CER=1.0（errors+=len(ref), chars+=len(ref)）。"""
    err = ch = 0
    for x in pos_rows:
        sim = float(x.get("max_sim", 0) or 0)
        ref = nfk(submit_norm(x.get("ref", "")))
        if not ref:
            continue
        if sim < THR:                                   # thr 拒
            err += len(ref); ch += len(ref)
        else:
            text = submit_norm(x.get("vanilla_text", ""))
            if with_gate and not is_valid_command(text, len_thr):  # gate 拒
                err += len(ref); ch += len(ref)
            else:
                err += editdistance.eval(nfk(text), ref); ch += len(ref)
    return err / ch if ch else 0


def neg_rr(neg_rows, len_thr, with_gate):
    """RR = 正确拒 / 总 neg。拒 = sim<thr 或 (sim≥thr 且 gate 拒)。"""
    n_rej = 0
    for r in neg_rows:
        sim = float(r.get("max_sim", 0) or 0)
        if sim < THR:
            n_rej += 1
        elif with_gate and not is_valid_command(r.get("text", "") or "", len_thr):
            n_rej += 1
    return n_rej / max(1, len(neg_rows))


def total_score(pos_rows, neg_rows, len_thr, with_gate, w=0.4):
    cer = pos_pool_cer(pos_rows, len_thr, with_gate)
    rr = neg_rr(neg_rows, len_thr, with_gate)
    return w * (1 - cer) + w * rr, cer, rr


def main():
    pos = json.load(open(os.path.join(_HERE, "exp_vanilla_full.json"), encoding="utf-8"))
    if isinstance(pos, dict):
        pos = pos.get("results", pos.get("rows", []))
    pos = [x for x in pos if "max_sim" in x]
    neg = json.load(open(os.path.join(_HERE, "out_neg_full", "result.json"), encoding="utf-8"))
    neg_rows = neg.get("results", neg) if isinstance(neg, dict) else neg

    pos_tr, pos_val = split_train_val(pos)
    neg_tr, neg_val = split_train_val(neg_rows)
    print(f"分割: pos train={len(pos_tr)} val={len(pos_val)} | neg train={len(neg_tr)} val={len(neg_val)}")

    # --- train 半扫 len_thr，选 train +gate TotalScore 最优（唯一 train 拟合参数）---
    print("\n=== train 半扫 len_thr（选最优 L）===")
    best_L, best_ts = 20, -1
    for L in [15, 18, 20, 22, 25, 30]:
        ts_now, _, _ = total_score(pos_tr, neg_tr, L, False)
        ts_gate, cer_g, rr_g = total_score(pos_tr, neg_tr, L, True)
        if ts_gate > best_ts:
            best_ts, best_L = ts_gate, L
        print(f"  L={L:>2}: train now={ts_now:.4f} +gate={ts_gate:.4f} "
              f"(Δ{ts_gate-ts_now:+.4f}, CER={cer_g:.3f} RR={rr_g:.3f})")
    print(f"→ train 选 len_thr={best_L} (train +gate TS={best_ts:.4f})")

    # --- val 半用 best_L 报泛化分数 ---
    print(f"\n=== val 半泛化分数（len_thr={best_L}，规则未看 val）===")
    ts_now, cer_now, rr_now = total_score(pos_val, neg_val, best_L, False)
    ts_gate, cer_gate, rr_gate = total_score(pos_val, neg_val, best_L, True)
    dval = ts_gate - ts_now
    print(f"  现状:  pos_CER={cer_now:.4f} RR={rr_now:.4f} TS={ts_now:.4f}")
    print(f"  +gate: pos_CER={cer_gate:.4f} RR={rr_gate:.4f} TS={ts_gate:.4f}")
    print(f"  ΔTS={dval:+.4f} ({dval/0.8*100:+.2f} 分/80满分) | ΔCER={cer_gate-cer_now:+.4f} ΔRR={rr_gate-rr_now:+.4f}")

    # --- bootstrap CI on val ΔTS（样本少，报方差）---
    rng = random.Random(42)
    B = 400
    deltas = []
    for _ in range(B):
        sp = [rng.choice(pos_val) for _ in pos_val] if pos_val else []
        sn = [rng.choice(neg_val) for _ in neg_val] if neg_val else []
        tn, _, _ = total_score(sp, sn, best_L, False)
        tg, _, _ = total_score(sp, sn, best_L, True)
        deltas.append(tg - tn)
    deltas.sort()
    print(f"  bootstrap(B={B}) ΔTS: mean={statistics.mean(deltas):+.4f} "
          f"p5={deltas[B // 20]:+.4f} p95={deltas[B * 19 // 20]:+.4f}")

    # --- val 多 L 稳定性（看 len_thr 不敏感性，定集成用哪个 L）---
    print(f"\n=== val 多 L 稳定性扫描（定集成 len_thr）===")
    for L in [18, 20, 22, 25, 30]:
        tn, cn, rn = total_score(pos_val, neg_val, L, False)
        tg, cg, rg = total_score(pos_val, neg_val, L, True)
        print(f"  L={L:>2}: val ΔTS={tg-tn:+.4f} (CER {cn:.3f}→{cg:.3f} {cg-cn:+.3f}, "
              f"RR {rn:.3f}→{rg:.3f} {rg-rn:+.3f})")

    # --- 判定 ---
    verdict = ("✅ 证泛化 → 进阶段2" if dval > 0.005
               else ("⚠️ 边界（微赚，谨慎）" if dval > 0 else "❌ 崩 → 放弃"))
    print(f"\n判定（val ΔTS > +0.005 即 +0.5分门槛）: {verdict}")

    # --- 对比：全 A 集（看与 PoC +2.3 的差距，口径差异说明）---
    ts_a_now, _, _ = total_score(pos, neg_rows, best_L, False)
    ts_a_gate, _, _ = total_score(pos, neg_rows, best_L, True)
    print(f"\n对比 全A集(len_thr={best_L}): ΔTS={ts_a_gate-ts_a_now:+.4f} "
          f"(PoC in-sample 报 +0.0256; 差距=hold-out 挤出的过拟合水分)")

    # --- pos 误拒 spot check（val 半，看被 gate 拒的 pos 原 CER）---
    pos_val_gate_rej = [x for x in pos_val
                        if float(x.get("max_sim", 0) or 0) >= THR
                        and not is_valid_command(submit_norm(x.get("vanilla_text", "")), best_L)]
    if pos_val_gate_rej:
        cers = []
        for x in pos_val_gate_rej:
            ref = nfk(submit_norm(x.get("ref", "")))
            cers.append(editdistance.eval(nfk(submit_norm(x.get("vanilla_text", ""))), ref) / len(ref)
                        if ref else 0)
        ge1 = sum(1 for c in cers if c >= 1.0)
        print(f"\npos val 被 gate 拒: {len(pos_val_gate_rej)} 条, 原 CER mean={sum(cers)/len(cers):.3f} "
              f"(CER≥1 占 {ge1}/{len(pos_val_gate_rej)}={ge1/len(cers):.0%} 反赚; <1 的={len(cers)-ge1} 需 spot check)")


if __name__ == "__main__":
    main()
