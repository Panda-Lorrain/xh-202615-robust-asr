"""BoH(bag-of-hallucinations) + delooping 拒识 hold-out 验证(2026-07-10 POC, RR 腿)。

目标: 抓 babble 循环幻觉漏拒 neg + 顺带拒 pos 死区循环幻觉(降 CER)。
集成方式: 叠加在 content_gate 之上的独立加拒通道(只在 sim≥thr 的 accept 后门控),
不改 sim/llm/content_gate 逻辑。复用 exp_content_gate_holdout 的 hold-out 框架
(uid md5 hash split + bootstrap CI + 多 seed)。

判定口径(与 content_gate 一致):
  pos CER 累计池: thr 拒 / gate 拒 / boh 拒 → CER=1.0(err+=len(ref), ch+=len(ref))。
  neg RR: 拒 = sim<thr 或 (sim≥thr 且 gate 拒) 或 (sim≥thr 且 gate 放行 且 boh 拒)。

诚实结论(本脚本输出): NO-GO 作主 gate —— delooping 抓的 7 条极端 CER pos 全已被
content_gate(len>22 / 非中文)捕获, 叠加之上 pos ΔCER≈0 / neg ΔRR≈0。保留价值仅
defense-in-depth(content_gate len_thr 放宽时的 0-FP 安全网)。详见函数 docstring。
"""
import json, sys, os, hashlib, editdistance, statistics, random
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from text_utils import to_simplified, digit_postproc, is_valid_command, bag_of_hallucinations_reject
from eval_metrics import normalize_text


def submit_norm(t):
    return digit_postproc(to_simplified(t or ""))


def nfk(t):
    return normalize_text(t or "")


THR = 0.27
LEN_THR = 22  # content_gate 集成默认(val 占优甜点近先验, 与 exp_content_gate_holdout 一致)


def uid_of(r):
    return str(r.get("uid") or r.get("utt_id") or r.get("id")
               or os.path.splitext(os.path.basename(r.get("recognition", "")))[0])


def split_train_val(rows, seed=42):
    """按 uid md5 hash + seed salt 分两半(与 exp_content_gate_holdout 完全一致)。"""
    train, val = [], []
    for r in rows:
        h = int(hashlib.md5((uid_of(r) + f"|seed={seed}").encode()).hexdigest(), 16)
        (train if h % 2 == 0 else val).append(r)
    return train, val


def _gate_stack(text, with_content_gate, with_boh, len_thr=LEN_THR):
    """三层门控顺序: content_gate 先, 放行则 boh 再判。返回 True=放行, False=拒。"""
    if with_content_gate and not is_valid_command(text, len_thr):
        return False
    if with_boh and bag_of_hallucinations_reject(text):
        return False
    return True


def pos_pool_cer(pos_rows, with_content_gate, with_boh, len_thr=LEN_THR):
    """pos 官方累计池 CER。"""
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
            if not _gate_stack(text, with_content_gate, with_boh, len_thr):
                err += len(ref); ch += len(ref)          # gate/boh 拒 → CER=1.0
            else:
                err += editdistance.eval(nfk(text), ref); ch += len(ref)
    return err / ch if ch else 0


def neg_rr(neg_rows, with_content_gate, with_boh, len_thr=LEN_THR):
    """neg RR。拒 = sim<thr 或 (sim≥thr 且 门控拒)。"""
    n_rej = 0
    for r in neg_rows:
        sim = float(r.get("max_sim", 0) or 0)
        if sim < THR:
            n_rej += 1
        elif not _gate_stack(submit_norm(r.get("text", "")), with_content_gate, with_boh, len_thr):
            n_rej += 1
    return n_rej / max(1, len(neg_rows))


def total_score(pos_rows, neg_rows, with_content_gate, with_boh, w=0.4):
    cer = pos_pool_cer(pos_rows, with_content_gate, with_boh)
    rr = neg_rr(neg_rows, with_content_gate, with_boh)
    return w * (1 - cer) + w * rr, cer, rr


def _spot_pos_rejected_by_boh_only(pos_rows):
    """被 boh 拒 但 content_gate 放行的 pos(即 boh 的边际拒, 看原 CER 代价)。"""
    out = []
    for x in pos_rows:
        if float(x.get("max_sim", 0) or 0) < THR:
            continue
        text = submit_norm(x.get("vanilla_text", ""))
        if not is_valid_command(text, LEN_THR):
            continue  # content_gate 已拒, 不算 boh 边际
        if bag_of_hallucinations_reject(text):
            ref = nfk(submit_norm(x.get("ref", "")))
            cer0 = editdistance.eval(nfk(text), ref) / len(ref) if ref else 0
            out.append((x.get("uid"), text, cer0))
    return out


def _spot_neg_caught_by_boh_only(neg_rows):
    """被 boh 拒 但 content_gate 放行的 neg(boh 边际抓回的漏拒)。"""
    out = []
    for r in neg_rows:
        if float(r.get("max_sim", 0) or 0) < THR:
            continue
        text = submit_norm(r.get("text", ""))
        if not is_valid_command(text, LEN_THR):
            continue
        if bag_of_hallucinations_reject(text):
            out.append((uid_of(r), text))
    return out


def main():
    pos = json.load(open(os.path.join(_HERE, "exp_vanilla_full.json"), encoding="utf-8"))
    pos = pos if isinstance(pos, list) else pos.get("results", pos.get("rows", []))
    pos = [x for x in pos if "max_sim" in x]
    neg = json.load(open(os.path.join(_HERE, "out_neg_vanilla_full", "result.json"), encoding="utf-8"))
    neg_rows = neg.get("results", neg) if isinstance(neg, dict) else neg

    print("=" * 72)
    print("BoH + delooping hold-out 验证 (叠加 content_gate 之上)")
    print("=" * 72)

    # --- A. 检测器本身是否工作: 全量上 delooping/BoH 各抓多少 ---
    print("\n[A] 检测器命中盘点(全量, 不分 train/val)")
    pos_loop = [x for x in pos if bag_of_hallucinations_reject(submit_norm(x.get("vanilla_text", "")))]
    neg_loop = [r for r in neg_rows if bag_of_hallucinations_reject(submit_norm(r.get("text", "")))]
    cers = [x["vanilla_cer"] for x in pos_loop]
    print(f"  pos 命中: {len(pos_loop)}/{len(pos)}"
          + (f"  CER min={min(cers):.2f} max={max(cers):.2f} mean={statistics.mean(cers):.2f}"
             f" (CER>=1: {sum(1 for c in cers if c >= 1)})" if cers else ""))
    print(f"  neg 命中: {len(neg_loop)}/{len(neg_rows)}")
    # 正确 pos 上的 FP
    correct = [x for x in pos if x.get("vanilla_cer", 1) < 0.5]
    fp = [x for x in correct if bag_of_hallucinations_reject(submit_norm(x.get("vanilla_text", "")))]
    print(f"  FP 检查: 正确 pos(CER<0.5, n={len(correct)}) 命中 {len(fp)} (期望 0)")

    # --- B. 与 content_gate 的重叠: boh 命中里多少已被 content_gate 拒 ---
    print("\n[B] 与 content_gate 重叠(关键: 决定边际收益)")
    boh_only_pos = _spot_pos_rejected_by_boh_only(pos)
    boh_only_neg = _spot_neg_caught_by_boh_only(neg_rows)
    overlap_pos = len(pos_loop) - len(boh_only_pos)
    print(f"  pos: boh 命中 {len(pos_loop)} = content_gate 已拒 {overlap_pos} + boh 边际新增 {len(boh_only_pos)}")
    print(f"  neg: boh 命中 {len(neg_loop)} = content_gate 已拒 {len(neg_loop) - len(boh_only_neg)} + boh 边际新增 {len(boh_only_neg)}")
    if boh_only_pos:
        print(f"  [boh 边际拒 pos] (代价: 这些 pos 原 CER 若 >1 则拒反赚, <1 则误拒)")
        for uid, t, c in boh_only_pos[:20]:
            print(f"    cer0={c:.2f} | {t[:50]!r}")
    if boh_only_neg:
        print(f"  [boh 边际抓 neg]")
        for uid, t in boh_only_neg[:20]:
            print(f"    | {t[:50]!r}")

    # --- C. hold-out: train 定(此处无参数可调, delooping 无参 + BoH 是先验), val 报泛化分 ---
    pos_tr, pos_val = split_train_val(pos)
    neg_tr, neg_val = split_train_val(neg_rows)
    print(f"\n[C] hold-out 分割: pos train={len(pos_tr)} val={len(pos_val)} | "
          f"neg train={len(neg_tr)} val={len(neg_val)}")

    print("\n=== val 半三档对比(现状 / +content_gate / +content_gate+boh) ===")
    ts0, c0, r0 = total_score(pos_val, neg_val, False, False)
    ts1, c1, r1 = total_score(pos_val, neg_val, True, False)
    ts2, c2, r2 = total_score(pos_val, neg_val, True, True)
    print(f"  现状(thr only):        CER={c0:.4f} RR={r0:.4f} TS={ts0:.4f}")
    print(f"  +content_gate:         CER={c1:.4f} RR={r1:.4f} TS={ts1:.4f}  (ΔTS={ts1-ts0:+.4f})")
    print(f"  +content_gate+boh:     CER={c2:.4f} RR={r2:.4f} TS={ts2:.4f}  "
          f"(vs +gate ΔTS={ts2-ts1:+.4f}, ΔCER={c2-c1:+.4f}, ΔRR={r2-r1:+.4f})")
    dval = ts2 - ts1

    # --- D. bootstrap CI on val ΔTS(+boh on top of +gate) ---
    rng = random.Random(42)
    B = 400
    deltas = []
    for _ in range(B):
        sp = [rng.choice(pos_val) for _ in pos_val] if pos_val else []
        sn = [rng.choice(neg_val) for _ in neg_val] if neg_val else []
        _, _, _ = total_score(sp, sn, True, False)
        tn, _, _ = total_score(sp, sn, True, False)
        tg, _, _ = total_score(sp, sn, True, True)
        deltas.append(tg - tn)
    deltas.sort()
    print(f"\n  bootstrap(B={B}) +boh ΔTS(on top of +gate): "
          f"mean={statistics.mean(deltas):+.4f} p5={deltas[B // 20]:+.4f} p95={deltas[B * 19 // 20]:+.4f}")

    # --- E. 多 seed hold-out 鲁棒性 ---
    print(f"\n=== 多 seed hold-out 10 划分 val ΔTS(+boh on top of +gate) ===")
    deltas_ms = []
    for sd in range(10):
        _ptr, pvl = split_train_val(pos, sd)
        _ntr, nvl = split_train_val(neg_rows, sd)
        _tn, _, _ = total_score(pvl, nvl, True, False)
        _tg, _, _ = total_score(pvl, nvl, True, True)
        deltas_ms.append(_tg - _tn)
    n_pos = sum(1 for d in deltas_ms if d > 0)
    nz = sum(1 for d in deltas_ms if abs(d) < 1e-9)
    print(f"  10 seed val ΔTS: min={min(deltas_ms):+.4f} max={max(deltas_ms):+.4f} "
          f"mean={sum(deltas_ms)/len(deltas_ms):+.4f} | 正:{n_pos} 零:{nz} 负:{10-n_pos-nz}")

    # --- F. 全 A 集三档(看整体口径) ---
    print(f"\n=== 全 A 集三档 ===")
    ta0, ca0, ra0 = total_score(pos, neg_rows, False, False)
    ta1, ca1, ra1 = total_score(pos, neg_rows, True, False)
    ta2, ca2, ra2 = total_score(pos, neg_rows, True, True)
    print(f"  现状:      CER={ca0:.4f} RR={ra0:.4f} TS={ta0:.4f}")
    print(f"  +gate:     CER={ca1:.4f} RR={ra1:.4f} TS={ta1:.4f} (ΔTS={ta1-ta0:+.4f})")
    print(f"  +gate+boh: CER={ca2:.4f} RR={ra2:.4f} TS={ta2:.4f} (ΔTS={ta2-ta1:+.4f})")

    # --- G. 判定 ---
    print("\n" + "=" * 72)
    if dval > 0.005:
        verdict = "✅ 证泛化 → 值得集成"
    elif dval > 0:
        verdict = "⚠️ 微赚(边界, 谨慎)"
    elif abs(dval) < 1e-9:
        verdict = "❌ NO-GO(与 content_gate 完全冗余, 边际=0)"
    else:
        verdict = "❌ NO-GO(负收益, 损 pos)"
    print(f"判定(val ΔTS(+boh on +gate) > +0.005): {verdict}")
    print("集成建议: 默认关(与 content_gate 冗余); 仅在 content_gate len_thr 放宽时作 0-FP 安全网开。")
    print("=" * 72)


if __name__ == "__main__":
    main()
