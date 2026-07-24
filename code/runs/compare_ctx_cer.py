#!/usr/bin/env python
"""对比 baseline qwen vs context引导 qwen 的 CER (累计池官方口径)。

ref + baseline qwen 从 poc_qwen_asr_full_result.json (含 sim/bucket 用于分桶);
context 候选为 --ctx 指定的 uid→text json (qwen_asr_backend.py 输出,可多个)。
取 pos uid 交集 (poc full 的 1350 条), 算累计池 CER + 分桶 + 改善/恶化 + 恶化案例。

用法:
  cd E:/midea_target_asr/code
  uv run python runs/compare_ctx_cer.py \
      --ctx runs/_qwen_ctx_scene.json --label scene \
      --ctx runs/_qwen_ctx_vocab.json --label vocab
"""
import json, argparse, unicodedata, os, re
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))


def to_posuid(k):
    """切片文件名→pos uid: utt{N}_cmd_{M}→cmd_{M}, 纯 cmd_{M} 不变(兼容 target_slices_full/qwen 两目录)。"""
    return re.sub(r'^utt\d+_', '', k)


def normalize(t):
    if t is None:
        t = ""
    t = unicodedata.normalize('NFKC', str(t)).lower().strip()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P') and not c.isspace())


def cer_dp(rn, hn):
    """单条 edit distance (normalize 后字符级)。"""
    m, n = len(rn), len(hn)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ri = rn[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == hn[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def pooled(refs, hyps):
    err, ln = 0, 0
    for r, h in zip(refs, hyps):
        rn, hn = normalize(r), normalize(h)
        err += cer_dp(rn, hn)
        ln += len(rn)
    return (err / ln if ln else 0.0, err, ln)


def bucket(sim):
    if sim is None:
        return 'unknown'
    if sim < 0.2:
        return '<0.2 死区'
    if sim < 0.4:
        return '[0.2,0.4) 主战场'
    return '>=0.4 接近解决'


def bucket_cers(uids, rows, hyps):
    bb = defaultdict(lambda: [0, 0])
    for u, h in zip(uids, hyps):
        rn = normalize(rows[u]['ref'])
        hn = normalize(h)
        e = cer_dp(rn, hn)
        b = bucket(rows[u].get('sim'))
        bb[b][0] += e
        bb[b][1] += len(rn)
    return bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', default=os.path.join(_HERE, 'poc_qwen_asr_full_result.json'))
    ap.add_argument('--ctx', action='append', default=[], help='uid→text json, 可多次')
    ap.add_argument('--label', action='append', default=[], help='对应 --ctx 标签')
    ap.add_argument('--baseline-ctx', default=None, help='baseline json(context=""); 不指定则用 poc qwen 字段')
    args = ap.parse_args()

    data = json.load(open(args.ref, encoding='utf-8'))
    rows = {r['uid']: r for r in data['rows']}
    uids = list(rows.keys())
    refs = [rows[u]['ref'] for u in uids]
    if args.baseline_ctx:
        bctx = {to_posuid(k): v for k, v in json.load(open(args.baseline_ctx, encoding='utf-8')).items()}
        base_hyps = [bctx.get(u, rows[u].get('qwen', '')) for u in uids]
        print(f"baseline: {args.baseline_ctx}")
    else:
        base_hyps = [rows[u].get('qwen', '') for u in uids]

    base_cer, base_err, base_len = pooled(refs, base_hyps)
    print(f"baseline qwen: CER={base_cer:.4f} (err={base_err} len={base_len} n={len(uids)})")
    base_bb = bucket_cers(uids, rows, base_hyps)
    print("baseline 分桶:")
    for b in ['<0.2 死区', '[0.2,0.4) 主战场', '>=0.4 接近解决', 'unknown']:
        if b in base_bb:
            e, l = base_bb[b]
            print(f"  {b}: CER={e / l:.4f}")

    labels = args.label if args.label else [f"ctx{i}" for i in range(len(args.ctx))]
    summary = {'baseline_cer': base_cer, 'n': len(uids), 'ctx': {}}
    for path, label in zip(args.ctx, labels):
        ctx = {to_posuid(k): v for k, v in json.load(open(path, encoding='utf-8')).items()}
        missing = [u for u in uids if u not in ctx]
        if missing:
            print(f"  [warn] {label}: {len(missing)} uid 缺失(用baseline填)")
        hyps = [ctx.get(u, rows[u].get('qwen', '')) for u in uids]
        c, e, l = pooled(refs, hyps)

        better = worse = same = 0
        per = []
        for u, r, bh, ch in zip(uids, refs, base_hyps, hyps):
            rn = normalize(r)
            bc = cer_dp(rn, normalize(bh))
            cc = cer_dp(rn, normalize(ch))
            d = cc - bc
            if d < 0:
                better += 1
            elif d > 0:
                worse += 1
            else:
                same += 1
            per.append((u, bc, cc, d))

        bb = bucket_cers(uids, rows, hyps)
        print(f"\n[{label}] {path}")
        print(f"  CER={c:.4f} Δ={c - base_cer:+.4f} ({(c - base_cer) / base_cer * 100:+.1f}%) n={len(uids)}")
        print(f"  改善{better} 恶化{worse} 不变{same}")
        print("  分桶(Δ vs base):")
        for b in ['<0.2 死区', '[0.2,0.4) 主战场', '>=0.4 接近解决', 'unknown']:
            if b in bb and b in base_bb:
                e2, l2 = bb[b]
                eb, lb = base_bb[b]
                print(f"    {b}: CER={e2 / l2:.4f} (base {eb / lb:.4f} Δ={e2 / l2 - eb / lb:+.4f})")
        per.sort(key=lambda x: -x[3])
        print("  恶化top5:")
        for u, bc, cc, d in per[:5]:
            print(f"    {u} {bc:.3f}→{cc:.3f} ref={rows[u]['ref'][:38]} | base={rows[u].get('qwen','')[:38]} | ctx={ctx.get(u,'')[:38]}")
        per.sort(key=lambda x: x[3])
        print("  改善top5:")
        for u, bc, cc, d in per[:5]:
            print(f"    {u} {bc:.3f}→{cc:.3f} ref={rows[u]['ref'][:38]} | base={rows[u].get('qwen','')[:38]} | ctx={ctx.get(u,'')[:38]}")

        summary['ctx'][label] = {'cer': c, 'delta': c - base_cer,
                                 'better': better, 'worse': worse, 'same': same,
                                 'missing': len(missing)}

    out = os.path.join(_HERE, '_ctx_compare_result.json')
    json.dump(summary, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n汇总存 {out}")


if __name__ == '__main__':
    main()
