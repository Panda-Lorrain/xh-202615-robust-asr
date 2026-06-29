"""对抗审查核查: langfix边界 vs 锁错人替代解释。
查 se6 的 max_sim/target_idx/speakers/stno_target_ratio, 排除"babble人声噪声→diar误检/声纹锁错→转写干扰人"。
跑完可删。"""
import json, ast
from collections import defaultdict

m = json.load(open('test_wav/dataset/final/final_manifest.json', encoding='utf-8'))
mm = {it['file']: it for it in m['items']}


def bn(f):
    return f.split(chr(92))[-1].split('/')[-1]


def parse(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except Exception:
        return x


d6 = {bn(r['recognition']): r for r in json.load(open('code/enroll_regen_se6.json', encoding='utf-8'))}

print("se6 按 (noise,overlap): n | max_sim均 | sim<0.2数 | speakers分布 | stno_target_ratio均")
for nt in ['white', 'pink', 'babble']:
    for ov in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sub = [f for f in mm if mm[f]['noise_type'] == nt and mm[f]['overlap_ratio'] == ov]
        if not sub:
            continue
        sims = [d6[f].get('max_sim', 0) for f in sub]
        low = sum(1 for s in sims if s < 0.2)
        nspk = defaultdict(int)
        stno = []
        for f in sub:
            spk = parse(d6[f].get('speakers', '[]'))
            nspk[len(spk)] += 1
            stno.append(d6[f].get('stno_target_ratio') or 0)
        print(f"  {nt:7s} ov{ov:.2f}: n={len(sub):2d} sim均={sum(sims)/len(sims):.3f} "
              f"sim<0.2:{low:2d} speakers={dict(nspk)} stno_t均={sum(stno)/len(stno):.2f}")

print("\n=== babble ov0 抽样(锁错人替代解释核查) ===")
n = 0
for f, it in mm.items():
    if it['noise_type'] == 'babble' and it['overlap_ratio'] == 0.0:
        r = d6[f]
        print(f" {f}")
        print(f"   max_sim={r.get('max_sim'):.3f} speakers={r.get('speakers')} "
              f"target_idx={r.get('target_idx')} stno_target={r.get('stno_target_ratio')}")
        print(f"   transcript[{len(r.get('transcript') or '')}]: {(r.get('transcript') or '')[:50]}")
        n += 1
        if n >= 4:
            break

print("\n=== white ov0 抽样(langfix生效基准对照) ===")
n = 0
for f, it in mm.items():
    if it['noise_type'] == 'white' and it['overlap_ratio'] == 0.0:
        r = d6[f]
        print(f" {f}  ref={it['target_ref']}")
        print(f"   max_sim={r.get('max_sim'):.3f} target_idx={r.get('target_idx')} "
              f"| transcript[{len(r.get('transcript') or '')}]: {(r.get('transcript') or '')[:50]}")
        n += 1
        if n >= 3:
            break
