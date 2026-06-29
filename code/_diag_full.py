"""全面诊断: SE post-fix 归因核查。供对抗审查。
回答: langfix生效范围 / se6最优稳健性 / babble崩因 / 5策略对比。跑完可删。"""
import json, sys, re
from collections import defaultdict
sys.path.insert(0, 'code')
from eval_metrics import cer

m = json.load(open('test_wav/dataset/final/final_manifest.json', encoding='utf-8'))
mm = {it['file']: it for it in m['items']}
zh = re.compile(r'[一-鿿]')


def bn(f):
    return f.split(chr(92))[-1].split('/')[-1]


def load(p):
    return {bn(r['recognition']): r for r in json.load(open(p, encoding='utf-8'))}


se0 = load('code/enroll_regen_se0.json')
se6 = load('code/enroll_regen_se6.json')


def cer_of(store, f):
    r = store.get(f)
    if not r:
        return None
    return cer(r.get('transcript') or '', mm.get(f, {}).get('target_ref', ''))


def analyze(store, label):
    cer_bins = {'正确(<0.5)': 0, '错字(0.5-1)': 0, '幻觉(>1)': 0}
    lang = {'中文': 0, '英文/纯非中文': 0, '空': 0}
    chars = []
    for f, r in store.items():
        it = mm.get(f)
        if not it:
            continue
        hyp = r.get('transcript') or ''
        c = cer(hyp, it['target_ref'])
        if c < 0.5:
            cer_bins['正确(<0.5)'] += 1
        elif c <= 1:
            cer_bins['错字(0.5-1)'] += 1
        else:
            cer_bins['幻觉(>1)'] += 1
        if not hyp.strip():
            lang['空'] += 1
        elif zh.search(hyp):
            lang['中文'] += 1
        else:
            lang['英文/纯非中文'] += 1
        chars.append(len(hyp))
    print(f"\n===== {label} =====")
    print(f"  CER构成: {cer_bins}  | 语言: {lang}  | avg chars={sum(chars)/len(chars):.1f}")
    print(f"  按 (noise, overlap) -> 中文数/正确数/avgCER:")
    for nt in ['white', 'pink', 'babble']:
        line = f"    {nt:7s}"
        for ov in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sub = [f for f in store if mm.get(f, {}).get('noise_type') == nt and mm.get(f, {}).get('overlap_ratio') == ov]
            if not sub:
                continue
            nzh = sum(1 for f in sub if zh.search(store[f].get('transcript') or ''))
            nc = sum(1 for f in sub if cer_of(store, f) < 0.5)
            avgc = sum(cer_of(store, f) for f in sub) / len(sub)
            line += f" | ov{ov:.2f}:{nzh}/{nc}/{avgc:.1f}"
        print(line)


analyze(se0, 'SE0 (=0 post-fix)')
analyze(se6, 'SE6 (=6 post-fix)')


def strat_overall(pick):
    cs = []
    for f in mm:
        s = pick(f)
        r = (se0 if s == '0' else se6).get(f)
        if r:
            cs.append(cer(r.get('transcript') or '', mm[f]['target_ref']))
    return sum(cs) / len(cs) if cs else float('nan')


print("\n===== 5 策略 overall CER (post-fix) =====")
print(f"  全se0:                          {strat_overall(lambda f: '0'):.3f}")
print(f"  全se6:                          {strat_overall(lambda f: '6'):.3f}")
print(f"  旧conditional(babble,white→0,pink→6): {strat_overall(lambda f: '6' if mm[f]['noise_type']=='pink' else '0'):.3f}")
print(f"  新conditional(white,pink→6,babble→0): {strat_overall(lambda f: '0' if mm[f]['noise_type']=='babble' else '6'):.3f}")
print(f"  精细(babble ov<=0.25→0,其余→6):       {strat_overall(lambda f: '0' if (mm[f]['noise_type']=='babble' and mm[f]['overlap_ratio']<=0.25) else '6'):.3f}")
