"""
全量 1350 条规则同音字纠错 A/B/C/D/E/组合 实测。
累计池 CER (sum err / sum ref len) = 官方口径 (NFKC + lower + 去 P*/空白)。
保守规则: 只改高置信, 宁可不改不可改坏。

用法:
  cd E:/midea_target_asr/code && uv run python runs/_rule_full.py
"""
import os, sys, json, unicodedata, re
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(_HERE)
DATA = os.path.join(_HERE, 'poc_qwen_asr_full_result.json')

# ---------- 官方口径 ----------
def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P') and not ch.isspace())

def lev(r, h):
    m, n = len(r), len(h)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ri = r[i - 1]
        for j in range(1, n + 1):
            if ri == h[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = min(prev[j], cur[j - 1], prev[j - 1]) + 1
        prev = cur
    return prev[n]

def pool_cer(rows, key):
    err = 0; ln = 0
    for r in rows:
        rn = normalize(r['ref'])
        hn = normalize(r[key])
        err += lev(rn, hn); ln += len(rn)
    return err, ln, (err / ln if ln else 0.0)

# ---------- 从 ref 对齐构建单字替换对 (qwen_char -> ref_char) ----------
def build_pairs(rows):
    pairs = Counter(); ex = defaultdict(list)
    for r in rows:
        rn = normalize(r['ref']); qn = normalize(r['qwen'])
        if len(rn) != len(qn): continue
        diffs = [(qn[i], rn[i]) for i in range(len(rn)) if rn[i] != qn[i]]
        if len(diffs) == 1:
            w, c = diffs[0]
            pairs[(w, c)] += 1
            ex[(w, c)].append((r['uid'], r['ref'], r['qwen']))
    return pairs, ex

def build_vocab(rows, min_freq=3, ns=(2, 3, 4)):
    wf = Counter()
    for r in rows:
        rn = normalize(r['ref'])
        for n in ns:
            for i in range(len(rn) - n + 1):
                wf[rn[i:i + n]] += 1
    return {w for w, c in wf.items() if c >= min_freq}, wf

# ---------- pypinyin (可选, 用于同音判定) ----------
try:
    from pypinyin import lazy_pinyin, Style
    _PIN = True
except Exception:
    _PIN = False

def is_homophone(a, b):
    if a == b: return True
    if not _PIN: return True
    pa = lazy_pinyin(a, style=Style.NORMAL, errors='ignore')
    pb = lazy_pinyin(b, style=Style.NORMAL, errors='ignore')
    if pa and pb and pa[0] == pb[0]:
        return True
    return False

# ================= Scheme A: 朴素单字词典 (freq>=2, 盲替) =================
def make_scheme_A(pairs):
    d = defaultdict(list)
    for (w, c), f in pairs.items():
        if f >= 2:
            d[w].append((c, f))
    for w in d: d[w].sort(key=lambda x: -x[1])
    return dict(d)

def apply_single(text, dct, vocab=None, require_bigram=False, require_homophone=False):
    t = normalize(text)
    out = list(t); ch = []
    for i, c in enumerate(out):
        if c in dct:
            best, bf = None, 0
            left = out[i - 1] if i > 0 else ''
            right = out[i + 1] if i < len(out) - 1 else ''
            for rep, f in dct[c]:
                if require_homophone and not is_homophone(c, rep):
                    continue
                if require_bigram and vocab is not None:
                    ok = ((left + rep) in vocab) or ((rep + right) in vocab)
                    if not ok: continue
                if f > bf:
                    best, bf = rep, f
            if best is not None:
                out[i] = best; ch.append((i, c, best))
    return ''.join(out), ch

# ================= Scheme C: trie 正向模糊匹配 =================
class Trie:
    def __init__(self):
        self.c = {}; self.end = False; self.word = None
    def add(self, w):
        n = self
        for ch in w:
            n = n.c.setdefault(ch, Trie())
        n.end = True; n.word = w

def build_trie(words):
    t = Trie()
    for w in words: t.add(w)
    return t

def _trie_one_diff(root, span):
    """trie 中与 span 恰好 1 位置不同的词, 返回 [(word,pos,trie_char)]."""
    res = []
    L = len(span)
    def walk_match(node, idx):
        # 严格匹配到结尾
        while idx < L:
            ch = span[idx]
            if ch in node.c:
                node = node.c[ch]; idx += 1
            else:
                return None
        return node if node.end else None
    def walk(node, idx, used_diff):
        if idx == L:
            if node.end and used_diff:
                res.append(node.word)
            return
        ch = span[idx]
        # match
        if ch in node.c:
            walk(node.c[ch], idx + 1, used_diff)
        # mismatch (only if not used)
        if not used_diff:
            for br, child in node.c.items():
                if br == ch: continue
                # 1 diff used at pos idx (trie char = br), then strict match rest
                end = walk_match(child, idx + 1)
                if end is not None and end.end:
                    res.append(end.word)
    walk(root, 0, False)
    out = []
    for w in res:
        for k in range(L):
            if w[k] != span[k]:
                out.append((w, k, w[k])); break
    return out

def trie_fuzzy_correct(text, trie, pairs_set, max_len=4):
    t = normalize(text); out = list(t); ch = []
    i = 0
    while i < len(t):
        replaced = False
        for L in range(max_len, 1, -1):
            if i + L > len(out): continue
            span = ''.join(out[i:i + L])
            cand = _trie_one_diff(trie, span)
            if not cand: continue
            best = None
            for word, pos, tc in cand:
                hc = span[pos]
                if (hc, tc) in pairs_set and is_homophone(hc, tc):
                    best = word; break
            if best is None:
                for word, pos, tc in cand:
                    hc = span[pos]
                    if (hc, tc) in pairs_set:
                        best = word; break
            if best is not None:
                for k, cc in enumerate(best):
                    if out[i + k] != cc:
                        ch.append((i + k, out[i + k], cc)); out[i + k] = cc
                i += L; replaced = True; break
        if not replaced:
            i += 1
    return ''.join(out), ch

# ================= Scheme D: 数字归一 (阿拉伯<->中文) =================
AR2CN = {'0':'零','1':'一','2':'二','3':'三','4':'四','5':'五','6':'六','7':'七','8':'八','9':'九'}
def _int2cn(s):
    s = s.lstrip('0') or '0'
    if s == '0': return '零'
    units = ['', '十', '百', '千']
    out = ''; n = len(s)
    for i, c in enumerate(s):
        d = AR2CN[c]; pos = n - 1 - i
        if d == '零': continue
        out += d + (units[pos] if pos < len(units) else '')
    return out
def _ar2cn(s):
    if '.' in s:
        a, b = s.split('.')
        return _int2cn(a) + '点' + ''.join(AR2CN[c] for c in b)
    return _int2cn(s)
def digit_normalize(text):
    t = unicodedata.normalize('NFKC', text)
    return re.sub(r'[0-9]+(\.[0-9]+)?', lambda m: _ar2cn(m.group(0)), t)

# ================= Scheme E: 品牌/功能名 锚点词精确修复 =================
BRAND_WORDS = ['轻干洗','净干洗','净呼吸','智控温','智清洁','防直吹','新风','无风感',
               '星香','一键净呼吸','AI净干洗','自清洁','柔风','速热','电辅热','舒适']

def brand_correct(text, pairs_set):
    t = normalize(text); out = list(t); ch = []
    for W in sorted(BRAND_WORDS, key=len, reverse=True):
        L = len(W)
        i = 0
        while i + L <= len(out):
            span = ''.join(out[i:i + L])
            if span == W:
                i += 1; continue
            diffs = [(k, span[k], W[k]) for k in range(L) if span[k] != W[k]]
            if len(diffs) == 1:
                k, hc, tc = diffs[0]
                if (hc, tc) in pairs_set and is_homophone(hc, tc):
                    out[i + k] = tc
                    ch.append((i + k, hc, tc))
                    i += L; continue
            i += 1
    return ''.join(out), ch

# ---------- 评估 ----------
def evaluate(rows, scheme_fn, name):
    err0, ln, cer0 = pool_cer(rows, 'qwen')
    corr_rows = []
    for r in rows:
        cor, ch = scheme_fn(r['qwen'])
        nr = dict(r); nr['corr'] = cor; nr['changes'] = ch
        corr_rows.append(nr)
    err1, _, cer1 = pool_cer(corr_rows, 'corr')
    imp = wor = unc = 0; worst = []
    for r in corr_rows:
        rn = normalize(r['ref'])
        e0 = lev(rn, normalize(r['qwen'])); e1 = lev(rn, normalize(r['corr']))
        if e1 < e0: imp += 1
        elif e1 > e0:
            wor += 1
            worst.append((e1 - e0, r['uid'], r['ref'], r['qwen'], r['corr'], r['changes']))
        else: unc += 1
    worst.sort(key=lambda x: -x[0])
    delta = cer1 - cer0
    print(f'\n=== {name} ===')
    print(f'  累计池 CER: {cer0:.6f} -> {cer1:.6f}  Δ={delta:+.6f}  (err {err0}->{err1}/{ln})')
    print(f'  改善 {imp} / 恶化 {wor} / 不变 {unc}  (净 {imp-wor})')
    return dict(name=name, cer0=cer0, cer1=cer1, delta=delta, improved=imp, worse=wor,
                unchanged=unc, worst=worst, err0=err0, err1=err1, ln=ln, corr_rows=corr_rows)

def main():
    with open(DATA, encoding='utf-8') as f:
        d = json.load(f)
    rows = d['rows']
    print(f'rows: {len(rows)}  pypinyin: {_PIN}')

    err0, ln, cer0 = pool_cer(rows, 'qwen')
    print(f'baseline qwen 累计池 CER = {cer0:.6f}  (err {err0} / len {ln})')

    pairs, ex = build_pairs(rows)
    pairs_set = set(pairs.keys())
    vocab, wf = build_vocab(rows, min_freq=3)
    print(f'单字对(freq>=2): {sum(1 for _,f in pairs.items() if f>=2)}; vocab(n>=2,f>=3): {len(vocab)}')

    dct_A = make_scheme_A(pairs)

    results = {}
    results['A'] = evaluate(rows, lambda t: apply_single(t, dct_A), 'A_朴素单字词典(freq>=2,盲替)')
    results['B'] = evaluate(rows, lambda t: apply_single(t, dct_A, vocab=vocab, require_bigram=True),
                            'B_bigram约束(替换后成高频家居词)')
    results['B+'] = evaluate(rows, lambda t: apply_single(t, dct_A, vocab=vocab, require_bigram=True, require_homophone=True),
                             'B+_bigram+同音双约束')
    trie = build_trie([w for w, c in wf.items() if c >= 3 and len(w) >= 2])
    results['C'] = evaluate(rows, lambda t: trie_fuzzy_correct(t, trie, pairs_set), 'C_trie词表正向模糊(1-diff+同音)')
    results['D'] = evaluate(rows, lambda t: (digit_normalize(t), []), 'D_数字归一(阿拉伯->中文)')
    results['E'] = evaluate(rows, lambda t: brand_correct(t, pairs_set), 'E_品牌功能名锚点修复')

    def combo_EBp(t):
        a, c1 = brand_correct(t, pairs_set)
        b, c2 = apply_single(a, dct_A, vocab=vocab, require_bigram=True, require_homophone=True)
        return b, c1 + c2
    results['E+B+'] = evaluate(rows, combo_EBp, '组合_E品牌+B+bigram同音单字')

    def combo_EBpD(t):
        a, c1 = brand_correct(t, pairs_set)
        b, c2 = apply_single(a, dct_A, vocab=vocab, require_bigram=True, require_homophone=True)
        cc = digit_normalize(b)
        return cc, c1 + c2
    results['E+B++D'] = evaluate(rows, combo_EBpD, '组合_E+B++D数字')

    print('\n\n========= 全量 1350 汇总 =========')
    print(f'{"方案":<30} {"CER前":>9} {"CER后":>9} {"Δ":>10} {"改":>4} {"恶":>4} {"净":>5}')
    for k, r in results.items():
        print(f'{r["name"]:<30} {r["cer0"]:.6f} {r["cer1"]:.6f} {r["delta"]:+.6f} {r["improved"]:>4} {r["worse"]:>4} {r["improved"]-r["worse"]:+5}')

    for k, r in results.items():
        if not r['worst']: continue
        print(f'\n--- {r["name"]} 改坏 top3 ---')
        for d_, uid, ref, qwen, cor, ch in r['worst'][:3]:
            print(f'  +{d_} {uid} changes={ch}')
            print(f'    ref : {ref[:50]}')
            print(f'    qwen: {qwen[:50]}')
            print(f'    corr: {cor[:50]}')

    best_key = min(results, key=lambda k: results[k]['delta'])
    best = results[best_key]
    print(f'\n=== 最强方案(按Δ): {best["name"]}  Δ={best["delta"]:+.6f}  改{best["improved"]}/恶{best["worse"]} ===')
    # 零回退方案中最强的
    safe = {k: r for k, r in results.items() if r['worse'] == 0 and r['delta'] < 0}
    if safe:
        sb = min(safe, key=lambda k: results[k]['delta'])
        print(f'=== 零回退最强: {results[sb]["name"]}  Δ={results[sb]["delta"]:+.6f}  改{results[sb]["improved"]}/恶{results[sb]["worse"]} ===')
        # 打印其改善案例
        print('   改善案例:')
        for r in results[sb]['corr_rows']:
            rn = normalize(r['ref'])
            if lev(rn, normalize(r['qwen'])) > lev(rn, normalize(r['corr'])):
                print(f'     {r["uid"]} changes={r["changes"]}  qwen={normalize(r["qwen"])[:30]} -> corr={normalize(r["corr"])[:30]}')

    out = {'baseline_cer': cer0, 'baseline_err': err0, 'baseline_len': ln,
           'schemes': {k: {kk: vv for kk, vv in r.items() if kk not in ('worst', 'corr_rows')} | {'worst_top5': r['worst'][:5]}
                       for k, r in results.items()}}
    with open(os.path.join(_HERE, '_rule_full_result.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nsaved -> runs/_rule_full_result.json')

if __name__ == '__main__':
    main()
