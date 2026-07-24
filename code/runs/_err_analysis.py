# -*- coding: utf-8 -*-
"""
Qwen3-ASR 转写错误分析 (官方累计池口径)
- normalize: NFKC + lower + 去所有 P* 标点和空白
- CER 累计池 = sum(err)/sum(len(ref))
- 对非满分行做错误分类,量化 rule/prompt/beam 理论可救上限
输出: runs/_err_analysis_result.json + 控制台摘要
"""
import json, unicodedata, re, sys
from collections import Counter, defaultdict

import editdistance
from pypinyin import lazy_pinyin, Style

DATA = "E:/midea_target_asr/code/runs/poc_qwen_asr_full_result.json"
OUT  = "E:/midea_target_asr/code/runs/_err_analysis_result.json"

# ---------- 官方 normalize ----------
def normalize(t):
    t = unicodedata.normalize('NFKC', t).lower()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P') and not c.isspace())

# ---------- pinyin 缓存 ----------
_pn_cache = {}
def pinyin_no_tone(ch):
    if ch in _pn_cache: return _pn_cache[ch]
    try:
        p = lazy_pinyin(ch, style=Style.NORMAL, errors='default')
        r = (p[0] if p else ch) or ch
    except Exception:
        r = ch
    _pn_cache[ch] = r
    return r
_pt_cache = {}
def pinyin_tone(ch):
    if ch in _pt_cache: return _pt_cache[ch]
    try:
        p = lazy_pinyin(ch, style=Style.TONE3, errors='default', neutral_tone_with_five=True)
        r = (p[0] if p else ch) or ch
    except Exception:
        r = ch
    _pt_cache[ch] = r
    return r

# ---------- 字符类别 ----------
ARABIC = set('0123456789')
CNUM   = set('一二三四五六七八九十百千万零两壹贰叁肆伍陆柒捌玖拾佰仟')
ASCII_LET = set('abcdefghijklmnopqrstuvwxyz')
# 美的/家居品牌词与功能名
BRAND_TERMS = ['智控温','轻干洗','净呼吸','防直吹','AI净','智清洁','自清洁','ECO','colmo','COLMO',
               '美的','美地','MIDEA','midea','AI','AI自','墅智','数智','星香','新风','无风感',
               '热启动','一键',' eco ']
# 近音混淆初始/韵母对(宽松)
def similar_pinyin(pa, pb):
    """拼音相近: 完全相同 / 只差声调 / 声母或韵母一步差(zh<->z,sh<->s,n<->l,in<->ing,en<->eng等)"""
    if not pa or not pb: return False
    if pa == pb: return True
    # 声调去掉后比
    if re.sub(r'[1-5]','',pa) == re.sub(r'[1-5]','',pb): return True
    a = re.sub(r'[1-5]','',pa); b = re.sub(r'[1-5]','',pb)
    pairs = [('zh','z'),('sh','s'),('ch','c'),('n','l'),('in','ing'),('en','eng'),('an','ang'),
             ('r','l'),('f','h'),('b','p'),('d','t'),('g','k'),('m','n'),('un','ong'),('v','u')]
    for x,y in pairs:
        if a.replace(x,y)==b or a.replace(y,x)==b: return True
    # 单字母差异
    if len(a)==len(b):
        nd=sum(1 for i in range(len(a)) if a[i]!=b[i])
        if nd<=1: return True
    return False

# ---------- 编辑脚本(DP 回溯) ----------
def edit_script(ref, hyp):
    """返回 ops: list of ('sub',a,b)|('del',a,None)|('ins',None,b)"""
    n,m=len(ref),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=i
    for j in range(m+1): dp[0][j]=j
    for i in range(1,n+1):
        ri=ref[i-1]
        for j in range(1,m+1):
            cost=0 if ri==hyp[j-1] else 1
            dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    ops=[]; i=n; j=m
    while i>0 or j>0:
        if i>0 and j>0 and ref[i-1]==hyp[j-1]:
            i-=1; j-=1
        elif i>0 and j>0 and dp[i][j]==dp[i-1][j-1]+1:
            ops.append(('sub',ref[i-1],hyp[j-1])); i-=1; j-=1
        elif i>0 and dp[i][j]==dp[i-1][j]+1:
            ops.append(('del',ref[i-1],None)); i-=1
        else:
            ops.append(('ins',None,hyp[j-1])); j-=1
    ops.reverse()
    return ops, dp[n][m]

# ---------- 循环幻觉检测 ----------
def has_cyclic(s, min_len=4, min_rep=2):
    # 检测 s 是否含长度>=min_len 的连续重复子串
    L=len(s)
    for l in range(min_len, L//min_rep+1):
        for i in range(0, L-l*min_rep+1):
            sub=s[i:i+l]
            if sub*(min_rep)==s[i:i+l*min_rep]:
                return True, sub
    # 宽松: 任意子串重复>=3次(短串)
    for l in range(3,8):
        for i in range(0, L-l*3+1):
            sub=s[i:i+l]
            if sub*3==s[i:i+l*3]:
                return True, sub
    return False, None

# ---------- 单 op 分类 ----------
def classify_op(op):
    typ,a,b = op
    if typ=='del':
        if a in ARABIC: return '数字错误'
        if a in ASCII_LET: return '英文品牌名'   # 英文 token 被吃成中文(colmo->库姆 的 l/m/o 删除)
        if a in CNUM: return '数字错误'
        return '删字'
    if typ=='ins':
        if b in ARABIC: return '数字错误'
        if b in ASCII_LET: return '英文品牌名'
        if b in CNUM: return '数字错误'
        return '增字'
    # sub
    # 数字
    if (a in ARABIC or a in CNUM) and (b in ARABIC or b in CNUM):
        return '数字错误'
    if a in ARABIC or b in ARABIC:
        # 阿拉伯数字 vs 非数字 也是数字错误
        return '数字错误'
    # 英文/品牌
    if (a in ASCII_LET and len(a)>=1) or (b in ASCII_LET):
        return '英文品牌名'
    # 拼音
    pa,pta = pinyin_no_tone(a), pinyin_tone(a)
    pb,ptb = pinyin_no_tone(b), pinyin_tone(b)
    # 完全同音(含声调)
    if pa==pb:
        return '同音字'   # 声母韵母完全同(声调可能异也算同音族;严格同调在下面)
    if similar_pinyin(pta, ptb) or similar_pinyin(pa,pb):
        return '近音字'
    return '近音字'

# ---------- 行级分类 ----------
def classify_row(ref, hyp, cer):
    """返回 (primary_category, op_category_counter)"""
    ops,_ = edit_script(ref, hyp)
    if not ops:
        return '标点', Counter()  # 仅标点差异(已被 normalize 吃掉)
    cat_of_op = [classify_op(o) for o in ops]
    cnt = Counter(cat_of_op)
    # 幻觉覆盖判定
    nr,nh=len(ref),len(hyp)
    cyclic,_ = has_cyclic(hyp)
    unrelated = cer > 0.80
    blown = (nh > 1.6*nr and cer>0.5)
    # 若多数字符无交集 -> 幻觉
    inter = len(set(ref) & set(hyp))
    jac = inter / max(1, len(set(ref) | set(hyp)))
    if cyclic or unrelated or blown or jac < 0.25:
        return '完全幻觉', Counter({'完全幻觉': sum(cnt.values())})
    # 否则取 op 数最多的类(平局按优先级)
    priority = ['数字错误','英文品牌名','同音字','近音字','删字','增字','重排','标点']
    best=None;bestv=-1
    for k in priority:
        v=cnt.get(k,0)
        if v>bestv:
            bestv=v; best=k
    if best is None:
        best='近音字'
    # 重排检测: 字符多重叠但编辑距离大且无明显主因
    if best in ('同音字','近音字') and jac>0.6 and cer>0.35 and cnt.get('删字',0)==0 and cnt.get('增字',0)==0:
        # 主要是替换且字符集高度重叠但顺序乱 -> 可能重排(罕见)
        pass
    return best, cnt

# ============================================================
def main():
    d = json.load(open(DATA, encoding='utf-8'))
    rows = d['rows']
    total_ref_len = 0
    total_err = 0
    perfect = 0
    # 累计
    cat_rows = Counter()          # 行数
    cat_cer_sum = defaultdict(list)  # 该类行 cer 列表
    cat_op_err = Counter()        # op 级错误归属(可加)
    examples = defaultdict(list)
    homophone_pairs = Counter()
    digit_examples = []
    brand_examples = []
    per_bucket_cat = defaultdict(Counter)
    # 第一遍: 统计累计池 + 分类
    for r in rows:
        nr = normalize(r['ref']); nq = normalize(r['qwen'])
        e = editdistance.eval(nr, nq)
        total_err += e; total_ref_len += len(nr)
        if nr == nq:
            perfect += 1
            continue
        cer = e / max(1, len(nr))
        primary, opcnt = classify_row(nr, nq, cer)
        cat_rows[primary] += 1
        cat_cer_sum[primary].append(cer)
        # op 级: 若幻觉, opcnt 已只含幻觉; 否则用 opcnt
        if primary == '完全幻觉':
            cat_op_err[primary] += e   # 幻觉行整行错误都归幻觉
        else:
            # opcnt 来自每个 op 的分类, 求和应==e
            cat_op_err.update(opcnt)
        per_bucket_cat[r.get('bucket','?')][primary] += 1
        # 收集样本与高频对
        if primary == '同音字':
            ops,_ = edit_script(nr,nq)
            for o in ops:
                if o[0]=='sub':
                    homophone_pairs[(o[1],o[2])] += 1
        if primary == '数字错误' and len(digit_examples)<8:
            digit_examples.append((r['uid'], r['ref'][:30], r['qwen'][:40]))
        if primary == '英文品牌名' and len(brand_examples)<8:
            brand_examples.append((r['uid'], r['ref'][:30], r['qwen'][:40]))
        if len(examples[primary]) < 6:
            examples[primary].append({'uid':r['uid'],'ref':r['ref'],'qwen':r['qwen'],
                                      'sim':round(r.get('sim',0),3),'cer':round(cer,3)})
    overall = total_err / total_ref_len if total_ref_len else 0
    nonperf = len(rows) - perfect
    # 输出每类
    cats_order = ['同音字','近音字','数字错误','英文品牌名','删字','增字','重排','完全幻觉','标点']
    correctable = {
        '同音字':'rule','近音字':'beam','数字错误':'rule','英文品牌名':'prompt',
        '删字':'beam','增字':'beam','重排':'none','完全幻觉':'none','标点':'rule',
    }
    results = []
    for c in cats_order:
        cnt = cat_rows.get(c,0)
        if cnt==0 and c not in ('标点',):
            # 仍记录占位但跳过(标点保留为0以确认)
            pass
        avg = (sum(cat_cer_sum[c])/len(cat_cer_sum[c])) if cat_cer_sum[c] else 0.0
        operr = cat_op_err.get(c,0)
        delta = -operr/total_ref_len if total_ref_len else 0.0
        results.append({
            'name':c,'count':cnt,'pct': round(cnt/nonperf,4) if nonperf else 0,
            'avg_cer': round(avg,4),
            'op_err': operr,
            'upper_bound_cer_delta': round(delta,4),
            'correctable_by': correctable[c],
        })
    # 方向上限(可加, op 级)
    def dir_upper(d):
        return round(sum(r['upper_bound_cer_delta'] for r in results if correctable[r['name']]==d),4)
    rule_u  = dir_upper('rule')
    prompt_u= dir_upper('prompt')
    beam_u  = dir_upper('beam')
    insolvable = cat_rows.get('完全幻觉',0)

    # 桶分布
    bucket_dist = {}
    for b,c in per_bucket_cat.items():
        bucket_dist[b] = dict(c)

    # 高频同音对 top20
    top_hom = [{'pair':f'{k[0]}->{k[1]}','n':v} for k,v in homophone_pairs.most_common(25)]

    out = {
        'total_rows': len(rows),
        'perfect': perfect,
        'nonperfect': nonperf,
        'cumulative_pool_CER_official': overall,    # 官方累计池
        'header_overall_qwen_meanRow': d['overall_qwen'],  # 文件 header(行均值,非官方)
        'note': 'header 0.3848 是 per-row CER 的算术均值; 官方累计池=0.3436',
        'categories': results,
        'rule_upper_bound': rule_u,
        'prompt_upper_bound': prompt_u,
        'beam_upper_bound': beam_u,
        'insolvable_count': insolvable,
        'top_homophone_pairs': top_hom,
        'digit_examples': digit_examples,
        'brand_examples': brand_examples,
        'bucket_category_dist': bucket_dist,
        'examples_by_cat': {k:v for k,v in examples.items()},
    }
    json.dump(out, open(OUT,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    # 控制台摘要
    op_sum = sum(r['op_err'] for r in results)
    print('=== cumulative pool CER (official) = %.4f  | header meanRow=%.4f ==='%(overall, d['overall_qwen']))
    print('perfect=%d nonperfect=%d | op_err_sum=%d total_err=%d (diff=%d, 应=0)'%(perfect,nonperf,op_sum,total_err,op_sum-total_err))
    print('%-10s %5s %6s %7s %8s  %s'%('cat','cnt','pct','avgCER','delta','corr'))
    for r in results:
        print('%-10s %5d %5.1f%% %7.3f %8.4f  %s'%(r['name'],r['count'],r['pct']*100,r['avg_cer'],r['upper_bound_cer_delta'],r['correctable_by']))
    print('rule_upper = %.4f  prompt_upper=%.4f  beam_upper=%.4f  insolvable=%d'%(rule_u,prompt_u,beam_u,insolvable))
    print('\ntop homophone pairs:')
    for h in top_hom[:20]:
        print('  ',h['pair'],h['n'])
    print('\nbucket x category:')
    for b,c in bucket_dist.items():
        print(' ',b, c)
    print('\nsaved ->', OUT)

if __name__=='__main__':
    main()
