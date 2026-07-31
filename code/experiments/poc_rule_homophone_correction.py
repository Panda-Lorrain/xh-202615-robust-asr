"""
POC: 规则同音字纠错
从A集ref构建家居指令同音字词典,用规则方式纠正ASR输出中的同音字错误。

用法:
  uv run python poc_rule_homophone_correction.py
"""
import os, sys, json, unicodedata
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().strip()
    return ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P') and not ch.isspace())

def cer(ref, hyp):
    r, h = list(normalize(ref)), list(normalize(hyp))
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[m][n] / m if m > 0 else 0

def build_homophone_dict(rows):
    """从A集ref构建同音字词典: 当ASR输出某字与ref某字不同,且是单字替换,记为同音字对"""
    pair_freq = Counter()
    for r in rows:
        if r['qwen_cer'] == 0:
            continue
        ref_n = normalize(r['ref'])
        hyp_n = normalize(r['qwen'])
        if len(ref_n) != len(hyp_n):
            continue
        diffs = [(i, ref_n[i], hyp_n[i]) for i in range(len(ref_n)) if ref_n[i] != hyp_n[i]]
        if len(diffs) == 1:
            pos, correct, wrong = diffs[0]
            pair_freq[(wrong, correct)] += 1

    # 只保留出现>=2次的同音字对
    homophone_dict = {}
    for (w, c), freq in pair_freq.items():
        if freq >= 2:
            if w not in homophone_dict:
                homophone_dict[w] = []
            homophone_dict[w].append((c, freq))

    # 按频率排序
    for w in homophone_dict:
        homophone_dict[w].sort(key=lambda x: -x[1])

    return homophone_dict

def build_vocab(rows):
    """从A集ref构建家居指令词汇表"""
    word_freq = Counter()
    for r in rows:
        ref_n = normalize(r['ref'])
        for n in [2, 3, 4]:
            for i in range(len(ref_n)-n+1):
                word_freq[ref_n[i:i+n]] += 1
    return {w for w, c in word_freq.items() if c >= 3 and len(w) >= 2}

def correct_with_homophone_dict(text, homophone_dict):
    """用同音字词典纠正ASR输出"""
    text_n = normalize(text)
    corrected = list(text_n)
    changes = []

    for i, ch in enumerate(corrected):
        if ch in homophone_dict:
            # 尝试替换为同音字
            best_replacement = None
            best_freq = 0
            for replacement, freq in homophone_dict[ch]:
                # 检查替换后是否形成更常见的双字组合
                left = corrected[i-1] if i > 0 else ''
                right = corrected[i+1] if i < len(corrected)-1 else ''

                # 检查左邻组合
                left_ok = True
                if left:
                    bigram = left + replacement
                    # 如果替换后形成常见词,更好
                    if bigram in ['空调', '风速', '温度', '模式', '打开', '关闭', '开启',
                                  '播放', '灯光', '窗帘', '食物', '什么', '适合', '推荐',
                                  '厨房', '客厅', '卧室', '儿童', '节能', '制热', '制冷',
                                  '风量', '风向', '风直', '防直', '智控', '净呼', '轻干',
                                  '星香', '新乡', '非餐', '备餐']:
                        left_ok = True

                if freq > best_freq:
                    best_replacement = replacement
                    best_freq = freq

            if best_replacement and best_freq >= 2:
                corrected[i] = best_replacement
                changes.append((i, ch, best_replacement))

    return ''.join(corrected), changes

def main():
    # 加载数据
    with open(os.path.join(_HERE, 'runs', 'poc_qwen_asr_full_result.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    rows = data['rows']

    # 构建同音字词典
    homophone_dict = build_homophone_dict(rows)
    print(f'同音字词典: {len(homophone_dict)} 个错字映射')
    for w, replacements in sorted(homophone_dict.items(), key=lambda x: -sum(f for _, f in x[1])):
        r_str = ', '.join(f'{c}({f}次)' for c, f in replacements[:3])
        print(f'  {w} → {r_str}')

    # 构建词汇表
    vocab = build_vocab(rows)
    print(f'\n家居词汇表: {len(vocab)} 个词')

    # 对A类158条做纠正
    with open(os.path.join(_HERE, 'runs', 'llm_correction_poc_input.json'), 'r', encoding='utf-8') as f:
        items = json.load(f)
    print(f'\n输入: {len(items)} 条')

    results = []
    for item in items:
        hyp = item['hyp']
        ref = item['ref']
        corrected, changes = correct_with_homophone_dict(hyp, homophone_dict)
        cer_before = cer(ref, hyp)
        cer_after = cer(ref, corrected)
        improved = cer_after < cer_before
        perfect = cer_after == 0

        results.append({
            'uid': item['uid'],
            'ref': ref,
            'hyp': hyp,
            'corrected': corrected,
            'changes': changes,
            'cer_before': cer_before,
            'cer_after': cer_after,
            'improved': improved,
            'perfect': perfect,
        })

    improved = sum(1 for r in results if r['improved'])
    perfect = sum(1 for r in results if r['perfect'])
    worse = sum(1 for r in results if r['cer_after'] > r['cer_before'])
    same = len(results) - improved - worse

    total_cer_before = sum(r['cer_before'] for r in results)
    total_cer_after = sum(r['cer_after'] for r in results)

    print(f'\n=== 规则同音字纠错结果 ===')
    print(f'总条数: {len(results)}')
    print(f'改善: {improved} ({improved/len(results)*100:.1f}%)')
    print(f'恶化: {worse} ({worse/len(results)*100:.1f}%)')
    print(f'不变: {same} ({same/len(results)*100:.1f}%)')
    print(f'完美: {perfect} ({perfect/len(results)*100:.1f}%)')
    print(f'平均CER: {total_cer_before/len(results):.4f} → {total_cer_after/len(results):.4f}')
    print(f'CER变化: {total_cer_after/len(results) - total_cer_before/len(results):.4f}')

    # 显示改善的条目
    improved_items = [r for r in results if r['improved']]
    improved_items.sort(key=lambda x: x['cer_before'] - x['cer_after'], reverse=True)
    print(f'\n=== 改善的条目 ===')
    for r in improved_items[:15]:
        print(f'  {r["uid"]} | CER {r["cer_before"]:.3f}→{r["cer_after"]:.3f} | 改动: {r["changes"]}')
        print(f'    ref: {r["ref"][:60]}')
        print(f'    hyp: {r["hyp"][:60]}')
        print(f'    cor: {r["corrected"][:60]}')
        print()

    # 显示恶化的条目
    worse_items = [r for r in results if r['cer_after'] > r['cer_before']]
    worse_items.sort(key=lambda x: x['cer_after'] - x['cer_before'], reverse=True)
    print(f'\n=== 恶化的条目 ===')
    for r in worse_items[:10]:
        print(f'  {r["uid"]} | CER {r["cer_before"]:.3f}→{r["cer_after"]:.3f} | 改动: {r["changes"]}')
        print(f'    ref: {r["ref"][:60]}')
        print(f'    hyp: {r["hyp"][:60]}')
        print(f'    cor: {r["corrected"][:60]}')
        print()

    # 保存
    output = {
        'n': len(results),
        'improved': improved,
        'worse': worse,
        'same': same,
        'perfect': perfect,
        'homophone_dict': {k: v for k, v in homophone_dict.items()},
        'results': results
    }
    out_path = os.path.join(_HERE, 'runs', 'rule_homophone_correction_result.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'结果已保存到 {out_path}')

if __name__ == "__main__":
    main()
