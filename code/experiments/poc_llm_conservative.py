#!/usr/bin/env python
"""精准保守 LLM 后纠正 POC (2026-07-24)

核心创新: LLM 改写 + 程序化裁剪(只留严格同音单字替换)
- 下行风险被裁剪锁死: 非同音/长度变化/重排 → 全回退原文(最坏=baseline)
- 上行 = LLM 正确识别的同音字修复
- 验证能否突破之前开放POC净负(10改善 vs 85恶化)

用法: .venv_llm/Scripts/python.exe poc_llm_conservative.py [--limit N]
"""
import os, sys, json, time, unicodedata, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "E:/hf_cache/Qwen2.5-3B-Instruct"
BASELINE = os.path.join(_HERE, "runs", "_qwen_ctx_baseline.json")   # no-context qwen 转写
REF = os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json")  # pos ref

try:
    from pypinyin import pinyin, Style
    HAS_PINYIN = True
except ImportError:
    HAS_PINYIN = False
    print("[warn] pypinyin 缺失 → 裁剪退化为'只保留相同字'(=不改), 仅验安全; 装: uv pip install pypinyin")

# 强约束提示词: 示例全部真同音(含声调 TONE3), 与程序裁剪口径一致
SYSTEM_PROMPT = """你是智能家居语音识别结果纠错器。严格规则(必须全部遵守):
1. 只做单字同音字替换(拼音含声调完全相同的字互换), 例如:
   - "开启制控温"→"开启智控温" (制zhì→智zhì, 智控温是功能名)
   - "洗衣机桶"→"洗衣机筒" (桶tǒng→筒tǒng)
   - "我要清干洗"→"我要轻干洗" (清qīng→轻qīng, 轻干洗是功能名)
   - "一键静呼吸"→"一键净呼吸" (静jìng→净jìng)
2. 输出字数必须与输入完全相同(不增不减不重排不改标点)
3. 只替换明显不合理的同音字(功能名/常用词被同音字写错)
4. 内容已合理则原样返回, 一个字都不改
5. 禁止: 增删字、重排语序、改数字、改标点、加任何解释
6. 只输出纠错结果文本, 不要前后缀不要解释"""


def normalize(t):
    if t is None:
        t = ""
    t = unicodedata.normalize('NFKC', str(t)).lower().strip()
    return ''.join(c for c in t if not unicodedata.category(c).startswith('P') and not c.isspace())


def cer_dp(rn, hn):
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
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ri == hn[j - 1] else 1))
        prev = cur
    return prev[n]


def is_homophone_strict(a, b):
    """严格同音(含声调 TONE3)。单字对比。"""
    if not HAS_PINYIN:
        return a == b  # 无 pypinyin: 只保留相同(=不改, 验证裁剪安全)
    if len(a) != 1 or len(b) != 1:
        return False
    pa = pinyin(a, style=Style.TONE3, errors='ignore')
    pb = pinyin(b, style=Style.TONE3, errors='ignore')
    if not pa or not pb or not pa[0] or not pb[0]:
        return False
    return pa[0][0] == pb[0][0]


def filter_to_homophone(orig, corrected):
    """程序裁剪: normalize 后长度相同 + 逐字比对, 只保留严格同音单字替换。
    返回 (裁剪后纯文本, n_keep, n_drop, len_mismatch)"""
    o = normalize(orig)
    c = normalize(corrected)
    if len(o) != len(c):
        return o, 0, 0, True   # 长度变 → 全回退
    res = list(o)
    n_keep = n_drop = 0
    for i in range(len(o)):
        if o[i] != c[i]:
            if is_homophone_strict(o[i], c[i]):
                res[i] = c[i]
                n_keep += 1
            else:
                n_drop += 1   # 非同音 → 回退
    return ''.join(res), n_keep, n_drop, False


class LLMCorrector:
    def __init__(self, model_path, device="cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        dtype = torch.float16 if "cuda" in device else torch.float32
        print(f"[load] {model_path} on {device} ({'fp16' if 'cuda' in device else 'fp32'})")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()

    def correct(self, text):
        import torch
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"纠错(只同音字单字替换,长度不变,合理则不改):\n{text}"}]
        inp = self.tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(inp, max_new_tokens=160, do_sample=False,
                                      temperature=1.0, top_p=1.0, repetition_penalty=1.0)
        gen = out[0][inp.shape[-1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只跑前N条(0=全量)")
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "_llm_conservative_result.json"))
    args = ap.parse_args()

    baseline = json.load(open(args.baseline, encoding='utf-8'))
    refdata = json.load(open(args.ref, encoding='utf-8'))
    rows = refdata['rows']
    if args.limit:
        rows = rows[:args.limit]
    print(f"baseline {len(baseline)} 条, 处理 pos {len(rows)} 条 | pypinyin={'yes' if HAS_PINYIN else 'NO'}")

    cor = LLMCorrector(args.model, args.device)

    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        uid, ref = r['uid'], r['ref']
        hyp = baseline.get(uid, r.get('qwen', ''))
        try:
            llm_raw = cor.correct(hyp)
        except Exception as e:
            llm_raw = hyp
            print(f"  {uid} LLM FAIL {type(e).__name__}")
        filtered, n_keep, n_drop, len_mm = filter_to_homophone(hyp, llm_raw)
        rn = normalize(ref)
        cb = cer_dp(rn, normalize(hyp))
        ca = cer_dp(rn, normalize(filtered))
        results.append({
            'uid': uid, 'ref': ref, 'hyp': hyp, 'llm_raw': llm_raw, 'filtered': filtered,
            'cer_before': cb, 'cer_after': ca,
            'improved': ca < cb, 'worse': ca > cb,
            'n_keep': n_keep, 'n_drop': n_drop, 'len_mismatch': len_mm,
        })
        if (i + 1) % 50 == 0 or i == len(rows) - 1:
            el = time.time() - t0
            imp = sum(1 for x in results if x['improved'])
            wor = sum(1 for x in results if x['worse'])
            tk = sum(x['n_keep'] for x in results)
            td = sum(x['n_drop'] for x in results)
            print(f"[{i+1}/{len(rows)}] {el:.0f}s 改善{imp} 恶化{wor} | LLM保留{tk}裁剪{td}")

    improved = sum(1 for x in results if x['improved'])
    worse = sum(1 for x in results if x['worse'])
    same = len(results) - improved - worse
    te_b = sum(x['cer_before'] for x in results)
    te_a = sum(x['cer_after'] for x in results)
    tl = sum(len(normalize(x['ref'])) for x in results)
    tot_keep = sum(x['n_keep'] for x in results)
    tot_drop = sum(x['n_drop'] for x in results)
    tot_lenmm = sum(1 for x in results if x['len_mismatch'])

    print(f"\n{'='*60}")
    print(f"精准保守 LLM 后纠正 (n={len(results)})")
    print(f"累计池CER: {te_b/tl:.4f} → {te_a/tl:.4f}  Δ={te_a/tl - te_b/tl:+.4f}")
    print(f"改善{improved}({improved/len(results)*100:.1f}%) 恶化{worse}({worse/len(results)*100:.1f}%) 不变{same}")
    print(f"LLM改动统计: 保留同音{tot_keep} | 裁剪非同音{tot_drop} | 长度不匹配全回退{tot_lenmm}条")
    print(f"裁剪率: LLM 改了 {tot_keep+tot_drop} 字, 程序只保留 {tot_keep} ({tot_keep/(tot_keep+tot_drop)*100 if tot_keep+tot_drop else 0:.0f}%)")

    improved_items = sorted([x for x in results if x['improved']], key=lambda x: x['cer_before']-x['cer_after'], reverse=True)
    worse_items = sorted([x for x in results if x['worse']], key=lambda x: x['cer_after']-x['cer_before'], reverse=True)
    print(f"\n--- 改善 top8 ---")
    for x in improved_items[:8]:
        print(f"  {x['uid']} {x['cer_before']:.2f}→{x['cer_after']:.2f} keep{x['n_keep']}drop{x['n_drop']}")
        print(f"    ref={x['ref'][:35]} hyp={x['hyp'][:35]} fil={x['filtered'][:35]}")
    print(f"\n--- 恶化 top8 (同音替换改错) ---")
    for x in worse_items[:8]:
        print(f"  {x['uid']} {x['cer_before']:.2f}→{x['cer_after']:.2f} keep{x['n_keep']}drop{x['n_drop']}")
        print(f"    ref={x['ref'][:35]} hyp={x['hyp'][:35]} fil={x['filtered'][:35]}")

    json.dump({'n': len(results), 'improved': improved, 'worse': worse, 'same': same,
               'cer_before_pooled': te_b/tl, 'cer_after_pooled': te_a/tl, 'delta': te_a/tl-te_b/tl,
               'tot_keep': tot_keep, 'tot_drop': tot_drop, 'len_mismatch': tot_lenmm,
               'results': results},
              open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"\n存 {args.out} (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
