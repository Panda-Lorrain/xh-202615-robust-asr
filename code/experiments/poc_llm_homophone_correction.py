"""
POC: LLM 家居同音字纠正
用 Qwen2.5-3B 对 ASR 输出做家居语义纠正,测试能救回多少 A 类同音字错误。

用法:
  uv run --venv .venv_llm python poc_llm_homophone_correction.py
"""
import os, sys, json, time, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repro import set_global_seed, resolve_model

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "E:/hf_cache/Qwen2.5-3B-Instruct"  # 本地缓存,避免重复下载

SYSTEM_PROMPT = """你是一个智能家居语音识别纠错器。你的任务非常简单:只替换错误的字,其他一切不变。

严格规则(必须全部遵守):
1. 只能做单字替换:把一个错字换成正确的字,不能增加字、删除字、移动字的位置
2. 输出长度必须与输入长度完全相同(字数一样)
3. 只替换语义明显不合理的字(如同音字导致不通顺)
4. 如果内容已经合理,直接返回原文,一个字都不要改
5. 不要添加或删除标点符号
6. 不要解释,只输出结果

判断标准:
- "灯光呈亮的色调" → "灯光常亮冷色调" (呈→常, 的→冷, 共换2字,长度不变)
- "关闭新乡空调" → "关闭星香空调" (新→星, 乡→香, 共换2字,长度不变)
- "洗衣机桶" → "洗衣机筒" (桶→筒, 换1字,长度不变)
- "我要看电影了" → "我要看电影了" (内容合理,不改,直接返回原文)
- "打开烟机打开洗衣机" → "打开烟机打开洗衣机" (内容合理,不改)

禁止:
- 禁止添加字("了"→"了。"❌)
- 禁止删除字("衣服啊"→"衣服"❌, 除非"啊"是明显多余的语气词)
- 禁止重排("风直吹"→"风吹直"❌)
- 禁止改品牌名/产品功能名("轻干洗""风直吹""智控温"是正确的功能名)
"""

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

class LLMCorrector:
    def __init__(self, model_path=DEFAULT_MODEL, device="cuda:0"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.device = device
        self.dtype = torch.float16 if "cuda" in str(device) else torch.float32
        print(f"[load] Qwen {model_path} on {device}")
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self.dtype, trust_remote_code=True
        ).to(device).eval()
        self._has_template = hasattr(self.tok, "apply_chat_template")
        print(f"[load] done")

    def correct(self, text, max_new_tokens=200):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"以下是语音识别结果,请只替换错误的字,其他不变。如果已经合理,直接返回原文:\n{text}"}]
        if self._has_template:
            inputs = self.tok.apply_chat_template(msgs, return_tensors="pt",
                                                   add_generation_prompt=True).to(self.device)
        else:
            prompt = f"System: {SYSTEM_PROMPT}\nUser: 请纠正以下语音识别结果:\n{text}\nAssistant:"
            inputs = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with __import__('torch').inference_mode():
            out = self.model.generate(inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False, temperature=1.0,
                                       top_p=1.0, repetition_penalty=1.0)
        gen = out[0][inputs.shape[-1]:]
        return self.tok.decode(gen, skip_special_tokens=True).strip()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(_HERE, "runs", "llm_correction_poc_input.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "llm_correction_poc_result.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N条(0=全部)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)
    items = json.load(open(args.input, encoding="utf-8"))
    if args.limit > 0:
        items = items[:args.limit]
    print(f"输入: {len(items)} 条")

    corrector = LLMCorrector(args.model, args.device)

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        hyp = item['hyp']
        ref = item['ref']
        corrected = corrector.correct(hyp)
        cer_before = cer(ref, hyp)
        cer_after = cer(ref, corrected)
        improved = cer_after < cer_before
        perfect = cer_after == 0

        results.append({
            'uid': item['uid'],
            'ref': ref,
            'hyp': hyp,
            'corrected': corrected,
            'cer_before': cer_before,
            'cer_after': cer_after,
            'improved': improved,
            'perfect': perfect,
        })

        if (i+1) % 20 == 0 or i == len(items)-1:
            elapsed = time.time() - t0
            improved_count = sum(1 for r in results if r['improved'])
            perfect_count = sum(1 for r in results if r['perfect'])
            print(f"[{i+1}/{len(items)}] {elapsed:.1f}s | 改善:{improved_count} | 完美:{perfect_count}")

    # 统计
    improved = sum(1 for r in results if r['improved'])
    perfect = sum(1 for r in results if r['perfect'])
    worse = sum(1 for r in results if r['cer_after'] > r['cer_before'])
    same = len(results) - improved - worse

    total_cer_before = sum(r['cer_before'] for r in results)
    total_cer_after = sum(r['cer_after'] for r in results)

    print(f"\n=== 结果 ===")
    print(f"总条数: {len(results)}")
    print(f"改善: {improved} ({improved/len(results)*100:.1f}%)")
    print(f"恶化: {worse} ({worse/len(results)*100:.1f}%)")
    print(f"不变: {same} ({same/len(results)*100:.1f}%)")
    print(f"完美(纠正后CER=0): {perfect} ({perfect/len(results)*100:.1f}%)")
    print(f"平均CER: {total_cer_before/len(results):.4f} → {total_cer_after/len(results):.4f}")
    print(f"CER变化: {total_cer_after/len(results) - total_cer_before/len(results):.4f}")

    # 保存
    output = {
        'n': len(results),
        'improved': improved,
        'worse': worse,
        'same': same,
        'perfect': perfect,
        'avg_cer_before': total_cer_before / len(results),
        'avg_cer_after': total_cer_after / len(results),
        'results': results
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {args.out}")

    # 显示改善最大的10条
    improved_items = [r for r in results if r['improved']]
    improved_items.sort(key=lambda x: x['cer_before'] - x['cer_after'], reverse=True)
    print(f"\n=== 改善最大的10条 ===")
    for r in improved_items[:10]:
        print(f"  {r['uid']} | CER {r['cer_before']:.3f}→{r['cer_after']:.3f}")
        print(f"    ref: {r['ref'][:60]}")
        print(f"    hyp: {r['hyp'][:60]}")
        print(f"    cor: {r['corrected'][:60]}")
        print()

    # 显示恶化最大的10条
    worse_items = [r for r in results if r['cer_after'] > r['cer_before']]
    worse_items.sort(key=lambda x: x['cer_after'] - x['cer_before'], reverse=True)
    print(f"\n=== 恶化最大的10条 ===")
    for r in worse_items[:10]:
        print(f"  {r['uid']} | CER {r['cer_before']:.3f}→{r['cer_after']:.3f}")
        print(f"    ref: {r['ref'][:60]}")
        print(f"    hyp: {r['hyp'][:60]}")
        print(f"    cor: {r['corrected'][:60]}")
        print()

if __name__ == "__main__":
    main()
