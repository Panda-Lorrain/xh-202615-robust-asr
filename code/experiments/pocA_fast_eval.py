"""POC A 快速评测 v2: batch 在 4060 无效(显存带宽饱和, batch=4 仍 6s/条无并行收益)。
改: 逐条 batch=1 + max_new=64 + 规模 150(e50核心不减 + 正例50 + a/b/c/d 50)。
env: TESTSET(默认150子集) / MAX_NEW(默认64) / BATCH_TRY(默认"1")。
复用 llm_reject SYSTEM_PROMPT/parse_verdict 保 prompt 一致(公平比较)。
输出 code/pocA_llm_reject_result.json = {n, secs_total, batch, max_new_tokens, rows} 兼容 analyze_pocA.py。
"""
import json, os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from llm_reject import SYSTEM_PROMPT, parse_verdict
import torch

# prompt v2: 修 v1 过严(误拒"空调十六度""风速自动""播放绝句唐杜甫"等合法指令)。
# 参数默认合理只拒明显荒谬, 短指令/省略合法, 播放任意内容合法。
# 同时修 llm_reject 过严 bug(对 content_gate/llm_or_sim 拒识链路也有益)。
SYSTEM_PROMPT_V2 = """你是一个智能家居语音助手的"指令合理性审核器"。判断一段【中文转写文本】是否是【应当被智能家居设备接受并执行的合法指令】。

核心原则: 宽松接受真实指令, 只拒明显非指令或明显超出常识的荒谬参数。

判断标准:
1. 家电实体: 是否提到可控家电/功能? 空调/灯/灯光/电视/扫地机器人/窗帘/空气净化器/热水器/闹钟/音箱/净水器/风扇/洗碗机/洗衣机/屏幕/显示屏/音乐/新风等。
2. 控制动作: 是否有控制意图? 打开/关闭/关/开/调到/调高/调低/开启/启动/暂停/播放/定/降低/升高/风速/风量/模式/摆风等。
3. 参数合理性(宽松): 含家电+动作即默认合理。正常参数(空调16-32度、亮度0-100%、风速任意档或自动、热水器洗澡水温、任何音量/定时)一律接受。只拒【明显超出常识/物理不可能】: 空调40度以上(如四十度/五十度/九十九度)、热水器100度沸水、亮度或风量或音量超100%(百分之两百)、闹钟0分钟或负数或超24小时。
4. 指令完整性(宽松): 短指令、省略房间/主语/量词都合法。"空调十六度""风速自动""开启左右摆风""打开屏幕""风速十"都是合法指令。不要求"具体房间"或"具体值"。
5. 播放类: 播放任意音频内容(歌曲/故事/诗词/电台/节目, 任何名称)都接受。"播放绝句唐杜甫""播放睡前故事""放周杰伦的歌"都接受。
6. 应拒: 纯闲聊/新闻/自言自语/与设备无关/疑问求助(空调怎么拆/灯泡哪买)/陈述事实(空调已经在制热了)/乱码/英文/空。

【接受 accept】(含家电+动作的真实指令, 含短/省略/播放类):
- "空调十六度" → accept(空调+温度16度正常)
- "风速自动" → accept(风速+自动档)
- "开启左右摆风" → accept(摆风功能)
- "播放绝句唐杜甫" → accept(播放+内容名)
- "所有灯的亮度降到五十" → accept(灯+亮度50%正常)
- "请把客厅空调调到二十六度" → accept
- "帮我定明天七点闹钟" → accept

【拒识 reject】(非指令/荒谬参数/闲聊/疑问/陈述):
- "今天天气真不错出去走走" → reject(闲聊)
- "空调调到四十度" → reject(空调40度超正常范围)
- "你家空调什么牌子" → reject(闲聊问询非控制)
- "空调已经在制热了" → reject(陈述事实非指令)
- "空调怎么拆开" → reject(疑问求助非控制指令)
- "把风速调到百分之两百" → reject(超100%不可能)

只输出JSON, 不要其他文字: {"entity":"<家电或none>","action":"<动作或none>","reason":"<一句话>","verdict":"accept或reject"}"""

PROMPT = SYSTEM_PROMPT_V2 if os.environ.get("PROMPT_VER", "v2") == "v2" else SYSTEM_PROMPT

MODEL = os.environ.get("MODEL_QWEN", "E:/hf_cache/Qwen2.5-3B-Instruct")
DEVICE = "cuda:0"
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
TESTSET = os.environ.get("TESTSET", os.path.join(_HERE, "llm_testset_pocA_150.json"))
BATCH_TRY = [int(x) for x in os.environ.get("BATCH_TRY", "1").split(",")]

ts = json.load(open(TESTSET, encoding="utf-8"))


def user_prompt(t):
    return f'请审核这条转写文本:\n"{t}"'


from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"[load] {MODEL}  max_new={MAX_NEW}  testset={os.path.basename(TESTSET)} ({len(ts)}条)  batch_try={BATCH_TRY}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"  # generate 必须 left pad
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
print(f"[load] done", flush=True)


def run_with_batch(B):
    """batch=B 跑全量, OOM 返回 None 触发降级。"""
    rows = []
    t0 = time.time()
    for i in range(0, len(ts), B):
        chunk = ts[i:i + B]
        msgs = [[{"role": "system", "content": PROMPT},
                 {"role": "user", "content": user_prompt(r["text"])}] for r in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        try:
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return None
            raise
        gen = out[:, enc["input_ids"].shape[1]:]
        raws = tok.batch_decode(gen, skip_special_tokens=True)
        for r, raw in zip(chunk, raws):
            v = parse_verdict(raw)
            rows.append({"text": r["text"], "gold": r["gold"], "pred": v["verdict"],
                         "layer": r["layer"], "entity": v.get("entity"),
                         "action": v.get("action"), "reason": (v.get("reason") or "")[:80]})
        done = min(i + B, len(ts))
        if done % 15 < B or done == len(ts):
            print(f"  [{done}/{len(ts)}] {time.time()-t0:.0f}s ({(time.time()-t0)/done*1000:.0f}ms/条 batch={B})", flush=True)
    return rows, time.time() - t0


final = None
for B in BATCH_TRY:
    torch.cuda.empty_cache()
    res = run_with_batch(B)
    if res is not None:
        rows, dt = res
        print(f"[ok] batch={B} 跑完 {len(rows)} 条 {dt:.0f}s", flush=True)
        final = (rows, dt, B)
        break
    print(f"[OOM] batch={B} 失败, 降级重试...", flush=True)

assert final is not None, "全降级仍失败?"
rows, dt, used_b = final
out = {"n": len(rows), "secs_total": round(dt, 2), "batch": used_b,
       "max_new_tokens": MAX_NEW, "testset": os.path.basename(TESTSET), "rows": rows}
with open(os.path.join(_HERE, "pocA_llm_reject_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[done] {len(rows)} 条 {dt:.0f}s (batch={used_b}, max_new={MAX_NEW}) -> pocA_llm_reject_result.json", flush=True)
