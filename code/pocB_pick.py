"""POC B 后处理: 读 enroll_infer --multi-voice 输出(每条 spk_texts={speaker:text}),
用 llm_reject(v2 prompt, 修 v1 过严) 判每段是否家居指令, 挑家居指令段(target),
算 multi-voice+LLM挑 CER vs argmax CER(对照 exp_vanilla)。

挑策略(当前考题: 1 target 指令 + 1 非指令):
  - 唯一 accept 段 → 选它
  - 多 accept → 选最长(未来多人各指令扩展再改)
  - 0 accept → 拒(CER=1.0, 但比 argmax 循环幻觉 CER 5-25 好得多)

输出 code/pocB_result.json + 打印逐条对比 + 均值 Δ。
"""
import json, os, sys, time, unicodedata
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import llm_reject

# ---- prompt v2(复制自 pocA_fast_eval, override llm_reject 的 v1 过严 prompt) ----
SYSTEM_PROMPT_V2 = """你是一个智能家居语音助手的"指令合理性审核器"。判断一段【中文转写文本】是否是【应当被智能家居设备接受并执行的合法指令】。

核心原则: 宽松接受真实指令, 只拒明显非指令或明显超出常识的荒谬参数。

判断标准:
1. 家电实体: 是否提到可控家电/功能? 空调/灯/灯光/电视/扫地机器人/窗帘/空气净化器/热水器/闹钟/音箱/净水器/风扇/洗碗机/洗衣机/屏幕/显示屏/音乐/新风等。
2. 控制动作: 是否有控制意图? 打开/关闭/关/开/调到/调高/调低/开启/启动/暂停/播放/定/降低/升高/风速/风量/模式/摆风等。
3. 参数合理性(宽松): 含家电+动作即默认合理。正常参数(空调16-32度、亮度0-100%、风速任意档或自动、热水器洗澡水温、任何音量/定时)一律接受。只拒【明显超出常识/物理不可能】: 空调40度以上(如四十度/五十度/九十九度)、热水器100度沸水、亮度或风量或音量超100%(百分之两百)、闹钟0分钟或负数或超24小时。
4. 指令完整性(宽松): 短指令、省略房间/主语/量词都合法。"空调十六度""风速自动""开启左右摆风""打开屏幕""风速十"都是合法指令。不要求"具体房间"或"具体值"。
5. 播放类: 播放任意音频内容(歌曲/故事/诗词/电台/节目, 任何名称)都接受。"播放绝句唐杜甫""播放睡前故事""放周杰伦的歌"都接受。
6. 应拒: 纯闲聊/新闻/自言自语/与设备无关/疑问求助(空调怎么拆/灯泡哪买)/陈述事实(空调已经在制热了)/乱码/英文/空/循环重复(如"手手骨骨骨"重复)。

【接受 accept】(含家电+动作的真实指令, 含短/省略/播放类):
- "空调十六度" → accept(空调+温度16度正常)
- "风速自动" → accept(风速+自动档)
- "开启左右摆风" → accept(摆风功能)
- "播放绝句唐杜甫" → accept(播放+内容名)
- "所有灯的亮度降到五十" → accept(灯+亮度50%正常)
- "请把客厅空调调到二十六度" → accept
- "帮我定明天七点闹钟" → accept

【拒识 reject】(非指令/荒谬参数/闲聊/疑问/陈述/循环幻觉):
- "今天天气真不错出去走走" → reject(闲聊)
- "空调调到四十度" → reject(空调40度超正常范围)
- "你家空调什么牌子" → reject(闲聊问询非控制)
- "空调已经在制热了" → reject(陈述事实非指令)
- "手手骨骨骨骨骨骨" → reject(循环重复非指令)

只输出JSON, 不要其他文字: {"entity":"<家电或none>","action":"<动作或none>","reason":"<一句话>","verdict":"accept或reject"}"""
llm_reject.SYSTEM_PROMPT = SYSTEM_PROMPT_V2
from llm_reject import LLMRejecter


def normalize(t):
    """官方口径: NFKC + lower + 去 P* 标点和空白(不繁简不数字)。"""
    t = unicodedata.normalize("NFKC", t).lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith("P") and not c.isspace())


def lev(a, b):
    """Levenshtein 编辑距离(不依赖 editdistance 库)。"""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def cer(pred, ref):
    pred, ref = normalize(pred), normalize(ref)
    if not ref:
        return 0.0 if not pred else 1.0
    return lev(pred, ref) / len(ref)


mv = json.load(open(os.path.join(_HERE, "pocB_multivoice_30.json"), encoding="utf-8"))
pairs = {p["id"]: p for p in json.load(open(os.path.join(_HERE, "pocB_pairs_30.json"), encoding="utf-8"))}
argmax_cer = {int(k): v for k, v in json.load(open(os.path.join(_HERE, "pocB_argmax_cer.json"), encoding="utf-8")).items()}

print(f"[load] Qwen2.5-3B (prompt v2) for {len(mv)} 条 multi-voice", flush=True)
rej = LLMRejecter(os.environ.get("MODEL_QWEN", "E:/hf_cache/Qwen2.5-3B-Instruct"), "cuda:0")

rows = []
for r in mv:
    uid = os.path.splitext(os.path.basename(r["recognition"]))[0]   # cmd_X
    pid = int(uid.split("_")[1])
    ref = pairs[pid]["ref"]
    spk_texts = r.get("spk_texts") or {}
    if not spk_texts:   # diar 单 speaker fallback
        spk_texts = {"0": r.get("transcript", "")}

    # 判每段
    verdicts = {}
    for spk, txt in spk_texts.items():
        t = (txt or "").strip()
        if not t:
            verdicts[spk] = ("reject", "")
        else:
            v = rej.reject(t)
            verdicts[spk] = (v["verdict"], t)

    accepts = [(spk, txt) for spk, (v, txt) in verdicts.items() if v == "accept"]
    if len(accepts) == 1:
        picked, mode = accepts[0][1], "unique_accept"
    elif len(accepts) > 1:
        picked, mode = max(accepts, key=lambda x: len(x[1]))[1], f"multi_accept({len(accepts)})_longest"
    else:
        picked, mode = "", "all_reject"

    cer_mv = cer(picked, ref)
    cer_arg = argmax_cer.get(pid)
    rows.append({"uid": uid, "ref": ref, "multivoice_picked": picked, "pick_mode": mode,
                 "spk_texts": spk_texts,
                 "verdicts": {s: v[0] for s, v in verdicts.items()},
                 "cer_multivoice": round(cer_mv, 4), "cer_argmax": cer_arg})
    print(f"  {uid} [{mode:24s}] mv={cer_mv:6.2f} arg={cer_arg:6.2f} "
          f"| ref={ref[:14]!r:16} picked={picked[:14]!r}", flush=True)

mv_mean = sum(r["cer_multivoice"] for r in rows) / len(rows)
arg_mean = sum(r["cer_argmax"] for r in rows) / len(rows)
from collections import Counter
modes = Counter(r["pick_mode"] for r in rows)
better = sum(1 for r in rows if r["cer_multivoice"] < r["cer_argmax"])
worse = sum(1 for r in rows if r["cer_multivoice"] > r["cer_argmax"])
print("\n" + "=" * 60)
print(f"POC B 30 条(多speaker失败条) CER 对比")
print("=" * 60)
print(f"argmax CER 均值:           {arg_mean:.3f}")
print(f"multi-voice+LLM挑 CER 均值: {mv_mean:.3f}")
print(f"Δ: {mv_mean - arg_mean:+.3f} ({'✓改善' if mv_mean < arg_mean else '✗恶化'})")
print(f"pick_mode 分布: {dict(modes)}")
print(f"改善 {better} 条 / 恶化 {worse} 条 / 持平 {len(rows)-better-worse} 条")

out = {"n": len(rows), "cer_argmax_mean": round(arg_mean, 4),
       "cer_multivoice_mean": round(mv_mean, 4), "delta": round(mv_mean - arg_mean, 4),
       "modes": dict(modes), "better": better, "worse": worse, "rows": rows}
with open(os.path.join(_HERE, "pocB_result.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n-> pocB_result.json")
