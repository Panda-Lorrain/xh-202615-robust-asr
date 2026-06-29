"""线C W5: LLM 语义拒识骨架(三路融合之"语义合理性"路)。

按美的 #1 论文《Reject or Not?》(arXiv:2512.10257, Midea AI Research) 的
"智能家居 query 拒识"框架, 用 Qwen2.5-3B-Instruct 对一段转写文本做
"是否合法智能家居指令"的二分类(accept=合法应执行 / reject=应拒识),
并给出理由。

本模块只做语义层(纯文本输入), 不碰音频/声纹。它设计为被 enroll_infer.py /
inference.py (pipeline) 在拿到 DiCoW 转写文本后调用, 与另外两路融合:
  - 声纹置信度(enrollment 余弦 sim) → enroll_infer 已有
  - LLM 语义合理性(本模块)
  - PVAD/STNO 检测(机制层) → pipeline 已有
三路在 pipeline 层加权融合最终拒识判定(见文末"接口设计")。

自适应 CoT(adaptive chain-of-thought, 草案, 对齐 #1 论文 §3 三层架构之第一层
"family-agnostic 语义边界 adapter" 的零样本近似):
  step1: 是否含【家电实体】(空调/灯/电视/扫地机器人/窗帘/空气净化器/热水器/闹钟/音箱...)
  step2: 是否含【控制动作】(打开/关闭/调到/启动/播放/定...+目标实体)
  step3: 参数是否在合理范围(温度0-40度、亮度0-100%、定时>0) — 超出=不合理=reject
  step4: 是否"非对设备说的闲话"(多人聊天/自言自语/与设备无关请求)
  结论: accept(含合法 实体+动作+合理参数 且是对设备说的) / reject(其余)

使用:
  # 单条(CPU 小测)
  python code/llm_reject.py --text "请把客厅空调温度调到二十六度"
  # 批量(GPU 跑测试集, 算 precision/recall/f1)
  python code/llm_reject.py --testset code/llm_reject_testset.json \
      --model E:/hf_cache/Qwen2.5-3B-Instruct --device cuda:0 \
      --out-json code/llm_reject_result.json

环境(独立 venv, 不碰主 DiCoW venv):
  E:/midea_target_asr/.venv_llm  (transformers 4.46.3 + torch 2.12)
"""
import os, sys, json, argparse, time

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "E:/hf_cache/Qwen2.5-3B-Instruct"

# ---- 自适应 CoT prompt(家电实体+动作+参数合理性+非闲话) ----
# few-shot 正负样例取自本数据集 target(合法)/nontarget(干扰闲话) + #1 论文 reject 类别样例
SYSTEM_PROMPT = """你是一个智能家居语音助手的"指令合理性审核器"。你的任务是判断一段【中文转写文本】是否是一条【应当被智能家居设备接受并执行的合法指令】。

判断标准(逐条 CoT 推理):
1. 家电实体: 文本是否提到可被控制的家电或功能? 例如: 空调、灯/灯光、电视、扫地机器人、窗帘、空气净化器、热水器、闹钟、音箱、净水器、风扇等。
2. 控制动作: 是否有针对该实体的明确控制意图? 例如: 打开/关闭/关小/调高/调到/启动/暂停/播放/定时等。
3. 参数合理性: 若带参数(温度、亮度、时间、时长等), 数值是否在合理范围? 例如"空调调到40度"明显不合理→应拒识; "调到二十六度"合理。
4. 对象合理性: 文本是否是"对设备发出的指令"? 还是"非对设备说的闲聊/自言自语/多人聊天/与设备功能无关的请求"? 闲聊与无关请求→应拒识。

【应接受 accept 的例子】(含家电实体+控制动作+合理参数, 对设备说的):
- "请把客厅的空调温度调到二十六度" → accept(空调+调到+26度合理)
- "小美小美, 打开卧室的灯" → accept(唤醒词+灯+打开)
- "帮我定一个明天早上七点的闹钟" → accept(闹钟+定时+时间合理)
- "打开热水器把水温调到四十五度" → accept(热水器+调到+45度合理)

【应拒识 reject 的例子】(闲聊/无关/无实体/无动作/参数不合理):
- "今天天气真不错, 我们出去走走吧" → reject(纯闲聊, 无家电无动作)
- "这个电影我觉得挺好看的, 推荐你也看看" → reject(闲聊)
- "你晚上想吃什么, 我来做" → reject(日常对话, 非设备指令)
- "那本书我看完了, 写得真不错" → reject(自言自语/闲聊)
- "空调调到四十度" → reject(参数不合理, 超正常范围)
- "这这这" → reject(无意义短句/ASR错误)

请按以下 JSON 格式输出, 只输出 JSON, 不要任何额外文字:
{"entity": "<识别到的家电实体, 没有则填 none>", "action": "<控制动作, 没有则填 none>", "reason": "<一句话理由>", "verdict": "accept 或 reject"}"""


def build_user_prompt(text: str) -> str:
    return f"请审核这条转写文本:\n\"{text}\""


def parse_verdict(raw: str) -> dict:
    """从模型输出里抽 JSON。容错: 模型可能包裹在 ```json ... ``` 或带前后缀。"""
    raw = raw.strip()
    # 去掉 markdown code fence
    if raw.startswith("```"):
        raw = raw.split("```")
        # 取最长且含 { 的段
        cand = [s for s in raw if "{" in s]
        raw = cand[0] if cand else raw[0]
    # 截取第一个 {...}
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j != -1 and j > i:
        raw = raw[i:j + 1]
    try:
        obj = json.loads(raw)
    except Exception:
        return {"entity": "none", "action": "none",
                "reason": f"PARSE_FAIL: {raw[:120]}", "verdict": "reject"}
    v = str(obj.get("verdict", "")).strip().lower()
    if v not in ("accept", "reject"):
        obj["verdict"] = "reject"  # 解析不出明确 accept → 保守拒识
    else:
        obj["verdict"] = v
    return obj


class LLMRejecter:
    """加载 Qwen2.5-3B-Instruct, 提供单条/批量语义拒识。"""

    def __init__(self, model_path=DEFAULT_MODEL, device="cuda:0", dtype=None,
                 load_model=True):
        self.model_path = model_path
        self.device = device
        import torch
        self.torch = torch
        self.dtype = dtype if dtype is not None else (torch.float16
                    if ("cuda" in str(device) and torch.cuda.is_available()) else torch.float32)
        if load_model:
            self._load()

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        t0 = time.time()
        print(f"[load] Qwen {self.model_path} on {self.device} ({self.dtype})")
        self.tok = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()
        # 试试 apply_chat_template(4.46 原生支持 Qwen2.5 chat template)
        self._has_template = hasattr(self.tok, "apply_chat_template")
        print(f"[load] done {time.time()-t0:.1f}s  has_chat_template={self._has_template}")

    def reject(self, text: str, max_new_tokens=160) -> dict:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(text)}]
        if self._has_template:
            inputs = self.tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt").to(self.device)
        else:
            # fallback: 手拼(不推荐, Qwen2.5 应有 template)
            prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(text)
            inputs = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False,  # 贪心, 可复现
                                      pad_token_id=self.tok.eos_token_id)
        gen = out[0][inputs.shape[-1]:]
        raw = self.tok.decode(gen, skip_special_tokens=True).strip()
        return {**parse_verdict(raw), "raw": raw}

    def reject_batch(self, texts, max_new_tokens=160):
        return [self.reject(t, max_new_tokens) for t in texts]


# ---- 三路融合接口(供 enroll_infer / pipeline 调用) ----
def fuse_three_ways(verdict_llm, sim=None, sim_threshold=0.5,
                    stno_target_ratio=None, stno_target_thresh=0.1,
                    w=(0.4, 0.4, 0.2)):
    """三路加权融合拒识(草案)。
    verdict_llm: 'accept'/'reject' (本模块)
    sim: enrollment 余弦 sim (声纹路, enroll_infer 的 max_sim)
    stno_target_ratio: PVAD/STNO 检测出的 target 帧占比 (机制路)
    返回 (final_verdict, score) — score 越高越倾向 accept。
    三路任一强 reject 信号可直接否决(保守拒识, 对齐美的"零误触发"偏好)。"""
    llm_accept = 1.0 if verdict_llm == "accept" else 0.0
    sim_accept = 1.0 if (sim is not None and sim >= sim_threshold) else 0.0
    stno_accept = 1.0 if (stno_target_ratio is not None
                          and stno_target_ratio >= stno_target_thresh) else 0.0
    w_llm, w_sim, w_stno = w
    score = w_llm * llm_accept + w_sim * sim_accept + w_stno * stno_accept
    final = "accept" if score >= 0.5 else "reject"
    return final, round(score, 3)


def _metrics(rows):
    """rows: list of {gold, pred} (accept/reject)。算 reject 为正类的 P/R/F1。"""
    tp = fp = fn = tn = 0
    for r in rows:
        g, p = r["gold"], r["pred"]
        if g == "reject" and p == "reject": tp += 1
        elif g == "accept" and p == "reject": fp += 1
        elif g == "reject" and p == "accept": fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(rows) if rows else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(rows)}


def main():
    ap = argparse.ArgumentParser(description="LLM 语义拒识(Qwen2.5-3B)")
    ap.add_argument("--text", help="单条文本(CPU/GPU 小测)")
    ap.add_argument("--testset", help="批量测试集 JSON(每条 {text, gold})")
    ap.add_argument("--infer-json", help="推理模式: 读 [{file,text}] 无 gold, 输出每条 verdict(对接 enroll 转写)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "llm_reject_result.json"))
    ap.add_argument("--no-load", action="store_true", help="不加载模型(只做 prompt/解析自测)")
    args = ap.parse_args()

    if args.no_load or (not args.text and not args.testset and not args.infer_json):
        # 纯逻辑自测: 验证 parse_verdict + SYSTEM_PROMPT 文本可用
        print("[selftest] parse_verdict 容错:")
        for s in ['{"verdict":"accept","entity":"空调","action":"调到","reason":"x"}',
                  '```json\n{"verdict":"reject"}\n```', '乱七八糟']:
            print(" ", parse_verdict(s))
        print("[selftest] SYSTEM_PROMPT 字符数:", len(SYSTEM_PROMPT))
        if not args.text and not args.testset:
            print("[hint] 加 --text '...' 单条测, 或 --testset code/llm_reject_testset.json 批量跑")
            return

    rej = LLMRejecter(args.model, args.device, load_model=not args.no_load)

    if args.text:
        r = rej.reject(args.text)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    if args.infer_json:
        # 推理模式: 无 gold, 对 enroll 转写批量判 accept/reject, 输出每条 verdict(供 fuse_eval 融合)
        rows = json.load(open(args.infer_json, encoding="utf-8"))
        print(f"[infer] {len(rows)} 条转写文本 on {args.device}")
        out, t0 = [], time.time()
        for i, r in enumerate(rows):
            text = r.get("text", "") or ""
            fkey = r.get("file") or os.path.basename(r.get("recognition", "") or "") or ""
            if text.strip():
                pred = rej.reject(text)
            else:
                pred = {"verdict": "reject", "entity": "none", "action": "none",
                        "reason": "EMPTY_TRANSCRIPT(空转写, 视为拒识)"}
            out.append({"file": fkey, "text": text, "pred": pred["verdict"],
                        "entity": pred.get("entity"), "action": pred.get("action"),
                        "reason": pred.get("reason")})
            if i == 0 or (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(rows)}] pred={pred['verdict']} | {text[:30]}")
        dt = time.time() - t0
        n_acc = sum(1 for r in out if r["pred"] == "accept")
        res = {"n": len(out), "secs_total": round(dt, 2),
               "secs_per_item": round(dt / max(len(out), 1), 3),
               "n_accept": n_acc, "n_reject": len(out) - n_acc, "rows": out}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\n[infer] accept={n_acc}/{len(out)} reject={len(out)-n_acc} "
              f"({dt:.0f}s, {dt/max(len(out),1):.2f}s/条) -> {args.out_json}")
        return

    # 批量跑测试集 + 算指标
    rows = json.load(open(args.testset, encoding="utf-8"))
    print(f"[batch] {len(rows)} 条 on {args.device}")
    out, t0 = [], time.time()
    for i, r in enumerate(rows):
        pred = rej.reject(r["text"])
        rec = {"text": r["text"], "gold": r["gold"], "pred": pred["verdict"],
               "entity": pred.get("entity"), "action": pred.get("action"),
               "reason": pred.get("reason")}
        out.append(rec)
        flag = "OK" if pred["verdict"] == r["gold"] else "MISS"
        print(f"  [{i+1}/{len(rows)}] {flag} gold={r['gold']} pred={pred['verdict']} | {r['text'][:30]}")
    dt = time.time() - t0
    m = _metrics(out)
    m["secs_total"] = round(dt, 2); m["secs_per_item"] = round(dt / len(rows), 3)
    res = {"metrics": m, "rows": out}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[metrics] {json.dumps(m, ensure_ascii=False)}")
    print(f"[done] {args.out_json}")


if __name__ == "__main__":
    main()
