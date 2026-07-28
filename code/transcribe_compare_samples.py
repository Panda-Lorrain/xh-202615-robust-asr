#!/usr/bin/env python
"""转写 cmd_2637 / cmd_146 的 6 个 wav, 让用户看 qwen3-asr 的具体输出文字。

复用 qwen_asr_backend.py 的模型加载逻辑 (code/.venv_qwen + Qwen3-ASR-1.7B,
language="Chinese", greedy, 与 enroll_infer --asr-backend qwen 主线一致)。

不归一 / 不去标点, 保留 qwen 原始输出。
"""
import os, json, sys, time
import torch

MODEL_PATH = r"E:/hf_cache/Qwen3-ASR-1.7B"

TARGETS = [
    ("cmd_2637", r"E:/midea_target_asr/code/runs/_pipeline_steps/cmd_2637",
     ["03_slice_full.wav", "05_sep_srcA.wav", "05_sep_srcB.wav"]),
    ("cmd_146", r"E:/midea_target_asr/code/runs/_pipeline_steps/cmd_146",
     ["03_slice_full.wav", "05_sep_srcA.wav", "05_sep_srcB.wav"]),
]

def main():
    seed = 42
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from qwen_asr import Qwen3ASRModel
    print(f"[load] Qwen3-ASR {MODEL_PATH} bf16 ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, device_map="cuda:0")

    out = {}
    t0 = time.time()
    for uid, dirp, names in TARGETS:
        out[uid] = {}
        for name in names:
            path = os.path.join(dirp, name)
            key = name.replace(".wav", "")
            if not os.path.exists(path):
                print(f"  MISSING: {path}", flush=True)
                out[uid][key] = "<file missing>"
                continue
            try:
                # 单条转写: model.transcribe(audio=path, language="Chinese") 返回 list[Result]
                res = model.transcribe(audio=path, language="Chinese", context="")
                txt = res[0].text.strip() if res else ""
                out[uid][key] = txt
                print(f"  {uid}/{name}: {txt!r}", flush=True)
            except Exception as e:
                msg = f"FAIL {type(e).__name__}: {str(e)[:80]}"
                out[uid][key] = msg
                print(f"  {uid}/{name} {msg}", flush=True)

    outp = r"E:/midea_target_asr/code/runs/_pipeline_steps/_transcript_compare_raw.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s -> {outp}", flush=True)

if __name__ == "__main__":
    main()
