#!/usr/bin/env python
"""下载 FireRedASR-LLM-L(1代,8.3B Qwen2-7B) 权重到 hf_cache(走 hf-mirror 国内镜像)。

结构(from_pretrained 要求 model_dir 下有 Qwen2-7B-Instruct 子目录):
  E:/hf_cache/FireRedASR-LLM-L/
    ├── model.pth.tar        (projector+config, 3.6GB)
    ├── asr_encoder.pth.tar  (ASR encoder)
    ├── cmvn.ark             (特征 CMVN)
    ├── config.yaml
    └── Qwen2-7B-Instruct/   (15.2GB, base LLM, FireRedASR 不改它只用 adapter)
"""
import os, time
# 用官方 huggingface.co 直连(本机可达, 小文件1.7s); hf-mirror 触发 huggingface_hub1.23 metadata bug
# 如需镜像可 export HF_ENDPOINT=https://hf-mirror.com 覆盖
from huggingface_hub import snapshot_download

BASE = r"E:/hf_cache/FireRedASR-LLM-L"

t0 = time.time()
print(f"[dl] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
print(f"[dl] FireRedASR-LLM-L → {BASE}")
snapshot_download("fireredteam/FireRedASR-LLM-L", local_dir=BASE,
                  max_workers=4)
print(f"[dl] Qwen2-7B-Instruct → {BASE}/Qwen2-7B-Instruct")
snapshot_download("Qwen/Qwen2-7B-Instruct",
                  local_dir=os.path.join(BASE, "Qwen2-7B-Instruct"),
                  max_workers=4)
print(f"[dl] DONE {time.time()-t0:.0f}s")
