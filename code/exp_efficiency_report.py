#!/usr/bin/env python
"""效率优化探索汇总报告 (2026-07-18)。

GPU: RTX 4060 Laptop (8GB VRAM, Ada Lovelace, compute 8.9)
Models: Whisper-large-v3-turbo (fp16), Qwen3-ASR-1.7B (bf16)
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))

def load_json(path):
    full = os.path.join(_HERE, path)
    if os.path.exists(full):
        return json.load(open(full, encoding="utf-8"))
    return None

print("""
================================================================
  效率优化探索汇总报告 (2026-07-18)
  GPU: RTX 4060 Laptop (8GB VRAM, Ada Lovelace, compute 8.9)
================================================================

⚠️⚠️ 2026-07-18 对抗审查勘误 (4-agent 复核, 下列原结论有多处方法学错误):
  1. [Part A] qwen torch.compile baseline 错用 0.0950(撞 exp_int8_qwen 的 bf16 数), 真实
     baseline 0.1161 → 实际改善 18-19%(非表里的 -0.2%)。但 compile_time 0.26s 物理不可能
     (典型 30-60s), 疑似 dynamo guard 失败回退 eager 静默 no-op, 须 TORCH_COMPILE_DEBUG=1
     + warmup 3 + 跑 30+ 条重测, 方知 qwen compile 真实收益。
  2. [Part C] "int8 严重负优化" 仅限 bitsandbytes LLM.int8() batch=1, 未测 GPTQ/AWQ/TensorRT,
     不能否定整个量化方向; 4060(Ada Lovelace) 结论不可外推 L20(Ampere); 样本仅 5 条方差大。
  3. [Part D/总结] batch=16 "5x" 仅 ASR 子进程, 全管线 1.76x(overall_rtf 0.25→0.142)。
  4. [总结] "唯一杠杆 batch" 与 Part A "ONNX 可行预期 2-3x" 自相矛盾(ONNX 未落地, 非证伪)。
  5. 官方按 batch=1 测 RTF → batch 红利不进效率腿分, 仅加快开发 A/B 迭代。官方 batch=1 口径
     下唯一确定杠杆是关 SE(省 30.6% RTF); ONNX/GPTQ-AWQ int4/speculative decoding/Qwen3-ASR
     蒸馏 0.5B 均 POC 未做。详见 docs/SE_bugfix_AB结果_2026-07-18.md + code/audit_se_bugfix.json。
  原始数字保留作过程记录, 结论以本勘误为准。

Part A: torch.compile
---------------------
| Backend      | Mode              | avg_RTF | vs baseline | Compile |
|-------------|-------------------|---------|-------------|---------|
| Whisper     | baseline (SDPA)   | 0.0562  | -           | -       |
| Whisper     | reduce-overhead   | 0.0549  | -2.2%       | 0.5s    |
| Whisper     | max-autotune      | 0.0545  | -3.0%       | 0.1s    |
| Qwen3-ASR   | baseline (SDPA)   | 0.0950  | -           | -       |
| Qwen3-ASR   | reduce-overhead   | 0.0948  | -0.2%       | 4.0s    |
| Qwen3-ASR   | max-autotune      | 0.0943  | -0.7%       | 0.3s    |

结论: torch.compile 对两个模型均无显著加速 (2-3% 以内, 噪声级别)。
原因: 模型已使用 SDPA (Scaled Dot Product Attention), PyTorch 2.5+ 内置的
     flash/mem-efficient kernel 已自动启用, compile 无法进一步优化。
CER: 无变化 (文本输出完全一致)。
推荐: 不采用。

Part B: Flash Attention 2
-------------------------
- flash-attn 包: 未安装 (Windows 需 CUDA Toolkit 编译, 成本高)
- PyTorch SDPA 状态:
    flash_sdp_enabled: True
    mem_efficient_sdp_enabled: True
    math_sdp_enabled: True
- Whisper 默认: WhisperSdpaAttention (已使用 SDPA)
- Qwen3-ASR 默认: Qwen3ASRAudioAttention → dispatch 到 SDPA
  (config._attn_implementation = "sdpa")

结论: SDPA 的 flash kernel 已在 PyTorch 层面启用, 等效于 Flash Attention 2。
     单独安装 flash-attn 包收益极小 (<5%), 且 Windows 编译复杂。
推荐: 不安装。SDPA 已是最佳方案。

Part C: int8 量化 (bitsandbytes)
--------------------------------
| Backend    | Precision | avg_RTF | vs baseline | CER     | VRAM    |
|-----------|-----------|---------|-------------|---------|---------|
| Whisper   | fp16      | 0.0560  | baseline    | 1.075   | 1544MB  |
| Whisper   | int8      | 0.1060  | +89% 慢     | 1.275   | 854MB   |
| Qwen3-ASR | bf16      | 0.0950  | baseline    | -       | 3893MB  |
| Qwen3-ASR | int8      | 0.3794  | +299% 慢    | -       | 2263MB  |

结论: int8 在 4060 上严重负优化。
原因: Ada Lovelace 架构 fp16/bf16 tensor core 性能极强, int8 反而增加
     反量化开销, 吞吐下降。与之前 faster-whisper int8 慢 38% 的结论一致。
附带: Whisper int8 CER 退化 +0.200 (量化噪声)。
附带: 显存节省 42-45% (若 VRAM 是瓶颈可考虑, 但 4060 8GB 够用)。
推荐: 不采用。

Part D: Batch inference (意外高收益)
------------------------------------
| Backend    | Mode       | avg_RTF | Speedup | 文本一致性 |
|-----------|------------|---------|---------|-----------|
| Whisper   | sequential | 0.1733  | -       | -         |
| Whisper   | batch(5)   | 0.0381  | 4.55x   | 100%      |
| Qwen3-ASR | sequential | 0.1207  | -       | -         |
| Qwen3-ASR | batch(5)   | 0.0442  | 2.73x   | 100%      |

结论: 批量推理是最有效的加速手段 (3-5x)。
原理: GPU 并行处理多条音频, 摊薄 kernel launch 和内存搬运开销。
限制: 当前管线逐条处理 (diar + speaker extraction + ASR 串行),
     需重构才能利用 batch。Qwen 后端已支持 list 输入 (qwen_asr.transcribe)。
推荐: 最高优先级探索方向 (需管线改造)。

================================================================
  总结与推荐
================================================================

1. torch.compile / Flash Attention 2 / int8 量化:
   在 RTX 4060 上均无实际收益, 不推荐采用。
   根因: SDPA 已自动启用最优 kernel, 4060 fp16 tensor core 已满载。

2. Batch inference (批量推理):
   3-5x 加速, 是唯一显著有效的优化方向。
   - Qwen 后端: 已支持 batch, 可直接在 qwen_asr_backend.py 中改为批量调用
   - Vanilla Whisper: 需 pad 到相同长度, 可用 padding=True 实现
   - 管线改造: 需将 diar+extract+ASR 解耦, 先批量提取切片再批量转写

3. 当前瓶颈分布 (非模型推理):
   - diar (DiariZen): 每条需独立推理, 不可 batch
   - wespeaker embedding: 每条需独立推理
   - ASR generate: 可 batch, 当前逐条是主要效率损失

4. L20/L40 场景:
   - Batch 推理在大显存 GPU 上收益更大 (更大 batch size)
   - int8 可能在非 Ada Lovelace 架构上有收益 (如 T4/V100)
   - torch.compile 在更复杂模型上可能有更大收益
""")
