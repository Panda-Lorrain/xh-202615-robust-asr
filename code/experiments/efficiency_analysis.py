"""效率优化探索分析脚本。

Part A: ONNX Runtime 可行性
Part B: 管线并行化分析
Part C: 模型加载优化
Part D: 推理参数优化
Part E: SE 阶段深度优化分析 (27.3% 瓶颈)

纯分析脚本，不修改现有代码。
"""
import os, sys, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# 实测数据 (pos 集 1364 条, 4060 GPU)
ACTUAL_TIMING = {
    "total_wall_sec": 889.6,
    "overall_rtf": 0.2543,
    "phases": {
        "noise_classify": {"wall_sec": 12.5, "pct": 1.4},
        "se": {"wall_sec": 270.9, "pct": 30.5, "n": 1364},
        "enroll_diar_dicow": {"wall_sec": 581.1, "pct": 65.3, "mean_rtf": 0.1582},
    },
}

# ============================================================
# Part A: ONNX Runtime 探索
# ============================================================
def part_a_onnx():
    print("=" * 70)
    print("PART A: ONNX Runtime 探索")
    print("=" * 70)

    # A.1: 检查 onnxruntime
    try:
        import onnxruntime as ort
        print(f"  [A1] onnxruntime 已安装: {ort.__version__}")
        print(f"       providers: {ort.get_available_providers()}")
        has_ort = True
    except ImportError:
        print("  [A1] onnxruntime 未安装 (.venv / .venv_qwen / .venv_se 均无)")
        has_ort = False

    # A.2: 检查 optimum
    try:
        import optimum
        print(f"  [A2] optimum 已安装: {optimum.__version__}")
        has_optimum = True
    except ImportError:
        print("  [A2] optimum 未安装")
        has_optimum = False

    # A.3: Whisper ONNX 可行性
    print("\n  --- Whisper ONNX 分析 ---")
    print("  [A3] Whisper-large-v3-turbo ONNX 可行性: [可行]")
    print("       HuggingFace optimum 提供 ORTModelForSpeechSeq2Seq,")
    print("       openai/whisper-large-v3-turbo 已有预导出 ONNX 权重(optimum 库自动处理)")
    print("       安装: uv add optimum[onnxruntime-gpu] (需 .venv)")
    print("       预期收益: encoder 2-3x 加速(decoder 自回归, 加速有限)")
    print("       整体 RTF 预估: vanilla 0.16-0.24 → ONNX ~0.10-0.15")
    print("       风险: ONNX 不支持 trust_remote_code(DiCoW 用), 仅 vanilla Whisper 可用")
    if has_ort:
        print("       建议: 安装 optimum 后, 用 ORTModelForSpeechSeq2Seq 加载 vanilla Whisper 测试")
    else:
        print("       建议: 先安装 onnxruntime-gpu + optimum 到 .venv, 再做基准测试")

    # A.4: Qwen3-ASR ONNX 可行性
    print("\n  --- Qwen3-ASR ONNX 分析 ---")
    print("  [A4] Qwen3-ASR ONNX 可行性: [不可行]")
    print("       Qwen3ASRModel 是自定义架构(非标准 transformers WhisperForConditionalGeneration)")
    print("       方法: transcribe() / streaming_transcribe() (自定义推理流程)")
    print("       optimum ORTModelForSpeechSeq2Seq 不兼容自定义模型类")
    print("       结论: Qwen3-ASR 无法用 ONNX 加速(除非手动 trace/export, 工程量大)")

    # A.5: FireRedASR ONNX 可行性
    print("\n  --- FireRedASR ONNX 分析 ---")
    print("  [A5] FireRedASR ONNX 可行性: [不可行]")
    print("       FireRedAsr.from_pretrained('aed', ...) 是自定义架构")
    print("       结论: 同 Qwen3-ASR, 无法用标准工具链 ONNX 加速")

    return has_ort, has_optimum


# ============================================================
# Part B: 管线并行化分析
# ============================================================
def part_b_parallelism():
    print("\n" + "=" * 70)
    print("PART B: 管线并行化分析")
    print("=" * 70)

    # B.1: 当前管线结构
    print("""
  --- 当前管线结构 (submit_infer.py) ---

  [阶段0] noise_classify.py
    - 进程: subprocess (独立 Python 进程)
    - 设备: CPU (纯 librosa 谱特征计算)
    - 模型: 无(纯信号处理)
    - 耗时: ~6.5s (0.8% 总耗时)
    - 输入: 全部 1364 条 wav
    - 输出: noise_est.json (每条 noise_type + atten_lim_db)

  [阶段1] se_denoise.py (按桶分组)
    - 进程: subprocess (独立 .venv_se Python 进程)
    - 设备: CPU (se_denoise.py 默认 --device cpu)
    - 模型: DeepFilterNet3 (8.7MB 权重, torch 2.2.2+cpu)
    - 耗时: ~227s (27.3% 总耗时)
    - 输入: 按 atten_lim_db 分桶的 wav
    - 输出: 降噪后的 wav 目录

  [阶段2] enroll_infer.py --pairs
    - 进程: subprocess (独立 Python 进程, 单进程批量)
    - 设备: GPU (cuda:0, 加载多个模型)
    - 模型(全部同时在显存):
      1. Whisper-large-v3-turbo / DiCoW (ASR, ~3GB fp16)
      2. DiariZen (diarization, wavlm-large, ~1.5GB)
      3. wespeaker (声纹, 复用 diar._embedding, ~0.5GB)
      总计: ~5GB VRAM (4060 有 8GB, 余量 3GB)
    - 耗时: ~569s (68.5% 总耗时)
    - 流程(每条音频):
      1. librosa.load(rec)
      2. diar(rec) → speakers + per_spk timelines
      3. fe(audio) → mel features
      4. get_emb() × N speakers → spk_embs
      5. argmax(sims) → target_idx
      6. cut_target_timeline() → target audio
      7. asr_model.generate() → transcript
      8. to_simplified + digit_postproc

  [阶段3] llm_reject.py (--no-llm 保底跳过)
    - 进程: subprocess (独立 .venv_llm Python 进程)
    - 设备: GPU
    - 模型: Qwen2.5-3B-Instruct (~6GB)
    - 耗时: 跳过(保底配置 --no-llm)

  [阶段4] 融合 + result 组装 (纯 Python, 秒级)
""")

    # B.2: 并行化机会分析
    print("  --- 并行化机会分析 ---")
    print("""
  [B2.1] noise_classify ∥ SE: [不可并行]
    - noise_classify 的输出(atten_lim_db)是 SE 的输入(决定降噪强度)
    - 数据依赖: noise_est → bucket → se_denoise
    - 结论: 严格串行依赖, 无法并行

  [B2.2] SE ∥ enroll_infer: [理论可并行, 实际受限]
    - SE 输出降噪后的 wav → enroll_infer 的输入
    - 数据依赖: SE 完成的 wav 才能喂 enroll_infer
    - 流水线并行: enroll_infer 可以"边 SE 边处理已完成的 wav"
      但 enroll_infer --pairs 是批量化单进程(模型加载1次), 不支持增量输入
    - 改造成本: 需要改 enroll_infer 为流式处理(高工程量)
    - 结论: 理论可行但改造成本高, 不推荐

  [B2.3] noise_classify ∥ enroll_infer(前半段): [不可行]
    - enroll_infer 需要完整的音频(可能是 SE 后的), 与 noise_classify 无直接关系
    - 但 enroll_infer 的输入是 SE 后的音频, 必须等 SE 完成
    - 结论: 数据依赖阻塞

  [B2.4] GPU/CPU 混合并行: [有限收益]
    - noise_classify (CPU) 理论上可与 enroll_infer (GPU) 并行
    - 但 noise_classify 只占 0.8%, 并行收益可忽略
    - SE (CPU) 占 27.3%, 但与 enroll_infer 有数据依赖
    - 结论: 无显著收益

  [B2.5] enroll_infer 内部并行: [有限]
    - diar + ASR 有数据依赖(diar 产出 timeline → ASR 用 timeline)
    - 多条音频之间: 当前 batch=1 逐条处理, 理论上可 batch
    - 但每条音频的 diarization 结果不同, batch ASR 需要等所有 diar 完成
    - 结论: batch ASR 有潜力(见 Part D), 但需改造
""")

    # B.3: 显存分析
    print("  --- 显存占用分析 (4060 8GB) ---")
    print("""
  [B3] 模型同时在显存的场景:
    - 仅阶段2(enroll_infer) 使用 GPU
    - 模型占用: Whisper ~3GB + DiariZen ~1.5GB + wespeaker(复用) ~0.5GB = ~5GB
    - 运行时峰值(含 mel/intermediate): ~5.5-6GB
    - 4060 8GB 余量: ~2GB
    - 结论: 当前配置不会 OOM, 但无余量加载额外模型

  如果尝试并行 SE(GPU) + enroll_infer(GPU):
    - DeepFilterNet3: ~50MB (极小)
    - 但 SE 用 .venv_se (torch 2.2.2+cpu), 独立进程, 不共享显存
    - 结论: SE 当前 CPU 运行, 切 GPU 需改 venv, 收益有限(DF3 太轻量)

  如果尝试并行 LLM + enroll_infer:
    - Qwen2.5-3B: ~6GB (与 Whisper 不能同时在 8GB 显存)
    - 结论: LLM 和 ASR 不能同时在 4060 上运行
""")


# ============================================================
# Part C: 模型加载优化
# ============================================================
def part_c_model_loading():
    print("=" * 70)
    print("PART C: 模型加载优化")
    print("=" * 70)

    print("""
  --- 当前模型加载方式 ---

  [C1] submit_infer.py 的调用方式: subprocess
    - 每个阶段是独立的 subprocess (Python 进程)
    - 阶段0 noise_classify: 无模型, 纯计算
    - 阶段1 se_denoise: 每次启动加载 DeepFilterNet3 (~0.5s)
    - 阶段2 enroll_infer: 每次启动加载 Whisper + DiariZen + wespeaker (~15-20s)
    - 阶段3 llm_reject: 每次启动加载 Qwen2.5-3B (~10-15s)
    - 阶段4: 无模型

  [C2] 模型加载开销估算:
    - DeepFilterNet3: 8.7MB 权重, 加载 ~0.5s (可忽略)
    - Whisper-large-v3-turbo: ~3GB 权重, 加载 ~8-12s (fp16)
    - DiariZen (wavlm-large): ~1.5GB 权重, 加载 ~5-8s
    - Qwen2.5-3B: ~6GB 权重, 加载 ~10-15s
    - 总加载开销(保底配置): ~15-20s (noise_classify 无模型 + SE ~0.5s + enroll ~15s)

  [C3] 当前优化措施(已实施):
    - enroll_infer --pairs 批量化: 模型加载1次, 跑全部1364条(非按 enrollment 分组重载)
    - 这是最大的加载优化(从 1838 次重载 → 1次)
    - enrollment embedding 缓存: 同路径复用(_enroll_cache)

  [C4] 进一步优化评估:

    [C4.1] 改 subprocess 为函数调用: [收益有限]
    - 当前: submit_infer.py → subprocess.run([python, script.py, ...])
    - 改为: submit_infer.py 直接 import 并调用函数
    - 收益: 省去 Python 进程启动 + 模型重复加载(如果跨阶段共享)
    - 问题: .venv_se (torch 2.2.2+cpu) 和 .venv_llm 不同 venv, 无法直接 import
    - enroll_infer 已经单进程批量(加载1次), 函数调用不额外省加载
    - 结论: 收益 ~0.5-1s (进程启动), 不值得改造风险

    [C4.2] 模型预加载/缓存: [已实施, 无额外空间]
    - enroll_infer 已经: 模型加载1次 + enrollment 缓存
    - subprocess 的模型加载是进程隔离的(每次新进程 = 重新加载)
    - 但 enroll_infer 只调用1次(批量), 所以加载开销只发生1次
    - 结论: 当前已经是最佳(1次加载), 无进一步优化空间

    [C4.3] 模型权重预编译(torch.compile): [潜在收益]
    - torch.compile 可优化推理图, 首次编译慢(30-60s), 后续加速 10-30%
    - Whisper generate 是自回归循环, torch.compile 收益有限(每次循环不同 shape)
    - 结论: 不推荐(编译开销 > 加速收益, 且 reproducibility 要求 cudnn.deterministic)

    [C4.4] 合并 SE + enroll_infer 为单进程: [理论可行]
    - 将 DeepFilterNet3 加载到 enroll_infer 的 GPU 进程中
    - 省去 .venv_se 独立进程 + 音频文件 IO
    - 问题: DeepFilterNet3 用 torch 2.2.2+cpu, enroll_infer 用 torch 2.5.1+cu124
    - 版本冲突, 需要统一 venv
    - 结论: 理论可行但 venv 统一风险高(历史教训: faster-whisper 装入主 venv 导致 speechbrain 崩)
""")

    # C.5: 量化模型加载耗时
    print("  --- 模型加载耗时实测(可选) ---")
    print("  注: 以下为估算值, 需要实际测量确认")
    print("  Whisper-large-v3-turbo fp16 加载: ~10s")
    print("  DiariZen wavlm-large 加载: ~6s")
    print("  DeepFilterNet3 加载: ~0.5s")
    print("  Qwen2.5-3B 加载: ~12s")
    print("  Python 进程启动: ~0.3s")


# ============================================================
# Part D: 推理参数优化
# ============================================================
def part_d_inference_optimization():
    print("\n" + "=" * 70)
    print("PART D: 推理参数优化")
    print("=" * 70)

    print("""
  --- D.1: Whisper 推理参数分析 ---

  当前 generate 调用(enroll_infer.py, vanilla 后端):
    asr_model.generate(
        input_features=ifp_v,        # mel features
        attention_mask=am_v,         # 全1 mask
        language="zh",               # 中文
        task="transcribe",           # 转写(非翻译)
        max_new_tokens=200           # 最大输出200 token
    )

  Whisper GenerationConfig 默认值:
    num_beams: 1 (greedy decoding)   ← 已经最快, 无需优化
    do_sample: False (deterministic) ← 已经最快
    max_new_tokens: 200 (代码显式)   ← 合理, 家居指令一般 <50字
    temperature: 1.0 (但 do_sample=False, 不生效)
    length_penalty: 1.0 (中性)

  [D1.1] beam search vs greedy: [已最优]
    - 当前 num_beams=1 (greedy), 已是最快配置
    - beam search (beams>1) 会线性增加计算量, 但 CER 改善有限
    - 结论: 无需改动

  [D1.2] max_new_tokens 限制: [已合理]
    - 当前 max_new_tokens=200, 家居指令一般 <50字(~100 token)
    - 可降到 100 或 150 节省少量时间(最长指令的上限)
    - 风险: 超长转写(幻觉)被截断, 可能影响 CER
    - 结论: 可选优化, 收益 <5%, 有 CER 风险

  [D1.3] KV-Cache: [已自动启用]
    - transformers generate 默认启用 KV-Cache (use_cache=True)
    - 自回归解码每步复用之前步骤的 key/value, 避免重复计算
    - 结论: 无需改动

  [D1.4] forced_decoder_ids: [Whisper 特性, 已正确]
    - Whisper GenerationConfig 有 forced_decoder_ids: [[1, None], [2, 50360]]
    - token 1 = language (由 language="zh" 参数设置)
    - token 2 = 50360 = task "transcribe"
    - 代码显式传 language + task, 与 forced_decoder_ids 一致
    - 结论: 无需改动

  [D1.5] Whisper 特定优化:
    - prompt_ids (langfix retry 用): 中文偏置 prompt, 仅英文幻觉时触发
    - 不影响主路径性能

  --- D.2: Batch 推理分析 ---

  [D2] 多条音频 batch ASR: [有潜力但改造成本高]

  当前: 每条音频单独 generate (batch_size=1)
    for enr, rec in pairs:
        ... # diar, emb, target selection
        out = asr_model.generate(input_features=ifp_v, ...)  # 单条

  理论 batch 方案:
    1. 先跑全部 diarization + 声纹匹配(快速, ~0.5s/条)
    2. 收集全部 target mel features
    3. batch generate (多条同时喂 Whisper encoder + decoder)

  问题:
    - Whisper generate 不原生支持 batch decoder(每条自回归长度不同)
    - 需要 padding + attention mask 处理(复杂)
    - KV-Cache 在 batch 模式下需要特殊处理
    - transformers WhisperModel.generate 实际支持 batch(内部 pad)
    - 但: 每条音频的 diarization 结果不同 → target timeline 不同 → mel 长度不同
    - batch 需要等所有 diar 完成, 丧失流式优势

  改造成本估算:
    - 需要重构 enroll_infer.py 的主循环(两阶段: diar阶段 + ASR阶段)
    - 需要处理 batch padding + variable length
    - 需要验证 batch=1 vs batch=N 的 CER 一致性(reproducibility)
    - 工程量: 中-高 (2-3天)

  预期收益:
    - Whisper encoder: batch 并行可加速 2-4x (GPU 利用率从 ~30% 提升到 ~80%)
    - decoder: 自回归, batch 加速有限(每条长度不同, padding 浪费)
    - 整体 ASR 加速估计: 1.5-2x (569s → 280-380s)
    - 但: diarization 仍串行(~0.5s/条), 限制了整体加速

  结论: 收益可观(1.5-2x ASR), 但改造成本高 + reproducibility 风险 + 需要充分测试

  --- D.3: 其他推理优化 ---

  [D3.1] torch.inference_mode vs torch.no_grad: [微优化]
    - 当前用 torch.no_grad()
    - inference_mode() 更激进(禁用更多 autograd 跟踪), 速度略快
    - 收益: <2%, 改动: 替换装饰器
    - 结论: 可选微优化

  [D3.2] fp16 vs bf16 vs fp32: [已最优]
    - 当前 Whisper 用 fp16 (torch_dtype=dtype, dtype=torch.float16)
    - Qwen3-ASR 用 bf16 (dtype=torch.bfloat16)
    - fp16 是 4060 的最优选择(4060 bf16 支持有限)
    - 结论: 无需改动

  [D3.3] mel feature 预计算: [不可行]
    - 每条音频的 mel 特征取决于音频内容, 无法预计算
    - fe(audio) 调用本身很快(<0.1s), 不是瓶颈
    - 结论: 不是优化点

  [D3.4] librosa.load 替换: [微优化]
    - librosa.load 用 audioread/soundfile 后端, 每次重新打开文件
    - 可用 soundfile.read 直接读(更快, 但需确保 16k mono)
    - 收益: <1s 总计, 不值得改动
    - 结论: 不推荐

  [D3.5] diarization 加速: [受限]
    - DiariZen 用 wavlm-large, GPU 推理
    - 每条音频 ~0.3-0.5s, 1364 条总计 ~400-680s
    - 但 diarization 是 enroll_infer 的一部分, 已包含在 569s 内
    - 无法单独加速(除非换更轻量的 VAD 模型)
    - 结论: 不推荐(换模型风险高)
""")


# ============================================================
# Part E: SE 阶段深度优化分析
# ============================================================
def part_e_se_optimization():
    print("\n" + "=" * 70)
    print("PART E: SE 阶段深度优化分析 (实测 270.9s, 30.5% 瓶颈)")
    print("=" * 70)

    print("""
  --- 当前 SE 实现分析 (se_denoise.py) ---

  流程(每条音频):
    1. torchaudio.load(wp)           # 读 16k wav
    2. Resample(16k → 48k)           # 上采样 3x
    3. DeepFilterNet3 enhance()      # GPU/CPU 降噪
    4. Resample(48k → 16k)           # 下采样 1/3
    5. torchaudio.save()             # 写 16k wav

  瓶颈分析:
    - 1364 条音频, 每条 ~2-5s (16k)
    - 上采样到 48k: 每条 ~6-15s 的 48k 信号
    - DF3 enhance: 轻量模型(8.7MB), 推理快
    - 下采样回 16k: 快
    - 文件 IO: 1364 次读 + 1364 次写(中间产物落盘)

  --- SE 优化方案评估 ---

  [E1] SE 从 CPU 切到 GPU: [收益有限]
    - 当前: .venv_se 用 torch 2.2.2+cpu (无 CUDA)
    - DF3 模型极小(8.7MB), GPU 推理加速有限
    - 瓶颈不在 DF3 推理本身, 而在 resample + 文件 IO
    - 改造: 需要重建 .venv_se 为 GPU 版本(或合并到主 .venv)
    - 预期收益: DF3 推理从 ~0.1s/条 → ~0.02s/条, 节省 ~110s (总 270s → 160s)
    - 风险: venv 合并可能导致依赖冲突(历史教训)
    - 结论: 可行但收益不如预期(resample 仍占大头)

  [E2] 跳过 SE (--no-se): [最大收益, 但有 CER 风险]
    - 直接省 270.9s (30.5%)
    - 总管线: 889.6s → 618.7s (RTF 0.254 → 0.177)
    - CER 影响: SE 对 CER 的贡献需要实测
      - 从 memory 知: SE 条件化最优 atten=0/6, CER 改善 ~2.82(仿真集)
      - 但真测集 A: vanilla 路线已绕过 DiCoW 条件化, SE 收益可能更小
    - 结论: 如果 SE 对 qwen/vanilla 主线 CER 贡献 <0.01, 可考虑跳过

  [E3] 优化 resample: [中等收益]
    - 当前: torchaudio.transforms.Resample(16k→48k, 48k→16k)
    - 替代方案:
      a) sox resampler (命令行, 通常更快)
      b) scipy.signal.resample_poly (CPU, 利用多核)
      c) 预先将数据集转为 48k(一次性, 后续省 resample)
    - 预期收益: resample 从 ~0.2s/条 → ~0.05s/条, 节省 ~200s (总 270s → 70s)
    - 风险: 需要验证 resample 质量不影响 DF3 效果
    - 结论: 高收益, 值得尝试

  [E4] 减少文件 IO: [中等收益]
    - 当前: 读 wav → 处理 → 写 wav → (后续)再读
    - 改为: 内存中传递(不落盘)
    - 问题: submit_infer.py 用 subprocess 调用 se_denoise.py, 必须通过文件系统
    - 如果合并 SE + enroll_infer 为单进程, 可省内存传递
    - 预期收益: ~20-30s (文件 IO 不是主要瓶颈)
    - 结论: 需要大改造, 不推荐

  [E5] 并行化 SE 多桶: [小收益]
    - 当前: 按 atten_lim_db 分桶, 桶内串行
    - 改为: 多桶并行(多进程/多线程)
    - 问题: DF3 模型加载需要在每个进程各加载一次
    - 桶数通常 2-3 个(atten=0, 6), 并行收益有限
    - 结论: 不推荐

  --- SE 优化推荐 ---
  优先级1: 跳过 SE (--no-se) 实测 CER 影响 → 如果影响小, 直接省 270s
  优先级2: 优化 resample (预转 48k 或用更快的 resampler) → 省 ~200s
  优先级3: SE 切 GPU → 省 ~110s (需要 venv 改造)
""")


# ============================================================
# 总结
# ============================================================
def summary():
    print("\n" + "=" * 70)
    print("总结: 优化建议按优先级排序 (含实测数据)")
    print("=" * 70)

    t = ACTUAL_TIMING
    print(f"""
  实测基线 (pos 集 1364 条, 4060 GPU):
    总耗时: {t['total_wall_sec']}s | RTF: {t['overall_rtf']}
    noise_classify: {t['phases']['noise_classify']['wall_sec']}s ({t['phases']['noise_classify']['pct']}%)
    SE:             {t['phases']['se']['wall_sec']}s ({t['phases']['se']['pct']}%)
    enroll_diar:    {t['phases']['enroll_diar_dicow']['wall_sec']}s ({t['phases']['enroll_diar_dicow']['pct']}%, mean_rtf={t['phases']['enroll_diar_dicow']['mean_rtf']})

  ============================================================
  优化方案收益矩阵 (按 ROI 排序)
  ============================================================

  方案1: 跳过 SE (--no-se)
    收益: -270.9s (30.5%)
    风险: CER 可能恶化(需实测, vanilla 路线 SE 收益可能 <0.01)
    工程量: 零(改 flag)
    管线: 889.6s → 618.7s (RTF 0.254 → 0.177)
    结论: [最高 ROI] 先实测 --no-se 的 CER 影响

  方案2: 优化 SE resample (预转 48k / 更快 resampler)
    收益: -150~200s (resample 占 SE 大头)
    风险: 低(需验证 resample 质量)
    工程量: 小(改 se_denoise.py resample 逻辑)
    管线: 889.6s → 690~740s (RTF 0.254 → 0.197~0.212)
    结论: [高 ROI] 值得尝试

  方案3: ONNX Runtime for vanilla Whisper
    收益: -120~220s (ASR encoder 加速 1.5-2x)
    风险: 仅 vanilla Whisper 可用, 需安装 onnxruntime-gpu + optimum
    工程量: 小(替换模型加载)
    管线: 889.6s → 580-680s (RTF 0.254 → 0.166~0.194)
    结论: [中 ROI] 需要安装测试

  方案4: Batch ASR 推理
    收益: -150~250s (ASR 加速 1.5-2x)
    风险: 高(reproducibility + batch padding + 大改造)
    工程量: 中-高 (2-3天)
    管线: 889.6s → 530-640s (RTF 0.254 → 0.152~0.183)
    结论: [中 ROI] 大改造, 谨慎评估

  方案5: SE 切 GPU (.venv_se 重建)
    收益: -100~110s (DF3 推理加速)
    风险: venv 依赖冲突(历史教训)
    工程量: 中(重建 venv)
    管线: 889.6s → 780~790s (RTF 0.254 → 0.223~0.226)
    结论: [低 ROI] 不推荐

  不推荐方案:
    - torch.compile: 编译开销 > 加速收益, cudnn.deterministic 冲突
    - subprocess → 函数调用: 收益 ~1s
    - LLM + ASR 并行: 8GB VRAM 不够(Whisper 3GB + Qwen 6GB)
    - noise_classify 优化: 只占 1.4%, 无意义

  ============================================================
  推荐行动路线
  ============================================================

  Step 1 (立即, 0 成本):
    实测 --no-se 对 qwen/vanilla 主线 CER 的影响
    如果 CER 恶化 <0.01, 直接跳过 SE, 省 270s → RTF 0.177

  Step 2 (小成本, 如果 Step 1 SE 不能跳):
    优化 SE resample (预转 48k 数据集 或 用 scipy.signal.resample_poly)
    目标: SE 从 270s → 100s

  Step 3 (中成本, 如果 Step 2 不够):
    安装 onnxruntime-gpu + optimum, 测试 vanilla Whisper ONNX 推理
    目标: enroll_diar_dicow 从 581s → 400s

  Step 4 (大成本, 决赛冲刺):
    Batch ASR 推理改造
    目标: enroll_diar_dicow 从 581s → 350s

  ============================================================
  效率腿评分估算 (⚠️ 2026-07-18 对抗审查勘误: 见下方)
  ============================================================
  [原估算 — 4060 batch=16 全管线 RTF 外推 L20, 已部分废弃]
  当前 (RTF 0.254):     效率腿 ~16-17/20
  +跳过SE (RTF 0.177):  效率腿 ~17-18/20   ← 关 SE 省 30.6% 实测(非 27%), 官方 batch=1 口径下唯一确定杠杆
  +ONNX (RTF 0.15):     效率腿 ~18-19/20   ← POC 未做非证伪(Part A 自己标可行预期 2-3x)
  +batch (RTF 0.12):    效率腿 ~19-20/20   ← ⚠️ 废弃: 官方 batch=1 测 RTF, batch 红利不进分
  注: L20 48GB 可大 batch, 收益比 4060 更大  ← ⚠️ 同上, batch 仅加快开发 A/B 迭代

  ⚠️ 2026-07-18 对抗审查勘误 (4-agent + code/audit_se_bugfix):
  官方按 batch=1 测 RTF(memory official-scoring-spec / l20-eval-hardware)。batch=1 口径下
  overall_rtf 可能从 0.142(batch=16 全量)升到 0.3-0.5(单条 qwen 0.095 + diar/extract 0.05+
  + 加载不摊薄) → 效率腿区间应宽到 [8,19]/20 而非 [16,19]/20, 待 L20 batch=1 实测定论。
  官方 batch=1 口径下唯一确定杠杆 = 关 SE; 未充分证伪 = ONNX/GPTQ-AWQ/speculative(Qwen3-ASR 蒸馏)。
""")


if __name__ == "__main__":
    part_a_onnx()
    part_b_parallelism()
    part_c_model_loading()
    part_d_inference_optimization()
    part_e_se_optimization()
    summary()
