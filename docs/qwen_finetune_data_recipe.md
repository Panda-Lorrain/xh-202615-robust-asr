# Qwen3-ASR 微调数据配方 + 外部数据盘点 + 算力估算

> **2026-07-27** · 外部训练已解禁(主办方 2026-07-27 确认, memory `external-training-allowed`)。
> 主线 ASR = Qwen3-ASR-1.7B (中文原生, CER 0.3436)。
> 死区 sim<0.4 (78.8% 大头) 中 **60% 失败是「ASR 错 + 接近地板」**, 细分:
>   ① **ASR 幻觉** (清洁→春洁, 音频清晰却转错)
>   ② **小声/被盖/重叠** (人耳可辨机器不能, 如 cmd_2096/2949/2571/2001)
> 本配方针对性合成训练对, 微调 Qwen3-ASR 攻这两类。
>
> 配套脚本: `code/data_aug_recipe.py` (已小样本 4 对自测通过, ~0.8 pair/s)。

---

## 1. 外部数据盘点

### 1.1 本机已有 (E:/)

| 资产 | 路径 | 规模 | 用途 |
|---|---|---|---|
| **Qwen3-ASR-1.7B** (微调目标) | `E:/hf_cache/Qwen3-ASR-1.7B/` | 4.4 GB | bf16 权重 + qwen_asr 推理代码 (venv_qwen) |
| FireRedASR-AED-L / LLM-L | `E:/hf_cache/FireRedASR-*` | - | 备选 ASR 后端 |
| Whisper-large-v3-turbo | `E:/hf_cache/whisper-large-v3-turbo/` | - | vanilla 主线 |
| SE_DiCoW / DiCoW_v3_2 | `E:/hf_cache/SE_DiCoW/`, `DiCoW_v3_2/` | - | 旧后端 |
| **datasetA 真测集** (禁训练, hold-out) | `E:/midea_target_asr/datasetA/` | pos 1364 + neg 474 | 仅作 hold-out 验证 + 词表提炼 |
| 程序噪声 white/pink/babble | `code/build_dataset.py` | 0 (代码内合成) | 增广噪声基线 (无外部依赖) |
| 中文 target/nontarget 测试音频 | `test_wav/zh_target_*.wav` (4), `zh_nontarget_*.wav` (2) | ~3s 各 | smoke 自测输入 |
| 现有 venv (主) | `code/.venv/` | torch 2.5.1+cu124, transformers 4.42.4, librosa/soundfile/datasets/cn2an/bitsandbytes | 缺 peft |
| 现有 venv (qwen) | `code/.venv_qwen/` | torch 2.6.0+cu124, transformers 4.57.6, qwen_asr, bitsandbytes | **缺 peft/datasets**, 微调用 |

**本机没有** (需下载): 中文干净语音 (Aishell 系列 / WenetSpeech / Tal ASR / Primeword / LibriTTS-zh), 真实噪声库 (MUSAN / WHAM! / WHAMR! / DNS-Challenge), 现成重叠数据 (Aishell1Mix / LibriMix 需脚本生成)。

### 1.2 需下载清单 (公开可用, license 标注)

#### A. 中文干净语音 (target + interferer 源)
| 数据集 | 规模 | license | 地址 | 用途 |
|---|---|---|---|---|
| **Aishell-1** | 178 h, 400 spk | Apache-2.0 | `kaldi-res/openslr.org/33/` / HF `kresnik/aishell` | target 干净 wav (主力, 中文) |
| **Aishell-2** | 1000 h, 1991 spk | 学术免费(需申请) | `aishell2.azurewebsites.net` | 大规模 target, 拒绝申请则用 Aishell-1 |
| **Aishell-3** | 85 h, 218 spk, **多说话人 TTS 数据** | CC BY-NC-SA 4.0 | `openslr.org/93/` / HF `voiceinspace/AISHELL-3` | **多声纹** (target+interferer 池, 模拟题目 20 种唤醒词) |
| **WenetSpeech** | 10000 h | CC BY-NC 4.0 (学术) | `wenet.org.cn/wenetspeech/` | 干扰源 (新闻/财经/闲聊, 非家居指令) |
| **PrimeWord-SC** | 99 h, 287 spk | 学术免费 | `openslr.org/47/` | 备用 target |
| **Tal ASR (MAD-TED)** | 26 h | CC BY-NC 4.0 | `openslr.org/77/` | 备用 target |
| **MagicData-RAMC** | 180 h 对话 | CC BY-NC-ND 4.0 | `openslr.org/123/` | **多人对话** (天然重叠) |

**优先推荐**: Aishell-1 (零成本, Apache) + Aishell-3 (多声纹 CC-BY-NC, 适合模拟题目 20 种唤醒词) + WenetSpeech 子集 (干扰)。

#### B. 噪声库 (背景噪声, 模拟 SNR −5~5dB)
| 数据集 | 规模 | license | 地址 | 用途 |
|---|---|---|---|---|
| **MUSAN** | 6 h noise + 5 h music + 5 h speech | CC BY 4.0 | `openslr.org/17/` | **首选**, env_noise + babble 真实源 |
| **WHAM!** | 80 h 真实环境噪声 | CC BY-NC 4.0 | `wham.whisper.ai` | env 噪声 (中文场景近似) |
| **WHAMR!** | WHAM + 混响 | CC BY-NC 4.0 | `whamr.whisper.ai` | 加混响 (题目可能含) |
| **DNS-Challenge** | 150+ h noise | CC BY 4.0 (部分) | `github.com/microsoft/DNS-Challenge` | 备用噪声 |

**优先推荐**: MUSAN (零成本 CC BY, 直接用) + WHAM! (更真实)。

#### C. 重叠数据 (现成 / 半现成)
| 数据集 | 规模 | license | 地址 | 备注 |
|---|---|---|---|---|
| **Aishell1Mix** | Aishell-1 衍生 | 同 Aishell-1 | 脚本生成 `github.com/JorisCos/LibriMix` (支持 aishell) | **推荐自己生成** (可控重叠率) |
| **LibriMix** | LibriSpeech 衍生 | CC BY 4.0 | `github.com/JorisCos/LibriMix` | 英文重叠, 仅作预训练参考 |

**推荐**: 不直接用现成重叠集 (题目需要 0–100% 重叠 × −5~5dB SNR × 家居指令 三维度精确可控), 用本配方脚本自己合成。

---

## 2. 增广配方 (对应 `code/data_aug_recipe.py`)

### 2.1 输入 / 输出

```
输入 (每条训练对):
  target_wav       干净中文单人音频 A (家居指令, 来自 Aishell-1/3)
  target_ref       A 的原始转录 (ground truth 文本)
  interferer_wav   干扰单人音频 B (闲话/新闻/财经, 非家居指令, 来自 WenetSpeech)
  noise_wav        噪声 (来自 MUSAN/WHAM!, 或程序噪声 white/pink/babble)

输出:
  enrollment.wav   ~1.5–2.5s (题目规格 ~1.8s), 从 A 切
  recognition.wav  A+B 重叠 (0–100%) + 加噪 (−5~5dB), A 可选小声化/快语速
  ref              A 的原始 transcript (微调 label)
  manifest.jsonl   所有增广参数 (可追溯/消融)
```

### 2.2 增广链 (5 步, 直击失败分布)

| 步骤 | 函数 | 失败模式对应 | 参数范围 |
|---|---|---|---|
| ① target 小声化 | `make_quiet` | cmd_2096/2949 类 (小声被盖) | gain −8~−3 dB (50% 触发) + 概率 0.3 叠 lowpass(3.2kHz) 模拟闷声 |
| ② target 快语速 | `make_fast` | cmd_2050 类 | librosa.time_stretch rate 1.1~1.4 (30% 触发) |
| ③ enrollment 切片 | `cut_enrollment` | 题目 ~1.8s 短 enrollment | 从 A 随机起点切 1.5–2.5s |
| ④ enrollment 污染 | `pollute_enrollment` | 题目 enrollment 可能含噪 | 概率 0.3 加噪 SNR 8~15 dB |
| ⑤ 重叠 + 加噪 | `mix_overlap` + `add_noise` | cmd_2637 类 (双人重叠) | overlap 0/25/50/75/100%, SNR −5~5dB |

### 2.3 失败分布对齐的采样权重 (来自 memory `overlap-is-cer-failure-rootcause`)

死区失败组分布: **重叠中位 45%, 75% 重叠, 97% 双人**, babble 是主因。采样权重设:

```
overlap_buckets = [0.0, 0.25, 0.5, 0.75, 1.0]
overlap_weights = [0.10, 0.10, 0.25, 0.30, 0.25]   # 偏中高重叠 (失败组主战场)

snr_buckets = [-5, -3, 0, 3, 5]
snr_weights = [0.20, 0.25, 0.30, 0.15, 0.10]       # 偏低 SNR (题目规格)

noise_types = [white, pink, babble]
noise_weights = [0.25, 0.25, 0.50]                  # babble 多 (题目主因)

quiet_prob  = 0.5;  quiet_db_range  = (-8, -3)      # 50% 训练对小声化
fast_prob   = 0.3;  fast_rate_range = (1.1, 1.4)    # 30% 训练对快语速
```

### 2.4 家居指令词表 (从 datasetA/pos.jsonl 模板提炼, 不取真实音频)

10 类: 空调 / 灯 / 洗衣机 / 温度 / 风速 / 窗帘 / 音乐 / 模式 / 通用。
配套 `sample_home_cmd(rng)` 拼装随机指令模板 (供后续 TTS 合成专属家居指令或文本替换参考)。当前骨架默认沿用 A 的原始 transcript (外部 Aishell 的文本非家居, 用模板替换后续可加)。

### 2.5 复用现有代码

- `simulate_pipeline.add_noise / mix_overlap / _fit_noise` (mixing 主力)
- `build_dataset.gen_white / gen_pink / gen_babble / load_env_noise` (噪声生成)

### 2.6 小样本自测结果 (2026-07-27)

```
$ code/.venv/Scripts/python.exe code/data_aug_recipe.py smoke --n 4
[build_pairs] 共 4 训练对 → code/_aug_smoke_out (耗时 5s)
[smoke] OK: 生成 4 训练对
```

manifest 抽样 (验证参数分布合理):
- `zh_target_01_k000`: overlap 0.75, SNR −5dB, pink 噪声, target −4.3dB 小声化
- `zh_target_01_k001`: overlap 0.75, SNR 0dB, babble 噪声
- `zh_target_02_k000`: overlap 0.5, SNR −3dB, babble, target −6dB 小声化
- `zh_target_02_k001`: overlap 0.5, SNR 0dB, babble, target −7dB + 1.30x 快语速

enrollment 时长 1.6–2.4s (题目 ~1.8s ✓), recognition 1.6–2.6s (合理), 速率 **~0.8 pair/s** (4060 CPU 单线程, 含 librosa.load + 合成 + 写盘)。

---

## 3. 微调方案

### 3.1 选型: LoRA 优先, full-FT 备选

| 维度 | LoRA (推荐先试) | Full FT |
|---|---|---|
| 显存 (Qwen3-ASR-1.7B, 4060 8GB) | **~5–6 GB 可行** (8-bit Adam + bf16 模型) | ~24 GB (4060 不可行) |
| 训练速度 | 快 (只训少量参数, ~1% 总参) | 慢 |
| 收敛上限 | 略低于 full (差 1–3%) | 上限略高 |
| 灾难性遗忘风险 | 低 (基座参数不动) | 高 |
| 落地复杂度 | peft + transformers 标准 API | accelerate + DDP |

**LoRA 优先理由**: ①4060 8GB 限制, ②任务窄 (小声/重叠增强, 不是换语种), ③hold-out A 集验证泛化, 防过拟合。

### 3.2 框架 & 依赖

```
venv_qwen (微调用):
  torch 2.6.0+cu124     ✓ 已装
  transformers 4.57.6   ✓ 已装 (Qwen3-ASR 兼容)
  accelerate 1.12.0     ✓ 已装
  bitsandbytes 0.49.2   ✓ 已装 (8-bit optim + QLoRA)
  peft                  ✗ 需补装: uv pip install peft
  datasets              ✗ 需补装: uv pip install datasets
  librosa 0.11.0        ✓ 已装
```

补装命令 (按全局规则用 uv, 在 venv_qwen 内):
```bash
code/.venv_qwen/Scripts/python.exe -m pip install --user peft datasets
# 或 uv 方式:
# uv pip install --python code/.venv_qwen/Scripts/python.exe peft datasets
```

### 3.3 LoRA 超参 (起点值, 需小规模 sweep)

```python
# 推荐起点 (基于 Qwen3-ASR 1.7B + 窄域增强任务经验)
lora_r = 16                # rank, 8/16/32 sweep
lora_alpha = 32            # = 2 × r
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # attention
                  "gate_proj", "up_proj", "down_proj"]     # MLP
lr = 1e-4                  # LoRA 典型, 5e-5 ~ 2e-4 sweep
epochs = 3
batch_size = 1             # 4060 8GB 限制
grad_accum = 16            # 等效 batch=16
warmup_ratio = 0.05
weight_decay = 0.01
bf16 = True                # 4060 支持 bf16
optim = "paged_adamw_8bit" # bitsandbytes, 8-bit Adam 省 50% 显存
gradient_checkpointing = True   # 必开, 换时间省显存
seed = 42
```

### 3.4 微调流程 (hold-out A 集, lessons-pitfalls §14)

```
数据划分 (绝不训练 A 集):
  外部增广训练集 (合成): 80% train + 20% val (合成集内部)
  A 集 (datasetA/pos 1364): hold-out, 只评测, 绝不进训练

微调步骤:
  1) 合成增广集 (data_aug_recipe.py build, ~1万–10万对)
  2) LoRA 微调 Qwen3-ASR-1.7B (venv_qwen + peft)
  3) 合成 val 集上选 checkpoint (loss + CER 双指标)
  4) A 集 hold-out 评测 (官方口径 normalize_text + CERMetric, eval_metrics.py)
  5) 与基线 (主线 CER 0.3436) 对比, 若 ΔCER ≤ −0.01 才上线 (考虑 CER ±0.04 噪声)

评估对照:
  - 基线 (未微调): 主线 transcribe CER 0.3436 / 含拒 0.5934
  - 微调后: 期望死区桶 (sim<0.4) CER ↓ + 整体 CER ≤ 0.30
  - 失败子集验证: cmd_2096/2949/2571/2001/2050 等 (小声/快语速/重叠) 单独评测
```

---

## 4. 数据量 & 算力估算

### 4.1 单对合成耗时 (本机 4060 实测)

```
合成速率: ~0.8 pair/s (librosa.load + 重叠 + 加噪 + 写 wav, 单线程 CPU)
含 target/interferer 双加载 + babble 多 nontarget 加载摊薄: ~0.5 pair/s (保守)

数据量预估:
  1 万对  ≈ 3.5 h (本机 4060 CPU)
  10 万对 ≈ 35  h (本机, 隔夜跑)
  100 万对 ≈ 14 天 (需租云, 或多进程并行 → ~2 天)
```

**推荐规模**: **1–3 万对** 起跑 (覆盖 overlap 5 桶 × SNR 5 桶 × noise 3 类 × 小声/快语速 × ~30 spk)。验证收益后再决定是否扩到 10 万。

### 4.2 微调 GPU 时长

| 阶段 | 4060 8GB | L20 48GB (官方评测机) | A100 40GB |
|---|---|---|---|
| 1 万对 LoRA 3 epoch | ~2 h (估计) | ~30 min | ~15 min |
| 10 万对 LoRA 3 epoch | ~20 h (隔夜) | ~5 h | ~2.5 h |
| 10 万对 full-FT | **不可行** (OOM) | ~20 h | ~8 h |

**显存明细 (4060, 1.7B 模型 LoRA + bf16 + grad ckpt + 8-bit Adam)**:
- 模型 bf16: ~3.4 GB
- LoRA 可训参 + Adam state: ~0.2 GB
- 激活 (grad ckpt, batch=1): ~1–2 GB
- 音频特征 + KV cache: ~0.5 GB
- 合计: **~5–6 GB, 4060 8GB 可跑**

### 4.3 成本 (4060 主跑, 必要时租 L20/A100)

```
4060 (本机): 免费, 1万对 LoRA ~2h, 10万对 ~20h 隔夜
租 AutoDL L20 (按 memory efficiency-portability-audit):
  ~¥3/h, 10万对 LoRA ~5h = ¥15
租 A100 40GB:
  ~¥8/h, 10万对 LoRA ~2.5h = ¥20, full-FT ~8h = ¥64
```

**结论**: **4060 本机跑 LoRA 完全可行**, 仅 full-FT 或 100 万对以上才需租卡。

---

## 5. 风险 & 防控

### 5.1 外部数据域匹配风险
- **风险**: Aishell-1 (朗读式干净录音) 与题目 (实际家居场景 + 真实噪声 + 多说话人) 域差异大, 合成训练可能不迁移。
- **防控**: ① 优先 Aishell-3 (多说话人) + MagicData-RAMC (对话) 减小域差; ② babble + env_noise (MUSAN/WHAM!) 拉近噪声域; ③ hold-out A 集早期止损 (1k 对 LoRA 即先验, 若 ΔCER < 0.005 则停, 不烧 10 万对算力)。

### 5.2 小声/快语速合成真实度
- **风险**: librosa.time_stretch + 简单增益/低通的"小声化"可能不够真实 (真实小声有气息 + 发音模糊 + 距离效应)。
- **防控**: ① 增广参数采样保守 (50%/30% 触发率, 不全员小声); ② 后续可接 RIR (房间冲激响应) 卷积模拟远场小声; ③ 拿 cmd_2096/2949 真实失败样本做合成参考对齐。

### 5.3 过拟合 + 灾难性遗忘
- **风险**: LoRA 训练过头会让 Qwen3-ASR 在干净语音上退化 (遗忘原始能力), 或在合成风格上过拟合。
- **防控**: ① hold-out A 集作为铁律 (lessons-pitfalls §14), 任何 epoch 都验; ② 合成集内部 80/20 切, val 集 loss 早停; ③ LoRA rank 16 (小), epoch 3 起步不深训; ④ 混入 10–20% 干净 (无增广) 训练对防遗忘。

### 5.4 评测口径风险
- **风险**: 合成集 CER 下降不代表 A 集下降 (合成 ref 是模板, A 集 ref 是真实家居)。
- **防控**: **唯一可信指标是 A 集 hold-out CER** (官方 normalize_text + CERMetric), 合成 val loss 仅作训练监控。CER ±0.04 噪声内不算改进。

### 5.5 微调对齐题目 enrollment→recognition 范式
- **风险**: Qwen3-ASR 是 ASR (无 enrollment 输入), 当前合成只增强 recognition → ref, **未利用 enrollment 信号**。微调后 enroll_infer 链路里 ASR 仍只吃切好的 target timeline, enrollment 是上游 enroll_infer 的事。
- **防控**: 本配方只攻「ASR 在已有 target timeline 上的转写鲁棒性」(死区 60% 失败中 ASR错那部分), 不解决"选错 target"(那部分归 enroll_infer/声纹, 已在主线 separately 处理)。范围清晰, 不混淆。

---

## 6. 落地步骤 (优先级排序)

1. **POC 阶段 (1 周内)**:
   - [ ] 补装 `peft` + `datasets` 到 venv_qwen
   - [ ] 下载 Aishell-1 (178h, Apache-2.0, ~15GB) + MUSAN (CC BY, ~6GB)
   - [ ] 装配 `target_manifest.jsonl` (Aishell-1 的 wav + transcript) + `interferer_manifest.jsonl` (WenetSpeech 子集) + `noise_manifest.jsonl` (MUSAN)
   - [ ] `data_aug_recipe.py build --n-per-target 10` 生成 1k 对小集
   - [ ] LoRA 微调 1k 对 (4060 ~1h) → A 集 hold-out 看 ΔCER
   - [ ] **决策点**: 若 ΔCER ≤ −0.01, 扩到 1–3 万对; 若 ≥ 0, 停 (配方不灵)

2. **正式阶段 (POC 通过后)**:
   - [ ] 1–3 万对训练 (4060 隔夜)
   - [ ] 多 epoch + sweep (lr / lora_r / quiet_prob)
   - [ ] 失败子集 (cmd_2096/2949/...) 单独报告
   - [ ] 集成 submit_infer: `--asr-backend qwen-lora` 加载 adapter

3. **备选方向 (POC 失败)**:
   - 改 full-FT (需租卡)
   - 加 RIR 远场卷积增广
   - 改 FireRedASR-L 中转 (备选 ASR 后端)

---

## 附录: 关键文件路径

- 配方脚本: `E:/midea_target_asr/code/data_aug_recipe.py` (含 smoke / build / demo-cmd 三个子命令)
- 仿真复用: `E:/midea_target_asr/code/simulate_pipeline.py` (add_noise/mix_overlap)
- 噪声复用: `E:/midea_target_asr/code/build_dataset.py` (gen_white/pink/babble/load_env_noise)
- 微调 venv: `E:/midea_target_asr/code/.venv_qwen/` (qwen_asr + transformers 4.57.6)
- Qwen3-ASR 权重: `E:/hf_cache/Qwen3-ASR-1.7B/` (4.4GB, bf16)
- A 集金标 (仅 hold-out): `E:/midea_target_asr/datasetA/pos.jsonl` + `pos/cmd_*.wav`
- 评测口径: `E:/midea_target_asr/code/eval_metrics.py` (官方 normalize_text + CERMetric)
- smoke 自测输出: `E:/midea_target_asr/code/_aug_smoke_out/manifest.jsonl` (4 对样本)
