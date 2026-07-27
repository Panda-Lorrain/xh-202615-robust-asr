# Qwen3-ASR LoRA 1k 对 POC 微调 — ΔCER 趋势验证 (2026-07-27)

> 任务: 外部训练解禁后, 验证「微调 Qwen3-ASR-1.7B 攻死区 60% 失败」是否值得扩到 1 万对。
>
> ⚠️ 本 POC 只看 ΔCER 趋势, 不求收敛。1k 对 + LoRA r=16 + 2 epoch / 200 step。
> A 集 1350 条 **只做 hold-out 评测, 绝不进训练** (lessons-pitfalls §14)。

## TL;DR — 判定: ❌ 退化 (+0.1472), 不扩 1 万, 换方向解决域不匹配

| 指标 | baseline qwen | LoRA 微调 | Δ |
|---|---|---|---|
| Overall CER | **0.3436** | **0.4908** | **+0.1472 ❌** |
| 死区 sim<0.4 (n=1064) | 0.3961 | 0.5850 | **+0.1889 ❌** |
| ├ sim<0.2 (n=396) | 0.4592 | 0.7238 | **+0.2646 ❌❌** |
| ├ sim[0.2,0.3) (n=332) | 0.3526 | 0.5775 | +0.2250 ❌ |
| ├ sim[0.3,0.4) (n=336) | 0.3676 | 0.4382 | +0.0706 |
| 主战场 sim≥0.4 (n=286) | 0.1823 | 0.2013 | +0.0190 (噪声内) |

**死区反而退化最严重** (sim<0.2 Δ+0.265), 与「攻死区 60%」目标完全相反。
主战场持平 → 模型基础能力保留, 退化来自合成域偏置损害真实数据判别。

## 1. 配置

- **基座**: Qwen3-ASR-1.7B (E:/hf_cache/Qwen3-ASR-1.7B)
- **主线对照**: qwen CER=0.3436 (官方池, 1350 条 pos 切片, 完美重现)
- **训练数据**: 1k 对合成 (Aishell-1 200 spk × 5 utt target, _aug_manifests_poc1k)
  - 配方 5 步链 (data_aug_recipe): 小声化(-8~-3dB, 50%) + 快语速(1.1-1.4x, 30%) +
    重叠(0-100% 偏中高) + 加噪(SNR -5~5dB 偏低, 50% 在 ≤0dB) + 短 enroll(1.5-2.5s)
  - noise 池: MUSAN noise 80 条; interferer 池: Aishell 30 spk × 8 utt = 240 条
  - ⚠️ 文本是新闻朗读非家居指令 (build_aishell_manifest 已注释; 家居指令化留后续)
- **LoRA**: peft r=16, alpha=32, target=q_proj+v_proj, dropout=0.05,
  task=CAUSAL_LM, **4.78M trainable / 2.04B total (0.23%)**
- **训练**: bf16 + gradient_checkpointing(use_reentrant=False) + paged_adamw_8bit +
  batch=1 + grad_accum=8, lr=1e-4, max_steps=200, epochs=2
- **训练耗时**: 200 step / 815s (~13.6min)
- **关键 fix (踩坑, 防 agent 重蹈)**:
  1. **PEFT 必须包 `base.thinker`** (Qwen3ASRThinkerForConditionalGeneration), 不能包
     顶层 `Qwen3ASRForConditionalGeneration` —— 顶层只 override generate() 无 forward()
  2. **数据 collator**: input_features 形状 **(mel=128, T)** 非 (T, mel); 时间维 = dim=-1
  3. qwen venv 缺 jiwer/editdistance/zhconv/cn2an → `uv pip install --python <venv>` 指定 venv 装

## 2. 训练动态 (loss 收敛 ✓ 但学到的是错误分布)

- loss_first 7.86 → loss_last_avg10 2.89 → **loss_min 0.009** (单条过拟合迹象)
- loss 轨迹在 3.0-4.0 区间稳定震荡, 后期 (step 175+) 出现 2.48-2.99 局部新低
- 模型确实 fit 了合成数据, 但 fit 的「小声+程序 babble」特征在真实死区不适用

## 3. 退化机制分析 (为什么死区反而更差)

### 3.1 数据域严重不匹配 (主因)
| 维度 | 训练合成 | 评测真实 |
|---|---|---|
| 文本域 | Aishell 新闻朗读 ("樊振东"/"苹果手机") | 家居指令 ("空调开到制热"/"风量调到百分之三十") |
| 重叠源 | mix_overlap 程序叠加 (静默段补零) | 真实双人同期录音 (相位/混响耦合) |
| 噪声 | MUSAN env + 程序 white/pink/babble | 题目真实 babble + 录音环境 |
| 小声 | make_quiet 低通+降增益 (程序模拟) | 真实录音电平 + 人声特征 (cmd_2096/2949) |
| 失败原因 | 合成"小声+程序 babble"是单因子 | 真实死区是「diar 切错+mel 摧毁+声纹提不出」复合 |

→ LoRA 把模型对「合成域特征」过拟合, 损害了真实数据的判别。

### 3.2 死区退化递增规律 (越死越退化)
- sim<0.2 (重死区): Δ +0.265 ← 退化最严重
- sim[0.2,0.3): Δ +0.225
- sim[0.3,0.4): Δ +0.071
- sim≥0.4 (主战场): Δ +0.019 (噪声内, 几乎不变)

→ 越难的数据 LoRA 偏置损害越大, 主战场因为音频相对干净不受影响。
→ 这与"攻死区 60%"目标完全相反, 死区反而被进一步摧毁。

### 3.3 样本对比 (samples)
- cmd_0/1/10/100: base 与 lora 接近 (正确条都正确, 失败条都失败)
- cmd_100 (拉杆洗烘套装, sim 0.318): base "那该是红头船了" → lora "那该起红套装了" (lora 多混入"装"字)
- cmd_101 (闭所有灯光, sim 0.486): base "闭所有灯光" → lora "关闭所有灯光" (lora 反而更准)
→ 退化在统计层面 (某些死区条 lora 多出幻觉字符), 非每条都崩

## 4. 判定与下一步

**当前 POC 判定**: ❌ **退化 (+0.1472), 不扩 1 万对, 必须先解决域不匹配**

扩 1 万对 (同配方) 只会过拟合更严重, 不解决问题。三个换方向:

### 🥇 方向 A: 换数据源 (最高优先级)
- **家居指令 TTS**: 用 data_aug_recipe.HOME_CMD_TEMPLATES 文本 + 中文 TTS
  (如 GPT-SoVITS/Qwen-TTS) 合成 target, 而非 Aishell 新闻朗读
- **真实双人重叠**: 用 Aishell-3 多人对话录音 (有真实相位/混响) 替代 mix_overlap
- **保留 MUSAN env** (这层域 OK), 替换 target 文本/音频 + 重叠机制

### 🥈 方向 B: 换训练范式
- **full-FT 而非 LoRA**: r=16 只 4.78M (0.23%) 容量小但已学错特征; full-FT 配合
  更低 lr (1e-5) + 早停可能更稳; 但 8GB 4060 跑 full-FT 显存紧张 (需租算力)
- **Whisper-Sidecar 路线** (memory non-voiceprint-target-selection 🥇):
  embedding 空间分离, 不在原 ASR 上 LoRA, 绕开域不匹配

### 🥉 方向 C: 缩窄目标
- 只针对"小声"单因子 (去掉重叠/加噪), 用真实小声样本 (从 A 集 cmd_2096/2949 风格
  参考, 但不能直接用 A 集训练) + TTS 合成, 测单因子可否改善
- 风险: 真实死区是复合故障, 单因子训练收益有限

## 5. 风险点与教训

1. **数据域 > 模型容量**: 1k 对 LoRA 容量小但已学到错特征; 域不匹配时增容/扩数据无效
2. **合成真实度上限**: make_quiet (低通+降增益) ≠ 真实小声; mix_overlap ≠ 真实重叠相位
3. **Aishell 是新闻朗读非家居**: build_aishell_manifest.py 已注释但 POC 仍用, 是已知妥协
4. **loss 收敛不等于评测改善**: 训练 loss 0.009-3.0 看似收敛, 但 ΔCER +0.1472 退化
   → POC 必须看 hold-out ΔCER, 不能只看 loss
5. **死区是复合故障**: sim<0.2 同时含 diar 切错 + mel 摧毁 + 声纹提不出, 单一合成配方覆盖不全

## 6. 产物 (绝对路径)

- `E:/midea_target_asr/code/exp_qwen_lora_poc.py` — 合成 + LoRA 训练 + 评测 主脚本
- `E:/midea_target_asr/code/_aug_manifests_poc1k/{target,interferer,noise}.jsonl` — 1k 对清单
- `E:/midea_target_asr/code/runs/_qwen_lora_poc/aug/manifest.jsonl` — 1k 训练对 + recognition/*.wav
- `E:/midea_target_asr/code/runs/_qwen_lora_poc/adapter/adapter/` — LoRA adapter (19MB)
- `E:/midea_target_asr/code/runs/_qwen_lora_poc/train_log.json` — 训练 loss 轨迹
- `E:/midea_target_asr/code/runs/_qwen_lora_poc/eval_result.json` — A 集评测 ΔCER + 分桶 + samples
- `E:/midea_target_asr/code/runs/_qwen_lora_poc/run.log` — 全程日志

## 7. 复现命令

```bash
# qwen venv 已装齐: torch2.6/transformers4.57/peft0.19/bnb0.49/jiwer/editdistance/zhconv/cn2an
cd E:/midea_target_asr/code
E:/midea_target_asr/code/.venv_qwen/Scripts/python.exe exp_qwen_lora_poc.py \
    --n-pairs 1000 --epochs 2 --grad-accum 8 --max-steps 200 --eval-limit 0
```
