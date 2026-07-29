# 自训中文 TSE 训练方案 (2026-07-28)

> **2026-07-29 Phase-2 POC**：已接入官方 WeSep pBSRNN + 离线
> CAM++ 512d enrollment embedding。speaker-disjoint 40 条 synthetic val
> 上 SI-SNRi `+1.0956 dB`，但冻结 Qwen3-ASR 官方累计池 CER 仅
> `1.2870→1.2671`，bootstrap CI 跨 0。无条件全段增强不准入主线。
> 下一实验固定为 overlap-only，并用输出 speaker cosine、能量/峰值/NaN
> 检测做 failure-aware fallback。相对 raw CER 绝对下降未达到 `0.05`
> 前，不启动 Phase-3 正式训练。
>
> 恢复执行细节以 `AGENT_HANDOFF.md` 顶部“下一个 Agent 从这里开始”为准。
> 特别注意：路由阈值只能在 speaker-disjoint val 上使用可上线特征拟合，
> 不能使用 ref、clean target 或 Qwen CER，后者只用于离线评估。
>
> **任务**：攻死区分离质量瓶颈。当前主线 qwen3-ASR pos CER 0.3436（含拒 0.5934），SepFormer 英文权重 sepformer-whamr16k 盲分离破坏 mel（B 类铁证：argmax CER 0 → SepFormer 后 0.6+）。CLAUDE.md 三红线：①中文数据 ②避免纯 SI-SDR 陷阱（EoW 感知-识别鸿沟）③数据对齐题目规格。
>
> **2026-07-29 状态修正**：阶段一训练正确性整改完成。旧冒烟只能证明程序不崩，
> 且存在监督错位和 enrollment 泄漏，**禁止按旧方案直接跑 100k steps**。
> 自写 V1 现在作为数据/损失正确性基准；阶段二改为 WeSep pBSRNN POC。

---

## TL;DR (5 行)

1. **阶段一已修正**：整句 recognition/clean 成对读取、动态 padding、真实长度 loss、正确 complex mask、独立 enrollment。
2. **数据分布已修正**：随机前/后重叠、显式 SIR、target-relative SNR，并输出 speaker-disjoint train/val 清单。
3. **旧结论撤回**：自写 CTC 只约束临时 encoder，不等价于 Qwen3 ASR-aware；旧 10-step loss 下降不构成模型有效证据。
4. **阶段二主推**：WeSep pBSRNN + 预训练中文 ECAPA/ERes2Net，只处理 overlap 区，分离失败回退原 mixture。
5. **阶段三候选**：在 Qwen3 AuT 表征内加入 enrollment-conditioned Sidecar/target activity head，直接以最终 CER 优化。

---

## 一、选型决策

### 候选对比 (基于 docs/research_speaker_separation.md Top5)

| 方案 | 中文 | 绕纯 SI-SDR | 短 enroll (~1.8s) | 现成实现 | 工程成本 | 选否 |
|------|------|------------|------------------|---------|---------|------|
| V1. STFT-mask TSE + 临时 CTC | 自训 | ⚠不对齐 Qwen3 | ✅ | 自写 | 低 | 仅正确性基准 |
| **V2. WeSep pBSRNN + 中文 ECAPA** | 自训 | ❌(需 Qwen CER 选模) | ✅ | 官方 toolkit | 中 | ⭐ 阶段二主推 |
| V2. Whisper-Sidecar 自训 | ✅(Whisper frozen) | ✅✅(纯 ASR) | ⚠(原 3s, 自训可调) | 有代码 (LingweiMeng) | 高 (4×GPU 数天) | 备选/升级 |
| TS-ASR-AD (Honda RI 2025) | 自训 | ✅(ASR+VAD) | ⚠(原 5s) | ❌ 无代码 | 高 | 远期 |
| SpEx+ (SpeechBrain) | 自训 | ❌(纯 SI-SNR) | ⚠(原 5s) | ✅ | 中 | 不选 (EoW 陷阱) |
| TSE-through-Pos-Neg-Enroll | 自训 | ❌(SI-SNR) | ✅ | ✅ | 中 | 不选 (EoW) |
| TSELM (已 POC 证伪) | ❌(英文) | ✅(CE 分类) | 4s | ✅ | 中 | 不选 |
| USEF-TSE (已 POC 证伪) | ❌(英文) | ❌(SI-SDR) | 部分 | ✅ | 低 | 不选 |

### V1 选型理由

- **一举解三瓶颈**：①中文数据训练（避 SepFormer 英文 OOD）②联合 ASR loss 绕纯 SI-SDR 陷阱 ③短 enrollment 友好（自写架构可任意长度）
- **复用现成组件**：speechbrain 1.0.2 已装，torchaudio 提供 STFT/MelSpectrogram，data_aug_recipe 增广链已就绪
- **可控可冒烟**：6.14M 参数小（vs Whisper-Sidecar 数百 M），4060 单卡可跑通完整链路
- **失败模式可定位**：自写架构每个组件可单独调试（mask net 出问题 vs ASR head）

### V2 为何留作升级

- Whisper-Sidecar 论文报告 Aishell1Mix zero-shot CER 28.94%（vs 我们主线 0.3436 = 34.36%），但 **2026-07-23 段记录 zero-shot NO-GO**：无公开权重 / 绑 Whisper 不能嫁接 qwen3 / 1.8s 硬失配 / clean 数据迁移空白
- **2026-07-27 战略转向自训**解除了 zero-shot NO-GO：自训中文权重可避开英文 OOD，但工程成本高（4×V100 论文规模）
- V1 跑通后若需要更强，V2 是清晰升级路径（替换 ASR encoder 为 frozen Whisper，替换 mask net 为 Sidecar 设计）

---

## 二、模型架构 (V1)

```
mix_wav ──STFT──► mix_ri [B,2,F,T]
                              │
  enroll_wav ──mel─► spk_enc ─► d_vec [B,D]
                              │   ├── (mix_ri, d_vec) → MaskNet (4 TCN blocks) → mask [B,2,F,T] (tanh)
                              │   │
                              ▼   ▼
                       est_ri = mix_ri * mask ──ISTFT──► est_wav [B,T]   ─► L_si_snr = -SI-SNR(est, clean_target)
                                          │
                                          ▼ log-mel
                                       ASR encoder (4 Conformer layers) + CTC head → ctc_logits ─► L_ctc
```

### 子模块详情 (见 `code/tse_train.py`)

| 模块 | 架构 | 参数量 | 备注 |
|------|------|-------|------|
| SpeakerEncoder | 4× Conv2d (32→64→128→128) + AvgPool + Linear | ~0.2M | mel → d_vec=192 |
| MaskNet | Conv2d(2→256) + d_vec broadcast + 4×TCNBlock(kernel=5, dilation=1,2,4,8) + Conv2d(256→2) + tanh | ~1.5M | 复数 mask in [-1,1] |
| ASREncoder | Linear(80→256) + 4× ASRLayer (depthwise Conv1D k=15 + FF + LayerNorm + residual) | ~1.5M | Conformer-lite |
| CTC head | Linear(256→V) | ~0.2M | V=字符级词表 |
| **合计** | | **~3.4M (实 6.14M 含 BN)** | 4060 8GB 充裕 |

### 关键设计点

1. **STFT 域 mask（非时域 TasNet）**：直接操作复数谱，可同时改幅度+相位；与 Whisper/Conformer 的 mel 输入自然衔接。
2. **dvec broadcast 到 mask net 输入**：把 enrollment 信息注入每个时频 bin，等效"基于 enrollment 引导的 mask 预测"。
3. **tanh mask ∈ [-1, 1]**：允许 mask 反相（极弱信号场景需要），同时有界保稳。
4. **ISTFT length=mix.size(-1)**：保证 est_wav 与 clean_target 完全等长对齐，SI-SNR 计算合法。
5. **log-mel = log(MelSpectrogram + 1e-6)**：标准 Kaldi-like 前端，n_fft=512 / hop=128 / n_mels=80 / win=512。

---

## 三、Loss 设计（核心，对齐"避免纯 SI-SDR"红线）

### 公式

```
L = α · (-SI-SNR(est, clean_target)) + β · CTC(log_softmax(asr_head(mel(est))), ref_tokens, blank=0)
```

推荐 α=β=1.0 起步。

### 为何不纯 SI-SDR

- **EoW (Ear-of-the-Werewolf) 陷阱**：纯 SI-SNR/SI-SDR 优化感知质量（人耳听着干净），但 mel 频谱失真导致 ASR 转错。**这是 SepFormer 在本项目失败的真因**（memory `overlap-is-cer-failure-rootcause`）。
- **CTC 项的作用**：直接把"分离为识别服务"压力传到 mask net，强制 mask 优化方向是 mel-友好而非 SI-SNR-友好。即使分离的 SI-SNR 略低，只要 mel 不破坏 ASR 就能赢。

### α/β 调度建议

| 阶段 | α (SI-SNR) | β (CTC) | 目的 |
|------|-----------|---------|------|
| Warmup (前 5k step) | 1.0 | 0.1 | 先让分离稳，CTC 太早反向会破坏 mask 初始化 |
| 主训练 (5k–80k) | 1.0 | 1.0 | 联合优化 |
| Fine-tune (80k–100k) | 0.5 | 2.0 | 让 ASR 主导，进一步压低 CER |
| 极端 (备选) | 0.0 | 1.0 | 退化为纯 ASR loss（V2 风格，但 mask 不再有 SI-SNR 约束可能不稳） |

**冒烟默认** α=β=1.0，验证基本可用，全量训练按上表分阶段调度。

### 其他可加 loss 项（保留 V1+ 扩展位）

- **L1 mel loss**：`||mel(est) - mel(clean_target)||_1`，进一步保 mel 保真度（α_mel=0.1）
- **Speaker consistency loss**：`1 - cos(spk_enc(mel(est)), d_vec)`，强制 est 仍含 target 声纹（α_spk=0.05）
- **V2 升级**：去掉 SI-SDR 项，纯 CTC + speaker consistency，对齐 Whisper-Sidecar 思路

---

## 四、数据准备 (基于 Aishell-1 + MUSAN)

### 数据流

```
Aishell-1 干净单人中文  ┐
                        ├──► build_aishell_manifest.py ──► {target,interferer,noise}.jsonl
MUSAN 噪声             ┘                                            │
                                                                    ▼
                                          tse_data_aug.py (3 件套合成)
                                                                    │
                                                                    ▼
                        {enrollment,recognition,clean_target}/*.wav + manifest.jsonl
                                                                    │
                                                                    ▼
                                                            tse_train.py (Dataset 直接吃 manifest)
```

### 三件套（TSE 训练必需，与 qwen 微调 2 件套不同）

| 文件 | 内容 | 用途 |
|------|------|------|
| `enrollment/<uid>.wav` | ~1.5–2.5s target 说话人参考（含可选污染） | speaker encoder 输入 |
| `recognition/<uid>.wav` | target + interferer + noise 混合 | mask net 输入 |
| `clean_target/<uid>.wav` | 与 recognition 等长的纯 target 波形（含小声/快预处理） | SI-SNR 监督 |

**⚠️ tse_data_aug.synthesize_tse_triple 关键**：clean_target 是预处理后的 target（小声/快语速已应用），与 recognition 中 target 贡献完全对齐；长度 = min(len(target), len(interferer))（mix_overlap 内部对齐），保证 SI-SNR 计算合法。

### 增广参数分布（题目规格 + 失败分布加权）

| 维度 | 取值 | 权重 | 题目对齐 |
|------|------|------|---------|
| overlap_ratio | [0.0, 0.25, 0.5, 0.75, 1.0] | [0.10, 0.10, 0.25, 0.30, 0.25] | 失败组重叠中位 45% / 75% 重叠 / 97% 双人 → 偏中高 |
| SNR (dB) | [-5, -3, 0, 3, 5] | [0.20, 0.25, 0.30, 0.15, 0.10] | 题目 −5~5dB → 偏低 |
| noise_type | [white, pink, babble] | [0.25, 0.25, 0.50] | 死区 babble 重 → babble 多 |
| target_gain_db | -8~-3 dB (prob 0.5) / 0 (prob 0.5) | - | 死区小声失败模式 |
| target_speed_rate | 1.1~1.4 (prob 0.3) / 1.0 (prob 0.7) | - | 死区快语速失败模式 |
| enroll_dur_sec | 1.5~2.5 均匀 | - | 题目 ~1.8s |
| enroll 污染 | prob 0.3, 污染 SNR 8~15 dB | - | 现场 enrollment 噪声 |

### 冒烟结果（test_wav/zh_target + zh_nontarget 4 条）

```
[smoke] OK: 4 三元组 → code/_tse_aug_smoke
  - zh_target_01_k000: enroll=2.39s recog=1.76s clean_tgt=1.76s (等长=True) overlap=0.75 snr=-5.0 noise=pink gain=-4.3 speed=1.00
  - zh_target_01_k001: enroll=2.12s recog=1.76s clean_tgt=1.76s (等长=True) overlap=0.75 snr=0.0 noise=babble gain=-5.8 speed=1.36
```

参数全部对齐题目规格。

### 全量数据规模建议

| 规模 | target 数 | 每 target 增广数 | 总对数 | 备注 |
|------|----------|-----------------|-------|------|
| **起步 POC** | 100 | 10 | 1k | 验证训练能收敛，1 epoch |
| **小规模** | 500 | 20 | 10k | 验证 CER 收益，单 GPU 数 h |
| **中等** | 1000 | 30 | 30k | 论文级 baseline，单 GPU ~10h |
| **大规模** | 2000+ | 50 | 100k+ | 全量 Aishell-1 train（~120 说话人） |

**A 集 N=全量**：Aishell-1 train ~ 12 万条单人朗读 → ~ 120 说话人 × 1000 条/人 → 可合成 100 万训练对（实际用 30-100k 足够）。

---

## 五、训练超参（推荐）

### 冒烟配置（已通过）

```bash
python code/tse_train.py \
  --manifest code/_tse_aug_smoke/manifest.jsonl \
  --out-dir code/_tse_train_smoke \
  --steps 10 --batch-size 2 --seg-samples 32000 \
  --device cuda --alpha 1.0 --beta 1.0 --lr 2e-4
```

### 小规模 POC (单 GPU, 10k 对, ~3h)

```bash
python code/tse_train.py \
  --manifest /path/full_manifest.jsonl \
  --out-dir /path/poc_out \
  --steps 30000 --batch-size 4 --seg-samples 48000 \
  --enroll-samples 28800 \
  --device cuda --alpha 1.0 --beta 1.0 --lr 2e-4 \
  --log-every 50
```

### 全量训练 (A100/L20, 100k 对, ~5-8h)

```bash
python code/tse_train.py \
  --manifest /path/full_manifest.jsonl \
  --out-dir /path/full_out \
  --steps 100000 --batch-size 16 --seg-samples 64000 \
  --enroll-samples 28800 \
  --mask-blocks 6 --asr-layers 6 \  # 加深
  --device cuda --alpha 1.0 --beta 1.0 --lr 5e-4 \
  --log-every 100
```

### 推荐超参表

| 超参 | 冒烟 | POC | 全量 | 备注 |
|------|------|-----|------|------|
| batch_size | 2 | 4 | 16 | 4060=2, L20=8, A100=16+ |
| seg_samples | 32000 (2.0s) | 48000 (3.0s) | 64000 (4.0s) | 越长越稳但显存大 |
| enroll_samples | 28800 (1.8s) | 28800 | 28800 | 固定对齐题目 |
| n_fft / hop | 512 / 128 | 同 | 同 | 16k 标准 |
| n_mels | 80 | 同 | 同 | 标准 |
| d_vec | 192 | 同 | 同 | ECAPA 标准 |
| mask_hidden | 256 | 256 | 384 | 全量加深 |
| mask_blocks | 4 | 4 | 6 | 全量加深 |
| asr_hidden | 256 | 256 | 384 | 全量加深 |
| asr_layers | 4 | 4 | 6 | 全量加深 |
| α (SI-SNR) | 1.0 | 1.0 | 调度 | 见 §三 |
| β (CTC) | 1.0 | 1.0 | 调度 | 见 §三 |
| lr | 2e-4 | 2e-4 | 5e-4 | Adam |
| grad clip | 5.0 | 5.0 | 5.0 | 防 CTC 早期爆炸 |
| steps | 10 | 30k | 100k | |

---

## 六、算力需求

### 4060 Laptop 8GB (本机, 仅冒烟)

- 6.14M 参数, batch=2, seg=32k → 显存占用 ~3 GB
- 0.6 s/step → 10k step ≈ 100 min（可跑 POC）
- **OK 冒烟/POC，全量训练太慢**

### L20 48GB (官方评测硬件, 用户可租)

- 估显存占用：batch=8, seg=48k → ~15 GB（充裕）
- 估速度：~0.2-0.3 s/step (L20 ≈ 2.5× 4060 for small model, memory `efficiency-portability-audit`)
- 10 万 step ≈ 6-8 h
- **推荐租 L20 跑全量**

### A100 80GB (云租, 通用最强)

- 估显存占用：batch=16, seg=64k → ~25 GB
- 估速度：~0.15-0.2 s/step
- 10 万 step ≈ 4-6 h
- **若费用允许，A100 是最快**

### 租算力建议

- **优先**：L20（与官方评测硬件一致，效率分可顺便测）
- **次选**：A100（最快，但与官方硬件有 gap）
- **避免**：V100/K80（旧卡，FP32 慢）/ T4（小卡 batch 上不去）
- **预算估算**：L20 ~ 8 元/h × 8h ≈ 64 元 / 训练 round；可负担多 round 调参

---

## 七、风险与不确定性

### 风险 1：CTC + 简化 ASR encoder 不及 Whisper/qwen 强

- **风险**：自写 4 层 Conformer-lite 比 Whisper-small/qwen3-ASR 弱，端到端 CER 可能不及主线 0.3436
- **缓解**：①加 external LM 解码 ②升级到 Whisper-frozen encoder (V2 思路) ③看 CTC 学到的字符错误率，只关心相对改善（vs vanilla baseline）
- **判定**：训练后做 hold-out POC，对比 (a) 当前主线 cascaded (b) V1 TSE+联合 ASR，若 (b) 死区 CER 改善 > 0.05 则有效

### 风险 2：STFT 复数 mask 路线能力上限

- **风险**：SpEx+ 时域 mask / TF-GridNet 多尺度卷积可能比 STFT mask 强（论文 SOTA 多是时域或时频联合）
- **缓解**：①V2 备选升级路径已规划 ②跑通 V1 后可单独换 MaskNet 为 SpEx+ separator（speechbrain 体系内）③死区主战场（双人重叠 75%）只要分离 + ASR CTC 都收敛就能改善
- **判定**：冒烟 + POC 阶段先验证收敛性，能力上限问题等 POC 数据出来再决定是否升级

### 风险 3：1.8s enrollment 极端短（论文多 3-5s）

- **风险**：speaker encoder 从 1.8s mel 提取的 d_vec 不稳，mask 引导失败
- **缓解**：①数据增广 enroll_dur 在 1.5-2.5 抖动（已实现）②可加 enroll pre-training（先用长 enroll 训，再 fine-tune 短 enroll）③可加 multi-cut enrollment（同一 target 多段 enroll 平均 d_vec）
- **判定**：训练后测不同 enroll 长度的 SI-SNR 分布，若 1.8s 显著差则加 pre-training

### 风险 4：Aishell-1 朗读 ≠ 家居指令域

- **风险**：Aishell-1 是新闻朗读，与题目家居指令（空调/灯/温度）文本/声学不匹配。**这是 qwen LoRA POC 翻车 Δ+0.147 的真因**（memory `external-training-allowed`）
- **缓解**：①本方案用 Aishell 仅作**分离训练**（声学层），文本是否家居指令对分离 mask 影响小（mask 学的是声纹引导，不是文本内容）②若需家居指令化，可后期用 TTS 合成家居指令（data_aug_recipe.HOME_CMD_TEMPLATES 已就绪）③用 WenentSpeech/MagicData-RAMC 扩充说话人多样性
- **判定**：hold-out 必须含题目样本（A 集几条 hold-out 不训，只评），若 hold-out ΔCER < 0 则域匹配 OK

### 风险 5：联合 loss 权重调不好（α/β 失衡）

- **风险**：CTC 太强压垮 SI-SNR，mask 学不出分离（est ≈ mix）；或 SI-SNR 太强 CTC 不收敛
- **缓解**：①warmup 调度（前 5k step α=1, β=0.1）②监控两个 loss 分项，都不应单调升 ③grad clip 5.0 防 CTC 早期爆炸（已实现）
- **判定**：训练日志看 sisnr 与 ctc 是否都收敛，loss 早期会震荡但 1k step 后应稳

### 风险 6：词表与中文 ASR 评估口径

- **风险**：字符级词表覆盖度（Aishell-1 ~ 4000 字 vs 中文 GB2312 6700+ 字），hold-out 遇到 OOV 字符 CER 飙
- **缓解**：①全量训练时词表用 WenetSpeech 全字表 ②训练前统计 manifest ref 覆盖度，缺字补 ③评估时用题目官方口径 normalize_text（已在 eval_metrics.py 实现）
- **判定**：词表覆盖率应 > 99%，否则扩数据

---

## 八、攻死区策略（重点，CLAUDE.md 红线 ③）

死区 sim<0.4（占 n_spk=2 的 99%，贡献 87% CER）特征：重叠 ≥75% / SNR ≤0dB / babble 重 / target 小声。

### 数据增广针对性（已在 tse_data_aug.py 实现）

- **overlap ≥0.5 概率 0.80**（题目失败组铁证）
- **SNR ≤0dB 概率 0.75**
- **babble noise 概率 0.50**（题目主因）
- **target 小声化概率 0.50**（-8~-3dB）
- **target 快语速概率 0.30**（1.1-1.4×）

### 训练策略（建议）

1. **Curriculum learning**：前 30k step 用 overlap≤0.5 简单样本，30k-60k 加 overlap=0.75，60k+ 加 overlap=1.0 + 死区 babble
2. **Hard negative mining**：60k+ step 优先采样 sim<0.4 区间的失败组合（需要第二阶段筛选 manifest）
3. **Domain adaptation**：60k+ step 切换到题目分布 1：1 的训练集（用题目 hold-out 复制+扰动，⚠️ 不能用 A 集本身训练，但可用 A 集相似分布的合成）

### 评估策略

- **Hold-out**：留 10% target 不训练，定期算 val SI-SNR + val CER
- **死区子集 POC**：训练后选 10-20 条题目死区样本（cmd_18/2098/2251/2687/2630 等）做推理，对比主线 qwen3 CER
- **判定阈值**：ΔCER > 0.05 = 有效，> 0.10 = 显著有效，可推进 V2 升级

---

## 九、产物清单

| 文件 | 状态 | 用途 |
|------|------|------|
| `docs/tse_train_plan.md` (本文件) | ✅ 完成 | 完整方案 |
| `code/tse_data_aug.py` | ✅ 阶段一纠错 + 冒烟通过 | 严格对齐 3 件套数据增广 |
| `code/tse_train.py` | ✅ 阶段一纠错 + 冒烟通过 | 正确性基准，不直接全量训练 |
| `tests/test_tse_phase1_logic.py` | ✅ 6 项通过 | 对齐/SIR/SNR/complex mask/split 不变量 |
| `code/_tse_aug_smoke_phase1/` | ✅ 4 三元组 | 修复后本机验证，gitignored |
| `code/_tse_train_smoke_phase1/` | ✅ 3 step CUDA | 修复后本机验证，gitignored |
| `code/build_aishell_manifest.py` | ✅ speaker-disjoint train/val | 输入清单装配 |
| `code/data_aug_recipe.py` | ✅ 已有 | 增广工具库（被 tse_data_aug 复用） |

## 十、下一步行动

1. 引入官方 WeSep，先复现 pBSRNN recipe 和推理接口。
2. 用修复后的生成器构造 1k train + speaker-disjoint val，不碰 A 集训练。
3. pBSRNN 输出统一送 Qwen3，和原 mixture、SepFormer 同框架比较 CER。
4. 死区 Qwen3 `ΔCER>0.05` 且 RR 下降不超过 1pp 才扩到 10k/全量。
5. 有效后再加入 overlap-only scene route 与 extraction-failure fallback。

---

## 附录 A：与 SepFormer (主线盲分离) 对比

| 维度 | SepFormer (主线) | 本方案 V1 |
|------|----------------|----------|
| 路线 | 盲分离 (无 enrollment) | TSE (enrollment 条件) |
| 训练数据 | WHAM+reverb 英文 | Aishell-1 中文 |
| Loss | 纯 SI-SDR | α·SI-SDR + β·CTC 联合 |
| 选 target | 后置 argmax sim 选路（25% 失效） | mask 内置 enrollment 引导（不需后置） |
| 输出 | 波形 (感知干净但 mel 失真) | 波形 + ASR hidden (联合优化 mel-友好) |
| 中文 OOD | 严重（"观影"→"关机"） | 训练解决 |
| 主线结果 | 死区 Δ-0.072（净负） | 待 POC |

## 附录 B：复用现有资源

- 调研基础：`docs/research_speaker_separation.md` (Top5, Whisper-Sidecar 最对症)
- 数据骨架：`code/build_aishell_manifest.py` + `code/data_aug_recipe.py` (已就绪, +22 行)
- 仿真工具：`code/simulate_pipeline.py` (mix_overlap / add_noise) + `code/build_dataset.py` (gen_white/pink/babble)
- SepFormer POC：`code/exp_sepformer_poc.py` (证伪用，本方案借鉴其评估框架)
- 评估口径：`code/eval_metrics.py` (cer_official)
- venv：`code/.venv_tse/` (torch 2.3.1+cu121, speechbrain 1.0.2, transformers 4.42.3, 全装齐)
