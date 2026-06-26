# 资料扩展·TS-ASR 路线与开源资产（2026-06-27）

> 在原有 8 篇精读之外，并行调研（SELD 精读 + jingkangqi github + 团队论文扩展）的重大增量。

---

## 一、🔥 最重要发现：TS-ASR（目标说话人 ASR）范式——原 8 篇里的空白

原 8 篇没有一篇是真正的「**enrollment → 只转写目标说话人**」端到端方案。这恰恰是本题最直接的范式。以下是完全开源、可直接当 baseline 的 TS-ASR 工作：

| 论文 | 来源 | 一句话 | 开源 | 对口度 |
|---|---|---|---|---|
| **SE-DiCoW**（Self-Enrolled DiCoW） | arxiv 2601.19194（BUT/JHU） | 在 Whisper 上加「自注册」——从混合音频自举说话人嵌入，**无需外部 enrollment clip**，完美契合本题「给定唤醒音频」 | ✅ [BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper) + HF 权重 | ⭐⭐⭐⭐⭐ |
| **DiCoW** | arxiv 2501.00114 | 用帧级 diarization 概率条件化 Whisper，做目标说话人 ASR | ✅ 同上 | ⭐⭐⭐⭐⭐ |
| **FDDT**（DiCoW 前身） | arxiv 2409.09543 | 把 diarization 当 decoder token 注入 Whisper | ✅ 同上 | ⭐⭐⭐⭐⭐ |
| **TS-RNNT**（流式神经 transducer TS-ASR） | arxiv 2209.04175（**NTT/东京工大**，⚠️非 CMU） | 端到端流式：Hadamard 积把声纹融进 RNN-T 编码器，RTF 与 vanilla RNNT 相同（0.40），**3 倍提速**；效率维度最优 | ⚠️ 论文未开源权重（ESPnet 实现）；SpeechBrain 另有同类 TS-ASR recipe | ⭐⭐⭐⭐⭐ |

→ **SE-DiCoW 是最契合本题的开源 baseline**：输入约定（enrollment 音频）、任务（只转写目标、重叠场景）几乎一一对应，且有现成权重。

---

## 二、杜俊团队 CHiME 系列（远场+重叠 pipeline 范式）

USTC-NERCSLIP 在 CHiME-7/8/9 的系统报告，全是「远场+重叠对话」全链路，可直接迁移：

| 论文 | 来源 | 价值 | 开源 |
|---|---|---|---|
| **CHiME-8 NOTSOFAR-1 系统** | arxiv 2409.02041 | diarization→分离→Whisper 三段式，提出 JDS（联合 diarization 与分离） | ✅ [rywang99/USTC-NERCSLIP_CHiME-8](https://github.com/rywang99/USTC-NERCSLIP_CHiME-8) |
| **Three-Stage Modular Diarization**（NOTSOFAR 期刊版） | CSL 2025 | 完整方法论，CER+效率双指标 | ✅ |
| **CHiME-8 MMCSG** | arxiv 2410.05986 | 智能眼镜两人对话，含目标说话人+多模态 | 部分 |
| **CHiME-9 MCoRec** | arxiv 2603.01415 | 多通道对话识别，最新 | 待查 |

---

## 三、MISP 2023 AVTSE（任务同构的挑战赛）

- **arxiv 2401.03697**（杜俊团队组织）：音视频 enrollment → 从混合中提取目标说话人语音。任务结构与本题同构（只是输出音频而非文本）。
- **数据集开源**，baseline 可复用。⭐⭐⭐⭐⭐

---

## 四、jingkangqi GitHub 资产评估（实测）

该账号（东南大学景康祺）**只有 2 个公开仓库**：DSENet、VSAEC。关键坑：

| 仓库 | 用途 | 预训练权重 | 数据/recipe | License |
|---|---|---|---|---|
| **DSENet** | DOA 引导 TSE（★★★★★ 直接命中） | ❌ 无 | ❌ 无 | ❌ 无 |
| **VSAEC** | 几何无关多通道 AEC（★★★） | ❌ 无 | ❌ 无 | ❌ 无 |

→ **都不能即插即用**（无 ckpt、无数据、无 License、DSENet 自发布 0 更新）。只能作**算法借鉴 + 代码骨架**。
- DSENet 迁移路径：把 **DOA/beamwidth embedding 换成「唤醒音频的说话人嵌入（ECAPA-TDNN/CAM++ 声纹）」**，即改造为 anchor-guided TSE。

---

## 五、社区成熟开源资产补充（有 ckpt + recipe，可当 baseline 起点）

- **Audio-WestLAI/target_speaker_extraction**：SpEx/SpEx+ 系列，声纹 anchor 的 TSE，最贴近「唤醒音频 anchor」范式。
- **SpeechBrain**：SepFormer / DPRNN / ECAPA-TDNN，均有预训练权重，可组合「声纹 anchor + 分离主干」。
- **Asteroid**：分离工具箱。

---

## 六、SELD 精读要点（shicheng-2023-apsipa.pdf，已下）

- **SS-SELD**：class-dependent 声分离（Conv-TasNet，每类一个）作前端 prompt，拼接到 ResNet-Conformer SELD。
- **结果**：STARSS23 上 SELDscore **0.279**（突破 ~0.30 瓶颈），F₂₀° 0.64；DCASE 2023 Task 3 **冠军系统核心**。
- **可借鉴**（声事件→说话人迁移）：① class-dependent → **speaker-dependent**（声纹条件分离）；② 分离特征作 prompt；③ 保留混合+叠加分离。
- **局限**：未开源；DOA 需多通道，分离前端可单通道；类别/说话人条件从离散标签→连续声纹嵌入需改造。

---

## 七、结论：资料面现在的两大候选范式

1. **美的 Personal VAD 路线**（#2 ASE-PVAD + #3 US-PVAD）：轻量、目标说话人检测为主干，后接 ASR。
2. **TS-ASR 路线**（SE-DiCoW / DiCoW / TS-RNNT）：端到端「enrollment → 只转写目标」，**完全开源、有现成权重**，最契合本题输入约定。

两者可融合（PVAD 做目标检测/拒识 + TS-ASR 做转写 + LLM 做语义拒识）。

---

## 八、SE-DiCoW + DiCoW 深度精读（2026-06-27 补充，已下 PDF）

### SE-DiCoW（arxiv 2601.19194，BUT/JHU）
- **自注册机制**：无需外部 enrollment clip，从混合音频里 argmax 选「目标说话人最活跃片段」作 X_se，经 cross-attention 注入 Whisper **每层 encoder**。
- **STNO mask**（Silence/Target/Non-target/Overlap）通过 **FDDT 仿射变换**调制 encoder 表征；非目标类用零初始化 W 显式压制（**内建拒识**）。
- 基座 **Whisper-large-v3-turbo**。
- **完全开源**：[github.com/BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper) + HF 权重 `BUT-FIT/SE_DiCoW`。
- 结果：Libri3Mix 全重叠 tcpWER 降 >75%，宏平均相对原 DiCoW 降 52.4%。

### DiCoW（arxiv 2501.00114，TS-ASR 基础范式）
- **抛弃 speaker embedding**，改用 **diarization 输出（who-spoke-when）** 的 STNO mask 条件化 Whisper——共享时间轴，少数据即可微调，对未见说话人泛化好。
- FDDT **抑制式初始化**（S/N 类零矩阵、T/O 类单位矩阵）不破坏 Whisper 预训练表示，收敛快。
- AMI/NOTSOFAR-1/Libri2Mix/LibriCSS 达 SOTA；开源同仓库。
- **关键缺口**：DiCoW 需要 diarization 输出 STNO，本题只给 enrollment 声纹 → 需前置「enrollment-conditioned 目标说话人 diarization」。

### 🔥 组合洞察：两条路线天然互补（最重要结论）

| 模块 | 美的 PVAD（#2/#3） | DiCoW/SE-DiCoW |
|---|---|---|
| 输入 | enrollment 声纹 | 混合音频 |
| 输出 | 帧级 TSS/NTSS（目标/非目标活动） | 目标说话人转写 |
| 需要的输入 | — | STNO mask |

→ **PVAD 的 TSS/NTSS 输出可直接转成 STNO mask，喂给 DiCoW**。即两条路线不是二选一，而是天然的前后端组合：

```
唤醒音频 → CAM++声纹 → Personal VAD(#2/#3) → 帧级 TSS/NTSS
                                              ↓ 转 STNO mask
识别音频 → DiCoW/SE-DiCoW(Whisper TS-ASR) → 目标转写
                                              ↓
                                     Qwen-2.5-3B LLM 语义拒识
```

这把「两大候选范式」升级为**一条组合主线**：**美的 PVAD（前端检测，产生 STNO）+ DiCoW/SE-DiCoW（后端 TS-ASR，Whisper 转写）+ LLM（语义拒识）**。每个环节都有开源/美的自研支撑。

### SE-DiCoW 当 baseline 的工作量
- **enrollment 接入**：把 self-enrollment（argmax 选片段）改成接受给定唤醒音频作 X_se，~1–2 周（架构兼容，只改数据/推理逻辑）。
- **中文适配**：Whisper 支持中文但 FDDT/cross-attention 未在中文训练，需少量中文多人数据微调，~2–4 周。
- 合计 **4–8 周**可产出中文 TS-ASR baseline；有现成 HF 权重起步。

---

## 九、FDDT / TS-RNNT / NOTSOFAR-1 精读补充（2026-06-27）

### FDDT（arxiv 2409.09543，BUT/JHU）—— DiCoW/SE-DiCoW 的基石
- 演进链第零块：首次提出「用 diarization STNO 条件化 Whisper，不用 speaker embedding」。
- **encoder 侧 FDDT 仿射变换**（逐层、逐类对角矩阵+偏置，按 STNO 凸组合）；**抑制式初始化**（S/N 零矩阵、T/O 单位矩阵）不破坏 Whisper。
- **最轻量**：单层偏置即可见效（b-only suppressive = 28.0）；NOTSOFAR ORC-WER 比 cascade baseline 低 12.9%。
- 开源 github.com/BUTSpeechFIT/TS-ASR-Whisper。价值：最快搭基线/消融；理解 DiCoW 为何换 decoder token 的前提。

### TS-RNNT（arxiv 2209.04175，**NTT/东京工大**）—— 效率维度最优
- ⚠️ 纠正：NTT，**非 CMU/SpeechBrain**（SpeechBrain recipe 是另一份衍生资产）。
- 流式端到端 TS-ASR：**Hadamard 积**把声纹 embedding 融进 RNN-T 编码器第 1 层，TSE 功能吸收进 ASR。
- 声纹**预注册**（只算一次，不进帧路径）→ **RTF 与 vanilla RNNT 完全相同（0.40），相对级联 TSE+ASR 3 倍提速**。
- 流式延迟 30–330ms；日文 CSJ 验证，SNR 0–20dB（未覆盖 −5dB）。
- 价值：**对标本题 20% 效率分的最优架构形态**；论文未开源权重，需 ESPnet 自实现或用 SpeechBrain recipe。

### NOTSOFAR-1（arxiv 2409.02041，杜俊团队）—— pipeline 范式对照
- CHiME-8 双赛道冠军（多通道 tcpWER 10.8%、单通道 22.2%）。
- 三段式 pipeline：diarization → 分离 → Enhanced Whisper；核心创新 **JDS（联合 diarization 与分离）**，CSS 26.68% → JDS 17.47%。
- 前端 WPE/MVDR/GSS/cACGMM；ASR = Whisper + WavLM + ConvNeXt + RPE + MoE 全家桶。
- **未开源系统代码**；重计算（9 系统融合），不适合实时。
- 价值：pipeline 范式（vs DiCoW 端到端）对照；**JDS「先验边界指导分离」思想可借鉴**；声纹匹配接口可改造为目标说话人。

### TS-ASR 路线四件套定位
| 维度 | FDDT | DiCoW/SE-DiCoW | TS-RNNT | NOTSOFAR-1 |
|---|---|---|---|---|
| 范式 | FDDT 仿射变换（encoder 侧）| FDDT+QKb；SE-DiCoW 加 cross-attn（自注册）| 端到端 RNN-T | pipeline 级联 |
| 效率 | 中 | 中（大模型） | **最优（流式轻量）** | 差（重计算） |
| 精度 | 基石 | **最高** | 中 | 高（冠军） |
| 开源 | 代码 | **代码+权重** | 未开源 | 未开源 |
| 本题用途 | 轻量基线 | 首选 baseline | 效率 baseline | pipeline 参考 |
