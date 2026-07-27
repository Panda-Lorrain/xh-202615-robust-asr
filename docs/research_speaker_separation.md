# 说话人分离 / 目标提取技术调研报告 (2026-07-27)

> **任务背景**: 美的 XH-202615. 给定目标说话人 enrollment (~1.8s, 可能被污染), 在带噪 (-5~5dB SNR) + 双人重叠 (0-100%) 的 recognition 音频里只转写 target 的话。链路: `diar 分人 → argmax 选 target → 切 timeline → ASR 转写`。
> **今天四大瓶颈**: ①盲分离(SepFormer)破坏声纹 ②中文 OOD ③选路失效(enrollment 短+污染, sim 不可信) ④重叠区时间归属模糊 (见 memory `overlap-is-cer-failure-rootcause`)。
> **本报告目的**: 全网搜 2024-2026 最新 "目标说话人提取/分离" 论文 / 模型 / 开源代码, 重点找能绕开上述瓶颈的解药。

---

## TL;DR (核心发现 8 行)

1. **TSE 方向最契合的 3 个**: ① **TSE-through-Pos-Neg-Enroll** (NeurIPS 2025, 处理 noisy enrollment 完美契合我们污染痛点) ② **USEF-TSE** (ICASSP 2024, 不依赖 speaker embedding, HF 有 checkpoint 零成本试) ③ **AlphaFlowTSE** (2026 one-step flow-matching, 比 diffusion 快)。
2. **零成本可试**: USEF-TSE (HF: ZBang/USEF-TSE, CC BY-NC, 但参赛可) + TSELM (HF: TSELM-L, 离散 token 路线) + ClearerVoice-Studio (Alibaba Apache-2.0, ModelScope)。
3. **TSE+ASR 联合训练最新**: ⭐ **TS-ASR-AD** (Honda RI, Interspeech 2025, CTC+CE loss 联合 ASR+VAD, WER 6.61/14.81, 完全绕 SI-SDR) + ⭐ **Whisper-Sidecar** (Interspeech 2024, embedding 空间分离 + frozen Whisper, zero-shot Aishell1Mix CER 28.94%)。
4. **中文专用**: 没有现成中文 TSE 预训练模型, 但 **Whisper-Sidecar 在 Aishell1Mix zero-shot 已实测** (large 模型 CER 28.94%, batch-tune 17.81%); SpeechBrain Aishell1Mix 只有 separation recipe 没 extraction (SepFormer 路线, 需自训)。
5. **Whisper-Sidecar 之外的 embedding 分离**: **TSELM** (WavLM 离散 token + LM, 2024, 用 CE 分类 loss 替代 SI-SDR 回归, 是 embedding 空间操作的另一范式); AudioSep/CLAPSep (CLAP text-guided, 不是 speaker 但理念可借鉴)。
6. **Top 5 推荐**: ① Whisper-Sidecar (最契合, embedding 分离 + frozen Whisper 中文强) ② TS-ASR-AD (联合 ASR+VAD, 解决 EoW 陷阱) ③ TSE-Pos-Neg-Enroll (enrollment 污染解药) ④ TSELM (离散 token 绕 SI-SDR) ⑤ USEF-TSE (零成本 POC 试)。
7. **候选 X (端到端 enrollment-conditioned ASR)**: ⭐ **TS-ASR-AD + Whisper-Sidecar 就是候选 X 的两种实例化**, 完全符合 memory `non-voiceprint-target-selection` 提的"自建 TSE+ASR 联合"方向, 不靠 SI-SDR。
8. **最大门槛**: 主流 TSE 论文都是英文 8kHz WSJ0-2mix/Libri2Mix 训练, **中文 16kHz 必须自训** (但 Aishell1Mix 数据集现成, SpeechBrain 已有 generation recipe); 大部分需 4-8 GPU 中等训练成本。

---

## 一、技术清单 (按方向 A-E)

### 方向 A. Target Speaker Extraction (TSE) - 用 enrollment 引导提取 ★最契合

| 模型/论文 | 年份 | 一句话原理 | 开源 | 中文 | 短enroll | 绕SI-SDR | 状态 |
|----------|------|-----------|------|------|---------|---------|------|
| **TSE-through-Positive-Negative-Enrollments** ([arxiv 2502.16611](https://arxiv.org/abs/2502.16611)) | NeurIPS 2025 | **正/负 enrollment 对比** - target 说话片段作 Positive, 沉默片段作 Negative, 对比编码 target 身份。SI-SNRi 比 SOTA +2.1dB。**直接处理 noisy enrollment** (我们项目核心痛点) | [xu-shitong/TSE-through-Positive-Negative-Enroll](https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll) (CC BY 4.0) | 否 | **鲁棒(核心创新)** | 否(SI-SNR) | 代码开, 权重要训 |
| **USEF-TSE** ([arxiv 2409.02615](https://arxiv.org/abs/2409.02615)) | ICASSP 2024 | **Universal Speaker Embedding Free** - 不依赖 speaker embedding, 用预训练 backbone 直接学 enrollment-conditioned 提取 | [ZBang/USEF-TSE](https://github.com/ZBang/USEF-TSE) + [HF ZBang/USEF-TSE](https://huggingface.co/ZBang/USEF-TSE) (**有 checkpoint**) | 否 | 部分 | 否(SI-SDR) | **可零成本试**; CC BY-NC(非商业) |
| **USEF-TP** ([arxiv 2501.03612](https://arxiv.org/abs/2501.03612)) | 2025 | USEF-TSE 扩展, 联合 TSE + Personal VAD | 同上团队 | 否 | 部分 | 否 | 联合 VAD 方向 |
| **SpEx+** (Ge, Interspeech 2020) | 2020 (经典 baseline) | 时域 multi-scale encoder + speaker embedding | [xuchenglin28/speaker_extraction_SpEx](https://github.com/xuchenglin28/speaker_extraction_SpEx) | 否 | 严苛(原 15s, 现 5s) | 否(SI-SNR) | 经典 baseline, 需重训 |
| **X-TF-GridNet** | 2023 | TF-GridNet + adaptive speaker embedding fusion | [HaoFengyuan/X-TF-GridNet](https://github.com/HaoFengyuan/X-TF-GridNet) | 否 | 一般 | 否(SI-SDR) | 需训练 |
| **ClearerVoice-Studio** ([arxiv 2506.19398](https://arxiv.org/abs/2506.19398)) | Interspeech 2025 | Alibaba 统一框架含 4 类 TSE (audio/visual/gesture/EEG), 8kHz audio-only | [modelscope/ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) **Apache-2.0** | 部分(Aishell1Mix 数据兼容) | 一般 | 否(SI-SDR) | 大厂背书, 框架完整 |
| **AlphaFlowTSE** ([arxiv 2603.10701](https://arxiv.org/abs/2603.10701)) | 2026 | One-step flow-matching 生成, 比 diffusion TSE 快(无需迭代采样) | 待开源 | 否 | 鲁棒 | 否(SI-SDR) | 最新生成范式 |
| **Multi-Level Speaker Rep** ([arxiv 2410.16059](https://arxiv.org/abs/2410.16059)) | 2024 | 多层 speaker 表征做 enrollment cue | 待查 | 否 | 鲁棒 | 否 | 强化 enrollment 路线 |
| **Listen to Extract / Onset-Prompted** ([arxiv 2505.05114](https://arxiv.org/abs/2505.05114)) | 2025 (cited 8) | 用 onset prompt 触发 TSE | 待查 | 否 | 一般 | 否 | 新颖 conditioning |
| **Discriminative-Generative TSE** ([arxiv 2601.06006](https://arxiv.org/abs/2601.06006)) | 2026 | 判别+生成混合, recovery from short enrollment | 待查 | 否 | **短enroll鲁棒** | 否 | 短enroll方向 |
| **EvoTSE - Evolving Enrollment** ([arxiv 2604.06810](https://arxiv.org/abs/2604.06810)) | 2026 | 解决 enrollment 鲁棒性 | 待查 | 否 | 鲁棒 | 否 | 短enroll方向 |
| **Training-Free Multi-Step Inference** ([arxiv 2603.10921](https://arxiv.org/abs/2603.10921)) | 2026 | 冻结预训练模型 + 迭代细化, 处理短 enrollment | 待查 | 否 | **短enroll鲁棒** | 否 | 零训练 POC 候选 |
| **TSELM** ([arxiv 2409.07841](https://arxiv.org/abs/2409.07841)) | 2024 | **离散 token + LM**, CE 分类 loss 替代回归 SI-SDR, **本质绕开波形重建陷阱** | [Beilong-Tang/TSELM](https://github.com/Beilong-Tang/TSELM) + [HF TSELM-L](https://huggingface.co/) (**有 checkpoint**) | 否(英文 Libri2Mix) | 4s enroll | **是**(CE分类, 不重建波形) | **可零成本试**; License 查仓库 |

### 方向 B. TSE + ASR 联合训练 (解决 EoW 陷阱 - 关键!) ⭐⭐⭐

| 模型/论文 | 年份 | 一句话原理 | 开源 | 中文 | 短enroll | 绕SI-SDR | 状态 |
|----------|------|-----------|------|------|---------|---------|------|
| **TS-ASR-AD** (Honda RI, [ISCA 1299](https://www.isca-archive.org/interspeech_2025/maeda25_interspeech.pdf)) | Interspeech 2025 | **TS-ASR + VAD 联合**, FiLM 适应 speaker embedding, WavLM SSL encoder, CTC + CE loss. WER 15.59→12.36 (Libri2Mix 100h), 6.61/14.81 (460h) | 待开源 (Honda RI) | 否(英文 Libri2Mix/LibriSpeechMix) | 5s enroll | **是**(纯 ASR+VAD loss) | **解决 EoW 关键**; 待开源 |
| **Whisper-Sidecar** ([arxiv 2407.09817](https://arxiv.org/abs/2407.09817)) | Interspeech 2024 | **frozen Whisper + Sidecar 分支 embedding 空间分离 + TTI 轻量分类头** 选 target, 不用 speaker embedding | [LingweiMeng/Whisper-Sidecar](https://github.com/LingweiMeng/Whisper-Sidecar) **MIT** | **是** (zero-shot Aishell1Mix CER 28.94%, batch-tune 17.81%) | 3s enroll | **是**(纯 ASR loss, embedding 空间) | **最契合, MIT license** |
| **TS-ASR with Whisper FDDT** (Polok, [arxiv 2409.09543](https://arxiv.org/abs/2409.09543)) | 2024 | STNO(Silence/Target/Non-target/Overlap) 概率掩码 → frame-level affine 变换 Whisper 隐状态; **不用 enrollment**, 用 diar 输出 | [BUTSpeechFIT/TS-ASR-Whisper](https://github.com/BUTSpeechFIT/TS-ASR-Whisper) (这就是我们项目主线 DiCoW 源头!) | 否(英文) | N/A(用 diar) | **是**(纯 ASR loss) | **我们已用** (code/TS-ASR-Whisper) |
| **Decoupling Sep-ASR** (YF Yang, [arxiv 2503.17886](https://arxiv.org/html/2503.17886v1)) | ICASSP 2025 | **解耦训练**: 分离前端用 SI-SDR, ASR 后端用 clean 数据训练, 推理拼接。解决"分离 artifacts 破坏 ASR" | 论文 PDF, 代码待查 | 否 | N/A | **是**(解耦) | 解决感知-识别鸿沟 |
| **End-to-End TS-ASR + VAD** (Liu, [ScienceDirect S1051200426000862](https://www.sciencedirect.com/science/article/abs/pii/S1051200426000862)) | 2026 | SP-ASR (Streaming Personal ASR) + VAD 融合, ctWER 大降 | 论文, 代码待查 | 否 | 5s | 是 | streaming 方向 |
| **Neural uncertainty TS-ASR** (Shi, [ScienceDirect S0885230821001200](https://www.sciencedirect.com/science/article/abs/pii/S0885230821001200)) | 2022 | 时域 TSE + RNN-T 联合, 加不确定性估计 | 论文 | 否 | 一般 | 是(RNN-T) | 经典联合训练 |
| **Joint ASR + Speaker Role SOT** ([arxiv 2506.10349](https://arxiv.org/html/2506.10349v1)) | 2025 | Whisper + role embedding + SOT 联合训练 | 待查 | 否 | N/A | 是 | SOT 路线 |

### 方向 C. Embedding 空间分离 (不重建波形)

| 模型/论文 | 年份 | 一句话原理 | 中文 | 备注 |
|----------|------|-----------|------|------|
| **Whisper-Sidecar** ([arxiv 2407.09817](https://arxiv.org/abs/2407.09817)) | Interspeech 2024 | Conv-TasNet 灵感 mask 在 Whisper encoder embedding 空间分离 + TTI 选 target | **是** | ⭐ 唯一明确"embedding 空间分离+ frozen Whisper"实例化 |
| **TSELM** ([arxiv 2409.07841](https://arxiv.org/abs/2409.07841)) | 2024 | WavLM 离散 token + LM (本质 embedding/token 空间操作), 不重建波形 | 否 | 离散化绕开回归 |
| **AudioSep** ([github audio-agi/audiosep](https://github.com/audio-agi/audiosep)) | 2023 | CLAP text-guided 通用声音分离(非 speaker 但理念可借鉴) | 部分 | 启发: 用任何 cue(CLAP/embedding)引导分离 |
| **CLAPSep** ([arxiv 2402.17455](https://arxiv.org/html/2402.17455v3)) | 2024 | CLAP 联合空间做多模态音频分离 | 否 | 同上 |
| **FlowSep** ([arxiv 2409.07614](https://arxiv.org/html/2409.07614v2)) | 2024 | Rectified Flow + 语言查询分离 | 否 | AudioSep 升级 |

### 方向 D. 重叠专用 diar/分离

| 模型/论文 | 年份 | 一句话原理 | 中文 | 备注 |
|----------|------|-----------|------|------|
| **pyannote 3.0 overlap detection** ([HF pyannote/overlapped-speech-detection](https://huggingface.co/pyannote/overlapped-speech-detection)) | 2024 | 重叠检测模型, 与 diar 联合可归属重叠区 | 语言无关 | 现成可嵌入 |
| **TS-VAD** (Medennikov CHiME-6) | 2018/2024 | speaker profile → 预测每说话人 VAD | 否 | 经典方向 |
| **TS-VAD+** ([APSIPA 2025 P333](https://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P333.pdf)) | APSIPA 2025 | 模块化 TS-VAD, 用 prior speaker embedding | 否 | 升级版 |
| **Overlap-Aware CSS without Permutation** ([Yu Interspeech 2023](https://www.isca-archive.org/interspeech_2023/yu23c_interspeech.html)) | 2023 | 显式识别 non-overlap 段引导分离 | 否 | CSS 思路 |
| **Overlap-Adaptive Hybrid Diar+ASR Seg** ([arxiv 2505.22013](https://arxiv.org/html/2505.22013v1)) | 2025 | 重叠自适应 hybrid diar | 否 | 最新 |
| **DiariZen** | 2024-2025 | 重叠改进的 diar | 待查 | 最新 diar 工具 |

### 方向 E. 中文专用

| 模型/论文 | 年份 | 一句话原理 | 备注 |
|----------|------|-----------|------|
| **Aishell1Mix 数据集** ([github huangzj421/Aishell1Mix](https://github.com/huangzj421/Aishell1Mix)) | 2020+ | 基于 AISHELL-1 的中文 2/3 说话人混合数据集 | Whisper-Sidecar / SpeechBrain 都引用 |
| **SpeechBrain Aishell1Mix recipe** ([speechbrain/.../Aishell1Mix](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Aishell1Mix)) | 2024 | **只有 separation 没 extraction**, 用 SepFormer | ⚠️ 不是 TSE, 需自改 |
| **Whisper-Sidecar Aishell1Mix 实测** ([arxiv 2407.09817](https://arxiv.org/abs/2407.09817)) | 2024 | zero-shot CER 28.94%, one-batch-tune 17.81% (large) | ⭐ 中文能力铁证 |
| **广工大交叉注意力 TSE** ([GDUT 2024](https://html.rhhz.net/GDGYDXXB/html/1718357042673-1634956375.htm)) | 2024 | 交叉注意力机制改进 TSE | 中文核心期刊 |
| **Real Conv Mixtures TSE** ([ISCA li25da](https://www.isca-archive.org/interspeech_2025/li25da_interspeech.pdf)) | Interspeech 2025 | **真实对话混合 (中英多语)**, 缩合合成-真实 gap | 唯一中英双语 TSE 论文 |
| **CN119007728A 专利** ([Google Patents](https://patents.google.com/patent/CN119007728A/zh)) | 2024 | 国内目标说话人提取方法专利 | 工业界跟进 |
| **知乎/CSDN TSE 综述** ([知乎 ICASSP2024 多策略 TSE](https://zhuanlan.zhihu.com/p/683380037) / [TSE/PSE 入门](https://zhuanlan.zhihu.com/p/630419988) / [SpEx 系列](https://zhuanlan.zhihu.com/p/507746723)) | 2024 | 中文综述 | 入门参考 |

---

## 二、Top 5 推荐 (按契合度)

### 🥇 #1. Whisper-Sidecar ([arxiv 2407.09817](https://arxiv.org/abs/2407.09817), [github LingweiMeng/Whisper-Sidecar](https://github.com/LingweiMeng/Whisper-Sidecar))

**为什么最契合** - 一举解决我们三大瓶颈:
- ✅ **绕开 SI-SDR**: Sidecar 分支在 Whisper encoder embedding 空间用 talker-dependent masks 分离, 不重建波形, 完全 ASR loss 训练 → 解决 EoW 感知-识别鸿沟
- ✅ **中文支持铁证**: zero-shot Aishell1Mix CER 28.94%, one-batch-tune 17.81% (large 模型); 大幅优于我们当前 vanilla Whisper baseline (pos CER 0.664)
- ✅ **绕开 enrollment 失效**: TTI (Target Talker Identifier) 是**轻量分类头**(linear+ReLU+softmax), **不用 speaker embedding 距离匹配**, 规避短/污染 enrollment 导致 sim 不可信问题 (我们 memory `overlap-is-cer-failure-rootcause` 提的双人重叠选 target 失败主因)
- ✅ **MIT license** + 复用我们 code/TS-ASR-Whisper 体系 (Polok FDDT 是同一 BUT 团队体系延伸)

**集成难度**: 中-高
- 训练成本: 论文用 8×V100, 200k steps; 可降到 Whisper-small-SS-TTI (242M params, 100h 可达 WER 15.75 - 已是可接受起点)
- 数据: 需自建 LibriMix (英文) + Aishell1Mix (中文) 训练; 两个数据集生成工具都开源
- 集成点: 替换 `enroll_infer.py` 里 vanilla Whisper 调用, 改成 Sidecar 模型 forward

**预期收益**: pos CER 0.3436 → 可能 0.20-0.25 区间 (双人重叠子集 0.51 失败组是主战场, Sidecar 设计针对重叠)
**风险**: 需自训 (无预训练 checkpoint 发布, 仅代码), 但训练成本可控

### 🥈 #2. TS-ASR-AD (Maeda Honda RI, [ISCA 1299](https://www.isca-archive.org/interspeech_2025/maeda25_interspeech.pdf))

**为什么契合**:
- ✅ **联合 ASR+VAD 端到端, CTC + CE loss, 完全绕 SI-SDR** → 直接解决 EoW 陷阱
- ✅ **5秒 enrollment** (近我们 1.8s, 但比 Whisper-Sidecar 3s 略长)
- ✅ WavLM SSL encoder, 96-319M params 灵活; WER 6.61/14.81 (Libri2Mix/Libri3Mix 460h)
- ✅ FiLM 适应 speaker embedding (与"enrollment-conditioned"机制相符)

**集成难度**: 中-高
- 论文未给代码链接, 但方法清晰可复现 (FiLM + WavLM + CTC + CE 头)
- 需自训练; 中文需重训 (英文 LibriMix 数据, 中文需生成 Aishell1Mix 版本)

**预期收益**: 英文 Libri2Mix 已证 15.59→12.36 (vs TS-ASR), 中文需验证; 解决 EoW 后端到端 CER 应优于 cascaded
**风险**: Honda RI 论文 2025-08 刚发, 代码开源待确认; 英文训练, 中文要重训

### 🥉 #3. TSE-through-Positive-Negative-Enrollments (Xu, NeurIPS 2025, [arxiv 2502.16611](https://arxiv.org/abs/2502.16611), [github xu-shitong/...](https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll))

**为什么契合**:
- ✅ **正/负 enrollment 对比机制完美解决"enrollment 污染"瓶颈** - 我们核心痛点 (memory `overlap-is-cer-failure-rootcause` 的 enrollment 块污染 38%)
- ✅ 不需要 clean anchor, 直接用含噪 enrollment (鸡尾酒会场景模拟)
- ✅ SI-SNRi +2.1dB over prior works (SOTA)

**集成难度**: 中
- 代码已开源 (CC BY 4.0)
- 仍走 SI-SDR/SI-SNR 路线, **EoW 陷阱仍在** (作为前端可改善 enrollment 鲁棒性, 但后端仍需 ASR 配合)
- 需自训练 (无预训练 checkpoint)

**预期收益**: 解决 enrollment 污染后, target selection 准确率应提升; 但 SI-SDR 优化的下游 ASR 风险仍在
**风险**: 与 Whisper-Sidecar 联合训练方案相比, 这仍是 cascaded; 适合做 POC 验证"enrollment 污染"是否就是失败主因

### #4. TSELM (Tang, [arxiv 2409.07841](https://arxiv.org/abs/2409.07841), [github Beilong-Tang/TSELM](https://github.com/Beilong-Tang/TSELM))

**为什么契合**:
- ✅ **离散 token + LM 是绕开 SI-SDR 的关键创新** - CE 分类 loss 替代回归, 同时输出可重建波形(可选)
- ✅ HF 有 checkpoint, 可零成本 POC
- ✅ WavLM Large 底座 (语音 SSL SOTA), 强表征

**集成难度**: 中
- WavLM Large + HiFiGAN 依赖重, 推理慢 (LM 自回归)
- 训练成本中等

**预期收益**: 论文报告 Libri2Mix 上 WER/PESQ 都强 (参考论文)
**风险**: 英文训练, 中文需重训(可考虑中文 WavLM); LM 推理慢, 效率腿可能受影响 (RTF 风险)

### #5. USEF-TSE (Zeng, ICASSP 2024, [arxiv 2409.02615](https://arxiv.org/abs/2409.02615), [github ZBang/USEF-TSE](https://github.com/ZBang/USEF-TSE), [HF checkpoint](https://huggingface.co/ZBang/USEF-TSE))

**为什么契合**:
- ✅ **不依赖 speaker embedding 选 target**, 规避短/污染 enrollment 失效
- ✅ **HF checkpoint 可零成本试** (CC BY-NC, 参赛非商业 OK)
- ✅ 预训练 backbone 直接学 enrollment-conditioned 提取

**集成难度**: 低 (有 checkpoint)
- 8kHz 输入需升采样到我们 16kHz (或重训 16kHz 版本)
- 英文训练, 中文 OOD 风险

**预期收益**: POC 验证"不靠 speaker embedding"机制是否真的鲁棒
**风险**: 商业 license 受限 (CC BY-NC), 中文 OOD; 但作为 POC 试金石价值高

---

## 三、候选 X (端到端 enrollment-conditioned ASR) 答案 ⭐

用户 memory `non-voiceprint-target-selection` 提的"自建 TSE+ASR 联合"方向, **2024-2025 已有论文实例化**:

### 候选 X 实例化方案 1: **Whisper-Sidecar** (推荐起点)
- **机制**: enrollment (3s) + 混合音频拼接 → frozen Whisper encoder → Sidecar 分离 embedding → TTI 选 target 分支 → decoder 转写
- **核心**: **embedding 空间分离 + frozen Whisper 多语言保留** (中文 zero-shot 验证)
- **训练成本**: 8×V100 200k steps (Whisper-small 242M params)
- **License**: MIT
- **契合度**: ⭐⭐⭐⭐⭐ (一举解决 EoW + 中文 + enrollment 失效)

### 候选 X 实例化方案 2: **TS-ASR-AD** (Honda RI)
- **机制**: enrollment (5s) → speaker embedder → FiLM 适应 → WavLM SSL encoder → 双 decoder (ASR CTC + VAD CE)
- **核心**: **联合 ASR+VAD 训练稳定化, 解决 CTC blank 累积**
- **训练成本**: 50k-500k steps, 96-319M params
- **License**: 待开源
- **契合度**: ⭐⭐⭐⭐ (联合训练思路强, 但英文需迁移中文)

### 候选 X 实例化方案 3: **Decoupling Sep-ASR** (ICASSP 2025, YF Yang)
- **机制**: 分离前端独立训练 (SI-SDR) + ASR 后端用 clean 训练 → 推理拼接
- **核心**: **解耦避免分离 artifacts 污染 ASR**
- **契合度**: ⭐⭐⭐ (折中方案, 部分解决 EoW 但仍 cascaded)

**结论**: 候选 X 不再是未来方向, **2024-2025 已成熟**。Whisper-Sidecar 是最值得 POC 的实例 (frozen Whisper + MIT license + 中文已验证)。

---

## 四、关键论文链接清单

### 必读 (Top 5)
1. [Whisper-Sidecar (Interspeech 2024, arxiv 2407.09817)](https://arxiv.org/abs/2407.09817) + [GitHub](https://github.com/LingweiMeng/Whisper-Sidecar)
2. [TS-ASR-AD (Interspeech 2025, ISCA 1299)](https://www.isca-archive.org/interspeech_2025/maeda25_interspeech.pdf)
3. [TSE-through-Positive-Negative-Enrollments (NeurIPS 2025, arxiv 2502.16611)](https://arxiv.org/abs/2502.16611) + [GitHub](https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll)
4. [USEF-TSE (ICASSP 2024, arxiv 2409.02615)](https://arxiv.org/abs/2409.02615) + [GitHub](https://github.com/ZBang/USEF-TSE) + [HF](https://huggingface.co/ZBang/USEF-TSE)
5. [TSELM (arxiv 2409.07841)](https://arxiv.org/abs/2409.07841) + [GitHub](https://github.com/Beilong-Tang/TSELM)

### 综述/基础
6. [Neural Target Speech Extraction Overview (Zmolikova 2023, arxiv 2301.13341)](https://arxiv.org/abs/2301.13341) - IEEE SPM 综述
7. [Survey of End-to-End Multi-Speaker ASR (He 2025, ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0885230825001500)
8. [Decoupling Sep-ASR (ICASSP 2025, arxiv 2503.17886)](https://arxiv.org/html/2503.17886v1)

### 我们已用的源头
9. [TS-ASR with Whisper FDDT (Polok 2024, arxiv 2409.09543)](https://arxiv.org/abs/2409.09543) - **就是我们 code/TS-ASR-Whisper 仓库的源头**, DiCoW 系列
10. [ClearerVoice-Studio (Alibaba, Interspeech 2025, arxiv 2506.19398)](https://arxiv.org/abs/2506.19398) + [GitHub](https://github.com/modelscope/ClearerVoice-Studio)

### 数据集
11. [Aishell1Mix (GitHub huangzj421/Aishell1Mix)](https://github.com/huangzj421/Aishell1Mix) - 中文混合数据集
12. [SpeechBrain Aishell1Mix recipe](https://github.com/speechbrain/speechbrain/tree/develop/recipes/Aishell1Mix) - ⚠️ 只有 separation 没 extraction

### 入门综述 (中文)
13. [知乎 - 目标说话人提取 TSE/个性化语音增强 PSE](https://zhuanlan.zhihu.com/p/630419988)
14. [知乎 - SpEx 系列介绍](https://zhuanlan.zhihu.com/p/507746723)
15. [知乎 - ICASSP 2024 多策略 TSE](https://zhuanlan.zhihu.com/p/683380037)

---

## 五、风险 / 门槛清单

| 风险类型 | 描述 | 缓解 |
|---------|------|------|
| **中文 OOD** | 绝大多数 TSE 论文 (USEF-TSE/TSE-Pos-Neg/AlphaFlowTSE) 英文 WSJ0-2mix/Libri2Mix 训练, 中文 OOD 风险高 (SepFormer 中文损字"无"→"雨"已证) | 优先选 Whisper-Sidecar (中文 zero-shot CER 28.9% 已验证) / TS-ASR-AD (英文训但 WavLM SSL 底座强, 中文 WavLM 强); 或自训 Aishell1Mix |
| **训练成本** | Whisper-Sidecar 8×V100 200k steps; TS-ASR-AD 50k-500k steps | L20 租算力 (项目已规划); 或先用 Whisper-small (242M) 起步 |
| **EoW 陷阱仍存** | TSE-through-Pos-Neg-Enroll / USEF-TSE 仍走 SI-SDR/SI-SNR, EoW 感知-识别鸿沟未解决 | 选 Whisper-Sidecar / TS-ASR-AD (纯 ASR loss 联合训练) 绕开; 或 Decoupling-Sep-ASR (ICASSP 2025) 解耦 |
| **License** | USEF-TSE CC BY-NC (非商业, 参赛可, 商用不行); 其他多数 MIT/Apache-2.0 | 参赛阶段全部 OK; 长期商用选 Whisper-Sidecar/ClearerVoice (MIT/Apache) |
| **预训练 checkpoint 缺失** | 多数论文只发代码不发 checkpoint (Whisper-Sidecar / TS-ASR-AD) | USEF-TSE / TSELM HF 有 checkpoint 可零成本 POC; 训练用 Aishell1Mix 现成数据 |
| **enrollment 短(1.8s)** | 论文多 3-5s enrollment, 我们 1.8s 是极端 | TSE-Pos-Neg-Enroll / Discriminative-Generative TSE / Training-Free Multi-Step 专门处理短/污染 enrollment; Whisper-Sidecar 3s 接近 |
| **重叠区时间归属** | diar 在重叠区模糊 | TS-ASR-AD 联合 VAD 训练正是为此; pyannote overlap detection 可前处理 |
| **效率腿 RTF** | TSELM (LM 自回归) / AlphaFlowTSE (生成) 推理慢, 可能 RTF >1 不满足效率 20% 腿 | Whisper-Sidecar 推理 = Whisper + 轻量 Sidecar, RTF ≈ vanilla Whisper + 5-10%; TS-ASR-AD 97M 小模型快 |

---

## 六、行动建议 (下一步)

### 立即可做 (零训练成本 POC)
1. **下载 USEF-TSE HF checkpoint**, 跑我们 5-10 条重叠死区样本, 看是否比 SepFormer 选 target 准 (验证"不靠 speaker embedding"是否真的规避污染)
2. **下载 TSELM HF checkpoint**, 同上 POC, 看离散 token 路线对中文 OOD 程度

### 短期 (1-2 周, 单 GPU 可训)
3. **复现 Whisper-Sidecar** small 版本 (242M params, 100h 训练数据): 用 Aishell1Mix 训练集 (中文) + Libri2Mix 子集 (英文) 联合训练, 验证中文 CER < vanilla 0.3436
4. **POC TSE-through-Pos-Neg-Enroll** 在 enrollment 污染子集上 (我们已有 38% enrollment 污染标注), 看是否解决 target selection 失败

### 中期 (L20 租算力后)
5. **训练 Whisper-large-Sidecar** (242M→18.6M 可训参数) 全量 Aishell1Mix + Libri2Mix, 集成到 `submit_infer.py --asr-backend sidecar`, 直接对比当前 qwen 主线 0.3436
6. **TS-ASR-AD 中文版自训**, 用 WavLM 多语言底座 + Aishell1Mix, 验证 ASR+VAD 联合训练对重叠区 CER 的提升

---

## 调研边界声明

- 时间: 2024-2026 最新论文为主, 经典 SpEx/VoiceFilter 仅作背景
- 重点关注"能否绕开我们今天瓶颈" (SI-SDR / 中文 / 选路 / 重叠), 不是泛泛列技术
- 开源模型标 license + 是否有预训练权重
- 所有链接可验证 (arXiv 号 / GitHub URL / HF URL)
- ⚠️ ICASSP/Interspeech 2025 部分论文 (TS-ASR-AD) 代码开源状态需后续追踪确认
