# 美的 AI 研究院 论文清单
> 题目：XH-202615《复杂交互场景的抗干扰语音指令识别技术》参考资料
> 整理日期：2026-06-26 ｜ **2026-06-27 核实修正：撤销虚构的「OOD ASR」——IEEE 10890695 实为异常声检测 DASM**
> 来源：https://ai.midea.com/#/papers + 用户提供的 16 个链接

## 一、下载状态总览

| 类别 | 数量 | 说明 |
|---|---|---|
| ✅ 已下载 PDF | **13 篇** | 10 篇命名核心 + US-PVAD(10848915) + SELD(10317385) + 1 篇相关参考(arxiv 2407.09807) |
| 🔒 付费墙未下载 | **5 篇** | **仅 jdkjjournal 与本题相关**；其余 4 篇（异常声检测×2、语音隐写、制冷噪声）均与本题无关 |

本地目录：`E:\midea_papers\`
文件命名规则：`序号_完整英文标题_原始ID.pdf`（仅去掉 Windows 非法字符 `?:`，不缩写、不翻译；分级与中文说明见本清单）

---

## 二、按相关度分级（重点先行）

### ⭐⭐⭐⭐⭐ 必精读（核心对口，直接关系 40%+40% 得分）

| # | 中文译名 / 说明 | 原标题（英文） | 来源 | 本地文件 |
|---|---|---|---|---|
| 1 | **智能家居语音助手「查询拒识」基准 + 基于 LLM 的改进方法** | Reject or Not: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs | arxiv 2512.10257 | `01_Reject or Not A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs_2512.10257.pdf` |
| 2 | **面向短注册语音的 Personal VAD 自适应声纹自增强** | Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech | arxiv 2601.12769 | `02_Adaptive Speaker Embedding Self-Augmentation for Personal Voice Activity Detection with Short Enrollment Speech_2601.12769.pdf` |
| 3 | **超短参考语音的 Personal VAD（US-PVAD，APSIPA 2024）** | Personal Voice Activity Detection With Ultra-Short Reference Speech | IEEE 10848915 | ✅ `Personal_Voice_Activity_Detection_With_Ultra-Short_Reference_Speech (1).pdf` |
| ~~4~~ | ~~带噪多说话人端到端 OOD 语音识别~~ | ~~End-to-End OOD Speech Recognition…~~ | ~~IEEE 10890695~~ | ❌ **已撤销：OCR 误认，10890695 实为异常声检测 DASM，见边缘 #13b** |

### ⭐⭐⭐⭐ 推荐

| # | 中文译名 / 说明 | 原标题（英文） | 来源 | 本地文件 |
|---|---|---|---|---|
| 5 | **噪声环境下的端到端方向感知 KWS（带空间先验，美的）** | End-to-End Direction-Aware Keyword Spotting with Spatial Priors in Noisy Environments | arxiv 2603.09505 | `03_End-to-End Direction-Aware Keyword Spotting with Spatial Priors in Noisy Environments_2603.09505.pdf` |
| 6 | **带噪多说话人场景下的端到端 DOA 引导语音提取（DSENet）** | End-to-End DOA-Guided Speech Extraction in Noisy Multi-Talker Scenarios | arxiv 2507.20926 | `04_End-to-End DOA-Guided Speech Extraction in Noisy Multi-Talker Scenarios_2507.20926.pdf` |
| 7 | **几何感知声学处理：联合回声消除与降噪的动态空间网络（VSAEC）** | Geometry-Aware Acoustic Processing: A Dynamic Spatial Network for Joint Echo Cancellation and Noise Suppression | midea_file | `07_Geometry-Aware Acoustic Processing A Dynamic Spatial Network for Joint Echo Cancellation and Noise Suppression_midea.pdf` |

### ⭐⭐⭐ 参考

| # | 中文译名 / 说明 | 原标题（英文） | 来源 | 本地文件 |
|---|---|---|---|---|
| 8 | **基于方差保持插值的扩散模型语音增强（VPIDM）** | Variance-Preserving-Based Interpolation Diffusion Models for Speech Enhancement | arxiv 2306.08527 | `05_Variance-Preserving-Based Interpolation Diffusion Models for Speech Enhancement_2306.08527.pdf` |
| 9 | **检索增强 + 自适应 CoT 的 ASR 命名实体纠错（RASTAR）** | Retrieval-Augmented Self-Taught Reasoning Model with Adaptive Chain-of-Thought for ASR Named Entity Correction | arxiv 2602.12287 | `06_Retrieval-Augmented Self-Taught Reasoning Model with Adaptive Chain-of-Thought for ASR Named Entity Correction_2602.12287.pdf` |
| 10 | **基于类别相关声分离的真实场景声事件定位与检测（SELD，APSIPA 2023）** | Improving Sound Event Localization and Detection with Class-Dependent Sound Separation for Real-World Scenarios | IEEE 10317385 | ✅ `shicheng-2023-apsipa.pdf` |
| 11 | **智慧家庭语音交互意图理解评测现状（中文综述）** | —— | jdkjjournal | 🔒（唯一相关的未下篇，走知网/期刊官网） |

### ⭐⭐ / ⭐ 次要或边缘（均与本题关系不大）

| # | 中文译名 / 说明 | 原标题（英文） | 来源 | 本地文件 |
|---|---|---|---|---|
| 12 | **当生成对抗网络遇到序列标注挑战** | When Generative Adversarial Networks Meet Sequence Labeling Challenges | ACL 2024 | `08_When Generative Adversarial Networks Meet Sequence Labeling Challenges_EMNLP2024.pdf` |
| 13 | **自监督增强扩散模型的异常声检测（SSDM）** | Self-Supervised Augmented Diffusion Model for Anomalous Sound Detection | IEEE 10849056 | 🔒 无关 |
| 13b | **扩散增强子中心建模的异常声检测（DASM，ICASSP 2025）** ⚠️原 OCR 误认为「OOD ASR」 | Diffusion Augmentation Sub-center Modeling for Unsupervised Anomalous Sound Detection with Partially Attribute-Unavailable Conditions | IEEE 10890695 | 🔒 无关 |
| 14 | **关于下一词预测的支持样本** | On Support Samples of Next Word Prediction | arxiv 2506.04047 | `10_On Support Samples of Next Word Prediction_2506.04047.pdf` |
| 15 | **细粒度语音情感识别的 emotion neural transducer** | Emotion Neural Transducer for Fine-Grained Speech Emotion Recognition | arxiv 2403.19224 | `09_Emotion Neural Transducer for Fine-Grained Speech Emotion Recognition_2403.19224.pdf` |
| 16 | **基于非目标转换的语音隐写（安全通信）** | Non-Target Conversion Based Speech Steganography for Secure Communication | IEEE 10849273 | 🔒 弱相关 |
| 17 | **制冷设备流致噪声的声音特性调谐（美的制冷业务）** | Tuning the sound characteristics of flow-induced noise in refrigeration equipment | ScienceDirect | 🔒 无关 |

---

## 三、出题方技术偏好（从论文归纳，6 条）

> 这批论文几乎全是美的 AI 研究院自己的研究——等于出题方把技术底牌摊开了。评委打分时心里认可的就是这些方向。

1. **核心路线 = Personal VAD + 短 enrollment 声纹**（#2 #3）——比 TSE 前端更轻量、更快，利好 20% 效率分。#3(US-PVAD) 更进一步不依赖外部 SV 模型。
2. **拒识用 LLM 做语义判断**（#1）——拒识不能只靠声纹阈值，要加「指令在智能家居场景是否合理」的判别。
3. **反 cascaded，主张端到端联合优化**（#5 #6 #7）——纯 pipeline（SV→TSE→ASR）非最优；#7 还做了 AEC+降噪联合。
4. **重视空间/DOA 多通道信息**（#5 #6 #7 #10）——四篇都在讲空间建模，⚠️ 强烈建议确认测试集是否多通道。
5. **扩散模型做低 SNR 增强**（#8）——对应 −5~5dB 考点，但需权衡效率（扩散通病慢）。
6. **鲁棒性靠数据增强广度**——题目要求 SNR −5~5dB、重叠 0–100%、多信道/语速适配；数据增强（噪声/混响/语速/信道类型）覆盖要广，防 A 集过拟合。
   > ⚠️ **撤销说明**：原第 6 条「OOD 鲁棒性是隐藏考点（#4）」系截图 OCR 误认——IEEE 10890695 实为异常声检测（DASM），并非 OOD 语音识别，与本题无关。「数据增强广度」建议本身仍成立（来自题目对 SNR/重叠/信道的鲁棒性要求），但不再援引该论文。

---

## 四、出题方学术团队

- **美的 AI 研究院**：Yu Gao、Wenbin Zhang、Rui Wang、Yi Xu、Xiaofeng Mou、Zhifei Zhang、Tianyi Wang、Jiawei Yin
- **合作方**：中科大 **Jun Du（杜俊）** 课题组（国内语音分离/增强/SELD 顶尖）、佐治亚理工 **Chin-Hui Lee（李锦辉）**、东南大学（景康祺）、东华大学、浙大、哈工程

→ 评委技术审美偏向**前端分离/提取 + 端到端联合优化**，纯后端 ASR 调参上限有限。

---

## 五、付费墙 5 篇说明

1. **jdkjjournal（智慧家庭语音意图综述）**——唯一与本题相关的未下篇，走知网 / 期刊官网（DOI 10.19784/j.cnki.issn1672-0172.2025.99.004）。
2. ~~**IEEE 10890695（OOD ASR）**~~ → **已撤销**：OCR 误认，实际为 **DASM 异常声检测（ICASSP 2025）**，与本题无关，无需找。
3. **IEEE 10849056（SSDM 异常声检测）、IEEE 10849273（语音隐写）、ScienceDirect（制冷噪声）**——均与本题无关，可不下。

> 已下的 US-PVAD(10848915)、SELD(10317385) 是用校园网（广州大学 IEEE 权限）拿到的；如需 jdkjjournal 走知网即可。
