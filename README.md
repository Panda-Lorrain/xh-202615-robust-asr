# XH-202615 复杂交互场景的抗干扰语音指令识别技术

> 🏆 **美的集团** 发榜｜挑战杯 / 竞赛赛道

## 📋 项目概述

本项目针对**复杂交互场景下的抗干扰语音指令识别**技术难题，提出了一套完整的 **Target-Speaker ASR（TS-ASR）+ 意图拒识** 解决方案。

### 🎯 核心任务

给定**目标说话人的唤醒音频**，在「带噪 + 多人重叠」的识别音频中：
- ✅ **只转写目标说话人**的语音内容
- ✅ **拒识非目标说话人**的干扰语音
- ✅ 在 **L20 GPU** 上实现实时/近实时推理

### 📊 评分维度

| 维度 | 权重 | 技术重点 |
|------|------|----------|
| 目标 CER | 40% | TS-ASR 转写精度 |
| 拒识率 | 40% | 非目标语音识别与拒绝 |
| 推理效率 | 20% | RTF、显存占用 |

---

## 🏗️ 技术架构

### 整体方案

```
┌─────────────────────────────────────────────────────────────┐
│  输入侧                                                      │
│  ┌──────────────┐                                            │
│  │  唤醒音频     │ ──→ CAM++ 声纹提取 ──→ 目标说话人嵌入        │
│  │ (enrollment) │      (处理超短参考 0.2s)                     │
│  └──────────────┘                                            │
│  ┌──────────────┐                                            │
│  │  识别音频     │ ──→ Personal VAD ──→ 帧级目标/非目标概率     │
│  │ (混合/带噪)   │     (条件化于目标嵌入)                       │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  转写侧（DiCoW / SE-DiCoW）                                  │
│  · Whisper-large-v3-turbo 作为 backbone                      │
│  · FDDT 仿射变换（STNO 条件化 encoder 每层）                  │
│  · cross-attention（目标嵌入作 enrollment 条件）              │
│  → 目标说话人 transcript                                      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  拒识侧（LLM 语义拒识 + 声纹置信度）                         │
│  · 判断指令是否针对当前设备                                    │
│  · 语义层 + 声纹层联合判断                                    │
└─────────────────────────────────────────────────────────────┘
```

### 核心技术栈

| 模块 | 技术方案 | 作用 |
|------|----------|------|
| **声纹提取** | CAM++ (192d) | 从超短唤醒音频提取说话人嵌入 |
| **语音活动检测** | Personal VAD (US-PVAD) | 帧级目标/非目标语音检测 |
| **TS-ASR** | DiCoW / SE-DiCoW | 目标说话人条件化语音识别 |
| **STNO 控制** | FDDT 仿射变换 | 通过 mask 控制转写/拒识 |
| **语义拒识** | LLM-based Reject-or-Not | 判断指令合理性与针对性 |
| **语音增强** | Diffusion-based SE | 低信噪比场景增强 |

---

## 📁 项目结构

```
xh-202615-robust-asr/
├── 📄 技术文档
│   ├── 00_技术路线总纲与行动地图.md    # 全局架构与行动计划
│   ├── 01_模块技术细节全解_答辩级.md   # 各模块详细技术方案
│   ├── 02_上限候选深读.md             # 差异化技术方向
│   └── 03_答辩FAQ与风险预案.md        # 答辩准备与风险应对
│
├── 📚 论文资料
│   ├── papers/                        # 原始 PDF 论文
│   ├── _txt/                          # 论文文本提取
│   └── paper_index.md                 # 论文索引与分类
│
├── 💻 代码实现
│   └── code/
│       ├── minimal_infer.py           # 最小推理验证脚本
│       ├── stno_experiment.py         # STNO 控制实验
│       ├── simulate_pipeline.py       # 数据仿真 pipeline
│       └── eval_metrics.py            # 评测指标计算
│
├── 📊 进度与结果
│   ├── PROGRESS.md                    # 开发进度记录
│   └── RESULTS.md                     # 实测结果与分析
│
└── 📖 论文精读
    ├── 核心论文精读与方案.md
    ├── 论文精读_US-PVAD_超短参考.md
    ├── 论文精读_增强与纠错路线.md
    └── 资料扩展_TS-ASR与开源资产.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.5+ (CUDA 12.4)
- Transformers 4.42+
- GPU: NVIDIA RTX 4060+ (8GB+)

### 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers librosa pyannote.audio
```

### 运行最小推理

```bash
cd code
python minimal_infer.py <audio.wav> [language]
```

**示例输出：**
```
[setup] device=cuda:0 dtype=torch.float16 model=E:/hf_cache/DiCoW_v3_2
[load] 2.6s | params=0.89G
[audio] EN2002a_30s.wav | 30.0s
[infer] 1.73s for 30.0s audio | RTF=0.058
[mem] peak GPU mem=2.13GB
[text] yeah yeah but i do not know about you but usually in windows right click does not do anything...
```

### STNO 控制实验

```bash
python stno_experiment.py
```

验证 FDDT/STNO 机制：
- **全-target**: 转写所有人（398 字）
- **前半 target + 后半 silence**: 只转前半（162 字）
- **全 non-target**: 完全拒识（0 字）✅

---

## 📈 实测结果

### DiCoW Baseline 性能

| 指标 | 值 | 备注 |
|------|-----|------|
| 模型参数量 | **0.89G** | Whisper-large-v3-turbo |
| 加载时间 | 2.6s | - |
| 推理时间 (30s) | 1.73s | RTX 4060 |
| **RTF** | **0.058** | 远快于实时 |
| 峰值显存 | 2.13GB | 8GB 显存绰绰有余 |

### STNO 控制验证

| STNO 构造 | 输出 | 结论 |
|-----------|------|------|
| 全-target | 398 字 | 完整转写 |
| 前半 target + 后半 silence | 162 字 | 精确控制 ✅ |
| 全 non-target | **0 字** | **内建拒识机制** ✅ |

**核心结论**: FDDT 的 STNO 条件化可验证、可控——target 转写、silence 跳过、non-target 直接产出空（拒识）。拒识不是后处理，是 FDDT 内建机制。

---

## 📚 核心论文

本项目基于以下关键论文：

1. **DiCoW** (2501.00114) - 目标说话人条件化 ASR
2. **SE-DiCoW** (2601.19194) - 增强版 DiCoW
3. **FDDT** (2409.09543) - 仿射变换控制机制
4. **US-PVAD** - 超短参考 Personal VAD
5. **Reject-or-Not** (2512.10257) - LLM 拒识基准
6. **RASTAR** (2602.12287) - 检索增强纠错

详见 [`papers/`](papers/) 目录和 [`paper_index.md`](paper_index.md)。

---

## 🎯 技术亮点

1. **STNO 内建拒识**: 通过 FDDT 的抑制式初始化，非目标帧被压到零输出，拒识成为模型内建能力
2. **超短 enrollment 支持**: CAM++ 支持 0.2s 参考音频，解决实际唤醒场景
3. **高效推理**: RTF=0.058，远快于实时，满足 20% 效率评分
4. **可控转写**: STNO mask 精确控制哪些段落被转写，机制可解释

---

## 📅 项目推进时间线

> 手机查看友好版。详细日志 [`PROGRESS.md`](PROGRESS.md)，结果 [`RESULTS.md`](RESULTS.md)。

### ✅ 已完成（06-27 → 06-30）

| 日期 | 阶段 | 关键产出 |
|---|---|---|
| 06-27 | W1 minimal 推理 | DiCoW 跑通：RTF=0.058 / 0.89G params / 峰值 2.13GB，解除"零实测"红线 |
| 06-27 | STNO 机制验证 | target→转 / silence→跳 / non-target→0字拒识（FDDT 内建拒识）|
| 06-27 | W6 评测 + W2 仿真 | `eval_metrics.py`(CER/RTF/拒识) + `simulate_pipeline.py`(SNR+重叠矩阵) |
| 06-28 | **T14** 完整端到端 pipeline | diar+STNO+DiCoW 真 target-speaker 转写，PIPELINE_EXIT=0 |
| 06-28 | 中文 CER=0 + 重叠诊断 | mimo-tts 合成；重叠 0/50/100% → CER 0.00/0.13/1.00（100% 单通道死区）|
| 06-28 | **T17** enrollment 锁定 target | `enroll_infer.py` 干净场景 sim0.816/CER=0；450 条画像 87%拒/4%正确 |
| 06-29 | **T18** 三线 de-risk | SE增强(CER 4.27→3.65) / CAM++证伪(维持 wespeaker) / LLM拒识 F1=0.878 |
| 06-29 | **T19** 集成 + langfix 修复 | `fuse_eval.py` 真实组合指标；修 DiCoW language 死代码 bug（英文 90%→72%）|
| 06-30 | **T20** SE 条件化 post-fix 重评 | =6 优于 =0（最优精细 2.022）；babble 归因深化（diar 误检+STNO 崩）|

### 🔄 当前重心（2026-06-30 起：研究 → 交付）

对照官方要求，**重心从"仿真性能深挖"转向"交付物标准化 + 真实评测"**：

- ⏳ **等测试集 A**（报名后发邮箱）——到手即用真 A 评测，取代仿真 450 条
- 🔧 **推理脚本标准化**（不依赖 A）：`submit_infer.py` 吃 enrollment+recognition → `result.json` + `timing.json`
- 📄 **设计报告 / 使用说明**（不依赖 A）：00-03 文档 + T14-T20 实验 → 比赛格式
- ⚡ **L20 耗时方案**：脚本显存自适应（48GB 大 batch）+ 租 AutoDL L40 验证

### ⏭ 未来里程碑

| 里程碑 | 预估 | 依赖 |
|---|---|---|
| 推理脚本标准化（json+耗时）| 1-2 天 | 无 |
| 设计报告 + 使用说明 | 2-3 天 | 无 |
| 测试集 A 真实评测 | 1-2 天 | **测试集 A** |
| L20/L40 耗时验证 | 1 天 | 租云 |
| 测试报告 | 1-2 天 | 真实 A 结果 |
| **作品提交** | — | ≤ **2026-09-05** |

### 📋 官方交付物清单

- [ ] 模型权重（DiCoW + diarizen + wespeaker + Qwen-2.5-3B）
- [ ] 推理脚本（python，吃 A → json 结果）
- [ ] json 测试结果 + 运行耗时
- [ ] 技术设计方案 + 测试验证方案 + 使用说明
- 评分：CER 40% / RR 40% / 效率 20%（L20 GPU）

---

## 🤝 贡献指南

本项目为竞赛项目，欢迎：
- 技术讨论与方案建议
- Bug 报告与修复
- 性能优化 PR

---

## 📄 License

本项目仅供学术研究与竞赛使用。

---

## 📧 联系方式

- GitHub: [@Panda_Lorrain](https://github.com/Panda_Lorrain)
- Issues: [项目 Issues](https://github.com/Panda_Lorrain/midea_target_asr/issues)

---

<div align="center">
  <sub>🏆 美的集团 XH-202615 挑战赛</sub>
</div>
