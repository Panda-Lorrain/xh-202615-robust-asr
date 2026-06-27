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

## 📅 开发进度

> W 编号对应 `00_技术路线总纲与行动地图.md` 行动地图（P0–P3）

- [x] **W1**: DiCoW baseline 最小推理跑通（RTF=0.058, params 0.89G, GPU 峰值 2.13GB）
- [x] **W1**: STNO 控制实验验证（target→转 / silence→跳 / non-target→0字拒识）
- [x] **W2**: 数据仿真 pipeline（`simulate_pipeline.py`：SNR −5~5dB + 重叠 0–100% + 批量）
- [x] **W6**: 评测脚本（`eval_metrics.py`：CER / RTF / 拒识 P·R·F1）
- [ ] **W2**: 完整 DiariZen pipeline（依赖/namespace/circular 已全解，90% 通，差 wespeaker 权重）
- [ ] 中文音频测试（edge-tts SSL 受阻，改用 mimo-tts）
- [ ] **W3**: Personal VAD 前端（STNO mask 真实来源）
- [ ] **W4**: enrollment 声纹（CAM++）
- [ ] **W5**: LLM 拒识系统（Qwen-2.5-3B）
- [ ] **W7**: 接入题目给定 enrollment
- [ ] **D1/D2**: 数据增强 + 中文家居微调
- [ ] **D5**: 效率优化（TS-RNNT 形态 / 量化 / 流式）

详见 [`PROGRESS.md`](PROGRESS.md)。

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
