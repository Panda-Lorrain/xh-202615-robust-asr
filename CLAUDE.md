# 美的语音比赛资料库（XH-202615）

## 项目概况
- **题目**：XH-202615《复杂交互场景的抗干扰语音指令识别技术》（美的集团发榜）
- **任务**：给定唤醒音频（enrollment，目标说话人），在带噪（SNR −5~5dB）+ 多说话人重叠（≤2人，0–100%）的识别音频中**只转写目标说话人指令、拒识非目标**
- **评分**：目标 CER 40%、拒识率 40%、推理效率 20%（L20 GPU）
- **当前阶段**：✅ 资料收集完成 + 文档体系就绪（2026-06-27）；技术路线待**通道数确认**后定稿

## 📂 文档导航（按此顺序读）
| 文档 | 作用 |
|---|---|
| **00_技术路线总纲与行动地图.md** | ⭐ 入口：全局架构 + 评分→模块映射 + 行动甘特（W1–W7）+ 单/多通道双预案 + 差异化矩阵 |
| **01_模块技术细节全解_答辩级.md** | M1–M7 每模块原理+公式+设计选择+开源资产+答辩问答 + 真实数据迁移 + 15 核心问答 |
| **02_上限候选深读.md** | 候选 X（端到端联合，主押）vs 候选 Y（音频大模型，探索），含前沿对标 |
| **03_答辩FAQ与风险预案.md** | 6 评委视角 FAQ（对抗验证）+ 10 类风险预案 + 完整性 critic（生成中） |
| `paper_index.md` | 全部论文索引（分级/完整标题/下载状态，已核实修正） |
| `核心论文精读与方案.md` | #1 拒识 / #2 ASE-PVAD / #3 KWS空间 / #4 DSENet |
| `论文精读_增强与纠错路线.md` | #6 RASTAR / #7 VSAEC / #8 VPIDM |
| `论文精读_US-PVAD_超短参考.md` | #3 US-PVAD + 与 #2 对比 |
| `资料扩展_TS-ASR与开源资产.md` | TS-ASR 四件套 + 组合主线 + github 资产 |

## 资料结构（E:\midea_papers\）
- **PDF 19 篇**：10 命名核心 + US-PVAD + SELD + 5 篇 TS-ASR（FDDT/DiCoW/SE-DiCoW/TS-RNNT/NOTSOFAR）+ CUSIDE-array + 智慧家庭语音意图综述
- **`_txt/` 16 篇全文**（pdftotext 提取）：10 美的论文 + FDDT/DiCoW/SE-DiCoW/TS-RNNT + 智慧家庭综述 + CUSIDE
- `pdf2txt.py` — 纯 Python zlib PDF 提取（无库时备用）

## 核心结论
1. **组合主线（稳健底盘，通道无关）**：`Personal VAD(产生STNO) + DiCoW/SE-DiCoW(Whisper TS-ASR转写) + Qwen-2.5-3B LLM语义拒识`。每环节开源。单通道即可跑通，是下限保障。
2. **评分→模块映射**：CER40%→DiCoW+中文家居微调+数据增强+可选RASTAR纠错；拒识40%→三路融合(声纹置信度+LLM语义合理性+PVAD检测)；效率20%→TS-RNNT形态(Hadamard积预注册,RTF=vanilla)+Whisper量化/蒸馏/流式。
3. **差异化策略（凭什么赢）**：D1数据增强极致 / D2中文家居微调 / D3端到端联合训练 / D4三路融合拒识+自适应CoT / D5效率优化。心法：比赛是"完成度+适配度+效率"竞赛，把开源baseline在中文/家居/极端SNR/超短enrollment调到极致。
4. **上限候选**：X=端到端联合训练（主押，契合出题方"反cascaded"审美，对标TS-ASR-AD Interspeech2025）；Y=音频大模型一体化（⚠️通用音频LLM不区分说话人需speaker-aware encoder改造，效率风险大，作探索分支）。
5. **双预案**：单通道→组合主线直接用；多通道→前端叠加DSENet/VSAEC空间提取。无论通道数，组合主线都是下限。
6. **出题方偏好**：PVAD+短enrollment、LLM拒识、端到端反cascaded、空间/DOA多通道、扩散增强。团队：美的(Yu Gao/Wenbin Zhang)+中科大杜俊+东南大学(景康祺)+GT李锦辉。
7. ⚠️ **单/多通道是分水岭，待确认**（决定空间路线DSENet/KWS能否用）—— 当务之急。

## 开源资产
- **CAM++ 声纹**（3D-Speaker / ModelScope，192d）
- **DSENet / VSAEC**（github.com/jingkangqi，⚠️ 无预训练权重/数据/License，仅代码骨架）
- **DiCoW / SE-DiCoW**（github.com/BUTSpeechFIT/TS-ASR-Whisper，✅ HF 权重 `BUT-FIT/SE_DiCoW`，基座 Whisper-large-v3-turbo）
- **Qwen-2.5-3B**（拒识基座）
- **SpeechBrain TS-ASR recipe**（含 VAD 分支，可运行工程 baseline）

## 工具
- **pdftotext**：`E:\poppler\poppler-26.02.0\Library\bin\pdftotext.exe`（git bash 调 exe 用 `E:/` 路径，**勿用 `/e/`**；传参路径用 `E:\` 或 `E:/`）
- Read 工具读 PDF 需 poppler（pdftoppm），已装好
- **Python 一律用 uv**（`uv run`/`uv add`），禁止裸 pip install（全局规则）
- pip/uv 默认源 SSL 失败时用清华源
- **PowerShell 工具当前不可用**，用 Bash 工具执行命令

## 教训（务必遵守）
- **截图 OCR 的标题必须用搜索独立核实**——曾把 IEEE 10890695 的「DASM 异常声检测」OCR 错成「OOD ASR」
- **技术细节/架构演进必须对照原文核实**——曾把 DiCoW 误记为"decoder token+cross-attn"，核实原文后纠正：**DiCoW = FDDT+QKb（encoder 侧）；cross-attention 是 SE-DiCoW 才加的**（解决重叠歧义）。答辩讲错架构演进很危险
- IEEE 付费墙论文用校园网（广州大学权限）下载
- 子 agent 精读统一用「pdftotext 提取 → Read → 7 节客观提炼」流程

## 下一步候选
1. **确认测试集通道数**（最关键，决定空间路线与整体架构走向）—— 向主办方确认或看 A 集数据
2. **搭 baseline**：W1 跑通 SE-DiCoW（`E:\midea_papers\` 下用 uv 建环境 + 拉 HF 权重 + 跑官方 demo）
3. 等 03 答辩文档就绪后做答辩演练
