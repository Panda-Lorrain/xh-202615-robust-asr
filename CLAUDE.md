# 美的目标说话人 ASR 参赛方案（XH-202615）

## 项目概况
- **题目**：XH-202615《复杂交互场景的抗干扰语音指令识别技术》（美的集团发榜）
- **任务**：给定唤醒音频（enrollment，目标说话人），在带噪（SNR −5~5dB）+ 多说话人重叠（≤2人，0–100%）的识别音频中**只转写目标说话人指令、拒识非目标**
- **评分**：目标 CER 40%、拒识率 40%、推理效率 20%（L20 GPU）
- **当前阶段**：✅ **组合主线端到端跑通 + 标准化交付 + 仿真水平画像已出**（2026-07-03）：T14 完整 pipeline（diar+STNO+DiCoW 真 target 转写）→ T17 enrollment 锁定（wespeaker 256d）→ T18 SE增强 / CAM++证伪（per-speaker sim 0.191<0.218 弃，维持 wespeaker）/ LLM拒识 F1=0.878 → T19 langfix（修 DiCoW language 死代码 bug）→ T20 SE 条件化 post-fix → T22 babble 归因（H3 确证：FDDT/STNO 低 target 覆盖劣化，**非 Whisper 基座问题**，vanilla Whisper 三角定位已证）。标准化入口 `code/submit_infer.py`（→ result.json+timing.json）。**本次 450 仿真集全量实测**（`--no-llm` 聚焦 CER）：可用率 14.0%（CER<0.5）/ babble 死区 0% / overlap 0.41→0.01 / SNR −5:2.7%→+5:24.7% / RTF 0.058 纯·0.27 端到端（4060）。可视化 `docs/progress_overview.png` + `docs/cer_progress_dashboard.html`。⚠️ **真瓶颈已归因、卡在外部依赖**：真实 A 集 + 通道数（仿真深挖边际递减，见 memory `stop-digging-on-sim-data`）

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

## 资料结构（E:\midea_target_asr\）
- **`papers/` PDF 19 篇**：10 命名核心 + US-PVAD + SELD + 5 篇 TS-ASR（FDDT/DiCoW/SE-DiCoW/TS-RNNT/NOTSOFAR）+ CUSIDE-array + 智慧家庭语音意图综述
- **`_txt/` 18 篇全文**（pdftotext 提取）：10 美的论文 + FDDT/DiCoW/SE-DiCoW/TS-RNNT + 智慧家庭综述 + CUSIDE + SELD
- `pdf2txt.py` — 纯 Python zlib PDF 提取（无库时备用）
- **`code/` 代码区**：⭐`submit_infer.py`（标准化推理入口，4 阶段 subprocess）/ `enroll_infer.py`（wespeaker 声纹+diar+DiCoW 转 target）/ `se_denoise.py`+`noise_classify.py`（DeepFilterNet3 条件化）/ `llm_reject.py`（Qwen2.5-3B）/ `fuse_eval.py`（多策略融合扫工作点）/ `eval_metrics.py`+`eval_full_test.py`（评测，⚠️ `eval_metrics.py` 无 CLI，复用需 `import cer()` 或用 `eval_full_test.py` 包装）/ `simulate_pipeline.py`（仿真）/ `make_readme_progress.py`（README 进度图生成）/ DiCoW-inference、TS-ASR-Whisper（开源仓库，gitignore 不入库）
- **`test_wav/` / `test_wav_clean/` 测试音频**

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
- **文档与实现要核实**：`eval_metrics.py` 实际**无 CLI**（`__main__` 仅自测），`交付/使用说明.md` 写的 `--result/--manifest/--out` 是理想用法未实现；复用评测用 `import cer()` 或 `eval_full_test.py` 包装
- **CER 均值会被幻觉扭曲，correct_rate 更诚实**：babble 重复循环幻觉使 hyp 超长、拉高 CER 均值，制造"SNR 越高 CER 反升"假象；看可用率用 correct_rate(CER<0.5 占比)，看绝对转写质量看 cer_accepted_only

## 下一步候选（2026-07-03：T14-T22 已完成，卡外部依赖）
1. ⏳ **等真实测试集 A**（报名后发邮箱）—— 到手即用 `eval_full_test.py` 真测，取代仿真 450 条（**最高优先级，外部阻塞**；绝对值不可从仿真外推）
2. ⏳ **确认测试集通道数**（决定 DSENet/KWS 空间路线能否用）—— 向主办方确认或看 A 集数据；判断线索见 `00` 总纲第四节（**外部阻塞**，单/多通道分水岭）
3. 🔧 **攻 babble 工程兜底**（T22 杠杆已明，**不依赖 A**）：提 STNO target 覆盖率 / 全程 language forcing(zh) / babble 专用源分离 —— babble 当前 0% 可用是最大短板，归因已清
4. ⚡ **L20 耗时验证**：推理脚本显存自适应（L20 48GB 大 batch）+ 租 AutoDL L40 验证端到端耗时（官方 L20 评效率分，本机仅 4060，见 memory `l20-eval-hardware`）
5. 📄 **答辩演练**：等 `03_答辩FAQ与风险预案.md` 就绪后做（README 已有进度概览图可作答辩开场素材）
