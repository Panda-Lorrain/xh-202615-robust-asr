# 美的目标说话人 ASR 参赛方案（XH-202615）

> ⚠️ **2026-07-08 更新**：主办方 CER 口径脚本到手并**坐实对齐**——`normalize_text`(NFKC+lower+去 P*标点和空白) + `CERMetric` 累计池(total_err/total_char, editdistance 库)。落地三件：①`eval_metrics.py` 加官方口径(逐行等价,12 边界全 Δ=0)；②`recompute_official_cer.py` 重算全量；③**修提交归一漏洞**(4-agent 对抗验证发现 cn2an/zhconv **原未声明依赖**→主办方环境必缺→digit_postproc 静默失效→CER 0.595 实际回 0.661；已建 `code/requirements.txt` + RuntimeWarning 告警 + to_submission SSOT)。**口径切换利好**：vanilla overall 0.664→**0.595** / thr0.27 含拒 **0.703** / H3 dicow sim[0.2,0.3) **1.609** 更稳 / digit_postproc -0.033 坐实。caliber-A 风险降级。详见 `AGENT_HANDOFF.md`【2026-07-08】段 + memory `official-scoring-spec`。

> ⚠️ **2026-07-07 晚更新**：MiMo-V2.5-ASR 调研闭环（全量 1362 条实测 CER **0.417 vs vanilla 0.661**，纯文字句验证非口径红利；云端不能进提交三红线 + 蒸馏不现实数据/容量/范式三障碍）+ **数字后处理集成**（vanilla 阿拉伯→中文数字对齐 ref，全量 0.661→**0.632** / 含数字句 0.739→0.608；`text_utils.digit_postproc` + enroll_infer，commit `0aea5ca`）。**最新状态/待办见 `AGENT_HANDOFF.md`【2026-07-07 晚】段（下个 agent 必读）**。

> ⚠️ **2026-07-07 更新**：统一 thr 选点 **T27 完成**（B 集 A 集模拟 + 5-agent 对抗验证 → **统一 thr=0.27** 区间 [0.26,0.29]，分 thr oracle 损失 6.46 分；已修守卫/run_baodi B 模式；⚠️ thr 待主办方 CER 口径定，caliber-A 假设未坐实是核心风险）。

> ⚠️ **2026-07-06 晚更新**：可复现性改造 T26 完成（核查 6 项硬要求全达标，fp16 run-twice delta=0 无需 fp32）+ vanilla 集成 T25 完成（pos CER 0.667 / neg RR 98.52%）。**最新状态/待办见 `AGENT_HANDOFF.md`【晚】段（下个 agent 必读）**。下面"当前阶段/下一步候选"部分已过时（vanilla 集成不再待办，可复现性已完成）。

## 项目概况
- **题目**：XH-202615《复杂交互场景的抗干扰语音指令识别技术》（美的集团发榜）
- **任务**：给定唤醒音频（enrollment，目标说话人），在带噪（SNR −5~5dB）+ 多说话人重叠（≤2人，0–100%）的识别音频中**只转写目标说话人指令、拒识非目标**
- **评分**：目标 CER 40%、拒识率 40%、推理效率 20%（L20 GPU）
- **当前阶段**：✅ **真实测试集 A 真测基线已出 + 组合主线架构极限确认**（2026-07-04）。历史：T14 pipeline(diar+STNO+DiCoW)→T17 wespeaker 锁定→T18 SE/CAM++证伪→T19 langfix→T20 SE 条件化→T22 babble 归因(H3 确证) + 仿真 450 画像(可用率 14%/babble 死区 0%)。**2026-07-04 真测**：datasetA 到手(pos 1364 测 CER + neg 474 测 RR，**单通道** 16k/16bit mono，enrollment ~1.8s 超短，唤醒词 20 种) → **单通道确认**(DSENet/VSAEC/DOA/KWS 空间路线全弃，组合主线是唯一路线)。真测适配(`make_pairs_from_datasetA`+`eval_datasetA`+繁简归一 zhconv) + Gap3 批量化(`enroll_infer --pairs` 模型加载1次，enroll 39→11s) + langfix 加强。**真实基线**：pos CER **1.25**(correct 31%)/neg RR **77%**(thr=0.2，可调 **98.5%**@thr=0.4 / 99.2%@thr=0.45)/RTF 0.16–0.24(4060，pos 全量 13.8min batch)。三方案攻短板全受挫：langfix 边际(英文 31.6%→18.5%，CER 仅降 0.028)/STNO 无效(babble 重 sim 0.024→0.038)/SE-DiCoW 架构不兼容(mt_num_speakers=2 多 speaker+self-enrollment 范式，短音频 OOD)。⚠️ **核心认知**：组合主线 cascaded(wespeaker+diar→STNO→DiCoW) 在极重 babble 下 CER **1.25–1.4 是能力极限**(target 声纹 median sim 0.28 / 低 sim 桶 correct 仅 30% + mel 退化转写崩；注：sim<0.06 仅 7.7% 非主流，答辩别引用 sim<0.06)，调参/小改破不了。**保底决策（2026-07-04 真测三档 + 3-agent 对抗审查修正）**：**关LLM**（`--no-llm`，**trade-off 非全面优于**：关LLM 赢 neg RR 98.5%>96.2% + RTF 0.24<1.01 4×；**开LLM 赢 pos 救回**——28 条 LLM 救回的 pos 里 26 条 CER=0.000 完美，原"pos 持平"错；选关LLM = 为效率20%+RR40% 牺牲 pos 救回）+ sim_thr 待主办方口径定（CER 均值→0.4/0.45 / correct→0.2 / pos 不许拒→0）。⚠️**提交用 `code/run_baodi.sh` 锁死 flag**（submit_infer 默认 llm_or_sim/thr0.2/llm ON = 灾难）；**CER 均值是幻觉陷阱**（correct_rate 才诚实 31%@thr0.2→14%@thr0.4）；**pos CER ~1.0 无 thr 能救**（架构极限）；**CER ±0.04 噪声**（langfix 边际 0.028 在噪声内不可靠）；**L20 batch=1 未实现**。详见 memory `baodi-config-no-llm`+`datasetA-spec`
- **2026-07-06 Phase 1 突破**：✅ **H3 强证伪**——DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】，改用 **vanilla Whisper-large-v3-turbo + 声纹切 target timeline** 路线（zero-training，全量 1362 条 pos）。**转写 CER 几乎减半**：vanilla **0.664** vs dicow 1.248；correct_rate vanilla **45.6%** vs dicow 31.4%；near_perfect vanilla 20.8% vs dicow 14.8%；**英文幻觉率 vanilla 0.59% vs dicow 18.80%**（DiCoW 条件化主动造孽，langfix 是治标，vanilla 从根消灭）。thr=0.20 overall CER：vanilla **0.711** vs dicow 1.241（vanilla 终于把 overall 拉到 <1）→ **CER 40% 腿从 ~0 分变 ~11 分**（线性 (1-0.711)×40，待主办方 CER 口径确认）。sim 分桶铁证 DiCoW 条件化最毒：sim[0.2,0.3) vanilla 0.746 vs dicow **1.606** / sim[0.3,0.4) vanilla 0.623 vs dicow **1.523**（Δ-0.90，条件化最反作用）。机制：diar+wespeaker 选 target（复用 enroll_infer）→ 切 target timeline 段（含重叠区）拼接 → vanilla Whisper 转写（去掉 stno_mask/FDDT 条件化）。**答辩弹药**：「cascaded 条件化机制在极重 babble 下反作用（sim 0.2-0.4 桶 CER 1.5-1.6、英文幻觉 18.8%），改用 target extraction + vanilla Whisper，CER 几乎减半」——契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。产物 `code/exp_vanilla_vs_dicow.py`+`analyze_vanilla_full.py`+`exp_vanilla_full.json`，memory `h3-dicow-conditioning-backfire-vanilla`

## 📂 文档导航（按此顺序读）
| 文档 | 作用 |
|---|---|
| **00_技术路线总纲与行动地图.md** | ⭐ 入口：全局架构 + 评分→模块映射 + 行动甘特（W1–W7）+ 单/多通道双预案 + 差异化矩阵 |
| **01_模块技术细节全解_答辩级.md** | M1–M7 每模块原理+公式+设计选择+开源资产+答辩问答 + 真实数据迁移 + 15 核心问答 |
| **02_上限候选深读.md** | 候选 X（端到端联合，主押）vs 候选 Y（音频大模型，探索），含前沿对标 |
| **03_答辩FAQ与风险预案.md** | 6 评委视角 FAQ（对抗验证）+ 10 类风险预案 + 完整性 critic（✅ 已完成，5 节，47-agent 对抗生成 + 2026-07-04 真测横幅） |
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
2. **评分→模块映射**：CER40%→DiCoW+中文家居微调+数据增强+可选RASTAR纠错；拒识40%→**声纹置信度(max_sim)锚信号 + LLM语义/PVAD辅助校验**（⚠️ decide_reject 实为 AND，LLM 只减拒不加拒，"三路融合"强项定位已证伪 GAP4，答辩勿列）；效率20%→TS-RNNT形态(Hadamard积预注册,RTF=vanilla)+Whisper量化/蒸馏/流式。
3. **差异化策略（凭什么赢）**：D1数据增强极致 / D2中文家居微调 / D3端到端联合训练（PoC 未做，未来方向）/ D4声纹锚信号拒识+LLM/PVAD辅助（⚠️ 实为 AND 非融合强项，GAP4 证伪）/ D5效率优化。心法：比赛是"完成度+适配度+效率"竞赛，把开源baseline在中文/家居/极端SNR/超短enrollment调到极致。
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
- **可选依赖必须显式声明 + 缺失可见（2026-07-08 workflow④ 教训）**：cn2an/zhconv 只在本地 .venv、项目无 requirements.txt → 主办方全新环境必缺 → digit_postproc 静默 graceful 跳过 → 提交 text 数字不归一 → 官方口径 CER 0.595 实际回 0.661（数字后处理 0.033 收益全丢）。规则：任何 graceful skip 的可选依赖必须 ①声明到 `code/requirements.txt` ②缺失时 RuntimeWarning 可见（非静默 except return）③提交链路末端（to_submission）SSOT 兜底

## 下一步候选（2026-07-06：Phase 1 突破，vanilla 路线 CER 减半，保底不再是仅选项）
1. ✅ ~~真实测试集 A~~（到手，真测完成）+ ✅ ~~通道数~~（**单通道确认**，空间路线全弃）
2. ⚠️ **CER 破局战略决策**（组合主线 CER 1.4 是架构极限）：①保底交付(langfix+sim_thr=0.4 RR 99%+答辩归因，确定) ②端到端联合 X(反 cascaded，CLAUDE.md 上限候选，大工程) ③babble 专用源分离(SepFormer 提 target mel，高成本) —— **2026-07-06 Phase 1 已开拓第④路线：vanilla Whisper + target timeline extraction（zero-training，CER 1.25→0.664 减半），无需大工程即可斩获大部分 CER 收益**
3. 🔧 **保底交付**（进行中）：langfix 保留 + sim_thr=0.4 调 RR 99% + 效率 RTF 0.2 + 答辩重点(babble 归因/单通道确认/工程优化 Gap3·繁简·langfix/诚实组合主线极限/Phase 1 vanilla 突破反 cascaded)
4. ⚡ **P2 Phase 1 落地任务**（Phase 1 后续，按优先级）：① **vanilla 集成 submit_infer**（最高优，`--asr-backend {dicow,vanilla}` 切换，把 0.664 变提交数字）② 声纹强化（CAM++ per-speaker / US-PVAD 改善 target timeline 切割，尤其低 sim 桶）③ 数字 initial_prompt 锦上添花（家居指令数字/温度场景）④ sim_thr 待主办方评测口径定（CER 均值→0.4 / correct→0.2 / pos 不许拒→0）
5. ⚡ **L20 耗时验证**：推理脚本显存自适应(L20 48GB 大 batch) + 租 AutoDL L40 验证端到端(官方 L20 评效率，本机 4060，memory `l20-eval-hardware`)
6. 📄 **答辩演练**：等 `03_答辩FAQ与风险预案.md` 就绪后做(README 进度图作开场)
