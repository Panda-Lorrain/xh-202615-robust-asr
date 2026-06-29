# Agent 交接文档（XH-202615 美的目标说话人 ASR）

> **下个 agent 第一时间读此文件 → CLAUDE.md → PROGRESS.md → RESULTS.md → 边缘部署规划.md → 项目阶段盘点.md**
> 更新：2026-06-29（T19，含对抗审查后归因修正）。T18 三线各自验证但**未集成**→ 用户选「集成三线+真实组合指标」→ `fuse_eval.py` 串成单一 pipeline 跑出真实组合指标。**对抗审查发现 + 修复了一个 critical bug**：DiCoW `language="zh"` 死代码失效致 90% 出英文（已修，english 90%→72%），但残留瓶颈=Whisper 硬噪声转写质量（需微调/SE-DiCoW）。**先读下方🆕 T19 速览**，完整数字见 RESULTS.md T19。

---

## ⚠️⚠️ 隔离声明（2026-06-30 用户明确指示，后续 agent 必读）

`docs/superpowers/specs/2026-06-29-final-exam-data-analysis-reframe-design.md` 与 `docs/superpowers/plans/2026-06-29-final-exam-deliverables.md` 这两份文档是用户**另一个独立的《Python与数据分析》课程期末作业**，与本项目（美的 XH-202615 参赛工程）**完全无关**。

- ❌ **不要执行**那份 plan——不要创建 `期末作业/` 目录、不写 `analysis.ipynb`、不写课程设计报告 / 答辩 PPT / 分工表
- ❌ **不要把项目方向**往「数据分析重构」带
- ✅ 本项目当前真正任务仍是**参赛工程**的下一步（见下方「T19 速览·待办」与 `PROGRESS.md`）：① SE 条件化 post-fix 重跑 ② 中文家居微调 ③ SE-DiCoW 接入 ④ 真实噪声再校准 noise_classify ⑤ 确认通道数 / 真实数据
- 这两份文件仅因用户在别处可能用到而保留在仓库中，**对本 loop 视作不存在**

---

## 🆕 T19 速览（2026-06-29 端到端集成 + 真实组合指标）

**做了什么**：`code/fuse_eval.py`（核心）把 **SE条件化 → 声纹enrollment锁定target(wespeaker) → DiCoW(Whisper-turbo+FDDT)转写 → LLM拒识(Qwen2.5-3B) → 多策略融合** 串成单一分阶段 pipeline（多 venv 编排），450 集跑出**首个真实组合指标**。配套：`llm_reject.py --infer-json`、`enroll_infer.py` 加 `stno_target_ratio`、`build_reject_set.py`（72条target缺席）、`noise_classify.py`（噪声估计器）、`diag_transcript.py`+`test_zh_force.py`（诊断+实验）。

**决定性发现（瓶颈重定，含对抗审查后的归因修正）**：
- LLM 拒 **449/450（99.8%）**，最优 correct_rate **仅 6-9%** → **融合/阈值旋钮无解**（LLM 不是太严——34条合成测 F1=0.878 健康，是转写垃圾）。
- ⚠️ **归因修正**：初版把英文幻觉判为"Whisper babble 模型漂移"是**错的**。Workflow 对抗审查逐行核 `generation.py` + 全量数据发现真因是 **DiCoW language 强制死代码 bug**——`language="zh"` 被静默忽略→`detect_language` 从退化音频误检英文→**450 集 90%(407) 输出英文**。**已打补丁修复**（`code/apply_dicow_langfix.py`，幂等；⚠️补丁在 HF cache，**cache 清了须重打此脚本**；注：DiCoW-inference/ 是嵌套 git 仓库，其 repro/ 内同名脚本无法被父仓库跟踪，故可跟踪副本放 code/ 根）。
- **修复非银弹（诚实）**：全量 english 90%→72%（chinese 39→125，3 倍），good<0.5 5.8%→7.8%（+9 条），raw CER 3.65→3.54（仅 −0.11）。简单条件（white/pink）从英文→正确中文；**难条件（babble/重叠/低SNR）即使强制中文也是错字垃圾** → 残留瓶颈=Whisper 硬噪声鲁棒性。
- `test_zh_force.py` 排除 initial_prompt（反而更差：前缀污染+重复循环）；三路融合现已在 450 跑通（enroll_regen 带 stno）但仍无济（转写垃圾）。
- **结论（两层瓶颈）**：① language bug（**已修**，必要但不充分）② Whisper 硬噪声转写质量（**残留，主导**，需微调/SE-DiCoW）。真实提升杠杆=①中文家居微调 ②SE-DiCoW（enrollment条件化攻重叠+babble死区）③更强babble SE——都需重投入，**待真实数据/通道数确认**。

**两个强阳交付**：
- ✅ **SE 条件化可部署**：`noise_classify.py` 谱平坦度估计器 **99.78% 准确**（white/pink/babble 三类谱平坦度无重叠），**可部署 CER 2.82≈oracle 2.82**（不再依赖 manifest noise_type，测试时可估）。
- ✅ **拒识侧 100% 真实拒识率**（72条target缺席集，sim/LLM/三路融合全对，0误放行）；发现 **stno 单独是坏拒识信号**（误放行非目标主导语音）、**sim 才是锚信号**。

**新 P0（集成后重定，T20 修正 babble 归因）**：① **babble diar 误检 + STNO 崩攻坚**（T20 证实 babble 英文幻觉真因更上游=DiariZen diar 把 babble 人声噪声误检为第2 speaker→`stno_target_ratio=0`→STNO target 帧清零→DiCoW 转写崩；杠杆：babble 强降噪前置 diar / 声纹 babble 鲁棒 / STNO 容错）② SE-DiCoW 接入（攻重叠+babble死区）③ 中文家居微调（攻 Whisper 转写层，**但 STNO 崩时无效，须在 ① 之后**）④ **确认通道数/真实数据**（决定空间路线+微调数据）。**融合框架已就绪，转写质量上去即生效**——组合主线工程闭环成立，下限取决于 babble diar + Whisper 带噪中文。

**待办（下一棒）**：① ✅ **SE条件化 post-fix 重跑已完成(2026-06-30 T20)**——旧 conditional post-fix overall **3.236**；post-fix 后 =6 全面优于 =0（最优精细二维 **2.022** / 新 conditional **2.609** 稳健推荐）；**归因深化修正 T19**：babble 英文幻觉真因更上游=**DiariZen diar 误检 babble 人声噪声为第2 speaker → stno_target_ratio=0 → STNO target 帧清零 → DiCoW 转写崩溃**（714字英文循环，非 langfix/Whisper 本身）；改进杠杆多元（babble 强降噪前置 diar / 声纹 babble 鲁棒 / STNO 容错 / SE-DiCoW），详见 RESULTS T20；② babble diar+STNO 崩攻坚（新 P0①，治本）；③ SE-DiCoW 接入（攻重叠+babble死区）；④ 真实噪声再校准 noise_classify（合成噪声谱干净，真实会差）；⑤ **确认通道数/真实数据**（决定空间路线+微调数据，当务之急）。融合框架已就绪，转写质量上去即生效。

---

## T18 速览（历史，2026-06-29 接手后多线并行铺开）

**三线 de-risk 全 READY**（Workflow 3 agent，各建独立 venv：`code/.venv_se` / `code/.venv_campp` / `.venv_llm`[项目根]）：

- **线C W5-LLM 拒识** ✅强阳：Qwen2.5-3B 零样本 34 条 **F1=0.878 / Recall=1.0**（最难 case 全对；5 误拒是合法复杂指令→prompt 调优）。拒识 40% 核心层已验证。脚本 `code/llm_reject.py`。
- **线A SE增强 一锤定音** ✅部分阳：DeepFilterNet3 降噪后 overall CER **4.27→3.65(Δ−0.62)**；**babble(人声,贴真实)Δ−4.20 巨大** / pink(稳态)+2.20 反伤；**diar-fail 33→0**（diarization 完全稳定）；CER 绝对值仍高(3.65)→瓶颈多元。**验证"瓶颈部分在音频质量"诊断**。脚本 `code/se_denoise.py`+`eval_se_cer.py`。
- **线B CAM++** ❌证伪(已定论)：per-speaker 公平对照(CAM++ 0.191 vs wespeaker 0.218, 正确率 0.00<0.04)→ **不值得替代 wespeaker, 主线维持 wespeaker**; sherpa-onnx 留边缘部署备用。脚本 `code/enroll_infer_campp.py`。
- **线A SE 条件化(新, 最优)** ✅：按 noise_type 分流(babble/white=0 全力, pink=6 温和) overall CER **2.82(−34%)**, 优于单一 =0(3.65)/=6(3.95); diar-fail 33→0。脚本 `code/merge_se_conditional.py`。

**新 P0 优先级**：① SE 前置条件化落地(快赢：babble/低SNR 启用，pink/white 用 `--atten-lim-db=6`) ② CAM++ per-speaker 公平对照(定论去留) ③ 中文家居微调(重，攻 Whisper 中文) ④ SE-DiCoW 接入(攻 100% 重叠死区)。

**数据增强暂缓**(用户定)：MUSAN/DEMAND+RIR+AISHELL 方案已评估存档(RESULTS T18)，等真实数据/通道数确认再定。

**坑**：HF 下载用 `curl -sSL 经代理+hf-mirror 直链`(snapshot_download 失败)；csukuangfj 仓 401 改 hf-mirror 无代理。

---

## 0. 项目一句话状态（T17 历史）
组合主线（STNO + DiCoW + 声纹enrollment + LLM拒识）的 **enrollment→target 锁定 + STNO 拒识链路已跑通验证**（干净场景完美），但 **带噪题目分布是当前瓶颈**：wespeaker 声纹在中文+噪声退化严重，导致 450 条全集 **87% 误拒、仅 4% 正确**。下一步核心是**攻带噪鲁棒性**（CAM++/阈值扫/enrollment增强/SE增强）。

---

## 1. 最新进展（T17，2026-06-28~29，已 commit）
| 产出 | 文件 | 状态 |
|---|---|---|
| **Part1** enrollment→wespeaker 锁定唯一 target | `code/enroll_infer.py` | ✅ 干净场景验证成功；带噪诊断完成 |
| **TTS 数据集**（趁 mimo-tts 限时免费）| `code/tts_dataset_gen.py` + `build_dataset.py` | ✅ 21 条 raw + 450 条矩阵 |
| **Part2** fork patch 固化 | `code/DiCoW-inference/repro/` | ✅ apply_patches.sh 幂等 EXIT=0 |
| **Part3** 项目阶段盘点 | `项目阶段盘点.md` | ✅ |
| **边缘部署规划** | `边缘部署规划.md` + memory | ✅ 战略级，待确认硬件 |
| **能力画像** | `code/eval_enrollment.py` + `enroll_wespeaker_full.json` | ✅ 450条：87%拒识/4%正确 |
| 阈值扫 + enrollment 加噪增强 | `code/enroll_infer.py --always-generate --enroll-augment` | 🔄 进行中（`enroll_aug_full.json`）|

---

## 2. 当前能力画像（诚实，答辩核心）
- **干净场景**：enrollment→锁定 target→转写「请把客厅的空调温度调到二十六度」(CER=0, sim=0.816)；不同人→兜底拒识空输出(sim=0.035)。**链路成立**。
- **题目分布 450 条**（带噪 −5~5dB + 重叠 0-100%）：**87% 拒识 / 4% 正确**。sim 随 overlap 单调降(0.32→0.12)，SNR 越低越差。
- **根因**：① wespeaker(VoxCeleb英文)声纹中文+噪声退化（噪声是主因，重叠次要）；② 阈值 0.5 在题目分布太严；③ 少数不拒识条出 Whisper 英文幻觉。
- **CAM++ 对比受阻**：modelscope 1.37.1 装上但 `from modelscope.pipelines import pipeline` import 挂起（无 proxy 也卡）。

---

## 3. 后续任务清单（优先级排序，直接可做）

### 🔴 P0 — 攻带噪鲁棒性（瓶颈已诊断：转移至 Whisper 转写质量）
1. ✅ **阈值扫 + enrollment 加噪增强（本 agent 已完成）**
   - **增强有效**：enrollment 加噪 emb → 均 sim 0.218→0.348(+59%)；同阈值0.5 拒识率 0.87→0.72，正确率 0.04→0.11
   - 阈值扫：拒识率 0.12(0.1)–0.72(0.5) 可调，但**正确率天花板~14%**
   - **关键结论——瓶颈转移**：阈值旋钮解不了根本，**真正瓶颈从"声纹拒识"转到"Whisper 带噪+重叠转写质量"** → 下一步必须 frontend SE 增强（下条升最高优先）
   - 数据：`code/enroll_aug_full.json`；最佳工作点 threshold~0.2
2. **解 CAM++ 集成**（验证 wespeaker→CAM++ 改进，原生中文更鲁棒假设）
   - modelscope 本环境 import 卡 → **方案A**：独立干净 venv（`uv venv`）装 modelscope 跑 CAM++ 抽 emb 存文件，主 venv 读；**方案B**：sherpa-onnx 的 campplus/ERes2Net ONNX（隔离，不碰 transformers，推荐先试）
   - 脚本 `code/campp_vs_wespeaker.py` 已就绪，只待加载方式解
3. **frontend SE 增强**（RASTAR/VSAEC，论文 #6/#7）再抽声纹/diarization — 攻噪声根源

### 🟡 P1 — 拒识/效率/微调
4. **W5 LLM 拒识**（Qwen-2.5-3B 语义拒识）— 拒识 40% 当前只有 STNO 这一层机制，缺语义层
5. **D2 中文家居微调**（DiCoW 在家居指令微调）— CER 40% 提升
6. **D5 效率/W8 边缘部署**（待确认目标硬件 MCU/网关/服务器 → 向主办方确认）

### 🟢 P2 — 待外部输入
7. **确认测试集通道数**（决定 DSENet/KWS 多通道能否用，待主办方或看 A 集数据）
8. **真实比赛数据**来时：用 `eval_metrics.py` + `final/final_manifest.json` ground truth 跑全面 CER/拒识率

---

## 4. 关键命令速查
```bash
# 环境（每次新终端先 source）
source code/setenv.sh && export HF_HUB_OFFLINE=1

# enrollment→target 推理（方案B，独立脚本）
code/.venv/Scripts/python.exe code/enroll_infer.py \
  --enrollment E:/midea_target_asr/test_wav/dataset/raw/enrollment/target_long_01.wav \
  --recognition <wav或folder> --always-generate --enroll-augment --out-json <out.json>

# 能力评估 + 扫阈值
code/.venv/Scripts/python.exe code/eval_enrollment.py --enroll-json <out.json> --threshold 0.2 --label X

# 完整 DiCoW pipeline（T14 验证过，PIPELINE_EXIT=0）
code/.venv/Scripts/python.exe code/DiCoW-inference/inference.py \
  --dicow-model E:/hf_cache/DiCoW_v3_2 --diarization-model E:/hf_cache/diarizen-wavlm-large-s80-md \
  --input-folder <wav> --output-folder <out> --device cuda --verbose

# TTS 合成数据（WSL 跑，mimo-tts 限时免费窗口注意时效）
wsl.exe -d Ubuntu-22.04 bash -lc 'python3 /mnt/e/midea_target_asr/code/tts_dataset_gen.py'
# 组装矩阵（Windows）
code/.venv/Scripts/python.exe code/build_dataset.py

# fork patch 固化校验
bash code/DiCoW-inference/repro/apply_patches.sh
```

---

## 5. 已知坑（务必避开）
- **modelscope 1.37 在本环境 import 挂起**（`pipelines` 卡，datasets 正常）→ CAM++ 待独立 venv/sherpa-onnx
- **ESC-50 真实环境音下载失败**（GitHub HTTP/2 中断）→ 数据集用程序噪声(white/pink/babble)；真实环境音放 `test_wav/dataset/env_noise/` 重跑 build_dataset 自动加入
- **git user = midea-overnight-loop**（非用户本人 Panda_Lorrain；本 overnight loop 项目用此身份是设计，手动 commit 注意）
- **DiCoW-inference/ 整棵 gitignore**，但 `repro/` 已 force add 入 git；改动 fork 代码后 re-run apply_patches.sh 重新导出
- **wespeaker 拼写 VoxCeleb 带 x**（曾拼错 voceleb 404）
- **setenv.sh 必须 source**（设 HF_HOME=E:/hf_cache 等，否则权重找不到）
- **uv 禁裸 pip**；PyPI SSL 失败用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **某些恶劣音频（ov100/snr-5）触发 DiariZen reconstruct `negative dimensions` crash** — enroll_infer 已加 try/except 容错跳过

---

## 6. 环境 & 资产位置
- **GPU**：RTX 4060 Laptop 8GB / torch 2.5.1+cu124 / Python 3.12 / `code/.venv`
- **权重**（全 E 盘）：`E:/hf_cache/DiCoW_v3_2`（0.89G）、`diarizen-wavlm-large-s80-md`、`hub/models--pyannote--wespeaker-voxceleb-resnet34-LM`（6.6M）
- **数据集**：`test_wav/dataset/raw/`（21条TTS种子，入git）/`final/`（450条矩阵，ignore，可 build_dataset 重建）/`final/final_manifest.json`（ground truth，入git）
- **文档**：`00_技术路线总纲与行动地图.md`（W1-W7 主路线）、`01-03` 模块/答辩、`papers/`（19 PDF）、`边缘部署规划.md`、`项目阶段盘点.md`
- **WSL**（Ubuntu-22.04）：mimo-tts 在此跑，key 在 `~/.hermes/.env`

---

## 7. 设计/决策记录（避免重复踩坑）
- **W3/W4 重新定义**：原"Personal VAD + CAM++"证伪（Personal VAD 无开源实现），改为 enrollment→wespeaker 锁定 target（复用已跑通组件）。设计文档 `docs/superpowers/specs/2026-06-28-enrollment-target-and-patch-fixation-design.md`
- **Part1 用方案B（独立脚本）非方案A（改pipeline.py）**：不碰跑通的 pipeline.py/inference.py，向后兼容天然
- **CAM++ 从"沉没成本"修正为"带噪鲁棒性数据驱动备选"**：wespeaker 中文+噪声退化实测后，CAM++ 原生中文有了引入理由
- **wespeaker 复用 diar._embedding**（不独立加载）：`diar._embedding(waveform[None,None])` → (batch,256)，零额外加载
- **STNO 数据结构**：`[N_speakers, 4, T@50Hz]` float32，4 行=(sil/target/nontarget/overlap) 每帧 one-hot
