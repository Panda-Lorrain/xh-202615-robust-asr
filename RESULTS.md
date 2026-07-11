# W1 实测结果（XH-202615 overnight loop）

> 首个真实推理数据，**解除"零实测"答辩红线**。明早看此文件 + PROGRESS.md。
> 生成：2026-06-27。

## 🎉 里程碑：DiCoW baseline 最小推理跑通

| 指标 | 值 | 意义 |
|---|---|---|
| 模型 | BUT-FIT/DiCoW_v3_2（Whisper-large-v3-turbo + FDDT） | TS-ASR 开源 baseline |
| **参数量** | **0.89G** | ✅ 印证 turbo ≈809M（**非 1.5B**，03 答辩红线 7 正确） |
| 加载时间 | 2.6s | |
| 推理（30s 音频） | 1.73s | |
| **RTF** | **0.058** | 远快于实时；本机 4060 已如此，L20（48GB）会更优。**20% 效率分前景极佳** |
| **峰值显存** | **2.13GB** | 8GB 4060 绰绰有余，L20 无压力 |
| 测试音频 | EN2002a_30s.wav（英文会议 30s） | DiariZen 自带样例 |

### 转写样例（全-target STNO，即转写整段所有人）
> "yeah yeah but i do not know about you but usually in windows right click does not do anything there is a it opens a menu..."（通顺，质量好）

### FDDT 配置印证（来自 config.json，答辩素材）
- `use_fddt: true` / `use_initial_fddt: true`
- `fddt_init: "disparagement"`（= 论文的**抑制式初始化**）
- `fddt_is_diagonal: true`（对角约束）
- `fddt_use_target/non_target/overlap/silence: true`（全四类 STNO）
- `encoder_layers: 32` + `decoder_layers: 4`（**turbo 配置**，印证 4 层 decoder）

### 说明（诚实）
- 这是**最小推理**：构造"全 target"STNO（整段当目标说话人），验证模型加载 + forward + generate + RTF，**非真 target-speaker 转写**。
- 真 target-speaker 需 DiariZen diarization 生成真实 STNO（完整 pipeline，下一步补）。

### 环境信息
- torch 2.5.1+cu124 / transformers 4.42.4 / python 3.12 / RTX 4060 Laptop 8GB
- 权重 `E:/hf_cache/DiCoW_v3_2`（model.safetensors 3.6G）
- 缓存全落 E 盘（HF_HOME/UV_CACHE_DIR，禁 C 盘）✅
- 脚本 `code/minimal_infer.py`

---

## 🎯 STNO 控制实验（验证 FDDT 机制，答辩黄金素材）

对 EN2002a（30s 多人会议）构造 4 种 STNO mask，验证 FDDT/STNO 如何控制转写（脚本 `code/stno_experiment.py`）：

| STNO 构造 | 输出 | 验证结论 |
|---|---|---|
| A 全-target（基线） | 398 字，转所有人 | 完整转写 |
| B 前半 target + 后半 silence | 162 字，**只转前半** | target 类→转，silence 类→跳过 ✅ |
| C 前半 silence + 后半 target | 165 字，**只转后半** | STNO 精确控制段 ✅ |
| D **全 non-target** | **0 字，完全空** | **non-target 类→直接拒识！** ✅ |

**核心结论（答辩可直接讲）**：
- FDDT 的 STNO 条件化**可验证、可控**：target 转写、silence 跳过、**non-target 直接产出空（拒识）**。
- 印证 config 的 `fddt_init: "disparagement"`（抑制式初始化）——非目标帧被压到零输出。**"拒识非目标"不是后处理，是 FDDT 内建机制**。
- 组合主线核心验证：STNO mask（来自 PVAD/diarization）直接控制 target-speaker 转写与拒识，机制成立。
- 这是完整 DiariZen pipeline 的**机制替代验证**（虽未跑真 diarization，但 STNO→转写/拒识 链路已证实，回应 03 答辩红线 4）。

## 下一步（overnight 续做）
1. **完整 pipeline**（DiariZen diarization → 真 STNO → 真 target-speaker）：装 pyannote/DiariZen，跑 inference.py
2. **中文音频测试**：验证 DiCoW 中文能力（题目是中文）
3. **W6 评测脚本**：CER 计算 + RTF 批量
4. **W2 数据仿真 pipeline**：中文多人 + SNR −5~5dB + 重叠 0-100%

---

## 🎯 中文重叠场景完整 pipeline 诊断（2026-06-28 T15，答辩核心素材）

> **题目核心场景验证**：带噪 + ≤2人重叠(0–100%)，只转 target 指令、拒识 non-target。
> 首次在**完整端到端 pipeline**（DiariZen diarization → STNO → DiCoW）上系统验证重叠率对 target-speaker 转写的影响。
> 脚本 `code/overlap_experiment.py`；结果 JSON `code/overlap_eval_result.json`；原始转写 `code/pipeline_overlap_out/`。

### 实验设置
- **音频源**：mimo-tts 合成（替代 SSL 受阻的 edge-tts）
  - target = 冰糖女声，家居指令「请把客厅的空调温度调到二十六度」
  - non-target = 苏打男声，干扰闲话「今天天气真不错，我们出去走走吧」
- **Pipeline**：DiariZen(wavlm-large diarization) → STNO mask → DiCoW_v3_2(Whisper-large-v3-turbo + FDDT) target-speaker 转写
- **诊断变量**：重叠率 3 档（0% / 50% / 100%），无加噪聚焦 diarization 分离能力
- **评测**：CER 最低的说话人识别为 target，算字符级 target CER（`eval_metrics.cer`）

### 结果：target CER 随重叠率单调退化
| 重叠率 | 构造 | target CER | target 转写 | 结论 |
|---|---|---|---|---|
| **0%（顺序）** | target → 0.3s 静音 → non-target | **0.000** | 请把客厅的空调温度调到二十六度（完全正确） | ✅ 完美分离+转写 |
| **50%（前段重叠）** | mix_overlap(0.5) | **0.133** | 清白客厅的空调温度调到二十六度（"请把"→"清白"） | ✅ 基本分离，轻微退化 |
| **100%（完全重叠）** | mix_overlap(1.0) | **1.000** | target 丢失，两 speaker 都转 non-target 闲话 | ❌ 单通道 diarization 死区 |

### 转写全貌（分说话人 + 时间戳）
**0% 顺序（seq_clean）** — 时间戳正确分开：
- Speaker 0 (0.00–2.84s)：「请把客厅的空调温度调到二十六度」← target ✓
- Speaker 1 (3.06–5.86s)：「今天天气真不错我们出去走走吧」← non-target ✓

**50% 部分重叠（partial50_clean）**：
- Speaker 0：「今天天气真不错」← non-target 前段
- Speaker 1：「清白客厅的空调温度调到二十六度」← target（"请把"误为"清白"）

**100% 完全重叠（full_clean）** — 分离失败：
- Speaker 0 / Speaker 1：均为「今天天气真不错我们出去走走吧」← target 指令完全丢失

### 结论与答辩意义
1. **完整端到端 pipeline 在可分离场景工作良好**：0% 重叠 target CER=0（完美），50% 重叠 CER=0.133（可用），证明 DiariZen + DiCoW 组合主线有效。
2. **target CER 随重叠率单调退化**（0 → 0.133 → 1.0），**100% 完全重叠是单通道 diarization 死区**——两人同帧说话、单麦克风无方向信息，diarization 退化为把主导声归到多个 speaker。
3. **这正是题目"重叠率 0–100%"的难点**，论证三条改进路线的必要性：
   - **多通道空间分离**（DSENet/VSAEC，DOA 引导）：完全重叠时靠方向信息分离（需多通道，待通道数确认）
   - **enrollment 条件化**（SE-DiCoW）：用目标声纹直接条件化，跳过"先分离再选 target"
   - **STNO non-target mask 拒识**：已验证 non-target→0字拒识（见上方 STNO 章节），是 diarization 路线的拒识补充
4. **诚实展示局限**：不回避 100% 失败，作为"问题诊断 → 改进方向"的答辩逻辑（评委更看重对难点的理解，而非完美数字）。

### 工程备注（极端场景）
- 完全重叠 + 短音频(2.7s) 还触发 Whisper 幻觉（早前 snr0 测试转出无关英文 "you can get some of the people..."）；加噪 −5dB 时 diarization 维度 crash（`negative dimensions are not allowed`）。极端场景需 robustness 改进（分块/更长上下文/异常兜底）。
- 当前 diarization 路线会转写**所有** speaker（含 non-target）；题目"拒识 non-target"需叠加 enrollment 声纹匹配或 STNO 拒识路线。

---

## 🛡️ 中文 STNO 拒识验证（2026-06-28 T16，拒识 40% 路线补全）

> **拒识占评分 40%**。STNO non-target mask 让 DiCoW 直接输出空 = 拒识非目标（FDDT 内建机制）。
> 此前证据为英文（stno_experiment.py D 组 EN2002a→0字），本节补**中文**证据。脚本 `code/zh_stno_reject.py`，结果 `code/zh_stno_reject_result.json`。

### 结果（zh_target_01「请把客厅的空调温度调到二十六度」）
| STNO 构造 | 输出 | 字数 | 结论 |
|---|---|---|---|
| target `([0,1]=1)` | 请把客厅的空调温度调到二十六度 | 15 | ✅ 正确转写 target（CER=0） |
| non-target `([0,2]=1)` | （空） | **0** | ✅ **拒识成功**（0字） |

### 答辩意义
1. **FDDT 的 STNO 条件化在中文上同样成立**：target 类→转写、non-target 类→直接产出空（拒识），与英文 EN2002a 结论一致。
2. **拒识不是后处理，是 FDDT 内建机制**（`fddt_init=disparagement` 抑制式初始化把非目标帧压到零输出）——组合主线拒识侧的核心论证。
3. **与重叠诊断（上方）互补**：diarization 路线转所有 speaker（需后续选 target），STNO 拒识路线直接 mask non-target→0字。两条路线共同支撑"只转 target、拒识 non-target"。

---

## 🎯 Part1: enrollment→wespeaker 锁定唯一 target（2026-06-28 T17，组合主线核心缺口补全）

**缺口**：此前 pipeline（T14）把 diarization 找出的所有 speaker 各转一遍，**无"enrollment→锁定唯一 target"**。本节实现并验证。脚本 `code/enroll_infer.py`（方案B独立脚本：复用 `diar._embedding`(wespeaker 256d) 抽声纹 + 余弦匹配选 target + 构造 target STNO + DiCoW 只转 target + 兜底拒识；不动 pipeline.py/inference.py，向后兼容）。

### 概念验证（干净场景）✅
| 场景 | enrollment | recognition | sim | 判定 | 结果 |
|---|---|---|---|---|---|
| 同人锁定 | 冰糖长enroll 10.2s | 冰糖 t_01 | **0.816** | TRANSCRIBE | 「请把客厅的空调温度调到二十六度」15字 CER=0 ✅ |
| 不同人拒识 | 苏打 n_01 | 冰糖 t_01 | **0.035** | REJECT | 0字空输出 ✅ |

判别力鲜明：同人 0.816 vs 不同人 0.035，阈值 0.5 完美居中。**干净场景 enrollment 锁定 + 兜底拒识完全成立**。

### 题目分布批量（t_01 × 5重叠 × 3SNR × white噪声，15条；3条 snr-5 触发 DiariZen reconstruct 边界 crash，已加容错跳过）
| 重叠\SNR | +5 | 0 | -5 |
|---|---|---|---|
| 0% | 0.475 ✓ | 0.431 ✓ | 0.205 ✓ |
| 25% | 0.345 ✓ | 0.349 ✓ | 0.261 ✗(选错) |
| 50% | 0.263 ✓ | 0.235 ✓ | crash |
| 75% | 0.054 ✓ | 0.327 ✗(选错) | crash |
| 100% | 0.263 =(死区) | 0.209 ✗ | crash |

（✓=选对 target，==两人 sim 相同无法区分，✗=选错 target；**全部 max_sim<0.5 → REJECT**）

### 全集 450 条能力画像（wespeaker，`eval_enrollment.py`）
| 维度 | n | 拒识率 | 均sim | 正确率(CER<0.5) |
|---|---|---|---|---|
| **总体** | 450 | **0.87** | 0.218 | **0.04** |
| overlap 0% | 90 | 0.82 | 0.322 | 0.12 |
| overlap 25% | 90 | 0.91 | 0.262 | 0.06 |
| overlap 50% | 90 | 0.89 | 0.203 | 0.00 |
| overlap 75% | 90 | 0.89 | 0.160 | 0.00 |
| overlap 100% | 90 | 0.86 | 0.122 | 0.00 |
| SNR −5 | 150 | 0.78 | 0.118 | 0.00 |
| SNR 0 | 150 | 0.98 | 0.209 | 0.01 |
| SNR +5 | 150 | 0.86 | 0.304 | 0.09 |

**当前能力基线（诚实）**：wespeaker enrollment→target 在题目分布（带噪 −5~5dB + 重叠）**基本失效——87% 拒识、仅 4% 正确**。sim 随 overlap 单调降，SNR 越低越差。少数不拒识条在恶劣音频上出 Whisper 英文幻觉（ov0/ov100/snr+5 的 CER>5）。**根因：wespeaker（VoxCeleb 英文）声纹在中文+噪声退化严重 + 阈值 0.5 在题目分布太严**。→ CAM++ 对比的强烈动机（原生中文可能更鲁棒）。

### 阈值扫 + enrollment 加噪增强（T17续，攻 87% 误拒）
**enrollment 加噪增强有效**：enrollment emb = 干净 + 3档加噪(10/5/0dB)均值 → 对带噪识别音频更匹配，**均 sim 0.218→0.348（+59%）**。同阈值 0.5：增强把**拒识率 0.87→0.72，正确率 0.04→0.11（近3倍）**。

| threshold | 拒识率 | 不拒CER | 正确率(CER<0.5) |
|---|---|---|---|
| 0.1 | 0.12 | 5.27 | 0.14 |
| 0.2 | 0.24 | 4.35 | 0.14 |
| 0.3 | 0.37 | 4.08 | 0.13 |
| 0.5 | 0.72 | 3.06 | 0.11 |

**关键洞察——瓶颈转移**：阈值旋钮可调拒识率（0.12–0.72），但**正确率天花板 ~14%**。降阈值→拒识少，但放进来的条 Whisper 转写仍错（CER爆/幻觉），**sim 高 ≠ 转写对**。**真正瓶颈已从"声纹拒识"转移到"Whisper 在带噪+重叠的转写质量"** → 下一步必须 **frontend SE 增强**（RASTAR/VSAEC 提升音频质量→改善 diarization + Whisper 转写），而非继续调阈值。最佳工作点 threshold~0.2。数据 `code/enroll_aug_full.json`。

### 关键发现（诚实）⚠️
1. **带噪 sim 普遍退化**：所有重叠/加噪条件 0.05–0.48，**全 <0.5 误拒**。阈值 0.5 在题目分布（−5~5dB）太严。
2. **噪声是退化主因**（非重叠）：ov000 无重叠但 snr-5 降到 0.205（vs 干净 0.816）；重叠是次要因素。
3. **多 speaker 选择偶反转**：高重叠 + 低 SNR 时两人 sim 接近/反转（ov025_snr-5、ov075_snr+0 选错）。
4. **ov100 死区**：两人 sim 相同（0.263），无法区分（与 T15 重叠诊断一致）。

### 改进方向（数据驱动，答辩素材）
- **CAM++ 引入有了数据理由**：wespeaker（VoxCeleb 英文训练）在中文+噪声退化严重 → CAM++（原生中英双语）可能更鲁棒。**修正此前"CAM++ 是沉没成本"判断为"带噪鲁棒性数据驱动备选"**。
- enrollment 加噪增强（同分布噪声） / 声纹多段质心融合 / 阈值分场景自适应 / frontend SE 增强（RASTAR/VSAEC）再抽声纹。

### CAM++ 对比实验（T17，本环境受阻）
- 装了 modelscope 1.37.1（清华源，未坏 DiCoW 依赖 transformers 4.42.4），但 `from modelscope.pipelines import pipeline` **import 挂起**（无 proxy 也卡，datasets import 正常）——疑似 modelscope 1.37 + Windows + torch2.5 的 import 兼容问题。
- 脚本 `code/campp_vs_wespeaker.py`（快速声纹 sim 对比）+ `code/eval_enrollment.py`（分档指标）就绪，待 modelscope import 解。
- **后续**：独立干净 venv 装 modelscope，或用 sherpa-onnx 的 campplus/ERes2Net ONNX（隔离不碰 transformers）。

### 答辩意义
概念验证成功（enrollment→锁定 target 完整链路打通，填补组合主线真正缺口），同时诚实展示带噪难点 + 明确改进路线（CAM++/增强/融合），体现工程深度。

---

## 📦 数据集生成（T17，W2 实质化）
- **mimo-tts 限时免费窗口**：合成 21 条 raw（3 enrollment 长/短 + 10 target 指令 + 8 nontarget 干扰），脚本 `code/tts_dataset_gen.py`（WSL 跑 MiMo-V2.5-TTS，冰糖女声/苏打男声）。
- `code/build_dataset.py` 组装 **450 条矩阵**（10 target × 5 重叠 × 3 SNR × 3 程序噪声）+ ground truth manifest（target_ref 供 eval_metrics 算 CER）。
- ESC-50 真实环境音下载失败（网络 HTTP/2 中断）→ fallback 程序噪声（white/pink/babble）；真实环境音待补（放 `test_wav/dataset/env_noise/` 重跑 build_dataset 自动加入）。

## 🔧 fork patch 固化（T17，Part2）
- Workflow 产出 `code/DiCoW-inference/repro/`：3 canonical patch + requirements-fork.txt + apply_patches.sh（幂等）+ REPRODUCE.md。
- 校验：跑 apply_patches.sh → 3 patch 全 SKIP（marker 已存在）+ wespeaker 权重校验 OK，**EXIT=0** ✅。可复现。

## 🏠 边缘部署规划（T17，战略级）
- 用户提出**终态目标是边缘部署**。产出 `边缘部署规划.md`：当前路线有轻量化种子（turbo 0.89G/RTF0.058 + wespeaker 6.6M 天生轻量）但缺系统性规划（目标硬件未定义、量化/蒸馏/ONNX/流式未排进行动地图）；建议新增 **W8 部署轻量化模块**（量化/蒸馏/ONNX导出/流式改造/硬件基准）；**待确认目标硬件**（家电 MCU / 边缘网关 / 本地服务器，算力差几个数量级）。记忆已存。

---

## 🎯 T18 三线 de-risk + 线A 一锤定音（2026-06-29，接手后多线铺开）

> 用户选「多线并行铺开」：Workflow 三线 de-risk（SE增强/CAM++/W5-LLM），各建独立 venv 调研+脚本+CPU 验证，GPU 实验主线串行（8GB 单卡约束）。三线全部 feasibility=READY。

### 线C W5-LLM 语义拒识（✅ 强阳性，拒识 40% 核心层）
- 部署 Qwen2.5-3B-Instruct（6.17GB，独立 .venv_llm），Prompt 按出题方 #1 论文（arXiv:2512.10257 Midea AI Research）13 类拒识 schema + 自适应 CoT 四步。
- **零样本 34 条测试集（16+/18-）**：**F1=0.878 / Precision=0.783 / Recall=1.0 / Accuracy=0.853**（GPU 74s）。vs 全 reject 基线 F1=0.69（+0.19）。
- **Recall=1.0**：该拒全拒（fn=0）——最难 case 全对（空调40度/热水器100度参数荒谬、「这这这」「汪汪汪」乱码非人声、各类闲聊）。
- 5 个误拒（fp）都是合法复杂指令（窗帘拉上一半/净化器睡眠模式/热水器45度/灯暖色最低亮度/十分钟后提醒关火）→ prompt 调优可解决；三路融合后声纹路纠正。
- 三路融合接口已设计：`fuse_three_ways(llm_verdict, max_sim, stno_target_ratio, w=0.4/0.4/0.2)`。
- 脚本 `code/llm_reject.py` + `build_llm_testset.py` + `llm_reject_testset.json` + `llm_reject_result.json`。

### 线A SE 增强 一锤定音（✅ 部分阳性，验证瓶颈诊断）
- 选型 DeepFilterNet3（8.7MB 权重，纯 CPU，450 条 50s）。独立 .venv_se。脚本 `code/se_denoise.py`（批量降噪 16k↔48k）+ `eval_se_cer.py`（CER 对比）。
- **全集 450 条 baseline vs 降噪后 CER 对比**（`code/se_baseline.json`/`se_denoised.json`）：

| 指标 | baseline | denoised | Δ | 结论 |
|---|---|---|---|---|
| **overall CER** | 4.2738 | 3.6545 | **−0.6194** | ✅ 降噪改善（瓶颈部分在音频质量） |
| SNR −5 | 5.976 | 3.969 | **−2.006** | ✅ 低 SNR 大幅受益 |
| SNR 0 | 4.092 | 3.277 | −0.815 | ✅ |
| SNR +5 | 2.754 | 3.717 | **+0.963** | ❌ 高 SNR 过消除伤语音 |
| overlap 0% | 6.198 | 3.828 | **−2.370** | ✅ 无重叠最大受益 |
| overlap 25% | 3.933 | 4.537 | +0.604 | ❌ |
| overlap 50% | 3.766 | 3.167 | −0.599 | ✅ |
| overlap 75% | 3.313 | 3.201 | −0.111 | ✅ |
| overlap 100% | 4.159 | 3.539 | −0.621 | ✅ |
| **noise babble**（人声） | 8.596 | 4.398 | **−4.197** | ✅✅ 巨大改善（最贴真实场景） |
| noise pink | 1.874 | 4.070 | **+2.196** | ❌❌ 稳态噪声过消除反伤 |
| noise white | 2.352 | 2.495 | +0.143 | ❌ 微伤 |
| **diar-fail 数** | 33 | **0** | — | ✅✅ 降噪后 diarization 完全稳定 |

- **诊断结论**：① 瓶颈**部分**在音频质量（Δ−0.62），SE 前置有效，**验证上个 agent "瓶颈转移到 Whisper 带噪转写" 诊断**；② 效果**高度依赖噪声类型**——babble（人声干扰，最贴题目真实场景）Δ−4.20 巨大改善（降噪大减 Whisper 幻觉），pink/white（稳态）过消除反伤；③ **diar-fail 33→0** 是强稳定性信号（即使 CER 改善有限，diar 稳定本身值得）；④ **CER 绝对值仍极高（3.65）**——瓶颈多元：重叠分离死区（单通道）+ Whisper 中文带噪能力 + 声纹误拒，SE 不是银弹。
- CER>1 说明 hyp 超长（Whisper 幻觉/重复），babble 噪声最易触发（把人声噪声听成语音），降噪后幻觉大减。

### 线B CAM++（⚠️ READY 但有证伪信号，决定性实验待做）
- sherpa-onnx（campplus.onnx 512d）**彻底绕过 modelscope import 挂起**（import 即时返回，ONNX runtime 自带不碰 transformers）。独立 .venv_campp。
- **但「原生中文带噪鲁棒性」假设——证据不支持**：① 8 对样本 margin 仅 +0.128（区分度中等）；② 450 矩阵整段 sim=0.121 **低于** wespeaker 0.218（**但不公平**：CAM++ 无 diar 整段 vs wespeaker per-speaker 分离后抽 emb）。
- 当前用的是 **VoxCeleb 版 CAM++**，非原生中文（要原生中文应换 CN-Celeb 训练的 wespeaker-cnceleb）。
- **决定性实验未做**：CAM++ per-speaker 公平对照（CAM++ 也走 diar 分离）——判定 CAM++ 是否值得换主线的唯一实验。
- 替代价值：sherpa-onnx OfflineSpeakerDiarization 一条龙（纯 ONNX，边缘部署友好）可作 DiariZen 轻量替代。
- 脚本 `code/test_campp_load.py`/`campp_vs_wespeaker*.py`/`campp_margin_diag.py`/`campp_enroll_full.py`。

### 聚焦建议（基于三线 + 线A 结果）
1. **SE 前置条件化落地（快赢，已验证）**：babble/低 SNR 档启用 SE，pink/white 用 `--atten-lim-db=6` 限制过消除；diar-fail 33→0 是 pipeline 稳定性强证，值得固化进主线。
2. **CAM++ per-speaker 公平对照（中，定论 CAM++ 去留）**：当前证伪信号基于不公平对比，需公平对照定论；若仍≤wespeaker 则维持 wespeaker，sherpa-onnx 留作边缘部署备用。
3. **中文家居微调（重，攻 Whisper 中文带噪）**：CER 绝对值高的主因之一，长期攻坚。
4. **SE-DiCoW 接入（中，攻 100% 重叠死区）**：enrollment 条件化跳过"先分离再选 target"。
- 3/4 是重投入，取决于真实数据/通道数（待确认）。

### 数据增强暂缓决策（用户定）
- 用户问能否解决「程序噪声非真实环境音 + TTS 分布≠真实远场」两个局限 → 评估：局限1（ESC-50）可换 MUSAN/DEMAND 实质解决；局限2（TTS）可 RIR 远场卷积 + AISHELL 真实语料大幅改善（但 100% 等于比赛数据需真实数据）。
- **用户选「暂缓·专注三线」**：数据增强方案已评估存档（轻量=MUSAN/DEMAND+RIR / 重=+AISHELL），等三线结果出来再定力度。

### 关键技术坑（后续复用）
- **HF 权重下载**：`huggingface_hub` snapshot_download 即使设 HF_ENDPOINT=hf-mirror 仍失败 → 改用 `curl -sSL 经代理(7897)+hf-mirror直链`（线C Qwen 6.17GB 此法下全）。
- **HF csukuangfj 仓 401**：代理鉴权注入，易误判仓不存在 → 改 hf-mirror 直连无代理。
- **新增 venv**：`.venv_se`(code/) / `.venv_campp`(code/) / `.venv_llm`(项目根)，已加 .gitignore。

### T18续：SE条件化落地 + CAM++ per-speaker 定论（2026-06-29）

**SE 条件化(任务1)——最优前置策略, CER −34%**
测 atten-lim=6(温和) + 按 noise_type 分流(450条):

| 策略 | overall CER | Δ vs baseline(4.274) |
|---|---|---|
| 降噪=0(全力) | 3.655 | −0.62 |
| 降噪=6(温和) | 3.949 | −0.32 |
| **条件化(babble/white=0, pink=6)** | **2.825** | **−1.45(改善34%)** |

分流最优: babble Δ−4.20(=0全力保持) / pink Δ−0.29(=6解过消除) / SNR−5 Δ−3.04 / 所有 overlap 档改善。**SE 前置最优落地 = 按噪声类型条件化** + diar-fail 33→0。但 CER 绝对值仍 2.82(>1 幻觉严重), 需叠加中文微调/重叠分离。脚本 `code/merge_se_conditional.py`。

**CAM++ per-speaker 公平对照(任务2)——证伪, 维持 wespeaker**
enroll_infer_campp.py(跨 venv: 主 .venv diarization + .venv_campp CAM++ emb, per-speaker 与 wespeaker 同一份分离音频 = 公平) 跑 450 条:
- CAM++ per-speaker: 均 sim **0.191** / 拒识率 0.92 / 正确率 0.00
- vs wespeaker: 均 sim **0.218** / 拒识率 0.87 / 正确率 0.04
- **CAM++ 0.191 < 0.218, 正确率 0.00 < 0.04 → 不值得替代 wespeaker(证伪)**; 唯一亮点 SNR−5 CAM++ 0.154 > wespeaker 0.118(低 SNR 略鲁棒), 但整体不如。
- 定论: 主线声纹维持 wespeaker; sherpa-onnx CAM++ 留边缘部署备用(纯 ONNX 轻量)。脚本 `code/enroll_infer_campp.py` + `emb_campp.py`。干净负面结果, 排除 CAM++ 替代路线, 聚焦 wespeaker + 中文微调 + 重叠分离。

---

## 🎯 T19 端到端集成 + 真实组合指标 + 瓶颈精准诊断（2026-06-29）

> **三线首次串成单一 pipeline + 跑出真实组合指标**（SE→enroll声纹锁定→DiCoW转写→LLM拒识→多策略融合）。三线此前三次单独验证（T18），但互不相连，450 集上没有单一组合分数。本节把它接通，**结果一锤定音地揭示了真正瓶颈**。

> ### ⚠️ 本节归因已修正（Workflow 对抗审查后发现 + 复测）
> 本节初版把英文幻觉归因为"Whisper 在 babble 上的模型漂移"——**这是错的**。对抗审查逐行核 generation.py + 全量数据交叉表发现真因是 **DiCoW `generation.py` 死代码 bug**：`language="zh"` 被静默忽略 → `detect_language` 从退化音频误检英文 → **450 集 90%(407) 输出英文**（干净音频检对了，故 T17 干净 CER=0 一直没暴露）。**已打补丁修复**（`repro/apply_dicow_langfix.py`）。
> **但修复非银弹**：全量 english 90%→72%（chinese 39→125，3 倍），good<0.5 5.8%→7.8%（+9 条），raw CER 3.65→3.54（仅 −0.11）。简单条件（white/pink）从英文→正确中文；**难条件（babble/重叠/低 SNR）即使强制中文也是错字垃圾**（"你把水深的均等温度好像"）→ 残留瓶颈是 **Whisper 硬噪声鲁棒性**（需微调/SE-DiCoW）。故真实瓶颈**两层**：① language bug（已修，必要但不充分）② Whisper 硬噪声转写质量（残留，主导）。下文"根因"小节按此两层理解；初版"babble 特有漂移/white 良好"的 cherry-pick 结论（基于 n=4）已废弃——全量上 white/pink 亦 ~85%+ 曾出英文（bug 致），babble 更甚（99%）。


### 做了什么
- `code/fuse_eval.py`（核心）：读 enroll_infer JSON（max_sim/transcript）+ LLM verdicts + manifest，对每个融合配置算 **cer_final（拒识计 1.0=漏 target）/ cer_accepted / correct_rate / reject_rate / RTF**，扫 sim_threshold × 策略（sim_only/llm_only/llm_or_sim/llm_and_sim/weighted/stno/three_way）排序找最优。
- `code/llm_reject.py` 加 `--infer-json` 推理模式（无 gold 批量判 verdict，对接 enroll 转写）。
- `code/enroll_infer.py` 加 `stno_target_ratio` 输出（三路第三信号）。
- `code/build_reject_set.py`：造 **72 条 target 缺席集**（苏打+噪声，配冰糖 enrollment→应拒识），补拒识 40% 的"真实拒识率"画面。
- `code/noise_classify.py`：谱平坦度噪声估计器（使 SE 条件化**可部署**）。

### 真实组合指标（450 集，target 恒在场）
**LLM 拒识 449/450（99.8%）**——这是决定性信号：

| 配置 | accept | reject | cer_final | correct(CER<0.5) | 解读 |
|---|---|---|---|---|---|
| sim_only(t=0.2) | 24% | 76% | 1.74 | 5% | 当前主线 |
| sim_only(t=0.15) | 39% | 61% | 2.13 | 6% | 放宽阈值 |
| llm_only | 0.2% | 99.8% | 1.00 | 0% | 全靠语义 |
| llm_or_sim | ≈sim_only | — | — | ≈6% | LLM 救不动 |

**最优 correct_rate 仅 6–9%**（sim_only t=0.15/oracle 条件化）。**融合/阈值旋钮无解**——LLM 不是太严（34 条合成测试 F1=0.878 健康），而是 **DiCoW 转写本身是垃圾**，LLM 正确地把垃圾判为非指令而拒识，但 target 在场 → 这些是"因转写垃圾导致的误拒"，**垃圾文本无可救**。

### 瓶颈精准诊断（根因锁定）
转写质量分布（`diag_transcript.py`）：**good(CER<0.5) 5.8% / garbage(CER≥2) 63.1%**。逐条看失败模式：

| 条件 | 转写 | 现象 |
|---|---|---|
| ov0/snr+5/white | "请把客厅的空调温度调到二十六鼻"(CER 0.07) / "把电视的声音关小一点"(CER 0.00) | ✅ **中文转写良好**（white 噪声 ov0 达 33% 优秀） |
| ov0/snr-5/babble | "more like i think it is a lot of issues" / "take off the water i think..." | ❌ **英文幻觉** |
| ov0/snr+5/babble | "i can you go to the other side" / "stutters are more efficient than I am" | ❌ **英文幻觉**（即使 SNR+5） |

**根因 = Whisper-large-v3-turbo（DiCoW 基座）在退化中文音频上语言漂移→英文幻觉**，尤其 **babble（类人声噪声）**触发最烈（与 SE 数据吻合：babble CER 8.6 最差）。enroll_infer 已传 `language="zh"`，但 Whisper 只强制首位 token，序列中段在 babble 下漂英文。**这不是融合/拒识能修的——是转写模型本身的鲁棒性问题。**

### 中文强制实验（`test_zh_force.py`，排除 prompt 方案）
试 `initial_prompt="以下是普通话的句子。"`（Whisper 标准中文强制手段）：**反而更差**（ov0 均CER 1.62→5.58，CER<0.5 占比 33%→0%）。原因：①prompt 文本被前缀进输出（"以下是普通话的句子。把电视..."）抬高 CER；②难音频上触发重复循环（"小小小小..."）。**排除 initial_prompt 方案**。

### SE 条件化可部署化（`noise_classify.py`）✅
谱平坦度把 white(0.39)/pink(0.11)/babble(0.03) 三类噪声**完全分离**（无重叠），babble vs stationary 阈值 0.053 达 **100% 准确**，三分类整体 **99.78%（449/450）**。用**估计的**噪声类型从已有 =0/=6 两版挑条合并：**可部署条件化 CER = 2.823 ≈ oracle 2.825**。→ **SE 条件化从"需 manifest（测试时不可知）"变成纯运行时可部署**。⚠️⚠️ **诚实 caveat（审查指出）**：① 99.78% 是**在合成 450 集 in-sample 测的**，分离阈值 `sep=(max(babble)+min(nonbabble))/2` 直接由该同集极值算出——合成噪声谱特征干净（white/pink/babble 各自谱平坦度无重叠），真实噪声分布不同，准确率必然下降；② 估计器**机制成立**（谱平坦度区分稳态/非稳态噪声在原理上有效），但**阈值需在真实噪声上重新校准**，99.78% 数字**不可直接外推到比赛**。

### target 缺席拒识测试（72 条，真实拒识率）✅ 强阳
`build_reject_set.py` 造 72 条 target 缺席音频（苏打+噪声，配冰糖 enrollment→正确行为=拒识）：

| 信号 | 拒识率 | cer_final | 解读 |
|---|---|---|---|
| **sim_only(t=0.2)** | **100%** | 0.000 | max_sim 均 0.026（苏打 vs 冰糖 enrollment），全部正确拒 |
| llm_only / 三路融合 | 100% | 0.000 | LLM 把闲聊/英文幻觉判 reject，三路(sim+LLM 都拒)正确 |
| stno_only(t=0.05) | 32%（**误放行 68%**） | 0.681 | ⚠️ stno 是坏拒识信号 |

**关键发现**：`stno_target_ratio` 衡量"所选 speaker 的独占帧占比"，但所选 speaker 可能是苏打（错选）→ 它说话多→stno 高却非 target → **stno 单独会误放行非目标**。**验证 sim 是拒识锚信号**（声纹匹配才是"target 在不在场"的真证据），stno 只能作辅助（三路里 sim+LLM 双否决时 stno 权重 0.2 仍正确拒）。→ **拒识侧 100% 真实拒识率，强阳**，平衡了 CER 侧的瓶颈。
**注**：72 条全是单说话人苏打（无第二非目标声音），真实比赛"≤2 说话人都非 target"会更难；本结果证明机制成立，真实数据来时需扩。

### 结论与下一步（集成后的方向重定）
1. **集成达成**：三线串成单一 pipeline，真实组合指标产出，机制全通。
2. **瓶颈铁定在 Whisper 转写质量（babble 英文幻觉），不在融合/拒识**——融合调参是死路，这与 T17"瓶颈转移到 Whisper 带噪转写"一致且更强证。
3. **真正提升 CER 的杠杆**（按可行性）：① 中文家居微调（让 Whisper 在 babble+中文上鲁棒，重但治本）② SE-DiCoW（enrollment 条件化，攻重叠+babble 死区）③ 更强 babble SE。三者都需重投入，**取决于真实数据/通道数确认**。
4. **可部署交付物**：SE 条件化（可部署 CER 2.82）+ 三路融合框架（接口齐，待转写质量上去即生效）+ 噪声估计器 + target 缺席拒识集。组合主线"只转 target、拒识 non-target"闭环工程上已成立，**下限取决于 Whisper 带噪中文能力**。

---

## T20：SE 条件化 post-fix 重评 + 归因深化（2026-06-30，对抗审查后）

> 背景：T19 修复 DiCoW language 死代码 bug（langfix）后，重跑 SE 条件化 post-fix。原任务=重跑 conditional（pink→=6）post-fix，预期 ~2.7。**结果颠覆预设：post-fix 后 =0/=6 优劣格局反转，并挖出 babble 失败的更上游根因（diarization 误检 babble 为 speaker + STNO target 帧清零，非 langfix/Whisper 本身）。**

### 1. 6 集 overall CER 交叉表

| 配置 | overall | white | pink | babble | 中文占比 | 正确率(CER<0.5) |
|---|---|---|---|---|---|---|
| pre-fix conditional（T18）| 2.825 | — | — | — | 17% | — |
| post-fix se0（=0 全力）| 3.542 | 3.230 | 2.201 | 5.196 | 30% | 7.8%（35/450）|
| post-fix se6（=6 温和）| **2.504** | 1.350 | 1.282 | 4.879 | 58% | 15.1%（68/450）|
| post-fix 旧 conditional | 3.236 | 3.230（se0）| 1.282（se6）| 5.196（se0）| 38% | — |

### 2. 5 策略对比（post-fix，450 条）

| 策略 | overall | 工程定位 |
|---|---|---|
| 全 se0（=0）| 3.542 | 最差 |
| 旧 conditional（babble,white→0, pink→6）| 3.236 | 原任务产出；规则系 pre-fix 经验，post-fix 后过时 |
| 新 conditional（white,pink→6, babble→0）| 2.609 | **稳健推荐**（仅改 white→=6）|
| 全 se6（=6）| 2.504 | 最简，但 babble 低重叠有牺牲 |
| 精细二维（babble ov≤0.25→0，其余→6）| **2.022** | oracle 最优；过拟合风险（每档 n=30）|

### 3. ⭐ 关键发现：babble 失败的更上游根因（修正 T19 归因）

T19 把 babble 英文幻觉归因为"Whisper 硬噪声鲁棒性"。post-fix 重评 + 字段核查（`code/_diag_lock.py`）发现**更上游的级联根因**，babble 条失败链：

1. **DiariZen diarization 误检**：babble ov0（overlap=0，本应单人）**100% 检出 2 个 speaker**（`speakers={2:30}`）——把 babble 人声噪声当成第 2 个 speaker。
2. **声纹 max_sim 崩**：babble 各档 max_sim **0.051–0.209**（white 0.335 / pink 0.436），大量条 <0.2（16–26/30）。
3. **STNO target 独占帧清零**：`stno_target_ratio`（=target 活跃且无其他 speaker 的帧占比，`enroll_infer.py:189`）在 babble 全档 = **0.00**——diar 把 target 与 babble-noise-speaker 判定帧级完全重叠，target 无任何独占帧。
4. **DiCoW 转写崩溃**：STNO mask 的 target 行空 → DiCoW 在退化信号上英文幻觉/重复循环（714 字 "i don't know if you..."、596 字 "oh no no..."）。

**抽样铁证**：babble ov0 snr+5 `max_sim=0.545`（声纹尚可）但仍 `stno_target=0` → 转写 596 字循环崩溃。**即使声纹锁住，STNO 崩也救不了**。对照：white ov0 snr+5 `max_sim=0.440` →「请把客厅的空调温度调到二十六度」**完全正确**。

**⚠️ 归因再修正（核实 se0 字段后）**：原以为"=0 防 diar 误检"——**错**。核查 se0(=0) babble：全档也 `speakers={2:30}`（diar 同样误检）、`stno_target≈0.01`（≈0，与 se6 一样）。即 **babble 的 diar 误检 + STNO≈0 是 =0/=6 都救不了的顽固现象**。se0 babble ov0 CER 较低（2.68 vs se6 7.84）的真因是 **=0 强降噪压制 babble 人声 → Whisper 幻觉更短**（se0 transcript 短），=6 保留 babble → Whisper 在丰富人声噪声上幻觉超长（se6 714 字循环）。**差异在幻觉长度/降噪层，不在 diar 层**。这反而强化结论：babble diar 误检需 SE 之外的更上游手段（源分离 / diar 模型改进 / STNO 构造绕过 diar）。精细策略 babble ov≤0.25 选 =0 的真实理由是"幻觉更短"而非"防误检"。

### 4. langfix 效果边界

langfix 对 **white/pink 低重叠有效**（se6 中文 93%、正确率 60%@ov0）；对 **babble 无效**——但真因**不是 langfix**，而是上述 diar+STNO 崩溃级联（langfix 无关：STNO 都崩了，强制中文 token 起始压不住 decoder 后续幻觉）。T19"修复非银弹，残留瓶颈=Whisper 硬噪声"需修正为：**残留瓶颈=babble 条 diar 误检 + STNO 崩（更上游），Whisper 转写质量是第二层**。

### 5. 结论与改进方向（诚实）

- **原任务完成**：旧 conditional post-fix = 3.236（规则过时，非最优）。
- **post-fix 最优**：精细二维 2.022（oracle，过拟合）/ 新 conditional 2.609（**稳健推荐**）/ 全 se6 2.504（最简）。
- **绝对值仍差**：se6 正确率仅 15.1%、幻觉 50%，系统远未可用——"2.504"是相对 se0 的改善，**非成功**。
- **真瓶颈（修正 T19）**：babble 条 **diarization 误检连续人声噪声为 speaker + STNO target 清零**（=0/=6 都存在，非单纯 Whisper）。改进杠杆（多元，T20 核实 se0 后修正）：① **babble 专用源分离/更强降噪**（⚠️ DeepFilterNet3 =0/=6 都救不了 diar 误检——se0 babble 全档也 speakers={2:30}+stno≈0；需针对性 babble 人声抑制或源分离模型）② 声纹 babble 鲁棒（CAM++/enrollment 增强，max_sim 0.05–0.21 太低）③ STNO 构造绕过 diar 误检（不依赖 diar 的 target 锁定）④ Whisper 中文微调（治转写层幻觉长度，但 diar 误检未解时收益有限）⑤ SE-DiCoW（enrollment 条件化可能绕过 diar）。
- **诚实声明**：450 条仿真数据，精细策略 oracle 选条过拟合风险；babble 误检根因基于字段关联（speakers=[0,1] + stno_target=0 + 英文幻觉的同现），因果链待真实数据/单步消融进一步确证（如同 T19 langfix bug 待对抗审查才定论）。

**产出文件**：`code/enroll_regen_se6.json`（=6 post-fix 转写）、`code/se_conditional_postfix.json`（旧 conditional post-fix）、`code/_diag_full.py` / `_diag_lock.py`（复现诊断）。

---

## Task 7 — submit_infer.py 集成验收(2026-07-01, --limit 3 烟雾)

标准化推理脚本 `submit_infer.py`(仅 stdlib subprocess 编排器)三档集成验收通过(本机 RTX 4060 Laptop 8GB, `--limit 3`, enrollment=`target_long_01.wav`, recognition=`test_wav/dataset/final` 前 3 条):

| 档 | 配置 | n | overall_rtf | audio/wall | phases |
|---|---|---|---|---|---|
| 最简 | `--no-se --no-llm` | 3 | **1.6737** | 8.16s / 13.66s | enroll_diar_dicow=13.61 (mean_rtf 0.333) |
| 带SE | `--no-llm` | 3 | **2.2283** | 8.16s / 18.18s | noise_classify=2.01 / se=4.00(n=3) / enroll=12.09 (mean_rtf 0.247) |
| 全量 | SE+LLM | 3 | **3.4793** | 8.16s / 28.39s | noise_classify=1.88 / se=3.21(n=3) / enroll=10.81 (mean_rtf 0.248) / llm=12.40 |

**结果文件 schema 校验通过**(三档 result.json/timing.json 均符合 `交付/使用说明.md §4`):
- 启用 SE 的两档(带SE/全量): `noise_type` 落盘为 `"pink"`/`"babble"`/`"pink"`、`atten_lim_db` 为 `6`/`0`/`6`(**非 null**, I1 修复后回连)。
- 最简档(`--no-se`): `noise_type`/`atten_lim_db` 为 `null`(SE 跳过未估噪声, **预期行为**)。
- `llm_verdict`: `--no-llm` 档为 `null`; 全量档 3 条均 `reject` —— 但 `rejected=false`(`llm_or_sim` 策略需 `llm!=accept` **且** `max_sim<0.2`, 此处 max_sim≈0.30 救回)。

**⚠️ 诚实声明**:
- `overall_rtf` 含**模型加载首次开销**分摊到仅 8.16s audio 的小批量假象(DiCoW+DiariZen 首次加载 + Qwen 首次加载在全量档贡献 ~10s wall)。450 条大批量 + L20 会显著降低。
- 纯 DiCoW minimal RTF=0.058(W1 实测)为推理下界参考;上表 mean_rtf 0.25–0.33 含 diar+enroll 开销,非纯转写。
- 三档转写印证 **T19/T20 诊断**: babble 条(utt0001)出现"乎的乎的..."长幻觉循环(~110 字重复, DiCoW 在 stno_target≈0 退化信号上的典型崩溃), pink snr-5 条(utt0002)出现英文漂移("so by taking a picture...")。max_sim 0.30 区间在 `llm_or_sim` 下救回 LLM 拒识(即 LLM 判 reject 但声纹未触发 → 不拒)。

**产出文件**:`code/submit_out_min/`(最简)、`code/submit_out_se/`(带SE)、`code/submit_out_verify/`(全量, 验收主目录) — 烟雾产物, 非提交结果。

---

## T22 — P2 babble 工程兜底：消融 + oracle 铁证 + 对抗审查反转（2026-07-02 晚）

**背景**：T20 把 babble 英文幻觉归因为「diar 误检幽灵 speaker → STNO target 帧清零 → FDDT 错标 overlap 通道」（H1）；T19 归因为「Whisper 硬噪声」（H2）。两者**从未单步消融区分**。P2 用消融 + 对抗审查 + oracle 铁证钉死。

**3 步**：①单步消融 `code/stno_ablation.py`（同一 babble 样本 t_01_n_07_ov000_snr+5，只改 STNO：A 现状/B 丢幽灵/C 单 spk，white 同 SNR+5 对照）②对抗审查（Workflow 3 agent）③oracle 铁证 `code/babble_oracle_test.py`（①babble+oracle 全程 target ②纯 babble ③干净 nontarget）。

| 实验 | STNO target行 | 输出 | 语言 |
|---|---|---|---|
| stno_ablation A 现状 | 0.000 | 53字"i mean you can't wait..." | 英文 |
| stno_ablation B 丢幽灵 | 0.067(diar派生) | 61字"i think you can do it..." | 英文 |
| stno_ablation C 单spk | 0.067(diar派生) | 同 B | 英文 |
| white 对照 A=B=C | 0.091 | 「先把客厅的空调温度调到二十六度」 | **中文** ✓ |
| **oracle ①** babble+全程target | **1.0** | 200字「我可以用它用它...」循环 | **中文** |
| **oracle ②** 纯babble(4中文叠加) | 全程 | 「我刚看了一个很有趣的...」片段 | **中文** |
| **oracle ③** 干净nontarget n07 | 全程 | 「帮我倒杯水好吗谢谢」 | **中文** ✓ |

**决定性发现（反转）**：
1. **H2 强形式证伪**：纯 babble（②，4 条**中文** nontarget 叠加，无 target）输出**中文片段**，不漂英文 → **babble 音频本身不让 Whisper/DiCoW 漂英文**。之前"babble 音频致英文幻觉"归因错误。（用户质疑"babble 含英文"也核实排除：n_01~08 全中文 voice=苏打，见 `build_dataset.py:52` gen_babble + `raw/manifest.json`。）
2. **STNO target 行覆盖率因果主导语言**：同一 babble 样本，diar 派生 0.067 → 英文；oracle 全程 1.0 → 中文（①）。唯一变量是 STNO 覆盖率。
3. **内容层另有瓶颈**：① 虽中文却"我可以用它用它..."循环崩 → **双层瓶颈：语言层（STNO 可修）+ 内容层（babble 音频质量/Whisper 鲁棒性）**。

**对抗审查救场（避免切向错误杠杆）**：原消融漏洞——C 用 diar 派生帧（非 oracle）/ B==C 退化（假冗余）/ 结果文件被 white 对照覆盖（babble H2 无产物）/ 无法区分 H2 vs **H3（language-drift，langfix 只锁首位 token 的残留）**。若无审查，会基于"H2 成立"切向 **SE-DiCoW**（错误杠杆）。memory `adversarial-review-before-milestone-commit` 完美体现。

**未 100% 排除**：①②③ 都用"全程 target STNO"，未干净排除 STNO 混淆；vanilla Whisper（无 FDDT）本地无未跑，babble 独立效应未彻底排除。彻底分 H2/H3 需 vanilla Whisper + prefix-forced decode。

**对 P2 的影响**：杠杆 A（修 STNO）**部分有效**——能救语言（英文→中文），内容需 SE/微调。但单纯"丢幽灵 speaker"（B/C）不够（diar 派生 0.067 仍英文），真正起作用的是"提高 target 行覆盖率"，机制疑为 H3（language drift 受 STNO 覆盖率影响），待确证。归因 slippery → 符合 `stop-digging-on-sim-data`，真瓶颈待真实 A 集/通道数。

**产出**：`code/stno_ablation.py`、`code/babble_oracle_test.py` + 本节结论。

### vanilla Whisper 三角定位（2026-07-03，H3 确证）

补审查 P0#1：vanilla whisper-large-v3-turbo（**无 FDDT/STNO**，DiCoW 基座原版）转同样三场景：

| 场景 | vanilla Whisper 输出 | DiCoW+diar STNO |
|---|---|---|
| ① babble 样本 t_01_n_07 | **「请把客厅的空调温度调整」(11字近全对)** | 英文幻觉 53字 |
| ② 纯 babble(4中文叠加) | 中文循环「我刚看了一个很有趣的千贵...」216字 | — |
| ③ 干净 target t_01 | 「请把客厅的空调温度调到26度」✓ | — |

**三角定论**：
- **H2 彻底证伪**：vanilla Whisper（无 STNO）在**同一个 babble 样本**上输出**正确中文** → Whisper 基座在 babble 上完全正常，不漂英文。babble 英文幻觉 **100% 是 DiCoW 的 FDDT/STNO 条件化病害**（H3 确证）。
- **机制**：DiCoW FDDT（每层 encoder 门控）在低覆盖 STNO（diar 派生 0.067）下，大量帧走 overlap/silence 通道 → encoder 表征劣化 → langfix 只锁首位 token 压不住 decoder 后续 → 英文漂移。vanilla 无 FDDT，不劣化，language=zh 正常生效。
- **杠杆指向（再修正）**：非 SE-DiCoW（H2 错），而是 ① 提高 STNO target 行覆盖率（避免 FDDT 劣化）② 更强 language forcing（constrained decode 锁全程 zh token）③ FDDT 鲁棒性。**洞察**：DiCoW STNO 条件化在 babble 上适得其反（vanilla 反而对），但 vanilla 失去 target-speaker 选择性，不能直接替代。
- 下载：vanilla 权重 `E:/hf_cache/whisper-large-v3-turbo`（hf-mirror+代理 12MB/s，~2 分钟；无代理仅 0.5MB/s）。脚本 `code/vanilla_whisper_test.py`。

---

## T23 — 2026-07-04 datasetA 真测三档 + 关LLM 保底决策（pos 1364 / neg 474 全量真测）

**背景**：datasetA 到手（单通道16k / enrollment ~1.8s），跑保底全量确认提交数字。三档对比（thr=0.2 基线 / thr=0.4 开LLM / thr=0.4 关LLM），评测 `eval_datasetA.py`（pos 误拒/空=CER1.0 + zhconv 繁简归一；neg RR=拒条/n）+ `analyze_pos_full.py`。

| 配置 | pos CER均值 | pos **correct**(CER<0.5) | pos 误拒 | pos cer_accepted | neg **RR** | neg 漏拒 | RTF(pos/neg) |
|---|---|---|---|---|---|---|---|
| thr=0.2 sim_only（基线）| 1.248 | **31.30%** | 30.0% | 1.354 | 77.00% | 23%（均值20.8字）| ~0.25 |
| thr=0.4 +LLM（llm_or_sim）| 0.987 | 15.84% | 76.9% | 0.944 | 96.20% | 3.8%（9.3字）| **1.01 / 0.70** |
| **thr=0.4 关LLM**（sim_only `--no-llm`）| 1.007 | 13.93% | 79.0% | 1.031 | **98.52%** | **1.5%**（11.4字）| **0.257 / 0.203** |

**⭐ 关LLM vs 开LLM = trade-off（⚠️ 3-agent 对抗审查修正：原「全面优于/pos 持平」被推翻，commit 前救场，详见末尾 7 GAP）**：
- **关LLM 赢**：neg RR +2.32pp（98.52% vs 96.20%，`decide_reject` llm_or_sim 拒=(llm≠accept)且(max_sim<thr)，LLM 把 11 条 neg 误判 accept 漏拒）；RTF **4×**（0.257 vs 1.01，LLM 阶段独占 pos 73%/neg 70%）。**Qwen2.5-3B 零样本拒识在 neg 上是负贡献**。
- **开LLM 赢 pos 救回（原"pos 持平"错）**：pos correct +1.91pp（15.84% vs 13.93%）、near_perfect +1.9pp（9.6% vs 7.7%）、cer_accepted_only -0.09（0.944 vs 1.031）。**铁证：28 条 LLM 救回的 pos（max_sim<0.4 但 llm=accept）里 26 条 CER=0.000 完美**（"关闭客厅空调""打开闹钟"，max_sim 低至 0.022/-0.039 —— Whisper 转对但声纹提不出被 sim_only 误杀，LLM 语义校验救回）。11 条 neg 泄漏（LLM 假接受）是语法合法家居指令幻觉（"打开洗碗机"），LLM 判不动 —— 固有盲区。
- → 保底选关LLM 真实理由 = **为效率20%+RR40% 牺牲 pos 救回**（RTF 4× 硬伤；pos 40% 反正架构极限放弃）。**不是"全面优于"**。若 LLM 量化/蒸馏降到 RTF~0.4，pos 救回可能净赢（开放问题）。

**⚠️ 两个陷阱（thr 选择关键）**：
1. **CER 均值是幻觉陷阱**：thr 0.2→0.4 时 pos CER 1.248→1.007 看似改善，实是"误拒把 babble 幻觉超长样本（hyp 超长重复循环，CER>>1 如 66/39/20）换成 CER=1.0"的数学假象。**correct_rate 才诚实：31.3%→13.9% 是真退化**（印证 CLAUDE.md 教训）。
2. **pos/neg thr 拉锯**：thr 高 → neg RR 升（拒识分↑）但 pos correct 降（CER 分↓）。同一 thr 无法两全。

**pos CER ~1.0 是架构极限**（sim_thr 全档 0.10–0.50 扫描 CER 都 ~1.0，无 thr 能救）：
- babble 89% 主导（n=1218/1364），cer_accepted_only 0.94–1.03（未拒的也几乎全错）
- 两极分化：完美(CER<0.05) **9.2%** / 灾难(CER≥0.5) **81.5%**，中间几乎空
- 极差样本全 babble：重复循环幻觉（id2323 hyp="先选民民民民..." ref="防直吹"）/ 英文漂移（id2429 hyp="i don't know what should" ref="把温度调到二十三度"）
- pos max_sim median 0.283，≥0.5 仅 7.7%，≥0.8 为 0 → 声纹在 babble 下普遍提不出 target
- 按 noise_type：babble correct 16.9% / pink 11.0% / white 1.6%；按唤醒词：小钱小钱 CER 2.73、小编小编 1.45（某些唤醒词 enrollment 质量差，全错）
- **印证 T22 仿真归因 + AGENT_HANDOFF 核心认知**：组合主线 cascaded 在极重 babble 下 CER 是架构极限（target 声纹 max_sim median 0.283（⚠️ sim<0.06 仅 7.7% 非主流，旧"sim<0.06 提不出"数字错，见下文 doc 数字修正）+ DiCoW FDDT 在 babble STNO 下 mel 退化转写崩）。且 correct 比预期更差（13.9–15.8% vs 交接 31%）。

**保底决策**：
- **确定**：关LLM（trade-off：为效率20%+RR40% 牺牲 pos 救回，**非"全面优于"**；用 wrapper `code/run_baodi.sh` 锁死 `--no-llm --sim-thr`，防 submit_infer 默认 flag 灾难）
- **thr 待主办方评测口径定**：CER 均值评分 → thr=0.4 或 0.45（within-noise 占优，RR 98.5%→99.2%）；correct_rate / pos 不许拒 → thr=0.2（31%）甚至 0
- **临时占位**（评测口径确认前）：关LLM thr=0.4（neg RR 98.5% / RTF 0.24 / pos CER 1.0 架构极限）
- **评分盘算**：pos CER ~1.0 → CER 40% 基本放弃；靠拒识 40%（RR 98.5%）+ 效率 20%（RTF 0.24）拿分。

**⚠️ 待确认（最高优先，thr 决策前提）**：向主办方问评测口径 —— CER 均值 vs correct_rate？pos 被拒算多少（当前保守 CER=1.0）？pos 是否允许拒（不许则 pos thr=0 全转写）？pos/neg 能否不同 thr？（`datasetA/readme.txt` 仅"pos 测 CER / neg 测 RR"，未明确 pos 被拒处理）

**产物**：`code/out_pos_full`+`out_neg_full`(thr=0.2 基线) / `out_pos_final`+`out_neg_final`(thr=0.4 开LLM) / `out_pos_noLLM`+`out_neg_noLLM`(thr=0.4 关LLM = 保底)；`code/analyze_pos_full.py`（max_sim 分布 + sim_thr 工作点扫描 + CER 分桶 + 噪声/唤醒词分组 + 极差样本）；`code/t1_fullrun.log`+`t1_nollm.log`；`code/run_baodi.sh`（保底 wrapper）；memory `baodi-config-no-llm`。

### ⚠️ 3-agent 对抗审查 7 GAP（commit 前救场，memory `adversarial-review-before-milestone-commit` 教科书案例）

原 T23 正文"关LLM 三项全胜/pos 持平"**被审查推翻**（见上"trade-off"修正）。其余 6 GAP：

1. **【高】CER ±0.04 噪声底限**（DiCoW fp16 generate 非确定性）：同 thr=0.4 跨 run，反推自 out_pos_full=0.9644 vs 真跑 out_pos_noLLM=1.0066，Δ=0.042（21/287 accepted utt transcript 不同）。→ **langfix"边际 CER 0.028"在噪声内不可靠**（英文率 31.6%→18.5% 可靠，因 max_sim 确定）；**pos CER thr 0.35–0.55 差异全在噪声内**（thr 对 pos CER 不可区分）。需多 seed / 确定性 decode（beam/温度0/固定 seed）。
2. **【高】L20 batch=1 硬编码未实现**：`enroll_infer.py:182` 单 utt 循环 + `:263` dicow.generate batch=1，全仓无 `--batch-size`/显存自适应。CLAUDE.md/memory"L20 48GB 大 batch"是 **TODO 未实现**。效率 20% 腿在 L20 实测前是猜测。
3. **【高】提交默认 flag = 灾难**：submit_infer 默认 strategy=llm_or_sim / sim_thr=0.2 / llm=ON → RTF 1.0 + neg RR 0.77 两腿崩。**保底必须显式 `--no-llm --sim-thr 0.4`，已写 wrapper `code/run_baodi.sh` 锁死**。
4. **【中】"三路融合拒识"被证伪**：`decide_reject` llm_or_sim 实为 **AND**（llm!=accept AND max_sim<thr），**LLM 只能减拒不能加拒**。final(LLM on) 全维度更差。**答辩别列三路融合为强项**（CLAUDE.md D4 差异化需改）。
5. **"99%@thr=0.4"高估**：实测 neg RR@thr=0.4 = **98.52%**（7 漏），99% 需 thr=0.45（99.16%）。**thr=0.45 within-noise 占优 thr=0.4**（RR +0.7% 免费，pos CER Δ 在噪声内）。
6. **【中】neg 漏拒口径**：7 漏拒含长新闻（"体育产业成资本新宠..."23 字 / "温必俧日前赴北京..."16 字），5/7 babble sim 膨胀 0.40–0.67。句准 vs char-weighted 未验证，若后者损失 >1.5%。**需问主办方**。
7. **pos CER 40% 全口径 conceded**（rejected=1.0 / 排除误拒 cer_acc 1.03 / 必须转写空 hyp 都~1.0），100% 押 RR+效率，无冗余 —— 是"保底"非"稳健"。

**doc 数字修正**：「target 声纹提不出 sim<0.06」**错** —— 实测 max_sim **median 0.283 / mean 0.286 / min -0.125**，sim<0.06 仅 7.7%。改用"median sim 0.28，低 sim 桶 correct 仅 30%"。**答辩别引用 sim<0.06**。

**thr<0.2 盲区**：submit_infer 在 max_sim<thr 短路跳 DiCoW（被拒 text 强制空），thr=0/0.1 correct 无法直测（间接外推 ~35-39%，不会明显>31%）。`analyze_pos_full.py` 工作点扫描 thr<0.2 行被污染（empty 当 cer=1.0），**引用需限定 thr≥0.2**。

---

## T24 — 2026-07-06 Phase 1 vanilla vs DiCoW 全量对比（H3 强证伪，CER 减半）

**背景**：T22 仿真 + T23 真测均显示组合主线 cascaded 在极重 babble 下 CER ~1.0–1.25 是架构极限。Phase 1 用 zero-training 思路验证：**去掉 DiCoW 的 FDDT/STNO 条件化，改用 vanilla Whisper-large-v3-turbo + 声纹切 target timeline**。脚本 `code/exp_vanilla_vs_dicow.py`（全量 1362 条 pos，always_generate 不拒；vanilla/dicow 同 max_sim 由 diar+wespeaker 计算）。

### 1. 转写质量（不拒，全 1362 条）

| 指标 | vanilla | dicow | Δ |
|---|---|---|---|
| **转写 CER** | **0.664** | 1.248 | **−0.584（几乎好一倍）** |
| **correct_rate (CER<0.5)** | **45.6%** | 31.4% | +14.2pp |
| near_perfect (CER<0.1) | **20.8%** | 14.8% | +6.0pp |
| **英文幻觉率** | **0.59%** | **18.80%** | **−18.21pp ← DiCoW 条件化主动造孽** |

**英文幻觉根因坐实**：DiCoW 条件化造 18.8% 英文幻觉，vanilla 仅 0.59%。之前 langfix 是在打 DiCoW 自己造的孽（治标），vanilla 路线从根消灭（治本）。

### 2. thr 工作点（含拒识拒=1.0 = 提交 overall CER）

| thr | vanilla overall CER | dicow overall CER | Δ |
|---|---|---|---|
| **0.20** | **0.711** | 1.241 | **−0.530 ← vanilla 终于把 overall 拉到 <1** |
| 0.30 | 0.774 | 1.093 | −0.319 |
| 0.40 | 0.867 | 0.964 | −0.097 |

**评分盘算**：thr=0.20 时 overall CER 0.711 → **CER 40% 腿从 ~0 分变 ~11 分**（线性 (1-0.711)×40 ≈ 11.6，待主办方 CER 口径确认）。

### 3. sim 分桶（DiCoW 条件化最毒的铁证）

| sim 桶 | vanilla CER | dicow CER | Δ |
|---|---|---|---|
| [0.2,0.3) | 0.746 | **1.606** | **−0.860** |
| [0.3,0.4) | 0.623 | **1.523** | **−0.900 ← 条件化最反作用** |
| ≥0.4（轻 babble） | 0.364 | 0.830 | −0.466（仍优） |

**机制**：sim 0.2–0.4 是 babble 重灾区，DiCoW 的 FDDT encoder 门控在低覆盖 STNO 下大量帧走 overlap/silence 通道 → encoder 表征劣化 → 英文漂移/重复循环幻觉；vanilla 无 FDDT，不劣化，language=zh 正常生效。**cascaded 条件化在中等 sim 桶反作用最烈**。

### 4. 路线机制（zero-training）

```
diar（DiariZen wavlm-large）+ wespeaker 声纹
  → 选 target speaker（复用 enroll_infer 锁定逻辑）
  → 切 target timeline 段（含 target 活跃的重叠区）拼接
  → vanilla Whisper-large-v3-turbo 转写（去掉 stno_mask/FDDT 条件化）
```

无任何训练/微调；与 DiCoW 共享前置 diar+声纹（公平对比），唯一变量是 ASR 后端（vanilla vs DiCoW+FDDT）。

### 5. 答辩弹药

「cascaded 条件化机制在极重 babble 下反作用（sim 0.2–0.4 桶 CER 1.5–1.6、英文幻觉 18.8%），改用 target extraction + vanilla Whisper，CER 几乎减半」——契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。比"端到端联合训练 X"轻得多（zero-training 即斩获大部分收益），是保底之上的现实破局路线。

### 6. 产物

- `code/exp_vanilla_vs_dicow.py` — 全量对比实验脚本（vanilla vs dicow 同 max_sim）
- `code/analyze_vanilla_full.py` — vanilla 全量深度分析（thr 工作点 + sim 分桶 + 英文幻觉）
- `code/exp_vanilla_full.json` — 实验结果数据
- memory `h3-dicow-conditioning-backfire-vanilla`

### 7. P2 待做（Phase 1 后续落地）

1. **vanilla 集成 submit_infer**（最高优）：`--asr-backend {dicow,vanilla}` 切换，把 0.664/0.711 变提交数字
2. **声纹强化**：CAM++ per-speaker / US-PVAD 改善 target timeline 切割（低 sim 桶当前是瓶颈）
3. **数字 initial_prompt**：家居指令数字/温度场景的锦上添花
4. **sim_thr 待主办方评测口径**：CER 均值→0.4 / correct→0.2 / pos 不许拒→0

---

## T25 — 2026-07-06 P2-① vanilla 集成 submit_infer 落地（提交数字 + 官方格式）

T24 §7 第1条落地。方案 A：`enroll_infer.py` 加 `--asr-backend {dicow,vanilla}`，共享 diar+声纹+选target，转写分叉（dicow=stno 条件化 / vanilla=切 target timeline 拼接无 mask）。vanilla 作提交主线，dicow 保留 fallback。新增 `text_utils.py`（繁简归一+timeline 切割，单测）+ `to_submission.py`（result.json→官方格式）。繁简归一顺带修 HANDOFF §8 坑4。pyarrow 预热避 WinError 6714。

### 全量提交数字（datasetA，vanilla 主线）

| 评分腿 | 指标 | 数字 | dicow 保底对比 |
|---|---|---|---|
| **pos CER 40%** | overall CER（thr=0 全转写，1364 条）| **0.667** | dicow 1.25（~0 分）|
| **neg RR 40%** | 句准拒识率（thr=0.4，474 条）| **98.52%** | dicow 98.5%（sim 复用持平）|
| **效率 20%** | RTF / duration（batch=1）| RTF **0.19–0.22** / pos 503.7s + neg 181.3s | dicow RTF 0.24 |

**CER 腿**：(1−0.667)×40 ≈ **13.3 分**（dicow ~0 分）→ vanilla 集成让 CER 腿从 0 分变 ~13 分。pos thr=0 全转写（误拒仅 0.88%），cer_accepted 0.664=overall（干净口径）。

### 验证证据

- 冒烟 100 条 vanilla transcript 与 exp 脚本逐条一致（集成正确）+ 繁简归一生效（输出简体）
- dicow 回归 100 条 `asr_backend=dicow` 确认走条件化路径（fallback 不坏）
- submission.json schema 完整（id/content/label/cer/final_cer/duration，id 无 utt 前缀，duration 对齐 batch=1）
- 全部单测 PASS（text_utils / submit_infer / to_submission）

### 提交命令

```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1
# pos 全量 vanilla（thr=0 全转写，CER 0.667）
BAODI_OK=1 code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs code/pos_pairs_datasetA.json --out-dir code/out_pos_vanilla_full \
  --no-llm --sim-thr 0 --strategy sim_only --asr-backend vanilla
# neg 全量 vanilla（thr=0.4，RR 98.5%）—— run_baodi 默认 vanilla
bash code/run_baodi.sh neg 0.4
# 转官方格式
code/.venv/Scripts/python.exe code/to_submission.py --result-json <out>/result.json --pairs code/<set>_pairs_datasetA.json
```

⚠️ thr 待主办方口径（memory `official-scoring-spec`）：pos 不许拒→thr=0（CER 0.667）/ CER 均值→thr=0.4（overall 0.867）。run_baodi 默认 thr=0.4 是 neg 口径，pos 全转写需显式 thr=0。官方格式 6 点待确认（label 语义/pos 被拒 cer/neg cer 填法/final_cer 算法/duration 含 SE?/pos+neg 交法）做成 `to_submission.py` 的 `SUBMISSION_DEFAULTS` 常量，主办方回复只改常量。

### 产物

- `code/{text_utils,to_submission}.py` + `tests/test_{text_utils,to_submission,submit_infer}_logic.py`
- `code/out_{pos,neg}_vanilla_full/{result,timing,submission}.json`
- spec `docs/superpowers/specs/2026-07-06-vanilla-backend-submit-infer-design.md` + plan `docs/superpowers/plans/2026-07-06-vanilla-backend-submit-infer.md`

---

## T26 — 2026-07-06 可复现性改造落地（FAQ 核查 6 项硬要求，5 Phase 全验证）

FAQ 2026-07-06 公布：**核查方式 = 完整复现结果比对**（非仅看代码），6 项硬要求（零外部依赖/种子固定/禁缓存/日志/显存/run-twice）。本次让 submit_infer 全链路可复现，过核查。spec `docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md`。

### 5 Phase 落地

| Phase | 内容 | 验证 |
|---|---|---|
| P1 repro.py | 公共模块 set_global_seed/resolve_model/peak_gpu_mib | 5 单测 PASS |
| P2 5 脚本+setenv | import repro + set_global_seed + --seed 透传 4 子进程 + 模型 resolve + 显存日志 | 冒烟 pos limit=3 全链路跑通 + batch/peak 字段齐全 |
| P3 DiCoW 部署 | 方案 C 文档化 REPRO_SETUP.md（submodule 阻塞：.gitignore `code/*/` negation 在 git 不生效）| — |
| P4 verify run-twice | 同 seed 跑两遍 enroll_infer 比对 | **text 一致 100%, CER delta=0**（fp16 确定，无需 fp32）|
| P5 回归 | pos/neg/dicow limit 验证 | pos CER 0.27(前100易样本)/neg RR 96%/dicow 不坏 |

### 关键结论

- **fp16 完全确定**：set_global_seed + cudnn.deterministic=True/benchmark=False 使 vanilla Whisper run-twice CER delta=0，**无需升 fp32**（效率腿不伤）。
- **模型走 HF repo id**：4 模型 default `from_pretrained(repo_id)`，本地 setenv 设 MODEL_* env override 复用缓存；DF3 例外（GitHub Rikorose/DeepFilterNet + env）。
- **B 集混合集**：submit_infer 天然支持（一个 manifest + 一个 thr），pos/neg 不作输入（C9）；统一 thr 选点是 follow-up。

### 约束 C1-C11（FAQ + 用户确认）

pos 拒 CER1.0 / 排名制 / B 集 dir1/dir2 混合统一 thr / pos-neg 不作输入 / batch 默认1（允许但须一致，RTF 用1测）/ CER=输出 vs 标准答案识别文本 / 主办方联网 HF / 三 venv 隔离。

### 范围外（follow-up）

统一 thr 选点（B 集必需）/ batch 推理加速 / 攻 CER（声纹强化 CAM++/US-PVAD）。

### 产物

- `code/repro.py` + `tests/test_repro_logic.py` + `code/verify_reproducibility.py` + `REPRO_SETUP.md`
- 5 脚本改造（submit_infer/enroll_infer/se_denoise/llm_reject/noise_classify）+ `code/setenv.sh`
- 5 commit: 609779c(spec) / 1e0ca55(P1) / fbff320(P2) / 88d676d(P4) / 7b7ca17(P3)
- memory `reproducibility-hardening`

---

## T27 — 2026-07-07 统一 thr 选点(B 集 A 集模拟 + 5-agent 对抗验证)

> B 集是赛事方最终评测题(参赛方拿不到, 否则作弊), pos/neg 混合不预分, **必须用单一 thr 处理所有音频**(FAQ C9)。本次用 A 集(已知 label)当验证集模拟 B 集混合场景, 选定稳健统一 thr, 并经 5-agent 对抗验证修正。

### 决策: 统一 thr=0.27(区间 [0.26,0.29])

| 项 | 值 | 说明 |
|---|---|---|
| **推荐统一 thr** | **0.27** | bootstrap 中位 0.28, IQR=[0.27,0.28], 80%CI=[0.27,0.34] |
| 细扫真峰(0.005) | 0.275 | 防 0.01 网格伪影(thr=0.28 argmax 是 Δ0.03 噪声峰) |
| thr=0.27 数字 | pos_CER **0.7418** / neg_RR **0.9051** / 总分 **46.53**(CER腿10.33+RR腿36.20) | 线性估算 (1-CER)×40+RR×40 |
| 取 0.27 而非 0.28 | 总分差 0.03(噪声内, ~15 样本翻转) | 0.27 在 pos 侧严格占优(pos_CER 0.7418<0.7496 / pos_correct 0.3013>0.294) + pos 单边下移场景 oracle_thr 恰为 0.27(抗 B 集 babble 加重) |
| 分 thr oracle | 52.99(pos thr=0 + neg thr=0.45 RR 0.9916) | A 集上界(用了 pos/neg label, B 集不可达) |
| **损失** | **6.46 分** | B 集必须统一 thr(无 label)的代价(原估 6.18 低估: neg 真 oracle 是 0.45 非 0.4) |

### 方法(纯后置拒识, 无需重跑推理)

pos_result(thr=0 全转写, 每条 max_sim+text 都在)+neg_result(有 max_sim)。扫任意 thr: pos `max_sim<thr→CER=1.0`(FAQ Q1 pos 拒=CER1.0), 否则 cer(text,ref); neg `max_sim<thr→正确拒`(neg 只算 RR, 不需 text)。后置模拟与 submit_infer `--always-generate` 前置拒识对 CER/RR 等价(已核实 enroll_infer:249 always-generate 恒真不跳转写)。

### 核心张力(决定 thr 选择的根因)

```
pos sim: p25=0.179 med=0.283 p75=0.380   ← pos sim<0.4 占 79%(升 thr 误拒他们)
neg sim: p25=0.054 med=0.116 p75=0.189   ← neg sim≥0.4 仅 1.5%(降 thr 漏拒少)
```
pos/neg sim 在 [0,0.4] **严重重叠** → 统一 thr 在重叠区做取舍: thr=0.28≈pos sim 中位数 → 拒掉约 47-50% pos 换 RR 腿。这是 pos/neg sim 分布结构性重叠决定的, 纯 thr 调参破不了(进一步提 RR 需灰区二级信号, 见 follow-up)。

### 稳健性证据(支持 thr=0.27 提交)

1. **bootstrap CI**(B=400 重采样选 thr): IQR=[0.27,0.28], 被选 thr 80% 落 [0.27,0.34] → thr 选点方差可控, 非过拟合尖峰
2. **真压力**(固定 thr\*=0.27, 替代退化的对称平移): sim 收缩 α=0.8 损失 1.20 / sim 扩张 α=1.2 损失 2.59(最大风险) / neg 重尾 5% 损失 0.03 / pos 单边下移 0.10 损失 0.00 → 固定 0.27 在形状变化下损失 <2.6 分
3. **双口径收敛**: overall_CER 与 correct_rate 两种 CER 口径独立都选 thr≈0.28(与 0.27 不可分) → 不依赖主办方最终 CER 口径
4. **跨噪声类型一致**: babble(n=1218) 0.28 / pink(n=82) 0.27 / white(n=64) 0.27 → 子群体一致(小样本仅看方向)

### 诊断(透明度)

- pos cer_text>1(babble 重复循环幻觉使 hyp 超长)占 **9.0%(123 条)** → 决定 overall_CER 口径最优 thr 漂移(若官方 per-sample CER 封顶 min(·,1.0), 最优 thr 下移至 0.20-0.25, 待主办方确认)
- pos/neg manifest 丢弃 0(1364/474 全匹配, 无静默丢失)
- thr=0.27 处 neg 漏拒 45/474(9.5%, label=accept 进提交); ⚠️ neg_result 在 thr=0.4 生成, [0.27,0.4) 段漏拒文本已置空, 漏拒转写内容质量无法评估(若主办方有漏拒严重度惩罚需重跑 neg≤0.27)

### 对抗验证(5-agent fan-out + 综合, 修正初版 thr=0.28)

初版推 thr=0.28, 经口径/稳健/方法论/基线/决策风险 5 维度独立审查发现:

| severity | 问题 | 处置 |
|---|---|---|
| 🔴 critical | submit_infer:201 保底守卫 `sim_thr<0.35→abort` 拦 thr=0.28, run_baodi 不 export BAODI_OK → 推荐策略跑不起来 | ✅ 已修: run_baodi.sh export BAODI_OK=1(opt-in)+守卫报错引导 B 模式 |
| 🔴 critical | 无真实"统一 thr 混合提交"产物(现有 out_pos/out_neg 是 split-thr 用了 label=oracle 上界, B 集禁用) | ✅ 已修路径: run_baodi.sh 加 B\|mixed 模式(混合 pairs 无 ref, 统一 thr); ⚠️ 端到端待 B 集到手跑通 |
| 🔴 critical | caliber-A 假设(pos 拒=CER1.0)未主办方坐实, thr=0.28≈pos sim 中位数拒掉~47%目标, 全部价值压此一条 | ⚠️ follow-up: 书面确认 pos 被拒 CER 计法(1.0?额外惩罚?必须转写?)+预生成 thr=0 fallback |
| 🟡 major | thr=0.28 是 0.01 网格伪影(与 0.27 Δ0.03 噪声内) | ✅ 已修: 报区间+bootstrap CI+0.005 细扫(真峰 0.275), 推荐 0.27 |
| 🟡 major | 无 bootstrap/held-out, thr 在全集选又在全集验证=in-sample 偏差 | ✅ 已修: bootstrap B=400 CI |
| 🟡 major | 压力测试数学退化(对称平移≡thr 反向移, 测不出独立泛化) | ✅ 已修: 真压力(方差缩放 α+neg 重尾注入+pos 单边下移) |
| 🟡 major | 权重比敏感(40:40 假设; RR-heavy 20:60→最优 thr 0.40 反超) | ⚠️ 标注假设边界, 待主办方口径 |
| 🟡 major | 效率腿剔除仅 --always-generate 下成立; 去 flag 后高 thr 可降 RTF | ⚠️ 标注假设; follow-up 可测去 flag RTF |
| 🟡 major | 灰区[0.2,0.4]选择性 LLM 未探索(全局 --no-llm 切掉), 可能挽回 2.87 RR 腿 | ⚠️ follow-up A/B 实验(可选高价值) |

### 提交用法

```bash
# B 集统一 thr=0.27(B 集到手后, make_pairs 产无 ref 混合 manifest)
BAODI_PAIRS=code/B_pairs_datasetB.json bash code/run_baodi.sh B 0.27
# 或默认路径: bash code/run_baodi.sh B   # thr 默认 0.27

# thr 待主办方口径定: RR-heavy→bash code/run_baodi.sh B 0.40 / pos不许拒→B 0
# A 集初评仍分 thr(oracle 上界): bash code/run_baodi.sh pos 0.4 ; bash code/run_baodi.sh neg 0.45
```

### follow-up must_fix(提交 B 集前)

1. 🔴 **闭环主办方口径**(零成本高杠杆): 书面确认 (a) pos 被拒 CER 计法(1.0?额外惩罚?必须转写?) (b) CER→分排名还是归一化 (c) CER:RR 权重比是否 40:40 (d) per-sample CER 是否封顶 min(·,1.0)。未确认前预生成 thr=0 fallback(pos 全转写, 口径C contingency)
2. 🔴 **B 集混合提交端到端跑通**: B 集到手 → make_pairs 产无 ref 混合 manifest(pos/neg 不作输入, utt_id 不冲突) → run_baodi B 0.27 → to_submission(cer 空, label 由 thr, final_cer 主办方算) → 自检整份用同一 thr
3. 🟡 **灰区选择性 LLM A/B**(可选高价值): 对 max_sim∈[0.2,0.4](~30% pos+5% neg) 跑 LLM 二次校验, 测能否救回 2.87 RR 腿(RTF 预估 0.24→0.35-0.45 仍<1.0)。即使不用也应答辩前测过

### 产物

- `code/scan_unified_thr.py`(v2: bootstrap CI+0.005 细扫+诊断+真压力+修 split_oracle) + `code/scan_unified_thr.json`
- `code/run_baodi.sh`(加 B\|mixed 模式 + export BAODI_OK=1) + `code/submit_infer.py`(守卫报错引导 B)
- spec `docs/superpowers/specs/2026-07-07-unified-thr-selection-design.md`(待补) + memory `unified-thr-decision`
- 对抗验证 workflow journal: `subagents/workflows/wf_5211d738-83b/`(5 agent findings + 综合)

---

## T28 — 2026-07-11 Qwen3-ASR 中文原生后端(候选2 全量证实, CER 腿+10分, 首个CER收益方向)

> 19路前沿探索(`docs/前沿探索报告_2026-07-10.md`)证伪 faster-whisper/BoH(零风险动作实测不成立)后, 候选2 中文原生 ASR 经全量数据证实为**首个真正 CER 收益方向**。Qwen3-ASR-1.7B drop-in 替换 vanilla 后端(复用 diar+wespeaker 切 target timeline, 不改 diar, zero-training, Apache2.0)。

### 全量数据(1350条 pos target 切片, 官方口径累计池 total_err/total_char, 与 vanilla 0.595 同口径)

| 桶 | n | Qwen3-ASR CER | vanilla CER | Δ |
|---|---|---|---|---|
| **overall** | 1350 | **0.3436** | 0.6635(未归一)/0.595(官方) | **−0.32** |
| 死区 <0.2 | 396 | 0.459 | 0.828 | −0.37 |
| 主战场 [0.2,0.4) | 668 | 0.360 | 0.718 | −0.36 |
| 接近解决 ≥0.4 | 286 | 0.182 | 0.375 | −0.19 |

- 探针60条主战场桶: CER 0.146 vs vanilla 0.454(Δ-0.31) → 全量证实
- Qwen 更优 55%(746/1350), RTF 0.289s/条(4060, L20 待测)
- **CER 40%腿(双口径, 2026-07-11 P0 核实坐实)**:
  - transcribe 不拒口径(pos 全转写, 诊断/能力上限): vanilla 16.2 → Qwen3-ASR **26.3** 分(**+10.1**)
  - **含拒 thr0.27 提交口径(排名公式实际用, pos 允许拒 2026-07-08 确认)**: vanilla 11.97 → Qwen3-ASR **16.26** 分(**+4.29**) ← 答辩/提交须报此口径, 勿用 transcribe 虚高
  - 详见下"双口径核实"子段 + `code/qwen_official_cer_workpoints.json`

### 关键发现

1. **ExtremeNoise 4× 鲁棒性真迁移**: Qwen3-ASR HF 官方 ExtremeNoise WER 16.17 vs Whisper-large-v3 63.17(≈4×), 本项目 babble 切片全桶大幅优(Δ-0.32)
2. **死区挑战"物理地板"**: 死区 Qwen3-ASR 0.459 比 MiMo 0.554 还强(Δ-0.37 vs vanilla 0.828) → 更强转写器能多救回一截; ⚠️ 0.459 仍不及格, 需对抗验证(别急着改"物理地板"答辩叙事)
3. **机制**: diar+wespeaker 切 target timeline → Qwen3-ASR 转写切片(language="Chinese") → to_simplified+digit_postproc 提交归一。drop-in, 不改 diar, zero-training

### ⚠️ 双口径核实（2026-07-11 P0 收尾, 修正 +10.1 水分 + 补 0.3436 不可复现隐患）

**缘起**: 7-agent 路线核实 workflow 发现 CLAUDE.md/handoff/RESULTS 头条「+10.1」是基于【不含拒 transcribe CER】0.3436 算的, 而提交进排名公式用【含拒 overall】(pos 允许拒 2026-07-08 确认); 且 0.3436 此前未入库(poc json 仅 per-sample 均值 0.3848, 非官方池)。`code/recompute_qwen_official.py` 独立坐实(同集合 1350 条公平对比):

| 口径 | qwen | vanilla | CER 腿 Δ | 用途 |
|---|---|---|---|---|
| transcribe 不拒(pos 全转写) | 0.3436 | 0.5954 | **+10.07** | 诊断/能力上限 |
| **含拒 thr0.27(提交)** | **0.5934** | 0.7007 | **+4.29** | **排名公式用** |

含拒 thr 扫描(官方累计池): thr0.20 qwen0.4912/vanilla0.6544 | **0.27 qwen0.5934/vanilla0.7007** | 0.30 qwen0.6435 | 0.35 qwen0.7221 | 0.40 qwen0.7993。全档 qwen 优于 vanilla, 低 thr 优势更大(-0.1632@0.20 → -0.0168@0.45)。

**归一零效应坐实**: 1350 条 qwen 输出 0 阿拉伯数字 0 真繁体(原生中文数字"二十五度"), digit_postproc/to_simplified 均 no-op, raw==归一==0.3436 逐位相等。对照 vanilla 150/1350 含阿拉伯数字, 归一收益 -0.033。提交侧 enroll_infer:384 已接归一, 无 cn2an/zhconv 式漏洞。

**死区 0.459 坐实(官方池)**: 死区[0,0.2) n=396 qwen 0.459 vs vanilla 0.784(Δ-0.325)。0.459 是官方累计池口径(per-row 均值 0.499 为另一口径), 挑战 spk-oracle-poc "物理地板"叙事(qwen 0.459 < oracle 0.607 完美选 target), 待 A2 对抗验证(最大威胁=中文家居 LM 先验幻觉)。

**提交数字(thr0.27, w1=w2=0.4)**: qwen CER腿16.26+RR腿36.20=52.46 | vanilla CER腿11.97+RR腿36.20=48.17 | Δ+4.29(效率腿20待L20)。neg RR 0.9051 与转写器无关(qwen 不转写 neg)。

### 🔬 死区对抗验证 A2（2026-07-11, 用户听音坐实 H1, 修正"物理地板"归因）

死区 sim<0.2(n=396) qwen 官方池 0.459 vs vanilla 0.784(Δ-0.33) 经听音对抗验证(`code/analyze_dead_zone_qwen.py`)，**不是 LM 幻觉红利，是真实转写突破(H1)**：

- **完美句 154/396=38.9%**(vs vanilla 10.9%, 3.6×), qwen vs vanilla win219/loss38/tie139。
- **听音坐实 H1**: 用户确认 cmd_2091(sim0.092 "儿童要少吃什么")+cmd_2137(sim**0.004** "打开睡眠模式") 音频清晰、qwen 听对 → 真实听音非 LM 幻觉。
- **关键反转**: cmd_2137 sim0.004(wespeaker 声纹近乎随机)但音频清晰 → **sim 是声纹代理 ≠ 音频质量**。
- **A 类(真摧毁+LM幻觉H2, 少数)**: 翻车条 cmd_2808"风速五十"→"邮政银行被打出个一比五"、cmd_2488"把空调关上"→"点一首刘德华的《冰雨》"(编造)。
- **归因修正**: spk-oracle-poc"死区物理地板不可破" → **vanilla 转写器 OOD 伪地板**(oracle 0.607/单spk 0.436 全程 vanilla 评估); qwen 凭 ExtremeNoise 训练突破到 0.459。
- **路线影响(2026-07-11 CAM++ POC 证伪, 声纹强化原理性关闭)**: A2 听音后疑声纹强化重开, 跑 CAM++ 真 POC(code/exp_spk_campp_deadzone.py, 396 死区条)。CAM++ 死区 sim 均值 0.39(vs wespeaker 0.13)但 **B类(转对)0.373 vs A类(翻车)0.374, B-A margin=-0.000**——两声纹器都无 B/A 区分力。**原理性: 声纹 emb 编码"who"不编码"audio clarity", B/A 区分在 mel 层声纹层看不到 → 任何声纹器都救不了 B 类(救 B 连 A 一起误放)**。A2 H1 突破成立但只能靠转写器。副产品: 支持"转写器置信度拒识>声纹 sim 拒识"(BLEMU/FA 置信度)。声纹强化方向关闭(七连受挫最新)。

### 集成(submit_infer --asr-backend qwen, drop-in 落地)

- enroll_infer 加 `--asr-backend qwen` 分支(切片存盘+text空→末尾 subprocess 调 code/qwen_asr_backend.py[code/.venv_qwen, venv隔离]批量转写→填transcript+提交归一)
- submit_infer choices 加 qwen(透传机制 line 158-159 已有)
- code/.venv_qwen: qwen-asr(transformers backend) + torch2.6+cu124 + Qwen3-ASR-1.7B 权重 E:/hf_cache/Qwen3-ASR-1.7B
- 验证: 5条 transcript 填充(Qwen3-ASR 识"制热""权志龙"等 vanilla 错的)
- code/.venv speechbrain lazy 修复: patch inspect.getmodule 固化 enroll_infer 顶部(解锁后续所有切片路线)

### 诚实边界

- Qwen3-ASR transcribe 0.344 是**纯转写诊断口径**(pos 全转写); 提交含拒 thr0.27=**0.5934**(CER腿+4.29, 已测见"双口径核实"); 归一零效应(0 阿拉伯数字 0 繁体, raw==归一==0.3436)
- RTF 0.289 慢于 vanilla 0.16-0.24, 效率腿时间分可能小失分(-1~2); L20 待测
- 55% 更优(45% 持平/更差, 如 cmd_66 抽油烟机/cmd_98 权志龙 Qwen 反劣)

### follow-up

1. ✅ submit_infer qwen 全流程 run-twice 验证(2026-07-11 完成: verify_reproducibility --backend qwen limit=10, **text 一致率 100%, CER delta=0**, 与 vanilla 对齐; 改 verify:47 choices 加 qwen + qwen_asr_backend 加 --seed 内联 + enroll_infer:377 透传)
2. 🟡 L20 RTF 真测(租 AutoDL L40)
3. 🟡 FireRedASR 横评定选型(FireRedASR 干净 CER 2.89% 略优 + RTF 0.087 更快)
4. ✅ Qwen3-ASR 提交归一后 overall + thr 含拒 overall(2026-07-11 P0 完成, 见"双口径核实"; 归一零效应, 含拒 thr0.27=0.5934, CER 腿真实 +4.29)
5. 🟡 死区 0.459 对抗验证

### 产物

- `code/recompute_qwen_official.py` + `code/qwen_official_cer_workpoints.json`(2026-07-11 P0: 官方口径坐实 transcribe 0.3436 + 含拒 thr 工作点 + 双口径诚实标注, 修 +10.1 水分)
- `code/poc_qwen_asr.py` + `code/poc_qwen_asr_full_result.json`(全量1350条逐条)
- `code/qwen_asr_backend.py`(code/.venv_qwen 批量转写)
- `code/enroll_infer.py`(--asr-backend qwen 分支 + patch inspect.getmodule 固化 + --save-target-audio)
- `code/submit_infer.py`(choices 加 qwen)
- E:/target_slices_full/(1350 切片, 外部 E 盘不入库)
- 报告 `docs/前沿探索报告_2026-07-10.md` + memory `cer-breakthrough-candidates`

---

