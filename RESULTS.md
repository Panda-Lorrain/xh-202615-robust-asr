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

### 关键发现（诚实）⚠️
1. **带噪 sim 普遍退化**：所有重叠/加噪条件 0.05–0.48，**全 <0.5 误拒**。阈值 0.5 在题目分布（−5~5dB）太严。
2. **噪声是退化主因**（非重叠）：ov000 无重叠但 snr-5 降到 0.205（vs 干净 0.816）；重叠是次要因素。
3. **多 speaker 选择偶反转**：高重叠 + 低 SNR 时两人 sim 接近/反转（ov025_snr-5、ov075_snr+0 选错）。
4. **ov100 死区**：两人 sim 相同（0.263），无法区分（与 T15 重叠诊断一致）。

### 改进方向（数据驱动，答辩素材）
- **CAM++ 引入有了数据理由**：wespeaker（VoxCeleb 英文训练）在中文+噪声退化严重 → CAM++（原生中英双语）可能更鲁棒。**修正此前"CAM++ 是沉没成本"判断为"带噪鲁棒性数据驱动备选"**。
- enrollment 加噪增强（同分布噪声） / 声纹多段质心融合 / 阈值分场景自适应 / frontend SE 增强（RASTAR/VSAEC）再抽声纹。

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
