# RR 提升方向 A — enrollment 污染检测自适应阈值 POC 设计

**日期**：2026-07-16
**状态**：设计已批准，待执行
**硬约束**：pos 目标 CER 主线（qwen 后端 0.5934）不能动；RR 必须在 Pareto 改进前提下提升

---

## 1. 背景与动机

- 当前提交配置（thr=0.27 / strategy=sim_only / 关 LLM）neg 拒识率 RR = **90.51%（429/474）**，**45 条漏拒**是唯一着力点。RR 与 ASR 后端无关（neg 在转写前已按声纹 thr 判拒）。
- 45 条漏拒画像（逐条实证，见调研）：
  - ~21 条：英文幻觉 / 新闻财经 / 乱码 → content_gate 规则可救
  - ~16 条：非指令碎片 → 需 LLM 语义才拒得掉
  - ~7-8 条 **TRAP**：文本本身就是合法指令（"把空调调低为18度"），任何内容/语义手段都救不了 → 物理地板
- 两个硬约束排除了大部分方向：
  1. 内容类方法（content_gate / LLM）在 **qwen 后端恶化**（gate Δ+0.024，已排除）；仅在 vanilla 后端是 Pareto，但 vanilla pos CER 0.666 > qwen 0.5934，合计分反降。
  2. 调 thr 是纯拉锯：sim 重叠带 [0.27,0.40) 内 neg 38 条 vs pos 433 条（**11:1**），每多拒 1 条 neg 陪葬 ~11 条 pos，无 free lunch。
- **收敛结论**：qwen 主线提 RR 只能靠**说话人信号类方向**。方向 A（enrollment 污染检测→污染项自适应 thr）是最高性价比候选，但收益未经验证 → 需本 POC。

## 2. 核心假设（POC 要证实/证伪）

> 45 条漏拒 neg 中相当一部分，其 **enrollment 音频被污染**（target + 非目标人同在）。污染的非目标人声纹被一起抽进 enrollment embedding，导致 recognition 中该非目标人的语音 sim 偏高（撞像 enrollment），sim_only 在 thr=0.27 判不过 → 漏拒。

**证据线索**：
- 45 漏拒的 max_sim 极值（0.516 / 0.537 / 0.668）远高于灰区主体 [0.27,0.40)，极似"非目标人声纹与 enrollment 撞像"。
- annotation_spec_v2 的 A-E1 根因（enrollment 污染）已人工坐实存在（cmd_2081 等），calibration_samples_v2.csv 的 `enw_可靠性` 字段亦有污染标注。

**假设成立的推论**：对污染 enrollment 的识别单独用更高 thr → 多拒这些高 sim neg；干净 enrollment 保持 0.27，其 pos 零影响 → Pareto 改进。

## 3. 已核实的设计约束

| 约束 | 事实（已核实） | 对 POC 的影响 |
|---|---|---|
| diar 覆盖范围 | `enroll_infer.py` 的 diar 只对 recognition 跑（`diar(rec)` L227），enrollment 仅抽单条 embedding（`diar._embedding` L175） | POC 须新写 enrollment 污染检测逻辑（不能直接复用现有 diar） |
| 校准真值 | `calibration_samples_v2.csv` 的 `enw_可靠性` 字段已人工听音（含污染/过短/干净判断），**26 条** | 作 Q1 检测器校准真值；样本偏小，F1 为初步估计 |
| 全量标签 | 标注 v2 全量未回收（spec §四"回收后"才能给 prevalence） | 无现成逐条污染标签，POC 自跑检测 |

## 4. POC 架构（3 子问题数据流）

### Q1：检测手段可行性
- **输入**：26 条 calibration 样本的 enrollment 音频（kws_X.wav）+ 其 `enw_可靠性` 真值。二值化：**污染=正、干净=负**。**过短（enrollment <1s）属另一失效模式，从 Q1 污染二分类校准集剔除并单独记录占比**（方向 A 只针对污染，不针对过短；过短归 A-E2，本 POC 不处理）
- **候选检测器**：
  - **D1 DiariZen diar**：复用 enroll_infer 的 `DiariZenPipeline`（L164），输入换成 enrollment 音频；`speakers ≥ 2` → 判污染
  - **D2 wespeaker 帧级聚类**：enrollment 切帧（如 0.5s 滑窗）→ 每帧 `diar._embedding`（L175）抽 emb → 余弦距离层次聚类 → 簇数 ≥ 2 → 判污染
- **输出**：D1 / D2 各自的 precision / recall / F1（vs 真值），以及各自对"过短"样本的混淆情况
- **Go 标准**：至少一种 **F1 > 0.7**；都不行 → 检测手段 No-Go

### Q2：污染与漏拒相关性
- **输入**：474 neg 的 enrollment 音频 + 各条 max_sim（`out_neg_vanilla_full/result.json`）
- **方法**：用 Q1 选定检测器对 474 neg enrollment 全跑，标污染。分组算污染率：
  - 组A = 45 条漏拒（max_sim ≥ 0.27）
  - 组B = 全 474 基线
- **输出**：组A 污染率 vs 组B 污染率 + 显著性（Fisher exact test）
- **Go 标准**：组A 污染率 **≥ 1.5× 组B 且 Fisher exact p < 0.05**（Go 下限）；**≥ 2×** 为强相关（高信心 Go）；**组A ≤ 1.2× 组B** → 弱相关 No-Go，转方向 B

### Q3：自适应 thr 的 RR / pos 代价
- **输入**：1364 pos 的 enrollment 音频 + 各条 max_sim + qwen 转写结果（`poc_qwen_asr_full_result.json`）
- **方法**：
  1. 对 pos enrollment 跑污染检测
  2. hold-out 分 train/val（8:2，**按 enrollment id 分组**防泄漏；memory lessons 强制要求）
  3. 模拟：污染 enr 的识别用 thr_high ∈ {0.30, 0.35, 0.40}，干净 enr 用 0.27（基线）
  4. 算 val 上：**ΔRR**（neg 侧多拒条数 → RR 提升 pp）+ **Δpos CER**（pos 侧污染 enr 被误拒导致的 CER 变化，含拒口径）
- **Go 标准**：存在 thr_high 使 **ΔRR > 0 且 Δpos CER ≤ 0.01**（含拒口径，hold-out val；远小于 memory 记载的 ±0.04 CER 噪声），或 ranking 分 w1·(1-CER)+w2·RR 不降 → Go + 给推荐 thr_high

## 5. 产物

- `code/poc_enrollment_pollution.py`：D1/D2 检测器实现 + 3 子问题执行 + 数据表输出（可复用 enroll_infer 的模型加载）
- 结果文档 `docs/POC_A_enrollment_pollution_结果_2026-07-16.md`：3 子问题数据表 + Go/No-Go 判断 + 推荐 thr_high + 预期 RR 提升 pp + pos 代价（hold-out val）

## 6. 风险与备案

| 风险 | 备案 |
|---|---|
| DiariZen 在 1.8s enrollment 不可靠（超短音频 diar 不稳） | 改 D2 wespeaker 帧聚类；两者都不行 → Q1 No-Go |
| 26 条校准里污染样本 < 5 → F1 不可靠 | 补听音到 30 条（扩 calibration 样本，+0.5 天） |
| Q2 弱相关（污染与漏拒无关） | Q2 即 No-Go，转方向 B（FA 置信度二次拒） |
| Q3 一提 thr_high，pos CER 就崩 | 说明污染 enr 的 pos 与 neg sim 高度重叠 → 方向 A 受限，转 B/C |
| hold-out 过拟合 A 集 | 按 enrollment id 分组 + 只报 val 指标 |

## 7. 工程量与时间

- **1–1.5 天**：写 D1/D2 检测器 + 跑 26 校准（Q1）+ 跑 474 neg（Q2）+ 跑 1364 pos + hold-out 模拟（Q3）+ 写结果文档
- 若需补听音校准：+0.5 天

## 8. 诚实声明

这是验证实验，**可能 No-Go**。即使 No-Go：
- 排除了"enrollment 污染是漏拒主因"假设，避免后续白做工
- Q1 的 enrollment 检测器可直接复用于方向 C（声纹多信号融合）

## 9. 后续路径（POC 完成后）

- **Go**：写方向 A 落地 spec（enrollment 污染检测集成进 enroll_infer，自适应 thr 逻辑接入 submit_infer 的 decide_reject），转 writing-plans 实施
- **部分 Go**（Q2 相关但 Q3 代价偏大）：收窄 thr_high 范围或结合方向 B
- **No-Go**：转方向 B（FA 置信度）POC

## 10. 数据来源（文件）

- neg 结果：`code/out_neg_vanilla_full/result.json`（474 条 max_sim）
- pos 结果：`code/poc_qwen_asr_full_result.json`（pos 逐条 sim + qwen 转写）
- enrollment/识别音频路径：`code/neg_pairs_datasetA.json` / pos pairs（`make_pairs_from_datasetA` 产出）
- 校准真值：`code/annot_pack/calibration_samples_v2.csv`（`enw_可靠性` 字段）
- diar / embedding 复用：`code/enroll_infer.py`（DiariZenPipeline L164，`diar._embedding` L175）

## 11. 验证方法（testing）

- Q1：检测器输出 vs 人听真值的 F1（标准分类指标）
- Q2：组间污染率 Fisher exact 显著性（p < 0.05 视为相关）
- Q3：hold-out val 上的 ΔRR / Δpos CER（不报 train，防过拟合）
- 全程数据可复现：固定随机种子（enrollment id 分组用固定 seed），结果文档附原始计数
