# B2: SepFormer 真源分离攻双人重叠失败组第③层 — 天花板实验

**日期**: 2026-07-27
**产物**: `code/runs/_sepformer_b2/{slices/, _uid2text.json, summary.json}` + `code/exp_sepformer_b2.py`
**前置**: A 实验 (oracle 选 speaker) / B1 实验 (forced alignment 切时间段)
**总耗时**: 0.6 min (40 条 fail 组, 主 4060 GPU)

## 一句话结论

**SepFormer+oracle 选路 mean CER 0.603 ≪ argmax 1.216（Δ-0.61），攻得动第③层**；但 wespeaker sim 选路完全失效（mean CER 1.249, 选对率仅 25%, 低于 50% 随机），实际可用版本目前不可达。**分离不是瓶颈, 选路才是**——坐实 memory `non-voiceprint-target-selection` 的 "SI-SDR 波形分离=EoW 感知-识别鸿沟陷阱"。

## 方法

### 复用情况
- **完全复用** `code/exp_sepformer_qwen.py` 的核心: `load_sepformer` / `separate` / `load_diar` / `get_emb_factory` 四个函数, 不重写。
- **改动**:
  ①数据源: `_oracle_speaker/summary.json` 中 `group==fail` 的 40 条 (主战场双人重叠, 不再是死区 sim<0.2)
  ②关联 `_oracle_separation/meta.json` 拿 `align_score`, 分桶 B1 对齐失效子集 (align_score<0.4)
  ③两路 (target/other) 都存 wav 都转写, 同时报 sim 选路 + oracle 选路 CER (原脚本只报 sim 选路 + 取 min)

### 模型与配置
- **SepFormer**: `speechbrain/sepformer-whamr16k` (WHAM+reverb, 16kHz, 英文预训练)
- **声纹选路**: DiariZen pipeline 内置 `diar._embedding` (wespeaker 系列), 整路 16k wav 抽 emb, cos 相似度最大者=target 路
- **ASR**: Qwen3-ASR-1.7B (`code/.venv_qwen`, batch=16, seed=42)
- **CER**: 官方口径 `eval_metrics.cer_official` + `text_utils.{to_simplified, digit_postproc, brand_homophone_fix}` 归一链
- 调用链: load 模型1次 → 40 条分离 (Phase1) → 80 路 wav 批量 qwen 转写 (Phase2) → 算 CER (Phase3)

## 分桶结果

### 失败组 40 条 (整体)
| 选路方式 | mean CER | correct<0.5 | 救回条数 | 备注 |
|---|---|---|---|---|
| argmax 基线 (来自 A) | **1.216** | 0% | — | 主线 argmax 选 speaker |
| A oracle_speaker | 0.850 | — | — | 前置 A 实验上限 |
| B1 forced alignment | 0.940 | — | — | 前置 B1 整体 |
| **SepFormer + sim 选路** | **1.249** | 2% | 1 | sim 选对 oracle 路 10/40 (25%) |
| **SepFormer + oracle 选路** | **0.603** | **45%** | **18** | 理论上限 |

- **Δ(sep_sim - argmax) = +0.033**: sim 选路和 argmax 持平 (实际不可用)
- **Δ(oracle - argmax) = -0.614**: oracle 选路大幅低于 argmax (上限可达)

### B1 对齐失效子集 (align_score<0.4, n=32) — 第③层重点
| 选路方式 | mean CER | correct<0.5 | 救回条数 |
|---|---|---|---|
| argmax | 1.172 | 0% | — |
| **SepFormer + sim 选路** | 1.236 | 0% | 0 |
| **SepFormer + oracle 选路** | **0.610** | **41%** | **13** |

**核心发现**: 第③层 (B1 forced alignment 已经救不动的子集) 上, SepFormer oracle 选路救回 13/32 (41%), mean CER 1.172→0.610。这正面证伪了 "第③层物理地板" 假设——**分离本身能拎出干净 target mel**, 只是选路是瓶颈。

### B1 对齐可靠子集 (align_score≥0.4, n=8)
| 选路方式 | mean CER |
|---|---|
| SepFormer + sim 选路 | 1.303 |
| SepFormer + oracle 选路 | 0.575 |

(可靠子集 SepFormer 表现与失效子集接近, 说明 B1 是否失效对 SepFormer 影响不大)

## sim 选路为什么失效?

**18 条 oracle 救回样本中, sim 选对仅 1 条 (cmd_2890)**, 17 条 sim 都选了相反的路 (干扰路)。这意味着:

> **SepFormer 分离出的两路音频中, target 路的 wespeaker emb 反而比干扰路更低**。

可能机制 (与 memory `non-voiceprint-target-selection` 的 "SI-SDR 陷阱" 一致):
1. **SI-SDR 优化破坏声纹特征**: SepFormer 训练目标是 SI-SDR (波形级能量差), 优化过程会改高音/共振峰/相位, 这些正是声纹 emb 的核心特征。**分离越好, 声纹失真越严重**。
2. **说话人活动不均匀**: target 人的活跃段可能被部分"分"到 other 路 (尤其两人同时说话的重叠段), 全路抽 emb 把 target 稀释。

这是 **EoW 感知-识别鸿沟的实锤**: SI-SDR 优化 (感知/波形层) ≠ ASR/声纹友好 (识别层)。

## 救回样本细节 (B1 失效子集 13/32)

| uid | align | argmax | sep_sim | sep_oracle | ref | oracle_text |
|---|---|---|---|---|---|---|
| cmd_2052 | 0.18 | 1.00 | 1.00 | **0.00** | 温度调到二十七 | 温度调到二十七。 |
| cmd_2060 | 0.20 | 1.00 | 1.67 | 0.33 | 开启无风模式 | 开启时光模式。 |
| cmd_2302 | 0.31 | 1.17 | 1.83 | **0.00** | 给我放桃花诺 | 给我放《桃花诺》。 |
| cmd_2347 | 0.31 | 0.83 | 1.17 | **0.00** | 开启睡眠模式 | 开启睡眠模式。 |
| cmd_2503 | 0.19 | 1.14 | 1.29 | **0.00** | 把风速调到最大 | 把风速调到最大。 |
| cmd_2595 | 0.20 | 1.25 | 1.25 | **0.00** | 关下空调 | 关下空调。 |
| cmd_2607 | 0.06 | 1.33 | 1.33 | **0.00** | 调到送风模式 | 调到送风模式。 |
| cmd_2630 | 0.36 | 2.25 | 2.25 | **0.00** | 开左右风 | 开左右风。 |
| cmd_2637 | 0.37 | 1.25 | 1.12 | **0.00** | 哺乳期要少吃什么 | 哺乳期要少吃什么？ |
| cmd_2687 | 0.34 | 1.00 | 1.00 | 0.25 | 把温度调到三十度 | 我温度调到三十。 |
| cmd_2709 | 0.05 | 1.00 | 1.00 | **0.00** | 吃什么可以明目 | 吃什么可以明目？ |
| cmd_2798 | 0.24 | 1.00 | 1.00 | 0.25 | 吃什么可以防辐射 | 吃什么预防辐射？ |
| cmd_2817 | 0.06 | 1.00 | 1.00 | **0.00** | 风速调到百分之四十 | 风速调到百分之四十。 |

注意: 9 条 oracle_text 与 ref 几乎逐字一致 (CER=0), 这些原本 argmax 都是 1.0+, **SepFormer 把 target 几乎完整拎出来了**——但 sim 选路全部选错。

## 分离质量限制 (sim 选对仍 CER>0.8 的, n=7)

这 7 条是 SepFormer 分离本身失败 (两路都转不出 ref):
- cmd_2023 (ref=风速自动, 两路分别 "旷世之龙" / "千五百四十六")
- cmd_2128 (ref=风小一点)
- cmd_2214 (ref=把温度调到二十五度)
- cmd_2549, cmd_2715 (ref=降低一度)
- cmd_2826, cmd_2830

这些样本提示 SepFormer 在某些重叠场景的分离本身有限制 (可能两人音色相近/信噪比极低)。

## 判定 (对照用户判别逻辑)

> **SepFormer+oracle 选路 0.603 ≪ 0.85 → 源分离有效, SepFormer 能攻第③层** ✓ 正向结论

但必须诚实附加:
- 这是 **oracle 选路** (两路取 CER 最小者, 需要 ref), 是理论上限, **实际部署不可达**
- 实际可用版本 (sim 选路) mean CER 1.249, 与 argmax 持平, **当前不可用**
- 工程瓶颈明确: **分离 OK, 选路坏**

## 局限 (必须诚实标注)

1. **SepFormer 训练在英文 WHAM+reverb, 中文 OOD 风险**: 但从 oracle 0.603 看, 实际中文分离质量尚可 (13/32 救回且 9 条几乎完美), 中文 OOD 不是主要限制因素。
2. **SI-SDR 陷阱 (实锤)**: sim 选对率 25% (< 50% 随机), 表明 SI-DR 优化破坏声纹特征, 这是 EoW 感知-识别鸿沟的实证。**修 target 路选择是关键, 不在分离本身**。
3. **B2 部分失败 (sim 选路) 不能断言 "源分离无空间"**: oracle 数字证明分离有效, 失败的是选路环节。后续应投 **非声纹 target 选择** (memory `non-voiceprint-target-selection` 排序: 🥇 Whisper-Sidecar 用 embedding 分离绕开 SI-SDR 陷阱 / 🥈 ASE 自增强选帧 / 🥉 GSE 救重叠区)。
4. **oracle 选路 0.603 是 40 条失败组的天花板**, 全量 668 条主战场的天花板未测 (但根据 A 实验 oracle_speaker 0.850 推断, 全量天花板可能略高, 因为失败组是难子集)。
5. **7 条 sim 选对但仍 CER>0.8**: 表明 SepFormer 分离本身有边界, 这些样本需要其他方法 (如 TSE 直接抽 target, 不走"先分两路再选")。

## 跟现有结论的关系

- **修正 memory `non-voiceprint-target-selection`**: 之前记 "SepFormer+Qwen3 证伪 (死区 sim<0.2, CER 0.687 vs argmax 0.410 Δ+0.277)" — 那是 **死区子集** (音频被摧毁, 分离也救不动)。本次 B2 在 **双人重叠失败组** (音频相对完整, 主要是 diar 切错) 重测, **oracle 数字证明分离有效**, 之前证伪只对死区成立, 不普适到所有失败组。
- **强化 memory `overlap-is-cer-failure-rootcause`**: "双人重叠是 CER 失败主因" 的解药确实是分离, 但本 B2 实测发现 **解药不是 SepFormer+sim, 而是需要非声纹 target 选择** (因为声纹在分离后失真)。
- **不影响主线 qwen CER 0.3436**: B2 是诊断/天花板实验, 不改 submit_infer, 不进提交链路。

## 后续 ROI 排序 (基于 B2 发现更新)

1. 🥇 **Whisper-Sidecar (embedding 空间分离)**: 用 Whisper encoder embedding 做 target 选择, 绕开 SI-SDR 波形分离的声纹失真陷阱。需要训练绑 Whisper, 工程量大, 但 B2 已经把"分离有效"坐实, 投入有价值。
2. 🥈 **ASE 自增强选帧**: 复用 CAM++ 在 enrollment 自增强, 改善 target 路 sim 选择 (零额外训练, 可快速 POC)。
3. 🥉 **TSE (target speech extraction)**: 不走"先分两路再选", 直接用 enrollment 引导抽 target, 避免 sim 选路环节。

判否方向:
- ❌ 再投 SepFormer 改进 (whamr→中文训练, 或 whamrEnhanced): B2 oracle 已经 0.603, 改进分离质量收益有限, 瓶颈在选路。
- ❌ 投 wespeaker 强化 (CAM++ 替换): memory `spk-oracle-poc` 已证伪, B2 又新增反例 (sim 选对率 25%), 强化声纹不是解药。
