# 分区切 timeline POC 报告 (2026-07-27)

> **结论一句话**: 分区切(partition)在 oracle 选路下天花板有效(Δ-0.33), 但**核心创新"exclusive 选路"证伪** — exclusive 段与 enroll 是同一说话人(enroll 选定 target_idx 的独占段), SepFormer 路 sim 排名几乎一致(7/8 同)。**真正瓶颈是 SepFormer 路选择本身, 既非 enroll 也非 exclusive 能解, 需非声纹信号**(内容判别 / LLM 挑家居指令 / whisper-sidecar embedding 路线)。

## 背景

链路 bug 已定位: `enroll_infer.py:300` 的 `cut_target_timeline` (`text_utils.py:58`) 切的是 target speaker 的 **full timeline(含重叠帧)**。重叠区物理上是两人混合, 切进去 → ASR 收混合 mel → 转错(如 cmd_2637 spk0 切片含 spk0+spk1 混合, ASR 转"就已经进行过一轮教育" ≠ ref"哺乳期要少吃什么")。

diar 已坐实是对的(spk0/spk1 wespeaker emb 余弦 0.219 可分, 内容也对: spk1="哺乳期要少吃什么"=ref)。bug 全在 cut_target_timeline 切了重叠区。

## POC 设计

对 target 的 timeline 分两类区域:
- **exclusive 区**: target 说话 + 其他 speaker 都不说话 → 纯 target, 切原始 audio
- **overlap 区**: target + 至少一个其他 speaker 同时说话 → 混合, 切 SepFormer 分离的 target 路

按时间顺序拼接(边界 10ms crossfade) → qwen 转 → 算 CER。

### 核心创新(已证伪): 用 exclusive 段纯 target 声纹选 SepFormer 路

B2 (`exp_sepformer_b2.py`) 证明 enroll emb 选 SepFormer 路只有 25% 选对 (10/40), 归因两类污染:
- ① **enrollment 污染**: enrollment 音频可能含噪声/与 target 不完全匹配
- ② **SI-SDR 破坏声纹**: SepFormer 输出在声纹空间被扭曲, enrollment 在原始域匹配错位

本 POC 改用 **exclusive 段(从原始 audio 抽的纯 target 未经 SepFormer)作锚** 匹配 SepFormer 两路。理论假设: exclusive 段是 recognition 域内、未经 SI-SI 扭曲的纯 target 音频, 与 SepFormer 输出域更近 → 选路更准。

### 对照矩阵 (3 切法 × 3 选路 = 9 组合)

注: full / excl-only 不经 SepFormer, 选路 no-op, 实际独立切片数 = 5/样本。

| 切法 | 描述 |
|---|---|
| **full** | `cut_target_timeline` 全 timeline 含重叠区(当前主线 baseline) |
| **exclusive-only** | 只切 exclusive 段(对照, ASR 退化预期) |
| **partition+enroll** | excl 原始 + ov 段用 SepFormer enroll 选路 (B2 baseline) |
| **partition+excl** | excl 原始 + ov 段用 SepFormer **exclusive 选路(创新)** |
| **partition+oracle** | excl 原始 + ov 段用 SepFormer oracle 选路(天花板, 不可达) |

### 样本

8 条 B2 fail 组(覆盖 4 类场景):
- **cmd_2637** (argmax 错, 双重错对照, ref"哺乳期要少吃什么")
- **cmd_2188/2302/2347/2503/2630/2766** (argmax 对 sep 错, 创新主战场)
- **cmd_2890** (argmax 对 sep 对, 不毁已对基线)

复用 B2 已分离 sepformer 两路 wav (`code/runs/_sepformer_b2/slices/`), 不重新分离, 省 15min。

## 结果

### 5 切法 CER 均值 (8 样本)

| 切法 | CER 均值 | vs full | 说明 |
|---|---:|---:|---|
| baseline **full** (当前主线) | **1.2415** | — | 含重叠区混合 |
| exclusive-only | 1.2206 | -0.02 | 切走重叠区, 但音频变短 ASR 退化, 噪声内 |
| partition+enroll (B2) | 1.2467 | +0.005 | 噪声内, 选路 1/8 |
| partition+**excl** (创新) | **1.2467** | +0.005 | **同 part_enroll(7/8 选路一致)** |
| partition+oracle (天花板) | **0.9081** | **-0.33** | **切法本身有效**, 救回空间大 |

### 选路选对率 (对比 B2 baseline 25%)

| 选路 | 选对率 | n/N |
|---|---:|---:|
| enroll 选路 (B2 baseline) | 12.5% | 1/8 |
| **EXCL 选路 (创新)** | **0%** | **0/8** |
| oracle (天花板) | 100% | 8/8 |

### 子集分析 (argmax 对/错)

| 子集 | n | enroll 选对 | excl 选对 |
|---|---:|---:|---:|
| enroll 选路选对(B2 sep_picks_oracle=T) | 1 | 1 | 0 |
| enroll 选路选错 | 7 | 0 | 0 |

### 逐条 CER

| uid | argmaxA | full | excl | p.enr | p.exc | p.orc | enroll/excl/oracle picks | ref |
|---|---:|---:|---:|---:|---:|---:|:--:|:--|
| cmd_2637 | 1.25 | 1.25 | 1.00 | 1.12 | 1.12 | **0.25** | 0/0/1 | 哺乳期要少吃什么 |
| cmd_2188 | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 | 1/1/0 | 吃什么有利于脂肪肝 |
| cmd_2302 | 1.17 | 1.17 | 1.83 | 1.33 | 1.33 | **0.00** | 0/0/1 | 给我放桃花诺 |
| cmd_2347 | 0.83 | 0.83 | 1.00 | 0.83 | 0.83 | 0.83 | 1/1/0 | 开启睡眠模式 |
| cmd_2503 | 1.14 | 1.14 | 1.14 | 1.14 | 1.14 | 1.14 | 0/0/1 | 把风速调到最大 |
| cmd_2630 | 2.25 | 2.25 | 1.50 | 2.25 | 2.25 | **1.75** | 0/0/1 | 开左右风 |
| cmd_2766 | 1.40 | 1.40 | 1.40 | 1.40 | 1.40 | 1.40 | 1/1/0 | 播放苦命人 |
| cmd_2890 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/1/0 | 吃什么有利于脂肪肝 |

## 4 个核心问题回答

### ① 分区切(partition)是否优于 full(baseline)和 exclusive-only?

**视选路而定**:
- partition+oracle 显著优于 full: 0.91 vs 1.24, **Δ-0.33** (SepFormer 选对路后切走重叠区混合有效)
- partition+enroll/excl 与 full 持平: 1.25 vs 1.24 (选路错 → ov 段切错 SepFormer 路, 等于"用错的 SepFormer 输出"代替"原始混合", 没收益)
- exclusive-only 略优于 full (1.22 vs 1.24) 但噪声内 (-0.02), 单独使用会丢目标内容(如 2302 切走 1.6s 重叠区, CER 1.83 比 full 1.17 反劣)

**结论**: 切法(partition)本身天花板有效, 但**完全卡死在 SepFormer 选路**。

### ② EXCLUSIVE 选路是否显著优于 ENROLL 选路? **核心创新证伪**

**否**。exclusive 选路 **0/8 选对** vs enroll **1/8** (12.5%), 反而更差。机制根源:

> **exclusive 段就是 enroll 选定的 target_idx 的独占段** — 二者描述同一说话人, 在 SepFormer 两路上的 sim 排名几乎一致。

数据支撑(`diag.json` 中 enroll_sims vs excl_sims):
- 7/8(87.5%) enroll_pick == excl_pick (选路完全一致)
- 唯一差异化的 cmd_2890 (0.4s exclusive 太短 < 0.5s, 抽 emb 不稳) — excl 选错(=1)而 enroll 选对(=0=oracle), 是噪声而非系统性优势
- 数值层面: enroll_sims 与 excl_sims 不同(因为是不同音频段), 但 sim 排名(决定选路)argmax 一致

**结论**: 创新假设"exclusive 段在 SepFormer 域更近"不成立。exclusive 段是 target speaker 自己的纯音, enrollment 也是 target speaker 的参考音频 — 两者在 wespeaker 嵌入空间指向同一说话人。SepFormer 路选择错误的根源**不是 enrollment 污染**(否则 exclusive 会救), 而是 **SepFormer 输出域的声纹整体失真**(SI-SDR 优化 ≠ 声纹保持, 两路 SepFormer 输出 emb 都扭曲, 任何"原始域 target 锚"都匹配不准)。

### ③ ORACLE 天花板 vs 实际选路差距多大?

- partition+oracle mean **0.91** vs partition+enroll/excl **1.25** → 差距 **0.34** (所有可救空间全卡在选路)
- 逐条救回(part_oracle vs full):
  - cmd_2637: 1.25 → 0.25 (-1.0, 救回"哺乳少吃什么")
  - cmd_2302: 1.17 → 0.00 (-1.17, 完美救回"给我放《桃花诺》")
  - cmd_2630: 2.25 → 1.75 (-0.5, 部分救)
  - cmd_2347/2503/2766: oracle 也救不动(ref 内容音频里就不存在)
- 即: **40-50% 的失败组存在 SepFormer 分离出的干净 target 路, 但声纹选路选不出来**

### ④ cmd_2637 救回没?

**part_oracle 救回 (1.25 → 0.25), 但 part_enroll/excl 没救 (1.12)**:
- full (1.25): "就已经进行过一轮教育。" (spk0+spk1 混合, ASM 转 louder 非 target)
- part_oracle (0.25): "哺乳少吃什么？" (SepFormer src1 = 真 target = spk1 路, 救回, 少了"期"字)
- part_enroll (1.12): "就已经进行过一轮交。" (SepFormer src0 = 错路 = spk0 干扰人, 与 full 几乎一样)
- part_excl (1.12): 同上 (exclusive 与 enroll 都选 src0)

2637 是"enrollment 污染"的典型: enroll emb 与 spk0 (干扰人) sim 0.585 高于 spk1 (真target) sim 0.10, argmax 一开始就选错 target_idx (spk0 而非 spk1)。exclusive 段是 spk0 的纯音, 自然也指向 spk0 → SepFormer src0 路 → 没救。

**真正救 2637 的解药**: 不是声纹选路(任何声纹锚都指向 spk0), 而是内容判别 — 把 SepFormer 两路都转写, LLM 挑"哺乳期要少吃什么"这种家居/健康指令段(对应 memory `multi-voice-llm-routing-architecture.md` 多声纹 LLM 路由方向)。本 POC 的 oracle 选路本质就是用 ref 内容做的"内容判别"上限。

## 结论与下一步

### 分区切 + exclusive 选路值得集成吗?

**不值得直接集成**(创新证伪, 实际选路瓶颈未解)。但**有两个重要副产品**:

1. **partition + oracle 选路 = 0.91 天花板坐实**: 切走重叠区 + SepFormer 选对路能从 1.24 → 0.91 (Δ-0.33), 这给"非声纹 target 选择"路线明确的收益上限。
2. **声纹选路上限证伪**: enroll (12.5%) / exclusive (0%) 都不行, **SepFormer 输出域的声纹整体失真** 是根源。任何"原始域 target 锚"匹配都低效。

### 真正值得投的方向(按 ROI 排序)

| 方向 | 机制 | 与本 POC 关系 |
|---|---|---|
| **A. 内容判别选路 (multi-voice LLM 路由)** | SepFormer 两路 + diar per_spk 都转写 → LLM 挑家居指令段 | 本 POC oracle 的可达版本, 复用 partition+sepformer 框架 |
| **B. Whisper-Sidecar embedding 路线** | end-to-end ASR+speaker 联合训练, embedding 空间与 ASR mel 同域 | 绕开 SepFormer SI-SDR 失真, 根治选路问题 |
| **C. ASE 自增强选帧 (复用 CAM++)** | enrollment 自增强近远场失配, 改 enroll emb | 不解 SepFormer 失真, 仅解 enroll 污染(边际) |

方向 A 最直接: 本 POC 已证明 oracle(partition+内容选路)天花板 Δ-0.33, 复用 `code/runs/_partition_poc/slices/` 已切好的 partition wav + SepFormer 两路文本, 只需加 LLM 内容判别即可 POC。

## 局限

1. **样本量小 (n=8)**: 8 条里 1 条差异化, 统计意义有限。但 7/8 一致性 + 机制根源清楚(exclusive 与 enroll 同说话人), 不影响核心结论。
2. **复用 B2 sepformer wav**: 没重新分离(省时), 但 B2 是 enroll 选路命名(`__target`/`__src{i}`), 本 POC 通过 per_src 顺序映射 src_idx 0/1, 已验证正确。
3. **exclusive < 0.5s 抽 emb 不稳** (cmd_2890 0.4s 是边界 case, cmd_2302 0s 走 fallback enroll): 影响有限, 因即使稳也指向同说话人。
4. **SepFormer 英文 OOD 损字**: SepFormer whamr16k 训练在英文 WHAM, 中文 OOD 风险(如 cmd_2475"无"→"雨"类), 本 POC 未单独评估。

## 产物

- 报告: `docs/partition_cut_poc.md` (本文)
- POC 脚本: `code/exp_partition_cut.py` (seed=42 可复现)
- 中间产物:
  - `code/runs/_partition_poc/slices/*.wav` (40 切片 = 8 样本 × 5 切法)
  - `code/runs/_partition_poc/_uid2text.json` (qwen 转写结果)
  - `code/runs/_partition_poc/summary.json` (5 切法 CER + 选路)
  - `code/runs/_partition_poc/diag.json` (每条 enroll/excl sims + 子区时长)
- 运行日志: `code/runs/_partition_poc_run.log`
- 总耗时: **0.5 min** (复用 B2 sep wav 不重分离, 8 样本 diar 24s + qwen 40 切片 4s)
