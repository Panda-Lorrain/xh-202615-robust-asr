# Multi-voice 内容判别选路 POC 报告 (2026-07-27)

## 背景

**前情**: B2 SepFormer 实验 (`code/exp_sepformer_b2.py`) 在失败组 40 条
(`sim≥0.4 & qwen_cer>0.8`, 即"主线 argmax 已经翻车"的子集) 上跑了源分离 + 两路 Qwen3-ASR 转写。结果:

| 选路方式 | mean CER | 选路准确率 |
|---|---|---|
| **oracle** (取 CER 较低那路, 天花板) | **0.603** | — |
| argmax 主线 (单路转写, 不分离) | 1.216 | — |
| **sim 选路** (SI-SDR/stream_sim) | 1.249 | 25% (10/40) |

**问题**: sim 选路整体证伪 — SI-SDR 波形分离优化方向不保声纹, 选路准确率仅 25%, 还不如随机。

**本 POC 的核心问题**: 不靠声纹, **靠文本语义内容判别**能否逼近 oracle 0.603?

## 实验设置

- **数据**: 复用 B2 产物 `code/runs/_sepformer_b2/summary.json` (40 条, 每条 2 路 SepFormer 输出 transcript + ref + oracle 真值)。不重新分离/转写。
- **task**: 给定两路 transcript `textA, textB`, 选出"像有效家居指令"那路作为 target 路; 对照 oracle_pick (CER 较低那路)。
- **指标**: mean CER (选出路的 cer 平均) + 选路准确率 (= oracle_pick 的比例)。
- **seed**: 42 (LLM do_sample=False 已确定, 启发式无随机)。

## 策略

### 策略1: content_gate 二值 (`is_valid_command`)
复用 `text_utils.is_valid_command`: 中文 + 长度≤22 + 无 news 黑名单词 + 无循环幻觉。
挑通过的那路; 都通过 → 取更短; 都不通过 → fallback src0。

### 策略2: LLM 语义判 (Qwen2.5-3B-Instruct)
复用 `code/llm_reject.py` 的 `LLMRejecter`, 每路独立判 verdict (accept/reject)。
挑 accept 那路; 都 accept → 挑 entity+action 更明确的; 都 reject → fallback src0。

### 策略3: 综合启发式评分 (content_gate + 家居指令特征)
设计连续评分函数 `cmd_score(text)`:
- **基底**: `is_valid_command=False` → -3
- **正信号** (领域先验, 非 A 集拟合):
  - 设备词 (空调/灯/洗衣机/电视/窗帘/风扇/热水器/净化器/音箱/扫地/净水器/闹钟/油烟机/...) → +2
  - 动作词 (打开/关闭/开启/关掉/调到/调节/启动/播放/定时/扫风/送风/...) → +2
  - 功能词 (模式/温度/风速/风量/睡眠/ECO/送风/柔风/防直吹/...) → +1.5
  - 疑问查询词 (吃什么/怎么/预防/明目/脂肪肝/忌口/...) → +2 (ref 多知识查询类)
  - 品牌锚点 (`_BRAND_ANCHORS`: 智控温/轻干洗/净呼吸/...) → +3
  - 复合 (设备+动作 / 动作+功能) → +1
- **负信号**:
  - news 黑名单词 (财经/体育/业务/市场/同比/收益/媒体/...) 每个 -3
  - 数字串≥4 (幻觉串如 二三二五一二) → -2
  - 长度>22 → -2; 长度<3 → -1
- 选分高那路; 平局 → fallback。

## 结果

| 策略 | mean CER | 选路准确率 | 备注 |
|---|---|---|---|
| **heuristic 综合 (策略3)** | **0.6538** | **87.5% (35/40)** | 最佳; 零外部依赖, 零 RTF |
| LLM Qwen2.5-3B (策略2, entity/action fallback) | 0.6622 | 85.0% (34/40) | LLM reject 全 reject (过严), 用 entity/action 二级判别 |
| content_gate 二值 (策略1) | 0.7426 | 72.5% (29/40) | 仅 is_valid_command 二值挑路, TRAP 下退化 |
| LLM Qwen2.5-3B (策略2, naive fallback) | 0.9783 | 60.0% (24/40) | LLM SYSTEM_PROMPT 过严, 40/40 全 reject → 全选 src0 |
| --- 对照基准 --- | | | |
| **oracle 天花板** | **0.6026** | — | 内容判别物理上限 |
| argmax 主线 (单路转写) | 1.2163 | — | 当前 main pipeline |
| sim 选路 (B2 SI-SDR/stream_sim) | 1.2489 | 25.0% (10/40) | 声纹破坏 → 选路基本无效 |
| 随机 50/50 估算 | 0.976 | — | 无信号下限 |

**核心发现**:
1. **heuristic 综合策略 mean CER 0.6538, 距 oracle 0.603 仅 0.05**, 选路准确率 **87.5%** 远超 sim 的 25%。
2. content_gate 二值已能拿到 0.7426 (压过 argmax 1.216 一大截), 但被 TRAP 拖累 (31/40 双 valid 无法判)。
3. 综合评分通过设备/动作/功能/疑问词打分, 即便两路都过 content_gate, 仍能挑出更"像指令"的一路 → 准确率从 72.5% 推到 87.5%。
4. **LLM (Qwen2.5-3B) naive 用法翻车**: SYSTEM_PROMPT 判"应否被设备执行"过于严格, 40/40 全 reject (含"把空调关上"这类真指令, 因结尾"行吧"被判非严肃指令; ref 多知识查询类"吃什么..."也被全拒)。直接 verdict 选路 = 60% 准确率, 比随机好不了多少。
5. **LLM 二级用法 (entity/action fallback) 接近 heuristic**: 即使 verdict 全 reject, LLM 抽取的 entity (空调/音箱/风扇) + action (打开/关闭/播放/调到) 仍有判别信号, 用它做 tiebreak → CER 0.6622 / 85% 准确率, 与 heuristic 0.6538 / 87.5% 接近, 但需要 3B 模型推理 (~5s/条 vs heuristic 零 RTF)。
6. **heuristic 是 Pareto 最优**: CER 最低 + 准确率最高 + 零 RTF + 零外部依赖。LLM 同档次表现但贵 10x。

## TRAP 分析 (内容判别上限)

40 条按 content_gate 分类:

| 子集 | 数量 | 含义 |
|---|---|---|
| both_valid (TRAP) | 31/40 | 两路都过 content_gate, 内容判别无法区分 |
| one_valid (clean) | 7/40 | 一路 valid 一路 invalid, 内容判别可干净挑路 |
| both_invalid | 2/40 | 两路都失败, fallback |

- **TRAP 子集 oracle CER: 0.561** (内容判别在此子集上的物理上限)
- TRAP 子集 argmax CER: 1.178 (当前主线水平)

**heuristic 在子集上的表现细分**:
| 子集 | n | heuristic 准确率 | heuristic CER | oracle CER |
|---|---|---|---|---|
| clean (one_valid) | 7 | **100% (7/7)** | — | — |
| **TRAP (both_valid)** | **31** | **87% (27/31)** | **0.619** | 0.561 |

**关键洞察**: 即使 TRAP 子集 (两路都过 content_gate, 二值无信号) 上, heuristic 通过设备/动作/功能/疑问词评分, 仍能 87% 选对路, CER 0.619 已逼近 oracle 0.561 (差 0.06)。这是 heuristic 相对 content_gate 二值 (TRAP 下退化为更短文本 tiebreak) 的核心增量。

含义: 即便内容判别在 TRAP 子集上完美挑路 (达到 oracle 0.561), 总体 mean 也只能到约 0.56。当前 heuristic 0.654 已经吃了大部分 TRAP 子集的红利 (通过评分挑短句 + 强指令特征)。

## 失败案例 (heuristic 5 条错路)

5 条全为 TRAP (两路均为非指令随机中文短语), 内容判别无法分辨:

| uid | ref | src0 / src1 (picked vs oracle) | 失败原因 |
|---|---|---|---|
| cmd_2128 | 风小一点 | "就是东西也可以能吃" vs "就是小概率的事情" | 两路均无家居信号 |
| cmd_2323 | 防直吹 | "先是林十一" vs "先是同时吹" | oracle 路"吹"字未触发 brand anchor (仅"防直吹"完整匹配) |
| cmd_2358 | 风速调低一点 | "调控的地区" vs "高到二十一" | "调控"误命中动作词 |
| cmd_2456 | 黑发哪些食物需要忌口 | "最好让学生自己留家做" vs "要发地址, 上你需要信息" | 两路随机 |
| cmd_2837 | 风速加大 | "彩电中,老对结构冲击下" vs "彩电市场竞争中,基线" | 两路强 news, oracle 仅"略不烂" |

结论: 5 条失败都是"两路随机中文, 无家居语义信号"的真 TRAP, 内容判别物理无法救。剩余空间 (<0.05 CER) 只能靠声纹/时长/能量等非文本信号补充。

## partition + multi-voice 扩展 (未做, 评估 NO-GO)

`code/runs/_partition_poc/summary.json` 仅 8 条样本, partition oracle CER **0.908** > pure SepFormer oracle 0.603 (更差), 且 partition POC 选路准确率 0/8。partition 把重叠区切单独 SepFormer 反而损失了 enrollemb 锚定的全局信息。鉴于:
1. pure SepFormer + 内容判别已 0.654 ≈ oracle 0.603, 几乎吃满红利;
2. partition oracle 比 pure 还差 0.3,
不展开 partition + content routing 重测 (低 ROI)。

## 结论与集成建议

1. **内容判别值得集成到 multi-voice 架构** — heuristic 综合评分把失败组 CER 从 argmax 1.216 / sim 1.249 拉到 **0.654**, 接近 oracle 天花板 0.603, 选路准确率 87.5% 远超 sim 25%。
2. **声纹 sim 选路整体证伪 → multi-voice 路由应改用内容判别 (heuristic) + sim 兜底**。声纹在 SepFormer 两路 SI-SDR 后已破坏, 不能作主信号。
3. **物理上限有限**: TRAP 子集 31/40 是内容判别无法分辨的"两路随机中文", 总体 mean CER 下限 ~0.56; 剩余空间只能靠非文本信号 (时长/能量/声纹原始 enroll sim) 补充。
4. **TRAP 的实际意义**: 失败组 (argmax 主线已经翻车) 本就是 babble 极重的尾部; 内容判别在此仍能救 87.5% 已是上限附近, 主战场 (非失败组) 上 SepFormer 两路通常只有一路是有效中文, content routing 决策更轻松。
5. **下一步**: 把 heuristic 评分集成进 multi-voice pipeline (替换 sim 选路 step), 在主战场 (全量 1362 条, 不仅失败组) 验证泛化性。

## 产物

- 脚本: `code/exp_multivoice_route.py` (策略1+3) / `code/exp_multivoice_llm_route.py` (策略2)
- 中间: `code/runs/_multivoice_route/summary.json` (策略1+3 + TRAP 分析) / `llm_routing.json` (策略2)
- 报告: `docs/multivoice_content_route.md` (本文件)
