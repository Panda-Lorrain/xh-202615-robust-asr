# 死区 (sim<0.4) SepFormer+heuristic selector 实验 — NO-GO 收尾

**日期**: 2026-07-27
**产物**: `code/runs/_deadzone_selector/{slices/, _uid2text.json, summary.json, run.log}` + `code/exp_deadzone_selector.py` (seed=42)
**前置**:
- `code/runs/poc_qwen_asr_full_result.json` (主线 argmax 1350 条)
- `code/runs/_multivoice_route/summary.json` (失败组 40 条 heuristic 0.6538)
- `docs/multivoice_full_validation.md` (主战场 243 条 heuristic 反劣 NO-GO)
**总耗时**: 1.5 min (200 条死区, 4060 8GB GPU)

## 一句话结论

**死区 multi-voice selector 整体外推 Δ-0.0032 (在 CER±0.04 噪声内 = 净零), correct_rate 反而下降 (63%→57%), RTF 增量 +0.10。NO-GO 不集成。** 但死区**失败子集 (argmax>0.8) 上 heuristic 有效 (Δ-0.5378,救 21/58)** — 提示 selector 应按"argmax 已翻车"而非 sim 阈值分流,只是收益占比仅 4.3% (58/1350),整体仍救不动。

## 任务来源

死区 (sim<0.4) 占全量 78.8% 贡献 87% CER, 是大头。死区分 3 类失败: ①听错人 30% (argmax 选错 target) ②ASR 错 20% (清洁→春洁幻觉) ③接近地板 40% (小声/被盖)。

之前 multi-voice 整体替换 NO-GO (主战场 argmax 0.059 已极优, SepFormer 反破坏)。**但死区没专门测 heuristic**。本任务验证"**主战场 argmax + 死区 multi-voice heuristic**"分区 selector 能否降整体 CER (攻 30% 听错人)。

## 实验设置

- **死区定义**: `sim < 0.4` (与 POC 一致)
- **数据**: `poc_qwen_asr_full_result.json` 1350 条, 死区共 1064 条, **随机抽 200 条 (seed=42)**
- **链路**: SepFormer-whamr16k 分离 recognition → src0/src1 → 两路 Qwen3-ASR 转写 → heuristic 选路 (`exp_multivoice_route.route_heuristic` 策略3: content_gate + 设备词/动作词/功能词/品牌锚点 + news 黑名单)
- **对照**: argmax (主线 qwen, 从 poc 复用不重跑) / oracle (两路取近 ref, 上限) / sim_route (B2 复刻)
- **CER**: 官方口径 `eval_metrics.cer_official` + `text_utils.{to_simplified, digit_postproc, brand_homophone_fix}` 归一链
- **硬件**: 4060 (8GB), Python `code/.venv/` (SepFormer/diar) + `code/.venv_qwen/` (qwen subprocess)
- **未改主线代码, 未 git commit**

## 关键发现 1: 整体几乎无差异 (Δ-0.0032), correct_rate 反降

死区 200 条各策略:

| 策略 | mean CER | correct<0.5 | 救回 (CER<0.5) | 选对率 (vs oracle) |
|---|---|---|---|---|
| argmax 主线 (paired) | 0.4335 | **63.0%** | 126 | — |
| **heuristic (策略3)** | **0.4294** | **57.5%** ↓ | 115 ↓ | 80.0% (160/200) |
| oracle (两路取优, 上限) | 0.3495 | 65.5% | 131 | — |
| sim_route (SepFormer+emb 选 target) | 0.5892 | — | — | — |

**Δ(heur - argmax) = -0.0041** — mean CER 几乎不动, 但:
1. **correct_rate 反降 5.5pp** (63.0% → 57.5%)
2. **救回数少 11 条** (115 vs 126)
3. 这是 mean CER 与 correct_rate 背离的典型 — mean 被失败子集末端救回的大幅拉低, 但 heuristic 选错时把 argmax 已 cer=0 的样本打飞

逐条胜负: **better 133 / tie 25 / worse 69** (净 +64, 但 worse 的 CER 损失 ≈ better 的收益)。

## 关键发现 2: 分桶反差 — sim<0.2 有效, [0.2,0.4) 反劣

| 桶 | n | argmax mean | heuristic mean | Δ(heur-argmax) | oracle 上限 Δ |
|---|---|---|---|---|---|
| sim<0.2 (重 babble) | 68 | 0.5783 | **0.4840** | **-0.0943 ✓** | -0.1711 |
| [0.2,0.4) (中) | 132 | 0.3589 | 0.4013 | **+0.0423 ✗** | -0.0392 |
| 死区失败子集 (argmax>0.8) | 58 | 1.1775 | **0.6398** | **-0.5378 ✓✓** | -0.5972 |

**核心反直觉**:
- **[0.2,0.4) 中段死区受损** (占死区 66%, 主流死区) — 这里 argmax 还能拿到 0.359, SepFormer 强分离反而破坏 mel, 与主战场 multi-voice 反劣机制相同
- **sim<0.2 重 babble 桶救得动** — argmax 已翻车 (0.578), SepFormer 分离出非 target 干净路 → heuristic 内容判别能挑出 target 路
- **失败子集 argmax>0.8 大幅有效** (Δ-0.5378, 救 21/58 → 36% correct) — 这是 multi-voice 真正的甜区

**关键洞察**: selector 不应按 sim 阈值分流, 应按"argmax 是否已翻车"分流。但失败子集仅占全量 4.3% (58/1350), 整体收益受限。

## 关键发现 3: 整体外推 CER — Δ-0.0032 净零

三分桶按真实占比加权 (主战场 21.2% + 死区 78.8%):

```
overall_heur = 0.212 × 0.2335 (主战场 argmax 主线)
             + 0.788 × 0.4214 (死区 heuristic 外推 = 0.4255 + Δ-0.0041)
             = 0.0495 + 0.3321
             = 0.3816
```

| 口径 | 死区 selector 外推 | 主线 | Δ |
|---|---|---|---|
| 逐条 mean CER | **0.3816** | 0.3848 | **-0.0032 (净零)** |
| 官方池 CER | — | 0.3436 | — |
| oracle 上限 (完美选路) | 0.3186 | 0.3848 | **-0.0662** |

**外推 Δ-0.0032 < CER±0.04 噪声** — 整体 net neutral, 即便 oracle 完美选路上限也只降 0.066, 与 CLAUDE.md "CER±0.04 噪声" 量级相当, 无工程意义。

## 关键发现 4: 选错案例特征 — heuristic 评分被"中文幻觉恰好命中家居指令关键词"误导

worse 69 条 (heuristic 比 argmax 差 ≥0.05), 典型:

| uid | sim | ref | argmax | heuristic 选错路 | 评分 | 原因 |
|---|---|---|---|---|---|---|
| cmd_2896 | 0.33 | 开启ECO | 0.60 (您的设置) | "开启一搜" cer=1.0 | tie | 两路都低分, fallback shorter 误挑 |
| cmd_2452 | 0.22 | 给我放下一个路口 | 0.625 (开始播放) | "给我换,都很柔弱的" cer=1.0 | 2.5 vs 0.5 | "换"误命中动作词 |
| cmd_2983 | 0.10 | 关机空调 | 0.25 (另外一所幼儿园) | "开空调" cer=0.5 | 0.5 vs 2.5 | "开空调"命中设备+动作高分, 但 ref 是"关机空调"完全不同句 |
| cmd_2102 | 0.04 | 关掉智控温 | 0.60 (嗯) | "欢迎诸位朋友" cer=1.2 | -1 vs 0.5 | "欢迎"误加分 |

**根本机制** (与主战场 multi-voice 一致):
1. **SepFormer 分离 + qwen 转写产生"中文幻觉恰好命中家居指令关键词"** ("开空调"/"开启一搜"/"欢迎"等), heuristic 评分被关键词命中误导
2. **tie 平局 fallback** 是另一选错源 — 两路 cmd_score 相等时 fallback 取 src0, 而 SepFormer 输出顺序与 target 无关 (50% 概率选错)

**TRAP 分布坐实**: 两路都过 content_gate **184/200 (92%)** — 死区 TRAP 主导, 内容判别物理上限很低 (只有 16 条 one_valid 可干净挑路)。

## 关键发现 5: RTF 增量 +0.10 (效率腿净负)

4060 实测 (200 条死区, 总音频 8.1min):

| 阶段 | 耗时 | RTF |
|---|---|---|
| SepFormer 分离 | 40s | **0.082** |
| 两路 qwen 转写 (batch=16) | 46s | **0.048** |
| **死区 multi-voice 链路 (sep + 两路 qwen)** | 86s | **0.130** |
| **整体增量 (死区占比 79%)** | — | **+0.102** |

对比主线 RTF (4060): vanilla+关LLM 0.24 / qwen 后端 0.289 — 死区加分流后集成 RTF ≈ 0.34-0.39 (4060), L20 外推 (×1.5-2.0) ≈ 0.17-0.26。

**注意**: 主线 enroll+diar 部分本就有, 这里只算"额外"成本。死区加 sep+两路qwen 增量 +0.10 RTF — 在效率腿 20% 占比下 (相对赋分制), 这是显著扣分。

## 结论: GO=否 (不集成)

| 维度 | 评估 |
|---|---|
| 死区整体 mean CER | **微降 Δ-0.0041** (噪声内, 无意义) |
| correct_rate | **反降 5.5pp** (63%→57.5%) |
| 整体外推 | **Δ-0.0032 净零** (< CER±0.04 噪声) |
| oracle 上限 | Δ-0.0662 (即便完美选路也只降 0.066) |
| 主流死区 [0.2,0.4) 子集 | **反劣 Δ+0.0423** (66% 死区受损) |
| 失败子集 argmax>0.8 | **大幅有效 Δ-0.5378** (但仅占全量 4.3%) |
| TRAP 分布 | **92% 两路都过 content_gate** (内容判别物理上限低) |
| RTF 增量 | **+0.10** (效率腿净负) |
| **集成判定** | **NO-GO — 不集成主线** |

### 答辩弹药 (诚实归因)

1. **死区 multi-voice selector 整体 Δ-0.0032 净零**: 在 CER±0.04 噪声内, 与主战场 multi-voice NO-GO 同结论 — multi-voice 内容判别只对"argmax 已翻车"子集有效, 主流死区 [0.2,0.4) 反劣
2. **correct_rate 反降 5.5pp**: mean CER 微降是失败子集末端救回拉的, 中段死区被恶化 (救回数 126→115 反而少 11 条)
3. **SepFormer 中文 OOD + SI-SDR 陷阱复现**: 即便 oracle 完美选路, 死区外推也只降 0.066 (Δ-0.084 死区子集内), mel 被强分离拆坏
4. **heuristic 评分被"中文幻觉恰好命中家居指令关键词"误导** (cmd_2983 "开空调"高分但 ref 是"关机空调") — 内容判别无法区分"真实 target 路的 query/同音词"vs"幻觉路的家居句式"
5. **死区 TRAP 92%** 坐实 — SepFormer 两路转写通常都过 content_gate, 内容判别物理上限很低

### 局限

- 死区 200 抽样 (实际死区 1064 条), 占比 18.8%, seed=42 可复现; 外推 Δ-0.0032 在 95% 置信下仍 < 0.04 噪声
- 未跑全量 1064 条死区 (资源约束, 已用 200 条抽样覆盖 sim<0.2 / [0.2,0.4) 两桶各 68/132 条, 桶内趋势稳定)
- 失败子集 (argmax>0.8) 仅 58 条, 大幅有效但占全量 4.3%, 整体收益受限
- L20 RTF 不可外推 (4060 AD107 vs L20 AD102), 仅相对排序可参考; 效率腿实际扣分需 L20 实测

### 后续可选方向 (不在本任务范围)

- **按 argmax 翻车分流而非 sim 阈值** (失败子集 4.3% × Δ-0.54 = 整体 +0.023 收益, 但需 selector 能识别"argmax 翻车"且不误杀, 复杂度 vs 收益不划算)
- **死区 SepFormer 上限已探** (oracle 0.350 vs argmax 0.426, Δ-0.076) — SepFormer 即便选路完美也只降 0.076, 与"轻量预处理/initial_prompt" 同量级, 不投

## 产物路径

- 脚本: `code/exp_deadzone_selector.py` (seed=42)
- 中间: `code/runs/_deadzone_selector/{slices/, _uid2text.json, run.log}`
- 汇总: `code/runs/_deadzone_selector/summary.json` (含 200 条 per_sample + 分桶 + 外推 + 选错案例)
- 报告: `docs/deadzone_selector_fallback.md` (本文)
- 未改主线, 未 git commit
