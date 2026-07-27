# Multi-voice 全量泛化验证报告 (2026-07-27)

## 任务来源

**前情**: multi-voice 内容判别选路 POC 在**失败组 40 条**已突破
(`code/runs/_multivoice_route/summary.json`):

| 选路 | mean CER | 选对率 |
|---|---|---|
| heuristic 综合 (策略3) | **0.6538** | **87.5% (35/40)** |
| oracle 天花板 | 0.6026 | — |
| argmax 主线 | 1.2163 | — |
| sim 选路 | 1.2489 | 25.0% |

**本任务**: 验证 heuristic 选路在**主战场**(非失败组)的泛化能力,测 RTF 成本,算整体外推 CER。
产物: `code/runs/_multivoice_full/summary.json` + `code/exp_multivoice_full.py` (seed=42)。

## 实验设置

- **主战场定义** (与失败组互补): `sim≥0.4 & qwen_cer<0.8` — argmax 主线已能解的子集
- **失败组定义** (复用 POC): `sim≥0.4 & qwen_cer>0.8` — argmax 主线已翻车
- **死区定义**: `sim<0.4` — 重 babble,主线 argmax mean 0.4255
- **数据**: `code/runs/poc_qwen_asr_full_result.json` 1350 条,seed=42 抽样
- **链路**: SepFormer-whamr16k 分离 recognition → sourceA/sourceB → 两路 Qwen3-ASR 转写 → heuristic 选路 (复用 `exp_multivoice_route.route_heuristic` 策略3)
- **硬件**: 4060 (8GB), Python `code/.venv/` (SepFormer/diar) + `code/.venv_qwen/` (qwen subprocess)

## 关键发现 1: 数据分布与任务预判不符

任务预判"主战场占 74.9%" — **实际三分桶分布**:

| 桶 | 条数 | 占比 | 主线 argmax mean CER |
|---|---|---|---|
| 死区 (sim<0.4) | 1064 | **78.8%** | 0.4255 |
| 主战场 (sim≥0.4 & cer<0.8) | 243 | 18.0% | **0.0588** |
| 失败组 (sim≥0.4 & cer>0.8) | 43 | 3.2% | 1.2204 |

- 主战场池实际只 243 条 (<260 任务额),seed=42 全抽 (243)
- **主战场 argmax mean CER 0.0588 已极优** (多数 sim≥0.4 样本 argmax 已完美转写,cer=0)
- 失败组占比仅 3.2% (不是 25.1%) — 整体改善天花板受限

## 关键发现 2: 主战场反退化 (Δ+0.1195)

主战场 243 条各策略 mean CER:

| 策略 | mean CER | median | correct<0.5 | 选对率 (vs oracle) |
|---|---|---|---|---|
| **argmax 主线 (paired)** | **0.0588** | 0.0 | **96.7%** | — |
| oracle (两路取优, 天花板) | 0.1025 | 0.0 | 91.4% | — |
| sim_route (SepFormer+emb 选 target) | 0.1577 | 0.0 | 86.4% | 88.1% |
| **heuristic (策略3)** | **0.1784** | 0.0 | 83.5% | 87.2% |
| content_gate (二值) | 0.7292 | 1.0 | 26.8% | 25.5% |

**核心反直觉**:
1. **oracle 都救不动 argmax**: SepFormer 分离后两路取优 mean 0.1025 vs argmax 主线 0.0588 = **Δ+0.0437** — SepFormer 强行分离反而破坏本来干净的 mel,即便 oracle 选路也回不到 argmax 水平
2. **heuristic 主战场 Δ+0.1195 (反劣)**: 选对率 87.2% 接近失败组 87.5%,但**选错代价放大** (主战场 oracle 路 cer=0 完美,heuristic 选错到 cer=1.0+,损失全量)
3. **sim_route 反而比 heuristic 略优**: sim≥0.4 子集声纹能区分 target (与死区/sim 选路证伪不冲突 — sim≥0.4 时 babble 弱,声纹信号保留)
4. **content_gate 灾难性退化** (0.7292): 主战场两路通常都通过 content_gate (都是有效中文),退到 tiebreak_shorter,而 shorter≠target

### 与失败组的对比 — 为何 POC 突破不能迁移

| 维度 | 失败组 (40条,有效) | 主战场 (243条,反退化) |
|---|---|---|
| argmax 主线 mean CER | 1.216 (已翻车) | 0.059 (已极优) |
| SepFormer 分离意义 | 强行分离出非 target 干净 mel → 救 Δ-0.56 | 强行分离破坏本来干净 mel → 恶化 Δ+0.04 |
| heuristic 选错代价 | 选错路 cer 也 1.0+ (本来灾难) | 选错路 cer=1.0+,但 oracle cer=0 (全量损失) |
| 两路文本特征 | 一路 news/财经词,一路家居词 (易区分) | 两路都是有效中文 (target 转写 vs 中文幻觉,内容都像指令) |

## 关键发现 3: 选错案例特征 (31/243)

选错 31 条,两大类:
- **tie 平局 fallback src0** (20/31 = 65%): 两路 cmd_score 相等 (典型都是 0.5 — 仅长度加分,无家居词/news词),fallback 取 src0,但 SepFormer 输出顺序与 target 无关,50% 概率选错
- **score 误判** (10/31 = 32%): 评分函数被"中文幻觉恰好是家居指令句式"误导

**典型 score 误判** (oracle cer=0 vs heuristic cer=1.0+):

| uid | ref | oracle 路 (丢) | heuristic 选错路 | 评分 | 原因 |
|---|---|---|---|---|---|
| cmd_2105 | 播放乐乐讲故事第三季 | 播放《乐乐讲故事》第三季 (cer=0) | 调到空调温度到二十七度 (cer=1.10) | 2.5 vs 8.0 | 幻觉路命中 空调+温度+动作 多关键词 |
| cmd_32 | LADY GAGA最新的专辑叫什么 | Lady Gaga 最新的专辑叫什么 (cer=0) | 打开电视机 (cer=1.0) | 0.0 vs 5.5 | 幻觉路命中 电视+打开 |
| cmd_164 | 帮我策划三天深圳旅游的行程 | 帮我策划...行程 (cer=0) | 苏宁会员 (cer=1.0) | -5.5 vs 0.5 | ref 是 query 类无家居词,评分低 |
| cmd_69 | 恢复 | 恢复 (cer=0) | 嗯 (cer=1.0) | -1.0 vs -1.0 | tie 双方都低分,fallback |

**根本机制**: heuristic 评"像家居指令",但主战场非 target 路 (SepFormer 输出的另一说话人) 也是中文短语,内容可能是另一说话人说的"打开电视机"等**真实家居指令句式** (背景里另一人在控制设备),评分反高于 target 路 (target 说的可能是 query/闲聊类 ref)。

## 关键发现 4: 整体外推 CER

三分桶按真实占比加权:

```
overall = 0.788 × 0.4255 (死区主线 argmax)
        + 0.180 × 0.1784 (主战场 multi-voice heuristic)
        + 0.032 × 0.6538 (失败组 multi-voice heuristic, 复用 POC)
        = 0.3353 + 0.0321 + 0.0208
        = 0.3883
```

| 口径 | multi-voice 外推 | 主线 | Δ |
|---|---|---|---|
| 逐条 mean CER | **0.3883** | 0.3848 | **+0.0035** (微涨) |
| 官方池 CER | — | 0.3436 | — |

**外推 Δ+0.0035 几乎无差异** — multi-voice 在失败组救的 0.56 CER (3.2% 占比 × 0.56 = 0.018 整体收益) 被主战场退化 (18.0% × 0.12 = 0.022 整体损失) 抵消还反向。

## 关键发现 5: RTF 成本

4060 实测 (主战场 243 条, 总音频 11.3min):

| 阶段 | 耗时 | RTF |
|---|---|---|
| SepFormer 分离 | 63s | 0.093 |
| 两路 qwen 转写 (batch=16) | 93s | 0.136 |
| **multi-voice 链路 (sep + 两路 qwen)** | 156s | **0.229** |

**口径注意**:
- 测得的 RTF 0.229 = SepFormer + 两路 qwen 的耗时, **不含**主线 enroll+diar (wespeaker 声纹+diarization) 部分
- 真实集成 RTF = 主线 enroll+diar (~0.05) + SepFormer (0.093) + 两路 qwen (0.136) ≈ **0.28** @4060
- 或对照主线 qwen 后端单路 RTF ~0.289, multi-voice 集成 ≈ 0.289 + 0.093 + 0.068 (qwen 第二路) ≈ **0.45** @4060
- L20 外推 (×1.5-2.0): **0.115-0.153** (4060 测得 0.229,绝对值不可外推仅排序可参考)

对比主线 RTF (4060): vanilla+关LLM 0.24 / qwen 后端 0.289 — multi-voice 集成 RTF 翻倍以上。

## 结论: GO=否 (不集成)

| 维度 | 评估 |
|---|---|
| 主战场泛化 | **退化** (heuristic Δ+0.1195,即便 oracle 也 Δ+0.0437) |
| 整体外推 CER | **微涨** Δ+0.0035 (在 CER±0.04 噪声内,无意义) |
| RTF 代价 | **翻倍** (~0.45 vs 主线 0.24-0.29) |
| 失败组救 Δ-0.56 | 占比仅 3.2%,整体收益 0.018,被主战场损失 0.022 抵消 |
| **集成判定** | **NO-GO** — 不集成主线 |

### 答辩弹药 (诚实归因)
1. **multi-voice 内容判别在失败组有效但不能泛化到主战场** — 失败组 argmax 已翻车 (1.216) 有救的空间,主战场 argmax 已极优 (0.059) SepFormer 强分离纯添乱
2. **SepFormer 中文 OOD + SI-SDR 陷阱复现**: 即便 oracle 选路,主战场分离后 cer 反而比 argmax 高 0.044 (mel 被拆坏)
3. **heuristic 评分对"中文幻觉恰好是家居指令句式"无辨别力**: 主战场非 target 路可能是背景里另一人说"打开电视机"等真指令,评分反高于 target 的 query/闲聊
4. **整体外推 Δ+0.0035 < CER±0.04 噪声** — multi-voice net neutral CER,但 RTF 翻倍,**效率腿净负**

### 局限
- 主战场池只 243 条 (任务额 260),但已覆盖 100% 池子
- 未跑死区 multi-voice (sim<0.4 子集,预判 SepFormer 在重 babble 更恶化,且死区已 oracle 证伪 — spk-oracle-poc 显示死区 sep+qwen 0.687 vs argmax 0.410 Δ+0.277)
- 失败组 POC 复用 40 条 (实际池 43 条,3 条未跑)
- RTF 测量未含 enroll+diar 部分 (主线已含,口径差异已在报告说明)

## 产物路径
- 脚本: `code/exp_multivoice_full.py` (seed=42)
- 中间: `code/runs/_multivoice_full/{slices/, _uid2text.json}`
- 汇总: `code/runs/_multivoice_full/summary.json`
- 报告: `docs/multivoice_full_validation.md` (本文)
- 未改主线代码, 未 git commit
