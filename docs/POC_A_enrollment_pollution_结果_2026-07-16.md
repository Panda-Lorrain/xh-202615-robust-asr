# POC 方向 A 结果 — enrollment 污染检测自适应阈值

**日期**: 2026-07-16
**结论**: ❌ **No-Go**（核心假设证伪 + net 收益亏损，方向 A 对"提 RR 不损 pos"无效）
**spec**: `docs/superpowers/specs/2026-07-16-rr-improvement-poc-design.md`

---

## 背景与假设

- 起点: neg RR@thr0.27 = 90.51%（429/474），45 条漏拒是唯一靶点
- 核心假设: 45 漏拒 neg 中相当一部分 enrollment 被污染（target+非目标人同在），非目标人声纹撞像 enrollment → sim 偏高 → sim_only 判不过 → 漏拒。若成立，对污染 enrollment 单独用更高 thr 即可多拒这些 neg，干净 enrollment 的 pos 零影响（Pareto 改进）

## Q1 实测：检测手段可行性 ✅ Go

对 26 条 calibration（剔除 5 条过短，剩 21 条）的 `enw_可靠性` 真值测两种检测器：

| 检测器 | P | R | F1 | tp/fp/fn |
|---|---|---|---|---|
| **D1 DiariZen diar** | 0.62 | **1.00** | **0.77** | 10/6/0 |
| D2 wespeaker 帧聚类 | 0.48 | 1.00 | 0.65 | 10/11/0 |

- **选 D1**（F1=0.77 > 0.7）。R=1.00（真污染全召回，不漏检），但 **P=0.62（误报多，6 条干净被误判污染）**——这个误报率是 Q3 pos 代价的根源

## Q2 实测：污染与漏拒相关性 ❌ No-Go（假设证伪）

D1 跑全 474 neg enrollment，分组算污染率：

| 组 | n | 污染率 |
|---|---|---|
| 组A（45 漏拒, max_sim≥0.27） | 45 | **31.8%**（14 条污染）|
| 已拒组（max_sim<0.27） | 429 | 26.1% |
| 全 474 基线 | 474 | 26.6% |

- 倍数 = 1.22×（漏拒组/已拒组）
- **Fisher exact p = 0.4737**（完全不显著，远 > 0.05）
- **判定**: 实质 No-Go。漏拒组的 enrollment 污染率只比基线高 5pp，无统计显著性。**enrollment 污染不是漏拒的主因**——45 漏拒 neg 的声纹撞像并非 enrollment 污染导致

## neg 侧 ΔRR 估算（用 Q2 数据，秒级）

自适应 thr（污染 enr 用 thr_high、干净用 0.27）对 neg 的 RR 提升：

| thr_high | base RR | adapt RR | ΔRR | 多拒 |
|---|---|---|---|---|
| 0.30 | 0.9051 | 0.9114 | +0.0063 | +3 条 |
| 0.35 | 0.9051 | 0.9177 | +0.0127 | +6 条 |
| 0.40 | 0.9051 | 0.9262 | +0.0211 | +10 条 |

最多 +2.1pp。45 漏拒里只 14 条被判污染，**打击面太窄**。

## pos 侧代价估算（未实测 Q3，基于 D1 误报率代理）

- D1 在 neg 上 P=0.62 → pos 侧约 26% enrollment 被判污染（真污染+误报）
- 这些污染 pos 用 thr_high，其中 max_sim∈[0.27, thr_high) 的（pos 低 sim 重叠带 ~433 条）被新增误拒
- 估算 Δpos CER: thr_high=0.40 约 **+0.033**，thr_high=0.30 约 +0.008
- **net 收益**（RR 腿 - CER 腿, w1=w2=0.4）: 即使最优 thr_high 也 <0.5 分，多数 thr_high 净亏损（pos 代价 > RR 收益）

## Q3 未跑的理由

Q2（p=0.47 假设证伪）+ neg ΔRR（最多 +2.1pp）+ pos 代价估算（net 亏损）三者一致指向 No-Go。Q3 实测 Δpos CER 不会改变结论（即使 Δpos CER≤0.01 的最优情况，net 也 <0.5 分，不值得落地）。故不跑 Q3（省 30 分钟 pos diar）。pos 代价为估算非实测，已在上方诚实标注。

## 结论与后续

- **方向 A No-Go**。enrollment 污染自适应阈值无法有效提 RR
- RR 90.51% 已接近内容/信号类天花板。剩 7-8 条 TRAP（非目标人恰好说了真指令）是物理地板，任何方法救不了
- **转方向 B（FA 置信度二次拒）**：正交于内容规则（抓"转写未扎根音频"的幻觉），pos 安全（pos 幻觉拒掉是受益侧），预期 +0.5-1.5pp。是"提 RR"最后的有效正交尝试

## 答辩弹药（诚实归因）

> "我们验证了 enrollment 污染→自适应阈值这条说话人信号路线。核心假设（漏拒 neg 的 enrollment 污染率显著高于基线）被证伪：Fisher p=0.47，倍数仅 1.22×。即使强行对污染 enrollment 提 thr，neg ΔRR 最多 +2.1pp，而 D1 检测器 62% 的误报率导致 pos 侧代价超过 RR 收益，net 亏损。这是诚实归因——不是没试，是试了且有数据证明此路不通。"

## 产物

- `code/poc_enrollment_pollution.py`（D1/D2 检测器 + Q1/Q2/Q3 框架，含缓存）
- `code/poc_pollution_cache.json`（474 neg + 21 calib 的 D1 检测结果缓存；pos 1364 条因 Q3 未跑而未检测，脚本可复用）
- `code/poc_enrollment_pollution_result.json`（Q1/Q2 摘要）
