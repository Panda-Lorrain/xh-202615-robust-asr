# 稳定性 / 鲁棒性测试设计（Stability & Robustness Test）

**日期**：2026-07-19
**状态**：spec 待用户审查 → 审查通过转 writing-plans 出实现计划
**前置**：`docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md`（已落地的可复现性改造）
**相关 memory**：`reproducibility-hardening`、`dataset-split-spec`、`lessons-pitfalls`、`se-bug-orphan-truth`

---

## 1. 背景与动机

用户观察到：项目至今未做过真正的稳定性测试。核查后属实——现有 `code/verify_reproducibility.py` 只跑过 **前 20 条 × 2 遍 × `enroll_infer` 单阶段**（默认 `--limit 20`），全量 1362 条 × 多遍 × 完整链路从未跑过。memory 里「run-twice delta=0」的真实覆盖率约 1.5%，统计上几乎不能说明全量行为。

用户的核心问题：**如果跑十遍全量，每遍结果会不会不一样？如果不一样，能否对照找出问题音频、借此提升模型能力？**

经 brainstorming 校准（见 §11 决策记录），本 spec 把目标收口为：

> **用稳定性测试量化系统残余非确定 + 定位波动音频并归因根因，落地只做工程修复 + 诊断归档，绝不碰训练（A 集是测试集，不能当训练集，否则过拟合）。**

## 2. 目标与范围

**目标**
1. 量化全量上 qwen 后端的残余非确定（同种子多遍是否真的 delta=0）
2. 用扰动矩阵定位波动音频，归因到 5 类根因
3. 工程修复 R1/R2（提升提交可信度 + 效率腿稳定性）
4. 诊断归档 R3/R4/R5（答辩弹药 + 未来 hold-out 拒识 / A 集外训练的输入）

**范围（IN）**
- A 阶段：qwen 全量 1362 条 × 10 遍，同 seed=42，`enroll_infer` 单阶段
- B 阶段四维扰动：batch 扫描 / 变种子 / 输入微扰 / enrollment 扰动
- 根因归因分析 + 报告 + 可视化
- 工程修复：`use_deterministic_algorithms`（R1 触发）/ 锁 batch=1（R2 触发）

**范围（OUT，明确不做）**
- ❌ 数据增强 / 模型训练 / 微调 —— A 集是测试集，训练=泄漏=过拟合；且项目本就 zero-training
- ❌ 拒识策略调整 —— 基于 A 集内容的提交规则，需独立 hold-out，本次只归档
- ❌ `--enroll-augment` 开关调整 —— 同上，本次只记录倾向
- ❌ 完整 submit_infer 四阶段链路 —— 本次聚焦 `enroll_infer` 单阶段，变量可控

## 3. 现状核查（设计前提）

**种子透传覆盖（完整 submit_infer 链路）**
- ✅ `enroll_infer`：`repro.set_global_seed` + `np.random.default_rng(seed)` + qwen 子进程透传 seed
- ✅ `se_denoise` / `noise_classify` / `llm_reject`：`set_global_seed`（`llm_reject` 还 `do_sample=False` 贪心）
- ✅ `qwen_asr_backend` / `firered_asr_backend`：内联 set_seed（`cudnn.deterministic=True` / `benchmark=False`）

**关键裂缝（残余非确定源）**
- ❌ **全项目无任何一处设 `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`** —— 这是消除 GPU 矩阵乘 / attention / reduction 非确定的硬开关，`cudnn.deterministic` 只管卷积
- ⚠️ Qwen3-ASR（bf16 + batch=16）、Whisper（fp16）、DiariZen 这些 attention-heavy 模型，残余非确定客观存在
- ⚠️ batch padding：不同 batch 内切片组合 → 不同 padding mask → 数值差异（B1 维度专门测这个）

**run-twice 现有验证粒度**：逐条 transcript 比对 + 逐条 CER delta（粒度 OK），但样本仅 20 条、单阶段、2 遍。

## 4. 架构

两个新脚本（纯增量，不动现有提交链路）：

```
code/stability_test.py    # 编排器：按扰动矩阵逐遍跑 enroll_infer，每遍产出一个 result JSON
code/analyze_stability.py # 分析器：汇总所有遍 → 定义波动音频 → 根因归因 → 报告 + 可视化
```

**数据流**
```
stability_test.py
  ├─ A: 同 seed=42 × 10 遍 enroll_infer --asr-backend qwen --pairs pos_pairs (全量)
  ├─ B1: batch=1/8/16/32 × 2 遍
  ├─ B2: seed=42/100/200/314/555 × 2 遍
  ├─ B3: 输入微扰(高斯噪 / ±1dB / 时间偏移) × 1 遍   [需先生成扰动音频]
  └─ B4: --enroll-augment on/off × 1 遍
     ↓ 每遍 → stability_matrix/<run-id>.json
analyze_stability.py
  ├─ 汇总所有 run-id.json
  ├─ 逐维度统计(match_rate / CER delta 分布 / 决策翻盘率)
  ├─ 逐条波动判定 + 根因归因决策树
  ├─ 出 stability_report.json + per_utt_volatility.json
  └─ 出 stability_dashboard.html + docs/稳定性测试报告_2026-07-19.md
```

## 5. 扰动矩阵

| 维度 | run-id 命名 | 配置 | 遍数 | 估时(4060) |
|---|---|---|---|---|
| A 同种子复现 | `A_s42_r{0-9}` | seed=42, batch=16, 原始音频 | 10 | ~2.3h |
| B1 batch 扫描 | `B1_b{1,8,16,32}_r{0,1}` | seed=42, batch 变(需 Phase 0 前置透传), 原始音频 | 8 | ~1.1h |
| B2 变种子 | `B2_s{42,100,200,314,555}_r{0,1}` | seed 变, batch=16, 原始音频 | 10 | ~1.15h |
| B3 输入微扰 | `B3_p{gauss,vol,time}` | seed=42, batch=16, 扰动音频 | 3 | ~0.7h + 生成 |
| B4 enrollment 扰动 | `B4_aug{on,off}` | seed=42, batch=16, --enroll-augment | 2 | ~0.3h |
| **合计** | | | **~33** | **~6h** |

**B3 输入微扰细则**（每条识别音频生成 3 种扰动版，缓存在 `stability_matrix/perturbed/<perturb>/<uid>.wav` 复用）
- `gauss`：叠加 -45 dB 高斯噪声（不可感知，测数值边界）
- `vol`：音量 ±1 dB（测能量敏感）
- `time`：时间偏移 ±20 ms（测对齐敏感）

## 6. 波动判定 + 根因归因

**波动音频判定（触发任一即入清单）**
1. 文本不稳定：N 遍 transcript 出现 ≥2 种不同文本
2. CER 高方差：对 ref 的逐遍 CER std > 0.1（或 max-min > 0.3）
3. 决策翻盘：accept/reject 在 N 遍里翻转（sim 跨 thr 反复横跳）

**根因归因决策树**（顺序判定，一条音频可多根因）
```
if A 同种子10遍翻车:           → R1 GPU残余非确定(工程缺陷)
elif B1 batch档间翻车:         → R2 batch padding敏感(工程)
elif B2 变种子翻车(A稳定):     → R5 数值边界(模型能力短板)
elif B3 输入微扰翻车:          → R3 输入泛化短板(模型能力短板)
if B4 enrollment扰动翻车:      → R4 声纹锁定不稳(叠加判定)
```

**5 类根因 + 修复映射**

| 根因 | 诊断信号 | 本次落地 |
|---|---|---|
| R1 GPU 残余非确定 | A 同种子翻车 | **工程修复**：`use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG` |
| R2 batch padding 敏感 | B1 档间翻车、同档稳定 | **工程修复**：提交锁 batch=1 |
| R3 输入泛化短板 | B3 翻车 | 诊断归档（未来 A 集外训练输入） |
| R4 声纹锁定不稳 | B4 翻车 + 低 sim | 诊断归档（记录 `--enroll-augment` 倾向，独立 hold-out） |
| R5 数值边界 | B2 翻车、A 稳定 | 诊断归档（未来 hold-out 拒识输入） |

**每条波动音频档案**（`per_utt_volatility.json`）
```json
{
  "uid": "rec_0123", "ref": "把空调调到二十六度",
  "sim_bucket": "[0.2,0.3)", "n_runs": 33,
  "n_distinct_transcripts": 3,
  "transcripts": {"把空调调到二十六度": 28, "把空调调到二十八度": 4, "": 1},
  "cer_mean": 0.12, "cer_std": 0.31, "cer_max": 1.0,
  "decision_flips": {"accept": 30, "reject": 3},
  "root_causes": ["R2_padding", "R5_boundary"],
  "fix_action": "submit_lock_batch1 + 归档(未来hold-out)"
}
```

## 7. hold-out 硬边界（防过拟合）

贯彻 memory `dataset-split-spec` / `lessons-pitfalls`「A 集调规则必须 hold-out 验证泛化」原则：
- 本次**不调整任何基于 A 集内容的提交规则**：拒识阈值 / 规则 / `--enroll-augment` 开关全不动
- 工程修复（`use_deterministic` / batch=1）不涉及 A 集内容，安全
- 诊断发现的所有「模式」只进报告 → 答辩弹药 + 未来 hold-out / A 集外训练决策输入
- R3/R4/R5 的「修复」全部推迟到独立 hold-out 流程或 A 集外训练项目，本次仅归档

## 8. 产物清单

| 产物 | 内容 | 用途 |
|---|---|---|
| `stability_matrix/<run-id>.json` × 33 | 每遍逐条结果 | 原始数据 |
| `stability_report.json` | 每维度 match_rate / CER delta / 翻盘率 + 波动清单(按严重度) + 根因分布 | 机读汇总 |
| `per_utt_volatility.json` | 每条波动音频完整档案 | 深挖 + 未来输入 |
| `stability_dashboard.html` | CER delta 分布 / 波动 sim 分桶 / 决策翻盘矩阵 / 根因堆叠图 | 可视化（dataviz 规范） |
| `docs/稳定性测试报告_2026-07-19.md` | 人读报告：诚实归因 + 量化 + 修复路线 + 答辩弹药 | 答辩 + 决策 |

## 9. 实现顺序

- **Phase 0**：
  - **前置改动（阻塞 B1）**：给 `enroll_infer` 加 `--asr-batch-size` 参数并透传给 qwen/firered 子进程。当前 `enroll_infer.py:413-415` 调 qwen 后端时未透传 batch-size，qwen 固定用默认 16，B1 维度无法实现（`qwen_asr_backend.py:24` 本身支持 `--batch-size`）。小改动 + 回归验证不影响现有 20 条 delta=0。
  - 写 `stability_test.py` + `analyze_stability.py`，小样本（20 条）dry-run 验证逻辑跑通
- **Phase 1**：跑 A 阶段同种子×10（~2.3h）→ 先评估 R1 严重度（决定是否立即修 `use_deterministic`）
- **Phase 2**：跑 B 阶段四维（~3.7h）
- **Phase 3**：分析 + 报告 + dashboard 可视化
- **Phase 4**：工程修复（R1/R2 触发时）+ 验证不破坏现有 20 条 delta=0 + commit

**容错**：每遍独立 run-id JSON，任一遍失败不影响其他遍分析（降级为 N-1 遍）；支持断点续跑（跳过已存在 run-id）。

## 10. 预判假设（待数据验证，写进报告）

1. 死区 / 低 sim 桶（sim<0.2，memory 说 babble 摧毁 mel）应是波动高发区
2. R1 若大面积出现 → `use_deterministic_algorithms` 必修（提交可信度硬伤）
3. R2 若显著 → batch=1 提交 vs batch=16 验证数字不一致（memory `efficiency_analysis.py:345` 未验证风险坐实）
4. R5 数值边界音频应与「接近 thr 的 sim」高度相关（决策边界附近）

## 11. 决策记录（why this design）

- **为什么混合 A+B 而非纯 A/B**：B 的鲁棒性分析必须先用 A 排除工程非确定污染，否则分不清波动来自 GPU 还是模型
- **为什么只 qwen 不双后端**：聚焦主线（CER 0.3436），vanilla 非当前提交后端，省一半成本
- **为什么 `enroll_infer` 单阶段不完整链路**：变量可控，SE/LLM 各自引入额外非确定会让根因归因变难；单阶段是 R1/R2 的主战场（ASR 模型 attention 非确定）
- **为什么落地只工程+诊断不碰训练**：A 集是测试集，训练=泄漏=过拟合（用户亲自纠正）；且项目本就 zero-training。稳定性测试的价值是工程可靠性 + 答辩 + 拒识稳定性（都对排名实打实），不是「用 A 集训模型」
- **为什么 hold-out 边界**：防止诊断中意外把 A 集模式泄漏进提交规则

## 12. 风险与对策

| 风险 | 对策 |
|---|---|
| 33 遍全量 ~6h，4060 跑一夜可能中断 | 每遍独立 JSON + 断点续跑（跳过已存在 run-id） |
| R1 修复 `use_deterministic` 可能拖慢推理（影响效率腿） | 修复后重测 RTF，若效率退化超阈值则只在提交链路开、验证链路关 |
| B3 输入微扰生成扰动音频占磁盘 | 扰动音频用完即删 / 缓存可配 |
| 分析时无意看到 ref 调规则（泄漏） | hold-out 硬边界：本次不改任何提交规则，只归档 |
