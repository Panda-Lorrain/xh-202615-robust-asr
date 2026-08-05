# Speaker-Aware Verification Fusion GO/NO-GO 探查

**日期**: 2026-08-05
**任务**: 判定「speaker-aware verification fusion / reranker（输出端 reranker / 帧级 verification / 更强 speaker encoder）」值不值得投 2-3 天
**结论**: **NO-GO**（弱 NO-GO，接近 tie，但 5-fold hold-out 把唯一正向证据打回零）

---

## 0. TL;DR（先看这个）

| 维度 | 结果 |
|---|---|
| **max_sim 整段 cosine（ResNet34-256d，现管线信号）** | **有信号**：通用区分 AUC=**0.862** CI[0.836,0.886]；但在 thr0.27 已拒子集里掉到 AUC=**0.725** |
| **stno_target_ratio（Personal-VAD 帧级 target 活跃）** | **反向信号** AUC=0.221（翻转后 0.779）：干扰人独白比 target 独白看起来"更像 target"（target 被 babble 淹没致 PVAD 失效） |
| **融合 max_sim + (-stno)** | 通用 AUC 0.862→**0.910**；operational 0.725→**0.846**；**确实有正交信息** |
| **in-sample best net Overall delta**（重新分类全部 mono） | **+0.0088**（替换 thr0.27，最优 alpha+thr） |
| **5-fold hold-out net delta**（防 alpha 过拟合） | **+0.0003 ± 0.0019** ← **真值，零，低于 ±0.04 噪声** |
| **operational recovery（在已拒子集救回 pos）** | **永不到 net 正**——precision 上限 0.75 < break-even 0.763，所有阈值 net negative |
| **强 speaker 信号可跑性** | 现有信号已接近天花板；ECAPA/帧级/CAM++ 跑得起但破不了精度天花板，详见 §4 |
| **判定** | **NO-GO**。enrollment 1.8s + babble + env 失配真磨平，speaker-aware 第三分支救不了 CER 腿。继续投效率/答辩/Phase-3 |

---

## 1. 数据来源 + 分组

**Per-sample 复用**（无任何 ASR/qwen 重跑）：

| 文件 | 内容 | 关键字段 |
|---|---|---|
| `code/runs/_scene_route_full/per_sample.json` | 1350 pos（scene-route 实验） | `uid, n_spk, max_sim, rejected_thr0.27, mainline_cer_transcribe, scene_route_cer_transcribe, ref, route_decision` |
| `code/runs/full_eval_20260730_pos/_work/enroll_all.json` | 1364 pos（enroll_infer 输出） | `max_sim, sims, stno_target_ratio, target_active_ratio, speakers, transcript, rejected` |
| `code/runs/full_eval_20260730_neg/_work/enroll_all.json` | 474 neg（enroll_infer 输出） | 同上 |

> ⚠️ 文档把 enroll 用的 encoder 写成 CAM++ 192d，**实测是 wespeaker ResNet34-256d**（参见 memory `a-set-is-real-recorded`），本探查按 ResNet34-256d 描述。

**分组**（uid 匹配 pos 两份 json，n_spk=1）：

| 组 | 定义 | 规模 |
|---|---|---|
| **POS n_spk=1 ALL** | 单人独白（target 独白，匹配上 enroll 的） | **543** |
| **POS 误拒 target 独白** | n_spk=1 + `rejected_thr0.27=True` + `mainline_cer_transcribe<0.3`（高质量 target 独白被 thr0.27 误拒，预期 147-185） | **147**（+38 条 CER≥0.3 是合理拒） |
| **NEG n_spk=1 ALL** | 单人独白（干扰人独白） | **333** |
| **NEG 正确拒** | n_spk=1 + `rejected=True`（thr0.27 拒对了） | **299** |
| **NEG 漏拒（HARD）** | n_spk=1 + `rejected=False`（干扰人独白被 thr0.27 漏放） | **34** |

`stno_target_ratio` 与 `target_active_ratio` 字段数值完全相同（mean abs diff = 0.0），下文统称 `stno`。

---

## 2. 信号区分力（核心）

### 2.1 TEST 1：通用区分（POS n_spk=1 ALL 543 vs NEG n_spk=1 ALL 333）

| 信号 | POS median (IQR) | NEG median (IQR) | AUC [95% CI] | 最佳 F1 阈值 | 阈值下 P / R |
|---|---|---|---|---|---|
| **max_sim** | 0.337 (0.216–0.442) | 0.106 (0.033–0.189) | **0.862 [0.836, 0.886]** | thr=0.151 | P=0.807, R=0.869 (TP=472, FP=113) |
| max_sim（高 P 档） | — | — | — | thr=0.336 | P=0.935, R=0.506 (TP=275, FP=19) |
| **stno_target_ratio** | 0.058 (0.045–0.075) | 0.093 (0.070–0.126) | **0.221 [0.191, 0.252]** （反向） | 翻转 AUC=0.779 | — |
| target_active_ratio | 同 stno | 同 stno | 0.221 | — | — |

**关键观察**：
- **max_sim 在通用层面确有强区分**：AUC 0.86，POS median 0.337 vs NEG median 0.106，肉眼可见 gap。NEG 落入 POS IQR 的只占 18.9%。
- **stno 反向**：NEG 干扰人独白的 target 活跃比例（median 0.093）反而**高于** POS target 独白（0.058）。机制：target 嵌在 babble 里，Personal-VAD 受 babble 干扰反而不认；干扰人是干净近讲，被 PVAD 误判为 target-like。这本身是个"audio cleanness"代理信号，翻转后 AUC 0.78，与 max_sim 相当。

### 2.2 TEST 2：operational recovery（POS 误拒 147 vs NEG 正确拒 299）

> 这才是 verification fusion 的真实战场：thr0.27 已经把样本拒了，fusion 想用第二信号救回 pos。

| 信号 | POS median (IQR) | NEG median (IQR) | AUC [95% CI] | 最佳 F1 op |
|---|---|---|---|---|
| **max_sim** | 0.163 (0.114–0.217) | 0.096 (0.024–0.157) | **0.725 [0.678, 0.769]** | thr=0.107: P=0.475 R=0.782（TP=115, FP=127） |
| max_sim（R≥0.5） | — | — | — | thr=0.159: P=0.540 R=0.551（TP=81, FP=69） |
| **stno_target_ratio** | 0.058 (0.047–0.075) | 0.093 (0.070–0.125) | **0.220 [0.182, 0.257]** （翻转 0.780） | — |

**关键观察**：
- **max_sim 在已拒子集里掉到 AUC 0.725**：因为按构造两组都 <0.27，max_sim 已经被"用掉"了大部分判别力，剩余信息有限。
- 单 max_sim 信号救回 pos，最好 F1 档 precision 仅 0.475–0.540 → 每救 1 个 pos 拖 ~1.6 个 neg 漏拒，**远未到 break-even 0.763**。

### 2.3 融合（max_sim + 翻转 stno）—— 有正交信息但破不了精度

**Fusion score** = `α·z(max_sim) + (1−α)·(−z(stno))`，z 是全样本 z-norm。

**通用区分（543 vs 333）**: best α=0.55, AUC **0.910** CI[0.890, 0.929]，比单 sim 0.862 涨 0.05。
**Operational（147 vs 299）**: best α=0.60, AUC **0.846** CI[0.809, 0.875]，比单 sim 0.725 涨 0.12。

→ **stno 确实提供正交信息**（融合显著涨 AUC），但 AUC 不等于 net score。

---

## 3. 净分影响（决策性）

**评分公式**：`Overall = 0.5·(1 − CER_pos) + 0.5·RR_neg`，POS 池 1364，NEG 池 474。

**Break-even 推导**（每救 1 pos vs 漏 1 neg）：
- 救 1 pos（CER 1.0→~0.1）：`ΔCER_overall = +0.9/1364 = +0.000660`
- 漏 1 neg（拒→放）：`ΔRR_overall = −1/474 = −0.002110`
- Break-even: `FP/TP = 0.9·474/1364 = 0.313` → **precision 必须 > 0.763 才 net 正**

### 3.1 Operational recovery（在已拒子集救 pos）—— **全阈值 net 负**

fusion α=0.60 PR 扫描典型点：

| thr | TP | FP | Precision | Recall | net Overall Δ |
|---|---|---|---|---|---|
| −0.662 | 139 | 132 | 0.513 | 0.946 | **−0.093** |
| −0.375 | 111 | 67 | 0.624 | 0.755 | **−0.034** |
| −0.160 | 74 | 33 | 0.692 | 0.503 | **−0.010** |
| −0.088 | 63 | 21 | **0.750** | 0.429 | **−0.0014** |
| 0.000 | 46 | 16 | 0.742 | 0.313 | −0.0017 |

**precision 上限 0.75（α=0.6 best 时）< break-even 0.763，任何阈值 net 都是负的**。哪怕用 fusion 把所有 rejected 嫌疑样本重打分，单点的 best 也只是 −0.0014（几乎零）。

### 3.2 重新分类全部 mono（替换 thr0.27，不是只救已拒）

**In-sample 最优**：α=0.55, thr=+0.049 → **net +0.0088**（CER_sum 220.6→196.5，FA=34 不变，d_cer=+0.0176，d_rr=0）。

**但 5-fold hold-out**（fit α+thr on 4/5, eval on 1/5）：

| fold | train_net | **test_net** | test d_cer | test d_rr |
|---|---|---|---|---|
| 0 | +0.0069 | **+0.0004** | −0.0014 | +0.0021 |
| 1 | +0.0106 | **−0.0006** | +0.0051 | −0.0063 |
| 2 | +0.0055 | **+0.0037** | +0.0053 | +0.0021 |
| 3 | +0.0115 | **−0.0023** | +0.0039 | −0.0084 |
| 4 | +0.0085 | **+0.0005** | +0.0116 | −0.0105 |
| **mean** | — | **+0.0003 ± 0.0019** | — | — |

**真值是零**。in-sample +0.0088 是 α 过拟合（约 2× 膨胀）。低于项目 ±0.04 噪声地板（CER 测量噪声，memory `stability-test-launched`），**集成后波动会盖过收益**。

### 3.3 HARD neg 漏拒 34 条——内容对称死结（与 multi-signal rescue 同因）

34 条干扰人独白被 thr0.27 漏放，是 fusion 想"额外拒掉"以博收益的唯一通道：

| sim 分桶 | 数量 |
|---|---|
| 0.27–0.35（刚过线） | 19 |
| 0.35–0.50 | 13 |
| ≥0.50 | 2 |

**stno<0.07（在两个信号上看着都像 target）的 HARD neg：10/34**——这些是任何 speaker 信号都救不了的"本质像 target"。

**HARD neg transcript 抽样**（触目惊心）：
- `打开烟机二档` / `关闭图灵烟机，关闭图灵烤箱` / `打开图灵烟机，打开图灵烤箱` / `观影模式` ← **干扰人说了完整家居指令**
- `我回家了` / `太冷了` / `过来，兄弟们` ← 短促指令型

→ 与 memory `multi-signal-reject-rescue` 同一死结：干扰人声学+内容都像 target，speaker 信号（包括更强 encoder）都分不开。

**理论上限**：即使完美拒掉 34 HARD neg（且不误拒任何 pos），`ΔOverall = +0.5·34/474 = +0.036`（≈ +1.8 分）。但 fusion sweep 显示，把 FA 从 34 降到 8（thr +0.488）同时把 CER_sum 推 196→312，net=**−0.006**；CER-RR 的杠杆（pos 池 1364 vs neg 池 474，neg 池小 2.9×）让任何"为救 pos / 拒 neg 而调 thr"的操作都亏。

---

## 4. 强 speaker 信号可跑性评估

| 候选 | 现有资源 | 是否需重训 | 预期能否破精度天花板 | 评估 |
|---|---|---|---|---|
| **帧级 enrollment 匹配**（max/percentile 帧 sim） | wespeaker ResNet34 已在 enroll_infer，可改 per-frame | 否 | **不能**。帧级 max 会挑最干净帧，target 嵌 babble 最干净帧仍 babble；neg 干扰人干净近讲整段都"像 target"。HARD neg 10/34 已在两信号上都像 target，帧级不会救这 10 条 | 不投 |
| **ECAPA-TDNN hidden + PLDA** | speechbrain 在 `code/.venv`（CLAUDE.md 提"speechbrain lazy 修复"） | PLDA 需说话人标注数据；A 集禁训（memory `lessons-pitfalls`§14），需外部集 | **不能**。预训练 ECAPA+cosine 性能 ≈ ResNet34；PLDA 需训外部数据，且 PLDA 在 1.8s enrollment + babble 下迁移性存疑 | 不投 |
| **Personal VAD（enrollment-conditioned 帧级 target 活跃）** | **已在管线**（`stno_target_ratio` 就是） | 否 | **已用尽**。融合 stno 把 AUC 推到 0.85，仍破不了精度。hold-out net=0 | 已用 |
| **CAM++ / 多 encoder 融合** | memory `a-set-is-real-recorded`：本机 venv 坏；且 07-23 spk-oracle POC 已证伪 CAM++ 在 Qwen3 下提升 sim≥0.2 帧为 0% | 需修 venv 或装 3D-Speaker | **不能**。CAM++ 在重 sim<0.2 死区提升为 0；本探查测的是 n_spk=1 mono（非死区），但 HARD neg 漏拒 34 已是"sim≥0.27 像 target"，CAM++ 大概率也判像 target | 不投 |

**结论**：**没有任何可跑（不重训）的强 speaker 信号能突破当前精度天花板**。理由是 §3.3 的 HARD neg 死结——这不是 encoder 不够强，是 enrollment 1.8s + babble + env 失配真磨平了 target 的声纹身份（target sim median 0.21，spk-oracle 早已证伪，memory `spk-oracle-poc`）。

---

## 5. **GO/NO-GO 判定**

### ❌ **NO-GO**

**理由（按重要性）**：

1. **唯一正向证据（in-sample net +0.0088）被 hold-out 打回零**（+0.0003 ± 0.0019），是 α 过拟合。低于 ±0.04 测量噪声地板，集成后波动盖过收益。
2. **Operational recovery 模式 precision 上限 0.75 < break-even 0.763**：所有阈值 net 负。这是结构性的——pos 池 1364 vs neg 池 474 杠杆不对称，每漏 1 neg 的 RR 罚（−1/474）≈ 2.9× 每救 1 pos 的 CER 收益（+0.9/1364）。
3. **HARD neg 34 条死结不可破**：10/34 在两信号上都像 target；含家居指令 transcript（"打开烟机二档"/"观影模式"）——speaker 信号（含更强 encoder）解不开，与 memory `multi-signal-reject-rescue` 同因。
4. **强 speaker encoder（ECAPA/帧级/CAM++）跑得起但破不了天花板**：spk-oracle 已证伪 sim 强化（memory `spk-oracle-poc`），HARD neg 已是"sim 高+stno 低"双重像 target，更强 encoder 只会更高确信"像 target"。
5. **enrollment 1.8s + babble + env 失配真磨平**：target sim median 0.21，整段 cosine 已失效；本探查证明帧级代理（stno）也只到 AUC 0.78（反向）；剩余空间不是"encoder 不够强"而是"信号不存在"。

**判定门槛复盘**（任务 §3）：
- AUC > 0.7？✅（max_sim 通用 0.86，融合 operational 0.85）
- 阈值下 P > 0.8 且 R > 0.5？❌（best F1 档 P=0.75/R=0.43，或 P=0.69/R=0.50；通用层有 P=0.935/R=0.506 但那是 sim 单信号，不能在已拒子集里用）
- → **AUC 过门槛但 PR 不过门槛**，且**net score hold-out 归零**，**NO-GO**。

### 与历史结论对齐

- 印证 memory `spk-oracle-poc`（sim 强化在死区无效）
- 印证 memory `multi-signal-reject-rescue`（内容对称死结，34 HARD neg 含家居指令）
- 印证 memory `cer-ceiling-oracle-fusion-net-negative`（融合 oracle gap 0.019，net 负）
- **CER 腿天花板再坐实一次**（CER 含拒 0.62 短板 = pos 误拒罚 CER=1.0，speaker-aware 路径救不了）

---

## 6. 推荐下一步（NO-GO 后）

| 优先级 | 方向 | 预期收益 | 备注 |
|---|---|---|---|
| 🔴 高 | **Phase-3 端到端 enrollment-aware**（冻结 Qwen encoder 前 1/3 + Sidecar + ASR loss 直接反传，memory `tse-poc-weak-go-overturns-perception-gap`/`tse-phase2-full-nogo`） | 唯一未试的 CER 路径；TSE POC 同环境 ΔCER 显著止跌 | 唯一科学复活路径，与 speaker 信号无关 |
| 🟡 中 | **效率腿 L20 验证**（memory `efficiency-portability-audit`） | 效率 20% 腿 | RTF 口径待问主办方 |
| 🟡 中 | **答辩弹药整理** | — | 含本探查作"speaker 信号已穷尽"诚实归因 |
| ⚪ 低 | 问主办方 w1/w2 权重（memory `unified-thr-decision`） | 调 thr 选点 | 若 RR-heavy 可考虑 thr 上移 |

**speaker-aware verification fusion / reranker / 强 speaker encoder 全部封死，不投 2-3 天**。

---

## 附录 A：方法学

- **AUC**: Mann-Whitney U（手写，含 ties 处理），与 sklearn 等价
- **Bootstrap CI**: 500 次重采样，2.5/50/97.5 百分位
- **5-fold hold-out**: 随机打乱后 5 等分，4/5 拟合 (α, thr)，1/5 eval，固定 seed=123
- **z-norm**: 在全样本上算（无监督统计，可接受）
- **CER 罚则**: rejected pos 计 CER=1.0；accepted pos 计实际 mainline CER
- **net Overall 公式**: `0.5·(ΔCER/1364) + 0.5·(ΔRR/474)`，w1=w2=0.5 假设

## 附录 B：复现脚本

- `code/runs/_speaker_aware_probe.py` — TEST 1/2 AUC + PR
- `code/runs/_speaker_aware_fusion.py` — 融合 α 网格 + in-sample net
- `code/runs/_speaker_aware_holdout.py` — 5-fold hold-out + HARD neg 体检
- 中间数据：`code/runs/_speaker_aware_probe.json`, `_speaker_aware_fusion.json`

---

**状态**: 探查完成，未 commit（任务约束：仅探查不跑模型，结果文件可留作答辩素材，commit 由用户决定）。

---

## ⚠️ 2026-08-05 反瓶颈审计归类（implementation-NO-GO，非 direction-NO-GO）

本探查的 NO-GO 已被 `docs/反瓶颈审计补充_speaker-aware_2026-08-05.md` 正式归类为 **`implementation-NO-GO`**（这套无训练 max_sim+stno 信号融合 / reranker / 强 speaker encoder），**不是 `direction-NO-GO`**——不要改写成"所有 speaker-aware 方法无解"。尚未封顶（`direction-unresolved`）：① Qwen decoder 置信度 / token entropy / no-speech 等 ASR 内部信号；② 真实/同环境非 A 数据的拒识校准器或 target-conditioned verifier；③ target embedding 进 ASR encoder + target activity + ASR loss 联合训练；④ 真实录音域 TSE。详见 memory `speaker-aware-fusion-impl-nogo`。
