# Oracle Speaker 天花板 POC 报告 (2026-07-27)

## 目的

坐实「双人重叠场景 diar 次分是 CER 失败主因」结论后, 量化一个上限:
**在 diar 现有输出里选对 speaker(而非 argmax), 能救回多少 CER?**

- 救得动 → 问题在「选 target 策略」, 改选 target 即可
- 救不动 → 必须改「分离本身」(diar 模型/前后处理/分离-转写联合)

不重新分离、不换模型, 纯用 diar 现有 speaker 输出做 oracle 上限估计。

## 方法

1. 数据源 `code/runs/poc_qwen_asr_full_result.json` (1350 条 pos, 已有 qwen_cer/sim/ref):
   - 失败组: `sim≥0.4 且 qwen_cer>0.8` → 取前 40 (实际 43, 取 40)
   - 对照组(成功): `sim≥0.4 且 qwen_cer<0.001` → 取前 20
2. 每条样本 (复刻 enroll_infer.py, 不动主线):
   - `diar(recognition)` → speakers + per_spk timelines
   - 每 speaker i `cut_target_timeline(audio, per_spk[i])` 切片(含重叠区) → `runs/_oracle_speaker/slices/cmd_N_spkI.wav`
   - 各 speaker 声纹(复用 `diar._embedding` + `collect_clean_audio` 抽独占帧), 与 enrollment 余弦 → 记录 argmax target_idx
3. 批量 Qwen3-ASR 转写所有切片(`.venv_qwen` 独立进程, `qwen_asr_backend.py --batch-size 16 --seed 42`)
4. 官方口径 CER: `submit_norm = brand_homophone_fix(digit_postproc(to_simplified(text)))` vs ref 同归一, `eval_metrics.CERMetric` 累计池
5. 每条:
   - `argmax_CER` = argmax speaker 的 CER (应≈POC qwen_cer, 交叉验证)
   - `oracle_CER = min(各 speaker CER)` (上限, 假设完美选 target)
   - `Δ = argmax_CER − oracle_CER` (>0.1 = 救回)

**可复现**: seed=42 全程固定; 切片 + uid2text + meta + summary 全落盘 `code/runs/_oracle_speaker/`。

## 数字 (FAIL n=40, SUCC n=20)

### 聚合

| 组 | n | argmax mean CER | oracle mean CER | Δ | rescued (Δ>0.1) | stuck (Δ≤0.1) | rescue_mean_drop |
|---|---|---|---|---|---|---|---|
| FAIL | 40 | **1.216** | **0.850** | **0.367** | **19** (47.5%) | 21 (52.5%) | 0.772 |
| SUCC | 20 | 0.000 | 0.000 | 0.000 | 0 | 20 | — |

SUCC sanity 通过: 成功组 argmax==oracle==0, 选不选都一样(本来 argmax 就对)。

### 救回幅度分布 (FAIL n=40)

| Δ 阈值 | n | % |
|---|---|---|
| Δ>0.0 | 19 | 48% |
| Δ>0.1 | 19 | 48% |
| Δ>0.3 | 16 | 40% |
| Δ>0.5 | 9 | 22% |
| Δ>0.8 | 7 | 18% |

### Oracle 天花板分桶

| oracle_CER | n | 占比 |
|---|---|---|
| >0.5 (救不动) | 31 | **78%** |
| ≤0.5 (救得动) | 9 | 22% |
| ≤0.1 (近完美) | 5 | 12% |

**即使 oracle 选 speaker, 78% (31/40) 的失败样本 CER 仍 >0.5。**

### 交叉验证

`|argmax_cer − poc_qwen_cer|`: mean=0.014, max=0.250 → argmax 逻辑与 POC 一致(误差个别条来自 collect_clean_audio 的 audio_len 用 librosa 估 vs enroll_infer 用 mel 帧数, 不影响结论方向)。

### 说话人结构

- FAIL: 39/40 双人 (97% 双人, 与项目已有结论「失败组 97% 双人」一致)
- SUCC: 19/20 单人 (95% 单人, 与「成功组 82% 单人」一致)

## 关键发现: wespeaker sim 在 19/19 救回案例里**主动误导**

所有 19 个救回案例(Δ>0.1)的模式一致:
- argmax 选的 speaker **sim 更高但 CER 更差** (高 sim = 错误目标)
- 真 target 的 speaker **sim 更低但 CER 接近 0** (低 sim 被埋没)

典型 (per_spk_cer 按 spk0/spk1 顺序):

| uid | n_spk | argmax_idx | per_spk_cer | argmax CER | oracle CER | sims |
|---|---|---|---|---|---|---|
| cmd_2837 | 2 | 1 | [0.00, 2.50] | 2.50 | 0.00 | spk0=0.27 / spk1=0.43 |
| cmd_2595 | 2 | 1 | [0.00, 1.25] | 1.25 | 0.00 | spk0=0.07 / spk1=0.44 |
| cmd_2637 | 2 | 0 | [1.25, 0.00] | 1.25 | 0.00 | spk0=0.59 / spk1=0.10 |
| cmd_2052 | 2 | 0 | [1.00, 0.00] | 1.00 | 0.00 | spk0=0.45 / spk1=0.13 |

→ **wespeaker 余弦 sim 在双人重叠 + 短 enrollment(~1.8s) 场景下有 ~48% 概率反向** (argmax 把干扰人当 target, 真 target 反而低 sim 被弃)。这与项目 `non-voiceprint-target-selection.md` 结论一致(声纹已证伪, 解药在非声纹)。

## 结论 (核心判别)

**两者都有, 但主次分明:**

1. **选 target 策略有真实但 bounded 的空间** (19/40 = 48% 救回, 平均降 0.77, 全量均值 Δ=0.367):
   - 这 19 条 wespeaker sim **主动反向**, 任何不依赖 sim 的 target 选择策略(如 LLM 挑家居指令、Whisper-Sidecar embedding、ASE 选帧)都能救
   - 上限: 全 FAIL 组均值 1.216 → 0.850 (改善 ≈30%)

2. **但 oracle 天花板仍 0.850 (>0.5)** → **改分离本身才是主因**:
   - 78% (31/40) 即使 oracle 也救不动, 双 speaker CER 都高(audio 摧毁)
   - 救得动的 9 条里只有 5 条能到 near-perfect (CER≤0.1)
   - 即使把「选 target 策略」做到完美, FAIL 组仍只到 0.85, 离可用 (<0.3) 远

**给主线的指引**:
- 「选 target 策略」是**值得投的次要杠杆**(无成本改 target_idx 已能拿 0.37 均值改善), 契合 `non-voiceprint-target-selection.md` 优先级 (ASE 选帧 / Whisper-Sidecar)
- 但**不能替代分离改善**: 真正的 CER 突破仍依赖 diar 模型升级或 target extraction 路线重构(原 vanilla+target timeline 即是此类)
- POC 印证了「sim 信号在双人重叠场景不可信」, 任何用 sim 做 target 选择/thr 的链路都有 ~48% 失败 inherent

## 样本 id 清单

### 救回组 (19 条, Δ>0.1)

cmd_2018, cmd_2052, cmd_2075, cmd_2188, cmd_2323, cmd_2347, cmd_247, cmd_2503, cmd_2549, cmd_2595, cmd_2602, cmd_2630, cmd_2637, cmd_2659, cmd_2709, cmd_2766, cmd_2830, cmd_2837, cmd_2890

### 救不动组 (21 条, Δ≤0.1, oracle CER 仍 >0.5 共 31 条)

cmd_2023, cmd_2060, cmd_2070, cmd_2128, cmd_2210, cmd_2214, cmd_2232, cmd_2251, cmd_2270, cmd_2302, cmd_2358, cmd_2456, cmd_2482, cmd_2560, cmd_2607, cmd_2687, cmd_2715, cmd_2798, cmd_2817, cmd_2826, cmd_2921

(其中 cmd_2921 是 FAIL 组里唯一的 n_spk=1 样本, 单人 diar 下无可选, oracle=argmax=1.4)

## 产物

- 主脚本: `code/exp_oracle_speaker_ceiling.py`
- 切片 + 转写: `code/runs/_oracle_speaker/slices/` (100 个 spk 切片 + `_uid2text.json`)
- 元数据: `code/runs/_oracle_speaker/meta.json` (每条 speakers/sims/argmax/per_spk)
- 汇总: `code/runs/_oracle_speaker/summary.json` + `agg.json`
- 本报告: `docs/oracle_speaker_ceiling_A.md`

## 复现

```
code/.venv/Scripts/python.exe code/exp_oracle_speaker_ceiling.py
```

(seed=42, 一次跑完约 16 分钟: 60 条 diar ~15s/条 + 100 切片 qwen batch 7s)
