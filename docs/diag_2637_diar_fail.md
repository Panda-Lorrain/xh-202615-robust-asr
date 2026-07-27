# cmd_2637 DiariZen diar 失效诊断报告

**日期**: 2026-07-27
**样本**: `datasetA/pos/cmd_2637.wav` (2.48s, 男女双人并行, 音量相当)
**任务**: 查清 spk0/spk1 都混人(用户原话: spk0 含"就已经进行过一轮交"+"哺乳期"两段内容)是**参数问题还是 diar 能力问题**。

## TL;DR 结论

**既不是参数问题, 也不是 diar 能力问题——是"下游用 FULL timeline 转写"的下游 bug**:
- DiariZen **正确分出 2 个聚类**, 独占段 wespeaker emb 余弦 **0.219(高度可分, 远<0.3)**;
- **14 组参数(default / force nspk2 / ahc 0.3~0.7 / Fa×2 / Fb×3 / AgglomerativeClustering×3)输出完全相同**——embedding 可分时聚类是 parameter-invariant 的;
- "spk0 混两人"是 **enroll_infer.py 用 full timeline(含 53% overlap 帧)送 ASR** → 在重叠区 ASR 拿到两人混合 mel → 转出混杂内容;
- _oracle_speaker 实验(用同样 diar 切分 + 各 spk 单独转写)证实 **spk0="就已经进行过一轮教育", spk1="哺乳期要少吃什么"(=target ref)**, 内容确实分开了;
- 修法不在 diar 参数, 在**下游改用 exclusive-only 切片 + 短段 ASR 兜底 + (可选)重叠区源分离**。

---

## 步骤1: 默认 diar 全套诊断

**DiariZen 默认 config** (VBxClustering, ahc=0.6, Fa=0.07, Fb=0.8, min=1, max=20, lda=128):

| 指标 | 值 | 解读 |
|---|---|---|
| 检测聚类数 | **2** | 题目保证 ≤2 人, 正确 |
| spk0 独占段 | 0.76s | [(0.01-0.35), (2.05-2.57)] 拼接, 纯一人 |
| spk1 独占段 | 0.38s | [(0.67-1.05)], 纯另一人 |
| **总独占段** | **1.14s / 2.48s = 46%** | **不是全程并行**(用户假设被推翻) |
| 重叠帧占比(active) | **53.7%** | 两人同时 active 的帧占一半 |
| 重叠帧占比(total) | 53.2% | |
| **spk0 vs spk1 wespeaker emb 余弦** | **0.219** | **高度可分**(<0.3 通常是不同人, 男女声铁定远低于此) |
| enr 余弦匹配 | spk0=0.585, spk1=0.101 | argmax 选 spk0 当 target |

**timeline 详情**:
- spk0: [(0.01, 0.67), (1.05, 2.57)] — 总 2.18s, 独占 0.76s
- spk1: [(0.35, 2.05)] — 总 1.70s, 独占 0.38s
- 两人 timeline 在 [0.35-0.67] 和 [1.05-2.05] **大量重叠**

**关键观察**: diar 给出的 spk0/spk1 timeline 本身就**互相重叠**——这正是重叠语音的预期行为(EEND 模型预测每帧多 speaker active 概率, 两聚类在重叠帧都被激活)。这不是 diar "没分开", 而是 diar 在老实报告"两人同时在说"。

### 独占段 vs 全 timeline 转写对照(whisper-large-v3-turbo)

| 段 | 时长 | whisper 转写 |
|---|---|---|
| spk0_exclusive | 0.76s | "就已经能叫" |
| spk0_full | 2.09s | "就已经进行过一轮教育" |
| spk1_exclusive | 0.38s | "72"(太短, ASR 退化) |
| spk1_full | 1.70s | "进行过一会儿" |
| recognition_original | 2.48s | "就已经进行过一轮较" |
| enrollment(kws_2637) | 1.82s | "两大夺冠热门加特林和博尔特"(幻觉) |

**对照 _oracle_speaker 实验历史结果**(同样 diar 切分, 各 spk 单独 qwen 转写):
- cmd_2637_spk0 → "**就已经进行过一轮教育**"
- cmd_2637_spk1 → "**哺乳期要少吃什么**" ← 完全等于 ref target!

**铁证**: diar 分出的两个聚类内容**确实不同**, spk1 才是真正的 target 内容(与 ref 完全一致), spk0 是非 target。

### 主线 enroll_infer 当时的输出

```
target_idx=0 (spk0), max_sim=0.585, transcript="就已经进行过一轮教育", ref="哺乳期要少吃什么"
```

主线选 spk0 当 target → 转写拿到非 target 内容 → CER 极高。但**这并不是 diar 把两人混进 spk0**, 而是:
1. diar 报告 spk0 timeline 含大量与 spk1 重叠的帧;
2. enroll_infer 把 spk0 的整段 timeline(含重叠区) 拼接送 whisper;
3. whisper 在重叠区只听到 louder 的非 target 声 → 转出"就已经进行过一轮教育";
4. **真正的 target 内容"哺乳期..."在 spk1**(其独占段仅 0.38s, 太短 ASR 抓不到)。

---

## 步骤2: 14 组参数扫描全部 identical

| config | n_spk | excl_s | ovl%act | cos_min | cos_max |
|---|---|---|---|---|---|
| default | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| force_nspk2 (min=max=2) | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| ahc_threshold=0.3 / 0.4 / 0.5 | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| ahc=0.3 + nspk2 | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| Fa=0.15 (2×) | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| Fb=1.5 (~2×) | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| Fa=0.15 + Fb=1.5 + nspk2 | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| Fb=0.3 (~×0.4) | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| Fb=0.3 + nspk2 | 2 | 1.14 | 0.537 | 0.219 | 0.219 |
| **AgglomerativeClustering** ahc=0.3 / 0.5 / 0.7 | 2 | 1.14 | 0.537 | 0.219 | 0.219 |

**14 组参数全部输出 identical**——连 1 帧的边界都没变。

**原因**: VBx/AHC 聚类在 wespeaker embedding 空间做。当两个聚类的 emb 余弦 0.219(非常远), 任何合理的阈值/方法都会得到完全相同的 2 聚类分配。聚类参数只在 emb 边界模糊(cos 接近阈值)时才起作用; 这里 emb 已经彻底分开了, 调参无效。

**强制 min=max=2 也不变**: 因为 default 已经检测出 2 类(min=1, max=20 的搜索空间也收敛到 2)。

---

## 步骤3: 结论

### 既不是参数问题, 也不是 diar 能力问题

**diar 是对的**, 三条铁证:
1. 检测出 2 个聚类(题目说 ≤2 人, 数量正确);
2. 独占段 emb 余弦 0.219(男女声本来就极易分, 这个数正常);
3. _oracle_speaker 实验中 spk0 / spk1 内容确实分开(非 target / target 完美对齐 ref)。

**用户"spk0 混两人"的感知来源**: 听 / 转写 spk0 的**完整 timeline**(2.18s, 含 1.42s 与 spk1 重叠的帧)。重叠区两人都在说, ASR 拿到的是混合 mel, 自然产出混合内容。但 spk0 的**独占段**(0.76s)是纯净的单人音频。

### 真正的失效点在下游

**enroll_infer.py 的失败链**:
1. 用 `diar_out.label_timeline(s)` 拿到 spk 完整 timeline(含重叠帧);
2. `cut_target_timeline` 把整段拼接送 whisper;
3. whisper 在重叠帧转出 louder 的非 target 内容 → CER 高;
4. 而 argmax(enr_sim) 选 spk0 (sim=0.585) 是因为 enrollment 的 wake word "小美小美" 听起来更接近 spk0(可能是噪声/音量/唤醒词发音差异, 也可能 spk1 独占段 0.38s 太短 emb 不稳)。

### 修法(都不是调 diar 参数)

| 路线 | 描述 | 可行性 |
|---|---|---|
| **A. exclusive-only 切片** | enroll_infer 改用 spk 的独占段(non-overlap) 送 ASR, 避开重叠污染 | 简单, 但 spk1 独占仅 0.38s ASR 退化(已验证转出"72"), 丢 53% 重叠区信息 |
| **B. 短段 ASR 兜底** | 独占段 <1s 时用 CTC/CTC+attention 模型(Qwen3-ASR / FireRedASR)对短切片更稳 | 中, 已有 qwen 后端 |
| **C. 源分离兜底重叠区** | 对 spk0 与 spk1 的重叠时间区跑 SepFormer / TF-GridNet 提出 2 路独立波形再分别 ASR | 重, 但对症(已记于 memory non-voiceprint-target-selection, 死区证伪但**双人重叠子集未测**, 应优先在这子集重测) |
| **D. speaker-aware ASR** | Whisper-Sidecar(emb 进 cross-attn)直接条件化 enrollment emb, 不依赖 diar 切 | 训练成本高(memory 路线🥇) |
| ❌ **E. 调 diar 参数** | 已验证 14 组全 identical, 无效 | **不要做** |

### 与历史 memory 的对照

- `overlap-is-cer-failure-rootcause` (2026-07-26) 说双人重叠是 CER 失败主因, 这里 2637 单条样本完全坐实;
- `non-voiceprint-target-selection` 路线🥇(Whisper-Sidecar) 正是为这种"重叠区声纹信号失效"场景设计的——本样本是它的典型受益案例;
- `cer-ceiling-oracle-fusion-net-negative` 说"主战场 22% 选错 target", 2637 是其中之一(spk0/spk1 都对、argmax 选错是另一个问题, 但即使选对 spk1, 独占 0.38s ASR 仍退化)。

---

## 产物清单

- `code/diag_2637_diar.py` — 主诊断脚本(14 config 扫描 + 独占段/重叠占比/emb 余弦/timeline 详情)
- `code/diag_2637_xscript.py` — 对照转写脚本(exclusive vs full)
- `code/runs/_diag_2637_diar/summary.json` — 14 config 完整指标
- `code/runs/_diag_2637_diar/transcription_check.json` — 各段 whisper 转写
- `code/runs/_diag_2637_diar/<config>/spk{i}_exclusive.wav` + `spk{i}_full.wav` — 各 config 下各 spk 的独占段 / 全 timeline 音频(可人工听验证)

## 复现命令

```bash
export HF_HOME=E:/hf_cache HF_HUB_CACHE=E:/hf_cache/hub MODEL_DIAR=E:/hf_cache/diarizen-wavlm-large-s80-md MODEL_VANILLA=E:/hf_cache/whisper-large-v3-turbo
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY
code/.venv/Scripts/python.exe code/diag_2637_diar.py all      # 14 config 扫描
code/.venv/Scripts/python.exe code/diag_2637_diar.py          # 单 default
code/.venv/Scripts/python.exe code/diag_2637_xscript.py       # 转写对照
```

seed=42, 不改主线, 不 commit。
