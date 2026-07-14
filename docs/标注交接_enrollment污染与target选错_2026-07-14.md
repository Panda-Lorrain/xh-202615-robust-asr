# 标注交接：enrollment 污染导致系统选错 target speaker

> **写给：负责人工标注环节的 Agent**（本文件自包含，你无需阅读本次诊断对话）
> **日期**：2026-07-14
> **作者**：诊断 Agent
> **结论先行**：今天定位到一条**此前未被证伪的失败链路**——enrollment（唤醒词音频）本身被污染（目标人与非目标人同时说话），系统从混合声纹出发在 recognition 里 argmax 选成了非目标人，于是转写出非目标人的话。这跟历史"七连受挫"完全不同（那七次都是声纹层尝试），是一个**文本/语义层的新角度**。本文件说明：① 问题机制与数据；② 它对当前标注流程的意义和指导；③ 标注结果可以拿来验证什么。

---

## 1. 一句话问题摘要

**enrollment(kws_X.wav) 音频里同时有目标人和非目标人两人在说话，当前系统对整条 enrollment 提一个混合声纹，话语更长的非目标人主导了声纹，导致系统在 recognition 里把非目标人选成 target，转写出非目标人的话。**

---

## 2. 背景

### 2.1 项目任务（XH-202615 美的目标说话人 ASR）
给定 enrollment（目标说话人念唤醒词，约 1.8s），在带噪（SNR −5~5dB）+ 多人重叠（≤2 人，0–100%）的 recognition 音频里**只转写目标说话人的家居指令、拒识非目标**。评分 = 目标 CER 40% + 拒识率 40% + 推理效率 20%。

数据集 A 为单通道 16k/16bit mono，pos 集 1364 条（测 CER），neg 集 474 条（测拒识率）。enrollment 超短（~1.8s），唤醒词共 20 种，题面强调"复杂交互场景"。

### 2.2 当前推理主线（被诊断的链路）
vanilla Whisper-large-v3-turbo + 声纹切 target timeline：`diar+wespeaker 在 recognition 里选 target speaker → 切 target 的 timeline 段 → vanilla Whisper 转写`。这条路线把 pos CER 从 DiCoW 条件化的 1.25 砍到 0.595~0.664（2026-07-06 Phase 1 突破），但 0.595 仍是天花板，大量失败条 CER>0.5。

### 2.3 标注环节现状（你必须先知道的上下文）
当前已有一套**多人标注分发 + 比对仲裁**工具链（2026-07-10/11 建，commit 至 b377054，已 push）：

| 文件 | 作用 |
|---|---|
| `code/build_annotator_pack.py` | 生成自包含标注包 `code/annot_pack/`（`标注.html` + `pos/` 2168 个音频 + `README.txt`），从 `code/error_analysis_pos_unfull.csv`（1084 条未满分）取数 |
| `code/compare_annotations.py` | 回收 2 份 `annot_<名>.csv` → 一致/分歧/未标比对 → `consistency_report.txt` + `merged_annotation.csv` + `annot_disputes.html`（分歧仲裁界面） |
| `code/build_cer_viewer.py` + `cer_distribution_viewer.html` | 全量 1364 条 CER 分布检视（分档柱状 + 逐条 ref/vanilla 对比 + 听音） |
| `code/annot_pack.zip`（116.6M） | 发给 2 个队员的包（gitignore，不入库） |

**每条音频当前分两块标注**（v2）：
- **① recognition 块**（9 类难点，1-9 快捷键）：音量小 / 语速快 / 语速慢 / babble强 / 重叠 / 英文干扰 / 静音未说话 / 循环幻觉 / 其他
- **② enrollment 块**（8 类难点，鼠标点选）：背景嘈杂 / **有其他说话人** / 唤醒词不清 / 音量小 / 唤醒词截断 / **多人同说** / 静音无有效语音 / 其他
- 两块各有"自然语言标注"自由文本框（词条形容不了时自由描述，回收后交 AI 二次分类）

流程：发包（zip → 2 队员各独立标全量 1084 → 导出 `annot_<名>.csv`）→ 比对（`compare_annotations.py`）→ 仲裁（`annot_disputes.html` → `arbitrated.csv`）→ 最终全集 = merged + arbitrated → 聚类难点分布 → 定模型路线。状态：**工具就绪 + 已分发，等队员标注回收**（参见 memory `multi-annotator-dispatch` 与 `docs/标注分发成果检验_2026-07-11.md`）。

> ⚠️ 本文件**不要求推翻重建这套标注流程**。第 4 节会说明在现有流程上**补什么维度**就能覆盖本次发现——具体补法（改包重发 vs. 用自然语言标注事后二次分类）**待与你（标注 Agent）及用户确认**。

---

## 3. 核心发现

### 3.1 机制链（5 步）

1. **enrollment 被污染**：`kws_X.wav` 里同时有两人——target 念唤醒词（如"小钱小钱"）+ 非目标人念新闻/闲聊（如"重审一天没有生产"）。这是题面"复杂交互场景"的体现，**enrollment 并不干净**。
2. **系统假设 enrollment 是干净单人**：`code/enroll_infer.py` 的 `get_enroll_emb`（约 187–209 行）对**整条 kws 提一个声纹**（`librosa.load` 整条 → `get_emb(w)` 一次成型），**没有做说话人分离、没有用唤醒词定位 target**。架构上隐含"enrollment = 干净单人"前提。
3. **混合声纹被非目标人主导**：enrollment ~1.8s 里唤醒词只占一小段，非目标人的闲聊话语更长、在混合声纹里占比更大。
4. **recognition 里 argmax 选错**：`enroll_infer.py` 约 248–251 行 `sims = torch.stack([...]); target_idx = int(torch.argmax(sims))`——在 recognition 各 speaker 声纹里挑"最像混合声纹"的，于是挑成了话语风格更接近混合声纹的**非目标人**。
5. **切错 timeline → 转出非目标人的话**：沿选错的 target 切 timeline → 拼接 → vanilla Whisper 转写 → 输出非目标人说的内容（如新闻片段），与 ref（target 的家居指令）南辕北辙，CER 往往 = 1.0~2.0。

### 3.2 全量统计数据（pos 1364 条）

来源：`code/out_pos_slices_full.json`（diar 分割结果 + per-speaker sim）+ `code/exp_vanilla_full.json`（vanilla 转写 + CER）。两者按 uid 对齐，按 diar 分出的 speaker 数分桶统计失败率（CER>0.5 视为失败）：

| diar 分桶 | 条数 | 占比 | CER>0.5 失败率 |
|---|---|---|---|
| **单 speaker**（欠分割或真单人） | 551 | ~40% | 25% |
| **多 speaker**（diar 分出 ≥2 人） | 811 | ~59% | **67%** |
| 失败总数 680 条 | — | — | 其中**多 speaker 占 80%** |

> 551 + 811 ≈ 1362（与 pos 1364 的差额是个别 diar 抛错跳过条）。

**关键结论**：主要失败**不是** diar 欠分割（单 speaker 桶失败率仅 25%），而是 **diar 成功分出了 2 人、但 argmax 选错了 target**（多 speaker 桶失败率 67%、占失败总数 80%）。**多 speaker 条是主战场。**

### 3.3 典型案例 cmd_2081（**注意：它是少数特例，不是主流！**）

`pos_pairs_datasetA.json` id=2081：
- `kws_txt`（唤醒词）：**"小钱小钱"**
- enrollment：`datasetA/pos/kws_2081.wav` — target 念"小钱小钱" + 非目标人念"重审一天没有生产"（用户亲耳确认两人同时说话）
- recognition：`datasetA/pos/cmd_2081.wav`，`ref`（应输出）：**"风速调高"**
- 系统实际输出（`exp_vanilla_full.json`）：`vanilla_text` = **"或销售手机的计划"**（= 非目标人的话，选错 target）
- `max_sim` = 0.365，`vanilla_cer` = 2.0
- ⚠️ **但在 `out_pos_slices_full.json` 里，cmd_2081 的 `speakers=[0]`（diar 只分出 1 个 speaker，属欠分割）**，所以它其实是 3.2 表里"单 speaker 桶"那 20% 的少数派。

**之所以挑 cmd_2081 当旗舰案例，是因为它 enrollment 污染可耳听坐实、转写错得最直观（新闻话 vs 家居指令）。但统计上，主流失败不是它这种**——主流是 3.4 那种 diar 分出 2 人、argmax 选错。

### 3.4 更典型的多 speaker 失败样例（**这才是主流，请重点理解这类**）

以下都是 diar 分出 2 人、argmax 选成了干扰人、转出干扰人的话：

| uid | ref（target 应输出） | vanilla 实际输出 | max_sim | 说明 |
|---|---|---|---|---|
| cmd_18 | 关闭灯光 | "我" | 0.058 | 两人 sim 都极低(0.062/0.055)，argmax 勉强选了 0 号，实际两人里哪个是 target 都分不清 |
| cmd_57 | 空调调为十二度 | "所以" | 0.324 | 转出干扰人短语气词 |
| cmd_237 | 打开清香烟机 | "把握上演字" | 0.289 | 转出干扰人的话，与家居指令完全无关 |

这类的共同特征：**输出内容根本不像家居指令，而像新闻/闲聊碎片**——这正是"选错 target、切了干扰人 timeline"的指纹。诊断时把"输出 = 干扰人话"作为可识别信号。

### 3.5 修法方向（**待 POC，不是已解决**）

#### 3.5.1 已否决：唤醒词定位 target

最初设想用 `kws_txt`（如"小钱小钱"）在 enrollment 里定位 target 段。**已否决（2026-07-14）**：`kws_txt` 是**开发集标注**，提交/评测时主办方**只给音频、不给唤醒词文本**，此路在真实评测走不通。

#### 3.5.2 新方向：多声纹 → 多路转写 → LLM 识别家居指令段

核心转变：**把"判断谁是 target"从【声纹 argmax】换成【LLM 凭内容识别家居指令】**——不再靠声纹赌 target，从根上绕过 3.1 的 argmax 选错。流程：

```
enrollment(kws) diar 分 N 个 speaker → 各提干净声纹 {e_1..e_N}
recognition diar 分 M 个 speaker → 各切 timeline 分别转写 → M 段文本 {T_1..T_M}
LLM(llm_reject) 判断每段 T_j 是否家居指令
当前考题: 保留唯一家居指令段(=target 的话), 拒非指令段
未来扩展: 多个家居指令段都识别(路由到不同机器)
```

**考题规则（用户 2026-07-14 阐明）**：当前考题保证 recognition 里**只有 target 说家居指令**、非目标说非家居指令（新闻/闲聊）——所以"LLM 识别家居指令段 = 识别 target 的话"在当前考题下可靠（消除了"非目标也说家居指令导致 LLM 分不清"的担忧）。未来主办方可能让多人各说不同家具的指令（如"拉窗帘"+"打开洗衣机"），**均需识别并路由到对应机器**——架构从"挑唯一指令段"泛化到"识别所有指令段"为此预留。

**为什么是新角度**：历史"七连受挫"全部是声纹/音频层尝试——langfix / STNO / SE-DiCoW / enroll-augment / wespeaker-oracle / SepFormer / CAM++ 真 POC——都被 POC 证伪（参见 memory `spk-oracle-poc`）。**没有一次是从内容/语义层（用 LLM 判文本是不是家居指令）入手的**，且本方向复用已有 `code/llm_reject.py`。enrollment 污染 → argmax 选错 target 这条链路，恰好不在那七次的攻防范围内。

**收益上限（诚实标注）**：本方向本质 = "全 speaker 转写、LLM 挑家居指令段"，其 oracle 版已被 `code/exp_spk_oracle.py`（2026-07-08）测过——作弊全 speaker 转写挑对的 CER **0.607** vs 当前 argmax **0.788**，Δ ≈ −0.18。⚠️ 这是 **2026-07-08 的历史 POC 数据**，且是 **vanilla 转写器**在 **sim<0.2 死区 60 条样本**上的数字（不是全量 1364）。两个诚实警告：
1. 0.607 仍 > 0.5 不及格——即便选对 target，极重 babble 仍常把 target mel 毁了，转写器照样翻车（2026-07-11 A2 听音修正进一步把它归因为"vanilla 转写器 OOD 地板"，部分死区条音频其实可辨但 wespeaker 声纹代理失败，换 Qwen3-ASR 能把死区压到 0.459）。
2. 所以新架构 POC 的**合理预期**是：吃掉"argmax 选错 target、漏转 target 那段"（Δ 向 0.18 靠拢），但**解决不了"选对了但 mel 毁了"那部分**——后者要靠转写器换型（候选 Qwen3-ASR / FireRedASR，见 `docs/前沿探索报告_2026-07-10.md`）。

---

## 4. 对人工标注的指导 / 意义（**本交接的核心**）

> 下面每条都基于第 2.3 节盘点到的现有标注流程；调查不到具体落地细节的，标"待与标注 Agent 确认"。

### 4.1 enrollment 块：必须区分"target 念唤醒词" vs "非目标人在说什么"

**现状**：现有 enrollment 8 类难点里已有"**有其他说话人**"和"**多人同说**"——这两个标签能检测到"enrollment 里不止一人"，**但分不清谁是 target、谁是干扰人**。本次发现的核心痛点正是这个角色区分缺失。

**指导**：
- 标 enrollment 时，**不能假设它干净**。听到两人同说，除勾"有其他说话人/多人同说"外，务必在**自然语言标注**里写清楚：
  - **谁在念唤醒词**（= target，例如"前半段女声念了'小钱小钱'"）；
  - **非目标人在说什么/什么时段**（例如"全程有男声在念新闻'重审一天没有生产'"）。
- 这层信息是新架构（多声纹 → LLM 路由）的**直接训练/验证信号**——要知道 enrollment 里每个 speaker 是谁、各说了什么，才能验证"enrollment diar 分多声纹"这步靠不靠谱、以及非目标人会不会也说家居指令。
- 如果标注包还能改（队员是否已开工？**待与你确认**）：建议给 enrollment 块加一个显式 checkbox"**enrollment 被污染（target+非目标人同在）**"，并在自然语言框提示写"target 时段 / 干扰人时段 / 干扰人内容"。
- 如果标注已在进行不能改包：退而求其次，回收后用自然语言标注做**事后二次分类**（标注流程原本就设计了"自然语言 → AI 二次分类"的回路，这里正好用上），把"有其他说话人/多人同说"的子集再细分出 target/干扰人角色。

### 4.2 recognition 块：记 target 真实指令 + 干扰人说了什么

**现状**：recognition 9 类难点偏向描述"为什么 target 指令转不出来"（音量/babble/重叠/英文/幻觉…），但**没有显式的"系统选错 target / 转出了干扰人的话"标签**——而这恰恰是失败主战场（多 speaker 选错占失败 80%）。

**指导**：
- 标 recognition 时，除了勾难点，**重点听并记录三件事**：
  1. **target 的真实指令**（即 ref，复核 ref 是否正确——有些 ref 本身可能存疑，发现请备注）；
  2. **干扰人说了什么**（尤其是当系统 vanilla 输出 = 干扰人话时，记下干扰人原话）；
  3. **每个 speaker（含干扰人）的话是不是家居指令**——新架构靠 LLM 凭"是不是家居指令"挑 target，这层标注是它的训练/验证数据，也用来坐实"非目标都不说家居指令"的考题假设（见 4.4）。
- 若 `vanilla_text`（系统输出）明显是新闻/闲聊、与家居指令无关（如"或销售手机的计划"、"把握上演字"），**高度怀疑是选错 target**，请在自然语言标注里写"**疑似选错 target，输出=干扰人话**"。这条信号能让你事后直接从标注里捞出"选错 target"子集，不用再回去重听。
- 同 4.1，如果包可改：建议加一个 recognition checkbox"**疑似选错 target（输出是干扰人话）**"；不可改则走自然语言二次分类。

### 4.3 优先级：盯住多 speaker 失败条

- 多 speaker 桶失败率 67%、占失败总数 80%——**这是标注性价比最高的子集**。
- 单 speaker 桶（如 cmd_2081 这种 diar 欠分割）只占失败的 20%，且其中相当部分是"target 唯一但 mel 被 babble 毁了"（memory `spk-oracle-poc` 单 speaker 控制组 CER 0.436），**不是选错 target 的问题、唤醒词定位救不了**——别在这类上花过多标注力气去区分 target/干扰人（它本来就只有一人）。
- **建议**：用"档筛选"功能优先标"死区 CER>1"档且 `out_pos_slices_full.json` 里 `speakers` 有 ≥2 个的条——这正是"diar 分出多人 + 选错 target"的高密度区。

### 4.4 标注结果能用来验证什么（回收后的产出）

1. **enrollment 污染 prevalence**：全量 1084 条里多少 enrollment 听到 ≥2 人？占比？（新架构 Step 1 多声纹分离的前提——污染普遍才值得做。）
2. **考题假设坐实（关键）**：非目标人是否真的**都不说家居指令**？用户阐明当前考题如此，但需标注确认——这是新架构"LLM 识别家居指令段 = 识别 target"可靠性的根基。若发现非目标也说家居指令，架构要回退声纹辅助。
3. **每个 speaker 内容 + 是否家居指令**：recognition 各 speaker 说了什么、哪段是家居指令——新架构 LLM 路由的训练/验证数据，也是评估"全 speaker 转写 + LLM 挑"能拿回多少分的直接靶子（对照 oracle 0.607）。
4. **系统选 target 准确率**：标注里"疑似选错 target"条数 / 多 speaker 总条数 = argmax 选错率，对照 2026-07-08 oracle POC 的 66.7% 选对率看全量是否吻合。
5. **干扰人话内容聚类**：干扰人都在说什么？（新闻/闲聊/数字？）——若同质，可衍生内容过滤拒识（呼应 `content_gate`）。

---

## 5. 关键文件清单（字段含义速查）

| 文件 | 用途 | 关键字段 |
|---|---|---|
| `code/pos_pairs_datasetA.json` | pos 1364 条配对清单（题目原始数据索引） | `id`（=cmd/kws 编号）/ `enrollment`（kws_X.wav 绝对路径）/ `recognition`（cmd_X.wav）/ `ref`（target 应输出的家居指令）/ `kws_txt`（唤醒词文本，如"小钱小钱"——⚠️ 仅开发集有、提交拿不到；唤醒词定位修法已否，此字段现仅用于开发期核对 target 身份） |
| `code/out_pos_slices_full.json` | 全量推理的 diar + 切片结果 | `recognition` / `enrollment` / `speakers`（diar 分出的 speaker id 列表，长度=1 是单 speaker/欠分割，≥2 是多 speaker）/ `sims`（各 speaker 与 enroll_emb 的余弦相似度字典）/ `target_idx`（argmax 选中的 speaker）/ `max_sim` / `stno_target_ratio` / `target_active_ratio` |
| `code/exp_vanilla_full.json` | 全量转写与 CER（vanilla vs dicow 对照） | `uid`（cmd_X）/ `ref` / `max_sim` / `vanilla_text`（系统实际输出）/ `dicow_text` / `vanilla_cer` / `dicow_cer` / `rtf` |
| `code/enroll_infer.py` | 被诊断的推理主入口 | `get_enroll_emb`（约 187–209 行，整条 kws 提一个声纹，无分离/无唤醒词定位——**机制根因**）；约 248–251 行 `argmax(sims)` 选 target（**误选发生处**） |
| `code/error_analysis_pos_unfull.csv` | 1084 条未满分音频清单（标注包取数源） | `uid` / `recognition_path` / `enrollment_path` / `ref` / `vanilla_text` / `vanilla_cer` / `max_sim` / `档`（CER 分档）/ `rec_sec` |
| `code/build_annotator_pack.py` / `compare_annotations.py` | 标注包生成 / 2 人比对+仲裁 | 见第 2.3 节 |
| `code/exp_spk_oracle.py`（历史，2026-07-08） | oracle 作弊选 target 的 POC | 产出 oracle_CER 0.607 vs argmax 0.788（死区 60 条样本，vanilla） |

---

## 6. 下一步（新架构 POC，待验证）

1. **等本标注交接落地**：先确认 4.1/4.2 的补法（改包重发 or 自然语言事后二次分类）——**待与标注 Agent 及用户确认**。
2. **新架构 POC**（不属于标注环节，列此只为让你知道标注数据将喂给什么）：
   - 思路：对多 speaker 失败条，**把 recognition 所有 speaker 都转写**（不只转 argmax 选的那个），用 `code/llm_reject.py` 判断每段是否家居指令，保留家居指令段，看 CER 变化。
   - 预期：吃掉"argmax 选错 target、漏转 target 那段"（Δ 向 0.18 靠拢）；不解决"选对了但 babble 毁 mel 转崩"（归转写器换型，候选 Qwen3-ASR/FireRedASR）。
   - 验证用本标注产出：4.4 第 2/3 条（考题假设坐实 + 每 speaker 指令性）作为评估靶。
3. **诚实边界**：oracle 0.607 是死区样本历史数据、仍不及格；新架构是**未被证伪的方向**但**不是已验证的解**。任何"问题已解决"的表述都是过度承诺。

---

## 附：调查核实记录（供你复核）

以下事实已被本次调查用工具逐一核对，引用时可直接信：
- `pos_pairs_datasetA.json` id=2081：`kws_txt="小钱小钱"`、`ref="风速调高"`、enrollment/recognition 路径 ✓
- `out_pos_slices_full.json` cmd_2081（约 10915 行）：`speakers=[0]`（**单 speaker，欠分割**）、`max_sim=0.3648...≈0.365`、`target_idx=0` ✓
- `exp_vanilla_full.json` cmd_2081（约 4453 行）：`vanilla_text="或销售手机的计划"`、`vanilla_cer=2.0`、`max_sim=0.365` ✓
- `enroll_infer.py` 187–209 行 `get_enroll_emb`：整条 kws → `get_emb(w)` 单次提声纹、无分离、无唤醒词定位 ✓；248–251 行 `argmax(sims)` 选 target ✓
- 多 speaker 样例 cmd_18（speakers=[0,1]，sims 0.062/0.055，ref"关闭灯光"→"我"）、cmd_57（ref"空调调为十二度"→"所以"）、cmd_237（ref"打开清香烟机"→"把握上演字"）均核对一致 ✓
- oracle 0.607 / argmax 0.788：源自 memory `spk-oracle-poc` 记载的 2026-07-08 `exp_spk_oracle.py` 死区 60 条样本 vanilla 数据，2026-07-11 A2 听音修正将其归因调整为"vanilla 转写器 OOD 地板" ✓

未独立核实、沿用户给定的事实（请你回收标注后用真实数据复核）：3.2 节全量分桶统计（551/811/25%/67%/680/80%）来自 `out_pos_slices_full.json + exp_vanilla_full.json` 的聚合，本次只抽核了 cmd_2081 等单条，未重跑全量分桶脚本。
