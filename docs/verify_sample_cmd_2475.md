# 验收包：cmd_2475（双人重叠 + 成功 CER=0）★ 对照 cmd_2637（双人重叠 + 失败）

> 让用户听「分离成功」长啥样, 和 2637 失败案例对比差异在哪个工位。
> 8 工位齐全 + SepFormer 分离两路 + 与 2637 的差异提示。

---

## 0. 样本选择依据

| 字段 | 值 | 备注 |
|---|---|---|
| **uid** | `cmd_2475` | |
| **ref** | `关掉无风感` | 真值转写 |
| **kws_txt**（唤醒词） | `嗨扫地机` | enrollment 是"嗨扫地机" |
| **poc_sim** | 0.439 | 主线 wespeaker argmax sim, ≥0.4 属"接近解决"桶 |
| **poc_qwen_cer** | 0.000 | qwen 转写 CER=0, 完美 |
| **n_spk** | 2 | diar 分出 2 人 |
| **overlap_rate** | **0.888** | ★ 重叠率 88.8%（极端重叠）|
| **audio_sec** | 2.14s | |
| **scan 范围** | 全 193 个成功候选（sim≥0.4 & qwen_cer<0.001） | |
| **挑选规则** | 双人样本中重叠率最高 | 次高 cmd_2468 overlap=0.720 / cmd_2105 overlap=0.667 |

> 备注：B1 早先判断「成功组重叠>10% 仅 12%」基本属实——193 个成功候选里只有约 30% 是双人（其余单人）；双人里大部分重叠 < 30%。本样本 88.8% 重叠是双人成功样本里**最极端**的，对照价值最大。

---

## 1. 8 工位逐项验收

> 所有产出在 `E:/midea_target_asr/code/runs/_verify_cmd_2475/`

### 工位 0：输入音频
- 做什么：复制原始 enrollment / recognition
- 产出：
  - `enrollment.wav`（1.91s, 唤醒词"嗨扫地机"）
  - `recognition.wav`（2.14s, 重叠双语音频）
- 怎么验：先听 enrollment 记住 target 音色（"嗨扫地机"是谁说的）；再听 recognition，**预期听到两人同时说话、几乎全程重叠**。

### 工位 1：enrollment diar + enroll 声纹
- 做什么：diar enrollment（看是否被污染）+ 从整段抽 enroll_emb（主线用法）
- 产出：
  - `enr_spk0.wav`（enrollment 第 1 个 speaker 段）
  - `enroll_diar.json`（diar 出 1 人, 1.91s, **无 enrollment 污染**）
- 怎么验：听 enr_spk0.wav 是否干净单一说话人；enrollment 没有混入第二人 → 主线 `enroll_emb` 抽自整段 1.91s 是安全的。

### 工位 2：recognition diar（★ 核心：是否正确分出 2 人）
- 做什么：diar recognition → 各 speaker 完整段 + 重叠率
- 产出：
  - `rec_spk0_full.wav`（spk0 全 timeline 拼接, 2.13s）
  - `rec_spk1_full.wav`（spk1 全 timeline 拼接, 1.72s）
- 关键数据：`speakers=[0,1] audio=2.14s total_spk=4.04s overlap=0.888`
- 怎么验：**重点听**——两人各自 full 段是否**确实是两个人**（音色不同）。2637 失败的核心就是 diar 欠分（两人混到同一 cluster），本条 diar 把两人正确分开了（spk0 = "关掉无风感" target, spk1 = 干扰人）。

### 工位 3：各 speaker emb 细节 ★（2637 缺的关键工位）
- 做什么：复刻 enroll_infer.py:252-262 + collect_clean_audio——抽 **speaker 独占非重叠帧**（避开污染）；若独占帧 <0.3s 触发 fallback 全 timeline；不足 1s 触发 np.tile。
- 产出：
  - `rec_spk0_excl_raw.wav` / `rec_spk1_excl_raw.wav` —— **不存在！** 因为 spk0 和 spk1 的独占帧都是 0 秒（overlap=88.8% 没有任何非重叠段）
  - `rec_spk0_emb_input.wav`（**fallback 到全 timeline**, 2.13s）
  - `rec_spk1_emb_input.wav`（**fallback 到全 timeline**, 1.72s）
- 关键数据：
  ```
  spk0: excl_sec=0.00s  fallback=True  tiled=False  emb_input=2.13s
  spk1: excl_sec=0.00s  fallback=True  tiled=False  emb_input=1.72s
  ```
- 怎么验：**听 emb_input.wav** —— spk0 和 spk1 的 emb_input 都包含两人重叠的声音（被互相污染）；这就是 88.8% 重叠下声纹抽取的真相：**没有干净独占帧, 只能从重叠全段抽**。 Wespeaker 仍能在「污染输入」下区分两人（因 diar 把同人的所有帧聚集, 即使穿插对方帧, 主成分仍是本人）。
- ⚠️ 对比 2637：2637 的诊断没存这个工位，所以看不到「emb 是从污染段抽的」事实；本工位把这个隐性环节暴露。

### 工位 4：选 target（余弦 argmax）
- 做什么：各 spk_emb vs enroll_emb 余弦 → argmax
- 产出：`sims.json`
  ```json
  {"sims": {"0": 0.4387, "1": 0.3768},
   "target_idx": 0, "target_speaker": 0, "max_sim": 0.4387}
  ```
- 怎么验：spk0 sim=0.439 vs spk1 sim=0.377 → **argmax 选 spk0**（gap 0.06 不算大但够分）。listen 验证 spk0_full.wav 是否就是 enrollment 那个人（"嗨扫地机"音色）。

### 工位 5：切 target timeline
- 做什么：cut_target_timeline(audio, per_spk[target_idx])（含重叠区拼接 → 喂 vanilla/qwen）
- 产出：
  - `target_slice.wav`（1.42s, **spk0 全 timeline 含重叠区**）
  - `假如选spk1_当target.wav`（对照, 若误选 spk1 会切出什么）
- 怎么验：听 target_slice.wav —— 应该能听到 target 说"关掉无风感"，**但背景里有另一人干扰声**（因为含重叠区）。听"假如选spk1"对照——是另一人的声音。

### 工位 6：ASR（qwen 转写 target_slice）
- 做什么：qwen3-ASR 转写 target_slice（独立 venv_qwen, seed=42）
- 产出：在 `_qwen_slices/_uid2text.json`（与 sep A/B 一起 batch 转写）
- 结果：`raw target text = "关掉无风感。"`
- 怎么验：对比 ref="关掉无风感"——**完美匹配**（仅句末多一个句号, CER≈0）。这就是"成功"的核心证据：尽管 target_slice 含重叠干扰, qwen3-ASR 鲁棒地把 target 指令转对了。

### 工位 7：后处理 4 步 ★（2637 缺的工位）
- 做什么：逐步应用 to_simplified → digit_postproc → brand_homophone_fix, 存中间结果
- 产出：`postprocess_steps.json`
  ```json
  {"raw": "关掉无风感。",
   "to_simplified": "关掉无风感。",
   "+digit_postproc": "关掉无风感。",
   "+brand_homophone_fix": "关掉无风感。"}
  ```
- 怎么验：本句无繁体/数字/品牌同音字 → 4 步全不变。看 json 字段确认每步链路真的跑过（不是 silent skip）。2637 没存这个工位, 无法看到中间步是否生效。

### 工位 8：拒识决策 ★
- 做什么：算 max_sim vs thr=0.27 + content_gate（is_valid_command）
- 产出：`reject_decision.json`
  ```json
  {"max_sim": 0.4387, "sim_thr": 0.27,
   "sim_rejected": false, "content_gate_on": true,
   "is_valid_command": true, "content_gate_reject": false,
   "final_rejected": false,
   "note": "pos 样本期望 final_rejected=False"}
  ```
- 怎么验：**pos 不应被拒**。max_sim=0.439 > thr=0.27 ✓；is_valid_command=True（"关掉无风感"是有效家居指令）✓；final_rejected=False ✓。三条件全过 → 正确 accept。

---

## 2. SepFormer 分离（让用户听分离效果）

- 做什么：用 speechbrain/sepformer-whamr16k 把 recognition 分离成 2 路 → 各路抽 wespeaker emb vs enroll_emb 算 sim → 各路 qwen 转写
- 产出：
  - `sep_sourceA.wav`（sim=0.376, picked=True, transcript=`关掉雨风感。`）
  - `sep_sourceB.wav`（sim=0.135, picked=False, transcript=`在当代建筑，现代。`）
  - `sepformer_result.json`
- 怎么验：
  - 听 `sep_sourceA.wav`：应该是 target 说"关掉无风感"——但转写为"关掉雨风感"（"雨" vs "无" 错一字, SepFormer 分离后仍有微量失真）。
  - 听 `sep_sourceB.wav`：应该是另一人, 内容完全不同（"在当代建筑..."）。
  - **关键观察**：SepFormer 选对了路（sourceA sim 0.376 > sourceB 0.135）, 但分离后音质略损导致 ASR 把"无"误转成"雨"。**主线 cut_target_timeline 不分离直接喂 qwen 反而更准**（"关掉无风感"完全正确）——印证了之前 SepFormer 证伪的结论（分离反伤 CER）。

---

## 3. 与 cmd_2637（失败）的对照 ★

| 维度 | cmd_2475（成功） | cmd_2637（失败） |
|---|---|---|
| 重叠率 | 88.8% | 高（双人对着说不同指令） |
| sim | 0.439 | 主线 sim 也通过 |
| **diar 是否正确分人** | ✅ 分出 spk0/spk1 两人, 各自聚类正确 | ❌ diar 欠分, 把两人混到 spk0, spk1 段都包含 A+B |
| 各 spk emb 来源 | fallback 全 timeline（excl=0s, 但本人主成分仍占主导） | 段内 A+B 混杂, 主成分被另一人稀释 |
| argmax 选 target | ✅ spk0 sim=0.439 > spk1 0.377, 正确选 target | ❌ 选错（选到 louder/更密集的 non-target） |
| ASR 转写 | "关掉无风感" ✅ 完美 | 错（转出另一人指令） |
| CER | 0.000 | 高（错人指令与 ref 完全不沾） |

### 用户该重点听哪几个工位的差异？

1. **工位 2（rec_spk{i}_full.wav）**——最大差异点。
   - 2475：spk0_full.wav 几乎全是 target 一人（穿插少量 spk1）；spk1_full.wav 是另一人。两人能听出来。
   - 2637：spk0_full 和 spk1_full 都同时有 A+B 两人声音（diar 没分开）。
   - **听点**：每条 full.wav 是不是"单一音色为主"。

2. **工位 3（rec_spk{i}_emb_input.wav）★ 补 2637 缺**——暴露 fallback 真相。
   - 2475：excl=0s, fallback 到全 timeline（两人重叠声都进 emb 了）。能听到背景里另一人。
   - 2637：未存这个工位, 看不到 wespeaker 实际抽的是被污染的段。
   - **听点**：emb_input.wav 背景里有没有另一人——有的话说明 emb 是在污染下抽的（这点 2475 也污染, 但主成分够强仍能选对）。

3. **工位 5（target_slice.wav）vs 工位 6（ASR 文本）**——验证"切对了人 → 转对了字"。
   - 2475：target_slice 是 target 说"关掉无风感"（背景有干扰） → qwen 转对。
   - 2637：target_slice 切的是 non-target → qwen 转出另一人指令。
   - **听点**：target_slice.wav 主音色是不是 enrollment 那个人。

4. **SepFormer 两路（sep_sourceA/B.wav）**——验证"分离能否救"。
   - 2475：sourceA/B 分离得相对清晰（但仍损字：无→雨）。
   - 2637：SepFormer 在 2637 重叠场景下基本失效（已证伪）。
   - **听点**：分离后两路是否更干净；但**注意分离反而让 ASR 错字**, 印证主线"不分离直接切 timeline + qwen"更优。

---

## 4. 关键论点（答辩弹药）

- **diar 正确分人是 CER 成功的前置条件**：2475 vs 2637 同样极端重叠, 唯一差别就是 diar 是否把两人分开。**重叠不是失败根因, diar 欠分才是**（呼应 memory `overlap-is-cer-failure-rootcause`）。
- **wespeaker 在 fallback 全 timeline（无独占帧）下仍能正确选 target**：因 diar 已把本人帧聚集为主成分, 即使穿插对方帧, 余弦仍能拉开 0.06+ gap。
- **主线 cut_target_timeline + qwen3-ASR 端到端 CER=0**：在 88.8% 极端重叠下仍 work, 证明 vanilla/timeline 切割路线（H3 反 cascaded）的价值。
- **SepFormer 分离反而损字**：分离后"无→雨", 不如不分离——支持此前 SepFormer 证伪决策。

---

## 5. 复现命令

```bash
source code/setenv.sh
code/.venv/Scripts/python.exe code/verify_sample_success.py
# 自动全扫 193 候选 → 选重叠率最高的 2-spk 成功样本 → 跑 8 工位 + SepFormer
# 产出: code/runs/_verify_cmd_2475/ + 本索引文件
# 单遍 ~3-5 min(diar 扫候选 ~3 min, qwen 转 3 切片 ~10s)
```

输出目录：`E:/midea_target_asr/code/runs/_verify_cmd_2475/`
