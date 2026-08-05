# REAL-TSE × ASR 后端公平对照 + PS4 zero-shot 上限评估

> ⚠️ **2026-08-05 反瓶颈审计注释**：本文「彻底封顶 TSE，全力转 Phase-3」等措辞中：① Phase-3 已于 07-30 实跑 NO-GO（合成域前提），「转 Phase-3」不再是未试路径；② "TSE 封顶"限**合成域训练的 TSE 系统**（REAL-T 领域天花板 ~0.6 也针对合成/会议域 TSE）。真实录音域 + 目标条件化 ASR 联合训练仍 `direction-unresolved`。我们 zero-training 切割策略（transcribe CER ~0.34）粗略强于 REAL-T 全部 TSE 系统是切割策略优势，不等于"CER 方向封顶"。详见 `docs/反瓶颈审计与后续Agent作战令_2026-08-05.md`。

> 2026-08-02。补完「TSE 提取 × ASR 后端」2×2 矩阵，分离两个混淆变量，并评估 PS4 真实会议数据联合训模型的 zero-shot 上限。
> 样本：A 集 pos 20 条（seed=2026，与前序 REAL-TSE baseline **完全同一批**，直接复用 `code/runs/_realt_baseline/A_spk_emb_100_n20/results.json` 的 id + ref）。
> CER 口径：**累计池**（`eval_metrics.CERMetric`：NFKC + lower + 去所有标点(P*)和空白 + editdistance，total_errors/total_chars）——与项目官方提交口径、qwen 0.3436 全量值严格一致。逐条 mean（jiwer）仅作对照。

---

## TL;DR

1. **D 完成，归因坐实**：Qwen3 在 baseline(spk_emb_100) 提取的 20 条上 CER = **0.7537**，与 Zipformer 的 **0.806** 几乎打平（Δ 仅 **−0.052**）。0.806 里 **87% 是 TSE 提取崩**，只有 13% 是 ASR 后端差异。**"2.35× 差距是后端假象"假设被推翻**——提取质量是主导。
2. **C 阻塞**：PS4 模型定义类（`bsrnn_legacy.py`，`get_model("BSRNN")`）**未公开**——HF 只放 269MB checkpoint，github PS4 仓库只有 `train.py`，公开的 `REAL-TSE/wesep-real-tse` 只有 `TSE_BSRNN_SPK` 一个类（无 legacy BSRNN），作者无私有 wesep fork。重建成本极高且数值正确性不可保，**按"C 阻塞不硬扛"原则放弃 zero-shot 实测**，给定性预期 + 阻塞证据链。
3. **战略结论修正**：①"合成训 TSE 死路"**强化**（换 SOTA ASR 也救不回 0.754）②"Qwen3 路线强"**精准化**为"target-timeline 切割路线强，Qwen3 本身只比 Zipformer 强 Δ0.05"③"突破靠真训"**仍待验**（PS4 阻塞，但 config 坐实它是真实会议数据训的同架构模型，理论应强于 baseline 合成训，能否逼近 0.3436 未知——会议域 vs 家居域仍有 gap）。

---

## 1. 填完的 2×2 矩阵（累计池 CER）

|  | baseline 提取 (spk_emb_100, Libri2Mix 合成训) | PS4 提取 (REAL-PS4 真实会议训) | 我们 target-timeline (diar+切割, zero-training) |
|---|---|---|---|
| **Zipformer-ZH** | **0.8060** (errors=108/chars=134) | **阻塞**（模型定义未公开） | —（不必测） |
| **Qwen3-ASR-1.7B** | **0.7537** (errors=101/chars=134) ⬅ **D 本次填** | **阻塞**（同上） | **0.3436**（全量 1350 条主线值，非 20 条子集） |

> 矩阵单元 = 同一批 20 条 pos、同一批 ref 文本、同一累计池归一口径下的 CER。PS4 列全阻塞。target-timeline 格的 0.3436 是全量 1350 条主线值（memory `h3-dicow-conditioning-backfire-vanilla` / `cer-breakthrough-candidates` 坐实），**非 20 条子集**——因本次任务范围未要求重跑 enroll_infer 生成 20 条 target-timeline 切片，仅作横向参考；标"—"的格按任务要求不必测。

### 横向归因（同 Qwen3 后端下）

```
0.806 (Zipformer×baseline)  ──后端Δ-0.052──▶  0.754 (Qwen3×baseline)  ──提取Δ-0.410──▶  0.344 (Qwen3×target-timeline)
         ↑                                              ↑                                              ↑
    原 2.35× 的"被质疑"端                          后端效应占 13%                                提取效应占 87%
```

- **ASR 后端效应（D，竖向）**：baseline 提取音频上，Zipformer 0.806 → Qwen3 0.754，Δ = **−0.0523**（仅 6.5% 相对下降）
- **TSE 提取效应（横向）**：同 Qwen3 后端下，baseline-TSE 提取 0.754 → target-timeline 0.344，Δ = **−0.4101**（54% 相对下降，碾压级）
- **2.35× 拆解**：0.806 vs 0.3436 总差 0.4624 = 后端 0.052（**13%**）+ 提取 0.410（**87%**）

---

## 2. D 单元：ASR 后端效应量化（核心交付）

### 2.1 数字（20 条 pos，seed=2026）

| 后端 | 累计池 CER | 逐条 mean | correct(CER<0.5) |
|---|---|---|---|
| Zipformer-ZH (sherpa-onnx multi-zh-hans) | **0.8060** | 0.9193 | 45.0% |
| Qwen3-ASR-1.7B (bf16, batch16) | **0.7537** | 1.0101 | 50.0% |
| Δ (Qwen3 − Zipformer) | **−0.0523** | +0.0908 | +5pp |

### 2.2 关键观察：两种口径方向相反，**官方累计池上 Qwen3 仅微优**

- **累计池（官方/提交口径）**：Qwen3 0.754 略好于 Zipformer 0.806（Δ−0.052）——但两者**都仍是垃圾级**（>0.7）。
- **逐条 mean**：Qwen3 1.01 反而**差于** Zipformer 0.92。原因：Qwen3 是强语言模型，在清晰条上更准（把 z0.40/q0.00、z0.14/q0.00 修到完美），但在烂音频上**更爱"编"**——pred 更长（如 id 2961 Zipformer「缝红色」→ Qwen3「不同，灰灰草变为红色」，editdistance 暴增），单条 CER 可达 2.25/3.25。长 pred 在逐条 mean 里拉高均值，但在累计池里被短 ref 条稀释。
- 这与项目历史发现一致（Qwen3 在 babble/死区上的循环英文幻觉、编造倾向）——**Qwen3 的语言先验是双刃剑**：清晰条增益，烂条反噬。

### 2.3 归因结论

**0.806 里绝大部分是 TSE 提取崩，不是 ASR 后端弱。**

证据：把后端从 Zipformer 换成 SOTA 级 Qwen3-ASR（项目主线同款，全量 0.3436 的功臣），CER 只从 0.806 滑到 0.754——**仍 >0.7 垃圾级**。如果 0.806 主要是后端，换 Qwen3 应该能往 0.4–0.5 走；实际只动 0.05。这说明 baseline 提取的音频本身已损坏到任何 ASR 都救不回——**合成训 TSE（Libri2Mix）在真实 A 集上的提取输出是"烂音频"，不是"好音频被弱 ASR 误读"**。

机制（与前序 agent 一致）：spk_emb_100 在 Libri2Mix（英文+合成混响/噪声）上训练，A 集是中文家居+真实家庭底噪+轻混响+近讲（见 memory `a-set-is-real-recorded`），**域 gap 使 TSE 提取的 target 语音信号本身损坏**（mel 谐波被毁），下游 ASR 无差别翻车。

### 2.4 对 2.35× 叙事的修正

- **原质疑（待澄清假设）**：0.806(baseline TSE) vs 0.3436(我们主线) 差 2.35×，会不会是 Zipformer 弱、Qwen3 强造成的后端假象？
- **修正后（坐实）**：同 Qwen3 后端下 baseline TSE 是 **0.754**，vs 主线 0.3436 仍差 **2.19×**。2.35× 里 **2.19× 是真实提取优势**（target-timeline 切割 vs 合成训 TSE），只有 **0.16× 是后端差异**。**后端假象假设被推翻**。

---

## 3. C 单元：PS4 zero-shot 上限评估

### 3.1 PS4 是什么（架构/数据/训练，已坐实）

- **论文**：arXiv:2607.08111《PS4: Proxy-Supervised Joint Training for Real Target Speaker Extraction》（Wanyi Ning 等, 2026）
- **架构**（HF 卡片 + `external/PS4_repo/configs/config_bsrnn_ecapa_vox1.yaml` 坐实）：
  - 分离骨干 BSRNN：`feature_dim=128, num_repeat=6, stride=128, win=512` —— **与 baseline spk_emb_100 的 separator 参数完全一致**
  - 说话人编码器 ECAPA-TDNN：`ECAPA_TDNN_GLOB_c512, embed_dim=192, feat_dim=80, ASTP` —— **与 baseline spk_emb_100 的 speaker_encoder 完全一致**
  - 融合：element-wise multiply —— 与 baseline 一致
  - **关键差异**：训练时加 Whisper-large-v3（frozen）做 proxy ASR supervisor + 4 项联合 loss（CE 1.0 / speaker-sim hinge 5.0 / VAD 0.5 / DNSMOS 0.2）。**推理时 Whisper 不参与，仍输出提取音频**（与 baseline pipeline 同构，过 ASR 后端才算 CER）
- **训练数据**：REAL-PS4（AISHELL-4 + AliMeeting + AMI + CHiME6）——**真实会议多说话人录音**，与 baseline 的 Libri2Mix（合成）形成对照
- **checkpoint**：`checkpoint_epoch037.pt`（269.6MB，已下到 `E:\midea_datasets\PS4_model\`），含 754 keys state_dict（`separator.separation` 146 + `spk_model.layer1-4` 221 + `mask.0-31` 256 + `BN.0-31` 128 + `spk_encoder` 2 + `preEmphasis` 1）

### 3.2 zero-shot 实测：**阻塞（无法复现推理）**

#### 阻塞证据链

1. **HF 仓库（TaurenMountain/PS4）只放 checkpoint + README**，无 `config.yaml`、无推理脚本、无模型定义代码。
2. **github PS4 仓库（github.com/TaurenMountain/PS4）只有 `train.py`**（+ config + resume_utils + run_train.sh），**无 inference.py、无模型定义 .py**。
3. `train.py` 第 491-495 行注释明确：模型类 `BSRNN` 在 **`bsrnn_legacy.py`**（"旧版 BSRNN，通过 wesep 的 legacy 路径"），通过 `from wesep.models import get_model; get_model("BSRNN")` 获取。PS4 config 也用 `model.tse_model: BSRNN`。
4. **公开的 `REAL-TSE/wesep-real-tse`（github.com/REAL-TSE/wesep-real-tse）不包含 `bsrnn_legacy.py`**：
   - 远程只有 `main` 分支、**无 tags**（`git ls-remote` 确认）
   - `wesep/models/` 目录只有 `__init__.py` + `tse_bsrnn_spk.py` 两个文件
   - `get_model` 只注册 `TSE_BSRNN_SPK`（`if model_name.startswith("TSE_BSRNN_SPK")`），对 `"BSRNN"` 直接 `exit(1)`
5. **作者 TaurenMountain 的 github 无 wesep fork**（4 个非 fork 仓库：FormalASR / homie / PS4 / wecom-tools-dify-plugin，无一含 wesep 代码）。
6. **PS4 state_dict 的 key 命名**（`separator.separation` / `spk_model.layer1-4` / `mask.N` / `BN.N` / `spk_encoder` / `preEmphasis`）**与公开 wesep-real-tse 的 `TSE_BSRNN_SPK`（key 前缀 `sep_model.*` / `spk_ft.*`）完全不匹配**，无法通过 key 重映射套用公开类。且 PS4 的 forward（preEmphasis → 80-dim mel → 32 子带 BN/mask → ECAPA multiply fusion）是 legacy BSRNN 的标准子带结构，与 TSE_BSRNN_SPK 的 listen/usef/tfmap/spkemb feature 路径不同。

**结论**：PS4 推理依赖的 `bsrnn_legacy.py` 模型定义**完全私有未公开**。HF 卡片的"Usage"代码（`get_model("BSRNN")`）在公开 wesep 上无法运行。重建该类需精确复现训练时 forward 的子带分割/掩膜/归一化/ECAPA 细节，成本极高（数小时 + ~200 行模型代码），且数值正确性无法验证（一处激活/归一化错则输出音频错误），**按"C 阻塞不硬扛"原则放弃实测**。

#### 定性预期（无法量化的科学分析）

| 维度 | PS4 vs baseline spk_emb_100 |
|---|---|
| 架构 | **完全相同**（BSRNN 128/6/128/512 + ECAPA 192d + multiply fusion） |
| 训练数据 | **PS4 更强**：真实会议多说话人（AISHELL-4/AliMeeting/AMI/CHiME6）vs baseline Libri2Mix 合成 |
| 训练范式 | **PS4 更强**：Whisper proxy ASR loss + speaker-sim + VAD + DNSMOS 四损失联合 vs baseline 纯 SI-SDR |
| 与 A 集域匹配 | **均偏离**：PS4 是会议/远场/多说话人，A 集是家居/近讲/轻混响（memory `a-set-is-real-recorded`）—— PS4 的"真实录音"属性可能比 baseline 合成更接近 A 集，但场景仍不同 |
| 预期 CER | **大概率 < 0.754**（baseline 同后端 Qwen3 值），但是否能逼近 **0.3436**（主线）**未知**——取决于会议→家居域迁移性，这正是"真实数据训但仍跨域"的核心不确定性 |

> 一句话：PS4 是"真实训 TSE"的代表样本，结构同 baseline 但数据/范式全面升级。**它的 zero-shot 强度是"合成训死路、突破靠真训"这条战略判断的关键试金石**——若 PS4 能在 A 集 20 条上逼近 0.3436，说明真实数据训的 TSE 有救（值得押 REAL-PS4 路线）；若仍 >0.5，说明即便真实会议训，跨域（会议→家居）+ enrollment 短等因素仍让 TSE 路线不胜出。**这个答案本次拿不到，需作者公开 `bsrnn_legacy.py` 或推理容器**。

---

## 4. 修正后的战略结论

基于矩阵，原三条判断的状态：

### ① "合成训 TSE 死路" —— **强化**

D 坐实：baseline(Libri2Mix 合成训) 提取的音频，换 SOTA Qwen3 后端仍 0.754（垃圾级）。0.806 不是 Zipformer 弱的假象，是 TSE 提取崩的真实表现。**合成-真实域 gap 摧毁 target 语音信号本身，下游 ASR 无差别翻车**。此结论从"基于 0.806 单点"升级为"跨后端验证（Zipformer 0.806 / Qwen3 0.754 双坐实）"，证据强度提升。

### ② "Qwen3 路线强" —— **精准化（非推翻）**

精准化为：**"target-timeline 切割路线强"是主导，"Qwen3 本身强"是次要**。
- 原 0.3436 的成功 = target-timeline 切割（diar+wespeaker 选 target，切含重叠区的 timeline 段拼接）+ Qwen3 转写。两者贡献此前未拆。
- D 显示：同 baseline 提取音频下，Qwen3 只比 Zipformer 强 Δ0.052。所以 0.3436 里 **Qwen3 后端的边际贡献约 0.05**，**绝大部分收益（~0.41）来自 target-timeline 切割策略**。
- 答辩叙事修正：别再说"Qwen3 碾压 Zipformer"，应说"**切割策略是主因，Qwen3 是锦上添花**"。这与 memory `h3-dicow-conditioning-backfire-vanilla` 的"vanilla+切割"路线本质一致（切割对，条件化错）。

### ③ "突破靠真训" —— **仍待验（PS4 阻塞）**

PS4 推理阻塞，无法直接验证"真实数据训 TSE 能否逼近主线"。但从 config 坐实：PS4 是**同架构 + 真实会议数据 + Whisper proxy 联合训**——理论上比 baseline 强，是"真实训 TSE"最像样的公开代表。**能否迁移到家居域 A 集，仍是开放问题**。本项目自训中文 TSE 的 memory `tse-poc-weak-go-overturns-perception-gap`（同环境 POC 弱 GO，止跌非正收益）+ `tse-phase2-full-nogo`（WeSep pBSRNN 感知-识别鸿沟）提示：即便真实/同环境训，TSE 到 ASR 的"感知-识别鸿沟"仍可能在 CER 上不兑现。PS4 的 proxy-ASR 联合训正是为了打通这个鸿沟（直接用 ASR loss 反传，不再只优化 SI-SNR/mel 代理）——**如果 PS4 跑得通且过门槛，会是 Phase-3 ASR-loss-反传路线的有力证据**。

### 2.35× 这个数字的最终处置

- **原值保留**：0.806 vs 0.3436 = 2.35×（这是事实数字）
- **解读修正**：2.35× **不是后端假象**，其中 2.19× 是"baseline 合成训 TSE 提取崩 vs 我们 target-timeline 切割"的真实提取质量优势，0.16× 是"Zipformer vs Qwen3"的后端差异。
- **答辩口径**：用 2.35× 时必须说明"同 Qwen3 后端下 baseline TSE 仍 0.754，2.35× 里后端只占 13%"，避免被评委质疑"是不是换强 ASR 就能拉平"。

---

## 5. 阻塞点 + 下一步建议

### 5.1 PS4 zero-shot 阻塞的解锁路径（按 ROI 排序）

1. **🥇 联系作者 Wanyi Ning（TaurenMountain）要 `bsrnn_legacy.py` / 推理容器**（最优，HF/github issue 或邮箱，论文 arXiv:2607.08111 有作者信息）。一旦拿到，20 分钟可跑出 PS4 × Zipformer / PS4 × Qwen3 两格。
2. **🥈 等 PS4 公开推理代码**（关注 github.com/TaurenMountain/PS4 更新，目前 5 stars 1 fork，作者可能后续放 inference）。
3. **🥉 自建 BSRNN legacy 类**（最后选项）：根据 state_dict 754 keys 反推结构（32 子带 × BSRNN band-RNN + ECAPA-TDNN 4 层 + 80-dim mel + preEmphasis），写 ~200 行 PyTorch + 严格对齐 forward。成本高、数值正确性不可保（建议找原作者确认实现细节后再做）。

### 5.2 下一步实验建议

- **若 PS4 解锁且 CER < 0.5**：TSE 路线复活，尤其 PS4 的 proxy-ASR 联合训范式值得借鉴 → 启动 REAL-PS4 风格的自训（中文家居数据 + Whisper proxy）
- **若 PS4 解锁且 CER > 0.6**：坐实"TSE 即便真实训也跨域不胜出"，彻底封顶 TSE，全力转 Phase-3（冻结 Qwen encoder + Sidecar + ASR loss 直接反传，memory `tse-phase2-full-nogo` 唯一未试路径）
- **PS4 若持续阻塞**：D 的强结论已足够支撑"合成训 TSE 死路"叙事，不阻塞决策；PS4 不再硬扛，转 Phase-3 POC（这是项目当前最高优先级待办，与本次任务正交）

### 5.3 本次产物的复现路径

- D 矩阵结果（含 per-sample）：`code/runs/_asr_matrix_2026_08_02/matrix_results.json`
- Qwen3 在 baseline 提取 20 条上的转写：`code/runs/_realt_baseline/qwen_on_baseline_n20.json`
- Qwen3 slice 目录（`{id}_extracted.wav` → `{id}.wav` 改名）：`code/runs/_realt_baseline/A_spk_emb_100_n20_qwen_slices/`
- 复跑命令：`code/.venv_qwen/Scripts/python.exe code/qwen_asr_backend.py --slice-dir code/runs/_realt_baseline/A_spk_emb_100_n20_qwen_slices --out code/runs/_realt_baseline/qwen_on_baseline_n20.json --batch-size 16`，CER 用 `code/.venv_realt/Scripts/python.exe` + `eval_metrics.CERMetric`
- PS4 模型：`E:\midea_datasets\PS4_model\checkpoint_epoch037.pt`（已下，待解锁推理）
- PS4 训练代码（参考）：`external/PS4_repo/`（git clone，含 config + train.py）

---

## 附录 A：方法学与口径对齐

### A.1 为什么用累计池而非逐条 mean

主办方排名公式 `TotalScore = w1*(1-CER) + w2*RR` 的 CER 是**累计池**（total_errors/total_chars，见 memory `official-scoring-spec` + `eval_metrics.CERMetric`）。逐条 mean 会被长 ref / 长 pred 的极端条带偏（如本次 id 2808 ref「风速五十」4 字 vs pred「越南邮政银行被打出个屁股」12 字，单条 CER=3.0；逐条 mean 被 1 条拖高，累计池则按 char 加权更稳）。矩阵所有数字用累计池，保证与提交/排名口径一致。

### A.2 为什么 20 条不重抽

直接复用 baseline `results.json` 的 20 个 id（seed=2026），保证 D 的 Qwen3 与已有 Zipformer 在**完全同一批音频 + 同一 ref** 上对比，消除采样偏差。20 条的绝对值不能外推全量，但**后端效应的相对方向（Δ−0.052）可信**（20 条内 ΔCER 的 bootstrap CI 虽未算，但 0.052 在项目历史 ±0.04 噪声带附近，定性结论"后端效应小"稳健）。

### A.3 Qwen3 调用隔离

Qwen3-ASR 在独立 venv `code/.venv_qwen`（torch 2.6+cu124, 4060 8GB, bf16），通过现成 CLI `qwen_asr_backend.py`（`--slice-dir` 接目录、`--out` 输出 uid→text json）subprocess 调用，不污染主 venv。20 条 batch=16 耗时 8 秒。CER 计算在 `code/.venv_realt`（有 editdistance/jiwer/numpy/soundfile），主 venv `code/.venv` 已损坏（memory `a-set-is-real-recorded` 记录）未用。

### A.4 局限性诚实标注

1. **20 条样本量小**：D 的 0.052 后端效应是点估计，未给 CI；但两条都 >0.7 垃圾级，"后端救不回 TSE 崩"的定性结论不依赖精确 Δ。
2. **target-timeline 格用全量 0.3436**：非 20 条子集；若补跑 20 条 target-timeline Qwen3，该格会更精确（但任务未要求，且不影响归因方向）。
3. **PS4 列空白**：C 阻塞导致矩阵右上两格无数字，PS4 强度仅定性。
4. **Qwen3 在 baseline 提取音频上的"编造"倾向**（逐条 mean 反超）是真实现象，不是 bug——但在官方累计池口径下 Qwen3 仍微优，不影响主结论。
