# AGENT 交接文档 — 美的目标说话人 ASR（XH-202615）

> **交接时间**：2026-07-07（统一 thr 选点 T27 完成 + push；接 2026-07-06 晚 T26 可复现性 + T25 vanilla 集成）
> **下个 agent 读序**：本文件【2026-07-07】段（↓）→ CLAUDE.md → 关键 memory（**unified-thr-decision** / h3-dicow-conditioning-backfire-vanilla / baodi-config-no-llm / reproducibility-hardening / submit-script-verification / official-scoring-spec）→ REPRO_SETUP.md
> **当前 git**：`master` @ `30c479d`（已 push origin，工作区干净）— 最近 commit：T27 统一 thr 选点（scan_unified_thr v2 / run_baodi B 模式 / 守卫 / RESULTS T27 / memory）。更早：89ab62a T26 指针 / 621ffc9 T26 / 58f8bed T25

---

## 【2026-07-07 最新】统一 thr 选点 T27 完成（B 集 A 集模拟 + 5-agent 对抗验证）

### 一句话现状
B 集按 FAQ C9 必须单一 thr → 用 A 集模拟 B 集混合场景选 thr → **统一 thr=0.27**（区间 [0.26,0.29]，bootstrap CI，与 0.28 不可分取 pos 占优）。5-agent 对抗验证修正初版 0.28（0.01 网格伪影）+ 发现 3 critical（守卫拦截/无混合产物/caliber-A 未坐实，前 2 已修代码）。下个 agent 焦点 = **闭环主办方 CER 口径（零成本高杠杆）+ B 集到手跑混合提交端到端**。

### T27 决策摘要
- **统一 thr=0.27**：pos_CER 0.7418 / neg_RR 0.9051 / 总分 46.53（线性估算 (1-CER)×40+RR×40）。分 thr oracle 52.99（A 集用了 label，B 集不可达）→ 损失 6.46 分 = B 集必须统一 thr 的代价
- **核心张力**：pos sim<0.4 占 79% / neg sim≥0.4 仅 1.5% → pos/neg sim 在 [0,0.4] 严重重叠 → thr=0.28≈pos sim 中位数拒掉~47% pos 换 RR。纯 thr 调参破不了
- **稳健证据**：bootstrap CI IQR=[0.27,0.28] / 真压力 max 损失 2.59（sim 扩张 α=1.2）/ 双口径收敛 / 跨噪声类型一致

### 已修代码（本 session，commit `30c479d` 已 push origin）
- `code/scan_unified_thr.py` v2：bootstrap CI(B=400)+0.005 细扫+诊断(cer_text>1 占 9.0%/丢弃数/neg 漏拒)+真压力(方差缩放+重尾)+修 split_oracle(neg 真 oracle 0.45)；产物 `code/scan_unified_thr.json`
- `code/run_baodi.sh`：加 `B|mixed` 模式（统一 thr 默认 0.27，混合 pairs 无 ref）+ export BAODI_OK=1（opt-in 绕守卫）
- `code/submit_infer.py`：守卫报错引导 B 模式（thr=0.27<0.35 不再裸调死路）

### ⚠️ follow-up must_fix（提交 B 集前）
1. 🔴 **闭环主办方口径**（零成本高杠杆）：书面确认 (a) pos 被拒 CER 计法（1.0?额外惩罚?必须转写? — memory official-scoring-spec 待问第1条，**caliber-A 是 thr=0.27 全部价值依托**）(b) CER→分排名还是归一化 (c) CER:RR 权重比是否 40:40（RR-heavy→最优 thr 上移 0.35-0.40）(d) per-sample CER 是否封顶 min(·,1.0)（封顶→thr 下移 0.20-0.25）。未确认前预生成 thr=0 fallback（pos 全转写，口径C contingency）
2. 🔴 **B 集混合提交端到端**：B 集到手 → make_pairs 产无 ref 混合 manifest（pos/neg 不作输入，utt_id 不冲突）→ `bash code/run_baodi.sh B 0.27` → to_submission（cer 空，label 由 thr，final_cer 主办方算）→ 自检整份同一 thr
3. 🟡 **灰区选择性 LLM A/B**（可选高价值）：对 max_sim∈[0.2,0.4]（~30% pos+5% neg）跑 LLM 二次校验，测能否救回 2.87 RR 腿（RTF 0.24→0.35-0.45 仍<1.0）。答辩前应测过避免被追问

详见 RESULTS.md T27 + memory `unified-thr-decision`。

### 下个 agent 全景待办（按优先级，合并 T27 follow-up + 历史方向）

1. 🔴 **闭环主办方 CER 口径**（零成本高杠杆，决定最终 thr 值）：见上 follow-up #1 的 4 问。**这是 thr=0.27 全部价值依托（caliber-A: pos 拒=CER1.0 未坐实），未确认前 thr 最终值不定**。已预生成 thr=0 fallback contingency
2. 🔴 **B 集混合提交端到端**（B 集到手后）：见上 follow-up #2。`make_pairs` 产无 ref 混合 manifest → `bash code/run_baodi.sh B 0.27` → to_submission → 自检整份同一 thr
3. 🟡 **灰区选择性 LLM A/B**（可选高价值）：见上 follow-up #3。对 max_sim∈[0.2,0.4] 跑 LLM 二次校验救 RR 腿
4. 🔧 **攻 CER 声纹强化**（本机可跑，研究性）：CAM++ per-speaker / US-PVAD 在 vanilla 路线下复评（先前 CAM++ 证伪 0.191<wespeaker 0.218 是 dicow 路线评的，vanilla 下声纹错→直接转错段权重更高，值得复评），攻低 sim 桶(0.2-0.4 CER 0.6-0.75)timeline 切割
5. 📄 **答辩 FAQ + 演练**：`03_答辩FAQ与风险预案.md` 待写；核心论点 = Phase1 vanilla 反 cascaded 突破 + 可复现性工程化 + T27 统一 thr（含 caliber-A 风险诚实披露）+ babble 归因 + 诚实组合主线极限
6. ⚡ **L20 端到端耗时真测**：submit_infer 显存自适应（L20 48G 大 batch）+ 租 AutoDL L40 验证（官方 L20 评效率，本机仅 4060，memory `l20-eval-hardware`）
7. ⚠️ **CER 进一步破局**（大工程，时间充裕再做）：端到端联合训练 X（反 cascaded，出题方偏好）/ SepFormer 提 target mel 再喂 vanilla

> 注：原【2026-07-06 晚】段 follow-up #1"统一 thr 选点"已由 T27 完成（本段）；#2-#5 对应上方 4-7。保底（关 LLM+thr=0.4，A 集分 thr RR 98.5%）仍作 fallback，但 B 集必须统一 thr=0.27。

---

## 【2026-07-06 晚 最新】可复现性改造完成 + FAQ 口径全确认

### 一句话现状
vanilla 路线集成（T25，pos CER 0.667 / neg RR 98.52%）+ 可复现性改造（T26，核查 6 项硬要求全达标，fp16 run-twice delta=0）双完成。下个 agent 焦点 = follow-up（统一 thr 选点 / 攻 CER / L20 真测）。

### FAQ 2026-07-06 已确认口径（所有 thr/路线决策依据，必读）
- **Q1 pos 拒 = CER 1.0**（字符级 Levenshtein，无额外惩罚）；neg 只 RR，pos 只 CER
- **Q2 排名制**（CER40+RR40+效率20 加权排名，不公布归一化公式）
- **Q4 统一 L20-46G**，其他算力内存不限；**batch 默认 1**（允许 batch 须结果一致，RTF 用 batch=1 测）
- **CER = 系统输出 vs 标准答案识别文本**，字符级；主办方过阵子给 CER 计算脚本
- **B 集不预分 pos/neg**（dir1/dir2 混合，结构同 A 但不给识别标签 + 不给 pos/neg 先验）→ **统一 thr，pos/neg 不作输入**（C9）
- **数据增广不限，可引外部样本**（vanilla zero-training 不受影响）
- **提交 JSON**（官方）：`{result:{results:[{id,content,label,cer}],final_cer,duration}}`，id=测试音频名，duration=batch=1 总推理时间
- **核查 = 完整复现结果比对**（非仅看代码），6 项硬要求

### 可复现性改造产物（T26，已 push）
- `code/repro.py`（公共模块：set_global_seed/resolve_model/peak_gpu_mib）+ `tests/test_repro_logic.py`（5 单测 PASS）
- 5 脚本改造（submit_infer/enroll_infer/se_denoise/llm_reject/noise_classify）：import repro + set_global_seed + `--seed` 透传 4 子进程 + 模型 `resolve_model`(env→HF repo id) + 显存日志
- `code/verify_reproducibility.py`（run-twice：limit=10 vanilla **text 一致 100%, CER delta=0**，fp16 确定**无需 fp32**）
- `REPRO_SETUP.md`（部署：DiCoW clone + 模型 HF + DF3 + 种子 + 验证）
- spec `docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md` + memory `reproducibility-hardening`

### 4 模型 HF repo id（代码 default，本地 setenv MODEL_* env override）
`openai/whisper-large-v3-turbo` / `BUT-FIT/DiCoW_v3_2` / `BUT-FIT/diarizen-wavlm-large-s80-md` / `Qwen/Qwen2.5-3B-Instruct`；DF3 例外（GitHub Rikorose/DeepFilterNet + `DF_MODEL_BASE_DIR` env）

### ⚠️ Phase 3 坑（DiCoW submodule 阻塞 → 方案 C）
`.gitignore` `code/*/` 通配的 negation `!code/DiCoW-inference/` 在 git 不生效（`git check-ignore` 确认仍被忽略）→ 切方案 C 文档化（`REPRO_SETUP.md` 写手动 clone `BUTSpeechFIT/DiCoW` + `--recursive` 拉 `Lakoc/DiariZen`/pyannote）。若要自动 submodule，试 `git add -f code/DiCoW-inference`（gitlink）+ 手动 `.gitmodules`。

### 下个 agent 待办（follow-up，按优先级）
1. 🔧 **统一 thr 选点**（B 集混合集必需）：扫统一 thr 优化 CER40+RR40 加权。A 集初评 pos/neg 分开 thr 合规（独立测试集），B 集必须统一。⚠️ **建议向主办方确认"A 集初评可否分 thr"**（合规风险）
2. 🔧 **攻 CER**（声纹强化）：CAM++ per-speaker / US-PVAD 改善低 sim 桶（0.2-0.4）timeline 切割。vanilla 路线下声纹错→直接转错段；先前 CAM++ 证伪是 DiCoW 路线评的，vanilla 下值得复评
3. ⚡ **L20 端到端耗时真测**：submit_infer（vanilla）显存自适应（L20 48G 大 batch）+ 租 AutoDL L40 验证（官方 L20 评效率，本机 4060，memory `l20-eval-hardware`）
4. 📄 **答辩 FAQ + 演练**：`03_答辩FAQ与风险预案.md` 待写；答辩核心 = Phase 1 vanilla 反 cascaded 突破 + 可复现性工程化 + 诚实归因
5. ⚠️ **CER 进一步破局**（大工程）：端到端联合 X / babble 专用源分离（SepFormer 提 target mel）

### 关键 commit（master，已 push）
`621ffc9` T26 / `7b7ca17` DiCoW 文档 / `88d676d` verify / `fbff320` 5脚本 / `1e0ca55` repro.py / `609779c` spec；更早 `58f8bed` T25 vanilla 集成 / `d7b1240` FAQ 待确认

### 保底仍有效（fallback）
关 LLM + thr=0.4（neg RR 98.5% / RTF 0.24，memory `baodi-config-no-llm`）；vanilla 主线 pos CER 0.667。提交用 `code/run_baodi.sh`（默认 vanilla，`BAODI_BACKEND=dicow` 切回）。pos/neg 分开跑（utt_id 冲突）。⚠️ 提交用 run_baodi 锁 flag（裸调默认 flag 灾难，memory `baodi-config-no-llm`）。

---

（↓ §0-§10 是 2026-07-06 白天 Phase 1 突破版历史交接，作参考；**最新状态以上面【晚】段为准**。其中 §6.1 "vanilla 集成最高优" 已完成 T25；§0 "保底为唯一选项" 已被 Phase 1 vanilla + 可复现性 T26 覆盖）

---

## 0. 一句话现状

真实测试集 A 到手 → 全量真测 → 组合主线 cascaded 在极重 babble 下 pos CER ~1.0 是架构极限 → **2026-07-06 Phase 1 突破**：zero-training 改用 **vanilla Whisper-large-v3-turbo + 声纹切 target timeline**（去掉 DiCoW FDDT/STNO 条件化），**CER 1.248 → 0.664 减半**，correct_rate 31%→**46%**，英文幻觉 18.8%→**0.59%**（DiCoW 条件化主动造孽坐实），thr=0.20 overall CER **0.711**（vanilla 终把 overall 拉到 <1，CER 40% 腿从 ~0 分变 ~11 分）。**保底（关LLM thr=0.4）仍备用**（neg RR 98.5% / RTF 0.24），但 Phase 1 给了一条现实破局路线。

> ⚠️ **2026-07-06 Phase 1 突破（覆盖旧"保底为唯一选项"，以下为准）**：H3 强证伪——DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】。sim 分桶铁证：sim[0.2,0.3) vanilla 0.746 vs dicow **1.606** / sim[0.3,0.4) vanilla 0.623 vs dicow **1.523**（Δ-0.90，条件化最反作用）。机制：diar+wespeaker 选 target → 切 target timeline 段（含重叠区）拼接 → vanilla Whisper 转写（去 FDDT）。**英文幻觉根因坐实**：DiCoW 条件化造 18.8% 英文幻觉，vanilla 仅 0.59%——langfix 是治标（打 DiCoW 自己造的孽），vanilla 路线从根消灭（治本）。**zero-training**：无需大工程即可斩获大部分 CER 收益，比端到端联合 X 轻得多。**答辩弹药**：「cascaded 条件化机制在极重 babble 下反作用，改用 target extraction + vanilla Whisper，CER 几乎减半」——契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。完整数据表 / thr 工作点 / sim 分桶 / 产物路径见 `RESULTS.md` T24 + memory `h3-dicow-conditioning-backfire-vanilla`。**P2 最高优 = vanilla 集成 submit_infer**（`--asr-backend vanilla`），把 0.664/0.711 变提交数字。
>
> ⚠️ **2026-07-04 真测保底仍有效（保留作 fallback）**：关LLM vs 开LLM = trade-off（原"全面优于/pos 持平"被审查推翻）—— 关LLM 赢 neg RR **98.5%**>96.2% + RTF **0.24**<1.01（4×）；开LLM 赢 pos 救回（28 条 LLM 救回的 pos 里 **26 条 CER=0.000 完美**）。⚠️**提交默认 flag=灾难 → 用 `code/run_baodi.sh` 锁死**。⚠️**CER 均值是幻觉陷阱**（correct_rate 才诚实）。**thr 待主办方评测口径定**。三档数字 / 归因 / 产物路径详见 `RESULTS.md` T23 + memory `baodi-config-no-llm`。

> ⚠️ **2026-07-04 真测更新（覆盖原"含LLM"保底，以下为准；含 3-agent 对抗审查修正）**：实测三档确认 **关LLM vs 开LLM = trade-off（原"全面优于/pos 持平"被审查推翻）** —— 关LLM 赢 neg RR **98.5%**>96.2% + RTF **0.24**<1.01（4×）；**开LLM 赢 pos 救回**（28 条 LLM 救回的 pos 里 **26 条 CER=0.000 完美**，原"pos 持平"错）。选关LLM = **为效率20%+RR40% 牺牲 pos 救回**（pos 反正架构极限放弃）。⚠️**7 GAP 见 RESULTS T23**：CER ±0.04 噪声（langfix 边际 0.028 不可靠）/ L20 batch=1 未实现 / **提交默认 flag=灾难 → 用 `code/run_baodi.sh` 锁死** / 三路融合证伪（llm_or_sim 是 AND）/ 99%@0.4 高估（实 98.5%）/ neg 漏拒口径未验证 / pos CER 全口径 conceded。⚠️**CER 均值是幻觉陷阱**：thr 升 = 误拒把 babble 幻觉超长样本（CER>>1）换成 CER=1.0，**correct_rate 才诚实**（thr=0.2 correct 31% → thr=0.4 correct 14% 真退化）。**pos CER ~1.0 无 thr 能救**（babble 89% 主导，cer_accepted 0.94，两极分化 9.2% 完美 vs 81.5% 灾难）。**thr 待主办方评测口径定**（CER 均值→0.4 / correct→0.2 / pos 不许拒→0）。三档数字 / 归因 / 产物路径详见 `RESULTS.md` T23 + memory `baodi-config-no-llm`。下个 agent：保底用**关LLM**（`submit_infer.py --no-llm --sim-thr 0.4`），别再走含LLM。

## 1. 当前真实基线（datasetA 全量真测，2026-07-04）

| 评分维度 | 权重 | 当前基线 | 备注 |
|---|---|---|---|
| **pos CER**（1364 条）| 40% | **1.25**（thr=0.2）/ **0.96**（thr=0.4）/ correct_rate 31% | enroll_all 全转写口径 1.40；离目标很远 |
| **neg RR**（474 条）| 40% | **77%**（thr=0.2）/ **99%**（thr=0.4）| sim_thr 调高即可达 99% |
| **效率 RTF** | 20% | **0.16–0.24**（4060，L20 会更快）| pos 全量 13.8min（batch 模式） |

**数据集关键事实**：单通道 16k/16bit mono（**100% 单通道 → DSENet/VSAEC/DOA/KWS 空间路线全弃**）；pos.jsonl id 0–2999 稀疏(n=1364)，neg id 1000–5399 稀疏(n=474)，**id 区间重叠 → pos/neg 必须分两次跑**（utt_id `cmd_N` 冲突）；enrollment ~1.8s 超短；唤醒词 20 种（非固定）。

## 2. 核心认知（最重要，决定一切后续）

**组合主线 cascaded 在极重 babble 下 CER 1.25–1.4 是架构能力极限**，调参/小改破不了。根因双重：
1. **wespeaker+diar 提不出 target 声纹**：pos sim median 仅 0.28（实测 median 0.283/mean 0.286/min -0.125，**sim<0.06 仅 7.7% 非主流**，答辩别引用 sim<0.06），30% <0.2 被误拒
2. **DiCoW mel 退化转写崩**：20% 输出英文（langfix 首位 token 锁不住漂移）+ 锁对 target 的高 sim 子集 CER 仍 0.43

**sim 与 CER 强相关**：sim≥0.5 → CER 0.43 / correct 76%；sim 0.2–0.3 → CER 1.63。**误拒非 CER 主因**（降 sim_thr 不降 CER，因低 sim 样本转写也崩）。**攻 babble 转写质量是唯一出路**，但要根本改进。

## 3. 三方案攻短板验证结论（全受挫，避免重试）

| 方案 | 验证结果 | 处置 |
|---|---|---|
| enroll 加噪增强（`--enroll-augment`）| **证伪**：babble 池用脏 cmd（带噪重叠）污染 emb，sim 反降 0.378→0.362 | 默认关闭，代码留 |
| langfix 加强（英文检测+prompt_ids 重生成）| **边际**：全量英文 31.6%→18.5%，但 CER 仅降 0.028（救回的中文也崩）| **保留**（确定边际收益 + LLM 拒识副作用正面）|
| STNO 放宽（`collect_clean_audio` 弱重叠帧）| **无效**：前 50 未触发（独占充分）；babble 重 50 条 sim 0.024→0.038 仍极低 | 已 revert |
| SE-DiCoW（cross-attn 解重叠）| **架构不兼容**：`mt_num_speakers=2` 多 speaker SCB + self-enrollment 范式，与 enroll_infer 单 target 范式根本不同；短音频 OOD 风险 | 放弃（4.4G 权重已下但用不了）|

## 4. 代码现状（本次 commit 的文件）

| 文件 | 改动 |
|---|---|
| `code/enroll_infer.py` | `--pairs` 批量化（模型加载1次+enroll_emb 缓存，enroll 39→11s）+ langfix（英文检测+prompt_ids 重生成+前缀去除）+ SE-DiCoW 探索分支（保留不触发）+ `_sample_babble`（增强，默认关）|
| `code/submit_infer.py` | Gap3 批量化（`run_enroll_infer_pairs` 单次调用）+ 透传 `--enroll-augment/--aug-noise-dir` |
| `code/apply_dicow_langfix.py` | TARGETS 加 SE_DiCoW/generation.py（死代码 bug 同源重打）|
| `code/eval_datasetA.py` | pos CER（误拒=1.0）/ neg RR + zhconv 繁简归一 |
| `code/make_pairs_from_datasetA.py` | jsonl(中文key) → `--pairs` manifest(英文key+绝对路径) |
| `code/analyze_pos_full.py` | max_sim 分布 + sim_thr 工作点扫描 + CER 分桶 + 分组诊断 |

**实验产物**（`code/*/` 被 gitignore，不入库）：`code/patches/`（三方案 patch + extract 脚本）、`code/out_pos_full/`、`code/out_neg_full/`、`code/patches/enroll_full_langfix.json`（langfix 全量结果）。

## 5. 保底配置 + 执行命令（最高优先级待办）

保底 = **langfix（已 apply）+ 关LLM（`--no-llm`）+ sim_thr=0.4**（neg RR **98.5%** / pos CER **1.0** 架构极限 / RTF **0.24**，2026-07-04 真测三档确认）。⚠️**关LLM 三项全胜开LLM**（neg RR 98.5%>96.2%、RTF 0.24<1.01 4倍快、pos 不变；Qwen2.5-3B 零样本拒识是负贡献），原"含LLM"保底已弃，详见 §0 更新 + RESULTS T23。最终 submit_infer 全量确认数字（**关LLM = sim_only**，命令务必带 `--no-llm`）：

```bash
cd E:/midea_target_asr && source code/setenv.sh && export HF_HUB_OFFLINE=1

# 0) 用户本地放 datasetA/ 后, 先生成 manifest（pos/neg pairs，含 label，gitignore 不入库）
code/.venv/Scripts/python.exe code/make_pairs_from_datasetA.py

# 1) pos 全量（thr=0.4，含 SE+LLM，4060 ~15min）
code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs code/pos_pairs_datasetA.json --out-dir code/out_pos_final --sim-thr 0.4

# 2) neg 全量（thr=0.4，~5min）—— pos/neg 分开跑(utt_id 冲突)
code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs code/neg_pairs_datasetA.json --out-dir code/out_neg_final --sim-thr 0.4

# 3) 评测
code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_pos_final/result.json code/pos_pairs_datasetA.json
code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_neg_final/result.json code/neg_pairs_datasetA.json
# 深度分析 pos:
code/.venv/Scripts/python.exe code/analyze_pos_full.py code/out_pos_full/result.json code/pos_pairs_datasetA.json
```

**关键评测语义待确认**：readme 说 "pos 测 CER"，未明确 pos 被拒算多少。当前 eval_datasetA 对 pos 被拒（空 text）算 CER=1.0。若主办方对 pos 不允许拒（必须转写），则 sim_thr 对 pos 应=0（全转写），只 neg 用 thr 拒。这影响保底 thr 选择，**建议向主办方确认**。

## 6. 待办优先级

1. 🔧 **P2-① vanilla 集成 submit_infer**（**最高优，Phase 1 落地**）：加 `--asr-backend {dicow,vanilla}` 切换，复用 enroll_infer 的 diar+声纹锁定 target timeline，转写用 vanilla Whisper-large-v3-turbo（去 FDDT/STNO 条件化）。把 CER 0.664 / overall 0.711 变成可提交数字。需测 vanilla 路径 RTF（基座 Whisper-large-v3-turbo 同 0.89G，预期与 DiCoW 相近或更快，无 FDDT 开销）
2. 🔧 **P2-② 声纹强化**（中优，攻低 sim 桶）：CAM++ per-speaker / US-PVAD 改善 target timeline 切割。低 sim 桶（0.2–0.4）即便 vanilla CER 仍 0.6–0.75，切割错了 vanilla 也救不回。先前 CAM++ per-speaker 证伪（0.191 < wespeaker 0.218），但那是为 DiCoW 路线评的；vanilla 路线下声纹错→直接转错段，权重更高，值得复评
3. 🔧 **P2-③ 数字 initial_prompt**（低优，锦上添花）：家居指令数字/温度场景（"调到二十六度"），vanilla 路线下可试 prompt（DiCoW 路线 T19 已证 prompt 反伤，vanilla 未测）
4. ⚠️ **P2-④ sim_thr 待主办方评测口径**：CER 均值→thr=0.4 / correct_rate→thr=0.2 / pos 不许拒→thr=0。**Phase 1 改变格局**：thr=0.20 vanilla overall CER 0.711 已 <1，不像 dicow 路线 pos CER 全档 ~1.0 无 thr 能救——vanilla 路线下 thr 选择更宽松
5. 🔧 **保底执行**（fallback，仍备用）：上面命令跑 pos+neg 全量 thr=0.4 关LLM，确认最终 CER/RR 提交数字（保底仍有效，Phase 1 失败时退路）
6. ⚡ **L20 耗时验证**：submit_infer（含 vanilla 后端）显存自适应（L20 48GB 大 batch）+ 租 AutoDL L40 验证端到端（官方 L20 评效率，本机仅 4060，memory `l20-eval-hardware`）
7. 📄 **答辩 FAQ + 演练**：`03_答辩FAQ与风险预案.md` 待写；答辩重点讲故事 = **Phase 1 vanilla 突破反 cascaded（新核心论点）** / babble 归因清晰 / 单通道确认 / 工程优化（Gap3·繁简·langfix）/ 诚实组合主线极限 + 端到端 X 是未来方向
8. ⚠️ **CER 进一步破局**（如要冲，大工程）：①端到端联合训练 X（反 cascaded，出题方偏好，`docs/02_上限候选深读.md`）②babble 专用源分离（SepFormer 提 target mel 再喂 vanilla/DiCoW，同时救 sim+转写）—— Phase 1 已用 zero-training 拿大部分收益，这两条留待时间充裕

## 7. 环境与工具规范（必读，省踩坑）

### venv（3 个独立，依赖冲突不可合并）
- `code/.venv`（主：enroll_infer/noise_classify/DiariZen/DiCoW，torch 2.5.1+cu124）
- `code/.venv_se`（se_denoise/DeepFilterNet3）/ `.venv_llm`（llm_reject/Qwen2.5-3B）
- **Python 一律用 uv**（`uv pip install --python code/.venv/Scripts/python.exe <pkg>`），禁止裸 pip

### 权重（全在 E 盘）
- `E:/hf_cache/{DiCoW_v3_2, diarizen-wavlm-large-s80-md, Qwen2.5-3B-Instruct, SE_DiCoW}`
- `E:/df_cache/DeepFilterNet`
- ⚠️ **DiCoW langfix 补丁**：`code/apply_dicow_langfix.py`（幂等，含 SE_DiCoW 路径）。**HF cache 清/重下必须重跑**，否则 `language="zh"` 失效 → 中文音频 90% 出英文
- ⚠️ **SE_DiCoW modules 补齐**：若重下 SE_DiCoW，trust_remote_code 的 .py 要复制到 `E:/hf_cache/modules/transformers_modules/SE_DiCoW/`（否则 `ModuleNotFoundError`）

### HF 下载（省重试，详见 memory `hf-download-method`）
```bash
source code/setenv.sh && export HF_HUB_OFFLINE=0 && unset HF_ENDPOINT && \
code/.venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('<repo>', local_dir='E:/hf_cache/<name>', max_workers=2)"
```
**根因**：setenv 默认 `HF_ENDPOINT=hf-mirror.com`，但镜像 head 请求 308 重定向回 huggingface.co 主站 → 无代理直连超时秒失败。**解法 = unset HF_ENDPOINT + 保留代理 7897 直连主站**。别再试镜像。

### git
- **身份**：本机 `git config user` 是 `midea-overnight-loop`（loop bot），**手动 commit 必须显式指定**：
  `git -c user.name=Panda_Lorrain -c user.email=2687571939@qq.com commit -m "..."`
- remote: `ghfast.top` 代理 GitHub（push 慢但能成）
- datasetA/ 278M 官方数据**不入库**（已 gitignore）；pos/neg_pairs_datasetA.json（含 label）也不入库

## 8. 踩过的坑（避免重试）

1. **enroll 加噪增强脏池**：用 datasetA/pos/cmd（带噪重叠）作 babble 池污染 enrollment emb，sim 反降。要试需干净 babble 源或更弱增强（SNR 20+），但前景不明（babble 重 sim 极低，增强难救）
2. **SE-DiCoW 多 speaker 不兼容**：`SCBs.py:148 x.view(B//S, S, T, F)` 硬要求 B 是 mt_num_speakers(=2) 倍数；uses_enrollments 是内部 self-enrollment（从 stno_mask target 行自动提），**不接受外部 enrollments kwarg**（实测 generate 报 `model_kwargs not used`）。与 enroll_infer 单 target 范式根本不同
3. **SE-DiCoW config 字段名**：是 `uses_enrollments`（复数 s），不是 sedicow agent 推断的 `use_enrollments`
4. **繁简虚高 CER**：Whisper-large-v3 输出繁体（空調開到），ref 是简体 → 字符级 CER 每个繁体字都算错。`eval_datasetA.py` 已加 zhconv 繁→简归一（冒烟 CER 0.111→0.016）。**最终提交推理输出也要转简体**
5. **评测 key 对齐**：submit_infer 把 recognition 复制为 `utt{N}_{cmd_id}.wav`，eval 时要 `re.sub(r'^utt\d+_', '', uid)` 去 utt 前缀才能匹配 manifest 的 `cmd_{id}`
6. **langfix prompt_ids 前缀**：DiCoW generation.py 的 prompt_ids 会经 batch_decode 复述出来（"以下是普通话的句子。"前缀），必须 decode 后手动去掉（已修）
7. **CER 均值被幻觉扭曲**：babble 重复循环幻觉使 hyp 超长拉高 CER 均值。看可用率用 correct_rate(CER<0.5)，看纯转写质量看 cer_accepted_only

## 9. 关键 memory（~/.claude/.../memory/，跨会话已更新）

- `datasetA-spec` — 数据集规格 + 单通道 + 真测基线 + 三方案验证结论（最全）
- `hf-download-method` — HF 下载正确方法（unset 镜像+代理直连）
- `diacow-language-force-bug` — DiCoW language 死代码 bug + langfix 补丁
- `l20-eval-hardware` — L20 评测硬件 + 耗时验证方案
- `stop-digging-on-sim-data` — 仿真深挖边际递减（真数据已到手）
- `git-identity-mismatch` — git 身份不一致（手动 commit 指定 Panda_Lorrain）
- `adversarial-review-before-milestone-commit` — 复杂诊断归因易错，commit 前对抗审查

## 10. 历史归因基础（T14-T22，详见 `PROGRESS.md` / `RESULTS.md`）

- **T14-T20**：pipeline 搭建（diar+STNO+DiCoW）+ wespeaker 锁定 + CAM++/SE 证伪 + langfix + SE 条件化
- **T22 babble 归因**（2026-07-02/03 仿真集，本次真测印证）：vanilla Whisper 三角定位证 Whisper 基座在 babble 上**不漂英文** → **H3 确证**（DiCoW FDDT/STNO 条件化在 babble 上适得其反）。STNO target 行覆盖率因果主导语言（全程1.0→中文 vs 0.067→英文）
- **2026-07-04 真测印证**：真数据 babble 89% 主导，20% 英文漂移（langfix 首位锁不住）+ sim 极低（锁不住 target），与 T22 仿真归因一致，且更严重（CER 1.25）。**仿真 → 真测，归因闭环**

---

## ⚠️ 隔离声明（沿用，下个 agent 必读）

`docs/superpowers/specs/2026-06-29-final-exam-*` 与 `docs/superpowers/plans/2026-06-29-final-exam-*` 是用户**另一个《Python与数据分析》课程期末作业**，与本项目（美的 XH-202615 参赛）**完全无关**。不要执行那份 plan、不要往「数据分析重构」带方向。对本 agent 视作不存在。

---

**给下一个 agent 的话**：⚠️ **2026-07-06 Phase 1 已破局**——组合主线 CER 1.4 是 **DiCoW 条件化路径**的极限（非任务极限），改用 vanilla Whisper + 声纹切 target timeline（zero-training）CER 已减半到 0.664、overall thr=0.2 拉到 0.711（CER 40% 腿 0→11 分）。**最高优 P2：vanilla 集成 submit_infer（`--asr-backend {dicow,vanilla}`）把 0.664 变提交数字**，无需大工程。保底（关 LLM+thr=0.4，RR 98.5%）仍作 fallback。langfix/STNO/enroll 增强/SE-DiCoW 在 cascaded 框架内试过无效，别重试。所有踩坑见第 8 节。
