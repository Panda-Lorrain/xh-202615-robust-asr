# AGENT 交接文档 — 美的目标说话人 ASR（XH-202615）

> 🔴 **2026-07-27 最新（当前恢复点）**：**消除信息隔阂 + 5 路轻量改进全证伪 + 推翻"到顶" + 战略转向自训中文 TSE**。用户挑战 memory 旧结论"0.3436 物理天花板/到顶"，要求逐环节用真数据核实 + 人耳听样本验证（不靠 memory 二手归因）。本轮 10+ commit 全 push（`fd17e9f` 最新）。
>
> **① 推翻"到顶"**：死区(sim<0.4, 占全量 78.8%) 真地板仅 ~10%（用户人耳听 13 条高 CER 死区样本，全听得清 target）→ 90% 可修，理论空间 **CER 0.3436 → ~0.15**。"mel 摧毁物理地板"归因被推翻（用户方法论纠正：地板判据=人耳能否听清，弃用"mel 摧毁"模糊词）。
>
> **② 真瓶颈链（不是 mel 摧毁）**：diar 分对（spk0/spk1 emb 余弦 0.219 男女可分，内容对）→ **cut_target_timeline 切 full timeline 含重叠区**（物理混合收进切片）→ SepFormer 分离后**选路失效**（sim 选对率 25%，SI-SDR 破坏声纹）。enrollment 污染假设证伪（拆 enroll 各段仍选 spk0）。详见 memory `overlap-is-cer-failure-rootcause`（最全，A/B1/B2/2637诊断/分区POC/multi-voice/死区诊断全链条）。
>
> **③ 得分 70-73**：CER 含拒 0.5934 → 腿 16.26（**短板**）；RR 0.9494（content_gate 扩词典）→ 腿 37.97（近满分）；效率 18-19（估，L20 未真测）= **72-73/100**。CER 腿是唯一提升杠杆。
>
> **④ 5 路轻量改进全证伪（共性：轻量实现证伪，方向 oracle 有效，要兑现得重投入）**：
> - **multi-voice 整体 NO-GO**：主战场 SepFormer 破坏 mel（0.178 vs argmax 0.059）；整体 Δ+0.0035 net neutral；RTF 翻倍
> - **selector fallback NO-GO**：死区 Δ-0.003 零收益
> - **SepFormer 盲分离**：oracle 选路 0.603（分离有效，救回 41%，9 条 CER=0）；sim 选路 1.249（25% 废）
> - **微调 qwen POC 退化**：合成数据域不匹配（Aishell 朗读≠家居/程序重叠≠真实相位/程序小声≠真实小声），Δ+0.147，**死区退化最严重**（与目标相反）；loss 收敛 0.009 但 hold-out 退化（hold-out 救了，没盲目扩 1 万）
> - **TSE 英文权重 zero-shot 中文证伪**：TSELM 6 条 0 救回（HiFiGAN 英文 kmeans 重建中文崩溃）；USEF-TSE 缺包；Whisper-Sidecar **核实无公开权重**
>
> **⑤ multi-voice 证伪诊断（用户质疑触发，澄清归因）**：用户怀疑"两路都家居指令不多"——**对**。"92% 双 accept"是 content_gate 宽松放行率（拒新闻用的），**真"两路都像家居指令"仅 25%**。选路（cmd_score）80% 准，不是大问题。**NO-GO 真因 = SepFormer 破坏 mel（B 类铁证：cmd_146 等 argmax CER=0 → SepFormer 后 0.6+），不是选路**。详见 `docs/verify_mv_fail.md`。
>
> **⑥ 方法论纠正（必守）**：地板判据=人耳可辨度（不用"mel 摧毁"）；hold-out 评测（loss 骗人，POC 必看 ΔCER）；外部训练允许（主办方 2026-07-27 确认，memory `external-training-allowed`）；**派 agent 不亲自跑**（核心负责人模式，memory `core-leader-delegate-mode`）。
>
> **🎧 关键待办（用户交代下一个 agent）**：
> 1. **用户正在听 B 类样本验证**：`code/runs/_verify_mv_fail/B_<uid>/`（cmd_146/2007/2129/277/73）——听 `argmax_target_slice.wav`（主线清晰）vs `sep_sourceA.wav`（SepFormer 破坏后），亲自验证"SepFormer 怎么把好的搞坏"。**听完用户会反馈**。
> 2. **下一步：启动自训中文 TSE**（用户确认要做，是最可执行的重投入）：Aishell1Mix 数据现成（`E:/midea_datasets/data_aishell` 14.5G + `musan_extracted` 10.3G 已下）+ 租算力（L20/A100 几天，用户愿租）+ 攻"分离"治本（绕开 SepFormer 中文 OOD + 选路失效）。**第 1-2 步本机能做**（生成 Aishell1Mix 重叠数据 + TSE 训练脚本，`data_aug_recipe.py`/`build_aishell_manifest.py` 已就绪），**第 3 步租算力训练**。TSE 调研见 `docs/research_speaker_separation.md`（Top5：Whisper-Sidecar 最对症 MIT+中文验证但要自训 / TS-ASR-AD=候选X / USEF-TSE）。
>
> **memory 重点读**：`overlap-is-cer-failure-rootcause`（最全） / `multivoice-content-routing-is-mainline`（multi-voice 全 NO-GO + 真因） / `external-training-allowed`（外部训练允许 + 微调 POC 退化） / `core-leader-delegate-mode`（派 agent 工作方式） / `non-voiceprint-target-selection`（Whisper-Sidecar 等，部分修正）。
>
> **本轮 docs**：`deadzone_diag` / `multivoice_full_validation` / `sepformer_ceiling_B2` / `partition_cut_poc` / `oracle_speaker_ceiling_A` / `separation_ceiling_B1` / `diag_2637_diar_fail` / `fix_cut_boundary` / `tse_poc` / `qwen_lora_poc` / `verify_mv_fail` / `verify_deadzone` / `verify_deadzone2` / `verify_sample_cmd_2475` / `research_speaker_separation` / `qwen_finetune_data_recipe` / `hf_cache_cleanup` / `链路验收清单`。

> 🔴 **2026-07-24 最新（当前恢复点）**：CER 后处理/解码侧探索闭环 — **6 路全证伪 + 规则E品牌同音修复集成（唯一确定净正 Δ-0.001）**。用户诉求"CER 占 40% 大头重点强化"。5-agent workflow（错误模式量化+规则实测+context设计+beam可行性+综合决策）+ 5 GPU 实验。**CER 0.3432（baseline batch 口径，对齐 poc 0.3436）→ 0.3422（规则E集成后）**，天花板坐实（55%=370 条无解幻觉 babble 摧毁 mel，45% 可救但全落 ±0.04 噪声地板）。① context C1 场景+词汇 Δ-0.0067（改善166>恶化69但主战场[0.2,0.4)回退+0.0044→门槛③失败噪声内）/ C2 纯词表 Δ+0.2042 灾难（死区热词回吐 CER 飙124，Issue#106 死区全面爆发坐实）② 零RTF rep_penalty1.1 Δ+0.0006 / no_repeat4 Δ+0.0001（qwen 幻觉是**无关型非循环型**，DiCoW"直击循环幻觉"假设不迁移）③ LLM 精准保守（Qwen2.5-3B+程序裁剪只留严格同音 TONE3 单字替换）Δ+0.0012 改7恶17——**改对7=品牌功能名（智控温/轻干洗/净呼吸/洗衣机筒）=规则E 已覆盖零增量，改错17=3B 过度纠正常用词（制热→智热/顶灯→鼎灯/最高→髙）+繁体+桶筒语境误判**，LLM 后纠正**严格劣于规则E**（3B 无判断"原文是否正确"能力，程序裁剪压不住）。✅**规则E集成** `text_utils.brand_homophone_fix`（美的功能名锚点+ pypinyin TONE3 严格同音，零回退 Δ-0.001 改9恶0，**领域知识非 A集统计合规可提交 B集**）→ enroll_infer L380(vanilla/dicow)+L425(qwen/firered) digit_postproc 后 + requirements 声明 pypinyin==0.55.0。**⚠️教训**：实验前必须验证切片目录=baseline 源（首轮 C1/C2 误用 target_slices_qwen CER0.4384 vs poc 源 target_slices_full 0.3436，白跑20min，诚实 baseline 交叉验证救了结论）。commit `93657ab` + 交接文档 commit。**CER 腿解码侧/后处理彻底 exhausted**，突破只剩更强 LLM(7B+效率代价)/模型替换(已证伪)/A集外训练(违规)。ROI 转效率腿(L20)+答辩腿（6路实证证伪+Issue#106 死区回吐发现=诚实归因硬弹药）。详见下【2026-07-24】段 + memory `cer-improvement-directions`。

> 🔴 **2026-07-23 最新（当前恢复点）**：Qwen3 重跑 CAM++/SepFormer **双证伪**——用户核心假设"是 DiCoW 差不是方法错"**不成立**。① CAM++ 声纹强化 GO=否：Qwen3 oracle（exp_spk_oracle_qwen.py，死区60+主战场60）显示 cer_gain 死区+0.146/主战场+0.106（选 target 是问题，部分推翻 vanilla "oracle 0.607 封闭"），但 miss 中正确 target sim≥0.2 仅 0%/21% → 声纹层提不出正确 target（babble 摧毁 who 信号），CAM++ 救不了。② SepFormer GO=否（更狠）：exp_sepformer_qwen.py（死区40）SepFormer+Qwen3 0.687 vs argmax 0.410（Δ+0.277），oracle 0.413≈argmax → 分离没拎出更干净 target。**新发现**：Qwen3 下选 target 是真问题（cer_gain +0.1~0.15），解药在**非声纹 target 选择**（diar 改进/enrollment-conditioned TSE 联合），留新方向线索。CER 腿 qwen 0.3436 天花板坐实，死区物理极限换更强转写器也救不了。答辩硬弹药。**另本轮还做了**: 目录整理(删~5.5G中间产物 + out_*/json/csv归 code/runs/ rename保留历史 + docs/实验结果汇总.md) / 效率迁移性分析(4-agent workflow: 4060→L20三不统一+家居RTF推算实时阈值0.2-0.3 L20外推0.09-0.12优秀) / 非声纹深探全证伪(ASE关键帧net更差 + CAM++替换wespeaker CER噪声内 + Whisper-Sidecar NO-GO无权重绑Whisper净亏) → **CER腿彻底到底, ROI转效率腿+答辩**。commit `6eb1fa6`+`dcff974` 全 push。详见下【2026-07-23】段 + memory `spk-oracle-poc`/`non-voiceprint-target-selection`/`efficiency-portability-audit`。

> 🔴 **2026-07-20 晚 最新（当前恢复点）**：CER 边际探索闭环 — 全量 oracle + ASE-PVAD 实跑**双证伪**，cascaded CER 近极限坐实。用户战略从"答辩 100 分"转向"**初赛进前 10-20 名**"（入围依据 = B 集成绩 + 脚本核查 + 客观指标，**A 集排行榜仅供参考不决定入围**，见 `待确认_主办方口径与外部输入.md`）。核心发现：① 主战场全量 668 oracle **GO=否**（60 条抽样 GO=是 是假象，cer_gain 虚高 3x）② 主战场 78% 损失是音频摧毁（切对 target 也救不回），22% 选错 target ③ ASE-PVAD（出题方 ICASSP2026 论文 #02）实跑**证伪**（救回 26/改错 33，CER +0.004，根因 zero-training：wespeaker 没学过用增强 embedding）④ **不是菜是架构极限**（出题方 NOTSOFAR CHiME-8 冠军死区也翻车），但端到端联合训练路线（没走）可能更强。**下一步用户定：效率腿 L20（最大杠杆，要租算力）/ 接受 CER 天花板保基本盘**。产物**未 commit**（本地）。详见下【2026-07-20 CER 边际探索】段 + memory `mainfield-oracle-full-debunked`。

> 🔴 **2026-07-20 最新状态（当前恢复点）**：接手收拾 + 答辩准备刷新(**7 commit 全 push**): A.收拾 07-18 游离改动 4 commit(content_gate 反转默认开 joint+0.826 `226e239` / qwen context 归档不启用 `7bba69f` / mainbattle oracle 归档 verdict 不可信 `bafcbf4` / 主办方问题清单 `3469756`) + B.答辩刷新 3 commit(README+FAQ 刷到 qwen 主线算分 53.3/80 `44a94e0` / 进度图重做真测版 3 子图 `089e6e7` / 补 SE bug 真相+时间线+技术亮点 `842706e`). **答辩材料全就绪**(算分/FAQ/README/进度图/时间线/技术亮点一致到 07-19 qwen 主线). **未入库**: out_smoke_fp32.json(孤立临时冒烟, 留本地). **下一步**: 答辩演练稿(待写) > 效率腿 L20(等租算力) > R4 hold-out.

> 🔴 **2026-07-19 最新状态（当前恢复点）**：稳定性/鲁棒性测试**闭环完成**(spec+plan+代码+26遍实跑+报告, 全 push)。核心: **R1=0**系统 greedy argmax 完全确定可复现(不修 use_deterministic) / **R2 纯仅2条**(batch1vs16 差异74条中72含 R3/R4 叠加 → 开发 batch16 数字基本可外推提交 batch1, submit 锁 batch1 3398c0d) / **R3 57%**输入微扰敏感(gauss 加性噪声54%主因破坏 mel)→模型泛化短板**归档**(A 集外训练才能修, A 集不能训练§14) / R5=0。**已 push 13+commit**(0097266→84d70d0+775e219)。详见下【2026-07-19 最新】段 + memory `stability-test-launched` + `docs/稳定性测试报告_2026-07-19.md`。**下一步按 ROI**: 答辩准备(最高) > 效率腿 L20(等租算力) > R4 hold-out。

> 🔴 **2026-07-18 最新状态**：效率腿探索 3 commit(a9dca73/031e4b1/c8c739d) + 对抗审查修正(6ce0636) + L20 阶段0 脚本(2c095de) **均已 push**。**当前在等用户租 L20 算力**——用户说"租了会给 SSH"。拿到后路径 A ssh 操控: `nohup bash code/deploy_l20.sh &`(无卡部署, Monitor 盯) → `SMOKE=1 bash code/run_efficiency_l20.sh`(冒烟) → 用户切 GPU → `bash code/run_efficiency_l20.sh`(全量+换算) → scp 回本机入库。**命门: 问主办方 RTF 口径(per-utt 计时 vs 总墙钟)**。SE orphan bug 真相(三机制, 非仅 mismatch)已入 memory `se-bug-orphan-truth`。详见下【2026-07-18 最新】段。

> **交接时间**：2026-07-15（**标注规范v2工具落地 + 官方CER脚本存档核对 + 当前算分**：改代码+commit+push。三件事：①标注v2工具实现（`build_annotator_pack_v2.py` 失败归因 A-X/B/C/D + enrollment污染诊断 + qwen后端 + 全量浏览 + CER=0隐藏，+ `compare_vs_gold.py`/`map_gold_to_v2.py`）；②用户贴主办方CER参考脚本原文 → 存档 `eval_metrics_official_ref.py` + 用官方原文重算 overall 0.3436 / 逐条 1350/1350 一致 + 修 `_norm_asr` 书名号显示 bug（官方口径书名号不扣分）；③当前分 qwen+thr0.27：CER腿 16.26 + RR腿 36.20 = 52.46/80 硬数字，效率待 L20 实测。详见【2026-07-15 最新】段）。
> **下个 agent 读序**：本文件【2026-07-14 最新】段（↓）→ `docs/标注交接_enrollment污染与target选错_2026-07-14.md`（标注 Agent 直接用）→ CLAUDE.md → 关键 memory（**multi-voice-llm-routing-architecture** / cer-breakthrough-candidates / multi-annotator-dispatch / spk-oracle-poc / content-gate-decision / official-scoring-spec / dataset-split-spec / reproducibility-hardening / mimo-asr-backend-potential / unified-thr-decision / h3-dicow-conditioning-backfire-vanilla / baodi-config-no-llm / submit-script-verification / lessons-pitfalls）→ REPRO_SETUP.md
> **当前 git**：`master`，本 session commit「标注规范v2工具落地 + 官方CER脚本存档核对」已 push（见 `git log -1`）。改动：`code/build_annotator_pack_v2.py`(新) + `code/eval_metrics_official_ref.py`(新,官方脚本存档) + `code/compare_vs_gold.py`+`code/map_gold_to_v2.py`(新) + `code/recompute_official_cer.json`(重算微调) + `AGENT_HANDOFF.md`。⚠️ `code/annot_pack/` 被 gitignore（标注HTML/音频不入库，仅本地）。下个 agent 先 `git status` 核对。

> **2026-07-16 续 session**：L20 效率腿准备(跨平台改造 8 处 + `efficiency_leg_calc.py` + `docs/L20效率实测_runbook_2026-07-15.md` + `setenv_linux.sh` + pyarrow/PY_SE 守卫) **本地未 commit/push**，`git status` 应见 6 改 + 3 新。详见下【2026-07-16 最新】段。

---

## 【2026-07-24 CER 后处理/解码侧 6 路证伪 + 规则E集成】CER 解码侧 exhausted

> **背景**：用户"CER 占 40% 大头重点强化"。5-agent workflow 深度分析 + 5 GPU 实验穷尽解码侧/后处理路线。**结论：CER 0.3422 天花板，规则E 是唯一确定净正（已集成）**。commit `93657ab` + 交接文档 commit。

### 1. 5-agent workflow 深度分析（subagent_tokens 263K）
- **error_pattern**（`runs/_err_analysis.py`）：1350 条程序化分类，55%(370)完全幻觉无解，可救近音154/删字73/同音53/英文品牌8/数字8；理论上限 rule-0.011/prompt-0.003/beam-0.059，现实 0.01-0.03。**header overall_qwen 0.3848 是 per-row 均值（幻觉行拉高），官方累计池=0.3436 不可引用 0.3848**
- **规则实测**（`runs/_rule_full.py`）：现有 poc_rule 158 抽样是乐观假象（Δ-0.0037），全量朴素 +0.0175 灾难（改31恶181）；最强 E（品牌锚点零回退）Δ-0.001 改10恶0 边际；D 数字归一纯 no-op（qwen 已输出中文数字）
- **context 设计**：官方范式=hotword 列表（Qwen3-ASR-Toolkit）；Issue#106 死区+裸词表→热词回吐（重 babble 正中靶心）；阿里云 Fun-ASR 姐妹产品实证"词匹配为主"
- **beam 可行性**：生产 NO-GO（RTF×2.8-3.5 净负，masked_scatter#29968 易碎官方未测），零 RTF 替代（rep/no_repeat/suppress 英文 token）ROI 更高

### 2. 5 GPU 实验（target_slices_full, baseline 0.3432）
| 路线 | ΔCER | 改/恶 | 证伪根因 |
|---|---|---|---|
| context C1 场景+词汇 | -0.0067 | 166/69 | 主战场[0.2,0.4)回退+0.0044，门槛③失败 |
| context C2 纯词表 | +0.2042 | 141/76 | 死区热词回吐 CER 飙124（Issue#106） |
| 零RTF rep_penalty1.1 | +0.0006 | 1/4 | qwen 幻觉无关型非循环型 |
| 零RTF no_repeat4 | +0.0001 | 0/1 | 同上 |
| LLM 精准保守(3B+裁剪) | +0.0012 | 7/17 | 改对=规则E覆盖零增量，改错=3B过度纠正 |

### 3. ✅ 规则E集成（唯一确定净正）
- `text_utils.brand_homophone_fix`：美的功能名锚点（AI净干洗/轻干洗/净呼吸/一键净呼吸/智控温/智清洁/防直吹/无风感/柔风/星香）+ pypinyin TONE3 严格同音；窗口恰好1字不同+同音→修复单字（零回退，最坏=原文）
- 实测 Δ-0.001 改9恶0；enroll_infer L380(vanilla/dicow 在线)+L425(qwen/firered 批量) digit_postproc 后；requirements 声明 pypinyin==0.55.0
- **领域知识非 A集统计**（合规可提交 B集）；排除洗衣机筒（桶/筒 ref 冲突 cmd_117 筒 vs cmd_157 桶）

### 4. qwen_asr_backend 加零RTF参数（Exp5 产物，保留备用）
- 加 `--rep-penalty`/`--no-repeat-ngram-size`（monkey-patch model.model.generate 注入，默认=原 greedy 向后兼容）
- 虽证伪（qwen 幻觉非循环型），但 flag 保留供未来 beam/n-best 实验复用

### 5. ⚠️ 教训（已入 memory）
- **实验前必须验证切片目录=baseline 源**：首轮 C1/C2 误用 enroll_infer 默认 target_slices_qwen（CER0.4384，另一 run 切片）vs poc 源 target_slices_full（0.3436），白跑20min。诚实 baseline 交叉验证（0.4384 vs 0.3436 对不上）暴露问题。

### 产物（commit `93657ab`）
- 脚本：`code/poc_llm_conservative.py` + `runs/{_err_analysis,_rule_full,compare_ctx_cer}.py` + 旧 `poc_llm_homophone_correction.py`/`poc_rule_homophone_correction.py`
- 集成：`text_utils.brand_homophone_fix` + enroll_infer L380/L425 + requirements pypinyin + qwen_asr_backend 零RTF flag
- memory：`cer-improvement-directions`（6 路结果 + 规则E集成 + 教训，全量更新）

### 下一步（按 ROI）
🥇 **效率腿**（L20 RTF 相对赋分；今天实测 qwen 10-14/s，零RTF参数确认不增耗时）> 🥈 **答辩腿**（6 路实证证伪 + Issue#106 死区回吐发现 = 诚实归因硬弹药）> ⛔ **CER 解码侧别再投**（exhausted）。CER 突破只剩更强 LLM(7B+ 预期噪声内)/模型替换(已证伪)/A集外训练(违规)。

---

## 【2026-07-23 目录整理 + 效率迁移性 + Qwen3 重跑 + 非声纹深探】CER 腿彻底到底

> **背景**：用户战略"初赛进前 10-20 名"。本 session 主攻 CER 破局（用户核心假设"是 DiCoW 差不是方法错"）+ 顺带目录整理/效率迁移性分析。**结论：CER 腿所有方向探底，qwen 0.3436 真天花板**。commit `6eb1fa6` + `dcff974`，全 push。

### 1. 目录整理（commit 6eb1fa6）
- 删 ~5.5G 中间产物（`out_*/_work/` 中间音频 + smoke/pycache/log/annot_pack 156M）
- `out_*/exp/poc/json/csv/html` 归 `code/runs/`（tracked **rename 保留历史**，git add -A 检测）
- `.gitignore` 加 `code/runs/out_*/` + `core_source/`（产物/重复副本不入库）
- `docs/实验结果汇总.md` 入库（精华表，删原始大文件留总结）
- `code/` 从 6.8G → ~1.3G

### 2. 效率迁移性分析（4-agent workflow + memory `efficiency-portability-audit`）
- **硬件核实**：4060=**AD107 8GB**（非AD104）/ L20=**AD102 中国特供**（算力≈L40的65%），同代 sm_89，量化 kernel 硅片级一致可迁移，但**绝对 RTF 不可外推只相对排序可**，4060→L20≈2.5×（非早期 L40 口径 3.9×）
- **三不统一**：① batch=16 **不进官方 batch=1 分**（最致命，4060 上 batch 红利到 L20 评分=0）② 绝对 RTF 不能外推 L20≠L40×1.5 ③ int8(bnb) **已证伪**（Whisper +89%/Qwen +299% 慢+损 CER）别再投
- **必迁移（已做）**：关 SE 省 30.6% RTF / --pairs 加载复用（1838次重载→1次）/ greedy+KV / SDPA（已等价FA2）/ fp16
- **命门（用户明天问主办方）**：RTF 口径（per-utt 纯推理符合家居真实）+ 映射公式 + 内存口径；用户澄清效率 20% **初赛算**（相对赋分制，比别人快就赢），官网 A 集排行榜暂不显示但初赛评分算

### 3. 家居 RTF 推算（答辩 + 问主办方依据）
- 家居**两阶段**：注册（enrollment 一次性）+ 识别（每条用预存声纹），效率应只算识别阶段 **per-utt 纯推理**（排除加载/enroll）
- 实时阈值 RTF **0.2-0.3**（说完<1s 响应），L20 外推 **0.09-0.12 瞬时优秀**

### 4. Qwen3 重跑双证伪（用户"是 DiCoW 差"假设证伪）
- **CAM++ 声纹强化 GO=否**：Qwen3 oracle（`exp_spk_oracle_qwen.py`，死区60+主战场60）cer_gain 死区+0.146/主战场+0.106（选 target 是真问题，部分推翻 vanilla "oracle 0.607 封闭"），但 miss 正确 target sim≥0.2 仅 0%/21% → 声纹提不出（babble 摧毁 who 信号）
- **SepFormer GO=否（更狠）**：`exp_sepformer_qwen.py`（死区40）+Qwen3 CER 0.687 vs argmax 0.410（Δ+0.277），oracle 0.413≈argmax → 分离没拎出更干净 target
- **新发现**：Qwen3 下选 target 是真问题（cer_gain+0.1~0.15），但解药**非声纹**（声纹层提不出）

### 5. 非声纹 target 选择深探（全证伪，CER 腿到底）
- **ASE 关键帧**（`exp_ase_keyframe_diag.py`）：用混合音远场关键帧选 target **net 更差**（死区53.3%/主战场56.7% < wespeaker 65%/68.3%，关键帧被 babble 污染），域失配假设不成立
- **CAM++ 替换 wespeaker**（`exp_campp_select_cer.py`）：选对率+10%（主战场78.3% vs 68.3%，**推翻 spk-oracle 笼统结论**——它证伪的是"用sim区分音频可辨度"非"选target speaker"）但 **CER 噪声内**（主战场-0.032/死区+0.043，全量持平），选 target 不是 CER 瓶颈解药（选错条 mel 摧毁主导）
- **Whisper-Sidecar NO-GO**（3 路 workflow 调研）：①**无公开预训练权重**（README 无，sidecar/TTI/soft-prompt 从零随机初始化，无法 zero-shot 必须自己训 400000 steps/4×GPU）②**绑死 Whisper 不能嫁接 Qwen3**（架构全不兼容），回退 Whisper 0.3436→0.595 **净亏 ~0.11**（target 选择天花板仅 0.14）③**1.8s enrollment 硬失配**（TTI 硬编码 3s）+ clean 数据（LibriMix clean subset）迁移空白，中文 target-talker 论文没验证

### 核心结论
**CER 0.3436 真天花板**——CAM++/SepFormer/ASE/非声纹/Sidecar 5+ 方向全探底，物理极限坐实。**诚实归因胜利**（别队盲投 SE/声纹/分离，我用硬证据证重 babble 下都救不了）。答辩级硬弹药。

### 产物（commit `6eb1fa6` + `dcff974`，全 push）
- 脚本：`code/exp_spk_oracle_qwen.py` + `exp_sepformer_qwen.py` + `exp_ase_keyframe_diag.py` + `exp_campp_select_cer.py`
- 结果：`code/runs/exp_spk_oracle_qwen_{dead,main}.json` + `exp_sepformer_qwen.json` + `exp_ase_keyframe_diag.json` + `exp_campp_select_cer.json`
- 文档：`docs/实验结果汇总.md`
- memory：`spk-oracle-poc`（补 Qwen3 段）/ `non-voiceprint-target-selection`（新+POC 结果）/ `efficiency-portability-audit`（新）

### 下一步（按 ROI）
🥇 **效率腿**（push 跨平台部署 = L20 前提 / 问主办方 RTF 口径+映射公式+内存口径 / 起草问主办方清单）> 🥈 **答辩**（03_答辩FAQ 演练稿，今天 5+ POC 坐实物理极限 = 诚实归因硬弹药）> ⛔ **CER 腿别再投**（到顶）。

---

## 【2026-07-20 CER 边际探索】主战场全量 oracle + ASE-PVAD 实跑双证伪（cascaded CER 近极限坐实）

> **背景**：用户战略从"答辩 100 分"转向"**初赛进前 10-20 名**"。入围规则澄清：前 10-20 名排名制（非分数线）；入围依据 = **B 集成绩 + 脚本核查 + 客观指标**（A 集排行榜仅供参考，不决定入围）；效率腿 = 时间 10% + 内存 10% 绝对测量（每条语音平均推理时间 + 模型内存占用，越快越省越好）。核心诉求"在死区/难题上多拿分"。本次 CER 边际探索两弹（全量 oracle 复测 + ASE-PVAD 实跑）**双证伪**，坐实 cascaded+zero-training 下 CER 近极限。

### 本次做了什么（**未 commit/push**，本地超前）
1. **understand workflow（3 agent, 193K tok）**：摸清 `exp_spk_oracle.py` 框架 + 数据可复用性 + sim 分桶。发现 60 条主战场抽样 GO=是（`exp_spk_oracle_mainbattle.json` 已存在，seed=42 单次抽样）。
2. **全量 668 oracle 复测**（`exp_spk_oracle_mainbattle_full.json`，6.27min）：**推翻 60 条 GO=是**，verdict **GO=否**。
3. **ASE-PVAD 论文精读 + POC 设计**（agent）：论文 arXiv 2601.12769（出题方 ICASSP2026 论文 #02），配方 1s 窗/0.2shift/单 keyframe cos 最大/λ=0.1 加性 Eq.6/5 迭代/帧来源 recognition 混合音频/推理期可做 zero-training 友好。
4. **ASE-PVAD 实跑**（`exp_ase_pvad_poc.py` 新 + `exp_ase_pvad_poc.json`，5.95min）：fork exp_spk_oracle + `ase_augment_keyframe` 函数（复用 enroll_infer collect_clean_audio/get_diarization_mask）+ 三档对比 + miss 救回细分 + min_best_sim 护栏。**证伪**。

### 核心发现（全量数据坐实）

**① 全量推翻抽样（adversarial-review 价值）**：
| 指标 | 60 条抽样 | 全量 668 |
|---|---|---|
| verdict | GO=是 | **GO=否** |
| cer_gain | 0.40 | **0.14**（虚高 3x） |
| miss_oracle_recognizable | 36% | **21%**（判据③ <30% 翻车） |
| argmax_CER | 0.95 | 0.65 |

**② 主战场损失分解（全量坐实）**：argmax_CER 0.65 = 选错 target 0.14（22%）+ 音频摧毁/mel 退化 0.51（78%）。主战场 78% 也是音频摧毁，与死区（[[spk-oracle-poc]]）同质 → cascaded CER 近极限。

**③ ASE-PVAD 实跑证伪**（出题方方法复现）：
| 档 | 选对率 | 累计池 CER |
|---|---|---|
| baseline argmax | 74.25% | 0.6491 |
| ASE-PVAD aug | **73.2%** ↓ | **0.6451** |
| oracle（上限） | 100% | 0.4926 |
- 救回 26 / 改错 33（net −7，改错 > 救回），CER +0.004（估算 +0.2，实跑 ≈0）
- flip 59 条（26 对/33 错）→ frame-level keyframe 选择不可靠
- 护栏 fallback 33 条（sim 反向被 min_best_sim=0.15 拦）
- **根因**：论文用训练过的 PVAD2.0 backbone 消费增强 embedding（模型学过），我们 zero-training 直接喂 argmax，wespeaker 没学过用增强 embedding → 引入噪声。印证 [[lessons-pitfalls]] §14（A 集不能训练）。

### 诚实归因（答辩弹药）
- **不是菜是 cascaded+zero-training 架构极限**：① 出题方 NOTSOFAR（CHiME-8 冠军，杜俊+iFlytek）死区 Water-tap 也翻车 ② 出题方自己方法 ASE-PVAD 实跑证伪 ③ oracle 0.51（完美选 target）仍高 → 音频摧毁 ④ ASE 论文有效靠训练 backbone，我们 zero-training。
- **caveat（必须诚实）**：端到端联合训练 / TSE+ASR 联合（要训练 + A 集外数据）可能更强——出题方反 cascaded 审美 + 团队背景（中科大杜俊做语音分离）暗示预期此解法，**学术队若走此路，死区/主战场可能比我们强**。这是路线选择 + zero-training 约束的代价（初赛时间窗 + A 集禁训练），非能力问题。
- **adversarial-review 价值**：全量复测阻止基于 60 条抽样误投 ASE-PVAD（估算 +0.2，实跑证伪）。**任何 oracle 结论必须全量坐实**（exp_spk_oracle.py 抽样 cer_gain 虚高 3x）。

### 产物（本地，**未 commit**）
- `code/exp_spk_oracle_mainbattle_full.json` + `.log`（全量 oracle，GO=否）
- `code/exp_ase_pvad_poc.py`（新，ASE-PVAD POC 脚本）+ `exp_ase_pvad_poc.json`（实跑证伪）+ `_smoke.json`/`.log`（冒烟，可删）
- `code/out_smoke_fp32.json`（孤立临时，之前 session 遗留）
- memory `mainfield-oracle-full-debunked.md`（新）+ MEMORY.md 索引

### 下个 agent 待办（按 ROI，用户定方向）
1. 🔴 **效率腿 L20 实测**（最大确定杠杆，+2-3 分）：用户租算力给 SSH → 阶段 0 脚本就绪（【2026-07-18】段 `deploy_l20.sh`/`run_efficiency_l20.sh`）。本机 4060 overall_rtf 0.142（关 SE），L20 外推 0.06-0.10（注: L40 口径; 真实 L20 ×1.5 ≈ 0.09-0.12, 见 runbook §6.2 勘误）。
2. 🟡 **接受 CER 天花板，保基本盘**：守可复现（07-19 闭环）+ qwen 0.5934 + RR 94.9%（content_gate）+ 关 SE 省 RTF。审 thr/content_gate 是否过拟合 A 集（B 集泛化）。
3. ⛔ **CER 边际别再投**： ASE 实跑证伪 + 全量坐实 78% 音频摧毁，剩余方向（qwen 后端 oracle / 严重桶）预期低。除非走端到端联合训练（大工程，A 集外数据，初赛时间风险高）。
4. 🟡 **commit 本次产物**（Panda_Lorrain 身份，[[git-identity-mismatch]]）：ASE-PVAD 脚本 + 全量 oracle + memory；或丢弃（证伪实验，memory 已记录价值）。

### 关键认知（本次坐实）
- **入围看 B 集不看 A 集**：A 集 53.3/80 是自测分，真正决定入围的是 B 集盲测 + 脚本核查。防过拟合 > A 集峰值分。
- **cascaded CER 近极限**：死区（物理地板）+ 主战场（78% 音频摧毁）都坐实，ASE/声纹强化全证伪。出题方夺冠系统都翻车。
- **zero-training 是死结**：出题方训练类方法（ASE-PVAD）在 zero-training 下失效，根因 A 集禁训练（§14）。唯一能动训练的是 A 集外数据（大工程）。
- **初赛最大杠杆 = 效率腿**（20 分悬空，确定能拿），非 CER 死磕。

---

## 【2026-07-19 最新】稳定性/鲁棒性测试闭环(R1=0 系统确定 / R2 纯2 可外推 / R3 归档)

> **背景**: 现有 run-twice 只 20条×2遍抽样(`verify_reproducibility.py` 默认 `--limit 20`), A 集从未全量多遍跑过; 全项目无 `use_deterministic_algorithms`。**主办方口径(美的_张志飞)**: 默认 batch=1 / 高 batch 结果一致才行 / RTF 按 batch=1 测。

### 本次 session 做了什么(13+ commit, 全 push)
- spec `0097266` + plan `dca7e95`(brainstorming→writing-plans 全流程闭环)
- 代码: enroll_infer 加 `--asr-batch-size`(`629e605`, 回归 qwen 20条×2 delta=0) + `stability_test.py` 编排器(`91aa44e`/`2562ea2`/`80b39d4`, A/B1/B2/B3/B4/all + 断点续跑) + `perturb_audio.py`(`b6b4116`, gauss/vol/time 微扰) + `analyze_stability.py`+`stability_dashboard.py`(`ac7e570`, 5根因决策树 + dataviz 合规)
- B1 改 batch=1 + slice_dir 每遍清 + submit 锁 batch=1(`3398c0d`)
- 报告 `docs/稳定性测试报告_2026-07-19.md`(`1b94d9b` 模板 + `84d70d0` 填数)
- 顺带: 之前 session enroll_infer 增强(context/diar-dtype/inference_mode/fp16)分离 commit `84c5409`

### 实跑 + 核心结论(26 遍全量 1364 条)
- **R1=0**(A 同种子10遍 + 变种子10遍 0 波动): 系统 greedy argmax **完全确定可复现**, 不修 `use_deterministic`(GPU 残余非确定 + 种子变化不改 argmax) → **Task11 跳过**(修了反拖效率腿)
- **R2 纯仅2条**(batch1vs16 差异74条5.43%, 但72条含 R3/R4 叠加): **开发口径 batch16 数字基本可外推提交 batch1**, submit 已锁 batch=1(`3398c0d`)
- **R3=717条57%**(输入微扰敏感, gauss 54%主因破坏 mel 谐波 / vol 9%保结构 / time 26%影响 diar): 模型泛化短板**归档**(A 集外训练才能修, A 集不能训练 §14)
- R4=99(enroll-aug 轻微7.26%), R5=0
- 波动740条 sim 分桶: 死区+低sim 53%, 高sim 18%(部分验证低 sim 更易波动); top 波动都是短指令("往左吹"/"向左吹风")

### hold-out 硬边界(防过拟合, 关键)
A 集是测试集, 本次**不改任何基于 A 集内容的提交规则**(拒识 thr/enhance 不动), 只工程修复(R1 跳过/R2 锁 batch1) + 诊断归档(R3/R4/R5)。**B3 加噪数据是 A 集派生, 不能训练**(§14, 用户亲自纠正)。

### 下一步(按 ROI)
🥇 答辩准备(稳定性测试4点弹药: 可复现性 R1=0 / batch 口径 R2 / 诚实归因 R3 / hold-out 纪律) > 🥈 效率腿 L20(等租算力, 阶段0 脚本就绪) > 🥉 R4 hold-out(轻量) > ⚠️ R3 修复(A 集外训练, 大工程 + CER 腿近天花板, 低 ROI)。

详见 `docs/稳定性测试报告_2026-07-19.md` + memory `stability-test-launched` + `code/stability_matrix/`(report.json/per_utt_volatility.json 740条/dashboard.html)。

---

## 【2026-07-18 最新】效率腿三 commit + SE orphan bug 真相(对抗审查 4-agent 修正)

> **结论**: 效率腿本机探索到头(官方 batch=1 口径下唯一确定杠杆=关 SE 省 30.6% RTF); **SE bugfix
> 揭示重大归因错误**——原"SE 恶化 CER / 对转写无害"经 4-agent 对抗审查 + `audit_se_bugfix.py` 一手
> 重算被推翻, 真相是三机制(sim mismatch 误拒 66% + DF3 过衰减致 diar 崩溃 22% + 转写恶化 12%)。
> 决策(关 SE)不变, 归因全部重写。**全部已 push**: 效率腿 3 commit(a9dca73/031e4b1/c8c739d) + 对抗审查修正(6ce0636) + L20 阶段0 脚本(2c095de)。

### 本次 session 做了什么

**A. 效率腿探索(3 commit, 已 push)**:
- `a9dca73` qwen ASR batch 推理(默认 batch=16): 4060 实测 61 条逐条 20s→batch 4s(**ASR 子进程 5x,
  全管线 1.76x**, overall_rtf 0.25→0.142)。OOM 自动 fallback, 文本 60/61 一致。
- `031e4b1` 效率优化探索: torch.compile / int8 / FA2(SDPA 自动启用)/ 管线并行。**⚠️ 对抗审查发现多条方法学缺陷(见下)**。
- `c8c739d` **SE orphaned-bug bugfix**: `submit_infer.py` 的 `rec_for_enroll` 是死变量(赋值后从未
  读取, 8da1e98 初版起潜伏), `enroll_pairs` 一直写原始音频路径, SE 输出 `se_out` 是孤儿目录 →
  **SE 全程空转 ~30.6% RTF 白烧**。此前所有 "--no-se 零差异" 结论均因此(SE 两分支输入相同)。
  修复: SE 生效时把 recognition 路径重映射到 se_out。

**B. 对抗审查(4-agent, 377K tok, 推翻核心归因)** — memory `adversarial-review-before-milestone-commit` 规则触发:
- **AB 归因 refuted**: 原"+0.1049 完全来自 sim mismatch 误拒, SE 对转写无害"**全错**。`audit_se_bugfix.py`
  一手重算: +0.1049 = **sim-drop 误拒 66%(+0.0765) + DF3 过衰减致 diar 崩溃 22%(+0.0252, 207 条
  SE-diar-fail / noSE 仅 2) + 转写恶化 12%(+0.0134) − lucky-accept 0.0102**。
- **"0 文本不一致"是假象**: `compare_se_bugfix.py:41` 字段名 bug(读 `transcript` 实为 `text`)
  → trivially 相等。实测 **109/383 (28.5%) 文本不同**(SE 改善 21 / 恶化 68, **净恶化**)。
- **"accepted-only SE 略好"是选择性偏差**: 两臂 accepted 集合不同(SE 432 vs noSE 720)。
  apples-to-apples 交集(383 条): SE **0.4281 vs noSE 0.3805, +0.0476 SE 反而恶化**。
- **效率结论缺陷**: batch=16 5x 仅 ASR 子进程(全管线 1.76x), **官方 batch=1 评测口径下不进分**;
  exp_efficiency_report qwen compile baseline 错用 0.0950(真 0.1161, 实 18-19% 但 compile_time
  0.26s 异常疑似 dynamo 回退); int8 只测 bnb 不可据此否定量化方向(L20 与 4060 同为 Ada AD102, 非架构差异问题); "唯一杠杆 batch" 与 Part A
  ONNX 可行自相矛盾。
- **决策确认**: 关 SE 是 Pareto 最优(分桶证每个 max_sim 桶 SE 都恶化 overall CER, 无反例); 双端
  SE 不值得测(收益上界 <0.017 在噪声内 + 仍付 30% RTF)。
- ⚠️ **审查自身也错 1 处**(我一手复核抓到): 审查称"翻转 644"系误算, 实测翻转 386(=337+49), doc 原数正确。

**C. 已修正(本 session, 已 commit 6ce0636 + push)**:
- `compare_se_bugfix.py`/`se_bugfix_record.py` 字段名 bug(`transcript`→`text`)
- `submit_infer.py` 删 `rec_for_enroll` 死变量(2 处)
- `run_baodi.sh:76-78` 注释归因(qwen 鲁棒 → orphan bug + 三机制 + 30.6%)
- `docs/SE_bugfix_AB结果_2026-07-18.md` 全面重写(权威归因 + 答辩表述)
- `exp_efficiency_report.py`/`efficiency_analysis.py` 加对抗审查勘误块
- `code/audit_se_bugfix.py` + `audit_se_bugfix.json`(新, 一手复核脚本/数据)

### 关键认知(本次坐实)
- **SE 三机制**(答辩弹药): sim mismatch 误拒(单端: recognition 过 SE + enrollment 不过) + DF3 过衰减
  致 diar 崩溃(207 条 ValueError, SE-RMS/orig mean 0.004) + 转写净恶化(交集 +0.0476)。**非仅 mismatch**。
- **batch 不进官方分**: 官方 batch=1 测 RTF, batch=16 红利只加快开发迭代。效率腿区间宽到
  [8,19]/20(batch=1 下 overall_rtf 可能 0.3-0.5), 待 L20 batch=1 实测。
- **对抗审查也会出错**: 必须一手复核关键数字(audit 脚本), 不能全信 agent。

### 下个 agent 待办
1. ✅ **对抗审查修正已 commit+push**(6ce0636): AB doc/代码/探索脚本/audit 全修, 无需再做。
2. ✅ **L20 阶段0 脚本已 commit+push**(2c095de): `deploy_l20.sh` + `run_efficiency_l20.sh` + runbook 勘误。
3. 🔴 **等用户租 L20 算力**(当前阻塞点): 用户说"先执行别的, 租了算力会给 SSH"。拿到后路径 A ssh 操控:
   `nohup bash code/deploy_l20.sh &`(无卡部署, Monitor 盯日志) → `SMOKE=1 bash code/run_efficiency_l20.sh`(冒烟5条)
   → 用户切 GPU 模式开机 → `bash code/run_efficiency_l20.sh`(全量 pos+neg + 换算) → scp timing/result.json 回本机入库。
   省钱: 无卡模式部署(~0.1元/h) + GPU 只开全量推理(~20min)。详见 `docs/L20效率实测_runbook_2026-07-15.md` 顶部勘误。
4. 🟡 **命门 — 问主办方**(用户去做): ① RTF 口径(per-utt 计时 vs 总墙钟, 决定 overall_rtf 怎么报 + 效率腿分数)
   ② 排名公式 w1/w2/w_eff(RR-heavy 则 thr 上移)。见 memory `official-scoring-spec`。
5. 🟡 **run_baodi 默认 BACKEND 不一致**(未决, 零提交影响): 默认 vanilla 但 SE A/B 只测 qwen, CLAUDE.md 主线 qwen。
   注释已标风险。用户定: 改 qwen / 补 vanilla+SE AB / 保持不动。
6. ⚪ 可选探索(L20 实测时): GPTQ/AWQ int4 / ONNX / speculative decoding / Qwen3-ASR 蒸馏 0.5B(均 POC 未做; L20 与 4060 同为 Ada AD102, int8 表现应相近; 量化方向须实测 GPTQ/AWQ/TensorRT 才能否定)。
7. ⛔ CER/RR 腿别投入(天花板, 见下【2026-07-16 续】段)。

---

## 【2026-07-16 续·RR+CER 双腿天花板坐实】(本 session, 用户问"RR 90%怎么提")

> **结论: CER+RR 两腿都近天花板, 继续榨边际 net 负。剩余 ROI = 效率腿(L20 runbook 已就绪, 见下段) + 答辩。** 2 commit + 3 memory 入库。

### RR 腿(40%) — 90.51% 接受 ⛔ 天花板
- **方向A(enrollment 污染自适应 thr, 说话人信号类)POC 证伪 No-Go** `code/poc_enrollment_pollution.py`: Q1 D1(DiariZen diar on enrollment)F1=0.77(R1.00/P0.62); Q2 **45漏拒neg enrollment污染率31.8% vs 基线26.6%, Fisher p=0.4737完全不显著** → 核心假设(污染致漏拒)证伪; neg ΔRR最多+2.1pp, D1误报致pos代价net亏损, Q3未跑。详见 `docs/POC_A_enrollment_pollution_结果_2026-07-16.md`。
- **方向B(FA置信度二次拒)天花板不投**: torchaudio MMS_FA+拼音技术可行, 但vanilla 45漏拒38条文本空仅7条非空, FA上限+1.48pp且4/7被content_gate覆盖; qwen主线漏拒是fluent伪指令(FA最弱场景)<0.5pp。
- **物理地板**: 45漏拒里7-8条TRAP(非目标人说了"打开烟机"真指令)谁都救不了。
- commit `75bf7ce`; memory `rr-ceiling-and-direction-A-debunked`。

### CER 腿(40%) — qwen 已优 0.5934 ⛔ 天花板
- **多后端融合 oracle gating net 负** `code/poc_oracle_fusion.py`(纯计算无GPU, sanity 0.3436/0.5934双通过): oracle_qf(qwen+firered)含拒gap**+0.0188**/oracle_qfv(+vanilla)+0.0250(排名公式吃含拒; transcribe gap大因不拒)。74%tie三后端同质。net负(现实CER腿+0.3~0.4 vs RTF翻倍效率腿-1~-2) → 枪毙作主线, 留答辩弹药。
- **FireRedASR替换已死**: firered transcribe 0.3501 > qwen 0.3436, sim桶全输(仅RTF快17%效率红利)。
- 杠杆=主战场[0.2,0.4)49.5%但qwen已吃下大部分(vanilla0.65→0.36); 死区30%地板(qwen0.459<oracle0.607已部分突破)。
- commit `955c03c`; memory `cer-ceiling-oracle-fusion-net-negative`。

### 全局(本次坐实)
- **CER16.26+RR36.20=52.46/80 两腿天花板**, 加效率20, 模型部分~67-70/100。
- **诚实归因=答辩弹药**: 验证了说话人信号(A p=0.47证伪)+FA(B天花板)+多后端融合(oracle含拒gap0.019 net负), 数据证CER+RR近物理极限, 单后端qwen+thr0.27是Pareto最优。契合出题方反cascaded审美。

### 下个 agent 待办
1. 🔴 **L20效率实测**(唯一可搏的20分, runbook已就绪见下段): 租AutoDL L20跑全量qwen → timing overall_rtf+peak_mem → `efficiency_leg_calc.py`换算。预计效率腿18-20/20。
2. 🟡 **答辩准备**(决赛100分ROI高): README进度图+`03_答辩FAQ`+风险预案演练。本次诚实归因(方向A/B证伪+多后端net负+死区地板+TRAP物理地板)是现成弹药。
3. ⛔ **CER/RR别再投入**(两腿天花板, net负或物理地板), 转效率+答辩。

---

## 【2026-07-16 最新】L20 效率腿实测准备就绪(跨平台 Linux-ready + runbook + 换算脚本)

> ⚠️ **2026-07-18 SE orphan bug 回溯**: 本段 :53 "保 SE 开(本机基线含 SE, 关则不可比)" / :61 "SE
> 占 28%" / :65 "--no-se A/B 验证关 SE 不损 CER/RR" 均基于 "SE 在生效" 前提。2026-07-18 发现
> `submit_infer.py` SE orphan bug(se_out 从未被消费, SE 全程空转) → 上述前提**不成立**: "基线含
> SE"是假(从未真含), "--no-se 零差异"是 SE 两分支都空转的 trivial 结果。本段 SE 相关结论**废弃**,
> 以【2026-07-18 最新】段 + `docs/SE_bugfix_AB结果_2026-07-18.md` 为准。(SE RTF 占比 30.6% 实测仍对, 但 "占" 实为 "白烧"。)

> 效率腿(20分)是当前**唯一可搏的剩余分**(CER腿16.26/RR腿36.20近天花板)。官方 L20-46G batch=1 测 RTF+显存; 本机仅 4060, 须租 AutoDL L20实测(或 L40 近似 ×1.5)。本次准备全部就绪, 8-agent 对抗审查过, **本地未 commit/push**(下个 agent 决定)。

### 做了什么(跨平台改造 + runbook + 换算脚本, 8-agent 审查修 6 finding)

**A. 跨平台路径改造(前置阻塞, 8 处)** — 提交链路原全是 Windows 硬编码(E:/ + Scripts/python.exe), Linux 跑不通。平台检测(os.name/OSTYPE) + env override, **Win 冒烟不破**:
- `submit_infer.py` PY_MAIN/PY_SE/PY_LLM · `enroll_infer.py` SLICE_DIR(Win 仍 E:/target_slices_\<be\> 不变)+ qwen/firered venv(PY_QWEN/PY_FIRERED) · `qwen_asr_backend.py` MODEL_QWEN3_ASR · `firered_asr_backend.py` MODEL_FIRERED · `se_denoise.py` DF3 路径 · `run_baodi.sh` source+PY(OSTYPE 分支) · **`setenv_linux.sh`(新, Linux 部署 env)**
- vanilla/dicow/diar 模型已走 `resolve_model`(env override), 本就跨平台

**B. `efficiency_leg_calc.py`(新)** — timing.json+result.json → 效率腿(20)分数区间。5 时间映射×4 内存映射, 双 RTF 口径。已用真实 vanilla 全量 timing 测通。

**C. `docs/L20效率实测_runbook_2026-07-15.md`** — L20 部署(3 venv: .venv/.venv_se/.venv_qwen + DiCoW submodule + 模型)+ 计时+换算+区间估算。结论: vanilla 4060 overall_rtf **0.224(实测含SE)** → L20 外推 ~0.06-0.10, **效率腿预计 18-20/20**。合计初评 **CER16.26+RR36.20+效率18-20 ≈ 70.5-72.5/100**。

**D. 8-agent 对抗审查(539K tok)修 4 confirmed + 2 medium**:
- 🔴 **SE 阶段实测机崩**(critical): run_baodi 不传 --no-se + runbook 漏 .venv_se + PY_SE 无守卫 → 修: runbook §3.3 补 `.venv_se`(df 包) + submit_infer 加 PY_SE 守卫。**保 SE 开**(本机基线含SE, 关则不可比; --no-se 是单独 A/B 项)
- 🔴 **pyarrow 未声明**(critical→high): enroll_infer:38 import, requirements 漏(本机 datasets 泄漏) → 修: `pyarrow==24.0.0`
- 🟡 **qwen "0.289" 口径错**(medium): 实为"转写 0.289 秒/条(仅转写)"非 RTF, §6.1/6.2 自相矛盾 → 修: runbook 澄清, qwen 真实 overall_rtf "待 L20 实测", **答辩勿引 0.289**
- 🟡 SLICE_DIR Win 跨平台(Win 不变) + calc 脚本错误信息

### 关键认知(本次坐实)
- **timing 双口径**: `overall_rtf`=总wall/总audio(端到端含SE+切timeline+ASR转写+加载摊薄, **主报告值**) vs `duration_infer_rtf`=sum(infer_sec)/audio(纯推理, **qwen 漏 ASR 转写低估**)。
- **qwen 真实 overall_rtf 本机从未实测**(无 out_pos_qwen_full/timing.json), "0.289" 是 poc_qwen_asr.py 转写秒/条, 待 L20 实测。
- **SE 占 vanilla RTF 28%**(phases se 222s/783s), --no-se 可砍(需验证 CER 不退化)。

### 下个 agent 待办
1. 🔴 **租 AutoDL L20 跑全量 qwen**(按 runbook §3-5): `BAODI_BACKEND=qwen bash code/run_baodi.sh pos/neg 0.27` → timing.json overall_rtf + result.json peak_mem → `efficiency_leg_calc.py` 换算效率腿真实分数。
2. 🟡 **--no-se A/B**(效率优化): 验证关 SE 不损 CER/RR → 砍 28% RTF。
3. 🟡 **push 本次 commit**(本地超前未 push)。
4. (原待办) 标注回收 / w1w2 问主办方。

---

## 【2026-07-15 最新】标注规范v2工具落地 + 官方CER脚本存档核对 + 当前算分

### 本次 session 做了什么（改代码 + commit + push）

**A. 标注规范 v2 工具落地**（实现 2026-07-14 设计的 v2 三层标注；`annot_pack/` 被 gitignore 不入库）：
- `build_annotator_pack_v2.py`（新）：生成标注 HTML，`--asr-backend {vanilla,qwen}` + `--range {dead_multi,all}`。失败归因（A-X选错target / B音频烂 / C解码崩程序判 / D无难点）+ enrollment污染诊断 + target_active_ratio<0.1警告 + qwen后端（比赛主线）+ 全量浏览（1084条+快捷跳转）+ CER=0已解决默认隐藏 + 档/cer/转写同backend一致。
- `compare_vs_gold.py` + `map_gold_to_v2.py`（新）：金标准对比/映射工具。
- 产物（本地）：`标注_v2_qwen.html`(101死区发包) / `标注_v2_qwen_all.html`(1084浏览) / `calibration_samples_v2_qwen.csv`(26金标准)。

**B. 官方 CER 脚本存档 + 核对**（用户贴主办方原文）：
- `eval_metrics_official_ref.py`（新）：主办方 2026-07-08 参考脚本**原文存档**（原封不动+来源注释+核对结果），作 `eval_metrics.py` 对照基准，勿改实现。
- 用官方原文重算 1350 条：overall **0.3436**，逐条 vs poc json qwen_cer **1350/1350 一致 0 差异** → 本仓库所有 CER 数字均出官方口径，依据链闭环到原文实体。
- **书名号不扣分铁证**：官方 `normalize_text` 去 Unicode P* 标点 → 书名号《》(Ps/Pe)被丢；官方自带测试场景1（`美 的 空调，真 省电！！！` vs `美的空调真省电` 断言 CER=0）自证标点不计入。
- 修 `_norm_asr` 显示 bug：枚举正则 `[，。！？...]` 漏《》→ 改用官方 `normalize_text`（去所有 P*），标注界面书名号清零。

**C. 当前算分**（`qwen_official_cer_workpoints.json` 已汇总，qwen+thr0.27 提交口径）：
- CER腿40 = **16.26**（含拒0.5934）/ RR腿40 = **36.20**（thr0.27, 90.51%）/ 效率腿20 = **待L20实测**
- **CER+RR = 52.46/80 硬数字**，加效率估 ~62-72 分。
- ⚠️ 提交用**含拒 0.5934**（CER腿16.26），非 transcribe 0.3436（诊断上限，评委按含拒复算会穿帮）。vs vanilla：qwen CER腿 +4.29。

### 下个 agent 待办（优先级）
1. 🔴 **L20效率实测**（唯一可搏的20分）：租 AutoDL L20 对齐官方 L20 测 RTF+内存。本机 4060 qwen RTF 0.289，L20 会更快。
2. 🟡 **标注回收**：annot_pack 发江/罗 v2 重标（v1 的 50 条作废），回收后 compare_annotations 对比（当前为 v1 列结构，需改 v2）。
3. 🟡 **w1/w2 权重**：向主办方确认排名公式权重（RR-heavy→thr 上移）；效率分映射口径（RTF/内存→0-10分）。
4. ⚠️ CER 死区 112 条是 A-X 切错 target 物理极限，qwen/vanilla 都救不了 → CER 腿 16.26 近天花板，别在 CER 再投入，转向效率/决赛答辩。

### 关键认知（本次坐实）
- 官方 CER 去所有标点+空白 → 任何格式美化（书名号/逗号/繁简）对提分=0，杠杆只在"转对字"。
- 去标点归一必须用 `unicodedata.category(ch).startswith("P")`，禁枚举正则白名单（必漏字符）。
- 详见 memory `annotation-spec-v2`（_norm_asr 书名号修复）+ `official-scoring-spec`（官方脚本存档核对，本次均更新）。

---

## 【2026-07-14】enrollment 污染诊断 + 架构转向多声纹 LLM 路由 + 标注交接文档

### 本次 session 做了什么（纯诊断 + 文档，**未改代码、未 commit**）

1. **诊断"输出非家居指令"**：
   - **demo**（`code/demo_web/inference_engine.py`）只有声纹 sim 单闸（`max_sim<0.27`），**没接 LLM/content_gate**（design 文档第 27 行 YAGNI 砍的）。sim 过线就无脑转写输出 → "而且这有缠了"/"好"这类非指令也出来。
   - **主线** `submit_infer.py:56 decide_reject` 有完整三档（sim_only/llm_or_sim/content_gate），但保底 `run_baodi.sh --no-llm` 关 LLM。
   - 概念澄清：**LLM(llm_reject) 是内容拒识（判文本是不是家居指令），不是说话人分离**；分离靠 wespeaker+diar。

2. **诊断 enrollment 污染 → argmax 选错 target**（用户听 cmd_2081 坐实）：
   - **cmd_2081 铁证**：ref"风速调高"，系统输出"或销售手机的计划"（vanilla+qwen 两个不同 ASR 都转非目标人 = **切错 timeline，不是转写器问题**）；`max_sim 0.365`、`vanilla_cer 2.0`、`speakers=[0]`（diar 欠分割）。
   - **机制**：`enroll_infer.py:187 get_enroll_emb` 整条 kws 提一个混合声纹（架构假设单人）→ `:249 argmax(sims)` 选错成非目标人 → 切错 timeline。
   - **全量统计**（pos 1364，`out_pos_slices_full.json`+`exp_vanilla_full.json`，未独立重跑脚本）：单 speaker 551(40%)/失败 25% | **多 speaker 811(59%)/失败 67%** | 失败 680 里**多 speaker 占 80%**。⚠️ **主战场是多 speaker 条 argmax 选错；cmd_2081 那种 diar 欠分割是 20% 少数派**，别误判主流。
   - 多 speaker 失败样例：cmd_18(关闭灯光→"我")/cmd_57(空调调为十二度→"所以")/cmd_237(打开清香烟机→"把握上演字")。

3. **架构转向（用户拍板，2026-07-14）**：
   - ❌ **唤醒词定位 target 已否**：kws_txt 是**开发集标注**，提交/评测只给音频不给文本，此路走不通（曾误提，用户纠正）。
   - ✅ **新架构**：enrollment 多声纹 → recognition 多路转写 → LLM 挑家居指令段。**核心：把"判 target"从声纹 argmax 换成 LLM 凭内容识别**，绕过 argmax 选错，复用 `code/llm_reject.py`。
   - **考题规则（用户阐明）**：当前考题保证 recognition 里**只有 target 说家居指令**、非目标说非指令（新闻/闲聊）→ "LLM 识别家居指令 = 识别 target"可靠；未来可能多人各说不同家具指令（拉窗帘+打开洗衣机）都要识别路由，架构从"挑唯一指令段"泛化到"识别所有指令段"预留。
   - **收益上限（诚实）**：≈ oracle **0.607**（`exp_spk_oracle.py` 全 speaker 转写挑对）vs argmax 0.788，Δ-0.18（死区 60 条样本 vanilla 数据，**仍不及格**）。救"选错 target 漏转"部分，**救不了**"选对但 babble 毁 mel 转崩"（归转写器换型 Qwen3-ASR/FireRedASR）。
   - **为什么是新角度**：七连受挫全是声纹层，本方向是内容/语义层，不在已证伪范围。

### 交付物
- **标注交接文档**：`docs/标注交接_enrollment污染与target选错_2026-07-14.md`（自包含，已更新到新架构）。标注维度：区分 enrollment 两人角色 / recognition 每 speaker 内容+是否家居指令 / 优先盯多 speaker 失败条。**标注 Agent 直接拿这个开工**。
- **memory**：`multi-voice-llm-routing-architecture.md`（架构方向持久化，含考题规则+收益上限+待 POC）。

### 关键数据/文件（下个 agent 复核用）
- cmd_2081：`pos_pairs_datasetA.json` id=2081（kws_txt"小钱小钱"/ref"风速调高"）；`out_pos_slices_full.json` ~10914 行（`speakers=[0]`）；`exp_vanilla_full.json` ~4452 行（vanilla_text"或销售手机的计划"/cer 2.0）。
- 全量分桶统计脚本未落盘，数字来自 `out_pos_slices_full.json`+`exp_vanilla_full.json` 现场聚合（单 speaker 551/多 speaker 811/失败 680 多占 80%），下个 agent 需要可重跑核实。
- 机制行号：`enroll_infer.py:187 get_enroll_emb`（整条提声纹，无分离）、`:249 argmax(sims)`（误选发生处）。

### 下一步（POC A 已完成 2026-07-14 续 session；POC B 条件 GO 待实现）
- ✅ **POC A 完成**（本续 session）：LLM 凭内容识别家居指令**判别逻辑根基成立**（防循环论证 e 层 reject 0.94 + 非家居 0.97 + 家居 precision 0.92）。prompt v1 过严（参数审查误拒"空调十六度""风速自动"）recall 0.25 → v2 放宽参数/短指令/播放 recall **0.75**（剩 12 误拒是 Qwen2.5-3B 对口语化/品牌/模式名识别能力边界，非 prompt 能根治）。**用户决策条件 GO POC B**（recall 0.75 略优于 argmax 选对 66.7%，净 CER 靠端到端验）。产物 `code/pocA_fast_eval.py`+`llm_testset_pocA.json`(400全)/`llm_testset_pocA_150.json`(跑用)+`analyze_pocA.py`+`pocA_analysis.json`+`docs/POC_A_llm判别力验证_设计与方法_2026-07-14.md`。副产品 memory `pos-ref-not-pure-commands`（pos.ref 31%非纯指令）+ 分桶坐实（多 spk 失败 67%/占 80%，原交接自承未核实现坐实）。⚠️ 执行教训：原逐条 max_new=160 长 SYSTEM_PROMPT 400 条 4060 要 30-50min 太慢，改 batch=1（主办方 batch=1 位，batch=4 实测无并行收益）+max_new=64+规模 150 ~7min。
- ⛔ **POC B 证伪（2026-07-14 续，multi-voice LLM 路由不 work）**：端到端 30 条最严重多 speaker 失败条（argmax CER 5.114，主因 vanilla 循环幻觉）。multi-voice+LLM挑(v2) CER 0.982（Δ-4.13）但 **29/30 all_reject**——价值=拒幻觉压 CER（死区 9.28→1.0），**不是挑对 target**。中等 15 条 14 all_reject（CER 0.96 略恶化 argmax 0.95）。**瓶颈=切片转写质量**（vanilla 重 babble 切片循环幻觉），LLM 正确拒乱码但无可挑——坐实死区是 babble 毁 mel 物理极限（[[spk-oracle-poc]]）。全量风险：POC A recall 0.75=25%误拒会让非失败条~1084 恶化。**新架构方向证伪（切片质量瓶颈+LLM recall 双证伪），搁置回主线**（Qwen3-ASR 0.3436 / vanilla 0.595 + 关LLM thr0.27）。产物 `code/enroll_infer.py --multi-voice flag` + `pocB_pick.py` + `pocB_multivoice_30.json` + `pocB_result.json`。教训：内容/语义层方法受切片质量上限制约（multi-voice 价值域=argmax选错条=切片质量差条=死结）。

- 🔄 **全量死区条确认修正（2026-07-15，修正 POC B 纯证伪解读）**：跑全量死区条 126（CER>1）：argmax CER **2.343→mv 0.987（Δ-1.355，改善 126/126 全改善）**。推全量 char-weighted（仅死区条启用 mv）vanilla **0.595→0.476（Δ-0.119）**。**修正**：POC B 30 条纠结"all_reject=挑不到 target=证伪"忽略了拒幻觉的累计池 CER 收益（幻觉 CER 2-10→拒 1.0，pos 允许拒）。但价值本质=**拒幻觉非挑 target**（124 all_reject 没挑到，切片都幻觉）。**修正后方向**：①"LLM 挑 target"仍证伪 ②**content_gate 加强幻觉检测**（拒 argmax 循环幻觉/超长/乱码输出）是可行 CER 改善方向（死区 -1.36/全量仅死区 -0.119），不需 multi-voice 多路，是 content_gate 增强（加循环重复检测，呼应 [[content-gate-decision]]）。方向从"multi-voice 多路"转向"单路 argmax+幻觉检测拒"（更轻量）。⚠️ multi-voice 不能全量启用（非死区 25% 误拒恶化），仅死区条（幻觉检测识别）启用才改善。产物 `code/pocB_result_deadzone.json`+`pocB_pairs_deadzone.json`+`pocB_multivoice_deadzone.json`。

- ✅ **content_gate v2 集成完成（2026-07-15，text_utils）**：幻觉检测集成 `is_valid_command`（循环 `_max_char_run>=4` + `_char_diversity<0.35` + NEWS_KW 财经词典扩展）+ `digit_postproc` 时间归一（14.55→十四点五十五）。全量 CER 净效果 **Δ-0.039**（digit_postproc 时间归一 -0.022 改善 114 数字句 + content_gate 拒 17 死区幻觉 -0.017），hold-out val Δ-0.0237 泛化 OK。官方口径 vanilla 提交 CER 0.7030→~0.666（CER 腿 +1.48），qwen 同理受益。content_gate 默认 BAODI_GATE=1 开(run_baodi)集成生效。⚠️ short 短碎片规则误拒 10% 非死区放弃；剩余 50% 死区(连贯干扰话)规则地板需 LLM。产物 `code/text_utils.py`(改)+`poc_content_gate_v2_eval.py`+`hallucination_detect_eval.py`+`hal_positives.json`+`deadzone_missed.json`。

- ⚠️ **content_gate v2 后端差异（2026-07-15 qwen 全量确认，修正"集成 CER-0.039"乐观估计）**：content_gate v2 **非普适**——vanilla ΔCER-0.0406 改善（死区 0.784 拒幻觉收益大+指令干净），**qwen Δ+0.0237 恶化**（死区 0.459 qwen 鲁棒+指令多样误拒 19>拒幻觉 16）。根因：qwen text 含标点(，。)致 len>22 误判（已修 `is_valid_command` 开头去标点 `re.sub(r"[^\w一-鿿]","",text)`，vanilla 无标点 no-op），但去标点后 qwen 仍误拒 19（NEWS_KW/循环/div 其他）。digit_postproc 时间归一对 qwen **no-op**（qwen 已中文数字，含阿拉伯数字 0/1350）。**结论**：qwen 主线（CER 0.3436 最低）**不用 content_gate**（恶化），仅 vanilla 保底受益（0.7030→~0.666）。启用策略：**后端条件**——vanilla 开（BAODI_GATE=1），**qwen 关**。集成 text_utils 对 qwen 主线无害（digit_postproc no-op+content_gate 关）。产物 `code/poc_content_gate_v2_qwen_eval.py`。
- 标注 Agent 拿 `docs/标注交接_enrollment污染与target选错_2026-07-14.md` 开工，回收后验证考题假设 + LLM 路由真实泛化（POC B 裁判）。

---

## 【2026-07-11】前沿探索(19路) + Qwen3-ASR 候选2 证实 + 集成落地 + P0 数字收尾

### ⚠️ 2026-07-11 P0 收尾（本 session 续：7-agent 路线核实 workflow + 双口径坐实 + wesep defer）

- **7-agent 路线核实 workflow**（ultracode，375K tokens/104 tool calls）：fan-out 评估 7 候选方向(A1 归一/A2 死区对抗/A3 run-twice/A4 提交数字/B1 FireRedASR/C1 wesep/C2 beam)，每 agent 核实 file:line 现状 + 收益/成本/风险/答辩价值。用户决策：**P0 数字收尾先做** + **wesep defer**。
- **两个洞堵上**：①**+10.1 是 transcribe 不拒口径虚高**——提交进排名公式用【含拒 overall】(pos 允许拒, 2026-07-08 确认)，qwen 含拒 thr0.27=**0.5934**(CER 腿 +4.29) vs transcribe 0.3436(+10.07 为诊断上限)；②**0.3436 此前不可复现**(poc json 仅 per-sample 0.3848)，`code/recompute_qwen_official.py` 独立坐实落盘 `qwen_official_cer_workpoints.json`。
- **归一零效应坐实**：1350 条 qwen 输出 0 阿拉伯数字 0 真繁体(原生中文"二十五度")，digit_postproc/to_simplified 均 no-op，raw==归一==0.3436 逐位相等。提交侧 enroll_infer:384 已接归一，无 cn2an/zhconv 式漏洞。
- **死区 0.459 坐实(官方累计池)**：n=396 qwen 0.459 vs vanilla 0.784；0.459 < oracle 0.607。✅ **A2 对抗验证已完成**(见 follow-up#5 + RESULTS A2 段): 用户听音 cmd_2091/2137 坐实 **H1 真实突破**(音频可辨 qwen 听对, 非LM幻觉), 死区是混合桶(B类声纹失败但音频可辨qwen突破 + A类真摧毁H2少数), spk-oracle-poc 物理地板修正为 vanilla OOD 伪地板。
- **含拒 thr 扫描**(官方池)：0.20 qwen0.4912/vanilla0.6544 | **0.27 qwen0.5934/vanilla0.7007** | 0.30 qwen0.6435 | 0.35 qwen0.7221 | 0.40 qwen0.7993。全档 qwen 优于 vanilla。
- **提交数字(thr0.27, w1=w2=0.4)**：qwen CER腿16.26+RR腿36.20=52.46 | vanilla 11.97+36.20=48.17 | Δ+4.29(效率腿20待L20)。neg RR 0.9051 与转写器无关。
- **下个 agent 焦点(A3/A2/B1/答辩固化/firered 集成 本 session 已全完成)**：🟡 03_答辩FAQ 下文红线/FAQ 按新数字全文细化(2026-07-11 横幅已加, 可选) / ⏸ L20 RTF 真测(租 AutoDL L20, qwen 0.289/firered 0.24 @4060) / ⏸ 等标注回收(定模型路线)。**C1 wesep / C2 beam / 声纹强化 均 defer/证伪关闭**。

### 一句话现状
用户做数据标注(1084条未满分, 分发2队员)期间, agent 完成"前沿探索者"任务闭环: 19路并行探索(报告 docs/前沿探索报告_2026-07-10.md) → 3 POC(faster-whisper/BoH no-go, **Qwen3-ASR +10分**) → code/.venv speechbrain 修复 → Qwen3-ASR 集成 enroll_infer/submit_infer → **2026-07-11 P0 数字收尾(7-agent 路线核实 + 双口径坐实 + wesep defer, 见上 P0 收尾段)**。**Qwen3-ASR 全量1350条 transcribe CER 0.3436(vs vanilla 0.5954); 含拒 thr0.27 提交口径 0.5934(CER 腿真实 +4.29, transcribe 不拒 +10.07 为诊断上限)**, drop-in 集成(submit_infer --asr-backend qwen)。首个经数据验证的 CER 突破。⚠️ 答辩/提交须报含拒 +4.29 口径, 勿用 transcribe +10.07 虚高。

### 前沿探索(19路, docs/前沿探索报告_2026-07-10.md)
12方向SOTA搜索 + 7组已有论文重读 + 综合路线图。候选裁决:
- 候选1 LLM后纠: 🔴 NO-GO(ASR-EC benchmark+Apple+Cambridge+langfix0.028 四重证伪"纯文本后纠天花板封顶")
- 候选2 中文原生ASR: 🟢✅ Qwen3-ASR 全量证实(+10分)
- 候选3 TSE: 🟡 降级(EoW 2026 arXiv:2602.15519 几乎1:1本项目场景证伪级联TSE→ASR对WER无效)
- 候选4 端到端: 🟡 Whisper-Sidecar落地实例(github LingweiMeng/Whisper-Sidecar, 开源+Aishell1Mix中文数据+3s enrollment)
- 候选5 RR/效率: faster-whisper/BoH no-go(见下)

### Qwen3-ASR 突破(候选2, 全量证实, 首个CER收益方向)
- 探针60条: 主战场桶(sim[0.2,0.4)) CER 0.146 vs vanilla 0.454, Δ-0.31
- 全量1350条官方口径累计池(total_err/total_char, 与vanilla 0.595同口径): overall **0.3436** vs vanilla 0.6635(未归一)/0.595(官方归一), **Δ-0.32**
- 分桶: 死区(<0.2,n=396) Qwen 0.459 vs vanilla 0.828(Δ-0.37, **比MiMo 0.554强, 挑战"物理地板"叙事, 需对抗验证**)/ 主战场[0.2,0.4)(n=668) 0.360 vs 0.718(Δ-0.36)/ 接近解决≥0.4(n=286) 0.182 vs 0.375
- Qwen更优 55%(746/1350), RTF 0.289s/条(4060, L20待测)
- **CER 40%腿双口径(2026-07-11 P0 坐实)**: transcribe 不拒 vanilla 16.2→qwen 26.3(**+10.07** 诊断上限) / **含拒 thr0.27 vanilla 11.97→qwen 16.26(+4.29 提交口径, 排名公式用)**。见上 P0 收尾段 + `qwen_official_cer_workpoints.json`
- 机制: 复用 enroll_infer diar+wespeaker 切 target timeline, Qwen3-ASR drop-in 转写切片(language="Chinese"), 不改diar, zero-training, Apache2.0可进提交
- HF核实: ExtremeNoise WER 16.17 vs Whisper-large-v3 63.17(≈4×), 中文原生+鲁棒对症babble
- 产物: code/poc_qwen_asr.py + poc_qwen_asr_full_result.json + E:/target_slices_full/(1350切片)

### code/.venv speechbrain 修复(关键解锁, 影响后续所有切片路线)
speechbrain 1.1 lazy proxy 注册 sys.modules, inspect.getmodule 遍历时触发 lazy resolve失败(ImportError, hasattr不捕获) → enroll_infer(librosa.load经lazy_loader→inspect)连锁崩。
**修复**: patch inspect.getmodule 固化 enroll_infer 顶部(捕获 ImportError/AttributeError 返None, 不破坏speechbrain正常lazy)。验证: enroll_infer直接跑通(diar+vanilla, cmd_0 TRANSCRIBE)。
**教训⚠️**: 装 faster-whisper到主 code/.venv 连带升级共享包触发此问题(已卸载faster-whisper+回滚typing_extensions)。**后续新依赖一律独立venv**(如 code/.venv_qwen), 不污染主venv。已记 lessons-pitfalls 待补。

### Qwen3-ASR 集成(drop-in 落地, submit_infer --asr-backend qwen 可提交)
- enroll_infer 加 --asr-backend qwen 分支: 切片存盘+text空 → 末尾 subprocess 调 code/qwen_asr_backend.py[code/.venv_qwen, venv隔离] 批量转写 → 填transcript + to_simplified/digit_postproc 提交归一
- code/qwen_asr_backend.py(code/.venv_qwen, Qwen3-ASR批量转写切片→uid2text.json)
- submit_infer choices 加 qwen(透传机制 line 158-159 已有, 非dicow即透传)
- 验证: 5条 transcript 填充(Qwen3-ASR识"制热""权志龙"等vanilla错的, 输出含标点官方口径去掉)
- code/.venv_qwen: qwen-asr(transformers backend, Windows兼容) + torch2.6.0+cu124(强制reinstall覆盖CPU版) + Qwen3-ASR-1.7B权重E:/hf_cache/Qwen3-ASR-1.7B

### ⚠️ follow-up(P0 收尾后状态: #4 已完成, #6 wesep 已 defer, 剩 A3/A2/B1 + 阻塞项)
1. ✅ **submit_infer qwen 全流程 run-twice 验证**(2026-07-11 完成: verify_reproducibility --backend qwen limit=10, **text 一致率 100%, CER delta=0**, 与 vanilla 对齐; 改 verify:47 choices 加 qwen + qwen_asr_backend 加 --seed 内联 set_seed(独立 venv 不依赖 repro.py) + enroll_infer:377 subprocess 透传 --seed)
2. 🟡 **L20 RTF 真测**(Qwen3-ASR RTF 0.289@4060 慢于vanilla 0.16-0.24, L20待测, 效率腿时间分可能小失分-1~2; 租AutoDL L20)
3. ✅ **FireRedASR 横评**(2026-07-11 完成, 见 RESULTS T29: firered 0.3501 ≈ qwen 0.3436 不可分, RTF 0.24 vs 0.289 firered 快 17%; B1 预判 45% no-go 未发生, WenetSpeech-meeting 训练对 babble 适应好; qwen 保持主线 firered drop-in 备选)
4. ✅ **Qwen3-ASR 提交归一后 overall**(2026-07-11 P0 完成: 归一零效应 0 阿拉伯数字 0 繁体 raw==归一==0.3436; 含拒 thr0.27=0.5934, CER腿真实 +4.29; 见 P0 收尾段)
5. ✅ **死区 Qwen3-ASR 0.459 对抗验证**(2026-07-11 完成: 纯分析+用户听音 cmd_2091/2137 坐实 **H1 真实听音**(非LM幻觉); 死区混合桶 B类声纹失败但音频可辨qwen突破 + A类真摧毁H2少数; **spk-oracle-poc 物理地板修正为 vanilla OOD 伪地板**; 连带声纹强化 CAM++ POC **证伪关闭**(B/A margin 0, 声纹 emb 编码who不编码audio clarity→任何声纹器都救不了B类, exp_spk_campp_deadzone.py); 产物 analyze_dead_zone_qwen.py + exp_spk_campp_deadzone.py)
6. ⏸ **wesep TSE POC**(2026-07-11 用户决策 defer/drop: 零 upside + emb-mismatch 可能产模糊结论 + EoW2026/SepFormer/STNO 三重同构预警 no-go 85%)
7. ⏸ **等用户标注回收**(1084条错误类型×sim分桶交叉表, 定thr+看死区可改进空间)

### 关联
docs/前沿探索报告_2026-07-10.md | memory cer-breakthrough-candidates(候选2全量数据+集成完成) | RESULTS.md T28 | REPRO_SETUP.md §2/§3(Qwen3-ASR+venv_qwen+patch)

---

## 【2026-07-10 最新】多人标注分发工具链（2队员全量标+比对仲裁）+ CER分布检视

### 一句话现状
用户决定把 1084 条未满分标注**从自己标改为分发给 2 个队员各标全量 + 交叉比对仲裁**（防误判）。工具链已交付 + mock 验证 + commit(53a6521)，等队员标注回收。副产品 CER 分布检视界面。

### 工具链（commit 53a6521）
- `code/build_annotator_pack.py`：读 error_analysis_pos_unfull.csv(1084) → 自包含标注包 `code/annot_pack/`（`标注.html`: 相对路径 `pos/*.wav` + 标注员ID输入框 + 全量导出带 annotator 列 + 文件名 `annot_<ID>.csv`; `pos/` 2168音频）。`--copy-audio` 拷音频。产物 `annot_pack.zip`(116.6M) 微信发队员（.gitignore 忽略不入库）。
- `code/compare_annotations.py`：回收 2 份 `annot_<名>.csv` → 比对(一致/分歧/未标) → `consistency_report.txt` + `merged_annotation.csv`(一致条定共识) + `annot_disputes.html`(分歧仲裁: 听音+两人难点对比+交集默认勾+导出 `arbitrated.csv`)。mock 验证通过(一致2/分歧3/未标1 逻辑对)。
- `code/build_cer_viewer.py` + `code/cer_distribution_viewer.html`（子agent生成）：全量1364条CER分布检视(分档柱状+逐条ref/vanilla对比+听音+档筛选/搜索/排序/分页)。

### 流程
1. 发包: `annot_pack.zip` → 2队员解压 → 双击 `标注.html`(Firefox优先; Chrome禁file://则 annot_pack 目录跑 `python -m http.server 8000` 开 `localhost:8000/标注.html`) → 填名 → 标1084难点(←→翻条/空格播放/1-9选难点) → 导出 `annot_<名>.csv` → 回收
2. 比对: `code/.venv/Scripts/python.exe code/compare_annotations.py annot_A.csv annot_B.csv` → 看一致率 + 仲裁分歧
3. 仲裁: `annot_disputes.html` 逐条听音定 → `arbitrated.csv`
4. 最终全集 = merged(一致) + arbitrated(仲裁) → 聚类难点分布 → 定模型路线

### CER 分布（全量1364，vanilla_cer 排序）
满分CER=0: 280(20.5%) / 轻微0-0.1: 12 / 中等0.1-0.5: 392(28.7%) / 严重0.5-1: 554(40.6%) / 死区1-2: 112 / 极重>2: 14。均值0.663/中位0.50。死区声纹sim<0.2占29.9%（吻合 [[spk-oracle-poc]] P1 oracle 30%物理死区）。

### ⚠️ follow-up
1. 🟡 等队员标注回收 → compare → 仲裁 → 聚类难点分布 → 定模型路线（接 [[cer-breakthrough-candidates]]）
2. ✅ 成果已全 push（6 commit 至 b377054：工具链/kws/v2分块/视觉重构/README/handoff）。annot_pack.zip 116.6M 待发组员。
3. 🔍 **下一个 agent 检验这些成果**：跑 `docs/标注分发成果检验_2026-07-11.md` 的检验脚本对照预期，按其第 8 节模板报告 PASS/FAIL（只检验+报告，不改实现）
4. ⏸ 模型路线全等标注聚类结论 + 用户决策

关联 memory: [[multi-annotator-dispatch]] [[cer-breakthrough-candidates]] [[dataset-split-spec]] [[lessons-pitfalls]]。

---

## 【2026-07-09 最新】数据驱动错误分析优先 + Paraformer/声纹注入调研（模型路线全延后）

### 一句话现状
用户听过死区样本(cmd_2102 音量小+语速快 / cmd_2788 语速慢+babble, rec_sec 4.28s 印证"拖很慢")后决策: **先人工过全部未满分音频提取难点(数据驱动错误分析), 模型路线(Paraformer/声纹注入/端到端/TSE/LLM纠正)全延后**。方向正确(先理解数据再调模型, 契合 lessons-pitfalls 先证机制)。

### Workflow B 调研结论(方向修正, tasks/wiupwi5ca.output, 4路WebSearch 454K tokens)
- ⚠️ **"Paraformer+声纹注入合并"工程不成立**: Paraformer 无任何 speaker-conditioned/TS 变体(CIF 非自回归, 注入全自定义无先例); 声纹注入有公开模板的是 **Whisper(TS-Whisper Ma ICASSP2024)**, 不是 Paraformer。
- **Paraformer drop-in(选项A)明确利好**: <1人天零训练(enroll_infer 已有 `--asr-backend` 机制, 集成点已核查); 中文原生(干净 CER SenseVoice 7.81% vs Whisper-turbo 21.71%, 2.8x, ⚠️babble 场景未证需 POC); 效率(RTF 0.008 vs 0.022, 2.6x); **数字ITN(use_itn 原生中文数字, 消除 cn2an/zhconv 依赖, 化解提交归一漏洞)**; License Apache 2.0。
- **声纹注入(走Whisper)双重红线**: ①enrollment 1.8s<5s 红线(Maeda2025 坐实) ②重babble下条件化反作用(H3铁证 DiCoW sim[0.2,0.3) CER1.606; FiLM/prompt同族)。选项B轻量注入1-2周, C端到端联合4-8周(上限候选X, 理论可能超MiMo但风险最高)。
- 🔴 **A集合规 GRAY ZONE(最高杠杆未知项, 必问主办方)**: 决定 B/C 可行性。推荐顺序: 选项A必做首选, B/C条件触发(待数据结论+合规+GPU+时间)。

### 未满分错误分析(用户主导, 数据驱动)
- pos 1364 条: 满分(CER=0) 280, **未满分(CER>0) 1084**。分档: 1_死区CER>1 **126** / 2_严重0.5-1 **554** / 3_中等0.1-0.5 **392** / 4_轻微0-0.1 **12**。
- 产物: `code/error_analysis_pos_unfull.csv`(分档, Excel 友好) + `code/error_annotator.html`(标注界面, 双击打开, **推荐 Firefox**; Chrome 禁 file:// 则 `python -m http.server`; 播放+难点多选+快捷键 ←→翻条/空格播放/1-9选难点+导出 error_annot_export.csv)。难点9类: 音量小/语速快/语速慢/babble强/重叠/英文干扰/静音未说话/循环幻觉/其他。
- 死区听音清单 18 条代表(tasks/wxl93pp09.output): 物理极限铁证(MiMo同翻车 ~61%) / 切错target·声纹反向(oracle本可救 ~28%, 唯一有改进空间) / 循环幻觉OOD / 英文水印。
- **下个 agent**: 等用户标注结果 → 聚类难点分布 → 决定攻哪个模型路线(音量小集中→前端增益可能救; babble强主导→物理地板守保底)。

### 产物(本 session, 未 commit)
- `code/extract_error_list.py` + `code/error_analysis_pos_unfull.csv`(1084条分档)
- `code/build_error_annotator.py` + `code/error_annotator.html`(标注界面)
- `memory/cer-breakthrough-candidates.md`(路线backlog+2026-07-09延后决策) + MEMORY.md 索引
- Workflow 产物: `tasks/wxl93pp09.output`(死区清单) + `tasks/wiupwi5ca.output`(4路调研+设计选项A/B/C)

### ⚠️ follow-up
1. 🟡 等**用户标注结果**(难点分布) → 决定模型路线; 2. 🟡 commit 本轮(Panda_Lorrain 身份); 3. ⏸ 模型路线全等数据结论+用户决策; 4. 🟡 A集合规待用户问主办方。

关联 memory: [[cer-breakthrough-candidates]] [[spk-oracle-poc]] [[h3-dicow-conditioning-backfire-vanilla]] [[mimo-asr-backend-potential]] [[content-gate-decision]] [[dataset-split-spec]] [[lessons-pitfalls]]。

---

## 【2026-07-08 最新】主办方 CER 口径坐实 + 提交归一漏洞修复

### 一句话现状
主办方 CER 口径脚本到手并坐实对齐（normalize_text: NFKC+lower+去 P* 标点和空白; CERMetric 累计池 total_err/total_char, editdistance 库）。落地 3 件：eval_metrics 加官方口径（逐行等价，12 边界 Δ=0）/ recompute_official_cer 重算全量 / **修 4-agent 对抗验证发现的提交归一漏洞**（cn2an/zhconv 原未声明依赖→主办方环境必缺→digit_postproc 静默失效→CER 0.595 实际回 0.661；已建 requirements.txt + RuntimeWarning 告警 + to_submission SSOT）。口径切换利好：vanilla overall 0.664→**0.595** / thr0.27 含拒 **0.703** / H3 dicow sim[0.2,0.3) **1.609** 更稳 / digit_postproc 收益 -0.033 坐实。**caliber-A 彻底解除**（2026-07-08 主办方问题二/三：pos 隐含允许拒 + 排名公式 `TotalScore=w1*(1-CER)+w2*RR` 公布 + RTF 按 batch=1 测）。

### 口径坐实（主办方脚本, 2026-07-08）
- 归一化：NFKC + lower + strip + 去所有 Unicode P* 标点和空白（含内部空格）。**不繁简归一、不数字归一**。
- 聚合：累计池 total_errors/total_chars（非逐条平均）。
- 库：editdistance.eval(norm_pred, norm_target)/len(norm_target)；空 target: errors==0→0 else→1.0；拒识条 pred 空→errors=len(ref)→CER=1.0 天然。
- 实测（1362 条）：overall 两口径差<0.01（可忽略）；ref 全简体+全中文数字+无标点（各 0/1364）；vanilla 243/1364 含繁体（提交侧 to_simplified 对冲）、49 含标点。

### 重算关键数字（recompute_official_cer.py，提交归一后, 累计池）
- 转写（不拒）：vanilla **0.5947** correct 48.8% / dicow **1.1894** correct 31.4% / 英文幻觉 vanilla 0.6% vs dicow 18.7%
- thr 含拒累计池：thr0.2 vanilla **0.6571** / thr0.27 **0.7030**（B 集统一 thr）/ thr0.4 0.8246
- sim 分桶：dicow sim[0.2,0.3)=**1.609** / [0.3,0.4)=1.522（H3 条件化反作用，官方口径更稳）；vanilla 0.659/0.641
- digit_postproc 收益：vanilla 仅繁简 0.6281 → +转数字 0.5947（**-0.033 坐实**）

### 已修代码（本 session，未 commit）
- `code/eval_metrics.py`：加 normalize_text / CERMetric（累计池）/ cer_pool / cer_official（照抄主办方，保留旧 cer）；cer_pool 防单条 str 误拆（workflow③）
- `code/recompute_official_cer.py`（新）：官方口径全量重算（转写/thr 工作点/sim 分桶/数字），产物 `recompute_official_cer.json`
- `code/requirements.txt`（新）：声明 cn2an/zhconv/editdistance/jiwer/numpy/soundfile/librosa/torch/transformers（修 workflow④ leak#1 根因）
- `code/text_utils.py`：digit_postproc / to_simplified 缺包 graceful 跳过改 RuntimeWarning（不再静默）
- `code/to_submission.py:49`：加 digit_postproc 成提交归一 SSOT（与 enroll_infer:317-319 / recompute submit_norm 对齐）
- `REPRO_SETUP.md §3`：补 `uv pip install -r code/requirements.txt` + cn2an/zhconv 必装警告

### 4-agent 对抗验证结论（workflow w1ry9l23r，全过）
①独立复算 `matches_mine=true`（0.5947/1.1894/thr0.27 0.703 零差异）；②转写一致性「核心一致，DF3 两文件均不前置，0.5947 可代表提交侧」+5 个 no-op 级低风险（attention_mask/seed/enroll-augment 等）；③口径实现逐行等价，12 边界全 Δ=0.00e+00；④提交漏洞发现 cn2an/zhconv 未声明（high，已修）+ to_submission 非 SSOT（medium，已修）。

### ⚠️ follow-up
1. 🟡 **commit + push**：本批改动未 commit（本地超前 `0aea5ca`）。建议合一个 commit「feat(eval): 官方 CER 口径对齐 + 提交归一漏洞修复」。注意 git 身份用 **Panda_Lorrain**（[[git-identity-mismatch]]），本机默认 midea-overnight-loop 需显式指定。
2. ✅ **caliber-A 2026-07-08 彻底解除**（主办方问题二/三确认）：pos **隐含允许被拒**（拒=CER1.0 无额外惩罚）+ 排名公式 **`TotalScore=w1*(1-CER)+w2*RR`** 公布（线性无惩罚项，per-sample 不封顶）+ **RTF 按 batch=1 测**（不做 batch 改造）。w1=w2（最可能 0.4/0.4）则 **thr=0.27 定稿**（T27 目标函数 (1-CER)×40+RR×40 与公布公式等价，无需重扫）；仅 RR-heavy 才上移。剩 w1/w2 具体值口头确认即闭环。
3. 🟡 **抽验（低风险）**：workflow② 提示 enroll_infer vanilla 与 exp json 的 attention_mask/seed 分歧（no-op 级），可抽 5 条 `enroll_infer --asr-backend vanilla` 输出 vs `exp_vanilla_full.json` 逐字比对彻底消除。

---

## 【2026-07-08 content_gate】转写内容有效性二次拒识（集成, 默认关, 详见 memory content-gate-decision）

### 一句话
RR 卡 90.5%（thr0.27）的 45 条漏拒 neg 转写多为新闻/英文/乱码 → 加 content_gate（sim≥thr 的 accept 再判转写是否家居指令, 非指令则拒）。**hold-out 证泛化（val +1.6 分, CI p5>0, L 不敏感）, 集成进 decide_reject 独立加拒通道, 默认关（BAODI_GATE=1 开）**。Pareto 改进：提 RR 不损 pos/效率, pos 侧顺带拒幻觉灾难降 CER。

### 用户方法论纠正（关键, 2026-07-08）
PoC 初版 +2.3 是 **in-sample 上界**（黑名单词看全 A 集漏拒 neg 定的）。用户指出过拟合风险（A 集是开发集 PDF 坐实, B 集不公开永拿不到, [[dataset-split-spec]]）。修正：**A 集分 train/val hold-out 验证泛化**（规则先验版, 唯一 train 拟合参数 len_thr, 最终用纯先验 20）。val ΔTS +0.0134, 全集 +0.0248 vs PoC +0.0256（过拟合水分仅 0.0008）。

### 集成（file:line）
- `text_utils.is_valid_command(text, len_thr=20)`：先验版（拒纯非中文/英文为主/通用非家居类目词繁简/超长, 默认保留）
- `submit_infer.decide_reject`（:55）：加 text+use_content_gate 参数 + 独立加拒通道（sim≥thr 且 not is_valid_command→拒, 不改原 sim/llm 逻辑）; :174 `--content-gate` flag（默认关）; :318 调用传 transcript
- `run_baodi.sh`：`BAODI_GATE=1` env 开关（保"锁死 flag 防灾难"语义）
- enroll_infer/to_submission 不改（transcript 字段, gate 在融合层）
- smoke 确认：sim0.371 '解剩所有的物料画面价格比较'（乱码）→ 拒; sim0.351 '关门的吗'（指令）→ 放

### hold-out 数字（code/exp_content_gate_holdout.py）
分割 pos 688/674 neg 228/246（uid md5 hash 固定）。val（len_thr=20 先验）：ΔTS +0.0134（+1.6 分/80 满分）| ΔRR +0.045（主力）| ΔCER +0.011（微亏, pos 误拒代价小）| bootstrap CI（B=400）p5=+0.007 p95=+0.024 稳赚 | L 不敏感（18-30 全正）| pos 误拒 9 条原 CER mean 0.98（CER≥1 占 89% 反赚）。

### 决策：默认关
hold-out 证泛化但 B 集未知（B 集干扰分布可能不同）→ 默认关, BAODI_GATE=1 开。这次提交保守不拿收益, 换不反噬 B 集。赛后 B 集复盘（赛事方测）再定默认开。

---

## 【2026-07-08 P1 oracle】证伪声纹强化攻死区 → 死区是 babble 摧毁 mel 物理极限

### 一句话
oracle POC（code/exp_spk_oracle.py, 60 条死区抽样, 详见 memory spk-oracle-poc）四组证据全指向 **GO=否**: 死区（pos sim<0.2, 29%）是 babble 摧毁 mel 的物理极限, 非选择器工程缺陷。**不投 CAM++/帧选择/US-PVAD**（避免五连受挫）。CER 腿破局只剩 SepFormer 源分离（高成本）或接受 vanilla 0.595 天花板。

### 四组证据
1. argmax 选对率 66.7%（多数选对）2. oracle_sim≥0.2 占 0%（正确 target 声纹也全不可识别）3. miss 20/20 声纹反向指错 4. 单 speaker 控制组 n=18 CER 0.436（target 唯一仍转不出）。
- oracle_CER 0.607（作弊完美选 target 仍>0.5 不及格）vs argmax_CER 0.788; diar 确定性 |Δ|=0.0025。

### 决策影响
- CER 腿声纹入口关闭（证伪 mimo-asr-backend-potential "声纹强化最高杠杆"）
- 答辩弹药升级: 单 spk 控制组 + miss 声纹反向 = 诚实归因硬证据（死区=物理极限非工程缺陷, 契合反 cascaded 审美）
- 下一步: 转 P2 答辩交付刷新（决赛 70 分）或 SepFormer 源分离 POC（高成本, 需另立项）

---

## 【2026-07-08 SepFormer+P2】SepFormer 源分离 POC 证伪 + 答辩交付刷新

### SepFormer = NO-GO（CER 天花板确认）
SepFormer POC（code/exp_sepformer_poc.py, 40 条死区）证伪: 盲分离拎不出 target（CER 0.859 vs vanilla 0.918 Δ-0.059 噪声带, correct 反降, oracle 0.752 反劣 diar-oracle 0.607 = 分离动作本身劣化 target）。**六连受挫**（langfix/STNO/SE-DiCoW/enroll-augment/声纹强化/SepFormer）→ 组合主线极重 babble 下 CER 能力极限。接受 vanilla 0.595 天花板。唯一剩 enrollment-conditioned TSE（SpEx/TF-GridNet）大工程（P1 单 spk 0.436 地板, 收益不确定, 超 POC 范围）。详见 memory spk-oracle-poc。

### P2 答辩交付刷新（4 文件, commit 5156fd5）
03FAQ（07-08 横幅 + 红线6 真测 + 风险12 L20未测/13 边缘部署）/ 使用说明（删虚构CLI + run_baodi 入口 + content_gate + JSON schema）/ 测试验证方案（仿真450→真测1362）/ 设计报告（主线 vanilla 反 cascaded）。
⚠️ follow-up: make_readme_progress.py 仍读仿真（PNG 标题 CER4.27）, 需重写读真测（答辩开场视觉矛盾风险）。PPT/演练待 ppt-master。

---

## 【2026-07-07 晚 最新】MiMo-V2.5-ASR 调研闭环 + 数字后处理集成

### 一句话现状
调研小米 MiMo-V2.5-ASR（开源 audio-LLM ASR）作 cascaded 后端候选：全量 1362 条实测 CER **0.417 vs vanilla 0.661**（纯文字句 0.428 vs 0.637 验证非口径红利），但云端不能进提交（L20/数据安全/可复现三红线）+ 蒸馏不现实（数据/容量/范式三障碍）。从对比中学到**数字后处理 quick win**（vanilla 阿拉伯→中文数字对齐 ref，全量 0.661→0.632），已集成 text_utils/enroll_infer（commit `0aea5ca` 未 push）。下个 agent 焦点 = T27 follow-up（闭环主办方口径/B 集混合提交）仍是最高优 + 本次 MiMo follow-up（cn2an 依赖/本地 RTF/答辩素材）中优。

### MiMo 调研结论（详见 memory `mimo-asr-backend-potential`）
- **定位**（源码核实）：MiMo 是通用/多说话人 ASR，**不支持 enrollment**（`asr_sft` 只 audio+audio_tag 两参数；WebSearch 说有 enrollment 是 LLM 幻觉）。非 TS-ASR，只能当 cascaded 后端转写器，配合 diar+声纹选 target。
- **全量实测**（`code/exp_mimo_asr.py`）：MiMo 0.417 vs vanilla 0.661；纯文字句 0.428 vs 0.637（真实能力，非口径）；英文幻觉 2.1%（vs DiCoW 18.8%）；逐条 mimo 更优 654/更差 130/持平 578。
- **不能进提交**：云端（`token-plan-cn.xiaomimimo.com`，tp-key 专用 base URL，非按量 `api.xiaomimimo.com`）不满足 L20 本地 RTF + 测试集上传 + 可复现性。进提交唯一路径=本地部署开源 MiMo，但 audio-LLM RTF 风险高，**未测**。
- **蒸馏不可行**：伪标签蒸馏传"输出标签"非"听音能力"，重 babble 是 student 盲区；数据硬伤（真实测试集不能用，仿真≠真实）。完整蒸馏不现实。
- **能力边界**：MiMo 依赖 target 切片质量，低 sim 重 babble（diar 切错）MiMo 也翻车。

### 已集成（commit `0aea5ca`，未 push）
- `code/text_utils.py` `digit_postproc`：百分比/纯数字(1-3 位,0-999)→中文，≥4 位幻觉串保留；graceful cn2an（未装不崩）；code-review 修正则加 `(?<!\d)` 防≥4 位末尾部分转换。
- `code/enroll_infer.py` 第 317 行 `to_simplified` 后统一调用（dicow+vanilla 覆盖，默认开无需 flag）。
- 效果：vanilla 全量 0.661→**0.632**，含数字句 0.739→**0.608**，121/315 条改善。稳健：两口径都不亏（不归一化数字→赚 0.029；归一化→平），对冲 caliber-A。
- `code/exp_mimo_asr.py` + `exp_mimo_asr_result.json`：实测脚本 + 全量结果（对照基准）。
- 修 DiariZen 恢复活路径（从 `.bak` 备份挪回 `code/DiCoW-inference/DiariZen/`，修 enroll_infer/submit_infer 的 diarizen import；`.bak` 保留双保险）。
- `.gitignore`：API keys 节（`.mimo_apikey`/`.env`）+ `code/_*.json`。

### ⚠️ follow-up（本次新增，中优）
1. 🟡 **cn2an 依赖记录**（可复现性）：已装 `code/.venv`，项目无 pyproject/requirements，复现需 `uv pip install cn2an`（text_utils docstring 已注）。建议建 `code/requirements.txt` 或补 `REPRO_SETUP.md`。
2. 🟡 **MiMo 本地部署 RTF**（进提交前提）：下开源 MiMo + 测 L20 RTF。RTF 可接受→集成 `--asr-backend mimo`；不行→仅答辩用。
3. 🟡 **答辩素材固化**：MiMo 对比（CER 减半/纯文字句验证/英文幻觉低）作"业界强基座对标，证明 cascaded 后端天花板可压"素材补进 `03_答辩FAQ`。

> T27 follow-up（🔴 闭环主办方口径 / 🔴 B 集混合提交 / 🟡 灰区 LLM）仍是最高优，见下【2026-07-07】段。本次未 push（`0aea5ca` 本地），下个 agent 决定是否合并 push。

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
6. ⚡ **L20 端到端耗时真测**：submit_infer 显存自适应（L20 48G 大 batch）+ 租 AutoDL L20 验证（官方 L20 评效率，本机仅 4060，memory `l20-eval-hardware`）
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
3. ⚡ **L20 端到端耗时真测**：submit_infer（vanilla）显存自适应（L20 48G 大 batch）+ 租 AutoDL L20 验证（官方 L20 评效率，本机 4060，memory `l20-eval-hardware`）
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
6. ⚡ **L20 耗时验证**：submit_infer（含 vanilla 后端）显存自适应（L20 48GB 大 batch）+ 租 AutoDL L20 验证端到端（官方 L20 评效率，本机仅 4060，memory `l20-eval-hardware`）
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

## 【2026-07-23 最新】用户战略复盘 + CER 提升新方向（待验证）

> **背景**：用户亲自听音频、分析数据集，重新审视之前证伪的实验。提出关键假设：**之前证伪的实验是在 DiCoW（较差模型）上做的，如果在 Qwen3-ASR（更好模型）上重跑，可能结果不同**（"同一种方法对学渣没用，对学霸可能有用"）。

### 本次做了什么
1. **用户亲自听音频**：听NEG集的11条"非目标说了家居指令"样本，确认任务本质是判断"是谁说的"而非"说了什么"
2. **LLM拒识分析**：详细列出LLM救回28条pos、放过11条neg的具体样本
3. **数据集A噪声分析**：POS集89% babble噪声，NEG集100%干净音频
4. **thr trade-off量化**：thr=0.27 vs thr=0.4对比，净亏4.65分（RR+3.21 vs CER-7.85）
5. **CER提升方向梳理**：更好的ASR模型、babble噪声优化方法

### 关键发现

**① LLM拒识根本局限**：
- LLM只能判断"说了什么"，不能判断"是谁说的"
- 任务要求判断"是谁说的"（目标说话人 vs 非目标）
- 所以LLM在这个任务里帮不上忙，关掉是正确的

**② NEG集比POS集简单**：
- POS集：89% babble噪声，声纹匹配困难，CER高
- NEG集：100%干净音频，声纹匹配容易，RR高
- 如果正式测试NEG集也有babble噪声，thr可能需要调整

**③ thr=0.27是最优点**：
- thr=0.27：CER腿13.38 + RR腿36.20 = 49.58分
- thr=0.4：CER腿5.52 + RR腿39.41 = 44.93分
- 净亏4.65分，调高thr得不偿失

**④ CER提升空间**：
- 当前669条pos被拒识（49%），是CER主要损失
- 被拒识样本平均sim=0.1754，低于thr=0.27
- sim在[0.27,0.4)的被拒识样本只有11条，提升空间有限

### CER提升方向（待验证）

**方向1：更好的ASR模型**
| 模型 | 优势 | 证据 |
|---|---|---|
| Qwen3-ASR-1.7B | 最对症babble | ExtremeNoise WER 4×优于Whisper |
| FireRedASR2-AED | 更快+中文强 | WenetSpeech-meeting 4.2×，RTF 0.087 |
| SenseVoice-Small | NAR极速+ITN | 数字归一化，解决cn2an漏洞 |

**方向2：针对babble噪声优化**
| 方法 | 原理 | 状态 |
|---|---|---|
| ASE-PVAD自增强 | 从混合语音收割target真实帧 | 未在Qwen3上测试 |
| CTC head反幻觉 | 强制单调对齐压insertion幻觉 | 未测试 |
| 家居热词注入 | 15264句家居指令构热词表 | 未测试 |

**方向3：在Qwen3-ASR上重跑之前证伪的实验**（用户核心想法）
| 实验 | 在DiCoW上的结果 | 在Qwen3-ASR上可能的结果 |
|---|---|---|
| 声纹强化（CAM++） | 证伪（sim提升有限） | 可能有效（Qwen3基础好，sim提升能转化为CER提升） |
| SE语音增强 | 证伪（CER+0.1049恶化） | 可能有效（Qwen3更鲁棒，能承受SE失真） |
| 源分离（SepFormer） | 证伪（EoW 2026证伪级联） | 可能有效（Qwen3+源分离可能协同） |
| LLM拒识 | 证伪（RR下降+RTF 4×） | 可能有效（Qwen3转写更准，LLM判断更准） |

### 下个 agent 待办
1. 🔴 **在Qwen3-ASR上重跑声纹强化POC**（验证用户核心想法，工程量最小）
2. 🔴 **在Qwen3-ASR上重跑SE语音增强POC**（验证用户核心想法）
3. 🟡 **FireRedASR2-AED横评**（方向1，可能比Qwen3更快）
4. 🟡 **ASE-PVAD自增强在Qwen3上测试**（方向2，出题方方法）
5. 🟡 **CTC head反幻觉在Qwen3上测试**（方向2，压insertion幻觉）

### 关键认知（本次坐实）
- **任务本质是判断"是谁说的"**：不是判断"说了什么"，所以LLM帮不上忙
- **之前证伪的结论可能只适用于DiCoW**：在更好的模型上重跑可能有效
- **NEG集比POS集简单**：当前RR高是因为NEG集干净，正式测试可能变化
- **CER主要瓶颈是49%的pos被拒识**：大部分是因为babble噪声导致sim变低

### 产物（本次聊天）
- `code/compare_llm_impact.py`（LLM拒识对比分析脚本）
- `code/export_datasetA_official.py`（数据集A导出CSV脚本）
- `code/datasetA_官方数据.csv`（数据集A官方原始数据）
- `C:\Users\26875\Desktop\音频分析\neg_非目标说家居指令\`（11条音频样本）

---

**给下一个 agent 的话**：⚠️ **2026-07-06 Phase 1 已破局**——组合主线 CER 1.4 是 **DiCoW 条件化路径**的极限（非任务极限），改用 vanilla Whisper + 声纹切 target timeline（zero-training）CER 已减半到 0.664、overall thr=0.2 拉到 0.711（CER 40% 腿 0→11 分）。**最高优 P2：vanilla 集成 submit_infer（`--asr-backend {dicow,vanilla}`）把 0.664 变提交数字**，无需大工程。保底（关 LLM+thr=0.4，RR 98.5%）仍作 fallback。langfix/STNO/enroll 增强/SE-DiCoW 在 cascaded 框架内试过无效，别重试。所有踩坑见第 8 节。

⚠️ **2026-07-23 新方向**：用户提出核心假设——之前证伪的实验是在DiCoW（较差模型）上做的，如果在Qwen3-ASR（更好模型）上重跑，可能结果不同。**优先验证这个假设**（声纹强化+SE在Qwen3上重跑），再决定是否投入其他方向。
