# 美的目标说话人 ASR 参赛方案（XH-202615）

> ⚠️ **2026-07-29 更新**：**TSE「唯一治本路径」阶段二全量验证 → 仍 NO-GO（声学翻正，CER 不动）**。07-28 结论"多信号死结→TSE 唯一治本、等租 L20 训练"在 07-29 实际训了 WeSep pBSRNN + 5 轮验证，**全量 NO-GO**，07-28"TSE 唯一治本/等算力"表述**已被推翻**。**① 阶段一纠错**：自写 V1 训练 4 个失真 bug 全修（独立裁剪致 SI-SNR 错位 / 2s 音频配整句 CTC / complex mask 误实现成实虚部逐元素乘 / enrollment 同句截取内容泄漏）+ speaker-disjoint train/val manifest + `tests/test_tse_phase1_logic.py` 6 项通过。**② 阶段二 WeSep pBSRNN**（160 train / 40 val speaker-disjoint triples，WeSep 固定 `99eca54`，CAM++ 512d 需 `joint_training=False,use_spk_transform=False`）：声学侧明确有效——中型 BSRNN val SI-SNRi **+1.0956 dB**，数据 balanced profile 回修后 **+2.264 dB**、Qwen mel L1 0.6146→0.4264（−31%）、wave L1 2.196→1.262（−43%）、输出 RMS 失配 5.04×→0.64×；**但识别侧纹丝不动**——冻结 Qwen 配对 CER 1.2870→1.2671（Δ**−0.0199**，bootstrap 95% CI [−0.1162,+0.0831] 跨 0），未过 ΔCER≤−0.05 门槛 → **无条件全段 TSE NO-GO**。**③ overlap-only + CAM++ cosine fallback**：SI-SNRi +0.48/+0.50 dB，ΔCER −0.0144/−0.0126 仍 NO-GO。**④ Qwen mel failure proxy + speaker-LOSO**（Qwen 自带 128-bin Whisper 前端 logmel_l2 做线上"该不该回退"判据，4 折 leave-one-speaker-out）：raw CER 1.2870→routed 1.2419（Δ**−0.0451**，CI 上界 +0.0109 仍>0），4 折阈值稳定 0.3148 比 CAM++ cosine 更贴 ASR 失败但仍未过双门槛，**不集成**。**⑤ 核心发现=感知-识别鸿沟坐实**：separator 声学收益（SI-SNRi/mel/wave 全面翻正）**没有传导到 ASR CER**，印证 EoW 论文警告；160 triples 规模下 TSE 到顶。**⚠️ 工程教训**：SI-SDR 尺度不变致 BSRNN 输出 RMS 达 mixture 3.61–6.52×（旧 PCM 导出隐式削波，overlap 拼接必须显式 gain matching）；Qwen 对同 SHA256 的 raw WAV 跨进程复跑仍有 +0.0126 CER 漂移 → 后续小收益实验**必须同进程配对**。**⑥ 唯一未试下一步=Phase-3**：冻结 Qwen audio encoder 前 1/3 层加小 Sidecar/STNO bias + target activity head（新增参数<5%），用 **ASR loss 直接反传**（不再只优化 SI-SNR/mel 代理）；若仍不过门槛则 TSE 路径彻底封顶。**提交线不变**：主线 qwen 含拒 thr0.27 CER 0.5934（CER 腿 16.26）；07-28 分场景路由 +0.83 未集成主线。**🎧 待办**：①Phase-3 ASR loss 反传 POC（TSE 唯一未试路径）②问主办方 w1/w2 权重 ③cmd_18 数据错配排查。**详见** `AGENT_HANDOFF.md`【2026-07-29】四段 + `docs/tse_train_plan.md` + memory `tse-phase2-full-nogo`。

> ⚠️ **2026-07-28 更新**：**分场景路由反转 multi-voice NO-GO（全量坐实）+ 多信号拒识 task6 强 NO-GO 放弃 + TSE 方案就绪等算力**。本轮 6 task 全量验证+定位真瓶颈，**未 commit**（用户把关身份和未验证标注）。**① 分场景路由★★★反转**：按 diar 分出 n_spk 路由——n_spk=1（单人 40%）走主线不分离；n_spk=2（重叠 60%）SepFormer 分离+二选一。机制坐实：SepFormer 单人破坏 Δ+0.165/重叠救回 Δ−0.157，无差别混算抵消看似 net neutral，**分场景后净正**。全量 1350 pos：transcribe CER **0.3427→0.2941（−14.2%）**/含拒 thr0.27 **0.5931→0.5727（−3.4%，CER 腿 16.26→17.09 +0.83）**。**外推 −20.7% 没坐实**（采样偏差：483 抽样里 43% 主战场，全 805 池里 99% 死区，全量坐实救了避免误集成）。**② 含拒收益小根因（收益漏斗）**：transcribe −14.2% 被四层吃到含拒 −3.4%——heuristic 选错 16.6%/死区占 n_spk=2 的 99% 收益小/n_spk=1 无空间/**含拒 thr0.27 拒掉 55% n_spk=2（445 条 CER=1.0 最大损失）**。**③ 多信号拒识 task6 强 NO-GO（放弃）**：task4 pos 侧有效（含拒 CER 0.5937→0.4468 −22%，CER 腿 +5.88，precision 83% 救回 187 条），但**task6 neg 474 条实测强 NO-GO**：neg RR 损失 **36.71%**（174/474，是代理上界 8.3% 的 4.4 倍），neg RR 0.9051→0.5380，**net −8.81**（pos +5.88 被 neg −14.68 反噬超额），收紧 kw 几乎无效（174→168）。**根因=架构性死结**：174 条错误 rescue 里 29-49% 是**干扰人说了完整家居指令**（"打开空气净化器""开启烹饪模式""把灯打开""打开洗碗机"），文本上和真 target 无法区分——内容信号"文本像指令"pos/neg 对称失效（pos: target 说指令→选指令路=target 有效；neg: 干扰人也说指令→选指令路=干扰人=漏拒 失效），"文本像指令"分不开是谁说的，声纹 sim 在 SepFormer 后失效，死结。**教训**：pos 另一路（干扰人说闲话）通过率 9.19% 代理 neg，但 neg 干扰人就是说家居话（题目设计）通过率 40.56%，pos/neg 干扰人行为不同不能 pos 代理 neg。**结论：多信号 rescue 放弃；分场景路由+0.83 仍成立（不依赖内容补救）；TSE 唯一治本**（enrollment 引导分离 target 绕开死结；⚠️此判断已被 07-29 阶段二全量 NO-GO 推翻，见上方 07-29 段）。**④ 选路天花板+真瓶颈定位（task2 坐实）**：选路 oracle 含拒 CER **0.5524**（heuristic 0.5733）选路侧到顶空间只 0.021；LLM 二选一选对率 78%<heuristic 84%（用户"LLM 更强"假设不成立）；真瓶颈=sep 质量（23.4% 样本两路都烂）+ thr 拒 55%。**⑤ TSE 方案就绪（task5 等算力）**：V1=STFT 复数 mask TSE+ECAPA-lite+联合 loss（−SI-SNR+CTC 避纯 SI-SDR 陷阱），三红线对齐，冒烟通过；推理红线 TSE 只分离→喂 qwen3 ASR；L20 ~8h ~64 元/round 等用户租，GO 判据死区子集 ΔCER>0.05。**⑥ cmd_18 数据错配**（enrollment 男声/recognition 女声，官方素材问题须排查）。**⑦ 用户方法论胜利**：全量坐实不信抽样（task1 修正 −20.7%→−14.2%）+人耳判据（听 cmd_146/2637 坐实分场景）。**🎧 待办**：①~~task6 neg 验证~~（已完成，强 NO-GO，多信号 rescue 放弃）②~~租 L20 训 TSE~~（07-29 已训完 WeSep pBSRNN + 5 轮全量 NO-GO，见上方 07-29 段）③问主办方 w1/w2/w_eff 权重。**详见** `AGENT_HANDOFF.md`【2026-07-28 最新】段 + memory `scene-route-overturns-multivoice-nogo`/`multi-signal-reject-rescue`/`multivoice-content-routing-is-mainline`/`overlap-is-cer-failure-rootcause`。

> ⚠️ **2026-07-27 更新**：**消除信息隔阂 + 推翻"到顶" + 5 路轻量改进全证伪 + 战略转向自训中文 TSE**。用户挑战 memory 旧结论"0.3436 物理天花板"，逐环节核实 + 人耳听样本。**① 推翻到顶**：死区(78.8%大头)真地板仅~10%(用户人耳13条坐实)，90%可修，理论空间 0.3436→~0.15。**② 真瓶颈链(非mel摧毁)**：diar分对(emb0.219可分)→cut_target_timeline切重叠区(物理混合)→SepFormer选路失效(sim25%,SI-SDR破坏声纹)。**③ 得分70-73**(CER腿16.26短板/RR37.97近满分/效率18-19估)。**④ 5路全证伪**：multi-voice整体/selector/SepFormer盲分离/微调POC(合成域不匹配Δ+0.147)/TSE英文权重(zero-shot中文OOD)——共性:轻量实现证伪,方向oracle有效,要兑现得重投入。**⑤ multi-voice诊断澄清**：真双指令仅25%(非"92%双accept",那是宽松gate)，NO-GO真因=SepFormer破坏mel(非选路)。**⑥ 外部训练允许**(主办方美的_张志飞2026-07-27确认,A集外数据训练合法)。**🎧 待办**：①用户正在听B类样本(`code/runs/_verify_mv_fail/B_*`)验证SepFormer破坏mel ②启动自训中文TSE(Aishell1Mix已下E:/midea_datasets+租算力,攻分离治本)。**详见** `AGENT_HANDOFF.md`【2026-07-27最新】段 + memory `overlap-is-cer-failure-rootcause`/`multivoice-content-routing-is-mainline`/`external-training-allowed`/`core-leader-delegate-mode`。本轮10+commit全push。

> ⚠️ **2026-07-19 更新**：稳定性/鲁棒性测试闭环(spec+plan+代码+26遍实跑+报告, push GitHub)。**核心结论**: ①**R1=0**系统 greedy argmax 完全确定可复现(A 同种子10遍+变种子10遍 0 波动, 不修 use_deterministic) ②**R2 纯仅2条**(batch1vs16 差异74条中72含 R3/R4 叠加)→开发 batch16 数字基本可外推提交 batch1, submit 锁 batch1(3398c0d) ③**R3 57%**输入微扰敏感(gauss 加性噪声54%主因破坏 mel 谐波)→模型泛化短板**归档**(A 集外训练才能修, A 集不能训练 lessons-pitfalls§14) ④R5=0 变种子不翻车。波动740条 sim 分桶死区+低sim占53%。**主办方口径(美的_张志飞)**:默认 batch1/高 batch 一致才行/RTF 按 batch1。答辩弹药:可复现性达标+batch 口径已修+诚实归因。详见 `docs/稳定性测试报告_2026-07-19.md` + memory `stability-test-launched`。

> ⚠️ **2026-07-18 更新**：效率腿探索闭环 + **SE orphan bug 真相**(推翻原归因) + L40 阶段0 就绪。①效率腿 3 commit(qwen batch ASR子进程5x/int8+compile探索/SE bugfix c8c739d) ②**SE bug 坐实**: `submit_infer.py` rec_for_enroll 死变量致 se_out 从未消费, SE 全程空转30.6%RTF; bugfix后真生效反而 overall CER+0.1049恶化——**三机制(sim mismatch误拒66%+DF3过衰减致diar崩溃22%+转写恶化12%),非仅mismatch**; 关SE Pareto最优(分桶无反例),双端SE不值得测 ③对抗审查修正(6ce0636)+`audit_se_bugfix.py`一手复核(审查自身也错1处翻转386,实证纠偏) ④L20阶段0脚本(`deploy_l20.sh`/`run_efficiency_l20.sh`)+runbook勘误(2c095de; 2026-07-23 l40→l20 正名+L20/L40≠折算勘误), **等用户租算力**(路径A ssh操控)。主线 qwen CER0.3436/含拒0.5934 不变。详见 `AGENT_HANDOFF.md`【2026-07-18最新】段 + memory `se-bug-orphan-truth` + `docs/SE_bugfix_AB结果_2026-07-18.md`。

> ⚠️ **2026-07-15 更新**：多声纹 LLM 路由方向验证闭环 + content_gate v2 集成 + 后端差异发现。①多声纹→LLM 挑 target 路线**证伪**(POC A 判别逻辑成立 e0.94/recall0.75, 但 POC B 端到端 124 all_reject 挑不到 target + 全量启用 25% 误拒恶化) ②全量死区 126 确认**修正 POC B 解读**(拒幻觉 CER 价值: argmax 2.34→mv 0.99 改善 126/126, 全量仅死区 0.595→0.476) ③content_gate v2 集成(幻觉检测循环/div+NEWS_KW+digit_postproc 时间归一+标点修复) **后端差异**: vanilla 改善 Δ-0.0406 / **qwen 恶化 Δ+0.024**(qwen 鲁棒死区少+指令多样误拒)→**qwen 主线不用 content_gate** ④死区物理极限三重坐实(spk-oracle+mimo+content_gate) ⑤副产品: pos.ref 31% 非纯指令 / 分桶坐实(多 spk 失败 67%/80%)。主线 qwen CER 0.3436(含拒 0.5934)不变。详见 `AGENT_HANDOFF.md`【2026-07-14/15】段 + memory `multi-voice-llm-routing-architecture` + `docs/session经验总结_2026-07-15.md`。

> ⚠️ **2026-07-11 更新**：前沿探索闭环 + **Qwen3-ASR 候选2 证实 + 集成落地 + P0 数字收尾(7-agent 核实双口径坐实)**。19路并行探索报告 `docs/前沿探索报告_2026-07-10.md`(候选1-5裁决+路线图)。3 POC: faster-whisper int8 **NO-GO**(4060 RTF慢38%+CER+0.0156) / BoH+delooping **NO-GO**(与content_gate冗余) / **Qwen3-ASR-1.7B 全量1350条 transcribe CER 0.3436 vs vanilla 0.5954; ⚠️双口径(2026-07-11 P0 坐实): transcribe 不拒 +10.07(诊断上限) vs 含拒 thr0.27 提交 0.5934→+4.29(排名公式实际用, pos 允许拒 2026-07-08 确认); 归一零效应(0阿拉伯数字0繁体 raw==归一==0.3436); 死区0.459<oracle0.607挑战物理地板待对抗验证; 产物 `recompute_qwen_official.py`+`qwen_official_cer_workpoints.json`**, drop-in集成(`submit_infer --asr-backend qwen`)。code/.venv speechbrain lazy修复(patch inspect.getmodule固化enroll_infer顶部, 解锁后续所有切片)。**教训: 新依赖一律独立venv(如code/.venv_qwen), 不污染主venv**(装faster-whisper到主venv连带升级共享包触发speechbrain lazy崩, 已回滚)。详见 AGENT_HANDOFF.md【2026-07-11最新】段 + memory `cer-breakthrough-candidates`。

> ⚠️ **2026-07-08 更新**：主办方 CER 口径脚本到手并**坐实对齐**——`normalize_text`(NFKC+lower+去 P*标点和空白) + `CERMetric` 累计池(total_err/total_char, editdistance 库)。落地三件：①`eval_metrics.py` 加官方口径(逐行等价,12 边界全 Δ=0)；②`recompute_official_cer.py` 重算全量；③**修提交归一漏洞**(4-agent 对抗验证发现 cn2an/zhconv **原未声明依赖**→主办方环境必缺→digit_postproc 静默失效→CER 0.595 实际回 0.661；已建 `code/requirements.txt` + RuntimeWarning 告警 + to_submission SSOT)。**口径切换利好**：vanilla overall 0.664→**0.595** / thr0.27 含拒 **0.703** / H3 dicow sim[0.2,0.3) **1.609** 更稳 / digit_postproc -0.033 坐实。caliber-A 风险降级。详见 `AGENT_HANDOFF.md`【2026-07-08】段 + memory `official-scoring-spec`。

> ⚠️ **2026-07-07 晚更新**：MiMo-V2.5-ASR 调研闭环（全量 1362 条实测 CER **0.417 vs vanilla 0.661**，纯文字句验证非口径红利；云端不能进提交三红线 + 蒸馏不现实数据/容量/范式三障碍）+ **数字后处理集成**（vanilla 阿拉伯→中文数字对齐 ref，全量 0.661→**0.632** / 含数字句 0.739→0.608；`text_utils.digit_postproc` + enroll_infer，commit `0aea5ca`）。**最新状态/待办见 `AGENT_HANDOFF.md`【2026-07-07 晚】段（下个 agent 必读）**。

> ⚠️ **2026-07-07 更新**：统一 thr 选点 **T27 完成**（B 集 A 集模拟 + 5-agent 对抗验证 → **统一 thr=0.27** 区间 [0.26,0.29]，分 thr oracle 损失 6.46 分；已修守卫/run_baodi B 模式；⚠️ thr 待主办方 CER 口径定，caliber-A 假设未坐实是核心风险）。

> ⚠️ **2026-07-06 晚更新**：可复现性改造 T26 完成（核查 6 项硬要求全达标，fp16 run-twice delta=0 无需 fp32）+ vanilla 集成 T25 完成（pos CER 0.667 / neg RR 98.52%）。**最新状态/待办见 `AGENT_HANDOFF.md`【晚】段（下个 agent 必读）**。下面"当前阶段/下一步候选"部分已过时（vanilla 集成不再待办，可复现性已完成）。

## 项目概况
- **题目**：XH-202615《复杂交互场景的抗干扰语音指令识别技术》（美的集团发榜）
- **任务**：给定唤醒音频（enrollment，目标说话人），在带噪（SNR −5~5dB）+ 多说话人重叠（≤2人，0–100%）的识别音频中**只转写目标说话人指令、拒识非目标**
- **评分**：目标 CER 40%、拒识率 40%、推理效率 20%（L20 GPU）
- **当前阶段**：✅ **真实测试集 A 真测基线已出 + 组合主线架构极限确认**（2026-07-04）。历史：T14 pipeline(diar+STNO+DiCoW)→T17 wespeaker 锁定→T18 SE/CAM++证伪→T19 langfix→T20 SE 条件化→T22 babble 归因(H3 确证) + 仿真 450 画像(可用率 14%/babble 死区 0%)。**2026-07-04 真测**：datasetA 到手(pos 1364 测 CER + neg 474 测 RR，**单通道** 16k/16bit mono，enrollment ~1.8s 超短，唤醒词 20 种) → **单通道确认**(DSENet/VSAEC/DOA/KWS 空间路线全弃，组合主线是唯一路线)。真测适配(`make_pairs_from_datasetA`+`eval_datasetA`+繁简归一 zhconv) + Gap3 批量化(`enroll_infer --pairs` 模型加载1次，enroll 39→11s) + langfix 加强。**真实基线**：pos CER **1.25**(correct 31%)/neg RR **77%**(thr=0.2，可调 **98.5%**@thr=0.4 / 99.2%@thr=0.45)/RTF 0.16–0.24(4060，pos 全量 13.8min batch)。三方案攻短板全受挫：langfix 边际(英文 31.6%→18.5%，CER 仅降 0.028)/STNO 无效(babble 重 sim 0.024→0.038)/SE-DiCoW 架构不兼容(mt_num_speakers=2 多 speaker+self-enrollment 范式，短音频 OOD)。⚠️ **核心认知**：组合主线 cascaded(wespeaker+diar→STNO→DiCoW) 在极重 babble 下 CER **1.25–1.4 是能力极限**(target 声纹 median sim 0.28 / 低 sim 桶 correct 仅 30% + mel 退化转写崩；注：sim<0.06 仅 7.7% 非主流，答辩别引用 sim<0.06)，调参/小改破不了。**保底决策（2026-07-04 真测三档 + 3-agent 对抗审查修正）**：**关LLM**（`--no-llm`，**trade-off 非全面优于**：关LLM 赢 neg RR 98.5%>96.2% + RTF 0.24<1.01 4×；**开LLM 赢 pos 救回**——28 条 LLM 救回的 pos 里 26 条 CER=0.000 完美，原"pos 持平"错；选关LLM = 为效率20%+RR40% 牺牲 pos 救回）+ sim_thr 待主办方口径定（CER 均值→0.4/0.45 / correct→0.2 / pos 不许拒→0）。⚠️**提交用 `code/run_baodi.sh` 锁死 flag**（submit_infer 默认 llm_or_sim/thr0.2/llm ON = 灾难）；**CER 均值是幻觉陷阱**（correct_rate 才诚实 31%@thr0.2→14%@thr0.4）；**pos CER ~1.0 无 thr 能救**（架构极限）；**CER ±0.04 噪声**（langfix 边际 0.028 在噪声内不可靠）；**L20 batch=1 未实现**。详见 memory `baodi-config-no-llm`+`datasetA-spec`
- **2026-07-06 Phase 1 突破**：✅ **H3 强证伪**——DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】，改用 **vanilla Whisper-large-v3-turbo + 声纹切 target timeline** 路线（zero-training，全量 1362 条 pos）。**转写 CER 几乎减半**：vanilla **0.664** vs dicow 1.248；correct_rate vanilla **45.6%** vs dicow 31.4%；near_perfect vanilla 20.8% vs dicow 14.8%；**英文幻觉率 vanilla 0.59% vs dicow 18.80%**（DiCoW 条件化主动造孽，langfix 是治标，vanilla 从根消灭）。thr=0.20 overall CER：vanilla **0.711** vs dicow 1.241（vanilla 终于把 overall 拉到 <1）→ **CER 40% 腿从 ~0 分变 ~11 分**（线性 (1-0.711)×40，待主办方 CER 口径确认）。sim 分桶铁证 DiCoW 条件化最毒：sim[0.2,0.3) vanilla 0.746 vs dicow **1.606** / sim[0.3,0.4) vanilla 0.623 vs dicow **1.523**（Δ-0.90，条件化最反作用）。机制：diar+wespeaker 选 target（复用 enroll_infer）→ 切 target timeline 段（含重叠区）拼接 → vanilla Whisper 转写（去掉 stno_mask/FDDT 条件化）。**答辩弹药**：「cascaded 条件化机制在极重 babble 下反作用（sim 0.2-0.4 桶 CER 1.5-1.6、英文幻觉 18.8%），改用 target extraction + vanilla Whisper，CER 几乎减半」——契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。产物 `code/exp_vanilla_vs_dicow.py`+`analyze_vanilla_full.py`+`exp_vanilla_full.json`，memory `h3-dicow-conditioning-backfire-vanilla`

## 📂 文档导航（按此顺序读）
| 文档 | 作用 |
|---|---|
| **docs/00_技术路线总纲与行动地图.md** | ⭐ 入口：全局架构 + 评分→模块映射 + 行动甘特（W1–W7）+ 单/多通道双预案 + 差异化矩阵 |
| **docs/01_模块技术细节全解_答辩级.md** | M1–M7 每模块原理+公式+设计选择+开源资产+答辩问答 + 真实数据迁移 + 15 核心问答 |
| **docs/02_上限候选深读.md** | 候选 X（端到端联合，主押）vs 候选 Y（音频大模型，探索），含前沿对标 |
| **docs/03_答辩FAQ与风险预案.md** | 6 评委视角 FAQ（对抗验证）+ 10 类风险预案 + 完整性 critic（✅ 已完成，5 节，47-agent 对抗生成 + 2026-07-04 真测横幅） |
| `docs/paper_index.md` | 全部论文索引（分级/完整标题/下载状态，已核实修正） |
| `docs/核心论文精读与方案.md` | #1 拒识 / #2 ASE-PVAD / #3 KWS空间 / #4 DSENet |
| `docs/论文精读_增强与纠错路线.md` | #6 RASTAR / #7 VSAEC / #8 VPIDM |
| `docs/论文精读_US-PVAD_超短参考.md` | #3 US-PVAD + 与 #2 对比 |
| `docs/资料扩展_TS-ASR与开源资产.md` | TS-ASR 四件套 + 组合主线 + github 资产 |

## 资料结构（E:\midea_target_asr\）
- **`_txt/` 论文全文文本（pdftotext 提取，供 Agent 精读）**：19 篇核心论文（10 命名核心 + US-PVAD + SELD + 5 篇 TS-ASR：FDDT/DiCoW/SE-DiCoW/TS-RNNT/NOTSOFAR + CUSIDE-array + 智慧家庭综述）；原件 PDF 已清理，索引见 `docs/paper_index.md`
- `pdf2txt.py` — 纯 Python zlib PDF 提取（无库时备用）
- **`code/` 代码区**：⭐`submit_infer.py`（标准化推理入口，4 阶段 subprocess）/ `enroll_infer.py`（wespeaker 声纹+diar+DiCoW 转 target）/ `se_denoise.py`+`noise_classify.py`（DeepFilterNet3 条件化）/ `llm_reject.py`（Qwen2.5-3B）/ `fuse_eval.py`（多策略融合扫工作点）/ `eval_metrics.py`+`eval_full_test.py`（评测，⚠️ `eval_metrics.py` 无 CLI，复用需 `import cer()` 或用 `eval_full_test.py` 包装）/ `simulate_pipeline.py`（仿真）/ `make_readme_progress.py`（README 进度图生成）/ DiCoW-inference、TS-ASR-Whisper（开源仓库，gitignore 不入库）
- **`test_wav/` / `test_wav_clean/` 测试音频**

## 核心结论
1. **组合主线（稳健底盘，通道无关）**：`Personal VAD(产生STNO) + DiCoW/SE-DiCoW(Whisper TS-ASR转写) + Qwen-2.5-3B LLM语义拒识`。每环节开源。单通道即可跑通，是下限保障。
2. **评分→模块映射**：CER40%→DiCoW+中文家居微调+数据增强+可选RASTAR纠错；拒识40%→**声纹置信度(max_sim)锚信号 + LLM语义/PVAD辅助校验**（⚠️ decide_reject 实为 AND，LLM 只减拒不加拒，"三路融合"强项定位已证伪 GAP4，答辩勿列）；效率20%→TS-RNNT形态(Hadamard积预注册,RTF=vanilla)+Whisper量化/蒸馏/流式。
3. **差异化策略（凭什么赢）**：D1数据增强极致 / D2中文家居微调 / D3端到端联合训练（PoC 未做，未来方向）/ D4声纹锚信号拒识+LLM/PVAD辅助（⚠️ 实为 AND 非融合强项，GAP4 证伪）/ D5效率优化。心法：比赛是"完成度+适配度+效率"竞赛，把开源baseline在中文/家居/极端SNR/超短enrollment调到极致。
4. **上限候选**：X=端到端联合训练（主押，契合出题方"反cascaded"审美，对标TS-ASR-AD Interspeech2025）；Y=音频大模型一体化（⚠️通用音频LLM不区分说话人需speaker-aware encoder改造，效率风险大，作探索分支）。
5. **双预案**：单通道→组合主线直接用；多通道→前端叠加DSENet/VSAEC空间提取。无论通道数，组合主线都是下限。
6. **出题方偏好**：PVAD+短enrollment、LLM拒识、端到端反cascaded、空间/DOA多通道、扩散增强。团队：美的(Yu Gao/Wenbin Zhang)+中科大杜俊+东南大学(景康祺)+GT李锦辉。
7. ⚠️ **单/多通道是分水岭，待确认**（决定空间路线DSENet/KWS能否用）—— 当务之急。

## 开源资产
- **CAM++ 声纹**（3D-Speaker / ModelScope，192d）
- **DSENet / VSAEC**（github.com/jingkangqi，⚠️ 无预训练权重/数据/License，仅代码骨架）
- **DiCoW / SE-DiCoW**（github.com/BUTSpeechFIT/TS-ASR-Whisper，✅ HF 权重 `BUT-FIT/SE_DiCoW`，基座 Whisper-large-v3-turbo）
- **Qwen-2.5-3B**（拒识基座）
- **SpeechBrain TS-ASR recipe**（含 VAD 分支，可运行工程 baseline）

## 工具
- **pdftotext**：`E:\poppler\poppler-26.02.0\Library\bin\pdftotext.exe`（git bash 调 exe 用 `E:/` 路径，**勿用 `/e/`**；传参路径用 `E:\` 或 `E:/`）
- Read 工具读 PDF 需 poppler（pdftoppm），已装好
- **Python 一律用 uv**（`uv run`/`uv add`），禁止裸 pip install（全局规则）
- pip/uv 默认源 SSL 失败时用清华源
- **PowerShell 工具当前不可用**，用 Bash 工具执行命令

## 教训（务必遵守）
- **截图 OCR 的标题必须用搜索独立核实**——曾把 IEEE 10890695 的「DASM 异常声检测」OCR 错成「OOD ASR」
- **技术细节/架构演进必须对照原文核实**——曾把 DiCoW 误记为"decoder token+cross-attn"，核实原文后纠正：**DiCoW = FDDT+QKb（encoder 侧）；cross-attention 是 SE-DiCoW 才加的**（解决重叠歧义）。答辩讲错架构演进很危险
- IEEE 付费墙论文用校园网（广州大学权限）下载
- 子 agent 精读统一用「pdftotext 提取 → Read → 7 节客观提炼」流程
- **文档与实现要核实**：`eval_metrics.py` 实际**无 CLI**（`__main__` 仅自测），`交付/使用说明.md` 写的 `--result/--manifest/--out` 是理想用法未实现；复用评测用 `import cer()` 或 `eval_full_test.py` 包装
- **CER 均值会被幻觉扭曲，correct_rate 更诚实**：babble 重复循环幻觉使 hyp 超长、拉高 CER 均值，制造"SNR 越高 CER 反升"假象；看可用率用 correct_rate(CER<0.5 占比)，看绝对转写质量看 cer_accepted_only
- **可选依赖必须显式声明 + 缺失可见（2026-07-08 workflow④ 教训）**：cn2an/zhconv 只在本地 .venv、项目无 requirements.txt → 主办方全新环境必缺 → digit_postproc 静默 graceful 跳过 → 提交 text 数字不归一 → 官方口径 CER 0.595 实际回 0.661（数字后处理 0.033 收益全丢）。规则：任何 graceful skip 的可选依赖必须 ①声明到 `code/requirements.txt` ②缺失时 RuntimeWarning 可见（非静默 except return）③提交链路末端（to_submission）SSOT 兜底

## 下一步候选（2026-07-06：Phase 1 突破，vanilla 路线 CER 减半，保底不再是仅选项）
1. ✅ ~~真实测试集 A~~（到手，真测完成）+ ✅ ~~通道数~~（**单通道确认**，空间路线全弃）
2. ⚠️ **CER 破局战略决策**（组合主线 CER 1.4 是架构极限）：①保底交付(langfix+sim_thr=0.4 RR 99%+答辩归因，确定) ②端到端联合 X(反 cascaded，CLAUDE.md 上限候选，大工程) ③babble 专用源分离(SepFormer 提 target mel，高成本) —— **2026-07-06 Phase 1 已开拓第④路线：vanilla Whisper + target timeline extraction（zero-training，CER 1.25→0.664 减半），无需大工程即可斩获大部分 CER 收益**
3. 🔧 **保底交付**（进行中）：langfix 保留 + sim_thr=0.4 调 RR 99% + 效率 RTF 0.2 + 答辩重点(babble 归因/单通道确认/工程优化 Gap3·繁简·langfix/诚实组合主线极限/Phase 1 vanilla 突破反 cascaded)
4. ⚡ **P2 Phase 1 落地任务**（Phase 1 后续，按优先级）：① **vanilla 集成 submit_infer**（最高优，`--asr-backend {dicow,vanilla}` 切换，把 0.664 变提交数字）② 声纹强化（CAM++ per-speaker / US-PVAD 改善 target timeline 切割，尤其低 sim 桶）③ 数字 initial_prompt 锦上添花（家居指令数字/温度场景）④ sim_thr 待主办方评测口径定（CER 均值→0.4 / correct→0.2 / pos 不许拒→0）
5. ⚡ **L20 耗时验证**：推理脚本显存自适应(L20 48GB 大 batch) + 租 AutoDL L40 验证端到端(官方 L20 评效率，本机 4060，memory `l20-eval-hardware`)
6. 📄 **答辩演练**：等 `03_答辩FAQ与风险预案.md` 就绪后做(README 进度图作开场)
