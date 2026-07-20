# 答辩 FAQ 与风险预案（XH-202615）

> 本文档由 **6 评委视角出题 → 47-agent 逐题对抗验证（skeptic 攻击）→ 10 类风险预案 → 完整性 critic** 生成。
> **它不是自夸式 FAQ，而是"被攻击过、暴露过真实弱点"的诚实答辩准备。** 凡是被对抗验证击穿的硬伤，本文档都标注 ⚠️ 并给出诚实应对——答辩时绝不回避。
> 创建：2026-06-27。配套：架构见 `00`、技术细节见 `01`、上限候选见 `02`。
>
> ⭐⭐⭐ **2026-07-19 最新横幅（稳定性/鲁棒性测试闭环 — 可复现性硬证据 + batch 口径验证 + 模型短板诚实归因）**：
> 1. **可复现性量化达标（R1=0）**：A 同种子 10 遍 + 变种子 5 个(42/100/200/314/555)×2 全量 1364 条，**transcript 零波动** —— greedy argmax 下系统完全确定。没设 `use_deterministic_algorithms` 也无所谓（GPU 残余非确定 + 种子变化都不改 argmax）。**回应评委「怎么保证可复现」的硬证据**。详见下文 ☆ 节 + `docs/稳定性测试报告_2026-07-19.md`。
> 2. **batch 口径验证（R2 纯仅 2 条）**：主办方默认 batch=1 / RTF 按 batch=1。我们历史数字都 batch=16 跑——验证 batch=1 vs 16 差异 74 条(5.43%)但 **72 条含 R3/R4 叠加**（本来就波动的边界音频），**纯 batch 差异仅 2 条** → 开发口径数字基本可外推提交 batch=1。submit 已锁 batch=1。
> 3. **诚实归因模型短板（R3 57%）**：输入微扰（gauss 加性噪声 54% 主因破坏 mel 谐波）过半翻车 —— 模型泛化短板，**归档**（A 集外训练才能修，A 集不能训练）。短指令尤甚。**答辩不掩饰，给根因 + 未来方向**。
> 4. **hold-out 纪律**：测试只用工程修复 + 诊断，不碰训练（A 集测试集泄漏红线）。
>
> ⭐⭐ **2026-07-11 最新横幅（Qwen3-ASR 突破全收尾 + 双 SOTA 横评 + 声纹强化证伪；覆盖下文 07-08/07-04 横幅的 CER/归因数字）**：
> 1. **Qwen3-ASR 替换 vanilla = 首个真实 CER 突破（候选2 全量证实）**：全量 1350 条官方口径 transcribe CER **0.3436** vs vanilla 0.5954（Δ-0.25，CER 40% 腿诊断口径 +10.07 分）。机制：复用 enroll_infer diar+wespeaker 切 target timeline，Qwen3-ASR-1.7B drop-in 转写（ExtremeNoise 4× 鲁棒迁移），zero-training，Apache2.0。已集成 `submit_infer --asr-backend qwen`，run-twice text 一致 100% delta=0（可复现性闭合）。⚠️ **双口径防穿帮**：提交用含拒 overall thr0.27=**0.5934**（CER 腿真实 **+4.29**），transcribe 0.3436(+10.07) 是诊断上限——**答辩/提交一律报含拒 0.5934/+4.29，勿用 transcribe 0.3436 虚高**（评委按含拒口径复算会穿帮 10 分）。归一零效应（0 阿拉伯数字 0 繁体，raw==归一==0.3436）。详见 RESULTS T28 + `recompute_qwen_official.py`。
> 2. **死区 sim<0.2 听音坐实 H1 真实突破（修正"物理地板"归因）**：用户听音 cmd_2091(sim0.092 "儿童要少吃什么")/cmd_2137(sim**0.004** "打开睡眠模式") 确认音频清晰、qwen 听对（非 LM 幻觉）。死区是混合桶：B 类(声纹失败但音频可辨, qwen 突破 H1) + A 类(真摧毁+LM 幻觉 H2 少数, 如 cmd_2808"风速五十"→"邮政银行被打出个一比五"编造)。**sim 是 wespeaker 声纹代理 ≠ 音频质量**（cmd_2137 sim0.004 音频清晰坐实）。→ **spk-oracle-poc"物理地板"修正为 vanilla 转写器 OOD 伪地板**（oracle 0.607 全程 vanilla 评估，换 qwen 压到 0.459）。**07-08 横幅#3 / 红线下"死区物理极限不可破"均以此修正为准**。详见 RESULTS T28 A2 段 + `analyze_dead_zone_qwen.py`。
> 3. **声纹强化 CAM++ 真 POC 证伪关闭（原理性，七连受挫）**：A2 听音后疑声纹强化重开，跑 CAM++ 真 POC（`exp_spk_campp_deadzone.py`，396 死区条）：B 类(qwen 转对) CAM++ sim 0.373 vs A 类(翻车) 0.374，**B-A margin=-0.000**。原理性结论：声纹 emb 编码"who"（说话人身份）不编码"audio clarity"（音频可辨性），B/A 区分在 mel/声学层、声纹层看不到 → **任何声纹器（CAM++/US-PVAD/wespeaker）都救不了 B 类**（救 B 连 A 类 qwen 幻觉灾难一起误放）。声纹强化方向关闭。副产品洞察：支持"转写器置信度拒识 > 声纹 sim 拒识"（BLEMU probe / FA 置信度方向）。
> 4. **FireRedASR-AED-L 横评 = 中文原生 ASR 双 SOTA 坐实**：firered transcribe **0.3501** ≈ qwen 0.3436（Δ+0.0065 噪声带不可分，72% 逐条持平），都远超 vanilla 0.595；RTF firered **0.24** vs qwen 0.289（4060，firered 快 17%）。B1 预判"无 babble 训练 45% no-go"**未发生**（WenetSpeech-meeting 训练对真实 babble 适应好）。选型：qwen 主线 + firered drop-in 备选（`firered_asr_backend.py` 镜像 qwen）。**答辩弹药：「中文原生 ASR 双 SOTA 横评选型，CER 都 ~0.35 vs vanilla 0.595」**。详见 RESULTS T29。
> 5. **答辩核心叙事刷新（最重要）**：① CER 从"vanilla 0.595 / 物理地板 / 架构极限 1.0 / pos CER 放弃"→"**Qwen3-ASR 真实突破：含拒提交 0.5934（CER 腿 +4.29）/ transcribe 0.3436（诊断 +10.07）**"——**07-04 横幅"CER ~1.0 架构极限 / 靠拒识+效率拿分 CER 放弃"已彻底推翻**，CER 腿不再放弃。② 死区从"物理地板不可破"→"vanilla OOD 伪地板，qwen 凭 ExtremeNoise 突破 H1"。③ 新增双 SOTA 选型严谨性 + 声纹强化七连受挫诚实归因弹药。⚠️ 答辩 CER 数字一律用**含拒 thr0.27=0.5934**。效率腿 20 分待 L20 RTF（qwen 0.289 / firered 0.24 @4060，L20 待测）。
>
> ⭐ **2026-07-08 最新横幅（决赛交付刷新，4 件新事；以下为准）**：
> 1. **主办方 CER 口径脚本到手并坐实 + caliber-A 彻底解除**：`normalize_text`(NFKC + lower + strip + 去所有 Unicode P* 标点和空白) + `CERMetric`(累计池 total_err/total_char，editdistance 库)，**不繁简归一、不数字归一**。`code/eval_metrics.py` 已照抄实现，4-agent 对抗验证逐行等价、12 边界 Δ=0。重算全量 1362 条（提交归一后）：vanilla 转写 overall **0.595**（原项目口径逐条均 0.664；correct 48.8%）/ dicow **1.189** / 英文幻觉 vanilla 0.6% vs dicow 18.7% / thr0.27 含拒累计池 **0.703** / H3 dicow sim[0.2,0.3)=**1.609** 反作用更稳。主办方问题二/三确认：**pos 隐含允许被拒**（拒=CER1.0 无额外惩罚）+ 排名公式 **`TotalScore = w1*(1−CER) + w2*RR`**（线性、无惩罚项、per-sample 不封顶）+ **RTF 按 batch=1 测** → caliber-A 彻底解除，**thr=0.27 定稿**（w1=w2 时 T27 目标函数与公布公式等价）。同时修了提交归一漏洞（cn2an/zhconv 原未声明依赖→主办方环境必缺→digit_postproc 静默失效→CER 0.595 实际回 0.661；已建 `code/requirements.txt` + RuntimeWarning + to_submission SSOT）。详见 memory `official-scoring-spec` + `recompute_official_cer.json`。
> 2. **content_gate（转写内容有效性二次拒识）集成 — hold-out 多 seed 证泛化**：对 sim≥thr 的 accept 再判转写是否有效家居指令（`text_utils.is_valid_command`），非指令（新闻/英文/乱码）则加拒。A 集分 train/val hold-out（回应"A 集是开发集"过拟合担忧），**10 个 seed 划分 10/10 全正**，val ΔTS min=+0.0019（+0.24 分）max=+0.0357 mean=**+0.0199（+2.5 分）**，bootstrap CI p5>0 稳赚，L 不敏感（18-30 全正）。集成进 `submit_infer.decide_reject` 独立加拒通道，**默认关（`BAODI_GATE=1` 开）**——保守不拿收益换不反噬 B 集（B 集干扰分布未知）。Pareto 改进：提 RR 不损 pos/效率。详见 memory `content-gate-decision`。
>    ⚠️ **2026-07-18 反转（覆盖上方"默认关"）**：qwen 后端 joint 验证净正 → **默认开（`BAODI_GATE=1`）**。`verify_content_gate_joint.py` 离线实测（qwen thr0.27 noSE，w1=w2=0.4）：pos CER 0.5934→0.6171（腿 −0.95，误拒 35 条 pos 多为 CER≥1 反赚）/ neg RR 0.9051→0.9494（腿 +1.77，gate 拒 21 条漏拒：信访/租赁物业/卖家协商等非家居）/ **JOINT +0.826**。对 w1/w2 鲁棒（净正只需 w2/w1>0.53，官方 0.4/0.4 远满足）。原"qwen 后端 gate 恶化 Δ+0.024"是 pos-only 评估漏 neg 侧，已纠正。`run_baodi.sh` `BAODI_GATE` 默认 1（commit `226e239`）。
> 3. **P1 oracle POC 证伪声纹强化 → 死区是 babble 摧毁 mel 的物理极限（诚实归因硬证据）**：oracle 实验（`code/exp_spk_oracle.py`，60 条死区抽样）四组证据全指向 GO=否：① argmax 选对率 66.7%（多数选对 target）② oracle_sim≥0.2 占 0%（正确 target 声纹也全不可识别）③ miss 20/20 声纹反向指错 ④ **单 speaker 控制组 n=18 CER 0.436**（target 唯一零歧义仍转不出 = 纯音频摧毁）。oracle_CER 0.607（作弊完美选 target 仍>0.5 不及格）。**不投 CAM++/帧选择/US-PVAD**（避免五连受挫）。答辩弹药升级：死区=物理极限非工程缺陷，契合反 cascaded 审美。详见 memory `spk-oracle-poc`。
> 4. **MiMo-V2.5-ASR 作诊断标尺（非后端，合规）补 E 类创新性**：用小米开源 audio-LLM ASR（MiMo CER 0.417 vs vanilla 0.661，纯文字句 0.428 vs 0.637 验证非口径红利）当**尺子**而非后端（云端不能进提交三红线：L20 本地 RTF/测试集上传/可复现性），同 target 切片输入控制变量定位瓶颈分布：**切片死区 30%（sim<0.2，连 SOTA MiMo 都翻车）/ 转写器为主 50%（sim[0.2,0.4)）/ 接近解决 20%（sim≥0.4）**。据此集中火力攻切片层而非盲目堆转写器。答辩弹药："用 SOTA 作诊断标尺定位瓶颈分布"= 学术诚实 + 合规 + 主动。详见 memory `mimo-asr-backend-potential`。
>
> ⚠️ **2026-07-04 真测更新（本文档创建于仿真期，多处已过时；答辩前必读本横幅 + 重读下文，标注「已过时」处按本横幅为准）**：
> - **已有全量真测基线**（pos 1364 / neg 474，详见 `RESULTS.md` T23 / memory `baodi-config-no-llm`）。第〇节总原则2「一行实测都没有」、红线6「零实测」**已过时**。
> - **保底改为关LLM**（⚠️ 3-agent 对抗审查修正：**trade-off 非全面优于** —— 关LLM 赢 neg RR **98.5%**>96.2% + RTF **0.24**<1.01 4×；**开LLM 赢 pos 救回**：28 条 LLM 救回的 pos 里 **26 条 CER=0.000 完美**（"关闭客厅空调"等 max_sim 低至 0.022 被 sim_only 误杀，LLM 语义救回），原"pos 持平"错；选关LLM = 为效率20%+RR40% 牺牲 pos 救回，pos 反正架构极限放弃）。**「三路融合拒识」被证伪**（llm_or_sim 是 AND，LLM 只减拒不增拒）——**答辩别列为强项**。
> - **SE-DiCoW 已证伪弃用**（架构不兼容：`mt_num_speakers=2` 多-speaker + self-enrollment 范式，与 `enroll_infer` 单-target 范式根本不同；短音频 OOD）—— 红线4/5、A4、B4 把 SE-DiCoW 当「主押/在用」的内容**已失效**，答辩**不得**再说在用 SE-DiCoW（诚实说法：尝试过、范式不兼容、放弃）。
> - **单通道已确认**（datasetA 100% 单通道 16k）—— 风险1「通道数未定（最关键未知项）」、第四节「多通道分支零覆盖」**已过时**；空间路线（DSENet/VSAEC/DOA/KWS）**全弃**。
> - **pos CER 真测值已出 = ~1.0（架构极限）**（babble 89% 主导，cer_accepted 0.94，sim_thr 全档扫描无 thr 能救）—— C1「目标 CER 需 W6 定」改为「真测 pos CER 1.0 是 cascaded 在极重 babble 下的架构极限」。**CER 均值是幻觉陷阱**（thr 升=误拒换幻觉超长，correct_rate 才诚实：31%@thr0.2 → 14%@thr0.4 真退化）。⚠️ **2026-07-06 Phase 1 已推翻此结论**：根因不是 cascaded 架构极限，而是 **DiCoW 的 FDDT/STNO 条件化反作用**（sim 0.2–0.4 桶 CER 1.5–1.6、英文幻觉 18.8%）；改用 vanilla Whisper + target extraction，CER 减半到 **0.664**（详见 A6 / 风险 11）。下文所有"CER ~1.0 架构极限 / 无 thr 能救"均以此修正为准。
> - **效率真测**：关LLM RTF **0.24**（4060，sim_only），L20 待测 —— 红线3「未实测」部分过时。
> - **第五节开场定位**仍为设计稿口吻，答辩前应改为：真测基线 + 关LLM + **诚实 CER 1.0 架构极限** + 拒识 98.5% / 效率 0.24 拿分 + 端到端联合 X 是未来方向。
> - **答辩核心叙事（真测后）**：babble 归因清晰（T22 H3 确证：vanilla Whisper 在 babble 正常，英文幻觉 100% 是 DiCoW FDDT/STNO 条件化病害）+ 单通道确认 + 工程优化（Gap3 批量化 / 繁简归一 zhconv / langfix）+ **诚实组合主线 CER 1.0 是架构极限**（靠拒识 40% + 效率 20% 拿分，CER 40% 放弃）+ 端到端联合 X 是冲 CER 的未来方向。
> - **待确认（最高优先，thr 决策前提）**：向主办方问评测口径 —— CER 均值 vs correct_rate？pos 被拒算多少？pos 是否允许拒？pos/neg 能否不同 thr？

---

## 〇、答辩总原则（先读，最重要）

1. **诚实 > 吹嘘**。被问到短板（级联次优、效率未实测、未见 0.2s 验证）时，**正面承认 + 给工程理由 + 给 deadline**，绝不包装。评委最懂工程，虚假低风险承诺一戳即穿。
2. **绝不引用未实测数据**。当前整个 pipeline 是设计稿，**一行实测都没有**。答辩中不得出现"实测 RTF=X""mask 准确率 88%"等编造数字——一律说"这是 W6 阶段（第 6 周）交付的实测项，届时给数"。
3. **架构矛盾正面承认**。级联 vs 端到端、DiCoW 抛弃声纹 vs PVAD 用 CAM++，这些"自相矛盾"评委一定会问——承认 + 解释工程分层，比强行辩护安全。
4. **技术细节禁记错**。Whisper-large-v3-turbo ≈ **809M**（不是 1.5B）；ASE-PVAD 论文只测 **0.5s/1s/1.5s**（无 0.2s 数据点）；SpeechBrain **没有** Personal VAD recipe。

---

## ☆、稳定性 / 可复现性 FAQ（2026-07-19 测试闭环 · 优势弹药）

> 本节是**优势弹药**（非红线硬伤）。26 遍全量实跑量化证明可复现性 + batch 口径一致 + 模型短板诚实归因。评委问可复现性 / batch / 鲁棒性 / 过拟合时用。详见 `docs/稳定性测试报告_2026-07-19.md`。

**Q1：你们管线用 fp16 + 多模型（diar/wespeaker/qwen），怎么保证可复现？不同跑次结果一样吗？**
A：实测量化达标。跑了 **A 同种子 10 遍 + 变种子 5 个 × 2 = 20 遍**全量 1364 条，**transcript 零波动（R1=0）**——greedy argmax 下系统完全确定。没设 `use_deterministic_algorithms` 也无所谓：GPU 矩阵乘/attention 的残余非确定（cuBLAS 原子加）+ 种子变化（42/100/200/314/555），都不足以改变 greedy argmax 结果。种子固定（random/numpy/torch/cuda/cudnn）+ greedy（`do_sample=False`）双保险。

**Q2：主办方默认 batch=1，你们验证用 batch=16，数字可信吗？**
A：已验证可外推。batch=1 vs batch=16 全量对比，差异 74 条(5.43%)，但其中 **72 条同时是 R3/R4 波动音频**（本来就翻车的边界音频，batch 变化叠加），**纯 batch 差异仅 2 条**。即开发口径 batch=16 的数字（主线 CER 0.5934 / thr 0.27 / 各 POC）基本可安全外推提交口径 batch=1。提交脚本已锁 batch=1（`submit_infer --asr-batch-size 1`，commit 3398c0d）。

**Q3：模型对输入扰动鲁棒吗？（加噪/音量/时间偏移）**
A：诚实说，这是**模型真实短板**，不掩饰。输入微扰（gauss 加性噪声 / ±1dB / ±20ms）**57% 音频转写变化**，主因 gauss 加性噪声 54%（随机噪声破坏 mel 谐波结构；整体增益 vol 仅 9% 因保结构，时间偏移 26% 影响 diar 切 target）。机制清晰：模型对加性噪声泛化有限，短指令尤甚（字符少一字错 CER 跳大）。**修复方向是 A 集外加噪训练**，但 A 集是测试集不能训练（泄漏），故本次只诊断归档，列为未来工作。

**Q4：你们怎么避免在测试集上过拟合？**
A：hold-out 硬纪律。A 集是开发/测试集：① 不用 A 集训练/数据增强（当前 zero-training 路线，直接用预训练权重 Qwen3-ASR/Whisper）；② 任何基于 A 集调的规则（thr / content_gate）必 hold-out 分 train/val 验证泛化（如 content_gate 10 seed 划分 10/10 全正，bootstrap CI p5>0）；③ 稳定性测试只用工程修复（可复现性/batch 锁定，不涉 A 集内容）+ 诊断归档，**绝不改基于 A 集内容的提交规则**。

---

## 一、⚠️ 答辩红线：9 个必须诚实应对的硬伤

> 这 9 点是 47-agent 对抗验证集中击穿的，**答辩最高危**。每条给"会被怎么问 + 诚实应对"。

### 红线 1：级联次优 vs 提交级联（循环论证）
- **会被问**："你们候选 X 承认级联有误差累积、押注端到端——既然承认级联次优，为什么还把它当主线提交？岂不是提交自己证伪的方案？"
- **诚实应对**：① 比赛看交付，不看架构信仰。真实数据下发前、X1 联合 PoC 出结果前，级联是唯一能跑通、能测三个指标的方案；跑通的次优 > 没跑通的最优。② X1 不是画饼，是排进关键路径的硬交付（第 8–10 周 PoC），是否取代级联**取决于 PoC 收益数据**，不取决于现在怎么吹。③ **不得**用"TS-ASR-AD 转写弱所以不做端到端"辩护——那是循环论证，等于否定自己"端到端是上限"。正确说法：TS-ASR-AD 证明方向可行，差异（CTC decoder vs Whisper 底座）用 X1 消融回答，不拍脑袋。

### 红线 2：FDDT 可微夸大
- **会被问**："你说 X1'FDDT 天然可微、无需改造、3–4 周平滑过渡'。那 PVAD 的 pairwise loss 和 Whisper 的 CE loss 梯度尺度差几个数量级？α/β/γ 怎么设？三分类→四分类的转换头可微吗？"
- **诚实应对**：**"天然可微"是不严谨的**。准确说法：FDDT 凸组合结构**降低了连接难度**，但梯度协调是主要工作量——需把 PVAD+转换头+Whisper 拼成同一计算图，处理 ① loss 加权 α/β/γ 调参 ② 梯度裁剪 ③ 大小模型学习率分组 ④ Whisper 用 LoRA 而非全参。3–4 周 PoC 是乐观估计。绝不包装成"零改造"。

### 红线 3：效率偷换概念
- **会被问**："你用'音频大模型慢'暗示级联不慢。但你主线是 Whisper-turbo + Qwen-3B 串行，RTF 到底多少？测过吗？"
- **诚实应对**：**不得用"对手慢"辩护"我不慢"**。诚实事实：级联主线本身有 RTF 风险，L20 实测是 W6 交付项。效率分用 turbo(蒸馏版，比 large-v3 快约 8 倍) + INT8/4 量化 + 流式 chunking 兜底；TS-RNNT 形态作上限冲刺但不 all-in。联合端到端是否比级联快是开放问题，不断言。

### 红线 4：M3 接口胶水（O 类判别 + self-enrollment 替换）
- **会被问**："PVAD 三分类怎么出 STNO 的 O 类？100% 重叠时两路概率都高，O 退化成阈值博弈；而且你们把 SE-DiCoW 的 self-enrollment 换成短唤醒音频，在最关键的重叠区同时削弱了两个支柱。"
- **诚实应对**：① O 类用**双路 PVAD**（目标路 + 非目标路，非目标声纹从识别音频聚类得到）联合推出，比单 PVAD+VAD 可靠；FDDT 凸组合是软过渡容错强于硬掩蔽。② self-enrollment **不是简单替换**——⚠️ SE-DiCoW 的 self-enrollment 已验证范式不兼容（`uses_enrollments` 不接受外部 enrollments kwarg、架构不兼容，见 B4），改靠 **sim_thr 拒识 + 双路 PVAD 兜底**（从识别音频聚类得到非目标声纹，与题目给定唤醒音频并用）。③ **诚实承认**：完全重叠区仍是最薄弱处，靠 ASE-PVAD 反哺声纹 + 拒识冗余兜底（声纹 max_sim 作锚信号，LLM 语义/PVAD 辅助校验；⚠️ 真测后 LLM 已关：`decide_reject` 是 AND 逻辑，LLM 只能减拒不加拒，不作"三路融合"主力），不靠单信号。

### 红线 5：声纹依赖矛盾（DiCoW 抛弃 emb vs PVAD 用 CAM++）
- **会被问**："DiCoW 卖点是抛弃 speaker embedding 避免跨空间映射，你们却在 PVAD 侧重新引入 CAM++ 声纹做 Cross-Attention+FiLM——一边借 DiCoW 论点一边复现它反对的做法，如何自洽？"
- **诚实应对**：DiCoW 批评的跨空间映射特指"声纹→ASR 序列生成空间"（高维、需海量多人数据泛化）；PVAD 的 Cross-Attention+FiLM 做的是**帧级三分类判别**（低维、只需判"这帧是不是这人"），映射难度和泛化要求天差地别。DiCoW 的批评**不直接适用于分类任务**。但追问触及真实优化点：超短 enrollment 走 US-PVAD 内生声纹路线（省 15.5M、0.2s 反超 2s baseline），逐步降低 CAM++ 依赖。

### 红线 6：数据捏造红线（最致命的诚信问题）
- **会被问**："你说 mask 在 −5dB 准确率 88–92%、贡献 60% CER——这些数据哪来的？文档里有吗？"
- **诚实应对**：**已解除**——2026-07-08 主办方 CER 口径脚本到手坐实后，全量真测 1362 条（提交归一后，累计池）：vanilla 转写 overall **0.595**（correct 48.8%）/ thr0.27 含拒 **0.703**（B 集统一 thr）/ dicow **1.189**（已证伪 fallback）/ neg RR **90.5%**（thr0.27 统一）/ RTF **0.2543**（4060 pos 全量 3498s 音频；**L20 待测**）。详见 `recompute_official_cer.json` + `RESULTS.md` T27。答辩**只报真测数字**。铁律不变：**未实测的指标（如 L20 RTF、各模块 ablation、mask 准确率）一律说"实测中/待交付"，绝不报虚数**——评委一句"打开评测脚本"就能让虚数穿帮。早期文档里的"mask 88-92%"等是设计期估算，**不作数**。

### 红线 7：参数量 / 资产事实错误
- **会被问**："Whisper-turbo 多大？你说 1.5B？" / "SpeechBrain 的 Personal VAD recipe 在哪？"
- **诚实应对**：记住正确数字——Whisper-large-v3-turbo ≈ **809M**（large-v3 1.55B 的 decoder 剪枝版，24→4 层）；**SpeechBrain 没有 Personal VAD recipe**（只有 ECAPA-TDNN/SepFormer 组件 + TS-RNNT 衍生 recipe），US-PVAD/ASE-PVAD 需自实现。基座参数和资产位置是基本功，记错等于"对方案不熟"。

### 红线 8：ASE-PVAD 兜底夸大
- **会被问**："你说 ASE-PVAD 兜底也能处理 0.2s，可 ASE-PVAD 论文只测了 0.5s/1s/1.5s，哪来的 0.2s？"
- **诚实应对**：**ASE-PVAD 全篇无 0.2s 数据点**（最强是 0.5s 经增强 REC 68.12%）。0.2s 场景**只能靠 US-PVAD 内生声纹路线**（论文实测 0.2s REC 62.74%）。不得说 ASE-PVAD 也能扛 0.2s。

### 红线 9：候选 X 的"新颖性"伪创新
- **会被问**："FDDT 可微是原作自带属性（凸组合带概率权重），你只是没破坏它，凭什么说'联合设计原创'？候选 X 真的比 TS-ASR-AD 新吗？"
- **诚实应对**：**FDDT 可微是 BUT-FIT 原作机制，不是本队发现**。正确表述：我们做的是"在 PVAD↔FDDT↔Whisper 这条具体链路上做联合微调 + 中文家居适配"，**系统级工程整合**而非单点理论原创。候选 X 的定位是"把 Interspeech 2025 的 TS-ASR-AD 范式落地到中文家居 TS-ASR+拒识场景"，是**工程方案**不是理论首创——评委问"研究还是工程"，诚实答工程。

---

## 二、六大类答辩 FAQ（综合原答 + 对抗改进版）

> 每题：**问题 → 诚实可用答案**（已剔除被验证击穿的夸大）。

### A. 架构选型与自洽性

**A1. 为什么是 PVAD+DiCoW+LLM 这个组合，不用纯 TSE 前端 / 纯端到端 / 纯音频大模型？**
答：组合主线每环节都有成熟开源资产，可独立验证、可降级、真实数据下发前可跑通——保下限。纯 TSE 前端（SpEx+）是备选；纯端到端（候选 X）是上限赌注但依赖真实数据；纯音频大模型（候选 Y）效率风险大作探索分支。**这是工程分层的稳健选择，不是技术信仰。**

**A2. 级联 vs 端到端的矛盾怎么解释？** → 见红线 1。

**A3. DiCoW 抛弃声纹 vs PVAD 用 CAM++ 的矛盾？** → 见红线 5。

**A4.（⚠️ 2026-07-04 已修正：SE-DiCoW 证伪弃用）为什么不押 DSENet+空间 TSE？**
答：**SE-DiCoW 已尝试并放弃**——`mt_num_speakers=2` 多-speaker + self-enrollment 范式与 `enroll_infer` 单-target 范式根本不兼容，短音频 OOD（详见 `RESULTS.md` T23 / `AGENT_HANDOFF.md` 第3节）。DSENet 无预训练权重、未测 −5dB、仅代码骨架，且**单通道已确认**（datasetA 100% 单通道）→ 空间路线全弃。实际主线用 **DiCoW_v3_2**（wespeaker 声纹 + DiariZen diar 锁 target → STNO → DiCoW 转），有权重、单通道可跑通、全开源。

**A6.（⭐ 2026-07-06 Phase 1 真测突破）为什么主线从 DiCoW 条件化改成 vanilla Whisper + target extraction？**
答：**2026-07-06 全量真测 1362 条 pos（zero-training）发现 DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】**，改用 vanilla Whisper-large-v3-turbo + 声纹切 target timeline 路线，**CER 几乎减半**：
- **转写 CER**：vanilla **0.664** vs DiCoW 1.248（Δ −0.58，几乎好一倍）
- **correct_rate**：vanilla **45.6%** vs DiCoW 31.4%；**near_perfect**：vanilla 20.8% vs DiCoW 14.8%
- **英文幻觉率**：vanilla **0.59%** vs DiCoW **18.80%** ← DiCoW 条件化主动造孽
- **sim 分桶（条件化最毒的证据）**：sim[0.2,0.3) vanilla 0.746 vs DiCoW **1.606**（Δ −0.86）；sim[0.3,0.4) vanilla 0.623 vs DiCoW **1.523**（Δ −0.90）；sim≥0.4 轻 babble vanilla 0.364 vs DiCoW 0.830（Δ −0.47，仍优）
- **overall CER（含拒=1.0）**：thr=0.20 vanilla **0.711** vs DiCoW 1.241（vanilla 终于把 overall 拉到 <1）→ **CER 40% 腿从 ~0 分变 ~11 分**（线性 (1-0.711)×40，待主办方 CER 口径确认）

**机制**：diar+wespeaker 选 target（复用 `enroll_infer` 逻辑）→ 切 target timeline 段（含重叠区）拼接 → vanilla Whisper 转写（去掉 stno_mask/FDDT 条件化）。**英文幻觉根因坐实**：DiCoW 条件化造 18.8% 英文幻觉，vanilla 仅 0.59%——之前 langfix 是在打 DiCoW 自己造的孽（治标），vanilla 路线从根消灭（治本）。

**答辩弹药**：「cascaded 条件化机制在极重 babble 下反作用（sim 0.2–0.4 桶 CER 1.5–1.6、英文幻觉 18.8%），改用 target extraction + vanilla Whisper，CER 几乎减半」——契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。**这推翻了"pos CER ~1.0 是架构极限"的旧结论**：极限是 DiCoW 条件化造的，不是 cascaded 本身的；vanilla 路线 zero-training 就把 CER 拉到 0.664，后续声纹强化（CAM++/US-PVAD）+ 数字 initial_prompt 还能再压。详见 memory `h3-dicow-conditioning-backfire-vanilla` + `code/exp_vanilla_full.json`。

### B. 技术深度

**B1. FDDT 凸组合为什么优于硬掩蔽？**
答：硬 input masking 留硬边界 artifact 伤 Whisper；FDDT 按 STNO 概率的凸组合是软过渡，+ 抑制式初始化（W_S,W_N≈0.1 对角压非目标，W_T,W_O=I）把"拒识非目标"先验焊进初始化，不破坏预训练。已核实 FDDT 原文 2409.09543。

**B2. DiCoW 为何抛弃 speaker embedding？**
答：声纹→ASR 表征的跨空间映射要泛化到未见说话人需海量多人数据，真实多人对话数据稀缺；DiCoW 改用 diarization 输出（who-spoke-when，时间轴共享）条件化，不需学跨空间映射，少数据微调、泛化好。

**B3. PVAD 三分类怎么出 STNO 四分类的 O 类？** → 见红线 4。

**B4.（⚠️ 2026-07-04 已修正：SE-DiCoW 证伪弃用）原计划用 SE-DiCoW cross-attention 解重叠，实际怎样？**
答：**SE-DiCoW 架构不兼容，已放弃**（见 A4）。原设想"题目给定唤醒音频替代 self-enrollment"**实测失败**——SE-DiCoW 的 `uses_enrollments` 是内部 self-enrollment（从 stno_mask target 行自动提），**不接受外部 enrollments kwarg**（generate 报 `model_kwargs not used`），且 `SCBs.py` 硬要求 batch 是 `mt_num_speakers(=2)` 倍数。完全重叠区（100%）仍是单通道死区，靠 sim_thr 拒识兜底（诚实承认是最薄弱处，靠拒识冗余而非单信号解决）。

**B5. TS-RNNT 为何 RTF=vanilla？**
答：声纹 h_target 预计算（预注册），只在 RNN-T encoder 第 n 层做一次 Hadamard 积（逐元素乘，近零开销），不进逐帧递归路径 → 复杂度与 vanilla RNNT 相同。

### C. 评分策略

**C1. 40% CER 怎么保证？目标多少？**
答：⚠️ **2026-07-04 真测已出**：pos CER ~1.0 是 cascaded 在极重 babble（89% 主导）下的**架构极限**，sim_thr 全档扫描无 thr 能救（cer_accepted 0.94）。原"SE-DiCoW 冲精度"路径**已失效**（SE-DiCoW 范式不兼容、已弃，见 A4）；实际主线 **DiCoW_v3_2**（wespeaker+DiariZen→STNO→DiCoW）。CER 40% 已诚实放弃，靠拒识 40% + 效率 20% 拿分；冲 CER 的未来方向是端到端联合 X。**CER 均值是幻觉陷阱**，看 correct_rate（31%@thr0.2 → 14%@thr0.4 真退化）。
⚠️ **2026-07-06 Phase 1 已推翻加粗"架构极限"结论**：根因是 **DiCoW FDDT/STNO 条件化反作用**（sim 0.2–0.4 桶 CER 1.5–1.6、英文幻觉 18.8%），不是 cascaded 本身。改用 **vanilla Whisper + target extraction（diar+声纹切 target timeline）**，全量 1362 条 zero-training：**转写 CER 0.664（vs DiCoW 1.248 减半）/ correct_rate 45.6%（vs 31.4%）/ 英文幻觉 0.59%（vs 18.80%）**，overall CER（含拒=1.0）**thr=0.20 → 0.711**（<1），**CER 40% 腿从 ~0 分变 ~11 分**（线性 (1-0.711)×40，待主办方口径确认）。后续可叠加：① 声纹强化 CAM++/US-PVAD ② 数字 initial_prompt。详见 A6 / 风险 11 / memory `h3-dicow-conditioning-backfire-vanilla`。
>    ⚠️ **2026-07-11 Qwen3-ASR 突破（覆盖上方 vanilla 0.664）**：换 Qwen3-ASR-1.7B（ExtremeNoise 4× 鲁棒迁移）drop-in 转写，全量 1350 条 **transcribe CER 0.3436**（诊断口径，CER 腿 (1−0.3436)×40=26.26）/ **含拒 thr0.27 提交 overall 0.5934**（CER 腿 **16.26**）。⚠️ 双口径：答辩/提交一律报含拒 0.5934（transcribe 0.3436 是诊断上限勿虚报，评委按含拒复算会穿帮 10 分）。详见 07-11 横幅。
>    ⚠️ **2026-07-18 content_gate 开（叠加）+ 当前算分**：gate 默认开后 pos CER 0.5934→0.6171（CER 腿 16.26→**15.32**）/ neg RR 0.9051→0.9494（RR 腿 36.20→**37.98**）。**当前算分（qwen+gate，w1=w2=0.4 假设，待主办方确认权重）：CER 腿 15.32 + RR 腿 37.98 = 53.3/80**（gate off 52.46/80 对比，gate 净 +0.84），效率腿 20 分待 L20 batch=1 RTF 实测（4060 关 SE overall_rtf 0.142，L20 待测）。

**C2. 40% 拒识率怎么定义？怎么测？** ⚠️ 这是当前定义最模糊处
答：**拒识率的精确定义（精确率/召回率/F1/TPR-FPR 权衡）需向主办方确认**——这是待确认项（00 文档第九节）。⚠️ 真测后拒识主力是**声纹 max_sim 阈值**（`decide_reject` = AND：`llm!=accept AND max_sim<thr`），LLM 语义/PVAD 仅辅助校验、且**只能减拒不加拒**——"三路融合"作主力强项已被 3-agent 对抗审查证伪（GAP4），答辩**不得**列为强项。thr 取值取决于评测口径（CER 均值→0.4/0.45 / correct→0.2 / pos 不许拒→0）。

**C3. 20% 效率怎么保证？** → 见红线 3。诚实：turbo + 量化 + 流式 + TS-RNNT 形态，W6 实测。

**C4. 三维度冲突（CER 好但效率差）怎么权衡？**
答：按 40/40/20 加权选总分最优单方案，依据实测而非拍板。CER/拒识各 40% 权重最高，**不可为效率牺牲 CER/拒识的核心**。

### D. 鲁棒性

**D1. SNR −5dB（几乎听不清）怎么办？**
答：数据增强覆盖（D1）+ Whisper 鲁棒性 + 可选 VPIDM 扩散增强（权衡效率）。诚实：−5dB 是极端档，性能下降是预期的，靠增强 + 联合训练 mitigate。

**D2. 0.2s 超短 enrollment 怎么办？** → 见红线 8。US-PVAD 内生声纹路线。

**D3. 100% 完全重叠怎么办？** → 见红线 4。最薄弱处，靠双路 PVAD + 拒识冗余兜底（⚠️ SE-DiCoW cross-attention 已弃用——见 B4 范式不兼容；现实际靠 sim_thr 拒识兜底，**不是三路融合**——LLM 在 AND 逻辑下只减拒不加拒），诚实承认仍难。

**D4. 未见说话人/方言泛化？**
答：DiCoW 不学跨空间映射天然对未见说话人友好；PVAD 用声纹条件化也泛化；数据增强覆盖方言/年龄/性别。方言是真实风险，靠增强广度。

### E. 创新性与对比 SOTA

**E1. DiCoW 开源谁都能用，你的新颖性在哪？**
答：**诚实**——单点技术无原创（见红线 9）。差异化在系统级：数据增强深度（D1）+ 中文家居微调（D2）+ 端到端联合（D3）+ 工程优化（Gap3 批量化 / 繁简归一 zhconv / langfix）+ 效率（D5）。⚠️ **原列的"三路融合拒识（D4）"已被 3-agent 对抗审查证伪**（`decide_reject` 是 AND，LLM 只减拒不加拒，不能作"三路融合"强项），**答辩不再列为差异化**。是**工程整合 + 适配深度**，不是理论首创。

**E2. 和 TS-ASR-AD（Interspeech 2025）比你强在哪？候选 X 真新吗？**
答：TS-ASR-AD 用轻量 CTC decoder，我们用 Whisper-large-v3-turbo 强底座（精度上限高）。候选 X 是把 TS-ASR-AD 范式**落地到中文家居 + 拒识**场景的工程方案，不是理论超越。诚实定位为工程。

**E3. 别的强队也读美的论文也用 DiCoW，你凭什么赢？**
答：完成度 + 适配度 + 效率。把开源 baseline 在中文/家居/极端 SNR/超短 enrollment 调到极致，比换新模型更易拉开差距。这是比赛本质。

### F. 工程落地与失败应对

**F1. 效率不达标怎么办？模型太大 L20 放不下？** → 见第三节风险预案 2、7。

**F2. 真实数据分布和仿真差很远怎么办？** → 风险预案 8。

**F3. 拒识误杀/漏拒怎么调？** → 风险预案 3。

**F4. 时间不够做什么取舍？** → 风险预案 10。核心：**严格按"对评分权重的边际贡献 × 完成概率"排序，不按技术新颖度排序**。保 W1–W7 底盘（40+40+20 稳健组合）优先于候选 X/Y。

**F5. 可复现性？无开源代码的模块（US-PVAD/ASE-PVAD/DSENet）怎么办？**
答：US-PVAD/ASE-PVAD 架构清晰需自实现（DPRNN/Conformer/FiLM 均开源组件）；DSENet 无权重仅作算法借鉴。诚实承认自实现风险，优先用有权重资产（主线 **DiCoW_v3_2**，HF 权重 `BUT-FIT/DiCoW_v3_2`、CAM++、Qwen；注：SE-DiCoW 权重虽有但范式不兼容已弃，见 B4）。

---

## 三、13 类风险预案（现象 / 根因 / 应对 / 备选 / 触发条件）

### 风险 1：测试集通道数未定（架构分水岭）
- **现象**：① 实际多通道但我方押单通道 → 漏做多通道增益；② 实际单通道但白做空间前端；③ 误判远场缺 AEC。早期信号：A 集样例出现"阵列/多路/6 麦/远场采集"措辞；wav 形状 [C,T]。
- **根因**：通道数未确认，前端策略锁死。
- **应对**：**第一时间向主办方确认**或分析 A 集 wav 头。组合主线通道无关（两种都跑得通），空间前端作增量。
- **备选**：单通道→组合主线直接用；多通道→前端叠加 DSENet/VSAEC（⚠️ 无权重，仅算法借鉴）+ 仿真造多通道数据。
- **触发**：数据下发即判。

### 风险 2：推理效率不达标（威胁 20%）
- **现象**：L20 单测 Whisper-turbo FP16 RTF 偏高；叠加 Qwen-3B 拒识后串行 RTF 超限。
- **根因**：两个重模块（turbo + 3B LLM）串行在同一前向链。
- **应对**：① turbo(蒸馏，快 8 倍) + INT8/4 量化（bitsandbytes/GPTQ）+ 流式 chunking——确定性收益；② 拒识用 3B 小模型 + 自适应 CoT（简单指令不 think）。
- **备选**：退 SpeechBrain 轻量 TS-ASR recipe 牺牲精度保效率；TS-RNNT 形态上限冲刺（不 all-in）。
- **触发**：W6 实测 RTF > 目标阈值。

### 风险 3：拒识误杀与漏拒
- **现象**：误杀=目标合理指令被拒（CER 反而升高）；漏拒=非目标/不合理被转（拒识率掉）。
- **根因**：声纹 max_sim 分布漂移（极重 babble 下 target sim 退化：median 0.28，sim<0.06 仅 7.7% 非主流）、阈值未校准。⚠️ 真测后拒识实为**单信号（max_sim）阈值**为主，LLM 在 AND 逻辑下只减拒不加拒——不是"三路信号分布漂移"。
- **应对**：验证集记录每个 reject 的 reason code（哪路触发）+ GT 比对；声纹路（enrollment 短/CAM++ 噪声污染）易误杀，给低权重或设保护逻辑；用真实数据校 thr（实测 thr=0.4 → neg RR **98.5%**，但 pos correct 31%@0.2 → 14%@0.4 真退化，trade-off 待主办方口径定）。
- **触发**：验证集误杀率或漏拒率超阈值。

### 风险 4：目标 CER 不达标
- **现象**：实体词/同音词错字（"美的"→"美迪"）集中；−5dB 段劣化最重。
- **应对**：中文家居微调（D2）+ 实体词库 RASTAR 纠错（M5）+ 数据增强覆盖噪声；按字符级 CER 拆分定位实体词错误。
- **触发**：仿真验证集 CER > 目标。

### 风险 5：超短 enrollment 声纹不稳
- **现象**：同说话人多次 enrollment 的 e_target 余弦自相似度方差大；低于 0.5s 档判别力不足。
- **应对**：走 US-PVAD 内生声纹路线（0.2s REC 62.74%）；ASE-PVAD 从混合音频选关键帧反哺（⚠️ 仅 0.5s+ 验证过）。
- **触发**：enrollment < 0.5s 或声纹自相似度方差大。

### 风险 6：100% 完全重叠
- **现象**：重叠帧 TSS≈NTSS 概率都高、PVAD 抖动；STNO 对两人几乎相同。
- **应对**：双路 PVAD 联合判 O；FDDT 凸组合软过渡容错。⚠️ 原列"SE-DiCoW cross-attention 解重叠"已证伪弃用（见 B4，范式不兼容），100% 重叠在单通道下是死区，靠 sim_thr 拒识兜底而非 cross-attention。
- **诚实**：最薄弱处，靠多源冗余兜底，性能下降预期。

### 风险 7：模型过大 L20 受限
- **现象**：turbo(~809M) + Qwen-3B 同时常驻显存吃紧；FP16 推理 >18GB；OOM。
- **应对**：INT8/4 量化降显存；模块分时加载（声纹/PVAD 轻量常驻，turbo/Qwen 按需）；KV cache 优化。
- **触发**：nvidia-smi 显存逼近 48GB 或 OOM。

### 风险 8：真实数据分布偏移
- **现象**：真实验证集 CER 较仿真劣化 ≥8 个点；系统性替换错误（实体词近音、英文泄漏）；STNO 帧级分布漂移。
- **应对**：真实数据 fine-tune（权重有初始化收敛快）；仿真参数据真实分布反推校准；域适应。
- **触发**：仿真→真实 CER 劣化超阈值。

### 风险 9：训练数据不足
- **现象**：仿真源语料规模/说话人对多样性不足（vs Libri3Mix）；重叠/噪声覆盖不全；过拟合仿真分布。
- **应对**：扩数据源（AISHELL-1/2/3 + WenetSpeech + MagicData-RAMC）；LLM 合成指令语料 + TTS 合成多人对话；增强广度。
- **触发**：仿真验证集过拟合或泛化差。

### 风险 10：时间不足的取舍
- **现象**：W1–W7 + 候选 X/Y + D1–D5 远超周期，被迫砍模块。
- **应对**：**严格按"对评分权重边际贡献 × 完成概率"排序**：① 保 W1（DiCoW_v3_2 跑通；⚠️ SE-DiCoW 已弃用，不再作 W1 地基）+ W2（数据）+ W6（评测）= 地基；② 保 CER/拒识核心（M4+M6）；③ 效率优化（M7）；④ 候选 X/Y 仅在底盘稳后投入。**绝不**为冲上限牺牲底盘。
- **触发**：进度落后于阶段里程碑。

### 风险 11：DiCoW 条件化在极重 babble 下反作用（⭐ 2026-07-06 Phase 1 已化解）
- **现象**：DiCoW 的 FDDT/STNO 条件化原设计是"用 STNO 概率凸组合抑制非目标帧"，但在 datasetA 极重 babble（sim 0.2–0.4 桶、目标声纹 median sim 0.28）下**条件化反而毒化转写**——sim[0.2,0.3) 桶 CER **1.606**、sim[0.3,0.4) 桶 CER **1.523**（均远高于 DiCoW 整体 1.248，是 DiCoW 最毒桶），且**主动制造 18.80% 英文幻觉**（vanilla 同数据仅 0.59%）。
- **根因**：极重 babble 下 STNO 概率对 target 的判别失效（mel 退化 + 声纹 sim 低 → 凸组合权重错乱），FDDT 把错乱权重焊进 Whisper 输入，比"不条件化"更糟。条件化是 DiCoW 的核心卖点，恰恰在题目最在意的极重 babble 区段反噬。
- **应对（已落地）**：改用 **vanilla Whisper-large-v3-turbo + target extraction** 路线——diar+wespeaker 选 target（复用 `enroll_infer` 逻辑）→ 切 target timeline 段（含重叠区）拼接 → vanilla Whisper 转写（去掉 stno_mask/FDDT 条件化）。**全量 1362 条 zero-training 真测：CER 0.664 vs DiCoW 1.248（几乎减半）、英文幻觉 0.59% vs 18.80%（从根消灭）、overall CER thr=0.20 → 0.711（CER 40% 腿从 ~0 分变 ~11 分）**。
- **备选**：① 声纹强化（CAM++/US-PVAD 内生声纹）把 target selection 准确率再拉一档；② 数字 initial_prompt（家居指令高频数字，锦上添花）；③ 若后续真测发现某些轻 babble 段 DiCoW 仍优，可按 sim 分桶路由（高 sim 桶走 DiCoW / 低 sim 桶走 vanilla）。
- **触发**：已触发并化解；后续每次 ablation 须监控英文幻觉率 + sim 分桶 CER 防回退。
- **产物**：`code/exp_vanilla_vs_dicow.py` + `code/analyze_vanilla_full.py` + `code/exp_vanilla_full.json` + memory `h3-dicow-conditioning-backfire-vanilla`。

### 风险 12：L20 端到端耗时未真测（威胁效率 20%）
- **现象**：官方在 **L20 48GB** 上评效率分（推理时间 10% + 内存 10%，RTF 按 **batch=1** 测），但本机只有 RTX 4060 Laptop 8GB，当前 RTF **0.2543**（4060 pos 全量 3498s 音频，关 LLM + vanilla + thr0.27）**不能直接当 L20 数字报**——L20 算力更强、显存更大，RTF 会更低，但**未实测**。
- **根因**：本机硬件 ≠ 评测硬件；submit_infer 显存自适应（L20 48G 可大 batch，但 RTF 按 batch=1 测，大 batch 不加分）未在 L20/L40 上验证。
- **诚实应对**：① 答辩报 RTF 0.2543 时**明确标注"4060 实测，L20 待测"**，绝不把 4060 数字当 L20 数字；② 给 L20 预期推断（算力比 4060 强数倍，RTF 必然更低，方向有利）；③ **租 AutoDL L40（L20 同芯 Ada）跑端到端 batch=1 验证**（memory `l20-eval-hardware`），赛前完成；④ 提交脚本已做显存自适应 + per-utt 显存日志（`repro.peak_gpu_mib`），L20 直接可跑。
- **备选**：若 L20 实测 RTF 仍偏高，turbo 已是蒸馏版（比 large-v3 快 ~8×），叠加 INT8/4 量化 + 流式 chunking 兜底。
- **触发**：L20 实测 RTF > 1.0（效率腿丢分）；或显存 OOM。

### 风险 13：边缘部署目标硬件未定义（落地可行性 20%）
- **现象**：决赛评分"应用价值和落地可行性 20%"需讲部署路径，但**赛事方未定义终态目标硬件**（家电 MCU / 边缘网关 / 本地服务器算力差几个数量级）。当前方案依赖 Whisper-large-v3-turbo(809M) + Qwen-3B，只能在 GPU 服务器/L20 跑，**离 MCU 量级差太远**。
- **根因**：题目聚焦"识别技术"非"端侧部署"，落地硬件口径空缺（memory `edge-deployment-end-goal`）。
- **诚实应对**：① **承认当前是服务器级方案**，落地分讲"云端/网关级可落地"+ 给 MCU 级降级路线（INT4 量化 + Whisper-tiny 蒸馏 + 流式 + 拒识前置 KWS 唤醒，作未来工作）；② **不夸大**"已可上家电"——评委追问"放洗碗机 MCU 上 RTF 多少"会穿帮；③ 向主办方确认目标硬件口径（若只是云端/网关，当前方案已达标）。
- **备选**：若确认需 MCU 级，量化蒸馏 + KWS 前置漏斗（唤醒词先过 KWS，仅命中再跑全 pipeline）。
- **触发**：评委追问部署硬件 / 落地分被压。

---

## 四、完整性 critic 指出的遗漏与补强

> critic 审完全部 QA + 风险后指出的最大盲区。

### ⚠️ 最大遗漏：多通道分支几乎零覆盖
- **问题**：22 个 QA 全程在单通道设定下答辩，但团队自己把"单/多通道"列为"架构分水岭"。一旦确认多通道，评委连环追问：
  - (a) 预案 B 说"多通道叠加 DSENet 是冲上限利器"，但 **DSENet 原文只验证 6 人重叠 3 通道圆阵 r=30mm、未测 −5dB、无预训练权重**——凭什么说"利器"？
  - (b) 若多通道成立，DSENet/KWS 空间是文档承认"最契合出题方审美的路线"，**为何主押 SE-DiCoW 不押 DSENet？** 是不是又回到"已知次优还提交"的矛盾？（⚠️ 2026-07-04 真测后此问已双失效：单通道确认 → DSENet 全弃；SE-DiCoW 范式不兼容 → 已弃。实际主线是 DiCoW_v3_2，此问不再适用。）
  - (c) **DOA 在重叠区同样有歧义**（两人同向），空间路线解决不了 100% 重叠——多通道不是银弹。
- **补强**：① 答辩前**必须确认通道数**；② 若多通道，准备 DSENet 可行性的诚实评估（无权重需从零复现，风险高）；③ 修正"利器"措辞为"潜在增量，受限于无权重+未测极端条件"；④ 准备"DOA 重叠歧义"的应对（空间路线与 PVAD 路线重叠区都难，互补而非替代）。

### 其他遗漏点（critic + 自查）
- **拒识率定义未澄清**（C2）——向主办方确认精确指标。
- **指令集范围/领域边界**未定——影响 M6 LLM"合理性"判别标准。
- **enrollment 格式**（整句唤醒词 vs 任意片段）未定——影响声纹策略。
- **可复现性/部署成本**——准备无权重模块（US-PVAD/DSENet）的自实现工作量说明。
- **数据合规**——若用 TTS 合成/LLM 生成语料，准备合规说明。

---

## 五、答辩一句话定位（开场可用）

> "我们的方案是**稳健的组合主线（保下限：wespeaker+DiariZen 锁 target → STNO → DiCoW 转 + sim_thr 拒识，全开源、可跑通、可降级）+ 端到端联合训练的差异化（冲上限：契合反 cascaded 审美，对标 TS-ASR-AD）**。我们诚实承认级联是次优架构、完全重叠是最薄弱处——这些都是 X1 阶段的硬交付，我们用数据回答，不空谈。"
>
> ⚠️ **2026-07-04 真测后修正定位（答辩实际开口用此版）**：组合主线（**关 LLM**，`--no-llm`）真测基线 = **neg RR 98.5%@thr0.4 + RTF 0.24（4060，sim_only）+ pos CER ~1.0（架构极限，诚实放弃 CER 40%）**。靠 RR 40% + 效率 20% 拿分，CER 40% 不再硬冲；"三路融合拒识"作强项已被 3-agent 对抗审查证伪（AND 逻辑、LLM 只减拒不加拒），答辩**不列**为强项；SE-DiCoW 已尝试放弃（范式不兼容）。冲 CER 的未来方向是端到端联合 X。trade-off 诚实交代：关 LLM 牺牲了 28 条 pos 救回（26 条 CER=0.000 完美），换 RR+效率。
>
> ⭐ **2026-07-06 Phase 1 真测突破（最新，开口优先讲）**：保底盘仍是关 LLM 保底（RR 98.5% + RTF 0.24），**但 pos CER 这条腿被 vanilla 路线救回来了**——全量 1362 条 zero-training 真测发现 DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】（sim 0.2–0.4 桶 CER 1.5–1.6、英文幻觉 18.8%），改用 **target extraction（diar+声纹切 target timeline）+ vanilla Whisper-large-v3-turbo**，CER 几乎减半：**转写 CER 0.664（vs DiCoW 1.248）/ correct_rate 45.6%（vs 31.4%）/ 英文幻觉 0.59%（vs 18.80%）**，overall CER（含拒=1.0）**thr=0.20 → 0.711**（vanilla 终于把 overall 拉到 <1），**CER 40% 腿从 ~0 分变 ~11 分**。⚠️ 这推翻了"pos CER ~1.0 是架构极限 / 无 thr 能救"的旧结论——根因是 DiCoW 条件化反作用，不是 cascaded 本身；vanilla zero-training 就破局，后续声纹强化（CAM++/US-PVAD）+ 数字 initial_prompt 还能再压。契合出题方反 cascaded 审美 + 诚实归因 + 真数据背书。详见 A6 + memory `h3-dicow-conditioning-backfire-vanilla`。

---

*本文档经 47-agent 对抗验证生成，标注 ⚠️ 处均为被攻击击穿、需诚实应对的高危点。答辩前请重读第一节（9 条红线）。*
