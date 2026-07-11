# AGENT 交接文档 — 美的目标说话人 ASR（XH-202615）

> **交接时间**：2026-07-11（**前沿探索闭环 + Qwen3-ASR 候选2 证实(CER 腿+10分) + 集成落地**：19路并行探索报告 docs/前沿探索报告_2026-07-10.md + faster-whisper/BoH no-go + code/.venv speechbrain 修复 + Qwen3-ASR 全量1350条官方口径 overall CER **0.344**(vs vanilla 0.595) + enroll_infer/submit_infer --asr-backend qwen 集成）。上一轮标注分发成果已 push(b377054), 等队员回收。
> **下个 agent 读序**：本文件【2026-07-11 最新】段（↓）→ CLAUDE.md → 关键 memory（**cer-breakthrough-candidates** / multi-annotator-dispatch / content-gate-decision / official-scoring-spec / dataset-split-spec / reproducibility-hardening / mimo-asr-backend-potential / unified-thr-decision / h3-dicow-conditioning-backfire-vanilla / spk-oracle-poc / baodi-config-no-llm / submit-script-verification / lessons-pitfalls）→ REPRO_SETUP.md
> **当前 git**：`master` @ `53a6521`（本地未 push）。2026-07-08/09/10 改动均已 commit（53a6521 多人标注工具链 / 14d0c58 error_analysis / 78e0576 官方口径）。⚠️ annot_pack/(2168音频+116M zip) 被 .gitignore(`*.zip`+`code/*/`)忽略不入库。下个 agent 先 `git status` 核对。

---

## 【2026-07-11 最新】前沿探索(19路) + Qwen3-ASR 候选2 证实 + 集成落地 + P0 数字收尾

### ⚠️ 2026-07-11 P0 收尾（本 session 续：7-agent 路线核实 workflow + 双口径坐实 + wesep defer）

- **7-agent 路线核实 workflow**（ultracode，375K tokens/104 tool calls）：fan-out 评估 7 候选方向(A1 归一/A2 死区对抗/A3 run-twice/A4 提交数字/B1 FireRedASR/C1 wesep/C2 beam)，每 agent 核实 file:line 现状 + 收益/成本/风险/答辩价值。用户决策：**P0 数字收尾先做** + **wesep defer**。
- **两个洞堵上**：①**+10.1 是 transcribe 不拒口径虚高**——提交进排名公式用【含拒 overall】(pos 允许拒, 2026-07-08 确认)，qwen 含拒 thr0.27=**0.5934**(CER 腿 +4.29) vs transcribe 0.3436(+10.07 为诊断上限)；②**0.3436 此前不可复现**(poc json 仅 per-sample 0.3848)，`code/recompute_qwen_official.py` 独立坐实落盘 `qwen_official_cer_workpoints.json`。
- **归一零效应坐实**：1350 条 qwen 输出 0 阿拉伯数字 0 真繁体(原生中文"二十五度")，digit_postproc/to_simplified 均 no-op，raw==归一==0.3436 逐位相等。提交侧 enroll_infer:384 已接归一，无 cn2an/zhconv 式漏洞。
- **死区 0.459 坐实(官方累计池)**：n=396 qwen 0.459 vs vanilla 0.784；0.459 < oracle 0.607。✅ **A2 对抗验证已完成**(见 follow-up#5 + RESULTS A2 段): 用户听音 cmd_2091/2137 坐实 **H1 真实突破**(音频可辨 qwen 听对, 非LM幻觉), 死区是混合桶(B类声纹失败但音频可辨qwen突破 + A类真摧毁H2少数), spk-oracle-poc 物理地板修正为 vanilla OOD 伪地板。
- **含拒 thr 扫描**(官方池)：0.20 qwen0.4912/vanilla0.6544 | **0.27 qwen0.5934/vanilla0.7007** | 0.30 qwen0.6435 | 0.35 qwen0.7221 | 0.40 qwen0.7993。全档 qwen 优于 vanilla。
- **提交数字(thr0.27, w1=w2=0.4)**：qwen CER腿16.26+RR腿36.20=52.46 | vanilla 11.97+36.20=48.17 | Δ+4.29(效率腿20待L20)。neg RR 0.9051 与转写器无关。
- **下个 agent 焦点(核实后优先级)**：🟡 A3 qwen run-twice(改 15 行+20min, FAQ 硬要求, verify_reproducibility:47 choices 加 qwen + seed 透传子进程) → 🟡 A2 死区对抗验证(纯分析+听音 30 条, 定答辩核心归因) → 🟡 B1 FireRedASR 横评(切片就绪, 选型+效率腿, 45% no-go) → ⏸ 等标注回收 / 03_答辩FAQ / L20 RTF。**C1 wesep / C2 beam 均 P2 defer**(wesep 零 upside + emb-mismatch 可能产模糊结论 + EoW2026/SepFormer/STNO 三重同构预警 no-go 85%)。

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
2. 🟡 **L20 RTF 真测**(Qwen3-ASR RTF 0.289@4060 慢于vanilla 0.16-0.24, L20待测, 效率腿时间分可能小失分-1~2; 租AutoDL L40)
3. 🟡 **FireRedASR 横评**(定 Qwen3-ASR vs FireRedASR 选型; FireRedASR干净CER 2.89%略优Qwen3-ASR 3.76% + RTF0.087更快; 需clone repo; 45% no-go 无 babble 训练)
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
