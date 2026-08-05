# AGENT_HANDOFF 2026-08-05：效率腿作战 + CER 收口后续

> **给下个 agent**：本文档自包含，读完即可干活。核心框架（implementation-NO-GO vs direction-NO-GO）以 `docs/反瓶颈审计与后续Agent作战令_2026-08-05.md` + `docs/反瓶颈审计补充_speaker-aware_2026-08-05.md` 为准，本文件不重复，只补「本次会话成果 + 效率作战 + 立即任务 + 环境/坑」。
> **当前焦点（用户指定）**：效率腿优化——还没到极限，最大未试空间在 ASR 推理引擎，不在量化。

> ⚠️ **2026-08-05 session B 续会话已更新本文件（见 §10）**：① scene-route **已集成提交主线**（commit c5af845，`run_baodi.sh` 默认 `BAODI_SCENE_ROUTE=1`）→ 提交线 CER **0.5952**/RR **0.9409**/Overall **0.6729**（+0.74 分）；§1「分场景候选 +0.99」已过时（+0.99 是 pos-only 上界，Overall 真值 +0.74）。② **code/.venv 已重建**（§7「venv 坏」过时），全链路可跑。③ thr 扫描 NO-GO（降 thr 换 CER 不划算）。最新状态见 §10。

---

## 0. TL;DR

- **当前定位**：Overall **0.6636**（主线，~第9）/ **0.6777**（分场景，第8）。RR 0.9473 已超榜第2/3（顶尖），**CER 含拒 0.62 是唯一短板**。
- **CER 低成本路径已 implementation-NO-GO**（拒识规则 n_spk/content、合成 TSE phase2/LoRA、speaker-conditioned encoder Phase-3、speaker-aware fusion），**但方向未封顶**——见反瓶颈审计 4 条 direction-unresolved。
- **CER 损失的本质**（反瓶颈审计已定性）：transcribe(不拒) CER 0.3442 vs 含拒 0.62，gap 是「目标句被拒按 CER=1 计罚」，不是 ASR 听错；核心困难是 pos 低 sim 目标句要放行、neg 低 sim 干扰人家居指令要拒掉，**身份判别**是真难点。
- **下个 agent 首要任务**：效率腿 profile + 优化（用户正在问，详见 §4）。

---

## 1. 锁定数字（别再算，直接用）

| 项 | 值 | 说明 |
|---|---|---|
| 评分公式 | **Overall = 0.5(1−CER) + 0.5×RR** | w1=w2=0.5，**从榜单反推确认**（榜1/榜2 验证吻合），别再问主办方 w1/w2 |
| CER 口径 | **累计池**（total_errors/total_chars） | 主办方脚本坐实；逐句平均 0.7230 是另一口径，**勿混用** |
| 主线 CER（累计池）| **0.5952** ⚠️§10 | scene_route **已集成默认开**(c5af845)；老主线(sr关)0.6201 |
| 分场景 CER | 0.5919(1350子集) | =现主线 scene_route；+0.99 是 pos-only 上界, Overall 真值 +0.74(§10) |
| RR | **0.9409** ⚠️§10 | scene_route 集成后(老主线 0.9473)；content_gate on thr0.27 |
| Overall | **0.6729**(scene_route,§10) / 老主线 0.6636 | 榜7 0.7018 / 榜8 0.6748 |
| transcribe CER（不拒，诊断）| 0.3436 | ASR SOTA 上限（FireRedASR 0.3501≈qwen），含拒 0.62 的 gap 全是 pos 误拒罚 |

---

## 2. 必须遵守的纪律（违反会出事）

1. **区分 implementation-NO-GO vs direction-NO-GO**（反瓶颈审计核心）：某套代码/参数/数据没过门槛 ≠ 整个方向无空间。旧结论里「封顶/到顶/别再投/exhausted」都是**具体实现/规则/合成域**层面，误读成方向封顶是被纠正的错。
2. **小收益实验三件套**：raw vs candidate **同进程配对**（Qwen 跨进程有 +0.0126 CER 漂移）+ **bootstrap CI** + **pos CER / neg RR / RTF 同看**（不联动看会漏掉 pos 救回却被 neg 反噬）。
3. **不把代理当成功**：SI-SNR/mel/loss 下降 ≠ ASR CER 改善（感知-识别鸿沟，phase2/Phase-3 教训）。
4. **CER 累计池**，勿混逐句（差 0.10，腿差 4.36 分）。
5. **batch=1**（官方 RTF 口径，大 batch 不加分）。
6. **A 集禁训练**（泄漏）；外部公开数据训练合法（主办方 2026-07-27 确认）。

---

## 3. 本次会话（2026-08-02~05）已完成的成果

1. **公开数据集调研**：REAL-T 生态（SLT2026 REAL-TSE Challenge ≈ 我们题目）。REAL-PS4（HF TaurenMountain/REAL-PS4，CC-BY-4.0，已公开）/ REAL-T（需 challenge 注册，2026-05-31 关闭，公开拿不到）。详见 `docs/公开数据集调研_目标说话人ASR_2026-08-02.md` + memory `target-speaker-asr-public-datasets`。
2. **REAL-TSE baseline 跨后端双崩**：官方合成训 baseline（Libri2Mix 训 BSRNN+ECAPA）在 A 集 20 条，Zipformer CER 0.806 / Qwen3 0.754（Δ仅 0.052）→ 合成训 TSE 死路，换 SOTA ASR 也救不回。详见 `docs/REAL-T数据获取与基线_2026-08-02.md` + `docs/REAL-T_ASR公平对照与PS4_2026-08-02.md`。
3. **PS4 leaderboard（免解锁）**：领域 TSE 天花板 ~0.6（第1 MERL 0.613 / 第2 PS4 0.639 / baseline 0.829）；baseline 在 REAL-T(0.829)≈在 A 集(0.806)难度可比 → 我们切割策略 0.3436 粗略比整个 REAL-T challenge 强。PS4 推理阻塞（模型类 bsrnn_legacy.py 私有未公开），邮件草稿 `docs/PS4作者请求邮件_2026-08-05.md` 待发。
4. **CER 真值纠正**：0.5934（07-28 旧）→ **0.6201**（累计池，07-30 全量独立复现）。`docs/全量提交评测_2026-08-05.md`。
5. **pos 误拒归因**：被 thr0.27 拒掉的 630 pos 里 55.9% 是误拒（352 条 CER<0.3，其中 283 条 CER=0 完美被罚 1.0）。`docs/pos误拒归因_2026-08-05.md`。
6. **n_spk=1 强制不拒 NO-GO**：neg 70.25% 是单人独白说家居指令（题目 trap），n_spk=1 不拒 net Overall −0.2572。`docs/neg_nspk验证_2026-08-05.md`。
7. **speaker-aware fusion NO-GO（implementation）**：max_sim+stno fusion AUC 0.846 但 precision 上限 0.75 < break-even 0.763（pos 池 1364 vs neg 池 474 杠杆不对称），5-fold hold-out net ΔOverall +0.0003±0.0019（零）。`docs/speaker_aware探查_2026-08-05.md`。**归类 implementation-NO-GO，未封死 decoder 置信度/真实数据拒识校准等（见反瓶颈审计补充）**。

---

## 4. ⚡ 效率腿优化作战（首要任务，用户指定）

### 4.1 现状 vs 极限
- RTF **0.24**（4060，关 LLM，pos 全量 13.8min batch）/ **L20 外推 0.09-0.12**，实时阈值 0.2-0.3，已优秀但**非极限**。
- Qwen3-ASR-1.7B 在 L20 batch=1 **理论 RTF ~0.05-0.08**，还有 **30-50% 空间**——但 pipeline 开销（enroll/diar/加载）把纯 ASF RTF 拉到 0.09-0.12。

### 4.2 已做 / 已证伪（别重复）
- ✅ batch ASR 子进程 5x / SE 关省 30% RTF / batch=1 锁定 / Gap3 enroll 模型加载 1 次（39→11s）
- ❌ **bnb int8**（Whisper +89% / Qwen +299% 慢；bnb int8 kernel 在 batch=1+decoder overhead 大）
- ❌ faster-whisper int8（CER +0.0156 + 慢）
- ⚠️ **「int8 证伪」特指 bnb int8，≠ 所有量化证伪**。ONNX int8 / TRT int8 / FP8 用不同（优化）kernel，没试。

### 4.3 第一步：profile 找瓶颈（必做，盲优化会投错地方）
在 4060 跑 `bash code/run_baodi.sh pos 0.27` 小样本（50-100 条），拆解 `submit_infer.py` 4 阶段（enroll_infer / se[已关] / asr / llm[已关]）耗时：
- 工具：`time.perf_counter` 包各阶段 / `nvidia-smi` / `py-spy record --pid` 火焰图
- 要回答：① ASR(Qwen3) 占比多少？② pipeline 是几个进程、模型加载几次？③ enroll_infer(wespeaker+diar+cut) 占比？④ 进程启动/内存拷贝开销？
- 产物：`docs/效率profile_2026-08-XX.md` + 各阶段耗时表

**✅ 2026-08-05 已完成**：`docs/效率profile_2026-08-05.md` 已拆出 A/B/C/C'/D。瓶颈是 Qwen 转写 C（可变成本约 64%）；Diar `torch.compile` 与 Qwen audio-tower `torch.compile` 已实测不集成；WSL vLLM 已打通并取得 conditional-GO。

**✅ 续跑全量 A（WSL，仅作候选复核）**：vLLM 正 1364 条 `CER=0.7251`、负 474 条 `RR=0.9051`；正端到端 `RTF=0.3255`（1138.5s），负端 sidecar 为 `load=73.2s + transcribe=101.8s`。WSL 必须临时关 cuDNN，负样本使用 `max_model_len=1536/gpu_mem=0.65`，且发现 Diar 与 vLLM 双模型竞争和 EngineCore 管道生命周期问题；**不切默认、不作纯 vLLM 质量归因**。全量细节见效率 profile §10.8。

### 4.4 未试优化（按潜力 × 低风险，profile 后针对投）
**A. ASR 推理引擎（最大潜力，若 ASR 是大头）**
| 手段 | 预期 | 精度 | 怎么试 |
|---|---|---|---|
| torch.compile | 20-40% | 零 | `qwen_model = torch.compile(qwen_model, mode="reduce-overhead")`，注意 audio encoder+decoder 结构坑 |
| ONNX Runtime（GPU EP）| 30-50% | 零 | `optimum` 导出 ONNX，onnxruntime-GPU 推理；验证输出等价 |
| FP8（L20 Ada sm_89 原生）| 1.5-2× | 小 | transformer-engine / torchao FP8，L20 实测（4060 不支持 FP8） |
| CUDAGraph | 10-30% | 零 | batch=1 下减 kernel launch 开销，`torch.cuda.make_graphed_callables` |
| ONNX/TRT int8（≠bnb）| 30-50% | 小 | onnxruntime dynamic_quantize / TRT int8，校准集用 A 集外中文 |

**已新增验证**：Qwen 官方 vLLM 后端在独立 WSL venv 可用；正式 backend batch=1 的 100 条纯转写约 30s→24s，文本归一化 CER 不变，但模型加载约 29s→68s，先 conditional-GO，待 L20/组委会 `duration` 口径后再默认集成。

**B. pipeline 合并（中潜力，若进程/加载开销大）**
- 确认 `submit_infer.py` 是否 4 阶段 subprocess（每阶段独立进程加载模型）。若是，合并成单进程加载 1 次、内存传 audio（省启动/加载/拷贝）。
- memory Gap3 只优化了 enroll 阶段加载，**整体合并未确认**。

**C. 激进（高成本，暂缓）**：Qwen3 early-exit / speculative decoding / 蒸馏到 0.6B（训练成本 + CER 风险）

### 4.5 约束
- **batch=1** 锁定（官方 RTF 口径）
- **绝对 RTF 不能 4060→L20 外推**（4060 AD107 ≠ L20 AD102，只能 L20 实测）；4060 数字只看**相对**改善
- **bnb int8 证伪**，别重试
- 每个优化必须 **RTF + CER/RR 同看**（加速不能掉精度）

### 4.6 GO 判据
- profile 出瓶颈 + 可执行优化清单（信息价值 ✓）
- 至少一个优化在 4060 显著降 RTF（>20%）且 CER/RR 不退化（同进程配对验证）
- 给出 L20 batch=1 实测计划（绝对数字要 L20 跑）

---

## 5. 下一步任务清单（按 priority）

### Task 1 ⚡ 效率 profile + 优化（§4，用户焦点，立即做）
**✅ profile 已完成**；Diar/audio-tower compile 为 implementation-NO-GO；Qwen vLLM 已完成 WSL 全量候选跑通但仍是 conditional-GO，受 WSL 显存/管道问题影响，不替换默认 Transformers；下一步是干净 L20/Linux 环境复核。

### Task 2 分场景路由提交 A/B 的 L20 batch=1 效率验证（反瓶颈审计⑤，integrate-GO 候选）
- 分场景路由 +0.99 质量分已坐实（CER 0.6168→0.5919，1350 共同样本），**只差 L20 batch=1 端到端 RTF 增量验证**（SepFormer 分离在 n_spk=2 上加多少 RTF）。
- 若 RTF 增量 <0.05 → **integrate-GO**，分场景进提交主线（Overall 0.6777，第8）。
- 依赖：L20 算力（用户租）。

### Task 3-5 反瓶颈审计 4 条 direction-unresolved（详见该文件，不重复）
按信息价值排序：
3. **同环境 TSE 全量验证**（same_env 800+train/200val 完整轮数 + A 集 1364 同进程配对）——弱 GO 的全量确认
4. **非内容拒识校准**（Qwen decoder 置信度 / token entropy / no-speech prob / 音频质量 / 双路相似度一致性，**非 A 数据标定**）——speaker-aware fusion 没探查这些 ASR 内部信号。**本轮已完成首个外部 AISHELL-1 dev speaker-disjoint 探针**：Qwen token meta 已接入可选 sidecar，base+decoder+audio 校准器 test rescue precision=0、Overall Δ=-0.0125（95% bootstrap CI [-0.0375, 0]），判为这套实现 `implementation-NO-GO`，不集成；真实域方向仍 `direction-unresolved`。详见 `docs/拒识校准探针_2026-08-05.md`。
5. **真实录音域 TSE / 目标条件化 ASR 联合训练**（REAL-PS4 / AISHELL-4 / AliMeeting / 自录家居，真实或同环境域 + target embedding/activity + ASR loss 约束 Qwen）——高风险高投入，需算力/录音

**别再做**：n_spk=1 不拒 / content rescue / bnb int8 / max_sim+stno fusion 重新扫（全 implementation-NO-GO）。

---

## 6. 待确认战略前提（需用户拍板，影响任务权重）

1. **效率腿进不进初评排名？**（最关键）
   - 榜单 Overall=0.5(CER+RR) 反推**不含效率**；但赛题写「初评 100 分 = CER40+RR40+效率20」，两者矛盾。
   - 若**进**：效率优化 + 分场景集成权重高（效率腿 ~18 分可争，分场景 +0.99 进主线）
   - 若**不进**（只 CER+RR 排名）：效率优化对初赛排名零贡献，Task 1/2 降优先级，转决赛答辩
   - 用户瞄一眼赛题文档可定。
2. **押不押真实数据方向？**（Task 3-5，高风险高投入，需算力/录音预算）

---

## 7. 环境状态 + 关键坑

### venv（本机有损坏，注意）
| venv | 状态 | 用途 |
|---|---|---|
| `code/.venv`（主）| **✅ 已重建**(2026-08-05 13:00, §10.4) | enroll_infer/scene_route/Qwen 全链路可跑；推翻"不存在/坏"旧状态 |
| `code/.venv_qwen` | **✅ 恢复完好**（torch2.6+cu124 cuda True, numpy2.4.6, transformers4.57, qwen_asr_backend import 验证通过）| Qwen3-ASR 推理；多装了 ~15 enroll_infer 依赖但不影响 Qwen3 |
| WSL `/home/lorrain/.venvs/midea_target_asr_qwen_vllm_wsl` | **✅ 独立可用**（Python3.10、Torch2.9.1+cu128、Transformers4.57.6、vLLM0.14.0、CUDA True）| Qwen3-ASR vLLM 效率候选；不与 Windows/旧 Qwen venv 混用 |
| `code/.venv_tse` | **⚠️ 被破坏**（装 PyPI pyannote.audio 致 torch→CPU、numpy→2.x、torchaudio→2.11）| 原 07-29 WeSep 训练用；TSE 已封顶不再需要，可不动；若复用需重装 torch2.6.0+cu124 |
| `code/.venv_realt` | 有（torch CPU + sherpa-onnx + wespeaker + soundfile）| REAL-TSE baseline/Zipformer 评测，CPU |

### 数据 / 评测
- A 集：`datasetA/pos.jsonl`（pos 1364，字段 `识别音频`=mixture / `唤醒音频`=enrollment）/ `neg_pairs_datasetA.json`（neg 474）；`datasetA/` 下音频，单通道 16k mono，enrollment ~1.8s
- 提交 wrapper：`code/run_baodi.sh pos|neg|B [thr]`（锁关 LLM + thr + sim_only；B 模式 thr0.27，pos/neg 默认 0.4）
- 评测：`code/eval_datasetA.py`（官方累计池 CER + RR）
- 现有 per-sample（复用宝藏，少跑推理）：`code/runs/_scene_route_full/per_sample.json`（pos 1350，sim/n_spk/transcribe CER/rejected/max_sim/sep_info）、`code/runs/full_eval_20260730_neg/_work/enroll_all.json`（neg 474，diar speakers/sim）

### 关键坑
- **CER 累计池 vs 逐句**：差 0.10，对标榜单用累计池
- **Qwen 跨进程漂移 +0.0126**：小收益实验必须同进程配对
- **SI-SDR 尺度不变**：BSRNN 输出 RMS 达 mixture 3.61-6.52×，overlap 拼接需显式 gain matching
- **可选依赖必声明**：cn2an/zhconv 缺失会静默失效（digit_postproc），见 `code/requirements.txt`

---

## 8. 关键文件索引

**本次会话产出（docs/）**：
- `效率profile_2026-08-05.md`（profile + WSL 三组实验 + vLLM 条件 GO）
- `公开数据集调研_目标说话人ASR_2026-08-02.md` / `REAL-T数据获取与基线_2026-08-02.md` / `REAL-T_ASR公平对照与PS4_2026-08-02.md`
- `全量提交评测_2026-08-05.md` / `pos误拒归因_2026-08-05.md` / `neg_nspk验证_2026-08-05.md` / `speaker_aware探查_2026-08-05.md`
- `PS4作者请求邮件_2026-08-05.md`（待用户发，GitHub issue 首选 github.com/TaurenMountain/PS4）
- **`反瓶颈审计与后续Agent作战令_2026-08-05.md` + `反瓶颈审计补充_speaker-aware_2026-08-05.md`（框架，必读）**

**外部资产（已下）**：
- `external/REAL-TSE-Challenge/`（baseline 权重 spk_emb_100 等 4 模型 + Zipformer-ZH + 评测脚本）
- `external/wesep-real-tse/` / `external/REAL-T/` / `external/PS4_repo/`
- `E:/midea_datasets/REAL-PS4/`（部分 42MB，全量 6.3GB 未下完，git-lfs 续传）
- `E:/midea_datasets/PS4_model/checkpoint_epoch037.pt`（269MB，待解锁 bsrnn_legacy.py）

**memory**：`official-scoring-spec`（评分+定位+死结）/ `target-speaker-asr-public-datasets` / `tse-phase3-sidecar-nogo` / `tse-poc-weak-go-overturns-perception-gap` / `efficiency-portability-audit`（4060→L20 迁移性）

---

## 10. 续会话（2026-08-05 session B）：thr 扫描 + scene-route 测试/集成 + venv 重建

> 本段为 this session 成果（用户"降 RR 提 CER"→ 转 scene-route 4060 本机测试/集成）。更新 §1/§5 Task2/§7/§9。

### 10.0 TL;DR
- **scene-route 已集成提交主线**（默认开，commit c5af845 push master）：CER **0.5952** / RR **0.9409** / Overall **0.6729**（+0.74 分）。
- **修正"分场景 +0.99"**：那是 pos-only CER 腿上界；首次实跑 neg 发现 RR −0.26 代价 → Overall 真值 **+0.74**。
- **thr 扫描 NO-GO**：降 thr 换 CER 最优 thr=0.23 仅 +0.6pp（±0.04 噪声带内）；RR 下降空间(0.95)是 CER 救回空间(0.276)的 3.4 倍，结构性换不动。**别再投 thr。**
- **code/.venv 已重建**（§7"venv 坏"过时），全链路可跑。

### 10.1 scene-route 质量测试（坐实 +0.74，修正 +0.99）
全量重算（pos 1350 离线 + neg 474 实跑，同 thr0.27+gate+累计池口径，锚点双向 MATCH Δ<0.0007）：

| 指标 | mainline | scene-route | Δ |
|---|---|---|---|
| 含thr0.27+gate CER（1364全量）| 0.6201 | 0.5952 | −0.0249 |
| RR（neg474 实跑）| 0.9473 | 0.9409 | −0.0063 |
| Overall | 0.6636 | **0.6729** | +0.0093 |

腿分：CER腿 +1.00 / RR腿 −0.26 / Overall +0.74。

机制二分干净（强印证）：n_spk=1（543条）Δ_gate=+0.0000（单人 identity 零副作用，文本 100% 一致）/ n_spk=2（805条）Δ_gate=−0.0429（SepFormer 重叠救回，全部收益来源）。

**neg RR −0.26 代价根因 = content symmetry trap neg 侧**：scene-route 在 neg 侧系统性把正确 gate-拒识翻成漏拒（3 条 reject→leak，0 反向）。SepFormer 给 neg 两路都是非目标干扰人，选路挑"更像指令"一路 = 挑乱码里偶然含家居关键词那路 → 误过 content_gate。3 例：信访/娱乐新闻/卖家协商（主线拒）→"信号物端/杨一梅/学生入学"乱码误过。与 multi-signal-reject-rescue task6 死结 / speaker-aware HARD neg 同源。**任何"选更像指令一路"的策略 neg 侧都有此陷阱**，解药候选=双路一致性检查（上限只 3 条/−0.26 分，ROI 低）。

### 10.2 scene-route 集成（commit c5af845，默认开）
- `run_baodi.sh` 加 `BAODI_SCENE_ROUTE` 门控（默认 1=开=提交主线），仿 BAODI_GATE/BAODI_SE 模式（行 90-104）。
- 提交跑法：`bash code/run_baodi.sh pos 0.27` + `bash code/run_baodi.sh neg 0.27`（默认 scene-route）→ 0.5952/0.9409/0.6729。
- 回退：`BAODI_SCENE_ROUTE=0 bash code/run_baodi.sh ...` → 老主线 0.6201/0.9473/0.6636。
- 冒烟验证：20 样本 15 eligible 全 routed 0 fallback，OFF 不触发（回退可用）。
- 过拟合风险低-中：机制驱动（n_spk 分水岭 + SepFormer 物理分离）占路由 2/3 层零 A 集拟合；heuristic 二选一 ~10 个手调 magic number 受三重保护（单人组零暴露 + 失败 fallback 主线 + 词典与 content_gate 共享 hold-out 背书）；neg 474 独立子集不反噬。
- **提交线变更**：原"主线=关 scene_route"→ 现"主线=开 scene_route"。

### 10.3 thr 扫描（降 thr 换 CER = NO-GO，别再投）
用户问"降 RR 提 CER"。全量重算 thr∈[0.10,0.30]（pos 1350 离线 + neg 474 + content_gate + 累计池）：
- 最优 thr=0.23：CER 0.5557 / RR 0.8945 / Overall 0.6694，ΔOverall +0.0058（+0.6pp），**在 ±0.04 噪声带内**。
- 根因（结构性）：CER 救回空间（0.6201→地板 0.3442 = 0.276）<< RR 下降空间（0.9473→0 = 0.9473），**3.4 倍**。1:1 权重下 thr=0.20 大降时 RR 损失(−0.135)反超 CER 收益(−0.102)→ Overall 反亏。
- thr=0.27 是历史对抗验证定的 pos/neg 平衡点，本质正确。**结论：不改 thr。**
- 产物：`code/data/thr_scan_overall.json` + `code/experiments/thr_scan_overall.py`。

### 10.4 code/.venv 已重建（§7"venv 坏"过时）
2026-08-05 13:00 `code/.venv` 重建（Python 3.12.13），实测能跑 enroll_infer + scene_route_backend + Qwen 子进程（code/.venv_qwen）全链路：speechbrain/diarizen/wespeaker/pyannote/transformers/torch 全 import OK，`is_valid_command('打开灯')=True` 通过。scene-route neg 474 实跑（SepFormer+Qwen 两路+选路 380s，139/139 路由成功）验证可用。
- **推翻**：CLAUDE.md 顶部 + 全量评测§5 + 本文件 §7「code/.venv 不存在/坏 = hard blocker」**已过时**。
- 后续重跑推理/全量验证/变种子复现 R5/scene-route 全量 → **直接用 code/.venv**，不必"离线重算绕开 venv"或"按 REPRO_SETUP.md 重建数小时"。
- 其他 venv 不变（.venv_qwen Qwen 子进程 OK / .venv_tse TSE 训练坏但本任务不需要 / .venv_realt wespeaker OK 缺 speechbrain）。

### 10.5 下个 agent 立即任务（更新 §5）
- **§5 Task 2（分场景 L20 验证）→ 状态变更**：scene-route 已本机集成（c5af845），不再"L20 A/B 候选"。**剩效率腿验证**：scene-route on vs off 的 RTF 增量（SepFormer 只对 n_spk=2 ~60% 样本跑 + 双路 Qwen），4060 能测（venv 在），决策 +0.74 质量净增益扣效率损失后是否仍正。用户暂不管效率，待命。
- §5 Task 1（效率 profile）仍首要，但现在 run_baodi 默认 scene-route，profile 要含 SepFormer 代价。
- §5 Task 3-5（反瓶颈 direction-unresolved）不变。
- **新增 NO-GO**：thr 调参（§10.3）。

### 10.6 留存未提交（working tree，不属 scene-route 集成）
`code/enroll_infer.py` + `code/qwen_asr_backend.py`（M，timing/profile 改动，不改 text/gate/CER/RR）/ `code/runs/efficiency_full_20260805_vllm_*/`（vllm 效率实验）/ `docs/效率profile_2026-08-05.md` / `code/_target_slices_qwen/`。留待单独评估/提交。

### 10.7 本次产物索引
- 集成：`code/run_baodi.sh`（commit c5af845）
- 测试数据（commit c5af845）：`code/data/scene_route_pos_eval.json` + `code/data/thr_scan_overall.json`
- 测试脚本（gitignore 本地）：`code/experiments/scene_route_pos_eval.py` + `code/experiments/thr_scan_overall.py`
- scene-route neg 实跑/冒烟（gitignore）：`code/runs/_scene_route_neg/` + `code/runs/_scene_route_smoke/`
- memory 新增：`scene-route-overall-true-gain`（+0.74 真值 + content symmetry trap neg 侧）/ `code-venv-rebuilt`

---

## 9. 给下个 agent 的一句话

> 先读 `docs/反瓶颈审计与后续Agent作战令_2026-08-05.md` 建立框架，再读本文件 **§10**（session B 最新：scene-route 已集成主线）。**当前提交线 = scene-route 默认开**（CER 0.5952/RR 0.9409/Overall 0.6729，c5af845，`bash run_baodi.sh pos/neg 0.27`）。首要任务仍效率腿（§4），但现含 scene-route 的 SepFormer+双路Qwen 代价——profile `run_baodi`（默认 scene-route）拆解 RTF，确认 +0.74 质量净增益扣效率损失后是否仍正。每步 RTF + CER/RR 同看，batch=1，**别重试 bnb int8 / thr 调参（均 NO-GO）**。venv 已重建可直接跑。
