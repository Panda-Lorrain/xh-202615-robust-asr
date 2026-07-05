# vanilla 后端集成 submit_infer — 设计文档（P2-①）

> **日期**：2026-07-06
> **状态**：已批准（用户 2026-07-06「按你的想法来 / 你先做你的」授权推进）
> **关联**：AGENT_HANDOFF §6.1 / memory `h3-dicow-conditioning-backfire-vanilla`
> **前置 spec**：`2026-07-01-submit-infer-and-deliverables-design.md`（submit_infer 原设计）
> **Phase 1 实验依据**：`code/exp_vanilla_vs_dicow.py`（全量 1362 条，vanilla CER 0.664 vs dicow 1.248）

---

## 1. 背景与目标

Phase 1 已强证伪：DiCoW 的 FDDT/STNO 条件化在极重 babble 下【反作用】（sim[0.2,0.3) 桶 CER 1.606、英文幻觉 18.8%）。改用 vanilla Whisper-large-v3-turbo + 声纹切 target timeline 路线，CER 几乎减半（0.664）、英文幻觉降到 0.59%。该结论已在 `exp_vanilla_vs_dicow.py` 全量验证，但**尚未进入标准化推理入口 `submit_infer.py`**——即 0.664 还不是可提交数字。

**目标**：把 vanilla 路线集成进 `submit_infer`，加 `--asr-backend {dicow,vanilla}` 开关，让保底提交能直接跑 vanilla 路线，产出官方格式的提交文件，把 CER 0.664 / overall 0.711 / batch=1 duration 变成提交数字。

---

## 2. 设计决策（已拍板）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 架构 | **方案 A**：`enroll_infer.py` 内加 `--asr-backend` | diar/wespeaker/选target 本就在此文件，共享零重复；改动最小 |
| 提交定位 | **vanilla 作提交主线** | CER 减半 + 契合反 cascaded 答辩论点；dicow 保留为 fallback + 答辩对比基线 |
| 验证 | **先 100 冒烟再全量 pos** | 两步走风险可控；冒烟先确认管线通 + CER≈0.664 + RTF 合理 |
| 繁简归一 | **dicow + vanilla 都加** | HANDOFF §8 坑4：最终提交输出要简体；一致化 |
| run_baodi 交互 | **默认 vanilla，`BAODI_BACKEND=dicow` 切回** | 一键出 vanilla 提交数字；环境变量保留 dicow fallback |
| 官方格式 | **新增 `to_submission.py` 适配** | 官方 schema 与现 result.json 不同，需最后一公里转换 |

---

## 3. 详细设计

### §1 架构核心（方案 A）

`enroll_infer.py` 加 `--asr-backend {dicow,vanilla}`（默认 `dicow`，向后兼容）。**diar→wespeaker→选 target 前置链路完全不动**，分叉点在「选完 target、算完 max_sim 之后」的转写步骤。

| | dicow 分支（现有，保留） | vanilla 分支（新增） |
|---|---|---|
| 加载模型 | `DiCoW_v3_2`（trust_remote_code） | `whisper-large-v3-turbo` |
| 转写输入 | 整条 mel + `stno_mask` 条件化 | **切 target timeline 段拼接**（`per_spk[target_idx]` 含重叠区）的 mel |
| 转写调用 | `dicow.generate(stno_mask=...)` | `model.generate()` 无 mask |
| SE-DiCoW 自登记 | 跑（仅当 `uses_enrollments`） | **跳过**（vanilla 无此机制） |
| langfix retry | 跑（英文漂移补救） | **不跑**（vanilla 英文幻觉 0.59%，langfix 是 dicow 治标，对 vanilla 是 YAGNI+副作用风险） |

**vanilla 分支逻辑**直接移植 `exp_vanilla_vs_dicow.py:139-153`（已验证 CER 0.664 的那版）：切 `per_spk[target_idx]` 含重叠区段拼接 → 退化保护（target 太短 <0.3s 退化整条）→ vanilla.generate。

**改动点（enroll_infer.py）**：
1. 加 `--asr-backend {dicow,vanilla}`（默认 dicow）、`--vanilla-model`（默认 `E:/hf_cache/whisper-large-v3-turbo`）
2. 模型加载分叉：按 backend 加载 DiCoW 或 vanilla（diar 模型两者共享）
3. 转写块加 `if backend==vanilla: ... else: <现有 dicow 逻辑>`
4. 转写后统一 `_to_simplified(text)`（繁→简，zhconv）
5. 每条输出加字段：`asr_backend`、`infer_sec`（= 现有 `dt`，单条纯推理，**不含模型加载**，对齐官方 batch=1 duration 口径）
6. `max_sim`/`rejected`/声纹匹配逻辑共享（vanilla/dicow 同一套，复用现有 217-230 行）

**输出 schema** 不变，新增 `asr_backend` + `infer_sec` 字段。

### §2 submit_infer 透传

- `submit_infer.py` 加 `--asr-backend`，透传给 `run_enroll_infer_pairs` → `enroll_infer`
- `result.json` 的 `config` 加 `asr_backend` 字段（提交可追溯）
- `timing.json` 加 `duration_infer_sec`（= 所有条 `infer_sec` 之和，对齐官方 duration 口径）
- **BAODI_OK 守卫不变**：backend 与 LLM/sim_thr 正交，dicow/vanilla 都得过守卫

### §3 繁简归一（顺带修 HANDOFF §8 坑4）

抽 `_to_simplified(text)` 工具（zhconv 繁→简），dicow 和 vanilla 转写后统一过一遍。保证 `result.json` 的 `text`、最终 submission 的 `content` 都是简体。zhconv 已在 `code/.venv`（已验证）。

### §4 run_baodi.sh 切 vanilla 作主线

- 默认加 `--asr-backend vanilla`
- 环境变量 `BAODI_BACKEND=dicow` 可切回 dicow（答辩对比基线 + fallback 必须保留 dicow 数字）
- 默认 thr 仍 0.4（与现保底一致）；thr 调优属 P2-④，待主办方口径，不在本次范围

### §5 验证计划（先冒烟再全量 + batch=1 duration + 格式校验）

1. **vanilla 冒烟**：`pos limit=100 --asr-backend vanilla`，校验：
   - CER 接近 exp 脚本 0.664（容差 ±0.05）
   - batch=1 duration（sum infer_sec）合理
   - 输出已繁简归一
2. **dicow 回归**：`pos limit=100 --asr-backend dicow`，确认改动没搞坏现有保底 fallback（行为与改前一致，CER≈1.25）
3. **全量 vanilla**：冒烟通过后 `pos 全量 1362 条`，产出最终提交 CER + duration 数字
4. **submission.json 格式校验**：跑完产出官方格式文件，schema 自检（字段齐全/类型对/id 无 `uttN_` 前缀/繁简已转/duration 含 batch=1 infer_sec）

### §6 测试策略（TDD）

**单测（纯函数，先写）**：
- `_to_simplified()`：繁→简（「空調開到」→「空调开到」）
- `_cut_target_timeline(audio, per_spk_idx, sr)`：抽出来，验证切含重叠区的拼接段（区别于抽声纹用的独占段 `collect_clean_audio`）、target 太短退化整条、空 timeline 退化整条
- backend 路由：模型加载/转写分支按 backend 选对（mock）

**集成**：冒烟 100 条端到端（真实模型，慢但必要，作为集成验证而非单测）

### §7 官方提交格式适配（新增模块 `code/to_submission.py`）

官方要求（2026-07-06 用户提供）：
```json
{"result":{"results":[{"id":"id1","content":"xxx","label":"xxx","cer":"xx"}],"final_cer":"xx","duration":"t"}}
```
- `id` ← recognition 音频名（去 `uttN_` 前缀，对齐 HANDOFF §8 坑5）
- `content` ← `text`（繁简归一后的简体；拒识条空字符串）
- `label` ← `rejected ? "reject" : "accept"`（参赛方拒识决策，默认推测）
- `cer` ← pos 条字符级 CER（拒识=1.0）/ neg 条空（ref 空，不评 CER 评 RR）
- `final_cer` ← pos 条 mean CER（默认推测）
- `duration` ← `sum(per-utt infer_sec)`，**不含模型加载**（对齐官方 batch=1 口径）

**待主办方确认口径**（做成代码常量 `SUBMISSION_DEFAULTS`，主办方回复只改常量）：

| 字段 | 默认推测值 | 不确定点 |
|---|---|---|
| `label` 语义 | `accept`/`reject` | 也可能是 `target`/`nontarget` 或回显 ref |
| pos 被错拒的 `cer` | `1.0` | 也可能主办方不允许 pos 拒（则 thr 对 pos 应=0） |
| neg 条 `cer` | 空 | ref 空，填 `0` 还是空未定 |
| `final_cer` 算法 | pos 条 mean CER | 是否含 neg、被拒条权重未定 |
| `duration` 含 SE 吗 | 含（pipeline 一环） | 「推理音频」是否含前端 SE 未定 |
| pos/neg 交法 | 一个文件混合（按 id） | 也可能分两个文件 |

---

## 4. scope 边界（本次不做）

- ❌ thr 调优（P2-④，待主办方口径）
- ❌ 声纹强化 CAM++/US-PVAD（P2-②，攻低 sim 桶）
- ❌ 数字 initial_prompt（P2-③，锦上添花）
- ❌ SE 逐条化（batch=1 严格化，当前 SE 仍批量，标注为 duration 测量 GAP）
- ❌ L20 真机验证（本机 4060 跑，L20 部署另算）

---

## 5. 环境就绪性（2026-07-06 已验证）

| 依赖 | 状态 |
|---|---|
| `E:/hf_cache/whisper-large-v3-turbo` | ✅ 在 |
| `E:/hf_cache/DiCoW_v3_2` | ✅ 在 |
| `E:/hf_cache/diarizen-wavlm-large-s80-md` | ✅ 在 |
| `code/pos_pairs_datasetA.json` / `neg_pairs_datasetA.json` | ✅ 在 |
| `zhconv` in `code/.venv` | ✅ 在 |
| `code/exp_vanilla_full.json`（全量逐条对比基准） | ✅ 在 |

---

## 6. 风险与回退

- **风险**：enroll_infer 是保底核心，改动可能搞坏 dicow 路径 → **缓解**：dicow 回归冒烟 100 条（§5 步骤2）必须通过才算集成成功
- **风险**：vanilla 分支移植 exp 脚本逻辑有偏差 → **缓解**：冒烟 CER 必须 ≈ exp 的 0.664（±0.05）
- **回退**：`--asr-backend dicow` 默认值 + dicow 路径不动，任何 vanilla 问题都可直接切回 dicow 保底
