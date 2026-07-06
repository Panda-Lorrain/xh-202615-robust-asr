# 可复现性改造（reproducibility hardening）— 设计文档

> **日期**：2026-07-06
> **状态**：已批准（用户 2026-07-06「可以，执行」授权推进；§6 DiariZen 选 git submodule；U4 判断 A 集分开/B 集统一 通过）
> **关联**：`待确认_主办方口径与外部输入.md`（FAQ 2026-07-06 公布）/ memory `submit-script-verification` / `baodi-config-no-llm`
> **前置 spec**：`2026-07-06-vanilla-backend-submit-infer-design.md`（T25，vanilla 集成，本改造的上层链路）
> **核实依据**：Explore agent 核实报告（5 模型 HF repo id 公开性 + 5 脚本集成点行号）

---

## 1. 背景与目标

FAQ 2026-07-06 公布：**提交脚本核查方式 = 完整复现结果并比对**（非仅看代码防作弊）。流程：审代码合规性 → 完整复现推理 → 复测全套指标（CER/RR/效率）。主办方不直接用上传的 json，而是运行 `submit_infer.py` 全新生成结果。**入围依据 = B 集成绩 + 脚本核查 + 客观指标**——核查不通过连入围都悬。

核查 6 项硬要求 + 当前现状（Explore agent 排查）：

| # | 要求 | 当前现状 |
|---|---|---|
| 1 | 零外部依赖（无私有文件/绝对路径） | ✗ 主线 4 脚本权重路径裸硬编码 `E:/hf_cache/...` |
| 2 | 随机种子固定 | ✗ 全无（仅 enroll_infer:177 局部 babble RNG） |
| 3 | 禁用本地缓存加速 | ✓ 无"读缓存跳过推理"逻辑 |
| 4 | 日志打印（max_sim/batch_size/耗时/显存/分支）| △ max_sim/RTF/拒识分支有；**batch_size/显存缺** |
| 5 | 峰值显存日志 | ✗ 7 脚本全无 |
| 6 | seed 固定后跑两遍比对（确定性验证）| ✗ 无 |

**目标**：让 `submit_infer` 全链路（5 脚本 + 4 子进程）可复现——种子固定、模型走 HF repo id（零本地路径）、结构化日志含显存、run-twice 验证 fp16 残余非确定、DiariZen 代码仓可拉取——通过主办方核查。

**范围**：仅可复现性 6 项 + DiariZen 仓库处理。不含统一 thr 选点 / batch 加速 / 攻 CER（声纹强化），见 §9。

---

## 2. 约束清单（C1–C11，设计必须满足）

| ID | 约束 | 来源 | 当前 |
|---|---|---|---|
| C1 | 所有样本输入流程一致（都用唤醒音频+识别音频）| 用户消息① | ✅ 已符合 |
| C2 | B 集混合集统一 thr（dir1/dir2 不给 pos/neg 先验）| 用户消息① | submit_infer 天然支持（§4.7）|
| C3 | pos 拒 = CER 1.0（删除错误，无额外惩罚）| FAQ Q1 + 用户消息① | ✅ eval/submission 已正确 |
| C4 | 效率初赛不优先（CER/RR 权重高）| 用户消息② | 种子策略不焦虑效率 |
| C5 | 零外部依赖（无私有文件/绝对路径，模型走 HF hub）| FAQ 核查 | ❌ 本次改 |
| C6 | 三 venv 隔离下 repro.py 可共享 import | agent 风险点 | repro.py 放 code/（§4.1）|
| C7 | batch_size 默认=1；允许 batch 但须结果一致；RTF 以 batch=1 测 | 用户补充 + FAQ Q4 | ✅ 已默认 batch=1 |
| C8 | CER = 系统输出 vs 标准答案识别文本，字符级 | 用户确认 | ✅ 已正确 |
| C9 | pos/neg 标签不得作为系统输入（B 集不给先验）| 用户转述主办方 | ✅ 脚本本就不读 pos/neg |
| C10 | 提交 JSON schema 固定（id/content/label/cer/final_cer/duration）| 用户转述官方格式 | ✅ to_submission 已符合 |
| C11 | 主办方核查环境能联网 HF | 用户 U1 答案 | 方案 A 成立，不打包模型 |

---

## 3. 设计决策（已拍板）

| 决策点 | 选择 | 理由 |
|---|---|---|
| 代码组织 | **方案 ② 公共模块 `code/repro.py`** | 符合 T25 抽 `text_utils.py` 先例；横切关注点归一；可独立单测 |
| 模型路径 | **HF repo id + env/CLI override** | 符合 FAQ「走 HF hub」(C5/C11)；本地 setenv 设 MODEL_* env 复用现有缓存免重下 |
| 种子/确定性 | **全 seed 固定 + fp16 保留 + run-twice 量化** | 兼顾确定与效率(C4)；工程标准做法；残余非确定(CUDA 原子加)用 verify 量化报告 |
| DiariZen 仓库 | **git submodule（A）** | 96M 代码 gitignore 不入库；submodule 指 BUT-FIT 公开仓库，主办方 `clone --recursive` 自动拉 |
| 初评 thr | **A 集分开（pos=0/neg=0.4）+ B 集统一** | A 集是两个独立测试集分别评 CER/RR，分别最优 thr 合法；B 集不给先验须统一(§4.7) |
| DF3 处理 | **例外：不走 HF repo id，保本地 + env** | `init_df` 接目录路径非 repo id；原始权重来自 GitHub；8.7MB 小 |

---

## 4. 详细设计

### §4.1 架构

新增 3 文件 + 改 5 脚本（各加 import + 调用点）：

| 文件 | 角色 |
|---|---|
| **`code/repro.py`**（新增）| 公共模块：种子/路径解析/显存计量/结构化日志 |
| **`code/verify_reproducibility.py`**（新增）| run-twice 验证脚本（§7）|
| **`tests/test_repro_logic.py`**（新增）| repro.py 单测（复用 T25 的 tests/ 结构）|
| `submit_infer.py` | 主进程 set_global_seed + `--seed` 透传 4 子进程 |
| `enroll_infer.py` | set_global_seed + 模型路径 resolve + 显存日志 |
| `se_denoise.py` | set_global_seed + DF3 env override |
| `llm_reject.py` | set_global_seed + Qwen repo id |
| `noise_classify.py` | set_global_seed（防御性，无模型/路径问题）|

**repro.py 放 `code/`** 解决 C6：5 脚本运行时 `code/` 自动在 `sys.path`（脚本所在目录），`from repro import ...` 三个 venv（`.venv`/`.venv_se`/`.venv_llm`）都能 import；各 venv 均含 torch/numpy/random。

### §4.2 repro.py 组件

```python
# code/repro.py — 可复现性公共模块（横切关注点归一）
import os, random

REPO_IDS = {
    "VANILLA": "openai/whisper-large-v3-turbo",
    "DICOW":   "BUT-FIT/DiCoW_v3_2",
    "DIAR":    "BUT-FIT/diarizen-wavlm-large-s80-md",
    "QWEN":    "Qwen/Qwen2.5-3B-Instruct",
}
_ENV_KEYS = {"VANILLA": "MODEL_VANILLA", "DICOW": "MODEL_DICOW",
             "DIAR": "MODEL_DIAR", "QWEN": "MODEL_QWEN"}

def set_global_seed(seed=42):
    """固定 random + numpy + torch + cuda + cudnn。延迟 import 兼容 submit_infer 主进程无 torch。"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # submit_infer 主进程：仅固定 random/numpy（子进程种子才影响推理）

def resolve_model(name):
    """name ∈ REPO_IDS → env override（本地缓存）或 repo id（主办方自动下载）。"""
    return os.environ.get(_ENV_KEYS[name], REPO_IDS[name])

def resolve_df_base_dir(fallback):
    """DF3 例外：不走 HF repo id，env override 或 fallback 本地目录。"""
    return os.environ.get("DF_MODEL_BASE_DIR", fallback)

def reset_peak_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

def peak_gpu_mib():
    """返回峰值显存(MiB)，无 CUDA/无 torch 返回 None。"""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except ImportError:
        pass
    return None
```

### §4.3 模型路径策略

**4 个 HF 模型**：常量/default 改 `resolve_model(...)`。DiCoW/Qwen 的 `trust_remote_code=True` 保留（HF repo 同样布局，无缝）。

**DF3 例外**（C5/D8）：`se_denoise.py` 的 `--model-base-dir` default 改 `resolve_df_base_dir(...)`，env `DF_MODEL_BASE_DIR` 本地 override；原始权重主办方从 GitHub `Rikorose/DeepFilterNet` release 获取（README 说明），不走 HF。

**setenv.sh 新增 env override**（本地开发复用现有缓存免重下）：
```bash
# 模型路径 override（可复现性：代码 default 走 HF repo id，本地用 env 指缓存）
export MODEL_VANILLA="E:/hf_cache/whisper-large-v3-turbo"
export MODEL_DICOW="E:/hf_cache/DiCoW_v3_2"
export MODEL_DIAR="E:/hf_cache/diarizen-wavlm-large-s80-md"
export MODEL_QWEN="E:/hf_cache/Qwen2.5-3B-Instruct"
export DF_MODEL_BASE_DIR="E:/df_cache/DeepFilterNet/Cache/DeepFilterNet3"
```

**argparse default 求值时机**：setenv.sh 在 python 启动前 source，env 已设，故 `default=resolve_model(...)`（import 时求值）能读到。submit_infer **不透传模型路径**给子进程（enroll_infer/se_denoise/llm_reject 各自 resolve），只透传 `--seed`。

### §4.4 种子策略（5 进程各自设 + --seed 透传）

**5 个 `set_global_seed(args.seed)` 调用点**（子进程独立，继承不了父进程 torch 状态）：

| 脚本 | 插入点（parse_args 后）| 备注 |
|---|---|---|
| submit_infer 主进程 | `:187` 后 | 主进程 numpy/random（expand_inputs/bucket_by_atten）；延迟 import torch |
| noise_classify | `:128` 后（parse_args 实测 :128，非 :59——:59 在 calibrate() 内）| 防御性（谱特征无随机）|
| se_denoise | `:59` 后（_patch_df 前）| DF3 随机性小 |
| enroll_infer | `:120` 后（device 前）| 核心推理进程 |
| llm_reject | `:193` 后（no-load 分支前）| do_sample=False 已确定 |

**`--seed` 透传链路**（submit_infer 4 个 `run_*` cmd 各加 `["--seed", str(args.seed)]`）：
- `run_noise_classify:114-117`、`run_se_bucket:120-124`、`run_enroll_infer_pairs:143-147`（+ 旧 `run_enroll_infer:127-133` folder 模式，两入口都加）、`run_llm:157-161`

**enroll_infer:177** augment RNG（硬编码 `np.random.default_rng(0)`）改 `np.random.default_rng(args.seed)`，避免双套种子。

**submit_infer argparse** 加 `--seed`（default 42，`:186` 附近）。

### §4.5 日志 + 显存（满足 FAQ「每条音频置信度/耗时/显存/分支」）

**enroll_infer per-utt**（GPU 主力）：
- 每条循环开始 `reset_peak_gpu()`（`:197` `t0=time.time()` 附近——记每条整条峰值，**不被 dicow langfix retry 的二次 generate 重置**）
- 每条结束后取 `peak = peak_gpu_mib()`
- `:319` print 补 `batch=1` + `peak_mem={peak}MiB`
- `:320-331` results dict 加字段 `"batch_size": 1, "peak_mem_mib": peak`

每条日志字段齐全：`max_sim / batch_size / infer_sec / peak_mem_mib / 拒识分支(verdict)` ✅

**se_denoise**（DF3 CPU 主跑，显存意义小）：`:104` enhance 后仅补结构化进度日志，不强求显存。
**llm_reject**：`:222` reject 后补显存（GPU 跑 Qwen）。
**noise_classify**：纯 CPU 谱特征，无显存。

### §4.6 DiariZen 仓库（git submodule，方案 A）

`code/DiCoW-inference`（96M，含 DiariZen + pyannote-audio）当前是裸 git clone，且 `.gitignore` 的 `code/*/` 通配**完全忽略它**（`git ls-files code/DiCoW-inference` 计数=0，未被主仓跟踪）。enroll_infer:38-41 注入其 sys.path，`from diarizen.pipelines.inference import` 依赖它。

**Plan agent 已确认结构**（2026-07-06 核实）：
- DiCoW-inference remote = `https://github.com/BUTSpeechFIT/DiCoW.git`（去 `ghfast.top/` 代理前缀，主办方 clone 不需代理）
- 三层嵌套 submodule：主仓 → DiCoW-inference → DiariZen（`https://github.com/Lakoc/DiariZen.git`）→ pyannote-audio。三层都需 `git clone --recursive`
- GitHub org = `BUTSpeechFIT`（HF org 是 `BUT-FIT`，勿混）

**实现步骤**：
1. **调 `.gitignore`**：`code/*/` 通配后加 negation `!code/DiCoW-inference/`（否则 submodule add 与 ignore 冲突）
2. 暂存现有 clone：`mv code/DiCoW-inference code/DiCoW-inference.bak`
3. `git submodule add https://github.com/BUTSpeechFIT/DiCoW.git code/DiCoW-inference`
4. `git -C code/DiCoW-inference submodule update --init --recursive`（拉 DiariZen + pyannote）
5. 验证：`from diarizen.pipelines.inference import DiariZenPipeline` 能 import
6. 确认无误删 `.bak`；`git add .gitmodules .gitignore` + commit
7. README 补「`git clone --recursive`」说明

**fallback**（嵌套 .git 冲突 / submodule add 拒绝 / DiariZen 子模块拉取失败）：退方案 C——README 写手动 clone + PYTHONPATH，requirements 补依赖。**不阻塞主改造**（6 项核查硬要求不受影响）。

### §4.7 B 集混合集模式（已天然支持，C2/C9）

**submit_infer 本就是「一个 manifest + 一个 thr 跑所有 pairs」**——B 集只要把 dir1/dir2 合成一个 manifest 跑即可，**无需改 submit_infer 结构**。脚本不读 manifest 的 pos/neg 字段（thr 是 CLI 配置），符合 C9。

**A 集初评**：pos/neg 是两个独立测试集（分别评 CER/RR），分别跑（pos manifest thr=0 / neg manifest thr=0.4）——评测设计允许，脚本内部不依赖 pos/neg 标签。
**B 集入围**：dir1/dir2 合并 manifest，统一 thr。

**统一 thr 选点**（B 集必需）是 follow-up（§9），不在本次改造范围——本次只保证脚本**能跑**混合集（已能），不优化统一 thr 值。

---

## 5. 集成点清单（行号，实现依据）

| 文件 | 改造点 | 行号 |
|---|---|---|
| `enroll_infer.py` | 删 DICOW_MODEL/DIAR_MODEL 常量→resolve_model | `:34-35` |
| | `--vanilla-model` default→resolve_model("VANILLA") | `:118` |
| | set_global_seed 插入 | `:120` 后 |
| | 模型加载（vanilla/DiCoW/DiariZen）用 resolved 路径 | `:127-142` |
| | augment RNG 消费 seed | `:177` |
| | reset_peak_gpu + peak_gpu_mib | generate 块前/`:319-331` |
| | **保留** pyarrow 预热（避 WinError 6714）| `:30` |
| `submit_infer.py` | 加 `--seed` argparse | `:186` 附近 |
| | 主进程 set_global_seed | `:187` 后 |
| | 4 个 run_* cmd 加 `--seed` | `:114-117/120-124/127-133/143-147/157-161` |
| | `--aug-noise-dir` default E:\→None（保底不用增强）| `:176` |
| `se_denoise.py` | `--model-base-dir` default→resolve_df_base_dir | `:54` |
| | set_global_seed 插入 | `:59` 后 |
| `llm_reject.py` | DEFAULT_MODEL→resolve_model("QWEN") | `:37` |
| | set_global_seed 插入 | `:193` 后 |
| `noise_classify.py` | set_global_seed 插入 + `--seed` | `:128` 后（parse_args 实测）|
| `setenv.sh` | 加 5 个 MODEL_* + DF_MODEL_BASE_DIR env | 末尾 |

---

## 6. 错误处理

| 场景 | 处理 |
|---|---|
| 模型下载失败（HF 不可达）| `from_pretrained` 抛异常；日志提示「设 MODEL_* env 指本地缓存 / 检查网络」|
| enroll_infer 单条 diar crash | 现有 try/except 保留，记 error 字段 |
| OOM | 现有 try/except + 显存日志辅助定位 |
| run-twice CER delta 超阈值（>0.01）| verify 报告 + 警告「fp16 残余非确定超阈，建议该段升 fp32」（决策辅助，不阻塞）|
| DiariZen submodule 拉取失败 | README fallback：手动 clone + PYTHONPATH |

---

## 7. 测试计划

### 单测（`tests/test_repro_logic.py`，纯函数，CPU 可跑，先写）
- `set_global_seed` 确定性：同 seed 同输入两次 → torch/numpy 输出一致（CPU 张量/数组）
- `resolve_model` env override：设 MODEL_VANILLA → 返回该值；不设 → 返回 repo id
- `resolve_df_base_dir`：同上
- `peak_gpu_mib`：无 CUDA 返回 None（不崩）
- REPO_IDS 4 个 key 正确

### 集成验证（`code/verify_reproducibility.py`，run-twice）
- `--pairs <pos_pairs> --limit 20 --seed 42`，同配置跑两遍 enroll_infer
- 比对：逐条 transcript 是否一致 + CER delta
- 报告：text 一致率 / max CER delta / 平均 delta
- 阈值：CER delta >0.01 警告（fp16 残余非确定量化）

### 回归验证
- 改造后跑 `pos limit=100 --asr-backend vanilla`，CER 接近 T25 的 0.664（容差 ±0.05）——证明种子/路径改造没破坏转写质量
- 跑 `neg limit=50 thr=0.4`，RR 接近 98.5%（容差 ±2pp）
- dicow 回归：`pos limit=100 --asr-backend dicow`，CER≈1.25（fallback 不坏）

---

## 8. 待确认（TBD，不阻塞本次改造）

主办方过阵子给 CER 计算脚本时明确：
- pos/neg 交一个还是两个 JSON（当前 to_submission 按"一个 result.json 转一个 submission.json"，pos/neg 分别转）
- `label` 确切值（当前 accept/reject）
- `id` 含不含扩展名（当前 cmd_N 去扩展名 + 去 utt 前缀）
- 效率分内存测法（峰值显存 vs 常驻）
- 初评/最终截止具体日期、入围名额

⚠️ **U4 合规风险**（建议向主办方确认一句）：A 集初评 pos/neg 两个测试集是否可分别选 thr？若主办方认为"分别 thr = 用先验"，切统一 thr（pos CER 会因误拒恶化，需重扫）。

---

## 9. 范围外（follow-up，记录不实现）

1. **统一 thr 选点**（C2 衍生）：扫统一 thr 最优化 CER40%+RR40% 加权，供 B 集提交用
2. **batch 推理加速**（C7 允许）：vanilla Whisper 多条 cut_target_timeline 拼批 generate 提速；初赛效率不优先，暂不实现；启用时需额外验 batch↔batch=1 一致
3. **攻 CER**（声纹强化 CAM++/US-PVAD 改善低 sim 桶 timeline 切割 + 数据增广）：另一独立任务线
4. **fp32 强制确定**：仅当 run-twice 显示 fp16 残余超阈时按段升级

---

## 10. 产物

- `code/repro.py`（公共模块）
- `code/verify_reproducibility.py`（run-twice 验证）
- `tests/test_repro_logic.py`（单测）
- 5 脚本改造（submit_infer/enroll_infer/se_denoise/llm_reject/noise_classify）+ setenv.sh
- `code/DiCoW-inference` 转 submodule + `.gitmodules`
- run-twice 验证报告（fp16 残余非确定量化数据，答辩「可复现」佐证）
