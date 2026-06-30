# 推理脚本标准化(submit_infer.py) + 交付文档骨架 — 设计

- **日期**: 2026-07-01
- **状态**: 已批准(用户 2026-07-01 设计评审后"执行")
- **范围**: P0 标准化推理脚本 + P1 交付文档骨架(对照 2026-06-30 战略转向「研究→交付物标准化」)
- **依赖**: 不依赖测试集 A(A 未到);不依赖通道数确认
- **关联**: `AGENT_HANDOFF.md` T19 待办①-③、`PROGRESS.md` 2026-06-30 战略转向、`README.md` 官方交付物清单

---

## 1. 目标

- **P0** — `code/submit_infer.py`:标准化**纯推理**脚本,吃 enrollment + recognition → `result.json` + `timing.json`。填两个空白:① 推理脚本这个交付物当前不存在(现仅有实验脚手架 `fuse_eval.py`/`enroll_infer.py`);② 端到端耗时空缺(只测过 minimal RTF=0.058,完整 pipeline 端到端 RTF 未知,而效率占评分 20%)。
- **P1** — `交付/` 三份文档骨架 + README 修正:把 00-03 文档料 + T14-T20 实验转成比赛交付格式,并修正 README"理想态架构 ≠ 实际实现"的穿帮风险。

## 2. 背景与约束

- **官方交付物**:模型权重 + 推理脚本 + json 测试结果 + 运行耗时 + 技术设计/测试验证方案 + 使用说明。评分 CER40% / RR40% / 效率20%(L20 GPU)。提交截止 2026-09-05。
- **现有已验证组件**(零侵入复用):
  | 组件 | venv | 接口要点 |
  |---|---|---|
  | `enroll_infer.py` | `code/.venv` | `--enrollment --recognition-folder` 批量,**一次加载模型处理整批**;输出 list[{recognition, max_sim, stno_target_ratio, transcript, rejected, rtf, ...}] |
  | `noise_classify.py` | `code/.venv` | 纯 librosa;`classify(flatness,lhr)→(noise_type,atten_lim_db)`;babble/white→0, pink→6 |
  | `se_denoise.py` | `code/.venv_se` | DeepFilterNet3;`--in-dir --out-dir --atten-lim-db`(⚠️ **目录统一 atten-lim,不支持逐条**) |
  | `llm_reject.py` | `.venv_llm` | `--infer-json [{file,text}] --out-json verdicts`;输出每条 accept/reject |
- **多 venv 不可合并**:`code/.venv` / `code/.venv_se` / `.venv_llm` 因依赖冲突(torch/transformers 版本、df、vllm)分立,单进程 import 全部不可行——这是当初分 venv 的原因。
- **诚实性能约束**:当前仿真绝对性能差(se6 正确率 15.1%)。P0 产出的是**可执行标准化交付物**,不是性能突破;性能提升待 A 集 + babble 工程兜底(单独的 P2,不在本 spec 范围)。

## 3. P0 架构决策

三方案对比:

| 方案 | 做法 | 裁决 |
|---|---|---|
| **A. subprocess 顶层编排器** | `submit_infer.py` 仅 stdlib,通过 `subprocess` 调各 venv 现成组件 | ✅ **采用** |
| B. 单进程 import 全部 | 一个 venv 里 import 所有组件 | ❌ 依赖冲突不可行(见上) |
| C. 重构一个大 Pipeline 类 | 把 enroll+SE+LLM 重写成类 | ❌ 丢弃已验证组件、风险高、违反 YAGNI |

**A 胜出关键**:① 零侵入现有组件;② 天然 venv 隔离;③ `enroll_infer.py` 支持 `--recognition-folder` 批量、一次加载模型处理整批,编排器只需"每阶段一次 subprocess"(非逐条),模型加载开销分摊到全批,不会慢。

## 4. P0 数据流

```
输入: --enrollment <wav> --recognition-folder <dir>   (一对多)
   或  --pairs <manifest.json> [{enrollment,recognition}] (一对一;A 集真形态未知,--pairs 最通用)
        │
        ▼
[阶段0 可选, --no-se 跳过] noise_classify 估每条噪声类型 → est_noise + atten_lim_db
        │
        ▼
[阶段1 可选, --no-se 跳过] SE 条件化降噪: 按 atten_lim_db 分桶(0 桶 / 6 桶)
   每桶一次 se_denoise subprocess → 临时目录;合并为"降噪后识别集"
        │
        ▼
[阶段2 核心] enroll_infer 转写(一次 code/.venv subprocess, 批量):
   enroll声纹锁定 → diar → DiCoW generate → 每条 {transcript, max_sim, stno_target_ratio, rtf}
   带 --always-generate(拒识条也转出文本,供 LLM/融合二次判断)
        │
        ▼
[阶段3 可选, --no-llm 跳过] llm_reject(.venv_llm subprocess): 每条 transcript → accept/reject
        │
        ▼
[阶段4 融合决策] rejected 按策略判定(见 §7)
        │
        ▼
输出: <out-dir>/result.json  +  <out-dir>/timing.json
```

## 5. P0 接口(submit_infer.py)

仅依赖 stdlib(`subprocess`/`json`/`argparse`/`tempfile`/`shutil`/`time`/`os`/`glob`),任意 python 可跑(它只编排)。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enrollment` | — | 目标说话人参考 wav(与 `--pairs` 二选一) |
| `--recognition-folder` | — | 识别 wav 目录(与 `--enrollment` 配对,一对多) |
| `--pairs` | — | `[{enrollment,recognition}]` manifest(一对一,最通用) |
| `--no-se` | False | 跳过 SE 条件化降噪 |
| `--no-llm` | False | 跳过 LLM 拒识(退化纯声纹+STNO) |
| `--sim-thr` | 0.2 | 声纹拒识阈值(T20 最优工作点) |
| `--strategy` | `llm_or_sim` | 融合策略:`llm_or_sim` / `sim_only` / `llm_only` |
| `--device` | `cuda:0` | 透传给 enroll_infer/llm |
| `--out-dir` | `code/submit_out` | 输出目录 |
| `--limit` | 0 | 只处理前 N 条(快测/本机低显存,0=全部) |
| `--work-dir` | `<out-dir>/_work` | 中间产物(SE 降噪后音频、各阶段 json) |

## 6. P0 SE 条件化逐条 atten-lim(gap 解法)

**gap**:`se_denoise.py` 只支持整目录统一 `--atten-lim-db`,而 SE 条件化要逐条不同(babble/white→0, pink→6)。

**解法(分桶,不改 se_denoise.py)**:
1. 阶段0 跑 `noise_classify` 估每条 `atten_lim_db`(0 或 6)
2. 按 `atten_lim_db` 把识别音频软链接/复制到两个临时桶目录:`_work/se_in_0/`、`_work/se_in_6/`
3. 每桶一次 `se_denoise --in-dir <桶> --out-dir <桶_out> --atten-lim-db <桶值>`
4. 把两桶输出合并回统一 `--recognition-folder` 视图,喂阶段2

`--no-se` 时直接用原始识别音频进阶段2。

> 注:SE 只作用于 recognition 音频;enrollment 参考音频为干净锚点,**不做 SE**。

## 7. P0 拒识融合决策

| strategy | rejected 判定 | 适用 |
|---|---|---|
| `llm_or_sim`(默认) | `(llm ≠ accept) AND (max_sim < sim_thr)` | LLM 救回 sim 误拒(fuse_eval T19 验证最优) |
| `sim_only` | `max_sim < sim_thr` | 无 LLM 退化 |
| `llm_only` | `llm ≠ accept` | 纯语义 |

- `--no-llm` 时强制 `sim_only`。
- `sim_thr` 默认 0.2(T20 最优工作点;fuse_eval 显示 0.1-0.5 可调)。
- `enroll_infer` 必须带 `--always-generate`,保证拒识条也有 transcript 供 LLM/融合二次判断(否则 LLM 无输入)。
- `score` 字段:当前 = `max_sim`;三路融合扩展时改为融合分(本 spec 不实现三路,留接口)。

## 8. P0 输出 schema

**`result.json`**:
```jsonc
{
  "task_id": "XH-202615",
  "generated_at": "<ISO8601>",
  "config": {"se": true, "llm": true, "strategy": "llm_or_sim", "sim_thr": 0.2, "device": "cuda:0"},
  "n_utt": 450,
  "results": [
    {"utt_id": "sample_001",            // = recognition 文件名去扩展名
     "enrollment": "enr_001.wav",
     "recognition": "rec_001.wav",
     "text": "请把客厅的空调温度调到二十六度",  // rejected=true 时为 ""
     "rejected": false,
     "score": 0.34,                     // 融合置信度(当前=max_sim)
     "max_sim": 0.34, "llm_verdict": "accept",
     "noise_type": "white", "atten_lim_db": 0,  // SE 关时为 null
     "diar_fail": false}
  ]
}
```

**`timing.json`**:
```jsonc
{
  "device": "NVIDIA RTX 4060 Laptop (8GB)",   // 透传,L20 上写 L20
  "n_utt": 450,
  "total_audio_sec": 1234.5,
  "total_wall_sec": 320.1,
  "overall_rtf": 0.259,                        // = total_wall/total_audio,端到端真实 RTF
  "phases": {"noise_classify": {"wall_sec": 5.1},
             "se": {"wall_sec": 50.2, "n": 450},
             "enroll_diar_dicow": {"wall_sec": 220.4, "mean_rtf": 0.18},
             "llm": {"wall_sec": 49.5}},
  "per_utt": [{"utt_id": "sample_001", "audio_sec": 2.7, "wall_sec": 0.8, "rtf": 0.30}]
}
```

## 9. P0 timing 测量

- 每个 subprocess 前后用 `time.perf_counter()` 包,记入 `phases`。
- `total_audio_sec` = 所有 recognition wav 时长之和(librosa.get_duration 或读 manifest)。
- `overall_rtf` = `total_wall_sec / total_audio_sec`(端到端,含 SE+enroll+LLM 全链路,区别于 enroll_infer 的纯转写 rtf)。
- `per_utt.rtf` 取自 enroll_infer 输出(纯转写 RTF);`per_utt.wall_sec` 按阶段分摊估算。

## 10. P1 交付文档

**诚实修正点**:README 当前"技术架构"是理想态(CAM++/Personal VAD/SE-DiCoW),实际实现是 wespeaker+enroll+DiCoW+SE条件化+LLM 融合(CAM++ T18 已证伪,PVAD 无开源实现)。报告写**实际实现**。

| 新建文档 | 内容来源 | 完成度 |
|---|---|---|
| `交付/设计报告.md` | 00 总纲 + 01 模块细节,改写成**实际实现版**;含实际数据流图、模块表(实际栈)、设计选择与权衡、差异化策略 | 骨架 + 关键章填实际;不确定处 `<!--TODO: ...-->` 标注 |
| `交付/使用说明.md` | submit_infer.py 用法 + 环境搭建(3 venv)+ 权重清单 + 输入输出格式 + 常见报错(已知坑) | submit_infer 定稿后填实 |
| `交付/测试验证方案.md` | eval_metrics.py 评测方法 + RESULTS T14-T20 仿真结果(诚实标注"仿真,待 A 集")+ A 集到手评测流程 | 现在即可填大半 |

README 同步:架构图理想态→实际;技术栈表删 CAM++/PVAD 改 wespeaker;`交付/` 文档链接;`submit_infer.py` 补进"快速开始"。

## 11. 文件清单

- **新建**:`code/submit_infer.py`、`交付/设计报告.md`、`交付/使用说明.md`、`交付/测试验证方案.md`
- **不改**:`enroll_infer.py`/`se_denoise.py`/`noise_classify.py`/`llm_reject.py`(编排器零侵入)
- **微调**:`README.md`(架构图/栈表/快速开始)

## 12. 不做(YAGNI)

- ❌ 显存自适应(L20 48GB 大 batch 自动调)→ 单独 P3,P0 仅 `--device` 手动
- ❌ 不重构任何已验证组件
- ❌ submit_infer 不内置评测(纯推理;评测用 `eval_metrics.py`,A 集到手后)
- ❌ 三路融合(stno 信号)→ 留接口(`score` 字段),本 spec 不实现

## 13. 验收标准

**P0**:
1. `submit_infer.py` 在 `test_wav/dataset/final/` 上 `--limit 5` 跑通(PIPELINE_EXIT=0)
2. 产出 `result.json` + `timing.json`,schema 与 §8 一致
3. `--no-se` / `--no-llm` 开关各自生效(产出的 config 字段反映)
4. `timing.json` 的 `overall_rtf` 有合理值(本机 4060 预期 < 1)
5. `--pairs` 模式可跑(单 enrollment 单 recognition)

**P1**:
1. `交付/` 三份文档建立,结构完整,关键章填实际内容
2. README 架构图/栈表修正为实际实现(无 CAM++/PVAD 穿帮)
3. 不确定处统一 `<!--TODO-->` 标注,不混入确定内容

## 14. 风险与缓解

- **风险**:A 集输入形态未知(一对多 vs 一对二)→ `--pairs` 最通用,覆盖两种;A 到手再定默认。
- **风险**:本机 4060 8GB 跑全量(SE+enroll+LLM)显存紧→ `--limit`/`--no-llm` 可降级;L20 上全开。
- **风险**:SE 条件化分桶产生中间音频占盘→ `--work-dir` 可控,跑完可清理(本 spec 默认保留供调试)。
