# W3/W4 enrollment→target 锁定 + fork patch 固化 + 阶段盘点 设计

> 日期：2026-06-28
> 状态：已批准（用户 brainstorming 后指令"执行"）
> 关联：PROGRESS.md T14–T16、CLAUDE.md 组合主线、RESULTS.md

---

## 1. 背景与颠覆性决策

原 CLAUDE.md 把 W3/W4 列为"Personal VAD 产生 STNO + CAM++ 声纹"。2026-06-28 brainstorming 探索（两路 agent 核实）发现该设想需重新定义：

1. **Personal VAD 无可用开源实现**：SpeechBrain 无 `personal_vad` recipe（只有标准 VAD `vad-crdnn-libriparty`），Google 原论文（arXiv:1908.04284）无官方代码/权重，社区实现（pirxus/personalVAD）不可靠。自研需训练 speaker-conditioned VAD，远超比赛预算。
2. **CAM++ 重接是沉没成本**：`pyannote/wespeaker-voxceleb-resnet34-LM`（256d，6.6M 参数，26MB）已在 DiariZen 内接通并端到端跑通（T14 `PIPELINE_EXIT=0`）。换 CAM++（192d，Apache-2.0，原生中英双语）需重接 DiariZen 整个 embedding 流程，wespeaker 成果推倒重来。CAM++ 留作 wespeaker 在真实中文测试集劣化时的二阶段备选。
3. **当前 pipeline 真正缺口**：`pipeline.py` 的 `preprocess` 拿到 DiariZen diarization 的 N 个 speaker 后，**每个 speaker 轮流当 target 各转一遍**（`get_stno_mask(mask, i)` for i in range(N)），没有"enrollment→声纹匹配→锁定唯一 target"。T14 说的"Speaker 3=主说话人"是事后人工挑选，非模型按 enrollment 选定。

**决策**：W3/W4 重新定义为 **enrollment→wespeaker 声纹匹配→锁定唯一 target speaker**，复用已跑通的 wespeaker + DiariZen diarization，填补组合主线"目标说话人 ASR"的真正缺口。答辩价值：从"全 speaker 转、人工挑 target"升级为"enrollment 自动锁定唯一 target + 兜底拒识"。

> 注：用户开场提"多通道测试"，但测试集通道数仍未确认（CLAUDE.md 标注当务之急），DSENet 空间路线暂不可行。本轮推进的"目标说话人"核心是**单通道下 enrollment→唯一 target 锁定**，与多通道正交。

---

## 2. STNO 数据结构（实现的事实基础，来自 pipeline.py 源码核实）

- **diarization_mask**：`get_diarization_mask` 产出 `[N_speakers, T@50Hz]` float32 二值 mask（50Hz=20ms/帧，由 whisper 100Hz log-mel 帧数 //2 得到）。
- **stno_mask**：`get_stno_mask(diar_mask, s_index)` 产出 `[4, T@50Hz]` float32，4 行每帧 one-hot：
  - `[0]` sil_frames：所有 speaker 都不说话
  - `[1]` target_spk：仅 target(s_index) 说话
  - `[2]` non_target_spk：仅非 target 说话
  - `[3]` overlapping_speech：target 与他人重叠
- **喂给 DiCoW**：`samples['stno_mask']` stack 成 `[N, 4, T]`，`input_features` 沿 batch repeat N 份，DiCoW `generate` 消费 `stno_mask` kwarg（trust_remote_code）。

---

## 3. Part 1 — enrollment→wespeaker 锁定唯一 target

### 方案选型
改造 `pipeline.py` 的 `DiCoWPipeline`（方案 A），加 enrollment 匹配；无 enrollment 时保持原"全 speaker 转"行为，**向后兼容**（现有 T14 复现命令不受影响）。弃独立脚本（B，逻辑重复）与改 DiariZen（C，侵入太深）。

### 数据流（改动集中在 `preprocess`，line 72–144）
```
__call__(inputs, enrollment=None, ...)
  ↓
① DiariZen diarization → N 个 speaker 时间段           [已有，不改]
  ↓
② if enrollment is not None:
     切每个 speaker 语音(按时间段) + enrollment → wespeaker 抽 embedding
     enrollment_emb vs 各 speaker_emb 余弦相似度 → argmax = target_idx
     if max_sim < reject_threshold: return 拒识(空文本)   [兜底拒识]
     只对 target_idx 跑 get_stno_mask → stno_mask [4,T]    [只转 target]
     input_features 不 repeat (batch=1)
   else:
     原行为: 全 speaker 各转                              [向后兼容]
```

### 关键改动点
- `pipeline.py`
  - `DiCoWPipeline.__init__` / `__call__` / `preprocess`：加 `enrollment`、`reject_threshold` 参数透传
  - 新方法 `get_wespeaker_embedding(audio)`：独立轻量加载 wespeaker（pyannote `Model.from_pretrained(wespeaker_bin)`，权重已在 `E:\hf_cache`），输入音频/片段 → 归一化 embedding
  - 新方法 `select_target_by_enrollment(...)`：切 speaker 片段 → 抽 emb → 余弦匹配 → 返回 `(target_idx, max_sim)` 或 `None`（拒识）
  - `preprocess`：enrollment 分支只产 1 份 stno_mask + 不 repeat input_features
- `inference.py`
  - 加 `--enrollment <wav>`（单 enrollment 用于文件夹内所有识别音频，符合比赛"同一目标说话人"场景）
  - 加 `--reject-threshold`（默认 0.5，可调；留扫阈值实验位）
  - 输出标注是否拒识（拒识时写空 + 标记，便于 W6 eval_metrics 统计）

### wespeaker 复用方式
**独立轻量加载**（不碰 DiariZen 内部 API）：用 pyannote.audio 的 wespeaker model 对象，对 enrollment 与各 speaker 切片分别 forward 抽 embedding。权重 `E:\hf_cache\hub\models--pyannote--wespeaker-voxceleb-resnet34-LM\snapshots\<hash>\pytorch_model.bin` 已在 cache，零下载。归一化后余弦相似度。

### 兜底拒识（对接拒识 40% 评分）
- 相似度阈值 `reject_threshold` 默认 0.5，`--reject-threshold` 可调
- max_sim < θ → target 不在场 → 输出空文本 + 拒识标记
- 留"扫阈值"实验位（脚本扫 θ∈[0.3,0.7] 看拒识率/CER 权衡），为答辩备料

### 验证计划
- **自匹配 sanity**：用同一段音频既当 enrollment 又当识别 → target_idx 应稳定锁定、相似度≈1
- **target vs non-target**：T15 的 zh_target 音频 + 另一段人声当 non-target → 锁定 target 正确转写
- **兜底拒识**：enrollment 与识别音频完全不同人 → max_sim 低 → 拒识空输出
- **向后兼容**：不带 `--enrollment` → 原 T14 行为不变
- 复用 W6 `eval_metrics.py` 算 CER/拒识率

---

## 4. Part 2 — fork patch 固化

5 个 patch 现状（grep 逐个核实）：源码级 patch 1/2/3 **真实存在**于 fork 代码，可直接导出 diff；patch 4 是依赖声明（非源码）；patch 5 是 wespeaker 权重（已在 cache）。

### 产出 `code/DiCoW-inference/repro/`
- `patches/01-pyannote-namespace.patch`（`pyannote/__init__.py`：declare_namespace→pkgutil.extend_path）
- `patches/02-pyannote-version.patch`（`pyannote/audio/__init__.py`：`__version__="0.0.0-fork-patch"`）
- `patches/03-diarizen-from_pretrained.patch`（`diarizen/pipelines/inference.py`：本地路径分支 + wespeaker `local_files_only`）
- `requirements-fork.txt`（toml/setuptools<81/einops/semver/matplotlib/pytorch-metric-learning + transformers 4.42.4/numpy<2/hf-hub<1.0）
- `apply_patches.sh`（幂等：检测已应用则跳过 → 应用 3 个 patch → 校验 wespeaker 权重在 cache）
- `REPRODUCE.md`（clone DiCoW-inference → uv 建环境装 requirements → apply_patches → setenv → 跑 T14 验证命令）

### 校验
apply_patches.sh 跑完后 `git -C DiCoW-inference diff` 应为空（patch 已在）或可重新干净应用；能用 T14 命令重新跑通 EN2002a。

---

## 5. Part 3 — 项目阶段盘点文档（新建 `项目阶段盘点.md`）

PROGRESS.md 保持 append 日志不动，新建独立结构化总览。结构：
1. **阶段总览表**：W1–W7（目标/产出/状态一眼看全）
2. **每阶段详述**：在做什么、产出物（文件+结论）、遗留、下一步
3. **里程碑时间线**：T0–T16（来自 PROGRESS.md 日志）
4. **当前能力边界**：能做什么 / 不能做什么 / 待确认项
5. **下一步优先级排序**

---

## 6. 验收标准
- Part 1：enrollment 锁定 + 兜底拒识 + 向后兼容 三场景均有实验产出（写 RESULTS.md），复用 eval_metrics 算指标
- Part 2：`apply_patches.sh` 幂等可用 + REPRODUCE.md 照做能跑通
- Part 3：`项目阶段盘点.md` 覆盖 W1–W7 全阶段，与 PROGRESS.md/RESULTS.md 一致

## 7. 风险
- wespeaker 独立加载 API 若与 pyannote 当前版本不符 → fallback 复用 DiariZen 内部 embedding model（实现时探查决定）
- 显存 8GB：wespeaker(26MB)+DiCoW(2.13GB)+DiariZen 同时驻留需监控；enrollment 抽 emb 可在 CPU 跑
- 拒识阈值 0.5 是经验初值，真实数据来前仅作 sanity，扫阈值实验定标
