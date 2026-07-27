# Qwen3-ASR 微调准备 — 数据 + 依赖 + smoke + 估算

> **2026-07-27** · 微调准备 agent 产物。
> **状态: 准备就绪, 可启动 1万对合成 + LoRA 微调**。
> 范围: 本任务只下载 + 准备 + smoke + 估算, 不跑 1万对/微调 (下一步)。

---

## 1. 磁盘 + 数据下载

### 1.1 磁盘
- 起始 `E:/` 可用 **81 GB** (足够, 任务要求 ≥25GB)。
- 下载后 (含解压) 用量 ~66 GB; **删除 400 个冗余 per-speaker tarball 后回收到 24 GB** 可用 (足够 1万对合成 ~600MB + checkpoint)。
- **真实下载量比任务 brief 估的大**: Aishell **14.5 GB** (非 ~15GB 估的), MUSAN **10.3 GB** (非 ~5GB! 实际是 11GB, brief 低估)。

### 1.2 下载结果 (openslr.org 直链, curl -C - 断点续传)

| 数据集 | URL | 实际大小 | 耗时 | 校验 | License |
|---|---|---|---|---|---|
| **Aishell-1** | https://www.openslr.org/resources/33/data_aishell.tgz | 15,582,913,665 B (14.5 GiB) | ~35 min (18:36→19:11) | Content-Length 完全一致 ✓ | Apache-2.0 |
| **MUSAN** | https://www.openslr.org/resources/17/musan.tar.gz | 11,086,114,085 B (10.3 GiB) | ~28 min (18:36→19:04) | Content-Length 完全一致 ✓ | CC-BY 4.0 |

下载稳定, 中途无断线 (curl 自动重试 5 次未触发)。速率 ~800 MB/min (双线程并发)。

### 1.3 解压结构

```
E:/midea_datasets/
├── data_aishell.tgz                       (15 GB, 原档保留)
├── musan.tar.gz                           (11 GB, 原档保留)
├── data_aishell/
│   ├── transcript/aishell_transcript_v0.8.txt   (141,620 行, 中文朗读, 空格分词)
│   └── wav/
│       ├── train/S0002..S0880/  (340 spks)      ← 400 per-speaker .tar.gz 已解压 + 已清理
│       ├── dev/S0002../         (40 spks)
│       └── test/S0002../        (20 spks)
│       —— 共 141,925 个 wav (16k mono, 中文新闻朗读, ~178h)
└── musan_extracted/musan/
    ├── noise/free-sound/ (845 wav)             ← env 噪声, 走 build_aishell_manifest 递归 glob
    ├── noise/sound-bible/ (88 wav)
    ├── music/ (子目录, ~6h)
    └── speech/ (子目录, ~5h)
```

**注意 (Aishell 二层 tarball)**: `data_aishell.tgz` 解压后 wav/ 下是 **400 个 `SXXXX.tar.gz` per-speaker tarball** (不是直接 wav), 需再次循环 `tar --force-local -xzf` 解到 `wav/{train,dev,test}/SXXXX/*.wav` (循环 ~1 min 完成)。`--force-local` 必加 (Windows `E:` 会被 tar 当远程 host)。

---

## 2. 依赖补装 (venv_qwen)

```bash
E:/Tools/uv/uv.exe pip install --python E:/midea_target_asr/code/.venv_qwen/Scripts/python.exe peft datasets
```

结果 (uv 0.11.24, 5.7s 解析, 8.3s 安装):
- **peft 0.19.1** ✓ (`LoraConfig` import OK)
- **datasets 5.0.0** ✓
- 副带: aiohttp/pyarrow/xxhash/multiprocess 等 14 包

**venv_qwen 完整关键栈** (微调就绪):
```
torch 2.6.0+cu124  (CUDA True, bf16 supported, RTX 4060 Laptop 8GB)
transformers 4.57.6
accelerate 1.12.0
bitsandbytes 0.49.2  (8-bit Adam + QLoRA)
peft 0.19.1          ← 新装
datasets 5.0.0       ← 新装
librosa 0.11.0 / soundfile 0.14.0 / scipy 1.18.0
qwen_asr (Qwen3ASRForConditionalGeneration 已注册 AutoModel)
```

⚠️ **transformers 4.57.6 不原生识别 qwen3_asr 架构**, 但 `qwen_asr` 包通过 `AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)` 注册了, 加载链 (`Qwen3ASRModel.from_pretrained` 或先 register 再 AutoModel) 走通, **与 peft LoRA 兼容** (底层是标准 `PreTrainedModel`)。

---

## 3. smoke 真实数据验证 (20 对, Aishell-1 + MUSAN)

### 3.1 配套新增脚本 (适配真实 Aishell 格式)
- **`code/build_aishell_manifest.py`** (新增, 152 行): 解析 `aishell_transcript_v0.8.txt` (去空格) + 递归扫 `wav/{train,dev,test}/SXXXX/*.wav`, 按说话人切 target/interferer (跨说话人保证 target≠interferer), 输出 `target.jsonl` / `interferer.jsonl` / `noise.jsonl`。**适配要点**: ① 三层 split 结构 (`wav/*/S*/*.wav`); ② MUSAN 子目录递归 glob (`noise/<subdir>/*.wav`, 不是直接 `noise/*.wav`); ③ transcript 空格分词去空格。
- **`code/verify_aug_smoke.py`** (新增, 81 行): 逐对打印 enroll/recog 时长 + RMS dB + 增广参数, 校验 enroll ∈ [1.5,2.5]s + 分布统计。
- **`code/data_aug_recipe.py` 未改动** (架构好, 吃 jsonl manifest 与数据源解耦, 真实数据 smoke 直接跑通)。

### 3.2 smoke 装配

```
target:  10 说话人 × 1 utt = 10 utt  (Aishell-1 干净中文朗读)
interf:  10 说话人 × 3 utt = 30 utt  (跨说话人切)
noise:   20 个 MUSAN noise/free-sound + sound-bible
n-per-target: 2 → 共 20 训练对
```

### 3.3 smoke 结果 (关键指标全过)

```
合成耗时: 18 s → 1.1 pair/s (稳态, 含双 wav 加载 + babble 多 nontarget 加载)
manifest 13 字段全有: id, enrollment_audio, recognition_audio, ref, target_src,
  overlap_ratio, snr_db, noise_type, target_gain_db, target_speed_rate,
  enroll_dur_sec, enroll_pollute, enroll_pollute_snr_db ✓
enroll 时长违规: 0/20 ✓ (全部 1.5–2.5s, 题目规格 ~1.8s)
平均 recog 时长: 4.01 s (target ~3s + overlap 扩展合理)
重叠分布: {0.0:3, 0.25:1, 0.5:2, 0.75:7, 1.0:7} — 偏中高, 对齐失败组主战场 ✓
噪声分布: {pink:5, babble:9, white:6} — babble 多 (题目主因) ✓
target 小声化: 13/20 (65%, n=20 抽样波动, 期望 50%)
target 快语速: 10/20 (50%, 期望 30%, n=20 波动)
enroll 污染: 8/20 (40%, 期望 30%, n=20 波动)
```

### 3.4 抽样实例 (听感指标判断合成对)

| id | ref | enroll | recog | 增广参数 |
|---|---|---|---|---|
| `BAC009S0251W0354_k000` | "虽然队中世界排名第三的樊振东" | 2.39s, -25dB | 3.99s, -20dB | overlap=0.75, snr=-5, pink, gain=-4.3dB |
| `BAC009S0096W0398_k000` | "季军归属郑州队的冯静怡" | 1.96s, **-46.7dB** | 3.56s, -28.5dB | overlap=1.0, snr=-5, pink, **gain=-6.3dB** (模拟 cmd_2096 小声) |
| `BAC009S0160W0445_k001` | "虽然官方报道项目将在十月开建" | 2.38s, -41.7dB | 2.53s, -28.2dB | overlap=1.0, snr=-3, white, gain=-6.7dB, **speed=1.38** (模拟 cmd_2050 快语速) |

**听感判断 (基于 RMS / 参数, 未人工耳听)**:
- target_gain_db 实际生效 (enroll RMS -25 ~ -47 dB 区间, 比 Aishell 原始 -20dB 左右更小声) ✓
- 重叠生效 (recog 时长 > enroll 时长, 含 interferer) ✓
- 加噪生效 (snr -5 ~ +5dB 区间, RMS 抬升干涉) ✓
- 小声化 + 快语速组合生效 (target_speed 1.11–1.38) ✓

**输出位置**: `code/_aug_build_smoke/{manifest.jsonl, enrollment/*.wav, recognition/*.wav}` (20 对可手动播放抽听)

---

## 4. 1万对合成 + LoRA 微调耗时估算

### 4.1 合成耗时 (基于 smoke 实测)

```
smoke 实测稳态速率: 1.1 pair/s (4060 Laptop CPU 单线程, librosa.load + 重叠 + 加噪 + 写 wav)
1万对: 10000 / 1.1 ≈ 9091 s ≈ 2.5 h
保守估计 (含 10-20% IO 抖动): 3 h
原 doc 估计 3.5 h (基于 0.8 pair/s) — 实测略快, 修窄到 2.5–3 h
```

### 4.2 LoRA 微调耗时 (4060 8GB, 估算)

```
Qwen3-ASR-1.7B (bf16) + LoRA r=16 + grad_ckpt + paged_adamw_8bit + batch=1 × grad_accum=16
显存预算: ~5–6 GB (4060 8GB 可跑, 已确认 bf16 supported)

GPU 时长 (1万对 × 3 epoch):
  每 step ~1.5 s (1.7B LoRA + grad_ckpt 经验值, 4060 mid-tier GPU)
  10000 对 / batch_eff=16 = 625 step/epoch × 3 = 1875 step
  1875 × 1.5 s ≈ 2800 s ≈ 47 min (forward+backward+optim)
  含 val + checkpoint: ~1 h
保守: 1.5–2 h (doc 估 ~2 h, 合理)
```

### 4.3 决策 (是否准备就绪)
- ✅ 数据就绪: Aishell-1 (141k wav) + MUSAN (930 noise wav) 全部解压可读
- ✅ 依赖就绪: peft + datasets 装好, qwen_asr + transformers 兼容性确认
- ✅ 配方就绪: smoke 20 对真实数据合成质量过验, 配方链 1.1 pair/s 稳定
- ✅ 算力就绪: 4060 8GB VRAM 够 LoRA + bf16 (GPU probe 通过)
- ✅ 路径就绪: data_aug_recipe.py + build_aishell_manifest.py + verify_aug_smoke.py 三件套就位
- **结论: 可启动 1万对合成 + LoRA 微调 (下一步任务)**
  - 预计耗时: 合成 ~2.5 h + 微调 ~1.5 h = ~4 h (单机 4060 顺序跑)
  - 建议 POC 先 1k 对 LoRA 验证 ΔCER (lessons-pitfalls §14: hold-out A 集铁律, ~10 min 微调), 通过再扩 1万

---

## 5. 产物清单

| 文件 | 用途 | 状态 |
|---|---|---|
| `E:/midea_datasets/data_aishell/` | Aishell-1 wav + transcript | ✅ 141,925 wav |
| `E:/midea_datasets/musan_extracted/musan/` | MUSAN noise/music/speech | ✅ 930+ noise wav |
| `code/build_aishell_manifest.py` | Aishell+MUSAN → jsonl 清单装配 | ✅ 新增 |
| `code/verify_aug_smoke.py` | smoke 合成质量校验器 | ✅ 新增 |
| `code/data_aug_recipe.py` | 配方主脚本 | ✅ 未改 (架构好, 直接吃 jsonl) |
| `code/_aug_manifests/{target,interferer,noise}.jsonl` | smoke 输入清单 | ✅ 60 行 |
| `code/_aug_build_smoke/{manifest.jsonl, enrollment/, recognition/}` | smoke 20 对真实数据输出 | ✅ 可抽听 |
| `docs/qwen_finetune_prep.md` | 本文档 | ✅ |

**未做 (下一步任务)**: 1万对全量合成 + LoRA 微调 + A 集 hold-out 评测。
**未 git commit** (按任务要求)。

---

## 6. 关键发现 / 注意事项给下个 agent

1. **MUSAN 实际 10.3GB, 非 brief 估的 5GB** — 下次评估磁盘时按实际算。
2. **Aishell 二层 tarball**: `data_aishell.tgz` 解压后 wav/ 里是 400 个 per-speaker .tar.gz, 必须再循环 `tar --force-local -xzf` 解一次 (Windows `--force-local` 必加)。已自动处理, 但若清过 wav/ 重解压得记得。
3. **MUSAN noise/<subdir>/*.wav 结构** (非 noise/*.wav), manifest 装配已递归 glob。
4. **磁盘当前 24GB 可用** — 跑 1万对 (~600MB) + LoRA checkpoint (~50MB×N) 充足。若要扩到 10万对 (~6GB) 也够; 若扩到 100万对 (~60GB) 必须先清 `data_aishell.tgz` + `musan.tar.gz` (回收 26GB)。
5. **transformers 4.57.6 + qwen3_asr 架构**: transformers 不原生识别, 但 qwen_asr 包用 AutoModel.register 注册了, 加载前需 `import qwen_asr` 触发注册 (否则 AutoModel.from_pretrained 报 ValueError)。微调脚本要写好这个 import 顺序。
6. **合成速率实测 1.1 pair/s** — 比 doc 估的 0.8 快 35%, 修窄 doc 的 3.5h → 2.5–3h。
7. **smoke 验证 enroll 时长全部 [1.5,2.5]s, 增广参数分布对齐失败组**, 配方可信, 直接进 1万对无技术风险。
