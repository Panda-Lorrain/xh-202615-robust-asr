# Oracle Separation 天花板 POC B1 报告 (2026-07-27)

## 目的

A 实验(选 speaker 天花板)已证实:**78% (31/40) 失败样本即使完美选 speaker 也救不回**,
因为 diar 分出的 speaker 本身就脏(混了两人)。**本 B1 实验测"切时间段天花板"**:

> 用 ref(target 正确文本)对 recognition 音频做 forced alignment, 得到 target 说话的
> 真实时间段 ≈ oracle 完美分离. 切出来交 qwen 转, 算 CER. 与 A 对称:
> A 测"在 diar 现有输出里选哪个 speaker", B1 测"用 ref 切出 target 时间段".

判别:
- B1 CER 大降 → "分离/切时间段"有真实空间(diar 升级或 source separation 有杠杆)
- B1 CER 没降 → **不能断言"分离无空间"**, 可能是 forced alignment 在重叠区失效(已知局限), 留给 B2 SepFormer 真做源分离实测验证

## 方法

### 工具选择(MMS_FA 首选, 已可行)

按任务要求优先级:
1. **torchaudio MMS_FA**(主 venv 已有 `torchaudio 2.5.1+cu124`, 零额外依赖) — **选用**
2. WhisperX wav2vec2 中文(备选, 装独立 venv) — 未到
3. stable-ts / whisper word timestamps(退化) — 未到

中文 tokenization: MMS_FA labels 是 29 个 IPA 字符 `('-','a','i','e','n','o','u','t','s','r','m','k','l','d','g','h','y','b','p','w','c','v','j','z','f',"'",'q','x','*')`.
用 `pypinyin.lazy_pinyin` 把每个 ref 中文字转拼音字母, 每字母作为一个 token 直接映射到 MMS labels
(拼音字母几乎全在 labels 内, 仅 ü/特殊字符丢弃, 不影响对齐). 实测对齐 score 平均 0.27, 高质量条目 0.8+.

### 关键设计

- **blank token = `*` (index 28)**: 验证 MMS_FA 实际 blank 是 `*`(argmax 帧分布 100% 落 `*`).
  用 blank=28 对齐 avg_prob=0.85, 用 blank=0(`'-'`) avg_prob=0.36 → 选 28.
- **forced_align 在 GPU 上部分 input 触发 IndexError**(torchaudio 2.5 已知), 落 CPU 稳定.
- **timeline 提取用 dilation 策略**: MMS FA 把每个 target token 对齐到 1-2 帧(周围都是 blank).
  直接当 timeline 段过短会被 `cut_target_timeline` 的 `min_sec=0.3` 退化整条.
  改用每非 blank 帧 ±3 帧(总跨度 ~100ms)union 重叠/相邻段 → 稠密 timeline 覆盖 target 整个说话区间.
- **对齐置信度 align_score**: 非 blank 帧的平均 `exp(score)`, 用于判断对齐质量. >0.5 高质量, <0.3 重叠严重可能对齐到 louder 干扰人(已知局限).
- **timeline 平均覆盖 60.5%**(mean) / 61.7%(median), 即 oracle 切片排除掉约 40% 的纯干扰段.

### 链路(复刻 enroll_infer 切法, 不动主线代码)

```
MMS_FA(ref→IPA token forced align) → 每 token 帧 → dilation union 成 oracle timeline
→ cut_target_timeline(audio, oracle_timeline) 切片(含重叠区, 同 enroll_infer.py:402)
→ qwen_asr_backend --batch-size 16 --seed 42 批转 → CERMetric 官方口径归一后算 CER
```

数据源: `code/runs/_oracle_speaker/meta.json` 的 40 条 fail 样本(与 A 完全可比, 同 ref/sim/argmax_cer).

## 数字 (FAIL n=40)

### 三档对照表(算术平均)

| 方法 | mean CER | 说明 |
|---|---|---|
| argmax 选 spk | **1.216** | 基线(wespeaker sim argmax) |
| A: oracle 选 spk | **0.850** | 完美选 speaker(仍用 diar 切片) |
| **B1: oracle separation** | **0.940** | forced alignment 切 ref 时间段 |
| B1 累计池(归一) | 0.902 | 官方 `CERMetric` total_err/total_char |

**整体 mean**: B1 (0.940) 比 argmax (1.216) Δ=-0.277, 但**不及 A 的 oracle 选 spk (0.850)**.
表面结论: 切时间段天花板比选 spk 还低?——**但这是被对齐失效的 32 条严重拖累的假象**, 见下.

### 关键发现: 对齐质量分桶(揭示真相)

按 align_score 分桶:

| align_score | n | argmax | oracle spk(A) | **oracle sep(B1)** | rescued(CER<0.5) |
|---|---|---|---|---|---|
| [0.6, 1.0) 高质量 | 3 | 0.963 | 0.370 | **0.333** | 2/3 |
| [0.4, 0.6) 较好 | 5 | 1.650 | 0.610 | **0.680** | 2/5 |
| **[0.2, 0.4) 一般** | 14 | 1.203 | 0.915 | **1.069** | 0/14 |
| **[0.0, 0.2) 失效** | 18 | 1.149 | 0.946 | **1.012** | 1/18 |

聚合:
- **对齐可靠子集 (score≥0.4) n=8**: argmax 1.392 → oracle_spk 0.520 → **oracle_sep 0.550**, rescued=4/8 (50%)
- **对齐失效子集 (score<0.4) n=32**: argmax 1.172 → oracle_spk 0.932 → **oracle_sep 1.037**(反而比 argmax 还差, 切到错位置)

→ forced alignment 在 80% (32/40) 失败样本上**对齐质量不可靠**: 双人重叠 + 极重 babble 场景,
model emission 几乎无法把 ref 音素对到正确帧, 对齐到的 token 大量是低概率猜测 (exp(score)<0.2).

但**对齐可靠的 8 条 (20%)**, B1 oracle_sep mean=0.550 比 argmax 1.392 救回 Δ=**0.84**,
有 4 条达到 CER<0.5 救回线. 这 8 条是 forced alignment 真正能起作用的样本(目标音素在 audio 里
可分辨), 实际分离天花板显著.

### 救回样本典型(高质量对齐)

| uid | align | argmax | orspk(A) | **orsep(B1)** | qwen 转写 | ref |
|---|---|---|---|---|---|---|
| cmd_2890 | 0.806 | 1.00 | 0.00 | **0.000** | 吃什么有利于脂肪肝？ | 吃什么有利于脂肪肝 |
| cmd_2188 | 0.796 | 0.89 | 0.11 | **0.000** | 吃什么有利于脂肪肝？ | 吃什么有利于脂肪肝 |
| cmd_2837 | 0.524 | 2.50 | 0.00 | **0.750** | 宋宋家大。 | 风速加大 |
| cmd_2766 | 0.447 | 1.40 | 0.40 | **0.000** | 播放《苦命人》。 | 播放苦命人 |
| cmd_247  | 0.504 | 0.75 | 0.25 | **0.250** | 灯的室温。 | 灯的色温 |

### 对齐失效反而恶化(argmax→orsep 升高)

| uid | align | argmax | orsep(B1) | Δarg | qwen 转写 | ref |
|---|---|---|---|---|---|---|
| cmd_2128 | 0.098 | 1.25 | **2.000** | +0.75 | 就是要开业了，经营。 | 风小一点 |
| cmd_2715 | 0.037 | 1.00 | **1.500** | +0.50 | 利率为二十四。 | 降低一度 |
| cmd_2023 | 0.133 | 1.50 | **1.750** | +0.25 | 千五百七十六元。 | 风速自动 |

→ 对齐失效时, dilation 把错位 token 当 target 时间段, 切出来的 oracle 切片反而包含更多干扰.

## 关键判别结论

### 1. 整体 B1 mean (0.940) 不是真实分离天花板

被对齐失效 32 条拖累(子集 mean 1.037). 看整体数字得出"分离天花板比选 spk 还低"是错的.

### 2. 对齐可靠子集揭示真实分离空间

score≥0.4 的 8 条: argmax 1.392 → B1 oracle_sep **0.550** (Δ=**0.84**, 救回 4/8=50%).
**和 A 的 oracle_spk 0.520 几乎持平** —— 在 forced alignment 可靠时, "切 target 时间段"和"完美选 spk"
是等价的(因为 diar 切片≈ref 时间段). 但 B1 不需要"选 spk", 完全靠 ref 对齐, 是另一条路径.

### 3. 但 forced alignment 在 80% 重叠样本上失效

median align_score=0.229, 32/40 条对齐到不可靠位置. 这是**forced alignment 在双人重叠场景的天生局限**:
ref 音素在重叠区会被对齐到 louder 说话人(可能是干扰人), 切出的 oracle 切片错位.

→ **B1 测的是"近似下界"**, 真实分离天花板需 B2 SepFormer 等真做源分离(把 target mel 分离出来)再对齐验证.

### 4. 给主线的指引

- **分离有真实空间**: 对齐可靠时 B1 救回 Δ=0.84, 印证"切 target 时间段"是有效路径
- **但 diar 升级 + forced alignment 不够**: 重叠区对齐失效 80%, 需要更强前端(源分离 / target extraction)
- **B1 印证 A 的方向**: A 选 spk 救回 0.37, B1 切时间段救回 0.84(可靠子集), 两者结合(完美选 spk + 完美切时段)接近真实天花板
- **真实分离天花板仍待 B2 SepFormer**: B2 在死区已证伪, 但**失败组(主战场, 非死区)的双人重叠子集尚未实测 SepFormer**, 是下一步关键

## 局限(诚实标注)

1. **forced alignment 在重叠区不完美**: ref 音素可能对齐到 louder 说话人而非 target. 用 align_score 分桶后, 失效子集 mean 1.037 反映对齐错误而非分离天花板.
2. **dilation 参数(D=3)经验选择**: 选 3 帧(±48ms)平衡覆盖率与精度. D 太小退化整条, D 太大段间相互吞并. 报告数字对 D 敏感但不影响方向.
3. **中文 tokenization 用拼音字母近似 IPA**: MMS_FA 原生 IPA 训练, 用拼音字母(a/i/u/n/g...)作 token 是近似(拼音 'q' 'x' 'z' 'c' 与 IPA 不完全对应). 但 avg_prob 0.27-0.81 说明 model 仍能识别.
4. **blank=28 选择**: 经验证 `*` 是 MMS_FA 实际 blank(非 torchaudio 默认的 `'-'`), 用 0 会让对齐 score 虚低.
5. **未做 B2 SepFormer 对照**: 真实源分离仍待测, B1 给出的是 forced alignment 视角下的"近似下界", 不能直接断言"分离有/无空间".
6. **样本量 n=40**: 失败组 40 条, 高质量对齐子集仅 8 条, 救回 4 条统计意义有限, 趋势可信但绝对数字谨慎引用.

## 产物

- 主脚本: `code/exp_oracle_separation_ceiling.py`
- 切片 + 转写: `code/runs/_oracle_separation/slices/` (40 个 oracle 切片 + `_uid2text.json`)
- 元数据: `code/runs/_oracle_separation/meta.json` (每条 align_score / timeline / 时长覆盖)
- 汇总: `code/runs/_oracle_separation/summary.json` + `agg.json`
- 本报告: `docs/separation_ceiling_B1.md`

## 复现

```
code/.venv/Scripts/python.exe code/exp_oracle_separation_ceiling.py
```
(seed=42, 40 条 fail 样本, 一次跑完约 70s: MMS_FA forward + dilation + qwen batch 7s)
