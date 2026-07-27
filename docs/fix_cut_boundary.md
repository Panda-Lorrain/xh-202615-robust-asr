# cut_target_timeline 边界"度"丢失诊断 + POC 修复验证报告

> 日期: 2026-07-27 | 任务: cmd_2098 切片"度"丢失诊断 + POC padding 修复验证
> POC 脚本: `code/exp_fix_cut_boundary.py` | 产物: `code/runs/_fix_cut_boundary/`
> 主线未改(`text_utils.cut_target_timeline` 只读); 不 git commit

## 1. 背景(用户报告)
用户听 cmd_2098 验收包发现: target 说"调到二十八度", 但切片
`code/runs/_verify_deadzone_cmd_2098/假如选spk1_当target.wav` 听感只含"调到二十八", 末尾"度"似乎被切掉。

## 2. 根因诊断(推翻用户原假设)

### 2.1 切片"度"实际未被切掉(qwen 已能恢复)
- **MD5 一致**: 重新生成的 `spk1_v1.wav` 与用户听的 `假如选spk1_当target.wav` hash 完全相同(`26887bdc...`), diar 可复现。
- **qwen 在 spk1_v1 上转写 = `"调到二十八度。"` CER=0**, 含完整"度"。
- 切片时长 760ms `[0.413, 1.172]s`, audio 全长 1.730s, **末段距 audio 末 558ms**。

### 2.2 真因: diar 边界 + 用户听觉感知(非 cut_target_timeline bug)
| speaker | diar 时间段 | 末段距 audio_end | 说明 |
|---|---|---|---|
| spk0 (argmax 错选) | `[0.593, 1.772]s` | -0.042s(超 audio 末) | diar 把"度"尾音划给了 spk0 |
| spk1 (实际 target) | `[0.413, 1.172]s` | **+0.558s** | spk1 末段被 diar 切短 558ms |

- "调到二十八度" 整句约 [0.4-1.5s], 跨越 spk1 末段 + spk0 重叠区。
- spk1 切片 `[0.413, 1.172]` 已含"度"主音(~220ms), 仅末尾 ~30-80ms 尾音落在 spk0 区。
- **听觉感知差异**: 人耳听完整"度"需 ~300ms 完整尾音, 切片只剩 ~220ms → 听感"度被切"; qwen 用 mel 上下文补全无碍。

### 2.3 已排除其他可能
- **(b) cut_target_timeline 内部截断**: 无。`cut_target_timeline` 与 raw diar timeline 拼接产物完全同 hash。
- **(d) collect_clean_audio 独占帧丢**: 无。那是抽声纹用, 不影响转写切片(走 `cut_target_timeline` 而非 `collect_clean_audio`)。
- **(c) padding 不足**: 部分成立(末段缺 30-80ms 尾音), 但 ASR 不需要即可恢复; padding 仅改听觉。

### 2.4 cmd_2098 CER=1.0 的真因(独立于"度"边界)
- argmax sim: spk0=0.0195 > spk1=-0.0102 → **argmax 选 spk0(错), spk1 是真 target**。
- spk0 切片转写 `"请你把温"` CER=1.0 — 锁错人。
- 这是 `non-voiceprint-target-selection` 问题(memory 已记), 非 cut 边界问题。

## 3. POC 修复: cut_target_timeline_v2(段尾 padding)

### 3.1 实现(`code/exp_fix_cut_boundary.py:60-90`)
```python
def cut_target_timeline_v2(audio, per_spk_timeline, sr=16000, min_sec=0.3, pad_sec=0.08):
    """每段尾向后扩 pad_sec, 夹到下段 start 或 audio 末; 首段额外前扩 pad_sec。"""
    segs = sorted((float(s), float(e)) for s, e in per_spk_timeline)
    n = len(audio)
    if not segs: return np.asarray(audio)
    clips = []
    for i, (s, e) in enumerate(segs):
        # 末段: 夹到 audio 末; 中间段: 夹到下段 start(覆盖 gap 不重叠)
        e_pad = min(e + pad_sec, n/sr) if i+1 == len(segs) else min(e + pad_sec, segs[i+1][0])
        # 首段额外前扩 pad_sec(捕首字); 其他段不前扩(避免与上段尾重叠)
        s_pad = max(0.0, s - pad_sec) if i == 0 else s
        clips.append(audio[int(s_pad*sr):int(e_pad*sr)])
    out = np.concatenate(clips)
    return out if len(out) >= sr*min_sec else np.asarray(audio)
```

### 3.2 cmd_2098 验证(`code/runs/_fix_cut_boundary/cmd_2098_qwen_compare.json`)
| pad_sec | spk1 切片时长 | qwen 文本 | CER vs ref | 含"度" |
|---|---|---|---|---|
| 0(v1) | 760ms | `调到二十八度。` | **0.000** | ✓ |
| 0.08 | 920ms | `调到二十八度。` | 0.000 | ✓ |
| 0.20 | 1080ms | `调到十八度。` | 0.167 | ✓(误听) |
| 0.40 | 1240ms | `调到二十八度。` | 0.000 | ✓ |
| 0.56 | 1400ms | `调到二十八度。` | 0.000 | ✓ |

**结论**: spk1(真 target) 各 pad 都含"度", v1 已 CER=0 — padding 对 cmd_2098 ASR 无收益(因 qwen 已恢复)。
spk0(argmax 错选) pad=0.56s 时变 `调到十八度` CER=0.167(因 spk0 段含真 target 的"度"音, 大 pad 跨越重叠区抓到)。

## 4. 30 条抽样 CER 对比(不退化验证)

### 4.1 pad_sweep(3 桶×10 = 30 条, seed=42)
| pad_sec | mean CER | Δ vs v1 | 判定 |
|---|---|---|---|
| 0(v1 baseline) | 0.2567 | — | — |
| 0.03 | 0.2972 | **+0.0406** | FAIL |
| 0.05 | 0.2721 | +0.0155 | FAIL |
| **0.08(默认)** | **0.2256** | **-0.0310** | PASS |
| 0.12 | 0.2616 | +0.0049 | PASS 边界 |
| 0.20 | 0.1868 | -0.0699 | PASS |

### 4.2 离群剔除(诚实性检验)
**cmd_2001 在 pad=0.08 时 CER 1.2→0.0(Δ=-1.2), 是离群主因**。剔除后:
| pad_sec | Δmean(剔除 cmd_2001) | 判定 |
|---|---|---|
| 0.03 | +0.0557 | FAIL |
| 0.05 | +0.0160 | FAIL |
| 0.08 | **+0.0093** | **FAIL 轻微退化** |
| 0.12 | +0.0189 | FAIL |
| 0.20 | -0.0309 | PASS 边界(噪声内) |

**剔除离群后, pad=0.08 实际轻微退化 +0.009**; 仅 pad=0.20 仍有 -0.03 改善但属噪声内。

### 4.3 v1 baseline 复核
30 条中 **29/30 v1 CER 与原 `poc_qwen_asr_full_result.json` 完全一致**(仅 cmd_2355 漂移 1.333 vs 1.5), 证明 v1 复刻忠实。

## 5. 结论与建议

### 5.1 主结论
1. **"度被切"是误诊**: qwen 在 byte-identical spk1 切片上 CER=0 已含"度"。人耳听感不全因 558ms 尾音被 diar 划给 spk0 重叠区。
2. **真正瓶颈是 argmax 选错 target**(spk0 vs 真 target spk1), 非 cut 边界。
3. **padding 不可靠**: pad_sweep 非单调(30ms 退化/80ms 改善/120ms 退化/200ms 改善), 剔除 cmd_2001 离群后 pad=0.08 实际 +0.009 轻微退化; 变化幅度全在项目已知 ±0.04 R3 噪声内。

### 5.2 集成建议: **NOT-INTEGRATE(不建议改主线)**
**风险评估**:
- 收益不可靠: 改善幅度在噪声内, 剔除离群后退化
- 引入非确定性: padding 末段扩到下段 start 或 audio 末, 随 spk 重叠分布变化引入不稳定
- 误导归因: 改 cut_target_timeline 会强化"cut 有 bug"错误叙事, 实际真因在 argmax 选 target
- 已知噪声地板: 项目 `stability-test-launched` 已坐实 R3 输入微扰 ±0.04 噪声, 当前 Δ 全在此区间

### 5.3 cmd_2098 类样本真正改进方向
- `non-voiceprint-target-selection`(memory): argmax 选错 target 是 CER 失败主因, 解药在非声纹 target 选择(Whisper-Sidecar/ASE 选帧/GSE), 非 cut 边界。
- 该方向已记入 memory, 本任务不动。

## 6. 产物清单
- `code/exp_fix_cut_boundary.py`(诊断+POC修复+pad_sweep验证, 448 行)
- `code/runs/_fix_cut_boundary/`:
  - `cmd_2098/`: diar_timeline.json + spk{0,1}_v1.wav + spk{0,1}_v2_pad{080,200,400,560}ms.wav + rec_last_{300,500,800}ms.wav + spk{0,1}_last_seg_*.wav
  - `cmd_2098_qwen_sweep/`: cmd_2098 各切片(qwen 转写输入)
  - `cmd_2098_qwen_compare.json`: qwen 文本+CER 对比
  - `sample30_pad{000,030,050,080,120,200}ms/`: 各 pad 切片目录
  - `sample30_meta.json` + `sample30_cer_compare.json`(含 pad_sweep + 剔除离群分析)
  - `FINAL_VERDICT.json`(最终判定)
- 不改主线 `text_utils.py`。不 git commit。
