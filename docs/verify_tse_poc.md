# TSE POC 听感验证索引

> **目的**: 让用户听 TSE 提取的 target audio vs 主线切片, 判断"TSE 提取干净吗? 是 target 一人吗?"。
> 用户耳朵是金标准 (cmd_2637/18/2098 主线翻车版用户已听过)。
>
> **生成时间**: 2026-07-27 TSE POC
> **TSELM (Tang et al., 2024)**: WavLM 离散化 + cross-attention (enroll 条件) + Conformer LM + HiFiGAN, **英文 Libri2Mix/LibriTTS 训练, zero-shot 中文 OOD 风险**, enroll 训练 4.05s 我们 1.5-2.85s (短 enroll 风险)。

## 听音方法
1. **主线翻车版** (`code/runs/_diag_2637/` 等已有切片) — 用户已听过
2. **TSE 提取版** (本次新增) — 见下表

每条听 3 件事:
- **干净度**: TSE 提取后是否还残留另一人/噪声?
- **目标性**: 提取的是 target 一人吗 (vs 干扰者)?
- **可懂度**: 能听出 ref 句吗 (vs 主线幻觉)?

## 听感索引表

| uid | ref | sim 主线 | 主线 CER | TSE 提取 wav | TSE CER | TSE qwen 转写 | 备注 |
|-----|-----|---------|---------|-------------|---------|-------------|------|
| cmd_2637 | 哺乳期要少吃什么 | 0.585 | 1.125 | `code/runs/_tse_poc/cmd_2637_tse.wav` | 1.125 | 吃一顿，起身过一阵子。 | **重叠区主战场, TSE 未救** |
| cmd_18 | 关闭灯光 | 0.058 | 1.000 | `code/runs/_tse_poc/cmd_18_tse.wav` | 1.000 | 啊。 | 死区物理地板 |
| cmd_2098 | 调到二十八度 | 0.020 | 1.000 | `code/runs/_tse_poc/cmd_2098_tse.wav` | 1.000 | 修理阀门。 | 死区物理地板 |
| cmd_2251 | 把温度调到三十度 | 0.604 | 1.000 | `code/runs/_tse_poc/cmd_2251_tse.wav` | 1.000 | 是行政区域内的。 | 额外重叠组 |
| cmd_2687 | 把温度调到三十度 | 0.579 | 1.000 | `code/runs/_tse_poc/cmd_2687_tse.wav` | 1.000 | 超前而下的较量。 | 额外重叠组 |
| cmd_2630 | 开左右风 | 0.567 | 2.250 | `code/runs/_tse_poc/cmd_2630_tse.wav` | 1.750 | 是他们经常工作。 | 极端幻觉组, 唯一 Δ-0.5 |

enrollment 参考: `datasetA/pos/kws_{N}.wav` (1.5-2.85s, **短于 TSELM 训练 4.05s**)
混合 audio: `datasetA/pos/cmd_{N}.wav` (1.7-2.5s)
原始混合 wav 对照: `datasetA/pos/cmd_{N}.wav`

## 用户判断回填区

> POC 已跑完 (2026-07-27), TSE CER 0/6 救回 (cmd_2630 边际改善但绝对值仍 1.75 灾区)。
> 用户听音判断每条 [✓干净 / ✗残留 / ✗错人 / ✗机器音]:
>
> - cmd_2637: ___
> - cmd_18: ___
> - cmd_2098: ___
> - cmd_2251: ___
> - cmd_2687: ___
> - cmd_2630: ___
>
> **Agent 初判依据**: TSE 输出全部非静音但能量异常放大 (cmd_2251/2687 放大 11-30x), qwen 转写全部"无关短语" → 怀疑 HiFiGAN 英文 kmeans 重建中文音素崩溃, 输出失真音/英文味音。需用户耳朵确认。
