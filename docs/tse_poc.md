# TSE (Target Speaker Extraction) POC 报告

> **任务**: 核实 TSE 模型权重可用性 + 零成本 POC(不训练, 用现成权重)
> **背景**: 链路瓶颈 ①盲分离 SepFormer 破坏声纹(SI-SDR)→选路失效25% ②中文 OOD ③重叠区物理混合。TSE = enrollment 引导直接提取 target, 对症解药。
> **日期**: 2026-07-27
> **执行**: TSE 调研+POC agent

## 步骤1: 权重核实结论

| 候选 | 权重 | 推理代码 | 中文 | 8kHz/16kHz | License | 部署难度 | 选用 |
|------|------|---------|------|-----------|---------|---------|------|
| **TSELM** (Tang 2024) | ✓ HF 公开 (1.87GB) | ✓ inference.py + 完整 model class | 英文 Libri2Mix | 16kHz | CC BY-NC-SA | 中 | **选用** |
| USEF-TSE (ICASSP 2024) | ✓ HF 1.8GB (chkpt 子目录) | ✗ HF 只放 chkpt+config+model.py, 缺完整 `models/ utils/` 包; GitHub 找不到 | 英文 wsj0-2mix/WHAM | **8kHz** (要降采样) | CC BY-NC | **极高** | 不选 |
| Whisper-Sidecar (Interspeech 2024) | ✗ **确认无公开权重** (memory `non-voiceprint-target-selection` 记的"无"对, 调研说"中文 zero-shot CER 28.94%"在 repo 页面未提及) | ✓ evaluation.py | - | 16kHz | MIT | 要自训 → 不可用 | 不选 |
| ClearerVoice-Studio SpEx+ (Alibaba) | ✓ HF 权重在 | ✗ ClearVoice 框架只有 AV_TSE (audio-visual 要视频), SpEx+ 训练目录无 inference.py, 要自己拼 | 英文 wsj0-2mix | 8kHz | Apache-2.0 | 高 | 不选 |
| ClearerVoice-Studio AV_TSE | ✓ HF | ✓ pip clearvoice | 英文 + 视频输入 | 16kHz | Apache-2.0 | 输入不匹配 (无视频) | 不选 |

**调研说"USEF-TSE 最易(权重 HF 1.8GB)"是错的**: HF 上只放 chkpt + config.yaml + model.py, 缺完整的 `models/` `utils/` 子包, 加载模型会 ImportError; GitHub 仓库也搜不到独立推理代码。所以 USEF-TSE 实际部署难度极高。
**调研说"Whisper-Sidecar 中文 zero-shot CER 28.94%"在 repo 没找到对应权重**, 要自训, 不可用。
**TSELM 是唯一"完整权重+完整推理代码"组合**, 选用。

## 步骤2: 部署

- venv: `code/.venv_tse/` (uv venv --python 3.10, torch 2.3.1+cu121, transformers 4.42.3, speechbrain, hyperpyyaml)
- 权重: `E:/hf_cache/tselm/`
  - `tselm_l.pth` (1.87GB, TSELM-L checkpoint)
  - `kmeans_wavlm_ckpt/` (kmeans 离散化, 解压自 .tar.gz)
  - `hifigan-wavlm-l1-3-7-18-23-k1000-LibriTTS/` (HiFiGAN 解码器, 解压自 .tar.gz)
  - `microsoft/wavlm-large` (HF 自动下载, ~1.2GB)
- config: `code/TSELM/config/tselm_l_poc.yaml` (替换占位路径)
- 推理脚本: `code/exp_tse_poc.py` (绕过 scp, 单条 mix+regi 调用)

## 步骤3: POC 样本

主线 CER 都~1.0 的翻车样本 (来自 `poc_qwen_asr_full_result.json`):

| uid | sim | 主线 qwen CER | ref | 翻车机制 |
|-----|-----|-------------|-----|---------|
| cmd_2637 | 0.585 | 1.125 | 哺乳期要少吃什么 | 重叠区 (memory `overlap-is-cer-failure-rootcause` 坐实) — **TSE 应救** |
| cmd_18 | 0.058 | 1.000 | 关闭灯光 | 死区物理地板 (sim<0.2 babble 摧毁 mel) — **TSE 期望救不回** |
| cmd_2098 | 0.020 | 1.000 | 调到二十八度 | 死区物理地板 |
| cmd_2251 | 0.604 | 1.000 | 把温度调到三十度 | 额外重叠组 |
| cmd_2687 | 0.579 | 1.000 | 把温度调到三十度 | 额外重叠组 |
| cmd_2630 | 0.567 | 2.250 | 开左右风 | 极端幻觉组 |

## 步骤4: POC 结果

| uid | TSE CER | 主线 CER | Δ | 救回? | TSE 转写 | ref |
|-----|--------|---------|---|------|---------|-----|
| cmd_2637 | 1.125 | 1.125 | +0.000 | ✗ | 吃一顿，起身过一阵子。 | 哺乳期要少吃什么 |
| cmd_18 | 1.000 | 1.000 | +0.000 | ✗ | 啊。 | 关闭灯光 |
| cmd_2098 | 1.000 | 1.000 | +0.000 | ✗ | 修理阀门。 | 调到二十八度 |
| cmd_2251 | 1.000 | 1.000 | +0.000 | ✗ | 是行政区域内的。 | 把温度调到三十度 |
| cmd_2687 | 1.000 | 1.000 | +0.000 | ✗ | 超前而下的较量。 | 把温度调到三十度 |
| cmd_2630 | 1.750 | 2.250 | **-0.500** | ✓ | 是他们经常工作。 | 开左右风 |

**汇总**: TSE 救回 1/6 (cmd_2630 Δ-0.5 但绝对值仍 1.75 灾区), 平均 TSE CER 1.146 vs 主线 1.229 (Δ-0.083 边际)。**TSELM 英文训练 zero-shot 中文 OOD 灾难性失败**。

### TSE 输出 wav 物理特性 (agent 初判听感依据)
| uid | 原始 rms | TSE rms | TSE max | 备注 |
|-----|---------|---------|---------|------|
| cmd_2637 | 0.033 | 0.100 | 0.65 | 能量放大 3x (HiFiGAN 重建后放大), 非静音 |
| cmd_18 | 0.025 | 0.019 | 0.23 | 接近原始能量, 但 qwen 转出"啊" → 提取噪声 |
| cmd_2098 | 0.036 | 0.074 | 0.45 | 能量放大 2x |
| cmd_2251 | 0.012 | 0.135 | 0.66 | **能量放大 11x**, HiFiGAN 重建放大过度 |
| cmd_2687 | 0.004 | 0.118 | 0.62 | **能量放大 30x**, 同上 |
| cmd_2630 | 0.035 | 0.102 | 0.81 | 能量放大 3x |

**Agent 初判**: TSE 输出全部非静音, 但能量异常放大 (cmd_2251/2687 放大 11-30x), 加上 qwen 转写全部"无关短语" → 强烈怀疑 HiFiGAN 用英文 kmeans 重建时把中文音素打散到错误 token, 输出失真音/英文味音 (英文味被 ASR 错认为中文乱短语)。需用户耳朵确认。

## 步骤5: 听感验证 (用户听, 见 docs/verify_tse_poc.md)

## 关键诚实标注

1. **TSELM 是英文 Libri2Mix/LibriTTS 训练**, zero-shot 中文是 OOD 场景 — POC 结果证实灾难性失败 (6/6 CER ≥1.0), 与预期 OOD 风险一致。
2. **enrollment 1.5-2.85s vs TSELM 训练 4.05s** — 不足部分用静音 pad (truc_wav F.pad), enrollment 表征弱化 (但本次失败主因不是 enroll 长度, 是 OOD)。
3. **样本量 6 条** — 小样本 POC, 但 6 条全 CER ≥1.0 + 转写全部乱短语 = 强信号, 不是统计噪声。
4. **HiFiGAN 重建能量异常放大** (cmd_2251/2687 放大 11-30x) — 可能是英文 kmeans 对中文音素表征崩溃。

## 结论

**TSE 治本路线"用现成英文权重 zero-shot 中文"被 POC 证伪**:
- TSELM 6 条 POC 全部 CER ≥1.0, 0/6 实质救回 (cmd_2630 Δ-0.5 但绝对值仍 1.75 灾区)
- 转写全部"无关短语" + HiFiGAN 能量异常放大 → 英文训练 OOD 失败的典型表现
- 与之前盲分离 SepFormer 证伪 (memory `overlap-is-cer-failure-rootcause`) 类似: 现成 TSE 权重都是英文训练, 中文 OOD 灾难性失败

**值得投入训练中文版吗? ROI 评估**:
- ✗ **门槛高**: TSELM 训练需要 100-360h 中文双说话人重叠语料 (类似 Libri2Mix 中文版, Aishell1Mix 等), 我们 A 集是测试集不能训练 (memory `lessons-pitfalls` §14), 需 A 集外数据。
- ✗ **训练成本**: 8 卡 GPU 数日 (论文用 9 卡 200 epoch), 不在剩余参赛时间内。
- ✓ **理论收益不确定**: 即使训练中文版, TSELM 在英文测试集 SI-SDRi 也只 17dB (仍有损失), 加上 enroll 1.8s 短约束, 未必能从 1.0 救回 < 0.5。
- ⚖ **替代方案**: memory `non-voiceprint-target-selection` 提到的 **Whisper-Sidecar (embedding 分离, 绕开 SI-SDR 鸿沟)** 也无公开权重需自训 → 同样 ROI 低。
- 🎯 **结论**: TSE 治本路线"等中文权重" = 等外训 + 自训, ROI 低, **不投入训练**。继续走 qwen 主线 CER 0.3436 (主战场已天花板) + 答辩腿归因 (TSE 是已被 POC 探索但证伪的方向, 答辩弹药)。

## 答辩弹药

- **诚实归因**: TSE 治本路线被探索但证伪 — 调研 4 个候选 (USEF-TSE / Whisper-Sidecar / TSELM / ClearerVoice SpEx+), 只有 TSELM 有完整权重+推理代码可 POC, 英文训练 zero-shot 中文 6 条 POC 全 CER≥1.0 救不回。
- **架构认知**: TSE 的 SI-SDR/重建范式 (WavLM 离散 + HiFiGAN 解码) 对中文 OOD 灾难性失败, 与盲分离 SepFormer 证伪一致 — 现成 TSE 权重都是英文训练, 中文需自训, ROI 低不投入。
- **数据诚实**: A 集是测试集不能训练 TSE 中文版 (memory `lessons-pitfalls` §14), 答辩可说"中文 TSE 需 A 集外大数据训练, 不在参赛范围"。

