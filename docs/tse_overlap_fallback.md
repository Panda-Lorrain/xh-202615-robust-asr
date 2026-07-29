# pBSRNN overlap-only + failure fallback POC（2026-07-29）

## 结论

固定 speaker-disjoint synthetic val 40 条上，overlap-only 和
overlap-only+CAM++ fallback 均未达到 Phase-2 GO 门槛：

| 系统 | SI-SNRi | P10 | Qwen CER | 相对 raw ΔCER | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| raw（原 Phase-2 固定输出） | 0 | 0 | 1.2870 | 0 | — |
| unconditional pBSRNN | +1.0956 | -0.9442 | 1.2671 | -0.0199 | [-0.1162, +0.0831] |
| overlap-only | +0.4760 | -0.5498 | 1.2726 | -0.0144 | [-0.0873, +0.0504] |
| overlap-only + fallback | +0.5012 | -0.3999 | 1.2744 | -0.0126 | [-0.0764, +0.0424] |

GO 要求为 ΔCER ≤ -0.05 且 CI 上界 < 0；本轮两个条件均未满足。
因此不得扩大 pBSRNN 或进入 Phase-3 正式训练。

## 实现

新增 `code/tse_overlap_fallback.py`：

- `infer`：只对 oracle/外部提供的 overlap intervals 加 0.25 s context
  运行 pBSRNN。
- `finalize`：用 CAM++ 输出/enrollment cosine、有限值、RMS/峰值比、
  clipping、区段时长做失败检测；短 cross-fade 仅发生在 overlap 内。
- `calibrate`：只在 validation clean target 上离线选 cosine 阈值；
  clean/ref/Qwen CER 不进入线上路由特征。
- `evaluate`：固定 ID 比较 raw、unconditional、overlap-only、
  overlap-only+fallback，并检查零 overlap 精确直通。

新增 `tests/test_tse_overlap_fallback.py`，覆盖 interval 归一、synthetic
oracle overlap 推导、非 overlap 样本精确直通、cross-fade 边界和失败原因。

## 实测发现

1. pBSRNN 使用 scale-invariant SI-SNR 训练，模型原始输出 RMS 是 mixture
   的 3.61–6.52 倍。旧全段导出写 PCM WAV 时会隐式削波。overlap 拼接前
   必须显式做局部 RMS gain matching；脚本保留原始幅度比供诊断。
2. 40 条中 38 条存在 overlap，2 条无 overlap；后两条在输出中逐点相等。
3. 输出/enrollment CAM++ cosine 范围 0.085–0.761，均值 0.300。
   validation SI-SNRi 最优单阈值为 0.193641，接受 30/38 个区段。
4. fallback 改善声学尾部（P10 -0.550→-0.400 dB），但 Qwen CER
   反而比不回退略差（-0.0126 vs -0.0144）。说明当前 CAM++ cosine
   不是可靠的 ASR 失败代理。
5. overlap-only 的离线 CER oracle fallback 可到 1.2202，仍有选路空间，
   但线上特征尚不能兑现；禁止用 CER/ref/clean target 直接路由。
6. 为排除 WAV 重编码污染，零 overlap 和全部回退样本改为直接复制原始
   WAV 字节。即使输入 SHA256 完全相同，Qwen 跨进程复跑 raw CER 仍从
   1.2870 漂到 1.2996（37 条相同、3 条变差，Δ+0.0126）。用该复跑 raw
   比较时，overlap-only ΔCER 为 -0.0271（CI [-0.0976,+0.0374]），
   fallback 为 -0.0253（CI [-0.0874,+0.0281]），结论仍是 NO-GO。
   后续小于约 0.013 的差异不能脱离同进程配对控制作结论。

## Qwen mel failure proxy + speaker-LOSO

新增 `code/tse_failure_proxy.py`，只比较 raw overlap 与增强候选，可在线
计算；ref 和已有 Qwen errors 只用于离线阈值拟合/评估。为避免同一
validation 选阈值又报结果，按 4 个 target speaker 做 leave-one-speaker-out。

使用 Qwen3-ASR 本地权重自带的精确 `WhisperFeatureExtractor`
（128-bin、n_fft=400、hop=160）后，6 个单特征代理中最好的
`logmel_l2` 得到：

| 路由 | Qwen CER | 相对 raw ΔCER | bootstrap 95% CI | 接受增强 |
|---|---:|---:|---:|---:|
| raw | 1.2870 | 0 | — | 0/40 |
| overlap-only 全接受 | 1.2726 | -0.0144 | [-0.0873,+0.0504] | 38/40 |
| Qwen log-mel L2 speaker-LOSO | **1.2419** | **-0.0451** | **[-0.1105,+0.0109]** | 28/40 |

4 折训练阈值为 0.315132 / 0.315132 / 0.315132 / 0.314810，稳定性尚可；
但 ΔCER 仍未达到 -0.05，且 CI 上界仍大于 0，因此按预注册红线仍是
**NO-GO**。另外，`logmel_l2` 是在同一 40 条上从 6 个代理中选出的最好
结果，存在 feature-selection 乐观偏差，不能把 -0.0451 当作已泛化收益。
产物为 `failure_proxy_qwenmel_loso.json`（runs 目录，默认不入 Git）。

## 复现命令

所有 Python 命令均通过 `uv run` 使用已有隔离环境：

```powershell
$env:UV_CACHE_DIR='E:\midea_target_asr\.uv-cache'
uv run --python code/.venv_tse/Scripts/python.exe python code/tse_overlap_fallback.py infer `
  --manifest code/_tse_phase2_val/manifest_campp.jsonl `
  --checkpoint code/runs/tse_wesep_phase2/poc64r3_e5/best.pt `
  --output-dir code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback

uv run --python code/.venv_campp/Scripts/python.exe python code/tse_overlap_fallback.py finalize `
  --candidates code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback/candidates.jsonl `
  --output-dir code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback/calibrated_routes `
  --cosine-min 0.1936412751674652

uv run --python code/.venv_tse/Scripts/python.exe python code/tse_failure_proxy.py `
  --manifest code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback/calibrated_routes/manifest.jsonl `
  --qwen-compare code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback/qwen_compare_overlap_only.json `
  --output code/runs/tse_wesep_phase2/poc64r3_e5/overlap_fallback/failure_proxy_qwenmel_loso.json
```

## 下一步

Qwen 精确 mel 代理已有信号但未过门槛。按 Phase-2 预注册规则，当前最有
价值的后续是回到数据/损失修正，不扩大模型：

1. 优先修训练目标与合成域，使 separator 本身的全接受 Qwen ΔCER 超过
   噪声地板；当前用路由从弱 separator 榨收益，样本量不足且容易过拟合。
2. 若仍要验证 frozen Qwen audio encoder embedding drift，只能把它作为
   最后一项独立代理，并使用新的 speaker-disjoint validation；不能继续在
   同一 40 条上挑特征/阈值后报告。
3. 未达到 ΔCER ≤ -0.05 且 CI 上界 < 0 前，不进入 Phase-3。
