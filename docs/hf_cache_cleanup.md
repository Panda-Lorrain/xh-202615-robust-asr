# E:/hf_cache 模型盘点与清理清单 (2026-07-27)

> **任务边界**: 仅盘点 + 列清单, **不删除任何文件**。本清单由用户确认后再执行。
> 主线判定依据: `submit_infer.py` + `enroll_infer.py` + `run_baodi.sh` + `setenv.sh` + `repro.py`。

## 总览
- **E:/hf_cache 总大小**: 29 GB
- **顶层条目数**: 15 (模型目录 11 + HF hub cache 1 + 辅助 3)
- **盘点口径**:
  - 🟢 主线 = `submit_infer → enroll_infer → --asr-backend {qwen, vanilla}` + `--no-llm --no-se` (memory `baodi-config-no-llm` + `run_baodi.sh` BACKEND 默认 vanilla, 用户可 BAODI_BACKEND=qwen 切 Qwen3-ASR 主线)
  - 主线 4 件: **Qwen3-ASR-1.7B** + **DiariZen (diarizen-wavlm-large-s80-md)** + **wespeaker (HF hub)** + **whisper-large-v3-turbo (vanilla fallback)**
  - 主线**不含**: LLM (`--no-llm` 关), SE (`--no-se` 关), DiCoW/SE_DiCoW 条件化 (Phase 1 后 vanilla+target timeline 替代)

---

## 🟢 必留 — 主线提交链路在用 (4 件, 共 **~6.6 GB**)

| # | 目录 | 大小 | 用途 / 代码位置 |
|---|---|---|---|
| 1 | `Qwen3-ASR-1.7B/` | 4.4 G | **qwen 后端主线** (CER 0.3436 / 含拒 0.5934 最佳)。`enroll_infer.py --asr-backend qwen` 切 target timeline → `qwen_asr_backend.py` 加载 (默认 `_DEFAULT_QWEN3`, 可 env MODEL_QWEN3_ASR 覆盖) |
| 2 | `diarizen-wavlm-large-s80-md/` | 531 M | **DiariZen diarization** (找 N 个 speaker 时间段)。`enroll_infer.py` line 172 `DiariZenPipeline.from_pretrained(args.diarization_model)`, `setenv.sh` MODEL_DIAR 指向此 |
| 3 | `hub/models--pyannote--wespeaker-voxceleb-resnet34-LM/` | 26 M | **wespeaker 声纹** (DiariZen `_embedding`)。`DiariZen/pipelines/inference.py` line 104-108 `hf_hub_download(repo_id="pyannote/wespeaker-voxceleb-resnet34-LM", filename="pytorch_model.bin", local_files_only=True)` — **强依赖 HF hub 路径, 不能删** |
| 4 | `whisper-large-v3-turbo/` | 1.6 G | **vanilla 后端保底默认** (`run_baodi.sh` BACKEND=vanilla)。`enroll_infer.py --asr-backend vanilla` 直接 `AutoModelForSpeechSeq2Seq.from_pretrained`, `setenv.sh` MODEL_VANILLA 指向此 |

---

## 🟡 谨慎删 — POC / 诊断 / 对比基线 (7 件, 共 **~22.0 GB**)

> **删了重跑对应实验要重新下载**。建议保留至少 DiCoW_v3_2 (argparse default) + Qwen2.5-3B-Instruct (LLM 实验链路完整保留)。

| # | 目录 | 大小 | 用途 / 哪个实验 | 风险点 |
|---|---|---|---|---|
| 5 | `DiCoW_v3_2/` | 7.2 G | `enroll_infer.py --dicow-model` default, **argparse `--asr-backend` 默认值仍是 `"dicow"`** (line 123); `repro.py` REPO_IDS["DICOW"]; `setenv.sh` MODEL_DICOW; 答辩反 cascaded 对比基线 (CER 1.25 vs vanilla 0.664); `apply_dicow_langfix.py` patch 其 generation.py | ⚠️ 删后裸调 enroll_infer (无 --asr-backend) 会崩; 需同步把 argparse default 改 "vanilla" |
| 6 | `SE_DiCoW/` | 4.4 G | SE A/B 实验链路 (memory `se-bug-orphan-truth`); `enroll_infer.py` line 334 检测 `uses_enrollments=True` 自动启用 cross-attn; `apply_dicow_langfix.py` 也在 patch 列表; `stno_ablation.py` | ⚠️ 关 SE 已是 baodi 默认, 但删后无法重跑 SE A/B 复核 |
| 7 | `Qwen2.5-3B-Instruct/` | 5.8 G | `llm_reject.py` DEFAULT_MODEL (`resolve_model("QWEN")`); `submit_infer.py` 阶段3 LLM 拒识 (默认 `--no-llm` 关, 但开 LLM 跑非保底实验需要); `setenv.sh` MODEL_QWEN | ⚠️ baodi 关 LLM 非主线, 但开 LLM 实验链路会断 |
| 8 | `FireRedASR-AED-L/` | 4.4 G | `firered_asr_backend.py` `_DEFAULT_FIRERED` (env MODEL_FIRERED 覆盖); `--asr-backend firered` 备选 (RTF 更优, 但 CER 已证伪非主线); memory `cer-breakthrough-candidates` | 删后无法重跑 firered 后端对比 |
| 9 | `sepformer-whamr16k/` | 108 M | SepFormer 源分离 POC: `exp_sepformer_b2.py` / `exp_sepformer_poc.py` / `exp_sepformer_qwen.py` / `exp_oracle_separation_ceiling.py` / `exp_deadzone_*.py` (memory `spk-oracle-poc` 双证伪 + `overlap-is-cer-failure-rootcause` 待重测子集) | 删后 SepFormer 重跑要重下 |
| 10 | `campplus/` | 28 M | CAM++ 声纹 POC: `exp_campp_select_cer.py` / `exp_ase_keyframe_diag.py` / `exp_spk_campp_deadzone.py` / `exp_spk_oracle*.py` (memory `spk-oracle-poc` Qwen3 证伪) | 删后 CAM++ 实验要重下 |
| 11 | `modules/` | 743 K | `transformers` `trust_remote_code` 自动写入的 DiCoW/SE_DiCoW Python 模块缓存 (`modules/transformers_modules/{DiCoW_v3_2,SE_DiCoW}/`) | 删了首次加载 DiCoW 会自动重建 (无网络也能恢复, 风险低) |

---

## 🔴 建议删 — 失败下载 / 重复占位 / 无用日志 (7 件, 共 **~3.2 MB**)

> 空间收益**几乎为零** (最大的 FireRedASR-LLM-L 仅 20K 是失败 stub), 仅做清理。删了无任何代码路径会崩。

| # | 目录 | 大小 | 为什么可删 |
|---|---|---|---|
| 12 | `FireRedASR-LLM-L/` | 20 K | **失败下载 stub**: `asr_encoder.pth.tar` 仅 1.4K (真实应 ~1GB), 只有 `.cache/huggingface/download/*.metadata` 占位。完整模型从未下完, `firered_llm_quant_backend.py` 跑不起来 (需完整 8.3B) |
| 13 | `models--Qwen--Qwen2.5-3B-Instruct/` (top-level) | 1 KB | HF hub 占位 (只 `refs/main` commit hash), 真实模型在 `Qwen2.5-3B-Instruct/` (5.8G) |
| 14 | `hub/models--BUT-FIT--SE_DiCoW/` | 1 KB | HF hub 占位 (只 `refs/main`), 真实模型在 `SE_DiCoW/` (4.4G) |
| 15 | `hub/models--BUT-FIT--diarizen-wavlm-large-s80-md/` | 1 KB | HF hub 占位, 真实模型在 `diarizen-wavlm-large-s80-md/` (531M) |
| 16 | `hub/models--FireRedTeam--FireRedASR-AED-L/` | 1 KB | HF hub 占位, 真实模型在 `FireRedASR-AED-L/` (4.4G) |
| 17 | `hub/models--Qwen--Qwen3-ASR-1.7B/` | 1 KB | HF hub 占位, 真实模型在 `Qwen3-ASR-1.7B/` (4.4G) |
| 18 | `xet/` | 3.2 M | HF Xet 内容寻址缓存日志 (`xet/logs/`), 自动重建无副作用 |

---

## 汇总

| 档位 | 件数 | 大小 | 说明 |
|---|---|---|---|
| 🟢 必留 (主线) | 4 | **6.6 GB** | 删了提交崩 |
| 🟡 谨慎删 (POC) | 7 | **22.0 GB** | 删了重跑对应实验要重下 |
| 🔴 建议删 (失败/占位) | 7 | **3.2 MB** | 删了无影响, 但空间收益几乎为零 |
| **合计** | **18** | **~29 GB** | 与 `du -sh /e/hf_cache` 一致 |

### "若全删🔴可腾空间"
**仅 ~3.2 MB** — 几乎可忽略 (最大单件 FireRedASR-LLM-L 才 20K)。🔴 的清理价值在整洁不在空间。

### "若进一步删🟡可腾空间"
**~22 GB** — 但会丢失: DiCoW 答辩对比基线 / SE A/B 复现链路 / LLM 拒识实验链路 / firered 备选 / SepFormer+CAM++ 全部 POC 复现能力。**不建议**。

---

## ⚠️ 误删风险提示 (用户决策前必看)

1. **`DiCoW_v3_2` 不要归🔴**: `enroll_infer.py` argparse `--asr-backend` **default 仍是 `"dicow"`** (line 123/207), `run_baodi.sh` 强制 BACKEND=vanilla 才绕过; 若用户裸调 `python enroll_infer.py` 不带参数会走 dicow → 删了崩。归🟡。删前需同步把 argparse default 改成 `"vanilla"` 或 `"qwen"`。

2. **`hub/models--pyannote--wespeaker-voxceleb-resnet34-LM` 必须保留**: 与其它 hub/ 占位不同, 这个**有真实 blob** (26M pytorch_model.bin), DiariZen 强依赖 `hf_hub_download(local_files_only=True)` 取它。**它是🟢主线, 删了 DiariZen 崩**。

3. **`modules/` 删除前测试**: trust_remote_code 加载 DiCoW 时查找此目录。理论上会自动重建 (DiCoW_v3_2/ 内有相同 .py), 但若整盘只读或权限问题会失败。建议归🟡保守。

4. **`SE_DiCoW` 与 `apply_dicow_langfix.py`**: langfix 补丁同时 patch `SE_DiCoW/generation.py`, 删后那脚本 `[skip]` 不崩但 langfix 覆盖面缩。归🟡。

5. **`Qwen2.5-3B-Instruct` 影响 `.venv_llm`**: `.venv_llm` (项目根, 非 code/) 是为 LLM 单独建的隔离 venv, 删模型后 venv 仍可用, 但 `submit_infer.py --no-llm` 解除(开LLM)会崩。

---

## 附录: 顶层条目全清单 (按大小降序)

```
7.2 G  DiCoW_v3_2/                              🟡 argparse default / 答辩基线
5.8 G  Qwen2.5-3B-Instruct/                    🟡 llm_reject (baodi 关 LLM)
4.4 G  FireRedASR-AED-L/                       🟡 firered 后端 POC
4.4 G  Qwen3-ASR-1.7B/                         🟢 qwen 后端主线
4.4 G  SE_DiCoW/                               🟡 SE A/B 复现
1.6 G  whisper-large-v3-turbo/                 🟢 vanilla 后端保底默认
531 M  diarizen-wavlm-large-s80-md/            🟢 DiariZen diarization
108 M  sepformer-whamr16k/                     🟡 SepFormer POC
28 M   campplus/                               🟡 CAM++ POC
26 M   hub/models--pyannote--wespeaker-...     🟢 wespeaker (HF hub)
3.2 M  xet/                                    🔴 HF Xet 日志
743 K  modules/                                🟡 trust_remote_code 缓存
20 K   FireRedASR-LLM-L/                       🔴 失败下载 stub
1 KB   models--Qwen--Qwen2.5-3B-Instruct/      🔴 HF hub 占位
1 KB   hub/models--BUT-FIT--SE_DiCoW/          🔴 HF hub 占位
1 KB   hub/models--BUT-FIT--diarizen-...       🔴 HF hub 占位
1 KB   hub/models--FireRedTeam--FireRedASR...  🔴 HF hub 占位
1 KB   hub/models--Qwen--Qwen3-ASR-1.7B/       🔴 HF hub 占位
```

---

## 附: 主线模型加载链路 (代码核实)

```
submit_infer.py
  ├── enroll_infer.py (PY_MAIN = code/.venv)
  │     ├── DiariZen: DiariZenPipeline.from_pretrained(MODEL_DIAR)            # 🟢 diarizen-wavlm-large-s80-md/
  │     │     └── _embedding ← hf_hub_download("pyannote/wespeaker-...")     # 🟢 hub/models--pyannote--wespeaker-...
  │     └── ASR backend (args.asr_backend):
  │           ├── dicow  (default): AutoModelForSpeechSeq2Seq(MODEL_DICOW)   # 🟡 DiCoW_v3_2/
  │           ├── vanilla (baodi) : AutoModelForSpeechSeq2Seq(MODEL_VANILLA) # 🟢 whisper-large-v3-turbo/
  │           ├── qwen   (主线)  : 切 target 段 → qwen_asr_backend.py        # 🟢 Qwen3-ASR-1.7B/
  │           └── firered        : 切 target 段 → firered_asr_backend.py     # 🟡 FireRedASR-AED-L/
  ├── se_denoise.py (PY_SE = code/.venv_se) — baodi --no-se 跳过
  │     └── DeepFilterNet3 (E:/df_cache/, 非 hf_cache, 另算)
  └── llm_reject.py (PY_LLM = ROOT/.venv_llm) — baodi --no-llm 跳过
        └── Qwen2.5-3B-Instruct (MODEL_QWEN)                                 # 🟡 Qwen2.5-3B-Instruct/
```
