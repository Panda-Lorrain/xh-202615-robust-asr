# 可复现性部署说明（主办方核查复现用）

> 对应 spec `docs/superpowers/specs/2026-07-06-reproducibility-hardening-design.md`（FAQ 2026-07-06 核查 = 完整复现结果比对）。
> 主办方环境：联网 HuggingFace（C11）、L20-46G、batch=1 计时。

## 1. DiCoW + DiariZen 代码（三层嵌套 submodule，enroll_infer.py 依赖）

`code/DiCoW-inference`（含 DiariZen + pyannote-audio）是第三方开源仓，主仓 `.gitignore` 不入库（`code/*/` 通配），需手动 clone：

```bash
git clone https://github.com/BUTSpeechFIT/DiCoW.git code/DiCoW-inference
cd code/DiCoW-inference
git submodule update --init --recursive   # 拉 DiariZen(Lakoc/DiariZen) + pyannote-audio
cd ../..
```

`enroll_infer.py:38-41` 自动把 `code/DiCoW-inference` + `DiariZen` + `DiariZen/pyannote-audio` 注入 `sys.path`，`from diarizen.pipelines.inference import DiariZenPipeline` 即可用。

## 2. 模型权重（4 个走 HF hub 自动下载 + DF3 例外）

代码 default 走 HF repo id，主办方环境联网 HF 自动下载（`from_pretrained`）：

| 模型 | HF repo id | 用途 |
|---|---|---|
| Whisper-large-v3-turbo | `openai/whisper-large-v3-turbo` | vanilla ASR 主线 |
| DiCoW | `BUT-FIT/DiCoW_v3_2` | dicow fallback（trust_remote_code）|
| DiariZen | `BUT-FIT/diarizen-wavlm-large-s80-md` | diarization + wespeaker 声纹 |
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | LLM 拒识（trust_remote_code）|
| **Qwen3-ASR-1.7B** | `Qwen/Qwen3-ASR-1.7B` | **中文原生 ASR 后端(`--asr-backend qwen`, 候选2 CER 腿+10分); Apache2.0; 走 code/.venv_qwen 独立环境** |

**DF3 例外**（非 HF 模型）：DeepFilterNet3 原始权重来自 GitHub [`Rikorose/DeepFilterNet`](https://github.com/rikorose/deepfilternet) release，`init_df(model_base_dir=...)` 接目录路径。下载后设 env `DF_MODEL_BASE_DIR` 指向。

**本地开发**：`code/setenv.sh` 设 `MODEL_VANILLA`/`MODEL_DICOW`/`MODEL_DIAR`/`MODEL_QWEN`/`DF_MODEL_BASE_DIR` env 指现有缓存（免重下），代码 `repro.resolve_model()` 优先读 env。

## 3. 环境（uv 管理，3 venv 隔离）

```bash
# Python 一律用 uv（禁止裸 pip）
# 主 venv(code/.venv: enroll_infer/DiariZen/vanilla/评测) 依赖已声明 code/requirements.txt
uv pip install -r code/requirements.txt   # cn2an/zhconv/editdistance/jiwer 等(2026-07-08 新增声明)
# 3+1 venv: code/.venv(主) / code/.venv_se(DeepFilterNet) / .venv_llm(Qwen LLM拒识) / **code/.venv_qwen(Qwen3-ASR 后端, 2026-07-11 新增)**
#
# code/.venv_qwen 建立(Qwen3-ASR transformers backend, Windows 兼容):
#   uv venv code/.venv_qwen --python 3.12
#   uv pip install --python code/.venv_qwen/Scripts/python.exe qwen-asr
#   uv pip install --python code/.venv_qwen/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu124 --reinstall  # 强制 reinstall 覆盖 CPU 版(qwen-asr 默认拉 CPU torch)
#   # 下权重: huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir E:/hf_cache/Qwen3-ASR-1.7B
#
# ⚠️ speechbrain lazy 修复(2026-07-11): code/.venv 的 speechbrain 1.1 lazy proxy 注册 sys.modules,
#   inspect.getmodule 遍历触发 lazy resolve ImportError → enroll_infer(librosa.load→lazy_loader→inspect)连锁崩。
#   已在 enroll_infer 顶部固化 patch inspect.getmodule(捕获返 None), 无需手动。enroll_infer --asr-backend qwen
#   内部 subprocess 调 code/.venv_qwen/python code/qwen_asr_backend.py(venv 隔离, 不污染主 venv)。
```

⚠️ **cn2an/zhconv 必装**（2026-07-08 workflow④ 发现并修复）：`text_utils.digit_postproc`/`to_simplified` 在缺包时 graceful 跳过 + RuntimeWarning 告警（不崩），但官方 CER 口径**不归一繁体/数字** → 缺包会让提交 content 残留繁体/阿拉伯数字，对齐简体中文数字 ref 时 CER 虚高（数字 ~0.03 全量、含数字句更甚；繁体更多）。`code/requirements.txt` 已声明，**务必 `uv pip install -r code/requirements.txt`**。`to_submission.py` 已加 digit_postproc 兜底（SSOT），enroll_infer:317-319 双归一。

## 4. 运行提交推理

```bash
source code/setenv.sh && export HF_HUB_OFFLINE=1   # 本地设 MODEL_* env; 主办方不设 → 代码走 HF repo id
bash code/run_baodi.sh pos    # pos 全量 vanilla, thr=0 全转写(CER 口径)
bash code/run_baodi.sh neg    # neg 全量 vanilla, thr=0.4 拒识(RR 口径)
code/.venv/Scripts/python.exe code/to_submission.py --result-json <out>/result.json --pairs code/<set>_pairs_datasetA.json
```

## 5. 可复现性验证（FAQ 核查硬要求 6：seed 固定跑两遍比对）

```bash
code/.venv/Scripts/python.exe code/verify_reproducibility.py \
  --pairs code/pos_pairs_datasetA.json --limit 20 --seed 42
# 期望: text 一致率 100%, CER delta ≤0.01(fp16 + cudnn.deterministic 已验证达标)
```

## 6. 随机种子（5 进程各自设，FAQ 核查硬要求 2）

`submit_infer --seed 42`（default），透传给 4 子进程（noise_classify/se_denoise/enroll_infer/llm_reject），各进程 `repro.set_global_seed()` 固定 torch+numpy+random+cuda+cudnn（deterministic=True, benchmark=False）。已验证 fp16 下 vanilla Whisper 完全确定（run-twice CER delta=0，无需 fp32）。
