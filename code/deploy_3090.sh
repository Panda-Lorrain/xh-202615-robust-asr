#!/usr/bin/env bash
# 3090 一键部署(幂等, 默认关 SE 跳过 .venv_se)。2026-08-06 FAQ 坐实官方评测硬件=RTX 3090 24G(Ampere sm_86); 直接在 3090 实测, 不外推(原 L40×1.5 折算作废)。
# 用法: 在项目根  bash code/deploy_3090.sh
#   BAODI_SE_DEPLOY=1 bash code/deploy_3090.sh   # 同时建 .venv_se(复现 SE A/B 才需要)
#   HF_ENDPOINT=https://hf-mirror.com bash code/deploy_3090.sh  # 走镜像(默认已设)
# 日志: 代码直接输出, 远端跑建议  bash code/deploy_3090.sh 2>&1 | tee /root/deploy.log
# 前置: RTX 3090 实例(官方评测硬件) + datasetA/ 已上传(或本机 scp 上来)。
# 详见 docs/L20效率实测_runbook_2026-07-15.md(顶部 2026-08-06 3090 勘误 + §3)。
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "[fatal] cd 项目根失败"; exit 1; }
source code/setenv_linux.sh

log(){ echo "[deploy $(date +%H:%M:%S)] $*"; }
done_(){ [[ -e "$1" ]]; }   # 幂等: 文件/目录存在即跳过
export PATH="$HOME/.local/bin:$PATH"

# ---- 0. uv ----
if ! command -v uv >/dev/null 2>&1; then
  log "装 uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ---- 1. 主 venv (enroll_infer / DiariZen / vanilla / 评测) ----
if ! done_ code/.venv/bin/python; then
  log "建 .venv(主) + requirements + speechbrain/onnxruntime"
  uv venv code/.venv --python 3.12
  uv pip install --python code/.venv/bin/python -r code/requirements.txt
  uv pip install --python code/.venv/bin/python speechbrain onnxruntime
else log ".venv 已存在, 跳过"; fi

# ---- 2. DiCoW/DiariZen submodule (diar 依赖) ----
if ! done_ code/DiCoW-inference/README.md; then
  log "clone DiCoW submodule"
  git clone --depth 1 https://github.com/BUTSpeechFIT/DiCoW.git code/DiCoW-inference
  (cd code/DiCoW-inference && git submodule update --init --recursive --depth 1)
else log "DiCoW 已存在, 跳过"; fi

# ---- 3. qwen venv (Qwen3-ASR, 独立隔离, GPU torch) ----
if ! done_ code/.venv_qwen/bin/python; then
  log "建 .venv_qwen (Qwen3-ASR + GPU torch cu124)"
  uv venv code/.venv_qwen --python 3.12
  uv pip install --python code/.venv_qwen/bin/python qwen-asr
  uv pip install --python code/.venv_qwen/bin/python torch --index-url https://download.pytorch.org/whl/cu124 --reinstall
else log ".venv_qwen 已存在, 跳过"; fi

# ---- 4. (可选) SE venv —— 默认关 SE, 跳过 ----
if [[ "${BAODI_SE_DEPLOY:-0}" == "1" ]]; then
  if ! done_ code/.venv_se/bin/python; then
    log "建 .venv_se (DeepFilterNet3, 仅 SE A/B 用)"
    uv venv code/.venv_se --python 3.12
    uv pip install --python code/.venv_se/bin/python df torch torchaudio soundfile numpy
  fi
else log "默认关 SE, 跳过 .venv_se (BAODI_SE_DEPLOY=1 启用)"; fi

# ---- 5. 权重 (HF, 国内默认走 mirror) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p /root/hf_cache
dl(){ if ! done_ "/root/hf_cache/$2"; then log "下 $1 → /root/hf_cache/$2"; huggingface-cli download "$1" --local-dir "/root/hf_cache/$2"; else log "$2 已存在, 跳过"; fi; }
log "下权重(国内走 hf-mirror, 大约 25G)"
dl openai/whisper-large-v3-turbo whisper-large-v3-turbo
dl BUT-FIT/diarizen-wavlm-large-s80-md diarizen-wavlm-large-s80-md
dl Qwen/Qwen3-ASR-1.7B Qwen3-ASR-1.7B
# scene_route 默认开(BAODI_SCENE_ROUTE=1)需要 SepFormer; speechbrain 在线拉取易被
# 镜像/断网卡住, 这里预下到 MODEL_SEPFORMER 指向的 sepformer-whamr16k 目录。
dl speechbrain/sepformer-whamr16k sepformer-whamr16k
# 可选(qwen 不用 DiCoW; 关 LLM 不用 Qwen2.5-3B):
# dl BUT-FIT/DiCoW_v3_2 DiCoW_v3_2
# dl Qwen/Qwen2.5-3B-Instruct Qwen2.5-3B-Instruct

# DF3 权重(GitHub release 非 HF, 仅 SE A/B 用)
if [[ "${BAODI_SE_DEPLOY:-0}" == "1" ]]; then
  if ! done_ "$DF_MODEL_BASE_DIR"; then
    log "[warn] DF3 权重需手动放 $DF_MODEL_BASE_DIR (GitHub Rikorose/DeepFilterNet release, 见 REPRO_SETUP.md)"
  fi
fi

# ---- 6. 数据 + pairs ----
if ! done_ code/pos_pairs_datasetA.json; then
  log "生成 pos/neg pairs manifest"
  if ! done_ datasetA; then echo "[error] datasetA/ 不存在, 先 scp 上传(278M)"; exit 1; fi
  code/.venv/bin/python code/make_pairs_from_datasetA.py
else log "pairs 已存在, 跳过"; fi

log "✅ 部署完成。下一步: SMOKE=1 bash code/run_efficiency_3090.sh  (冒烟 5 条)"
log "          然后:        bash code/run_efficiency_3090.sh          (全量, 直接 3090 实测)"
