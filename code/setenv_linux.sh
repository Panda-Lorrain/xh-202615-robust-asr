#!/usr/bin/env bash
# ===== Linux 部署环境变量(AutoDL L40 / 主办方 L20) =====
# 对应 Windows 版 code/setenv.sh。路径默认 /root(AutoDL 默认; 按实际部署改)。
# 用法: 在项目根 `source code/setenv_linux.sh`
#
# 跨平台路径改造(2026-07-15): submit_infer/enroll_infer/*_asr_backend/se_denoise 已支持
# 平台检测 + env override, 配合本文件在 Linux 跑通 qwen 全流程 batch=1 实测效率腿。

# ---- HuggingFace 缓存(落 /root, 避免系统盘满)----
export HF_HOME="/root/hf_cache"
export HF_HUB_CACHE="/root/hf_cache/hub"
export HF_DATASETS_CACHE="/root/hf_cache/datasets"
# 主办方环境联网 HF(C11)直接下; 本地已下好设 HF_HUB_OFFLINE=1 免重下(默认 0 联网)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}

# ---- uv / pip / torch 缓存 ----
export UV_CACHE_DIR="/root/uv_cache"
export UV_PYTHON_INSTALL_DIR="/root/uv_python"
export PIP_CACHE_DIR="/root/pip_cache"
export TORCH_HOME="/root/torch_cache"
export XDG_CACHE_HOME="/root/xdg_cache"

# ---- 模型路径 override(本地缓存免重下; 主办方不设 → 代码走 HF repo id 自动下载)----
# resolve_model() 读 MODEL_* env(REPO_IDS 作 fallback)。按实际下载位置改。
export MODEL_VANILLA="/root/hf_cache/whisper-large-v3-turbo"
export MODEL_DICOW="/root/hf_cache/DiCoW_v3_2"
export MODEL_DIAR="/root/hf_cache/diarizen-wavlm-large-s80-md"
export MODEL_QWEN="/root/hf_cache/Qwen2.5-3B-Instruct"
# Qwen3-ASR / FireRedASR(独立 venv 后端, qwen_asr_backend.py / firered_asr_backend.py 读)
export MODEL_QWEN3_ASR="/root/hf_cache/Qwen3-ASR-1.7B"
export MODEL_FIRERED="/root/hf_cache/FireRedASR-AED-L"
# 可选 scene route；启用 --scene-route 前必须确认该缓存已存在，代码不会隐式下载
export MODEL_SEPFORMER="/root/hf_cache/sepformer-whamr16k"
# DeepFilterNet3(SE 阶段, 非 HF, GitHub Rikorose/DeepFilterNet release)
export DF_MODEL_BASE_DIR="/root/df_cache/DeepFilterNet3"
export DF_CACHE_DIR="/root/df_cache/DeepFilterNet/Cache"

# ---- 不设本地代理(AutoDL 走平台网关; 主办方直连。Windows 版的 7897 代理在此注释掉)----
# unset HTTPS_PROXY HTTP_PROXY ALL_PROXY

echo "[setenv_linux] HF_HOME=$HF_HOME | MODEL_QWEN3_ASR=$MODEL_QWEN3_ASR | OFFLINE=$HF_HUB_OFFLINE"
