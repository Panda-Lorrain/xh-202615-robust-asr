#!/usr/bin/env bash
# ===== C 盘重定向：所有下载/缓存落 E 盘（禁 C 盘）=====
# 用法：在项目根目录执行 `source code/setenv.sh`

# HuggingFace 模型权重 → E 盘
export HF_HOME="E:/hf_cache"
export HF_HUB_CACHE="E:/hf_cache/hub"
export HF_DATASETS_CACHE="E:/hf_cache/datasets"
export HF_ENDPOINT="https://hf-mirror.com"   # 国内镜像 fallback（直连超时再走它）

# uv / pip 缓存 → E 盘
export UV_CACHE_DIR="E:/uv_cache"
export UV_PYTHON_INSTALL_DIR="E:/uv_python"
export PIP_CACHE_DIR="E:/pip_cache"

# torch hub → E 盘
export TORCH_HOME="E:/torch_cache"
export XDG_CACHE_HOME="E:/xdg_cache"

# 代理（GitHub / HF 直连用）
export HTTPS_PROXY="http://127.0.0.1:7897"
export HTTP_PROXY="http://127.0.0.1:7897"
export ALL_PROXY="http://127.0.0.1:7897"

# transformers / huggingface 走镜像时关代理（避免代理→镜像绕路）；默认开代理
# 若 hf-mirror 也慢，可临时 unset HTTPS_PROXY

# ===== 可复现性：模型路径 override（代码 default 走 HF repo id，本地用 env 指缓存免重下）=====
# 提交时主办方环境不设这些 env → 代码自动走 HF repo id 下载（C11 联网 HF）
export MODEL_VANILLA="E:/hf_cache/whisper-large-v3-turbo"
export MODEL_DICOW="E:/hf_cache/DiCoW_v3_2"
export MODEL_DIAR="E:/hf_cache/diarizen-wavlm-large-s80-md"
export MODEL_QWEN="E:/hf_cache/Qwen2.5-3B-Instruct"
# 可选 scene route；启用 --scene-route 前必须确认该缓存已存在，代码不会隐式下载
export MODEL_SEPFORMER="E:/hf_cache/sepformer-whamr16k"
# DF3 例外：非 HF 模型，原始权重来自 GitHub Rikorose/DeepFilterNet
export DF_MODEL_BASE_DIR="E:/df_cache/DeepFilterNet/Cache/DeepFilterNet3"

echo "[setenv] HF_HOME=$HF_HOME | UV_CACHE_DIR=$UV_CACHE_DIR | proxy=$HTTPS_PROXY"
