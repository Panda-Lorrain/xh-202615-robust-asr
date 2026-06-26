#!/usr/bin/env bash
# ===== C 盘重定向：所有下载/缓存落 E 盘（禁 C 盘）=====
# 用法：每个需要下载的命令前 `source /e/midea_papers/code/setenv.sh`

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

echo "[setenv] HF_HOME=$HF_HOME | UV_CACHE_DIR=$UV_CACHE_DIR | proxy=$HTTPS_PROXY"
