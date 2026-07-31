#!/usr/bin/env bash
# CosyVoice 2 WSL 部署脚本 (Apache-2.0)。临时工具, 跑完可删。
# 路径: WSL 内 ~/cosyvoice/ (IO 快), 模型 ModelScope 下。
set -euo pipefail
cd "$HOME"
mkdir -p cosyvoice && cd cosyvoice
echo "=== [1/5] venv python3.10 ==="
if [ ! -d .venv_cosyvoice ]; then
  uv venv --python 3.10 .venv_cosyvoice
fi
# shellcheck disable=SC1091
source .venv_cosyvoice/bin/activate
python --version

echo "=== [2/5] clone CosyVoice (recursive, github->ghfast.top via 7897) ==="
# WSL ~/.gitconfig 配了 http.proxy=ghproxy.com (实测超时); 清空 proxy 走 https_proxy=7897,
# 并把 github.com 重写到 ghfast.top (7897 代理实测通, 主仓+Matcha-TTS 子模块都走它)。
export GIT_CONFIG_COUNT=3
export GIT_CONFIG_KEY_0="http.proxy"
export GIT_CONFIG_VALUE_0=""
export GIT_CONFIG_KEY_1="http.https://github.com.proxy"
export GIT_CONFIG_VALUE_1=""
export GIT_CONFIG_KEY_2="url.https://ghfast.top/https://github.com/.insteadof"
export GIT_CONFIG_VALUE_2="https://github.com/"
if [ ! -d CosyVoice ]; then
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
fi
cd CosyVoice
git submodule update --init --recursive

echo "=== [3/5] install requirements ==="
# --index-strategy unsafe-best-match: requirements.txt 有多个 extra-index-url
# (pytorch cu121 + onnxruntime-cuda-12), uv 默认只用第一个含某包的 index,
# 会导致 protobuf==4.25 等在 PyPI 有但在 onnxruntime index 无版本的包解析失败。
# openai-whisper/antlr4/wget 等 sdist 的 setup.py 用 pkg_resources, 但 uv 默认
# build isolation 环境不装 setuptools → ModuleNotFoundError: pkg_resources。
# 修复: 预装 setuptools/wheel + --no-build-isolation 让 sdist 复用主 venv 的 setuptools。
uv pip install "setuptools<70" wheel numpy cython
uv pip install --index-strategy unsafe-best-match --no-build-isolation -r requirements.txt
uv pip install ttsfrd wetext modelscope

echo "=== [4/5] download CosyVoice2-0.5B via modelscope ==="
python -c "from modelscope import snapshot_download; p=snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B'); print('MODEL_AT', p)"

echo "=== [5/5] smoke load CosyVoice2 ==="
export PYTHONPATH="$PWD/third_party/Matcha-TTS:${PYTHONPATH:-}"
python -c "from cosyvoice.cli.cosyvoice import CosyVoice2; m=CosyVoice2('pretrained_models/CosyVoice2-0.5B'); print('MODEL_LOADED_OK', type(m).__name__)"
echo "=== DEPLOY_DONE ==="
