#!/usr/bin/env bash
# demo 网站启动器: source setenv + uvicorn 起在 0.0.0.0:7860
# 公网: 另起 cloudflared quick tunnel(见 README), 把 https://*.trycloudflare.com 发给访客
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "cd code/ 失败"; exit 1; }   # 进 code/
source setenv.sh                 # code/setenv.sh(HF缓存指E盘 + 代理)
export HF_HUB_OFFLINE=1          # 模型已缓存, 离线跑避免每次校验网络
echo "[demo] 启动 uvicorn 0.0.0.0:7860 (首次加载模型约 40s)..."
exec .venv/Scripts/python.exe -m uvicorn demo_web.server:app --host 0.0.0.0 --port 7860
