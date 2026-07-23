#!/usr/bin/env bash
# L20 效率腿实测一键(默认关 SE, qwen 后端, 全量 batch=1 口径 + 换算; L40 可近似, RTF×1.5)。2026-07-18 阶段0 准备, 2026-07-23 l40→l20 正名。
# 用法: 在项目根
#   SMOKE=1 bash code/run_efficiency_l20.sh   # 冒烟 5 条(确认管线通, ~1min, 强烈建议先跑)
#   bash code/run_efficiency_l20.sh           # 全量 pos+neg + 换算(~15-30min on L20)
#   ONLY=pos bash code/run_efficiency_l20.sh  # 只跑 pos
# 前置: deploy_l20.sh 已跑完。
# 产出: code/out_{pos,neg}_baodi/{timing.json,result.json} + /root/efficiency_*.txt
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "[fatal] cd 项目根失败"; exit 1; }
source code/setenv_linux.sh

log(){ echo "[run $(date +%H:%M:%S)] $*"; }
PY=code/.venv/bin/python
export BAODI_OK=1   # 绕过 submit_infer 守卫(run_baodi 自动设, 冒烟直调也要)

# ---- 0. 冒烟模式: 切前 5 条 pairs 直调 submit_infer ----
if [[ "${SMOKE:-0}" == "1" ]]; then
  log "🔥 冒烟: 切前 5 条 pos pairs 直调 submit_infer(关SE, qwen)"
  $PY -c "import json; d=json.load(open('code/pos_pairs_datasetA.json')); json.dump(d[:5], open('code/_smoke_pairs.json','w',encoding='utf-8'), ensure_ascii=False)"
  $PY code/submit_infer.py --pairs code/_smoke_pairs.json --out-dir code/out_smoke \
    --no-llm --sim-thr 0.27 --strategy sim_only --asr-backend qwen --no-se
  log "✅ 冒烟完成 → code/out_smoke/。确认 result.json + timing.json 产出且 5 条都跑通, 再跑全量(去 SMOKE)"
  exit 0
fi

# ---- 1. 全量 pos + neg (默认关 SE, qwen 后端) ----
if [[ "${ONLY:-}" != "neg" ]]; then
  log "pos 全量 qwen (thr0.27, 默认关SE) → out_pos_baodi"
  BAODI_BACKEND=qwen bash code/run_baodi.sh pos 0.27 2>&1 | tee /root/run_pos.log
fi
if [[ "${ONLY:-}" != "pos" ]]; then
  log "neg 全量 qwen (thr0.27) → out_neg_baodi"
  BAODI_BACKEND=qwen bash code/run_baodi.sh neg 0.27 2>&1 | tee /root/run_neg.log
fi

# ---- 2. 换算效率腿分数(timing.json + result.json → 时间腿+内存腿区间) ----
calc(){ if [[ -f "$1" ]]; then log "换算 $(basename $(dirname $1))"; $PY code/efficiency_leg_calc.py "$1" --result "${1%/*}/result.json" 2>&1 | tee "/root/efficiency_$(basename $(dirname $1)).txt"; else log "[warn] $1 不存在, 跳过"; fi; }
calc code/out_pos_baodi/timing.json
calc code/out_neg_baodi/timing.json

# ---- 3. 报告主数字(overall_rtf = 效率腿命门) ----
log "===== 主数字(overall_rtf) ====="
for s in pos neg; do
  tj=code/out_${s}_baodi/timing.json
  [[ -f "$tj" ]] && echo "  $s: overall_rtf=$($PY -c "import json;print(json.load(open('$tj')).get('overall_rtf'))" 2>/dev/null)  wall=$($PY -c "import json;print(json.load(open('$tj')).get('total_wall_sec'))" 2>/dev/null)s"
done

log "✅ 完成。拉回本机换算/入库:"
log "  scp -P <port> root@<host>:/root/midea_target_asr/code/out_{pos,neg}_baodi/{timing,result}.json ./"
log "  scp -P <port> root@<host>:/root/efficiency_*.txt ./"
log "⚠️ batch=1 口径(官方怎么测 RTF)若主办方尚未一锤定音, 可补测纯逐条 ASR:"
log "  改 enroll_infer 调 qwen_asr_backend 传 --batch-size 0(默认16), 重跑 pos 看 overall_rtf 变化"
