#!/usr/bin/env bash
# 保底提交 wrapper —— 锁死关LLM + thr + sim_only，防 submit_infer 默认 flag 灾难。
# 对抗审查 GAP3（memory baodi-config-no-llm）：submit_infer 默认 strategy=llm_or_sim /
#   sim_thr=0.2 / llm=ON → RTF 1.0 + neg RR 0.77 两腿崩。保底必须显式 --no-llm --sim-thr 0.4 --strategy sim_only。
# 用法：
#   bash code/run_baodi.sh pos        # pos 全量，thr=0.4（默认）
#   bash code/run_baodi.sh neg 0.45   # neg 全量，thr=0.45（within-noise 占优，RR 99.2%）
# 评测：
#   code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_<set>_baodi/result.json code/<set>_pairs_datasetA.json
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "cd 失败"; exit 1; }
source code/setenv.sh
export HF_HUB_OFFLINE=1

SET="${1:?用法: run_baodi.sh pos|neg [thr(默认0.4)]}"
THR="${2:-0.4}"

# thr 范围校验（防误传 2.0/-1 导致全 utt 同样被拒/放，result.json 静默错误输出）
if ! awk "BEGIN{exit !($THR>=0 && $THR<=1)}" 2>/dev/null; then
  echo "[error] THR 必须 [0,1]，当前 '$THR'"; exit 1
fi

case "$SET" in
  pos) PAIRS=code/pos_pairs_datasetA.json; OUT=code/out_pos_baodi ;;
  neg) PAIRS=code/neg_pairs_datasetA.json; OUT=code/out_neg_baodi ;;
  *) echo "未知集合 '$SET'（应为 pos 或 neg）"; exit 1 ;;
esac

# pairs 存在性校验（缺失则 submit_infer.py L48 FileNotFoundError，此处提前友好报错）
if [[ ! -f "$PAIRS" ]]; then
  echo "[error] pairs 文件不存在: $PAIRS"; exit 1
fi

echo "[baodi] 关LLM(--no-llm) + thr=$THR + strategy=sim_only  → $OUT  (保底：防 submit_infer 默认 flag 灾难)"
exec code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs "$PAIRS" --out-dir "$OUT" --no-llm --sim-thr "$THR" --strategy sim_only
