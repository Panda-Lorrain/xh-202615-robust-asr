#!/usr/bin/env bash
# 保底提交 wrapper —— 锁死关LLM + thr + sim_only,防 submit_infer 默认 flag 灾难。
# 对抗审查 GAP3(memory baodi-config-no-llm): submit_infer 默认 strategy=llm_or_sim /
#   sim_thr=0.2 / llm=ON → RTF 1.0 + neg RR 0.77 两腿崩。保底必须显式 --no-llm --sim-thr --strategy sim_only。
#
# 三种模式:
#   bash code/run_baodi.sh pos [thr]      # A 集 pos, thr 默认 0.4(A 集分 thr 初评)
#   bash code/run_baodi.sh neg [thr]      # A 集 neg, thr 默认 0.4(0.45→RR 99.2%)
#   bash code/run_baodi.sh B   [thr]      # B 集(混合 pos/neg 无 label), 统一 thr 默认 0.27
#   BAODI_BACKEND=dicow bash code/run_baodi.sh pos        # 切 dicow fallback/答辩对比基线
#   BAODI_PAIRS=xxx bash code/run_baodi.sh B 0.27         # 自定义 B 集 pairs 路径
#
# ⚠️ 统一 thr=0.27(T27 对抗验证推荐, bootstrap CI 区间[0.26,0.29], 细扫真峰0.275):
#    B 集按 FAQ C9 不预分 pos/neg → 必须单一 thr。0.27 在 40:40 线性估算下最优
#    (与 0.28 不可分, 取 0.27 因 pos 侧占优 + 抗 babble 下移)。
#    thr 待主办方口径定: RR-heavy→0.35-0.40 / CER-heavy 或 pos不许拒→0。
#    见 RESULTS.md T27 + scan_unified_thr.json。
#
# 评测(A 集; B 集无 ref 主办方算):
#   code/.venv/Scripts/python.exe code/eval_datasetA.py code/out_<set>_baodi/result.json code/<set>_pairs_datasetA.json
set -euo pipefail
cd "$(dirname "$0")/.." || { echo "cd 失败"; exit 1; }
source code/setenv.sh
export HF_HUB_OFFLINE=1

SET="${1:?用法: run_baodi.sh pos|neg|B [thr]}"
# B 集(混合)用统一 thr 默认 0.27(T27 推荐); pos/neg(A 集分 thr)默认 0.4
if [[ "$SET" == "B" || "$SET" == "mixed" ]]; then
  THR="${2:-0.27}"
else
  THR="${2:-0.4}"
fi
# vanilla 作提交主线(CER 0.664 减半, 反 cascaded); BAODI_BACKEND=dicow 切回 fallback/答辩对比基线
BACKEND="${BAODI_BACKEND:-vanilla}"

# thr 范围校验(防误传 2.0/-1 导致全 utt 同样被拒/放, result.json 静默错误输出)
if ! awk "BEGIN{exit !($THR>=0 && $THR<=1)}" 2>/dev/null; then
  echo "[error] THR 必须 [0,1]，当前 '$THR'"; exit 1
fi

case "$SET" in
  pos) PAIRS=code/pos_pairs_datasetA.json; OUT=code/out_pos_baodi ;;
  neg) PAIRS=code/neg_pairs_datasetA.json; OUT=code/out_neg_baodi ;;
  B|mixed) PAIRS="${BAODI_PAIRS:-code/B_pairs_datasetA.json}"
           OUT="${BAODI_OUT:-code/out_B_baodi}"
           echo "[baodi] ⚠️ B 集混合模式: 统一 thr=$THR (T27 推荐 0.27, bootstrap CI [0.26,0.29])"
           if [[ ! -f "$PAIRS" ]]; then
             echo "[error] B 集 pairs 不存在: $PAIRS"
             echo "        B 集到手后用 make_pairs_from_datasetB.py 生成(无 ref 混合 manifest, pos/neg 不作输入),"
             echo "        或 BAODI_PAIRS=<path> bash code/run_baodi.sh B $THR 指定"
             exit 1
           fi ;;
  *) echo "未知集合 '$SET'（应为 pos/neg/B）"; exit 1 ;;
esac

# pairs 存在性校验(pos/neg; B 已在上面校验)
if [[ "$SET" != "B" && "$SET" != "mixed" && ! -f "$PAIRS" ]]; then
  echo "[error] pairs 文件不存在: $PAIRS"; exit 1
fi

# run_baodi 是受控入口(锁 --no-llm/--sim-thr/--strategy), BAODI_OK=1 opt-in 绕过
# submit_infer 保底守卫(守卫防裸调默认 flag 灾难; 统一 thr=0.27 < 守卫阈值 0.35,
# 故 B 集统一 thr 必须经 run_baodi 或显式 BAODI_OK=1, 已由 T27 对抗验证背书)
export BAODI_OK=1

echo "[baodi] backend=$BACKEND 关LLM(--no-llm) + thr=$THR + strategy=sim_only  → $OUT  (vanilla 主线 / BAODI_BACKEND=dicow 切回)"
exec code/.venv/Scripts/python.exe code/submit_infer.py \
  --pairs "$PAIRS" --out-dir "$OUT" --no-llm --sim-thr "$THR" --strategy sim_only \
  --asr-backend "$BACKEND"
