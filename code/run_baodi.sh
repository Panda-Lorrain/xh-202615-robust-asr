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
# 跨平台 setenv + 主 venv python(原硬编码 Win .venv/Scripts/python.exe, Linux/L20 阻塞)
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* || "${OS:-}" == "Windows_NT" ]]; then
  source code/setenv.sh
  PY="${PY:-code/.venv/Scripts/python.exe}"
else
  source code/setenv_linux.sh
  PY="${PY:-code/.venv/bin/python}"
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}  # 默认离线(本地已下权重); 联网下权重前 HF_HUB_OFFLINE=0

SET="${1:?用法: run_baodi.sh pos|neg|B [thr]}"
# B 集(混合)用统一 thr 默认 0.27(T27 推荐); pos/neg(A 集分 thr)默认 0.4
if [[ "$SET" == "B" || "$SET" == "mixed" ]]; then
  THR="${2:-0.27}"
else
  THR="${2:-0.4}"
fi
# qwen 作提交主线(07-11 起主线已切, CER 0.6201); BAODI_BACKEND=vanilla 切回早期保底基线, =dicow 答辩对比
BACKEND="${BAODI_BACKEND:-qwen}"

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

# content_gate: 默认开(2026-07-18 反转原决策). qwen后端joint验证净正+0.826(w1=w2=0.4, 扩词后实测):
#   neg RR 0.9051→0.9494(+1.77腿分, gate拒21条漏拒:女排亚俱杯/信访/租赁物业/卖家协商等非家居)
#   pos CER 0.5934→0.6171(-0.95腿分, 误拒35条pos多为CER≥1反赚). verify_content_gate_joint.py实测.
#   对 w1/w2 鲁棒: 净正只需 w2/w1>0.53, 官方 w1=w2=0.4 远满足(待主办方确认权重不翻车).
#   原"qwen后端gate恶化Δ+0.024"是pos-only评估(poc_content_gate_v2_qwen_eval漏neg侧); vanilla hold-out val+1.6证泛化→B集同开.
#   BAODI_GATE=0 显式关(回退对比).
GATE_FLAG="--content-gate"
if [[ "${BAODI_GATE:-1}" == "0" ]]; then GATE_FLAG=""; fi
# --no-se(2026-07-18): qwen 后端跳过 DeepFilterNet3 语音增强。SE 占 ~30.6% RTF(timing.json
# 219.8s/718.2s 实测)。bugfix(commit c8c739d)前 SE 输出 se_out 是孤儿目录从未被消费 →
# "50 条 A/B 零差异"实因 SE 两分支都空转, 非 qwen 鲁棒(旧注释错归因已废弃)。bugfix 后 SE
# 真生效, 全量 A/B(1364 pos, thr0.27) SE 反而 overall CER +0.1049(三机制: sim mismatch
# 误拒 66% + DF3 过衰减致 diar 崩溃 22% + 转写恶化 12%) + RTF +45% → qwen 主线关 SE。
# 详见 docs/SE_bugfix_AB结果_2026-07-18.md + code/audit_se_bugfix.{py,json}。
# BAODI_SE=1 可恢复 SE(⚠️ 仅 qwen 后端做过 A/B; vanilla/dicow 后端 SE 效果未测)。
SE_FLAG="--no-se"
if [[ "${BAODI_SE:-0}" == "1" ]]; then SE_FLAG=""; fi
echo "[baodi] backend=$BACKEND 关LLM(--no-llm) + thr=$THR + strategy=sim_only + content_gate=${BAODI_GATE:-1} + se=${BAODI_SE:-off}  → $OUT"
exec "$PY" code/submit_infer.py \
  --pairs "$PAIRS" --out-dir "$OUT" --no-llm --sim-thr "$THR" --strategy sim_only \
  --asr-backend "$BACKEND" $GATE_FLAG $SE_FLAG
