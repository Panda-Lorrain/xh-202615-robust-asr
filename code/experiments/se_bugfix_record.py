#!/usr/bin/env python
"""后台编排: 等 SE bugfix run 完成 → 跑评测+对比 → 写结果文档。
独立于 Claude Code API 运行(用户要关服务), 机器开着即自动完成。
2026-07-18 SE orphaned-bug bugfix A/B 记录。
"""
import json, os, subprocess, time, statistics, sys

ROOT = r"E:\midea_target_asr"
SE_RESULT = os.path.join(ROOT, "code", "out_pos_SE_bugfixed_1364", "result.json")
NO_RESULT = os.path.join(ROOT, "code", "out_pos_noSE_1364", "result.json")
PAIRS = os.path.join(ROOT, "code", "pos_pairs_datasetA.json")
VENV_PY = os.path.join(ROOT, "code", ".venv", "Scripts", "python.exe")
EVAL_SCRIPT = os.path.join(ROOT, "code", "eval_datasetA.py")
COMPARE_SCRIPT = os.path.join(ROOT, "code", "compare_se_bugfix.py")
DOC = os.path.join(ROOT, "docs", "SE_bugfix_AB结果_2026-07-18.md")

# 1. 等 bugfix run 完成(最多等 40 分钟)
print(f"[wait] 等待 {SE_RESULT} ...", flush=True)
waited = 0
while not os.path.exists(SE_RESULT):
    time.sleep(30)
    waited += 30
    if waited > 2400:
        print("[timeout] 40 分钟未出 result, 放弃", flush=True)
        sys.exit(1)
print(f"[wait] result.json 出现 (等了 {waited}s)", flush=True)
time.sleep(10)  # 等文件写完

# 2. 跑 CER 评测(两个 run 都评)
def run_cer(result_json, label):
    try:
        out = subprocess.check_output([VENV_PY, EVAL_SCRIPT, result_json, PAIRS],
                                      stderr=subprocess.STDOUT, cwd=ROOT, timeout=120,
                                      encoding="utf-8", errors="replace")
        # 解析关键行
        lines = {}
        for line in out.split("\n"):
            if ":" in line and any(k in line for k in ["overall CER", "correct_rate", "near_perfect", "仅 accepted", "误拒率"]):
                lines[line.split(":")[0].strip()] = line.split(":", 1)[1].strip()
        print(f"[eval] {label}: {lines}", flush=True)
        return lines, out
    except Exception as e:
        print(f"[eval] {label} FAIL: {e}", flush=True)
        return {}, str(e)

se_cer, se_out = run_cer(SE_RESULT, "SE(bugfixed)")
no_cer, no_out = run_cer(NO_RESULT, "noSE")

# 3. 跑对比(sim/拒识/文本差异)
def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return d.get("results", d) if isinstance(d, dict) else d
def uid(r):
    return os.path.splitext(os.path.basename(r.get("recognition", "")))[0]

try:
    se = {uid(r): r for r in load(SE_RESULT)}
    no = {uid(r): r for r in load(NO_RESULT)}
    common = set(se) & set(no)
    sim_diff = [se[u].get("max_sim", 0) - no[u].get("max_sim", 0) for u in common]
    sim_changed = sum(1 for d in sim_diff if abs(d) > 1e-6)
    se_rej = sum(1 for u in common if se[u].get("rejected"))
    no_rej = sum(1 for u in common if no[u].get("rejected"))
    cross = sum(1 for u in common if se[u].get("rejected") != no[u].get("rejected"))
    text_diff = sum(1 for u in common
                    if not se[u].get("rejected") and not no[u].get("rejected")
                    and se[u].get("text", "") != no[u].get("text", ""))  # 修字段名 bug(原 'transcript'→'text')
except Exception as e:
    sim_diff, sim_changed, se_rej, no_rej, cross, text_diff = [], 0, 0, 0, 0, 0
    common = set()
    print(f"[compare] FAIL: {e}", flush=True)

# 4. 读 timing
def read_rtf(d):
    tj = os.path.join(os.path.dirname(d), "timing.json")
    if os.path.exists(tj):
        t = json.load(open(tj, encoding="utf-8"))
        return t.get("overall_rtf"), t.get("total_wall_sec"), t.get("phases", {})
    return None, None, {}

se_rtf, se_wall, se_phases = read_rtf(SE_RESULT)
no_rtf, no_wall, no_phases = read_rtf(NO_RESULT)

# 5. 决策
try:
    se_overall = float(se_cer.get("overall CER", "999").split()[0])
    no_overall = float(no_cer.get("overall CER", "999").split()[0])
except Exception:
    se_overall = no_overall = None

if se_overall is not None and no_overall is not None:
    delta = se_overall - no_overall  # 负=SE更好
    if delta < -0.02:
        decision = "🔴 SE 真生效显著降 CER → 建议用 SE(去 --no-se 默认, BAODI_SE=1)"
    elif delta > 0.02:
        decision = "⛔ SE 真生效反而恶化 CER → 保持 --no-se"
    else:
        decision = "🟡 SE 真生效对 CER 无显著影响 → 保持 --no-se(省 27% RTF)"
else:
    decision = "⚠️ CER 解析失败, 需人工核对"

# 6. 写文档
os.makedirs(os.path.dirname(DOC), exist_ok=True)
md = f"""# SE orphaned-bug bugfix A/B 结果 (2026-07-18)

## 背景: 发现并修复 SE 输出从未被使用的潜伏 bug
- **Bug**: `submit_infer.py` 的 `rec_for_enroll` 变量赋值后从未读取, enroll_infer 始终读 `rec_in`(原始音频), SE 输出 `se_out` 是孤儿目录 → **SE 全程空转(27% RTF 白烧), 从 vanilla/qwen 后端启用起就如此**。
- **修复**: SE 生效时把 enroll_pairs 的 recognition 路径重映射到 `se_out`, 让 enroll_infer 真正读到降噪音频。
- **影响**: 上个模型 "--no-se 零 CER 影响 / 50 条字节一致 / sim 0 差异" 均由此 bug 决定(SE 本就没生效)。

## A/B 结果 (qwen 后端, thr=0.27, 全量 1364 条 pos)

| 指标 | SE(bugfixed, 真生效) | noSE | 差异 |
|---|---|---|---|
| overall CER | {se_cer.get('overall CER','?')} | {no_cer.get('overall CER','?')} | {f'{se_overall-no_overall:+.4f}' if se_overall and no_overall else '?'} |
| cer(仅accepted) | {se_cer.get('cer (仅 accepted)','?')} | {no_cer.get('cer (仅 accepted)','?')} | - |
| correct_rate | {se_cer.get('correct_rate (CER<0.5)','?')} | {no_cer.get('correct_rate (CER<0.5)','?')} | - |
| 误拒率 | {se_cer.get('误拒率 (pos 被拒,伤CER)','?')} | {no_cer.get('误拒率 (pos 被拒,伤CER)','?')} | - |
| 拒识数 | {se_rej}/{len(common)} | {no_rej}/{len(common)} | 翻转{cross}条 |
| overall RTF | {se_rtf:.4f} | {no_rtf:.4f} | {f'{(se_rtf-no_rtf)/no_rtf*100:+.0f}%' if se_rtf and no_rtf else '?'} |
| wall(s) | {se_wall:.0f} | {no_wall:.0f} | - |

### sim/文本变化
- max_sim 差异(SE-noSE): mean={statistics.mean(sim_diff):+.4f} std={statistics.stdev(sim_diff):.4f} (有变化 {sim_changed}/{len(sim_diff)})
- 两边都 accepted 的条里文本不一致: {text_diff} 条

## 决策
{decision}

## 说明
- 此文档由 `code/se_bugfix_record.py` 在 SE bugfix run 完成后**自动生成**(独立于 API 服务)。
- 原始评测输出见 `code/out_pos_SE_bugfixed_1364/_cer_eval.txt`(若需人工复核)。
- bugfix commit: 见 `git log` (submit_infer.py SE 重映射)。
"""
with open(DOC, "w", encoding="utf-8") as f:
    f.write(md)
print(f"[done] 文档已写: {DOC}", flush=True)
print(f"[decision] {decision}", flush=True)
