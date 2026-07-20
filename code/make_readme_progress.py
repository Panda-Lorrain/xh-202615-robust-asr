"""生成 README 进度概览 PNG（真测版 3 子图，2026-07-20 重做）。

替换原仿真期版本（baseline 4.274 仿真集）。本版全部用 datasetA 全量真测，
**口径统一官方累计池**（NFKC + lower + 去所有 Unicode P* 标点空白 + editdistance 累计池）：
  - 子图1 CER 路线突破：DiCoW 条件化(1.1894) → vanilla+target(0.5947) → Qwen3-ASR(0.3436)
    源 code/recompute_official_cer.json + code/qwen_official_cer_workpoints.json
  - 子图2 sim 分桶对比：vanilla vs qwen × 死区/主战场/高 sim（从 poc_qwen rows 现算累计池）
    源 code/poc_qwen_asr_full_result.json
  - 子图3 稳定性五维波动率：R1/R2/R3/R4/R5（R1/R5=0 可复现达标，R3=57% 诚实归档）
    源 code/stability_matrix/stability_report.json

从仓库根运行: code/.venv/Scripts/python.exe code/make_readme_progress.py
palette = dataviz reference（CVD-safe，worst adjacent ΔE=24.2）。
"""
import json, unicodedata, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- palette (dataviz reference, light surface) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#2a78d6"        # categorical slot1 blue
ACCENT_DEEP = "#1c5cab"   # blue step 550（强调最优/最高）
CRITICAL = "#d03b3b"      # status critical（短板）
GOOD = "#2e8b57"          # status good（达标）

_HERE = os.path.dirname(os.path.abspath(__file__))


# ============ 官方累计池 CER（NFKC+lower+去 P* 空白，editdistance 累计池） ============
def normalize(t):
    t = unicodedata.normalize("NFKC", t or "").lower()
    return "".join(c for c in t if not unicodedata.category(c).startswith("P") and not c.isspace())


def lev(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return dp[n]


def pool_cer(preds, refs):
    """累计池 = sum(lev) / sum(len(ref))，与主办方 CERMetric 一致。"""
    err = ch = 0
    for p, r in zip(preds, refs):
        rn = normalize(r)
        if not rn: continue
        err += lev(normalize(p), rn); ch += len(rn)
    return err / ch if ch else 0.0


# ============ 子图1: CER 路线突破（官方累计池，确认数字） ============
ROUTE_LABELS = ["DiCoW 条件化\n(2026-07-04)", "vanilla+target\n(2026-07-06)", "Qwen3-ASR\n(2026-07-11)"]
ROUTE_VALS = [1.1894, 0.5947, 0.3436]

# ============ 子图2: sim 分桶官方累计池（从 poc_qwen rows 现算） ============
_d = json.load(open(os.path.join(_HERE, "poc_qwen_asr_full_result.json"), encoding="utf-8"))
_rows = _d["rows"]
_BUCKET_DEF = [("<0.2 死区", lambda s: s < 0.2),
               ("[0.2,0.4) 主战场", lambda s: 0.2 <= s < 0.4),
               (">=0.4 接近解决", lambda s: s >= 0.4)]
BUCKET_LABELS, VANILLA_POOL, QWEN_POOL, BUCKET_N = [], [], [], []
for _label, _pred in _BUCKET_DEF:
    _sub = [r for r in _rows if _pred(float(r.get("sim", 0) or 0))]
    BUCKET_LABELS.append(_label)
    VANILLA_POOL.append(pool_cer([r.get("vanilla", "") for r in _sub], [r.get("ref", "") for r in _sub]))
    QWEN_POOL.append(pool_cer([r.get("qwen", "") for r in _sub], [r.get("ref", "") for r in _sub]))
    BUCKET_N.append(len(_sub))

# ============ 子图3: 稳定性五维波动率（源 stability_matrix/stability_report.json） ============
STAB_LABELS = ["R1 GPU非确定\n(同种子×10)", "R2 batch桥\n(1 vs 16)", "R3 输入微扰\n(gauss主因)",
               "R4 声纹锁定\n(enroll-aug)", "R5 变种子\n(5种子×2)"]
STAB_VALS = [0.0, 5.43, 57.11, 7.26, 0.0]  # %


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=SURFACE)
fig.subplots_adjust(left=0.05, right=0.985, top=0.74, bottom=0.20, wspace=0.32)

# --- 子图1: CER 路线突破 ---
ax = axes[0]; style_ax(ax)
colors = [CRITICAL, ACCENT, ACCENT_DEEP]
bars = ax.bar(ROUTE_LABELS, ROUTE_VALS, color=colors, width=0.6, edgecolor=SURFACE, linewidth=1.5)
ax.set_title("CER 路线突破（官方累计池, zero-training）\nDiCoW 1.19 → vanilla 0.59 → qwen 0.34",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("CER（越低越好）", fontsize=9.5, color=MUTED); ax.set_ylim(0, 1.4)
for b, v in zip(bars, ROUTE_VALS):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")

# --- 子图2: sim 分桶对比 ---
ax = axes[1]; style_ax(ax)
x = np.arange(len(BUCKET_LABELS)); w = 0.36
b1 = ax.bar(x - w / 2, VANILLA_POOL, w, color=MUTED, edgecolor=SURFACE, linewidth=1.2, label="vanilla")
b2 = ax.bar(x + w / 2, QWEN_POOL, w, color=ACCENT_DEEP, edgecolor=SURFACE, linewidth=1.2, label="Qwen3-ASR")
ax.set_title("各 sim 桶 CER 对比（官方累计池）\nqwen 各桶均提升，死区 OOD 伪地板突破",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("CER（越低越好）", fontsize=9.5, color=MUTED); ax.set_ylim(0, 1.05)
ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(BUCKET_LABELS, BUCKET_N)], fontsize=8.4)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
for bars_grp in (b1, b2):
    for b in bars_grp:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8, color=INK)

# --- 子图3: 稳定性五维波动率 ---
ax = axes[2]; style_ax(ax)
colors = [GOOD if v == 0 else (CRITICAL if v > 20 else ACCENT) for v in STAB_VALS]
bars = ax.bar(STAB_LABELS, STAB_VALS, color=colors, width=0.6, edgecolor=SURFACE, linewidth=1.5)
ax.set_title("稳定性五维波动率（26 遍全量 1364 条）\nR1/R5=0 可复现达标 · R3=57% 诚实归档",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("波动率 %（越低越稳）", fontsize=9.5, color=MUTED); ax.set_ylim(0, 65)
for b, v in zip(bars, STAB_VALS):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=8.8, color=INK, fontweight="bold")

fig.suptitle("XH-202615 抗干扰语音指令识别 — 真测进度概览（datasetA 全量）",
             fontsize=14, color=INK, fontweight="bold", y=0.965)
fig.text(0.5, 0.035,
         "datasetA 全量真测（1364 pos / 474 neg）｜官方累计池 CER（NFKC + lower + 去 P*）｜"
         "当前算分 qwen+content_gate: CER 腿 15.32 + RR 腿 37.98 = 53.3/80（w1=w2=0.4 假设），效率腿待 L20 batch=1 RTF",
         ha="center", fontsize=7.5, color=MUTED, style="italic")

_out = os.path.join(_HERE, "..", "docs", "progress_overview.png")
fig.savefig(_out, dpi=150, facecolor=SURFACE)
print(f"[saved] {_out}")
print(f"[子图2 分桶 vanilla/qwen/n] " + str(list(zip(BUCKET_LABELS,
      [round(v, 3) for v in VANILLA_POOL], [round(q, 3) for q in QWEN_POOL], BUCKET_N))))
