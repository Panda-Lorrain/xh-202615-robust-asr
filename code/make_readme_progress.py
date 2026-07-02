"""生成 README 进度概览 PNG（3 子图：工程迭代 CER / 重叠率退化 / 噪声死区）。
数据对齐 docs/cer_progress_dashboard.html 的 inline 数据 + eval.json。
从仓库根运行: code/.venv/Scripts/python.exe code/make_readme_progress.py
palette = dataviz 验证过的 reference palette（CVD-safe，worst adjacent ΔE=24.2）。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体（win11 必有 Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- palette (dataviz reference, light surface) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
ACCENT = "#2a78d6"        # categorical slot1 blue
ACCENT_DEEP = "#1c5cab"   # blue step 550（强调最优/最高）
CRITICAL = "#d03b3b"      # status critical（死区）

EVAL = json.load(open("code/submit_out_full/eval.json", encoding="utf-8"))

# 工程迭代 CER（全场景均值，与 dashboard dataA 一致）
ITER_LABELS = ["baseline\n裸跑", "+SE\n全局", "+langfix\n修bug", "+SE se6\n温和", "+SE\n条件化", "+精细\noracle"]
ITER_VALS = [4.274, 3.655, 3.542, 2.504, 2.609, 2.022]

# overlap 可用率
OV_LABELS = ["0%", "25%", "50%", "75%", "100%"]
OV_VALS = [EVAL["by_overlap"][k]["correct_rate"] for k in ["0.0", "0.25", "0.5", "0.75", "1.0"]]

# noise 可用率
NZ_KEYS = ["pink", "white", "babble"]
NZ_VALS = [EVAL["by_noise"][k]["correct_rate"] for k in NZ_KEYS]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=SURFACE)
fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.20, wspace=0.30)

# --- 子图1：工程迭代 CER ---
ax = axes[0]
style_ax(ax)
colors = [ACCENT] * 5 + [ACCENT_DEEP]
bars = ax.bar(ITER_LABELS, ITER_VALS, color=colors, width=0.62,
              edgecolor=SURFACE, linewidth=1.5)
ax.set_title("工程迭代 CER（全场景均值）\nbaseline 4.27 → oracle 2.02",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("CER（越低越好）", fontsize=9.5, color=MUTED)
ax.set_ylim(0, 5)
for b, v in zip(bars, ITER_VALS):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}",
            ha="center", va="bottom", fontsize=8.6, color=INK, fontweight="bold")

# --- 子图2：重叠率退化 ---
ax = axes[1]
style_ax(ax)
ov_colors = [ACCENT_DEEP] + [ACCENT] * 4
bars = ax.bar(OV_LABELS, [v * 100 for v in OV_VALS], color=ov_colors,
              width=0.6, edgecolor=SURFACE, linewidth=1.5)
ax.set_title("可用率随重叠率坍塌\nov0 41% → ov100 1%（单通道死区）",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("可用率 %（CER<0.5 占比）", fontsize=9.5, color=MUTED)
ax.set_ylim(0, 50)
ax.set_xlabel("目标说话人重叠占比", fontsize=9.5, color=MUTED)
for b, v in zip(bars, OV_VALS):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v * 100:.0f}%",
            ha="center", va="bottom", fontsize=8.6, color=INK, fontweight="bold")

# --- 子图3：噪声死区 ---
ax = axes[2]
style_ax(ax)
nz_colors = [ACCENT, ACCENT, CRITICAL]
bars = ax.bar(NZ_KEYS, [v * 100 for v in NZ_VALS], color=nz_colors,
              width=0.55, edgecolor=SURFACE, linewidth=1.5)
ax.set_title("babble 全灭（FDDT/STNO 低覆盖劣化）\npink 25% / white 17% / babble 0%",
             fontsize=10.5, color=INK, pad=8, fontweight="bold")
ax.set_ylabel("可用率 %", fontsize=9.5, color=MUTED)
ax.set_ylim(0, 32)
for b, v in zip(bars, NZ_VALS):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6, f"{v * 100:.0f}%",
            ha="center", va="bottom", fontsize=8.6, color=INK, fontweight="bold")

fig.suptitle("XH-202615 抗干扰语音指令识别 — 实测进度概览",
             fontsize=14, color=INK, fontweight="bold", y=0.965)
fig.text(0.5, 0.035,
         "450 条 mimo-tts 仿真集（非真实 A 集）｜可用率 = CER<0.5 占比，比 CER 均值更诚实"
         "（babble 幻觉拉高均值）｜RTF = 0.058 纯推理 / 0.27 端到端（RTX 4060）",
         ha="center", fontsize=7.8, color=MUTED, style="italic")

fig.savefig("docs/progress_overview.png", dpi=150, facecolor=SURFACE)
print("[saved] docs/progress_overview.png")
