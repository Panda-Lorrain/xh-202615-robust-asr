#!/usr/bin/env python
"""稳定性测试 dashboard 生成器(dataviz合规 validated palette): 读 report+per_utt → 单文件 HTML(4图)。

spec §8。调色板取自 dataviz reference palette(CVD-validated): categorical 8色固定顺序 /
blue sequential / blue ordinal ramp; CSS变量 light/dark chrome; 文字用 ink token 不用 series色。
用法: code/.venv/Scripts/python.exe code/stability_dashboard.py
"""
import os, json

_HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(_HERE, "stability_matrix")

TEMPLATE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>稳定性测试 Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root { --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9; --kpi:#1a1a19; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --page:#0d0d0d; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --kpi:#000; }
}
body { font-family: system-ui,-apple-system,"Segoe UI",sans-serif; margin:24px; background:var(--page); color:var(--ink); }
h1 { font-size:20px; margin:0 0 2px; } h2 { font-size:13px; color:var(--ink2); font-weight:normal; margin:4px 0 0; }
.kpis { display:flex; gap:8px; flex-wrap:wrap; margin:14px 0; }
.kpi { background:var(--kpi); color:#fff; padding:10px 14px; border-radius:8px; }
.kpi b { font-size:18px; font-variant-numeric:tabular-nums; } .kpi span { font-size:11px; opacity:.8; margin-left:6px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
.card { background:var(--surface); padding:14px; border-radius:8px; border:1px solid var(--grid); }
.card h3 { font-size:13px; margin:0 0 10px; } .note { font-size:11px; color:var(--muted); margin-top:6px; }
</style></head><body>
<h1>稳定性/鲁棒性测试 Dashboard</h1>
<h2>B1(batch=1) vs A(batch=16) 不一致率 = 开发口径数字能否外推提交口径(主办方默认 batch=1, RTF 按 batch=1 测)</h2>
<div class="kpis">
  <div class="kpi"><b>__N_RUNS__</b><span>总 run</span></div>
  <div class="kpi"><b>__N_VOL__</b><span>波动音频</span></div>
  <div class="kpi"><b>__B1_DIFF__</b><span>B1vsA 不一致 (__B1_PCT__%)</span></div>
</div>
<div class="grid">
  <div class="card"><h3>各维度波动/不一致率(%)</h3><canvas id="dim"></canvas></div>
  <div class="card"><h3>根因分布</h3><canvas id="root"></canvas></div>
  <div class="card"><h3>波动音频 max_sim 分桶</h3><canvas id="sim"></canvas><div class="note">验证死区/低sim更易波动假设</div></div>
  <div class="card"><h3>波动音频 CER std (top20)</h3><canvas id="std"></canvas></div>
</div>
<script>
const css = getComputedStyle(document.documentElement);
const ink = getComputedStyle(document.body).color;
const grid = css.getPropertyValue('--grid').trim();
const surf = css.getPropertyValue('--surface').trim();
Chart.defaults.color = ink;
const dimRates=__DIM_RATES__, dimLabels=__DIM_LABELS__;
const rootLabels=__ROOT_LABELS__, rootData=__ROOT_DATA__, rootColors=__ROOT_COLORS__;
const simLabels=__SIM_LABELS__, simData=__SIM_DATA__, simColors=__SIM_COLORS__;
const stdLabels=__STD_LABELS__, stdData=__STD_DATA__;
const BAR='#2a78d6';
new Chart(document.getElementById('dim'),{type:'bar',data:{labels:dimLabels,datasets:[{data:dimRates,backgroundColor:BAR,borderRadius:4,maxBarThickness:48}]},options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.parsed.y+'%'}}},scales:{y:{beginAtZero:true,ticks:{callback:v=>v+'%'},grid:{color:grid}},x:{grid:{display:false}}}}});
new Chart(document.getElementById('root'),{type:'doughnut',data:{labels:rootLabels,datasets:[{data:rootData,backgroundColor:rootColors,borderColor:surf,borderWidth:2}]},options:{plugins:{legend:{position:'right',labels:{color:ink}}}}});
new Chart(document.getElementById('sim'),{type:'bar',data:{labels:simLabels,datasets:[{data:simData,backgroundColor:simColors,borderRadius:4,maxBarThickness:64}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,grid:{color:grid}},x:{grid:{display:false}}}}});
new Chart(document.getElementById('std'),{type:'bar',data:{labels:stdLabels,datasets:[{data:stdData,backgroundColor:BAR,borderRadius:4,maxBarThickness:32}]},options:{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{beginAtZero:true,grid:{color:grid}},y:{grid:{display:false}}}}});
</script></body></html>"""


def main():
    report = json.load(open(os.path.join(MATRIX, "stability_report.json"), encoding="utf-8"))
    per_utt = json.load(open(os.path.join(MATRIX, "per_utt_volatility.json"), encoding="utf-8"))

    dim_stats = report["dim_stats"]
    dim_labels = list(dim_stats.keys())
    dim_rates = [round(v.get("volatile_rate", v.get("diff_rate", 0)) * 100, 2) for v in dim_stats.values()]
    root_dist = report["root_cause_distribution"]
    buckets = {"<0.2": 0, "[0.2,0.3)": 0, "[0.3,0.4)": 0, ">=0.4": 0}
    for u in per_utt.values():
        s = u.get("max_sim", 0)
        if s < 0.2: buckets["<0.2"] += 1
        elif s < 0.3: buckets["[0.2,0.3)"] += 1
        elif s < 0.4: buckets["[0.3,0.4)"] += 1
        else: buckets[">=0.4"] += 1
    b1va = dim_stats.get("B1_vs_A(batch1vs16)", {})
    top20 = list(per_utt.values())[:20]

    # validated reference palette (dataviz): categorical 固定顺序 / blue ordinal ramp
    CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
    SIM_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#104281"]  # blue ordinal step 250/400/550/650
    n_root = max(1, len(root_dist))

    html = (TEMPLATE
            .replace("__N_RUNS__", str(report["n_runs_total"]))
            .replace("__N_VOL__", str(report["n_volatile_utts"]))
            .replace("__B1_DIFF__", f"{b1va.get('diff_n','?')}/{b1va.get('total_n','?')}")
            .replace("__B1_PCT__", f"{b1va.get('diff_rate',0)*100:.2f}")
            .replace("__DIM_RATES__", repr(dim_rates))
            .replace("__DIM_LABELS__", repr(dim_labels))
            .replace("__ROOT_LABELS__", repr(list(root_dist.keys())))
            .replace("__ROOT_DATA__", repr(list(root_dist.values())))
            .replace("__ROOT_COLORS__", repr(CAT[:n_root]))
            .replace("__SIM_LABELS__", repr(list(buckets.keys())))
            .replace("__SIM_DATA__", repr(list(buckets.values())))
            .replace("__SIM_COLORS__", repr(SIM_RAMP))
            .replace("__STD_LABELS__", repr([u["ref"][:12] for u in top20]))
            .replace("__STD_DATA__", repr([u.get("cer_std", 0) for u in top20])))
    out = os.path.join(MATRIX, "stability_dashboard.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"[done] dashboard → {out}")


if __name__ == "__main__":
    main()
