"""生成 vanilla Whisper CER 分布检视界面 HTML (2026-07-10)。

数据源: exp_vanilla_full.json (全量 1364 per-sample)。
输出: code/cer_distribution_viewer.html (self-contained, 双击打开, 无后端)。

两区域:
  A. 顶部 CER 分档柱状图(CSS div 宽度) + 关键统计行
  B. 下方逐条检视: uid/档badge/CER/max_sim/ref(绿)/vanilla_text(红)/双音频控件
     交互: 档筛选 / uid 搜索 / CER 升降序 / 分页(50/页)
     只读检视(无标注), 重点分布 + ref vs 模型输出对比。

音频: file:///E:/midea_target_asr/datasetA/pos/<uid>.wav (recognition)
      file:///E:/midea_target_asr/datasetA/pos/kws_<N>.wav (enrollment, uid cmd→kws)
"""
import json, os, statistics

DATA_JSON = r"E:\midea_target_asr\code\exp_vanilla_full.json"
OUT = r"E:\midea_target_asr\code\cer_distribution_viewer.html"
POS_DIR = r"E:\midea_target_asr\datasetA\pos"

# ---- 分档定义(顺序即展示顺序) ----
BUCKETS = [
    ("perfect", "满分 CER=0",   "#2e7d32"),  # 深绿
    ("mild",    "轻微 0-0.1",   "#66bb6a"),  # 浅绿
    ("medium",  "中等 0.1-0.5", "#fdd835"),  # 黄
    ("severe",  "严重 0.5-1",   "#fb8c00"),  # 橙
    ("dead",    "死区 1-2",     "#e53935"),  # 红
    ("extreme", "极重 >2",      "#8b0000"),  # 深红
]


def bucket_of(cer):
    if cer <= 0:
        return "perfect"
    if cer <= 0.1:
        return "mild"
    if cer <= 0.5:
        return "medium"
    if cer <= 1:
        return "severe"
    if cer <= 2:
        return "dead"
    return "extreme"


def main():
    raw = json.load(open(DATA_JSON, encoding="utf-8"))
    n = len(raw)

    rows = []
    for r in raw:
        uid = r.get("uid", "")
        cer = float(r.get("vanilla_cer") or 0)
        sim = r.get("max_sim")
        sim = float(sim) if sim is not None else None
        rec = "file:///" + (POS_DIR + "\\" + uid + ".wav").replace("\\", "/")
        enw_uid = uid.replace("cmd", "kws")
        enw = "file:///" + (POS_DIR + "\\" + enw_uid + ".wav").replace("\\", "/")
        rows.append({
            "uid": uid,
            "ref": r.get("ref", "") or "",
            "hyp": (r.get("vanilla_text", "") or "")[:500],
            "cer": round(cer, 4),
            "sim": (round(sim, 4) if sim is not None else None),
            "bk": bucket_of(cer),
            "rec": rec,
            "enw": enw,
        })

    # ---- 分档统计 ----
    from collections import Counter
    cc = Counter(r["bk"] for r in rows)
    bucket_stats = []  # (key, label, color, count, pct, width%)
    for key, label, color in BUCKETS:
        c = cc[key]
        pct = c / n * 100
        bucket_stats.append((key, label, color, c, pct, pct))
    max_pct = max((c / n * 100) for _, _, _, c, _, _ in bucket_stats) or 1

    cers = [r["cer"] for r in rows]
    sims_valid = [r["sim"] for r in rows if r["sim"] is not None]
    mean_cer = statistics.mean(cers)
    median_cer = statistics.median(cers)
    perfect = cc["perfect"]
    perfect_rate = perfect / n * 100
    dead_sim = sum(1 for s in sims_valid if s < 0.2)
    dead_sim_rate = dead_sim / n * 100

    # ---- 柱状图 HTML ----
    bars = []
    for key, label, color, c, pct, _ in bucket_stats:
        w = pct / max_pct * 100  # 相对最宽档归一, 视觉更清楚
        bars.append(f"""
      <div class="bar-row">
        <div class="bar-label" style="border-left:6px solid {color}">{label}</div>
        <div class="bar-track"><div class="bar-fill" style="width:{w:.2f}%;background:{color}"></div></div>
        <div class="bar-count">{c} <span class="bar-pct">({pct:.1f}%)</span></div>
      </div>""")
    bars_html = "\n".join(bars)

    # ---- 统计行 ----
    stats_html = f"""
    <div class="stats">
      <div class="stat"><div class="stat-num">{n}</div><div class="stat-lbl">总条数</div></div>
      <div class="stat"><div class="stat-num">{perfect} <span class="unit">({perfect_rate:.1f}%)</span></div><div class="stat-lbl">满分率 (CER=0)</div></div>
      <div class="stat"><div class="stat-num">{mean_cer:.4f}</div><div class="stat-lbl">CER 均值</div></div>
      <div class="stat"><div class="stat-num">{median_cer:.4f}</div><div class="stat-lbl">CER 中位数</div></div>
      <div class="stat"><div class="stat-num">{dead_sim} <span class="unit">({dead_sim_rate:.1f}%)</span></div><div class="stat-lbl">max_sim&lt;0.2 (死区声纹)</div></div>
    </div>"""

    # ---- 筛选下拉选项 ----
    filt_opts = '<option value="">全部档 ({})</option>'.format(n)
    for key, label, color, c, pct, _ in bucket_stats:
        filt_opts += f'\n      <option value="{key}">{label} ({c}, {pct:.1f}%)</option>'

    bucket_map_js = {key: {"label": label, "color": color} for key, label, color in BUCKETS}

    # ---- HTML 模板 ----
    TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>vanilla Whisper CER 分布检视</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#f4f5f7;color:#222;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:18px}
.header{background:#fff;border-radius:8px;padding:14px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.header h1{margin:0 0 4px;font-size:18px}
.header .sub{color:#666;font-size:13px}
.section{background:#fff;border-radius:8px;padding:16px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.section h2{margin:0 0 12px;font-size:15px;color:#444;border-left:4px solid #1976d2;padding-left:8px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:8px}
.stat{flex:1;min-width:140px;background:#f8f9fb;border-radius:6px;padding:10px 12px;text-align:center}
.stat-num{font-size:20px;font-weight:700;color:#1976d2}
.stat-num .unit{font-size:13px;font-weight:400;color:#888}
.stat-lbl{font-size:12px;color:#666;margin-top:2px}
.chart{margin-top:6px}
.bar-row{display:flex;align-items:center;gap:8px;margin:5px 0}
.bar-label{width:120px;font-size:13px;padding:3px 6px;border-radius:3px;background:#f0f0f0}
.bar-track{flex:1;height:24px;background:#eee;border-radius:4px;overflow:hidden}
.bar-fill{height:100%;border-radius:4px;transition:width .4s}
.bar-count{width:110px;font-size:13px;text-align:right}
.bar-count .bar-pct{color:#888;font-size:12px}
.toolbar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:12px}
.toolbar label{font-size:13px;color:#555}
.toolbar select,.toolbar input{padding:5px 8px;border:1px solid #ccc;border-radius:4px;font-size:13px}
.toolbar input{flex:1;min-width:160px}
.btn{padding:5px 12px;border:1px solid #888;background:#fff;border-radius:4px;cursor:pointer;font-size:13px}
.btn:hover{background:#f0f0f0}
.btn.active{background:#1976d2;color:#fff;border-color:#1976d2}
.pager{display:flex;justify-content:space-between;align-items:center;margin:10px 0;font-size:13px;color:#555}
.list{display:flex;flex-direction:column;gap:12px}
.item{border:1px solid #e0e0e0;border-radius:6px;padding:12px 14px;background:#fff}
.item-head{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:8px}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;color:#fff;font-size:12px;font-weight:600;white-space:nowrap}
.uid{font-weight:700;font-family:Consolas,monospace;font-size:13px}
.metric{font-size:13px;color:#555}
.metric b{color:#1976d2}
.field{padding:7px 10px;border-radius:3px;margin:4px 0;font-size:14px;word-break:break-all}
.field .flbl{font-size:12px;color:#777;font-weight:700;margin-right:4px}
.ref-field{background:#e8f5e9;border-left:4px solid #4caf50}
.hyp-field{background:#ffebee;border-left:4px solid #f44336}
.audio-row{display:flex;gap:12px;margin-top:8px;flex-wrap:wrap}
.audio-cell{flex:1;min-width:260px}
.audio-cell .albl{font-size:12px;color:#777;margin-bottom:2px}
.audio-cell audio{width:100%}
.empty{padding:30px;text-align:center;color:#999}
.hint{color:#999;font-size:12px;margin-top:6px}
</style></head><body><div class="wrap">

<div class="header">
  <h1>vanilla Whisper CER 分布检视 — datasetA pos 全量</h1>
  <div class="sub">vanilla Whisper-large-v3-turbo + 声纹切 target timeline 路线。CER 越低越好(0=完美转写), max_sim 越高=目标声纹越清晰锁住。</div>
</div>

<div class="section">
  <h2>A. CER 分档分布总览</h2>
  <div class="chart">__BARS__</div>
  __STATS__
</div>

<div class="section">
  <h2>B. 逐条检视</h2>
  <div class="toolbar">
    <label>档筛选:<select id="filt" onchange="resetPage()"><__FILT__></select></label>
    <label>排序:
      <button class="btn active" id="sortAsc" onclick="setSort(true)">CER 升序 ▲</button>
      <button class="btn" id="sortDesc" onclick="setSort(false)">CER 降序 ▼</button>
    </label>
    <label>搜索 uid:<input id="search" type="text" placeholder="如 cmd_2788" oninput="resetPage()"></label>
    <span class="hint">每页 50 条</span>
  </div>
  <div class="pager">
    <span id="info"></span>
    <span>
      <button class="btn" onclick="prevPage()">◀ 上一页</button>
      <button class="btn" onclick="nextPage()">下一页 ▶</button>
    </span>
  </div>
  <div class="list" id="list"></div>
  <div class="pager">
    <span id="info2"></span>
    <span>
      <button class="btn" onclick="prevPage()">◀ 上一页</button>
      <button class="btn" onclick="nextPage()">下一页 ▶</button>
    </span>
  </div>
</div>

</div>
<script>
const DATA = __DATA__;
const BK = __BK__;
const PAGE = 50;
let sortAsc = true, page = 0;

function esc(s){
  if(s==null) s='';
  return String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function getView(){
  const filt = document.getElementById('filt').value;
  const q = document.getElementById('search').value.trim().toLowerCase();
  let v = DATA.filter(d => !filt || d.bk === filt);
  if(q) v = v.filter(d => d.uid.toLowerCase().indexOf(q) >= 0);
  v = v.slice().sort((a,b)=> sortAsc ? a.cer-b.cer : b.cer-a.cer);
  return v;
}
function render(){
  const v = getView();
  const total = v.length;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  if(page >= pages) page = pages - 1;
  if(page < 0) page = 0;
  const start = page * PAGE;
  const slice = v.slice(start, start + PAGE);
  const list = document.getElementById('list');
  if(!slice.length){
    list.innerHTML = '<div class="empty">无匹配条目</div>';
  } else {
    list.innerHTML = slice.map(d => {
      const b = BK[d.bk];
      const simTxt = (d.sim===null||d.sim===undefined) ? '<i>null</i>' : d.sim;
      const hypTxt = d.hyp ? esc(d.hyp) : '<i style="color:#999">(空)</i>';
      return '<div class="item">'
        + '<div class="item-head">'
        +   '<span class="badge" style="background:'+b.color+'">'+b.label+'</span>'
        +   '<span class="uid">'+esc(d.uid)+'</span>'
        +   '<span class="metric">CER <b>'+d.cer+'</b></span>'
        +   '<span class="metric">max_sim <b>'+simTxt+'</b></span>'
        + '</div>'
        + '<div class="field ref-field"><span class="flbl">ref (正确答案):</span>'+esc(d.ref)+'</div>'
        + '<div class="field hyp-field"><span class="flbl">vanilla_text (模型输出):</span>'+hypTxt+'</div>'
        + '<div class="audio-row">'
        +   '<div class="audio-cell"><div class="albl">recognition (带噪识别音频)</div><audio controls preload="none" src="'+d.rec+'"></audio></div>'
        +   '<div class="audio-cell"><div class="albl">enrollment (目标唤醒词, 音色参考)</div><audio controls preload="none" src="'+d.enw+'"></audio></div>'
        + '</div>'
        + '</div>';
    }).join('');
  }
  const infoTxt = '第 '+(start+1)+'–'+Math.min(start+PAGE,total)+' 条 / 共 '+total+' 条 (第 '+(page+1)+'/'+pages+' 页)';
  document.getElementById('info').textContent = total ? infoTxt : '无匹配';
  document.getElementById('info2').textContent = total ? infoTxt : '';
}
function resetPage(){ page = 0; render(); }
function setSort(asc){ sortAsc = asc; document.getElementById('sortAsc').classList.toggle('active',asc); document.getElementById('sortDesc').classList.toggle('active',!asc); page = 0; render(); }
function prevPage(){ if(page>0){page--;render();} }
function nextPage(){ const pages=Math.max(1,Math.ceil(getView().length/PAGE)); if(page<pages-1){page++;render();} }
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT') return;
  if(e.key==='ArrowRight'){e.preventDefault();nextPage();}
  if(e.key==='ArrowLeft'){e.preventDefault();prevPage();}
});
render();
</script></body></html>"""

    html = (TPL
            .replace("__BARS__", bars_html)
            .replace("__STATS__", stats_html)
            .replace("__FILT__", filt_opts)
            .replace("__DATA__", json.dumps(rows, ensure_ascii=False))
            .replace("__BK__", json.dumps(bucket_map_js, ensure_ascii=False)))

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    # ---- 终端报告 ----
    print(f"HTML -> {OUT} ({len(rows)} 条嵌入)")
    print(f"size: {os.path.getsize(OUT)/1024:.1f} KB")
    print("--- 分档统计 ---")
    for key, label, color, c, pct, _ in bucket_stats:
        print(f"  {label}: {c} 条 ({pct:.1f}%)")
    print(f"总条数: {n}")
    print(f"满分率: {perfect_rate:.2f}%")
    print(f"CER 均值: {mean_cer:.4f} / 中位数: {median_cer:.4f}")
    print(f"max_sim<0.2 占比: {dead_sim_rate:.2f}%")


if __name__ == "__main__":
    main()
