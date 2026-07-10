"""2人标注比对 + 争议仲裁 (2026-07-10)。

流程: 2队员各导出 annot_<ID>.csv (build_annotator_pack.py) → 本脚本比对 → 一致条定共识 / 分歧条进仲裁界面。
输出 (code/annot_pack/compare/):
  consistency_report.txt  — 一致性统计(一致/分歧/未标 占比)
  merged_annotation.csv   — 全量1084: 一致条填共识难点, 分歧条标 DISPUTE + 两人选项
  annot_disputes.html     — 分歧条仲裁界面(听音+两人难点对比+交集高亮+仲裁多选, 默认勾交集)
                             仲裁完导出 arbitrated.csv → 与 merged 合并得最终全集
用法:
  code/.venv/Scripts/python.exe code/compare_annotations.py code/annot_pack/annot_张三.csv code/annot_pack/annot_李四.csv
"""
import csv, json, os, sys

POS_DIR = r"E:\midea_target_asr\datasetA\pos"
OUT = r"E:\midea_target_asr\code\annot_pack\compare"
META_CSV = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
HP = ["音量小", "语速快", "语速慢", "babble强", "重叠", "英文干扰", "静音/未说话", "循环幻觉", "其他"]


def load(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        uid = r["uid"]
        tags = set(t for t in (r.get("难点类别", "") or "").split(";") if t)
        d[uid] = tags
    return d


def file_uri(p):
    return "file:///" + p.replace("\\", "/") if p else ""


paths = sys.argv[1:]
if len(paths) < 2:
    print("用法: compare_annotations.py <annot_A.csv> <annot_B.csv> [...]"); sys.exit(1)
annots = [(os.path.basename(p).replace("annot_", "").replace(".csv", ""), load(p)) for p in paths]
names = [n for n, _ in annots]
print(f"读 {len(names)} 份标注: {', '.join(names)}")

# 元信息(档/CER/sim/ref/vanilla/音频路径)
META = {}
for r in csv.DictReader(open(META_CSV, encoding="utf-8-sig")):
    META[r["uid"]] = r

all_uids = set()
for _, d in annots:
    all_uids |= set(d.keys())
all_uids |= set(META.keys())  # 保全量1084

rows_out, disputes = [], []
n_cons = n_disp = n_miss = 0
for uid in sorted(all_uids):
    m = META.get(uid, {})
    per = {n: d.get(uid, set()) for n, d in annots}
    sets = list(per.values())
    n_marked = sum(1 for s in sets if s)
    if n_marked == 0:
        status, n_miss = "全未标", n_miss + 1
    elif all(s == sets[0] for s in sets):
        status, n_cons = "一致", n_cons + 1
    else:
        status, n_disp = "分歧", n_disp + 1
    nonempty = [s for s in sets if s]
    inter = set.intersection(*nonempty) if nonempty else set()
    row = {"uid": uid, "档": m.get("档", ""), "CER": m.get("vanilla_cer", ""),
           "max_sim": m.get("max_sim", ""), "状态": status,
           "共识难点": ";".join(sorted(inter)) if status == "一致" else ("DISPUTE" if status == "分歧" else "")}
    for n, s in per.items():
        row[f"{n}_难点"] = ";".join(sorted(s))
    row["备注"] = ""
    rows_out.append(row)
    if status == "分歧":
        rec = m.get("recognition_path", os.path.join(POS_DIR, uid + ".wav"))
        enw = m.get("enrollment_path", os.path.join(POS_DIR, uid.replace("cmd", "kws") + ".wav"))
        disputes.append({
            "uid": uid, "档": m.get("档", ""), "cer": m.get("vanilla_cer", ""),
            "sim": m.get("max_sim", ""), "ref": m.get("ref", ""),
            "vanilla": (m.get("vanilla_text", "") or "")[:500],
            "rec": file_uri(rec), "enw": file_uri(enw),
            "per": {n: sorted(s) for n, s in per.items()}, "inter": sorted(inter),
        })

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "merged_annotation.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)

total = len(all_uids)
rep = (f"比对人数: {len(names)} ({', '.join(names)})\n总条数: {total}\n"
       f"一致: {n_cons} ({n_cons/total*100:.1f}%) — 共识已定\n"
       f"分歧: {n_disp} ({n_disp/total*100:.1f}%) — 需仲裁\n全未标: {n_miss}\n\n"
       f"下一步: 打开 annot_disputes.html 逐条听音仲裁 → 导出 arbitrated.csv\n"
       f"最终全集 = merged_annotation.csv(一致条) + arbitrated.csv(仲裁条)")
open(os.path.join(OUT, "consistency_report.txt"), "w", encoding="utf-8").write(rep)
print(rep)

# 争议仲裁 HTML
TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>分歧仲裁</title>
<style>
body{font-family:system-ui;margin:0;background:#fafafa}
.card{max-width:900px;margin:auto;background:#fff;padding:18px;min-height:100vh;box-sizing:border-box}
.nav{display:flex;justify-content:space-between;align-items:center;margin:8px 0}
.meta{color:#555;font-size:13px;margin:6px 0}.meta b{color:#1976d2}
.ref{background:#e8f5e9;padding:8px 10px;border-left:4px solid #4caf50;margin:6px 0;border-radius:3px}
.hyp{background:#ffebee;padding:8px 10px;border-left:4px solid #f44336;margin:6px 0;border-radius:3px;word-break:break-all}
audio{width:100%;margin:4px 0}
.per{background:#f5f5f5;padding:8px;border-radius:4px;margin:6px 0}
.ptag{display:inline-block;background:#bbb;color:#fff;padding:2px 8px;border-radius:8px;margin:2px;font-size:12px}
.inter{background:#fff3e0;padding:8px;border-left:4px solid #fb8c00;margin:6px 0;border-radius:3px}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.tag{padding:4px 10px;border:1px solid #888;border-radius:12px;cursor:pointer;font-size:13px;user-select:none}
.tag.on{background:#43a047;color:#fff;border-color:#43a047}
button{padding:6px 14px;margin:2px;cursor:pointer;border:1px solid #888;border-radius:4px;background:#fff}
textarea{width:100%;height:36px;margin:6px 0;box-sizing:border-box}
progress{width:100%;height:6px}.hint{color:#999;font-size:12px}
</style></head><body><div class="card">
<div class="nav"><button onclick="prev()">◀ (←)</button><span id="pos">0/0</span><button onclick="next()">(→) ▶</button></div>
<progress id="prog" max="100" value="0"></progress>
<div style="margin:6px 0"><button onclick="exportArb()">⬇ 导出仲裁CSV</button><span class="hint">←→翻条 / 空格播放 / 1-9 仲裁难点 (默认勾两人交集)</span></div>
<div class="meta" id="meta"></div>
<div class="per" id="per"></div>
<div class="inter" id="inter"></div>
<div class="ref" id="ref"></div>
<div class="hyp" id="hyp"></div>
<div class="hint">recognition (带噪):</div><audio id="rec" controls></audio>
<div class="hint">enrollment (音色参考):</div><audio id="enw" controls></audio>
<div class="tags" id="arb"></div>
<textarea id="note" placeholder="仲裁备注..." oninput="saveNote()"></textarea>
</div><script>
const DATA=__DATA__; const HP=__HP__; const KEY='arb_v1';
let idx=0; let store=JSON.parse(localStorage.getItem(KEY)||'{}');
const $=id=>document.getElementById(id);
function cur(){const d=DATA[idx]; return [d, store[d.uid]||{tags:d.inter.slice(),note:''}];}
function render(){
  const [d,s]=cur();
  $('pos').textContent=(idx+1)+'/'+DATA.length;
  $('prog').value=(idx+1)/DATA.length*100;
  $('meta').innerHTML='<b>'+d['档']+'</b> | '+d.uid+' | CER <b>'+d.cer+'</b> | sim '+d.sim;
  let ph=''; Object.entries(d.per).forEach(([n,t])=>{ph+='<div><b>'+n+':</b> '+(t.length?t.map(x=>'<span class="ptag">'+x+'</span>').join(''):'<i>未标</i>')+'</div>';});
  $('per').innerHTML=ph;
  $('inter').innerHTML='<b>两人共同选:</b> '+(d.inter.length?d.inter.join('、'):'<i>无(完全分歧)</i>');
  $('ref').innerHTML='<b>ref(正确):</b> '+d.ref;
  $('hyp').innerHTML='<b>vanilla(ASR输出):</b> '+(d.vanilla||'');
  $('rec').src=d.rec; $('enw').src=d.enw;
  $('note').value=s.note;
  $('arb').innerHTML=HP.map((h,j)=>'<span class="tag '+(s.tags.includes(h)?'on':'')+'" onclick="toggle('+j+')">'+(j+1)+'.'+h+'</span>').join('');
}
function toggle(j){
  const [d,s]=cur(); store[d.uid]=store[d.uid]||{tags:d.inter.slice(),note:''};
  const h=HP[j]; const k=store[d.uid].tags.indexOf(h);
  if(k>=0) store[d.uid].tags.splice(k,1); else store[d.uid].tags.push(h);
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function saveNote(){const [d,s]=cur(); store[d.uid]=store[d.uid]||{tags:d.inter.slice(),note:''}; store[d.uid].note=$('note').value; localStorage.setItem(KEY,JSON.stringify(store));}
function next(){if(idx<DATA.length-1){idx++;render();}}
function prev(){if(idx>0){idx--;render();}}
function exportArb(){
  let csv='uid,仲裁难点,备注\n';
  DATA.forEach(d=>{const s=store[d.uid]||{tags:d.inter.slice(),note:''};
    if(s.tags.length||s.note){const note=(s.note||'').replace(/[\n,]/g,' ');csv+=[d.uid,s.tags.join(';'),note].join(',')+'\n';}});
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='arbitrated.csv';a.click();
}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;
  if(e.key==='ArrowRight'){e.preventDefault();next();}if(e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();$('rec').play();}if(/^[1-9]$/.test(e.key)){toggle(parseInt(e.key)-1);}});
render();
</script></body></html>"""

html = TPL.replace("__DATA__", json.dumps(disputes, ensure_ascii=False)).replace("__HP__", json.dumps(HP, ensure_ascii=False))
open(os.path.join(OUT, "annot_disputes.html"), "w", encoding="utf-8").write(html)
print(f"\n分歧仲裁界面 -> {os.path.join(OUT, 'annot_disputes.html')} ({len(disputes)} 条分歧)")
print(f"全部输出目录: {OUT}")
