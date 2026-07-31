"""2人标注比对 + 争议仲裁 v2 (rec/enw 分块, 2026-07-10)。

流程: 2队员各导出 annot_<ID>.csv (build_annotator_pack.py v2, 含 rec/enw 两块难点+自然语言)
      → 本脚本比对 → 一致(两块都一致)条定共识 / 分歧(任一块不一致)条进仲裁界面。
输出 (code/annot_pack/compare/):
  consistency_report.txt  — 一致性统计
  merged_annotation.csv   — 全量1084: 一致条填 rec_共识/enw_共识, 分歧条标 DISPUTE + 两人两块选项
  annot_disputes.html     — 分歧条仲裁界面(rec块+enw块各: 听音+两人难点/自然语言对比+交集默认勾)
                             仲裁完导出 arbitrated.csv(uid, rec_仲裁, enw_仲裁, 备注)
用法:
  code/.venv/Scripts/python.exe code/compare_annotations.py code/annot_pack/annot_A.csv code/annot_pack/annot_B.csv
"""
import csv, json, os, sys

POS_DIR = r"E:\midea_target_asr\datasetA\pos"
OUT = r"E:\midea_target_asr\code\annot_pack\compare"
META_CSV = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
POS_PAIRS = r"E:\midea_target_asr\code\pos_pairs_datasetA.json"
HP_REC = ["音量小", "语速快", "语速慢", "babble强", "重叠", "英文干扰", "静音/未说话", "循环幻觉", "其他"]
HP_ENW = ["背景嘈杂", "有其他说话人", "唤醒词不清", "音量小", "唤醒词截断", "多人同说", "静音/无有效语音", "其他"]


def load(path):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig")):
        uid = r["uid"]
        rec = set(t for t in (r.get("rec_难点", "") or "").split(";") if t)
        enw = set(t for t in (r.get("enw_难点", "") or "").split(";") if t)
        d[uid] = {"rec": rec, "enw": enw,
                  "rec_note": (r.get("rec_自然语言", "") or ""),
                  "enw_note": (r.get("enw_自然语言", "") or "")}
    return d


def file_uri(p):
    return "file:///" + p.replace("\\", "/") if p else ""


paths = sys.argv[1:]
if len(paths) < 2:
    print("用法: compare_annotations.py <annot_A.csv> <annot_B.csv> [...]"); sys.exit(1)
annots = [(os.path.basename(p).replace("annot_", "").replace(".csv", ""), load(p)) for p in paths]
names = [n for n, _ in annots]
print(f"读 {len(names)} 份标注: {', '.join(names)}")

META = {}
for r in csv.DictReader(open(META_CSV, encoding="utf-8-sig")):
    META[r["uid"]] = r
KWS = {}
if os.path.isfile(POS_PAIRS):
    for p in json.load(open(POS_PAIRS, encoding="utf-8")):
        KWS[p["id"]] = p.get("kws_txt", "")

all_uids = set(META.keys())
for _, d in annots:
    all_uids |= set(d.keys())

rows_out, disputes = [], []
n_cons = n_disp = n_miss = 0
for uid in sorted(all_uids):
    m = META.get(uid, {})
    per = {n: d.get(uid, {"rec": set(), "enw": set(), "rec_note": "", "enw_note": ""}) for n, d in annots}
    rec_sets = [per[n]["rec"] for n in names]
    enw_sets = [per[n]["enw"] for n in names]
    n_marked = sum(1 for n in names if (per[n]["rec"] or per[n]["enw"]))
    rec_eq = all(s == rec_sets[0] for s in rec_sets)
    enw_eq = all(s == enw_sets[0] for s in enw_sets)
    if n_marked == 0:
        status, n_miss = "全未标", n_miss + 1
    elif rec_eq and enw_eq:
        status, n_cons = "一致", n_cons + 1
    else:
        status, n_disp = "分歧", n_disp + 1
    rec_inter = set.intersection(*[s for s in rec_sets if s]) if any(rec_sets) else set()
    enw_inter = set.intersection(*[s for s in enw_sets if s]) if any(enw_sets) else set()
    row = {"uid": uid, "档": m.get("档", ""), "CER": m.get("vanilla_cer", ""), "max_sim": m.get("max_sim", ""),
           "状态": status,
           "rec_共识": ";".join(sorted(rec_inter)) if status == "一致" else ("DISPUTE" if status == "分歧" else ""),
           "enw_共识": ";".join(sorted(enw_inter)) if status == "一致" else ("DISPUTE" if status == "分歧" else "")}
    for n in names:
        row[f"{n}_rec"] = ";".join(sorted(per[n]["rec"]))
        row[f"{n}_enw"] = ";".join(sorted(per[n]["enw"]))
    rows_out.append(row)
    if status == "分歧":
        rec_path = m.get("recognition_path", os.path.join(POS_DIR, uid + ".wav"))
        enw_path = m.get("enrollment_path", os.path.join(POS_DIR, uid.replace("cmd", "kws") + ".wav"))
        _id = int(uid.replace("cmd_", "")) if uid.startswith("cmd_") else -1
        disputes.append({
            "uid": uid, "档": m.get("档", ""), "cer": m.get("vanilla_cer", ""), "sim": m.get("max_sim", ""),
            "ref": m.get("ref", ""), "vanilla": (m.get("vanilla_text", "") or "")[:500],
            "rec": file_uri(rec_path), "enw": file_uri(enw_path), "kws": KWS.get(_id, ""),
            "per": {n: {"rec": sorted(per[n]["rec"]), "enw": sorted(per[n]["enw"]),
                        "rec_note": per[n]["rec_note"], "enw_note": per[n]["enw_note"]} for n in names},
            "rec_inter": sorted(rec_inter), "enw_inter": sorted(enw_inter),
        })

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "merged_annotation.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)

total = len(all_uids)
rep = (f"比对人数: {len(names)} ({', '.join(names)})\n总条数: {total}\n"
       f"一致(两块都一致): {n_cons} ({n_cons/total*100:.1f}%) — 共识已定\n"
       f"分歧(任一块不一致): {n_disp} ({n_disp/total*100:.1f}%) — 需仲裁\n"
       f"全未标: {n_miss}\n\n"
       f"下一步: 打开 annot_disputes.html 逐条听音仲裁(rec块+enw块) → 导出 arbitrated.csv\n"
       f"最终全集 = merged_annotation.csv(一致条) + arbitrated.csv(仲裁条)")
open(os.path.join(OUT, "consistency_report.txt"), "w", encoding="utf-8").write(rep)
print(rep)

TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>分歧仲裁</title>
<style>
body{font-family:system-ui;margin:0;background:#fafafa}
.card{max-width:900px;margin:auto;background:#fff;padding:18px;min-height:100vh;box-sizing:border-box}
.nav{display:flex;justify-content:space-between;align-items:center;margin:8px 0}
.meta{color:#555;font-size:13px;margin:6px 0}.meta b{color:#1976d2}
.ref{background:#e8f5e9;padding:8px 10px;border-left:4px solid #4caf50;margin:6px 0;border-radius:3px}
.hyp{background:#ffebee;padding:8px 10px;border-left:4px solid #f44336;margin:6px 0;border-radius:3px;word-break:break-all}
audio{width:100%;margin:4px 0}
.section{border:1px solid #e0e0e0;border-radius:6px;padding:10px;margin:10px 0}
.section h3{margin:0 0 8px;font-size:15px;border-bottom:2px solid #1976d2;padding-bottom:4px}
.section.enw h3{border-bottom-color:#fb8c00}
.per{background:#f5f5f5;padding:6px 8px;border-radius:4px;margin:4px 0;font-size:13px}
.ptag{display:inline-block;background:#bbb;color:#fff;padding:2px 8px;border-radius:8px;margin:1px;font-size:11px}
.inter{background:#fff3e0;padding:6px;border-left:4px solid #fb8c00;margin:4px 0;border-radius:3px;font-size:13px}
.note2{color:#777;font-style:italic;font-size:12px;margin:2px 0 2px 12px}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0}
.tag{padding:4px 10px;border:1px solid #888;border-radius:12px;cursor:pointer;font-size:13px;user-select:none;background:#fff}
.tag.on{background:#43a047;color:#fff;border-color:#43a047}
button{padding:6px 14px;margin:2px;cursor:pointer;border:1px solid #888;border-radius:4px;background:#fff}
textarea{width:100%;height:36px;margin:6px 0;box-sizing:border-box}
progress{width:100%;height:6px}.hint{color:#999;font-size:12px}.lbl{font-size:12px;color:#666;margin-top:4px}
</style></head><body><div class="card">
<div class="nav"><button onclick="prev()">◀ (←)</button><span id="pos">0/0</span><button onclick="next()">(→) ▶</button></div>
<progress id="prog" max="100" value="0"></progress>
<div style="margin:6px 0"><button onclick="exportArb()">⬇ 导出仲裁CSV</button><span class="hint">←→翻条 / 空格播recognition / 1-9 仲裁rec难点(默认勾两人交集)</span></div>
<div class="meta" id="meta"></div>
<div class="section rec">
 <h3>① recognition 仲裁</h3>
 <div class="ref" id="ref"></div>
 <div class="hyp" id="hyp"></div>
 <audio id="rec" controls></audio>
 <div id="per_rec"></div>
 <div class="lbl">仲裁难点（默认勾两人共同选的，可改）：</div>
 <div class="tags" id="rec_arb"></div>
</div>
<div class="section enw">
 <h3>② enrollment 仲裁</h3>
 <div class="hint">唤醒词: <b id="kws" style="color:#fb8c00"></b></div>
 <audio id="enw" controls></audio>
 <div id="per_enw"></div>
 <div class="lbl">仲裁难点（默认勾两人共同选的）：</div>
 <div class="tags" id="enw_arb"></div>
</div>
<textarea id="note" placeholder="仲裁备注..." oninput="saveNote()"></textarea>
</div><script>
const DATA=__DATA__, HP_REC=__HP_REC__, HP_ENW=__HP_ENW__, KEY='arb_v2';
let idx=0; let store=JSON.parse(localStorage.getItem(KEY)||'{}');
const $=id=>document.getElementById(id);
function cur(){const d=DATA[idx];return [d, store[d.uid]||{rec_tags:d.rec_inter.slice(),enw_tags:d.enw_inter.slice(),note:''}];}
function render(){
  const [d,s]=cur();
  $('pos').textContent=(idx+1)+'/'+DATA.length;
  $('prog').value=(idx+1)/DATA.length*100;
  $('meta').innerHTML='<b>'+d['档']+'</b> | '+d.uid+' | CER <b>'+d.cer+'</b> | sim '+d.sim;
  $('ref').innerHTML='<b>ref:</b> '+d.ref;
  $('hyp').innerHTML='<b>vanilla:</b> '+(d.vanilla||'');
  $('rec').src=d.rec; $('enw').src=d.enw; $('kws').textContent=d.kws||'(无)';
  $('note').value=s.note;
  let pr='';Object.entries(d.per).forEach(([n,v])=>{pr+='<div class="per"><b>'+n+' rec难点:</b> '+(v.rec.length?v.rec.map(t=>'<span class="ptag">'+t+'</span>').join(''):'<i>未标</i>');if(v.rec_note)pr+='<div class="note2">「'+v.rec_note+'」</div>';pr+='</div>';});
  pr+='<div class="inter"><b>共同选:</b> '+(d.rec_inter.length?d.rec_inter.join('、'):'<i>无</i>')+'</div>';
  $('per_rec').innerHTML=pr;
  let pe='';Object.entries(d.per).forEach(([n,v])=>{pe+='<div class="per"><b>'+n+' enw难点:</b> '+(v.enw.length?v.enw.map(t=>'<span class="ptag">'+t+'</span>').join(''):'<i>未标</i>');if(v.enw_note)pe+='<div class="note2">「'+v.enw_note+'」</div>';pe+='</div>';});
  pe+='<div class="inter"><b>共同选:</b> '+(d.enw_inter.length?d.enw_inter.join('、'):'<i>无</i>')+'</div>';
  $('per_enw').innerHTML=pe;
  $('rec_arb').innerHTML=HP_REC.map((h,j)=>'<span class="tag '+(s.rec_tags.includes(h)?'on':'')+'" onclick="toggle(\'rec\','+j+')">'+(j+1)+'.'+h+'</span>').join('');
  $('enw_arb').innerHTML=HP_ENW.map((h,j)=>'<span class="tag '+(s.enw_tags.includes(h)?'on':'')+'" onclick="toggle(\'enw\','+j+')">'+(j+1)+'.'+h+'</span>').join('');
}
function toggle(block,j){
  const [d]=cur(); store[d.uid]=store[d.uid]||{rec_tags:d.rec_inter.slice(),enw_tags:d.enw_inter.slice(),note:''};
  const o=store[d.uid]; const hp=block==='rec'?HP_REC:HP_ENW; const key=block+'_tags';
  const k=o[key].indexOf(hp[j]);
  if(k>=0) o[key].splice(k,1); else o[key].push(hp[j]);
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function saveNote(){const [d]=cur();store[d.uid]=store[d.uid]||{rec_tags:d.rec_inter.slice(),enw_tags:d.enw_inter.slice(),note:''};store[d.uid].note=$('note').value;localStorage.setItem(KEY,JSON.stringify(store));}
function next(){if(idx<DATA.length-1){idx++;render();}}
function prev(){if(idx>0){idx--;render();}}
function exportArb(){
  let csv='uid,rec_仲裁,enw_仲裁,备注\n';
  DATA.forEach(d=>{const s=store[d.uid]||{rec_tags:d.rec_inter.slice(),enw_tags:d.enw_inter.slice(),note:''};
    const note=(s.note||'').replace(/[\n,]/g,' ');
    csv+=[d.uid,s.rec_tags.join(';'),s.enw_tags.join(';'),note].join(',')+'\n';});
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='arbitrated.csv';a.click();
}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;
  if(e.key==='ArrowRight'){e.preventDefault();next();}if(e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();$('rec').play();}if(/^[1-9]$/.test(e.key)){toggle('rec',parseInt(e.key)-1);}});
render();
</script></body></html>"""

html = (TPL.replace("__DATA__", json.dumps(disputes, ensure_ascii=False))
           .replace("__HP_REC__", json.dumps(HP_REC, ensure_ascii=False))
           .replace("__HP_ENW__", json.dumps(HP_ENW, ensure_ascii=False)))
open(os.path.join(OUT, "annot_disputes.html"), "w", encoding="utf-8").write(html)
print(f"\n分歧仲裁界面 -> {os.path.join(OUT, 'annot_disputes.html')} ({len(disputes)} 条分歧)")
print(f"全部输出目录: {OUT}")
