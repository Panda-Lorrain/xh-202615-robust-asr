"""生成未满分音频标注界面 HTML (2026-07-09)。
读 error_analysis_pos_unfull.csv → self-contained error_annotator.html (无后端, 双击打开)。
功能: recognition/enrollment 播放 + 难点多选 + 备注 + 上下条导航(←→) + 空格播放 + 数字键选难点 + 档筛选 + localStorage 自动存 + 导出标注 CSV。
难点类别(HP)默认 9 类, 用户可改本数组。
用法: 双击 error_annotator.html (推荐 Firefox; Chrome 若禁 file:// 则在 E:\\midea_target_asr 跑 python -m http.server 再开 localhost:8000/code/error_annotator.html)。标注存浏览器, 导出 CSV 带 uid/档/CER/sim/难点/备注。
"""
import json, csv

CSV = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
OUT = r"E:\midea_target_asr\code\error_annotator.html"

rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
for r in rows:
    r["rec_src"] = "file:///" + r["recognition_path"].replace("\\", "/")
    r["enw_src"] = "file:///" + r["enrollment_path"].replace("\\", "/")
HP = ["音量小", "语速快", "语速慢", "babble强", "重叠", "英文干扰", "静音/未说话", "循环幻觉", "其他"]

TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>未满分音频标注</title>
<style>
body{font-family:system-ui;margin:0;background:#fafafa}
.card{max-width:900px;margin:auto;background:#fff;padding:18px;min-height:100vh;box-sizing:border-box}
.nav{display:flex;justify-content:space-between;align-items:center;margin:8px 0}
.meta{color:#555;font-size:13px;margin:6px 0}
.meta b{color:#1976d2}
.ref{background:#e8f5e9;padding:8px 10px;border-left:4px solid #4caf50;margin:6px 0;border-radius:3px}
.hyp{background:#ffebee;padding:8px 10px;border-left:4px solid #f44336;margin:6px 0;border-radius:3px;word-break:break-all}
audio{width:100%;margin:4px 0}
.tags{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}
.tag{padding:4px 10px;border:1px solid #888;border-radius:12px;cursor:pointer;font-size:13px;user-select:none}
.tag.on{background:#1976d2;color:#fff;border-color:#1976d2}
button{padding:6px 14px;margin:2px;cursor:pointer;border:1px solid #888;border-radius:4px;background:#fff}
button:hover{background:#f0f0f0}
textarea{width:100%;height:36px;margin:6px 0;box-sizing:border-box}
progress{width:100%;height:6px}
.hint{color:#999;font-size:12px;margin:4px 0}
</style></head><body><div class="card">
<div class="nav">
 <button onclick="prev()">◀ 上一条 (←)</button>
 <span id="pos">0/0</span>
 <button onclick="next()">下一条 (▶)</button>
</div>
<progress id="prog" max="100" value="0"></progress>
<div style="margin:6px 0">
 档筛选:<select id="filt" onchange="applyFilt()"><option value="">全部</option></select>
 <button onclick="exportCsv()">⬇ 导出标注CSV</button>
 <span class="hint">快捷键: ←→翻条 / 空格播放recognition / 1-9 选难点</span>
</div>
<div class="meta" id="meta"></div>
<div class="ref" id="ref"></div>
<div class="hyp" id="hyp"></div>
<div class="hint">recognition (带噪, 听这个判断 target 说了什么):</div>
<audio id="rec" controls></audio>
<div class="hint">enrollment (target 唤醒词, 音色参考):</div>
<audio id="enw" controls></audio>
<div class="tags" id="tags"></div>
<textarea id="note" placeholder="备注..." oninput="save()"></textarea>
</div>
<script>
const DATA = __DATA__;
const HP = __HP__;
const KEY = 'err_annot_v1';
let filt="", idx=0, view=DATA.map((_,i)=>i);
let store=JSON.parse(localStorage.getItem(KEY)||'{}');
const $=id=>document.getElementById(id);
function applyFilt(){
  filt=$('filt').value;
  view=DATA.map((_,i)=>i).filter(i=>!filt||DATA[i]['档']===filt);
  idx=0; render();
}
function render(){
  if(!view.length){$('meta').textContent='无';return;}
  const i=view[idx]; const d=DATA[i]; const key=d['uid'];
  $('pos').textContent=(idx+1)+'/'+view.length+' (全量第'+(i+1)+'条)';
  $('prog').value=(idx+1)/view.length*100;
  $('meta').innerHTML='<b>'+d['档']+'</b> | '+d['uid']+' | CER <b>'+d['vanilla_cer']+'</b> | sim '+d['max_sim']+' | '+d['rec_sec']+'s';
  $('ref').innerHTML='<b>ref(正确):</b> '+d['ref'];
  $('hyp').innerHTML='<b>vanilla(ASR输出):</b> '+(d['vanilla_text']||'').slice(0,500);
  $('rec').src=d['rec_src']; $('enw').src=d['enw_src'];
  $('note').value=(store[key]&&store[key].note)||'';
  const tags=(store[key]&&store[key].tags)||[];
  $('tags').innerHTML=HP.map((h,j)=>'<span class="tag '+(tags.includes(h)?'on':'')+'" onclick="toggle('+j+')">'+(j+1)+'.'+h+'</span>').join('');
}
function toggle(j){
  const i=view[idx]; const key=DATA[i]['uid'];
  store[key]=store[key]||{tags:[],note:''};
  const h=HP[j]; const k=store[key].tags.indexOf(h);
  if(k>=0) store[key].tags.splice(k,1); else store[key].tags.push(h);
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function save(){
  const i=view[idx]; const key=DATA[i]['uid'];
  store[key]=store[key]||{tags:[],note:''};
  store[key].note=$('note').value;
  localStorage.setItem(KEY,JSON.stringify(store));
}
function next(){if(idx<view.length-1){idx++;render();}}
function prev(){if(idx>0){idx--;render();}}
function exportCsv(){
  let csv='uid,档,CER,max_sim,难点类别,备注\n';
  DATA.forEach(d=>{
    const s=store[d['uid']];
    if(s && (s.tags.length||s.note)){
      const note=(s.note||'').replace(/[\n,]/g,' ');
      csv+=[d['uid'],d['档'],d['vanilla_cer'],d['max_sim'],(s.tags.join(';')||''),note].join(',')+'\n';
    }
  });
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='error_annot_export.csv'; a.click();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA') return;
  if(e.key==='ArrowRight'){e.preventDefault();next();}
  if(e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();$('rec').play();}
  if(/^[1-9]$/.test(e.key)){toggle(parseInt(e.key)-1);}
});
[...new Set(DATA.map(d=>d['档']))].sort().forEach(b=>{
  const o=document.createElement('option'); o.value=b;
  o.textContent=b+' ('+DATA.filter(d=>d['档']===b).length+')';
  $('filt').appendChild(o);
});
applyFilt();
</script></body></html>"""

html = TPL.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace("__HP__", json.dumps(HP, ensure_ascii=False))
open(OUT, "w", encoding="utf-8").write(html)
print(f"HTML -> {OUT} ({len(rows)} 条数据嵌入)")
