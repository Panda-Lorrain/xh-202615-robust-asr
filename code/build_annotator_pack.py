"""生成多人标注分发包 (2026-07-10)。

需求: 2 个队员各独立标全部 1084 条未满分 → 回收后 compare_annotations.py 比对找分歧 → 用户仲裁。
本脚本生成自包含标注包 code/annot_pack/:
  标注.html — 数据内嵌(1084条), 音频相对路径 pos/*.wav (队员解压到任意位置都能播),
              标注员ID输入框(存localStorage), 导出CSV带 annotator 列 + 文件名 annot_<ID>.csv
              (全量1084行, 含未标空行便于比对时查漏)
  pos/      — 1084 条 cmd_X.wav + kws_X.wav (--copy-audio 拷贝)
队员流程: 解压→双击标注.html→填ID→逐条听音选难点(←→翻条/空格播放/1-9选难点)→导出CSV→发回。
用法:
  code/.venv/Scripts/python.exe code/build_annotator_pack.py              # 仅生成HTML(快, 验证用)
  code/.venv/Scripts/python.exe code/build_annotator_pack.py --copy-audio  # 额外拷音频(2168文件, 数分钟)
"""
import json, csv, os, shutil, argparse

CSV = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
PACK = r"E:\midea_target_asr\code\annot_pack"
POS_OUT = os.path.join(PACK, "pos")
HTML_OUT = os.path.join(PACK, "标注.html")

ap = argparse.ArgumentParser()
ap.add_argument("--copy-audio", action="store_true", help="拷贝1084条音频到 pos/")
args = ap.parse_args()

rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
for r in rows:
    r["rec_src"] = "pos/" + os.path.basename(r["recognition_path"])
    r["enw_src"] = "pos/" + os.path.basename(r["enrollment_path"])
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
.idbox{padding:4px 8px;border:1px solid #1976d2;border-radius:4px;font-size:14px}
</style></head><body><div class="card">
<div class="nav">
 <button onclick="prev()">◀ 上一条 (←)</button>
 <span id="pos">0/0</span>
 <button onclick="next()">下一条 (▶)</button>
</div>
<progress id="prog" max="100" value="0"></progress>
<div style="margin:6px 0;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
 <span>标注员:</span><input id="annotid" class="idbox" placeholder="填你的名字(导出文件名会带它)" oninput="saveId()">
 <span>档筛选:</span><select id="filt" onchange="applyFilt()"><option value="">全部</option></select>
 <button onclick="exportCsv()">⬇ 导出标注CSV</button>
 <span class="hint">←→翻条 / 空格播放 / 1-9 选难点</span>
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
$('annotid').value=localStorage.getItem('annot_id')||'';
function saveId(){localStorage.setItem('annot_id',$('annotid').value);}
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
  const id=($('annotid').value.trim())||'unknown';
  let csv='annotator,uid,档,CER,max_sim,难点类别,备注\n';
  DATA.forEach(d=>{
    const s=store[d['uid']]||{tags:[],note:''};
    const note=(s.note||'').replace(/[\n,]/g,' ');
    csv+=[id,d['uid'],d['档'],d['vanilla_cer'],d['max_sim'],(s.tags.join(';')||''),note].join(',')+'\n';
  });
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});
  const a=document.createElement('a'); a.href=URL.createObjectURL(b);
  a.download='annot_'+id+'.csv'; a.click();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT') return;
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

os.makedirs(PACK, exist_ok=True)
html = TPL.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace("__HP__", json.dumps(HP, ensure_ascii=False))
open(HTML_OUT, "w", encoding="utf-8").write(html)
print(f"HTML -> {HTML_OUT} ({len(rows)} 条数据嵌入, 音频相对路径 pos/*.wav)")

if args.copy_audio:
    os.makedirs(POS_OUT, exist_ok=True)
    n = 0
    miss = 0
    for r in rows:
        for k in ("recognition_path", "enrollment_path"):
            src = r[k]
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(POS_OUT, os.path.basename(src)))
                n += 1
            else:
                miss += 1
    print(f"拷音频 {n} 个文件 -> {POS_OUT}" + (f" (缺 {miss} 个源文件)" if miss else ""))
    print(f"\n打包完成: {PACK}")
    print("  发给队员: 把 annot_pack 整个文件夹压缩成 zip, 网盘发给 2 个队员")
    print("  队员步骤: 解压 → 双击 标注.html → 填名字 → 听音标难点 → 导出 annot_<名字>.csv → 发回")
