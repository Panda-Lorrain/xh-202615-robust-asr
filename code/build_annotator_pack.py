"""生成多人标注分发包 (2026-07-10, v2 分块标注)。

需求: 2 队员各独立标全部 1084 条 → compare_annotations.py 比对 → 用户仲裁。
v2 改动: recognition 和 enrollment **分开标注**(各有难点+自然语言标注);
         "备注"→"自然语言标注"(词条形容不了时自由描述, 回收后交 Claude 二次分类总结)。

本脚本生成自包含标注包 code/annot_pack/:
  标注.html — 数据内嵌(1084条), 音频相对路径 pos/*.wav, 标注员ID,
              recognition 块(难点9类 + 自然语言标注) + enrollment 块(难点8类 + 自然语言标注),
              导出 CSV 列: annotator,uid,档,CER,max_sim,rec_难点,rec_自然语言,enw_难点,enw_自然语言
  pos/      — 1084 条 cmd_X.wav + kws_X.wav (--copy-audio)
难点类别:
  recognition(9): 音量小/语速快/语速慢/babble强/重叠/英文干扰/静音未说话/循环幻觉/其他
  enrollment(8):  背景嘈杂/有其他说话人/唤醒词不清/音量小/唤醒词截断/多人同说/静音无有效语音/其他
队员流程: 解压→双击标注.html→填ID→逐条标两块(难点+自然语言)→导出CSV→发回。
快捷键: ←→翻条 / 空格播放recognition / 1-9选recognition难点 / enrollment难点鼠标点
用法:
  code/.venv/Scripts/python.exe code/build_annotator_pack.py              # 仅生成HTML
  code/.venv/Scripts/python.exe code/build_annotator_pack.py --copy-audio  # 额外拷音频
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
# 唤醒词参考文本 (enrollment 音频对应的文字, 从 pos_pairs_datasetA.json 的 kws_txt 按 id 取)
POS_PAIRS = r"E:\midea_target_asr\code\pos_pairs_datasetA.json"
KWS = {}
if os.path.isfile(POS_PAIRS):
    for p in json.load(open(POS_PAIRS, encoding="utf-8")):
        KWS[p["id"]] = p.get("kws_txt", "")
for r in rows:
    r["rec_src"] = "pos/" + os.path.basename(r["recognition_path"])
    r["enw_src"] = "pos/" + os.path.basename(r["enrollment_path"])
    _id = int(r["uid"].replace("cmd_", "")) if r["uid"].startswith("cmd_") else -1
    r["kws_txt"] = KWS.get(_id, "")

HP_REC = ["音量小", "语速快", "语速慢", "babble强", "重叠", "英文干扰", "静音/未说话", "循环幻觉", "其他"]
HP_ENW = ["背景嘈杂", "有其他说话人", "唤醒词不清", "音量小", "唤醒词截断", "多人同说", "静音/无有效语音", "其他"]

TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>未满分音频标注</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f0f2f5;color:#2c3e50;line-height:1.6;font-size:14px}
.card{max-width:940px;margin:0 auto;padding:20px;min-height:100vh}
.nav{display:flex;justify-content:space-between;align-items:center;background:#fff;border-radius:12px;padding:12px 18px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.nav #pos{font-weight:700;color:#1976d2;font-size:15px}
progress{width:100%;height:7px;-webkit-appearance:none;appearance:none;border:none;border-radius:4px;margin-bottom:14px;display:block}
progress::-webkit-progress-bar{background:#e4e7ed;border-radius:4px}
progress::-webkit-progress-value{background:linear-gradient(90deg,#1976d2,#42a5f5);border-radius:4px}
.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#fff;border-radius:12px;padding:12px 18px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.tlbl{color:#78909c;font-size:13px;font-weight:600}
.toolbar .hint{margin-left:auto}
.idbox{padding:6px 12px;border:1.5px solid #cfd8dc;border-radius:8px;font-size:14px;outline:none;transition:border-color .2s;background:#fff}
.idbox:focus{border-color:#1976d2;box-shadow:0 0 0 3px rgba(25,118,210,.12)}
select{padding:6px 12px;border:1.5px solid #cfd8dc;border-radius:8px;font-size:13px;background:#fff;outline:none;cursor:pointer;color:#455a64}
.meta{background:#fff;border-radius:12px;padding:12px 18px;margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,.06);font-size:14px;color:#546e7a}
.meta b{color:#1976d2;font-weight:700}
.section{background:#fff;border-radius:14px;padding:20px;margin-bottom:20px;box-shadow:0 2px 12px rgba(0,0,0,.08);border:1px solid #eceff1;overflow:hidden}
.section h3{margin:-20px -20px 18px;padding:14px 20px;font-size:15px;font-weight:700;color:#fff;background:#1976d2}
.section.enw h3{background:#fb8c00}
.ref{background:#f1f8f4;border:1px solid #c8e6c9;border-left:4px solid #4caf50;padding:10px 14px;margin:10px 0;border-radius:8px}
.ref b{color:#2e7d32}
.hyp{background:#fdf3f4;border:1px solid #ffcdd2;border-left:4px solid #f44336;padding:10px 14px;margin:10px 0;border-radius:8px;word-break:break-all}
.hyp b{color:#c62828}
audio{width:100%;margin:12px 0;display:block}
.lbl{font-size:11px;color:#90a4ae;font-weight:700;margin:14px 0 6px;text-transform:uppercase;letter-spacing:.8px}
.tags{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 4px}
.tag{padding:6px 14px;border:1.5px solid #cfd8dc;border-radius:20px;cursor:pointer;font-size:13px;user-select:none;background:#fff;color:#546e7a;transition:all .15s;font-weight:500}
.tag:hover{border-color:#1976d2;color:#1976d2}
.tag.on{background:#1976d2;color:#fff;border-color:#1976d2;box-shadow:0 2px 6px rgba(25,118,210,.3)}
.section.enw .tag:hover{border-color:#fb8c00;color:#fb8c00}
.section.enw .tag.on{background:#fb8c00;border-color:#fb8c00;box-shadow:0 2px 6px rgba(251,140,0,.3)}
textarea{width:100%;margin:8px 0;padding:10px 14px;border:1.5px solid #e0e0e0;border-radius:10px;font-size:14px;font-family:inherit;resize:vertical;outline:none;transition:border-color .2s;line-height:1.5;display:block}
textarea:focus{border-color:#1976d2;box-shadow:0 0 0 3px rgba(25,118,210,.1)}
.section.enw textarea:focus{border-color:#fb8c00;box-shadow:0 0 0 3px rgba(251,140,0,.1)}
button{padding:8px 16px;cursor:pointer;border:1.5px solid #cfd8dc;border-radius:8px;background:#fff;color:#455a64;font-size:13px;font-weight:500;transition:all .15s}
button:hover{background:#f5f7fa;border-color:#1976d2;color:#1976d2}
button.primary{background:#1976d2;color:#fff;border-color:#1976d2;font-weight:600}
button.primary:hover{background:#1565c0;border-color:#1565c0;color:#fff}
.hint{color:#90a4ae;font-size:12px}
</style></head><body><div class="card">
<div class="nav">
 <button onclick="prev()">◀ 上一条 (←)</button>
 <span id="pos">0/0</span>
 <button onclick="next()">下一条 (▶)</button>
</div>
<progress id="prog" max="100" value="0"></progress>
<div class="toolbar">
 <span class="tlbl">标注员</span><input id="annotid" class="idbox" placeholder="填你的名字(导出文件名会带它)" oninput="saveId()">
 <span class="tlbl">档筛选</span><select id="filt" onchange="applyFilt()"><option value="">全部</option></select>
 <button class="primary" onclick="exportCsv()">⬇ 导出标注CSV</button>
 <span class="hint">←→翻条 / 空格播recognition / 1-9选rec难点</span>
</div>
<div class="meta" id="meta"></div>

<div class="section rec">
 <h3>① recognition（带噪识别音频 — 标 target 说了什么 + 识别难点）</h3>
 <div class="ref" id="ref"></div>
 <div class="hyp" id="hyp"></div>
 <audio id="rec" controls></audio>
 <div class="lbl">难点（可多选，1-9 快捷键）：</div>
 <div class="tags" id="rec_tags"></div>
 <div class="lbl">自然语言标注（9类形容不了时自由描述；回收后由 AI 二次分类）：</div>
 <textarea id="rec_note" placeholder="例: target 的话被背景人声盖住一半..." oninput="save('rec_note',this.value)"></textarea>
</div>

<div class="section enw">
 <h3>② enrollment（声纹唤醒词音频 — 标 target 音色/唤醒词质量难点）</h3>
 <div class="hint">唤醒词参考文本: <b id="kws" style="color:#fb8c00"></b></div>
 <audio id="enw" controls></audio>
 <div class="lbl">难点（可多选，鼠标点击）：</div>
 <div class="tags" id="enw_tags"></div>
 <div class="lbl">自然语言标注：</div>
 <textarea id="enw_note" placeholder="例: 唤醒词后半段有第二个说话人插入..." oninput="save('enw_note',this.value)"></textarea>
</div>
</div>
<script>
const DATA=__DATA__, HP_REC=__HP_REC__, HP_ENW=__HP_ENW__, KEY='err_annot_v2';
let filt="", idx=0, view=DATA.map((_,i)=>i);
let store=JSON.parse(localStorage.getItem(KEY)||'{}');
const $=id=>document.getElementById(id);
$('annotid').value=localStorage.getItem('annot_id')||'';
function saveId(){localStorage.setItem('annot_id',$('annotid').value);}
function applyFilt(){filt=$('filt').value;view=DATA.map((_,i)=>i).filter(i=>!filt||DATA[i]['档']===filt);idx=0;render();}
function cur(){const i=view[idx];const d=DATA[i];return [d, store[d.uid]||{rec_tags:[],rec_note:'',enw_tags:[],enw_note:''}];}
function render(){
  if(!view.length){$('meta').textContent='无';return;}
  const [d,s]=cur(); const i=view[idx];
  $('pos').textContent=(idx+1)+'/'+view.length+' (全量第'+(i+1)+'条)';
  $('prog').value=(idx+1)/view.length*100;
  $('meta').innerHTML='<b>'+d['档']+'</b> | '+d.uid+' | CER <b>'+d.vanilla_cer+'</b> | sim '+d.max_sim+' | '+d.rec_sec+'s';
  $('ref').innerHTML='<b>ref(正确答案):</b> '+d.ref;
  $('hyp').innerHTML='<b>vanilla(ASR输出):</b> '+(d.vanilla_text||'').slice(0,500);
  $('rec').src=d.rec_src; $('enw').src=d.enw_src;
  $('kws').textContent=d.kws_txt||'(无)';
  $('rec_note').value=s.rec_note; $('enw_note').value=s.enw_note;
  $('rec_tags').innerHTML=HP_REC.map((h,j)=>'<span class="tag '+(s.rec_tags.includes(h)?'on':'')+'" onclick="toggle(\'rec\','+j+')">'+(j+1)+'.'+h+'</span>').join('');
  $('enw_tags').innerHTML=HP_ENW.map((h,j)=>'<span class="tag '+(s.enw_tags.includes(h)?'on':'')+'" onclick="toggle(\'enw\','+j+')">'+(j+1)+'.'+h+'</span>').join('');
}
function toggle(block,j){
  const [d]=cur(); store[d.uid]=store[d.uid]||{rec_tags:[],rec_note:'',enw_tags:[],enw_note:''};
  const o=store[d.uid]; const hp=block==='rec'?HP_REC:HP_ENW; const key=block+'_tags';
  const k=o[key].indexOf(hp[j]);
  if(k>=0) o[key].splice(k,1); else o[key].push(hp[j]);
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function save(field,val){const [d]=cur();store[d.uid]=store[d.uid]||{rec_tags:[],rec_note:'',enw_tags:[],enw_note:''};store[d.uid][field]=val;localStorage.setItem(KEY,JSON.stringify(store));}
function next(){if(idx<view.length-1){idx++;render();}}
function prev(){if(idx>0){idx--;render();}}
function exportCsv(){
  const id=($('annotid').value.trim())||'unknown';
  let csv='annotator,uid,档,CER,max_sim,rec_难点,rec_自然语言,enw_难点,enw_自然语言\n';
  DATA.forEach(d=>{
    const s=store[d.uid]||{rec_tags:[],rec_note:'',enw_tags:[],enw_note:''};
    const rn=(s.rec_note||'').replace(/[\n,]/g,' ');
    const en=(s.enw_note||'').replace(/[\n,]/g,' ');
    csv+=[id,d.uid,d['档'],d.vanilla_cer,d.max_sim,s.rec_tags.join(';'),rn,s.enw_tags.join(';'),en].join(',')+'\n';
  });
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='annot_'+id+'.csv';a.click();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight'){e.preventDefault();next();}
  if(e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();$('rec').play();}
  if(/^[1-9]$/.test(e.key)){toggle('rec',parseInt(e.key)-1);}
});
[...new Set(DATA.map(d=>d['档']))].sort().forEach(b=>{
  const o=document.createElement('option');o.value=b;
  o.textContent=b+' ('+DATA.filter(d=>d['档']===b).length+')';
  $('filt').appendChild(o);
});
applyFilt();
</script></body></html>"""

os.makedirs(PACK, exist_ok=True)
html = (TPL.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
           .replace("__HP_REC__", json.dumps(HP_REC, ensure_ascii=False))
           .replace("__HP_ENW__", json.dumps(HP_ENW, ensure_ascii=False)))
open(HTML_OUT, "w", encoding="utf-8").write(html)
print(f"HTML -> {HTML_OUT} ({len(rows)} 条, recognition+enrollment 分块标注 v2)")

if args.copy_audio:
    os.makedirs(POS_OUT, exist_ok=True)
    n = miss = 0
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
    print("  队员步骤: 解压 → 双击 标注.html → 填名字 → 标两块(难点+自然语言) → 导出 annot_<名字>.csv → 发回")
