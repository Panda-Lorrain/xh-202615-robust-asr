"""生成 v2 标注分发包 (2026-07-15, 失败归因+音频刻画+target锁定诊断)。

v1→v2 变革: ①范围收窄到多speaker死区条(106条,失败率67%占80%高价值子集)
  ②标签体系重构: recognition(主环节A-X/B2/C/D + A根因 + 说话人/场景维度 + 干扰人)
                 enrollment(可靠性污染/过短/不清/干净 + 污染时角色区分)
  ③C类程序预填(循环/英文幻觉自动判) + target_active_ratio<0.1警告(可能A-X切错)
  ④26条金标准样例可切换查看(队员对照标)
产物 code/annot_pack/: 标注_v2.html(106条内嵌) + pos/ + README_v2.txt
用法:
  code/.venv/Scripts/python.exe code/build_annotator_pack_v2.py              # 仅HTML
  code/.venv/Scripts/python.exe code/build_annotator_pack_v2.py --copy-audio  # 拷音频
"""
import json, csv, os, re, shutil, argparse

CSV = r"E:\midea_target_asr\code\error_analysis_pos_unfull.csv"
SLICES = r"E:\midea_target_asr\code\out_pos_slices_full.json"
GOLD_CSV = r"E:\midea_target_asr\code\annot_pack\calibration_samples_v2.csv"
PACK = r"E:\midea_target_asr\code\annot_pack"
POS_OUT = os.path.join(PACK, "pos")

ap = argparse.ArgumentParser()
ap.add_argument("--copy-audio", action="store_true")
ap.add_argument("--asr-backend", choices=["vanilla", "qwen"], default="vanilla",
                help="转写源: vanilla(csv原始未归一) 或 qwen(归一后中文,比赛主线)")
ap.add_argument("--range", choices=["dead_multi", "all"], default="dead_multi",
                help="dead_multi=死区多speaker(发包106条) / all=全量未满分(自己浏览)")
args = ap.parse_args()
_rng = "_all" if args.range == "all" else ""
HTML_OUT = os.path.join(PACK, "标注_v2" + ("_qwen" if args.asr_backend == "qwen" else "") + _rng + ".html")
GOLD_CSV = os.path.join(PACK, "calibration_samples_v2" + ("_qwen" if args.asr_backend == "qwen" else "") + ".csv")

# ===== slice 数据(预填用) =====
sl = {}
for s in json.load(open(SLICES, encoding="utf-8")):
    base = os.path.basename(s.get("recognition", "")).replace(".wav", "")
    sl[base] = s

# ===== 唤醒词 =====
KWS = {}
for p in json.load(open(r"E:\midea_target_asr\code\pos_pairs_datasetA.json", encoding="utf-8")):
    KWS[p["id"]] = p.get("kws_txt", "")


def has_loop(t):
    return bool(re.search(r'(.{3,}?)\1', t or ''))


def has_eng(v, r):
    return bool(re.search(r'[a-zA-Z]{2,}', v or '')) and not bool(re.search(r'[a-zA-Z]{2,}', r or ''))


# ===== qwen 转写 + 归一链 (--asr-backend qwen 时启用) =====
QWEN = {}
if args.asr_backend == "qwen":
    _pj = json.load(open(r"E:\midea_target_asr\code\poc_qwen_asr_full_result.json", encoding="utf-8"))
    QWEN = {x["uid"]: {"qwen": x.get("qwen", ""), "qwen_cer": x.get("qwen_cer", "")} for x in _pj["rows"]}


def _norm_asr(text):
    """归一链: 繁→简 → 数字(qwen多为no-op) → 官方normalize_text(NFKC+lower+去所有P*标点+空白)。
    ⚠️ 2026-07-15: 末步从枚举正则改为复用官方口径——原正则[，。！？...]漏《》〈〉「」『』【】等
    中文标点,致 disp 残留书名号、标注界面把 qwen 自发加的《》当多余字误判扣分(官方口径已去标点,
    实际不扣; qwen_cer 字段本就用官方口径算, 显示层归一没跟上)。"""
    try:
        from text_utils import to_simplified, digit_postproc
        t = to_simplified(text or "")
        t = digit_postproc(t)
    except Exception:
        t = text or ""
    from eval_metrics import normalize_text
    return normalize_text(t)


def _tier(cer):
    """按 CER 分档。qwen 版用 qwen_cer 重分,避免档(vanilla)与cer(qwen)打架。
    ⚠️ 2026-07-16: 拆出 5_满分CER=0 档——原 c<=0.1 一律归轻微,致 417 条 qwen 已转对的满分
    样本全挤在轻微(435)里淹没真小错(仅18条)。现 CER==0 独立成档,轻微=(0,0.1]。"""
    try:
        c = float(cer)
    except (TypeError, ValueError):
        return "未分档"
    if c > 1:
        return "1_死区CER>1"
    if c > 0.5:
        return "2_严重0.5-1"
    if c > 0.1:
        return "3_中等0.1-0.5"
    if c > 0:
        return "4_轻微0-0.1"
    return "5_满分CER=0"


# ===== 筛选: dead_multi=死区×多speaker(发包) / all=全量未满分(浏览) =====
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
data = []
for r in rows:
    s = sl.get(r["uid"], {})
    ref = r.get("ref", "") or ""
    if args.asr_backend == "qwen":
        q = QWEN.get(r["uid"], {})
        disp = _norm_asr(q.get("qwen", r.get("vanilla_text", "")))
        cer_val = q.get("qwen_cer") if "qwen_cer" in q else r.get("vanilla_cer", "")
        档 = _tier(cer_val)
    else:
        disp = (r.get("vanilla_text", "") or "")[:500]
        cer_val = r.get("vanilla_cer", "")
        档 = r["档"]
    if args.range == "dead_multi":
        if "死区" not in 档:
            continue
        if len(s.get("speakers", [])) < 2:
            continue
    auto_c = []
    if has_loop(disp):
        auto_c.append("C1循环")
    if has_eng(disp, ref):
        auto_c.append("C2英文")
    _id = int(r["uid"].replace("cmd_", "")) if r["uid"].startswith("cmd_") else -1
    try:
        _solved = float(cer_val) == 0
    except (TypeError, ValueError):
        _solved = False
    data.append({
        "uid": r["uid"], "档": 档, "cer": cer_val, "sim": r.get("max_sim", ""),
        "ref": ref, "vanilla": disp, "rec_src": "pos/" + os.path.basename(r["recognition_path"]),
        "enw_src": "pos/" + os.path.basename(r["enrollment_path"]), "kws": KWS.get(_id, ""),
        "spk_n": len(s.get("speakers", [])), "tar_act": round(s.get("target_active_ratio", 0) or 0, 3),
        "auto_c": ";".join(auto_c), "rec_sec": r.get("rec_sec", ""), "solved": _solved,
    })
print(f"筛选: {len(data)} 条 ({'全量未满分' if args.range=='all' else '死区×多speaker'})")

# ===== 26条金标准样例(只读对照) =====
gold = []
if os.path.isfile(GOLD_CSV):
    for r in csv.DictReader(open(GOLD_CSV, encoding="utf-8-sig")):
        gold.append({k: r[k] for k in ("uid", "rec_主环节", "rec_根因", "目标说话人", "场景", "干扰人内容", "enw_可靠性", "置信")})

# ===== v2 标签选项 =====
MAIN = ["A-X选错target", "B2重叠抢麦", "C模型解码崩", "D无难点"]
CAUSE = ["A-E1 enrollment污染", "A-E2过短/质量差", "A-R1目标被压制", "A-R2音色混淆", "A-?根因不明"]
FORCE = ["轻声", "正常", "大声"]
SPEED = ["慢", "正常", "快"]
CLEAR = ["清晰", "含糊"]
ACCENT = ["标准", "方言", "口音"]
ENV = ["安静室内", "家电运行", "户外", "办公", "多人环境"]
BG = ["无", "人声babble", "音乐", "电视节目", "噪声"]
ENW_REL = ["E-污染", "E-过短", "E-不清", "E-干净"]

TPL = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>v2标注-失败归因</title>
<style>
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f0f2f5;color:#2c3e50;line-height:1.6;font-size:14px}
.card{max-width:960px;margin:0 auto;padding:18px;min-height:100vh}
.nav{display:flex;justify-content:space-between;align-items:center;background:#fff;border-radius:12px;padding:10px 16px;margin-bottom:8px;box-shadow:0 2px 8px rgba(0,0,0,.06);flex-wrap:wrap;gap:6px}
.nav #pos{font-weight:700;color:#1976d2}
progress{width:100%;height:6px;-webkit-appearance:none;border:none;border-radius:4px;margin-bottom:10px;display:block}
progress::-webkit-progress-bar{background:#e4e7ed;border-radius:4px}
progress::-webkit-progress-value{background:linear-gradient(90deg,#1976d2,#42a5f5);border-radius:4px}
.toolbar{display:flex;align-items:center;gap:8px;flex-wrap:wrap;background:#fff;border-radius:12px;padding:10px 16px;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.tlbl{color:#78909c;font-size:12px;font-weight:600}
.toolbar .hint{margin-left:auto;font-size:11px}
.idbox,.sel{padding:5px 10px;border:1.5px solid #cfd8dc;border-radius:8px;font-size:13px;outline:none;background:#fff}
.idbox:focus{border-color:#1976d2}
.warn{background:#fff8e1;border:1px solid #ffe082;border-left:4px solid #ffb300;padding:8px 12px;border-radius:8px;margin:8px 0;font-size:13px}
.warn b{color:#e65100}
.prefill{background:#ede7f6;border:1px solid #b39ddb;border-left:4px solid #7e57c2;padding:8px 12px;border-radius:8px;margin:8px 0;font-size:13px}
.meta{background:#fff;border-radius:12px;padding:10px 16px;margin-bottom:14px;box-shadow:0 2px 8px rgba(0,0,0,.06);font-size:13px;color:#546e7a}
.meta b{color:#1976d2}
.section{background:#fff;border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,.08);border:1px solid #eceff1}
.section h3{margin:-18px -18px 14px;padding:12px 18px;font-size:15px;color:#fff;background:#1976d2;border-radius:14px 14px 0 0}
.section.enw h3{background:#fb8c00}
.ref{background:#f1f8f4;border:1px solid #c8e6c9;border-left:4px solid #4caf50;padding:8px 12px;margin:8px 0;border-radius:8px;font-size:13px}
.ref b{color:#2e7d32}
.hyp{background:#fdf3f4;border:1px solid #ffcdd2;border-left:4px solid #f44336;padding:8px 12px;margin:8px 0;border-radius:8px;word-break:break-all;font-size:13px}
.hyp b{color:#c62828}
audio{width:100%;margin:8px 0;display:block}
.lbl{font-size:11px;color:#90a4ae;font-weight:700;margin:10px 0 4px;text-transform:uppercase;letter-spacing:.6px}
.sub{font-size:12px;color:#5c6bc0;margin:8px 0 3px;font-weight:600}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin:3px 0}
.tag{padding:5px 12px;border:1.5px solid #cfd8dc;border-radius:18px;cursor:pointer;font-size:13px;user-select:none;background:#fff;color:#546e7a;transition:all .12s}
.tag:hover{border-color:#1976d2;color:#1976d2}
.tag.on{background:#1976d2;color:#fff;border-color:#1976d2}
.tag.main.on{background:#d32f2f;border-color:#d32f2f}
.section.enw .tag:hover{border-color:#fb8c00;color:#fb8c00}
.section.enw .tag.on{background:#fb8c00;border-color:#fb8c00}
.dim{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0}
.dim .dg{flex:1;min-width:120px}
.dim .dl{font-size:11px;color:#78909c;margin-bottom:2px}
textarea{width:100%;margin:5px 0;padding:8px 12px;border:1.5px solid #e0e0e0;border-radius:8px;font-size:13px;font-family:inherit;resize:vertical;outline:none}
textarea:focus{border-color:#1976d2}
button{padding:6px 14px;cursor:pointer;border:1.5px solid #cfd8dc;border-radius:8px;background:#fff;color:#455a64;font-size:13px;font-weight:500}
button:hover{background:#f5f7fa;border-color:#1976d2;color:#1976d2}
button.primary{background:#1976d2;color:#fff;border-color:#1976d2;font-weight:600}
button.gold{background:#43a047;color:#fff;border-color:#43a047}
.hint{color:#90a4ae;font-size:11px}
.gold-modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.5);z-index:99;overflow:auto}
.gold-modal .inner{max-width:920px;margin:30px auto;background:#fff;border-radius:14px;padding:20px}
.gold-modal table{border-collapse:collapse;width:100%;font-size:12px}
.gold-modal th,.gold-modal td{border:1px solid #ddd;padding:5px 7px;text-align:left;vertical-align:top}
.gold-modal th{background:#43a047;color:#fff;position:sticky;top:0}
</style></head><body><div class="card">
<div class="nav">
 <button onclick="prev()">◀ 上一条(←)</button><span id="pos">0/0</span><button onclick="next()">下一条(▶)</button>
</div>
<progress id="prog" max="100" value="0"></progress>
<div class="toolbar">
 <span class="tlbl">标注员</span><input id="annotid" class="idbox" placeholder="你的名字" oninput="localStorage.setItem('annot_id',this.value)">
 <span class="tlbl">档</span><select id="filt" class="sel" onchange="applyFilt()"><option value="">全部</option></select>
 <label style="font-size:12px;color:#546e7a;user-select:none;cursor:pointer"><input type="checkbox" id="showSolved" onchange="applyFilt()"> 显示已解决(cer=0)</label>
 <button class="primary" onclick="exportCsv()">⬇ 导出CSV</button>
 <span class="tlbl">跳转</span><input id="jump" class="idbox" style="width:96px" placeholder="uid/条号" onkeydown="if(event.key==='Enter')jumpTo()">
 <button class="gold" onclick="document.getElementById('gm').style.display='block'">📖 26条金标准样例</button>
 <span class="hint">←→翻 / 空格播rec / 主环节1-4 / 跳转回车</span>
</div>
<div class="meta" id="meta"></div>
<div id="prefill_box"></div>

<div class="section rec">
 <h3>① recognition（带噪识别音频）— 失败归因 + 音频刻画</h3>
 <div class="ref" id="ref"></div><div class="hyp" id="hyp"></div>
 <audio id="rec" controls></audio>
 <div class="lbl">主环节（对CER贡献最大的失败，单选 / 快捷键1-4）</div>
 <div class="tags" id="main_tags"></div>
 <div id="cause_box">
   <div class="sub">A-X 根因（多选，判不出留空）</div>
   <div class="tags" id="cause_tags"></div>
   <div class="sub">干扰人说了什么（A-X 时填，判定依据）</div>
   <textarea id="interfer" placeholder="例：背景男声念新闻'重审一天没有生产'" oninput="save('interfer',this.value)"></textarea>
 </div>
 <div class="lbl">目标说话人特征</div>
 <div class="dim">
   <div class="dg"><div class="dl">力度</div><div class="tags" id="force_tags"></div></div>
   <div class="dg"><div class="dl">语速</div><div class="tags" id="speed_tags"></div></div>
   <div class="dg"><div class="dl">清晰度</div><div class="tags" id="clear_tags"></div></div>
   <div class="dg"><div class="dl">口音</div><div class="tags" id="accent_tags"></div></div>
 </div>
 <div class="lbl">场景</div>
 <div class="dim">
   <div class="dg"><div class="dl">环境</div><div class="tags" id="env_tags"></div></div>
   <div class="dg"><div class="dl">背景声</div><div class="tags" id="bg_tags"></div></div>
 </div>
 <div class="lbl">备注（机制推断/特殊观察）</div>
 <textarea id="rec_note" placeholder="例：声纹识别错人，转写的是背景人..." oninput="save('rec_note',this.value)"></textarea>
</div>

<div class="section enw">
 <h3>② enrollment（唤醒词音频）— 声纹来源可靠性 ⚠️ 不能假设干净</h3>
 <div class="hint">唤醒词: <b id="kws" style="color:#fb8c00"></b></div>
 <audio id="enw" controls></audio>
 <div class="lbl">enrollment 可靠性（单选）</div>
 <div class="tags" id="enw_rel_tags"></div>
 <div id="role_box">
   <div class="sub">target 时段（谁/何时念唤醒词）</div>
   <textarea id="enw_target" placeholder="例：前半段女声念了'小钱小钱'" oninput="save('enw_target',this.value)"></textarea>
   <div class="sub">干扰人时段/内容</div>
   <textarea id="enw_interfer" placeholder="例：全程男声念新闻'重审一天没有生产'" oninput="save('enw_interfer',this.value)"></textarea>
 </div>
</div>
</div>

<div class="gold-modal" id="gm"><div class="inner">
 <h3 style="margin-top:0">📖 26 条金标准样例（负责人标注，对照参考）</h3>
 <p class="hint">这是负责人已确认的标注范例。你标注时对照这个标准，理解每个标签怎么用。</p>
 <table><thead><tr><th>uid</th><th>主环节</th><th>根因</th><th>目标说话人</th><th>场景</th><th>干扰人内容</th><th>enrollment</th></tr></thead>
 <tbody id="gm_body"></tbody></table>
 <button onclick="document.getElementById('gm').style.display='none'" style="margin-top:12px">关闭</button>
</div></div>

<script>
const DATA=__DATA__, MAIN=__MAIN__, CAUSE=__CAUSE__,
      FORCE=__FORCE__, SPEED=__SPEED__, CLEAR=__CLEAR__, ACCENT=__ACCENT__, ENV=__ENV__, BG=__BG__, ENW_REL=__ENW_REL__,
      GOLD=__GOLD__, KEY='err_annot_v3';
let filt="", idx=0, view=DATA.map((_,i)=>i);
let store=JSON.parse(localStorage.getItem(KEY)||'{}');
const $=id=>document.getElementById(id);
$('annotid').value=localStorage.getItem('annot_id')||'';
function applyFilt(){filt=$('filt').value;const sw=$('showSolved').checked;view=DATA.map((_,i)=>i).filter(i=>{if(!sw&&DATA[i].solved&&filt!=='5_满分CER=0')return false;return !filt||DATA[i]['档']===filt;});idx=0;render();}
function empty(){return{rec_main:'',rec_cause:[],force:'',speed:'',clear:'',accent:'',env:'',bg:'',interfer:'',rec_note:'',enw_rel:'',enw_target:'',enw_interfer:''};}
function cur(){const i=view[idx];const d=DATA[i];return [d, store[d.uid]||(store[d.uid]=empty())];}
function tagsRender(elid, opts, field, multi, extraOn){
  const [d,s]=cur(); const el=$(elid); if(!el)return;
  const fn=multi?'togg':'pick';
  el.innerHTML=opts.map(o=>'<span class="tag'+(extraOn?' main':'')+(s[field]===o||(multi&&s[field].includes(o))?' on':'')+'" onclick="'+fn+'(\''+field+'\',\''+o.replace(/'/g,"\\'")+'\')">'+o+'</span>').join('');
}
function pick(field,val){
  const [d]=cur(); store[d.uid][field]=store[d.uid][field]===val?'':val;
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function togg(field,val){
  const [d]=cur(); const a=store[d.uid][field]; const k=a.indexOf(val);
  if(k>=0)a.splice(k,1);else a.push(val);
  localStorage.setItem(KEY,JSON.stringify(store)); render();
}
function save(field,val){const [d]=cur();store[d.uid][field]=val;localStorage.setItem(KEY,JSON.stringify(store));}
function render(){
  if(!view.length){$('meta').textContent='无';return;}
  const [d,s]=cur(); const i=view[idx];
  $('pos').textContent=(idx+1)+'/'+view.length+' (第'+(i+1)+'条)';
  $('prog').value=(idx+1)/view.length*100;
  $('meta').innerHTML='<b>'+d['档']+'</b> | '+d.uid+' | CER <b>'+d.cer+'</b> | sim '+d.sim+' | '+d.rec_sec+'s | '+d.spk_n+'speakers';
  $('ref').innerHTML='<b>ref(target应说):</b> '+d.ref;
  $('hyp').innerHTML='<b>__ASR_LABEL__(系统输出):</b> '+(d.vanilla||'');
  $('rec').src=d.rec_src; $('enw').src=d.enw_src; $('kws').textContent=d.kws||'(无)';
  // 预填提示
  let pf='';
  if(d.auto_c) pf+='<div class="prefill">🤖 程序预判 C 类幻觉: <b>'+d.auto_c+'</b>（已自动勾选"模型解码崩"，请复核——若 target_active_ratio 低，实为切错 target）</div>';
  if(d.tar_act<0.1) pf+='<div class="warn">⚠️ target_active_ratio=<b>'+d.tar_act+'</b>（极低，几乎没切到 target）→ 大概率 <b>A-X 选错 target</b>，ASR 崩成幻觉只是表象，别直接归 C！</div>';
  $('prefill_box').innerHTML=pf;
  // 默认勾C(若auto_c且未标)
  if(d.auto_c && !s.rec_main){s.rec_main='C模型解码崩';}
  // 各tag组
  tagsRender('main_tags',MAIN,'rec_main',false,true);
  tagsRender('cause_tags',CAUSE,'rec_cause',true,false);
  tagsRender('force_tags',FORCE,'force',false,false);
  tagsRender('speed_tags',SPEED,'speed',false,false);
  tagsRender('clear_tags',CLEAR,'clear',false,false);
  tagsRender('accent_tags',ACCENT,'accent',false,false);
  tagsRender('env_tags',ENV,'env',false,false);
  tagsRender('bg_tags',BG,'bg',false,false);
  tagsRender('enw_rel_tags',ENW_REL,'enw_rel',false,false);
  // 显隐
  $('cause_box').style.display=(s.rec_main==='A-X选错target')?'block':'none';
  $('role_box').style.display=(s.enw_rel==='E-污染')?'block':'none';
  $('interfer').value=s.interfer||''; $('rec_note').value=s.rec_note||'';
  $('enw_target').value=s.enw_target||''; $('enw_interfer').value=s.enw_interfer||'';
}
function next(){if(idx<view.length-1){idx++;render();}}
function prev(){if(idx>0){idx--;render();}}
function jumpTo(){
  const v=$('jump').value.trim(); if(!v)return;
  let key=v.startsWith('cmd_')?v:(/^\d+$/.test(v)?'cmd_'+v:v);
  let i=view.findIndex(k=>DATA[k].uid===key);
  if(i<0 && /^\d+$/.test(v)){ i=parseInt(v)-1; if(i<0||i>=view.length)i=-1; }
  if(i>=0){idx=i;render();$('jump').value='';}else{alert('未找到 '+v+' (当前筛选下)');}
}
function exportCsv(){
  const id=($('annotid').value.trim())||'unknown';
  let csv='annotator,uid,档,CER,max_sim,rec_主环节,rec_根因,力度,语速,清晰度,口音,环境,背景声,干扰人内容,rec备注,enw_可靠性,enw_target时段,enw_干扰人内容\n';
  DATA.forEach(d=>{const s=store[d.uid]||empty();
    const cl=v=>(v||'').toString().replace(/[\n,]/g,' ');
    csv+=[id,d.uid,d['档'],d.cer,d.sim,s.rec_main,s.rec_cause.join(';'),s.force,s.speed,s.clear,s.accent,s.env,s.bg,cl(s.interfer),cl(s.rec_note),s.enw_rel,cl(s.enw_target),cl(s.enw_interfer)].join(',')+'\n';});
  const b=new Blob([csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='annot_v2_'+id+'.csv';a.click();
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.key==='ArrowRight'){e.preventDefault();next();}
  if(e.key==='ArrowLeft'){e.preventDefault();prev();}
  if(e.key===' '){e.preventDefault();$('rec').play();}
  if(/^[1-4]$/.test(e.key)){pick('rec_main',MAIN[parseInt(e.key)-1]);}
});
[...new Set(DATA.map(d=>d['档']))].sort().forEach(b=>{const o=document.createElement('option');o.value=b;o.textContent=b+' ('+DATA.filter(d=>d['档']===b).length+')';$('filt').appendChild(o);});
// 金标准样例表
$('gm_body').innerHTML=GOLD.map(g=>'<tr><td>'+g.uid+'</td><td><b>'+g['rec_主环节']+'</b></td><td>'+g['rec_根因']+'</td><td>'+g['目标说话人']+'</td><td>'+g['场景']+'</td><td>'+g['干扰人内容']+'</td><td>'+g['enw_可靠性']+'</td></tr>').join('');
applyFilt();
</script></body></html>"""

os.makedirs(PACK, exist_ok=True)


def J(x):
    return json.dumps(x, ensure_ascii=False)


html = (TPL
        .replace("__DATA__", J(data))
        .replace("__MAIN__", J(MAIN)).replace("__CAUSE__", J(CAUSE))
        .replace("__FORCE__", J(FORCE)).replace("__SPEED__", J(SPEED))
        .replace("__CLEAR__", J(CLEAR)).replace("__ACCENT__", J(ACCENT))
        .replace("__ENV__", J(ENV)).replace("__BG__", J(BG))
        .replace("__ENW_REL__", J(ENW_REL)).replace("__GOLD__", J(gold)).replace("__ASR_LABEL__", args.asr_backend))
open(HTML_OUT, "w", encoding="utf-8").write(html)
print(f"HTML -> {HTML_OUT} ({len(data)} 条多speaker死区, v2 失败归因标注)")

README = """══════════════════════════════════════════
       v2 标注 · 失败归因 + target 锁定诊断
══════════════════════════════════════════

■ 范围
本包只标 106 条「多 speaker 死区(CER>1)」——这是失败主战场
(diar 分出≥2人、argmax 选错 target 占失败 80%)。性价比最高。

■ 核心思路（和 v1 不同）
v1 标"音频里有什么难点"(音量/语速/babble…)——对改进模型没指导意义。
v2 标"模型在哪一环失败了"——直接告诉你该改哪个模块：
  A-X 选错 target → 改前端选 target(多声纹路由)  ← 主战场
  B2 重叠抢麦     → 改前端分离
  C 模型解码崩    → 改 ASR 解码/换后端
  D 无难点        → 边界噪声

■ 每条怎么标（看规范详版: annotation_spec_v2.md）

【① recognition 块】
1. 先看顶部预填：
   · 🤖程序预判C类(循环/英文幻觉) → 已自动勾"模型解码崩"
   · ⚠️target_active_ratio<0.1 → 大概率是 A-X 选错 target，别直接归C！
2. 主环节单选(1-4)：对比 ref 和 __ASR_LABEL__输出
   · __ASR_LABEL__输出是别人的话(新闻/闲聊，不像家居指令) → A-X 选错 target
   · __ASR_LABEL__输出是 target 但错乱 → B2 重叠
   · __ASR_LABEL__输出是循环/英文/编造幻觉 → C 模型崩
   · 音频清晰错误小 → D
3. A-X 时填根因(多选) + 干扰人说了什么
4. 目标说话人特征(力度/语速/清晰/口音) + 场景(环境/背景声)

【② enrollment 块】⚠️ 不能假设干净！
1. 可靠性单选：听到两人同说 → E-污染(关键！)
2. E-污染 时填：target 时段(谁念唤醒词) + 干扰人时段/内容

■ 26 条金标准样例
顶部「📖26条金标准样例」按钮 → 负责人已标好的范例，对照着理解标签用法。

■ 步骤
1) 解压 → 双击「标注_v2.html」(Firefox 优先;Chrome 不响见下)
2) 填标注员名字 → 逐条标 → 导出 annot_v2_你的名字.csv → 发回

Chrome 音频不响：解压目录开 cmd 跑 python -m http.server 8000 再访问 localhost:8000/标注_v2.html

■ ⚠ 自动存浏览器，别清缓存；可分多次标。
══════════════════════════════════════════
"""
_readme_p = os.path.join(PACK, "README_v2" + ("_qwen" if args.asr_backend == "qwen" else "") + ".txt")
open(_readme_p, "w", encoding="utf-8").write(README.replace("__ASR_LABEL__", args.asr_backend))
print(f"README -> {_readme_p}")

if args.copy_audio:
    os.makedirs(POS_OUT, exist_ok=True)
    n = miss = 0
    for r in rows:
        s = sl.get(r["uid"], {})
        if args.range == "dead_multi":
            if "死区" not in r["档"]:
                continue
            if len(s.get("speakers", [])) < 2:
                continue
        for k in ("recognition_path", "enrollment_path"):
            src = r[k]
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(POS_OUT, os.path.basename(src)))
                n += 1
            else:
                miss += 1
    print(f"拷音频 {n} 个 -> {POS_OUT}" + (f" (缺 {miss})" if miss else ""))
print(f"\n打包完成: {PACK}")
print("  发队员: 压缩 annot_pack 成 zip → 网盘发 2 队员")
print("  队员: 解压 → 标注_v2.html → 填名 → 标 → 导出 annot_v2_<名>.csv → 发回")
