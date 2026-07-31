"""build_verify_mv_fail.py — multi-voice 证伪验收包构建.

3 类样本 (A 分离失败 / B mel破坏 / C 双accept) 各取 4-8 条, 每条存:
  enrollment.wav / recognition_original.wav / sep_sourceA.wav / sep_sourceB.wav / argmax_target_slice.wav
  + summary.json (含两路转写/heuristic评分/oracle/字段含义)

输出: code/runs/_verify_mv_fail/{A,B,C}_<uid>/
索引: docs/verify_mv_fail.md

边界: 只读 summary.json + 拷贝现有 wav, 不跑模型, 不改主线, 不 commit.
"""
import json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from text_utils import is_valid_command, _CONTENT_GATE_NEWS_BLACK, _BRAND_ANCHORS
from exp_multivoice_route import (
    cmd_score, _strip_punct, DEVICE_KW, ACTION_KW, FEATURE_KW, QUERY_KW,
    _news_hits, _digit_run_len, NEWS_BLACK_EXT, route_heuristic,
)

OUT_ROOT = os.path.join(HERE, "runs", "_verify_mv_fail")
DOC_PATH = os.path.join(PROJ, "docs", "verify_mv_fail.md")

# 占位, main() 里填充
qwen_text_by_uid = {}
qwen_cer_by_uid = {}

MV_FULL_SLICES = os.path.join(HERE, "runs", "_multivoice_full", "slices")
DZ_SLICES = os.path.join(HERE, "runs", "_deadzone_selector", "slices")
ARGMAX_SLICES = os.path.join(PROJ, "code", "stability_matrix", "_slices")

# ========= 样本选择 =========
# A类: 分离失败 (SepFormer 两路都没拿到 target, oracle_cer>0.8) — multivoice_full
A_UIDS = ["cmd_181", "cmd_2533", "cmd_2786", "cmd_286", "cmd_2997"]

# B类: mel破坏 (主战场 sim≥0.4 & argmax<0.2 但 oracle>0.5) — multivoice_full
B_UIDS = ["cmd_146", "cmd_2007", "cmd_2129", "cmd_277", "cmd_73"]

# C类: 双accept (两路都过 content_gate, TRAP_both_valid) — deadzone_selector
# 覆盖谱: (a)真·非指令gate放过 (b)两路都像指令的真TRAP (c)短文本gate过松
C_UIDS = [
    "cmd_2627",  # src0='新疆地区天气'(天气,非指令) gate放过; heuristic正确选src1
    "cmd_2488",  # src0='点一首刘德华的《冰雨》'(点歌,非指令) gate放过; heuristic正确选src1
    "cmd_2983",  # src0='另外一所幼儿园'(非指令) gate放过; heuristic正确选src1
    "cmd_2541",  # src0='关屏显'(真指令) src1='合约的持续进行'(财经,gate放过!); heuristic选src0
    "cmd_2896",  # src0='您的设置' src1='开启一搜' (两路都像指令,真TRAP) heuristic平局选短→src0错
    "cmd_2452",  # src0='开始播放'(像指令) src1='给我换,都很柔弱的'(非指令) gate放过; heuristic选src0
    "cmd_2102",  # src0='嗯'(L=1, gate过松放过) src1='欢迎诸位朋友'(非指令) gate放过; 都烂
    "cmd_2605",  # src0='纺织吹模式' src1='养殖模式' (两路都命中"模式"特征词,真TRAP) heuristic平局→src1错
]


def _score_breakdown(text):
    """复刻 cmd_score 内部, 拆出每个信号命中细节供用户判读。"""
    if not text or not text.strip():
        return {"empty": True, "score": -5.0}
    raw = text
    valid = is_valid_command(raw)
    dev_hit = [w for w in DEVICE_KW if w in raw]
    act_hit = [w for w in ACTION_KW if w in raw]
    fea_hit = [w for w in FEATURE_KW if w in raw]
    qry_hit = [w for w in QUERY_KW if w in raw]
    bra_hit = [w for w in _BRAND_ANCHORS if w in raw]
    news = _news_hits(raw)
    digit_run = _digit_run_len(raw)
    L = len(_strip_punct(raw))
    s = 0.0
    if not valid:
        s -= 3.0
    if dev_hit: s += 2.0
    if act_hit: s += 2.0
    if fea_hit: s += 1.5
    if qry_hit: s += 2.0
    if bra_hit: s += 3.0
    if dev_hit and act_hit: s += 1.0
    if act_hit and fea_hit: s += 1.0
    if news: s -= 3.0 * news
    if digit_run >= 4: s -= 2.0
    if 3 <= L <= 15: s += 0.5
    elif L > 22: s -= 2.0
    elif L < 3: s -= 1.0
    return {
        "text": raw, "score": round(s, 3), "content_gate_pass": valid,
        "len_after_strip_punct": L,
        "device_kw_hit": dev_hit, "action_kw_hit": act_hit,
        "feature_kw_hit": fea_hit, "query_kw_hit": qry_hit,
        "brand_anchor_hit": bra_hit,
        "news_blacklist_hits": news, "digit_run_len": digit_run,
        "news_words_matched": [w for w in (_CONTENT_GATE_NEWS_BLACK + NEWS_BLACK_EXT) if w in raw],
    }


def _copy_wav(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        return True
    return False


def _build_one(category, uid, src_dir, slice_naming, summary_record, doc_lines):
    """建一个样本包. slice_naming: 'srcA'/'src0'."""
    out_dir = os.path.join(OUT_ROOT, f"{category}_{uid}")
    os.makedirs(out_dir, exist_ok=True)

    # 拷贝 wav
    uid_num = int(uid.split("_")[1])
    enr_src = os.path.join(PROJ, "datasetA", "pos", f"kws_{uid_num}.wav")
    rec_src = os.path.join(PROJ, "datasetA", "pos", f"cmd_{uid_num}.wav")
    if slice_naming == "srcA":
        s1_src = os.path.join(src_dir, f"{uid}__srcA.wav")
        s2_src = os.path.join(src_dir, f"{uid}__srcB.wav")
        s1_name, s2_name = "sep_sourceA.wav", "sep_sourceB.wav"
    else:
        s1_src = os.path.join(src_dir, f"{uid}__src0.wav")
        s2_src = os.path.join(src_dir, f"{uid}__src1.wav")
        s1_name, s2_name = "sep_sourceA.wav", "sep_sourceB.wav"  # 统一命名 A/B
    argmax_src = os.path.join(ARGMAX_SLICES, f"{uid}.wav")

    files = {
        "enrollment.wav": _copy_wav(enr_src, os.path.join(out_dir, "enrollment.wav")),
        "recognition_original.wav": _copy_wav(rec_src, os.path.join(out_dir, "recognition_original.wav")),
        s1_name: _copy_wav(s1_src, os.path.join(out_dir, s1_name)),
        s2_name: _copy_wav(s2_src, os.path.join(out_dir, s2_name)),
        "argmax_target_slice.wav": _copy_wav(argmax_src, os.path.join(out_dir, "argmax_target_slice.wav")),
    }

    # 构 summary
    per_src = summary_record.get("per_src", [])
    per_src_detail = []
    for i, s in enumerate(per_src):
        text = s.get("text", "")
        slice_wav = s1_name if i == 0 else s2_name
        per_src_detail.append({
            "src_idx": s.get("src_idx", i),
            "slice_uid": s.get("slice_uid"),
            "slice_wav": slice_wav,
            "text": text,
            "cer_vs_ref": s.get("cer"),
            **_score_breakdown(text),
        })

    # 主线 argmax 转写 (优先 poc_qwen_asr_full_result rows[].qwen, 其次 deadzone summary 的 argmax_qwen_text_poc)
    argmax_text = qwen_text_by_uid.get(uid) or summary_record.get("argmax_qwen_text_poc") or summary_record.get("argmax_qwen_text") or ""
    argmax_cer = (qwen_cer_by_uid.get(uid)
                  if qwen_cer_by_uid.get(uid) is not None
                  else (summary_record.get("argmax_qwen_cer_poc")
                        if summary_record.get("argmax_qwen_cer_poc") is not None
                        else summary_record.get("argmax_cer")))

    oracle_idx = (summary_record.get("oracle_idx")
                  if summary_record.get("oracle_idx") is not None
                  else summary_record.get("oracle_src_idx"))
    oracle_cer = summary_record.get("oracle_cer")
    heur_idx = summary_record.get("heuristic_idx")
    heur_cer = summary_record.get("heuristic_cer")
    heur_reason = summary_record.get("heuristic_reason")
    heur_scores = summary_record.get("heuristic_scores")

    # 重跑 route_heuristic 拿 scores (若 summary 没有)
    if heur_scores is None and per_src_detail:
        scores = [round(d["score"], 3) for d in per_src_detail]
        heur_idx_re, heur_reason_re = route_heuristic(per_src_detail)
        heur_scores = scores
        if heur_idx is None:
            heur_idx = heur_idx_re
        if heur_reason is None:
            heur_reason = heur_reason_re

    summary_out = {
        "category": category,
        "uid": uid,
        "uid_num": uid_num,
        "ref": summary_record.get("ref", ""),
        "kws_txt": summary_record.get("kws_txt", ""),  # may be empty
        "sim_argmax_target": summary_record.get("sim"),
        "audio_sec": summary_record.get("audio_sec"),
        "files": files,
        "argmax_main_line": {
            "transcript": argmax_text,
            "cer_vs_ref": argmax_cer,
            "note": "主线 enroll_infer qwen 在 target_slice 上的转写 (argmax 选 target)",
            "audio": "argmax_target_slice.wav",
        },
        "sepformer_two_paths": per_src_detail,
        "oracle": {"picked_src_idx": oracle_idx, "cer": oracle_cer,
                   "note": "oracle 选 CER 较低那路 (天花板)"},
        "heuristic_selector": {
            "picked_src_idx": heur_idx, "cer": heur_cer, "reason": heur_reason,
            "scores": heur_scores,
            "note": "route_heuristic: 取 cmd_score 高的那路; 平局 tiebreak 更短",
        },
        "字段含义": {
            "ref": "数据集真值 (目标说话人实际说的家居指令)",
            "argmax_main_line.cer": "主线 enroll_infer qwen 后端在 argmax target 切片上的 CER (主线基准)",
            "sepformer_two_paths[*].cer_vs_ref": "SepFormer 两路分离后各路 qwen 转写 vs ref 的 CER",
            "sepformer_two_paths[*].content_gate_pass": "is_valid_command(text) — content_gate 是否放行(像家居指令)",
            "sepformer_two_paths[*].score": "cmd_score 综合评分(越高越像指令): -3 if !gate, +设备2+动作2+功能1.5+查询2+品牌3+复合, -3×news词-2长数字串, ±长度",
            "oracle.picked_src_idx": "SepFormer 两路里 CER 较低的那路 (选路天花板, 实战不可得)",
            "heuristic_selector": "multi-voice 内容判别选路: 比两路 cmd_score, 高的那路赢; 平局取更短(content_gate 通过的)",
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    # doc 行
    return summary_out


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    global qwen_text_by_uid, qwen_cer_by_uid

    mv_full = json.load(open(os.path.join(HERE, "runs", "_multivoice_full", "summary.json"), encoding="utf-8"))
    dz_sel = json.load(open(os.path.join(HERE, "runs", "_deadzone_selector", "summary.json"), encoding="utf-8"))

    mv_by_uid = {r["uid"]: r for r in mv_full["results"]}
    dz_by_uid = {r["uid"]: r for r in dz_sel["results"]}

    # argmax 主线 qwen 转写 (来自 poc_qwen_asr_full_result.json rows[].qwen)
    qfull = json.load(open(os.path.join(HERE, "runs", "poc_qwen_asr_full_result.json"), encoding="utf-8"))
    qwen_text_by_uid = {r["uid"]: r.get("qwen", "") for r in qfull.get("rows", [])}
    qwen_cer_by_uid = {r["uid"]: r.get("qwen_cer") for r in qfull.get("rows", [])}

    # 补 kws_txt
    pos_jsonl = os.path.join(PROJ, "datasetA", "pos.jsonl")
    kws_map = {}
    if os.path.exists(pos_jsonl):
        for line in open(pos_jsonl, encoding="utf-8"):
            d = json.loads(line)
            kws_map[d["id"]] = d.get("唤醒文本", "")

    doc_lines = ["# multi-voice 证伪验收包索引 (verify_mv_fail)", "",
                 "> 客观摆样本, 不预设结论. 用户亲自听/看 3 类证伪依据是否成立.",
                 "",
                 "- 数据源: code/runs/_multivoice_full + _deadzone_selector",
                 "- 每条样本目录: enrollment.wav / recognition_original.wav / sep_sourceA.wav / sep_sourceB.wav / argmax_target_slice.wav + summary.json",
                 "- argmax_target_slice.wav 来自 code/stability_matrix/_slices/cmd_<N>.wav (主线 enroll_infer qwen 切的 target timeline)",
                 "",
                 "## A 类 — 分离失败 (SepFormer 两路都没拿到 target, oracle_cer>0.8)", ""]
    doc_lines.append(f"共 {len(A_UIDS)} 条. 共同特征: SepFormer 把目标说话人话分到了**两路都不像**的位置, 两路转写都偏离 ref → 选路再准也救不回 (oracle 也 >0.8). 这是**分离本身失败**, 不是 heuristic 选路问题.")
    doc_lines.append("")
    for uid in A_UIDS:
        r = dict(mv_by_uid[uid])
        r["kws_txt"] = kws_map.get(int(uid.split("_")[1]), "")
        s = _build_one("A", uid, MV_FULL_SLICES, "srcA", r, doc_lines)
        doc_lines.append(f"### {uid}  (ref: `{s['ref']}`)")
        doc_lines.append(f"- sim(argmax target) = `{s.get('sim_argmax_target'):.3f}`  audio_sec = `{s.get('audio_sec')}`")
        doc_lines.append(f"- 主线 argmax qwen: `{s['argmax_main_line']['transcript']}`  CER = `{s['argmax_main_line']['cer_vs_ref']}`")
        for d in s["sepformer_two_paths"]:
            doc_lines.append(f"- SepFormer {d['slice_wav']}: `{d['text']}`  CER=`{d['cer_vs_ref']}`  score=`{d['score']}`  gate_pass=`{d['content_gate_pass']}`")
        doc_lines.append(f"- oracle 选 src{s['oracle']['picked_src_idx']} (CER={s['oracle']['cer']})  heuristic 选 src{s['heuristic_selector']['picked_src_idx']} (CER={s['heuristic_selector']['cer']}, reason={s['heuristic_selector']['reason']})")
        doc_lines.append(f"- **该听哪个**: 听 enrollment.wav(目标说话人声纹) → recognition_original.wav(完整混音) → sep_sourceA/B.wav(SepFormer 两路). 判断: SepFormer 两路里有任何一路能听清 ref 吗? 若都听不出 → 分离失败坐实.")
        doc_lines.append("")

    doc_lines += ["", "## B 类 — mel 被破坏 (主战场 argmax 好, SepFormer 后变差)", "",
                  f"共 {len(B_UIDS)} 条. 筛选: sim≥0.4 (主战场, argmax 选对 target) & 主线 argmax CER<0.2 (原本好转写), 但 SepFormer 两路 oracle 都 >0.5. 显示 SepFormer 重分离破坏了原本能转好的 target mel (artifact/失真).", ""]
    for uid in B_UIDS:
        r = dict(mv_by_uid[uid])
        r["kws_txt"] = kws_map.get(int(uid.split("_")[1]), "")
        s = _build_one("B", uid, MV_FULL_SLICES, "srcA", r, doc_lines)
        am = s['argmax_main_line']
        oc = s['oracle']['cer']
        delta = round((oc or 0) - (am['cer_vs_ref'] or 0), 3)
        doc_lines.append(f"### {uid}  (ref: `{s['ref']}`)")
        doc_lines.append(f"- sim = `{s.get('sim_argmax_target'):.3f}`  argmax CER = `{am['cer_vs_ref']}` (好转写)  →  SepFormer oracle CER = `{oc}` (变差, Δ=+{delta})")
        doc_lines.append(f"- 主线 argmax 转写: `{am['transcript']}`")
        for d in s["sepformer_two_paths"]:
            doc_lines.append(f"- SepFormer {d['slice_wav']}: `{d['text']}`  CER=`{d['cer_vs_ref']}`  score=`{d['score']}`")
        doc_lines.append(f"- oracle 选 src{s['oracle']['picked_src_idx']} (CER={s['oracle']['cer']})  heuristic 选 src{s['heuristic_selector']['picked_src_idx']} (CER={s['heuristic_selector']['cer']})")
        doc_lines.append(f"- **该听哪个**: 听 argmax_target_slice.wav (主线切的, qwen 能转好) → sep_sourceA/B.wav (SepFormer 两路). 判断: 主线切片里 ref 听得清吗? SepFormer 两路是不是反而糊了/有 artifact?")
        doc_lines.append("")

    doc_lines += ["", "## C 类 — ★heuristic 双 accept (两路都过 content_gate) — 用户重点核查", "",
                  f"共 {len(C_UIDS)} 条 (从 deadzone_selector 200 条 TRAP_both_valid 集里抽, 该集占 184/200=92%). 筛选: 两路都过 `is_valid_command` (即 TRAP_both_valid). 用户质疑: 两路真的都像家居指令吗? 还是 content_gate 门槛太低把非指令也放过了?",
                  "",
                  "**核查要点**: `content_gate_pass=True` 只代表 \"没命中明显非指令信号(英文主导/news黑名单词/超长/字循环/低多样)\" — 它是为**主流程拒绝新闻/英文幻觉**设计的宽松 gate, **不是判断 \"真家居指令\" 的严格判别器**. 真正选路的是 `cmd_score` (设备词+动作词+品牌锚点).",
                  "",
                  "**所以 92% 双accept 的解读**:",
                  "  - 字面: 92% 的样本两路都过了宽松 content_gate (True)",
                  "  - 实质: 大部分情况**一路是真指令, 另一路是 gate 没拦住的天气/点歌/财经/幼教等短句** → cmd_score 仍能正确区分",
                  "  - 用户该判: 看下面每条两路原文, 真两路都像指令的有几条? gate 放过非指令的有几条?",
                  ""]
    # C类对照表
    doc_lines.append("### C 类两路转写对照表 (用户肉眼判断)")
    doc_lines.append("")
    doc_lines.append("| uid | ref | src0 (sep_sourceA) | src1 (sep_sourceB) | src0 score | src1 score | oracle | heuristic选 | 解读 |")
    doc_lines.append("|---|---|---|---|---|---|---|---|---|")
    for uid in C_UIDS:
        r = dict(dz_by_uid[uid])
        r["kws_txt"] = kws_map.get(int(uid.split("_")[1]), "")
        s = _build_one("C", uid, DZ_SLICES, "src0", r, doc_lines)
        d0, d1 = s["sepformer_two_paths"][0], s["sepformer_two_paths"][1]
        # 解读
        verdict = []
        if not d0["content_gate_pass"] and not d1["content_gate_pass"]:
            verdict.append("两路 gate 都拒(不应在 C 类)")
        elif d0["content_gate_pass"] and d1["content_gate_pass"]:
            # 检查是否真两路都像指令
            like0 = d0["score"] >= 2.0
            like1 = d1["score"] >= 2.0
            if like0 and like1:
                verdict.append("**真双指令** (两路 score≥2)")
            elif like0 or like1:
                verdict.append("**单边像指令** (另一路 gate 放过非指令)")
            else:
                verdict.append("**两路都不像** (gate 双放过短/弱文本)")
        heur_correct = s["heuristic_selector"]["picked_src_idx"] == s["oracle"]["picked_src_idx"]
        verdict.append("heuristic选" + ("对" if heur_correct else "错"))
        doc_lines.append(f"| {uid} | `{s['ref']}` | `{d0['text']}` | `{d1['text']}` | {d0['score']} | {d1['score']} | src{s['oracle']['picked_src_idx']} | src{s['heuristic_selector']['picked_src_idx']} | {' / '.join(verdict)} |")
    doc_lines.append("")

    # C类逐条详情
    doc_lines += ["", "### C 类逐条详情 (含 heuristic 评分拆解)", ""]
    for uid in C_UIDS:
        # 重新读刚写的 summary
        with open(os.path.join(OUT_ROOT, f"C_{uid}", "summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        doc_lines.append(f"#### {uid}  (ref: `{s['ref']}`  sim={s.get('sim_argmax_target'):.3f})")
        d0, d1 = s["sepformer_two_paths"][0], s["sepformer_two_paths"][1]
        # 判读本条
        like_cmd = []  # 哪路像真指令
        for tag, d in [("src0", d0), ("src1", d1)]:
            sigs = (d["device_kw_hit"] or d["action_kw_hit"] or d["feature_kw_hit"]
                    or d["query_kw_hit"] or d["brand_anchor_hit"])
            if d["score"] >= 2.0 and sigs:
                like_cmd.append(tag)
        oc = s['oracle']; hs = s['heuristic_selector']
        heur_correct = hs['picked_src_idx'] == oc['picked_src_idx']
        # 概括本条解读
        if len(like_cmd) == 2:
            case = "真双指令 (两路都命中设备/动作/功能词)"
        elif len(like_cmd) == 1:
            wrong = "src0" if "src0" not in like_cmd else "src1"
            case = f"单边像指令 (真指令={like_cmd[0]}, 另一路 {wrong} gate 放过非指令)"
        else:
            case = "两路都不像 (gate 双放过弱/短文本, 无设备/动作词)"
        if heur_correct:
            if "tie" in hs['reason']:
                why = "选对 (平局 tiebreak 短句, 撞对)"
            elif len(like_cmd) == 1:
                why = f"选对 (cmd_score 看设备/动作词区分, 真指令={like_cmd[0]})"
            else:
                why = "选对"
        else:
            if "tie" in hs['reason']:
                why = "选错 (平局 tiebreak 更短, 撞错)"
            else:
                # cmd_score 给非 target 路 higher score (动作词误命中)
                picked = hs['picked_src_idx']
                picked_tags = []
                d_picked = d0 if picked == 0 else d1
                if d_picked["action_kw_hit"]: picked_tags.append(f"动作词{d_picked['action_kw_hit']}")
                if d_picked["device_kw_hit"]: picked_tags.append(f"设备词{d_picked['device_kw_hit']}")
                why = f"选错 (cmd_score 给 src{picked} 更高分因 {'/'.join(picked_tags) if picked_tags else '弱信号'}, 但 oracle 指向另一路)"
        for d in s["sepformer_two_paths"]:
            doc_lines.append(f"- **{d['slice_wav']}**: `{d['text']}`")
            doc_lines.append(f"  - CER vs ref = `{d['cer_vs_ref']}`  content_gate_pass = `{d['content_gate_pass']}`  cmd_score = `{d['score']}`  L(去标点) = `{d['len_after_strip_punct']}`")
            hits = []
            if d["device_kw_hit"]: hits.append(f"设备词{d['device_kw_hit']}(+2)")
            if d["action_kw_hit"]: hits.append(f"动作词{d['action_kw_hit']}(+2)")
            if d["feature_kw_hit"]: hits.append(f"功能词{d['feature_kw_hit']}(+1.5)")
            if d["query_kw_hit"]: hits.append(f"查询词{d['query_kw_hit']}(+2)")
            if d["brand_anchor_hit"]: hits.append(f"品牌锚{d['brand_anchor_hit']}(+3)")
            if d["news_blacklist_hits"]: hits.append(f"news黑名单命中{d['news_words_matched']}(-{3*d['news_blacklist_hits']})")
            if d["digit_run_len"] >= 4: hits.append(f"长数字串({d['digit_run_len']})(-2)")
            if not d["content_gate_pass"]: hits.append("gate拒(-3)")
            doc_lines.append(f"  - 评分信号: {' / '.join(hits) if hits else '无强信号 (仅长度±0.5)'}")
        doc_lines.append(f"- oracle 选 src{oc['picked_src_idx']} (CER={oc['cer']})  |  heuristic 选 src{hs['picked_src_idx']} (CER={hs['cer']}, reason=`{hs['reason']}`)")
        doc_lines.append(f"- **本条判读**: {case} → heuristic {why}")
        doc_lines.append(f"- **该听哪个**: 听 sep_sourceA.wav vs sep_sourceB.wav. 肉眼/耳判: 真指令那路听得清吗? 非指令那路真的是天气/点歌/财经/幼教等 gate 漏放的吗?")
        doc_lines.append("")

    doc_lines += ["", "## 汇总 — 用户判断后填", "",
                  "- A 类 (分离失败): ___ 条坐实 SepFormer 没分出 target / ___ 条其实有一路能听清 (分离OK只是选路问题)",
                  "- B 类 (mel 破坏): ___ 条 argmax 切片明显比 SepFormer 两路清晰 / ___ 条差不多 (SepFormer 没明显破坏)",
                  "- C 类 (双 accept): ___ 条真双家居指令 / ___ 条单边像指令 gate 放过非指令 / ___ 条两路都不像",
                  "",
                  "## 结论 (不下判断, 用户填)", "",
                  "- multi-voice 整体 NO-GO 的 3 条依据是否成立:",
                  "  - A 分离失败: 成立 / 部分成立 / 不成立",
                  "  - B mel 破坏: 成立 / 部分成立 / 不成立",
                  "  - C 双accept 92%: 数字准确但解读需修正 (gate 宽松 vs 真双指令) / 数字本身有问题",
                  ]

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_lines))
    print(f"[OK] 写出索引 {DOC_PATH}")
    print(f"[OK] 样本包在 {OUT_ROOT}/")
    # 列出
    for cat, uids in [("A", A_UIDS), ("B", B_UIDS), ("C", C_UIDS)]:
        print(f"  {cat}类: {len(uids)} 条 -> {', '.join(uids)}")


if __name__ == "__main__":
    main()
