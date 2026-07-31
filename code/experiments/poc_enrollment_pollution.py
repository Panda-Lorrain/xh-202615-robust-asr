#!/usr/bin/env python3
"""POC 方向A: enrollment 污染检测 → 污染项自适应提 thr, 能否不损 pos CER 地提 neg RR。

验证 3 子问题(对应 spec docs/superpowers/specs/2026-07-16-rr-improvement-poc-design.md):
  Q1 检测手段可行性: 1.8s enrollment 上 DiariZen diar(D1) / wespeaker 帧聚类(D2) 能否检测≥2人污染, F1>0.7?
  Q2 污染与漏拒相关性: 45 漏拒 neg(max_sim≥0.27) 的 enrollment 污染率 vs 全474基线, ≥1.5× 且 Fisher p<0.05?
  Q3 自适应 thr 代价: 污染 enr 用 thr_high∈{0.30,0.35,0.40}, 干净用0.27, hold-out val 上 ΔRR>0 且 Δpos CER≤0.01?

环境: code/.venv/Scripts/python.exe (复用 enroll_infer 同环境, DiariZen+wespeaker+GPU)
用法:
  python poc_enrollment_pollution.py --stage load   # smoke 验证数据加载
  python poc_enrollment_pollution.py --stage q1     # 26条calibration 算D1/D2 F1
  python poc_enrollment_pollution.py --stage q2     # 474neg 污染与漏拒相关性+Fisher
  python poc_enrollment_pollution.py --stage q3     # 1364pos hold-out thr_high扫描
  python poc_enrollment_pollution.py --stage all    # 全跑(q1→q2→q3)
输出: code/poc_pollution_cache.json (diar结果缓存, 避免重跑) + stdout 数据表
"""
# ---- speechbrain lazy patch (复制自 enroll_infer.py:24-29, 否则 librosa 经 lazy_loader→inspect 崩) ----
import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import os, sys, json, csv, argparse, time
import numpy as np
import torch
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
# DiariZen / pyannote 加入 sys.path (复制自 enroll_infer.py:44-48)
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p): sys.path.insert(0, _p)

from repro import set_global_seed, resolve_model
from eval_metrics import CERMetric
from text_utils import to_simplified, digit_postproc

CACHE_PATH = os.path.join(_HERE, "poc_pollution_cache.json")
NEG_RESULT = os.path.join(_HERE, "out_neg_vanilla_full", "result.json")
POS_RESULT = os.path.join(_HERE, "out_pos_vanilla_full", "result.json")
QWEN_ROWS = os.path.join(_HERE, "poc_qwen_asr_full_result.json")
CALIB_CSV = os.path.join(_HERE, "annot_pack", "calibration_samples_v2.csv")
NEG_PAIRS = os.path.join(_HERE, "neg_pairs_datasetA.json")
POS_PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")


# ============ 模型加载 (复制 enroll_infer.py:163-177 接口) ============
def load_models(device_str="cuda:0"):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    from diarizen.pipelines.inference import DiariZenPipeline
    print(f"[load] DiariZen {resolve_model('DIAR')} on {device}")
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1: w = w[None, None]
        elif w.dim() == 2: w = w[None]
        with torch.no_grad():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0).cpu().numpy()

    return diar, get_emb, device


# ============ 数据加载 ============
def load_neg():
    """out_neg_vanilla_full/result.json → [{uid, enrollment, max_sim}]。n=474。"""
    d = json.load(open(NEG_RESULT, encoding="utf-8"))
    out = []
    for r in d["results"]:
        uid = "_".join(r["utt_id"].split("_")[1:])  # utt0000_cmd_1000 → cmd_1000
        out.append({"uid": uid, "enrollment": r["enrollment"], "max_sim": r["max_sim"]})
    return out

def load_pos():
    """join out_pos_vanilla_full(enrollment+max_sim) × poc_qwen rows(ref+qwen) by cmd_X。n≈1350。"""
    pos_res = json.load(open(POS_RESULT, encoding="utf-8"))["results"]
    enr_by_uid = {"_".join(r["utt_id"].split("_")[1:]): r["enrollment"] for r in pos_res}
    sim_by_uid = {"_".join(r["utt_id"].split("_")[1:]): r["max_sim"] for r in pos_res}
    qrows = json.load(open(QWEN_ROWS, encoding="utf-8"))["rows"]
    out = []
    for r in qrows:
        uid = r["uid"]
        if uid not in enr_by_uid:
            continue
        out.append({"uid": uid, "enrollment": enr_by_uid[uid],
                    "max_sim": r["sim"], "ref": r["ref"], "qwen": r["qwen"]})
    return out

def load_calib_truth(pos_data):
    """calibration_samples_v2.csv → [(enr_path, is_polluted)]。
    uid=cmd_X → enrollment 从 pos_data 映射; 过短样本剔除(方向A只看污染)。"""
    uid2enr = {p["uid"]: p["enrollment"] for p in pos_data}
    out = []
    n_skip = 0
    for r in csv.DictReader(open(CALIB_CSV, encoding="utf-8-sig")):
        uid = r["uid"]
        rel = r.get("enw_可靠性", "") or ""
        if "过短" in rel:  # 剔除过短(另一失效模式, 方向A只看污染)
            n_skip += 1
            continue
        if uid not in uid2enr:
            n_skip += 1
            continue
        polluted = ("污染" in rel) and ("非污染" not in rel)  # 排除"(非污染,用户确认)"误匹配
        out.append((uid2enr[uid], polluted, uid, rel))
    print(f"[calib] 加载 {len(out)} 条 (剔除过短/找不到enr {n_skip} 条)")
    return out


# ============ 缓存 (diar 慢, 1838次必须缓存) ============
def load_cache():
    if os.path.exists(CACHE_PATH):
        return json.load(open(CACHE_PATH, encoding="utf-8"))
    return {}

def save_cache(cache):
    json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ============ 检测器 ============
def detect_pollution_diar(diar, enr_path):
    """D1: enrollment 跑 DiariZen diar, speakers≥2 → 污染。返回 (is_polluted|None, n_spk)。None=diar失败。"""
    try:
        diar_out = diar(enr_path)
        n = len(diar_out.labels())
        return (n >= 2, n)
    except Exception as e:
        print(f"  [diar-fail] {os.path.basename(enr_path)}: {type(e).__name__} {str(e)[:60]}")
        return (None, None)

def detect_pollution_cluster(get_emb, enr_path, frame_sec=0.3, sim_thresh=0.5):
    """D2: enrollment 切 0.3s 帧 → wespeaker emb → 余弦层次聚类 → 簇≥2 污染。"""
    from sklearn.cluster import AgglomerativeClustering
    wav, _ = librosa.load(enr_path, sr=16000)
    fl = int(frame_sec * 16000)
    if len(wav) < fl * 2:
        return (False, 1)  # 太短无法分帧
    frames = [wav[i:i+fl] for i in range(0, len(wav)-fl+1, fl)]
    embs = np.stack([get_emb(f) for f in frames])
    dist = 1.0 - embs @ embs.T
    dist = np.clip(dist, 0, 2)
    labels = AgglomerativeClustering(
        n_clusters=None, metric="precomputed", linkage="average",
        distance_threshold=1.0 - sim_thresh).fit_predict(dist)
    n = len(set(labels))
    return (n >= 2, n)


def get_polluted(detect_fn_name, detect_fn, items, cache, key_prefix):
    """对 items(含 enrollment) 跑检测, 带缓存。返回 {uid: is_polluted|None}。"""
    polluted = {}
    t0 = time.time()
    for i, it in enumerate(items):
        enr = it["enrollment"]
        ck = f"{key_prefix}::{enr}"
        if ck in cache:
            polluted[it["uid"]] = cache[ck]
            continue
        if detect_fn_name == "D1":
            res, _ = detect_pollution_diar(detect_fn, enr)  # detect_fn=diar
        else:
            res, _ = detect_fn(enr)  # detect_fn=get_emb wrapper
        cache[ck] = res
        polluted[it["uid"]] = res
        if (i+1) % 50 == 0:
            print(f"  [{detect_fn_name}] {i+1}/{len(items)} ({(i+1)/(time.time()-t0):.1f}/s)")
            save_cache(cache)  # 周期性落盘防中断丢失
    save_cache(cache)
    return polluted


# ============ Q1: 检测器 F1 ============
def run_q1(diar, get_emb, cache):
    print("\n" + "="*60 + "\nQ1: 检测手段可行性 (26条 calibration)\n" + "="*60)
    pos_data = load_pos()
    truth = load_calib_truth(pos_data)  # [(enr, polluted, uid, rel)]
    if not truth:
        print("[Q1] 无校准样本, No-Go"); return None

    def eval_detector(name, polluted_pred):
        tp=fp=fn_=0
        for (enr, gt, uid, rel), pred in zip(truth, polluted_pred):
            if pred is None: continue
            if pred and gt: tp+=1
            elif pred and not gt: fp+=1
            elif (not pred) and gt: fn_+=1
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn_) if tp+fn_ else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        print(f"  {name}: P={prec:.2f} R={rec:.2f} F1={f1:.2f} (tp={tp} fp={fp} fn={fn_})")
        return {"P":round(prec,3),"R":round(rec,3),"F1":round(f1,3),"tp":tp,"fp":fp,"fn":fn_}

    # D1
    d1_items = [{"uid":uid,"enrollment":enr} for enr,_,uid,_ in truth]
    d1_pred = []
    for enr,_,uid,_ in truth:
        ck = f"D1::{enr}"
        if ck in cache: d1_pred.append(cache[ck])
        else:
            r,_ = detect_pollution_diar(diar, enr); cache[ck]=r; d1_pred.append(r)
    save_cache(cache)
    d1 = eval_detector("D1_diar", d1_pred)

    # D2
    d2_pred = []
    for enr,_,uid,_ in truth:
        ck = f"D2::{enr}"
        if ck in cache: d2_pred.append(cache[ck])
        else:
            r,_ = detect_pollution_cluster(get_emb, enr); cache[ck]=r; d2_pred.append(r)
    save_cache(cache)
    d2 = eval_detector("D2_cluster", d2_pred)

    go = (d1["F1"] > 0.7) or (d2["F1"] > 0.7)
    print(f"\n[Q1 判定] Go={go} (标准: 任一 F1>0.7)")
    return {"D1":d1, "D2":d2, "go":go}


# ============ Q2: 污染与漏拒相关性 ============
def run_q2(diar, cache, thr=0.27):
    print("\n" + "="*60 + f"\nQ2: 污染与漏拒相关性 (474 neg, thr={thr})\n" + "="*60)
    from scipy.stats import fisher_exact
    neg = load_neg()
    # 跑 474 neg enrollment D1 检测(带缓存)
    polluted = {}
    t0=time.time()
    for i, it in enumerate(neg):
        ck = f"D1::{it['enrollment']}"
        if ck in cache: polluted[it["uid"]] = cache[ck]; continue
        r,_ = detect_pollution_diar(diar, it["enrollment"]); cache[ck]=r; polluted[it["uid"]]=r
        if (i+1)%50==0:
            print(f"  [D1 neg] {i+1}/{len(neg)} ({(i+1)/(time.time()-t0):.1f}/s)"); save_cache(cache)
    save_cache(cache)

    leak = [it for it in neg if it["max_sim"] >= thr]   # 组A: 漏拒
    nonleak = [it for it in neg if it["max_sim"] < thr]  # 组: 已拒
    def cnt(items):
        p=c=0
        for it in items:
            v = polluted.get(it["uid"])
            if v is True: p+=1
            elif v is False: c+=1
        return p, c
    pA, cA = cnt(leak)        # 漏拒组: 污染/干净
    pN, cN = cnt(nonleak)     # 已拒组: 污染/干净
    rA = pA/(pA+cA) if pA+cA else 0
    rN = pN/(pN+cN) if pN+cN else 0
    rAll = (pA+pN)/(pA+cA+pN+cN)
    mult = rA/rN if rN>0 else float('inf')
    # Fisher: 漏拒组 vs 已拒组 的污染富集
    table = [[pA, cA], [pN, cN]]
    _, p = fisher_exact(table)

    print(f"  组A(漏拒 n={len(leak)}): 污染率 {rA:.1%} ({pA}污染/{pA+cA}有效)")
    print(f"  已拒组(n={len(nonleak)}):  污染率 {rN:.1%} ({pN}/{pN+cN})")
    print(f"  全474基线污染率: {rAll:.1%}")
    print(f"  倍数(漏拒/已拒)={mult:.2f}x   Fisher p={p:.4f}")
    go = (mult >= 1.5) and (p < 0.05)
    strong = mult >= 2.0 and p < 0.05
    nogo = mult <= 1.2
    verdict = "强相关Go" if strong else ("Go" if go else ("弱相关No-Go" if nogo else "中等,看Q3"))
    print(f"\n[Q2 判定] {verdict} (标准: ≥1.5×且p<0.05=Go, ≥2×=强, ≤1.2×=No-Go)")
    return {"rA":round(rA,3),"rN":round(rN,3),"rAll":round(rAll,3),"mult":round(mult,2),
            "p":round(p,4),"n_leak":len(leak),"go":go,"strong":strong,"verdict":verdict,
            "polluted_neg":polluted}


# ============ Q3: 自适应 thr 代价 (hold-out) ============
def submit_norm(text):
    return digit_postproc(to_simplified(text or ""))

def pool_adaptive(hyps, refs, sims, polluted_flags, thr_high, thr_low=0.27):
    """自适应 thr 含拒 CER(官方累计池)。每条 thr = thr_high if 污染 else thr_low。"""
    m = CERMetric()
    for h, r, s, pol in zip(hyps, refs, sims, polluted_flags):
        thr = thr_high if pol else thr_low
        m.update(["" if s < thr else h], [r])
    res = m.compute()
    per = res["per_sample"]; n = len(per)
    correct = sum(1 for x in per if x["cer"] < 0.5)/n if n else 0.0
    return res["cer"], correct

def neg_rr_adaptive(neg_sims, neg_pol, thr_high, thr_low=0.27):
    rej = sum(1 for s, pol in zip(neg_sims, neg_pol)
              if s < (thr_high if pol else thr_low))
    return rej/len(neg_sims) if neg_sims else 0.0

def run_q3(diar, cache, q2_result, seed=42):
    print("\n" + "="*60 + "\nQ3: 自适应 thr 代价 (1364 pos hold-out)\n" + "="*60)
    pos = load_pos()
    neg = load_neg()
    print(f"[Q3] pos={len(pos)} neg={len(neg)}")

    # pos enrollment 污染检测(D1, 带缓存)
    pos_pol = {}
    t0=time.time()
    for i, it in enumerate(pos):
        ck=f"D1::{it['enrollment']}"
        if ck in cache: pos_pol[it["uid"]]=cache[ck]; continue
        r,_=detect_pollution_diar(diar, it["enrollment"]); cache[ck]=r; pos_pol[it["uid"]]=r
        if (i+1)%100==0:
            print(f"  [D1 pos] {i+1}/{len(pos)} ({(i+1)/(time.time()-t0):.1f}/s)"); save_cache(cache)
    save_cache(cache)

    # neg 污染(复用Q2缓存)
    neg_pol = q2_result["polluted_neg"] if q2_result else {}
    if not neg_pol:
        neg_pol = {}
        for it in neg:
            ck=f"D1::{it['enrollment']}"
            neg_pol[it["uid"]] = cache.get(ck)

    # hold-out by uid (8:2, seed)
    rng = np.random.default_rng(seed)
    uids = sorted(p["uid"] for p in pos)
    rng.shuffle(uids)
    val_uids = set(uids[int(len(uids)*0.8):])

    # 基线(thr_low=0.27 全量)
    neg_sims = [it["max_sim"] for it in neg]
    neg_pol_l = [bool(neg_pol.get(it["uid"], False)) for it in neg]
    base_rr = neg_rr_adaptive(neg_sims, neg_pol_l, 0.27, 0.27)  # 等价全0.27
    base_pos_cer_val = _pos_cer(pos, pos_pol, val_uids, 0.27, 0.27)
    print(f"[基线 thr=0.27 全量] neg RR={base_rr:.4f}  pos val CER={base_pos_cer_val[0]:.4f} correct={base_pos_cer_val[1]:.1%}")

    print(f"\n  {'thr_high':<9}{'ΔRR':<9}{'pos CER(val)':<14}{'Δpos CER':<10}{'correct':<9}{'verdict'}")
    rows = []
    for thr_high in [0.30, 0.35, 0.40]:
        rr = neg_rr_adaptive(neg_sims, neg_pol_l, thr_high, 0.27)
        cer, cor = _pos_cer(pos, pos_pol, val_uids, thr_high, 0.27)
        drr = rr - base_rr
        dcer = cer - base_pos_cer_val[0]
        verdict = "Go" if (drr > 0 and dcer <= 0.01) else ("ΔCER超" if dcer>0.01 else "RR无增益")
        print(f"  {thr_high:<9.2f}{drr:<+9.4f}{cer:<14.4f}{dcer:<+10.4f}{cor:<9.1%}{verdict}")
        rows.append({"thr_high":thr_high,"RR":round(rr,4),"ΔRR":round(drr,4),
                     "pos_cer_val":round(cer,4),"Δpos_cer":round(dcer,4),
                     "correct":round(cor,3),"verdict":verdict})
    go_rows = [r for r in rows if r["verdict"]=="Go"]
    print(f"\n[Q3 判定] Go thr_high = {[r['thr_high'] for r in go_rows] or '无'}")
    return {"base_rr":round(base_rr,4),"base_pos_cer_val":round(base_pos_cer_val[0],4),
            "rows":rows,"go_thr_high":[r['thr_high'] for r in go_rows]}

def _pos_cer(pos, pos_pol, val_uids, thr_high, thr_low):
    hyps=[];refs=[];sims=[];pol=[]
    for p in pos:
        if p["uid"] not in val_uids: continue
        v = pos_pol.get(p["uid"])
        if v is None: continue  # diar失败条跳过
        hyps.append(submit_norm(p["qwen"])); refs.append(p["ref"])
        sims.append(p["max_sim"]); pol.append(bool(v))
    if not hyps: return (0,0)
    return pool_adaptive(hyps, refs, sims, pol, thr_high, thr_low)


# ============ main ============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["load","q1","q2","q3","all"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--recompute", action="store_true", help="清缓存重跑")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seed(args.seed)

    cache = {} if args.recompute else load_cache()

    if args.stage == "load":
        neg = load_neg(); pos = load_pos()
        print(f"[load] neg={len(neg)} pos={len(pos)} cache={len(cache)}")
        calib = load_calib_truth(pos)
        print(f"[load] calib truth={len(calib)}")
        print("[load] OK, 数据加载就绪")
        return

    diar, get_emb, device = load_models()
    q1=q2=q3=None
    if args.stage in ("q1","all"):
        q1 = run_q1(diar, get_emb, cache)
    if args.stage in ("q2","all"):
        q2 = run_q2(diar, cache)
    if args.stage in ("q3","all"):
        q3 = run_q3(diar, cache, q2, args.seed)

    summary = {"q1":q1, "q2":({k:v for k,v in q2.items() if k!="polluted_neg"} if q2 else None),
               "q3":q3, "seed":args.seed}
    out_path = os.path.join(_HERE, "poc_enrollment_pollution_result.json")
    json.dump(summary, open(out_path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[已写] {out_path}")

if __name__ == "__main__":
    main()
