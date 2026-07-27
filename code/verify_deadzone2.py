"""verify_deadzone2.py — 第二批 M1 死区验收包(10 条, 5 深 + 5 浅)。

【背景】第一批 5 条(cmd_0/18/2000/2041/2098)中 3 条 M1(cmd_18/2000/2098)用户人耳全可辨 →
真地板比例可能远低于估算 25-30%。本批再抽 10 条 M1 验收, 把真地板比例钉死。

【挑样】M1 判据(同 exp_deadzone_diag.py:_classify_failure):
  - sim<0.4(死区)
  - qwen_cer>=0.8(高 CER)
  - char_overlap(qwen, ref)<0.3(字面完全不沾边)
  - 排除第一批 cmd_0/18/2000/2041/2098
挑 10 条: 5 深(sim<0.1) + 5 浅(sim∈[0.20,0.22])。

【复刻】enroll_infer.py:24-29 (_safe_getmodule) + 170-172 (load diar) + 178-187 (get_emb)
       exp_deadzone_diag.py:phase_verify (8 工位)
单条样本, diar 加载一次循环 10 条。

【输出】
  code/runs/_verify_deadzone2_<uid>/   8 工位(避免覆盖第一批 _verify_deadzone_<uid>)
  docs/verify_deadzone2.md             索引(听音回填表)

种子 42  不改主线代码  不 git commit
"""
import inspect as _inspect
_orig_getmodule = _inspect.getmodule
def _safe_getmodule(*a, **k):
    try: return _orig_getmodule(*a, **k)
    except (ImportError, AttributeError): return None
_inspect.getmodule = _safe_getmodule

import os, sys, json, time, argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import pyarrow  # 预热: 避免 pyannote 扫 sys.path 触发 WinError 6714

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

DICOW_INF = os.path.join(_HERE, "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p): sys.path.insert(0, _p)

from enroll_infer import get_diarization_mask, collect_clean_audio
from text_utils import cut_target_timeline
from repro import set_global_seed, resolve_model


# ====================== 10 条 M1 样本(5 深 + 5 浅) ======================
# 挑选原则: 覆盖 sim 深浅 + ref 多样(空调/温度/播放/食物) + qwen 乱词型(M1b)与循环幻觉(M1a)
SAMPLE_UIDS = [
    # ---- DEEP sim<0.1 (5 条) ----
    "cmd_2029",  # sim=0.013 ref='风往中间吹'      qwen='租户与业主。'
    "cmd_2010",  # sim=0.026 ref='把空调关闭'      qwen='是谁轻易骗？'
    "cmd_2096",  # sim=0.035 ref='风速开到最大'    qwen='加了点儿血嘛。'
    "cmd_2832",  # sim=0.049 ref='把温度调到十八度' qwen='这里是萤火虫。'
    "cmd_2949",  # sim=0.071 ref='播放减压放松音乐合集' qwen='一百四十九。'
    # ---- SHALLOW sim∈[0.20,0.22] (5 条) ----
    "cmd_2571",  # sim=0.200 ref='打开显示屏'      qwen='联合工会。'
    "cmd_2754",  # sim=0.200 ref='关掉智清洁'      qwen='春节。'
    "cmd_2050",  # sim=0.202 ref='开开空调'        qwen='你那边呢？'
    "cmd_2001",  # sim=0.208 ref='给我讲乱世枭雄杜月笙' qwen='住的小区车位早已供不应求。'
    "cmd_2586",  # sim=0.200 ref='富含硒的食物有哪些' qwen='色彩饱和度为。'
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qwen-full", default=os.path.join(_HERE, "runs", "poc_qwen_asr_full_result.json"))
    ap.add_argument("--pairs", default=os.path.join(_HERE, "pos_pairs_datasetA.json"))
    args = ap.parse_args()

    print(f"\n{'='*70}\n[verify_deadzone2] 第二批 M1 死区验收包 (10 条)")
    set_global_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 数据
    qfull = json.load(open(args.qwen_full, encoding="utf-8"))
    rows_map = {r["uid"]: r for r in qfull["rows"]}
    pairs = json.load(open(args.pairs, encoding="utf-8"))
    uid2pair = {os.path.splitext(os.path.basename(p["recognition"]))[0]: p for p in pairs}

    # 加载 diar (一次)
    print(f"[load] DiariZen {resolve_model('DIAR')}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(resolve_model("DIAR")).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1: w = w[None, None]
        elif w.dim() == 2: w = w[None]
        with torch.inference_mode():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    verify_summaries = []
    t0 = time.time()
    for n, uid in enumerate(SAMPLE_UIDS):
        if uid not in uid2pair:
            print(f"[skip] {uid} 不在 pairs"); continue
        if uid not in rows_map:
            print(f"[skip] {uid} 不在 qwen_full"); continue
        pair = uid2pair[uid]
        enr, rec = pair["enrollment"], pair["recognition"]
        out_dir = os.path.join(_HERE, "runs", f"_verify_deadzone2_{uid}")
        if os.path.isdir(out_dir):
            import shutil; shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        r = rows_map[uid]

        # 工位 1+2: 原始 enrollment + recognition
        w_enr, _ = librosa.load(enr, sr=16000)
        w_rec, sr = librosa.load(rec, sr=16000)
        sf.write(os.path.join(out_dir, "enrollment.wav"), w_enr.astype(np.float32), 16000)
        sf.write(os.path.join(out_dir, "recognition.wav"), w_rec.astype(np.float32), 16000)

        # diar (recognition)
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [{uid}] diar FAIL {e}")
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]
        audio_len = len(w_rec) // 320  # 50Hz 帧数近似
        try:
            diar_mask = get_diarization_mask(per_spk, audio_len)
        except Exception as e:
            print(f"  [{uid}] diar_mask fail: {e}")
            diar_mask = torch.zeros(len(speakers), 1)

        # 各 speaker 声纹 + 工位 3/4/5
        enr_emb = get_emb(w_enr)
        spk_embs, spk_sims = [], []
        spk_emb_info = []
        # enrollment diar (污染检测)
        enr_diar = None
        try: enr_diar = diar(enr)
        except Exception: pass
        enr_n_spk = len(list(enr_diar.labels())) if enr_diar is not None else 1
        if enr_diar is not None and enr_n_spk > 1:
            enr_per_spk = [enr_diar.label_timeline(s) for s in list(enr_diar.labels())]
            enr_diar_mask = get_diarization_mask(enr_per_spk, len(w_enr) // 320)
            for j in range(enr_n_spk):
                seg = collect_clean_audio(w_enr, enr_diar_mask, j)
                if seg is not None and len(seg) > 0:
                    sf.write(os.path.join(out_dir, f"enr_spk{j}.wav"), seg.astype(np.float32), 16000)
        for i in range(len(speakers)):
            # 工位 4: rec_spk_full
            full_segs = [w_rec[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
            full_audio = np.concatenate(full_segs) if full_segs else np.zeros(sr, dtype=np.float32)
            sf.write(os.path.join(out_dir, f"rec_spk{i}_full.wav"), full_audio.astype(np.float32), 16000)
            # 工位 5: rec_spk_excl_raw
            excl = collect_clean_audio(w_rec, diar_mask, i)
            excl_sec = round(len(excl) / sr, 2) if excl is not None else 0
            if excl is not None and len(excl) > 0:
                sf.write(os.path.join(out_dir, f"rec_spk{i}_excl_raw.wav"), excl.astype(np.float32), 16000)
            # 抽 emb(优先独占, fallback full)
            emb_input = excl if (excl is not None and len(excl) >= sr * 0.3) else full_audio
            min_len = sr
            if len(emb_input) < min_len:
                emb_input = np.tile(emb_input, min_len // len(emb_input) + 1)[:min_len]
            emb = get_emb(emb_input)
            spk_embs.append(emb)
            sim = float(torch.dot(enr_emb, emb))
            spk_sims.append(sim)
            spk_emb_info.append({
                "speaker": speakers[i], "sim": round(sim, 4),
                "excl_sec": excl_sec, "fallback_full": excl is None or len(excl) < sr * 0.3,
                "tiled_to_1s": len(emb_input) < min_len * 2,
            })

        # argmax 选 target
        target_idx = int(np.argmax(spk_sims)) if spk_sims else 0
        max_sim = spk_sims[target_idx] if spk_sims else 0
        # 工位 6: target_slice
        target_audio = cut_target_timeline(w_rec, per_spk[target_idx], sr=sr)
        sf.write(os.path.join(out_dir, "target_slice.wav"), target_audio.astype(np.float32), 16000)

        # 工位 7: 假如选别的 speaker
        if len(speakers) >= 2:
            for other_idx in range(len(speakers)):
                if other_idx == target_idx: continue
                other_audio = cut_target_timeline(w_rec, per_spk[other_idx], sr=sr)
                sf.write(os.path.join(out_dir, f"假如选spk{other_idx}_当target.wav"),
                         other_audio.astype(np.float32), 16000)

        # 重叠率
        if len(speakers) >= 2 and diar_mask.shape[1] > 0:
            overlap_rate = float(((diar_mask.sum(dim=0) >= 2)).float().mean())
        else:
            overlap_rate = 0.0

        # 工位 8: postprocess_steps.json + summary.json
        steps = {
            "uid": uid, "ref": r.get("ref"), "kws_txt": pair.get("kws_txt"),
            "enrollment": enr, "recognition": rec,
            "audio_sec": round(len(w_rec) / sr, 2),
            "enr_sec": round(len(w_enr) / sr, 2),
            "enr_n_spk": enr_n_spk,
            "n_spk_rec": len(speakers), "speakers": speakers,
            "spk_emb_info": spk_emb_info,
            "target_idx": target_idx, "target_speaker": speakers[target_idx] if speakers else None,
            "max_sim": round(max_sim, 4), "all_sims": [round(s, 4) for s in spk_sims],
            "overlap_rate": round(overlap_rate, 4),
            "qwen_text_argmax": r.get("qwen"),
            "qwen_cer_argmax": r.get("qwen_cer"),
            "vanilla_text": r.get("vanilla"),
            "vanilla_cer": r.get("vanilla_cer"),
            "poc_sim": r.get("sim"),
            "bucket": r.get("bucket"),
            "note": "M1 疑似人耳不可辨(字面 overlap<0.3 + CER>=0.8) — 待用户听 recognition 定性真地板/可修",
        }
        with open(os.path.join(out_dir, "postprocess_steps.json"), "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(steps, f, ensure_ascii=False, indent=2)
        verify_summaries.append(steps)
        elapsed = time.time() - t0
        print(f"  [{n+1}/{len(SAMPLE_UIDS)}] {uid} sim={max_sim:.3f} n_spk={len(speakers)} "
              f"target=spk{speakers[target_idx] if speakers else '-'} "
              f"overlap={overlap_rate:.2f} qwen_CER={r.get('qwen_cer')} ({elapsed:.0f}s)")

    # ---- docs/verify_deadzone2.md 索引 ----
    doc_path = os.path.join(_ROOT, "docs", "verify_deadzone2.md")
    write_index(verify_summaries, doc_path)
    print(f"\n[done] {len(verify_summaries)}/{len(SAMPLE_UIDS)} 条验收包, 总耗时 {(time.time()-t0)/60:.1f}min")
    print(f"[doc] 索引 {doc_path}")
    return verify_summaries


def write_index(summaries, doc_path):
    """复刻 exp_deadzone_diag.py phase_verify 的索引, 改成第二批语境."""
    lines = [
        "# 第二批 M1 死区验收包索引(听音核实 — 用户必答)\n",
        f"> 生成: {time.strftime('%Y-%m-%d %H:%M')} by verify_deadzone2.py\n",
        "> 每样本目录 `code/runs/_verify_deadzone2_<uid>/`, 8 工位文件清单见下表.\n",
        "\n## 背景\n",
        "死区(sim<0.4)占全量 78.8% 贡献 87% CER, 是 CER 大头。死区诊断把死区失败模式分 5 类,",
        "其中 **M1(疑似人耳不可辨)共 324 条(30.5%)** — 机器转写与 ref 字面完全不沾边(乱词/循环幻觉)。",
        "**第一批 3 条 M1(cmd_18/2000/2098)用户人耳全听得清 target**, 真地板比例可能远低于估算 25-30%。",
        "本批再抽 10 条 M1, 覆盖死区深(sim<0.1)/浅(sim∈[0.20,0.22]), 把真地板比例钉死。\n",
        "\n## 🔑 核心问题(每条样本必答)\n",
        "**听每条 `recognition.wav`, 对照下面的 ref 文本, 回答: target 说的话, 人耳能否听清?**\n",
        "- ✅ **人耳能听清 ref** → 即便机器 CER 很高(乱词/完全不沾边), 也是机器能力不够 = **可修(训练/微调/增强/分离能攻)**",
        "- ❌ **人耳听不清 ref**(纯噪音/完全淹没) → **真地板**(任何模型都救不了)",
        "\n这是判定死区天花板的**唯一可靠依据**(2637/2475/18/2000/2098 都是用户耳朵先于数据定位)。\n",
        "\n## 8 工位文件清单\n",
        "| 工位 | 文件 | 听什么 |",
        "|---|---|---|",
        "| 1 | `enrollment.wav` | 目标说话人参考音(原 wav, 锁定 target 音色) |",
        "| **2** | **`recognition.wav`** | **识别音频原 wav(双人重叠+噪声) — 核心验收, 听 target 说的话能否听清** |",
        "| 3 | `enr_spk{i}.wav` | enrollment 是否被 diar 拆多 speaker(若有则 enrollment 污染) |",
        "| 4 | `rec_spk{i}_full.wav` | diar 切出的各 speaker 全 timeline 段(含重叠区) |",
        "| 5 | `rec_spk{i}_excl_raw.wav` | diar 切出的各 speaker 独占帧(避重叠, 抽声纹用) |",
        "| 6 | `target_slice.wav` | argmax 选 target 切出的 timeline 切片(喂 ASR 的实际音频) |",
        "| 7 | `假如选spk{i}_当target.wav` | 假如选其他 speaker 当 target(对照, 听是否另一人才是对的) |",
        "| 8 | `postprocess_steps.json` + `summary.json` | sims / argmax 选谁 / 重叠率 / qwen 转写 / ref / CER |",
        "\n## 样本列表(10 条 — 5 死区深 + 5 死区浅, 机器转写都是乱词型)\n",
        "| uid | sim | bucket | qwen CER | n_spk | overlap | argmax target | **ref(请听 recognition 对照)** | 机器转写(qwen) | 机器初判 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        cer = s.get("qwen_cer_argmax")
        sim = s.get("max_sim", 0)
        ref = s.get("ref", "")
        qwen_text = s.get("qwen_text_argmax", "")
        # 机器初判(只作提示, 用户听为准)
        if s.get("n_spk_rec", 0) >= 2 and s.get("overlap_rate", 0) > 0.3:
            reason = f"🔥 重叠率 {s['overlap_rate']*100:.0f}% + 机器高 CER — 待听重叠区 target 能否辨"
        elif cer is not None and cer >= 0.8:
            reason = "🔥 机器乱词完全不沾边 — 待听 target 说 ref 人耳能否辨(真地板 vs 可修)"
        else:
            reason = "机器转写接近但有错(可纠正类)"
        lines.append(
            f"| {s['uid']} | {sim:.3f} | {s.get('bucket','-')} | {cer} | "
            f"{s.get('n_spk_rec','-')} | {s.get('overlap_rate',0)*100:.0f}% | "
            f"spk{s.get('target_speaker','-')} | **{ref}** | {qwen_text} | {reason} |"
        )
    lines.append("\n## 用户听完后请回填\n")
    lines.append("| uid | bucket | ref | 人耳能否听清 target 说的 ref? (能/部分/不能) | 真地板 or 可修 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(f"| {s['uid']} | {s.get('bucket','-')} | {s.get('ref','')} | _待填_ | _待填_ | |")

    lines.append("\n## 真地板比例估算(用户回填后更新)\n")
    lines.append("> 第一批 3 条 M1 全可辨 → 真地板 0%; 本批 10 条若也多数可辨, 真地板比例估算:")
    lines.append("- 若本批 0-2 条听不清: 真地板估算 <10% (M1 几乎全是机器能力不够, 死区可修子集大幅上调)")
    lines.append("- 若本批 3-5 条听不清: 真地板估算 ~15-25% (与原 SepFormer oracle 估算 25-30% 接近)")
    lines.append("- 若本批 6-10 条听不清: 真地板估算 >40% (M1 主导真地板, 死区天花板硬)")
    lines.append("\n> 综合第一批 + 第二批共 13 条 M1 听音结果, 给出死区 324 条 M1 的真地板/可修拆分。")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
