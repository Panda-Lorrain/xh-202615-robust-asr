"""Phase 1 实验：vanilla Whisper + 声纹切 target vs DiCoW 条件化（验证 H3）

H3（T22 babble 归因）：DiCoW 的 STNO/FDDT 条件化在极重 babble 下【反作用】——
vanilla Whisper-large-v3-turbo 在 babble 下反而正常。

本脚本复用 enroll_infer 的 diar + 声纹匹配选 target，但转写用【vanilla Whisper】
（去掉 DiCoW 条件化）：切 target timeline 段拼接 → vanilla Whisper 转写。
对比 out_pos_full（DiCoW 条件化）同样本的 CER。

若 vanilla 显著优于 DiCoW → H3 证伪 DiCoW 条件化，改用 target extraction + vanilla 路线
（零训练、答辩大招）。

用法：
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/exp_vanilla_vs_dicow.py \\
    --pairs code/pos_pairs_datasetA.json --limit 100 \\
    --dicow-result code/out_pos_full/result.json
"""
import os, sys, json, time, argparse
import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
import pyarrow  # 预热：避免后续 import pyannote 时扫描 sys.path 的 DiariZen 目录触发 WinError 6714

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

# 复用 enroll_infer 的工具函数（diar_mask / collect_clean_audio）
from enroll_infer import get_diarization_mask, collect_clean_audio

# diar sys.path（DiariZen + pyannote）
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

VANILLA_MODEL = "E:/hf_cache/whisper-large-v3-turbo"
DIAR_MODEL = "E:/hf_cache/diarizen-wavlm-large-s80-md"


def cer_of(text, ref):
    from eval_datasetA import _norm_zh
    from eval_metrics import cer
    t = _norm_zh(text or "")
    r = _norm_zh(ref or "")
    if not t:
        return 1.0
    return cer(t, r)


def main():
    ap = argparse.ArgumentParser(description="vanilla Whisper vs DiCoW 条件化对比（H3 验证）")
    ap.add_argument("--pairs", required=True, help="pos_pairs_datasetA.json")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out-json", default=os.path.join(_HERE, "exp_vanilla_result.json"))
    ap.add_argument("--dicow-result", default=os.path.join(_HERE, "out_pos_full/result.json"),
                    help="DiCoW 条件化结果（对比基准）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vanilla-model", default=VANILLA_MODEL)
    ap.add_argument("--diar-model", default=DIAR_MODEL)
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"[load] vanilla Whisper {args.vanilla_model}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.vanilla_model, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.vanilla_model)
    fe = AutoFeatureExtractor.from_pretrained(args.vanilla_model)

    print(f"[load] DiariZen {args.diar_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diar_model).to(device)

    def get_emb(wav_np):
        w = torch.from_numpy(np.ascontiguousarray(wav_np.astype(np.float32))).to(device)
        if w.dim() == 1:
            w = w[None, None]
        elif w.dim() == 2:
            w = w[None]
        with torch.no_grad():
            emb = diar._embedding(w)
        emb = torch.as_tensor(emb, device=device, dtype=torch.float32)
        return torch.nn.functional.normalize(emb, dim=-1).squeeze(0)

    pair_rows = json.load(open(args.pairs, encoding="utf-8"))[:args.limit]

    # DiCoW 结果（对比基准）按 recognition basename 索引
    dicow_res = {}
    if os.path.exists(args.dicow_result):
        for r in json.load(open(args.dicow_result, encoding="utf-8")).get("results", []):
            dicow_res[os.path.splitext(os.path.basename(r.get("recognition", "")))[0]] = r

    results = []
    for row in pair_rows:
        enr, rec = row["enrollment"], row["recognition"]
        uid = os.path.splitext(os.path.basename(rec))[0]
        ref = row.get("ref", "") or ""
        t0 = time.time()

        # enrollment 声纹
        w, _ = librosa.load(enr, sr=16000)
        enroll_emb = get_emb(w)

        # diar
        audio, sr = librosa.load(rec, sr=16000)
        dur = len(audio) / sr
        try:
            diar_out = diar(rec)
        except Exception as e:
            print(f"  [diar-fail] {uid}: {type(e).__name__} {str(e)[:80]}")
            results.append({"uid": uid, "ref": ref, "error": str(e)[:120],
                            "vanilla_text": "", "dicow_text": dicow_res.get(uid, {}).get("text", "")})
            continue
        speakers = list(diar_out.labels())
        per_spk = [diar_out.label_timeline(s) for s in speakers]

        # diar_mask + 声纹匹配选 target（复用 enroll_infer 逻辑）
        audio_len = len(audio) // 320  # 50Hz 帧率（16000 / 320 = 50）
        diar_mask = get_diarization_mask(per_spk, audio_len)
        spk_embs = []
        for i in range(len(speakers)):
            seg = collect_clean_audio(audio, diar_mask, i, sr)
            if seg is None or len(seg) < sr * 0.3:
                segs = [audio[int(s * sr):int(e * sr)] for s, e in per_spk[i]]
                seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
            min_len = sr
            if len(seg) < min_len:
                seg = np.tile(seg, min_len // len(seg) + 1)[:min_len]
            spk_embs.append(get_emb(seg))
        sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
        target_idx = int(torch.argmax(sims))
        max_sim = float(sims[target_idx])

        # 切 target timeline（含重叠区）拼接喂 vanilla —— 不用 DiCoW 条件化
        target_segs = sorted([(float(s), float(e)) for s, e in per_spk[target_idx]])
        if target_segs:
            target_audio = np.concatenate([audio[int(s * sr):int(e * sr)] for s, e in target_segs])
        else:
            target_audio = audio
        if len(target_audio) < sr * 0.3:
            target_audio = audio  # target 太短退化整条

        # vanilla Whisper 转写（无条件化）
        ifp = fe(target_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        with torch.no_grad():
            out = model.generate(input_features=ifp, language=args.language,
                                 task="transcribe", max_new_tokens=200)
        text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()

        dt = time.time() - t0
        dicow_text = (dicow_res.get(uid, {}).get("text") or "").strip()
        v_cer = cer_of(text, ref)
        d_cer = cer_of(dicow_text, ref)
        flag = "✓vanilla更优" if v_cer < d_cer - 0.01 else ("✗dicow更优" if v_cer > d_cer + 0.01 else "≈持平")
        print(f"[{uid}] sim={max_sim:.3f} ({dur:.1f}s→{len(target_audio)/sr:.1f}s, {dt:.1f}s) {flag}")
        print(f"  vanilla({v_cer:.2f}): {text[:55]}")
        print(f"  dicow  ({d_cer:.2f}): {dicow_text[:55]}")
        print(f"  ref        : {ref[:55]}")
        results.append({"uid": uid, "ref": ref, "max_sim": max_sim,
                        "vanilla_text": text, "dicow_text": dicow_text,
                        "vanilla_cer": v_cer, "dicow_cer": d_cer, "rtf": dt / dur})

    # 汇总对比
    valid = [r for r in results if "vanilla_cer" in r]
    print(f"\n===== CER 对比（{len(valid)} 条）=====")
    if valid:
        for label, key in [("vanilla", "vanilla_cer"), ("dicow", "dicow_cer")]:
            cs = [r[key] for r in valid]
            overall = sum(cs) / len(cs)
            correct = sum(1 for c in cs if c < 0.5) / len(cs)
            near = sum(1 for c in cs if c < 0.1) / len(cs)
            print(f"  {label:8}: overall_cer={overall:.4f}  correct(CER<0.5)={correct:.2%}  near_perfect(<0.1)={near:.2%}")
        better = sum(1 for r in valid if r["vanilla_cer"] < r["dicow_cer"] - 0.01)
        worse = sum(1 for r in valid if r["vanilla_cer"] > r["dicow_cer"] + 0.01)
        tie = len(valid) - better - worse
        print(f"  逐条: vanilla 更优 {better} / 更差 {worse} / 持平 {tie}（共 {len(valid)}）")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] → {args.out_json}")


if __name__ == "__main__":
    main()
