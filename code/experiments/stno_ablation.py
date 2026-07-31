"""单步消融实验(stno_ablation.py): 区分 H1 vs H2 —— P2 babble 工程兜底的决定性实验。

【背景】babble(人声噪声) ov0(本应单人)时, DiariZen 100% 误检"幽灵speaker2" →
STNO 的 target 行被错标: target 帧本该进 target 通道(stno[:,1]), 却因 anyone_else=0
全跑进 overlap 通道(stno[:,3])。FDDT.py:41-63 是门控线性变换, 每帧 hidden 按 STNO 四行
加权混合各通道线性层 → target 帧用 overlap 通道(为"两人重叠"训练)处理 → 条件化错位 → 退化乱转。
  H1: STNO 错标主导, 修 STNO(target 帧归位)可救转写
  H2: babble 音频本身让 Whisper 硬噪声幻觉, 修 STNO 无用
T19 倾向 H2, T20 倾向 H1, 至今未用单步消融区分。本脚本回答(P2 必须先确证)。

【方法】拿一条 babble ov0 snr+5(已知 sim 锁对 target, stno=0, 转写崩), 同一 mel 特征 + 同一
target 锁定下, 只改 STNO mask, 构造三种喂 DiCoW generate 对比:
  A 现状      get_stno_mask(diar_mask, target_idx)              → 复现崩(target帧→overlap通道)
  B 丢幽灵    检测与target帧级overlap率>thr/时长<min_dur的speaker, 丢之重算  → H1则target帧归位
  C 单spk     只留target行, get_stno_mask                        → H1则target帧全归位(上界)
判定(对 ref 算字符 CER):
  A 崩 且 (B 或 C 的 CER<0.5)            → H1 成立, 杠杆A有效
  B、C 的 CER 都 >0.6                     → H2 成立, 需 SE-DiCoW/更强 babble SE
  其它                                    → 混合, 细看

【运行】(需先 source setenv 设 HF_HOME 等; 与 enroll_infer 同环境)
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/stno_ablation.py
"""
import os
import sys
import json
import time
import argparse
import difflib

import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# 兜底 HF 缓存(setenv 已设则不覆盖), 让脚本尽量独立可跑
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
from enroll_infer import (  # 复用: 路径常量 + STNO/diar 函数 + DiCoW-inference/DiariZen 的 sys.path
    DICOW_MODEL, DIAR_MODEL, get_diarization_mask, get_stno_mask, collect_clean_audio,
)


# ---- 三种 STNO 构造 ----
def stno_A_status_quo(diar_mask, target_idx):
    """现状: 原始 diar_mask 直接构 STNO(target 帧被幽灵 speaker 拖进 overlap 通道)。"""
    return get_stno_mask(diar_mask, target_idx)


def stno_B_drop_ghost(diar_mask, target_idx, overlap_thr=0.8, min_dur=0.5, frame_rate=50):
    """杠杆A: 丢幽灵 speaker 后重算 STNO。
    幽灵判据(仅用现有 diar_mask): 与 target 帧级 overlap 率 > overlap_thr, 或该 speaker 总时长 < min_dur 秒。
    返回 (stno_mask, diagnostics, dropped_idx)。"""
    target_frames = diar_mask[target_idx] > 0
    target_n = max(int(target_frames.sum().item()), 1)
    diags, keep_idx, dropped = [], [], []
    for j in range(diar_mask.shape[0]):
        if j == target_idx:
            keep_idx.append(j)
            continue
        jf = diar_mask[j] > 0
        ow = int((jf & target_frames).sum().item()) / target_n   # 与 target 的帧级 overlap 率
        dur = float(jf.sum().item()) / frame_rate                 # 该 speaker 总时长(秒)
        ghost = bool(ow > overlap_thr or dur < min_dur)
        diags.append({"idx": j, "overlap_with_target": round(ow, 3),
                      "duration_s": round(dur, 3), "ghost": ghost})
        (dropped if ghost else keep_idx).append(j)
    new_mask = diar_mask[keep_idx].clone()
    new_tidx = keep_idx.index(target_idx)
    return get_stno_mask(new_mask, new_tidx), diags, dropped


def stno_C_single_speaker(diar_mask, target_idx):
    """强制单 speaker 假设(上界): 只留 target 行 → target 帧全部正确归位, 无 overlap 错标。"""
    single = diar_mask[target_idx:target_idx + 1].clone()
    return get_stno_mask(single, 0)


# ---- 复用 enroll_infer 的 generate / embedding ----
def get_emb_factory(diar, device):
    """复制自 enroll_infer.py:116-125: 用 diar._embedding(wespeaker) 抽声纹, L2 归一化。"""
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
    return get_emb


def run_generate(dicow, tok, ifp, stno, device, dtype, language="zh", max_new_tokens=200):
    """复用 enroll_infer.py:200-207 的 generate 调用。同一 mel/锁定下只换 STNO。"""
    am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
    with torch.no_grad():
        out = dicow.generate(input_features=ifp, attention_mask=am,
                             stno_mask=stno[None].to(device, dtype),
                             language=language, task="transcribe", max_new_tokens=max_new_tokens)
    seqs = out["sequences"] if isinstance(out, dict) else out
    return tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()


def char_cer(ref, hyp):
    """字符级 CER = 编辑距离 / len(ref)。自包含(不依赖外部包)。"""
    r, h = list(ref), list(hyp)
    m, n = len(r), len(h)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ri = r[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ri == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / m


def main():
    ap = argparse.ArgumentParser(description="STNO 单步消融: 区分 H1(STNO 错标) vs H2(babble 音频)")
    ap.add_argument("--recognition",
                    default=os.path.join(_ROOT, "test_wav/dataset/final/t_01_n_07_ov000_snr+5_babble.wav"),
                    help="消融样本(babble ov0 snr+5, sim 锁对 stno=0 崩)")
    ap.add_argument("--enrollment",
                    default=os.path.join(_ROOT, "test_wav/dataset/raw/enrollment/target_long_01.wav"))
    ap.add_argument("--ref", default="请把客厅的空调温度调到二十六度", help="target 参考文本(算 CER)")
    ap.add_argument("--dicow-model", default=DICOW_MODEL)
    ap.add_argument("--diarization-model", default=DIAR_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--overlap-thr", type=float, default=0.8, help="幽灵判据: 与 target 帧级 overlap 率阈值")
    ap.add_argument("--min-dur", type=float, default=0.5, help="幽灵判据: speaker 最短时长(秒)")
    ap.add_argument("--out-json", default=os.path.join(_HERE, "stno_ablation_result.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print(f"[load] DiCoW {args.dicow_model} on {device}")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.dicow_model)
    fe = AutoFeatureExtractor.from_pretrained(args.dicow_model)

    print(f"[load] DiariZen {args.diarization_model}")
    from diarizen.pipelines.inference import DiariZenPipeline
    diar = DiariZenPipeline.from_pretrained(args.diarization_model).to(device)
    get_emb = get_emb_factory(diar, device)

    enroll_wav, _ = librosa.load(args.enrollment, sr=16000)
    enroll_emb = get_emb(enroll_wav)
    print(f"[enrollment] {args.enrollment} ({len(enroll_wav)/16000:.1f}s)")

    audio, sr = librosa.load(args.recognition, sr=16000)
    dur = len(audio) / sr
    print(f"[rec] {os.path.basename(args.recognition)} ({dur:.1f}s) ref={args.ref}")

    # diar → speakers + diar_mask
    diar_out = diar(args.recognition)
    speakers = list(diar_out.labels())
    per_spk = [diar_out.label_timeline(s) for s in speakers]
    ifp = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
    audio_len = ifp.shape[-1] // 2          # 50Hz 帧数
    diar_mask = get_diarization_mask(per_spk, audio_len)
    print(f"[diar] speakers={speakers} diar_mask={tuple(diar_mask.shape)}")

    # 选 target(声纹匹配, 复用 enroll_infer.py:170-184)
    spk_embs = []
    for i in range(len(speakers)):
        seg = collect_clean_audio(audio, diar_mask, i, sr)
        if seg is None or len(seg) < sr * 0.3:
            segs = [audio[int(s*sr):int(e*sr)] for s, e in per_spk[i]]
            seg = np.concatenate(segs) if segs else np.zeros(sr, dtype=np.float32)
        if len(seg) < sr:
            seg = np.tile(seg, sr // len(seg) + 1)[:sr]
        spk_embs.append(get_emb(seg))
    sims = torch.stack([torch.dot(enroll_emb, e) for e in spk_embs])
    target_idx = int(torch.argmax(sims))
    max_sim = float(sims[target_idx])
    print(f"[match] target={speakers[target_idx]} sim={max_sim:.3f} (sims={[round(float(s),3) for s in sims]})")

    # 三种 STNO
    stno_A = stno_A_status_quo(diar_mask, target_idx)
    stno_B, ghost_diags, dropped = stno_B_drop_ghost(diar_mask, target_idx, args.overlap_thr, args.min_dur)
    stno_C = stno_C_single_speaker(diar_mask, target_idx)
    rt = {"A_现状": float(stno_A[1].mean()), "B_丢幽灵": float(stno_B[1].mean()), "C_单spk": float(stno_C[1].mean())}
    print(f"[ghost-diag] {ghost_diags} dropped={dropped}")
    print(f"[stno target_row_ratio] A={rt['A_现状']:.3f}  B={rt['B_丢幽灵']:.3f}  C={rt['C_单spk']:.3f}")

    # 三种 generate(同一 mel + 同一锁定, 只换 STNO)
    texts, cers, times = {}, {}, {}
    for name, stno in [("A_现状", stno_A), ("B_丢幽灵", stno_B), ("C_单spk", stno_C)]:
        t0 = time.time()
        text = run_generate(dicow, tok, ifp, stno, device, dtype, args.language)
        dt = time.time() - t0
        texts[name] = text
        cers[name] = round(char_cer(args.ref, text), 3)
        times[name] = round(dt, 1)
        print(f"[generate {name}] {len(text)}字 CER={cers[name]} ({dt:.1f}s): {text[:80]}")

    # 判定 H1 vs H2
    a_crash = cers["A_现状"] > 0.6 or len(texts["A_现状"]) > 60
    b_save = cers["B_丢幽灵"] < 0.5
    c_save = cers["C_单spk"] < 0.5
    print("\n=== 判定 ===")
    print(f"A 复现崩(CER>0.6或>60字): {a_crash}  | B 恢复(CER<0.5): {b_save}  | C 恢复(CER<0.5): {c_save}")
    if a_crash and (b_save or c_save):
        verdict = "H1 成立: STNO 错标主导, 杠杆A(丢幽灵/单spk)有效, 转写恢复"
    elif not b_save and not c_save:
        verdict = "H2 成立: babble 音频本身致 Whisper 幻觉, 修 STNO 无用, 需 SE-DiCoW/更强 babble SE"
    else:
        verdict = "混合: 部分恢复, 需细看(可能 STNO 错标 + 音频双层瓶颈)"
    print(f"结论: {verdict}")

    result = {
        "recognition": args.recognition, "enrollment": args.enrollment, "ref": args.ref,
        "speakers": speakers, "target_idx": target_idx, "max_sim": max_sim,
        "sims": [round(float(s), 4) for s in sims],
        "ghost_diag": ghost_diags, "dropped": dropped,
        "stno_target_ratio": {k: round(v, 4) for k, v in rt.items()},
        "transcripts": texts, "cer_vs_ref": cers, "gen_time_s": times,
        "a_crash": a_crash, "b_save": b_save, "c_save": c_save, "verdict": verdict,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[done] -> {args.out_json}")


if __name__ == "__main__":
    main()
