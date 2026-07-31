#!/usr/bin/env python
"""POC: faster-whisper (CTranslate2) vs HF transformers vanilla 等价性 + RTF
===========================================================
目的：验证把 vanilla 后端从 HF transformers generate() 换成 faster-whisper CTranslate2，
是否零/低 CER 风险 + 效率（RTF/VRAM）收益。独立脚本，不改主推理链。

对比维度（相同输入 = 原始 recognition 音频，不切片，严格测引擎+量化差异）：
  - text 完全一致率（delta=0 的直接证据）
  - CER mean（官方口径近似，HF vs FW）
  - RTF mean（FW 应更快）
  - 峰值显存

第一版只测 FW-int8_float16（最大效率收益 / 最大 CER 风险配置）。
若 int8 delta 大，再转 fp16 测纯引擎差异。

用法: code/.venv/Scripts/python.exe code/poc_faster_whisper.py [N=30]
"""
import json, time, sys, os, gc, unicodedata, subprocess
import torch
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from text_utils import to_simplified, digit_postproc

def submit_norm(t):
    """提交归一：繁→简 + 数字后处理（与 enroll_infer/to_submission SSOT 对齐）。"""
    return digit_postproc(to_simplified(t or ""))

POS_PAIRS = "code/pos_pairs_datasetA.json"
HF_MODEL = "E:/hf_cache/whisper-large-v3-turbo"
CT2_INT8 = "E:/fw_cache/whisper-large-v3-turbo-ct2-int8"
N_DEFAULT = 30
DEVICE = "cuda"


def norm_text(s):
    """官方口径近似：NFKC + lower + 去所有 P* 标点和空白。"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(c for c in s if not unicodedata.category(c).startswith("P") and not c.isspace())


def cer_single(hyp, ref):
    import editdistance
    h, r = norm_text(hyp), norm_text(ref)
    if not r:
        return 0.0
    return editdistance.eval(h, r) / len(r)


def load_pairs(n):
    rows = json.load(open(POS_PAIRS, encoding="utf-8"))
    out = []
    for r in rows:
        wav = r.get("recognition", "")
        ref = r.get("ref", "") or ""
        if os.path.exists(wav) and ref:  # 只取 recognition 存在 + 有 ref 的
            out.append((wav, ref))
        if len(out) >= n:
            break
    return out


def peak_mib():
    """nvidia-smi 全局 GPU 显存(MiB)。ctranslate2 显存不走 torch.cuda，必须 nvidia-smi。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def run_hf(audios):
    from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor
    print(f"[HF] load {HF_MODEL} fp16 ...")
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(HF_MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    tok = AutoTokenizer.from_pretrained(HF_MODEL)           # 复现 enroll_infer（走 tokenizer.json，避开损坏的 vocab.json）
    fe = AutoFeatureExtractor.from_pretrained(HF_MODEL)
    out = []
    for i, (wav, ref) in enumerate(audios):
        audio = librosa.load(wav, sr=16000)[0]
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, torch.float16)
        am = torch.ones(1, inputs.shape[-1], dtype=torch.bool, device=DEVICE)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            seqs = model.generate(input_features=inputs, attention_mask=am,
                                  language="zh", task="transcribe", max_new_tokens=200)
        torch.cuda.synchronize(); dt = time.time() - t0
        text = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
        out.append((text, dt, len(audio) / 16000))
        print(f"  HF[{i}] {dt:.2f}s | {norm_text(text)[:40]}")
    peak = peak_mib()
    del model; gc.collect(); torch.cuda.empty_cache()
    return out, peak


def run_fw(audios, ct2_path, compute_type):
    from faster_whisper import WhisperModel
    print(f"[FW] load {ct2_path} compute_type={compute_type} ...")
    torch.cuda.reset_peak_memory_stats()
    model = WhisperModel(ct2_path, device=DEVICE, compute_type=compute_type)
    out = []
    for i, (wav, ref) in enumerate(audios):
        audio = librosa.load(wav, sr=16000)[0]
        t0 = time.time()
        segs, _ = model.transcribe(audio, language="zh", task="transcribe",
                                   beam_size=1, vad_filter=False)
        text = "".join(s.text for s in segs).strip()
        dt = time.time() - t0
        out.append((text, dt, len(audio) / 16000))
        print(f"  FW[{i}] {dt:.2f}s | {norm_text(text)[:40]}")
    peak = peak_mib()
    del model; gc.collect(); torch.cuda.empty_cache()
    return out, peak


def compare(name, a, b, refs, peak_a, peak_b):
    # 过提交归一(to_simplified+digit_postproc)后比，消除繁简输出差异，反映真实提交 CER
    na = [norm_text(submit_norm(x[0])) for x in a]
    nb = [norm_text(submit_norm(y[0])) for y in b]
    same = sum(1 for h, f in zip(na, nb) if h == f)
    cer_a = sum(cer_single(submit_norm(x[0]), r) for x, r in zip(a, refs)) / len(a)
    cer_b = sum(cer_single(submit_norm(y[0]), r) for y, r in zip(b, refs)) / len(b)
    rtf_a = sum(x[1] for x in a) / sum(x[2] for x in a)
    rtf_b = sum(x[1] for x in b) / sum(x[2] for x in b)
    print(f"\n{'='*60}\n=== {name} (n={len(a)}, 提交归一后) ===")
    print(f"  text 完全一致: {same}/{len(a)} = {100*same/len(a):.1f}%")
    print(f"  CER mean:  HF={cer_a:.4f}  FW={cer_b:.4f}  Δ={cer_b-cer_a:+.4f}")
    print(f"  RTF mean:  HF={rtf_a:.3f}  FW={rtf_b:.3f}  ({rtf_b/rtf_a:.2f}x, 越小越快)")
    print(f"  GPU 显存(nvidia-smi):  HF={peak_a}MiB  FW={peak_b}MiB")
    diffs = [(i, na[i], nb[i]) for i in range(len(a)) if na[i] != nb[i]]
    if diffs:
        print(f"  不一致样本(前5, 归一后):")
        for i, h, f in diffs[:5]:
            print(f"    [{i}] HF={h[:30]} || FW={f[:30]}")
    return dict(same=same, n=len(a), cer_a=cer_a, cer_b=cer_b,
                rtf_a=rtf_a, rtf_b=rtf_b, peak_a=peak_a, peak_b=peak_b)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_DEFAULT
    audios = load_pairs(n)
    if len(audios) < n:
        print(f"⚠️ 只拿到 {len(audios)} 条音频（pos_pairs 路径问题？），继续用这些")
    refs = [r for _, r in audios]
    print(f"POC faster-whisper vs HF vanilla | {len(audios)} 条 pos recognition 原始音频(不切片, 严格测引擎差异)\n")

    hf, peak_hf = run_hf(audios)

    results = {"n": len(audios), "hf": {"cer": None, "rtf": None, "peak": peak_hf}}
    if os.path.exists(CT2_INT8):
        fw, peak_fw = run_fw(audios, CT2_INT8, "int8_float16")
        r = compare("HF-fp16 vs FW-int8_float16", hf, fw, refs, peak_hf, peak_fw)
        results["fw_int8"] = r
    else:
        print(f"⚠️ ct2 int8 权重不存在: {CT2_INT8}（转换未完成？）")

    out_path = "code/poc_faster_whisper_result.json"
    json.dump(results, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n结果存 {out_path}")
    print("\n判定标准: text 一致率>95% 且 ΔCER<0.01 → int8 可用(零风险换效率); 否则需转 fp16 测纯引擎差异")


if __name__ == "__main__":
    main()
