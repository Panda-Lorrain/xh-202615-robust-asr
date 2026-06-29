"""实验: 强制 Whisper/DiCoW 中文, 抑制英文幻觉(转写质量瓶颈的根因)。

诊断(code/diag_transcript.py): 63% 转写是垃圾, 多为英文幻觉(Whisper-large-v3-turbo
在退化中文音频上语言漂移→英文)。enroll_infer 已传 language="zh", 但 Whisper 只强制
首位 token, 序列中段仍可漂英文。本脚本测更强约束是否压住英文:

策略:
  A baseline : language="zh" (当前 enroll_infer)
  B +prompt  : language="zh" + initial_prompt="以下是普通话的句子。"(标准中文强制 prompt)
  C +suppress: language="zh" + prompt + suppress_tokens 抑制英文 token 区间(可选)

在 ov0 单说话人(无重叠)条目上用 all-target STNO 直转 target, 隔离"转写质量"问题。
若 B/C 显著降 CER → 集成进 enroll_infer 重跑。

用法:
  source code/setenv.sh && export HF_HUB_OFFLINE=1
  code/.venv/Scripts/python.exe code/test_zh_force.py --n 6
"""
import os, sys, json, argparse, glob, time
import torch
import numpy as np
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DICOW_MODEL = "E:/hf_cache/DiCoW_v3_2"
DICOW_INF = os.path.join(_ROOT, "code", "DiCoW-inference")
for _p in (DICOW_INF, os.path.join(DICOW_INF, "DiariZen"), os.path.join(DICOW_INF, "DiariZen", "pyannote-audio")):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)
sys.path.insert(0, _HERE)
from eval_metrics import cer  # noqa

MANIFEST = os.path.join(_ROOT, "test_wav", "dataset", "final", "final_manifest.json")
FINAL_DIR = os.path.join(_ROOT, "test_wav", "dataset", "final")
ZH_PROMPT = "以下是普通话的句子。"


def all_target_stno(T):
    """[4, T]: target 行全 1(整段当 target), 其余 0。"""
    s = torch.zeros(4, T)
    s[1] = 1.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="每条件取几条(ov0)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dicow-model", default=DICOW_MODEL)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    mani = {it["file"]: it for it in json.load(open(MANIFEST, encoding="utf-8"))["items"]}
    # 取 ov0(无重叠, target 独占) 的条目, 覆盖 snr×noise
    ov0 = [it for it in mani.values() if it["overlap_ratio"] == 0.0]
    # 挑 snr{-5,0,5} × noise{white,babble} 各 n 条
    picks = []
    for snr in [-5, 0, 5]:
        for nt in ["white", "babble"]:
            cands = [it for it in ov0 if it["snr_db"] == snr and it["noise_type"] == nt][:args.n]
            picks.extend(cands)
    print(f"[test] {len(picks)} 条 ov0(单说话人) | 策略 A=baseline(zh) B=+中文prompt")

    print(f"[load] DiCoW {args.dicow_model}")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.dicow_model, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(args.dicow_model)
    fe = AutoFeatureExtractor.from_pretrained(args.dicow_model)
    prompt_ids = tok(ZH_PROMPT, add_special_tokens=False).input_ids

    rows = []
    for it in picks:
        wav, _ = librosa.load(os.path.join(FINAL_DIR, it["file"]), sr=16000)
        ifp = fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        T = ifp.shape[-1] // 2
        stno = all_target_stno(T)[None].to(device, dtype)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
        ref = it["target_ref"]
        out_row = {"file": it["file"], "snr": it["snr_db"], "noise": it["noise_type"], "ref": ref}

        for tag, extra in [("A_zh", {}), ("B_zh+prompt", {"prompt_ids": torch.tensor(prompt_ids, device=device)})]:
            with torch.no_grad():
                o = dicow.generate(input_features=ifp, attention_mask=am, stno_mask=stno,
                                   language="zh", task="transcribe", max_new_tokens=200, **extra)
            seqs = o["sequences"] if isinstance(o, dict) else o
            txt = tok.batch_decode(seqs, skip_special_tokens=True)[0].strip()
            out_row[tag] = txt
            out_row[tag + "_cer"] = round(cer(txt, ref), 3)
        rows.append(out_row)
        print(f"\n[{it['file']}] snr={it['snr_db']} noise={it['noise_type']}  ref: {ref}")
        print(f"  A(zh)      CER={out_row['A_zh_cer']:.2f}: {out_row['A_zh'][:48]}")
        print(f"  B(zh+prompt) CER={out_row['B_zh+prompt_cer']:.2f}: {out_row['B_zh+prompt'][:48]}")

    # 汇总
    a_cers = [r["A_zh_cer"] for r in rows]
    b_cers = [r["B_zh+prompt_cer"] for r in rows]
    print(f"\n=== 汇总({len(rows)} 条) ===")
    print(f"  A(zh)        均CER={np.mean(a_cers):.3f}  CER<0.5占比={np.mean([c<0.5 for c in a_cers]):.2%}")
    print(f"  B(zh+prompt) 均CER={np.mean(b_cers):.3f}  CER<0.5占比={np.mean([c<0.5 for c in b_cers]):.2%}")
    improved = sum(1 for a, b in zip(a_cers, b_cers) if b < a - 0.05)
    print(f"  B 改善条数: {improved}/{len(rows)}")
    json.dump(rows, open(os.path.join(_HERE, "test_zh_force_result.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
