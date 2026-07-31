"""vanilla Whisper-large-v3-turbo(无 FDDT/STNO)转 babble —— 区分 Whisper×babble 层 vs DiCoW 特异。

审查 P0 #1: DiCoW 基座是 whisper-large-v3-turbo。本脚本用【原版 Whisper】(无 FDDT 条件化、无 STNO)
转写同样三场景, 干净测试"babble 音频是否让 Whisper 基座本身漂英文":
  - vanilla 出中文 → Whisper 基座在 babble 上不漂英文 → babble 英文是 DiCoW FDDT/STNO 病因(→ H3 方向, 杠杆=修 STNO/constrained decode)
  - vanilla 出英文 → Whisper×babble 层本身漂 → H2 方向, 杠杆=SE/微调(模型层)

对照 babble_oracle_test.py(DiCoW): DiCoW+oracle全程target→中文; DiCoW+diar低覆盖STNO→英文。
本脚本补"无 STNO 条件"这一臂, 三角定位。

运行(需先下载 vanilla 到 E:/hf_cache/whisper-large-v3-turbo):
  source code/setenv.sh
  code/.venv/Scripts/python.exe code/vanilla_whisper_test.py
"""
import os
import sys
import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
VANILLA = "E:/hf_cache/whisper-large-v3-turbo"


def is_chinese(s):
    return sum(1 for c in s if '一' <= c <= '鿿') > max(len(s) * 0.3, 3)


def main():
    if not os.path.isdir(VANILLA):
        print(f"[fatal] vanilla 权重未下载: {VANILLA}")
        print("        先跑: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai/whisper-large-v3-turbo --local-dir " + VANILLA)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"[load] vanilla Whisper-large-v3-turbo {VANILLA} on {device}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(VANILLA, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(VANILLA)
    fe = AutoFeatureExtractor.from_pretrained(VANILLA)

    def transcribe(wav, label):
        ifp = fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
        with torch.no_grad():
            out = model.generate(input_features=ifp, attention_mask=am,
                                 language="zh", task="transcribe", max_new_tokens=200)
        text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        print(f"  [{label}] {len(text)}字 | {'中文' if is_chinese(text) else '非中文'}: {text[:90]}")
        return text

    # ① babble 样本(target中文 + babble噪声)
    print("\n=== ① babble 样本(t_01_n_07_ov000_snr+5) ===")
    babble_wav, _ = librosa.load(
        os.path.join(_ROOT, "test_wav/dataset/final/t_01_n_07_ov000_snr+5_babble.wav"), sr=16000)
    t1 = transcribe(babble_wav, "babble样本")

    # ② 纯 babble(4 中文叠加, 无 target)
    print("\n=== ② 纯 babble(gen_babble=4中文叠加) ===")
    sys.path.insert(0, _HERE)
    from build_dataset import gen_babble
    nt_dir = os.path.join(_ROOT, "test_wav/dataset/raw/nontarget")
    nt_wavs = [librosa.load(os.path.join(nt_dir, f"n_{i:02d}.wav"), sr=16000)[0] for i in range(1, 9)]
    pure_babble = gen_babble(nt_wavs, 16 * 16000, np.random.default_rng(42))
    t2 = transcribe(pure_babble, "纯babble")

    # ③ 干净 target t_01(基准: vanilla Whisper 转干净中文)
    print("\n=== ③ 干净 target t_01(请把客厅...) 基准 ===")
    clean, _ = librosa.load(os.path.join(_ROOT, "test_wav/dataset/raw/target/t_01.wav"), sr=16000)
    t3 = transcribe(clean, "干净target")

    print("\n=== 判定 ===")
    if is_chinese(t1):
        print("① babble样本 → 中文 ⇒ Whisper基座在babble上不漂英文 → babble英文是DiCoW FDDT/STNO病因(H3方向, 杠杆=STNO/constrained decode)")
    else:
        print("① babble样本 → 非中文 ⇒ Whisper×babble层本身漂英文(H2方向, 杠杆=SE/微调)")
    print(f"② 纯babble → {'中文' if is_chinese(t2) else '非中文'}")
    print(f"③ 干净target → {'中文(基准OK)' if is_chinese(t3) else '非中文(模型问题)'}")


if __name__ == "__main__":
    main()
