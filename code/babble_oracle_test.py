"""铁证实验(babble_oracle_test.py): 钉死 H1/H2 + 回应"babble含英文"质疑。

三个场景, 都用 oracle STNO(target 行=全程, 排除 diar 误检/漏检干扰):
  ① babble 样本(t_01_n_07, target中文+babble噪声) + oracle全程target
     → 中文: H1(diar漏检target帧是问题, 完美STNO能救); 英文: H2(完美STNO也救不了babble)
  ② 纯 babble(gen_babble=4条中文nontarget叠加, 无target) + 全程STNO
     → 中文片段: 转写babble内容; 英文: babble触发英文幻觉(非转写, 因babble是中文)
  ③ 纯干净 nontarget(n_07「帮我倒杯水」, 无噪声) + 全程STNO
     → 中文: DiCoW能正确转中文人声(基准); 英文: 模型连干净中文都漂(更深问题)

【为什么不是 stno_ablation 的 C】stno_ablation 的 C 用 diar_mask[target](0.067, diar检出),
  但 diar 在 babble 上误检→其 target 帧检出本身可疑(漏检?). 本脚本①用全程target=真oracle。

运行: source code/setenv.sh && code/.venv/Scripts/python.exe code/babble_oracle_test.py
"""
import os
import sys
import numpy as np
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoFeatureExtractor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, _HERE)
from enroll_infer import DICOW_MODEL, get_stno_mask
from build_dataset import gen_babble


def stno_full_target(audio_len):
    """oracle STNO: 单 speaker, target 行=全程(ov0 时 target 全程说话的 ground truth)。"""
    diar_mask = torch.ones(1, audio_len)
    return get_stno_mask(diar_mask, 0)


def is_chinese(s):
    return sum(1 for c in s if '一' <= c <= '鿿') > max(len(s) * 0.3, 3)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    print(f"[load] DiCoW {DICOW_MODEL} on {device}")
    dicow = AutoModelForSpeechSeq2Seq.from_pretrained(
        DICOW_MODEL, trust_remote_code=True, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(DICOW_MODEL)
    fe = AutoFeatureExtractor.from_pretrained(DICOW_MODEL)

    def transcribe(wav, label):
        ifp = fe(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        audio_len = ifp.shape[-1] // 2
        stno = stno_full_target(audio_len)
        am = torch.ones(1, ifp.shape[-1], dtype=torch.bool, device=device)
        with torch.no_grad():
            out = dicow.generate(input_features=ifp, attention_mask=am,
                                 stno_mask=stno[None].to(device, dtype),
                                 language="zh", task="transcribe", max_new_tokens=200)
        text = tok.batch_decode(out["sequences"] if isinstance(out, dict) else out,
                                skip_special_tokens=True)[0].strip()
        print(f"  [{label}] {len(text)}字 | {'中文' if is_chinese(text) else '非中文'}: {text[:90]}")
        return text

    # ① babble 样本 + oracle 全程 target(修 stno_ablation C 的 diar 帧不准漏洞)
    print("\n=== ① babble 样本(t_01_n_07) + oracle STNO(target行=全程) ===")
    babble_wav, _ = librosa.load(
        os.path.join(_ROOT, "test_wav/dataset/final/t_01_n_07_ov000_snr+5_babble.wav"), sr=16000)
    t1 = transcribe(babble_wav, "babble样本+oracle全程target")

    # ② 纯 babble(4 条中文 nontarget 叠加, 无 target)
    print("\n=== ② 纯 babble(gen_babble=4中文叠加, 无target) + 全程STNO ===")
    nt_dir = os.path.join(_ROOT, "test_wav/dataset/raw/nontarget")
    nt_wavs = [librosa.load(os.path.join(nt_dir, f"n_{i:02d}.wav"), sr=16000)[0] for i in range(1, 9)]
    rng = np.random.default_rng(42)
    pure_babble = gen_babble(nt_wavs, 16 * 16000, rng)
    t2 = transcribe(pure_babble, "纯babble_4中文叠加")

    # ③ 纯干净 nontarget 中文(基准)
    print("\n=== ③ 纯干净 nontarget n_07(帮我倒杯水) + 全程STNO ===")
    n07, _ = librosa.load(os.path.join(nt_dir, "n_07.wav"), sr=16000)
    t3 = transcribe(n07, "干净nontarget_n07")

    print("\n=== 判定 ===")
    print(f"① babble+oracle全程target → {'中文 ⇒ H1(diar漏检是问题,完美STNO能救, 前H2结论错)' if is_chinese(t1) else '非中文 ⇒ H2确证(完美STNO也救不了babble)'}")
    print(f"② 纯babble(4中文叠加)    → {'中文片段(转写babble内容)' if is_chinese(t2) else '非中文 ⇒ babble触发英文幻觉(非转写, babble是中文)——钉死质疑'}")
    print(f"③ 干净nontarget           → {'中文(DiCoW转中文人声正常, 基准OK)' if is_chinese(t3) else '非中文 ⇒ 模型连干净中文都漂(更深问题)'}")


if __name__ == "__main__":
    main()
