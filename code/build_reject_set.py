"""造 target 缺席拒识测试集: 识别音频只含 non-target(苏打)说话 + 噪声, 无 target(冰糖)。

背景: 450 矩阵里 target(冰糖)恒在场, 只能测"误拒率"(target 在场却被拒);
拒识评分 40% 的另一半"真实拒识率"(target 缺席时正确拒)需要一个 target 缺席集。
配 冰糖 enrollment(target_long_01) 跑 enroll_infer → pipeline 应拒识(sim 低/STNO 无 target)。

构造: 每条 nontarget(苏打闲话) 单独 + 程序噪声(white/pink/babble)按 SNR, 不含冰糖。
输出: test_wav/dataset/reject/*.wav + reject_manifest.json
      (每条 target_present=False, target_ref="" → 正确输出=空/拒识)。

用法:
  code/.venv/Scripts/python.exe code/build_reject_set.py
"""
import os, sys, json, glob, random
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RAW = os.path.join(_ROOT, "test_wav", "dataset", "raw")
REJECT_DIR = os.path.join(_ROOT, "test_wav", "dataset", "reject")
SR = 16000
SNRS = [-5, 0, 5]
NOISE_TYPES = ["white", "pink", "babble"]

sys.path.insert(0, _HERE)
from simulate_pipeline import add_noise  # noqa
from build_dataset import gen_white, gen_pink, gen_babble  # noqa


def main():
    os.makedirs(REJECT_DIR, exist_ok=True)
    raw_manifest = json.load(open(os.path.join(RAW, "manifest.json"), encoding="utf-8"))
    nontargets = [m for m in raw_manifest["items"] if m["role"] == "nontarget_cmd"]
    if not nontargets:
        raise SystemExit("[fatal] raw 缺 nontarget wav")
    for m in nontargets:
        m["path"] = os.path.join(RAW, "nontarget", m["name"] + ".wav")

    nt_wavs = [librosa.load(m["path"], sr=SR)[0] for m in nontargets]
    rng = random.Random(7)
    nprng = np.random.default_rng(7)
    items, cnt = [], 0

    for nt in nontargets:
        nt_wav = librosa.load(nt["path"], sr=SR)[0]
        for snr in SNRS:
            for noise_t in NOISE_TYPES:
                n = len(nt_wav)
                if noise_t == "white":
                    noise = gen_white(n, nprng)
                elif noise_t == "pink":
                    noise = gen_pink(n, nprng)
                else:
                    noise = gen_babble(nt_wavs, n, nprng)
                noise = noise.astype(np.float32)
                if len(noise) < n:
                    noise = np.pad(noise, (0, n - len(noise)))
                else:
                    noise = noise[:n]
                final = add_noise(nt_wav, noise, snr)
                name = f"{nt['name']}_solo_snr{snr:+d}_{noise_t}.wav"
                sf.write(os.path.join(REJECT_DIR, name), final, SR)
                items.append({
                    "file": name,
                    "target_ref": "",            # 无 target → 正确输出=空
                    "target_present": False,     # 应被拒识
                    "speaker": nt.get("voice", "苏打"),
                    "snr_db": snr,
                    "noise_type": noise_t,
                    "overlap_ratio": 0.0,
                })
                cnt += 1

    manifest = {"sr": SR, "count": cnt, "set": "target_absent_reject",
                "enrollment": "冰糖(target_long_01)", "items": items}
    out_json = os.path.join(REJECT_DIR, "reject_manifest.json")
    json.dump(manifest, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] {cnt} 条 target 缺席音频 → {REJECT_DIR}")
    print(f"       manifest → {out_json} (配冰糖 enrollment, 正确=pipeline 拒识)")


if __name__ == "__main__":
    main()
