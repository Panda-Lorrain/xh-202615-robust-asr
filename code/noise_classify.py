"""噪声类型估计器: 从识别音频估计 noise_type(babble/white/pink) → 选 SE atten-lim。

背景: SE 条件化最优(babble/white→=0 全力, pink→=6 温和, CER 2.82)靠 manifest 的
noise_type, 但**测试时不可知**。本模块用谱特征估计噪声类型, 使条件化**可部署**。

原理(谱平坦度 SFM = 几何均值/算术均值, 越高越"白"):
  babble(人声) : 非稳态, 谐波/共振峰多 → 谱平坦度**低**(频谱峰谷多)
  white        : 全频带均匀 → 谱平坦度**最高**(接近 1)
  pink         : 1/f 衰减, 低频能量多 → 谱平坦度**中**, 低/高频能量比高

用法:
  # 校准: 在 450 集(已知 noise_type)上看各类型谱特征分布, 定阈值
  code/.venv/Scripts/python.exe code/noise_classify.py --calibrate
  # 估计单条/目录: 输出每条估计 noise_type + 推荐 atten-lim
  code/.venv/Scripts/python.exe code/noise_classify.py --in-dir test_wav/dataset/final --out code/noise_est.json
"""
import os, sys, json, argparse, glob
import numpy as np
import librosa

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_MANIFEST = os.path.join(_ROOT, "test_wav", "dataset", "final", "final_manifest.json")


def spectral_features(wav, sr=16000, n_fft=2048):
    """返回 (flatness_mean, centroid_mean, low_high_ratio)。low_high_ratio = 低频(<=1kHz)/高频(>1kHz) 能量比。"""
    wav = np.asarray(wav, dtype=np.float32)
    if len(wav) < n_fft:
        wav = np.pad(wav, (0, n_fft - len(wav)))
    # 谱平坦度(每帧几何/算术均值, 取均值)
    flat = librosa.feature.spectral_flatness(y=wav, hop_length=512)
    flatness = float(flat.mean())
    # 谱质心(亮度, 越高越"亮")
    cent = librosa.feature.spectral_centroid(y=wav, sr=sr, hop_length=512)
    centroid = float(cent.mean())
    # 低/高频能量比(STFT 分频带)
    S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    low_mask = freqs <= 1000
    e_low = float(S[low_mask].sum())
    e_high = float(S[~low_mask].sum())
    lhr = e_low / max(e_high, 1e-9)
    return flatness, centroid, lhr


def classify(flatness, lhr, f_babble=0.06, f_pink=0.18):
    """三段阈值判噪声类型 → atten-lim。
    flatness 低(<f_babble)→babble→=0; 中(f_babble~f_pink + 低频多)→pink→=6; 高→white→=0。
    返回 (noise_type, atten_lim_db)。white 可部署用 =0(实测 =0/=6 对 white 近中性)。"""
    if flatness < f_babble:
        return "babble", 0
    if flatness < f_pink and lhr > 3.0:    # 中等平坦 + 低频能量多 → pink
        return "pink", 6
    return "white", 0


def calibrate(manifest_path, audio_dir):
    mani = json.load(open(manifest_path, encoding="utf-8"))
    by_nt = {"white": [], "pink": [], "babble": []}
    for it in mani["items"]:
        wav, _ = librosa.load(os.path.join(audio_dir, it["file"]), sr=16000)
        f, c, lhr = spectral_features(wav)
        by_nt[it["noise_type"]].append((f, c, lhr, it["file"]))
    print("=== 谱特征分布(按 noise_type) ===")
    print(f"{'type':>8} {'n':>4} {'flatness':>22} {'centroid':>22} {'low/high':>18}")
    summary = {}
    for nt, rows in by_nt.items():
        if not rows:
            continue
        fl = [r[0] for r in rows]; ce = [r[1] for r in rows]; lh = [r[2] for r in rows]
        summary[nt] = {"flatness": (np.mean(fl), np.std(fl), np.min(fl), np.max(fl)),
                       "centroid": (np.mean(ce), np.std(ce)),
                       "lhr": (np.mean(lh), np.std(lh))}
        print(f"{nt:>8} {len(rows):>4} "
              f"{np.mean(fl):.4f}±{np.std(fl):.4f}[{np.min(fl):.3f},{np.max(fl):.3f}] "
              f"{np.mean(ce):.0f}±{np.std(ce):.0f} "
              f"{np.mean(lh):.2f}±{np.std(lh):.2f}")
    # 找分离阈值: babble 上界 vs pink/white 下界
    bab_f = [r[0] for r in by_nt.get("babble", [])]
    nonbab_f = [r[0] for nt in ("white", "pink") for r in by_nt.get(nt, [])]
    if bab_f and nonbab_f:
        sep = (max(bab_f) + min(nonbab_f)) / 2
        acc = sum(1 for f in bab_f if f < sep) / len(bab_f)
        acc2 = sum(1 for f in nonbab_f if f >= sep) / len(nonbab_f)
        print(f"\n[babble vs stationary] 平坦度分离阈值 ≈ {sep:.4f} "
              f"| babble<{sep:.3f} 命中率 {acc:.2%} | stationary>={sep:.3f} 命中率 {acc2:.2%}")
    return summary


def estimate_dir(in_dir, out):
    wavs = sorted(glob.glob(os.path.join(in_dir, "*.wav")))
    rows = []
    correct = total = 0
    mani_map = {}
    mpath = os.path.join(in_dir, "final_manifest.json")
    if os.path.exists(mpath):
        mani_map = {it["file"]: it["noise_type"] for it in json.load(open(mpath, encoding="utf-8"))["items"]}
    for wp in wavs:
        wav, _ = librosa.load(wp, sr=16000)
        f, c, lhr = spectral_features(wav)
        nt, atten = classify(f, lhr)
        gold = mani_map.get(os.path.basename(wp))
        rows.append({"file": os.path.basename(wp), "flatness": round(f, 4),
                     "centroid": round(c, 1), "lhr": round(lhr, 2),
                     "est_noise": nt, "atten_lim_db": atten, "gold": gold})
        if gold:
            total += 1
            if gold == nt:
                correct += 1
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    acc = correct / total if total else 0
    print(f"[estimate] {len(rows)} 条 → {out} | 噪声类型估计准确率 {correct}/{total} = {acc:.2%}")
    if total:
        from collections import Counter
        cm = Counter((r["gold"], r["est_noise"]) for r in rows if r["gold"])
        print("  混淆矩阵 (gold→est):", dict(cm))
    return rows


def main():
    ap = argparse.ArgumentParser(description="噪声类型估计器(谱平坦度)→SE atten-lim")
    ap.add_argument("--calibrate", action="store_true", help="在 450 集校准谱特征分布/阈值")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--audio-dir", default=os.path.join(_ROOT, "test_wav", "dataset", "final"))
    ap.add_argument("--in-dir", help="估计目录每条噪声类型→atten-lim")
    ap.add_argument("--out", default=os.path.join(_HERE, "noise_est.json"))
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.manifest, args.audio_dir)
        return
    if args.in_dir:
        estimate_dir(args.in_dir, args.out)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
