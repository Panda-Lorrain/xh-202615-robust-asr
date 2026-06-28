"""组装 TTS 数据集矩阵 — 把 raw 单人 wav 混成「重叠×SNR×噪声」识别集 (W2)。

读 raw/manifest.json, 对每条 target 指令:
  与随机 nontarget 干扰按 overlap_ratio 重叠 → 加程序噪声(按 SNR) → 输出
矩阵: overlap [0,25,50,75,100]% × snr [-5,0,5] × noise [white,pink,babble,(env)]
输出: test_wav/dataset/final/*.wav + final_manifest.json (含 ground truth: target 参考文本)

噪声源:
  - 程序噪声 white/pink/babble (零依赖, 立即可用)
  - 若 test_wav/dataset/env_noise/ 下有真实环境音(ESC-50/MUSAN 下载), 自动加入 noise_type="env"
复用 simulate_pipeline.add_noise / mix_overlap。

Windows 跑: code/.venv/Scripts/python.exe code/build_dataset.py
"""
import os, json, glob, random
import numpy as np
import librosa
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
RAW = os.path.join(_ROOT, "test_wav", "dataset", "raw")
FINAL = os.path.join(_ROOT, "test_wav", "dataset", "final")
ENV_NOISE_DIR = os.path.join(_ROOT, "test_wav", "dataset", "env_noise")
SR = 16000

OVERLAPS = [0.0, 0.25, 0.5, 0.75, 1.0]
SNRS = [-5, 0, 5]
NOISE_TYPES = ["white", "pink", "babble"]

# 复用 simulate_pipeline 的混合/加噪
import sys
sys.path.insert(0, _HERE)
from simulate_pipeline import add_noise, mix_overlap


def gen_white(n, rng):
    return rng.standard_normal(n).astype(np.float32)


def gen_pink(n, rng):
    """1/f 粉红噪声 (FFT 滤波)。"""
    white = rng.standard_normal(n)
    f = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # 避免除零
    f = f / np.sqrt(freqs)
    pink = np.fft.irfft(f, n=n)
    return pink.astype(np.float32)


def gen_babble(nontarget_wavs, n, rng, n_spk=4):
    """多人 babble: 随机选 n_spk 条 nontarget 叠加平均。"""
    pool = [w for w in nontarget_wavs if len(w) > 0]
    k = min(n_spk, len(pool))
    if k == 0:
        return rng.standard_normal(n).astype(np.float32) * 0.05
    chosen = rng.choice(len(pool), size=k, replace=False)
    out = np.zeros(n, dtype=np.float32)
    for idx in chosen:
        w = pool[idx]
        if len(w) < n:
            reps = n // len(w) + 1
            w = np.tile(w, reps)[:n]
        else:
            w = w[:n]
        out += w / k
    return out


def load_env_noise(rng):
    """加载 env_noise/ 下真实环境音(可选)。返回 (list[np.ndarray], list[str])。"""
    envs, names = [], []
    if os.path.isdir(ENV_NOISE_DIR):
        for p in sorted(glob.glob(os.path.join(ENV_NOISE_DIR, "*.wav"))):
            try:
                w, _ = librosa.load(p, sr=SR)
                if len(w) > 0:
                    envs.append(w)
                    names.append(os.path.splitext(os.path.basename(p))[0])
            except Exception as e:
                print(f"  [skip env] {p}: {e}")
    return envs, names


def main():
    os.makedirs(FINAL, exist_ok=True)
    manifest_path = os.path.join(RAW, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        raw_manifest = json.load(f)
    items = raw_manifest["items"]
    # tts_dataset_gen.py 在 WSL 跑, manifest 的 path 是 /mnt/e/... WSL 路径;
    # build_dataset 在 Windows 跑 → 用 role+name 重建 Windows 路径(避免 WSL/Win 路径不匹配)
    _ROLE_DIR = {"enrollment_long": "enrollment", "enrollment_short": "enrollment",
                 "target_cmd": "target", "nontarget_cmd": "nontarget"}
    for m in items:
        m["path"] = os.path.join(RAW, _ROLE_DIR.get(m["role"], "."), m["name"] + ".wav")

    target_cmds = [m for m in items if m["role"] == "target_cmd"]
    nontargets = [m for m in items if m["role"] == "nontarget_cmd"]
    if not target_cmds or not nontargets:
        raise SystemExit(f"[fatal] raw 缺 target/nontarget wav (target={len(target_cmds)} nt={len(nontargets)})")

    # 预加载 nontarget wav (供 babble + 重叠)
    nt_wavs = [librosa.load(m["path"], sr=SR)[0] for m in nontargets]

    envs, env_names = load_env_noise(random.Random(0))
    noise_types = list(NOISE_TYPES)
    if envs:
        noise_types.append("env")
        print(f"[env] 加载 {len(envs)} 条真实环境音: {env_names[:5]}...")
    else:
        print(f"[env] 无真实环境音, 仅用程序噪声 (white/pink/babble)。可后续放 wav 到 {ENV_NOISE_DIR} 重跑")

    rng = random.Random(42)
    nprng = np.random.default_rng(42)
    final_items = []
    cnt = 0

    for t in target_cmds:
        t_wav, _ = librosa.load(t["path"], sr=SR)
        for ov in OVERLAPS:
            for snr in SNRS:
                for noise_t in noise_types:
                    # 随机选一条 nontarget 干扰
                    nt_idx = rng.randrange(len(nt_wavs))
                    nt_wav = nt_wavs[nt_idx]
                    mixed = mix_overlap(t_wav, nt_wav, ov)

                    n = len(mixed)
                    if noise_t == "white":
                        noise = gen_white(n, nprng)
                    elif noise_t == "pink":
                        noise = gen_pink(n, nprng)
                    elif noise_t == "babble":
                        noise = gen_babble(nt_wavs, n, nprng)
                    elif noise_t == "env":
                        e = envs[rng.randrange(len(envs))]
                        noise = np.tile(e, n // len(e) + 1)[:n] if len(e) < n else e[:n]
                    else:
                        continue
                    noise = noise.astype(np.float32)
                    if len(noise) < n:
                        noise = np.pad(noise, (0, n - len(noise)))
                    else:
                        noise = noise[:n]

                    final = add_noise(mixed, noise, snr)
                    name = (f"{t['name']}_{nontargets[nt_idx]['name']}"
                            f"_ov{int(ov * 100):03d}_snr{snr:+d}_{noise_t}.wav")
                    sf.write(os.path.join(FINAL, name), final, SR)
                    final_items.append({
                        "file": name,
                        "target_ref": t["ref"],          # ground truth: 目标指令文本(算 CER 用)
                        "target_voice": raw_manifest["voices"]["target"],
                        "interferer": nontargets[nt_idx]["name"],
                        "overlap_ratio": ov,
                        "snr_db": snr,
                        "noise_type": noise_t,
                    })
                    cnt += 1
        print(f"  [target {t['name']}] 完成 {len(OVERLAPS) * len(SNRS) * len(noise_types)} 条")

    final_manifest = {
        "sr": SR,
        "matrix": {"overlaps": OVERLAPS, "snrs": SNRS, "noise_types": noise_types},
        "n_target_cmds": len(target_cmds),
        "count": cnt,
        "items": final_items,
    }
    out_json = os.path.join(FINAL, "final_manifest.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[done] 生成 {cnt} 条识别音频 → {FINAL}")
    print(f"        矩阵: {len(target_cmds)} target × {len(OVERLAPS)} overlap × {len(SNRS)} snr × {len(noise_types)} noise")
    print(f"        manifest → {out_json}")
    print("[用途] final_manifest.json 的 target_ref 是 ground truth, 配 eval_metrics.py 算 CER;")


if __name__ == "__main__":
    main()
