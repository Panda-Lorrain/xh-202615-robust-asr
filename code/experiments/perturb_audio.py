#!/usr/bin/env python
"""B3 输入微扰: 对 pos_pairs 的 recognition 音频生成扰动版(gauss/vol/time), 输出新 pairs JSON。

spec §5 B3。扰动音频缓存在 code/stability_matrix/perturbed/<perturb>/<uid>.wav。
分析时 stability_test.py --phase B3 调本脚本生成新 pairs 后跑 enroll_infer。

用法:
  code/.venv/Scripts/python.exe code/perturb_audio.py --perturb gauss
"""
import os, json, wave, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(_HERE, "pos_pairs_datasetA.json")
OUT_BASE = os.path.join(_HERE, "stability_matrix", "perturbed")


def read_wav_mono(p):
    with wave.open(p, "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def write_wav(p, x, sr):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    xi = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(xi.tobytes())


def perturb_gauss(x, sr):   # 叠加 -45 dB 高斯噪(不可感知, 测数值边界)
    rng = np.random.default_rng(42)
    return x + rng.standard_normal(len(x)) * (10 ** (-45 / 20))

def perturb_vol(x, sr):     # +1 dB(测能量敏感)
    return x * (10 ** (1 / 20))

def perturb_time(x, sr):    # 前补 20ms 静音(测对齐敏感)
    shift = int(0.020 * sr)
    return np.concatenate([np.zeros(shift), x])[:len(x)]


PERTURBS = {"gauss": perturb_gauss, "vol": perturb_vol, "time": perturb_time}


def main():
    ap = argparse.ArgumentParser(description="B3 输入微扰音频生成")
    ap.add_argument("--perturb", required=True, choices=list(PERTURBS))
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条(0=全部, dry-run 用 5)")
    args = ap.parse_args()
    fn = PERTURBS[args.perturb]
    rows = json.load(open(args.pairs, encoding="utf-8"))
    if args.limit > 0:
        rows = rows[:args.limit]
    out_dir = os.path.join(OUT_BASE, args.perturb)
    new_rows = []
    for r in rows:
        uid = os.path.splitext(os.path.basename(r["recognition"]))[0]
        dst = os.path.join(out_dir, f"{uid}.wav")
        if not os.path.exists(dst):
            x, sr = read_wav_mono(r["recognition"])
            write_wav(dst, fn(x, sr), sr)
        new_rows.append({"id": r.get("id"), "enrollment": r["enrollment"],
                         "recognition": dst, "ref": r.get("ref", ""),
                         "kws_txt": r.get("kws_txt", "")})
    out_pairs = os.path.join(_HERE, "stability_matrix", f"_pairs_B3_{args.perturb}.json")
    os.makedirs(os.path.dirname(out_pairs), exist_ok=True)
    json.dump(new_rows, open(out_pairs, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[perturb {args.perturb}] {len(new_rows)} 条 → {out_pairs}")


if __name__ == "__main__":
    main()
