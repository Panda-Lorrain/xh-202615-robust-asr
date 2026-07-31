#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 _aug_build_smoke/manifest.jsonl 的合成质量 (smoke 检查器)

逐对打印: enroll/recog 时长, ref 文本, 增广参数;
计算 enrollment 长度是否落在 1.5–2.5s, recognition 是否含重叠/噪声。
"""
from __future__ import annotations
import os, sys, json, glob, argparse
import librosa
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--play", action="store_true", help="打印可手动播放的路径")
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"[verify] 共 {len(rows)} 对")
    print("=" * 80)
    n_bad_enroll = 0
    sum_recog_dur = 0.0
    by_noise = {}
    overlap_hist = {}
    quiet_cnt = 0
    fast_cnt = 0
    polluted_cnt = 0
    for i, r in enumerate(rows):
        try:
            ye, sr_e = librosa.load(r["enrollment_audio"], sr=None)
            yr, sr_r = librosa.load(r["recognition_audio"], sr=None)
        except Exception as e:
            print(f"[{i}] LOAD FAIL {e}")
            continue
        enroll_dur = len(ye) / sr_e
        recog_dur = len(yr) / sr_r
        sum_recog_dur += recog_dur
        enroll_rms_db = 20 * np.log10(max(np.sqrt(np.mean(ye**2)), 1e-9) + 1e-12)
        recog_rms_db = 20 * np.log10(max(np.sqrt(np.mean(yr**2)), 1e-9) + 1e-12)
        by_noise[r["noise_type"]] = by_noise.get(r["noise_type"], 0) + 1
        ov = r["overlap_ratio"]
        overlap_hist[ov] = overlap_hist.get(ov, 0) + 1
        if r.get("target_gain_db", 0.0) != 0.0:
            quiet_cnt += 1
        if r.get("target_speed_rate", 1.0) != 1.0:
            fast_cnt += 1
        if r.get("enroll_pollute", False):
            polluted_cnt += 1
        enroll_ok = 1.5 <= enroll_dur <= 2.5
        if not enroll_ok:
            n_bad_enroll += 1
        if i < 6 or not enroll_ok:
            print(f"[{i}] id={r['id']}")
            print(f"     ref=\"{r['ref']}\" ({len(r['ref'])} chars)")
            print(f"     enroll {enroll_dur:.2f}s rms={enroll_rms_db:+.1f}dB "
                  f"{'OK' if enroll_ok else 'BAD'} | "
                  f"recog {recog_dur:.2f}s rms={recog_rms_db:+.1f}dB")
            print(f"     overlap={ov} snr={r['snr_db']}dB noise={r['noise_type']} "
                  f"gain={r.get('target_gain_db',0):+.1f}dB "
                  f"speed={r.get('target_speed_rate',1.0):.2f} "
                  f"polluted={r.get('enroll_pollute', False)}")
            if args.play:
                print(f"     -> {r['enrollment_audio']}")
                print(f"     -> {r['recognition_audio']}")
    print("=" * 80)
    print(f"[verify] enroll 时长违规 (不在 1.5–2.5s): {n_bad_enroll}/{len(rows)}")
    print(f"[verify] 平均 recog 时长: {sum_recog_dur/max(len(rows),1):.2f}s")
    print(f"[verify] 噪声类型分布: {by_noise}")
    print(f"[verify] 重叠分布: {sorted(overlap_hist.items())}")
    print(f"[verify] target 小声化: {quiet_cnt}/{len(rows)} "
          f"({100*quiet_cnt/max(len(rows),1):.0f}%)")
    print(f"[verify] target 快语速: {fast_cnt}/{len(rows)} "
          f"({100*fast_cnt/max(len(rows),1):.0f}%)")
    print(f"[verify] enroll 污染: {polluted_cnt}/{len(rows)} "
          f"({100*polluted_cnt/max(len(rows),1):.0f}%)")
    print(f"[verify] manifest 字段: {sorted(rows[0].keys()) if rows else 'EMPTY'}")


if __name__ == "__main__":
    main()
