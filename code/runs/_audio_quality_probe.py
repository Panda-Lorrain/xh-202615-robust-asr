"""Audio-quality multi-dim reject-calibration probe (direction-4 supplement).

Tests whether pure acoustic-quality signals (SNR / bandwidth / spectral
centroid / flatness / clipping / silence / ZCR) computed on the target
timeline slice can discriminate:
  TEST1 general:     pos_n1 (target mono)  vs  neg_n1 (interferent mono)
  TEST2 operational: pos_frej (false-rej @ thr0.27, mainline CER<0.3)
                     vs  neg_n1_rej (correctly-rej neg)

Break-even: recovery-precision > 0.763 on TEST2 (lever: 1 pos saved
= +0.9/1364; 1 neg leaked = -1/474). Per-signal TEST2 Pmax > 0.763
=> worth fusing; else audio-quality direction NO-GO.

Reuses AUC framework (auc_mwu/bootstrap_auc/thr_pr/summarize) from
code/runs/_speaker_aware_probe.py.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ----复用 AUC 框架(从 _speaker_aware_probe 复制,避免 import 副作用)----
def rankdata(a):
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(len(a), dtype=np.intp)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)

def auc_mwu(scores_pos, scores_neg):
    sp = np.asarray(scores_pos, dtype=float)
    sn = np.asarray(scores_neg, dtype=float)
    n1, n2 = len(sp), len(sn)
    if n1 == 0 or n2 == 0:
        return float('nan')
    alls = np.concatenate([sp, sn])
    r = rankdata(alls)
    r_sum = r[:n1].sum()
    U = r_sum - n1 * (n1 + 1) / 2.0
    return U / (n1 * n2)

def bootstrap_auc(sp, sn, B=500, seed=42):
    rng = np.random.default_rng(seed)
    sp = np.asarray(sp, dtype=float); sn = np.asarray(sn, dtype=float)
    aucs = []
    for _ in range(B):
        sp_s = rng.choice(sp, size=len(sp), replace=True)
        sn_s = rng.choice(sn, size=len(sn), replace=True)
        aucs.append(auc_mwu(sp_s, sn_s))
    return (float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 50)),
            float(np.percentile(aucs, 97.5)))

def thr_pr(sp, sn, direction='gt'):
    sp = np.asarray(sp, dtype=float); sn = np.asarray(sn, dtype=float)
    out = []
    thresholds = sorted(set(list(sp) + list(sn)))
    for t in thresholds:
        if direction == 'gt':
            tp = int((sp >= t).sum()); fp = int((sn >= t).sum())
        else:
            tp = int((sp <= t).sum()); fp = int((sn <= t).sum())
        fn = len(sp) - tp; tn = len(sn) - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append({'thr': float(t), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'precision': prec, 'recall': rec, 'f1': f1})
    return out

def pmax_recall_half(sp, sn, direction='gt'):
    """Max precision at recall>=0.5 = recovery-precision upper bound."""
    pr = thr_pr(sp, sn, direction)
    rec_05 = [r for r in pr if r['recall'] >= 0.5]
    if not rec_05:
        return None
    best = max(rec_05, key=lambda r: r['precision'])
    return {
        'thr': best['thr'], 'precision': best['precision'],
        'recall': best['recall'], 'tp': best['tp'], 'fp': best['fp'],
        'fn': best['fn'], 'tn': best['tn'],
    }

def best_f1(sp, sn, direction='gt'):
    pr = thr_pr(sp, sn, direction)
    if not pr:
        return None
    b = max(pr, key=lambda r: r['f1'])
    return {'thr': b['thr'], 'precision': b['precision'], 'recall': b['recall'],
            'f1': b['f1'], 'tp': b['tp'], 'fp': b['fp'], 'fn': b['fn'], 'tn': b['tn']}


# ----音频质量信号提取----
def _stft_power(x, sr, n_fft=512, hop=160, win=None):
    """Frame -> FFT power spectrum. 16k/32ms/10ms ~ standard."""
    if win is None:
        win = np.hanning(n_fft + 1)[:-1]
    n_frames = max(1, 1 + (len(x) - n_fft) // hop)
    frames = np.zeros((n_frames, n_fft), dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        e = min(s + n_fft, len(x))
        frames[i, :e - s] = x[s:e]
    frames = frames * win
    spec = np.fft.rfft(frames, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    return power  # (n_frames, n_fft/2+1)

def extract_signals(path):
    """Return dict of audio-quality signals. None on read failure."""
    try:
        x, sr = sf.read(path, dtype='float32', always_2d=False)
    except Exception as e:
        return {'_error': str(e)}
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        # resample linearly (slices should already be 16k mono)
        idx = np.arange(len(x)) * (16000 / sr)
        idx_i = idx.astype(np.int64)
        idx_i = idx_i[idx_i < len(x)]
        x = x[idx_i]
        sr = 16000
    x = x.astype(np.float64)
    n = len(x)
    if n < 64:
        return {'_error': f'too_short n={n}'}

    out = {}
    # 1. rms (whole-signal)
    out['rms'] = float(np.sqrt(np.mean(x ** 2)) + 1e-12)

    # 2. peak
    out['peak'] = float(np.max(np.abs(x)))

    # 3. clipping_rate (|sample|>0.99 of full-scale int16)
    out['clipping_rate'] = float(np.mean(np.abs(x) > 0.99))

    # frame-level RMS for SNR/silence
    n_fft = 512
    hop = 160
    # frame rms (32ms window, 10ms hop)
    n_frames = max(1, 1 + (n - n_fft) // hop)
    frame_rms = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s + n_fft]
        frame_rms[i] = np.sqrt(np.mean(seg ** 2)) + 1e-12

    # 4. snr_est: WADA-style (top-decile / bottom-decile RMS)
    top = np.percentile(frame_rms, 90)
    bot = np.percentile(frame_rms, 10)
    out['snr_est'] = float(top / (bot + 1e-12))

    # 5. silence_ratio: frames with rms < 0.1 * median frame rms
    med = np.median(frame_rms)
    out['silence_ratio'] = float(np.mean(frame_rms < 0.1 * med))

    # zero crossing rate (frame-level mean)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s + n_fft]
        if len(seg) > 1:
            sign = np.sign(seg)
            zcr[i] = np.mean(np.abs(np.diff(sign)) > 0) * (sr / 2)
    out['zero_crossing_rate'] = float(np.mean(zcr))

    # spectral features (use same frames)
    power = _stft_power(x, sr, n_fft=n_fft, hop=hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # (n_fft/2+1,)
    total_power = power.sum(axis=1) + 1e-12   # (n_frames,)

    # 6. high_freq_ratio: >4kHz energy / total
    hf_mask = freqs >= 4000.0
    hf_e = power[:, hf_mask].sum(axis=1)
    out['high_freq_ratio'] = float((hf_e / total_power).mean())

    # 7. spectral_centroid (freq-weighted mean)
    sc = (power * freqs[None, :]).sum(axis=1) / total_power
    out['spectral_centroid'] = float(np.mean(sc))

    # 8. spectral_flatness: exp(mean(log(p))) / mean(p) per frame, averaged
    p_safe = np.maximum(power, 1e-12)
    gm = np.exp(np.mean(np.log(p_safe), axis=1))
    am = np.mean(p_safe, axis=1)
    out['spectral_flatness'] = float(np.mean(gm / (am + 1e-12)))

    return out


def collect(uids, slice_map):
    out = {}
    miss = []
    for u in uids:
        p = slice_map.get(u)
        if p is None or not os.path.isfile(p):
            miss.append(u)
            out[u] = None
            continue
        out[u] = extract_signals(p)
    return out, miss


def vec(feats_dict, key):
    """Extract one signal vector across uid dict (skip None/error)."""
    vals = []
    for u, f in feats_dict.items():
        if f is None or '_error' in f:
            continue
        v = f.get(key)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            continue
        vals.append((u, v))
    return vals


def run_signal(name, pos_feats, neg_feats, key):
    pos_pairs = vec(pos_feats, key)
    neg_pairs = vec(neg_feats, key)
    if len(pos_pairs) < 5 or len(neg_pairs) < 5:
        return {'error': f'insufficient samples pos={len(pos_pairs)} neg={len(neg_pairs)}'}
    sp_u, sp = zip(*pos_pairs)
    sn_u, sn = zip(*neg_pairs)
    sp = np.array(sp); sn = np.array(sn)

    # auto-detect direction: gt vs lt by which gives higher AUC
    auc_gt = auc_mwu(sp, sn)
    direction = 'gt' if auc_gt >= 0.5 else 'lt'
    auc_lo, auc_med, auc_hi = bootstrap_auc(sp, sn, B=500, seed=42)
    # report AUC as "pos has higher score" sense (flip if direction=lt)
    auc_report_lo, auc_report_med, auc_report_hi = (
        auc_lo if direction == 'gt' else 1 - auc_hi,
        auc_med if direction == 'gt' else 1 - auc_med,
        auc_hi if direction == 'gt' else 1 - auc_lo,
    )

    pr = pmax_recall_half(sp, sn, direction)
    f1 = best_f1(sp, sn, direction)

    return {
        'n_pos': len(sp), 'n_neg': len(sn),
        'pos_mean': float(sp.mean()), 'pos_median': float(np.median(sp)),
        'pos_std': float(sp.std()),
        'pos_p10': float(np.percentile(sp, 10)),
        'pos_p25': float(np.percentile(sp, 25)),
        'pos_p75': float(np.percentile(sp, 75)),
        'pos_p90': float(np.percentile(sp, 90)),
        'neg_mean': float(sn.mean()), 'neg_median': float(np.median(sn)),
        'neg_std': float(sn.std()),
        'neg_p10': float(np.percentile(sn, 10)),
        'neg_p25': float(np.percentile(sn, 25)),
        'neg_p75': float(np.percentile(sn, 75)),
        'neg_p90': float(np.percentile(sn, 90)),
        'auc_gt': float(auc_gt),
        'direction': direction,
        'auc_lo': auc_report_lo, 'auc_med': auc_report_med, 'auc_hi': auc_report_hi,
        'pmax_recall_half': pr,
        'best_f1': f1,
        'pos_uids': list(sp_u), 'neg_uids': list(sn_u),
    }


SIGNAL_KEYS = [
    'rms', 'snr_est', 'high_freq_ratio', 'spectral_centroid',
    'spectral_flatness', 'clipping_rate', 'silence_ratio',
    'zero_crossing_rate', 'peak',
]


def main():
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)

    with open('code/runs/_decoder_probe/subset_uids.json') as f:
        subsets = json.load(f)
    with open('code/runs/_decoder_probe/slice_paths.json') as f:
        slice_map_full = json.load(f)
    pos_slices = slice_map_full['pos']
    neg_slices = slice_map_full['neg']

    print('[load] subsets:', {k: len(v) for k, v in subsets.items()})
    print(f'[load] pos slice_map entries: {len(pos_slices)} | neg: {len(neg_slices)}')

    out_dir = Path('code/runs/_audio_quality_probe')
    out_dir.mkdir(parents=True, exist_ok=True)

    # extract features for all four subsets
    feats = {}
    miss_report = {}
    for sub_name, uids, smap in [
        ('pos_n1', subsets['pos_n1'], pos_slices),
        ('pos_frej', subsets['pos_frej'], pos_slices),
        ('neg_n1', subsets['neg_n1'], neg_slices),
        ('neg_n1_rej', subsets['neg_n1_rej'], neg_slices),
    ]:
        print(f'\n[extract] {sub_name} n={len(uids)} ...')
        f, miss = collect(uids, smap)
        feats[sub_name] = f
        miss_report[sub_name] = miss
        ok = sum(1 for v in f.values() if v is not None and '_error' not in v)
        print(f'  ok={ok}  missing={len(miss)}  error={sum(1 for v in f.values() if v and "_error" in v)}')

    # ---- TEST1 general: pos_n1 vs neg_n1 ----
    print('\n' + '=' * 78)
    print('TEST 1: GENERAL  pos_n1 (target mono) vs neg_n1 (interferent mono)')
    print('=' * 78)
    test1 = {}
    for key in SIGNAL_KEYS:
        r = run_signal(key, feats['pos_n1'], feats['neg_n1'], key)
        test1[key] = r
        if 'error' in r:
            print(f'  {key:22s}  ERROR {r["error"]}')
            continue
        pmax = r['pmax_recall_half']
        pmax_s = f"Pmax@R>=0.5 = {pmax['precision']:.3f} (thr={pmax['thr']:.4g}, R={pmax['recall']:.3f}, FP={pmax['fp']}/{r['n_neg']})" if pmax else "Pmax@R>=0.5 = N/A"
        print(f"  {key:22s}  AUC={r['auc_med']:.4f} CI[{r['auc_lo']:.4f},{r['auc_hi']:.4f}]  dir={r['direction']}  {pmax_s}")

    # ---- TEST2 operational: pos_frej vs neg_n1_rej ----
    print('\n' + '=' * 78)
    print('TEST 2: OPERATIONAL  pos_frej (false-rej) vs neg_n1_rej (correctly-rej)')
    print('=' * 78)
    test2 = {}
    for key in SIGNAL_KEYS:
        r = run_signal(key, feats['pos_frej'], feats['neg_n1_rej'], key)
        test2[key] = r
        if 'error' in r:
            print(f'  {key:22s}  ERROR {r["error"]}')
            continue
        pmax = r['pmax_recall_half']
        pmax_s = f"Pmax@R>=0.5 = {pmax['precision']:.3f} (thr={pmax['thr']:.4g}, R={pmax['recall']:.3f}, FP={pmax['fp']}/{r['n_neg']})" if pmax else "Pmax@R>=0.5 = N/A"
        print(f"  {key:22s}  AUC={r['auc_med']:.4f} CI[{r['auc_lo']:.4f},{r['auc_hi']:.4f}]  dir={r['direction']}  {pmax_s}")

    # ---- verdict ----
    BREAKEVEN = 0.763
    test2_pmaxs = []
    for key, r in test2.items():
        if 'error' in r:
            continue
        pmax = r.get('pmax_recall_half')
        if pmax is None:
            continue
        test2_pmaxs.append((key, pmax['precision'], r['auc_med'], r['direction']))
    test2_pmaxs.sort(key=lambda t: t[1], reverse=True)
    best = test2_pmaxs[0] if test2_pmaxs else None
    passes = best and best[1] > BREAKEVEN

    print('\n' + '=' * 78)
    print(f'TEST 2 Pmax ranking (break-even = {BREAKEVEN}):')
    for k, p, a, d in test2_pmaxs:
        flag = '  <-- PASS' if p > BREAKEVEN else ''
        print(f'  {k:22s}  Pmax={p:.4f}  AUC={a:.4f}  dir={d}{flag}')
    if best:
        print(f'\nBEST TEST2: {best[0]}  Pmax={best[1]:.4f}  AUC={best[2]:.4f}  dir={best[3]}')
        print(f'passes_breakeven(>0.763): {bool(passes)}')

    # ---- anomaly observation: pos_frej 是否为 pos_n1 中最安静的弱子集 ----
    # compare rms/centroid/snr distributions of pos_frej vs pos_n1_all
    pos_n1_all_feats = feats['pos_n1']
    pos_frej_feats = feats['pos_frej']
    print('\n[anomaly] pos_frej (147) vs pos_n1 ALL (543) on key signals:')
    for key in ['rms', 'snr_est', 'spectral_centroid', 'silence_ratio', 'high_freq_ratio']:
        a_vals = [v[key] for v in pos_n1_all_feats.values() if v and '_error' not in v and key in v]
        f_vals = [v[key] for v in pos_frej_feats.values() if v and '_error' not in v and key in v]
        if not a_vals or not f_vals:
            continue
        a_med = float(np.median(a_vals)); f_med = float(np.median(f_vals))
        a_mean = float(np.mean(a_vals)); f_mean = float(np.mean(f_vals))
        delta_pct = (f_med - a_med) / (abs(a_med) + 1e-12) * 100
        print(f'  {key:22s}  pos_n1 median={a_med:.4g}  pos_frej median={f_med:.4g}  Δmedian={delta_pct:+.1f}%')

    # ---- save outputs ----
    # full per-uid feature dump
    full_feats = {
        sub: {u: v for u, v in feats[sub].items()}
        for sub in feats
    }
    # summary
    summary = {
        'break_even': BREAKEVEN,
        'group_sizes': {
            'pos_n1': len(subsets['pos_n1']),
            'pos_frej': len(subsets['pos_frej']),
            'neg_n1': len(subsets['neg_n1']),
            'neg_n1_rej': len(subsets['neg_n1_rej']),
        },
        'missing_slices': miss_report,
        'test1_general': {k: {kk: vv for kk, vv in r.items() if kk not in ('pos_uids', 'neg_uids')} for k, r in test1.items()},
        'test2_operational': {k: {kk: vv for kk, vv in r.items() if kk not in ('pos_uids', 'neg_uids')} for k, r in test2.items()},
        'test2_pmax_ranking': [{'signal': k, 'pmax': p, 'auc': a, 'direction': d} for k, p, a, d in test2_pmaxs],
        'best_test2_signal': best[0] if best else None,
        'best_test2_pmax': best[1] if best else None,
        'best_test2_auc': best[2] if best else None,
        'best_test2_direction': best[3] if best else None,
        'passes_breakeven': bool(passes),
        'verdict': (
            f'PASS (worth fusion) — best TEST2 signal {best[0]} Pmax={best[1]:.4f} > 0.763'
            if passes else
            f'NO-GO (audio-quality single-signal) — best TEST2 signal {best[0] if best else "NONE"} '
            f'Pmax={best[1] if best else 0:.4f} <= 0.763. All audio-quality signals fail '
            f'recovery-precision break-even on operational recovery.'
        ),
    }

    with open(out_dir / 'probe.json', 'w', encoding='utf-8') as f:
        json.dump({
            'features_per_uid': {sub: {u: v for u, v in feats[sub].items()}
                                  for sub in feats},
            'signals_tested': SIGNAL_KEYS,
        }, f, ensure_ascii=False, indent=2)

    with open(out_dir / 'auc_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # human-readable auc_report.txt
    lines = []
    lines.append('AUDIO-QUALITY MULTI-DIM REJECT-CALIBRATION PROBE')
    lines.append('=' * 78)
    lines.append('')
    lines.append('Subsets (same as decoder probe):')
    for k, v in summary['group_sizes'].items():
        lines.append(f'  {k:14s}  n={v}')
    lines.append(f'  Missing slices: pos_n1={len(miss_report["pos_n1"])} pos_frej={len(miss_report["pos_frej"])} neg_n1={len(miss_report["neg_n1"])} neg_n1_rej={len(miss_report["neg_n1_rej"])}')
    lines.append('')
    lines.append(f'Break-even recovery-precision = {BREAKEVEN}  (1 pos saved = +0.9/1364; 1 neg leaked = -1/474)')
    lines.append('')
    lines.append('TEST1 general (pos_n1 vs neg_n1):')
    lines.append(f'  {"signal":22s}  {"AUC":>7s}  {"95%CI":>17s}  {"dir":>4s}  {"Pmax":>6s}  {"@thr":>10s}  {"R":>5s}  {"FP":>10s}')
    for k in SIGNAL_KEYS:
        r = test1.get(k, {})
        if 'error' in r:
            lines.append(f'  {k:22s}  ERROR: {r["error"]}')
            continue
        pm = r.get('pmax_recall_half') or {}
        lines.append(f'  {k:22s}  {r["auc_med"]:7.4f}  [{r["auc_lo"]:.4f},{r["auc_hi"]:.4f}]  {r["direction"]:>4s}  {pm.get("precision",float("nan")):6.3f}  {pm.get("thr",float("nan")):10.4g}  {pm.get("recall",float("nan")):5.3f}  {pm.get("fp","?")}/{r["n_neg"]}')
    lines.append('')
    lines.append('TEST2 operational (pos_frej vs neg_n1_rej):')
    lines.append(f'  {"signal":22s}  {"AUC":>7s}  {"95%CI":>17s}  {"dir":>4s}  {"Pmax":>6s}  {"@thr":>10s}  {"R":>5s}  {"FP":>10s}')
    for k in SIGNAL_KEYS:
        r = test2.get(k, {})
        if 'error' in r:
            lines.append(f'  {k:22s}  ERROR: {r["error"]}')
            continue
        pm = r.get('pmax_recall_half') or {}
        flag = '  *PASS' if (pm.get('precision', 0) or 0) > BREAKEVEN else ''
        lines.append(f'  {k:22s}  {r["auc_med"]:7.4f}  [{r["auc_lo"]:.4f},{r["auc_hi"]:.4f}]  {r["direction"]:>4s}  {pm.get("precision",float("nan")):6.3f}  {pm.get("thr",float("nan")):10.4g}  {pm.get("recall",float("nan")):5.3f}  {pm.get("fp","?")}/{r["n_neg"]}{flag}')
    lines.append('')
    lines.append('TEST2 Pmax ranking:')
    for k, p, a, d in test2_pmaxs:
        flag = '  *PASS' if p > BREAKEVEN else ''
        lines.append(f'  {k:22s}  Pmax={p:.4f}  AUC={a:.4f}  dir={d}{flag}')
    lines.append('')
    if best:
        lines.append(f'BEST TEST2 signal: {best[0]}  Pmax={best[1]:.4f}  AUC={best[2]:.4f}  dir={best[3]}')
    lines.append(f'passes_breakeven(>0.763): {bool(passes)}')
    lines.append('')
    lines.append('VERDICT:')
    lines.append(f'  {summary["verdict"]}')
    (out_dir / 'auc_report.txt').write_text('\n'.join(lines), encoding='utf-8')

    print(f'\n[done] wrote {out_dir/"probe.json"}, {out_dir/"auc_summary.json"}, {out_dir/"auc_report.txt"}')
    print(f'\nVERDICT: {summary["verdict"]}')
    return summary


if __name__ == '__main__':
    main()
