"""Speaker-aware GO/NO-GO probe. Compare false-rejected pos vs true neg on speaker signals."""
import json, re
import numpy as np
from collections import Counter

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
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 50)), float(np.percentile(aucs, 97.5))

def thr_pr(sp, sn, direction='gt'):
    """Predict pos if score OP thr. Return list of dicts at meaningful thresholds."""
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

def summarize(name, sp, sn, direction='gt', do_pr=False):
    sp = np.array([x for x in sp if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    sn = np.array([x for x in sn if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    lo, med, hi = bootstrap_auc(sp, sn, B=500, seed=42)
    print(f'\n=== {name} ===')
    print(f'  POS  n={len(sp):4d}  mean={sp.mean():.4f}  median={np.median(sp):.4f}  std={sp.std():.4f}  p10={np.percentile(sp,10):.4f}  p25={np.percentile(sp,25):.4f}  p75={np.percentile(sp,75):.4f}  p90={np.percentile(sp,90):.4f}  min={sp.min():.4f}  max={sp.max():.4f}')
    print(f'  NEG  n={len(sn):4d}  mean={sn.mean():.4f}  median={np.median(sn):.4f}  std={sn.std():.4f}  p10={np.percentile(sn,10):.4f}  p25={np.percentile(sn,25):.4f}  p75={np.percentile(sn,75):.4f}  p90={np.percentile(sn,90):.4f}  min={sn.min():.4f}  max={sn.max():.4f}')
    print(f'  AUC (pos if score {direction} thr)  = {med:.4f}  95% CI [{lo:.4f}, {hi:.4f}]')
    if do_pr:
        pr = thr_pr(sp, sn, direction)
        # pick operating points: max F1, and recall>=0.5 with max precision
        best_f1 = max(pr, key=lambda r: r['f1'])
        rec_05 = [r for r in pr if r['recall'] >= 0.5]
        best_p_at_r05 = max(rec_05, key=lambda r: r['precision']) if rec_05 else None
        print(f'  best-F1 op: thr={best_f1["thr"]:.4f}  P={best_f1["precision"]:.3f}  R={best_f1["recall"]:.3f}  TP={best_f1["tp"]}/{len(sp)}  FP={best_f1["fp"]}/{len(sn)}')
        if best_p_at_r05:
            print(f'  R>=0.5 op:   thr={best_p_at_r05["thr"]:.4f}  P={best_p_at_r05["precision"]:.3f}  R={best_p_at_r05["recall"]:.3f}  TP={best_p_at_r05["tp"]}/{len(sp)}  FP={best_p_at_r05["fp"]}/{len(sn)}')
        # overlap: % of NEG within [p25_pos, p75_pos]
        p25, p75 = np.percentile(sp, 25), np.percentile(sp, 75)
        ov = ((sn >= p25) & (sn <= p75)).mean()
        print(f'  NEG within POS IQR [{p25:.4f}, {p75:.4f}]: {ov*100:.1f}%  (higher = more overlap = worse separation)')
    return lo, med, hi

# ---- Load ----
with open('code/runs/_scene_route_full/per_sample.json') as f:
    pos_ps = json.load(f)
with open('code/runs/full_eval_20260730_pos/_work/enroll_all.json') as f:
    pos_enr = json.load(f)
with open('code/runs/full_eval_20260730_neg/_work/enroll_all.json') as f:
    neg_enr = json.load(f)

def uid_from_rec(rec):
    m = re.search(r'cmd_(\d+)', rec or '')
    return 'cmd_' + m.group(1) if m else None

pos_by_uid = {p['uid']: p for p in pos_ps}
pos_enr_by_uid = {}
for n in pos_enr:
    u = uid_from_rec(n.get('recognition'))
    if u:
        pos_enr_by_uid[u] = n
common = set(pos_by_uid) & set(pos_enr_by_uid)
print('POS per_sample:', len(pos_by_uid), '| POS enroll_all:', len(pos_enr_by_uid), '| common uid:', len(common))

pos_n1, pos_frej = [], []
for u in common:
    ps = pos_by_uid[u]; en = pos_enr_by_uid[u]
    if ps.get('n_spk') != 1:
        continue
    rec = {
        'uid': u,
        'max_sim_en': en.get('max_sim'),
        'stno_target_ratio': en.get('stno_target_ratio'),
        'target_active_ratio': en.get('target_active_ratio'),
        'mainline_cer': ps.get('mainline_cer_transcribe'),
        'rejected_ps': ps.get('rejected_thr0.27'),
        'transcript': en.get('transcript'),
        'ref': ps.get('ref'),
    }
    pos_n1.append(rec)
    if ps.get('rejected_thr0.27') == True and ps.get('mainline_cer_transcribe') < 0.3:
        pos_frej.append(rec)

neg_n1 = []
for n in neg_enr:
    if len(n.get('speakers', [])) == 1:
        neg_n1.append({
            'uid': uid_from_rec(n.get('recognition')),
            'max_sim': n.get('max_sim'),
            'stno_target_ratio': n.get('stno_target_ratio'),
            'target_active_ratio': n.get('target_active_ratio'),
            'rejected': n.get('rejected'),
            'transcript': n.get('transcript'),
        })

print('POS n_spk=1 matched:', len(pos_n1))
print('POS false-rejected (n_spk=1 + rejected@0.27 + mainline CER<0.3):', len(pos_frej))
print('NEG n_spk=1 (interferent mono):', len(neg_n1))
neg_n1_rej = [n for n in neg_n1 if n['rejected'] == True]
neg_n1_notrej = [n for n in neg_n1 if n['rejected'] == False]
print('  NEG n_spk=1 rejected (correctly):', len(neg_n1_rej))
print('  NEG n_spk=1 NOT rejected (leakage):', len(neg_n1_notrej))

print('\n' + '=' * 70)
print('TEST 1: GENERAL discrimination')
print('  POS n_spk=1 ALL target monologue  vs  NEG n_spk=1 ALL interferent mono')
print('=' * 70)
sp_sim = [p['max_sim_en'] for p in pos_n1]
sn_sim = [n['max_sim'] for n in neg_n1]
auc1_sim = summarize('max_sim (whole-seg cosine, ResNet34-256d)', sp_sim, sn_sim, 'gt', do_pr=True)

sp_stno = [p['stno_target_ratio'] for p in pos_n1]
sn_stno = [n['stno_target_ratio'] for n in neg_n1]
auc1_stno = summarize('stno_target_ratio (enrollment-cond Personal-VAD frame target activity)', sp_stno, sn_stno, 'gt', do_pr=True)

sp_ta = [p['target_active_ratio'] for p in pos_n1]
sn_ta = [n['target_active_ratio'] for n in neg_n1]
auc1_ta = summarize('target_active_ratio', sp_ta, sn_ta, 'gt')

print('\n' + '=' * 70)
print('TEST 2: OPERATIONAL recovery')
print('  POS false-rejected 147  vs  NEG correctly-rejected 299')
print('  Q: among samples thr0.27 already rejected, can a 2nd signal recover pos')
print('    without re-accepting neg?')
print('=' * 70)
sp_sim_op = [p['max_sim_en'] for p in pos_frej]
sn_sim_op = [n['max_sim'] for n in neg_n1_rej]
auc2_sim = summarize('max_sim', sp_sim_op, sn_sim_op, 'gt', do_pr=True)

sp_stno_op = [p['stno_target_ratio'] for p in pos_frej]
sn_stno_op = [n['stno_target_ratio'] for n in neg_n1_rej]
auc2_stno = summarize('stno_target_ratio', sp_stno_op, sn_stno_op, 'gt', do_pr=True)

# Sanity: check if stno and target_active are identical
diff = np.mean(np.abs(np.array(sp_stno_all := [p['stno_target_ratio'] for p in pos_n1 if p['stno_target_ratio'] is not None]) -
                      np.array([p['target_active_ratio'] for p in pos_n1 if p['target_active_ratio'] is not None])))
print('\nstno vs target_active mean abs diff (POS n1):', diff)

# Save arrays
out = {
    'group_sizes': {
        'pos_n1_all': len(pos_n1), 'pos_frej': len(pos_frej),
        'neg_n1_all': len(neg_n1), 'neg_n1_rej': len(neg_n1_rej),
        'neg_n1_notrej': len(neg_n1_notrej),
    },
    'test1_general': {
        'max_sim': {'auc_ci': auc1_sim, 'pos': sp_sim, 'neg': sn_sim},
        'stno_target_ratio': {'auc_ci': auc1_stno, 'pos': sp_stno, 'neg': sn_stno},
    },
    'test2_operational': {
        'max_sim': {'auc_ci': auc2_sim, 'pos': sp_sim_op, 'neg': sn_sim_op},
        'stno_target_ratio': {'auc_ci': auc2_stno, 'pos': sp_stno_op, 'neg': sn_stno_op},
    },
    'samples': {
        'pos_frej': pos_frej[:20],
        'neg_n1_notrej': neg_n1_notrej[:20],
    },
}
with open('code/runs/_speaker_aware_probe.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\nSaved -> code/runs/_speaker_aware_probe.json')
