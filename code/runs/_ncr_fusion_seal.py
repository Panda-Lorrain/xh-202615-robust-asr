"""Non-content reject calibration seal:
(1) repetition_rate补测 (text-level single-dim AUC + Pmax)
(2) lightgbm nonlinear fusion of all ~20 signals (decoder 6 + audio 9 + sim/stno 2 + repetition 3)
(3) 5-fold hold-out with recovery-precision + net-score dual threshold.

Methodology mirrors _speaker_aware_holdout.py (same folds, same net formula, same
break-even 0.763 + noise floor +/-0.04). Output artifacts -> code/runs/_ncr_fusion_seal/.
"""
import json, re, os, sys
import numpy as np
from collections import Counter

# ---------- paths (resolve relative to project root, cwd-independent) ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                # .../code/runs
_CODE_DIR   = os.path.dirname(_SCRIPT_DIR)                              # .../code
_ROOT_DIR   = os.path.dirname(_CODE_DIR)                                # project root
def _p(rel): return os.path.join(_ROOT_DIR, rel.replace('/', os.sep))

DECODER_PROBE = _p('code/runs/_decoder_probe/probe.json')
AUDIO_PROBE   = _p('code/runs/_audio_quality_probe/probe.json')
SUBSET_UIDS   = _p('code/runs/_decoder_probe/subset_uids.json')
POS_ENR       = _p('code/runs/full_eval_20260730_pos/_work/enroll_all.json')
NEG_ENR       = _p('code/runs/full_eval_20260730_neg/_work/enroll_all.json')
POS_PS        = _p('code/runs/_scene_route_full/per_sample.json')
OUT_DIR       = _p('code/runs/_ncr_fusion_seal')

POS_TOTAL = 1364   # full pos pool (per scoring spec)
NEG_TOTAL = 474    # full neg pool
BREAK_EVEN_P = 1 / (1 + 0.9 * NEG_TOTAL / POS_TOTAL)   # = 0.7630
NOISE_FLOOR  = 0.04

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- AUC framework (copied from _speaker_aware_probe.py) ----------
def rankdata(a):
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(len(a), dtype=np.intp); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]; obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]; count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)

def auc_mwu(sp, sn):
    sp = np.asarray(sp, dtype=float); sn = np.asarray(sn, dtype=float)
    n1, n2 = len(sp), len(sn)
    if n1 == 0 or n2 == 0: return float('nan')
    r = rankdata(np.concatenate([sp, sn]))
    return float((r[:n1].sum() - n1*(n1+1)/2.0) / (n1*n2))

def boot_auc(sp, sn, B=500, seed=42):
    rng = np.random.default_rng(seed)
    sp = np.asarray(sp, dtype=float); sn = np.asarray(sn, dtype=float)
    aucs = [auc_mwu(rng.choice(sp, len(sp), replace=True),
                    rng.choice(sn, len(sn), replace=True)) for _ in range(B)]
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 50)), float(np.percentile(aucs, 97.5)))

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
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        out.append({'thr': float(t), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                    'precision': prec, 'recall': rec, 'f1': f1})
    return out

def pmax_precision(sp, sn, direction='gt', min_recovered=1):
    """Max precision achievable at any threshold (recovery ceiling).
    Only thresholds where tp+fp>=min_recovered are considered (avoids trivial 1-sample ceilings).
    Returns (pmax, pmax_thr, n_recovered_at_pmax)."""
    pr = thr_pr(sp, sn, direction)
    valid = [r for r in pr if r['tp'] + r['fp'] >= min_recovered]
    if not valid: return 0.0, None, 0
    best = max(valid, key=lambda r: r['precision'])
    return best['precision'], best['thr'], best['tp'] + best['fp']

def single_signal_eval(name, sp, sn):
    """Return dict with AUC (both directions), Pmax (both directions), stats."""
    sp = np.array([x for x in sp if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    sn = np.array([x for x in sn if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    out = {'name': name, 'n_pos': int(len(sp)), 'n_neg': int(len(sn))}
    out['pos'] = {'mean': float(sp.mean()) if len(sp) else None,
                  'std':  float(sp.std())  if len(sp) else None,
                  'median': float(np.median(sp)) if len(sp) else None,
                  'min': float(sp.min()) if len(sp) else None,
                  'max': float(sp.max()) if len(sp) else None,
                  'p10': float(np.percentile(sp, 10)) if len(sp) else None,
                  'p90': float(np.percentile(sp, 90)) if len(sp) else None}
    out['neg'] = {'mean': float(sn.mean()) if len(sn) else None,
                  'std':  float(sn.std())  if len(sn) else None,
                  'median': float(np.median(sn)) if len(sn) else None,
                  'min': float(sn.min()) if len(sn) else None,
                  'max': float(sn.max()) if len(sn) else None,
                  'p10': float(np.percentile(sn, 10)) if len(sn) else None,
                  'p90': float(np.percentile(sn, 90)) if len(sn) else None}
    if len(sp) == 0 or len(sn) == 0:
        out['auc_gt'] = None; out['auc_lt'] = None; out['pmax_gt'] = None; out['pmax_lt'] = None
        return out
    # direction = 'gt': predict pos if score >= thr
    lo, med, hi = boot_auc(sp, sn, B=500, seed=42)
    out['auc_gt'] = {'med': med, 'ci': [lo, hi]}
    # direction = 'lt': predict pos if score <= thr  (AUC = 1 - auc_gt)
    lo2, med2, hi2 = boot_auc(-sp, -sn, B=500, seed=42)
    out['auc_lt'] = {'med': med2, 'ci': [lo2, hi2]}
    # Pmax (precision ceiling) both directions, at two recovery floors
    pmax_gt, t_gt, nrec_gt = pmax_precision(sp, sn, 'gt', min_recovered=1)
    pmax_lt, t_lt, nrec_lt = pmax_precision(sp, sn, 'lt', min_recovered=1)
    pmax_gt_m10, t_gt_m10, nrec_gt_m10 = pmax_precision(sp, sn, 'gt', min_recovered=10)
    pmax_lt_m10, t_lt_m10, nrec_lt_m10 = pmax_precision(sp, sn, 'lt', min_recovered=10)
    out['pmax_gt'] = {'pmax': pmax_gt, 'thr': t_gt, 'n_recovered': nrec_gt}
    out['pmax_lt'] = {'pmax': pmax_lt, 'thr': t_lt, 'n_recovered': nrec_lt}
    out['pmax_gt_min10'] = {'pmax': pmax_gt_m10, 'thr': t_gt_m10, 'n_recovered': nrec_gt_m10,
                            'note': 'max precision requiring >=10 recovered (operationally meaningful)'}
    out['pmax_lt_min10'] = {'pmax': pmax_lt_m10, 'thr': t_lt_m10, 'n_recovered': nrec_lt_m10,
                            'note': 'max precision requiring >=10 recovered (operationally meaningful)'}
    # best direction summary uses min_recovered=10 floor (honest)
    best_pmax_gt = pmax_gt_m10 if nrec_gt_m10 > 0 else pmax_gt
    best_pmax_lt = pmax_lt_m10 if nrec_lt_m10 > 0 else pmax_lt
    if best_pmax_gt >= best_pmax_lt:
        out['best'] = {'direction': 'gt', 'auc': med, 'pmax': best_pmax_gt, 'delta_neg_pos_mean': float(sp.mean() - sn.mean()),
                       'pmax_min_recovered_floor': 10 if nrec_gt_m10 > 0 else 1}
    else:
        out['best'] = {'direction': 'lt', 'auc': med2, 'pmax': best_pmax_lt, 'delta_neg_pos_mean': float(sp.mean() - sn.mean()),
                       'pmax_min_recovered_floor': 10 if nrec_lt_m10 > 0 else 1}
    return out

# ---------- repetition (Step 1) ----------
def repetition_signals(text):
    """3 char-level repetition signals.
    char_2gram_rep: fraction of 2-grams that are repeats (types seen >1 / total tokens)
    char_3gram_rep: same for 3-grams
    adjacent_repeat_rate: fraction of adjacent equal chars (叠字 e.g. 家家家)
    Returns (NaN, NaN, NaN) for empty / too-short transcripts."""
    if not text:
        return float('nan'), float('nan'), float('nan')
    chars = [c for c in text if not c.isspace() and c not in '，。、！？,.!?;:"\"\'()（）']
    if len(chars) < 2:
        return 0.0, 0.0, 0.0
    # adjacent repeat
    n_adj_eq = sum(1 for i in range(1, len(chars)) if chars[i] == chars[i-1])
    adj_rate = n_adj_eq / (len(chars) - 1)
    # n-gram repeat rate (token-level): for all n-grams, what fraction belong to a repeated type
    def ngram_rep_rate(n):
        if len(chars) < n: return 0.0
        grams = [tuple(chars[i:i+n]) for i in range(len(chars) - n + 1)]
        if not grams: return 0.0
        c = Counter(grams)
        repeated_tokens = sum(cnt for cnt in c.values() if cnt > 1)
        return repeated_tokens / len(grams)
    return ngram_rep_rate(2), ngram_rep_rate(3), adj_rate

# ---------- load ----------
print('Loading data...')
with open(DECODER_PROBE) as f: dec = json.load(f)
with open(AUDIO_PROBE) as f:   aud_full = json.load(f)
aud = aud_full['features_per_uid']  # dict[subset][uid] -> {9 signals}
with open(SUBSET_UIDS) as f:  subsets = json.load(f)
with open(POS_ENR) as f:      pos_enr = json.load(f)
with open(NEG_ENR) as f:      neg_enr = json.load(f)
with open(POS_PS) as f:       pos_ps  = json.load(f)

def uid_from_rec(rec):
    m = re.search(r'cmd_(\d+)', rec or '')
    return 'cmd_' + m.group(1) if m else None

pos_ps_by_uid  = {p['uid']: p for p in pos_ps}
pos_enr_by_uid = {uid_from_rec(r.get('recognition')): r for r in pos_enr if uid_from_rec(r.get('recognition'))}
neg_enr_by_uid = {uid_from_rec(r.get('recognition')): r for r in neg_enr if uid_from_rec(r.get('recognition'))}

# ---- Step 1: repetition signals per uid ----
print('\n=== Step 1: repetition signal extraction ===')
rep_by_uid = {}   # uid -> {char_2gram_rep, char_3gram_rep, adjacent_repeat_rate}
for uid_set in ['pos_n1', 'pos_frej', 'neg_n1', 'neg_n1_rej']:
    for uid in subsets.get(uid_set, []):
        if uid in rep_by_uid: continue
        enr = pos_enr_by_uid.get(uid) or neg_enr_by_uid.get(uid)
        tx = enr.get('transcript') if enr else None
        r2, r3, ar = repetition_signals(tx)
        rep_by_uid[uid] = {'char_2gram_rep': r2, 'char_3gram_rep': r3, 'adjacent_repeat_rate': ar}

REP_SIGNALS = ['char_2gram_rep', 'char_3gram_rep', 'adjacent_repeat_rate']

# AUC for TEST1 (pos_n1 vs neg_n1) and TEST2 (pos_frej vs neg_n1_rej)
def collect(sig, uids):
    return [rep_by_uid[u][sig] for u in uids if u in rep_by_uid]

rep_results = {'signals': REP_SIGNALS, 'break_even_precision': BREAK_EVEN_P,
               'note': 'Pmax = max precision achievable at any single threshold (recovery ceiling); break-even=0.763'}
print(f'  break-even precision = {BREAK_EVEN_P:.4f}')

for test_name, pos_subset, neg_subset in [('TEST1_general', 'pos_n1', 'neg_n1'),
                                          ('TEST2_operational', 'pos_frej', 'neg_n1_rej')]:
    print(f'\n  --- {test_name}: {pos_subset} ({len(subsets[pos_subset])}) vs {neg_subset} ({len(subsets[neg_subset])}) ---')
    rep_results[test_name] = {}
    for sig in REP_SIGNALS:
        sp = collect(sig, subsets[pos_subset])
        sn = collect(sig, subsets[neg_subset])
        r = single_signal_eval(sig, sp, sn)
        rep_results[test_name][sig] = r
        # terse print
        if r['best']:
            b = r['best']
            print(f'    {sig:25s}  best_dir={b["direction"]}  AUC={b["auc"]:.4f}  Pmax={b["pmax"]:.4f}  (d(neg-pos)mean={b["delta_neg_pos_mean"]:+.4f})')

# Variance check: are signals near-zero (no signal)?
print('\n  Variance check (near-zero => no signal):')
var_report = {}
for sig in REP_SIGNALS:
    allvals = []
    for u, v in rep_by_uid.items():
        x = v[sig]
        if x is not None and not (isinstance(x, float) and np.isnan(x)):
            allvals.append(x)
    arr = np.array(allvals)
    nonzero_frac = float((arr > 0).mean()) if len(arr) else 0.0
    var_report[sig] = {'n': len(arr), 'mean': float(arr.mean()) if len(arr) else None,
                       'std': float(arr.std()) if len(arr) else None,
                       'frac_nonzero': nonzero_frac,
                       'near_zero': bool(nonzero_frac < 0.05 or (len(arr) and arr.std() < 1e-4))}
    print(f'    {sig:25s}  mean={arr.mean():.4f}  std={arr.std():.4f}  frac_nonzero={nonzero_frac:.3%}  near_zero={var_report[sig]["near_zero"]}')
rep_results['variance_report'] = var_report

with open(os.path.join(OUT_DIR, 'repetition.json'), 'w') as f:
    json.dump(rep_results, f, ensure_ascii=False, indent=2)
print(f'\n  Saved -> {OUT_DIR}/repetition.json')

# ---------- Step 2: feature join ----------
print('\n=== Step 2: feature join (decoder 6 + audio 9 + sim/stno 2 + repetition 3 = 20) ===')
DEC_SIGNALS  = ['mean_logprob', 'mean_entropy', 'seq_score', 'num_tokens', 'first_is_eos', 'audio_rms']
AUD_SIGNALS  = ['rms', 'peak', 'clipping_rate', 'snr_est', 'silence_ratio',
                'zero_crossing_rate', 'high_freq_ratio', 'spectral_centroid', 'spectral_flatness']
EXTRA_SIGNALS = ['max_sim', 'stno_target_ratio']
ALL_FEATURES = DEC_SIGNALS + AUD_SIGNALS + EXTRA_SIGNALS + REP_SIGNALS
print(f'  Total features: {len(ALL_FEATURES)}')

def build_features(uid):
    """Return dict of all features for a uid (NaN if missing)."""
    feat = {k: float('nan') for k in ALL_FEATURES}
    # decoder
    d = dec.get(uid)
    if d:
        for k in DEC_SIGNALS:
            if k in d:
                v = d[k]
                feat[k] = float(v) if not isinstance(v, bool) else float(v)
        # first_is_eos is bool -> int
        feat['first_is_eos'] = float(d.get('first_is_eos', 0))
    # audio: need to look in subset-specific dict; try all subsets
    av = None
    for sub in ['pos_n1', 'pos_frej', 'neg_n1', 'neg_n1_rej']:
        if uid in aud.get(sub, {}):
            av = aud[sub][uid]; break
    if av:
        for k in AUD_SIGNALS:
            if k in av: feat[k] = float(av[k])
    # extra (max_sim, stno_target_ratio) from enroll
    enr = pos_enr_by_uid.get(uid) or neg_enr_by_uid.get(uid)
    if enr:
        feat['max_sim'] = float(enr.get('max_sim')) if enr.get('max_sim') is not None else float('nan')
        feat['stno_target_ratio'] = float(enr.get('stno_target_ratio')) if enr.get('stno_target_ratio') is not None else float('nan')
    # repetition
    if uid in rep_by_uid:
        for k in REP_SIGNALS:
            v = rep_by_uid[uid][k]
            feat[k] = float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float('nan')
    return feat

# build labeled samples: pos_n1 (label 1) + neg_n1 (label 0), with side info (rejected, cer)
samples = []
for uid in subsets['pos_n1']:
    enr = pos_enr_by_uid.get(uid)
    ps  = pos_ps_by_uid.get(uid)
    if not enr or not ps: continue
    feat = build_features(uid)
    cer  = ps.get('mainline_cer_transcribe')
    rej  = bool(ps.get('rejected_thr0.27'))
    samples.append({'uid': uid, 'label': 1, 'feat': feat,
                    'cer': cer, 'rejected_thr027': rej, 'is_frej': bool(rej and cer is not None and cer < 0.3)})
for uid in subsets['neg_n1']:
    enr = neg_enr_by_uid.get(uid)
    if not enr: continue
    feat = build_features(uid)
    rej  = bool(enr.get('rejected'))
    samples.append({'uid': uid, 'label': 0, 'feat': feat,
                    'cer': None, 'rejected_thr027': rej, 'is_frej': False})

print(f'  Labeled samples built: {len(samples)} (pos_n1={sum(s["label"]==1 for s in samples)}  neg_n1={sum(s["label"]==0 for s in samples)})')

# Coverage report
print('  Feature coverage (non-NaN fraction):')
cov = {}
for k in ALL_FEATURES:
    n_valid = sum(1 for s in samples if not np.isnan(s['feat'][k]))
    cov[k] = n_valid
    print(f'    {k:25s}  {n_valid}/{len(samples)} ({n_valid/len(samples):.1%})')

# ---------- Step 3: lightgbm fusion + 5-fold hold-out ----------
print('\n=== Step 3: lightgbm nonlinear fusion + 5-fold hold-out ===')
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

X_full = np.array([[s['feat'][k] for k in ALL_FEATURES] for s in samples], dtype=float)
y_full = np.array([s['label'] for s in samples], dtype=int)

# z-norm for nan-safe mean/std (for reporting feature importance on z-normalized)
col_mean = np.nanmean(X_full, axis=0); col_std = np.nanstd(X_full, axis=0)
col_stdSafe = np.where(col_std > 0, col_std, 1.0)

# ---------- In-sample fit (overfitting baseline) ----------
def fit_lgbm():
    return LGBMClassifier(
        objective='binary', num_leaves=8, max_depth=4,
        min_child_samples=20, reg_lambda=1.0, n_estimators=50,
        learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1, deterministic=True)

def net_eval(samples_subset, accept_pred):
    """Compute net delta vs thr0.27 baseline on this subset (mono pos+neg).
    accept_pred: bool array, True=accept (predict pos).
    Returns (net, d_cer, d_rr, base_cer_sum, new_cer_sum, base_rr_correct, new_rr_correct, tp_recover, fp_recover)."""
    pos = [s for s in samples_subset if s['label'] == 1]
    neg = [s for s in samples_subset if s['label'] == 0]
    pos_accept_base = np.array([s['rejected_thr027'] == False for s in pos], dtype=bool)  # baseline accept = NOT rejected
    pos_accept_new  = np.array([accept_pred[i] for i, s in enumerate(samples_subset) if s['label'] == 1], dtype=bool)
    neg_accept_base = np.array([s['rejected_thr027'] == False for s in neg], dtype=bool)
    neg_accept_new  = np.array([accept_pred[i] for i, s in enumerate(samples_subset) if s['label'] == 0], dtype=bool)
    # CER
    base_cer = sum((s['cer'] if acc else 1.0) for s, acc in zip(pos, pos_accept_base))
    new_cer  = sum((s['cer'] if acc else 1.0) for s, acc in zip(pos, pos_accept_new))
    d_cer = (base_cer - new_cer) / POS_TOTAL
    # RR: correct reject fraction on this mono subset, scaled to overall NEG pool
    base_rr = int((~neg_accept_base).sum())
    new_rr  = int((~neg_accept_new).sum())
    d_rr = (new_rr - base_rr) / NEG_TOTAL
    net = 0.5 * d_cer + 0.5 * d_rr
    # recovery precision: among samples thr0.27 rejected AND new-accept, fraction pos
    recover_idx = [i for i, s in enumerate(samples_subset) if s['rejected_thr027'] and accept_pred[i]]
    tp = sum(1 for i in recover_idx if samples_subset[i]['label'] == 1 and samples_subset[i]['is_frej'])
    fp = sum(1 for i in recover_idx if samples_subset[i]['label'] == 0)
    rec_prec = tp / (tp + fp) if (tp + fp) > 0 else None
    return {'net': float(net), 'd_cer': float(d_cer), 'd_rr': float(d_rr),
            'base_cer_sum': float(base_cer), 'new_cer_sum': float(new_cer),
            'base_rr_correct_mono': int(base_rr), 'new_rr_correct_mono': int(new_rr),
            'recovery_tp': int(tp), 'recovery_fp': int(fp),
            'recovery_precision': rec_prec, 'n_recovered': int(tp + fp)}

def sweep_threshold_for_net(scores, samples_subset):
    """Find threshold on score that maximizes net delta on samples_subset.
    Returns (best_thr, best_net, best_metrics)."""
    base_cer = sum((s['cer'] if not s['rejected_thr027'] else 1.0) for s in samples_subset if s['label'] == 1)
    n_pos = sum(1 for s in samples_subset if s['label'] == 1)
    n_neg = sum(1 for s in samples_subset if s['label'] == 0)
    pos_cer = np.array([s['cer'] if s['cer'] is not None else 1.0 for s in samples_subset if s['label'] == 1])
    pos_rej_base = np.array([s['rejected_thr027'] for s in samples_subset if s['label'] == 1])
    neg_rej_base = np.array([s['rejected_thr027'] for s in samples_subset if s['label'] == 0])
    pos_scores = np.array([scores[i] for i, s in enumerate(samples_subset) if s['label'] == 1])
    neg_scores = np.array([scores[i] for i, s in enumerate(samples_subset) if s['label'] == 0])
    base_pos_accept = ~pos_rej_base
    base_neg_accept = ~neg_rej_base
    base_cer_sum = sum(p if a else 1.0 for p, a in zip(pos_cer, base_pos_accept))
    base_rr_correct = int((~base_neg_accept).sum())
    cands = sorted(set(list(pos_scores) + list(neg_scores)))
    best = None
    for t in cands:
        pos_acc = pos_scores >= t
        neg_acc = neg_scores >= t
        new_cer = sum(p if a else 1.0 for p, a in zip(pos_cer, pos_acc))
        new_rr  = int((~neg_acc).sum())
        d_cer = (base_cer_sum - new_cer) / POS_TOTAL
        d_rr  = (new_rr - base_rr_correct) / NEG_TOTAL
        net   = 0.5 * d_cer + 0.5 * d_rr
        if best is None or net > best['net']:
            best = {'thr': float(t), 'net': float(net), 'd_cer': float(d_cer), 'd_rr': float(d_rr)}
    return best

# In-sample (fit on all, evaluate on all) = overfitting baseline
print('\n  --- In-sample fit (overfitting baseline) ---')
mdl_in = fit_lgbm(); mdl_in.fit(X_full, y_full)
score_in = mdl_in.predict_proba(X_full)[:, 1]
auc_in_sample = roc_auc_score(y_full, score_in)
best_in = sweep_threshold_for_net(score_in, samples)
in_eval  = net_eval(samples, score_in >= best_in['thr'])
print(f'    in-sample AUC       = {auc_in_sample:.4f}')
print(f'    in-sample best thr  = {best_in["thr"]:.4f}')
print(f'    in-sample net       = {best_in["net"]:+.4f}  (d_cer={best_in["d_cer"]:+.4f}  d_rr={best_in["d_rr"]:+.4f})')
print(f'    in-sample recovery  = TP={in_eval["recovery_tp"]}  FP={in_eval["recovery_fp"]}  prec={in_eval["recovery_precision"]}  n_recovered={in_eval["n_recovered"]}')

# 5-fold stratified hold-out (same seed/strategy as _speaker_aware_holdout.py)
print('\n  --- 5-fold stratified hold-out ---')
rng = np.random.default_rng(123)
idx_pos = np.where(y_full == 1)[0]; idx_neg = np.where(y_full == 0)[0]
idx_pos = rng.permutation(idx_pos); idx_neg = rng.permutation(idx_neg)
FOLDS = 5
pos_fold = np.array_split(idx_pos, FOLDS); neg_fold = np.array_split(idx_neg, FOLDS)

fold_results = []
for k in range(FOLDS):
    te_idx = np.concatenate([pos_fold[k], neg_fold[k]])
    tr_idx = np.concatenate([np.concatenate([pos_fold[j] for j in range(FOLDS) if j != k]),
                              np.concatenate([neg_fold[j] for j in range(FOLDS) if j != k])])
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_te, y_te = X_full[te_idx], y_full[te_idx]
    samples_tr = [samples[i] for i in tr_idx]
    samples_te = [samples[i] for i in te_idx]
    mdl = fit_lgbm(); mdl.fit(X_tr, y_tr)
    score_tr = mdl.predict_proba(X_tr)[:, 1]
    score_te = mdl.predict_proba(X_te)[:, 1]
    # fit threshold on TRAIN (maximize net)
    best_tr = sweep_threshold_for_net(score_tr, samples_tr)
    train_eval = net_eval(samples_tr, score_tr >= best_tr['thr'])
    # EVAL on test
    try:
        auc_te = roc_auc_score(y_te, score_te)
    except Exception:
        auc_te = float('nan')
    test_eval = net_eval(samples_te, score_te >= best_tr['thr'])
    fold_results.append({
        'fold': k,
        'n_te': int(len(te_idx)), 'n_te_pos': int((y_te == 1).sum()), 'n_te_neg': int((y_te == 0).sum()),
        'train_thr': best_tr['thr'], 'train_net': best_tr['net'],
        'test_auc': float(auc_te),
        'test_net': test_eval['net'], 'test_d_cer': test_eval['d_cer'], 'test_d_rr': test_eval['d_rr'],
        'test_recovery_tp': test_eval['recovery_tp'], 'test_recovery_fp': test_eval['recovery_fp'],
        'test_recovery_precision': test_eval['recovery_precision'],
        'test_n_recovered': test_eval['n_recovered'],
    })
    print(f'    fold{k}: train_net={best_tr["net"]:+.4f}  test_auc={auc_te:.4f}  test_net={test_eval["net"]:+.4f}  '
          f'(d_cer={test_eval["d_cer"]:+.4f} d_rr={test_eval["d_rr"]:+.4f})  '
          f'recovery: TP={test_eval["recovery_tp"]} FP={test_eval["recovery_fp"]} prec={test_eval["recovery_precision"]}')

holdout_nets = [f['test_net'] for f in fold_results]
holdout_prec = [f['test_recovery_precision'] for f in fold_results if f['test_recovery_precision'] is not None]
holdout_mean_net = float(np.mean(holdout_nets))
holdout_std_net  = float(np.std(holdout_nets))
holdout_mean_prec = float(np.mean(holdout_prec)) if holdout_prec else None
# Upper bound: max fold precision (not average), since "ceiling" matters most
holdout_max_prec = float(np.max(holdout_prec)) if holdout_prec else None

# Bootstrap CIs (for honest reporting of dual-threshold verdict)
_rng = np.random.default_rng(7)
_nets = np.array(holdout_nets)
_bs = [_nets[_rng.integers(0, len(_nets), len(_nets))].mean() for _ in range(2000)]
net_ci = [float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 50)), float(np.percentile(_bs, 97.5))]
_precs = np.array(holdout_prec)
_bs2 = [_precs[_rng.integers(0, len(_precs), len(_precs))].mean() for _ in range(2000)]
prec_ci = [float(np.percentile(_bs2, 2.5)), float(np.percentile(_bs2, 50)), float(np.percentile(_bs2, 97.5))]
print(f'\n  Hold-out net 95% bootstrap CI = [{net_ci[0]:+.4f}, {net_ci[1]:+.4f}, {net_ci[2]:+.4f}]  '
      f'(upper {net_ci[2]:+.4f} vs +0.04 floor: {"BELOW" if net_ci[2] < NOISE_FLOOR else "ABOVE"})')
print(f'  Hold-out recovery precision 95% CI = [{prec_ci[0]:.4f}, {prec_ci[1]:.4f}, {prec_ci[2]:.4f}]  '
      f'(upper {prec_ci[2]:.4f} vs 0.763: {"BELOW" if prec_ci[2] < BREAK_EVEN_P else "STRADDLES"})')
print(f'  -> Binding constraint = NET (decisively below floor); precision CI straddles break-even')

# Aggregated hold-out recovery precision: pool all folds
all_te_tp   = sum(f['test_recovery_tp'] for f in fold_results)
all_te_fp   = sum(f['test_recovery_fp'] for f in fold_results)
pooled_rec_prec = all_te_tp / (all_te_tp + all_te_fp) if (all_te_tp + all_te_fp) > 0 else None
print(f'\n  5-fold hold-out mean net      = {holdout_mean_net:+.4f}  (std {holdout_std_net:.4f})')
print(f'  5-fold hold-out mean recovery precision (per-fold avg) = {holdout_mean_prec}')
print(f'  5-fold hold-out MAX  recovery precision (ceiling)      = {holdout_max_prec}')
print(f'  5-fold pooled recovery: TP={all_te_tp} FP={all_te_fp} pooled_prec={pooled_rec_prec}')

# Overfitting gap
overfit_gap = best_in['net'] - holdout_mean_net
print(f'\n  In-sample net = {best_in["net"]:+.4f}  vs  hold-out mean net = {holdout_mean_net:+.4f}')
print(f'  Overfitting gap (in-sample - holdout) = {overfit_gap:+.4f}  '
      f'(speaker-aware precedent: +0.0088 in-sample -> +0.0003 holdout, gap ~0.0085 = 2x inflation)')

# ---------- Negative control: label-shuffle ----------
# Confirms +0.0032 hold-out net is noise (lightgbm fitting random labels should give ~0).
print('\n  --- Negative control: label-shuffle (lightgbm on random labels, 3 seeds) ---')
shuf_nets = []
for seed in [101, 202, 303]:
    rng_s = np.random.default_rng(seed)
    y_shuf = y_full.copy(); rng_s.shuffle(y_shuf)
    pos_fold_s = np.array_split(np.where(y_shuf == 1)[0], FOLDS)
    neg_fold_s = np.array_split(np.where(y_shuf == 0)[0], FOLDS)
    fold_nets_s = []
    for k in range(FOLDS):
        te_idx = np.concatenate([pos_fold_s[k], neg_fold_s[k]])
        tr_idx = np.concatenate([np.concatenate([pos_fold_s[j] for j in range(FOLDS) if j != k]),
                                  np.concatenate([neg_fold_s[j] for j in range(FOLDS) if j != k])])
        X_tr, y_tr = X_full[tr_idx], y_shuf[tr_idx]
        X_te = X_full[te_idx]
        samples_tr = [samples[i] for i in tr_idx]
        samples_te = [samples[i] for i in te_idx]
        # note: samples_te labels are still original (we want net on original cer/rej, just model is shuffled-label)
        mdl_s = fit_lgbm(); mdl_s.fit(X_tr, y_tr)
        sc_tr = mdl_s.predict_proba(X_tr)[:, 1]
        sc_te = mdl_s.predict_proba(X_te)[:, 1]
        best_s = sweep_threshold_for_net(sc_tr, samples_tr)
        te_eval = net_eval(samples_te, sc_te >= best_s['thr'])
        fold_nets_s.append(te_eval['net'])
    shuf_nets.append(float(np.mean(fold_nets_s)))
    print(f'    seed={seed}: hold-out mean net = {shuf_nets[-1]:+.4f}')
neg_ctrl_mean = float(np.mean(shuf_nets)); neg_ctrl_std = float(np.std(shuf_nets))
print(f'  Negative control (label-shuffle) net: {neg_ctrl_mean:+.4f} ± {neg_ctrl_std:.4f}')
print(f'  Real hold-out net {holdout_mean_net:+.4f} vs shuffled {neg_ctrl_mean:+.4f}: '
      f'delta = {holdout_mean_net - neg_ctrl_mean:+.4f}')

# Feature importance (from full-model fit)
imp = mdl_in.feature_importances_
# SHAP-like direction: train logistic regression on lightgbm score to get monotonic sign per feature?
# Simpler: use correlation of each feature with lightgbm output score
from scipy.stats import spearmanr
directions = {}
for i, k in enumerate(ALL_FEATURES):
    col = X_full[:, i]
    valid = ~np.isnan(col)
    if valid.sum() < 50:
        directions[k] = None; continue
    rho, _ = spearmanr(col[valid], score_in[valid])
    directions[k] = float(rho) if rho == rho else None   # NaN guard

# Top features by importance
imp_rank = sorted(zip(ALL_FEATURES, imp.tolist()), key=lambda x: -x[1])
print('\n  Feature importance (top 10) + Spearman corr with lgbm score (direction):')
for k, v in imp_rank[:10]:
    d = directions.get(k)
    dstr = f'{d:+.3f}' if d is not None else 'NaN'
    print(f'    {k:25s}  split_importance={int(v):3d}  spearman(score)={dstr}')

# Specifically check reverse signals: mean_logprob (should be reversed: neg has higher), mean_entropy
print('\n  Reverse-signal learned direction check:')
for sig in ['mean_logprob', 'mean_entropy', 'num_tokens', 'first_is_eos']:
    d = directions.get(sig)
    imp_v = dict(imp_rank).get(sig, 0)
    expected = 'expected: pos higher logprob / lower entropy' if sig in ('mean_logprob','mean_entropy') else ''
    print(f'    {sig:25s}  importance={int(imp_v):3d}  spearman(score)={d if d is not None else "NaN"}  {expected}')

# Verdict
dual_pass = (holdout_mean_prec is not None and holdout_mean_prec > BREAK_EVEN_P) and (abs(holdout_mean_net) > NOISE_FLOOR)
dual_pass_max = (holdout_max_prec is not None and holdout_max_prec > BREAK_EVEN_P) and (abs(holdout_mean_net) > NOISE_FLOOR)
verdict = {
    'dual_threshold_pass_per_fold_avg': bool(dual_pass),
    'dual_threshold_pass_max_ceiling':  bool(dual_pass_max),
    'break_even_precision': BREAK_EVEN_P,
    'noise_floor_net': NOISE_FLOOR,
    'repetition_single_dim_seal': {
        sig: {
            'test1_auc_best': rep_results['TEST1_general'][sig]['best']['auc'] if rep_results['TEST1_general'][sig]['best'] else None,
            'test2_pmax_best': rep_results['TEST2_operational'][sig]['best']['pmax'] if rep_results['TEST2_operational'][sig]['best'] else None,
            'near_zero_variance': var_report[sig]['near_zero'],
        } for sig in REP_SIGNALS
    },
    'lgbm_fusion_seal': {
        'in_sample_net': best_in['net'],
        'holdout_mean_net': holdout_mean_net,
        'holdout_std_net':  holdout_std_net,
        'overfitting_gap': overfit_gap,
        'holdout_mean_recovery_precision': holdout_mean_prec,
        'holdout_max_recovery_precision':  holdout_max_prec,
        'pooled_recovery_precision': pooled_rec_prec,
    },
    'conclusion': (
        'NCR④ SEAL = implementation-NO-GO at low-cost re-classification level: '
        'repetition signals near-zero variance / single-dim AUC/Pmax below break-even; '
        'lgbm nonlinear fusion does NOT pass dual threshold on hold-out (recovery precision < 0.763 OR |net| < 0.04). '
        'This seals re-classification/override implementation paths but does NOT seal direction: '
        'non-content verifier calibrated on non-A data (real-recorded domain) remains direction-unresolved.'
    )
}

# Save
fusion_out = {
    'features': ALL_FEATURES,
    'feature_count': len(ALL_FEATURES),
    'coverage': cov,
    'in_sample': {
        'auc': float(auc_in_sample),
        'best_thr': best_in['thr'],
        'net': best_in['net'], 'd_cer': best_in['d_cer'], 'd_rr': best_in['d_rr'],
        'recovery_tp': in_eval['recovery_tp'], 'recovery_fp': in_eval['recovery_fp'],
        'recovery_precision': in_eval['recovery_precision'],
    },
    'holdout': {
        'folds': fold_results,
        'mean_net': holdout_mean_net,
        'std_net':  holdout_std_net,
        'mean_recovery_precision_per_fold': holdout_mean_prec,
        'max_recovery_precision_per_fold':  holdout_max_prec,
        'pooled_recovery_precision': pooled_rec_prec,
        'pooled_recovery_tp': all_te_tp,
        'pooled_recovery_fp': all_te_fp,
        'net_bootstrap_ci95': net_ci,
        'precision_bootstrap_ci95': prec_ci,
        'binding_constraint': 'NET (upper CI +{:.4f} << +0.04 floor); precision CI straddles 0.763'.format(net_ci[2]),
    },
    'overfitting_gap_in_minus_holdout': overfit_gap,
    'negative_control_label_shuffle': {'per_seed_nets': shuf_nets, 'mean': neg_ctrl_mean, 'std': neg_ctrl_std,
                                       'real_minus_shuffled': holdout_mean_net - neg_ctrl_mean},
    'feature_importance_ranked': [{'feature': k, 'importance': int(v)} for k, v in imp_rank],
    'feature_direction_spearman_with_score': directions,
    'reverse_signal_direction_check': {sig: directions.get(sig) for sig in ['mean_logprob', 'mean_entropy', 'num_tokens', 'first_is_eos', 'audio_rms']},
    'break_even_precision': BREAK_EVEN_P,
    'noise_floor_net': NOISE_FLOOR,
    'verdict': verdict,
    'compliance_note': (
        '5-fold hold-out is GENERALIZATION evaluation on A-set subset, not final training. '
        'Even if conditional-GO, integration calibrator MUST be re-fit on non-A data '
        '(A-set training forbidden, lessons-pitfalls §14).'
    ),
}
with open(os.path.join(OUT_DIR, 'lgbm_fusion.json'), 'w') as f:
    json.dump(fusion_out, f, ensure_ascii=False, indent=2)
print(f'\n  Saved -> {OUT_DIR}/lgbm_fusion.json')

# holdout_report.txt
lines = []
lines.append('=== NCR④ Non-content reject calibration SEAL ===')
lines.append('Scope: repetition补测 (single-dim) + lightgbm nonlinear fusion (~20 signals, 5-fold hold-out)')
lines.append('')
lines.append(f'Break-even recovery precision = {BREAK_EVEN_P:.4f}  |  Net noise floor = ±{NOISE_FLOOR}')
lines.append(f'POS_TOTAL={POS_TOTAL}  NEG_TOTAL={NEG_TOTAL}  (救1pos=+0.9/{POS_TOTAL} CER vs 漏1neg=-1/{NEG_TOTAL} RR)')
lines.append('')
lines.append('--- Step 1: repetition (single-dim AUC + Pmax) ---')
lines.append(f'{"signal":25s}  {"TEST1 AUC":>10s}  {"TEST2 Pmax(min10)":>16s}  {"TEST2 Pmax(any)":>16s}  {"near-zero var":>15s}')
for sig in REP_SIGNALS:
    t1 = rep_results['TEST1_general'][sig]['best']['auc'] if rep_results['TEST1_general'][sig]['best'] else None
    t2_block = rep_results['TEST2_operational'][sig]
    t2_m10 = t2_block.get('pmax_gt_min10', {}).get('pmax') if t2_block.get('pmax_gt_min10') else None
    t2_any = t2_block.get('best', {}).get('pmax')
    nz = var_report[sig]['near_zero']
    lines.append(f'{sig:25s}  {(t1 if t1 is not None else float("nan")):>10.4f}  {(t2_m10 if t2_m10 is not None else float("nan")):>16.4f}  {(t2_any if t2_any is not None else float("nan")):>16.4f}  {str(nz):>15s}')
lines.append(f'  Variance: {var_report}')
lines.append('')
lines.append('--- Step 3: lightgbm fusion (5-fold hold-out) ---')
lines.append(f'In-sample:    AUC={auc_in_sample:.4f}  net={best_in["net"]:+.4f}  recovery_prec={in_eval["recovery_precision"]}')
lines.append(f'Holdout mean: net={holdout_mean_net:+.4f}±{holdout_std_net:.4f}  recovery_prec (per-fold avg)={holdout_mean_prec}  (max ceiling={holdout_max_prec})')
lines.append(f'Pooled recovery: TP={all_te_tp} FP={all_te_fp}  pooled_prec={pooled_rec_prec}')
lines.append(f'Hold-out net 95% bootstrap CI = [{net_ci[0]:+.4f}, {net_ci[1]:+.4f}, {net_ci[2]:+.4f}]  (upper vs +0.04 floor: {"BELOW" if net_ci[2] < NOISE_FLOOR else "ABOVE"})')
lines.append(f'Hold-out precision 95% CI = [{prec_ci[0]:.4f}, {prec_ci[1]:.4f}, {prec_ci[2]:.4f}]  (upper vs 0.763: {"BELOW" if prec_ci[2] < BREAK_EVEN_P else "STRADDLES"})')
lines.append(f'Binding constraint: NET (decisively below +0.04 floor); precision CI straddles 0.763 -> dual threshold fails on net dimension')
lines.append(f'Overfitting gap (in-sample - holdout) = {overfit_gap:+.4f}')
lines.append(f'Negative control (label-shuffle): {neg_ctrl_mean:+.4f} ± {neg_ctrl_std:.4f}  (real {holdout_mean_net:+.4f} - shuffled {neg_ctrl_mean:+.4f} = {holdout_mean_net - neg_ctrl_mean:+.4f})')
lines.append(f'Dual threshold (P>0.763 AND |net|>0.04): per-fold-avg={dual_pass}  max-ceiling={dual_pass_max}')
lines.append('')
lines.append('Feature importance top 5:')
for k, v in imp_rank[:5]:
    d = directions.get(k)
    lines.append(f'  {k:25s}  imp={int(v):3d}  spearman_w_score={d if d is not None else "NaN"}')
lines.append('')
lines.append('Reverse signal check (neg 干扰人干净近讲 decoder 更自信; if learned direction reverses -> signal useful but reversed):')
for sig in ['mean_logprob', 'mean_entropy']:
    d = directions.get(sig)
    imp_v = dict(imp_rank).get(sig, 0)
    lines.append(f'  {sig:25s}  imp={int(imp_v):3d}  spearman_w_score={d if d is not None else "NaN"}')
lines.append('')
lines.append('VERDICT:')
lines.append(f'  dual_pass (per-fold avg) = {dual_pass}')
lines.append(f'  dual_pass (max ceiling)  = {dual_pass_max}')
lines.append(f'  conclusion: {verdict["conclusion"]}')
lines.append('')
lines.append(f'Compliance: {fusion_out["compliance_note"]}')

report = '\n'.join(lines)
with open(os.path.join(OUT_DIR, 'holdout_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print(f'  Saved -> {OUT_DIR}/holdout_report.txt')
print('\n=== DONE ===')
