"""Fusion analysis + net-score impact for speaker-aware verification."""
import json, re
import numpy as np

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
pos_enr_by_uid = {uid_from_rec(n.get('recognition')): n for n in pos_enr if uid_from_rec(n.get('recognition'))}

# Build mono samples
pos_n1 = []
for u, ps in pos_by_uid.items():
    en = pos_enr_by_uid.get(u)
    if not en or ps.get('n_spk') != 1: continue
    if en.get('max_sim') is None or en.get('stno_target_ratio') is None: continue
    pos_n1.append({
        'uid': u, 'sim': en['max_sim'], 'stno': en['stno_target_ratio'],
        'cer': ps.get('mainline_cer_transcribe'),
        'rejected_thr0.27': ps.get('rejected_thr0.27'),
    })

neg_n1 = []
for n in neg_enr:
    if len(n.get('speakers', [])) != 1: continue
    if n.get('max_sim') is None or n.get('stno_target_ratio') is None: continue
    neg_n1.append({
        'uid': uid_from_rec(n.get('recognition')),
        'sim': n['max_sim'], 'stno': n['stno_target_ratio'],
        'rejected_thr0.27': n.get('rejected'),
    })

POS_TOTAL = 1364   # pos pool
NEG_TOTAL = 474    # neg pool

print('Pos n1 mono (model-complete):', len(pos_n1))
print('Neg n1 mono (model-complete):', len(neg_n1))

def rankdata(a):
    sorter = np.argsort(a, kind='mergesort')
    inv = np.empty(len(a), dtype=np.intp); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]; obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]; count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)

def auc_mwu(sp, sn):
    sp = np.asarray(sp); sn = np.asarray(sn)
    n1, n2 = len(sp), len(sn)
    if n1 == 0 or n2 == 0: return float('nan')
    r = rankdata(np.concatenate([sp, sn]))
    return (r[:n1].sum() - n1*(n1+1)/2.0) / (n1*n2)

def boot_auc(sp, sn, B=500, seed=42):
    rng = np.random.default_rng(seed)
    sp = np.asarray(sp); sn = np.asarray(sn)
    aucs = [auc_mwu(rng.choice(sp, len(sp), replace=True),
                   rng.choice(sn, len(sn), replace=True)) for _ in range(B)]
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 50)), float(np.percentile(aucs, 97.5)))

# ---------------- FUSION ----------------
# Score = a*sim + b*(-stno)  [stno inverted since lower-stno => more target-like]
# Grid search a in [0..1], b=1-a (1-D). Z-normalize first.
all_sim = np.array([p['sim'] for p in pos_n1] + [n['sim'] for n in neg_n1])
all_stno = np.array([p['stno'] for p in pos_n1] + [n['stno'] for n in neg_n1])
sim_mu, sim_sd = all_sim.mean(), all_sim.std()
stno_mu, stno_sd = all_stno.mean(), all_stno.std()

def z(x, mu, sd): return (x - mu) / (sd if sd > 0 else 1.0)

# Build arrays for operational (rejected) test
pos_frej = [p for p in pos_n1 if p['rejected_thr0.27'] == True and p['cer'] < 0.3]
neg_rej = [n for n in neg_n1 if n['rejected_thr0.27'] == True]
pos_frej_sim = np.array([z(p['sim'], sim_mu, sim_sd) for p in pos_frej])
pos_frej_stno = np.array([z(p['stno'], stno_mu, stno_sd) for p in pos_frej])
neg_rej_sim = np.array([z(n['sim'], sim_mu, sim_sd) for n in neg_rej])
neg_rej_stno = np.array([z(n['stno'], stno_mu, stno_sd) for n in neg_rej])

print('\n--- FUSION grid (operational: 147 pos_frej vs 299 neg_rej) ---')
print('score = a*z(sim) + (1-a)*(-z(stno))   [stno inverted]')
best = None
for a in np.linspace(0, 1, 21):
    score_pos = a * pos_frej_sim + (1 - a) * (-pos_frej_stno)
    score_neg = a * neg_rej_sim + (1 - a) * (-neg_rej_stno)
    lo, med, hi = boot_auc(score_pos, score_neg, B=300)
    if best is None or med > best[1]:
        best = (a, med, lo, hi)
    print(f'  a={a:.2f}  AUC={med:.4f}  CI[{lo:.4f},{hi:.4f}]')
print(f'  best a={best[0]:.2f}  AUC={best[1]:.4f}  CI[{best[2]:.4f},{best[3]:.4f}]')

# Use best alpha for detailed PR
a = best[0]
sp_score = a * pos_frej_sim + (1 - a) * (-pos_frej_stno)
sn_score = a * neg_rej_sim + (1 - a) * (-neg_rej_stno)
print(f'\nDetailed PR at fusion alpha={a:.2f} (operational):')
print('  thr              TP      FP    Precision  Recall   Net-score-delta')
for t in np.linspace(sp_score.min(), sp_score.max(), 30):
    tp = int((sp_score >= t).sum()); fp = int((sn_score >= t).sum())
    fn = len(sp_score) - tp
    prec = tp/(tp+fp) if tp+fp>0 else 0
    rec = tp/(tp+fn) if tp+fn>0 else 0
    # Net score delta: each TP saves (1 - 0.1) on CER for that sample / POS_TOTAL
    # Each FP costs 1/NEG_TOTAL on RR
    # weights 0.5, 0.5
    cer_gain = 0.9 * tp / POS_TOTAL    # approximating avg recovered cer ~0.1
    rr_loss = fp / NEG_TOTAL
    net = 0.5 * cer_gain - 0.5 * rr_loss
    print(f'  {t:+.3f}   TP={tp:3d}  FP={fp:3d}   P={prec:.3f}   R={rec:.3f}   net={net:+.4f}')

# ---------------- FULL n_spk=1 (not just rejected) using fusion ----------------
# Re-classify ALL mono: replace thr0.27 with fused score threshold
pos_all_sim = np.array([z(p['sim'], sim_mu, sim_sd) for p in pos_n1])
pos_all_stno = np.array([z(p['stno'], stno_mu, stno_sd) for p in pos_n1])
neg_all_sim = np.array([z(n['sim'], sim_mu, sim_sd) for n in neg_n1])
neg_all_stno = np.array([z(n['stno'], stno_mu, stno_sd) for n in neg_n1])

print('\n--- FUSION grid (all n_spk=1: 543 pos vs 333 neg) ---')
best_all = None
for a in np.linspace(0, 1, 21):
    score_pos = a * pos_all_sim + (1 - a) * (-pos_all_stno)
    score_neg = a * neg_all_sim + (1 - a) * (-neg_all_stno)
    lo, med, hi = boot_auc(score_pos, score_neg, B=300)
    if best_all is None or med > best_all[1]:
        best_all = (a, med, lo, hi)
print(f'  best fusion alpha={best_all[0]:.2f}  AUC={best_all[1]:.4f}  CI[{best_all[2]:.4f},{best_all[3]:.4f}]')
print(f'  (sim alone AUC = 0.8618 from Test 1)')

# ---------------- NET SCORE SIMULATION ----------------
# Current thr0.27 baseline:
#  - pos: 147 false-rejected (CER=1.0), others CER=actual
#  - neg: count rejected vs total
# Simulate replacing thr0.27 with fused score threshold on mono samples ONLY.
# (n_spk=2 path is scene-route, untouched.)
print('\n--- NET SCORE SIM: replace mono thr0.27 with fusion threshold ---')
# Need full mono arrays with cer/rejected status
pos_mono = pos_n1  # 543 with cer + rejected
neg_mono = neg_n1  # 333 with rejected
# Baseline mono contribution
pos_mono_base_cer = sum(1.0 if p['rejected_thr0.27'] else p['cer'] for p in pos_mono)  # rejected contributes CER=1.0
# But this is only mono pos (543 of 1364). For fair score we need ALL pos but we only have mono. Scale? Better: compute mono-only delta and note scope.

# Actually: we want to estimate the DELTA from re-classifying mono samples. The 543 mono pos + 333 mono neg.
# Among mono pos currently rejected by thr0.27: how many are false-rejected (cer<0.3) vs correctly rejected?
pos_mono_rej = [p for p in pos_mono if p['rejected_thr0.27']]
pos_mono_frej = [p for p in pos_mono_rej if p['cer'] < 0.3]
pos_mono_crej = [p for p in pos_mono_rej if p['cer'] >= 0.3]  # correctly rejected (bad transcript anyway)
print(f'  pos_mono rejected total: {len(pos_mono_rej)}  false-rej(CER<0.3): {len(pos_mono_frej)}  correctly-rej(CER>=0.3): {len(pos_mono_crej)}')

# Baseline CER sum on mono pos (using mainline_cer_transcribe)
def cer_sum(samples, accept_pred):
    s = 0.0
    for i, p in enumerate(samples):
        if accept_pred[i]:
            s += p['cer']  # accepted: actual CER
        else:
            s += 1.0       # rejected: CER=1.0 penalty
    return s

a_best = best_all[0]
# Compute fused score for ALL mono
pos_mono_score = a_best * np.array([z(p['sim'], sim_mu, sim_sd) for p in pos_mono]) + (1-a_best) * (-np.array([z(p['stno'], stno_mu, stno_sd) for p in pos_mono]))
neg_mono_score = a_best * np.array([z(n['sim'], sim_mu, sim_sd) for n in neg_mono]) + (1-a_best) * (-np.array([z(n['stno'], stno_mu, stno_sd) for n in neg_mono]))

# Baseline (thr0.27 on sim alone)
pos_mono_base_accept = np.array([p['sim'] >= 0.27 for p in pos_mono])
neg_mono_base_accept = np.array([n['sim'] >= 0.27 for n in neg_mono])
base_cer = cer_sum(pos_mono, pos_mono_base_accept)
base_rr_correct = (~neg_mono_base_accept).sum()  # neg correctly rejected

print('\n  Baseline (thr0.27 on sim):')
print(f'    pos_mono CER sum = {base_cer:.1f}  (avg {base_cer/len(pos_mono):.4f})')
print(f'    neg_mono correctly rejected = {base_rr_correct}/{len(neg_mono)} = RR {base_rr_correct/len(neg_mono):.4f}')
print(f'    neg_mono false accepted = {neg_mono_base_accept.sum()}')

# Sweep fusion threshold
print('\n  Fusion threshold sweep (only affects mono; n_spk=2 untouched):')
print('    thr        pos_CER_sum  pos_mono_avg_CER  neg_FA  neg_RR_mono   net_delta_overall')
best_net = (-1e9, None)
for t in np.linspace(min(pos_mono_score.min(), neg_mono_score.min()), max(pos_mono_score.max(), neg_mono_score.max()), 40):
    pos_acc = pos_mono_score >= t
    neg_acc = neg_mono_score >= t
    cer = cer_sum(pos_mono, pos_acc)
    rr_corr = (~neg_acc).sum()
    # delta on overall (assume pos pool 1364, neg pool 474; only mono affected)
    # n_spk=2 pos CER sum unchanged; n_spk=2 neg RR unchanged. So delta only on mono.
    d_cer = (base_cer - cer) / POS_TOTAL   # positive = improvement
    d_rr = (rr_corr - base_rr_correct) / NEG_TOTAL
    net = 0.5 * d_cer + 0.5 * d_rr
    if net > best_net[0]:
        best_net = (net, {'t': float(t), 'cer': cer, 'rr_corr': int(rr_corr), 'pos_acc': int(pos_acc.sum()), 'neg_acc': int(neg_acc.sum()), 'd_cer': float(d_cer), 'd_rr': float(d_rr)})
    if any(abs(t - x) < 0.05 for x in [-1.0, -0.5, 0.0, 0.5, 1.0]):
        print(f'    {t:+.3f}   CER={cer:6.1f}  avg={cer/len(pos_mono):.4f}  FA={int(neg_acc.sum()):3d}  RR_mono={rr_corr/len(neg_mono):.4f}  net={net:+.4f}')
print(f'\n  BEST NET overall delta: {best_net[0]:+.4f}')
for k, v in best_net[1].items():
    print(f'    {k}: {v}')

# What-if: precision needed for break-even
print('\n--- Break-even precision analysis ---')
print('  For each TP (recovered pos, cer 1.0->0.1): dCER_overall = +0.9/1364 = +0.000660')
print('  For each FP (neg flipped reject->accept): dRR_overall = -1/474 = -0.002110')
print('  Break-even when 0.9*TP/1364 = FP/474, i.e. FP/TP = 0.9*474/1364 =', 0.9*474/1364)
print('  => break-even precision = TP/(TP+FP) =', 1/(1+0.9*474/1364))
print('  NEED P > 0.763 for net positive')

# Save summary
summary = {
    'operational_best_fusion_alpha': float(best[0]),
    'operational_best_fusion_auc': float(best[1]),
    'operational_best_fusion_auc_ci': [float(best[2]), float(best[3])],
    'all_mono_best_fusion_alpha': float(best_all[0]),
    'all_mono_best_fusion_auc': float(best_all[1]),
    'all_mono_best_fusion_auc_ci': [float(best_all[2]), float(best_all[3])],
    'break_even_precision': 1/(1+0.9*474/1364),
    'best_net_delta_overall': float(best_net[0]),
    'best_net_op': best_net[1],
}
with open('code/runs/_speaker_aware_fusion.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\nSaved -> code/runs/_speaker_aware_fusion.json')
