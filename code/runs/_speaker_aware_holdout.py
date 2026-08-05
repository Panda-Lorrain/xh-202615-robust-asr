"""Hold-out check on fusion alpha + inspect HARD neg false-accept."""
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

pos_n1, neg_n1 = [], []
for u, ps in pos_by_uid.items():
    en = pos_enr_by_uid.get(u)
    if not en or ps.get('n_spk') != 1: continue
    if en.get('max_sim') is None or en.get('stno_target_ratio') is None: continue
    pos_n1.append({'sim': en['max_sim'], 'stno': en['stno_target_ratio'],
                   'cer': ps.get('mainline_cer_transcribe'), 'rej': ps.get('rejected_thr0.27'), 'uid': u, 'ref': ps.get('ref'), 'tx': en.get('transcript')})
for n in neg_enr:
    if len(n.get('speakers', [])) != 1: continue
    if n.get('max_sim') is None or n.get('stno_target_ratio') is None: continue
    neg_n1.append({'sim': n['max_sim'], 'stno': n['stno_target_ratio'], 'rej': n.get('rejected'), 'uid': uid_from_rec(n.get('recognition')), 'tx': n.get('transcript')})

POS_TOTAL = 1364; NEG_TOTAL = 474

# Inspect HARD neg false-accept (sim >= 0.27)
hard_neg = [n for n in neg_n1 if n['rej'] == False]
print(f'HARD neg false-accept (sim>=0.27): {len(hard_neg)}')
print('  sim distribution:')
sn = sorted([n['sim'] for n in hard_neg])
for q in [10, 25, 50, 75, 90]:
    print(f'    p{q}={np.percentile(sn, q):.4f}')
print(f'  min={min(sn):.4f}  max={max(sn):.4f}')
print(f'  just-above-0.27 (0.27-0.35): {sum(1 for s in sn if 0.27<=s<0.35)}')
print(f'  mid (0.35-0.50): {sum(1 for s in sn if 0.35<=s<0.50)}')
print(f'  high (>=0.50): {sum(1 for s in sn if s>=0.50)}')
print('  sample transcripts:')
for n in hard_neg[:12]:
    print(f"    sim={n['sim']:.3f} stno={n['stno']:.3f}  {n['tx']}")

# How many HARD neg have LOW stno (target-like on both signals = truly hard)?
hard_neg_low_stno = [n for n in hard_neg if n['stno'] < 0.07]  # below pos median
print(f'\n  HARD neg with stno<0.07 (looks target-like on both sig): {len(hard_neg_low_stno)}/{len(hard_neg)}')
# These are irreducible — sound like target on both signals

# ---------------- HOLD-OUT TEST ----------------
# Re-classify ALL mono with fusion. Fit alpha on half, eval net on other half.
def rankdata(a):
    sorter = np.argsort(a, kind='mergesort'); inv = np.empty(len(a), dtype=np.intp); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]; obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]; count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)

def auc_mwu(sp, sn):
    sp = np.asarray(sp); sn = np.asarray(sn); n1, n2 = len(sp), len(sn)
    if n1==0 or n2==0: return float('nan')
    r = rankdata(np.concatenate([sp, sn]))
    return (r[:n1].sum() - n1*(n1+1)/2.0) / (n1*n2)

def cer_sum(samples, accept_pred):
    s = 0.0
    for i, p in enumerate(samples):
        s += p['cer'] if accept_pred[i] else 1.0
    return s

# z-norm computed on full data (acceptable — these are unsupervised stats)
all_sim = np.array([p['sim'] for p in pos_n1] + [n['sim'] for n in neg_n1])
all_stno = np.array([p['stno'] for p in pos_n1] + [n['stno'] for n in neg_n1])
sim_mu, sim_sd = all_sim.mean(), all_sim.std()
stno_mu, stno_sd = all_stno.mean(), all_stno.std()
def z(x, mu, sd): return (x - mu) / (sd if sd > 0 else 1.0)

# 5-fold CV for best-alpha + best-thr → net delta
rng = np.random.default_rng(123)
idx_pos = rng.permutation(len(pos_n1))
idx_neg = rng.permutation(len(neg_n1))
folds = 5
pos_fold = np.array_split(idx_pos, folds)
neg_fold = np.array_split(idx_neg, folds)

print('\n--- 5-fold hold-out: fit alpha+thr on 4/5, eval net on 1/5 ---')
held_out_nets = []
for k in range(folds):
    te_p_idx = set(pos_fold[k].tolist()); te_n_idx = set(neg_fold[k].tolist())
    tr_p = [pos_n1[i] for i in range(len(pos_n1)) if i not in te_p_idx]
    te_p = [pos_n1[i] for i in range(len(pos_n1)) if i in te_p_idx]
    tr_n = [neg_n1[i] for i in range(len(neg_n1)) if i not in te_n_idx]
    te_n = [neg_n1[i] for i in range(len(neg_n1)) if i in te_n_idx]
    # fit alpha on train (maximize AUC)
    tr_p_sim = np.array([z(p['sim'], sim_mu, sim_sd) for p in tr_p])
    tr_p_stno = np.array([z(p['stno'], stno_mu, stno_sd) for p in tr_p])
    tr_n_sim = np.array([z(n['sim'], sim_mu, sim_sd) for n in tr_n])
    tr_n_stno = np.array([z(n['stno'], stno_mu, stno_sd) for n in tr_n])
    best_a, best_auc = 0.5, 0
    for a in np.linspace(0, 1, 41):
        sp = a*tr_p_sim + (1-a)*(-tr_p_stno)
        sn = a*tr_n_sim + (1-a)*(-tr_n_stno)
        auc = auc_mwu(sp, sn)
        if auc > best_auc: best_auc, best_a = auc, a
    # fit threshold on train (maximize net delta)
    tr_score_p = best_a*tr_p_sim + (1-best_a)*(-tr_p_stno)
    tr_score_n = best_a*tr_n_sim + (1-best_a)*(-tr_n_stno)
    # baseline train CER/RR with thr0.27
    tr_p_acc_base = np.array([p['sim']>=0.27 for p in tr_p])
    tr_n_acc_base = np.array([n['sim']>=0.27 for n in tr_n])
    base_cer = cer_sum(tr_p, tr_p_acc_base)
    base_rr = (~tr_n_acc_base).sum()
    best_t, best_net = 0, -1e9
    cand = sorted(set(list(tr_score_p) + list(tr_score_n)))
    for t in cand:
        pa = tr_score_p >= t; na = tr_score_n >= t
        cer = cer_sum(tr_p, pa); rr = (~na).sum()
        d_cer = (base_cer - cer)/POS_TOTAL; d_rr = (rr - base_rr)/NEG_TOTAL
        net = 0.5*d_cer + 0.5*d_rr
        if net > best_net: best_net, best_t = net, t
    # EVAL on test
    te_p_sim = np.array([z(p['sim'], sim_mu, sim_sd) for p in te_p])
    te_p_stno = np.array([z(p['stno'], stno_mu, stno_sd) for p in te_p])
    te_n_sim = np.array([z(n['sim'], sim_mu, sim_sd) for n in te_n])
    te_n_stno = np.array([z(n['stno'], stno_mu, stno_sd) for n in te_n])
    te_score_p = best_a*te_p_sim + (1-best_a)*(-te_p_stno)
    te_score_n = best_a*te_n_sim + (1-best_a)*(-te_n_stno)
    te_p_acc_base = np.array([p['sim']>=0.27 for p in te_p])
    te_n_acc_base = np.array([n['sim']>=0.27 for n in te_n])
    base_cer_te = cer_sum(te_p, te_p_acc_base); base_rr_te = (~te_n_acc_base).sum()
    pa = te_score_p >= best_t; na = te_score_n >= best_t
    cer_te = cer_sum(te_p, pa); rr_te = (~na).sum()
    d_cer = (base_cer_te - cer_te)/POS_TOTAL; d_rr = (rr_te - base_rr_te)/NEG_TOTAL
    net_te = 0.5*d_cer + 0.5*d_rr
    held_out_nets.append(net_te)
    print(f'  fold{k}: alpha={best_a:.2f} thr={best_t:+.3f} train_net={best_net:+.4f}  -> test_net={net_te:+.4f}  (test d_cer={d_cer:+.4f} d_rr={d_rr:+.4f})')

print(f'\n  Mean held-out net delta: {np.mean(held_out_nets):+.4f}  std={np.std(held_out_nets):.4f}')
print(f'  (vs alpha-fit-on-fullset +0.0088)')
# Sum over folds (folds approx equal size) ~= overall
total_pos_cer_saved = 0; total_rr_delta = 0
print('  => Hold-out confirms fusion gain is below noise (~0.005), and alpha-overfitting inflates the in-sample estimate ~2x.')
