"""Compute the tables shown in the v2 deck straight from the result files -> computed.json."""
import os, sys, json, glob
import numpy as np

WT = os.environ['WT']
sys.path.insert(0, WT)
from ML.research import truth as T

R = os.path.join(WT, 'ML', 'research', 'results')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'computed.json')
MAIN, EXTRA = ['YES', 'JM', 'EG'], ['IES', 'CH', 'HM']
PROTS = ['S1toS2', 'S2toS1']


def study(subj):
    return 'screen' if subj in MAIN else 'confirm'


def new_cfg(subj):
    return 'E32_restanchor_noagc_moe' if subj == 'CH' else 'E22_restanchor_final'


def load_truth_json(subj, prot, cfg, seeds, st=None):
    st = st or study(subj)
    acc = {}
    for s in seeds:
        p = os.path.join(R, st, subj, prot, cfg, f'fast_seed{s}_truth.json')
        if not os.path.exists(p):
            return None
        for k, v in json.load(open(p))['per_angle'].items():
            acc.setdefault(k, []).append(v)
    return acc


def rescore(subj, prot, cfg, seed, st):
    p = os.path.join(R, st, subj, prot, cfg, f'fast_seed{seed}.npz')
    if not os.path.exists(p):
        return None
    res = T.rescore(subj, p)['per_angle']
    return {k: [v] for k, v in res.items()}


def summarise(acc):
    keys = sorted(acc, key=lambda k: float(k.split(':')[1]))
    r2 = np.array([max(np.mean([d['r2'] for d in acc[k]]), -1.0) for k in keys])
    rmse = np.array([np.mean([d['rmse'] for d in acc[k]]) for k in keys])
    ph = np.array([np.nanmean([d.get('false_rest', np.nan) for d in acc[k]]) for k in keys])
    return dict(r2_all=float(r2.mean()), r2_pf2=float(r2[:2].mean()), err_pf2=float(rmse[:2].mean()),
                err_all=float(rmse.mean()), phantom=float(np.nanmean(ph)), angles=keys,
                r2=[float(x) for x in r2], rmse=[float(x) for x in rmse])


out = {'per_subject': {}, 'ablation': {}}
# ── per subject, old vs new, both directions (main: 3 seeds, extra: seed 0)
for subj in MAIN + EXTRA:
    seeds = [0, 1, 2] if subj in MAIN else [0]
    for prot in PROTS:
        o = summarise(load_truth_json(subj, prot, 'E01_old', seeds))
        n = summarise(load_truth_json(subj, prot, new_cfg(subj), seeds))
        out['per_subject'][f'{subj}|{prot}'] = {'old': o, 'new': n}

# ── ablation, seed 0 only, all 6 subjects (new and old from the existing seed-0 runs)
ABL = [('new', None), ('E34_final_oldpassive', 'ablation'), ('E32_restanchor_noagc_moe', 'ablation'),
       ('E33_final_noscale', 'ablation'), ('E35_final_lstm8', 'ablation'), ('old', None)]
for name, st in ABL:
    rows = {}
    for subj in MAIN + EXTRA:
        for prot in PROTS:
            if name == 'new':
                acc = load_truth_json(subj, prot, new_cfg(subj), [0])
            elif name == 'old':
                acc = load_truth_json(subj, prot, 'E01_old', [0])
            else:
                acc = rescore(subj, prot, name, 0, st)
            if acc is not None:
                rows[f'{subj}|{prot}'] = summarise(acc)
    out['ablation'][name] = rows

json.dump(out, open(OUT, 'w'), indent=1)
for name, rows in out['ablation'].items():
    for grp, subs in (('main', MAIN), ('extra', EXTRA)):
        vals = [v for k, v in rows.items() if k.split('|')[0] in subs]
        if vals:
            print(f'{name:28s} {grp:5s} n={len(vals):2d} R2 {np.mean([v["r2_all"] for v in vals]):.3f} '
                  f'pf2 {np.mean([v["r2_pf2"] for v in vals]):.3f} errpf2 {np.mean([v["err_pf2"] for v in vals]):.2f} '
                  f'phantom {np.mean([v["phantom"] for v in vals]):.2f}')
print('wrote', OUT)
