"""
Build the interactive results explorer: EXPERIMENTS/dashboard/index.html (one self-contained
file, no server, no install; open it in any browser, also on a Mac).

    python EXPERIMENTS/code/presentation_metrics.py     # first: scores every run
    python EXPERIMENTS/code/build_dashboard.py

Embedded data
  * every experiment's settings (passive table, EMG scaling, weighting, window, dropout,
    learning rate, extra input, code base), read from configs/*.json;
  * every run x test position: R2, MAE, RMSE, bias, demeaned R2/MAE, peak torque
    (results/presentation_runs.csv, answer key = passive trials of the tested session);
  * waveforms: measured voluntary torque and every run's prediction, sampled every 0.2 s,
    stored as int16 (0.01 Nm steps) in base64.
The page template is dashboard/template.html; the data replace the /*__DATA__*/ marker.
"""
import os
import sys
import json
import glob
import base64

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
sys.path.insert(0, HERE)
import presentation_metrics as PM                       # noqa: E402
from run_experiment import BASE                          # noqa: E402

RES = os.path.join(EXP, 'results')
DASH = os.path.join(EXP, 'dashboard')
GRID = np.round(np.arange(0.5, 88.9, 0.2), 2)             # common time grid (s) for all waveforms
SUBJ = PM.SUBJ


def params(cfg, code='framework'):
    return dict(code=code,
                passive={'legacy': 'original', 'static': 'clean'}.get(cfg['passive'], cfg['passive']),
                emg={'trainmax': 'max', 'agc_auto': 'autogain'}.get(cfg['emg'], cfg['emg']),
                weighting={'none': 'none', 'scale': 'equal'}.get(cfg['balance'], cfg['balance']),
                window=round(cfg['window'] / 100, 2), dropout=cfg['dropout'], lr=cfg['lr'] or 0.003,
                extra={None: 'none', 'passive_trial_value': 'passive'}.get(cfg.get('extra_input'), 'other'))


def b64(a):
    return base64.b64encode(np.clip(np.round(a * 100), -32767, 32767).astype('<i2').tobytes()).decode()


def on_grid(t, y):
    i = np.clip(np.searchsorted(t, GRID), 0, len(t) - 1)
    return y[i]


def main():
    exps = {'X00_baseline': dict(id='X00_baseline', title='Last semester (original code, frozen copy)', params=params(dict(BASE), 'original'))}
    for f in sorted(glob.glob(os.path.join(EXP, 'configs', '*.json'))):
        spec = json.load(open(f, encoding='utf-8'))
        cfg = dict(BASE)
        cfg.update(spec.get('changes', {}))
        e = os.path.basename(f)[:-5]
        if os.path.isdir(os.path.join(RES, e)):
            exps[e] = dict(id=e, title=spec.get('title', e), hypothesis=spec.get('hypothesis', ''), params=params(cfg))

    D = pd.read_csv(os.path.join(RES, 'presentation_runs.csv'))
    D['experiment'] = D.model.replace({'original': 'X00_baseline'})
    D = D[D.experiment.isin(exps)]
    cols = ['experiment', 'subject', 'direction', 'seed', 'position_rad', 'r2', 'mae', 'rmse', 'bias', 'r2_demeaned', 'mae_demeaned', 'peak_torque', 'toes_down']
    D = D[cols].drop_duplicates(['experiment', 'subject', 'direction', 'seed', 'position_rad'])
    rows = [[r.experiment, r.subject, r.direction, int(r.seed), round(r.position_rad, 3), round(r.r2, 4), round(r.mae, 4), round(r.rmse, 4),
             round(r.bias, 4), round(r.r2_demeaned, 4), round(r.mae_demeaned, 4), round(r.peak_torque, 2), int(r.toes_down)] for r in D.itertuples()]

    # waveforms: positions per subject x tested session, the measured key once, every run's prediction
    pos, key, wave = {}, {}, {}
    for (e, s, d, seed), _ in D.groupby(['experiment', 'subject', 'direction', 'seed']):
        f = os.path.join(RES, e, s, d, f'seed{seed}.npz')
        if not os.path.exists(f):
            continue
        tr = sorted(PM.trials(e, s, f), key=lambda x: x[0])
        k = f'{s}|{d}'
        if k not in pos:
            pos[k] = [round(p, 3) for p, *_ in tr]
        if len(tr) != len(pos[k]):
            continue
        if k not in key and e in ('X00f_baseline_framework', 'D01_cleanpassive+autogain+equalweight', 'C01_autogain+equalweight'):
            key[k] = b64(np.concatenate([on_grid(t, y) for _, t, y, _ in tr]))
        wave[f'{e}|{s}|{d}|{seed}'] = b64(np.concatenate([on_grid(t, p) for _, t, _, p in tr]))
    for k in pos:                                          # any run can supply the key if the preferred ones are missing
        if k not in key:
            s, d = k.split('|')
            e = next(x for x in exps if os.path.exists(os.path.join(RES, x, s, d, 'seed0.npz')))
            tr = sorted(PM.trials(e, s, os.path.join(RES, e, s, d, 'seed0.npz')), key=lambda x: x[0])
            key[k] = b64(np.concatenate([on_grid(t, y) for _, t, y, _ in tr]))

    data = dict(built=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), subjects=SUBJ, experiments=exps,
                cols=cols, rows=rows, grid=[float(GRID[0]), 0.2, len(GRID)], positions=pos, key=key, wave=wave)

    # extra tables for the React explorer (dashboard_react/): learning curves, earlier exploration, bias/variance
    lcs = []
    for f in sorted(glob.glob(os.path.join(RES, 'learning_curves', '*', '*', '*', 'seed*.json'))):
        r = json.load(open(f))
        lcs.append(dict(experiment=r['experiment'], subject=r['subject'], direction=r['direction'], seed=r['seed'],
                        best_epoch=r['best_epoch'], epochs=r['epochs'],
                        cols=['epoch', 'mae_fit', 'mae_val', 'mae_test', 'mse_fit', 'mse_val', 'mse_test', 'lr'],
                        history=[[h['epoch'], round(h['mae_fit'], 4), round(h['mae_val'], 4), round(h['mae_test'], 4),
                                  round(h['mse_fit'], 4), round(h['mse_val'], 4), round(h['mse_test'], 4), h['lr']] for h in r['history']]))
    extra = dict(learning_curves=lcs)
    p = os.path.join(RES, 'round2_rescored.csv')
    if os.path.exists(p):
        R2 = pd.read_csv(p)
        extra['round2'] = dict(cols=['study', 'experiment', 'subject', 'direction', 'seed', 'position_rad', 'r2', 'mae', 'toes_down'],
                               rows=[[r.study, r.experiment, r.subject, r.direction, int(r.seed), round(r.position_rad, 3), round(r.r2, 4),
                                      round(r.mae, 4), int(r.toes_down)] for r in R2.itertuples()])
    p = os.path.join(RES, 'families_rescored.csv')
    if os.path.exists(p):
        FA = pd.read_csv(p)
        extra['families'] = dict(cols=['model', 'subject', 'seed', 'position_rad', 'r2', 'mae', 'toes_down'],
                                 rows=[[r.model, r.subject, int(r.seed), round(r.position_rad, 3), round(r.r2, 4), round(r.mae, 4), int(r.toes_down)]
                                       for r in FA.itertuples()])
    p = os.path.join(RES, 'bias_variance_runs.csv')
    if os.path.exists(p):
        BV = pd.read_csv(p)
        extra['bias_variance'] = dict(cols=['model', 'subject', 'position_rad', 'runs', 'mse_runs', 'systematic', 'run_to_run', 'offset2',
                                            'r2_ensemble', 'mae_ensemble', 'toes_down'],
                                      rows=[[r.model, r.subject, round(r.position_rad, 3), int(r.runs), round(r.mse_runs, 5), round(r.systematic, 5),
                                             round(r.run_to_run, 5), round(r.offset2, 5), round(r.r2_ensemble, 4), round(r.mae_ensemble, 4), int(r.toes_down)]
                                            for r in BV.itertuples()])
    react = os.path.join(EXP, 'dashboard_react', 'src', 'data')
    os.makedirs(react, exist_ok=True)
    with open(os.path.join(react, 'results.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(data, **extra), fh, separators=(',', ':'))
    print(f'React data -> dashboard_react/src/data/results.json ({os.path.getsize(os.path.join(react, "results.json")) / 1e6:.1f} MB)')

    tpl = open(os.path.join(DASH, 'template.html'), encoding='utf-8').read()
    out = tpl.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':')))
    open(os.path.join(DASH, 'index.html'), 'w', encoding='utf-8').write(out)
    print(f'{len(exps)} experiments, {len(rows)} run x position rows, {len(wave)} waveforms -> dashboard/index.html '
          f'({os.path.getsize(os.path.join(DASH, "index.html")) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
