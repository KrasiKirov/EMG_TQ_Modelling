"""y07: extra baseline checks added after the v2 fact-check (chapters c08, c11-c15).

Reads  : DWT/EXPERIMENTS/results/all_evaluations.csv          (REF_train_mean rows: predict session 1's average)
         DWT/EXPERIMENTS/results/X00_baseline/<subj>/S1toS2/seed<k>.json   (the original code's own scores)
         DWT/EXPERIMENTS/results/learning_curves/X00f_baseline_framework/<subj>/S1toS2/seed0.json
         VWT/EXPERIMENTS/dashboard_react/src/data/results.json (X00 runs scored on the passive_trial key)
Writes : data2/facts_y07.json, data2/y07.npz
Run    : .venv311/Scripts/python.exe extract2/y07_reference_checks.py

Sections
  ref        a guess the model could have made: session 1's average active torque at each position, no EMG
             (score.py REF_train_mean, passive_trial key), against the baseline's 10-seed median RMSE and bias
  shift      how far the true average moved between sessions (REF bias = S1 fit mean - S2 key mean) vs the
             model's offset (median bias over 10 seeds), toes-down cells
  per_trial  R2 on fit / validation / test for every position, rebuilt baseline X00f seed 0 (final_per_trial)
  pooled     the original code's own pooled R2 (all 8 positions together) per person, 10 seeds
  own_target JM -0.078: R2 against the model's own (buggy) target, per seed, from the original code's output
  vaf        the original code's VAF next to R2 (YES -0.486 seed 1), the same quantity as shape-only R2
  dropout    training loss with dropout on (keras_loss) vs the real fit error (mse_fit) at the kept epoch
"""
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
VWT = HERE.parents[2]
DWT = pathlib.Path('C:/Users/dramo/Desktop/Projects/krasi_ml/EMG_TQ_Modelling/.claude/worktrees/dorsiflexion-r2-drop-341137')
RES = DWT / 'EXPERIMENTS/results'
OUT = VWT / 'EXPERIMENTS/pipeline_video/data2'
SUBJ = ['CH', 'EG', 'HM', 'IES', 'JM', 'YES']

ev = pd.read_csv(RES / 'all_evaluations.csv')
ev['pos'] = ev.position_rad.round(3)
raw = json.load(open(VWT / 'EXPERIMENTS/dashboard_react/src/data/results.json'))
df = pd.DataFrame(raw['rows'], columns=raw['cols'])
df['pos'] = df.position_rad.round(3)
x00 = df[(df.experiment == 'X00_baseline') & (df.direction == 'S1toS2')]
cell = x00.groupby(['subject', 'pos']).agg(rmse=('rmse', 'median'), bias=('bias', 'median'), r2=('r2', 'median'),
                                          td=('toes_down', 'first'), n=('seed', 'nunique')).reset_index()
facts, checks = {}, []

# ── ref: session-1 average, no EMG ───────────────────────────────────────────
ref = ev[(ev.experiment == 'REF_train_mean') & (ev.key == 'passive_trial') & (ev.direction == 'S1toS2')]
ref = ref.groupby(['subject', 'pos']).agg(ref_rmse=('rmse', 'first'), ref_r2=('r2', 'first'),
                                          ref_bias=('bias', 'first'), ref_n=('seed', 'nunique')).reset_index()
checks.append(dict(check='REF_train_mean has one row per cell (no seeds)', ok=bool((ref.ref_n == 1).all())))
m = cell.merge(ref, on=['subject', 'pos'], how='left')
checks.append(dict(check='every X00 cell has a REF row', ok=bool(m.ref_rmse.notna().all()), n=len(m)))
m['lstm_better'] = m.rmse < m.ref_rmse
rows = [dict(subject=r.subject, pos=r.pos, toes_down=int(r.td), lstm_rmse=r.rmse, ref_rmse=r.ref_rmse, ref_r2=r.ref_r2,
             lstm_r2=r.r2, lstm_bias=r.bias, s1_minus_s2_mean=r.ref_bias, lstm_better=bool(r.lstm_better))
        for r in m.itertuples()]
facts['ref'] = dict(
    definition='REF_train_mean (score.py): predict, at each position, the average session-1 training target there; '
               'scored on the passive_trial key like X00. bias = mean(pred - key), so REF bias = S1 mean - S2 mean.',
    cells=rows,
    n_cells_ref_beats_lstm=int((~m.lstm_better).sum()),
    cells_ref_beats_lstm=[f'{r.subject} {r.pos:+.3f}' for r in m[~m.lstm_better].itertuples()],
    ref_r2_median_all=float(m.ref_r2.median()), ref_r2_median_toes_down=float(m[m.td == 1].ref_r2.median()),
    lstm_over_ref_rmse_median_toes_down=float((m[m.td == 1].rmse / m[m.td == 1].ref_rmse).median()),
    lstm_over_ref_rmse_median_other=float((m[m.td == 0].rmse / m[m.td == 0].ref_rmse).median()))

# ── shift: did the true average move, or is the offset the model's own? ──────
td = m[m.td == 1].copy()
facts['shift'] = dict(
    note='s1_minus_s2_mean = how much the true average differs between session 1 (fit part) and session 2, '
         'with the sign of a bias (positive = session 1 higher); lstm_bias = median over 10 seeds of mean(pred - key)',
    cells=[dict(subject=r.subject, pos=r.pos, s1_minus_s2_mean=r.ref_bias, lstm_bias=r.bias,
                unexplained=r.bias - r.ref_bias) for r in td.itertuples()])

# ── per_trial: fit / val / test R2 per position, X00f seed 0 ─────────────────
pt = []
for s in SUBJ:
    lc = json.load(open(RES / f'learning_curves/X00f_baseline_framework/{s}/S1toS2/seed0.json'))
    f = lc['final_per_trial']
    fit = sorted(f['fit'], key=lambda d: d['angle'])
    val = sorted(f['val'], key=lambda d: d['angle'])
    tst = sorted(f['test'], key=lambda d: d['angle'])
    for k, (a, b, c) in enumerate(zip(fit, val, tst)):
        assert a['trial'] == b['trial'] and abs(a['angle'] - c['angle']) < 0.02
        pt.append(dict(subject=s, rank=k, toes_down=int(k < 2), angle_s1=a['angle'], angle_s2=c['angle'],
                       r2_fit=a['r2'], r2_val=b['r2'], r2_test=c['r2'], mae_fit=a['mae'], mae_val=b['mae'],
                       mae_test=c['mae'], keras_best_epoch=lc['best_epoch']))
ptd = pd.DataFrame(pt)
facts['per_trial'] = dict(
    source='X00f_baseline_framework seed 0, final_per_trial (kept epoch, dropout off); fit/val on the training '
           'target, test on the clean key (learning_curves.py)',
    rows=pt,
    toes_down_median=dict(r2_fit=float(ptd[ptd.toes_down == 1].r2_fit.median()),
                          r2_val=float(ptd[ptd.toes_down == 1].r2_val.median()),
                          r2_test=float(ptd[ptd.toes_down == 1].r2_test.median())),
    other_median=dict(r2_fit=float(ptd[ptd.toes_down == 0].r2_fit.median()),
                      r2_val=float(ptd[ptd.toes_down == 0].r2_val.median()),
                      r2_test=float(ptd[ptd.toes_down == 0].r2_test.median())))

# ── pooled R2, own-target R2, VAF: the original code's own output ────────────
pooled, own, vaf = {}, [], None
for s in SUBJ:
    vals = []
    for k in range(10):
        j = json.load(open(RES / f'X00_baseline/{s}/S1toS2/seed{k}.json'))
        vals.append(j['overall_original']['r2'])
        if s == 'JM':
            key = min(j['per_position_original'], key=lambda p: abs(float(p) + 0.078))
            own.append(dict(seed=k, position=key, r2_own_target=j['per_position_original'][key]['r2']))
        if s == 'YES' and k == 1:
            key = min(j['per_position_original'], key=lambda p: abs(float(p) + 0.486))
            vaf = dict(position=key, **j['per_position_original'][key])
    pooled[s] = dict(min=min(vals), max=max(vals), median=float(np.median(vals)), seeds=vals)
yes_pos = x00[(x00.subject == 'YES') & (x00.pos == -0.486)]
facts['pooled'] = dict(note="the original code's 'overall' R2: all 8 positions of session 2 pooled, own target",
                       per_subject=pooled, yes_m486_r2_min=float(yes_pos.r2.min()), yes_m486_r2_max=float(yes_pos.r2.max()))
o = [d['r2_own_target'] for d in own]
facts['own_target'] = dict(note='JM -0.078: R2 against the target the network was trained toward (legacy table), '
                                'from the original code\'s per_position_original', per_seed=own,
                           median=float(np.median(o)), n_negative=int(sum(v < 0 for v in o)), min=min(o), max=max(o),
                           clean_key_median=float(x00[(x00.subject == 'JM') & (x00.pos == -0.078)].r2.median()))
facts['vaf'] = dict(note='VAF = (1 - var(error)/var(truth)) x 100 in evaluation/metrics.py = shape-only R2', **vaf)
checks.append(dict(check='VAF equals shape-only R2 (YES -0.486 seed 1)',
                   ok=bool(abs(vaf['vaf'] / 100 - float(x00[(x00.subject == 'YES') & (x00.pos == -0.486) & (x00.seed == 1)].r2_demeaned.iloc[0])) < 1e-3)))

# ── dropout: printed training loss vs real fit error ─────────────────────────
dl = {}
for s in SUBJ:
    lc = json.load(open(RES / f'learning_curves/X00f_baseline_framework/{s}/S1toS2/seed0.json'))
    h = next(e for e in lc['history'] if e['epoch'] == lc['best_epoch'])
    dl[s] = dict(keras_loss=h['keras_loss'], mse_fit=h['mse_fit'], ratio=h['keras_loss'] / h['mse_fit'],
                 keras_val_loss=h['keras_val_loss'], mse_val=h['mse_val'])
facts['dropout_loss'] = dict(note='at the kept epoch, X00f seed 0: keras_loss = training loss with dropout on; '
                                  'mse_fit = the same fit windows with dropout off', per_subject=dl,
                             ratio_min=min(v['ratio'] for v in dl.values()), ratio_max=max(v['ratio'] for v in dl.values()))

# ── CH -0.326: task flip between sessions ────────────────────────────────────
ch = m[(m.subject == 'CH') & ((m.pos + 0.326).abs() < 0.01)].iloc[0]
facts['ch_flip'] = dict(pos=float(ch.pos), ref_r2=float(ch.ref_r2), lstm_r2_median=float(ch.r2))

facts['checks'] = checks
OUT.mkdir(exist_ok=True)
(OUT / 'facts_y07.json').write_text(json.dumps({'y07': facts}, indent=1, default=float), encoding='utf-8')
np.savez_compressed(OUT / 'y07.npz',
                    td_subject=np.array([r['subject'] for r in facts['shift']['cells']]),
                    td_pos=np.array([r['pos'] for r in facts['shift']['cells']]),
                    td_s1_minus_s2=np.array([r['s1_minus_s2_mean'] for r in facts['shift']['cells']]),
                    td_lstm_bias=np.array([r['lstm_bias'] for r in facts['shift']['cells']]),
                    td_ref_rmse=td.ref_rmse.to_numpy(), td_lstm_rmse=td.rmse.to_numpy(),
                    pt_subject=ptd.subject.to_numpy().astype(str), pt_rank=ptd["rank"].to_numpy().astype(int),
                    pt_r2=ptd[['r2_fit', 'r2_val', 'r2_test']].to_numpy())

# ── readable summary ─────────────────────────────────────────────────────────
print('REF beats LSTM at:', facts['ref']['cells_ref_beats_lstm'])
print(td[['subject', 'pos', 'rmse', 'ref_rmse', 'ref_r2', 'bias', 'ref_bias']].round(3).to_string(index=False))
print('per-trial toes-down fit/val/test median R2:', {k: round(v, 3) for k, v in facts['per_trial']['toes_down_median'].items()})
print(ptd[ptd.toes_down == 1][['subject', 'angle_s2', 'r2_fit', 'r2_val', 'r2_test']].round(3).to_string(index=False))
print('pooled R2:', {s: (round(v['min'], 3), round(v['max'], 3)) for s, v in pooled.items()})
print('JM own-target R2 per seed:', [round(v, 2) for v in o], 'median', round(np.median(o), 2))
print('VAF YES -0.486 seed1:', vaf)
print('dropout loss ratio:', {s: round(v['ratio'], 1) for s, v in dl.items()})
print('CH flip:', facts['ch_flip'])
print('checks:', checks)
