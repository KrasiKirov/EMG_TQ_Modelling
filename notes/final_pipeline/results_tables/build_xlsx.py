"""
Build all_results.xlsx from all_runs_per_position.csv (made by make_tables.py).

Sheets
------
README                       what everything means
Summary                      per configuration, per group of people, per answer key (formulas)
Per subject                  per configuration × person × direction × key (formulas)
Per position (mean of runs)  per angle, averaged over the training runs (values + helper formulas)
Per run per position         every training run, every angle, every key (values)

The per-run values are computed by make_tables.py from the saved predictions in
ML/research/results; the summary sheets are Excel formulas on top of them.
"""

import os
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_tables import PLAIN  # noqa: E402  (only the dictionary; make_tables runs only as a script)

df = pd.read_csv(os.path.join(HERE, 'all_runs_per_position.csv'))
KEY_PLAIN = {'passive_trial': 'Passive trials of the tested session (supervisor)',
             'in_trial': 'Relaxed moments inside the trial (report 2)',
             'own_target': "Each version's own target (last semester's way)"}
STUDY_PLAIN = {'screen': 'main 3 people, 3 runs', 'confirm': 'extra 3 people, 1 run', 'ablation': 'remove-one-change test, 1 run',
               'final_check': 'FINAL check, 1 run', 'passive': 'passive-trials-only versions'}

# toes-down flag: the 2 most negative angles of that person's tested session
df['toes_down'] = 0
for (sj, ses), x in df.groupby(['subject', 'tested_session']):
    angs = sorted(x['angle_rad'].unique())[:2]
    df.loc[x.index[x['angle_rad'].isin(angs)], 'toes_down'] = 1

FONT = 'Arial'
HDR = Font(name=FONT, bold=True, color='FFFFFF')
HFILL = PatternFill('solid', fgColor='1F2A44')
BODY = Font(name=FONT)
BOLD = Font(name=FONT, bold=True)


def write_table(ws, frame, start_row=1, widths=None):
    for j, c in enumerate(frame.columns, 1):
        cell = ws.cell(row=start_row, column=j, value=c)
        cell.font, cell.fill = HDR, HFILL
        cell.alignment = Alignment(wrap_text=True, vertical='center')
    for i, row in enumerate(frame.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            ws.cell(row=i, column=j, value=v).font = BODY
    ws.freeze_panes = ws.cell(row=start_row + 1, column=1)
    for j, c in enumerate(frame.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(c, max(10, min(40, len(str(c)) + 2)))


wb = Workbook()
# ── README ───────────────────────────────────────────────────────────────────
ws = wb.active
ws.title = 'README'
lines = [
    ('EMG → voluntary ankle torque: complete results', True),
    ('', False),
    ('Every number comes from the saved predictions in ML/research/results, re-scored by notes/final_pipeline/results_tables/make_tables.py.', False),
    ('', False),
    ('ANSWER KEYS (what the prediction is compared with, per test trial)', True),
    ('passive_trial = measured torque − passive torque from the passive trials of the TESTED session (clean static holds; YES: lab worksheet values). Supervisor\'s choice. IES has no passive trials, so nothing is subtracted for IES.', False),
    ('in_trial = measured torque − smooth line through the EMG-detected relaxed moments of the same trial (report 2).', False),
    ("own_target = each version's own test target (how last semester scored).", False),
    ('', False),
    ('METRICS (per test trial = one angle of one tested session)', True),
    ('R² = 1 − Σ(prediction − key)² / Σ(key − mean of key)².  1 = perfect, 0 = as good as a flat line at the average, below 0 = worse.', False),
    ('MSE = mean of (prediction − key)², in Nm².   RMSE = square root of MSE, in Nm (typical size of the miss).', False),
    ('NRMSE = RMSE ÷ (98th − 2nd percentile of the key): the error as a fraction of the torque range of that trial.', False),
    ('Bias = mean of (prediction − key), in Nm (negative = predicts too much toes-down push).', False),
    ('', False),
    ('OTHER COLUMNS', True),
    ('run = training run (1, 2, 3): the same training repeated from a different random start.', False),
    ('direction = which session the model learned from and which it was tested on.', False),
    ('toes_down = 1 for the 2 most toes-down (most negative) angles of that person.', False),
    ('In the summary sheets, R² of each angle is first averaged over runs, then floored at −1 (so one catastrophic angle cannot dominate), then averaged.', False),
    ('', False),
    ('STUDIES', True),
] + [(f'{k}: {v}', False) for k, v in STUDY_PLAIN.items()] + [('', False), ('CONFIGURATIONS', True)] + \
    [(f'{k}: {v}', False) for k, v in PLAIN.items() if k in set(df['config'])]
for i, (t, b) in enumerate(lines, 1):
    c = ws.cell(row=i, column=1, value=t)
    c.font = BOLD if b else BODY
ws.column_dimensions['A'].width = 160

# ── Per position (mean of runs) ─────────────────────────────────────────────
mean = (df.groupby(['study', 'config', 'description', 'subject', 'direction', 'tested_session', 'angle_rad', 'angle_deg',
                    'toes_down', 'key'], as_index=False)
          .agg(runs=('run', 'nunique'), r2=('r2', 'mean'), mse=('mse', 'mean'), rmse=('rmse', 'mean'),
               nrmse=('nrmse', 'mean'), bias=('bias', 'mean')))
mean = mean.sort_values(['key', 'study', 'config', 'subject', 'direction', 'angle_rad']).reset_index(drop=True)
wsP = wb.create_sheet('Per position (mean of runs)')
write_table(wsP, mean, widths={'description': 45, 'direction': 22})
cols = {c: get_column_letter(j) for j, c in enumerate(mean.columns, 1)}
fcol = get_column_letter(len(mean.columns) + 1)
wsP.cell(row=1, column=len(mean.columns) + 1, value='r2_floored').font = HDR
wsP.cell(row=1, column=len(mean.columns) + 1).fill = HFILL
N = len(mean) + 1
for i in range(2, N + 1):
    wsP[f'{fcol}{i}'] = f'=MAX({cols["r2"]}{i},-1)'
    wsP[f'{fcol}{i}'].font = BODY
rng = lambda c: f"'Per position (mean of runs)'!${c}$2:${c}${N}"
R = {c: rng(cols[c]) for c in ('key', 'study', 'config', 'subject', 'direction', 'toes_down', 'rmse', 'mse', 'nrmse')}
R['r2f'] = rng(fcol)

# ── Per subject (formulas) ──────────────────────────────────────────────────
ps = mean[['key', 'study', 'config', 'description', 'subject', 'direction', 'runs']].drop_duplicates().reset_index(drop=True)
wsS = wb.create_sheet('Per subject')
heads = list(ps.columns) + ['R² (8 angles)', 'R² toes-down', 'MSE (Nm²)', 'RMSE (Nm)', 'RMSE toes-down (Nm)', 'NRMSE']
write_table(wsS, pd.DataFrame(columns=heads), widths={'description': 45, 'direction': 22})
for i, row in enumerate(ps.itertuples(index=False), 2):
    for j, v in enumerate(row, 1):
        wsS.cell(row=i, column=j, value=v.item() if isinstance(v, np.generic) else v).font = BODY
    crit = (f'{R["key"]},$A{i},{R["study"]},$B{i},{R["config"]},$C{i},{R["subject"]},$E{i},{R["direction"]},$F{i}')
    f = {'H': f'=AVERAGEIFS({R["r2f"]},{crit})', 'I': f'=AVERAGEIFS({R["r2f"]},{crit},{R["toes_down"]},1)',
         'J': f'=AVERAGEIFS({R["mse"]},{crit})', 'K': f'=AVERAGEIFS({R["rmse"]},{crit})',
         'L': f'=AVERAGEIFS({R["rmse"]},{crit},{R["toes_down"]},1)', 'M': f'=AVERAGEIFS({R["nrmse"]},{crit})'}
    for c, v in f.items():
        wsS[f'{c}{i}'] = v
        wsS[f'{c}{i}'].font = BODY
        wsS[f'{c}{i}'].number_format = '0.000'

# ── Summary (formulas on Per subject) ───────────────────────────────────────
NS = len(ps) + 1
srng = lambda c: f"'Per subject'!${c}$2:${c}${NS}"
GROUPS = [('Main 3 (YES, JM, EG)', ['YES', 'JM', 'EG']), ('Extra: CH + HM', ['CH', 'HM']), ('IES (no passive trials)', ['IES'])]
summ = []
for (key, study, cfg, desc), x in ps.groupby(['key', 'study', 'config', 'description'], sort=False):
    for gname, subs in GROUPS:
        if x['subject'].isin(subs).any():
            summ.append((key, study, cfg, desc, gname, ','.join(s for s in subs if s in set(x['subject']))))
wsM = wb.create_sheet('Summary', 1)
heads = ['key', 'study', 'config', 'description', 'group', 'people', 'R² (8 angles)', 'R² toes-down', 'MSE (Nm²)', 'RMSE (Nm)',
         'RMSE toes-down (Nm)', 'NRMSE']
write_table(wsM, pd.DataFrame(columns=heads), widths={'description': 50, 'group': 22, 'people': 12})
for i, (key, study, cfg, desc, gname, people) in enumerate(summ, 2):
    for j, v in enumerate((key, study, cfg, desc, gname, people), 1):
        wsM.cell(row=i, column=j, value=v).font = BODY
    subs = people.split(',')
    for col, src in zip('GHIJKL', 'HIJKLM'):
        parts = [f'SUMIFS({srng(src)},{srng("A")},$A{i},{srng("B")},$B{i},{srng("C")},$C{i},{srng("E")},"{s}")' for s in subs]
        cnts = [f'COUNTIFS({srng("A")},$A{i},{srng("B")},$B{i},{srng("C")},$C{i},{srng("E")},"{s}")' for s in subs]
        wsM[f'{col}{i}'] = f'=({"+".join(parts)})/({"+".join(cnts)})'
        wsM[f'{col}{i}'].font = BODY
        wsM[f'{col}{i}'].number_format = '0.000'

# ── Per run per position (values) ───────────────────────────────────────────
wsR = wb.create_sheet('Per run per position')
runcols = ['key', 'study', 'config', 'description', 'subject', 'direction', 'run', 'tested_session', 'angle_rad', 'angle_deg',
           'toes_down', 'r2', 'mse', 'rmse', 'nrmse', 'bias', 'std_key', 'n']
write_table(wsR, df.sort_values(['key', 'study', 'config', 'subject', 'direction', 'run', 'angle_rad'])[runcols],
            widths={'description': 45, 'direction': 22})

out = os.path.join(HERE, 'all_results.xlsx')
wb.save(out)
print('saved', out, 'rows', len(df), 'summary rows', len(summ), 'per-subject rows', len(ps))
