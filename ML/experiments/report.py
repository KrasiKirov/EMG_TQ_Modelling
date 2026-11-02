"""Build the summary tables (markdown / Discord code blocks) from results.

python -m ML.experiments.report            → prints all tables
python -m ML.experiments.report --discord  → also returns chunked messages
"""

import os
import sys
import json
import glob

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ML.experiments.harness import RESULTS_DIR

WEAK = {'YES': ['-0.486', '-0.381'], 'JM': ['-0.787', '-0.644', '-0.360']}

# (run name, short description, honest?) — screening runs (harness, fast protocol)
METHODS = [
    ('baseline',                 'Baseline (current pipeline)',                           'yes'),
    ('clip12',                   'Clip EMG artefacts (env <= 1.2x train max)',            'yes'),
    ('posw_invvar',              'Per-angle loss weights only',                           'yes'),
    ('knotpos',                  'Set-point angle instead of measured angle',             'yes'),
    ('H_pgain_smooth_posnorm',   'Passive gain + drift-free train labels + per-angle std', 'yes'),
    ('H_pgain_posw',             'Passive gain + per-angle loss weights',                 'yes'),
    ('H_pgain_smooth_posw',      '... + drift-free training labels',                      'yes'),
    ('H_pgain_smooth_posw_knot', '... + set-point angle',                                 'yes'),
    ('H_anc_pgain_smooth_floortrain_posw', '... + zero-anchored LSTM',                    'yes'),
    ('ml_moe',                   'Mixture-of-experts head per angle',                     'yes'),
    ('gain_snr_agc_posw',        'Causal EMG auto-gain + per-angle weights',              'causal-transductive'),
    ('rezero_const',             'Re-zero target on in-trial rest (constant)',            'NO (uses test torque)'),
    ('rezero_drift',             'Re-zero target on in-trial rest (drift)',               'NO (uses test torque)'),
    ('drift_posw',               'Re-zero (drift) + per-angle weights',                   'NO (uses test torque)'),
    ('anc_floor_drift',          'Re-zero (drift) + anchored LSTM',                       'NO (uses test torque)'),
]


def _load(path):
    with open(path) as f:
        return json.load(f)


def harness_r2(subject, name, protocol='fast'):
    p = os.path.join(RESULTS_DIR, f'{subject}__{name}__{protocol}.json')
    if not os.path.exists(p):
        return None
    r = _load(p)
    return {k: v['r2_mean'] for k, v in r['summary'].items()}


def methods_table():
    hdr = f"{'Method':<52} {'Honest':<22} " + ' '.join(
        f"{s+' '+k:>12}" for s in WEAK for k in WEAK[s])
    lines = [hdr, '-' * len(hdr)]
    for name, desc, honest in METHODS:
        cells = []
        for s in WEAK:
            r = harness_r2(s, name)
            for k in WEAK[s]:
                cells.append(f"{r[k]:>+12.2f}" if r and k in r else f"{'—':>12}")
        if all(c.strip() == '—' for c in cells):
            continue
        lines.append(f"{desc[:52]:<52} {honest[:22]:<22} " + ' '.join(cells))
    return '\n'.join(lines)


def pipeline_table(subject, names, protocol):
    rows = []
    for n in names:
        p = os.path.join(RESULTS_DIR, 'pipeline', f'{subject}__{n}__{protocol}.json')
        if os.path.exists(p):
            rows.append((n, _load(p)))
    if not rows:
        return f'(no pipeline results for {subject} {protocol})'
    keys = [k for k in rows[0][1]['summary'] if k != 'pooled']
    hdr = f"{'pos':>7} | " + ' | '.join(f"{n[:22]:^22}" for n, _ in rows)
    sub = f"{'':>7} | " + ' | '.join(f"{'R2':>6} {'MSE':>6} {'RMSE':>6} " for _ in rows)
    lines = [hdr, sub, '-' * len(sub)]
    for k in keys + ['pooled']:
        cells = []
        for _, r in rows:
            s = r['summary'][k]
            cells.append(f"{s['r2']:>+6.2f} {s['mse']:>6.2f} {s['rmse']:>6.2f} ")
        lines.append(f"{k:>7} | " + ' | '.join(cells))
    return '\n'.join(lines)


def main():
    print('=== Screening (harness, fast protocol, mean of 3 seeds): per-position R² at the weak angles')
    print(methods_table())
    for prot in ('fast', 'orig'):
        for s in ('YES', 'JM', 'IES', 'EG', 'CH', 'HM'):
            files = glob.glob(os.path.join(RESULTS_DIR, 'pipeline', f'{s}__*__{prot}.json'))
            if files:
                names = sorted({os.path.basename(f).split('__')[1] for f in files},
                               key=lambda n: (n != 'legacy', n))
                print(f'\n=== {s} — real pipeline, {prot} protocol (mean of seeds)')
                print(pipeline_table(s, names, prot))


if __name__ == '__main__':
    main()
