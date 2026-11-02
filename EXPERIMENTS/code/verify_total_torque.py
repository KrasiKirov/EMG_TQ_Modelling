"""
Pre-batch checks for the total-torque study (EXPERIMENTS/configs/TT*.json).

    python EXPERIMENTS/code/verify_total_torque.py --subjects YES JM

For every subject, direction and TT configuration it checks that
 1. the target of every train / validation / test window equals the processed
    measured torque at the window's last sample (no passive subtraction), and
    differs from the D01 active target (measured - clean passive table);
 2. inputs and target are aligned: the window's last input step is the
    target sample (angle channel == measured angle at that sample);
 3. fixed EMG scales (TT02, TT03) come from the training part only: changing
    the validation part and the whole other session leaves them unchanged and
    they equal a direct recomputation; TT04's source-session scale uses the
    training part only and its target-session scale only target-session EMG;
 4. auto-gain is off: every feature equals envelope / constant scale;
 5. the balancing factors (training SD per position, floor 0.1 Nm) come from
    training targets only, and fit_predict multiplies predictions back to Nm
    (a network that always outputs 1.0 must return exactly SD of the
    position of each test window).
Writes a JSON report next to the results (EXPERIMENTS/results/total_torque_checks.json).
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.dirname(EXP)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np                                   # noqa: E402

from run_experiment import load_config, drop_last_val_window   # noqa: E402
from ML.research.run import prepare                 # noqa: E402
from ML.research.data import load_subject           # noqa: E402
from ML.research.passive import PassiveModel        # noqa: E402
from ML.research import emg_norm, splits, train as RT, config as RC   # noqa: E402

TT = ['TT01_total_w200_bal_rawemg', 'TT02_total_w200_bal_trainmax', 'TT03_total_w200_bal_trainp99',
      'TT04_total_w200_bal_sessionp99_offline']


def check(cond, msg, out):
    out.append({'ok': bool(cond), 'check': msg})
    if not cond:
        print('FAIL', msg, flush=True)


def nm_restore_check(data):
    """Constant-output network -> predictions must equal SD of each test window's position."""
    from tensorflow.keras import Sequential, layers, initializers
    orig_build, orig_p = RT.build_model, dict(RC.PROTOCOLS['orig'])

    def build(name, n_steps, n_features, knots):
        return Sequential([layers.Input((n_steps, n_features)), layers.Flatten(),
                           layers.Dense(1, kernel_initializer='zeros', bias_initializer=initializers.Constant(1.0),
                                        trainable=False)])
    RT.build_model = build
    RC.PROTOCOLS['orig'] = dict(orig_p, epochs=1)
    try:
        small = dict(data)
        for p in ('train', 'val'):                    # every 50th window: all positions, fast
            sl = slice(None, None, 50)
            small[f'X_{p}'], small[f'y_{p}'] = data[f'X_{p}'][sl], data[f'y_{p}'][sl]
            small[f'meta_{p}'] = {k: v[sl] for k, v in data[f'meta_{p}'].items()}
        pred, info = RT.fit_predict(small, 'lstm', 'scale', None, 'orig', 0)
        if not np.all(np.isfinite(info['sd_per_knot'])):
            return False
    finally:
        RT.build_model, RC.PROTOCOLS['orig'] = orig_build, orig_p
    knots = np.array(info['knots'])
    sd = np.array(info['sd_per_knot'])
    return np.allclose(pred, sd[RT.snap(data['meta_test']['angle'], knots)], rtol=1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=['YES', 'JM'])
    ap.add_argument('--directions', nargs='+', default=['S1toS2', 'S2toS1'])
    a = ap.parse_args()
    results = []
    for s in a.subjects:
        b = load_subject(s)
        recs = {r['idx']: r for r in b['active']}
        static = PassiveModel('static', b)
        for d in a.directions:
            trs = splits.PROTOCOLS[d]['train']
            train_recs = [r for r in b['active'] if r['session'] in trs]
            for exp in TT:
                spec, cfg = load_config(exp)
                tag = f'{s} {d} {exp}'
                out = []
                data, details = prepare(s, d, {k: cfg[k] for k in ('passive', 'emg', 'window')})
                data = drop_last_val_window(data)
                check(cfg['passive'] == 'none' and details['passive']['kind'] == 'none', f'{tag}: passive none', out)
                check('agc' not in cfg['emg'] and not details['emg'].get('auto_decision'), f'{tag}: auto-gain off', out)
                for p in ('train', 'val', 'test'):
                    m = data[f'meta_{p}']
                    tq = np.array([recs[i]['tq'][k] for i, k in zip(m['trial'], m['last'])], np.float32)
                    act = np.array([recs[i]['tq'][k] - static.predict(recs[i])[k]
                                    for i, k in zip(m['trial'][:200], m['last'][:200])], np.float32)
                    pos = np.array([recs[i]['pos'][k] for i, k in zip(m['trial'], m['last'])], np.float32)
                    check(np.array_equal(data[f'y_{p}'], tq), f'{tag}: {p} target == measured torque', out)
                    check(s == 'IES' or not np.allclose(data[f'y_{p}'][:200], act),
                          f'{tag}: {p} target differs from D01 active target', out)
                    check(np.array_equal(data[f'X_{p}'][:, -1, 4], pos), f'{tag}: {p} last input step == target sample', out)
                # features == env / constant scale (no adaptive gain), scale from training part only
                kind = cfg['emg']
                feats, info = emg_norm.normalise(kind, b, train_recs, b['active'])
                ratios = []
                for r in b['active']:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        q = r['env_raw'] / feats[r['idx']]
                    ratios.append(np.nanmedian(q, axis=0))
                    ok = np.allclose(feats[r['idx']] * ratios[-1], r['env_raw'], rtol=1e-9, atol=0)
                    if not ok:
                        break
                check(ok, f'{tag}: features = envelope / constant per-channel scale', out)
                if kind in ('trainmax_tr', 'trainp99', 'session_p99'):
                    pert = {'active': []}
                    for r in b['active']:
                        r2 = dict(r, env_raw=r['env_raw'].copy())
                        if r['session'] in trs:
                            r2['env_raw'][emg_norm._train_part(r):] *= 7.0      # validation part
                        else:
                            r2['env_raw'] *= 7.0                                 # other session
                        pert['active'].append(r2)
                    tr2 = [r for r in pert['active'] if r['session'] in trs]
                    _, info2 = emg_norm.normalise(kind, pert, tr2, pert['active'])
                    if kind == 'session_p99':
                        src = trs[0]
                        tgt = [x for x in info['session_scale'] if x != src][0]
                        check(np.allclose(info['session_scale'][src], info2['session_scale'][src]),
                              f'{tag}: source-session scale ignores validation part and other session', out)
                        check(np.allclose(np.array(info2['session_scale'][tgt]), 7 * np.array(info['session_scale'][tgt])),
                              f'{tag}: target-session scale uses target-session EMG only', out)
                    else:
                        src = np.concatenate([r['env_raw'][:emg_norm._train_part(r)] for r in train_recs])
                        direct = src.max(0) if kind == 'trainmax_tr' else np.percentile(src, 99, axis=0)
                        check(np.allclose(info['scale'], info2['scale']), f'{tag}: fixed scale ignores val + other session', out)
                        check(np.allclose(info['scale'], direct), f'{tag}: fixed scale == direct recomputation', out)
                # balancing factors from training targets only
                knots = RT.angle_groups(data['train_angles'])
                k_tr = RT.snap(data['meta_train']['angle'], knots)
                sd = np.array([max(np.std(data['y_train'][k_tr == j].astype(np.float64)), 0.1) for j in range(len(knots))])
                check(np.all(np.isfinite(sd)) and all(np.any(k_tr == j) for j in range(len(knots))),
                      f'{tag}: every training position has training windows and a finite balancing SD', out)
                if exp == TT[0]:
                    check(nm_restore_check(data), f'{tag}: predictions multiplied back to Nm', out)
                n_bad = sum(not c['ok'] for c in out)
                print(f'{tag}: {len(out) - n_bad}/{len(out)} checks passed', flush=True)
                results.append({'subject': s, 'direction': d, 'experiment': exp, 'checks': out,
                                'balancing_sd_nm': sd.tolist(), 'emg': details['emg']})
    ok = all(c['ok'] for r in results for c in r['checks'])
    with open(os.path.join(EXP, 'results', 'total_torque_checks.json'), 'w') as f:
        json.dump({'all_passed': ok, 'runs': results}, f, indent=1, default=float)
    print('ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
