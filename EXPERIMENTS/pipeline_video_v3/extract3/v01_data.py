"""Real data for the v3 video (windows, the fit/validation split, shuffling, the 10 s calibration).

    python extract3/v01_data.py            # project venv (.venv311), PYTHONPATH = repository root, EMG_DATA_DIR set

Writes data3/v01.npz and data3/facts_v01.json. No training: it reads the recordings (ML.research.data) and the
saved N0 runs (EXPERIMENTS/results/N0_total_trainmax), the final model before calibration.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIDEO = HERE.parent
ROOT = VIDEO.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'EXPERIMENTS' / 'normalization'))
from ML.research.data import load_subject            # noqa: E402
from calibration_test import correct                 # noqa: E402

RES = ROOT / 'EXPERIMENTS' / 'results' / 'N0_total_trainmax'
WINDOW, TRAIN_FRAC = 20, 0.8


def main():
    out, facts = {}, {}

    # one session-1 trial of EG, as the final model sees it: envelopes / session-1 maximum, angle, total torque
    b = load_subject('EG')
    s1 = [r for r in b['active'] if r['session'] == 'S1']
    emg_max = np.max(np.stack([r['env_max1k'] for r in s1]), axis=0)
    rec = min(s1, key=lambda r: abs(r['angle'] - (-0.038)))
    env = rec['env_raw'] / emg_max
    n = len(rec['tq'])
    nwin = n - WINDOW + 1
    n_fit = int(nwin * TRAIN_FRAC)
    out.update(eg_t=rec['t'] - rec['t'][0], eg_env=env.astype(np.float32), eg_angle=rec['pos'].astype(np.float32),
               eg_tq=rec['tq'].astype(np.float32))
    facts['eg_trial'] = dict(angle_rad=round(float(rec['angle']), 3), samples=int(n), seconds=round(n / 100, 1),
                             windows=int(nwin), fit_windows=n_fit, val_windows=int(nwin - n_fit),
                             fit_seconds=round(n_fit / 100, 1), val_seconds=round((nwin - n_fit) / 100, 1),
                             cut_sample=int(n_fit + WINDOW - 1), cut_time_s=round((n_fit + WINDOW - 1) / 100, 2),
                             tq_at_cut=round(float(rec['tq'][n_fit + WINDOW - 1]), 2),
                             tq_range=[round(float(rec['tq'].min()), 1), round(float(rec['tq'].max()), 1)],
                             session1_trials=len(s1), session1_angles=sorted(round(float(r['angle']), 3) for r in s1))

    # window counts and epochs of the final model's runs (N0, 6 people x 3 seeds)
    runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(RES / '*' / 'S1toS2' / 'seed*.json')))]
    facts['n_windows'] = runs[0]['n_windows']
    facts['same_counts_all_runs'] = all(r['n_windows'] == runs[0]['n_windows'] for r in runs)
    ep = [r['train_info']['epochs'] for r in runs]
    best = [r['train_info']['best_epoch'] for r in runs]
    facts['n0_epochs'] = dict(runs=len(runs), epochs_min=min(ep), epochs_max=max(ep), best_min=min(best),
                              best_max=max(best), patience=25, epoch_limit=200, batch=8,
                              steps_per_epoch=runs[0]['n_windows']['train'] // 8,
                              all_stop_best_plus_25=all(e == b_ + 25 for e, b_ in zip(ep, best)))
    out['n0_epochs'], out['n0_best'] = np.array(ep), np.array(best)

    # calibration candidates: every session-2 trial of the N0 runs, seed 0
    cands = []
    for f in sorted(glob.glob(str(RES / '*' / 'S1toS2' / 'seed0.npz'))):
        subj = Path(f).parts[-3]
        z = np.load(f)
        y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
        angles = sorted(np.unique(ang))
        for a in angles:
            m = ang == a
            yy, pp, tt = y[m], p[m], t[m] - t[m][0]
            cal, score = tt < 10, tt >= 30
            pc = correct(pp[cal], yy[cal], pp, 'gain+offset')
            ga, _ = np.polyfit(pp[cal], yy[cal], 1)
            cands.append(dict(subject=subj, angle=round(float(a), 3), toes=int(a in angles[:2]),
                              mae_raw=float(np.mean(abs(pp - yy)[score])), mae_cal=float(np.mean(abs(pc - yy)[score])),
                              gain=float(np.clip(ga, 0.5, 2)), sd_cal=float(np.std(yy[cal])), swing=float(np.ptp(yy))))
    cands.sort(key=lambda c: c['mae_raw'] - c['mae_cal'], reverse=True)

    # the calibration example shown in the video: YES, session 2, -0.381 rad, seed 0
    z = np.load(RES / 'YES' / 'S1toS2' / 'seed0.npz')
    y, p, ang, t = (z[k].astype(float) for k in ('y_own_target', 'y_pred', 'angle', 't'))
    m = np.isclose(ang, -0.381, atol=0.01)
    yy, pp, tt = y[m], p[m], t[m] - t[m][0]
    cal, score = tt < 10, tt >= 30
    ga, gb = np.polyfit(pp[cal], yy[cal], 1)
    a_ = float(np.clip(ga, 0.5, 2))
    b_ = float(np.mean(yy[cal] - a_ * pp[cal]))
    pc = a_ * pp + b_
    out.update(cal_t=tt.astype(np.float32), cal_y=yy.astype(np.float32), cal_p=pp.astype(np.float32),
               cal_pc=pc.astype(np.float32))
    facts['calibration_example'] = dict(subject='YES', angle_rad=-0.381, seed=0, gain=round(a_, 3), offset=round(b_, 2),
                                        mae_raw_scored=round(float(np.mean(abs(pp - yy)[score])), 2),
                                        mae_cal_scored=round(float(np.mean(abs(pc - yy)[score])), 2),
                                        bias_raw_scored=round(float(np.mean((pp - yy)[score])), 2),
                                        seconds=round(float(tt[-1]), 1))
    facts['group'] = dict(toes_mae=[0.86, 0.35], toes_r2=[0.81, 0.94], other_mae=[1.01, 0.75], people_improved='6 of 6',
                          source='EXPERIMENTS/normalization/calibration_summary.md (gain+offset, 10 s, scored from 30 s)')
    facts['calibration_candidates'] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()}
                                       for c in cands[:15]]

    (VIDEO / 'data3').mkdir(exist_ok=True)
    np.savez_compressed(VIDEO / 'data3' / 'v01.npz', **out)
    (VIDEO / 'data3' / 'facts_v01.json').write_text(json.dumps(facts, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in facts.items() if k != 'calibration_candidates'}, indent=1))
    for c in facts['calibration_candidates']:
        print(c)


if __name__ == '__main__':
    main()
