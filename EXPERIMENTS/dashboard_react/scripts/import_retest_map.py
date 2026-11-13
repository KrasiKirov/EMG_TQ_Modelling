"""Import the 0.5 s window + retest passive map runs into the presentation catalogue.

Reads the collection written by ``python -m ML.training.parallel_benchmark`` (one
folder per subject and seed). Only arrays and JSON are read; no model is loaded.
Every per-trial score is recomputed from the saved predictions and checked against
the run's own metrics.json.
Usage: python scripts/import_retest_map.py <run collection folder> [--workers 8]
"""
from datetime import datetime
from pathlib import Path
import argparse, base64, hashlib, json, os
import numpy as np

DASH = Path(__file__).resolve().parents[1]
ID = 'main/session-passive'
NOTES = [
    'YES: trials 27, 29, 31 and 33 are the retest pushes at p1–p4. Their notes in the recording wrongly say '
    '“passive”; their angles and torque match the session-1 pushes. Session-1 passive values come from the lab worksheet.',
    'IES: the recording has no passive trials. The passive table in the IES metadata document (measured before '
    'session 1) is used for both sessions.',
    'JM: the still-hold rule already ignores the movement in passive trial 21, so its passive map needs no fix.',
    'CH: added to the subject list; nothing else changed.',
]

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def metrics(y, p):
    e = p-y
    ss = np.sum((y-y.mean())**2)
    ed = (p-p.mean())-(y-y.mean())
    assert ss > 0
    return dict(r2=float(1-np.sum(e**2)/ss), mae=float(np.mean(abs(e))),
                rmse=float(np.sqrt(np.mean(e**2))), bias=float(np.mean(e)),
                r2_demeaned=float(1-np.sum(ed**2)/ss), mae_demeaned=float(np.mean(abs(ed))),
                peak=float(np.max(abs(y))))

def packed(a):
    return base64.b64encode(np.asarray(a, dtype='<f4').tobytes()).decode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--workers', type=int, default=8, help='jobs that ran at once (for the timing note)')
    args = parser.parse_args()
    folders = sorted(p.parent for p in args.runs.resolve().glob('*/config.json'))
    # A folder without metrics is a job still training (or interrupted); import only finished runs.
    jobs = [f for f in folders if any(f.glob('*/metrics.json'))]
    assert jobs, 'No finished runs found'
    if len(jobs) < len(folders):
        print(f'Skipping {len(folders)-len(jobs)} unfinished run folder(s)')
    data = read(DASH/'src/data/presentation-catalog.json')
    old = data['experiments'][ID]
    # Idempotent: replace this collection only; preserve all controlled and GRU data.
    data['rows'] = [r for r in data['rows'] if r['exp'] != ID]
    for field in ['positions','targets','predictions','means','times']:
        data[field] = {k:v for k,v in data[field].items() if not k.startswith(ID+'|')}
    raw_main = read(DASH/'src/data/results.json')
    t0,dt,n = raw_main['grid']; grid = t0+dt*np.arange(n)
    histories, native, coverage, created, configs = {}, [], {}, [], []
    worst_delta = 0.
    for job in jobs:
        cfg = read(job/'config.json')
        assert cfg['window'] == 50 and cfg['session_passive_calibration'] and cfg['evaluate_retest']
        assert cfg['target_mode'] == 'active_torque'
        (file,) = job.glob('*/test_predictions.npz')
        folder = file.parent
        meta = read(folder/'metrics.json')
        subject = folder.name.split('_')[0]; seed = int(meta['seed'])
        records = {r['trial']:r for r in read(job/f'{subject}_manifest.json')['trials']}
        mode = read(job/f'{subject}_preprocessing.json')['passive_calibration_mode']
        provenance = read(job/'provenance.json')
        created.append(provenance['created']); configs.append(cfg)
        z = np.load(file, allow_pickle=False)
        assert all(k in z for k in ['y_true','y_pred','measured_torque','subject','trial','position_id','time_s'])
        assert len({len(z[k]) for k in z.files}) == 1 and np.all(z['subject'] == subject)
        y,p = z['y_true'].astype(float),z['y_pred'].astype(float)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        pooled = metrics(y,p); orig = meta['splits']['test']['overall']
        worst_delta = max(worst_delta, *(abs(pooled[k]-orig[k]) for k in ['r2','mae','rmse','bias']))
        ident = f'{ID}|{subject}|S1toS2'; key = f'{ident}|{seed}'
        trials = sorted(np.unique(z['trial']), key=lambda tr: records[int(tr)]['mean_position_rad'])
        poses = [round(records[int(tr)]['mean_position_rad'],3) for tr in trials]
        assert len(set(poses)) == len(poses), 'Multiple trials at one angle need explicit handling'
        data['positions'][ident] = poses
        targets, predictions, trial_rmse = [], [], []
        expected = raw_main['positions'].get(f'{subject}|S1toS2',poses)
        for trial,pos in zip(trials,poses):
            mask = z['trial']==trial
            yy,pp,tt = y[mask],p[mask],z['time_s'][mask]
            assert np.all(np.diff(tt)>0)
            row = metrics(yy,pp)
            original_trial = meta['splits']['test']['trials'][f'{subject}/{trial}']
            worst_delta = max(worst_delta, *(abs(row[k]-original_trial[k]) for k in ['r2','mae','rmse','bias']))
            trial_rmse.append(row['rmse'])
            closest = min(range(len(expected)), key=lambda i:abs(expected[i]-pos))
            row.update(exp=ID,subj=subject,dir='S1toS2',seed=seed,pos=pos,
                       toes=int(closest<2),position_id=str(z['position_id'][mask][0]),
                       trial=int(trial),n=int(mask.sum()),time_start=float(tt[0]),time_end=float(tt[-1]))
            data['rows'].append(row)
            data['means'][f'{key}|{pos}'] = [float(yy.mean()),float(pp.mean())]
            # The first ~1.5 s is excluded by this pipeline. Do not extrapolate it.
            def sample(values):
                out=np.interp(grid,tt,values)
                out[(grid<tt[0])|(grid>tt[-1])]=np.nan
                # Preserve any excluded intervals as gaps, never bridge them.
                right=np.clip(np.searchsorted(tt,grid),1,len(tt)-1)
                out[(tt[right]-tt[right-1])>.015]=np.nan
                return out
            targets.append(sample(yy)); predictions.append(sample(pp))
        blob = packed(np.concatenate(targets)); target_id = hashlib.sha256(blob.encode()).hexdigest()[:16]
        data['targetPool'][target_id] = blob; data['targets'][key] = target_id
        data['predictions'][key] = packed(np.concatenate(predictions))
        data['times'][key] = dict(seconds=float(meta['training_seconds']),prep=None,
                                  definition='model construction + compilation + fit; excludes preparation/prediction/evaluation')
        histories[key] = read(folder/'history.json')
        native.append(dict(subject=subject,seed=seed,rmse=float(np.mean(trial_rmse)),worst=float(max(trial_rmse))))
        c = coverage.setdefault(subject, dict(positions=len(poses),seeds=[],positionIds=[],passiveMode=mode))
        assert c['positions'] == len(poses) and c['passiveMode'] == mode, f'{subject}: runs disagree on coverage'
        c['seeds'] = sorted(c['seeds']+[seed]); c['positionIds'] = sorted(set(c['positionIds'])|set(z['position_id'].tolist()))
    assert worst_delta < 1e-4, f'Recomputed scores differ from metrics.json by {worst_delta}'
    seeds = sorted({s for c in coverage.values() for s in c['seeds']})
    config = {k:v for k,v in configs[0].items() if k not in ['data_dir','output','overrides','subjects','seeds','plots']}
    config.update(subjects=sorted(coverage), seeds=seeds)
    subjects = sorted({k.split('|')[0] for k in raw_main['positions']})
    entry = dict(id=ID, title='0.5 s window + retest passive map', group='0.5 s window + retest map',
        kind='retest-map', native=ID, model='lstm', sessionPassive=True, params=dict(old['params'],window=.5),
        config=config, revision=read(jobs[0]/'provenance.json')['git_head'][:7],
        purpose='Predict voluntary ankle torque from the last 0.5 s of muscle activity, scored against a target '
                'whose passive part is re-measured in the test session.',
        target='Active torque = measured torque − passive torque at that angle. Session 1 (training) uses its own passive map; '
               'session 2 (test) uses a passive map measured again in session 2.',
        protocol=f'Train on session 1, test on session 2. 50 samples at 100 Hz (0.5 s) of four muscle envelopes plus angle; '
                 f'LSTM 8×2, dropout 0.3; up to 60 epochs, patience 12, batch 64; seeds {seeds[0]}–{seeds[-1]}.',
        collection=dict(date=max(datetime.fromisoformat(c) for c in created).astimezone().strftime('%Y-%m-%d'), runs=len(jobs), rows=sum(r['exp']==ID for r in data['rows']),
            coverage=coverage, notes=NOTES, native=native, histories=histories,
            unavailable={s:dict(reason='Not run for this method') for s in subjects if s not in coverage},
            environment=dict(platform=read(jobs[0]/'provenance.json')['platform'], cpu_count_visible=os.cpu_count(),
                             concurrency=f'{args.workers} jobs at a time (ML/training/parallel_benchmark.py)',
                             threads_per_job='1 CPU thread',
                             training_duration_definition='wall-clock seconds around model construction, compile and fit')))
    data['experiments'][ID] = entry
    data.pop('teammateImported', None)
    used = set(data['targets'].values()); data['targetPool'] = {k:v for k,v in data['targetPool'].items() if k in used}
    (DASH/'src/data/presentation-catalog.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(runs=len(jobs),subjects={s:dict(positions=c['positions'],seeds=len(c['seeds']),passive=c['passiveMode'])
                                                 for s,c in coverage.items()},
                          position_scores=entry['collection']['rows'],max_metric_difference=worst_delta),indent=2))

if __name__=='__main__':main()
