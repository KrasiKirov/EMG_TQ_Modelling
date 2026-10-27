"""Validate and import a teammate handoff directory. Reads arrays; never loads models or trains.
Usage: python scripts/import_teammate.py <extracted dashboard_export_20261026 folder>
"""
from pathlib import Path
import argparse, base64, hashlib, json
import numpy as np

DASH = Path(__file__).resolve().parents[1]
ID = 'main/session-passive'

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
    parser.add_argument('export', type=Path)
    args = parser.parse_args()
    root = args.export.resolve()
    inv = read(root/'experiment_inventory.json')['requested_experiment']
    configs = [p for p in root.glob('*/config.json') if read(p).get('session_passive_calibration')]
    assert len(configs) == 1, 'Expected one requested run collection'
    run_root = configs[0].parent
    cfg = read(configs[0])
    assert cfg['window'] == 50 and cfg['seeds'] == [11,23,37]
    assert cfg['target_mode'] == 'active_torque' and cfg['evaluate_retest']
    data = read(DASH/'src/data/presentation-catalog.json')
    old = data['experiments'][ID]
    # Idempotent: replace this collection only; preserve all controlled and GRU data.
    data['rows'] = [r for r in data['rows'] if r['exp'] != ID]
    for field in ['positions','targets','predictions','means','times']:
        data[field] = {k:v for k,v in data[field].items() if not k.startswith(ID+'|')}
    raw_main = read(DASH/'src/data/results.json')
    t0,dt,n = raw_main['grid']; grid = t0+dt*np.arange(n)
    check_rows, histories, native, coverage = [], {}, [], {}
    files = sorted(run_root.glob('*/test_predictions.npz'))
    assert len(files) == 12
    for file in files:
        folder = file.parent
        meta = read(folder/'metrics.json')
        subject = folder.name.split('_')[0]; seed = int(meta['seed'])
        manifest = read(run_root/f'{subject}_manifest.json')
        records = {r['trial']:r for r in manifest['trials']}
        z = np.load(file, allow_pickle=False)
        required = ['y_true','y_pred','measured_torque','subject','trial','position_id','sample_index','time_s']
        assert all(k in z for k in required)
        assert len({len(z[k]) for k in z.files}) == 1
        assert np.all(z['subject'] == subject)
        y,p = z['y_true'].astype(float),z['y_pred'].astype(float)
        assert np.isfinite(y).all() and np.isfinite(p).all()
        pooled = metrics(y,p); orig = meta['splits']['test']['overall']
        delta = max(abs(pooled[k]-orig[k]) for k in ['r2','mae','rmse','bias'])
        assert delta < 1e-4, (folder.name,delta)
        ident = f'{ID}|{subject}|S1toS2'; key = f'{ident}|{seed}'
        trials = sorted(np.unique(z['trial']), key=lambda tr: records[int(tr)]['mean_position_rad'])
        poses = [round(records[int(tr)]['mean_position_rad'],3) for tr in trials]
        assert len(set(poses)) == len(poses), 'Multiple trials at one angle need explicit handling'
        data['positions'][ident] = poses
        targets,predictions, trial_rmse = [],[],[]
        expected = raw_main['positions'].get(f'{subject}|S1toS2',poses)
        for trial,pos in zip(trials,poses):
            mask = z['trial']==trial
            yy,pp,tt = y[mask],p[mask],z['time_s'][mask]
            assert np.all(np.diff(tt)>0)
            row = metrics(yy,pp)
            original_trial=meta['splits']['test']['trials'][f'{subject}/{trial}']
            trial_delta=max(abs(row[k]-original_trial[k]) for k in ['r2','mae','rmse','bias'])
            assert trial_delta<1e-4, (folder.name,int(trial),trial_delta)
            trial_rmse.append(row['rmse'])
            closest = min(range(len(expected)), key=lambda i:abs(expected[i]-pos))
            row.update(exp=ID,subj=subject,dir='S1toS2',seed=seed,pos=pos,
                       toes=int(closest<2),position_id=str(z['position_id'][mask][0]),
                       trial=int(trial),n=int(mask.sum()),time_start=float(tt[0]),time_end=float(tt[-1]))
            data['rows'].append(row)
            data['means'][f'{key}|{pos}'] = [float(yy.mean()),float(pp.mean())]
            # The first ~1.5 s is excluded by this audited pipeline. Do not extrapolate it.
            def sample(values):
                out=np.interp(grid,tt,values)
                out[(grid<tt[0])|(grid>tt[-1])]=np.nan
                # Preserve any excluded intervals as gaps, never bridge them.
                right=np.clip(np.searchsorted(tt,grid),1,len(tt)-1)
                out[(tt[right]-tt[right-1])>.015]=np.nan
                return out
            targets.append(sample(yy));predictions.append(sample(pp))
        blob = packed(np.concatenate(targets)); target_id=hashlib.sha256(blob.encode()).hexdigest()[:16]
        data['targetPool'][target_id]=blob;data['targets'][key]=target_id
        data['predictions'][key]=packed(np.concatenate(predictions))
        data['times'][key]=dict(seconds=float(meta['training_seconds']),prep=None,
                               definition='model construction + compilation + fit; excludes preparation/prediction/evaluation')
        histories[key]=read(folder/'history.json')
        native.append(dict(subject=subject,seed=seed,rmse=float(np.mean(trial_rmse)),worst=float(max(trial_rmse))))
        coverage[subject]=dict(positions=len(poses),seeds=cfg['seeds'],positionIds=sorted(set(z['position_id'].tolist())))
        check_rows.append(dict(run=folder.name,samples=len(y),positions=len(poses),max_metric_difference=delta,
                               prediction_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    config_public = {k:v for k,v in cfg.items() if k not in ['data_dir','output','overrides']}
    environment = read(root/'environment_extra.json')
    entry = dict(old,kind='teammate',title='Teammate · 0.5 s LSTM',group='Teammate · reproduced 26 Oct',
        purpose='Reproduce the audited 0.5 s LSTM with reviewed trial decisions and retest-session passive calibration.',
        model='lstm',config=config_public,revision=inv['source_revision'],
        target='Active torque = measured torque − the applicable passive map. Training uses the source-session curve; evaluation uses the retest-session curve.',
        protocol='Source session → retest; 50 samples at 100 Hz; seeds 11, 23, 37; maximum 60 epochs, patience 12, batch 64.',
        params=dict(old['params'],window=.5),
        handoff=dict(date='2026-10-26',status=inv['status'],coverage=coverage,
                     unavailable=read(root/'unavailable_and_coverage.json'),environment=environment,
                     runs=len(files),rows=len([r for r in data['rows'] if r['exp']==ID]),
                     native=native,histories=histories,checks=check_rows,
                     archive='dashboard_export_20261026.zip',
                     historicalNote='An older 0.2 s, source-map-only collection is preserved in the handoff archive and excluded from the presentation.'))
    data['experiments'][ID]=entry
    used=set(data['targets'].values());data['targetPool']={k:v for k,v in data['targetPool'].items() if k in used}
    data['teammateImported']='2026-10-26'
    output=DASH/'src/data/presentation-catalog.json'
    output.write_text(json.dumps(data,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(runs=len(files),subjects=coverage,position_scores=entry['handoff']['rows'],
                         max_metric_difference=max(r['max_metric_difference'] for r in check_rows)),indent=2))

if __name__=='__main__':main()
