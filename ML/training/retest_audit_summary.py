"""Aggregate the source audit without fitting corrections to active retest labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ML.preprocessing.dataset_builder import lookup_passive_torque

ROOT=Path(__file__).resolve().parents[1]/'reports/retest_mechanisms_20261019'
rows=[]; calibration=[];quality=[]
for s in ['YES','EG','IES','JM','HM']:
    d=ROOT/'audit'/s
    a=json.loads((d/'raw_audit.json').read_text())
    if 'independently_labelled_passive_maps' in a:
        a['raw_label_passive_maps']=a.pop('independently_labelled_passive_maps')
    a['caveat']='YES raw-label maps include inferred active records and are NOT valid calibration maps. Other maps still require source-quality review.'
    (d/'raw_audit.json').write_text(json.dumps(a,indent=2))
    df=pd.read_csv(d/'historical_decomposition.csv')
    for pos,group in df.groupby('position'):
        row=dict(subject=s,position=pos,n=int(group.n.iloc[0]),seeds=len(group))
        for key in ['r2','vaf','rmse','mae','abs_bias','residual_sd','target_sd','bias_fraction']:
            row[key+'_mean']=group[key].mean();row[key+'_seed_sd']=group[key].std(ddof=1)
        rows.append(row)
    traces={split:np.load(d/f'{split}_traces.npz') for split in ['train','validation','retest']}
    for trial in a['trials']:
        for split,z in traces.items():
            m=z['trial']==trial['trial']
            if not m.any():continue
            row=dict(subject=s,trial=trial['trial'],split=split,comment=trial['header']['comment'],
                position_sd_raw=trial['position_sd'],n=int(m.sum()),
                measured_sd=z['measured'][m].std(),active_sd=z['target'][m].std(),
                passive_sd=z['passive'][m].std(),passive_range=np.ptp(z['passive'][m]),
                passive_sd_without_last_second=z['passive'][m & (z['time']<89)].std(),
                within_trial_median_dt=float(np.median(np.diff(z['time'][m]))))
            for c,x in trial['channels'].items():
                row[c+'_extrema_fraction']=x['fraction_at_observed_extrema']
                row[c+'_equal_fraction']=x['consecutive_equal_fraction']
            quality.append(row)
    dist=pd.read_csv(d/'distributions.csv')
    for pos in dist.position.unique():
        maps=a['raw_label_passive_maps']
        if s=='YES':
            meta=json.loads((d/'metadata.json').read_text())
            rec=meta['reconstruction']['retest_records']
            first=sorted((r['mean_angle_rad'],p) for r,p in zip(rec,[2.6,-10.1,4.3,.35,1.8,7.9,-2.6,-3.5]))
            second=meta['passive_entries']
        else:first,second=maps['test'],maps['retest']
        if first and second:
            t1=lookup_passive_torque(first,pos);t2=lookup_passive_torque(second,pos)
            calibration.append(dict(subject=s,position=pos,session1_passive=t1,session2_passive=t2,
                 implied_training_target_increase=t2-t1,
                 source='worksheet' if s=='YES' else 'raw labelled passive trials; interpolation/extrapolation',
                 session1_extrapolation=pos<first[0][0] or pos>first[-1][0],
                 session2_extrapolation=pos<second[0][0] or pos>second[-1][0]))
pd.DataFrame(rows).to_csv(ROOT/'historical_all_positions.csv',index=False)
pd.DataFrame(calibration).to_csv(ROOT/'independent_calibration_differences.csv',index=False)
pd.DataFrame(quality).to_csv(ROOT/'signal_quality.csv',index=False)
print(pd.DataFrame(rows).groupby('subject').head(1)[['subject','position','r2_mean','r2_seed_sd','rmse_mean','rmse_seed_sd','abs_bias_mean','target_sd_mean','residual_sd_mean','bias_fraction_mean']].to_string(index=False))

# Descriptive conditional EMG comparison; these bins never train a correction.
conditioned=[]
for s,strong in [('EG',-.038),('IES',.144),('JM',.055)]:
    folder=ROOT/'audit'/s
    traces={k:np.load(folder/f'{k}_traces.npz') for k in ['train','validation','retest']}
    positions=json.loads((folder/'metadata.json').read_text())['operating_positions']
    for pos in [min(positions),min(positions,key=lambda p:abs(p-strong))]:
        masks={k:np.argmin(abs(z['position'][:,None]-np.array(positions)[None,:]),axis=1)==positions.index(pos)
               for k,z in traces.items()}
        first_session=np.concatenate([traces[k]['target'][masks[k]] for k in ['train','validation']])
        bounds=np.percentile(first_session,[0,25,50,75,100]);bounds[0]=-np.inf;bounds[-1]=np.inf
        for split,z in traces.items():
            for quartile in range(4):
                mask=masks[split] & (z['target']>=bounds[quartile]) & (z['target']<bounds[quartile+1])
                if not mask.any():continue
                row=dict(subject=s,position=pos,split=split,training_target_quartile=quartile+1,
                         n=int(mask.sum()),target_mean=z['target'][mask].mean())
                row.update({c+'_mean':z['emg'][mask,j].mean() for j,c in enumerate(['MG','LG','SOL','TA'])})
                conditioned.append(row)
pd.DataFrame(conditioned).to_csv(ROOT/'emg_conditioned_on_training_torque_bins.csv',index=False)
