"""Summarize completed paired trials; never select seeds or omit positions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'reports/retest_mechanisms_20261019'

def main():
    frames=[];timings=[]
    for path in sorted((ROOT/'experiment').glob('*/*/seed*/metrics.csv')):
        # Metrics files can precede run completion; use completed runs only.
        if not (path.parent/'run.json').exists():continue
        frames.append(pd.read_csv(path));run=json.loads((path.parent/'run.json').read_text())
        epochs=json.loads((path.parent/'epoch_timings.json').read_text())
        run.update(epoch_wall_median=np.median([r['wall_s'] for r in epochs]),
                   epoch_wall_max=max(r['wall_s'] for r in epochs),
                   max_wall_minus_monotonic=max(abs(r['wall_s']-r['monotonic_s']) for r in epochs))
        timings.append(run)
    if not frames:return
    allrows=pd.concat(frames,ignore_index=True)
    allrows.to_csv(ROOT/'exploratory_all_metrics.csv',index=False)
    pd.DataFrame(timings).to_csv(ROOT/'run_timings.csv',index=False)
    paired=[];gates=[]
    for (s,seed),group in allrows.groupby(['subject','seed']):
        modes=set(group['mode'])-{'baseline'}
        for candidate in modes:
            a=group[group['mode']=='baseline'];b=group[group['mode']==candidate]
            m=a.merge(b,on=['subject','seed','split','position'],suffixes=('_baseline','_candidate'))
            if len(m)!=len(a) or len(m)!=len(b):raise ValueError('Incomplete matched positions')
            m['candidate']=candidate
            for key in ['r2','rmse','mae','bias','residual_sd','vaf']:
                m['delta_'+key]=m[key+'_candidate']-m[key+'_baseline']
            paired.append(m)
            v=m[(m.split=='validation')&(m.position!='pooled')]
            # Use the candidate target variance for both residual scores. For
            # calibration experiments, translating baseline prediction and target
            # together into the candidate reference leaves residuals unchanged.
            denom=v.target_sd_candidate**2
            ratio=(v.mse_candidate/denom).mean()/(v.mse_baseline/denom).mean()
            worst=(v.rmse_candidate/v.rmse_baseline).max()
            eligible=bool(ratio<=.9 and worst<=1.1)
            gates.append(dict(subject=s,seed=int(seed),candidate=candidate,
                validation_relative_mse_ratio=ratio,worst_position_validation_rmse_ratio=worst,
                validation_gate_pass=eligible,
                decision='Eligible for review of all-position retest trade-offs before confirmation' if eligible else 'Do not launch ten-seed confirmation: prespecified validation gate failed'))
            r=m[(m.split=='retest')&(m.position!='pooled')].copy()
            r['position']=r.position.astype(float);r=r.sort_values('position')
            fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
            for j,key in enumerate(['r2','rmse','bias']):
                for suffix,label in [('baseline','Baseline'),('candidate',candidate)]:
                    ax[j].plot(r.position,r[key+'_'+suffix],'-o',label=label,ms=4)
                ax[j].set_xlabel('Angle (rad)');ax[j].set_ylabel({'r2':'R²','rmse':'RMSE (Nm)','bias':'Prediction − target (Nm)'}[key]);ax[j].axhline(0,color='k',lw=.5)
            ax[0].legend(fontsize=8);fig.suptitle(f'{s}: every retest position | matched seed {seed}, exploratory')
            fig.savefig(ROOT/f'{s}_seed{seed}_all_positions.png',dpi=150);plt.close(fig)
    if paired:pd.concat(paired,ignore_index=True).to_csv(ROOT/'paired_all_positions.csv',index=False)
    (ROOT/'confirmation_gate.json').write_text(json.dumps(gates,indent=2))
    print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
