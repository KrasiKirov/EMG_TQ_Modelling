import pandas as pd, numpy as np
R = pd.read_csv('ladder_runs.csv'); R['r2'] = R.r2.clip(lower=0)
sel = {('X00f_baseline_framework',0):'baseline', ('A0_active_trainmax',0):'balanced', ('A0_active_trainmax',1):'balanced+cal', ('X00f_baseline_framework',1):'baseline+cal'}
for (m,c),name in sel.items():
    for td in (1,0):
        x = R[(R.model==m)&(R.cal==c)&(R.toes_down==td)]
        print(f'{name:13s} td={td} MAE 5-95 {np.percentile(x.mae,5):.2f}-{np.percentile(x.mae,95):.2f}  1-99 {np.percentile(x.mae,1):.2f}-{np.percentile(x.mae,99):.2f}  R2 5-95 {np.percentile(x.r2,5):.2f}-{np.percentile(x.r2,95):.2f}  n={len(x)}')
