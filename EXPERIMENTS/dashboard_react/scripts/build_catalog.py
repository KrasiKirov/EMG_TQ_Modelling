"""Export saved historical predictions and timing metadata; never trains a model.
Run from dashboard_react: python scripts/build_catalog.py.
The main generated results.json remains untouched. Additional data and provenance
go to src/data/catalog.json. Report-only main entries retain their native metric.
"""
from pathlib import Path
import sys,json,base64,subprocess,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DASH=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'EXPERIMENTS/code'))
import presentation_metrics as PM
from ML.research.experiments import get as get_config

GRID=np.round(np.arange(.5,88.9,.2),2)
main=json.loads((DASH/'src/data/results.json').read_text())
out={'experiments':{},'rows':[],'positions':{},'targets':{},'targetPool':{},'predictions':{},'means':{},'times':{},'grid':list(main['grid'])}

def packed(a):
    return base64.b64encode(np.asarray(a,dtype='<f4').tobytes()).decode()
def at_grid(t,y):
    return y[np.clip(np.searchsorted(t,GRID),0,len(t)-1)]
def timing(exp,subject,direction,seed,sec,prep=None,definition='fit + prediction'):
    if sec is not None and np.isfinite(sec):
        out['times'][f'{exp}|{subject}|{direction}|{seed}']={'seconds':float(sec),'prep':prep,'definition':definition}
def add(exp,subject,direction,seed,trials):
    trials=sorted(trials,key=lambda x:x[0])
    ident=f'{exp}|{subject}|{direction}'
    poses=[round(float(x[0]),3) for x in trials]
    out['positions'][ident]=poses
    targets=[];predictions=[]
    for index,(pos,t,y,p) in enumerate(trials):
        pos=round(float(pos),3); y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
        metrics=PM.metrics(y,p)
        out['rows'].append(dict(exp=exp,subj=subject,dir=direction,seed=int(seed),pos=pos,
            r2=metrics['r2'],mae=metrics['mae'],rmse=metrics['rmse'],bias=metrics['bias'],
            r2_demeaned=metrics['r2_demeaned'],mae_demeaned=metrics['mae_demeaned'],peak=metrics['peak_torque'],toes=int(index<2)))
        out['means'][f'{ident}|{seed}|{pos}']=[float(y.mean()),float(p.mean())]
        targets.append(at_grid(t,y));predictions.append(at_grid(t,p))
    blob=packed(np.concatenate(targets));target_id=hashlib.sha256(blob.encode()).hexdigest()[:16]
    out['targetPool'][target_id]=blob
    out['targets'][f'{ident}|{seed}']=target_id
    out['predictions'][f'{ident}|{seed}']=packed(np.concatenate(predictions))

# Main experiment times and full-resolution means for an honest demeaning view.
for exp in main['experiments']:
    for f in (ROOT/'EXPERIMENTS/results'/exp).glob('*/S*to*/seed*.json'):
        r=json.loads(f.read_text());s,d=f.parts[-3:-1];seed=int(f.stem[4:])
        sec=r.get('train_s',r.get('train_info',{}).get('train_s'))
        timing(exp,s,d,seed,sec,r.get('prep_s'),definition='fit + prediction + evaluation' if exp=='X00_baseline' else 'fit + prediction')
        npz=f.with_suffix('.npz')
        if npz.exists():
            for pos,t,y,p in PM.trials(exp,s,str(npz)):
                out['means'][f'{exp}|{s}|{d}|{seed}|{round(pos,3)}']=[float(np.mean(y)),float(np.mean(p))]
print('Main timings/means',len(out['times']),flush=True)

family_names={'H_pgain_posw':'LSTM 8×2 · family reference','F_gru8':'GRU 8×2','F_lstm32':'LSTM 32×2','F_mlp':'MLP · feed-forward network','F_tcn':'TCN · temporal convolution'}
for model,title in family_names.items():
    eid='family/'+model
    for f in (ROOT/'ML/experiments/results').glob(f'*__{model}__fast.npz'):
        meta=json.loads(f.with_suffix('.json').read_text());s=meta['subject'];opts=meta['opts'];z=np.load(f)
        out['experiments'].setdefault(eid,{'id':eid,'title':title,'group':'Model families · report 1','kind':'family','native':model,
            'purpose':'Compare saved network configurations on the same family-study data and targets. Architecture-specific dropout settings differ; this is not a perfectly isolated cell-type change.',
            'model':meta['model'],'config':opts,'target':'Each saved run’s own training target. Compare within this family study; not a direct ranking against the controlled experiments.',
            'protocol':'Fast: batch 64, up to 150 epochs, patience 15; 0.2 s history; source session 1 → retest session 2.',
            'params':dict(code='family',passive='gain',emg='max',weighting='weights',window=opts['window']/100,dropout={'F_lstm32':.2,'F_mlp':.2,'F_tcn':.1}.get(model,.3),lr=.003,extra='none')})
        for k,pred in enumerate(z['preds']):
            seed=int(meta['seeds'][k]['seed']);tr=[]
            for trial in np.unique(z['test_trial']):
                mask=z['test_trial']==trial;pos=np.median(z['test_pos'][mask]);t=(np.arange(mask.sum())+opts['window']-1)/100
                tr.append((pos,t,z['y_test'][mask],pred[mask]))
            add(eid,s,'S1toS2',seed,tr)
            timing(eid,s,'S1toS2',seed,meta['seeds'][k]['info'].get('train_s'),definition='model.fit only; excludes prediction/preparation')
print('Families exported',flush=True)

# Keep studies separate: screening and confirmation are different run collections.
table=main['round2'];ix={c:i for i,c in enumerate(table['cols'])}
groups=sorted({(r[ix['study']],r[ix['experiment']]) for r in table['rows']})
for study,model in groups:
    eid=f'research/{study}/{model}';cfg=get_config(model)
    for f in (ROOT/'ML/research/results'/study).glob(f'*/S*to*/{model}/fast_seed*.npz'):
        meta=json.loads(f.with_suffix('.json').read_text());s=meta['subject'];d=meta['protocol'];seed=meta['seed']
        out['experiments'].setdefault(eid,{'id':eid,'title':model.replace('_',' '),'group':'Earlier study · '+study,'kind':'research','native':model,
            'purpose':cfg['q'],'model':cfg['model'],'config':cfg,'target':'Clean passive-trial map of the tested session, as used for the controlled experiment scores. Training labels may use a different passive model.',
            'protocol':'Fast: batch 64, up to 150 epochs, patience 15. Screening, confirmation and ablations remain separate entries.',
            'params':dict(code='research',passive=cfg['passive'],emg=cfg['emg'],weighting=cfg['balance'],window=cfg['window']/100,dropout=.3,lr=.003,extra='none')})
        add(eid,s,d,seed,list(PM.trials(model,s,str(f))))
        info=meta['train_info'];timing(eid,s,d,seed,info.get('train_s'),info.get('prep_s'))
    print(study,model,flush=True)

# Report-only teammate experiments. Do not invent per-run values or waveforms.
git=['git','-c','safe.directory='+str(ROOT).replace('\\','/')]
revision=subprocess.check_output(git+['rev-parse','--short','origin/main'],cwd=ROOT,text=True).strip()
reports={f:subprocess.check_output(git+['show','origin/main:ML/'+f],cwd=ROOT,text=True,encoding='utf-8') for f in ['WINDOW50_VALIDATION.md','PASSIVE_RECALIBRATION_AUDIT.md']}
base_rows={20:[],50:[]}
for line in reports['WINDOW50_VALIDATION.md'].splitlines():
    cells=[x.strip() for x in line.strip('|').split('|')]
    if len(cells)==7 and cells[0] in main['subjects']:
        for w,i in [(20,1),(50,2)]:base_rows[w].append({'subject':cells[0],'rmse':float(cells[i]),'worst':float(cells[4 if w==20 else 5])})
session_rows=[]
for line in reports['PASSIVE_RECALIBRATION_AUDIT.md'].splitlines():
    cells=[x.strip() for x in line.strip('|').split('|')]
    if len(cells)==4 and cells[0] in main['subjects']:session_rows.append({'subject':cells[0],'sourceRmse':float(cells[1]),'rmse':float(cells[2])})
for id,title,window,rows,session in [('main/window20','Teammate · 0.2 s reference',.2,base_rows[20],False),('main/window50','Teammate · 0.5 s history',.5,base_rows[50],False),('main/session-passive','Teammate · 0.5 s + retest passive map',.5,session_rows,True)]:
    out['experiments'][id]={'id':id,'title':title,'group':'Teammate · main','kind':'report','native':id,'model':'LSTM 8×2','revision':revision,'reportRows':rows,'sessionPassive':session,
        'purpose':'Use the tested session’s passive calibration for the retest target.' if session else 'Compare 20 versus 50 samples of history under the teammate’s audited benchmark.',
        'target':'Retest measured torque minus retest-session passive map.' if session else 'Retest measured torque minus source-session passive map.',
        'protocol':'60 epochs maximum, patience 12, batch 64, train stride 5; paired seeds 11, 23 and 37. The within-subject LSTM target is standardised using the training-set mean and standard deviation, then transformed back to Nm.',
        'params':dict(code='main',passive='session' if session else 'source',emg='max',weighting='none',window=window,dropout=.3,lr=.003,extra='none'),
        'reportText':reports['PASSIVE_RECALIBRATION_AUDIT.md' if session else 'WINDOW50_VALIDATION.md']}

path=DASH/'src/data/catalog.json'
path.write_text(json.dumps(out,separators=(',',':'),allow_nan=False),encoding='utf-8')
print('Saved',len(out['experiments']),'extra entries,',len(out['rows']),'scores,',len(out['times']),'timings,',round(path.stat().st_size/1e6,1),'MB',flush=True)
