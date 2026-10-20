"""Reproducible, isolated retest audit and matched exploratory experiments.

From repository parent: .venv/bin/python -m ML.training.retest_mechanisms
See the saved manifest for the frozen protocol and output directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from ML.config import SUBJECTS, DATA_DIR
from ML.training import load_subject, train_model, set_global_seed
from ML.preprocessing.flb_reader import read_flb, _read_header, _read_data
from ML.preprocessing.dataset_builder import classify_trials, get_passive_torque_map, lookup_passive_torque
from ML.evaluation.metrics import compute_metrics, snap_to_operating_points

ROOT = Path(__file__).resolve().parents[1]
FOCUS = {'YES': [-.486, -.381, .138], 'EG': [-.739, -.038],
         'IES': [-.591, .144], 'JM': [-.787, .055], 'HM': [-.789, .052]}
SPLITS = {'train': ('X_train', 'y_train'), 'validation': ('X_val', 'y_val'),
          'retest': ('X_test', 'y_test')}


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False))


def sha(path):
    with open(path, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape or not len(y):
        raise ValueError('Metrics require matching, nonempty one-dimensional arrays')
    assert np.isfinite(y).all() and np.isfinite(pred).all()
    residual = pred - y
    mse, bias, variance = np.mean(residual**2), residual.mean(), residual.var(ddof=0)
    assert np.isclose(mse, bias**2 + variance, rtol=1e-12, atol=1e-12)
    out = compute_metrics(y, pred)
    # Independent definitions, not ratios of averaged metrics.
    np.testing.assert_allclose(out['r2'], 1 - mse / y.var(), rtol=1e-12, atol=1e-12)
    return dict(out, n=len(y), mse=mse, bias=bias, residual_sd=np.sqrt(variance),
                target_sd=y.std(ddof=0), bias_fraction=bias**2/mse if mse else 0.)


def bounded_weights(X, y, positions):
    snapped, known = snap_to_operating_points(X[:, -1, 4], positions)
    iqrs = {p: max(float(np.subtract(*np.percentile(y[snapped == p], [75, 25]))), .1)
            for p in known if np.any(snapped == p)}
    median = np.median(list(iqrs.values()))
    raw = np.array([median / iqrs[p] for p in snapped])
    lo, hi = 0., 1e5
    for _ in range(80):
        scale = (lo+hi)/2
        if np.clip(raw*scale, .5, 2.).mean() > 1: hi = scale
        else: lo = scale
    weights = np.clip(raw*(lo+hi)/2, .5, 2.).astype('float32')
    assert abs(weights.mean()-1) < 1e-6
    return weights, {str(p): {'iqr': iqrs[p], 'weight': float(weights[snapped == p].mean())} for p in iqrs}


def source_audit(subject, out):
    path = Path(DATA_DIR) / SUBJECTS[subject][0]
    headers = []
    with open(path, 'rb') as f:
        while (h := _read_header(f)) is not None:
            headers.append(h)
            _read_data(f, h)
    frames = read_flb(path, SUBJECTS[subject][1])
    records = []
    for i, (frame, header) in enumerate(zip(frames, headers)):
        rec = dict(trial=i+1, header=header, samples=len(frame),
                   mean_angle=float(frame.position.mean()), position_sd=float(frame.position.std(ddof=0)),
                   torque_mean=float(frame.torque.mean()), torque_sd=float(frame.torque.std(ddof=0)),
                   finite=bool(np.isfinite(frame.to_numpy()).all()), channels={})
        assert abs(header['domainIncr']-.001) < 1e-9 and header['nReal'] == 1
        np.testing.assert_allclose(np.diff(frame.time), .001, atol=1e-10)
        for c in ['gm','gl','sol','ta']:
            v = frame[c].to_numpy()
            rec['channels'][c] = dict(min=float(v.min()), max=float(v.max()), sd=float(v.std()),
                fraction_at_observed_extrema=float(np.mean((v==v.min())|(v==v.max()))),
                consecutive_equal_fraction=float(np.mean(np.diff(v)==0)))
        records.append(rec)
    _, passive, _ = classify_trials(frames)
    maps = {s: get_passive_torque_map(passive, session=s) for s in ['test','retest']}
    dump(out/'raw_audit.json', dict(source=str(path), sha256=sha(path), trials=records,
         raw_label_passive_maps=maps,
         caveat='YES raw-label maps include inferred active records and are NOT valid calibration maps. Other maps still require source-quality review.'))


def distribution_rows(subject, data, audit):
    rows=[]
    for split, (xkey,ykey) in SPLITS.items():
        trace=audit['traces'][split]; X=data[xkey]; y=data[ykey]
        assert X.shape==(len(y),50,5) and len(trace['time'])==len(y)
        np.testing.assert_array_equal(trace['target'],y)
        np.testing.assert_allclose(trace['measured']-trace['passive'], y, atol=1e-5)
        snapped,_=snap_to_operating_points(X[:,-1,4],data['operating_positions'])
        for pos in sorted(set(snapped)):
            m=snapped==pos
            row=dict(subject=subject,split=split,position=float(pos),n=int(m.sum()),
                     trials=','.join(map(str,sorted(set(trace['trial'][m])))),
                     target_mean=float(y[m].mean()),target_sd=float(y[m].std(ddof=0)),
                     measured_mean=float(trace['measured'][m].mean()), measured_sd=float(trace['measured'][m].std()),
                     passive_mean=float(trace['passive'][m].mean()),passive_sd=float(trace['passive'][m].std()),
                     angle_sd=float(trace['position'][m].std()))
            for j,c in enumerate(['MG','LG','SOL','TA']):
                v=trace['emg'][m,j]
                row.update({f'{c}_mean':float(v.mean()), f'{c}_sd':float(v.std()),
                            f'{c}_p95':float(np.percentile(v,95)), f'{c}_above1':float(np.mean(v>1)),
                            f'{c}_above09':float(np.mean(v>.9))})
            rows.append(row)
    return rows


def plots(subject, data, audit, out, predictions=None, label=None):
    figure_label = label
    out.mkdir(parents=True,exist_ok=True)
    positions=data['operating_positions']
    for desired in FOCUS[subject]:
        pos=min(positions,key=lambda p:abs(p-desired))
        emg_max = 1.25
        for split,(xkey,_) in SPLITS.items():
            assigned,_=snap_to_operating_points(data[xkey][:,-1,4],positions)
            emg_max=max(emg_max,float(audit['traces'][split]['emg'][assigned==pos].max()))
        fig, axes=plt.subplots(4,3,figsize=(16,11),sharey='row',layout='constrained')
        for j,(split,(xkey,ykey)) in enumerate(SPLITS.items()):
            t=audit['traces'][split]
            snapped,_=snap_to_operating_points(data[xkey][:,-1,4],positions)
            mask=snapped==pos; ii=np.flatnonzero(mask)
            # Concatenated trials are broken rather than joined by fictitious lines.
            for trial in np.unique(t['trial'][mask]):
                k=ii[t['trial'][ii]==trial]; k=k[::10]
                x=t['time'][k]
                for key,label in [('measured','Measured'),('passive','Passive'),('target','Target')]:
                    axes[0,j].plot(x,t[key][k],label=label if trial==t['trial'][ii[0]] else None,lw=.8)
                if predictions is not None:
                    pred=predictions[split]
                    axes[0,j].plot(x,pred[k],label='Prediction' if trial==t['trial'][ii[0]] else None,lw=.8)
                    axes[1,j].plot(x,pred[k]-data[ykey][k],lw=.8)
            axes[0,j].set_title(f'{split} (N={len(ii):,})')
            axes[0,j].legend(fontsize=8,ncol=2)
            axes[1,j].axhline(0,color='k',lw=.5)
            if predictions is None:
                axes[1,j].text(.5,.5,'No model in source audit',ha='center',transform=axes[1,j].transAxes)
            axes[1,j].set_xlabel('Within-trial time (s)')
            axes[2,j].hist(data[ykey][mask],bins=40,density=True,alpha=.7)
            for c in range(4):
                axes[3,j].hist(t['emg'][mask,c],bins=np.linspace(0,emg_max,70),
                               density=True,histtype='step',label=['MG','LG','SOL','TA'][c])
            axes[3,j].legend(fontsize=8)
        for row,label in enumerate(['Torque (Nm)','Prediction − target (Nm)','Target density','EMG density']):axes[row,0].set_ylabel(label)
        for row in [0,1,2,3]:
            lims=[a.get_xlim() for a in axes[row]]
            for a in axes[row]: a.set_xlim(min(l[0] for l in lims),max(l[1] for l in lims))
        for a in axes[2]: a.set_xlabel('Active torque (Nm)')
        for a in axes[3]: a.set_xlabel('EMG / session-1 maximum')
        fig.suptitle(f'{subject}: {pos:+.3f} rad | '+(figure_label or ('source audit' if predictions is None else 'seed 11 exploratory prediction')))
        fig.savefig(out/f'{subject}_{pos:+.3f}.png',dpi=130);plt.close(fig)


def historical(subject,data,out):
    source=(ROOT/'reports/yes_eight_positions_reconstructed_20261013/YES/metrics.json' if subject=='YES'
            else ROOT/f'reports/repeated_runs/{subject}/metrics.json')
    runs=json.loads(source.read_text()); rows=[]
    snapped,_=snap_to_operating_points(data['X_test'][:,-1,4],data['operating_positions'])
    for run in runs:
        assert run['seed'] in [11,23,37,53,71,89,101,113,127,139]
        for pos0,m in run['per_position'].items():
            pos=float(pos0); closest=min(data['operating_positions'],key=lambda p:abs(p-pos))
            assert abs(pos-closest)<1e-9
            y=data['y_test'][snapped==closest].astype('float64')
            assert len(y)==m['n']
            mse=m['rmse']**2; var=y.var(); residual_var=var*(1-m['vaf']/100)
            assert abs((1-mse/var)-m['r2'])<3e-5
            bias2=max(0,mse-residual_var)
            rows.append(dict(subject=subject,seed=run['seed'],position=pos,**m,
                target_sd=np.sqrt(var),residual_sd=np.sqrt(max(0,residual_var)),
                abs_bias=np.sqrt(bias2),bias_fraction=bias2/mse,
                train_s=run['train_s'],bias_sign='unavailable: predictions not saved'))
    pd.DataFrame(rows).to_csv(out/'historical_decomposition.csv',index=False)
    dump(out/'historical_verification.json',dict(source=str(source),sha256=sha(source),
        n_runs=len(runs),n_test=len(data['y_test']),positions=data['operating_positions'],
        checks='All saved per-position N and R2 agree with rebuilt target variance; signed bias unavailable.'))


def audit_subject(subject, root):
    out=root/'audit'/subject;out.mkdir(parents=True,exist_ok=False)
    source_audit(subject,out)
    # Preserve the historical audit explicitly; clean validation is now the
    # default for ordinary training and evaluation entry points.
    audit={};data=load_subject(subject,yes_eight_positions=subject=='YES',
                               passive_session='retest', validation_gap_s=None,
                               audit=audit)
    dump(out/'metadata.json',{k:v for k,v in data.items() if not k.startswith(('X_','y_'))})
    dump(out/'trials.json',audit['trials'])
    pd.DataFrame(distribution_rows(subject,data,audit)).to_csv(out/'distributions.csv',index=False)
    for split,t in audit['traces'].items():np.savez_compressed(out/f'{split}_traces.npz',**t)
    historical(subject,data,out)
    plots(subject,data,audit,out/'plots')
    print(f'AUDIT COMPLETE {subject}',flush=True)


class EpochClock(tf.keras.callbacks.Callback):
    def __init__(self,path): super().__init__();self.path=path;self.rows=[]
    def on_epoch_begin(self,epoch,logs=None):self.wall=time.time();self.mono=time.perf_counter();self.cpu=time.process_time()
    def on_epoch_end(self,epoch,logs=None):
        row=dict(epoch=epoch+1,wall_start=self.wall,wall_s=time.time()-self.wall,
                 monotonic_s=time.perf_counter()-self.mono,cpu_s=time.process_time()-self.cpu,**(logs or {}))
        self.rows.append(row);dump(self.path,self.rows)
        if (epoch+1)%10==0:print(f'{self.path.parent}: epoch {epoch+1}, val_loss={row["val_loss"]:.4f}',flush=True)


def experiment(subject,root,epochs,seed,candidate='weighted'):
    if candidate == 'session_calibration' and subject != 'YES':
        raise ValueError('Worksheet calibration experiment is restricted to YES')
    out=root/'experiment'/subject;out.mkdir(parents=True,exist_ok=True)
    # This registered historical experiment compares against the shared
    # retest-map baseline; keep that choice explicit now that ordinary loads
    # use session-matched passive calibration by default.
    audit={};data=load_subject(subject,yes_eight_positions=subject=='YES',
                               passive_session='retest', validation_gap_s=2.,
                               audit=audit)
    dump(out/'metadata.json',{k:v for k,v in data.items() if not k.startswith(('X_','y_'))})
    dump(out/'trials.json',audit['trials'])
    pd.DataFrame(distribution_rows(subject,data,audit)).to_csv(out/'distributions.csv',index=False)
    weights,ledger=bounded_weights(data['X_train'],data['y_train'],data['operating_positions'])
    dump(out/'weights.json',ledger)
    # Frozen before any candidate retest predictions: same validation MSE monitor,
    # architecture, callbacks, seed, minibatches, epochs and raw segment protocol.
    for mode in ['baseline',candidate]:
        dest=out/mode/f'seed{seed}';dest.mkdir(parents=True,exist_ok=False)
        mode_data = dict(data)
        mode_audit = dict(audit, traces={s: dict(t) for s,t in audit['traces'].items()})
        if mode == 'session_calibration':
            # Values read independently from worksheet Test rows, not fitted
            # to active torque. Position pairing inherits the frozen YES ledger.
            worksheet_values = [2.6, -10.1, 4.3, .35, 1.8, 7.9, -2.6, -3.5]
            records = data['reconstruction']['retest_records']
            session1_map = sorted((r['mean_angle_rad'],tq) for r,tq in zip(records,worksheet_values))
            dump(dest/'calibration.json',dict(session1_map=session1_map,session2_map=data['passive_entries'],
                source='Worksheet Test and Re-Test passive columns; position pairing uses reconstructed retest mean angles',
                source_sha256=data['reconstruction']['worksheet_sha256'],
                limitation='No original nominal position-to-trial log; do not treat inferred identities as confirmed.'))
            for split in ['train','validation']:
                tr=mode_audit['traces'][split]
                tr['passive']=np.array([lookup_passive_torque(session1_map,p) for p in tr['position']])
                tr['target']=(tr['measured'].astype('float32')-tr['passive'].astype('float32'))
                mode_data[SPLITS[split][1]]=tr['target']
        tf.keras.backend.clear_session();set_global_seed(seed)
        clock=EpochClock(dest/'epoch_timings.json')
        start=time.time();mono=time.perf_counter();cpu=time.process_time()
        model,history=train_model(mode_data['X_train'],mode_data['y_train'],mode_data['X_val'],mode_data['y_val'],
            seed=seed,epochs=epochs,batch_size=64,lr=.003,verbose=0,
            sample_weight=weights if mode=='weighted' else None,extra_callbacks=[clock])
        train_s=time.perf_counter()-mono
        model.save(dest/'model.keras')
        dump(dest/'history.json',history.history)
        rows=[];predictions={}
        for split,(xkey,ykey) in SPLITS.items():
            pred=model.predict(mode_data[xkey],batch_size=512,verbose=0).ravel();predictions[split]=pred
            trace=mode_audit['traces'][split]
            np.savez_compressed(dest/f'{split}.npz',**trace,prediction=pred)
            snapped,_=snap_to_operating_points(data[xkey][:,-1,4],data['operating_positions'])
            rows.append(dict(subject=subject,mode=mode,seed=seed,split=split,position='pooled',**metrics(mode_data[ykey],pred)))
            for pos in sorted(set(snapped)):
                m=snapped==pos
                rows.append(dict(subject=subject,mode=mode,seed=seed,split=split,position=pos,**metrics(mode_data[ykey][m],pred[m])))
        pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
        plots(subject,mode_data,mode_audit,dest/'plots',predictions,label=f'{mode} | seed {seed} | exploratory')
        dump(dest/'run.json',dict(subject=subject,mode=mode,seed=seed,epochs_completed=len(history.history['loss']),
             best_epoch=int(np.argmin(history.history['val_loss']))+1,train_s=train_s,total_s=time.perf_counter()-mono,
             cpu_s=time.process_time()-cpu,started_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),
             ended_utc=datetime.now(timezone.utc).isoformat(),batch_size=64,lr=.003,max_epochs=epochs,
             validation_gap_s=2.,edge_guards_s=1.,window_ms=500,filter_order=4,
             timing_caveat='Local CPU wall time under caffeinate; epoch wall/monotonic/CPU logged. Not standardized benchmark.'))
        print(f'EXPERIMENT COMPLETE {subject} {mode} seed {seed}: {train_s:.1f}s',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['audit','experiment'],required=True)
    p.add_argument('--subjects',nargs='+',default=['YES','EG','IES','JM','HM'],choices=list(FOCUS))
    p.add_argument('--output',type=Path,default=ROOT/'reports/retest_mechanisms_20261019')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--seed',type=int,default=11)
    p.add_argument('--candidate',choices=['weighted','session_calibration'],default='weighted')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    manifest=dict(stage=a.stage,subjects=a.subjects,seed=a.seed,epochs=a.epochs,platform=platform.platform(),
        tensorflow=tf.__version__,numpy=np.__version__,devices=[str(d) for d in tf.config.list_physical_devices()],
        threads=dict(intra=1,inter=1),started_utc=datetime.now(timezone.utc).isoformat(),
        command=' '.join(__import__('sys').argv),
        sources={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'training/__init__.py',
                 ROOT/'preprocessing/dataset_builder.py',ROOT/'preprocessing/emg_envelope.py',ROOT/'models/lstm.py',ROOT/'config.py']},
        inference='Retest was already inspected. All new comparisons exploratory, not independent validation.',
        gate='Proceed to ten seeds only if validation mean per-position MSE/target_variance falls >=10%, no position validation RMSE rises >10%, and retest trade-offs warrant confirmation. No retest tuning.',
        candidate=a.candidate,
        weighting='Train-only inverse IQR; clip weights to [0.5,2] with mean=1; no retest label corrections.',
        calibration='YES optional: worksheet Test map for training/validation and Re-Test map for retest. Train/validation errors are relative to each protocol target; retest targets identical. No active-label fitting.',
        protocol='For experiments: raw 80/20 split, 2 s central gap, independent filtering, training-only max; 1 s session-1 edge guards. Retest unchanged.')
    dump(a.output/f'manifest_{a.stage}_{"_".join(a.subjects)}_seed{a.seed}.json',manifest)
    for subject in a.subjects:
        (audit_subject(subject,a.output) if a.stage=='audit' else experiment(subject,a.output,a.epochs,a.seed,a.candidate))
    manifest['completed_utc']=datetime.now(timezone.utc).isoformat()
    dump(a.output/f'manifest_{a.stage}_{"_".join(a.subjects)}_seed{a.seed}.json',manifest)

if __name__=='__main__': main()
