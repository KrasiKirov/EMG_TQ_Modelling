"""
Learning curves for the bias / variance / overfitting check.

    python EXPERIMENTS/code/learning_curves.py --exp D01_cleanpassive+autogain+equalweight --subject JM --seed 0
    python EXPERIMENTS/code/learning_curves.py --plot

Trains exactly like run_experiment.py (same data, same split, same settings, same
early stopping on the validation part) and, after every epoch, measures the error of
the current model WITH DROPOUT OFF on three sets:
  fit   first 80 % of every training-session trial (what the weights are fitted on)
  val   last 20 % of every training-session trial (early stopping; same session, unseen)
  test  every trial of the other session (only monitored here, never used for any choice)
Keras' own 'loss' is measured with dropout on and averaged while the weights change,
so it cannot be compared with 'val_loss'; the numbers here can.

fit / val errors are against the target the model is trained on; test errors against
the primary answer key (passive trials of the tested session, as score.py).  Errors in
Nm (MAE) and Nm^2 (MSE), after undoing the per-position scaling of 'equal weight'.

How to read them
  fit error high                    -> bias (model too simple / inputs not informative)
  val error well above fit error    -> overfitting to the fitted samples
  test error well above val error   -> the session changed (electrodes, passive torque,
                                       effort) - not fixable by regularising the model
Outputs results/learning_curves/<exp>/<subject>/<direction>/seed<k>.json and, with
--plot, plots/learning_curves_<direction>.png + reports/learning_curves.md
"""
import os
import sys
import glob
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(EXP))
sys.path.insert(0, HERE)

import numpy as np                                      # noqa: E402

OUT = os.path.join(EXP, 'results', 'learning_curves')
FS = 100.0


def primary_key(subject, meta):
    """passive_trial answer key for every test window (same definition as score.py)"""
    from ML.research.data import load_subject
    from ML.research.passive import PassiveModel
    b = load_subject(subject)
    st = PassiveModel('static', b)
    recs = {r['idx']: r for r in b['active']}
    y = np.empty(len(meta['trial']))
    for i in np.unique(meta['trial']):
        m = meta['trial'] == i
        r = recs[int(i)]
        key = r['tq'] - st.predict(r)
        y[m] = key[np.rint((meta['t'][m] - r['t'][0]) * FS).astype(int)]
    return y


def per_trial(y, p, trial):
    out = []
    for i in np.unique(trial):
        m = trial == i
        e = p[m] - y[m]
        ss = np.sum((y[m] - y[m].mean()) ** 2)
        out.append(dict(trial=int(i), r2=float(1 - np.sum(e ** 2) / ss), mae=float(np.mean(np.abs(e))),
                        mse=float(np.mean(e ** 2)), n=int(m.sum())))
    return out


def run(a):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(a.threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, a.threads))
    from ML.research.run import prepare
    from ML.research import train as RT
    from run_experiment import load_config, make_builder, drop_last_val_window

    spec, cfg = load_config(a.exp)
    t0 = time.time()
    data, details = prepare(a.subject, a.direction, {k: cfg[k] for k in ('passive', 'emg', 'window')})
    data = drop_last_val_window(data)
    # the per-position scale used by balance='scale' (same computation as RT.fit_predict)
    knots = RT.angle_groups(data['train_angles'])
    k = {p: RT.snap(data[f'meta_{p}']['angle'], knots) for p in ('train', 'val', 'test')}
    y_tr = data['y_train'].astype(np.float64)
    sd = np.array([max(np.std(y_tr[k['train'] == j]), 0.1) for j in range(len(knots))])
    f = {p: (sd[k[p]] if cfg['balance'] == 'scale' else np.ones(len(k[p]))) for p in k}
    sets = {'fit': (data['X_train'], y_tr, f['train'], data['meta_train']['trial']),
            'val': (data['X_val'], data['y_val'].astype(np.float64), f['val'], data['meta_val']['trial']),
            'test': (data['X_test'], primary_key(a.subject, data['meta_test']), f['test'], data['meta_test']['trial'])}

    class Curves(tf.keras.callbacks.Callback):
        """after each epoch: errors on fit / val / test with dropout off (model.predict)"""
        def __init__(self):
            super().__init__()
            self.rows, self.final = [], {}

        def _pred(self, X, fac):
            return self.model.predict(X, batch_size=4096, verbose=0).ravel().astype(np.float64) * fac

        def on_epoch_end(self, epoch, logs=None):
            row = dict(epoch=epoch + 1, keras_loss=float(logs['loss']), keras_val_loss=float(logs['val_loss']),
                       lr=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
            for name, (X, y, fac, _) in sets.items():
                e = self._pred(X, fac) - y
                row[f'mse_{name}'], row[f'mae_{name}'] = float(np.mean(e ** 2)), float(np.mean(np.abs(e)))
            self.rows.append(row)

        def on_train_end(self, logs=None):          # runs after EarlyStopping restored the best weights
            for name, (X, y, fac, trial) in sets.items():
                self.final[name] = per_trial(y, self._pred(X, fac), trial)

    cb = Curves()

    def build(name, n_steps, n_features, kn):
        model = make_builder(cfg)(name, n_steps, n_features, kn)
        fit = model.fit

        def fit_with_curves(*args, callbacks=None, **kw):  # same fit call, one extra (read-only) callback
            return fit(*args, callbacks=list(callbacks or []) + [cb], **kw)
        model.fit = fit_with_curves
        return model
    RT.build_model = build
    pred, info = RT.fit_predict(data, cfg['model'], cfg['balance'], cfg['pf_extra'], cfg['train_protocol'], a.seed,
                                lr=cfg['lr'], rest_weight=cfg['rest_weight'])
    angles = {int(i): float(ang) for p in ('train', 'val', 'test') for i, ang in zip(data[f'meta_{p}']['trial'], data[f'meta_{p}']['angle'])}
    for name in cb.final:
        for r in cb.final[name]:
            r['angle'] = angles[r['trial']]
    out = os.path.join(OUT + ('_smoke' if a.max_epochs else ''), a.exp, a.subject, a.direction)
    os.makedirs(out, exist_ok=True)
    res = dict(experiment=a.exp, subject=a.subject, direction=a.direction, seed=a.seed, config=cfg,
               best_epoch=info['best_epoch'], epochs=info['epochs'], history=cb.rows, final_per_trial=cb.final,
               n_windows={p: int(len(v[1])) for p, v in sets.items()}, seconds=time.time() - t0)
    with open(os.path.join(out, f'seed{a.seed}.json.tmp'), 'w') as fh:
        json.dump(res, fh, indent=1)
    os.replace(os.path.join(out, f'seed{a.seed}.json.tmp'), os.path.join(out, f'seed{a.seed}.json'))
    print(f'[{a.exp} {a.subject} {a.direction} s{a.seed}] {info["epochs"]} ep (best {info["best_epoch"]}) {time.time() - t0:.0f}s')


def plot():
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = sorted(glob.glob(os.path.join(OUT, '*', '*', '*', 'seed*.json')))
    R = [json.load(open(f)) for f in files]
    if not R:
        print('no learning curves yet')
        return
    subj = [s for s in ['YES', 'EG', 'JM', 'IES', 'CH', 'HM'] if any(r['subject'] == s for r in R)]
    exps = sorted({r['experiment'] for r in R}, key=lambda e: (not e.startswith('X00'), e))
    col = {'fit': '#1f9d55', 'val': '#2a78d6', 'test': '#c0392b'}
    lab = {'fit': 'fitting part (80 % of session used for training)', 'val': 'validation part (last 20 %, same session)',
           'test': 'other session (test, only monitored)'}
    rows = []
    for d in sorted({r['direction'] for r in R}):
        fig, axs = plt.subplots(len(subj), len(exps), figsize=(6.5 * len(exps), 2.7 * len(subj)), squeeze=False)
        for i, s in enumerate(subj):
            for j, e in enumerate(exps):
                ax = axs[i, j]
                rr = [r for r in R if r['subject'] == s and r['experiment'] == e and r['direction'] == d]
                for r in rr:
                    h = pd.DataFrame(r['history'])
                    for p in ('fit', 'val', 'test'):
                        ax.plot(h.epoch, h[f'mae_{p}'], color=col[p], lw=1.1, label=lab[p] if (i == 0 and r is rr[0]) else None)
                    ax.axvline(r['best_epoch'], color='k', lw=0.7, ls='--')
                    fin = {p: pd.DataFrame(r['final_per_trial'][p]) for p in ('fit', 'val', 'test')}
                    td = sorted(fin['test'].angle)[:2]
                    rows.append(dict(direction=d, experiment=e, subject=s, seed=r['seed'], best_epoch=r['best_epoch'], epochs=r['epochs'],
                                     **{f'mae_{p}': fin[p].mae.mean() for p in fin},
                                     **{f'r2_median_{p}': fin[p].r2.median() for p in fin},
                                     r2_test_toesdown=fin['test'][fin['test'].angle.isin(td)].r2.mean()))
                ax.set_title(f'{s} · {e.split("_")[0]}', loc='left', fontsize=10, weight='bold')
                ax.set_ylabel('MAE (Nm)')
        for ax in axs[-1]:
            ax.set_xlabel('epoch (dashed = epoch kept by early stopping)')
        axs[0, 0].legend(fontsize=8, loc='upper right')
        fig.suptitle(f'Learning curves, dropout off · {d}', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(EXP, 'plots', f'learning_curves_{d}.png'), dpi=110)
        plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(EXP, 'results', 'learning_curves_summary.csv'), index=False)
    L = ['# Learning curves: bias, variance and overfitting', '',
         'Generated by `code/learning_curves.py --plot`. Errors of the kept model (best validation epoch), dropout off. '
         'MAE = mean over the positions of each set; R² = median over positions (raw, not floored).', '',
         '| direction | model | subject | seed | epochs (kept) | MAE fit | MAE val | MAE test | R² fit | R² val | R² test | R² test toes-down |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in T.sort_values(['direction', 'experiment', 'subject']).itertuples():
        L.append(f'| {r.direction} | {r.experiment.split("_")[0]} | {r.subject} | {r.seed} | {r.epochs} ({r.best_epoch}) | {r.mae_fit:.2f} | {r.mae_val:.2f} | '
                 f'{r.mae_test:.2f} | {r.r2_median_fit:.3f} | {r.r2_median_val:.3f} | {r.r2_median_test:.3f} | {r.r2_test_toesdown:.3f} |')
    open(os.path.join(EXP, 'reports', 'learning_curves.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp')
    ap.add_argument('--subject')
    ap.add_argument('--direction', default='S1toS2', choices=['S1toS2', 'S2toS1'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--max-epochs', type=int, default=None, help='smoke test only: cap the number of epochs')
    ap.add_argument('--plot', action='store_true')
    a = ap.parse_args()
    if a.plot:
        plot()
    else:
        if a.max_epochs:
            from ML.research import config as RC
            RC.PROTOCOLS['orig'] = dict(RC.PROTOCOLS['orig'], epochs=a.max_epochs)
        run(a)
