"""Training for one prepared run: the three fitters (windowed LSTM, carried-state LSTM, lagged ridge) and the
validation trim made before fitting. Called by run.py."""
import json

import numpy as np

from EXPERIMENTS.protocol.protocol import STUDIES


def drop_last_val_window(data):
    """Remove the last validation window of every trial (original split)."""
    tr = data['meta_val']['trial']
    keep = np.ones(len(tr), bool)
    for i in np.unique(tr):
        keep[np.where(tr == i)[0][-1]] = False
    data = dict(data)
    data['X_val'], data['y_val'] = data['X_val'][keep], data['y_val'][keep]
    data['meta_val'] = {k: v[keep] for k, v in data['meta_val'].items()}
    return data


def fit_ridge(data, protocol_file=STUDIES['study_v2'].protocol_file):
    """Study v2 A1: ridge on lagged envelopes (ML.models.baselines.LaggedRidge); alpha chosen on the session-1
    validation block from the protocol's grid. Deterministic: the seed is not used."""
    from ML.models.baselines import LaggedRidge
    with open(protocol_file, encoding='utf-8') as f:
        rp = json.load(f)['ridge']
    m = LaggedRidge(rp['alphas'], rp['lags_steps']).fit(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    vp = m.predict(data['X_val'])
    info = {'epochs': 0, 'alpha': m.alpha, 'val_mae_by_alpha': {str(k): v for k, v in m.val_mae.items()}, 'n_params': int(len(m.weights)),
            'val_pred': vp, 'val_mae': float(np.mean(np.abs(vp - data['y_val'])))}
    return m.predict(data['X_test']), info


def fit_carry(data, cfg, seed):
    """state='carry': train ML.research.stateful on whole trials; return test predictions on the same samples
    (and in the same order) as the windowed model's data['y_test'] / data['meta_test']."""
    from ML.research import stateful as ST
    from ML.research.train import angle_groups
    knots = angle_groups(data['train_angles'])
    train, test = data['streams']['train'], data['streams']['test']
    preds, info = ST.fit_predict_carry(train, test, chunk=cfg['chunk'] or cfg['window'], units=cfg['units'],
                                       layers=cfg['layers'], dropout=cfg['dropout'], seed=seed, lr=cfg['lr'])
    vs = {s['rec']['idx']: v for s, v in zip(train, info.pop('val_streams'))}
    mv = data['meta_val']
    val_pred = np.array([vs[int(i)][j] for i, j in zip(mv['trial'], mv['last'])], np.float64)
    info.update(val_pred=val_pred, val_mae=float(np.mean(np.abs(val_pred - data['y_val']))))
    by_trial = {s['rec']['idx']: (s, pr) for s, pr in zip(test, preds)}
    meta = data['meta_test']
    pred = np.empty(len(meta['trial']))
    for i in np.unique(meta['trial']):
        rows = np.where(meta['trial'] == i)[0]
        s, pr = by_trial[int(i)]
        pred[rows] = pr[meta['last'][rows]]
        assert np.allclose(np.asarray(s['y'])[meta['last'][rows]], data['y_test'][rows])
    info.update(knots=knots.tolist())
    return pred, info, data


def fit(data, cfg, seed, *, save_weights=None, protocol_file=STUDIES['study_v2'].protocol_file):
    """Fit the model ``cfg`` asks for on ``data`` (already through drop_last_val_window): returns (test predictions,
    info, data). info['val_pred'] holds the validation predictions."""
    from ML.research import train as RT
    if cfg['state'] == 'carry':
        return fit_carry(data, cfg, seed)
    if cfg['model'] == 'ridge_lagged':
        pred, info = fit_ridge(data, protocol_file)
        return pred, info, data
    pred, info = RT.fit_predict(data, cfg['units'], cfg['layers'], cfg['dropout'], seed=seed, lr=cfg['lr'],
                                batch=cfg['batch'], save_weights=save_weights)
    return pred, info, data
