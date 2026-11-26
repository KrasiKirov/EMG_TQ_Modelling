"""Training for one prepared run: the model builder, the three fitters (windowed LSTM, carried-state LSTM, lagged
ridge) and the data adjustments made before fitting. Called by run.py; bodies moved unchanged from it."""
import json

import numpy as np

from EXPERIMENTS.protocol.protocol import STUDIES


def make_builder(cfg):
    """Model builder plugged into ML.research.train (same call signature)."""

    def build(name, n_steps, n_features, knots):
        if name != 'lstm':                              # 'lstm8': the original model
            from ML.research.train import build_original
            return build_original(name, n_steps, n_features, knots)
        from ML.models.lstm import build_lstm
        return build_lstm(n_steps, n_features, cfg['units'], cfg['layers'], cfg['dropout'])
    return build


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


def add_passive_input(data, subject):
    """Tier-2 option 'extra_input': 'passive_trial_value'.
    Appends one constant input channel per window: the passive torque of that
    trial's angle taken from the PASSIVE TRIALS of the trial's own session (the
    clean static-hold table, the same table the primary answer key uses).  It is
    measured before the active trial, so it is available at prediction time.
    IES has no passive trials -> 0 for every window."""
    from ML.research.data import load_subject
    from ML.research.passive import map_lookup
    maps = load_subject(subject)['maps']['static']
    data = dict(data)
    for part in ('train', 'val', 'test'):
        meta = data[f'meta_{part}']
        p = np.array([map_lookup(maps[s], np.array([a]))[0] if maps.get(s) else 0.0
                      for s, a in zip(meta['session'], meta['angle'])], np.float32)
        X = data[f'X_{part}']
        data[f'X_{part}'] = np.concatenate([X, np.repeat(p[:, None, None], X.shape[1], axis=1)], axis=2)
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
    from ML.config import TRAIN_PROTOCOLS
    from ML.research import stateful as ST
    from ML.research.train import angle_groups
    knots = angle_groups(data['train_angles'])
    train, test = data['streams']['train'], data['streams']['test']
    p = TRAIN_PROTOCOLS[cfg['train_protocol']]
    preds, info = ST.fit_predict_carry(train, test, chunk=cfg['chunk'] or cfg['window'],
                                       seed=seed, epochs=p['epochs'], early=p['early'], lr=cfg['lr'] or p['lr'],
                                       lr_factor=p['lr_factor'], lr_patience=p['lr_patience'],
                                       units=cfg['units'], layers=cfg['layers'], dropout=cfg['dropout'],
                                       clipnorm=cfg['clipnorm'])
    vs = {s['rec']['idx']: v for s, v in zip(train, info.pop('val_streams'))}
    mv = data['meta_val']
    val_pred = np.array([vs[int(i)][j] for i, j in zip(mv['trial'], mv['last'])], np.float64)
    info.update(val_pred=val_pred, val_mae=float(np.mean(np.abs(val_pred - data['y_val']))), clipnorm=cfg['clipnorm'])
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
    pred, info = RT.fit_predict(data, model_name=cfg['model'], protocol=cfg['train_protocol'], seed=seed,
                                lr=cfg['lr'], shuffle=cfg['shuffle'], save_weights=save_weights,
                                clipnorm=cfg['clipnorm'], batch=cfg['batch'], build=make_builder(cfg))
    return pred, info, data
