"""
Keras training of the windowed LSTM on plain MSE (and the ridge baseline, ridge.py).

Training angles closer than KNOT_TOL are merged into knots (used by the ridge basis and stored in the run info).
"""

import numpy as np

from .config import PROTOCOLS
from .ridge import fit_ridge

KNOT_TOL = 0.02   # rad: training trials closer than this share an angle group


def angle_groups(train_angles):
    """Merge near-identical training angles into knots (sorted)."""
    knots = []
    for a in sorted(train_angles):
        if knots and a - knots[-1][-1] < KNOT_TOL:
            knots[-1].append(a)
        else:
            knots.append([a])
    return np.array([np.mean(k) for k in knots])


def build_original(name, n_steps, n_features, knots):
    """The original two-layer LSTM-8 for ``name`` 'lstm' / 'lstm8'."""
    if name not in ('lstm', 'lstm8'):
        raise ValueError(f'unknown model {name!r}')
    from ML.models.lstm import build_lstm
    return build_lstm(n_steps, n_features, units=8, layers=2, dropout=0.3, name='ankle_torque_lstm')


def fit_predict(data, model_name='lstm8', protocol='fast', seed=0, verbose=0, lr=None, shuffle=True,
                save_weights=None, clipnorm=None, batch=None, build=None):
    """Train on data['X_train'] / ['X_val'] and predict X_test; returns (y_pred_test (Nm), info).

    model_name    'lstm' / 'lstm8' (built by ``build``) or 'ridge' (ridge.py)
    lr            overrides the protocol's learning rate
    shuffle       False feeds the windows in stored order (shuffle ablation)
    save_weights  path for the best-epoch weights
    clipnorm      gradient-norm clipping of Nadam (None = off)
    batch         overrides the protocol's batch size
    build         model builder (name, n_steps, n_features, knots) -> keras model; default ``build_original``
    Also returns the validation predictions (Nm, best-epoch weights) as info['val_pred'] and their MAE.
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    tf.keras.utils.set_random_seed(seed)
    p = PROTOCOLS[protocol]
    knots = angle_groups(data['train_angles'])
    y_tr, y_va = data['y_train'].astype(np.float64), data['y_val'].astype(np.float64)

    if model_name == 'ridge':
        pred, info = fit_ridge(data, knots)
        info.update(knots=knots.tolist(), model='ridge')
        return pred, info
    model = (build or build_original)(model_name, data['X_train'].shape[1], data['X_train'].shape[2], knots)
    model.compile(optimizer=Nadam(learning_rate=lr or p['lr'], clipnorm=clipnorm), loss='mse')
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=p['lr_factor'], patience=p['lr_patience'],
                             min_lr=1e-6, verbose=0),
           EarlyStopping(monitor='val_loss', patience=p['early'], restore_best_weights=True, verbose=0)]
    val = (data['X_val'], y_va.astype(np.float32))
    hist = model.fit(data['X_train'], y_tr.astype(np.float32), sample_weight=None,
                     validation_data=val, epochs=p['epochs'], batch_size=batch or p['batch'],
                     callbacks=cbs, shuffle=shuffle, verbose=verbose)
    if save_weights:
        model.save_weights(save_weights)
    pred = model.predict(data['X_test'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    pv = model.predict(data['X_val'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    best = int(np.argmin(hist.history['val_loss']))
    info = {'epochs': len(hist.history['loss']), 'best_epoch': best + 1,
            'best_val_loss': float(hist.history['val_loss'][best]),
            'knots': knots.tolist(), 'n_params': int(model.count_params()),
            'lr': lr or p['lr'],
            'updates': int(model.optimizer.iterations.numpy()),
            'clipnorm': clipnorm, 'optimizer_clipnorm': model.optimizer.clipnorm, 'max_epochs': p['epochs'],
            'batch': batch or p['batch'], 'shuffle': bool(shuffle), 'val_pred': pv, 'val_mae': float(np.mean(np.abs(pv - y_va)))}
    return pred, info
