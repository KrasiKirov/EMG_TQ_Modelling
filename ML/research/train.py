"""
Keras training with optional per-angle loss balancing, and a ridge baseline.

Plain MSE is dominated by the strong angles (YES: torque sd 15 Nm at +0.14 rad vs 0.4 Nm at -0.49 rad).
balance: 'none' plain MSE (original), 'weights' sample weight 1 / var(target of its angle), 'scale' train on
target / sd(angle) and scale predictions back. pf_extra = (n, factor) multiplies the weight of the n most
plantarflexed angles. A window has its trial's angle; test windows use the nearest training angle.
"""

import numpy as np

from . import config as C
from .models import build as build_model

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


def snap(angles, knots):
    return np.argmin(np.abs(np.asarray(angles)[:, None] - knots[None, :]), axis=1)


def _weights(k, n_knots, mode, pf_extra, var_per_knot=None):
    """Per-window loss weights; ``var_per_knot`` comes from the training part and also weights validation."""
    if mode == 'weights':
        w = 1.0 / var_per_knot[k]
    else:
        w = np.ones(len(k))
    if pf_extra:
        n_pf, factor = pf_extra
        w = w * np.where(k < n_pf, factor, 1.0)          # knots sorted: lowest angle first
    return w


RIDGE_LAG_STEP = 2                          # use every 2nd step of the window
RIDGE_ALPHAS = np.logspace(-8, 2, 21)        # wide enough that the choice is not at the edge


def hat_basis(angles, knots):
    """Piecewise-linear 'hat' weights over the training angles (as the MoE AngleGate)."""
    a = np.clip(np.asarray(angles, float), knots[0], knots[-1])
    B = np.zeros((len(a), len(knots)))
    for j, k in enumerate(knots):
        left = knots[j - 1] if j > 0 else k - 1.0
        right = knots[j + 1] if j < len(knots) - 1 else k + 1.0
        B[:, j] = np.clip(np.minimum((a - left) / (k - left), (right - a) / (right - k)), 0, 1)
    return B


def ridge_features(X, knots):
    """[lagged EMG, lagged EMG squared] x angle hat basis, plus the basis (polynomial EMG-torque model as in
    Clancy et al. 2012, IEEE TBME 59:1, with one coefficient set per training angle)."""
    E = X[:, ::-1, :4][:, ::RIDGE_LAG_STEP, :].reshape(len(X), -1)    # newest step first
    F = np.concatenate([E, E ** 2], axis=1)
    B = hat_basis(X[:, -1, 4], knots)
    return np.concatenate([(F[:, :, None] * B[:, None, :]).reshape(len(X), -1), B], axis=1)


def fit_ridge(data, knots, w_tr=None, w_va=None):
    """Weighted ridge, alpha chosen on validation MSE. Use balance='none': each angle already has its own
    coefficients."""
    F_tr, F_va = ridge_features(data['X_train'], knots), ridge_features(data['X_val'], knots)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
    Z_tr, Z_va = (F_tr - mu) / sd, (F_va - mu) / sd
    y_tr, y_va = data['y_train'].astype(np.float64), data['y_val'].astype(np.float64)
    w_tr = np.ones(len(y_tr)) if w_tr is None else w_tr
    w_va = np.ones(len(y_va)) if w_va is None else w_va
    ym = np.average(y_tr, weights=w_tr)
    G = (Z_tr * w_tr[:, None]).T @ Z_tr
    b = (Z_tr * w_tr[:, None]).T @ (y_tr - ym)
    lam, V = np.linalg.eigh(G)                  # one decomposition, stable for small alpha
    lam, Vb = np.maximum(lam, 0.0), V.T @ b
    best = None
    for al in RIDGE_ALPHAS:
        coef = V @ (Vb / (lam + al * len(y_tr)))
        err = float(np.average((Z_va @ coef + ym - y_va) ** 2, weights=w_va))
        if best is None or err < best[0]:
            best = (err, al, coef)
    err, al, coef = best
    pred = (ridge_features(data['X_test'], knots) - mu) / sd @ coef + ym
    return pred, {'alpha': float(al), 'alpha_at_edge': bool(al in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1])),
                  'val_mse': err, 'n_features': int(len(coef)), 'epochs': 0}


def epoch_indices(n, n_epoch):
    """Indices of one epoch of exactly ``n_epoch`` windows taken in time order from ``n`` windows, wrapping to
    the first window when the set runs out (every epoch starts at the first window)."""
    return np.resize(np.arange(n), n_epoch)


def fit_predict(data, model_name='lstm8', balance='none', pf_extra=None,
                protocol='fast', seed=0, verbose=0, init_weights=None, lr=None, rest_weight=None, shuffle=True,
                save_weights=None, epoch_windows=None, epoch_mode='passes', clipnorm=None, batch=None):
    """Train on data['X_train'] / ['X_val'] and predict X_test; returns (y_pred_test (Nm), info).

    init_weights  optional .weights.h5 to start from
    lr            overrides the protocol's learning rate
    rest_weight   multiplies the loss weight of windows ending in EMG-detected rest; 0 drops them
    shuffle       False feeds the windows in stored order (shuffle ablation)
    save_weights  path for the best-epoch weights
    epoch_windows one epoch = round(N / n_train) whole passes (at least 1), so settings with few windows get
                  about as many updates per epoch as one with N windows
    epoch_mode    'passes' (above, v1) or 'exact': every epoch is exactly N windows, cycling in time order
    clipnorm      gradient-norm clipping of Nadam (None = off)
    batch         overrides the protocol's batch size
    Also returns the validation predictions (Nm, best-epoch weights) as info['val_pred'] and their MAE.
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Nadam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    tf.keras.utils.set_random_seed(seed)
    p = C.PROTOCOLS[protocol]
    if rest_weight == 0:                      # drop rest windows from train / val
        data = dict(data)
        for part in ('train', 'val'):
            keep = ~data[f'meta_{part}']['rest']
            data[f'X_{part}'], data[f'y_{part}'] = data[f'X_{part}'][keep], data[f'y_{part}'][keep]
            data[f'meta_{part}'] = {k: v[keep] for k, v in data[f'meta_{part}'].items()}
    knots = angle_groups(data['train_angles'])
    k_tr = snap(data['meta_train']['angle'], knots)
    k_va = snap(data['meta_val']['angle'], knots)
    k_te = snap(data['meta_test']['angle'], knots)
    y_tr, y_va = data['y_train'].astype(np.float64), data['y_val'].astype(np.float64)

    sd = np.array([max(np.std(y_tr[k_tr == j]), 0.1) for j in range(len(knots))])
    if balance == 'scale':
        t_tr, t_va = y_tr / sd[k_tr], y_va / sd[k_va]
    else:
        t_tr, t_va = y_tr, y_va
    use_w = balance == 'weights' or bool(pf_extra) or bool(rest_weight)
    var_k = np.array([max(np.var(y_tr[k_tr == j]), 1e-3) if np.any(k_tr == j) else 1.0 for j in range(len(knots))])
    w_tr = _weights(k_tr, len(knots), balance, pf_extra, var_k) if use_w else None
    w_va = _weights(k_va, len(knots), balance, pf_extra, var_k) if use_w else None
    if rest_weight:                           # down-weight rest windows
        w_tr = w_tr * np.where(data['meta_train']['rest'], rest_weight, 1.0)
        w_va = w_va * np.where(data['meta_val']['rest'], rest_weight, 1.0)
    if use_w:   # same normalisation constant for train and validation
        c = w_tr.mean()
        w_tr, w_va = w_tr / c, w_va / c

    if model_name == 'ridge':
        pred, info = fit_ridge(data, knots, w_tr, w_va)
        info.update(knots=knots.tolist(), model='ridge')
        return pred, info
    model = build_model(model_name, data['X_train'].shape[1], data['X_train'].shape[2], knots)
    if init_weights is not None:
        model.load_weights(init_weights)
    model.compile(optimizer=Nadam(learning_rate=lr or p['lr'], clipnorm=clipnorm), loss='mse')
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=p['lr_factor'], patience=p['lr_patience'],
                             min_lr=1e-6, verbose=0),
           EarlyStopping(monitor='val_loss', patience=p['early'], restore_best_weights=True, verbose=0)]
    val = (data['X_val'], t_va.astype(np.float32)) + ((w_va.astype(np.float32),) if use_w else ())
    X_fit, t_fit = data['X_train'], t_tr.astype(np.float32)
    w_fit = None if w_tr is None else w_tr.astype(np.float32)
    passes = max(1, int(round(epoch_windows / len(X_fit)))) if epoch_windows else 1
    if epoch_windows and epoch_mode == 'exact':
        sel = epoch_indices(len(X_fit), int(epoch_windows))
        X_fit, t_fit = X_fit[sel], t_fit[sel]
        w_fit = None if w_fit is None else w_fit[sel]
        passes = len(sel) / len(data['X_train'])
    elif epoch_mode != 'passes':
        raise ValueError(f'epoch_mode must be passes or exact, got {epoch_mode!r}')
    elif passes > 1:                            # whole passes, each in stored order
        X_fit, t_fit = np.concatenate([X_fit] * passes), np.concatenate([t_fit] * passes)
        w_fit = None if w_fit is None else np.concatenate([w_fit] * passes)
    hist = model.fit(X_fit, t_fit, sample_weight=w_fit,
                     validation_data=val, epochs=p['epochs'], batch_size=batch or p['batch'],
                     callbacks=cbs, shuffle=shuffle, verbose=verbose)
    if save_weights:
        model.save_weights(save_weights)
    pred = model.predict(data['X_test'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    pv = model.predict(data['X_val'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    if balance == 'scale':
        pred = pred * sd[k_te]
        pv = pv * sd[k_va]
    best = int(np.argmin(hist.history['val_loss']))
    info = {'epochs': len(hist.history['loss']), 'best_epoch': best + 1,
            'best_val_loss': float(hist.history['val_loss'][best]),
            'knots': knots.tolist(), 'sd_per_knot': sd.tolist(), 'n_params': int(model.count_params()),
            'init_weights': init_weights, 'lr': lr or p['lr'],
            'passes_per_epoch': passes, 'updates': int(model.optimizer.iterations.numpy()),
            'clipnorm': clipnorm, 'optimizer_clipnorm': model.optimizer.clipnorm, 'max_epochs': p['epochs'],
            'batch': batch or p['batch'], 'shuffle': bool(shuffle), 'epoch_windows': epoch_windows,
            'epoch_mode': epoch_mode, 'val_pred': pv, 'val_mae': float(np.mean(np.abs(pv - y_va)))}
    return pred, info
