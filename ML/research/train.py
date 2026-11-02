"""
Training with per-angle loss balancing.

Why balancing: with plain MSE the loss is dominated by the strong angles
(YES: torque std 15 Nm at +0.14 rad vs 0.4 Nm at -0.49 rad), so the weak,
plantarflexed angles barely influence training.

balance
-------
none      plain MSE in Nm² (original pipeline)
weights   sample weight 1 / var(target of its angle)   (angle-weighted MSE)
scale     train on target / sd(angle) and multiply predictions back
          (equivalent idea, used with the mixture-of-experts head)

pf_extra : (n_angles, factor) — multiply the loss weight of the ``n_angles``
most plantarflexed training angles by ``factor`` on top of the balancing
(tests "push the model harder towards plantarflexion").

Angles are grouped by trial: a window belongs to the angle of the trial it
was cut from; test windows use the nearest training angle.
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
    """Per-window loss weights.  ``var_per_knot``: target variance of each
    angle in the TRAINING part; the same values weight the validation part
    so that early stopping optimises the same objective as training."""
    if mode == 'weights':
        w = 1.0 / var_per_knot[k]
    else:
        w = np.ones(len(k))
    if pf_extra:
        n_pf, factor = pf_extra
        w = w * np.where(k < n_pf, factor, 1.0)          # knots sorted: lowest angle first
    return w


# ── classical baseline: ridge regression on lagged envelopes ────────────────

RIDGE_LAG_STEP = 2                          # use every 2nd step of the window
RIDGE_ALPHAS = np.logspace(-8, 2, 21)        # wide enough that the choice is not at the edge


def hat_basis(angles, knots):
    """Piecewise-linear 'hat' weights over the training angles (same rule as
    the MoE AngleGate): one-hot at a knot, linear blend in between."""
    a = np.clip(np.asarray(angles, float), knots[0], knots[-1])
    B = np.zeros((len(a), len(knots)))
    for j, k in enumerate(knots):
        left = knots[j - 1] if j > 0 else k - 1.0
        right = knots[j + 1] if j < len(knots) - 1 else k + 1.0
        B[:, j] = np.clip(np.minimum((a - left) / (k - left), (right - a) / (right - k)), 0, 1)
    return B


def ridge_features(X, knots):
    """[lagged EMG, lagged EMG squared] x angle hat basis, plus the basis.

    Linear + quadratic terms follow the polynomial EMG-torque models of
    Clancy et al. 2012 (IEEE TBME 59:1); one coefficient set per training
    angle (blended in between) lets gain and muscle mix change with angle."""
    E = X[:, ::-1, :4][:, ::RIDGE_LAG_STEP, :].reshape(len(X), -1)    # newest step first
    F = np.concatenate([E, E ** 2], axis=1)
    B = hat_basis(X[:, -1, 4], knots)
    return np.concatenate([(F[:, :, None] * B[:, None, :]).reshape(len(X), -1), B], axis=1)


def fit_ridge(data, knots, w_tr=None, w_va=None):
    """Weighted ridge, alpha chosen on the validation part (weighted MSE).

    Use it with balance='none': every angle has its own coefficients, so the
    angles are already fitted separately, and per-angle weights would only
    change how strongly each angle is regularised."""
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


def fit_predict(data, model_name='lstm8', balance='none', pf_extra=None,
                protocol='fast', seed=0, verbose=0, init_weights=None, lr=None, rest_weight=None):
    """Train one model on data['X_train'] / ['X_val'] and predict X_test.

    init_weights : optional .weights.h5 file to start from (fine-tuning a
                   pre-trained model, see transfer.py); None = random init
    lr           : optional learning rate overriding the protocol's
    rest_weight  : None = all windows count normally.  A number w multiplies
                   the loss weight of windows that end in EMG-detected rest
                   (data['meta_*']['rest']); w = 0 removes them from training
                   and validation (relaxed-window ablation).

    Returns (y_pred_test (Nm), info dict)."""
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
    model.compile(optimizer=Nadam(learning_rate=lr or p['lr']), loss='mse')
    cbs = [ReduceLROnPlateau(monitor='val_loss', factor=p['lr_factor'], patience=p['lr_patience'],
                             min_lr=1e-6, verbose=0),
           EarlyStopping(monitor='val_loss', patience=p['early'], restore_best_weights=True, verbose=0)]
    val = (data['X_val'], t_va.astype(np.float32)) + ((w_va.astype(np.float32),) if use_w else ())
    hist = model.fit(data['X_train'], t_tr.astype(np.float32),
                     sample_weight=None if w_tr is None else w_tr.astype(np.float32),
                     validation_data=val, epochs=p['epochs'], batch_size=p['batch'],
                     callbacks=cbs, shuffle=True, verbose=verbose)
    pred = model.predict(data['X_test'], verbose=0, batch_size=4096).ravel().astype(np.float64)
    if balance == 'scale':
        pred = pred * sd[k_te]
    best = int(np.argmin(hist.history['val_loss']))
    info = {'epochs': len(hist.history['loss']), 'best_epoch': best + 1,
            'best_val_loss': float(hist.history['val_loss'][best]),
            'knots': knots.tolist(), 'sd_per_knot': sd.tolist(), 'n_params': int(model.count_params()),
            'n_train_per_knot': [int(np.sum(k_tr == j)) for j in range(len(knots))],
            'history': {k: [float(x) for x in v] for k, v in hist.history.items()},
            'init_weights': init_weights, 'lr': lr or p['lr']}
    return pred, info
