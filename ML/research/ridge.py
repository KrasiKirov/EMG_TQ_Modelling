"""Per-angle polynomial ridge baseline (model 'ridge'): lagged EMG and EMG² times a hat basis over the training
angles, alpha chosen on validation MSE."""

import numpy as np

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


def fit_ridge(data, knots):
    """Ridge, alpha chosen on validation MSE. Each angle has its own coefficients."""
    F_tr, F_va = ridge_features(data['X_train'], knots), ridge_features(data['X_val'], knots)
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-8
    Z_tr, Z_va = (F_tr - mu) / sd, (F_va - mu) / sd
    y_tr, y_va = data['y_train'].astype(np.float64), data['y_val'].astype(np.float64)
    ym = y_tr.mean()
    G = Z_tr.T @ Z_tr.copy()                    # copy: keeps the general product (Z.T @ Z rounds differently)
    b = Z_tr.T @ (y_tr - ym)
    lam, V = np.linalg.eigh(G)                  # one decomposition, stable for small alpha
    lam, Vb = np.maximum(lam, 0.0), V.T @ b
    best = None
    for al in RIDGE_ALPHAS:
        coef = V @ (Vb / (lam + al * len(y_tr)))
        err = float(np.mean((Z_va @ coef + ym - y_va) ** 2))
        if best is None or err < best[0]:
            best = (err, al, coef)
    err, al, coef = best
    pred = (ridge_features(data['X_test'], knots) - mu) / sd @ coef + ym
    return pred, {'alpha': float(al), 'alpha_at_edge': bool(al in (RIDGE_ALPHAS[0], RIDGE_ALPHAS[-1])),
                  'val_mse': err, 'n_features': int(len(coef)), 'epochs': 0}
