"""Study v2 baseline A1: ridge regression on lagged EMG envelopes (NumPy only)."""
import numpy as np


def lagged_features(X, lags_steps=range(0, 51, 5)):
    """Study v2 baseline A1 (lagged ridge) features: the envelopes at each lag, the current angle and each current envelope x angle.

    X: (n, steps, channels) windows ending at the target sample, last channel = angle, steps > max lag.
    With 4 envelopes and 11 lags: 44 + 1 + 4 = 49 features. Uses no sample after the target.
    """
    lags = list(lags_steps)
    assert X.shape[1] > max(lags), f'windows of {X.shape[1]} steps are too short for lag {max(lags)}'
    env = [X[:, -1 - k, :-1] for k in lags]
    cur = X[:, -1]
    return np.column_stack(env + [cur[:, -1:], cur[:, :-1] * cur[:, -1:]])


class LaggedRidge:
    """Ridge on lagged_features; alpha chosen on the validation block (lowest MAE), then refit is not needed:
    each alpha is fitted on the training block only."""

    def __init__(self, alphas, lags_steps=range(0, 51, 5)):
        self.alphas, self.lags = [float(a) for a in alphas], list(lags_steps)

    def _fit_one(self, F, y, alpha):
        mean, scale = F.mean(axis=0), np.maximum(F.std(axis=0), 1e-8)
        z = np.column_stack([np.ones(len(y)), (F - mean) / scale])
        penalty = np.eye(z.shape[1]) * alpha
        penalty[0, 0] = 0
        return mean, scale, np.linalg.solve(z.T @ z + penalty, z.T @ y)

    def fit(self, X, y, X_val, y_val):
        F, Fv = lagged_features(X, self.lags), lagged_features(X_val, self.lags)
        fits, self.val_mae = {}, {}
        for a in self.alphas:
            fits[a] = self._fit_one(F, y, a)
            mean, scale, w = fits[a]
            self.val_mae[a] = float(np.mean(np.abs(w[0] + ((Fv - mean) / scale) @ w[1:] - y_val)))
        self.alpha = min(self.alphas, key=lambda a: (self.val_mae[a], a))
        self.mean, self.scale, self.weights = fits[self.alpha]
        return self

    def predict(self, X):
        return self.weights[0] + ((lagged_features(X, self.lags) - self.mean) / self.scale) @ self.weights[1:]
