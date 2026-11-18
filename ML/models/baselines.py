"""Small NumPy baselines with explicit, non-pickle serialization."""
import numpy as np


def static_features(X):
    current = X[:, -1]
    return np.column_stack([current, current[:, :-1] * current[:, -1:]])


class RidgeModel:
    def __init__(self, alpha=1.):
        self.alpha = float(alpha)

    def fit(self, X, y):
        features = static_features(X)
        self.mean = features.mean(axis=0)
        self.scale = np.maximum(features.std(axis=0), 1e-8)
        z = np.column_stack([np.ones(len(y)), (features - self.mean) / self.scale])
        penalty = np.eye(z.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        return self

    def predict(self, X):
        features = (static_features(X) - self.mean) / self.scale
        return self.weights[0] + features @ self.weights[1:]

    def save(self, path):
        np.savez(path, kind='ridge', alpha=self.alpha, mean=self.mean, scale=self.scale, weights=self.weights)


def lagged_features(X, lags_steps=range(0, 51, 5)):
    """Study v2 A1 (D22): the envelopes at each lag, the current angle and each current envelope x angle.

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


class PositionModel:
    def fit(self, X, y, groups):
        points = sorted((float(np.mean(X[groups == g, -1, -1])), float(np.mean(y[groups == g])))
                        for g in np.unique(groups))
        # Coincident source angles receive equal group weight, not arbitrary
        # duplicate-knot interpolation dependent on their torque ordering.
        positions, torques = map(np.asarray, zip(*points))
        self.positions = np.unique(positions)
        self.torques = np.array([torques[positions == p].mean() for p in self.positions])
        return self

    def predict(self, X):
        # Explicitly clamp unseen angles to the nearest training support endpoint.
        return np.interp(X[:, -1, -1], self.positions, self.torques)

    def save(self, path):
        np.savez(path, kind='position', positions=self.positions, torques=self.torques)


def load_baseline(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data['kind']) == 'ridge':
            model = RidgeModel(float(data['alpha']))
            for attr in ('mean', 'scale', 'weights'):
                setattr(model, attr, data[attr])
        else:
            model = PositionModel()
            model.positions, model.torques = data['positions'], data['torques']
    return model
