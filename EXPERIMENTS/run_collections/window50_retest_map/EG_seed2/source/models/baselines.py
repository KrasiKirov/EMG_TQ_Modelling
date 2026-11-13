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
