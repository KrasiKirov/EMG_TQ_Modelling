"""Alternative model families (same inputs: 20-step window × [4 EMG, angle])."""

from tensorflow.keras import layers, Model, Sequential


def mlp(n_steps, n_features, **kw):
    """Plain feed-forward net on the flattened window."""
    return Sequential([layers.Input(shape=(n_steps, n_features)), layers.Flatten(),
                       layers.Dense(64, activation='relu'), layers.Dropout(0.2),
                       layers.Dense(32, activation='relu'), layers.Dense(1)])


def tcn(n_steps, n_features, **kw):
    """Small causal temporal CNN (dilations 1-2-4, receptive field 15 steps)."""
    inp = layers.Input(shape=(n_steps, n_features))
    h = inp
    for d in (1, 2, 4):
        h = layers.Conv1D(16, 3, dilation_rate=d, padding='causal', activation='relu')(h)
        h = layers.Dropout(0.1)(h)
    h = layers.Lambda(lambda x: x[:, -1, :])(h)
    return Model(inp, layers.Dense(1)(h))


def gru8(n_steps, n_features, **kw):
    """Same size as the baseline LSTM (8-8), GRU cells."""
    return Sequential([layers.Input(shape=(n_steps, n_features)),
                       layers.GRU(8, return_sequences=True), layers.Dropout(0.3),
                       layers.GRU(8), layers.Dropout(0.3), layers.Dense(1)])
