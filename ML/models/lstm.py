"""The study's network: stacked LSTM layers, each followed by dropout, then one linear output (ankle torque, Nm).

build_lstm(n_steps, n_features, 8, 2, 0.3) is the original model (1,001 parameters). The mixture-of-experts and
anchored variants were removed in 2026-11; see tag pre-cleanup-2026-11.
"""
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Sequential


def build_lstm(n_steps, n_features, units, layers, dropout, name='lstm_experiment'):
    """Input (n_steps, n_features) -> [LSTM(units, tanh) -> Dropout(dropout)] x layers -> Dense(1)."""
    lay = [Input(shape=(n_steps, n_features))]
    for i in range(layers):
        lay.append(LSTM(units, activation='tanh', return_sequences=i < layers - 1))
        lay.append(Dropout(dropout))
    lay.append(Dense(1))
    return Sequential(lay, name=name)
